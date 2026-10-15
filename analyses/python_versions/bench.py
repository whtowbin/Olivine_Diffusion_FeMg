"""FORSTER on Python 3.13 to 3.15: the JIT, free threading, and Numba kernels without the GIL.

    uv run python analyses/python_versions/bench.py matrix --venv-root <dir outside the repo>

``matrix`` makes one throwaway venv per interpreter in TARGETS under ``--venv-root``,
installs FORSTER's dependencies one package at a time (a wheel, else a source build,
else recorded as failed and left out), installs FORSTER from this checkout without
dependencies, records whether importing each package leaves the GIL disabled (each
import in a fresh process), and runs ``bench.py measure`` in the venv. Free-threaded
venvs also try the two GUI candidates (PySide6, nicegui). Every result goes to
``results.csv`` next to this file, one measurement per row (query it with DuckDB).
``--targets`` reruns some interpreters and replaces only their rows; ``--max-load``
waits (up to ``--max-wait`` minutes) for the 1-minute load average to drop below it
before measuring, since other processes on the machine skew every timing.

``measure`` benchmarks the interpreter running it on the packaged template example
(AZ18 WHT01_Ol_63_prof_1), with three time-stepping backends swapped in at run time:

- ``numba-nogil``: the installed `forster._kernels` (which release the GIL);
- ``numba-gil``: the same functions recompiled without ``nogil`` (FORSTER before
  2026-10-15), to show what releasing the GIL is worth;
- ``numpy``: the NumPy steppers, as with ``FORSTER_NO_NUMBA=1`` or without Numba.

Timings are medians over ``--repeats`` rounds, the backends (and worker counts)
interleaved within each round so that a change in machine load hits them all alike.
The first call, which loads or compiles the Numba kernels, is timed separately.

- ``fit``: `fit_run` of the template run, read from the tables `write_template` writes;
- ``scan125``: `scan_run` with the step (Gaussian, sigma 2 um), edge and core Fo
  (uniform +-0.003 and +-0.002) uncertain, 5 points each: 125 candidates;
- ``batch64``: 64 candidates (8 step positions x 8 edge Fo, on the scan's grid) run by
  `run_candidates` in 16 chunks of 4 from a ThreadPoolExecutor with 1, 4 and 8
  workers; each timing repeats the batch until it lasts about 0.2 s and reports the
  time per batch; the outputs must be bit for bit those of 1 worker;
- ``heartbeat_*``: a thread that sleeps until its next 5 ms tick, as a GUI event loop
  does, records how late it wakes while one worker thread repeats `fit_run`
  (``fit``), the 64-candidate batch on 1 worker (``batch``) or a 0.25 s explicit
  kernel call (``kernel``, Numba only) for 2 s; ``idle`` has no worker.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = HERE / "results.csv"
COLUMNS = (
    "round",
    "target",
    "python",
    "metric",
    "backend",
    "threads",
    "value",
    "unit",
    "detail",
)

# label: (uv python request, extra environment)
TARGETS = {
    "3.13": ("3.13", {}),
    "3.14": ("3.14", {"PYTHON_JIT": "0"}),
    "3.14+jit": ("3.14", {"PYTHON_JIT": "1"}),
    "3.15": ("3.15", {"PYTHON_JIT": "0"}),
    "3.15+jit": ("3.15", {"PYTHON_JIT": "1"}),
    "3.14t": ("3.14t", {}),
    "3.15t": ("3.15t", {}),
}
DEPENDENCIES = ("numpy", "scipy", "pandas", "matplotlib", "openpyxl", "pykrige", "numba")
GUI_PACKAGES = {"PySide6": "PySide6.QtWidgets", "nicegui": "nicegui"}
# Dependencies left out when an install fails with them: orjson refuses to build for
# free-threaded Python, and nicegui falls back to the json module without it.
DROP_ON_FAILURE = {"nicegui": ("orjson",)}
# Modules whose import is checked, each in a fresh process, for re-enabling the GIL.
GIL_MODULES = (
    "numpy",
    "numpy.random",  # Cython-built, unlike numpy's core
    "scipy.linalg",
    "pandas",
    "matplotlib.pyplot",
    "openpyxl",
    "pykrige",
    "numba",
    "forster",
    "forster._kernels",
)
SOURCE_BUILD_TIMEOUT_S = 900
THREADS = (1, 4, 8)
CHUNK = 4  # candidates per task in the batch
BATCH_TIMING_S = 0.2  # each batch timing repeats the batch to last about this long
TICK_S = 0.005
HEARTBEAT_S = 2.0
KERNEL_STEPS = 250_000  # one explicit kernel call of about 0.25 s


# ---------------------------------------------------------------- measure (in a venv)


def _row(metric: str, value: Any, unit: str = "", backend: str = "", threads: Any = "", **kw):
    return {
        "metric": metric,
        "backend": backend,
        "threads": threads,
        "value": value,
        "unit": unit,
        "detail": json.dumps(kw, sort_keys=True) if kw else "",
    }


def _gil_holding(kernels: Any) -> SimpleNamespace:
    """The kernels recompiled without nogil. The inner `_d` and `_stage` become native
    calls inside them, so the GIL is held for the whole outer call."""
    from numba import njit

    out = SimpleNamespace(GAMMA=kernels.GAMMA)
    for name in ("central_steps", "dipra_steps", "tr_bdf2_steps"):
        setattr(out, name, njit(nogil=False)(getattr(kernels, name).py_func))
    return out


@contextlib.contextmanager
def _backend(kernels: Any) -> Iterator[None]:
    from forster import compiled

    saved = compiled._kernels
    compiled._kernels = kernels
    try:
        yield
    finally:
        compiled._kernels = saved


def _backends() -> dict[str, Any]:
    from forster import compiled

    out: dict[str, Any] = {}
    if compiled._kernels is not None:
        out["numba-nogil"] = compiled._kernels
        out["numba-gil"] = _gil_holding(compiled._kernels)
    out["numpy"] = None
    return out


def _example(tmp: Path):
    import forster
    from forster import Uncertainty, candidate_grid

    forster.write_template(tmp)
    run = forster.read_runs(tmp / "parameters.csv")[0]
    profiles = forster.read_profiles(tmp / "profiles.csv")
    run125 = replace(
        run,
        uncertainties={
            "step1_um": Uncertainty(2.0),
            "edge_fo": Uncertainty(0.003, "uniform"),
            "core_fo": Uncertainty(0.002, "uniform"),
        },
        scan_points=5,
    )
    batch = candidate_grid(
        run.initial, {"step1_um": Uncertainty(2.0), "edge_fo": Uncertainty(0.003, "uniform")}, 8
    ).candidates
    return run, profiles, run125, batch


def _batch(run, profiles, candidates, grid, threads: int) -> list:
    from forster.scan import run_candidates

    profile = profiles[run.profile_name]
    chunks = [candidates[i : i + CHUNK] for i in range(0, len(candidates), CHUNK)]

    def work(chunk):
        return run_candidates(
            profile, run.conditions, run.orientation, chunk, grid, run.diffusivity, run.scheme
        )

    with ThreadPoolExecutor(threads) as pool:
        return list(pool.map(work, chunks))


def _flat(results: list) -> list:
    return [a for r in results for a in r.at_points]


def _identical(a: list, b: list) -> bool:
    return len(a) == len(b) and all(
        x.shape == y.shape and x.tobytes() == y.tobytes() for x, y in zip(a, b, strict=True)
    )


def _heartbeat(work: Callable[[], Any], duration_s: float = HEARTBEAT_S) -> dict[str, float]:
    """Lateness of a thread waking every TICK_S while another thread repeats ``work``
    for ``duration_s``."""
    late: list[float] = []
    stop = threading.Event()

    def beat():
        due = time.perf_counter() + TICK_S
        while not stop.is_set():
            wait = due - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            now = time.perf_counter()
            late.append(now - due)
            due = now + TICK_S

    iterations = 0

    def worker():
        nonlocal iterations
        end = time.perf_counter() + duration_s
        while time.perf_counter() < end:
            work()
            iterations += 1

    beater = threading.Thread(target=beat)
    beater.start()
    time.sleep(0.05)
    late.clear()
    busy = threading.Thread(target=worker)
    busy.start()
    busy.join()
    stop.set()
    beater.join()
    ms = sorted(x * 1e3 for x in late)
    return {
        "p50_ms": ms[len(ms) // 2],
        "p99_ms": ms[min(len(ms) - 1, int(0.99 * len(ms)))],
        "max_ms": ms[-1],
        "beats": len(ms),
        "iterations": iterations,
    }


def _kernel_call(kernels, run) -> Callable[[], Any] | None:
    """One explicit-kernel call of KERNEL_STEPS steps on 151 nodes (Numba only)."""
    import numpy as np

    from forster import compiled

    if kernels is None:
        return None
    D, _ = run.diffusivity.functions(run.conditions, run.orientation)
    params = compiled._parameters(D)
    if params is None:
        return None
    x = np.linspace(0.0, 150.0, 151)
    start = np.where(x < 20.3, 0.82, 0.88)
    dx = 1e-6
    dt = 0.4 * dx**2 / float(D(np.array([0.88]))[0])
    every = 100
    n = KERNEL_STEPS // every
    return lambda: kernels.central_steps(start, *params, 0.82, 0.88, dt, dx, n, every)


def _environment_rows() -> list[dict]:
    jit = getattr(sys, "_jit", None)
    rows = [
        _row("free_threaded_build", int(bool(sysconfig.get_config_var("Py_GIL_DISABLED")))),
        _row("jit_available", int(bool(jit and jit.is_available()))),
        _row("jit_enabled", int(bool(jit and jit.is_enabled()))),
        _row("cpu_count", os.cpu_count()),
        _row("loadavg_1min_start", os.getloadavg()[0]),
    ]
    from forster import compiled

    rows.append(_row("numba_kernels", int(compiled.AVAILABLE)))
    if compiled._kernels is not None:
        import numba

        nogil = compiled._kernels.central_steps.targetoptions.get("nogil", False)
        rows.append(_row("numba_kernels_nogil", int(nogil), numba=numba.__version__))
    if hasattr(sys, "_is_gil_enabled"):
        rows.append(_row("gil_enabled_during_measure", int(sys._is_gil_enabled())))
    return rows


def _interleaved(jobs: dict[tuple, Callable[[], Any]], repeats: int) -> dict[tuple, list]:
    """Seconds per call of each job, the jobs taking turns in every round."""
    times: dict[tuple, list] = {key: [] for key in jobs}
    for _ in range(repeats):
        for key, fn in jobs.items():
            start = time.perf_counter()
            fn()
            times[key].append(time.perf_counter() - start)
    return times


def _in_backend(kernels: Any, fn: Callable[[], Any], times: int = 1) -> Callable[[], Any]:
    def call():
        with _backend(kernels):
            for _ in range(times):
                fn()

    return call


def _warm(run, profiles, run125, batch, backends) -> tuple[list, dict, Any]:
    """First calls (Numba compiles or loads its cache), results, and batch repeats."""
    from forster import fit_run, scan_run

    rows, nominal, grid = [], {}, None
    for name, kernels in backends.items():
        with _backend(kernels):
            start = time.perf_counter()
            nominal[name] = fit_run(run, profiles)
            rows.append(_row("fit_first_call", time.perf_counter() - start, "s", name))
            scan = scan_run(run125, profiles, time_days=nominal[name].time_days)
            rows.append(
                _row(
                    "result_days",
                    nominal[name].time_days,
                    "days",
                    name,
                    scan_days=scan.time_days,
                    scan_candidates=len(scan.runs.candidates),
                )
            )
            grid = scan.runs.grid  # the same for every backend
    return rows, nominal, grid


def _batch_checks(run, profiles, batch, grid, backends) -> tuple[list, dict]:
    """Bit-for-bit check of every worker count against 1, and repeats per timing."""
    rows, inner = [], {}
    for name, kernels in backends.items():
        with _backend(kernels):
            start = time.perf_counter()
            reference = _flat(_batch(run, profiles, batch, grid, 1))
            inner[name] = max(1, math.ceil(BATCH_TIMING_S / (time.perf_counter() - start)))
            for threads in THREADS[1:]:
                same = _identical(_flat(_batch(run, profiles, batch, grid, threads)), reference)
                rows.append(_row("batch64_identical", int(same), "", name, threads))
    return rows, inner


def _timing_rows(run, profiles, run125, batch, backends, repeats) -> tuple[list, Any]:
    from forster import fit_run, scan_run

    rows, nominal, grid = _warm(run, profiles, run125, batch, backends)
    check_rows, inner = _batch_checks(run, profiles, batch, grid, backends)
    rows += check_rows
    jobs: dict[tuple, Callable[[], Any]] = {}
    for name, kernels in backends.items():
        days = nominal[name].time_days
        jobs["fit", name, ""] = _in_backend(kernels, lambda: fit_run(run, profiles))
        jobs["scan125", name, ""] = _in_backend(
            kernels, lambda d=days: scan_run(run125, profiles, time_days=d)
        )
        for threads in THREADS:
            jobs["batch64", name, threads] = _in_backend(
                kernels, lambda t=threads: _batch(run, profiles, batch, grid, t), inner[name]
            )
    for (metric, name, threads), times in _interleaved(jobs, repeats).items():
        n = inner[name] if metric == "batch64" else 1
        per = [t / n for t in times]
        rows.append(
            _row(
                metric,
                statistics.median(per),
                "s",
                name,
                threads,
                min=min(per),
                max=max(per),
                repeats=len(per),
                calls_per_timing=n,
            )
        )
    return rows, grid


def _heartbeat_rows(run, profiles, batch, grid, backends) -> list:
    from forster import fit_run

    rows = []
    for name, kernels in backends.items():
        work = {
            "fit": lambda: fit_run(run, profiles),
            "batch": lambda: _batch(run, profiles, batch, grid, 1),
            "kernel": _kernel_call(kernels, run),
        }
        with _backend(kernels):
            for label, fn in work.items():
                if fn is None:
                    continue
                fn()  # compile and warm
                r = _heartbeat(fn)
                rows.append(_row(f"heartbeat_{label}", r["max_ms"], "ms max", name, **r))
    return rows


def _numpy25_shim() -> bool:
    """NumPy 2.5 turned float() of an array with ndim > 0 from a DeprecationWarning (since
    1.25) into a TypeError. `forster.orientation.vector_direction` calls float() on the 1x1
    np.matrix that `Shortest_angle` returns, so reading a template with Euler angles
    fails. Until src is fixed, make `Shortest_angle` return the same value as a 0-d array.
    Returns whether the shim was needed."""
    import numpy as np

    from forster import orientation

    try:
        float(np.matrix([[1.0]]))
        return False
    except TypeError:
        original = orientation.Shortest_angle
        orientation.Shortest_angle = lambda uvw, axis: np.asarray(original(uvw, axis)).reshape(())
        return True


def measure(repeats: int) -> list[dict]:
    import warnings

    import numpy as np

    warnings.simplefilter("ignore")
    rows = _environment_rows()
    rows.append(_row("numpy25_orientation_shim", int(_numpy25_shim()), numpy=np.__version__))
    idle = _heartbeat(lambda: time.sleep(0.01))
    rows.append(_row("heartbeat_idle", idle["max_ms"], "ms max", **idle))
    backends = _backends()
    with tempfile.TemporaryDirectory() as tmp:
        run, profiles, run125, batch = _example(Path(tmp))
        timing, grid = _timing_rows(run, profiles, run125, batch, backends, repeats)
        rows += timing
        rows += _heartbeat_rows(run, profiles, batch, grid, backends)
    rows.append(_row("loadavg_1min_end", os.getloadavg()[0]))
    return rows


# ---------------------------------------------------------------- matrix (orchestration)


def _sh(cmd: list[str], timeout: float | None = None, env: dict | None = None):
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout, env=env, check=False
        )
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(cmd, 124, "", f"timed out after {timeout} s")


def _last_error(result) -> str:
    if result.returncode < 0:
        return f"killed by signal {-result.returncode} (11 is SIGSEGV)"
    lines = [x.strip() for x in (result.stderr or "").splitlines() if x.strip()]
    errors = [x for x in lines if "error" in x.lower() or "because" in x.lower()]
    return (errors or lines or ["?"])[-1][:300]


def _version(python: str, dist: str) -> str:
    r = _sh([python, "-c", f"import importlib.metadata as m; print(m.version({dist!r}))"])
    return r.stdout.strip() if r.returncode == 0 else ""


def _install(python: str, package: str, venv: Path) -> dict:
    """Install ``package`` from a wheel if one exists, else from source (cut off after
    SOURCE_BUILD_TIMEOUT_S). Whether a wheel exists is asked with a fresh resolution, so
    a rerun over an installed package reports the same."""
    base = ["uv", "pip", "install", "--python", python]
    start = time.perf_counter()
    probe = _sh(
        [*base, "--dry-run", "--reinstall-package", package, "--only-binary", ":all:", package],
        timeout=600,
    )
    has_wheel = probe.returncode == 0
    how = "wheel" if has_wheel else "source build"
    if has_wheel:
        result = _sh([*base, "--only-binary", ":all:", package], timeout=600)
    else:
        # --no-cache: build afresh, not from a wheel an earlier (maybe patched) build cached
        result = _sh([*base, "--no-cache", package], timeout=SOURCE_BUILD_TIMEOUT_S)
        if result.returncode != 0 and "SIGSEGV" in result.stderr:
            # Cython's compiled wheel crashed building pentapy (pykrige's build dependency)
            # on 3.15.0b3; Cython as pure Python builds it.
            first_error = _last_error(result)
            result = _sh([*base, "--no-binary", "cython", package], timeout=SOURCE_BUILD_TIMEOUT_S)
            how = f"source build, Cython from source (compiled Cython: {first_error})"
    if result.returncode != 0 and package in DROP_ON_FAILURE:
        dropped = DROP_ON_FAILURE[package]
        overrides = venv / "drop-overrides.txt"
        overrides.write_text("".join(f"{d}; python_version < '3'\n" for d in dropped))
        first_error = _last_error(result)
        result = _sh(
            [*base, "--overrides", str(overrides), package], timeout=SOURCE_BUILD_TIMEOUT_S
        )
        how = f"{how} without {', '.join(dropped)} (with it: {first_error})"
    ok = result.returncode == 0
    return {
        "how": how if ok else "failed",
        "version": _version(python, package) if ok else "",
        "seconds": round(time.perf_counter() - start, 1),
        "no_wheel_reason": "" if has_wheel else _last_error(probe),
        "error": "" if ok else _last_error(result),
    }


def _gil_after_import(python: str, module: str, env: dict) -> dict:
    code = (
        "import sys, time; t = time.perf_counter(); import "
        + module
        + "; print(time.perf_counter() - t, int(sys._is_gil_enabled()))"
    )
    r = _sh([python, "-W", "default", "-c", code], timeout=300, env={**env, "MPLBACKEND": "Agg"})
    if r.returncode != 0:
        return {"ok": False, "error": _last_error(r)}
    seconds, gil = r.stdout.split()[-2:]
    warned = [x.strip() for x in r.stderr.splitlines() if "GIL" in x]
    return {"ok": True, "gil": int(gil), "import_s": float(seconds), "warning": warned[:1]}


def _venv(label: str, venv_root: Path) -> Path:
    return venv_root / TARGETS[label][0]


def _setup(label: str, venv_root: Path) -> tuple[str, list[dict]]:
    request = TARGETS[label][0]
    venv = _venv(label, venv_root)
    python = str(venv / "bin" / "python")
    if not Path(python).exists():
        r = _sh(["uv", "venv", "-q", "--python", request, str(venv)], timeout=900)
        if r.returncode != 0:
            raise RuntimeError(f"{label}: uv venv failed: {_last_error(r)}")
    rows = []
    packages = list(DEPENDENCIES) + (list(GUI_PACKAGES) if request.endswith("t") else [])
    for package in packages:
        info = _install(python, package, venv)
        rows.append(_row("install", info["how"], backend=package, **info))
    r = _sh(["uv", "pip", "install", "--python", python, "--no-deps", "-e", str(REPO)])
    rows.append(_row("install", "editable" if r.returncode == 0 else "failed", backend="forster"))
    return python, rows


def _wait_for_load(max_load: float | None, max_wait_min: float) -> None:
    if max_load is None:
        return
    end = time.monotonic() + 60 * max_wait_min
    while os.getloadavg()[0] > max_load and time.monotonic() < end:
        print(f"   load {os.getloadavg()[0]:.1f} > {max_load}; waiting", flush=True)
        time.sleep(30)


def _run_target(label: str, venv_root: Path, repeats: int, load: tuple) -> list[dict]:
    request, extra = TARGETS[label]
    print(f"== {label}: setting up", flush=True)
    python, rows = _setup(label, venv_root)
    env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PYTHON_JIT")}
    env |= extra
    env["NUMBA_CACHE_DIR"] = str(_venv(label, venv_root) / "numba_cache")
    modules = list(GIL_MODULES)
    if request.endswith("t"):
        modules += [*GUI_PACKAGES.values(), "forster._kernels, nicegui"]
    for module in modules:
        info = _gil_after_import(python, module, env)
        rows.append(_row("gil_after_import", info.get("gil", ""), backend=module, **info))
    _wait_for_load(*load)
    print(f"== {label}: measuring (load {os.getloadavg()[0]:.1f})", flush=True)
    r = _sh([python, str(Path(__file__)), "measure", "--repeats", str(repeats)], env=env)
    if r.returncode != 0:
        rows.append(_row("measure", "failed", error=_last_error(r)))
        print(r.stderr[-3000:], flush=True)
    else:
        rows += [json.loads(line) for line in r.stdout.splitlines() if line.startswith("{")]
    version = _sh([python, "-c", "import sys; print(sys.version.split()[0])"]).stdout.strip()
    for row in rows:
        row.update(target=label, python=version)
    return rows


def matrix(labels: list[str], venv_root: Path, repeats: int, load: tuple, rnd: int) -> None:
    """Rows for ``labels`` in round ``rnd`` replace any earlier ones; others are kept."""
    if venv_root.resolve().is_relative_to(REPO):
        raise SystemExit("--venv-root must be outside the repository")
    venv_root.mkdir(parents=True, exist_ok=True)
    kept = []
    if RESULTS.exists():
        with RESULTS.open() as f:
            kept = [
                r
                for r in csv.DictReader(f)
                if not (r["target"] in labels and r.get("round", "1") == str(rnd))
            ]
    rows = []
    for label in labels:
        rows += [r | {"round": rnd} for r in _run_target(label, venv_root, repeats, load)]
        _write(kept + rows)
    print(f"{len(rows)} rows for {labels} written to {RESULTS} ({platform.platform()})")


def _write(rows: list[dict]) -> None:
    order = {label: i for i, label in enumerate(TARGETS)}
    rows = sorted(rows, key=lambda r: (int(r.get("round") or 1), order.get(r["target"], 99)))
    with RESULTS.open("w", newline="") as f:
        writer = csv.DictWriter(f, COLUMNS)
        writer.writeheader()
        writer.writerows({k: r.get(k, "") for k in COLUMNS} for r in rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("measure", help="benchmark this interpreter; JSON lines on stdout")
    m.add_argument("--repeats", type=int, default=7)
    x = sub.add_parser("matrix", help="set up a venv per interpreter and measure each")
    x.add_argument("--venv-root", type=Path, required=True)
    x.add_argument("--targets", nargs="+", default=list(TARGETS), choices=list(TARGETS))
    x.add_argument("--repeats", type=int, default=7)
    x.add_argument("--round", type=int, default=1, help="a later round keeps earlier rounds")
    x.add_argument("--max-load", type=float, default=None)
    x.add_argument("--max-wait", type=float, default=30.0, help="minutes")
    s = sub.add_parser("setup", help="only make the venvs and install (matrix redoes this fast)")
    s.add_argument("--venv-root", type=Path, required=True)
    s.add_argument("--targets", nargs="+", default=list(TARGETS), choices=list(TARGETS))
    args = parser.parse_args()
    if args.command == "measure":
        for row in measure(args.repeats):
            print(json.dumps(row), flush=True)
    elif args.command == "setup":
        for label in args.targets:
            _, rows = _setup(label, args.venv_root)
            print(label, [(r["backend"], r["value"]) for r in rows], flush=True)
    else:
        load = (args.max_load, args.max_wait)
        matrix(args.targets, args.venv_root, args.repeats, load, args.round)


if __name__ == "__main__":
    main()
