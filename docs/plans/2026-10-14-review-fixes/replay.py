"""Differential replay for the group C refactors (plan: ../2026-10-14-review-fixes.md).

Runs the same inputs through whichever `forster` is first on the path and saves, for
every array and number of every result, its shape, dtype and a SHA-256 of its bytes (the
curves alone would be 4 GB); `compare` checks two such files for bitwise equality. To
compare a baseline commit with the working tree:

    git worktree add /tmp/forster-base <commit>
    PYTHONPATH=/tmp/forster-base/src uv run python replay.py run base.npz
    uv run python replay.py run new.npz
    uv run python replay.py compare base.npz new.npz

Run each with and without FORSTER_NO_NUMBA=1. A baseline run twice is the control.
Inputs: every round robin and paper run with its own scheme and grid and again with
central, dipra and implicit (correlated method); growth stages (central and implicit)
and a changing rim; a Ni element fit and a joint Fo + Ni fit; explicit and implicit
initial-condition scans; `choose_grid` for every scheme.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
import sys
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "tests"))


def flatten(obj, prefix: str, out: dict) -> None:
    """Every number, string and array reachable from ``obj``, keyed by its path."""
    if obj is None or (callable(obj) and not dataclasses.is_dataclass(obj)):
        return
    if isinstance(obj, np.ndarray):
        out[prefix] = obj
    elif isinstance(obj, bool | int | float | str | np.floating | np.integer):
        out[prefix] = np.asarray(obj)
    elif dataclasses.is_dataclass(obj):
        for f in dataclasses.fields(obj):
            flatten(getattr(obj, f.name), f"{prefix}.{f.name}", out)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            flatten(v, f"{prefix}[{k}]", out)
    elif isinstance(obj, list | tuple):
        for i, v in enumerate(obj):
            flatten(v, f"{prefix}[{i}]", out)
    elif hasattr(obj, "_asdict"):
        flatten(obj._asdict(), prefix, out)


def cases():
    """(label, thunk) for every input."""
    from conftest import PAPER_DATA, PAPER_PARAMS, ROUND_ROBIN_DATA, ROUND_ROBIN_PARAMS

    import forster
    from forster import (
        Conditions,
        ElementInitial,
        ElementProfile,
        Grid,
        InitialCondition,
        Orientation,
        Profile,
        choose_grid,
        fit_element,
        fit_joint,
        fit_profile,
        fit_run,
        read_profiles,
        read_runs,
        write_template,
    )
    from forster.boundary import RimPath, Stage
    from forster.io import scan_run
    from forster.scan import Uncertainty, candidate_grid, scan_initial_conditions

    day = 86400.0
    tables = [
        (read_runs(ROUND_ROBIN_PARAMS), read_profiles(ROUND_ROBIN_DATA)),
        (read_runs(PAPER_PARAMS), read_profiles(PAPER_DATA, sheet="Olivine")),
    ]
    for runs, profiles in tables:
        for run in runs:
            yield f"run/{run.name}/own", lambda r=run, p=profiles: fit_run(r, p)
            for scheme in ("central", "dipra", "implicit"):
                yield (
                    f"run/{run.name}/{scheme}",
                    lambda r=run, p=profiles, s=scheme: fit_run(
                        replace(r, scheme=s, uncertainty="correlated"), p
                    ),
                )
    rr_runs, rr_profiles = tables[0]
    for run in rr_runs[:4]:
        m = rr_profiles[run.profile_name]
        profile = Profile(m.distance_um, m.fo, name=run.name)
        for scheme in ("central", "dipra", "implicit"):
            yield (
                f"choose_grid/{run.name}/{scheme}",
                lambda r=run, p=profile, s=scheme: choose_grid(
                    p, r.conditions, r.orientation, r.initial, scheme=s
                ),
            )

    rr1200 = Conditions(T_C=1200, P_Pa=4.2e7, fO2_Pa=0.00138)
    along = Orientation(90, 90, 0)
    x = np.arange(0.0, 200.0, 1.0)
    dummy = Profile(x, np.full(len(x), 0.88) + 1e-4 * np.sin(x), name="dummy")
    early = InitialCondition((40, 199), edge_fo=0.86, core_fo=0.90, steps_um=(40,))
    growth = Stage(6.0 * day, rim_um=0.0, zone_value=0.82)
    yield (
        "stages/central",
        lambda: fit_profile(
            dummy,
            rr1200,
            along,
            early,
            Grid(1.0, 20.0, 10),
            stop_early=False,
            stages=[growth],
            scheme="central",
        ),
    )
    yield (
        "stages/implicit",
        lambda: fit_profile(
            dummy, rr1200, along, early, Grid(1.0, 0.05 * day, 10), stages=[growth]
        ),
    )
    whole = InitialCondition((0, 199), edge_fo=0.86, core_fo=0.90, steps_um=(40,))
    yield (
        "rimpath/implicit",
        lambda: fit_profile(
            dummy,
            rr1200,
            along,
            whole,
            Grid(1.0, 0.05 * day, 20),
            rim_fo=RimPath((0.0, 3.0), (0.86, 0.82)),
        ),
    )

    fo_initial = InitialCondition((0, 299), edge_fo=0.80, core_fo=0.90, steps_um=(150,))
    points = np.arange(2.0, 298.0, 4.0)
    rng = np.random.default_rng(7)
    shape = 0.5 + 0.5 * np.tanh((points - 150) / 25)
    fo_profile = Profile(points, 0.80 + 0.10 * shape + rng.normal(0, 0.001, len(points)))
    ni_profile = ElementProfile(
        points, 0.40 - 0.10 * shape + rng.normal(0, 0.002, len(points)), "Ni"
    )
    ni = ElementInitial(edge=0.40, core=0.30)
    fit_grid = Grid(dx_um=1.0, dt_s=60.0, max_time_days=30)
    yield (
        "element/ni",
        lambda: fit_element(ni_profile, rr1200, along, fo_initial, ni, fit_grid),
    )
    yield (
        "element/ni_fitted_concentrations",
        lambda: fit_element(ni_profile, rr1200, along, fo_initial, None, fit_grid),
    )
    yield (
        "joint/fo_ni",
        lambda: fit_joint(
            fo_profile, [ni_profile], rr1200, along, fo_initial, {"Ni": ni}, fit_grid
        ),
    )

    import tempfile

    import pandas as pd

    tmp = Path(tempfile.mkdtemp())
    write_template(tmp / "t")
    template_profiles = read_profiles(tmp / "t" / "profiles.csv")
    for scheme in ("implicit", "central"):
        params = pd.read_csv(tmp / "t" / "parameters.csv").assign(
            dx_um=np.nan, dt_s=np.nan, scheme=scheme, steps_sd_um=1.0, scan_points=3
        )
        params.to_csv(tmp / f"{scheme}.csv", index=False)
        (run,) = read_runs(tmp / f"{scheme}.csv")
        yield (
            f"scan_run/{scheme}",
            lambda r=run: scan_run(
                r, template_profiles, time_days=fit_run(r, template_profiles).time_days
            ),
        )
    run = rr_runs[0]
    m = rr_profiles[run.profile_name]
    grid_c = candidate_grid(
        run.initial,
        {"step1_um": Uncertainty(4.0, "uniform"), "edge_fo": Uncertainty(0.003, "normal")},
        points=3,
    )
    yield (
        "scan/central_two_parameters",
        lambda: scan_initial_conditions(
            Profile(m.distance_um, m.fo),
            run.conditions,
            run.orientation,
            grid_c,
            replace(run.grid, max_time_days=150),
            scheme="central",
        ),
    )
    _ = forster


def _digest(v: np.ndarray) -> str:
    sha = hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()
    return f"{v.dtype.str} {v.shape} {sha}"


def run(path: str) -> None:
    import forster

    warnings.simplefilter("ignore")
    print(
        f"forster from {Path(forster.__file__).parent}; numba off: "
        f"{os.environ.get('FORSTER_NO_NUMBA') == '1'}"
    )
    out: dict = {}
    for label, thunk in cases():
        try:
            flatten(thunk(), label, out)
        except Exception as error:  # a case that raises is part of the behaviour
            out[f"{label}!error"] = np.asarray(f"{type(error).__name__}: {error}")
    digests = {k: np.asarray(_digest(v)) for k, v in out.items()}
    np.savez(path, **digests)
    print(f"{len(out)} arrays from {len({k.split('.')[0] for k in out})} cases -> {path}")


def compare(a_path: str, b_path: str) -> int:
    a, b = np.load(a_path), np.load(b_path)
    only_a, only_b = sorted(set(a.files) - set(b.files)), sorted(set(b.files) - set(a.files))
    differ = []
    for key in sorted(set(a.files) & set(b.files)):
        if str(a[key]) != str(b[key]):
            differ.append(key)
    print(
        f"{len(a.files)} vs {len(b.files)} arrays; only in first: {len(only_a)}, "
        f"only in second: {len(only_b)}, different: {len(differ)}"
    )
    for key in (only_a + only_b + differ)[:40]:
        print("  ", key)
    return 1 if only_a or only_b or differ else 0


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2])
    else:
        sys.exit(compare(sys.argv[2], sys.argv[3]))
