# Python 3.14/3.15, the JIT, free threading, and a GUI worker thread

Status (2026-10-15): measured. One code change: the five Numba kernels in
`src/forster/_kernels.py` now release the GIL (`nogil=True`). Nothing else in `src` or
`pyproject.toml` changed. The benchmark is `analyses/python_versions/bench.py`, and every
number below comes from its output, `analyses/python_versions/results.csv` (one
measurement per row, two rounds).

## Summary

- **nogil kernels:** the output is bit for bit unchanged. A 0.25 s kernel call no
  longer freezes other threads: a 5 ms heartbeat thread was up to 215-242 ms late with
  the GIL-holding kernels and at most 5.5 ms late with nogil (p99 1.3 ms, the same as
  idle). On standard CPython, 64 candidates on 4 threads now run 1.7x faster than on
  one thread. With the GIL-holding kernels, more threads made the batch 15% slower.
- **The JIT** (3.14, `PYTHON_JIT=1`) changes nothing measurable. A NumPy-path fit took
  101.4 ms without it and 101.7 ms with it; the Numba path was 14.4-14.6 ms either way.
  3.15's JIT could not be measured, because SciPy, pandas and numpy.random crash on
  import in 3.15.0b3 (below).
- **3.13 to 3.14:** the NumPy path is 4-5% faster (fit 106.8 to 101.4 ms; 125-candidate
  scan 757 to 725 ms). The Numba path is the same (fit 14.3 ms, scan 37.5 ms).
- **Free threading (3.14t)** works for FORSTER's own stack. Every dependency imports
  with the GIL still disabled, and the full test suite passes (239 tests). Batches
  scale 3.1-3.3x on 4 threads and 3.6-4.1x on 8, but single-thread NumPy-path work is 11-15%
  slower. pykrige has no free-threaded wheel, so users would need a C compiler.
  PySide6 has no free-threaded wheel at all, and nicegui installs only without orjson.
- **3.15.0b3** cannot run FORSTER. Every Cython-built extension (numpy.random, SciPy,
  pandas, pykrige) segfaults in `import_array` on both the default and the
  free-threaded build. That includes the scientific-python nightly wheels.
- **GUI recommendation:** run fits in a worker thread on standard CPython 3.13 or 3.14,
  with Numba installed. A subprocess is not needed for responsiveness, and free
  threading is not needed for speed.
- **Found on the way (needs a fix in `src`):** NumPy 2.5 breaks `read_runs` for any
  table with Euler angles, on every Python version. Details are in the last section.

## Setup

- **Machine:** Apple M4 Pro, 12 cores (8 performance, 4 efficiency), macOS 27.
  - Other Claude sessions were running on the machine, so `bench.py` waited for the
    1-minute load average to fall below 10 before measuring each interpreter. The
    recorded load was 4.6-7.9 for the rows used here.
  - A 3.14t run at load 9.8-10.9 was discarded and rerun at 5.3.
- **Interpreters:** uv's builds of 3.13.11, 3.14.6, 3.14.6+freethreaded, 3.15.0b3 and
  3.15.0b3+freethreaded. Each got a throwaway venv outside the repo.
- **Packages:** numpy 2.5.4, scipy 1.18.1, pandas 3.0.6, matplotlib 3.11.2, numba
  0.68.0 and pykrige 1.7.3, all the newest from PyPI. These are newer than the project
  lock (numpy 2.3.3), and NumPy 2.5 needed a shim (last section).
- **Workload:** the packaged template example (`write_template`, then `read_runs`,
  `read_profiles` and `fit_run`), with three time-stepping backends swapped in at run
  time in the same process:
  - `numba-nogil`: the new kernels;
  - `numba-gil`: the same functions recompiled without `nogil`, which is FORSTER
    before this change;
  - `numpy`: the fallback used without Numba or with `FORSTER_NO_NUMBA=1`.
- **Timing:** medians of 7 rounds, with the backends and thread counts interleaved
  within each round. The two rounds agree to within 3% for every row below except
  3.14t, whose first round ran at a higher load; its range is shown.

## Part 1: nogil kernels

### Output unchanged

- **Bitwise check:** the kernels were run on fixed inputs before and after the edit,
  each time with a fresh Numba cache directory. The inputs were two diffusivity laws;
  `central_steps` and `dipra_steps` at 1, 2 and 3 substeps; `tr_bdf2_steps` with a
  moving rim; `_stage`; and `_d`. All 14 output arrays are byte-identical.
- **Tests:** `tests/test_compiled.py` and `tests/test_differential.py` pass (30 tests),
  and so does the full suite.
- **In the benchmark:** every backend gives the same fitted time (5.1367 days) and the
  same scan result. For every backend, the batch outputs on 4 and 8 threads are
  bit for bit those of one thread.

### Thread scaling

The batch is 64 candidates (8 step positions x 8 edge Fo, on the scan grid), run by
`forster.scan.run_candidates` in 16 chunks of 4 from a `ThreadPoolExecutor`. Times are
milliseconds per batch.

| Interpreter | Backend | 1 thread | 4 threads | 8 threads | Speed-up at 4 |
|---|---|---|---|---|---|
| 3.13 / 3.14 | numba-gil (before) | 16.8 | 19.5 | 20.0 | 0.86x |
| 3.13 / 3.14 | numba-nogil (now) | 16.4 | 9.8 | 11.0 | 1.7x |
| 3.13 / 3.14 | numpy | 365-381 | 368-383 | 370-384 | 1.0x |
| 3.14t | numba (either) | 16.2-17.1 | 5.0-6.0 | 4.0-4.8 | 2.8-3.3x |
| 3.14t | numpy | 431-453 | 135-161 | 95-114 | 3.0x |

- **The limit on standard CPython:** in a profile of the serial batch, the kernels took
  62% of the time. The other 38% is Python that holds the GIL: setting up each run,
  the residual sums and the tail extension. Only the kernel time can overlap, which
  caps 4 threads at 1/(0.38 + 0.62/4) = 1.9x, so 1.7x is close to that limit. 8 threads are no
  better than 4.
- **Size of the work:** this is small. A 125-candidate scan takes 37 ms with Numba on
  one thread, so parallel scans matter only for much larger scans or many profiles.

### Responsiveness

A heartbeat thread sleeps until its next 5 ms tick, the way an event loop waits, and
records how late it wakes. It ran for 2 s while one worker thread repeated a workload.
The table gives the p99 lateness and, in brackets, the worst value over both rounds.
Idle, with no worker, the floor is p99 1.3 ms. macOS overshoots short sleeps.
Isolated maxima of 10-25 ms also turned up when idle (10.4 ms once), so they come from
the shared machine.

| Worker runs | Backend | 3.13 | 3.14 | 3.14t |
|---|---|---|---|---|
| one 0.25 s explicit kernel call | numba-gil (before) | 242 (242) | 231 (231) | 1.4 (10) |
| one 0.25 s explicit kernel call | numba-nogil (now) | 1.3 (5.5) | 1.3 (1.3) | 1.3 (1.3) |
| `fit_run` of the template, repeated | numba-gil | 2.3 (24) | 2.2 (2.6) | 1.3 (1.6) |
| `fit_run` of the template, repeated | numba-nogil | 1.8 (15) | 1.8 (1.8) | 1.3 (1.3) |
| `fit_run` of the template, repeated | numpy | 8.7 (14) | 8.4 (27) | 1.3 (3.8) |
| 64-candidate batch, 1 thread | numba-gil | 1.7 (1.8) | 1.7 (1.8) | 1.3 (1.3) |
| 64-candidate batch, 1 thread | numba-nogil | 1.4 (1.6) | 5.9 (12) | 1.3 (1.3) |
| 64-candidate batch, 1 thread | numpy | 12.8 (13) | 12.6 (13) | 1.3 (1.4) |

- **A GIL-holding kernel freezes every other thread for its whole call.** In the
  template fit each call is short: `_run_past_best_fit` advances in blocks of a few
  hundred steps, about 1-2 ms. That is why the template fit showed only small delays
  even before this change. A long run (a fine grid, `stop_early=False`, many substeps
  per row) makes one long call, and the event loop then stops for its full length.
  With nogil the delay is gone at any call length.
- **The NumPy fallback** holds the GIL in Python code, which hands it over every 5 ms
  (`sys.getswitchinterval()`). The event loop is then late by 8-15 ms at p99. That is
  noticeable but usable.

## Part 2: interpreters and packages

### Installs (macOS arm64, 2026-10-15)

| | 3.13 | 3.14 | 3.15.0b3 | 3.14t | 3.15.0b3t |
|---|---|---|---|---|---|
| numpy, scipy, pandas, matplotlib, openpyxl | wheel | wheel | wheel | wheel | wheel |
| numba | wheel | wheel | wheel | wheel | source build |
| pykrige | wheel | wheel | source build (1) | source build | source build |
| PySide6 | not tried | not tried | not tried | none (2) | none (2) |
| nicegui | not tried | not tried | not tried | without orjson (3) | without orjson (3) |
| `import forster` works | yes | yes | no: segfault | yes | no: segfault |

1. pykrige's build dependency pentapy segfaulted while building with Cython's compiled
   wheel. It built once Cython was installed from source (`--no-binary cython`).
2. PySide6 6.12 ships only `abi3` wheels, and free-threaded builds cannot load `abi3`.
3. orjson 3.13 refuses to build for free-threaded Python ("does not support
   free-threaded Python"). nicegui falls back to the json module without it.
   - Installed with a uv override that drops orjson, nicegui 3.18 imports on 3.14t
     with the GIL still disabled.
   - It served a page with FORSTER imported. 2,503 page fetches made while a worker
     thread ran 50 fits all succeeded.

### GIL after import on free-threaded builds

Each import was run in a fresh process, and `sys._is_gil_enabled()` read afterwards.

- **3.14t:** numpy, numpy.random, scipy.linalg, pandas, matplotlib.pyplot, openpyxl,
  pykrige, numba, `forster`, `forster._kernels`, nicegui and `forster._kernels`
  together with nicegui all left the GIL **disabled**. No extension re-enabled it.
- **3.15t:** numpy core, openpyxl and nicegui left it disabled. numpy.random, SciPy,
  pandas, matplotlib, pykrige, numba and FORSTER crashed with SIGSEGV.
  - The crash is in Cython's generated `__pyx_f_5numpy_import_array` (faulthandler C
    stack). The default 3.15.0b3 build shows the same crash, and so do the numpy 2.6,
    scipy 2.0 and pandas 3.2 nightly wheels.
  - So 3.15.0b3 is unusable for this stack until Cython-built wheels catch up.

### Timings (ms, median; Numba nogil kernels unless stated)

| | 3.13 | 3.14 | 3.14 + JIT | 3.14t | 3.15 / 3.15 + JIT / 3.15t |
|---|---|---|---|---|---|
| `import forster._kernels` (s) | 0.59 | 0.56 | 0.57 | 0.61 | crashes |
| first `fit_run` (Numba cache warm) | 94 | 98 | 100 | 84-89 | |
| `fit_run`, Numba | 14.3 | 14.5 | 14.5 | 14.1-15.0 | |
| `fit_run`, NumPy | 106.8 | 101.4 | 101.7 | 112-117 | |
| 125-candidate scan, Numba | 37.5 | 38.6 | 37.6 | 37.1-38.0 | |
| 125-candidate scan, NumPy | 757 | 725 | 726 | 835-872 | |
| 64-candidate batch, 8 threads, Numba | 11.0 | 10.8 | 11.0 | 4.0-4.8 | |
| heartbeat p99 during a 0.25 s kernel | 1.3 | 1.3 | 1.3 | 1.3 | |

- **Compile cost:** compiling the kernels from scratch, which is what the uncached
  GIL-holding copy does on every run, takes 1.0 s on first use. With `cache=True` and
  a warm cache, the first fit costs 84-100 ms.
- **JIT confirmed on:** `sys._jit.is_enabled()` was True in the 3.14 + JIT rows. The
  free-threaded builds have no JIT (`is_available()` False).

## What the JIT changes

Nothing measurable for FORSTER on 3.14:

- **Numba path:** the time is in compiled kernels and NumPy calls, which CPython's JIT
  does not touch.
- **NumPy path:** a fit runs in about 100 ms of small-array NumPy calls, with little
  pure-Python looping for the JIT to speed up. 3.14 with the JIT is within 0.3% of 3.14
  without it.

3.15's JIT could not be tried with FORSTER (the import crash above). Nothing here
suggests it would matter, because no hot loop in FORSTER is interpreted Python once
Numba is installed.

## Is free threading viable today?

For computation on 3.14t, yes:

- every dependency keeps the GIL disabled;
- the test suite passes, with the NumPy 2.5 shim;
- batches scale 3.6-4.1x on 8 threads, where standard CPython reaches 1.7x.

What blocks it:

- **PySide6:** no free-threaded wheels, so a Qt GUI cannot run on 3.14t.
- **pykrige:** no free-threaded wheels for 3.14t or 3.15t, so installing FORSTER needs
  a C toolchain. FORSTER needs pykrige only for the legacy kriging uncertainty, but
  `forster.engine` imports it at module load.
- **orjson**, for nicegui: it works around this by falling back to the json module.
- **3.15t:** every Cython-built wheel crashes on 3.15.0b3.
- **Cost:** NumPy-path work is 11-15% slower on one thread. On the Numba path the
  difference is within the noise.

## Recommendation for the GUI worker

1. **Run fits in a worker thread** (`QThread` or `concurrent.futures`; in nicegui,
   `run.io_bound` or `asyncio.to_thread`) **on standard CPython 3.13 or 3.14, with
   Numba installed.** With the nogil kernels the event loop is never late by more
   than the idle floor during kernel work. FORSTER's Python parts take the GIL in
   slices of 5 ms or less, so a 16 ms frame is not missed. The template fit takes
   14 ms and a 125-candidate scan 38 ms, so the worker is idle most of the time.
2. **Do not use a subprocess for responsiveness.** A fresh process pays about 0.6 s to
   import FORSTER plus about 0.1 s for the first fit before its first result. A
   subprocess is worth it only to kill a runaway fit: `fit_run` has no cancellation
   hook, while `scan_run` has a `progress` callback that can raise. Even then, a single
   long-lived worker process beats one per fit.
3. **Without Numba** (the NumPy fallback) a thread still works: the event loop is late
   by 8-15 ms at p99. Keep `numba` in the GUI's install (`forster[fast]`); the fits are
   also 7x faster with it.
4. **Free-threaded Python is not worth it for the GUI yet.** The work is too small to
   need 4-8 cores, PySide6 does not exist for it, and pykrige would need a compiler.
   Revisit when PySide6 and pykrige ship free-threaded wheels. The same thread-based
   worker design then gains multi-core scans with no change.
5. **Pick the toolkit on its merits, not on threading.** Both PySide6 and nicegui work
   with a worker thread on standard CPython. Only nicegui runs on 3.14t today, and only
   without orjson.

## Found on the way: NumPy 2.5 breaks Euler-angle tables

- **The breakage:** NumPy 2.5 turned `float()` of an array with ndim > 0 from a
  DeprecationWarning (deprecated since 1.25) into a TypeError.
  `forster.orientation.vector_direction` calls `float(min(Shortest_angle(...), ...))`,
  and `Shortest_angle` returns a 1x1 `np.matrix`. So `read_runs` fails on any row
  given as Euler angles, which includes the packaged template, and so does
  `forster template` followed by `forster fit`.
- **Who it reaches:** `pyproject.toml` asks for `numpy>=2.3.3`, so a fresh
  `pip install` today gets 2.5.4 and is broken on every Python version.
- **Test results:**
  - with numpy 2.5.4 on 3.13, 30 tests fail and 3 error, all from this line;
  - with a one-line shim (`Shortest_angle` returning `np.asarray(x).reshape(())`),
    all 239 pass on 3.13 and on 3.14t;
  - the shim gives the same orientation as NumPy 2.3 to the last bit (56.98977666896446,
    76.75037863975008, 36.2306213045472 degrees).
- **The fix (not made here, since `src/forster/orientation.py` was out of scope):**
  `float(np.asarray(x).item())`, or index `[0, 0]`, in `vector_direction`, plus CI
  against the newest NumPy. `bench.py` applies the shim only when NumPy needs it and
  records that it did (`numpy25_orientation_shim`).

## Reproducing

```
uv run python analyses/python_versions/bench.py matrix --venv-root <dir outside the repo> \
    --max-load 10 --max-wait 3 --round 1
```

`--targets` picks interpreters, and `--round 2` adds a second round without replacing
the first. Query `results.csv` with DuckDB, for example:

```
uv run --no-project --with duckdb python -c "import duckdb; print(duckdb.sql('''
  select target, backend, threads, value from 'analyses/python_versions/results.csv'
  where metric = 'batch64' order by all'''))"
```
