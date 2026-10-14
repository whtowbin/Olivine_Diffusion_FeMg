"""End-to-end timings; run once with FORSTER_NO_NUMBA=1 and once without."""
import sys, time, warnings
t_start = time.perf_counter()
from dataclasses import replace
sys.path.insert(0, "tests"); warnings.simplefilter("ignore")
from conftest import PAPER_DATA, PAPER_PARAMS, ROUND_ROBIN_DATA, ROUND_ROBIN_PARAMS
from forster import RunDefaults, compiled, fit_run, read_profiles, read_runs, scan_run
from forster.grid import GridRequest
from forster.scan import Uncertainty
t_import = time.perf_counter() - t_start
sets = [(read_runs(ROUND_ROBIN_PARAMS, defaults=RunDefaults(max_time_days=730)), read_profiles(ROUND_ROBIN_DATA)),
        (read_runs(PAPER_PARAMS), read_profiles(PAPER_DATA, sheet="Olivine"))]
times = {}
t = time.perf_counter()
first = None
for runs, profiles in sets:
    for run in runs:
        t1 = time.perf_counter(); fit_run(replace(run, grid=GridRequest()), profiles)
        if first is None: first = time.perf_counter() - t1
times["35 default fits (chosen grid)"] = time.perf_counter() - t
t = time.perf_counter()
for runs, profiles in sets:
    for run in runs:
        fit_run(run, profiles)
times["35 fits on the tables' own grids"] = time.perf_counter() - t
runs, profiles = sets[0]
run = replace({r.name: r for r in runs}["KS20-527-8_transect-2"], grid=GridRequest(),
              uncertainties={"step1_um": Uncertainty(2.0), "step2_um": Uncertainty(2.0), "edge_fo": Uncertainty(0.003, "uniform")})
t = time.perf_counter(); scan = scan_run(run, profiles, time_days=70.0)
times[f"scan, {len(scan.runs.candidates)} candidates"] = time.perf_counter() - t
print("scan result", round(scan.time_days, 6), [round(v, 6) for v in scan.profile_days], [round(v, 6) for v in scan.marginal_days])
print(f"numba={compiled.AVAILABLE} import {t_import:.2f} s; first fit {first:.2f} s")
for k, v in times.items(): print(f"  {k}: {v:.3f} s")
