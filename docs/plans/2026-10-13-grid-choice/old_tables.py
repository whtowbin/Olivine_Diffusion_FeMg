"""The existing parameter tables, with their own grids: central (old default) vs implicit (new)."""
import sys, time, warnings
from dataclasses import replace
import pandas as pd
sys.path.insert(0, "tests"); warnings.simplefilter("ignore")
from conftest import PAPER_DATA, PAPER_PARAMS, ROUND_ROBIN_DATA, ROUND_ROBIN_PARAMS
from forster import RunDefaults, fit_run, read_profiles, read_runs
sets = [(read_runs(ROUND_ROBIN_PARAMS, defaults=RunDefaults(max_time_days=730)), read_profiles(ROUND_ROBIN_DATA)),
        (read_runs(PAPER_PARAMS), read_profiles(PAPER_DATA, sheet="Olivine"))]
ref = pd.read_csv(__file__.replace("old_tables.py", "implicit_sweep2.csv")).set_index("run").reference_days
rows = []
for runs, profiles in sets:
    for run in runs:
        out = dict(run=run.name, dx_um=run.grid.dx_um, dt_s=run.grid.dt_s, reference_days=ref.get(run.name))
        for scheme in ("central", "implicit"):
            t0 = time.perf_counter()
            res = fit_run(replace(run, scheme=scheme), profiles)
            out[f"{scheme}_days"], out[f"{scheme}_s"] = res.time_days, time.perf_counter() - t0
        rows.append(out); print(out, flush=True)
pd.DataFrame(rows).to_csv(__file__.replace(".py", ".csv"), index=False)
