"""All real profiles: chosen final grid, explicit (central) vs implicit; time per fit and fitted times."""
import sys, time, warnings
from dataclasses import replace
import pandas as pd
sys.path.insert(0, "tests"); warnings.simplefilter("ignore")
from conftest import PAPER_DATA, PAPER_PARAMS, ROUND_ROBIN_DATA, ROUND_ROBIN_PARAMS

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _private import write_rows  # noqa: E402  (round robin rows to the submodule)
from forster import RunDefaults, fit_run, read_profiles, read_runs
from forster.grid import GridRequest

sets = [(read_runs(ROUND_ROBIN_PARAMS, defaults=RunDefaults(max_time_days=730)), read_profiles(ROUND_ROBIN_DATA)),
        (read_runs(PAPER_PARAMS), read_profiles(PAPER_DATA, sheet="Olivine"))]
rows = []
for runs, profiles in sets:
    for run in runs:
        out = dict(run=run.name)
        for scheme in ("central", "implicit"):
            r = replace(run, grid=GridRequest(), scheme=scheme)
            t0 = time.perf_counter()
            try:
                res = fit_run(r, profiles)
            except Exception as e:  # report, keep going
                out[f"{scheme}_error"] = repr(e)[:120]
                continue
            out[f"{scheme}_s"] = time.perf_counter() - t0
            out[f"{scheme}_days"] = res.time_days
            out[f"{scheme}_low"] = res.time_days + res.time_minus_s / 86400
            out[f"{scheme}_high"] = res.time_days + res.time_plus_s / 86400
            out[f"{scheme}_nodes"] = len(res.model_x_um)
            out[f"{scheme}_rows"] = len(res.curves)
            if scheme == "implicit":
                g = res.inputs["grid"]
                ref = fit_run(replace(r, grid=replace(g, dx_um=g.dx_um / 3, dt_s=g.dt_s / 2)), profiles)
                out["reference_days"] = ref.time_days
        rows.append(out); print(out, flush=True)
write_rows(rows, Path(__file__).stem + "2.csv")
