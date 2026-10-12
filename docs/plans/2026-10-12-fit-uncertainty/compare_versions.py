"""Fit the 17 round robin profiles under the three time-uncertainty rules.

v1: engine.Best_fit_Chi2_old (until March 2026), kriged grid, chi2 band at
    min + sqrt(2(N-1)).
v2: engine.Best_fit_Chi2 = fit_profile(uncertainty="legacy") (March to October
    2026), kriged grid, reduced chi2 within min + 1.
v3: fit_profile(uncertainty="correlated"), the default from 2026-10-12.

Writes round_robin_versions.csv next to this file. Times in days; *_minus and
*_plus are offsets from the best fit. Run from the repo root:
    uv run python docs/plans/2026-10-12-fit-uncertainty/compare_versions.py
"""

import sys
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "tests"))
from conftest import ROUND_ROBIN_DATA, ROUND_ROBIN_PARAMS  # noqa: E402
from test_differential import BATCH_DEFAULTS  # noqa: E402  (pins uncertainty="legacy")

from forster import engine, fit_run, read_profiles, read_runs  # noqa: E402

DAY = 86400.0
profiles = read_profiles(ROUND_ROBIN_DATA)
rows = []
for run in read_runs(ROUND_ROBIN_PARAMS, defaults=BATCH_DEFAULTS):
    legacy = fit_run(run, profiles)
    n_grid = len(legacy.model_x_um)
    lo = int(np.searchsorted(legacy.interp_x_um, legacy.model_x_um[0]))
    sigma = legacy.interp_fo_sigma[lo : lo + n_grid]
    t1, _, _ = engine.Best_fit_Chi2_old(legacy.curves, legacy.fit_fo, sigma, legacy.dt_s)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        new = fit_run(replace(run, uncertainty="correlated"), profiles)
    rows.append(
        {
            "run": run.name,
            "n_points": len(new.fit_x_um),
            "N_grid": n_grid,
            "v1_best": t1[0] / DAY,
            "v1_minus": t1[1] / DAY,
            "v1_plus": t1[2] / DAY,
            "v2_best": legacy.time_days,
            "v2_minus": legacy.time_minus_days,
            "v2_plus": legacy.time_plus_days,
            "v3_best": new.time_days,
            "v3_minus": new.time_minus_days,
            "v3_plus": new.time_plus_days,
            "white_Fo": new.noise.white_sigma * 100,
            "corr_Fo": new.noise.correlated_sigma * 100,
            "L_um": new.noise.correlation_length_um,
            "warnings": "; ".join(str(w.message)[:60] for w in caught),
        }
    )
    del legacy, new  # each run holds a few hundred MB of curves
    print(run.name, flush=True)
pd.DataFrame(rows).to_csv(HERE / "round_robin_versions.csv", index=False, float_format="%.4g")
