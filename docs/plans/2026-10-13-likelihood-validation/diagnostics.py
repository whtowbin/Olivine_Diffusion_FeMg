"""Noise-model diagnostics on every real profile (17 round robin, 18 Towbin and Plank 2026).

For each run fitted with the default (correlated) method:

- the fitted noise model and its correlation length L against the point spacing,
  the profile span and the diffusion length sqrt(D t) at the best fit;
- whether the noise-model grid-edge warning fired;
- residual whiteness before and after whitening: lag-1 autocorrelation of the raw
  residuals r and of z = chol(C)^-1 r, and a Kolmogorov-Smirnov test of z
  against N(0, 1) (if the error model is adequate, z looks like white noise);
- best fit and 1-sigma range, against the white-noise (rho = 0) best fit.

Writes real_profiles.csv next to this file. Run from the repo root:
    uv run python docs/plans/2026-10-13-likelihood-validation/diagnostics.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "tests"))
from conftest import (  # noqa: E402
    PAPER_DATA,
    PAPER_PARAMS,
    ROUND_ROBIN_DATA,
    ROUND_ROBIN_PARAMS,
)

from forster import RunDefaults, fit_run, read_profiles, read_runs  # noqa: E402
from forster.uncertainty import chi2_per_step, interpolation_matrix  # noqa: E402

DAY = 86400.0
SETS = {
    # The round robin batch modelled up to 2 years (several fits pass 200 days).
    "round robin": (
        read_runs(ROUND_ROBIN_PARAMS, defaults=RunDefaults(max_time_days=730)),
        read_profiles(ROUND_ROBIN_DATA),
    ),
    "Towbin and Plank 2026": (
        read_runs(PAPER_PARAMS),
        read_profiles(PAPER_DATA, sheet="Olivine"),
    ),
}


def lag1(v):
    return float(np.corrcoef(v[:-1], v[1:])[0, 1])


rows = []
for label, (runs, profiles) in SETS.items():
    for run in runs:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            r = fit_run(run, profiles)
        notes = sorted({str(w.message)[:70] for w in caught if issubclass(w.category, UserWarning)})
        x, y, noise = r.fit_x_um, r.fit_fo, r.noise
        W = interpolation_matrix(r.model_x_um, x)
        best = r.indices[0]
        resid = y - W @ r.curves[best]
        chol = np.linalg.cholesky(noise.covariance(x))
        z = np.linalg.solve(chol, resid)
        white_best = int(np.argmin(chi2_per_step(r.curves, W, y)))
        D, _ = run.diffusivity.functions(run.conditions, run.orientation)
        d_mid = float(D(np.array([np.mean(y)]))[0])
        diff_len = np.sqrt(d_mid * max(r.time_s, r.dt_s)) * 1e6
        spacing = float(np.median(np.diff(x)))
        rows.append(
            {
                "set": label,
                "run": run.name,
                "n_points": len(x),
                "spacing_um": spacing,
                "span_um": float(x.max() - x.min()),
                "time_days": r.time_days,
                "low_days": r.time_days + r.time_minus_days,
                "high_days": r.time_days + r.time_plus_days,
                "white_time_days": white_best * r.dt_s / DAY,
                "white_sigma_fo": noise.white_sigma,
                "corr_sigma_fo": noise.correlated_sigma,
                "L_um": noise.correlation_length_um,
                "L_over_spacing": noise.correlation_length_um / spacing,
                "diffusion_length_um": diff_len,
                "L_over_diffusion_length": noise.correlation_length_um / diff_len,
                "lag1_raw": lag1(resid),
                "lag1_whitened": lag1(z),
                "ks_p_whitened": float(stats.kstest(z, "norm").pvalue),
                "warnings": "; ".join(notes),
            }
        )
        print(f"{label:22s} {run.name:40s} L={noise.correlation_length_um:6.1f} um", flush=True)
        del r

table = pd.DataFrame(rows)
table.to_csv(HERE / "real_profiles.csv", index=False, float_format="%.4g")
pd.set_option("display.width", 250)
print(table.drop(columns="warnings").round(3).to_string(index=False))
