"""Validate the correlated-noise likelihood across sampling densities and noise types.

Synthetic profiles are made from the forward model at a known time on two real
round robin geometries, with known noise added, and fitted with
`forster.uncertainty.fit_correlated` (the package default). For each case:

- coverage: how often the 1-sigma range contains the true time (target 68%);
- width of the range as a share of the true time;
- the fitted noise model (white sigma, correlated sigma, correlation length L)
  against the true one;
- how often the noise-model warning fires.

Three interval rules are compared on the same correlated chi-squared curve: the
default (rise by `interval_threshold`, the Student-t small-sample correction), a rise
of exactly 1 (the rule before 2026-10-13), and the Bayesian posterior with the noise
scale integrated out (`posterior_interval`). A "white" baseline is also computed:
independent errors with sigma fitted to the residuals (rho = 0), delta chi2 = 1.

For the two systematic initial-condition errors, the same data are also fitted with
`forster.scan` over 13 candidates (step position -6..+6 um, or edge Fo -0.006..+0.006),
giving the profile-likelihood and marginal ranges.

Writes coverage.csv and trials.csv next to this file. Run from the repo root:
    uv run python docs/plans/2026-10-13-likelihood-validation/validate.py [trials]
"""

from __future__ import annotations

import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "tests"))
from conftest import ROUND_ROBIN_DATA, ROUND_ROBIN_PARAMS  # noqa: E402

from forster import fit_run, read_profiles, read_runs  # noqa: E402
from forster.model import Profile  # noqa: E402
from forster.scan import fit_candidates, run_candidates, step_offsets  # noqa: E402
from forster.uncertainty import (  # noqa: E402
    NoiseModel,
    chi2_per_step,
    delta_chi2_interval,
    fit_correlated,
    interpolation_matrix,
    posterior_interval,
)

SUMMARISE_ONLY = len(sys.argv) > 1 and sys.argv[1] == "summarise"  # reuse trials.csv
TRIALS = int(sys.argv[1]) if len(sys.argv) > 1 and not SUMMARISE_ONLY else 200
SEED = 20261013

runs = {r.name: r for r in read_runs(ROUND_ROBIN_PARAMS)}
profiles = read_profiles(ROUND_ROBIN_DATA)


def model(run, initial=None):
    """Curves (time steps 0..3 x the measured best fit), grid, best row, measured x."""
    if initial is not None:
        run = replace(run, initial=initial)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = fit_run(run, profiles)
    return r


def white_interval(curves, W, y):
    """Independent errors, sigma fitted at the best fit, delta chi2 = 1."""
    rss = chi2_per_step(curves, W, y)
    s2 = rss.min() / len(y)
    return delta_chi2_interval(rss / s2)


# Geometries: a profile's model, the positions sampled, the white noise level.
ks = runs["KS20-527_2-transect-1"]
ks_fit = model(ks)
k_true = ks_fit.indices[0]
ks_curves = ks_fit.curves[: 3 * k_true].copy()
ks_x_meas = ks_fit.fit_x_um
sh = runs["SH63olv066-1"]
sh_fit = model(sh)
sh_k = sh_fit.indices[0]
sh_curves = sh_fit.curves[: 3 * sh_k].copy()

GEOMETRIES = {
    # name: (curves, model_x, x positions, k_true, white sigma Fo fraction)
    "KS 6 pts (50 um)": (ks_curves, ks_fit.model_x_um, ks_x_meas[::5], k_true, 0.002),
    "KS 10 pts (30 um)": (ks_curves, ks_fit.model_x_um, ks_x_meas[::3], k_true, 0.002),
    "KS 29 pts (10 um, measured)": (ks_curves, ks_fit.model_x_um, ks_x_meas, k_true, 0.002),
    "KS 100 pts (3 um)": (ks_curves, ks_fit.model_x_um, np.arange(1.5, 299, 3.0), k_true, 0.004),
    "KS 300 pts (1 um)": (ks_curves, ks_fit.model_x_um, np.arange(0.5, 299.5, 1.0), k_true, 0.006),
    "SH63 298 pts (0.5 um, measured)": (
        sh_curves,
        sh_fit.model_x_um,
        sh_fit.fit_x_um,
        sh_k,
        0.006,
    ),
}

# Noise: correlated misfit amplitude (Fo fraction) and length (um); 0 = white only.
NOISE = {
    "white only": (0.0, 1.0),
    "smooth misfit, L = 3 um": (0.004, 3.0),
    "smooth misfit, L = 10 um": (0.004, 10.0),
    "smooth misfit, L = 30 um": (0.004, 30.0),
}

# Systematic initial-condition errors on the KS profile: the data come from a model
# with a shifted step or edge, the fit uses the nominal one.
ks_init = ks.initial
SYSTEMATIC = {
    "step shifted +3 um": replace(ks_init, steps_um=(ks_init.steps_um[0] + 3,)),
    "edge Fo +0.003": replace(ks_init, edge_fo=ks_init.edge_fo + 0.003),
}
SCAN_CANDIDATES = {
    "step shifted +3 um": step_offsets(ks_init, range(-6, 7)),
    "edge Fo +0.003": [
        replace(ks_init, edge_fo=ks_init.edge_fo + d / 1000) if d else ks_init for d in range(-6, 7)
    ],
}


def run_case(geometry, noise_name, setup=None, scan_runs=None):
    """``setup`` overrides the geometry's (curves, model_x, truth): for the systematic cases
    everything comes from the scan's candidate runs, so the truth, the nominal fit and the
    scan share one model grid (the grid depends on the farthest point of the profile)."""
    curves, model_x, x, k, white = GEOMETRIES[geometry]
    if setup is None:
        truth = interpolation_matrix(model_x, x) @ curves[k]
    else:
        curves, model_x, truth = setup
    W = interpolation_matrix(model_x, x)
    amp, length = NOISE.get(noise_name, (0.0, 1.0))
    true_noise = NoiseModel(white, amp, length)
    chol = np.linalg.cholesky(true_noise.covariance(x))
    rng = np.random.default_rng(SEED)
    rows = []
    for trial in range(TRIALS):
        y = truth + chol @ rng.standard_normal(len(x))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = fit_correlated(curves, model_x, x, y)
        edge = any("spans the profile" in str(w.message) for w in caught)
        span, wspan = fit.interval, white_interval(curves, W, y)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            one = delta_chi2_interval(fit.chi2, 1.0, end_is_limit=False)
            post = posterior_interval(fit.chi2, len(x))
        scan = None
        if scan_runs is not None:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scan = fit_candidates(replace(scan_runs, fo=y))
        rows.append(
            {
                "geometry": geometry,
                "noise": noise_name,
                "trial": trial,
                "n_points": len(x),
                "spacing_um": float(np.median(np.diff(x))),
                "true_white": white,
                "true_corr": amp,
                "true_L_um": length if amp > 0 else np.nan,
                "k_true": k,
                "best": span.best,
                "low": span.low,
                "high": span.high,
                "covered": span.low <= k <= span.high,
                "white_best": wspan.best,
                "white_low": wspan.low,
                "white_high": wspan.high,
                "white_covered": wspan.low <= k <= wspan.high,
                "one_low": one.low,
                "one_high": one.high,
                "one_covered": one.low <= k <= one.high,
                "post_low": post.low,
                "post_high": post.high,
                "post_covered": post.low <= k <= post.high,
                "scan_best_row": scan.best_row if scan else np.nan,
                "scan_best_step_um": scan.best_initial.steps_um[0] if scan else np.nan,
                "scan_profile_low": scan.profile.low if scan else np.nan,
                "scan_profile_high": scan.profile.high if scan else np.nan,
                "scan_marginal_low": scan.marginal.low if scan else np.nan,
                "scan_marginal_high": scan.marginal.high if scan else np.nan,
                "scan_profile_covered": (
                    scan.profile.low <= k <= scan.profile.high if scan else np.nan
                ),
                "scan_marginal_covered": (
                    scan.marginal.low <= k <= scan.marginal.high if scan else np.nan
                ),
                "scan_profile_width": (
                    (scan.profile.high - scan.profile.low) / k if scan else np.nan
                ),
                "scan_marginal_width": (
                    (scan.marginal.high - scan.marginal.low) / k if scan else np.nan
                ),
                "fit_white": fit.noise.white_sigma,
                "fit_corr": fit.noise.correlated_sigma,
                "fit_L_um": fit.noise.correlation_length_um,
                "edge_warning": edge,
            }
        )
    return rows


rows = []
t0 = time.perf_counter()
if SUMMARISE_ONLY:
    GEOMETRIES, SYSTEMATIC = {}, {}  # skip the fits; read trials.csv below
for geometry in GEOMETRIES:
    for noise_name in NOISE:
        rows += run_case(geometry, noise_name)
        print(f"{time.perf_counter() - t0:7.1f}s  {geometry} / {noise_name}", flush=True)
scan_grid = replace(ks.grid, max_time_days=3 * k_true * ks.grid.dt_s / 86400)
for name, shifted_initial in SYSTEMATIC.items():
    candidates = SCAN_CANDIDATES[name]
    i_nominal = candidates.index(ks_init)
    i_truth = next(
        i
        for i, c in enumerate(candidates)
        if np.allclose(c.steps_um, shifted_initial.steps_um)
        and np.isclose(c.edge_fo, shifted_initial.edge_fo)
    )
    for geometry in ("KS 29 pts (10 um, measured)", "KS 300 pts (1 um)"):
        x = GEOMETRIES[geometry][2]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scan_runs = run_candidates(
                Profile(x, np.full(len(x), 0.85)),
                ks.conditions,
                ks.orientation,
                candidates,
                scan_grid,
                ks.diffusivity,
                stop_early=False,
            )
        # Model at the points is the "curve" on a grid of the points themselves.
        setup = (
            scan_runs.at_points[i_nominal],
            scan_runs.x_um,
            scan_runs.at_points[i_truth][k_true],
        )
        rows += run_case(geometry, name, setup=setup, scan_runs=scan_runs)
        print(f"{time.perf_counter() - t0:7.1f}s  {geometry} / {name}", flush=True)

if SUMMARISE_ONLY:
    trials = pd.read_csv(HERE / "trials.csv")
else:
    trials = pd.DataFrame(rows)
    trials.to_csv(HERE / "trials.csv", index=False)


def summarise(g):
    k = g.k_true.iloc[0]
    return pd.Series(
        {
            "n_points": g.n_points.iloc[0],
            "spacing_um": g.spacing_um.iloc[0],
            "coverage": g.covered.mean(),
            "white_coverage": g.white_covered.mean(),
            "delta1_coverage": g.one_covered.mean(),
            "posterior_coverage": g.post_covered.mean(),
            # astype(float): object columns of numpy bools would "sum" as a logical OR.
            "scan_profile_coverage": g.scan_profile_covered.astype(float).mean(),
            "scan_marginal_coverage": g.scan_marginal_covered.astype(float).mean(),
            "median_width_pct": 100 * ((g.high - g.low) / k).median(),
            "white_median_width_pct": 100 * ((g.white_high - g.white_low) / k).median(),
            "delta1_median_width_pct": 100 * ((g.one_high - g.one_low) / k).median(),
            "posterior_median_width_pct": 100 * ((g.post_high - g.post_low) / k).median(),
            "scan_profile_median_width_pct": 100 * g.scan_profile_width.median(),
            "scan_marginal_median_width_pct": 100 * g.scan_marginal_width.median(),
            "median_bias_pct": 100 * ((g.best - k) / k).median(),
            "median_fit_L_um": g.fit_L_um.where(g.fit_corr > 0).median(),
            "share_with_correlation": (g.fit_corr > 0).mean(),
            "median_fit_white": g.fit_white.median(),
            "median_fit_corr": g.fit_corr.median(),
            "edge_warning_share": g.edge_warning.mean(),
            "trials": len(g),
        }
    )


coverage = (
    trials.groupby(["geometry", "noise"], sort=False)
    .apply(summarise, include_groups=False)
    .reset_index()
)
coverage.to_csv(HERE / "coverage.csv", index=False, float_format="%.4g")
pd.set_option("display.width", 220)
print(coverage.round(3).to_string(index=False))
