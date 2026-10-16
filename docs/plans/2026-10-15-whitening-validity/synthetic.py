"""Part B: coverage of forster's default range when the noise model is wrong.

Profiles of known age are made with the forward model (implicit scheme, grid from
`forster.grid.choose_grid`) on three real geometries (point positions only):

- "KS 29 pts": KS20-527_2-transect-1, 29 points 10 um apart (round robin);
- "paper 17 pts": GCB1A R31_MP_Ol5_prof_2, 17 points 2.9 um apart (Towbin and Plank 2026);
- "SH63 298 pts": SH63olv066-1, 298 points 0.5 um apart (round robin, two steps).

The true time is the fit to that profile's data on the chosen grid. Noise of a known
kind is added (cases below) and each synthetic profile is fitted with the unchanged
default method, `forster.uncertainty.fit_correlated_points` on the model at the points
for every time step (what `fit_profile` does, without early stopping, which does not
change the fit). Recorded per trial:

- default: best fit, 1-sigma and 2-sigma ranges (rise by `interval_threshold`), fitted
  noise model, the profile-wide-misfit warning, and whiteness diagnostics of the
  whitened residuals at the best fit;
- baseline: independent errors, sigma fitted at the best fit (rho = 0), delta chi2 = 1
  (1 sigma) and 4 (2 sigma);
- two candidate changes, computed on the same data for comparison only (not forster):
  "kernel by AIC" picks the noise kernel (white, squared-exponential, exponential,
  Matern 3/2) with the lowest AIC at the white best fit, then proceeds as the default;
  "outlier clip" refits once without points whose standardised leave-one-out residual
  under the fitted noise model exceeds 4.

Usage (repo root):
    uv run python docs/plans/2026-10-15-whitening-validity/synthetic.py measure
    uv run python docs/plans/2026-10-15-whitening-validity/synthetic.py run [trials]
    uv run python docs/plans/2026-10-15-whitening-validity/synthetic.py summarise

At most 4 worker processes, one BLAS thread each. `run` appends to trials.csv per
chunk and skips chunks already there, so it can be resumed.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.linalg as sla  # noqa: E402
from common import HERE, fit_kernel, kernel_matrix, whiten, whiteness  # noqa: E402
from scipy import stats  # noqa: E402

CACHE = Path(os.environ.get("WHITENING_CACHE", "/tmp")) / "whitening_geometries"
TRIALS_CSV = HERE / "trials.csv"
COVERAGE_CSV = HERE / "coverage.csv"
SEED = 20261015
WORKERS = 4
CHUNK = 25  # trials per task
TIME_FACTOR = 3  # model up to 3 x the true time, as the 2026-10-13 validation
AMP = 0.004  # smooth misfit amplitude, Fo fraction (0.4 mol% Fo)
CLIP_Z = 4.0

GEOMETRIES = {
    # name: (source, run name, white noise sd as Fo fraction)
    "KS 29 pts (10 um)": ("round_robin", "KS20-527_2-transect-1", 0.002),
    "paper 17 pts (2.9 um)": ("paper", "GCB1A R31_MP_Ol5_prof_2", 0.001),
    "SH63 298 pts (0.5 um)": ("round_robin", "SH63olv066-1", 0.006),
}

# Case name -> (generator kind, parameters). "misfit" draws a stationary Gaussian
# process of the given kernel and length with sd AMP; "envelope" multiplies such a
# misfit by a Gaussian bump of sd one diffusion length around each step (peak 1).
CASES = {
    "white (control)": ("white", {}),
    "sq-exp L=3 (control)": ("misfit", {"kernel": "sqexp", "L": 3.0}),
    "sq-exp L=10 (control)": ("misfit", {"kernel": "sqexp", "L": 10.0}),
    "sq-exp L=30 (control)": ("misfit", {"kernel": "sqexp", "L": 30.0}),
    "exponential L=3": ("misfit", {"kernel": "exp", "L": 3.0}),
    "exponential L=10": ("misfit", {"kernel": "exp", "L": 10.0}),
    "exponential L=30": ("misfit", {"kernel": "exp", "L": 30.0}),
    "Matern 3/2 L=3": ("misfit", {"kernel": "matern32", "L": 3.0}),
    "Matern 3/2 L=10": ("misfit", {"kernel": "matern32", "L": 10.0}),
    "Matern 3/2 L=30": ("misfit", {"kernel": "matern32", "L": 30.0}),
    "near-step sq-exp L=3": ("envelope", {"kernel": "sqexp", "L": 3.0}),
    "near-step sq-exp L=10": ("envelope", {"kernel": "sqexp", "L": 10.0}),
    # 3x the amplitude (1.2 mol% Fo at the step): near/far scatter like the real SH63 data
    "strong near-step L=3": ("envelope", {"kernel": "sqexp", "L": 3.0, "amp": 3 * AMP}),
    "strong near-step L=10": ("envelope", {"kernel": "sqexp", "L": 10.0, "amp": 3 * AMP}),
    "Student-t3 white": ("student", {}),
    "1-2 outliers (5 sd)": ("outliers", {}),
    "sd grows with |Fo - core|": ("hetero_fo", {}),
    "sd rises 4x to the rim": ("hetero_rim", {}),
}


# ------------------------------------------------------------------------- geometries


def build_geometries():
    """Model at the points for every time step, per geometry; cached as .npz."""
    from references import load

    import forster
    from forster.grid import GridRequest
    from forster.model import Grid
    from forster.uncertainty import interpolation_matrix, points_in_domain

    CACHE.mkdir(parents=True, exist_ok=True)
    data = load()
    for gname, (_, run_name, white) in GEOMETRIES.items():
        path = CACHE / f"{run_name.replace(' ', '_')}.npz"
        if path.exists():
            continue
        run, profiles, _ = data[run_name]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # The true time: the default fit to this profile with its own grid, moved to
            # the nearest step of the chosen grid (implicit scheme, choose_grid at that
            # time). A blank GridRequest would need a coarse fit to estimate the time,
            # which on SH63 exceeds the curve-memory limit, so the time is passed in.
            own = forster.fit_run(run, profiles)
            from forster.io import _profile

            prof = _profile(run, profiles)
            req = GridRequest().resolve(
                prof,
                run.conditions,
                run.orientation,
                run.initial,
                run.diffusivity,
                time_days=own.time_days,
                scheme="implicit",
            )
            dt = req.grid.dt_s
            k_true = int(round(own.time_s / dt))
            grid = Grid(req.grid.dx_um, dt, TIME_FACTOR * k_true * dt / 86400 + dt / 86400)
            res = forster.fit_profile(
                prof,
                run.conditions,
                run.orientation,
                run.initial,
                grid,
                diffusivity=run.diffusivity,
                scheme="implicit",
                stop_early=False,
                keep_curves=True,
            )
        x, _ = points_in_domain(res.model_x_um, res.fit_x_um, res.fit_fo)
        x = np.sort(x)
        at_points = (
            res.curves[: TIME_FACTOR * k_true + 1] @ interpolation_matrix(res.model_x_um, x).T
        )
        D, _ = run.diffusivity.functions(run.conditions, run.orientation)
        truth = at_points[k_true]
        d_mid = float(D(np.array([np.mean(truth)]))[0])
        np.savez(
            path,
            at_points=at_points,
            x=x,
            k_true=k_true,
            dt_s=dt,
            dx_um=req.grid.dx_um,
            white=white,
            steps=np.array(run.initial.steps_um),
            core_fo=run.initial.core_fo,
            diffusion_length_um=np.sqrt(d_mid * k_true * dt) * 1e6,
        )
        print(
            f"{gname}: {len(x)} pts, k_true={k_true}, dt={dt:.4g}s, dx={req.grid.dx_um:.3g}um, "
            f"t={k_true * dt / 86400:.3f} d, rows={len(at_points)}, "
            f"l={np.sqrt(d_mid * k_true * dt) * 1e6:.2f} um",
            flush=True,
        )


_GEO = {}


def geometry(gname):
    if gname not in _GEO:
        run_name = GEOMETRIES[gname][1]
        g = dict(np.load(CACHE / f"{run_name.replace(' ', '_')}.npz"))
        g["k_true"] = int(g["k_true"])
        _GEO[gname] = g
    return _GEO[gname]


# ------------------------------------------------------------------------------ noise


def _gp_chol(kind, x, L):
    return np.linalg.cholesky(kernel_matrix(kind, x, L) + 1e-10 * np.eye(len(x)))


def make_noise(g, case, rng):
    """Synthetic error vector for one trial. The same rng stream order for every case,
    so cases share their white-noise draw for a given trial."""
    x, s = g["x"], float(g["white"])
    n = len(x)
    e_white = rng.standard_normal(n)
    e_corr = rng.standard_normal(n)
    kind, p = CASES[case]
    if kind == "white":
        return s * e_white
    if kind in ("misfit", "envelope"):
        f = p.get("amp", AMP) * _gp_chol(p["kernel"], x, p["L"]) @ e_corr
        if kind == "envelope":
            ell = float(g["diffusion_length_um"])
            env = np.exp(-0.5 * ((x[:, None] - g["steps"][None, :]) / ell) ** 2).max(axis=1)
            f = f * env
        return s * e_white + f
    if kind == "student":
        return s * rng.standard_t(3, n) / np.sqrt(3.0)
    if kind == "outliers":
        k = int(rng.integers(1, 3))
        idx = rng.choice(n, size=k, replace=False)
        e = s * e_white
        e[idx] += 5 * s * rng.choice([-1.0, 1.0], size=k)
        return e
    if kind == "hetero_fo":
        truth = g["at_points"][g["k_true"]]
        w = np.abs(truth - float(g["core_fo"]))
        w = w + 0.1 * w.max()
    elif kind == "hetero_rim":
        w = 1 + 3 * (x.max() - x) / (x.max() - x.min())
    else:
        raise ValueError(kind)
    sd = s * w / np.sqrt(np.mean(w**2))  # rms sd = s
    return sd * e_white


# ------------------------------------------------------------------------------- fits


def _covered(span, k):
    return bool(span.low <= k <= span.high)


def loo_z(cov, r):
    """Standardised leave-one-out residuals of a Gaussian process with covariance cov:
    [C^-1 r]_i / sqrt([C^-1]_ii) (Rasmussen and Williams 2006, eq. 5.12)."""
    ci = sla.cho_solve(sla.cho_factor(cov, lower=True), np.eye(len(r)))
    return (ci @ r) / np.sqrt(np.diag(ci))


def one_trial(gname, case, trial):
    from forster.uncertainty import (
        TWO_SIGMA,
        delta_chi2_interval,
        fit_correlated_points,
        interval_threshold,
        white_rss,
    )

    g = geometry(gname)
    gi = list(GEOMETRIES).index(gname)
    rng = np.random.default_rng([SEED, gi, trial])
    at, x, k = g["at_points"], g["x"], g["k_true"]
    y = at[k] + make_noise(g, case, rng)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = fit_correlated_points(at, x, y, end_is_limit=False)
    edge = any("spans the profile" in str(w.message) for w in caught)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rss = white_rss(at, y)
        s2 = rss.min() / len(y)
        w1 = delta_chi2_interval(rss / s2, 1.0, end_is_limit=False)
        w2 = delta_chi2_interval(rss / s2, 4.0, end_is_limit=False)
    best = fit.interval.best
    r_best = y - at[best]
    cov = fit.noise.covariance(x)
    wh = whiteness(whiten(cov, r_best))
    row = {
        "geometry": gname,
        "case": case,
        "trial": trial,
        "n_points": len(x),
        "k_true": k,
        "best": best,
        "low": fit.interval.low,
        "high": fit.interval.high,
        "covered": _covered(fit.interval, k),
        "low2": fit.interval2.low,
        "high2": fit.interval2.high,
        "covered2": _covered(fit.interval2, k),
        "white_best": w1.best,
        "white_low": w1.low,
        "white_high": w1.high,
        "white_covered": _covered(w1, k),
        "white_low2": w2.low,
        "white_high2": w2.high,
        "white_covered2": _covered(w2, k),
        "fit_white": fit.noise.white_sigma,
        "fit_corr": fit.noise.correlated_sigma,
        "fit_L_um": fit.noise.correlation_length_um,
        "edge_warning": edge,
        **{f"diag_{kk}": v for kk, v in wh.items()},
    }
    # Candidate 1: kernel chosen by AIC at the white best fit (grid search only, as forster).
    r_w = y - at[int(np.argmin(rss))]
    fits = {
        kind: fit_kernel(kind, x, r_w, refine=False)
        for kind in ("white", "sqexp", "exp", "matern32")
    }
    chosen = min(fits, key=lambda kk: fits[kk].aic)
    row["aic_kernel"] = chosen
    row["aic_sqexp_minus_best"] = fits["sqexp"].aic - min(f.aic for f in fits.values())
    if chosen in ("sqexp", "white") and np.isclose(fits[chosen].white_sigma, fit.noise.white_sigma):
        a1, a2 = fit.interval, fit.interval2
    else:
        chol = np.linalg.cholesky(fits[chosen].covariance(x))
        z = sla.solve_triangular(chol, (at - y).T, lower=True)
        chi2 = np.einsum("pk,pk->k", z, z)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a1 = delta_chi2_interval(chi2, interval_threshold(len(x)), end_is_limit=False)
            a2 = delta_chi2_interval(
                chi2, interval_threshold(len(x), level=TWO_SIGMA), end_is_limit=False
            )
    row |= {
        "aic_best": a1.best,
        "aic_low": a1.low,
        "aic_high": a1.high,
        "aic_covered": _covered(a1, k),
        "aic_covered2": _covered(a2, k),
    }
    # Candidate 2: drop points with |standardised LOO residual| > CLIP_Z, refit once.
    lz = np.abs(loo_z(cov, r_best))
    keep = lz <= CLIP_Z
    row["n_clipped"] = int((~keep).sum())
    row["max_loo_z"] = float(lz.max())
    if keep.all():
        c1, c2 = fit.interval, fit.interval2
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cf = fit_correlated_points(at[:, keep], x[keep], y[keep], end_is_limit=False)
        c1, c2 = cf.interval, cf.interval2
    row |= {
        "clip_best": c1.best,
        "clip_low": c1.low,
        "clip_high": c1.high,
        "clip_covered": _covered(c1, k),
        "clip_covered2": _covered(c2, k),
    }
    return row


def run_chunk(gname, case, start, stop):
    return [one_trial(gname, case, t) for t in range(start, stop)]


# --------------------------------------------------------------------------- commands


def measure():
    build_geometries()
    for gname in GEOMETRIES:
        for case in ("white (control)", "exponential L=10", "1-2 outliers (5 sd)"):
            t = time.perf_counter()
            rows = run_chunk(gname, case, 0, 3)
            dt = (time.perf_counter() - t) / 3
            print(
                f"{gname:24s} {case:22s} {dt:6.3f} s/trial  covered={[r['covered'] for r in rows]}",
                flush=True,
            )


def run(trials):
    build_geometries()
    done = set()
    if TRIALS_CSV.exists():
        old = pd.read_csv(TRIALS_CSV, usecols=["geometry", "case", "trial"])
        done = set(zip(old.geometry, old.case, old.trial, strict=True))
    # Slowest geometry first so the pool stays busy at the end.
    order = sorted(GEOMETRIES, key=lambda gname: -len(geometry(gname)["x"]))
    tasks = [
        (gname, case, s, min(s + CHUNK, trials))
        for gname in order
        for case in CASES
        for s in range(0, trials, CHUNK)
        if not all((gname, case, t) in done for t in range(s, min(s + CHUNK, trials)))
    ]
    print(f"{len(tasks)} chunks to run", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(run_chunk, *task): task for task in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            rows = [r for r in fut.result() if (r["geometry"], r["case"], r["trial"]) not in done]
            pd.DataFrame(rows).to_csv(
                TRIALS_CSV, mode="a", header=not TRIALS_CSV.exists(), index=False
            )
            g, c, s, e = futures[fut]
            print(
                f"{time.perf_counter() - t0:7.1f}s [{i}/{len(tasks)}] {g} / {c} {s}-{e}", flush=True
            )
    summarise()


def _band(p, n):
    """Central 95% range of the coverage share for n trials at true probability p."""
    lo, hi = stats.binom.ppf([0.025, 0.975], n, p) / n
    return float(lo), float(hi)


def summarise():
    t = pd.read_csv(TRIALS_CSV).drop_duplicates(["geometry", "case", "trial"])

    def agg(g):
        k = g.k_true.iloc[0]
        n = len(g)
        return pd.Series(
            {
                "trials": n,
                "coverage_1s": g.covered.mean(),
                "white_coverage_1s": g.white_covered.mean(),
                "aic_coverage_1s": g.aic_covered.mean(),
                "clip_coverage_1s": g.clip_covered.mean(),
                "coverage_2s": g.covered2.mean(),
                "white_coverage_2s": g.white_covered2.mean(),
                "aic_coverage_2s": g.aic_covered2.mean(),
                "clip_coverage_2s": g.clip_covered2.mean(),
                "median_bias_pct": 100 * ((g.best - k) / k).median(),
                "white_median_bias_pct": 100 * ((g.white_best - k) / k).median(),
                "median_width_pct": 100 * ((g.high - g.low) / k).median(),
                "white_median_width_pct": 100 * ((g.white_high - g.white_low) / k).median(),
                "aic_median_width_pct": 100 * ((g.aic_high - g.aic_low) / k).median(),
                "clip_median_width_pct": 100 * ((g.clip_high - g.clip_low) / k).median(),
                "median_width2_pct": 100 * ((g.high2 - g.low2) / k).median(),
                "share_aic_not_sqexp": (~g.aic_kernel.isin(["sqexp", "white"])).mean(),
                "share_sqexp_within2": (g.aic_sqexp_minus_best <= 2).mean(),
                "share_lb5_p_lt_05": (g.diag_lb_p_lag5 < 0.05).mean(),
                "share_shapiro_p_lt_05": (g.diag_shapiro_p < 0.05).mean(),
                "share_clipped": (g.n_clipped > 0).mean(),
                "edge_warning_share": g.edge_warning.mean(),
                "median_fit_white": g.fit_white.median(),
                "median_fit_corr": g.fit_corr.median(),
                "median_fit_L_um": g.fit_L_um.where(g.fit_corr > 0).median(),
                "band68_lo": _band(0.6827, n)[0],
                "band68_hi": _band(0.6827, n)[1],
                "band95_lo": _band(0.9545, n)[0],
                "band95_hi": _band(0.9545, n)[1],
            }
        )

    cov = t.groupby(["geometry", "case"], sort=False).apply(agg, include_groups=False).reset_index()
    cov["case"] = pd.Categorical(cov.case, list(CASES))
    cov["geometry"] = pd.Categorical(cov.geometry, list(GEOMETRIES))
    cov = cov.sort_values(["geometry", "case"])
    cov.to_csv(COVERAGE_CSV, index=False, float_format="%.4g")
    pd.set_option("display.width", 250)
    cols = ["geometry", "case", "trials", "coverage_1s", "white_coverage_1s", "coverage_2s"]
    cols += ["white_coverage_2s", "median_bias_pct", "median_width_pct", "aic_coverage_1s"]
    cols += ["clip_coverage_1s"]
    print(cov[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "measure"
    if cmd == "measure":
        measure()
    elif cmd == "run":
        run(int(sys.argv[2]) if len(sys.argv) > 2 else 300)
    elif cmd == "summarise":
        summarise()
    else:
        raise SystemExit(f"unknown command {cmd!r}")
