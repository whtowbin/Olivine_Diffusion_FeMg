"""Follow-up to part B: a candidate fix and a candidate warning for misfit concentrated
near the step (the failure the real residuals show, part A4).

Candidate noise model ("near-step white"): the white noise variance may grow near the
step(s),

    C = s^2 (D_g + rho^2 K_L),   D_g = diag(1 + g e_i),
    e_i = max over steps of exp(-(x_i - step)^2 / (2 l^2)),

with l = sqrt(D t) at the white best fit, and g on the grid G_GRID (g = 0 is forster's
model). s has the same closed form, (g, rho, L) are fitted by maximum likelihood at the
white best fit, and the range uses the same threshold. Note that the synthetic
"near-step" cases use the same bump shape for their misfit, so for them the candidate
has the right shape by construction; the heteroscedastic cases do not.

Candidate noise model ("near-step correlated"): the correlated misfit's amplitude may
grow near the step instead,

    C = s^2 (I + rho^2 S_h K_L S_h),   S_h = diag(sqrt(1 + h e_i)),

with h on H_GRID (h = 0 is forster's model). This is the shape of the synthetic
"near-step" misfit, so it is the model's best case.

Candidate warning: after the default fit, the whitened residuals within 2 l of a step
have a larger variance than the rest (Brown-Forsythe p < 0.05 and ratio > 1).

Writes zone_trials.csv and zone_coverage.csv (synthetic), zone_real.csv (real profiles).

    uv run python docs/plans/2026-10-15-whitening-validity/zone_noise.py measure
    uv run python docs/plans/2026-10-15-whitening-validity/zone_noise.py run [trials]
    uv run python docs/plans/2026-10-15-whitening-validity/zone_noise.py real
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.linalg as sla  # noqa: E402
from common import HERE, kernel_matrix, length_grid, whiten  # noqa: E402
from scipy import stats  # noqa: E402
from synthetic import GEOMETRIES, SEED, _band, build_geometries, geometry, make_noise  # noqa: E402

from forster.uncertainty import (  # noqa: E402
    RHO_GRID,
    TWO_SIGMA,
    delta_chi2_interval,
    fit_correlated_points,
    interval_threshold,
    white_rss,
)

G_GRID = np.array([0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
H_GRID = np.array([0.0, 1.0, 3.0, 10.0, 30.0, 100.0])
ZONE_LENGTHS = 2.0
WORKERS = 4
CHUNK = 25
CASES = [
    "white (control)",
    "sq-exp L=10 (control)",
    "exponential L=10",
    "near-step sq-exp L=3",
    "near-step sq-exp L=10",
    "strong near-step L=3",
    "strong near-step L=10",
    "sd grows with |Fo - core|",
    "sd rises 4x to the rim",
]
TRIALS_CSV = HERE / "zone_trials.csv"


def bump(x, steps, ell):
    return np.exp(-0.5 * ((x[:, None] - np.asarray(steps)[None, :]) / ell) ** 2).max(axis=1)


def fit_zone_noise(x, r, e, mode="white"):
    """ML noise model with a near-step term, at residuals r. ``mode`` "white":
    C = s^2 (diag(1 + g e) + rho^2 K); "corr": C = s^2 (I + rho^2 S K S), S = diag(sqrt(1 +
    g e)). g = 0 is forster's model in both. Returns (covariance, g, delta AIC of the
    fitted model against the best g = 0 model, one extra parameter; < 0 favours it)."""
    n = len(r)
    lengths = length_grid(x)
    Ks = [kernel_matrix("sqexp", x, L) for L in lengths]
    grid = G_GRID if mode == "white" else H_GRID
    eye = np.eye(n)

    def parts(g):
        if mode == "white":
            return np.diag(1.0 + g * e), [K for K in Ks]
        sc = np.sqrt(1.0 + g * e)
        return eye, [sc[:, None] * K * sc[None, :] for K in Ks]

    best = (np.inf, None, None, None, None)
    nll0 = np.inf  # best with g = 0 (forster's model)
    for g in grid:
        base, mats = parts(g)
        dg = np.diag(base)
        s2 = max(float(np.sum(r**2 / dg)) / n, 1e-30)
        nll = 0.5 * n * np.log(s2) + 0.5 * float(np.log(dg).sum())
        if g == 0:
            nll0 = min(nll0, nll)
        if nll < best[0]:
            best = (nll, s2, g, 0.0, None)
        for i, M in enumerate(mats):
            for rho in RHO_GRID[1:]:
                chol = np.linalg.cholesky(base + rho**2 * M)
                z = sla.solve_triangular(chol, r, lower=True)
                s2 = max(float(z @ z) / n, 1e-30)
                nll = 0.5 * n * np.log(s2) + float(np.log(np.diag(chol)).sum())
                if g == 0:
                    nll0 = min(nll0, nll)
                if nll < best[0]:
                    best = (nll, s2, g, rho, i)
    nll, s2, g, rho, i = best
    base, mats = parts(g)
    cov = s2 * base + (s2 * rho**2 * mats[i] if rho > 0 else 0.0)
    # AIC change for the one extra parameter (< 0: the near-step term is preferred)
    return cov, float(g), float(2 * (nll - nll0) + 2)


def zone_warning(x, z, near):
    """Variance ratio of whitened residuals near / far from the step, and the
    Brown-Forsythe p-value; NaN if either zone has fewer than 3 points."""
    if near.sum() < 3 or (~near).sum() < 3:
        return np.nan, np.nan
    ratio = float(np.mean(z[near] ** 2) / np.mean(z[~near] ** 2))
    return ratio, float(stats.levene(z[near], z[~near], center="median").pvalue)


def _ranges(at, y, cov):
    chol = np.linalg.cholesky(cov)
    z = sla.solve_triangular(chol, (at - y).T, lower=True)
    chi2 = np.einsum("pk,pk->k", z, z)
    n = len(y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        i1 = delta_chi2_interval(chi2, interval_threshold(n), end_is_limit=False)
        i2 = delta_chi2_interval(chi2, interval_threshold(n, level=TWO_SIGMA), end_is_limit=False)
    return i1, i2


def zone_fit(at, x, y, steps, ell_at, default):
    """Both candidate fits and the warning. ``ell_at(row)``: diffusion length at a row."""
    w_row = int(np.argmin(white_rss(at, y)))
    ell = ell_at(w_row)
    e = bump(x, steps, ell)
    out = {}
    for mode in ("white", "corr"):
        cov, g, daic = fit_zone_noise(x, y - at[w_row], e, mode)
        out[mode] = (*_ranges(at, y, cov), g, daic)
    # Warning from the default fit's whitened residuals at its best fit.
    near = np.abs(x[:, None] - np.asarray(steps)[None, :]).min(axis=1) <= ZONE_LENGTHS * ell
    r = y - at[default.interval.best]
    zd = whiten(default.noise.covariance(x), r)
    ratio, p = zone_warning(x, zd, near)
    rms = (
        float(np.sqrt(np.mean(r[near] ** 2) / np.mean(r[~near] ** 2)))
        if near.sum() >= 3 and (~near).sum() >= 3
        else np.nan
    )
    return out, ratio, p, rms


def one_trial(gname, case, trial):
    g = geometry(gname)
    gi = list(GEOMETRIES).index(gname)
    rng = np.random.default_rng([SEED, gi, trial])  # same draws as synthetic.py
    at, x, k = g["at_points"], g["x"], g["k_true"]
    y = at[k] + make_noise(g, case, rng)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = fit_correlated_points(at, x, y, end_is_limit=False)
    ell_true = float(g["diffusion_length_um"])
    out, ratio, p, rms = zone_fit(
        at, x, y, g["steps"], lambda row: ell_true * np.sqrt(max(row, 1) / k), d
    )
    row = {
        "geometry": gname,
        "case": case,
        "trial": trial,
        "k_true": k,
        "covered": d.interval.low <= k <= d.interval.high,
        "covered2": d.interval2.low <= k <= d.interval2.high,
        "width": (d.interval.high - d.interval.low) / k,
        "near_far_var_ratio": ratio,
        "near_far_bf_p": p,
        "near_far_rms_raw": rms,
    }
    for mode, (i1, i2, gam, daic) in out.items():
        row |= {
            f"{mode}_delta_aic": daic,
            f"{mode}_best": i1.best,
            f"{mode}_covered": i1.low <= k <= i1.high,
            f"{mode}_covered2": i2.low <= k <= i2.high,
            f"{mode}_width": (i1.high - i1.low) / k,
            f"{mode}_g": gam,
        }
    return row


def run_chunk(gname, case, start, stop):
    return [one_trial(gname, case, t) for t in range(start, stop)]


def measure():
    build_geometries()
    for gname in GEOMETRIES:
        t = time.perf_counter()
        run_chunk(gname, "near-step sq-exp L=10", 0, 3)
        print(f"{gname:24s} {(time.perf_counter() - t) / 3:.3f} s/trial", flush=True)


def run(trials):
    build_geometries()
    done = set()
    if TRIALS_CSV.exists():
        old = pd.read_csv(TRIALS_CSV, usecols=["geometry", "case", "trial"])
        done = set(zip(old.geometry, old.case, old.trial, strict=True))
    order = sorted(GEOMETRIES, key=lambda gname: -len(geometry(gname)["x"]))
    tasks = [
        (gname, case, s, min(s + CHUNK, trials))
        for gname in order
        for case in CASES
        for s in range(0, trials, CHUNK)
        if not all((gname, case, t) in done for t in range(s, min(s + CHUNK, trials)))
    ]
    print(f"{len(tasks)} chunks", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(run_chunk, *task): task for task in tasks}
        for i, fut in enumerate(as_completed(futures), 1):
            pd.DataFrame(fut.result()).to_csv(
                TRIALS_CSV, mode="a", header=not TRIALS_CSV.exists(), index=False
            )
            if i % 20 == 0 or i == len(tasks):
                print(f"{time.perf_counter() - t0:7.1f}s [{i}/{len(tasks)}]", flush=True)
    summarise()


def summarise():
    t = pd.read_csv(TRIALS_CSV).drop_duplicates(["geometry", "case", "trial"])
    t["warn"] = (t.near_far_bf_p < 0.05) & (t.near_far_var_ratio > 1)

    def agg(g):
        n = len(g)
        w = g[g.warn]
        nw = g[~g.warn]
        return pd.Series(
            {
                "trials": n,
                "coverage_1s": g.covered.mean(),
                "white_cand_coverage_1s": g.white_covered.mean(),
                "corr_cand_coverage_1s": g.corr_covered.mean(),
                "coverage_2s": g.covered2.mean(),
                "white_cand_coverage_2s": g.white_covered2.mean(),
                "corr_cand_coverage_2s": g.corr_covered2.mean(),
                "median_width_pct": 100 * g.width.median(),
                "white_cand_median_width_pct": 100 * g.white_width.median(),
                "corr_cand_median_width_pct": 100 * g.corr_width.median(),
                "corr_cand_median_bias_pct": 100 * ((g.corr_best - g.k_true) / g.k_true).median(),
                "white_cand_share_g_positive": (g.white_g > 0).mean(),
                "corr_cand_share_h_positive": (g.corr_g > 0).mean(),
                "median_near_far_rms_raw": g.near_far_rms_raw.median(),
                "share_corr_daic_lt_m2": (g.corr_delta_aic < -2).mean(),
                "share_corr_daic_lt_m6": (g.corr_delta_aic < -6).mean(),
                "coverage_1s_if_daic_lt_m2": g.covered[g.corr_delta_aic < -2].mean(),
                "coverage_1s_if_daic_ge_m2": g.covered[g.corr_delta_aic >= -2].mean(),
                "warning_share": g.warn.mean(),
                "coverage_1s_if_warned": w.covered.mean() if len(w) else np.nan,
                "coverage_1s_if_not_warned": nw.covered.mean() if len(nw) else np.nan,
                "band68_lo": _band(0.6827, n)[0],
                "band68_hi": _band(0.6827, n)[1],
            }
        )

    cov = t.groupby(["geometry", "case"], sort=False).apply(agg, include_groups=False)
    cov = cov.reset_index()
    cov["case"] = pd.Categorical(cov.case, CASES)
    cov["geometry"] = pd.Categorical(cov.geometry, list(GEOMETRIES))
    cov = cov.sort_values(["geometry", "case"])
    cov.to_csv(HERE / "zone_coverage.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 250)
    print(cov.round(3).to_string(index=False))


def real():
    """The candidate on the 35 real profiles plus the example: does it change the fit?"""
    import tempfile
    from pathlib import Path

    from references import load

    import forster
    from forster.uncertainty import interpolation_matrix, points_in_domain

    cases = [(n, run, prof, src) for n, (run, prof, src) in load().items()]
    with tempfile.TemporaryDirectory() as tmp:
        forster.write_template(tmp)
        ex_prof = forster.read_profiles(Path(tmp) / "profiles.csv")
        ex_run = forster.read_runs(Path(tmp) / "parameters.csv")[0]
    cases.append((f"packaged example ({ex_run.name})", ex_run, ex_prof, "example"))
    rows = []
    for name, run, prof, src in cases:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = forster.fit_run(run, prof, keep_curves=True)
        x, y = points_in_domain(res.model_x_um, res.fit_x_um, res.fit_fo)
        o = np.argsort(x)
        x, y = x[o], y[o]
        at = res.curves @ interpolation_matrix(res.model_x_um, x).T
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = fit_correlated_points(at, x, y, end_is_limit=False)
        D, _ = run.diffusivity.functions(run.conditions, run.orientation)
        d_mid = float(D(np.array([np.mean(y)]))[0])

        def ell_at(row, d_mid=d_mid, dt=res.dt_s):
            return float(np.sqrt(d_mid * max(row, 1) * dt) * 1e6)

        out, ratio, p, rms = zone_fit(at, x, y, run.initial.steps_um, ell_at, d)
        dt_d = res.dt_s / 86400
        row = {
            "source": src,
            "run": name,
            "n_points": len(x),
            "time_days": d.interval.best * dt_d,
            "low_days": d.interval.low * dt_d,
            "high_days": d.interval.high * dt_d,
            "near_far_var_ratio": ratio,
            "near_far_bf_p": p,
            "near_far_rms_raw": rms,
        }
        for mode, (i1, _, gam, daic) in out.items():
            row |= {
                f"{mode}_time_days": i1.best * dt_d,
                f"{mode}_low_days": i1.low * dt_d,
                f"{mode}_high_days": i1.high * dt_d,
                f"{mode}_g": gam,
                # AIC change for one extra parameter; < 0: the candidate is preferred
                f"{mode}_delta_aic": daic,
                f"{mode}_width_ratio": (i1.high - i1.low) / (d.interval.high - d.interval.low),
                f"{mode}_time_ratio": i1.best / d.interval.best,
            }
        rows.append(row)
        print(
            f"{name:44s} corr: h={row['corr_g']:5.1f} dAIC={row['corr_delta_aic']:7.2f} "
            f"width x{row['corr_width_ratio']:.2f} time x{row['corr_time_ratio']:.2f}",
            flush=True,
        )
    pd.DataFrame(rows).to_csv(HERE / "zone_real.csv", index=False, float_format="%.5g")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "measure"
    {"measure": measure, "summarise": summarise, "real": real}.get(
        cmd, lambda: run(int(sys.argv[2]) if len(sys.argv) > 2 else 1000)
    )()
