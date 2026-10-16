"""Part A: is the squared-exponential noise model adequate for real residuals?

For the 35 real profiles of the grid-choice study (17 round robin, 18 Towbin and Plank
2026; loaded as `analyses/ic_sampling/references.py` does) plus the packaged example
(`forster.write_template`):

1. fit with the default method (`forster.fit_run`: correlated likelihood, implicit
   scheme, each run's own grid);
2. at the white (least-squares) best fit, where forster fits its noise model, fit four
   noise models by maximum likelihood: white only, white + squared-exponential (the
   current one), white + exponential (Matern 1/2), white + Matern 3/2; compare AIC.
   Repeated at the correlated best fit as a check;
3. whiten the residuals at the correlated best fit with forster's fitted noise model and
   test the result for whiteness (Ljung-Box lags 1-5, runs test, Shapiro-Wilk, var z,
   max |z|);
4. compare residual scatter near the step(s) and the rim (within 2 diffusion lengths)
   with the scatter elsewhere.

Writes real_kernels.csv (one row per profile x kernel x residual set) and
real_profiles.csv (one row per profile). Round robin profiles are private: only
diagnostic numbers are written, never their data. Run from the repo root:

    uv run python docs/plans/2026-10-15-whitening-validity/real_profiles.py
"""

from __future__ import annotations

import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from common import HERE, KERNELS, fit_kernel, whiten, whiteness
from synthetic import loo_z
from references import load
from scipy import stats

import forster
from forster.uncertainty import (
    correlated_chi2,
    fit_noise_model,
    interpolation_matrix,
    points_in_domain,
    white_rss,
)

ZONE_LENGTHS = 2.0  # "near" = within this many diffusion lengths
N_BOOT = 200  # parametric-bootstrap replicates per profile
BOOT_ROWS = 2000  # time steps kept (evenly thinned) for the bootstrap refits
SEED = 20261015
# Which tail of each statistic is "bad": low p-values, large |z|, and so on.
BOOT_TAILS = {
    "lb_p_lag5": "low",
    "runs_p": "low",
    "shapiro_p": "low",
    "var_z": "two",
    "max_abs_z": "high",
    "kurtosis_z": "high",
    "lag1_z": "two",
    "step_var_ratio_z": "high",
}


def example_run():
    with tempfile.TemporaryDirectory() as tmp:
        forster.write_template(tmp)
        profiles = forster.read_profiles(Path(tmp) / "profiles.csv")
        run = forster.read_runs(Path(tmp) / "parameters.csv")[0]
    return run, profiles


def zone_stats(x, r, z, near):
    """Scatter of raw residuals r and whitened z inside ``near`` against outside it."""
    far = ~near
    out = {"n_near": int(near.sum()), "n_far": int(far.sum())}
    if near.sum() < 3 or far.sum() < 3:
        return out | {"rms_ratio_raw": np.nan, "var_ratio_z": np.nan, "bf_p_z": np.nan}
    return out | {
        "rms_ratio_raw": float(np.sqrt(np.mean(r[near] ** 2) / np.mean(r[far] ** 2))),
        "var_ratio_z": float(np.mean(z[near] ** 2) / np.mean(z[far] ** 2)),
        # Brown-Forsythe (Levene with the median), on the whitened residuals, which are
        # independent if the model holds; the raw residuals are correlated.
        "bf_p_z": float(stats.levene(z[near], z[far], center="median").pvalue),
    }


def boot_pvalues(at_points, best, x, noise, near_step, observed, seed):
    """Parametric bootstrap of the whiteness statistics under forster's own fitted model.

    The nominal tests assume z is exactly iid N(0, 1), but the noise model was fitted to
    the same residuals, which makes z "too white". Here data are drawn from the best-fit
    curve plus noise from the fitted covariance, and the whole default pipeline is rerun
    (white best fit, noise model, generalised chi2, best fit, whitening). The p-value is
    the share of replicates at least as extreme as the observed statistic.
    """
    stride = max(1, len(at_points) // BOOT_ROWS)
    rows = np.unique(np.concatenate([np.arange(0, len(at_points), stride), [best]]))
    model = at_points[rows]
    truth = at_points[best]
    chol = np.linalg.cholesky(noise.covariance(x))
    rng = np.random.default_rng(seed)
    reps = {k: [] for k in BOOT_TAILS}
    for _ in range(N_BOOT):
        y = truth + chol @ rng.standard_normal(len(x))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chi2, nz = correlated_chi2(model, x, y)
        r = y - model[int(np.argmin(chi2))]
        z = whiten(nz.covariance(x), r)
        w = whiteness(z)
        w["step_var_ratio_z"] = zone_stats(x, r, z, near_step).get("var_ratio_z", np.nan)
        for k in BOOT_TAILS:
            reps[k].append(w[k])
    out = {}
    for k, tail in BOOT_TAILS.items():
        v = np.asarray(reps[k], dtype=float)
        o = observed.get(k, np.nan)
        v = v[np.isfinite(v)]
        if not np.isfinite(o) or len(v) == 0:
            out[f"boot_p_{k}"] = np.nan
            continue
        lo, hi = (np.sum(v <= o) + 1) / (len(v) + 1), (np.sum(v >= o) + 1) / (len(v) + 1)
        out[f"boot_p_{k}"] = float({"low": lo, "high": hi, "two": min(1.0, 2 * min(lo, hi))}[tail])
    return out


def analyse(name, run, profiles, source, seed):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = forster.fit_run(run, profiles, keep_curves=True)
    notes = sorted({str(w.message)[:60] for w in caught if issubclass(w.category, UserWarning)})
    x, y = points_in_domain(res.model_x_um, res.fit_x_um, res.fit_fo)
    order = np.argsort(x)
    x, y = x[order], y[order]
    W = interpolation_matrix(res.model_x_um, x)
    at_points = res.curves @ W.T
    white_row = int(np.argmin(white_rss(at_points, y)))
    best = res.indices[0]
    r_white = y - at_points[white_row]
    r_best = y - at_points[best]
    # Check: our squared-exponential grid fit reproduces forster's noise model exactly.
    check = fit_noise_model(x, r_white)
    assert np.isclose(check.white_sigma, res.noise.white_sigma), name
    kern_rows = []
    fits = {}
    for where, r in (("white best fit", r_white), ("correlated best fit", r_best)):
        for kind in KERNELS:
            f = fit_kernel(kind, x, r)
            fits[(where, kind)] = f
            kern_rows.append(
                {
                    "source": source,
                    "run": name,
                    "residuals_at": where,
                    "kernel": kind,
                    "n_points": len(x),
                    "loglik": f.loglik,
                    "aic": f.aic,
                    "white_sigma": f.white_sigma,
                    "correlated_sigma": f.correlated_sigma,
                    "length_um": f.length_um,
                }
            )
    z = whiten(res.noise.covariance(x), r_best)
    D, _ = run.diffusivity.functions(run.conditions, run.orientation)
    d_mid = float(D(np.array([np.mean(y)]))[0])
    diff_len = float(np.sqrt(d_mid * max(res.time_s, res.dt_s)) * 1e6)
    steps = np.array(run.initial.steps_um)
    rim = run.initial.domain_um[0]
    near_step = (np.abs(x[:, None] - steps[None, :]) <= ZONE_LENGTHS * diff_len).any(axis=1)
    near_rim = (x - rim <= ZONE_LENGTHS * diff_len) & ~near_step
    step_z = {f"step_{k}": v for k, v in zone_stats(x, r_best, z, near_step).items()}
    rim_z = {f"rim_{k}": v for k, v in zone_stats(x, r_best, z, near_rim).items()}
    aics = {k: fits[("white best fit", k)].aic for k in KERNELS}
    row = {
        "source": source,
        "run": name,
        "n_points": len(x),
        "spacing_um": float(np.median(np.diff(x))),
        "time_days": res.time_days,
        "diffusion_length_um": diff_len,
        "white_sigma_fo": res.noise.white_sigma,
        "corr_sigma_fo": res.noise.correlated_sigma,
        "L_um": res.noise.correlation_length_um,
        **{f"aic_{k}": v for k, v in aics.items()},
        "best_kernel": min(aics, key=aics.get),
        "lag1_raw": float(np.corrcoef(r_best[:-1], r_best[1:])[0, 1]),
        **whiteness(z),
        **step_z,
        **rim_z,
        "warnings": "; ".join(notes),
    }
    # Where the most extreme point is: standardised leave-one-out residual under the
    # fitted model (not order-dependent, unlike the Cholesky z), and its distance from
    # the nearest step in diffusion lengths.
    lz = loo_z(res.noise.covariance(x), r_best)
    i_max = int(np.argmax(np.abs(lz)))
    row["max_abs_loo_z"] = float(np.abs(lz[i_max]))
    row["n_loo_z_gt4"] = int((np.abs(lz) > 4).sum())
    row["max_loo_z_steps_away_l"] = float(np.min(np.abs(x[i_max] - steps)) / diff_len)
    row.update(boot_pvalues(at_points, best, x, res.noise, near_step, row, seed))
    return row, kern_rows


def main():
    t0 = time.perf_counter()
    cases = [(n, run, prof, src) for n, (run, prof, src) in load().items()]
    run, prof = example_run()
    cases.append(("packaged example (" + run.name + ")", run, prof, "example"))
    rows, kern = [], []
    for i, (name, run, prof, src) in enumerate(cases):
        row, k = analyse(name, run, prof, src, SEED + i)
        rows.append(row)
        kern += k
        print(
            f"{time.perf_counter() - t0:6.1f}s {src:11s} {name:44s} "
            f"best={row['best_kernel']:8s} LB5 p={row['lb_p_lag5']:.3f} "
            f"SW p={row['shapiro_p']:.3f} max|z|={row['max_abs_z']:.2f} "
            f"boot: LB {row['boot_p_lb_p_lag5']:.3f} SW {row['boot_p_shapiro_p']:.3f} "
            f"max|z| {row['boot_p_max_abs_z']:.3f}",
            flush=True,
        )
    pd.DataFrame(rows).to_csv(HERE / "real_profiles.csv", index=False, float_format="%.5g")
    pd.DataFrame(kern).to_csv(HERE / "real_kernels.csv", index=False, float_format="%.6g")
    print(f"done in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
