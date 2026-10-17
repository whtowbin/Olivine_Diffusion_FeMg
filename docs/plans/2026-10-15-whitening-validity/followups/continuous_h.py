"""Follow-up 1: fit the near-step growth h continuously instead of on (0, 1, 3, 10, 30, 100).

    uv run python docs/plans/2026-10-15-whitening-validity/followups/continuous_h.py real
    uv run python docs/plans/2026-10-15-whitening-validity/followups/continuous_h.py null [draws]
    uv run python docs/plans/2026-10-15-whitening-validity/followups/continuous_h.py extended

`real`: the 36 real profiles of the study (35 runs plus the packaged example), each fitted
with `forster.fit_run` (keep_curves). At the white best fit, exactly as FORSTER's
`_near_step` does (diffusion length from `forster.grid.diffusion_length_um` at the middle
of the initial condition's Fo range), the residuals go to:
  - `forster.uncertainty.near_step_check` (the package, grid h);
  - nearstep.NearStep (this prototype): grid h again (must match the package: the control)
    and continuous h, with its 95% profile-likelihood interval;
and the time and 1-sigma range under the default, grid-h and continuous-h covariances.
Paper and example rows go to real_continuous_h.csv here; round robin rows go only to the
private folder (Diffusion Round Robin/outputs/near_step_followups/).

`null`: false alarms and detection, grid against continuous h, on synthetic profiles with
the same pipeline (white best fit, residuals, check):
  - "paper 17 pts": the study's GCB1A R31_MP_Ol5_prof_2 geometry and true time (positions
    only), the study's seeds (same draws as synthetic.py);
  - "synthetic 30 pts": the noise-models guide geometry (30 points 2.5 um apart, 4 mol%
    step at 25 um, l = 10 um), from power.py.
Cases: white noise, two stationary misfits (controls) and the study's near-step cases.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import itertools  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
REPO = STUDY.parents[2]
PRIVATE = REPO / "Diffusion Round Robin" / "outputs" / "near_step_followups"
SCRATCH = Path(os.environ.get("FOLLOWUP_CACHE", "/tmp")) / "near_step_followups"
os.environ.setdefault("WHITENING_CACHE", str(SCRATCH))
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "analyses" / "ic_sampling"))
sys.path.insert(0, str(STUDY))
sys.path.insert(0, str(HERE))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from nearstep import NearStep, chi2_rows  # noqa: E402

from forster.uncertainty import (  # noqa: E402
    TWO_SIGMA,
    delta_chi2_interval,
    fit_correlated_points,
    interpolation_matrix,
    interval_threshold,
    near_step_check,
    white_rss,
)

WORKERS = 2


def ranges(chi2, n):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        i1 = delta_chi2_interval(chi2, interval_threshold(n), end_is_limit=False)
        i2 = delta_chi2_interval(chi2, interval_threshold(n, level=TWO_SIGMA), end_is_limit=False)
    return i1, i2


# ------------------------------------------------------------------------- real profiles


def real():
    from references import load

    import forster
    from forster.grid import diffusion_length_um, fo_range

    cases = [(n, run, prof, src) for n, (run, prof, src) in load().items()]
    with tempfile.TemporaryDirectory() as tmp:
        forster.write_template(tmp)
        ex_prof = forster.read_profiles(Path(tmp) / "profiles.csv")
        ex_run = forster.read_runs(Path(tmp) / "parameters.csv")[0]
    cases.append((f"packaged example ({ex_run.name})", ex_run, ex_prof, "example"))
    rows = []
    for name, run, prof, src in cases:
        t0 = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = forster.fit_run(run, prof, keep_curves=True)
        x, y = np.asarray(res.fit_x_um, float), np.asarray(res.fit_fo, float)
        at = res.curves @ interpolation_matrix(res.model_x_um, x).T
        w_row = int(np.argmin(white_rss(at, y)))
        days = max(w_row, 1) * res.dt_s / 86400
        ell = diffusion_length_um(
            days, [(run.diffusivity, fo_range(run.initial))], run.conditions, run.orientation
        )
        r = y - at[w_row]
        steps = run.initial.steps_um
        pkg = near_step_check(x, r, steps, ell)
        ns = NearStep(x, r, steps, ell)
        out = ns.fit()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d = fit_correlated_points(at, x, y, end_is_limit=False)
        n = len(x)
        dt_d = res.dt_s / 86400
        g1, _ = ranges(chi2_rows(at, y, out.cov_grid), n)
        c1, _ = ranges(chi2_rows(at, y, out.cov_cont), n)
        row = {
            "source": src,
            "run": name,
            "n_points": n,
            "n_near": out.n_near,
            "n_far": out.n_far,
            "diffusion_length_um": ell,
            "package_reported_daic": res.near_step_delta_aic,
            "package_daic": pkg.delta_aic,
            "package_h": pkg.h,
            "grid_daic": out.daic_grid,
            "grid_h": out.h_grid,
            "cont_daic": out.daic_cont,
            "cont_h": out.h_cont,
            "h_lo95": out.h_lo95,
            "h_hi95": out.h_hi95,
            "white_sd_molpct": 100 * out.s,
            "rho": out.rho,
            "L_um": out.L,
            "misfit_far_molpct": 100 * out.a_far,
            "misfit_peak_molpct": 100 * out.a_peak,
            "sqrt_1_plus_h": float(np.sqrt(1 + out.h_cont)),
            "implied_near_far_rms": out.implied_rms_ratio,
            "measured_near_far_rms": out.measured_rms_ratio,
            "warn_grid": out.daic_grid < -6,
            "warn_cont": out.daic_cont < -6,
            "time_default_d": d.interval.best * dt_d,
            "low_default_d": d.interval.low * dt_d,
            "high_default_d": d.interval.high * dt_d,
            "time_grid_d": g1.best * dt_d,
            "low_grid_d": g1.low * dt_d,
            "high_grid_d": g1.high * dt_d,
            "time_cont_d": c1.best * dt_d,
            "low_cont_d": c1.low * dt_d,
            "high_cont_d": c1.high * dt_d,
            "n_evals": out.n_evals,
            "seconds": time.perf_counter() - t0,
        }
        rows.append(row)
        print(
            f"{name:42s} n={n:3d} pkg {pkg.delta_aic:7.2f} (h {pkg.h:5.0f}) grid "
            f"{out.daic_grid:7.2f} cont {out.daic_cont:7.2f} h {out.h_cont:8.1f} "
            f"[{out.h_lo95:.3g}, {out.h_hi95:.3g}] {row['seconds']:.1f}s",
            flush=True,
        )
    t = pd.DataFrame(rows)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    t.to_csv(PRIVATE / "real_continuous_h_all.csv", index=False, float_format="%.5g")
    t[t.source != "round_robin"].to_csv(
        HERE / "real_continuous_h.csv", index=False, float_format="%.5g"
    )
    gap = (t.package_daic - t.grid_daic).abs().max()
    rep = (t.package_daic - t.package_reported_daic).abs().max()
    print(f"control: |package - prototype grid dAIC| max {gap:.2e}; vs fit_run's own {rep:.2e}")


# --------------------------------------------------------------------------- null cases

PAPER = "paper 17 pts (2.9 um)"
NULL_CASES = [
    "white (control)",
    "sq-exp L=3 (control)",
    "sq-exp L=10 (control)",
    "near-step sq-exp L=3",
    "near-step sq-exp L=10",
    "strong near-step L=3",
    "strong near-step L=10",
]


def paper_geometry():
    """The study's paper geometry (synthetic.build_geometries for that one only)."""
    import synthetic

    path = synthetic.CACHE / "GCB1A_R31_MP_Ol5_prof_2.npz"
    if not path.exists():
        keep = synthetic.GEOMETRIES
        synthetic.GEOMETRIES = {PAPER: keep[PAPER]}
        try:
            synthetic.build_geometries()
        finally:
            synthetic.GEOMETRIES = keep
    return synthetic.geometry(PAPER)


def _null_trial(gname, g, case, trial):
    from synthetic import SEED, make_noise

    at, x, k = g["at_points"], g["x"], int(g["k_true"])
    if gname == PAPER:
        rng = np.random.default_rng([SEED, 1, trial])  # synthetic.py's draws for this geometry
    else:
        rng = np.random.default_rng([SEED, 30, trial])
    y = at[k] + make_noise(g, case, rng)
    w_row = int(np.argmin(white_rss(at, y)))
    ell = float(g["diffusion_length_um"]) * np.sqrt(max(w_row, 1) / k)
    out = NearStep(x, y - at[w_row], g["steps"], ell).fit()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = fit_correlated_points(at, x, y, end_is_limit=False)
    c1, _ = ranges(chi2_rows(at, y, out.cov_cont), len(x))
    return {
        "geometry": gname,
        "case": case,
        "trial": trial,
        "grid_daic": out.daic_grid,
        "grid_h": out.h_grid,
        "cont_daic": out.daic_cont,
        "cont_h": out.h_cont,
        "covered": d.interval.low <= k <= d.interval.high,
        "cont_covered": c1.low <= k <= c1.high,
    }


def _null_chunk(gname, cases, start, stop):
    if gname == PAPER:
        g = paper_geometry()
    else:
        from power import geometry as power_geometry

        g = power_geometry(30)
    return [_null_trial(gname, g, c, t) for c in cases for t in range(start, stop)]


def null(draws):
    paper_geometry()  # build once before the workers start
    from power import geometry as power_geometry

    power_geometry(30)
    tasks = []
    edges = np.linspace(0, draws, 9).astype(int)
    for gname in (PAPER, "synthetic 30 pts (2.5 um)"):
        for a, b in itertools.pairwise(edges):
            tasks.append((gname, NULL_CASES, int(a), int(b)))
    t0 = time.perf_counter()
    with ProcessPoolExecutor(WORKERS) as ex:
        rows = [r for f in [ex.submit(_null_chunk, *task) for task in tasks] for r in f.result()]
    print(f"{time.perf_counter() - t0:.0f} s")
    t = pd.DataFrame(rows)
    t.to_csv(HERE / "null_trials.csv", index=False, float_format="%.5g")
    summarise_null(t)


def summarise_null(t=None):
    if t is None:
        t = pd.read_csv(HERE / "null_trials.csv")
    g = t.groupby(["geometry", "case"], sort=False)
    s = pd.DataFrame(
        {
            "draws": g.size(),
            "grid_lt_m6_pct": 100 * g.grid_daic.apply(lambda v: (v < -6).mean()),
            "cont_lt_m6_pct": 100 * g.cont_daic.apply(lambda v: (v < -6).mean()),
            "cont_lt_m8_pct": 100 * g.cont_daic.apply(lambda v: (v < -8).mean()),
            "grid_q99": g.grid_daic.quantile(0.01),
            "cont_q99": g.cont_daic.quantile(0.01),
            "median_shift": g.apply(lambda d: (d.cont_daic - d.grid_daic).median()),
            "max_shift": g.apply(lambda d: (d.cont_daic - d.grid_daic).min()),
            "grid_h100_pct": 100 * g.grid_h.apply(lambda v: (v == 100).mean()),
            "cont_h_gt100_pct": 100 * g.cont_h.apply(lambda v: (v > 100).mean()),
            "cov1_default": 100 * g.covered.mean(),
            "cov1_cont": 100 * g.cont_covered.mean(),
        }
    ).reset_index()
    s.to_csv(HERE / "null_summary.csv", index=False, float_format="%.4g")
    pd.set_option("display.width", 250)
    print(s.round(2).to_string(index=False))


EXTENDED = (0.0, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0)


def extended(draws=300):
    """Is a longer grid of h as good as the continuous fit? power.py's geometry, three
    point counts, white noise and two strong near-step misfits; also the time per check."""
    import power

    from forster.uncertainty import near_step_check as pkg_check

    rows = []
    for n in (12, 17, 35):
        g = power.geometry(n)
        at, x = g["at_points"], g["x"]
        for name in ("white", "1.2 mol%, L=3", "1.2 mol%, L=10"):
            for t in range(draws):
                rng = np.random.default_rng([power.SEED, n, t])  # power.py's draws
                y = at[power.K_TRUE] + power.noise(g, name, rng)
                w = int(np.argmin(white_rss(at, y)))
                ell = power.ELL_UM * np.sqrt(max(w, 1) / power.K_TRUE)
                r = y - at[w]
                t0 = time.perf_counter()
                pkg_check(x, r, [power.STEP_UM], ell)
                t_pkg = time.perf_counter() - t0
                ns = NearStep(x, r, [power.STEP_UM], ell)
                t0 = time.perf_counter()
                out = ns.fit()
                t_cont = time.perf_counter() - t0
                f0 = ns.at(0.0).nll
                ext = 2 * (min(ns.at(h).nll for h in EXTENDED) - f0) + 2
                rows.append(
                    {
                        "n": n,
                        "noise": name,
                        "trial": t,
                        "grid_daic": out.daic_grid,
                        "extended_daic": ext,
                        "cont_daic": out.daic_cont,
                        "ms_package": 1e3 * t_pkg,
                        "ms_continuous": 1e3 * t_cont,
                    }
                )
    t = pd.DataFrame(rows)
    g = t.groupby(["noise", "n"], sort=False)
    s = pd.DataFrame(
        {
            "draws": g.size(),
            "warn_grid_pct": 100 * g.grid_daic.apply(lambda v: (v < -6).mean()),
            "warn_extended_pct": 100 * g.extended_daic.apply(lambda v: (v < -6).mean()),
            "warn_cont_pct": 100 * g.cont_daic.apply(lambda v: (v < -6).mean()),
            "max_cont_minus_extended": g.apply(lambda d: (d.extended_daic - d.cont_daic).max()),
            "ms_package": g.ms_package.median(),
            "ms_continuous": g.ms_continuous.median(),
        }
    ).reset_index()
    s.to_csv(HERE / "extended_grid.csv", index=False, float_format="%.4g")
    print(s.round(2).to_string(index=False))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "real"
    if cmd == "real":
        real()
    elif cmd == "null":
        null(int(sys.argv[2]) if len(sys.argv) > 2 else 2000)
    elif cmd == "summarise":
        summarise_null()
    elif cmd == "extended":
        extended()
