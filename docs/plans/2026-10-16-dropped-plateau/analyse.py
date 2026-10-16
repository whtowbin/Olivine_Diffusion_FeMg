"""Core-side points dropped from published runs: can they come back?

Several Towbin and Plank (2026) runs end the model domain (``run.initial.domain_um[1]``)
before the last measured point, so the core-side points past it are not fitted. For
every paper run with more than one point past its domain end this script

1. fits the run as published and with the domain extended to the end of the data
   (`findings.extended_run`), with FORSTER's default fit (`forster.fit_run`);
2. classifies the dropped points as "flat" (the core plateau continues) or "feature"
   (`flat_test`) from the residuals of the extended fit. Past the published domain end
   the model is flat at the held core Fo, so these residuals are the data minus the core
   level;
3. if the points become a feature only at the far end, also refits with the domain ended
   just before the feature;
4. as a sensitivity check, refits the extended run with the core Fo set to the mean of
   every plateau point (published and dropped);
5. where the near-step check warns (dAIC < -6), adds the range of the near-step noise
   model (docs/references/noise-model-findings.md section 3), which keeps its coverage
   when a long plateau makes the default range too narrow;
6. writes results.csv and profiles.png beside this file.

Run from the repository root:

    uv run python docs/plans/2026-10-16-dropped-plateau/analyse.py

Public paper profiles only (analyses/xenolith_archive). About 10 s, one process.
"""

from __future__ import annotations

import itertools
import math
import sys
import warnings
from dataclasses import replace
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "docs" / "references" / "figures" / "noise_findings"))

import findings as F  # noqa: E402  (sets the BLAS thread variables first)
import matplotlib as mpl  # noqa: E402

mpl.use("Agg")
import matplotlib.lines  # noqa: E402
import matplotlib.patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

import forster  # noqa: E402
from forster.grid import diffusion_length_um, fo_range  # noqa: E402
from forster.model import fe_mg_diffusivity  # noqa: E402
from forster.uncertainty import interpolation_matrix, points_in_domain, white_rss  # noqa: E402

MIN_DROPPED = 2  # "more than one point past the domain end"
P_FEATURE = 0.05  # any test below this calls the dropped points a feature
LEVEL_S = 1.0  # mean residual of the dropped points must be within this many s
POINT_S = 2.0  # ... and all but at most one dropped point within this many s
NEAR_STEP_WARN = F.NEAR_STEP_WARN

# ----------------------------------------------------------------------------- fitting


def near_step_range(run, profiles, steps) -> dict:
    """1- and 2-sigma range of the near-step noise model (findings._fit_three, "corr":
    the misfit may grow within about 2 diffusion lengths of a step), on every row."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fr = forster.fit_run(run, profiles, keep_curves=True)
    x, y = points_in_domain(fr.model_x_um, fr.fit_x_um, fr.fit_fo)
    at = fr.curves @ interpolation_matrix(fr.model_x_um, x).T
    w_row = int(np.argmin(white_rss(at, y)))
    species = [(fe_mg_diffusivity(run.diffusivity), fo_range(run.initial))]
    days = max(w_row, 1) * fr.dt_s / F.DAY
    ell = diffusion_length_um(days, species, run.conditions, run.orientation)
    cov, _, _ = F.fit_zone_noise(x, y - at[w_row], F.bump(x, steps, ell), "corr")
    i1, i2 = F._ranges(at, y, cov)
    d = fr.dt_s / F.DAY
    return {"ns_time_d": i1.best * d, "ns_lo1_d": i1.low * d, "ns_hi1_d": i1.high * d,
            "ns_lo2_d": i2.low * d, "ns_hi2_d": i2.high * d}  # fmt: skip


def fit(run, profiles) -> dict:
    """FORSTER's default fit of ``run``: the numbers reported and the curves drawn."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fr = forster.fit_run(run, profiles)
    lo1, hi1 = fr.time_range_days(sigmas=1)
    lo2, hi2 = fr.time_range_days(sigmas=2)
    prof = profiles[run.profile_name]
    x_all, y_all = np.asarray(prof.distance_um, float), np.asarray(prof.fo, float)
    step = max(run.initial.steps_um)
    out = {
        "domain_end_um": float(run.initial.domain_um[1]), "model_end_um": float(fr.model_x_um[-1]),
        "core_fo_mol": 100 * run.initial.segment_fo[-1], "n_fitted": len(fr.fit_x_um),
        "time_d": fr.time_days, "lo1_d": lo1, "hi1_d": hi1, "lo2_d": lo2, "hi2_d": hi2,
        "near_step_daic": fr.near_step_delta_aic, "s_mol": 100 * fr.noise.white_sigma,
        "a_mol": 100 * fr.noise.correlated_sigma, "L_um": fr.noise.correlation_length_um,
        "core_end_l": fr.core_end_lengths, "modelled_d": fr.modelled_days,
        "plateau_start_um": F.plateau_start_um(fr, step),
        "noise": fr.noise, "model_x": fr.model_x_um, "best": fr.best_fit, "x_all": x_all,
        "y_all": y_all, "resid_all": y_all - np.interp(x_all, fr.model_x_um, fr.best_fit),
    }  # fmt: skip
    if fr.near_step_delta_aic is not None and fr.near_step_delta_aic < NEAR_STEP_WARN:
        out |= near_step_range(run, profiles, run.initial.steps_um)
    return out


# ------------------------------------------------------------------------ classifying


def runs_test_p(signs: np.ndarray) -> float:
    """One-sided Wald-Wolfowitz runs test, exact: P(as few sign runs or fewer) with the
    signs in random order. Few runs = residuals of one sign clustered together (a step, a
    trend or a second plateau). NaN with fewer than 4 points or all signs the same."""
    n, n_pos = len(signs), int(signs.sum())
    if n < 4 or n_pos in (0, n):
        return math.nan

    def count_runs(s):
        return 1 + int(np.sum(s[1:] != s[:-1]))

    observed, total, hits = count_runs(signs), 0, 0
    for pos in itertools.combinations(range(n), n_pos):
        s = np.zeros(n, bool)
        s[list(pos)] = True
        total += 1
        hits += count_runs(s) <= observed
    return hits / total


def gls(X, r, cov) -> tuple[np.ndarray, np.ndarray]:
    """Generalised least squares of r on the columns of X under covariance cov:
    coefficients and their covariance."""
    Ci = np.linalg.inv(cov)
    A = X.T @ Ci @ X
    beta = np.linalg.solve(A, X.T @ Ci @ r)
    return beta, np.linalg.inv(A)


def line_p(x, r, noise) -> tuple[float, float]:
    """Line r = b0 + b1 (x - mean x) through the dropped points under FORSTER's fitted
    noise (white plus correlated); slope (per um) and p of b0 = b1 = 0 (2 d.o.f.). A small
    p: the points sit off the core level or slope, beyond what the fitted noise explains."""
    X = np.column_stack([np.ones_like(x), x - x.mean()])
    beta, V = gls(X, r, noise.covariance(x))
    return float(beta[1]), float(stats.chi2.sf(float(beta @ np.linalg.solve(V, beta)), 2))


def shift_p(x, r, noise, old_end) -> tuple[float, float]:
    """Level step at the published domain end, across the whole plateau: r = b0 + b1 [x >
    old_end] under the fitted noise; b1 (Fo fraction) and its two-sided p."""
    X = np.column_stack([np.ones_like(x), (x > old_end).astype(float)])
    beta, V = gls(X, r, noise.covariance(x))
    z = beta[1] / math.sqrt(V[1, 1])
    return float(beta[1]), float(2 * stats.norm.sf(abs(z)))


def flat_test(x, r, noise, x_plat=None, old_end=None) -> dict:
    """Is the core plateau flat over the dropped points (x, residuals r)? "Flat" needs all
    of: (1) mean residual within LEVEL_S s of the core level; (2) all but at most one point
    within POINT_S s; (3) with 3 or more points, the GLS line test p > P_FEATURE; (4) the
    runs test p > P_FEATURE (4 or more points); and (5), given the plateau start
    ``x_plat``, no level step at ``old_end`` between the published plateau points and the
    dropped ones (p > P_FEATURE). s is the fitted white-noise sd."""
    s = noise.white_sigma
    n = len(x)
    out = {"n": n, "mean_s": float(r.mean() / s), "n_over_2s": int(np.sum(np.abs(r) > POINT_S * s))}
    slope, p_line = line_p(x, r, noise) if n >= 3 else (math.nan, math.nan)
    span = float(x.max() - x.min()) if n > 1 else 0.0
    out |= {"trend_s": slope * span / s, "p_line": p_line, "p_runs": runs_test_p(r > 0)}
    out["shift_s"] = out["p_shift"] = math.nan
    ok = abs(out["mean_s"]) <= LEVEL_S and out["n_over_2s"] <= 1
    ok &= not p_line <= P_FEATURE  # NaN (too few points) passes
    ok &= not out["p_runs"] <= P_FEATURE
    if x_plat is not None:
        shift, p = shift_p(x_plat[0], x_plat[1], noise, old_end)
        out |= {"shift_s": shift / s, "p_shift": p}
        ok &= p > P_FEATURE
    out["flat"] = bool(ok)
    return out


def classify(ext: dict, old_end: float) -> tuple[dict, float | None]:
    """The flat test on the points past ``old_end`` (residuals of the extended fit), and,
    if they fail it, a domain end before the feature when there is a clear one: the
    longest run of dropped points from the old end outward that passes, followed by points
    that fail on their own and sit at a different level (GLS level step between the two
    groups, p < P_FEATURE). The end is halfway between the two groups; None otherwise."""
    x, r = ext["x_all"], ext["resid_all"]
    o = np.argsort(x)
    x, r = x[o], r[o]
    ok = np.isfinite(r)
    on_plateau = ok & (x >= ext["plateau_start_um"])
    past = ok & (x > old_end)
    xp, rp = x[past], r[past]
    whole = flat_test(xp, rp, ext["noise"], (x[on_plateau], r[on_plateau]), old_end)
    if whole["flat"]:
        return whole, None
    for k in range(len(xp) - 1, 0, -1):
        if flat_test(xp[:k], rp[:k], ext["noise"])["flat"]:
            cut = 0.5 * (xp[k - 1] + xp[k])
            _, p = shift_p(xp, rp, ext["noise"], cut)
            if not flat_test(xp[k:], rp[k:], ext["noise"])["flat"] and p < P_FEATURE:
                return whole, cut
            break
    return whole, None


# --------------------------------------------------------------------------- the runs


def affected(runs, profiles):
    for run in runs:
        prof = profiles.get(run.profile_name)
        if prof is None:
            continue
        x = np.asarray(prof.distance_um, float)
        n_past = int(np.sum(x > run.initial.domain_um[1]))
        if n_past >= MIN_DROPPED:
            yield run, prof, n_past


def with_end(run, end_um):
    lo, _ = run.initial.domain_um
    return replace(run, initial=replace(run.initial, domain_um=(lo, float(end_um))))


def with_core(run, core_fo):
    seg = (*run.initial.segment_fo[:-1], float(core_fo))
    return replace(run, initial=replace(run.initial, segment_fo=seg))


def analyse(runs, profiles):
    rows, curves = [], {}
    for run, prof, n_past in affected(runs, profiles):
        old_end = float(run.initial.domain_um[1])
        ext_run = F.extended_run(run, prof)
        pub, ext = fit(run, profiles), fit(ext_run, profiles)
        test, partial = classify(ext, old_end)
        fits = {"published": pub, "extended to data end": ext}
        if partial is not None:
            fits["extended to before feature"] = fit(with_end(run, partial), profiles)
        x, y = ext["x_all"], ext["y_all"]
        on_plateau = x >= ext["plateau_start_um"]
        levels = {
            "plateau_mean_published_mol": 100 * float(np.nanmean(y[on_plateau & (x <= old_end)])),
            "dropped_mean_mol": 100 * float(np.nanmean(y[x > old_end])),
            "plateau_mean_all_mol": 100 * float(np.nanmean(y[on_plateau])),
        }
        core = levels["plateau_mean_all_mol"] / 100
        fits["extended, core Fo = plateau mean"] = fit(with_core(ext_run, core), profiles)
        curves[run.name] = (fits, old_end, test)
        for label, f in fits.items():
            rows.append({
                "run": run.name, "fit": label, "class": "flat" if test["flat"] else "feature",
                "domain_start_um": run.initial.domain_um[0],
                "data_start_um": float(np.nanmin(prof.distance_um)),
                "data_end_um": float(np.nanmax(prof.distance_um)),
                "published_domain_end_um": old_end, "points_past_end": n_past,
                "last_step_um": max(run.initial.steps_um),
                **{f"dropped_{k}": v for k, v in test.items() if k not in ("n", "flat")},
                **levels,
                **{k: v for k, v in f.items() if np.isscalar(v)},
                "time_change_pct": 100 * (f["time_d"] / pub["time_d"] - 1),
            })  # fmt: skip
    return pd.DataFrame(rows), curves


# --------------------------------------------------------------------------- the figure

OKABE = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
C_PUB, C_EXT, C_PART = OKABE[5], OKABE[6], OKABE[3]
C_DROP = "#F6EFA6"
mpl.rcParams.update({
    "font.size": 7.5, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5, "legend.fontsize": 6.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "0.92", "grid.linewidth": 0.5, "legend.frameon": False,
    "lines.linewidth": 1.1, "savefig.dpi": 200, "pdf.fonttype": 42,
})  # fmt: skip


def short(name: str) -> str:
    return name.replace("AZ18 ", "").replace("_prof_1", "").replace("_prof 1", "")


def _panel(ax, ax_r, fits, old_end, test, name):
    pub, ext = fits["published"], fits["extended to data end"]
    x, y, r = ext["x_all"], ext["y_all"], ext["resid_all"] / ext["noise"].white_sigma
    inside = (x >= pub["model_x"][0]) & (x <= pub["model_x"][-1])
    for a in (ax, ax_r):
        a.axvspan(old_end, x.max(), color=C_DROP, lw=0, zorder=0)
        a.axvline(old_end, color=C_PUB, lw=0.8, ls=":")
    for a, v in ((ax, 100 * y), (ax_r, r)):
        a.plot(x[inside], v[inside], "o", ms=2.6, color="0.25", zorder=3)
        a.plot(x[~inside], v[~inside], "o", ms=2.6, mfc="white", mec="0.25", mew=0.7, zorder=3)
    ax.plot(ext["model_x"], 100 * ext["best"], color=C_EXT, lw=1.1)
    ax.plot(pub["model_x"], 100 * pub["best"], color=C_PUB, lw=1.1, ls="--")
    change = 100 * (ext["time_d"] / pub["time_d"] - 1)
    sub = f"{pub['time_d']:.3g} d published → {ext['time_d']:.3g} d extended ({change:+.0f}%)"
    part = fits.get("extended to before feature")
    if part is not None:
        ax.plot(part["model_x"], 100 * part["best"], color=C_PART, lw=1.0, ls="-.")
        for a in (ax, ax_r):
            a.axvline(part["model_end_um"], color=C_PART, lw=0.8, ls=":")
        sub += f"; to {part['model_end_um']:.0f} µm: {part['time_d']:.3g} d"
    verdict = "flat plateau" if test["flat"] else f"not flat (mean {test['mean_s']:+.1f} s)"
    ax.set_title(f"{short(name)}: dropped points {verdict}\n{sub}", loc="left", fontsize=6.6)
    ax_r.axhspan(-2, 2, color="0.9", lw=0, zorder=0)
    ax_r.axhline(0, color=C_EXT, lw=0.8)
    lim = max(3.5, float(np.nanmax(np.abs(r))) + 0.7)
    ax_r.set_ylim(-lim, lim)
    ax.set_xlim(min(0, x.min()) - 2, x.max() + 3)
    ax_r.set_xlim(ax.get_xlim())
    ax.tick_params(labelbottom=False)


def figure(curves, out: Path):
    names = list(curves)
    ncol, nrow = 2, math.ceil(len(names) / 2)
    fig = plt.figure(figsize=(7.2, 2.35 * nrow + 0.75), layout="constrained")
    outer = fig.add_gridspec(nrow, ncol)
    for i, name in enumerate(names):
        inner = outer[i // ncol, i % ncol].subgridspec(2, 1, height_ratios=(2.4, 1), hspace=0.05)
        ax, ax_r = fig.add_subplot(inner[0]), fig.add_subplot(inner[1])
        _panel(ax, ax_r, *curves[name], name)
        ax.set_ylabel("Fo (mol%)")
        ax_r.set_ylabel("resid. / s")
        if i // ncol == nrow - 1 or i + ncol >= len(names):
            ax_r.set_xlabel("Distance from rim (µm)")
    handles = [
        mpl.lines.Line2D([], [], marker="o", ls="", ms=3, color="0.25",
                         label="point fitted in the published run"),
        mpl.lines.Line2D([], [], marker="o", ls="", ms=3, mfc="white", mec="0.25",
                         label="point outside the published model domain"),
        mpl.patches.Patch(color=C_DROP, label="dropped core-side points (past the blue dotted "
                          "line: published domain end)"),
        mpl.lines.Line2D([], [], color=C_PUB, ls="--", label="best fit, published domain"),
        mpl.lines.Line2D([], [], color=C_EXT, label="best fit, domain extended to the last point"),
        mpl.patches.Patch(color="0.9", label="lower strips: (data - extended fit) / white-noise "
                          "sd s; band ±2 s"),
    ]  # fmt: skip
    if any("extended to before feature" in c[0] for c in curves.values()):
        handles.append(mpl.lines.Line2D([], [], color=C_PART, ls="-.",
                       label="best fit, domain ended before the feature"))  # fmt: skip
    fig.legend(handles=handles, loc="outside upper center", ncol=2, fontsize=6.3)
    fig.savefig(out)
    plt.close(fig)


REPORT_COLS = ["run", "fit", "class", "model_end_um", "n_fitted", "core_fo_mol", "time_d",
               "lo1_d", "hi1_d", "lo2_d", "hi2_d", "near_step_daic", "s_mol", "a_mol", "L_um",
               "core_end_l", "time_change_pct", "ns_time_d", "ns_lo1_d", "ns_hi1_d", "ns_lo2_d",
               "ns_hi2_d"]  # fmt: skip
TEST_COLS = ["run", "published_domain_end_um", "data_end_um", "points_past_end",
             "plateau_start_um", "dropped_mean_s", "dropped_n_over_2s", "dropped_trend_s",
             "dropped_p_line", "dropped_p_runs", "dropped_shift_s", "dropped_p_shift",
             "core_fo_mol",
             "plateau_mean_published_mol", "dropped_mean_mol", "plateau_mean_all_mol"]  # fmt: skip


def survey(runs, profiles) -> pd.DataFrame:
    """Every paper run: model domain against the measured data at both ends."""
    rows = []
    for run in runs:
        x = np.asarray(profiles[run.profile_name].distance_um, float)
        lo, hi = run.initial.domain_um
        rows.append({"run": run.name, "domain_start_um": lo, "domain_end_um": hi,
                     "data_start_um": float(x.min()), "data_end_um": float(x.max()),
                     "points_before_start": int(np.sum(x < lo)),
                     "points_past_end": int(np.sum(x > hi)),
                     "past_end_um": float(x.max() - hi)})  # fmt: skip
    return pd.DataFrame(rows)


def main():
    runs, profiles = F.load_paper()
    survey(runs.values(), profiles).to_csv(HERE / "all_runs.csv", index=False, float_format="%.4g")
    table, curves = analyse(runs.values(), profiles)
    table.to_csv(HERE / "results.csv", index=False, float_format="%.6g")
    figure(curves, HERE / "profiles.png")
    with pd.option_context("display.width", 260, "display.max_columns", 40):
        print(table[[c for c in REPORT_COLS if c in table]].round(3).to_string(index=False))
        first = table.drop_duplicates("run")
        print(first[TEST_COLS].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
