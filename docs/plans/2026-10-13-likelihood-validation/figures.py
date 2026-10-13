"""Figures for docs/references/likelihood-method.md.

Reads coverage.csv, trials.csv and real_profiles.csv (from validate.py and
diagnostics.py) and refits one round robin profile for the worked example. Writes
PNG files to docs/references/figures/likelihood/. Run from the repo root:
    uv run python docs/plans/2026-10-13-likelihood-validation/figures.py
"""

# ruff: noqa: RUF001  (figure labels use typographic sigma, minus, times and dashes)
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adjustText import adjust_text

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
OUT = REPO / "docs" / "references" / "figures" / "likelihood"
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(REPO / "tests"))
from conftest import ROUND_ROBIN_DATA, ROUND_ROBIN_PARAMS  # noqa: E402

from forster import RunDefaults, fit_run, read_profiles, read_runs  # noqa: E402
from forster.uncertainty import (  # noqa: E402
    chi2_per_step,
    delta_chi2_interval,
    interpolation_matrix,
    interval_threshold,
)

OKABE_ITO = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
CORRELATED, WHITE, GREY = "#0072B2", "#D55E00", "0.55"
SCAN = "#CC79A7"  # Okabe-Ito reddish purple
mpl.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 9.5,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "0.92",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
        "savefig.dpi": 200,
        "pdf.fonttype": 42,
    }
)
DAY = 86400.0
coverage = pd.read_csv(HERE / "coverage.csv")
trials = pd.read_csv(HERE / "trials.csv")
real = pd.read_csv(HERE / "real_profiles.csv")


def save(fig, name):
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


# 1. Worked example ----------------------------------------------------------------
run = {r.name: r for r in read_runs(ROUND_ROBIN_PARAMS, defaults=RunDefaults(max_time_days=730))}[
    "KS20-527_2-transect-1"
]
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    res = fit_run(run, read_profiles(ROUND_ROBIN_DATA))
x, y, noise, dt = res.fit_x_um, res.fit_fo, res.noise, res.dt_s
W = interpolation_matrix(res.model_x_um, x)
curves = res.curves[: 3 * res.indices[0]]
rss = chi2_per_step(curves, W, y)
white_chi2 = rss / (rss.min() / len(y))
threshold = interval_threshold(len(y))  # the default rule, used for both curves
white = delta_chi2_interval(white_chi2, threshold, end_is_limit=False)
corr = res.sum_chi2[: len(curves)]
corr_span = delta_chi2_interval(corr, threshold, end_is_limit=False)
cov = noise.covariance(x)
resid = y - W @ res.curves[res.indices[0]]
xs = np.linspace(x.min(), x.max(), 400)
k_star = noise.correlated_sigma**2 * np.exp(
    -0.5 * (xs[:, None] - x[None, :]) ** 2 / noise.correlation_length_um**2
)
smooth = k_star @ np.linalg.solve(cov, resid)  # estimated smooth misfit (GP mean)

days = np.arange(len(curves)) * dt / DAY
w_lo, w_hi = white.low * dt / DAY, white.high * dt / DAY
c_lo, c_hi = corr_span.low * dt / DAY, corr_span.high * dt / DAY
fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.3), layout="constrained")
BELOW = {"loc": "upper center", "bbox_to_anchor": (0.5, -0.24), "ncol": 1}
ax = axes[0]
ax.plot(res.model_x_um, res.initial * 100, color=GREY, ls="--", lw=1, label="initial step")
ax.plot(
    res.model_x_um, res.best_fit * 100, color=CORRELATED, label=f"best fit, {res.time_days:.1f} d"
)
ax.plot(x, y * 100, "o", color="black", ms=3.5, label=f"EPMA ({len(x)} points)")
ax.set(xlabel="Distance from rim (µm)", ylabel="Fo (mol%)", title="a  Profile and best fit")
ax.legend(**BELOW)
ax = axes[1]
ax.axhline(0, color=GREY, lw=0.8)
ax.plot(x, resid * 100, "o", color="black", ms=3.5, label="residual")
ax.plot(xs, smooth * 100, color=CORRELATED, label="smooth misfit (fitted)")
ax.fill_between(
    xs,
    (smooth - noise.white_sigma) * 100,
    (smooth + noise.white_sigma) * 100,
    color=CORRELATED,
    alpha=0.15,
    lw=0,
    label="± white noise s",
)
ax.set(
    xlabel="Distance from rim (µm)",
    ylabel="Data − model (Fo mol%)",
    title=f"b  Residuals: s {noise.white_sigma * 100:.2f},\n"
    f"a {noise.correlated_sigma * 100:.2f} Fo, L {noise.correlation_length_um:.0f} µm",
)
ax.legend(**BELOW)
ax = axes[2]
ax.plot(days, white_chi2 - white_chi2.min(), color=WHITE, ls="--", label="independent errors")
ax.plot(days, corr - corr.min(), color=CORRELATED, label="correlated errors")
ax.axhline(threshold, color=GREY, lw=0.8)
ax.annotate(
    f"Δχ² = {threshold:.2f}",
    (0.42 * res.time_days, threshold),
    xytext=(2, 2),
    textcoords="offset points",
    ha="left",
    va="bottom",
    fontsize=8,
    color="0.35",
)
for lo, hi, color in ((w_lo, w_hi, WHITE), (c_lo, c_hi, CORRELATED)):
    ax.axvspan(lo, hi, color=color, alpha=0.12, lw=0)
ax.set(
    ylim=(0, 6),
    xlim=(0.4 * res.time_days, 1.6 * res.time_days),
    xlabel="Time (days)",
    ylabel="χ² − minimum",
    title=f"c  χ² against time\n1σ: {w_lo:.0f}–{w_hi:.0f} d → {c_lo:.0f}–{c_hi:.0f} d",
)
ax.legend(**BELOW)
fig.suptitle(
    "KS20-527_2-transect-1: correlated residuals widen the 1σ range "
    f"{(c_hi - c_lo) / (w_hi - w_lo):.1f}×",
    x=0.0,
    ha="left",
    fontsize=10,
    fontweight="bold",
)
save(fig, "fig1_example.png")

# 2. Coverage --------------------------------------------------------------------
geometries = list(dict.fromkeys(coverage.geometry))
panels = [
    "white only",
    "smooth misfit, L = 3 um",
    "smooth misfit, L = 10 um",
    "smooth misfit, L = 30 um",
    "step shifted +3 um",
    "edge Fo +0.003",
]
titles = [
    "White\nnoise",
    "+ misfit\nL = 3 µm",
    "+ misfit\nL = 10 µm",
    "+ misfit\nL = 30 µm",
    "Step\n3 µm off",
    "Edge Fo\n0.3 off",
]
n_trials = int(coverage.trials.iloc[0])
band = 196 * np.sqrt(0.68 * 0.32 / n_trials)  # 95% sampling band around 68%, in %
smooth_rows = coverage[coverage.noise.str.startswith("smooth")]
fig, axes = plt.subplots(1, len(panels), figsize=(7.2, 3.4), sharey=True, layout="constrained")
ypos = {g: i for i, g in enumerate(reversed(geometries))}
for ax, noise_name, title in zip(axes, panels, titles, strict=True):
    sub = coverage[coverage.noise == noise_name]
    ax.axvspan(68 - band, 68 + band, color="0.9", lw=0)
    ax.axvline(68, color=GREY, lw=0.8)
    yy = [ypos[g] for g in sub.geometry]
    ax.plot(
        sub.white_coverage * 100,
        yy,
        "o",
        mfc="white",
        mec=WHITE,
        mew=1.2,
        ls="none",
        label="independent errors",
    )
    ax.plot(sub.coverage * 100, yy, "o", color=CORRELATED, ls="none", label="correlated errors")
    ax.set(xlim=(0, 100), title=title, xticks=[0, 50, 100])
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(range(len(geometries)))
axes[0].set_yticklabels([g.replace(" um", " µm") for g in reversed(geometries)])
axes[2].set_xlabel("Coverage of the true time (%)", loc="left")
fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2)
dense = smooth_rows[smooth_rows.n_points >= 29]
lo_c, hi_c = dense.coverage.min() * 100, dense.coverage.max() * 100
lo_w, hi_w = dense.white_coverage.min() * 100, dense.white_coverage.max() * 100
step = coverage[coverage.noise == "step shifted +3 um"].coverage * 100
fig.suptitle(
    f"With ≥29 points and smooth misfit, coverage is {lo_c:.0f}–{hi_c:.0f}% (target 68%) against "
    f"{lo_w:.0f}–{hi_w:.0f}% for\nindependent errors. A 3 µm step error is not covered "
    f"({step.min():.0f}–{step.max():.0f}%).",
    x=0.0,
    ha="left",
    fontsize=10,
    fontweight="bold",
)
save(fig, "fig2_coverage.png")

# 3. Correlation length recovery ---------------------------------------------------
smooth_trials = trials[trials.noise.str.startswith("smooth")]
fig, grid = plt.subplots(2, 3, figsize=(7.2, 4.6), sharex=True, sharey=True, layout="constrained")
axes = grid.ravel()
for ax, g in zip(axes, geometries, strict=True):
    sub = smooth_trials[smooth_trials.geometry == g]
    spacing = sub.spacing_um.iloc[0]
    ax.axhspan(0.3, spacing, color="0.93", lw=0)
    ax.plot([1, 100], [1, 100], color=GREY, lw=0.8)
    for true_L, s in sub.groupby("true_L_um"):
        found = s[s.fit_corr > 0]
        share = len(found) / len(s)
        if len(found):
            q = found.fit_L_um.quantile([0.25, 0.5, 0.75])
            ax.errorbar(
                true_L,
                q[0.5],
                yerr=[[q[0.5] - q[0.25]], [q[0.75] - q[0.5]]],
                fmt="o",
                color=CORRELATED,
                ms=4,
                capsize=2,
                lw=1,
            )
        ax.annotate(f"{share:.0%}", (true_L, 0.42), ha="center", fontsize=7, color="0.35")
    ax.set(
        xscale="log", yscale="log", xlim=(1.5, 60), ylim=(0.3, 300), title=g.replace(" um", " µm")
    )
    ax.title.set_fontsize(8.5)
for ax in grid[:, 0]:
    ax.set_ylabel("Fitted L (µm)")
fig.supxlabel(
    "True correlation length L (µm). Dots: median fitted L, bars: IQR; grey band: "
    "below the point spacing;\n% = share of trials where a correlated term was found",
    fontsize=8,
)
fig.suptitle(
    "L is recovered once it exceeds the point spacing; shorter misfit is absorbed as white noise",
    x=0.0,
    ha="left",
    fontsize=10,
    fontweight="bold",
)
save(fig, "fig3_length.png")

# 4. Real profiles -----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), layout="constrained")
ax = axes[0]
sets = {"round robin": ("o", OKABE_ITO[5]), "Towbin and Plank 2026": ("s", OKABE_ITO[1])}
found = real[real.corr_sigma_fo > 0]
ax.plot([0.3, 30], [0.3, 30], color=GREY, lw=0.8)
for label, (marker, color) in sets.items():
    s = found[found.set == label]
    ax.plot(s.spacing_um, s.L_um, marker, color=color, ls="none", label=label, alpha=0.85)
long = found[found.L_over_diffusion_length > 3]
texts = [
    ax.text(
        row.spacing_um,
        row.L_um,
        row.run.replace("AZ18 ", "").replace("_prof_1", "").replace("_prof_2", ""),
        fontsize=6.5,
        color="0.3",
    )
    for _, row in long.iterrows()
]
n_ge = int((found.L_um >= 0.999 * found.spacing_um).sum())
ax.set(
    xscale="log",
    yscale="log",
    xlabel="Median point spacing (µm)",
    ylabel="Fitted L (µm)",
    title=f"a  {len(found)} of {len(real)} profiles need a\ncorrelated term; "
    f"{n_ge} have L ≥ spacing.\nLabelled: L > 3× diffusion length",
    xlim=(0.3, 30),
)
ax.title.set_fontsize(8.5)
ax.legend(loc="lower right")
adjust_text(
    texts,
    ax=ax,
    x=found.spacing_um.to_numpy(),
    y=found.L_um.to_numpy(),
    expand=(1.4, 1.8),
    arrowprops={"arrowstyle": "-", "lw": 0.5, "color": "0.5"},
)
ax = axes[1]
order = real.sort_values("lag1_raw").reset_index(drop=True)
yy = np.arange(len(order))
ax.hlines(yy, order.lag1_whitened, order.lag1_raw, color="0.8", lw=1)
ax.plot(order.lag1_raw, yy, "o", mfc="white", mec=WHITE, mew=1.1, ls="none", label="raw residuals")
ax.plot(order.lag1_whitened, yy, "o", color=CORRELATED, ls="none", label="whitened")
ax.axvline(0, color=GREY, lw=0.8)
ax.set(
    yticks=[],
    xlabel="Lag-1 autocorrelation of residuals",
    ylabel=f"{len(real)} real profiles",
    xlim=(-0.5, 1.0),
)
n_raw = int((order.lag1_raw.abs() > 0.3).sum())
n_white = int((order.lag1_whitened.abs() > 0.3).sum())
ax.set_title(
    f"b  Profiles with |lag-1| > 0.3:\n{n_raw} raw, {n_white} after whitening", fontsize=8.5
)
ax.legend(loc="lower right")
fig.suptitle(
    "Real profiles: the fitted noise model leaves near-white residuals",
    x=0.0,
    ha="left",
    fontsize=10,
    fontweight="bold",
)
save(fig, "fig4_real_profiles.png")
print("figures written to", OUT)

# 5. Interval rules and the initial-condition scan ----------------------------------
RULES = {
    "delta1_coverage": ("Δχ² = 1", "o", "white", WHITE),
    "coverage": ("small-sample (default)", "o", CORRELATED, CORRELATED),
    "posterior_coverage": ("Bayesian posterior", "D", "white", OKABE_ITO[3]),
}
fig, axes = plt.subplots(
    1,
    3,
    figsize=(7.2, 3.4),
    sharey=False,
    layout="constrained",
    gridspec_kw={"width_ratios": [1, 1, 1.1]},
)
for ax, noise_name, title in zip(
    axes[:2],
    ["white only", "smooth misfit, L = 30 um"],
    ["a  White noise", "b  Misfit, L 30 µm"],
    strict=True,
):
    sub = coverage[coverage.noise == noise_name]
    yy = [ypos[g] for g in sub.geometry]
    ax.axvspan(68 - band, 68 + band, color="0.9", lw=0)
    ax.axvline(68, color=GREY, lw=0.8)
    for col, (label, marker, face, edge) in RULES.items():
        ax.plot(sub[col] * 100, yy, marker, mfc=face, mec=edge, mew=1.2, ls="none", label=label)
    ax.set(xlim=(40, 80), title=title, xlabel="Coverage (%)")
    ax.grid(axis="y", visible=False)
axes[0].set_yticks(range(len(geometries)))
axes[0].set_yticklabels(
    [g.replace(" um", " µm").replace(", measured", "") for g in reversed(geometries)]
)
axes[1].set_yticks(range(len(geometries)))
axes[1].set_yticklabels([])
ax = axes[2]
sys_rows = coverage[coverage.noise.isin(["step shifted +3 um", "edge Fo +0.003"])].reset_index()
labels = [
    f"{'step +3 µm' if r.noise.startswith('step') else 'edge +0.3 Fo'}, {int(r.n_points)} pts"
    for r in sys_rows.itertuples()
]
yy = np.arange(len(sys_rows))[::-1]
ax.axvspan(68 - band, 68 + band, color="0.9", lw=0)
ax.axvline(68, color=GREY, lw=0.8)
ax.plot(sys_rows.coverage * 100, yy, "o", color=CORRELATED, ls="none", label="nominal fit")
ax.plot(
    sys_rows.scan_profile_coverage * 100,
    yy,
    "s",
    mfc="white",
    mec=SCAN,
    mew=1.2,
    ls="none",
    label="scan, profile",
)
ax.plot(
    sys_rows.scan_marginal_coverage * 100, yy, "^", color=SCAN, ls="none", label="scan, marginal"
)
ax.set(xlim=(0, 100), yticks=yy, title="c  Wrong start", xlabel="Coverage (%)")
ax.set_yticklabels(labels)
ax.yaxis.tick_right()
ax.grid(axis="y", visible=False)
h1, l1 = axes[0].get_legend_handles_labels()
h2, l2 = ax.get_legend_handles_labels()
fig.legend(h1 + h2[1:], l1 + l2[1:], loc="outside lower center", ncol=5, fontsize=7.5)
sparse = coverage[(coverage.noise == "white only") & (coverage.n_points == 6)].iloc[0]
step_scan = sys_rows[sys_rows.noise.str.startswith("step")]
fig.suptitle(
    f"The small-sample threshold lifts 6-point coverage from {sparse.delta1_coverage * 100:.0f}% "
    f"to {sparse.coverage * 100:.0f}%;\nscanning the step lifts a 3 µm step error from "
    f"{step_scan.coverage.min() * 100:.0f}–{step_scan.coverage.max() * 100:.0f}% to "
    f"{step_scan.scan_profile_coverage.min() * 100:.0f}–"
    f"{step_scan.scan_profile_coverage.max() * 100:.0f}% (profile range)",
    x=0.0,
    ha="left",
    fontsize=9.5,
    fontweight="bold",
)
save(fig, "fig5_rules_and_scan.png")

# 6. Methods on real profiles ---------------------------------------------------------
methods = pd.read_csv(HERE / "real_methods.csv")
order = [
    "legacy",
    "delta chi2 = 1",
    "small-sample (default)",
    "Bayesian posterior",
    "step scan, profile",
    "step scan, marginal",
]
style = {
    "legacy": ("0.5", "o"),
    "delta chi2 = 1": (WHITE, "o"),
    "small-sample (default)": (CORRELATED, "o"),
    "Bayesian posterior": (OKABE_ITO[3], "D"),
    "step scan, profile": (SCAN, "s"),
    "step scan, marginal": (SCAN, "^"),
}
profiles_shown = list(dict.fromkeys(methods.run))
fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), layout="constrained")
for ax, name in zip(axes.ravel(), profiles_shown, strict=True):
    sub = methods[methods.run == name].set_index("method").loc[order]
    yy = np.arange(len(order))[::-1]
    for y_, (m, row) in zip(yy, sub.iterrows(), strict=True):
        color, marker = style[m]
        ax.hlines(y_, row.low_days, row.high_days, color=color, lw=2)
        ax.plot(row.best_days, y_, marker, color=color, ms=4)
    ax.set_yticks(yy)
    ax.set_yticklabels(order if ax in axes[:, 0] else [], fontsize=7.5)
    n = int(sub.n_points.iloc[0])
    short = (
        name.replace("AZ18 ", "").replace("_prof_1", "").replace("KS20-527_2-transect-", "KS2 t")
    )
    ax.set_title(f"{short} ({n} pts)", fontsize=8)
    ax.set_xlabel("Time (days)", fontsize=8)
    ax.grid(axis="y", visible=False)
ks = methods[methods.run == "KS20-527_2-transect-1"].set_index("method")
fig.suptitle(
    "Where the step position is uncertain it dominates: on KS20-527_2-transect-1\n"
    f"the scan moves the time from {ks.loc['small-sample (default)', 'best_days']:.0f} d (step "
    f"{ks.loc['step scan, marginal', 'nominal_steps_um']} µm) to "
    f"{ks.loc['step scan, profile', 'best_days']:.0f} d (step "
    f"{ks.loc['step scan, marginal', 'best_steps_um']} µm)",
    x=0.0,
    ha="left",
    fontsize=9.5,
    fontweight="bold",
)
save(fig, "fig6_real_methods.png")
print("figures 5-6 written")
