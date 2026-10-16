# ruff: noqa: RUF001  # sigma, minus, en dash and times signs are wanted in figure text
"""Figures for the noise-models guide (docs/references/noise-models-guide.md).

What each kind of misfit looks like in a FORSTER fit, and which noise model suits which
real profile. Run from the repository root:

    uv run python docs/references/figures/noise_models/make_figures.py            # public
    uv run python docs/references/figures/noise_models/make_figures.py --private  # round robin

Public figures (this folder) use only synthetic profiles and published Towbin and Plank
(2026) profiles (analyses/xenolith_archive):

- fig1_gallery_stationary.png, fig2_gallery_other.png: one synthetic geometry (30 points
  2.5 um apart, true time 3.6 d, diffusion length 10 um), one column per kind of
  misfit; gallery_coverage.csv holds the Monte Carlo numbers printed on them;
- fig3_real_models.png: paper profiles where white noise is enough, where a smooth
  misfit is needed, and where the misfit is much longer than the diffusion length;
- fig4_real_near_step.png: paper profiles where the near-step term is clearly preferred,
  fitted with the default and the near-step noise model; real_examples.csv holds the
  numbers.

``--private`` makes the same views for round robin profiles (other labs' unpublished
data). They are written only to ``Diffusion Round Robin/outputs/noise_models/``, inside
the private submodule, and never to this folder.

Noise models (docs/references/likelihood-method.md, section 2):
- default: C = s^2 I + a^2 K_L, K_L squared-exponential, fitted by maximum likelihood at
  the white (least-squares) best fit;
- near-step: the misfit amplitude may grow near the steps,
  C = s^2 (I + rho^2 S K_L S), S = diag(sqrt(1 + h e)), e a Gaussian bump of width l
  (the diffusion length at the white best fit) around each step
  (docs/plans/2026-10-15-whitening-validity/zone_noise.py, mode "corr"; the same model
  as `forster.uncertainty.near_step_check`).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import sys  # noqa: E402
import warnings  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib as mpl  # noqa: E402

mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy.linalg as sla  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
STUDY = REPO / "docs" / "plans" / "2026-10-15-whitening-validity"
PRIVATE_OUT = REPO / "Diffusion Round Robin" / "outputs" / "noise_models"
sys.path.insert(0, str(STUDY))
sys.path.insert(0, str(REPO / "tests"))

from common import kernel_matrix, whiten  # noqa: E402
from synthetic import AMP, CASES, make_noise  # noqa: E402
from zone_noise import ZONE_LENGTHS, _ranges, bump, fit_zone_noise  # noqa: E402

import forster  # noqa: E402
from forster.grid import diffusion_length_um, fo_range  # noqa: E402
from forster.model import Grid, InitialCondition, Profile  # noqa: E402
from forster.uncertainty import (  # noqa: E402
    fit_correlated_points,
    interpolation_matrix,
    near_step_check,
    points_in_domain,
    white_rss,
)

DAY = 86400.0
OKABE = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
C_DEFAULT = OKABE[5]  # blue: default noise model
C_NEAR = OKABE[6]  # vermillion: near-step noise model
C_MISFIT = OKABE[3]  # green: fitted smooth misfit
C_ZONE = "#F6EFA6"  # pale yellow: within 2 diffusion lengths of a step
C_TRUE = "0.35"

mpl.rcParams.update(
    {
        "font.size": 7.5,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "0.92",
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "lines.linewidth": 1.1,
        "savefig.dpi": 200,
        "pdf.fonttype": 42,
    }
)

# ----------------------------------------------------------------------------- analysis


@dataclass
class Analysis:
    """One profile fitted with the default and the near-step noise model."""

    name: str
    x: np.ndarray  # fitted points (um), sorted
    y: np.ndarray  # Fo fraction
    at: np.ndarray  # model at the points, rows x points
    model_x: np.ndarray
    curves: np.ndarray  # model on its grid, every row
    dt_s: float
    steps: tuple[float, ...]
    ell_best: float  # diffusion length at the default best fit (um)
    default: object  # forster.uncertainty.CorrelatedFit
    ns_cov: np.ndarray
    ns_h: float
    ns_daic: float
    ns_i1: object
    ns_i2: object

    def days(self, row: float) -> float:
        return float(row) * self.dt_s / DAY

    @property
    def near(self) -> np.ndarray:
        d = np.abs(self.x[:, None] - np.asarray(self.steps)[None, :]).min(axis=1)
        return d <= ZONE_LENGTHS * self.ell_best

    @property
    def r_default(self) -> np.ndarray:
        return self.y - self.at[self.default.interval.best]

    @property
    def z_default(self) -> np.ndarray:
        return whiten(self.default.noise.covariance(self.x), self.r_default)

    @property
    def z_near(self) -> np.ndarray:
        return whiten(self.ns_cov, self.y - self.at[self.ns_i1.best])

    def misfit_mean(self) -> np.ndarray:
        """Smooth misfit the default model sees in the residuals: a^2 K C^-1 r."""
        nm = self.default.noise
        if nm.correlated_sigma <= 0:
            return np.zeros_like(self.x)
        K = kernel_matrix("sqexp", self.x, nm.correlation_length_um)
        cov = nm.covariance(self.x)
        return nm.correlated_sigma**2 * K @ sla.cho_solve(sla.cho_factor(cov), self.r_default)


def analyse(name, at, x, y, model_x, curves, dt_s, steps, length_at_row) -> Analysis:
    """Default fit, and the near-step model fitted at the white best fit (as the warning
    in `forster.model._near_step` does). ``length_at_row(row)``: diffusion length (um)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        d = fit_correlated_points(at, x, y, end_is_limit=False)
    w_row = int(np.argmin(white_rss(at, y)))
    ell_w = length_at_row(w_row)
    r_w = y - at[w_row]
    ns_cov, h, daic = fit_zone_noise(x, r_w, bump(x, steps, ell_w), "corr")
    check = near_step_check(x, r_w, steps, ell_w)  # what FORSTER reports
    if not np.isclose(check.delta_aic, daic, atol=1e-6):
        raise RuntimeError(f"{name}: near-step dAIC {daic} differs from forster's {check}")
    i1, i2 = _ranges(at, y, ns_cov)
    return Analysis(
        name, x, y, at, model_x, curves, dt_s, tuple(steps),
        length_at_row(d.interval.best), d, ns_cov, h, daic, i1, i2,
    )  # fmt: skip


def analyse_run(run, profiles) -> Analysis:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = forster.fit_run(run, profiles, keep_curves=True)
    x, y = points_in_domain(res.model_x_um, res.fit_x_um, res.fit_fo)
    o = np.argsort(x)
    x, y = x[o], y[o]
    at = res.curves @ interpolation_matrix(res.model_x_um, x).T
    species = [(run.diffusivity, fo_range(run.initial))]

    def length(row):
        return diffusion_length_um(max(row, 1) * res.dt_s / DAY, species, run.conditions,
                                   run.orientation)  # fmt: skip

    a = analyse(run.name, at, x, y, res.model_x_um, res.curves, res.dt_s,
                run.initial.steps_um, length)  # fmt: skip
    if not np.isclose(a.days(a.default.interval.best), res.time_days, rtol=1e-6):
        raise RuntimeError(f"{run.name}: refit {a.days(a.default.interval.best)} d "
                           f"!= fit_run {res.time_days} d")  # fmt: skip
    reported = res.near_step_delta_aic
    if reported is not None and not np.isclose(reported, a.ns_daic, atol=1e-6):
        print(f"note: {run.name}: near-step dAIC {a.ns_daic:.3f} here, {reported:.3f} in fit_run")
    return a


# ------------------------------------------------------------------- synthetic geometry

SYN_N = 30
SYN_DX = 2.5
SYN_ELL = 10.0  # diffusion length at the true time (um)
SYN_STEP = 25.0
SYN_WHITE = 0.001  # 0.1 mol% Fo, the median white noise fitted to the paper profiles
SYN_SEED = 20261016
GALLERY = {
    # study case name (synthetic.CASES) -> (panel title, short description)
    "white (control)": ("White noise only", "sd 0.1 mol%"),
    "sq-exp L=3 (control)": ("Smooth misfit, L = 3 µm", "shorter than l (10 µm)"),
    "sq-exp L=10 (control)": ("Smooth misfit, L = 10 µm", "about l: trades off with t"),
    "exponential L=10": ("Rough misfit, L = 10 µm", "exponential kernel"),
    "strong near-step L=3": ("Misfit near the step", "peak 1.2 mol%, L = 3 µm"),
    "1-2 outliers (5 sd)": ("1–2 outliers", "shifted by 5 × white sd"),
    "sd rises 4x to the rim": ("Noise grows to the rim", "sd 4× larger at the rim"),
}
STATIONARY = list(GALLERY)[:4]
OTHER = list(GALLERY)[4:]


def synthetic_geometry() -> dict:
    """The forward model for one realistic geometry (conditions of the packaged example,
    a normally zoned 4 mol% step), every time step to 3x the true time."""
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        forster.write_template(tmp)
        run = forster.read_runs(Path(tmp) / "parameters.csv")[0]
    initial = InitialCondition(domain_um=(0.0, 90.0), edge_fo=0.86, core_fo=0.90,
                               steps_um=(SYN_STEP,))  # fmt: skip
    species = [(run.diffusivity, fo_range(initial))]
    per_day = diffusion_length_um(1.0, species, run.conditions, run.orientation) ** 2
    t_days = SYN_ELL**2 / per_day
    k_true = 400
    dt = t_days * DAY / k_true
    grid = Grid(dx_um=SYN_ELL / 8, dt_s=dt, max_time_days=3 * t_days + dt / DAY)
    x = SYN_DX * (1 + np.arange(SYN_N))
    prof = Profile(x, np.full_like(x, 0.88), name="synthetic")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = forster.fit_profile(prof, run.conditions, run.orientation, initial, grid,
                                  diffusivity=run.diffusivity, scheme="implicit",
                                  stop_early=False, keep_curves=True,
                                  check_near_step=False)  # fmt: skip
    at = res.curves @ interpolation_matrix(res.model_x_um, x).T
    return {
        "x": x, "white": SYN_WHITE, "steps": np.array([SYN_STEP]), "core_fo": 0.90,
        "diffusion_length_um": SYN_ELL, "at_points": at, "k_true": k_true,
        "model_x": res.model_x_um, "curves": res.curves, "dt_s": dt, "t_days": t_days,
        "length": lambda row: SYN_ELL * np.sqrt(max(row, 1) / k_true),
    }  # fmt: skip


def synthetic_trial(g, case, trial) -> Analysis:
    rng = np.random.default_rng([SYN_SEED, trial])  # same white draw for every case
    y = g["at_points"][g["k_true"]] + make_noise(g, case, rng)
    return analyse(case, g["at_points"], g["x"], y, g["model_x"], g["curves"], g["dt_s"],
                   tuple(g["steps"]), g["length"])  # fmt: skip


def outlier_points(g, case, trial) -> np.ndarray:
    """Indices of the points ``make_noise`` shifted (outlier case only)."""
    if CASES[case][0] != "outliers":
        return np.array([], dtype=int)
    e = make_noise(g, case, np.random.default_rng([SYN_SEED, trial]))
    e0 = make_noise(g, "white (control)", np.random.default_rng([SYN_SEED, trial]))
    return np.flatnonzero(np.abs(e - e0) > 1e-12)


def true_sd(g, case) -> np.ndarray:
    """Standard deviation of the noise actually added at each point (Fo fraction)."""
    x, s = g["x"], float(g["white"])
    kind, p = CASES[case]
    if kind in ("white", "outliers", "student"):
        return np.full_like(x, s)
    if kind == "misfit":
        return np.full_like(x, np.hypot(s, p.get("amp", AMP)))
    if kind == "envelope":
        env = np.exp(-0.5 * ((x[:, None] - g["steps"][None, :]) / SYN_ELL) ** 2).max(axis=1)
        return np.hypot(s, p.get("amp", AMP) * env)
    if kind == "hetero_rim":
        w = 1 + 3 * (x.max() - x) / (x.max() - x.min())
        return s * w / np.sqrt(np.mean(w**2))
    raise ValueError(case)


def covers(span, k) -> bool:
    return bool(span.low <= k <= span.high)


def gallery_coverage(g, trials) -> pd.DataFrame:
    rows = []
    k = g["k_true"]
    for case in GALLERY:
        for t in range(trials):
            a = synthetic_trial(g, case, t)
            d = a.default
            rows.append(
                {
                    "case": case, "trial": t,
                    "covered": covers(d.interval, k), "covered2": covers(d.interval2, k),
                    "near_covered": covers(a.ns_i1, k), "near_covered2": covers(a.ns_i2, k),
                    "near_step_daic": a.ns_daic,
                    "width": (d.interval.high - d.interval.low) / k,
                    "near_width": (a.ns_i1.high - a.ns_i1.low) / k,
                    "white_sigma": d.noise.white_sigma,
                    "correlated_sigma": d.noise.correlated_sigma,
                    "L_um": d.noise.correlation_length_um,
                }
            )  # fmt: skip
        print(f"  {case}: done", flush=True)
    t = pd.DataFrame(rows)
    summ = t.groupby("case", sort=False).agg(
        trials=("trial", "size"),
        coverage_1s=("covered", "mean"),
        coverage_2s=("covered2", "mean"),
        near_step_coverage_1s=("near_covered", "mean"),
        near_step_coverage_2s=("near_covered2", "mean"),
        share_daic_lt_m6=("near_step_daic", lambda v: (v < -6).mean()),
        median_width_1s=("width", "median"),
        median_near_step_width_1s=("near_width", "median"),
        median_white_sigma=("white_sigma", "median"),
        median_correlated_sigma=("correlated_sigma", "median"),
        median_L_um=("L_um", "median"),
    )
    # The draw shown in the gallery: the first whose near-step dAIC is closest to the
    # case's median, so the diagnostic shown is typical of that kind of misfit.
    dev = (t.near_step_daic - t.groupby("case").near_step_daic.transform("median")).abs()
    shown = t.assign(dev=dev).sort_values(["dev", "trial"]).groupby("case").trial.first()
    summ["shown_trial"] = shown
    return summ.reset_index()


# ------------------------------------------------------------------------------ drawing


def pct(v: np.ndarray) -> np.ndarray:
    return 100 * np.asarray(v)


def msize(a: Analysis) -> float:
    """Marker size: smaller for dense profiles, so neighbouring points stay apart."""
    return 2.4 if len(a.x) <= 60 else 1.3


def shade_zone(ax, a: Analysis):
    for s in a.steps:
        ax.axvspan(s - ZONE_LENGTHS * a.ell_best, s + ZONE_LENGTHS * a.ell_best,
                   color=C_ZONE, lw=0, zorder=0)  # fmt: skip


def band(a: Analysis, span) -> tuple[np.ndarray, np.ndarray]:
    lo = int(np.floor(span.low))
    hi = min(int(np.ceil(span.high)), len(a.curves) - 1)
    block = a.curves[lo : hi + 1]
    return block.min(axis=0), block.max(axis=0)


def draw_profile(ax, a: Analysis, truth=None, near=False):
    xlim = (min(a.x.min(), 0.0), a.x.max() + 0.03 * np.ptp(a.x))
    m = (a.model_x >= xlim[0] - 1) & (a.model_x <= xlim[1] + 1)
    mx = a.model_x[m]
    for span, alpha in ((a.default.interval2, 0.18), (a.default.interval, 0.4)):
        lo, hi = band(a, span)
        ax.fill_between(mx, pct(lo[m]), pct(hi[m]), color=C_DEFAULT, alpha=alpha, lw=0)
    if near:
        lo, hi = band(a, a.ns_i1)
        ax.fill_between(mx, pct(lo[m]), pct(hi[m]), color=C_NEAR, alpha=0.3, lw=0)
        ax.plot(mx, pct(a.curves[a.ns_i1.best][m]), color=C_NEAR, lw=1.0, ls=(0, (4, 1.5)))
    if truth is not None:
        ax.plot(mx, pct(truth[m]), color=C_TRUE, lw=1.0, ls=(0, (1.2, 1.2)))
    ax.plot(mx, pct(a.curves[a.default.interval.best][m]), color=C_DEFAULT, lw=1.0)
    ax.plot(a.x, pct(a.y), "o", ms=msize(a), mfc="white", mec="black", mew=0.6, zorder=5)
    for s in a.steps:
        ax.axvline(s, color="0.6", lw=0.5, ls=":", zorder=0)
    ax.set_xlim(*xlim)


def draw_raw(ax, a: Analysis, sd=None, outliers=()):
    shade_zone(ax, a)
    s = a.default.noise.white_sigma
    ax.axhspan(-pct(s), pct(s), color="0.85", lw=0, zorder=0.5)
    ax.axhline(0, color="0.5", lw=0.5)
    if sd is not None:
        for sign in (-1, 1):
            ax.plot(a.x, sign * 2 * pct(sd), color=C_TRUE, lw=0.7, ls=(0, (3, 2)))
    ax.plot(a.x, pct(a.misfit_mean()), color=C_MISFIT, lw=1.0)
    ax.plot(a.x, pct(a.r_default), "o", ms=msize(a), color="black", zorder=5)
    if len(outliers):
        ax.plot(a.x[outliers], pct(a.r_default[outliers]), "o", ms=6, mfc="none",
                mec=OKABE[7], mew=1.0, zorder=6)  # fmt: skip


def log_time_axis(ax, lo, hi):
    """Log axis from lo to hi with plain-number ticks at 1, 2 and 5 per decade."""
    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.xaxis.set_major_locator(mpl.ticker.LogLocator(base=10, subs=(1.0, 2.0, 5.0)))
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())


def draw_white(ax, a: Analysis, near=False):
    shade_zone(ax, a)
    for v in (-2, 2):
        ax.axhline(v, color="0.6", lw=0.5, ls="--")
    ax.axhline(0, color="0.5", lw=0.5)
    ax.plot(a.x, a.z_default, "o", ms=msize(a), color=C_DEFAULT, zorder=5)
    if near:
        ax.plot(a.x, a.z_near, "^", ms=msize(a) + 0.2, mfc="none", mec=C_NEAR, mew=0.7, zorder=6)


def draw_times(ax, a: Analysis, scale: float, near: bool, true_row=None):
    """Best fit (dot), 1-sigma (thick) and 2-sigma (thin) ranges, in units of ``scale``
    rows (the true time) or days (scale = rows per day)."""
    items = [("default", C_DEFAULT, a.default.interval, a.default.interval2)]
    if near:
        items.append(("near-step", C_NEAR, a.ns_i1, a.ns_i2))
    for j, (_, c, i1, i2) in enumerate(items):
        yv = -j
        ax.plot([i2.low / scale, i2.high / scale], [yv, yv], color=c, lw=1.0,
                solid_capstyle="butt")  # fmt: skip
        ax.plot([i1.low / scale, i1.high / scale], [yv, yv], color=c, lw=4.0, alpha=0.6,
                solid_capstyle="butt")  # fmt: skip
        ax.plot([i1.best / scale], [yv], "o", ms=3.2, color=c, mec="white", mew=0.5)
    if true_row is not None:
        ax.axvline(true_row / scale, color=C_TRUE, lw=0.9, ls=(0, (1.2, 1.2)))
    ax.set_yticks([-j for j in range(len(items))], [it[0] for it in items])
    ax.set_ylim(-len(items) + 0.4, 0.6)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)


def fmt_daic(v: float) -> str:
    return f"{v:+.1f}".replace("-", "−")


def noise_text(a: Analysis) -> str:
    nm = a.default.noise
    if nm.correlated_sigma > 0:
        corr = f"a = {pct(nm.correlated_sigma):.2f} mol%, L = {nm.correlation_length_um:.1f} µm"
    else:
        corr = "no correlated term (a = 0)"
    return f"s = {pct(nm.white_sigma):.2f} mol%, {corr}"


# ------------------------------------------------------------------------------ gallery


def gallery_figure(g, summ: pd.DataFrame, cases, path: Path, title: str, share_raw=True):
    k = g["k_true"]
    truth = g["curves"][k]
    fig = plt.figure(figsize=(7.2, 7.6), layout="constrained")
    gs = fig.add_gridspec(5, len(cases), height_ratios=[1.35, 1.0, 1.0, 0.42, 0.62])
    axes = np.empty((5, len(cases)), dtype=object)
    for j in range(len(cases)):
        for i in range(5):
            share = {}
            if j > 0 and i < 4 and (share_raw or i != 1):
                share = {"sharey": axes[i, 0]}
            axes[i, j] = fig.add_subplot(gs[i, j], **share)
    for j, case in enumerate(cases):
        row = summ.set_index("case").loc[case]
        shown = int(row.shown_trial)
        a = synthetic_trial(g, case, shown)
        head, sub = GALLERY[case]
        ax = axes[0, j]
        draw_profile(ax, a, truth=truth)
        ax.set_title(f"{head}\n{sub}", fontsize=7, loc="left")
        draw_raw(axes[1, j], a, sd=true_sd(g, case), outliers=outlier_points(g, case, shown))
        draw_white(axes[2, j], a)
        for i in (1, 2):
            axes[i, j].set_xlim(*ax.get_xlim())
        for i in range(3):
            axes[i, j].set_xticks([0, 25, 50, 75])
        draw_times(axes[3, j], a, k, near=True, true_row=k)
        log_time_axis(axes[3, j], 0.1, 10)
        covered = "yes" if covers(a.default.interval, k) else "NO"
        nm = a.default.noise
        corr = (f"a {pct(nm.correlated_sigma):.2f}, L {nm.correlation_length_um:.0f} µm"
                if nm.correlated_sigma > 0 else "a 0 (white only)")  # fmt: skip
        warn = "  (warns)" if a.ns_daic < -6 else ""
        lines = [
            f"Fit: s {pct(nm.white_sigma):.2f} mol%, {corr}",
            f"Near-step ΔAIC {fmt_daic(a.ns_daic)}{warn}",
            f"1σ covers true time: {covered}",
            f"In {int(row.trials)} draws like this one:",
            f"  default 1σ covers {100 * row.coverage_1s:.0f}%, 2σ {100 * row.coverage_2s:.0f}%",
            f"  near-step 1σ {100 * row.near_step_coverage_1s:.0f}%, "
            f"2σ {100 * row.near_step_coverage_2s:.0f}%",
            f"  warning fires in {100 * row.share_daic_lt_m6:.0f}%",
        ]
        axt = axes[4, j]
        axt.axis("off")
        axt.text(0, 1, "\n".join(lines), va="top", ha="left", fontsize=6.3,
                 transform=axt.transAxes, linespacing=1.35)  # fmt: skip
        for i in range(3):
            axes[i, j].set_xlabel("")
            if j > 0 and (share_raw or i != 1):
                axes[i, j].tick_params(labelleft=False)
        axes[2, j].set_xlabel("Distance from rim (µm)")
        if j > 0:
            axes[3, j].tick_params(labelleft=False)
    axes[0, 0].set_ylabel("Fo (mol%)")
    axes[1, 0].set_ylabel("Residual (mol% Fo)")
    axes[2, 0].set_ylabel("Whitened residual z")
    axes[3, 0].set_xlabel("")
    for j in range(len(cases)):
        axes[3, j].set_xlabel("Time / true time", labelpad=1)
    axes[2, 0].set_ylim(-5.5, 5.5)
    # one legend for the figure, outside the data
    handles = [
        mpl.lines.Line2D([], [], ls="none", marker="o", ms=3, mfc="white", mec="black",
                         label="data"),
        mpl.lines.Line2D([], [], color=C_TRUE, ls=(0, (1.2, 1.2)), label="true model / time"),
        mpl.lines.Line2D([], [], color=C_TRUE, lw=0.7, ls=(0, (3, 2)), label="± 2 × true noise sd"),
        mpl.lines.Line2D([], [], ls="none", marker="o", ms=5, mfc="none", mec=OKABE[7],
                         label="outlier added"),
        mpl.lines.Line2D([], [], color=C_DEFAULT, label="best fit (default noise model)"),
        mpl.patches.Patch(color=C_DEFAULT, alpha=0.4, label="1σ range"),
        mpl.patches.Patch(color=C_DEFAULT, alpha=0.18, label="2σ range"),
        mpl.lines.Line2D([], [], color=C_MISFIT, label="smooth misfit the model sees"),
        mpl.patches.Patch(color="0.85", label="± white sd s"),
        mpl.patches.Patch(color=C_ZONE, label="within 2 l of the step"),
        mpl.lines.Line2D([], [], color=C_NEAR, lw=3, alpha=0.6, label="near-step noise model"),
    ]  # fmt: skip
    if not any(CASES[c][0] == "outliers" for c in cases):
        handles = [h for h in handles if h.get_label() != "outlier added"]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=6.3,
               handlelength=1.8, columnspacing=1.2)  # fmt: skip
    fig.suptitle(title, fontsize=8.5, fontweight="bold", x=0.01, ha="left")
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


# --------------------------------------------------------------------------- real rows

REAL_PUBLIC = {
    "fig3_real_models.png": (
        "Real paper profiles: which noise model each needs",
        [
            ("AZ18 WHT06_ol_overgrowth2_prof_1", "(a) White noise is enough", False),
            ("AZ18 WHT01_bigol1_overgrowth_prof_1", "(b) A smooth misfit is needed", False),
            ("AZ18 WHT01_Ol_192_prof_1",
             "(c) Misfit much longer than the diffusion length: check the initial condition",
             False),
        ],
    ),
    "fig4_real_near_step.png": (
        "Real paper profiles where the near-step term is clearly preferred (ΔAIC < −6)",
        [
            ("Xenocryst_ol6_prof_1", "(d1) Near-step misfit", True),
            ("AZ18 WHT06_ol43_prof 1", "(d2) Near-step misfit", True),
            ("AZ18 WHT06_ol48_prof 1", "(d3) Near-step misfit", True),
        ],
    ),
}  # fmt: skip
REAL_PRIVATE = {
    "rr_near_step.png": (
        "Round robin SH63olv066 (PRIVATE): near-step misfit",
        [(f"SH63olv066-{i}", "Near-step misfit", True) for i in (1, 2, 3, 4)],
    ),
    "rr_long_misfit.png": (
        "Round robin KS20-527_2 transects (PRIVATE): long misfit against a typical one",
        [
            ("KS20-527_2-transect-3", "Misfit much longer than the diffusion length", False),
            ("KS20-527_2-transect-1", "For comparison: a typical transect", True),
        ],
    ),
}


def real_figure(analyses: dict, spec, path: Path):
    title, rows = spec
    fig = plt.figure(figsize=(7.2, 2.15 * len(rows) + 0.45), layout="constrained")
    subs = fig.subfigures(len(rows), 1)
    subs = np.atleast_1d(subs)
    for sub, (name, label, near) in zip(subs, rows, strict=True):
        a = analyses[name]
        d = a.default
        axs = sub.subplots(1, 4, width_ratios=[1.35, 1.0, 1.0, 0.85])
        draw_profile(axs[0], a, near=near)
        axs[0].set_ylabel("Fo (mol%)")
        draw_raw(axs[1], a)
        axs[1].set_ylabel("Residual (mol% Fo)")
        draw_white(axs[2], a, near=near)
        axs[2].set_ylabel("Whitened residual z")
        zmax = max(3.2, 1.1 * np.abs(a.z_default).max(),
                   1.1 * np.abs(a.z_near).max() if near else 0)  # fmt: skip
        axs[2].set_ylim(-zmax, zmax)
        for ax in axs[:3]:
            ax.set_xlabel("Distance from rim (µm)")
            ax.set_xlim(*axs[0].get_xlim())
        draw_times(axs[3], a, DAY / a.dt_s, near=near)
        axs[3].set_xlabel("Time (days)")
        lo = a.days(min(d.interval2.low, a.ns_i2.low if near else np.inf))
        hi = a.days(max(d.interval2.high, a.ns_i2.high if near else 0))
        pad = 0.12 * (hi - lo)
        axs[3].set_xlim(max(lo - pad, 0.0), hi + pad)
        axs[3].xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
        t0, t1, t2 = (a.days(v) for v in (d.interval.best, d.interval.low, d.interval.high))
        t2l, t2h = a.days(d.interval2.low), a.days(d.interval2.high)
        short = name.replace("AZ18 ", "")
        stats = (f"{len(a.x)} points; {noise_text(a)}; l = {a.ell_best:.1f} µm; "
                 f"near-step ΔAIC {fmt_daic(a.ns_daic)}")  # fmt: skip
        times = f"default: {t0:.3g} d, 1σ {t1:.3g}–{t2:.3g}, 2σ {t2l:.3g}–{t2h:.3g} d"
        if near:
            n0, n1, n2 = (a.days(v) for v in (a.ns_i1.best, a.ns_i1.low, a.ns_i1.high))
            times += (f"   |   near-step model: {n0:.3g} d, 1σ {n1:.3g}–{n2:.3g} d "
                      f"({100 * (n0 / t0 - 1):+.0f}%)")  # fmt: skip
        sub.suptitle(f"{label}: {short}\n{stats}\n{times}", x=0.01, ha="left", fontsize=7,
                     fontweight="normal", linespacing=1.3)  # fmt: skip
    handles = [
        mpl.lines.Line2D([], [], ls="none", marker="o", ms=3, mfc="white", mec="black",
                         label="data"),
        mpl.lines.Line2D([], [], color=C_DEFAULT, label="default fit"),
        mpl.patches.Patch(color=C_DEFAULT, alpha=0.4, label="1σ"),
        mpl.patches.Patch(color=C_DEFAULT, alpha=0.18, label="2σ"),
        mpl.lines.Line2D([], [], color=C_NEAR, ls=(0, (4, 1.5)), label="near-step fit, 1σ"),
        mpl.lines.Line2D([], [], color=C_MISFIT, label="smooth misfit the model sees"),
        mpl.patches.Patch(color="0.85", label="± s"),
        mpl.patches.Patch(color=C_ZONE, label="within 2 l of a step"),
        mpl.lines.Line2D([], [], ls="none", marker="^", ms=3, mfc="none", mec=C_NEAR,
                         label="z, near-step model"),
    ]  # fmt: skip
    if not any(near for _, _, near in rows):
        handles = [h for h in handles if "near-step" not in h.get_label()]
    fig.legend(handles=handles, loc="outside lower center", ncol=5, fontsize=6.3,
               handlelength=1.8, columnspacing=1.2)  # fmt: skip
    fig.suptitle(title, fontsize=8.5, fontweight="bold", x=0.01, ha="left")
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path}")


def summary_rows(analyses: dict) -> pd.DataFrame:
    out = []
    for name, a in analyses.items():
        d, nm = a.default, a.default.noise
        out.append(
            {
                "run": name, "n_points": len(a.x),
                "white_sigma_fo": nm.white_sigma, "correlated_sigma_fo": nm.correlated_sigma,
                "correlation_length_um": nm.correlation_length_um,
                "diffusion_length_um": a.ell_best,
                "L_over_l": nm.correlation_length_um / a.ell_best,
                "near_step_daic": a.ns_daic, "near_step_h": a.ns_h,
                "time_days": a.days(d.interval.best),
                "low_days": a.days(d.interval.low), "high_days": a.days(d.interval.high),
                "low2_days": a.days(d.interval2.low), "high2_days": a.days(d.interval2.high),
                "near_time_days": a.days(a.ns_i1.best),
                "near_low_days": a.days(a.ns_i1.low), "near_high_days": a.days(a.ns_i1.high),
                "near_low2_days": a.days(a.ns_i2.low),
                "near_high2_days": a.days(a.ns_i2.high),
                "lag1_raw": float(np.corrcoef(a.r_default[:-1], a.r_default[1:])[0, 1]),
                "lag1_whitened": float(np.corrcoef(a.z_default[:-1], a.z_default[1:])[0, 1]),
                "max_abs_z": float(np.abs(a.z_default).max()),
                "near_far_rms_raw": float(np.sqrt(np.mean(a.r_default[a.near] ** 2)
                                                  / np.mean(a.r_default[~a.near] ** 2)))
                if a.near.sum() >= 3 and (~a.near).sum() >= 3 else np.nan,
            }
        )  # fmt: skip
    return pd.DataFrame(out)


# --------------------------------------------------------------------------------- main


def public(trials: int):
    from conftest import PAPER_DATA, PAPER_PARAMS

    print("synthetic geometry ...", flush=True)
    g = synthetic_geometry()
    print(f"  true time {g['t_days']:.3f} d, {len(g['curves'])} rows", flush=True)
    cov_csv = HERE / "gallery_coverage.csv"
    if cov_csv.exists() and pd.read_csv(cov_csv).trials.min() >= trials:
        summ = pd.read_csv(cov_csv)
    else:
        print(f"Monte Carlo, {trials} draws per case ...", flush=True)
        summ = gallery_coverage(g, trials)
        summ.to_csv(cov_csv, index=False, float_format="%.4g")
    gallery_figure(g, summ, STATIONARY, HERE / "fig1_gallery_stationary.png",
                   "Stationary misfit: the same size everywhere along the profile")  # fmt: skip
    gallery_figure(g, summ, OTHER, HERE / "fig2_gallery_other.png",
                   "Misfit that is not the same everywhere, and outliers",
                   share_raw=False)  # fmt: skip

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runs = {r.name: r for r in forster.read_runs(PAPER_PARAMS)}
        profiles = forster.read_profiles(PAPER_DATA, sheet="Olivine")
    names = [n for _, rows in REAL_PUBLIC.values() for n, _, _ in rows]
    analyses = {n: analyse_run(runs[n], profiles) for n in names}
    for fname, spec in REAL_PUBLIC.items():
        real_figure(analyses, spec, HERE / fname)
    summary_rows(analyses).to_csv(HERE / "real_examples.csv", index=False, float_format="%.5g")


def private():
    from conftest import HAS_ROUND_ROBIN, ROUND_ROBIN_DATA, ROUND_ROBIN_PARAMS

    if not HAS_ROUND_ROBIN:
        raise SystemExit("the round robin submodule is not checked out")
    PRIVATE_OUT.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runs = {}
        for r in forster.read_runs(ROUND_ROBIN_PARAMS,
                                   defaults=forster.RunDefaults(max_time_days=730)):  # fmt: skip
            runs.setdefault(r.name, r)
        profiles = forster.read_profiles(ROUND_ROBIN_DATA)
    names = [n for _, rows in REAL_PRIVATE.values() for n, _, _ in rows]
    analyses = {n: analyse_run(runs[n], profiles) for n in names}
    for fname, spec in REAL_PRIVATE.items():
        real_figure(analyses, spec, PRIVATE_OUT / fname)
    summary_rows(analyses).to_csv(PRIVATE_OUT / "rr_examples.csv", index=False,
                                  float_format="%.5g")  # fmt: skip


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument(
        "--private",
        action="store_true",
        help="round robin figures, written only inside the private submodule",
    )
    ap.add_argument("--trials", type=int, default=500, help="Monte Carlo draws per case")
    args = ap.parse_args()
    if args.private:
        private()
    else:
        public(args.trials)
