"""Figures for the near-step follow-ups (paper and synthetic profiles only).

    uv run python docs/plans/2026-10-15-whitening-validity/followups/figures.py

fig1_continuous_h.png: (a) the near-step dAIC as a function of h for the three paper
profiles where the warning fires; (b) the share of synthetic white-noise profiles whose
dAIC falls below a threshold, grid h against continuous h.
fig2_power.png: detection rate and default 1-sigma coverage against the number of points.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
REPO = STUDY.parents[2]
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "analyses" / "ic_sampling"))
sys.path.insert(0, str(HERE))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from nearstep import NearStep  # noqa: E402

plt.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
OI = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#000000"]
WARNED = ["Xenocryst_ol6_prof_1", "AZ18 WHT06_ol43_prof 1", "AZ18 WHT06_ol48_prof 1"]
SHORT = {
    "Xenocryst_ol6_prof_1": "Xenocryst_ol6 prof 1",
    "AZ18 WHT06_ol43_prof 1": "WHT06_ol43",
    "AZ18 WHT06_ol48_prof 1": "WHT06_ol48",
}


def warned_curves():
    """dAIC(h) = 2 (nll(h) - nll(0)) + 2 for the three warned paper profiles, residuals
    at the white best fit exactly as FORSTER's `_near_step`."""
    from references import load

    import forster
    from forster.grid import diffusion_length_um, fo_range
    from forster.uncertainty import interpolation_matrix, white_rss

    data = load()
    out = {}
    hs = np.logspace(-2, 4, 121)
    for name in WARNED:
        run, prof, _ = data[name]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = forster.fit_run(run, prof, keep_curves=True)
        x, y = np.asarray(res.fit_x_um, float), np.asarray(res.fit_fo, float)
        at = res.curves @ interpolation_matrix(res.model_x_um, x).T
        w = int(np.argmin(white_rss(at, y)))
        ell = diffusion_length_um(
            max(w, 1) * res.dt_s / 86400,
            [(run.diffusivity, fo_range(run.initial))],
            run.conditions,
            run.orientation,
        )
        ns = NearStep(x, y - at[w], run.initial.steps_um, ell)
        f0 = ns.at(0.0).nll
        r = ns.fit()
        out[name] = (hs, np.array([2 * (ns.at(h).nll - f0) + 2 for h in hs]), r)
    return out


def fig1():
    real = pd.read_csv(HERE / "real_continuous_h.csv").set_index("run")
    null = pd.read_csv(HERE / "null_trials.csv")
    curves = warned_curves()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.7), layout="constrained")

    ax = axes[0]
    grid_h = np.array([1, 3, 10, 30, 100])
    for i, name in enumerate(WARNED):
        hs, d, r = curves[name]
        lab = f"{SHORT[name]}: h = {r.h_cont:.0f}, dAIC {r.daic_grid:.1f} to {r.daic_cont:.1f}"
        ax.plot(hs, d, color=OI[i], lw=1.4, label=lab)
        gd = np.interp(np.log10(grid_h), np.log10(hs), d)
        ax.plot(grid_h, gd, "o", ms=3.5, color=OI[i], mfc="white", mew=1)
        ax.plot(r.h_cont, r.daic_cont, "*", ms=9, color=OI[i], mec="black", mew=0.4)
        assert abs(r.daic_cont - real.loc[name, "cont_daic"]) < 1e-2  # CSV keeps 5 digits
    ax.axhline(-6, color="0.4", lw=0.8, ls="--")
    ax.text(0.012, -5.8, "warning threshold (-6)", va="bottom", fontsize=6.5, color="0.3")
    ax.axvspan(100, 1e4, color="0.93", zorder=0)
    ax.text(130, 2.6, "beyond the\ncurrent grid", fontsize=6.5, color="0.35", va="top")
    ax.set_xscale("log")
    ax.set_xlim(0.01, 1e4)
    ax.set_ylim(-16, 3.5)
    ax.set_xlabel("near-step growth h (misfit variance at a step = (1 + h) x far)")
    ax.set_ylabel("dAIC of the near-step model at this h")
    ax.legend(loc="lower left", frameon=False, fontsize=6.5, handlelength=1.5)
    ax.set_title("(a) Warned paper profiles: dAIC is flat past h = 100", loc="left")

    ax = axes[1]
    thr = np.linspace(-14, -2, 121)
    g = null[null.geometry == "paper 17 pts (2.9 um)"]
    cases = [
        ("strong near-step L=3", OI[3], "1.2 mol% at step, L = 3 um"),
        ("strong near-step L=10", OI[4], "1.2 mol% at step, L = 10 um"),
        ("sq-exp L=10 (control)", "0.5", "same misfit everywhere"),
        ("white (control)", OI[0], "white noise only"),
    ]
    for case, colr, lab in cases:
        c = g[g.case == case]
        for col, ls in (("grid_daic", "-"), ("cont_daic", "--")):
            rate = [100 * (c[col] < t).mean() for t in thr]
            ax.plot(thr, rate, ls=ls, color=colr, lw=1.3, label=lab if ls == "-" else None)
    ax.plot([], [], color="black", ls="-", label="solid: grid h")
    ax.plot([], [], color="black", ls="--", label="dashed: continuous h")
    n_draw = int(g.groupby("case").size().min())
    ax.axhline(1, color="0.4", lw=0.8, ls=":")
    ax.axvline(-6, color="0.4", lw=0.8, ls=":")
    ax.set_yscale("log")
    ax.set_ylim(0.03, 100)
    ax.set_xlabel("warning threshold on dAIC")
    ax.set_ylabel("draws flagged (%)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncols=2, frameon=False, fontsize=6.5)
    ax.set_title(f"(b) Paper 17-pt geometry, {n_draw} draws per case", loc="left")
    fig.savefig(HERE / "fig1_continuous_h.png", dpi=220)
    plt.close(fig)


def fig2():
    s = pd.read_csv(HERE / "power_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5), layout="constrained")
    cases = [
        ("1.2 mol%, L=3", OI[3], "-", "o", "1.2 mol% at step"),
        ("0.8 mol%, L=3", OI[1], "-", "s", "0.8 mol% at step"),
        ("0.4 mol%, L=3", OI[2], "-", "^", "0.4 mol% at step"),
        ("1.2 mol%, L=10", OI[4], "--", "D", "1.2 mol%, L = 10 um"),
        ("smooth 0.4 mol%, L=10, everywhere", "0.55", ":", "v", "same misfit everywhere"),
        ("white", "0.3", ":", "x", "white noise only"),
    ]
    for name, c, ls, m, lab in cases:
        g = s[s.noise == name].sort_values("n")
        axes[0].plot(g.n, g.grid_warn_pct, ls=ls, marker=m, ms=3.5, color=c, lw=1.3, label=lab)
        axes[1].plot(g.n, g.cov1_default, ls=ls, marker=m, ms=3.5, color=c, lw=1.3, label=lab)
    draws = int(s.draws.min())
    se = 100 * np.sqrt(0.6827 * 0.3173 / draws)
    axes[1].axhspan(68.27 - 1.96 * se, 68.27 + 1.96 * se, color="0.9", zorder=0)
    axes[1].axhline(68.27, color="0.4", lw=0.8, ls="--")
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks([12, 17, 25, 35, 50, 100])
        ax.set_xticklabels(["12", "17", "25", "35", "50", "100"])
        ax.set_xlabel("points in the profile (same span)")
    axes[0].set_ylabel("warning fires (dAIC < -6), % of draws")
    axes[0].set_ylim(-2, 102)
    axes[0].set_title("(a) Detection rises with the number of points", loc="left")
    axes[1].set_ylabel("default 1-sigma range covers true time (%)")
    axes[1].set_title("(b) ...the coverage loss does not", loc="left")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncols=3, frameon=False)
    fig.savefig(HERE / "fig2_power.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    os.environ.setdefault("MPLBACKEND", "Agg")
    which = [a for a in sys.argv[1:] if a in ("1", "2")] or ["1", "2"]
    if "1" in which:
        fig1()
    if "2" in which:
        fig2()
