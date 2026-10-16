"""Figures for the whitening-validity study. Reads coverage.csv, real_kernels.csv and
real_profiles.csv; writes fig1_coverage.png, fig2_aic.png, fig3_whitened.png here.
No measured profile is plotted (the round robin data are private): only fit
statistics. Run from the repo root:

    uv run python docs/plans/2026-10-15-whitening-validity/figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OKABE = ["#000000", "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]
mpl.rcParams.update(
    {
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "0.92",
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "savefig.dpi": 200,
        "pdf.fonttype": 42,
    }
)


def fig_coverage():
    cov = pd.read_csv(HERE / "coverage.csv")
    zone = pd.read_csv(HERE / "zone_coverage.csv")
    geoms = list(dict.fromkeys(cov.geometry))
    cases = list(dict.fromkeys(cov.case))
    y = np.arange(len(cases))[::-1]
    fig, axes = plt.subplots(
        2, len(geoms), figsize=(7.2, 7.0), sharey=True, layout="constrained", sharex="row"
    )
    for row, (lvl, target, lo, hi, xlim) in enumerate(
        (
            ("1s", 68.27, "band68_lo", "band68_hi", (0, 101.5)),
            ("2s", 95.45, "band95_lo", "band95_hi", (25, 101.5)),
        )
    ):
        for ax, gname in zip(axes[row], geoms, strict=True):
            g = cov[cov.geometry == gname].set_index("case").loc[cases]
            ax.axvspan(100 * g[lo].iloc[0], 100 * g[hi].iloc[0], color="0.85", lw=0)
            ax.axvline(target, color="0.4", lw=0.8)
            ax.scatter(
                100 * g[f"white_coverage_{lvl}"],
                y,
                marker="o",
                facecolor="white",
                edgecolor=OKABE[1],
                s=22,
                lw=1.2,
                label="independent errors (baseline)",
                zorder=3,
            )
            ax.scatter(
                100 * g[f"coverage_{lvl}"],
                y,
                marker="D",
                color=OKABE[5],
                s=20,
                label="correlated likelihood (default)",
                zorder=4,
            )
            zc = zone[zone.geometry == gname].set_index("case")
            yz = [y[cases.index(c)] for c in zc.index]
            ax.scatter(
                100 * zc[f"corr_cand_coverage_{lvl}"],
                yz,
                marker="^",
                color=OKABE[3],
                s=22,
                label="candidate: misfit amplitude grows near the step (500 trials)",
                zorder=5,
            )
            for i in range(len(cases)):
                if i in (3, 9, 13):  # separate controls / kernels / near-step / white-noise groups
                    ax.axhline(y[i] - 0.5, color="0.7", lw=0.5)
            ax.set_xlim(*xlim)
            ax.set_ylim(-0.7, len(cases) - 0.3)
            if row == 0:
                ax.set_title(gname, fontsize=8)
            ax.set_xlabel(f"{'1' if lvl == '1s' else '2'}-sigma coverage (%)")
        axes[row][0].set_yticks(y, cases)
    n = int(cov.trials.min())
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncol=2)
    fig.suptitle(
        "The kernel's shape barely matters; misfit concentrated near the step does\n"
        f"(coverage of the true time, {n} synthetic profiles per case; line: target; "
        "grey: 95% sampling range)",
        fontsize=9,
    )
    fig.savefig(HERE / "fig1_coverage.png")
    plt.close(fig)


def fig_aic():
    k = pd.read_csv(HERE / "real_kernels.csv")
    k = k[k.residuals_at == "white best fit"]
    p = k.pivot(index="run", columns="kernel", values="aic")
    d = p.sub(p.min(axis=1), axis=0)
    d = d.sort_values("white")
    src = k.drop_duplicates("run").set_index("run").source.loc[d.index]
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(6.0, 7.0), layout="constrained")
    ax.axvspan(0, 2, color="0.88", lw=0)
    styles = {
        "white": ("white only", "x", OKABE[0]),
        "sqexp": ("squared-exponential (current)", "D", OKABE[5]),
        "exp": ("exponential (Matern 1/2)", "o", OKABE[6]),
        "matern32": ("Matern 3/2", "s", OKABE[3]),
    }
    off = {"white": 0.0, "sqexp": -0.2, "exp": 0.2, "matern32": 0.0}
    for kind, (lab, m, c) in styles.items():
        face = c if m in ("x", "D") else "none"
        ax.scatter(
            d[kind],
            y + off[kind],
            marker=m,
            s=18,
            label=lab,
            **({"color": c} if m == "x" else {"facecolor": face, "edgecolor": c}),
            lw=1.0,
            zorder=3,
        )
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(-0.2, 300)
    names = [
        n.replace("AZ18 ", "") + (" (RR)" if s == "round_robin" else "")
        for n, s in zip(d.index, src, strict=True)
    ]
    ax.set_yticks(y, names, fontsize=6)
    ax.set_ylim(-0.7, len(d) - 0.3)
    ax.set_xlabel("AIC above the best of the four noise models (0 = best; grey: within 2)")
    n_fam = int((np.minimum(d.white, d.sqexp) <= 2).sum())
    n_kern = int(((p.sqexp - p[["sqexp", "exp", "matern32"]].min(axis=1)) <= 2).sum())
    ax.set_title(
        f"Current model (white, or white + sq-exp) within 2 AIC\nof the best: {n_fam} of {len(d)}"
        f" profiles; sq-exp within 2 of the best\ncorrelated kernel: {n_kern} of {len(d)}",
        fontsize=8,
    )
    fig.legend(loc="outside lower center", ncol=2)
    fig.savefig(HERE / "fig2_aic.png")
    plt.close(fig)


def fig_whitened():
    r = pd.read_csv(HERE / "real_profiles.csv")
    tests = {
        "boot_p_lb_p_lag5": "Ljung-Box\nlags 1-5",
        "boot_p_runs_p": "runs\ntest",
        "boot_p_shapiro_p": "Shapiro-\nWilk",
        "boot_p_var_z": "var z",
        "boot_p_max_abs_z": "largest\n|z|",
        "boot_p_kurtosis_z": "excess\nkurtosis",
        "boot_p_step_var_ratio_z": "near-step /\nfar var z",
    }
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.2, 3.4), layout="constrained", width_ratios=(2.3, 1)
    )
    rng = np.random.default_rng(0)
    colours = {"round_robin": OKABE[6], "paper": OKABE[5], "example": OKABE[3]}
    labels = {"round_robin": "round robin", "paper": "Towbin & Plank 2026", "example": "example"}
    for i, col in enumerate(tests):
        for src, g in r.groupby("source"):
            v = g[col].dropna()
            ax1.scatter(
                i + rng.uniform(-0.18, 0.18, len(v)),
                v,
                s=10,
                color=colours[src],
                label=labels[src] if i == 0 else None,
                alpha=0.85,
                lw=0,
            )
        n_fail = int((r[col] < 0.05).sum())
        n_all = int(r[col].notna().sum())
        ax1.annotate(
            f"{n_fail}/{n_all}",
            (i, 1.0),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            annotation_clip=False,
        )
    ax1.axhline(0.05, color="0.3", lw=0.8, ls="--")
    ax1.set_yscale("log")
    ax1.set_ylim(3e-3, 1.0)
    ax1.set_xticks(range(len(tests)), list(tests.values()))
    ax1.set_ylabel("bootstrap p-value (dashed: 0.05)")
    ax1.set_title("Whitened residuals: share with p < 0.05 above each test", fontsize=8, pad=14)
    fig.legend(loc="outside lower center", ncol=3)
    for src, g in r.groupby("source"):
        ax2.scatter(g.lag1_raw, g.lag1_z, s=12, color=colours[src], lw=0)
    ax2.axhline(0, color="0.5", lw=0.6)
    ax2.axvline(0, color="0.5", lw=0.6)
    ax2.plot([-0.3, 0.8], [-0.3, 0.8], color="0.7", lw=0.6, ls=":")
    ax2.set_xlabel("lag-1 autocorr., raw")
    ax2.set_ylabel("lag-1 autocorr., whitened")
    ax2.set_title("Raw vs whitened lag-1", fontsize=8)
    fig.savefig(HERE / "fig3_whitened.png")
    plt.close(fig)


if __name__ == "__main__":
    fig_aic()
    fig_whitened()
    if (HERE / "coverage.csv").exists():
        fig_coverage()
