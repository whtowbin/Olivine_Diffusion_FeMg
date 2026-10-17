"""Figures for the population-prior note, from the CSVs analyse.py writes.

    uv run python docs/plans/2026-10-16-population-priors/figures.py

spread.png: one value per crystal against its within-crystal error, for the groups with
    a spread worth showing (each panel its own Fo scale).
refits.png: time ranges with the wide prior and with a population prior.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
mpl.rcParams.update(
    {
        "font.size": 7.5,
        "axes.titlesize": 7.5,
        "axes.labelsize": 7.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "legend.fontsize": 7,
        "pdf.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)
BLUE, ORANGE, GREY = "#0072B2", "#E69F00", "0.55"
PANELS = [  # (group, role, title)
    ("AZ18 WHT01", "core", "WHT01 cores"),
    ("AZ18 WHT06", "core", "WHT06 cores"),
    ("Xenocryst", "core", "Xenocryst cores"),
    ("AZ18 WHT01 (reverse)", "rim", "WHT01 high-Fo rims"),
    ("AZ18 WHT01 (normal)", "rim", "WHT01 low-Fo rims"),
    ("Xenocryst (normal)", "rim", "Xenocryst low-Fo rims"),
]


def short_name(name: str) -> str:
    return (
        name.replace("AZ18 ", "")
        .replace("WHT01_", "")
        .replace("WHT06_", "")
        .replace("Xenocryst_", "")
        .replace("ol_overgrowth", "og")
        .replace("_overgrowth", "_og")
        .replace("_prof_", " p")
        .replace("_prof ", " p")
    )


def spread_figure() -> None:
    cv = pd.read_csv(HERE / "crystal_values.csv")
    sp = pd.read_csv(HERE / "spread.csv")
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6), layout="constrained")
    for ax, (group, role, title) in zip(axes.flat, PANELS, strict=True):
        d = cv[(cv["group"] == group) & (cv["role"] == role)].sort_values("value")
        s = sp[(sp["group"] == group) & (sp["role"] == role)].iloc[0]
        ax.axvspan(
            100 * (s["mean_fo"] - s["between_sd"]),
            100 * (s["mean_fo"] + s["between_sd"]),
            color="0.88",
            lw=0,
        )
        ax.axvline(100 * s["mean_fo"], color="0.6", lw=0.8)
        for i, (_, c) in enumerate(d.iterrows()):
            measured = c["source"].startswith("measured")
            ax.errorbar(
                100 * c["value"],
                i,
                xerr=100 * c["error"],
                fmt="o" if measured else "s",
                ms=3.5,
                mfc=BLUE if measured else "white",
                mec=BLUE,
                ecolor="k",
                elinewidth=0.8,
                capsize=2,
            )
        ax.set_yticks(range(len(d)), [short_name(n) for n in d["crystal"]])
        ax.set_ylim(-0.7, len(d) - 0.3)
        ax.set_xlabel("Fo (mol%)")
        ax.margins(x=0.15)
        verdict = "differ" if s["p_same_value"] < 0.01 else "agree"
        ax.set_title(
            f"{title}: {verdict}\nn = {len(d)}, sd between = {s['ratio']:.1f} x error",
            loc="left",
        )
    big = sp[sp["crystals"] >= 3]
    cores = big[big["role"] == "core"]
    rims = big[big["role"] == "rim"]
    fig.suptitle(
        f"Cores differ between crystals beyond their errors in "
        f"{int((cores['p_same_value'] < 0.01).sum())} of {len(cores)} samples with 3+ crystals; "
        f"rims agree within error in {int((rims['p_same_value'] > 0.05).sum())} of {len(rims)}",
        x=0.01,
        ha="left",
        fontsize=8.5,
    )
    handles = [
        plt.Line2D([], [], marker="o", ls="", mfc=BLUE, mec=BLUE, ms=4),
        plt.Line2D([], [], marker="s", ls="", mfc="white", mec=BLUE, ms=4),
        plt.Rectangle((0, 0), 1, 1, color="0.88"),
    ]
    fig.legend(
        handles,
        [
            "measured plateau (bar: 1 standard error)",
            "fitted, no plateau measured (bar: 1 posterior sd)",
            "group mean +- 1 sd between crystals",
        ],
        loc="outside lower center",
        ncol=2,
        frameon=False,
    )
    fig.savefig(HERE / "spread.png", dpi=220)
    plt.close(fig)


def _interval(ax, y, lo, best, hi, color, dashed=False):
    ls = (0, (2, 1)) if dashed else "-"
    ax.plot([lo, hi], [y, y], color=color, lw=1.6, ls=ls, solid_capstyle="butt")
    ax.plot(best, y, "o", ms=3.2, mfc="white" if dashed else color, mec=color)


def _fixed(nc, case, run):
    """The fixed-noise refit of this case (check_noise.py), or None."""
    m = nc[(nc["case"] == case) & (nc["run"] == run)]
    return None if m.empty else m.iloc[0]


def refit_figure() -> None:
    r = pd.read_csv(HERE / "refits.csv")
    full = r[r["case"] == "full"].sort_values(["role", "wide_best_days"])
    trunc = r[r["case"] == "truncated"].sort_values("full_best_days")
    borrowed = r[r["case"].str.startswith("borrowed")].set_index("case")
    nc = pd.read_csv(HERE / "noise_check.csv")
    fig, axes = plt.subplots(
        1, 2, figsize=(7.2, 4.6), layout="constrained", gridspec_kw={"width_ratios": [1, 1]}
    )

    ax = axes[0]
    for i, (_, s) in enumerate(full.iterrows()):
        _interval(ax, i - 0.22, s.wide_low_days, s.wide_best_days, s.wide_high_days, GREY)
        _interval(ax, i + 0.08, s.pop_low_days, s.pop_best_days, s.pop_high_days, BLUE)
        f = _fixed(nc, "full", s.run) if s.role == "core" else None
        if f is not None:
            _interval(
                ax, i + 0.36, f.fixed_low_days, f.fixed_best_days, f.fixed_high_days, BLUE, True
            )
    ax.set_yticks(range(len(full)), [f"{short_name(s.run)} ({s.role})" for s in full.itertuples()])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Time (days): best fit (dot) and 68% range")
    ratio = full["range_ratio"].copy()  # fixed-noise refit where there is one
    for k, s in full.iterrows():
        f = _fixed(nc, "full", s.run) if s.role == "core" else None
        if f is not None:
            ratio[k] = (f.fixed_high_days - f.fixed_low_days) / s.wide_range_days
    ax.set_title(
        f"(a) Full profiles, prior from the other crystals\n"
        f"range >10% narrower in {int((ratio < 0.9).sum())}, "
        f">10% wider in {int((ratio > 1.1).sum())}, of {len(full)}",
        loc="left",
    )

    ax = axes[1]
    labels = []
    for i, (_, s) in enumerate(trunc.iterrows()):
        _interval(ax, i - 0.3, s.full_low_days, s.full_best_days, s.full_high_days, "k")
        _interval(ax, i - 0.08, s.wide_low_days, s.wide_best_days, s.wide_high_days, GREY)
        _interval(ax, i + 0.14, s.pop_low_days, s.pop_best_days, s.pop_high_days, BLUE)
        f = _fixed(nc, "truncated", s.run)
        if f is not None:
            _interval(
                ax, i + 0.36, f.fixed_low_days, f.fixed_best_days, f.fixed_high_days, BLUE, True
            )
        labels.append(short_name(s.run))
    if len(borrowed):
        j = len(trunc) + 0.6
        og1, henry = borrowed.loc["borrowed (og1)"], borrowed.loc["borrowed (henry)"]
        _interval(ax, j - 0.25, og1.wide_low_days, og1.wide_best_days, og1.wide_high_days, GREY)
        _interval(ax, j, og1.pop_low_days, og1.pop_best_days, og1.pop_high_days, BLUE)
        _interval(
            ax, j + 0.25, henry.pop_low_days, henry.pop_best_days, henry.pop_high_days, ORANGE
        )
        ax.axhline(len(trunc) - 0.2, color="0.8", lw=0.6)
    ticks = list(range(len(trunc))) + ([len(trunc) + 0.6] if len(borrowed) else [])
    labels += ["WHT06 og2 p1\n(no core plateau)"] if len(borrowed) else []
    ax.set_yticks(ticks, labels)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Time (days): best fit (dot) and 68% range")
    ax.set_title(
        "(b) Core barely measured (profile cut 5 um into\n"
        "the core plateau); bottom: no core plateau",
        loc="left",
    )
    handles = [
        plt.Line2D([], [], color=GREY, lw=1.6, marker="o", ms=3.2),
        plt.Line2D([], [], color=BLUE, lw=1.6, marker="o", ms=3.2),
        plt.Line2D([], [], color="k", lw=1.6, marker="o", ms=3.2),
        plt.Line2D([], [], color=ORANGE, lw=1.6, marker="o", ms=3.2),
        plt.Line2D([], [], color=BLUE, lw=1.6, ls=(0, (2, 1)), marker="o", ms=3.2, mfc="white"),
    ]
    fig.legend(
        handles,
        [
            "wide prior (segment Fo sd 1 mol%)",
            "population prior (og2: overgrowth1's core +- 0.12)",
            "reference: whole profile, wide prior",
            "og2: Henry's core value +- 0.12",
            "population prior, noise model held at the wide fit's",
        ],
        loc="outside lower center",
        ncol=2,
        frameon=False,
    )
    fig.savefig(HERE / "refits.png", dpi=220)
    plt.close(fig)


if __name__ == "__main__":
    spread_figure()
    refit_figure()
