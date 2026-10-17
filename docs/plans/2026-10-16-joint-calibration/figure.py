"""The README figure: the ESS does not tell an accurate Laplace range from a wrong one.

uv run python docs/plans/2026-10-16-joint-calibration/figure.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rules  # noqa: E402

from forster.sampling import ideal_ess_fraction  # noqa: E402

# Okabe-Ito colours and distinct markers for the cases where Laplace was ever inaccurate
HARD = {
    "GCB1A R31_MP_Ol5 | base": ("#0072B2", "o", "GCB1A Ol5, base"),
    "GCB1A R31_MP_Ol5 | no-ref": ("#009E73", "D", "GCB1A Ol5, no-ref"),
    "GCB1A R31_MP_Ol5 | edge": ("#D55E00", "^", "GCB1A Ol5, edge"),
    "Xenocryst_ol6 | edge": ("#CC79A7", "v", "Xenocryst ol6, edge"),
    "WHT01 pseudo-4 | base": ("#E69F00", "s", "WHT01 pseudo-crystal, base"),
    "WHT01 pseudo-4 | no-ref": ("#56B4E9", "P", "WHT01 pseudo-crystal, no-ref"),
}
EASY = ("0.6", "o", "other cases (4)")


def style(case):
    return HARD.get(case, EASY)


def tolerance_lines(ax, label: bool) -> None:
    """5% (the tolerance the recommendation uses) dashed, 2% (secondary) dotted."""
    for tol, ls, text in ((0.05, "--", "5% tolerance (used)"), (0.02, ":", "2% (secondary)")):
        ax.axhline(100 * tol, color="0.3", lw=0.7, ls=ls, zorder=1)
        if label:
            ax.annotate(
                text,
                (0.0045, 100 * tol),
                xytext=(0, 2),
                textcoords="offset points",
                va="bottom",
                color="0.3",
                fontsize=6,
            )


def main() -> None:
    caches = [HERE / "cache_paper.json"]
    if (HERE / "cache_stress.json").exists():
        caches.append(HERE / "cache_stress.json")
    scored = rules.score(rules.rows_frame(caches))
    hard = [c for c in HARD if c in set(scored.case)]
    easy = [c for c in dict.fromkeys(scored.case) if c not in HARD]
    EASY_LABEL = f"other cases ({len(easy)})"
    plt.rcParams.update(
        {
            "font.size": 7,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(7.2, 3.0), layout="constrained", gridspec_kw={"width_ratios": [1, 1.15]}
    )

    # a: ESS fraction against Laplace's error, 1024 samples, every seed
    lap = scored[(scored.method == "laplace") & (scored.n == rules.N0)]
    for case in easy + hard:
        g = lap[lap.case == case]
        colour, marker, label = style(case)
        ax1.scatter(
            g.ess_fraction,
            100 * g.time_err,
            s=14,
            marker=marker,
            facecolor=colour,
            edgecolor="white",
            linewidth=0.3,
            label=label if case in HARD else (EASY_LABEL if case == easy[0] else None),
            zorder=3 if case in HARD else 2,
        )
    ax1.set_xscale("log")
    tolerance_lines(ax1, label=True)
    ds = sorted(set(lap.d))
    lo = 0.25 * ideal_ess_fraction(max(ds)) / ideal_ess_fraction(4)
    hi = 0.25 * ideal_ess_fraction(min(ds)) / ideal_ess_fraction(4)
    ax1.axvspan(lo, hi, color="0.85", lw=0, zorder=0)
    ax1.annotate(
        f"grey band: the current\nrule's threshold (d = {min(ds)}-{max(ds)});"
        "\nnested sampling left of it",
        (0.0045, 7.4),
        va="top",
        color="0.35",
        fontsize=6,
    )
    ax1.set_xlim(0.004, 0.7)
    ax1.set_ylim(0, 7.5)
    ax1.set_xlabel("ESS / samples (1024 samples; log scale)")
    ax1.set_ylabel("Laplace error: largest of p16, p50, p84\nagainst the nested reference (%)")
    ax1.set_title("a  The ESS barely predicts Laplace's error", loc="left", fontsize=7.5)

    # b: the hard cases' error at each sample count, and the fallback's
    columns = [("Laplace\n1024", "laplace", 1024), ("Laplace\n2048", "laplace", 2048)]
    columns += [
        ("Laplace\n4096", "laplace", 4096),
        ("fallback:\nnested,\nnlive 200", "nested200", 0),
    ]
    for j, (_, method, n) in enumerate(columns):
        for i, case in enumerate(hard):
            g = scored[(scored.case == case) & (scored.method == method)]
            if n:
                g = g[g.n == n]
            x = j + (i - (len(hard) - 1) / 2) * 0.12
            colour, marker, _ = style(case)
            ax2.scatter(
                np.full(len(g), x),
                100 * g.time_err,
                s=11,
                marker=marker,
                facecolor=colour,
                edgecolor="white",
                linewidth=0.3,
                zorder=3,
            )
    tolerance_lines(ax2, label=False)
    ax2.set_xticks(range(len(columns)), [c[0] for c in columns])
    ax2.set_xlim(-0.5, len(columns) - 0.5)
    ax2.set_ylim(0, 7.5)
    ax2.set_ylabel("error against the nested reference (%)")
    worst = scored[(scored.method == "laplace") & (scored.n == 4096)].time_err.max()
    ax2.set_title(
        f"b  Harder cases: at 4096 samples the worst run is {100 * worst:.1f}% off",
        loc="left",
        fontsize=7.5,
    )
    fig.legend(loc="outside right center", frameon=False, fontsize=6.5, title="crystal, priors")
    out = HERE / "ess_vs_error.png"
    fig.savefig(out, dpi=200)
    print(out)


if __name__ == "__main__":
    main()
