"""The melt-contact figures again, with Oeser et al. (2026) added.

figures/with_oeser/
- fig1_bowties.png: bowties at the RR1200 conditions, as fig1: (a) the preset's fit, with
  Oeser's points shown but not fitted; (b) Oeser's depth-profile runs added; (c) those
  plus a melt term for Spandler and O'Neill (2010).
- fig2_fo2.png: (a) data and laws against fO2 at 1200 C, Fo90, 1 atm, [001], points
  moved by the fit of (c); (b) the fO2 exponent of each fit, and each study's own.
- fig3_offsets.png: each study's residual against DC07 at its own conditions, with
  Oeser's depth-profile and linescan runs as separate rows.

Oeser's linescan runs (7, 14, 15: convolution corrected) are drawn faded and never
fitted. Oeser's tables give no error on D, so its points have no error bars. Shea et
al.'s rows are not fitted here. The power-law fits are fit_models.fit_random, as the
preset; analyses/femg_refit/fit_extended.py fits the same sets by its own machinery.

Run: uv run python analyses/melt_contact/oeser_figures.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "femg_refit"))

import fit_extended as X  # noqa: E402
import fit_models as F  # noqa: E402
import fo2_figures as FO  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import melt_figures as MF  # noqa: E402
import melt_sensitivity as M  # noqa: E402

OUT = HERE / "figures" / "with_oeser"
LN10 = math.log(10)
OESER, OESER_LS = "oeser2026", "oeser2026_linescan"
GREEN = "#117733"
STYLE = {
    **MF.STYLE,
    OESER: (GREEN, "X", "Oeser et al. 2026, depth profiles"),
    OESER_LS: (GREEN, "x", "Oeser et al. 2026, linescan runs (not fitted)"),
}
NAMES = {
    "oeser_all": "+ Oeser, all runs",
    "oeser_dp": "+ Oeser, depth profiles",
    "melt_fixed_spandler": "melt term: Spandler (no Oeser)",
    "oeser_dp_melt": "+ Oeser dp, melt term: Spandler",
}


def data():
    """Points to draw (Oeser's linescan runs as their own source), and the fitted sets."""
    base = F.tamed()
    oes_all, oes_dp = X.oeser(False), X.oeser(True)
    shown = X.oeser(False)
    shown.loc[shown["run"].isin(X.LINESCAN_RUNS), "source"] = OESER_LS
    pts = pd.concat([base, M.shea_femg_rows(M.load_shea()), shown], ignore_index=True)
    sigma = F._y_sigma_ln(pts) / LN10
    sigma[(pts["source"] == MF.SHEA).to_numpy()] = MF.SHEA_SIGMA
    sigma[pts["source"].isin([OESER, OESER_LS]).to_numpy()] = 0.0  # not given
    sets = {
        "base": base,
        "oeser_all": pd.concat([base, oes_all], ignore_index=True),
        "oeser_dp": pd.concat([base, oes_dp], ignore_index=True),
    }
    return pts, sigma, sets


def fits(sets: dict) -> dict:
    form = F.FORMS["dc07_free"]
    dp = sets["oeser_dp"]
    return {
        "baseline": F.fit_random(form, sets["base"]),
        "oeser_all": F.fit_random(form, sets["oeser_all"]),
        "oeser_dp": F.fit_random(form, dp),
        "melt_fixed_spandler": F.fit_random(
            M.melt_form(form, sets["base"], (M.SPANDLER,)), sets["base"]
        ),
        "oeser_dp_melt": F.fit_random(M.melt_form(form, dp, (M.SPANDLER,)), dp),
    }


def panels(pts: pd.DataFrame, fit: dict) -> list[dict]:
    solid = ~pts["source"].isin([MF.SHEA, OESER, OESER_LS]).to_numpy()
    with_dp = solid | (pts["source"] == OESER).to_numpy()
    spec = (
        (
            "a",
            "baseline",
            "previous preset: Oeser not fitted",
            solid,
            "faded: Oeser and Shea, not fitted",
        ),
        ("b", "oeser_dp", "+ Oeser, depth-profile runs", with_dp, "faded: Oeser linescans, Shea"),
        (
            "c",
            "oeser_dp_melt",
            "+ Oeser dp, melt term: Spandler",
            with_dp,
            "faded: Oeser linescans, Shea",
        ),
    )
    return [
        {"key": k, "variant": v, "fit": fit[v], "title": t, "in_fit": m, "note": note}
        for k, v, t, m, note in spec
    ]


def bowtie_title(rows: list[dict]) -> str:
    a, b, c = rows
    return (
        f"Oeser's depth-profile runs barely move the law at RR1200 "
        f"({a['RR1200_1200C_vs_dc07']:+.2f} → {b['RR1200_1200C_vs_dc07']:+.2f} log above DC07) "
        f"and narrow it (±{a['band68_1200C']:.2f} → ±{b['band68_1200C']:.2f});\n"
        f"with a melt term the lab spread τ is {c['tau_log10']:.2f}, "
        f"not {a['tau_log10']:.2f}, and melt contact sits {c['melt_log10']:+.2f} above solid"
    )


def exponents(sets: dict, fit: dict) -> pd.DataFrame:
    form = F.FORMS["dc07_free"]
    rows = [("dc07_free", v, *MF._exponent(form, fit[v])) for v in fit]
    every = FO.load()
    dohmen = every[(every["source"] == "dohmen2007_part1") & (every["log10_fO2_Pa"] > -10)]
    rows.append(("study's own", "Dohmen 2007 (T, axis removed)", *FO._within_study(dohmen, True)))
    petry = every[every["source"] == "petry2004"]
    rows.append(("study's own", "Petry 2004 (T removed)", *FO._within_study(petry, False)))
    for label, df in (("depth profiles", X.oeser(True)), ("all runs", X.oeser(False))):
        rows.append(
            ("study's own", f"Oeser 2026, {label} (T, axis removed)", *FO._within_study(df, True))
        )
    return pd.DataFrame(rows, columns=["form", "variant", "exponent", "se"])


def fig_fo2(pts, sigma, fit, ex, rng) -> None:
    form = F.FORMS["dc07_free"]
    n = len(form.params)
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(12, 6.4), layout="constrained", width_ratios=[1.25, 1]
    )
    fo2 = np.linspace(-10, 4, 71)
    v = F.Vars.of(pd.DataFrame({"log10_fO2_Pa": fo2, **MF.AT_1200}))
    rr1200, paper = FO.CONDITIONS_FO2["RR1200"], FO.CONDITIONS_FO2["Paper"]
    lo_rr, hi_rr = min(rr1200, paper) - MF.RR_BELOW_RR1200, max(rr1200, paper)
    a.axvspan(lo_rr, hi_rr, color="0.9", lw=0, zorder=0)
    a.text(
        (lo_rr + hi_rr) / 2,
        0.02,
        "round robin\nand paper fO2",
        transform=a.get_xaxis_transform(),
        ha="center",
        fontsize=7,
        color="0.35",
    )
    best0, solid0, _ = MF.curves(form, fit["baseline"], v, rng)
    MF.band(a, fo2, solid0, MF.BASE, "previous preset, 68%")
    a.plot(fo2, best0, color=MF.BASE, lw=1.4, label="previous preset")
    m_fit = fit["oeser_dp_melt"]
    best, solid, melt = MF.curves(form, m_fit, v, rng)
    MF.band(a, fo2, solid, MF.SOLID, "+ Oeser dp, solid contact, 68%")
    a.plot(fo2, best, color=MF.SOLID, lw=1.8, label="+ Oeser dp, solid contact")
    MF.band(a, fo2, melt, MF.MELT, "with melt term, 68%")
    a.plot(
        fo2, best + m_fit["beta"][n], color=MF.MELT, lw=1.6, label="melt contact (Spandler term)"
    )
    y = [F.preset_log10_d("dohmen_chakraborty2007", 1200.0, 1e5, g, 0.9) for g in fo2]
    a.plot(fo2, y, color="0.15", ls=":", lw=1.0, label="DC07 (default)")
    moved = F.moved(pts, form, m_fit["beta"][:n], **MF.AT_1200)
    in_fit = ~pts["source"].isin([MF.SHEA, OESER_LS]).to_numpy()
    MF.points(a, pts["log10_fO2_Pa"].to_numpy(), moved, sigma, pts, in_fit, style=STYLE)
    a.set_xlabel("log10 fO2 (Pa)")
    a.set_ylabel("log10 D[001] (m²/s) at 1200 °C, Fo90, 1 atm")
    a.set_title(
        "(a) points moved in T, Fo, P and axis by the fit with Oeser dp and a melt term", loc="left"
    )
    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", fontsize=6.5, frameon=False, ncol=5)
    MF.exponent_panel(b, ex, NAMES)
    b.legend(loc="center left", fontsize=6.5, frameon=False)  # clear of the rows
    b.set_title("(b) fO2 exponent of each fit and study", loc="left")
    get = ex.set_index("variant")["exponent"]
    own = get["Oeser 2026, depth profiles (T, axis removed)"]
    fig.suptitle(
        f"Adding Oeser's depth-profile runs moves the fitted fO2 exponent {get['baseline']:.3f} → "
        f"{get['oeser_dp']:.3f}; their own exponent is {own:.2f}.\nWith the linescan runs in, the "
        f"joint exponent falls to {get['oeser_all']:.3f} (they act through temperature)",
        fontsize=10,
    )
    fig.savefig(OUT / "fig2_fo2.png", dpi=200)
    plt.close(fig)


ROWS = (
    *MF.ROWS[:4],
    (OESER, "", "Oeser et al. 2026, depth profiles", False),
    (OESER_LS, "", "Oeser et al. 2026, linescan runs", False),
    *MF.ROWS[4:],
)


def offsets_title(summary: pd.DataFrame) -> str:
    """The headline leaves out Oeser's linescan runs, the suspect row; the row is shown."""
    melt = summary[summary["melt"] & (summary["set"] != "8 axis rows")]
    solid = summary[~summary["melt"] & (summary["source"] != OESER_LS)]
    ls = summary.loc[summary["source"] == OESER_LS, "mean"].iloc[0]
    return (
        f"Melt-contact studies sit {melt['mean'].min():+.2f} to {melt['mean'].max():+.2f} "
        f"log above DC07;\nsolid-source studies {solid['mean'].min():+.2f} to "
        f"{solid['mean'].max():+.2f}\n"
        f"(Oeser's linescan runs, convolution corrected: {ls:+.2f})"
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(MF.SEED)
    pts, sigma, sets = data()
    fit = fits(sets)
    table = MF.fig_bowties(
        pts, sigma, panels(pts, fit), rng, OUT / "fig1_bowties.png", bowtie_title, STYLE
    )
    table.to_csv(OUT / "fig1_bowties.csv", index=False)
    ex = exponents(sets, fit)
    ex.to_csv(OUT / "fig2_fo2_exponents.csv", index=False)
    fig_fo2(pts, sigma, fit, ex, rng)
    solid_pts = pts[pts["source"] != MF.SHEA]
    off = MF.fig_offsets(
        MF.offsets(solid_pts), rng, ROWS, STYLE, OUT / "fig3_offsets.png", offsets_title
    )
    off.to_csv(OUT / "fig3_offsets.csv", index=False)
    print(table.round(3).to_string(index=False))
    print(ex.round(3).to_string(index=False))
    print(off.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
