"""Figures for the melt-contact analysis (docs/references/melt-contact-diffusion.md).

fig1_bowties.png: Arrhenius bowties of the dc07_free random-study fit at the KS20
conditions (fO2 1.4e-3 Pa, Fo85, 42 MPa, [001]), drawn as log D minus the default law
(Dohmen & Chakraborty 2007) so differences of 0.1 log show: (a) as the femg_refit2026
preset, melt contact inside the between-study spread; (b) a melt term for Spandler and
O'Neill (2010); (c) one melt term for Spandler and Shea et al. (2023).
fig2_fo2.png: (a) data and laws against fO2 at 1200 C, Fo90, 1 atm, [001]; (b) the fO2
exponent of every variant, against each study's own and the published exponents.
fig3_offsets.png: every study's Fe-Mg residual against DC07 at its own conditions, so
Jurewicz and Watson (1988), whose fO2, Fo and orientation we do not have, can be placed:
their residuals, and Shea et al.'s full set, are Shea et al.'s own (their Fig 6, read
from the vector graphics by extract_shea2023_figs.py).

Bands are quantiles over draws from each fit's covariance (bowties.py). Points are
moved to a panel's conditions by that panel's fit; a melt term does not move them
(each point keeps its own contact). Shea et al.'s 8 profiles with a known axis are shown in every
panel, fainter where the fit leaves them out. Each figure's numbers are saved beside it.

Run: uv run python analyses/melt_contact/melt_figures.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "femg_refit"))

import bowties as B  # noqa: E402
import fit_models as F  # noqa: E402
import fo2_figures as FO  # noqa: E402
import melt_sensitivity as M  # noqa: E402

from forster import DIFFUSIVITY_PRESETS  # noqa: E402

OUT = HERE / "figures"
LN10 = math.log(10)
SEED = 20261014
SHEA = "shea2023"
SHEA_SIGMA = 0.3  # log10: Shea et al.'s error bars span 0.6 log units
STYLE = {**F.STUDY_STYLE, SHEA: ("#000000", "P", "Shea et al. 2023 (8 profiles with a known axis)")}
SOLID, MELT, BASE = "#0072B2", "#882255", "0.45"
T_GRID = np.linspace(700, 1600, 91)
KS20 = B.PANELS["b"][1:]  # log10 fO2 (Pa), X_Fo, P (Pa)
AT_1200 = {"T_K": 1473.15, "X_Fo": 0.9, "P_Pa": 1e5, "axis": "[001]"}
PANELS = {
    "a": ("baseline", (), "as the preset: melt inside the lab spread"),
    "b": ("melt_fixed_spandler", (M.SPANDLER,), "melt term: Spandler"),
    "c": ("melt_fixed_both", (M.SPANDLER, SHEA), "melt term: Spandler + Shea"),
}
RR_BELOW_KS20 = 0.9  # log units: the round robin runs reach 0.9 below KS20 (fo2_figures)
NEAR = 0.5  # log units either side of that span counted as "near our fO2"
FORM_NAME = {"dc07_free": "DC07 form: ", "mutch_x6": "Mutch form: "}
FORM_LEGEND = {"dc07_free": "DC07 form", "mutch_x6": "Mutch form, 6x", "study's own": "one study"}
VARIANT_NAME = {
    "baseline": "as the preset",
    "no_spandler": "without Spandler",
    "melt_fixed_spandler": "melt term: Spandler",
    "shea_random": "Shea as a sixth lab",
    "melt_fixed_both": "melt term: Spandler + Shea",
}
plt.rcParams.update({"pdf.fonttype": 42, "font.size": 9, "axes.titlesize": 9})


def data() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """The TaMED set, it plus Shea's 8 profiles with a known axis, and each row's log10 D error."""
    base = F.tamed()
    both = pd.concat([base, M.shea_femg_rows(M.load_shea())], ignore_index=True)
    sigma = F._y_sigma_ln(both) / LN10
    sigma[(both["source"] == SHEA).to_numpy()] = SHEA_SIGMA
    return base, both, sigma


def curves(form: F.Form, fit: dict, v: F.Vars, rng: np.random.Generator):
    """log10 D over parameter draws: solid contact, and with the melt term (or None)."""
    n = len(form.params)
    draws = B._draws(fit, rng)
    solid = (form.offset(v)[:, None] + form.design(v) @ draws[:, :n].T) / LN10
    melt = solid + draws[:, n] if draws.shape[1] > n else None
    best = (form.offset(v) + form.design(v) @ fit["beta"][:n]) / LN10
    return best, solid, melt


def band(ax, x, sample, colour, label68, label95=None):
    lo95, lo68, hi68, hi95 = np.quantile(sample, [0.025, 0.16, 0.84, 0.975], axis=1)
    if label95:
        ax.fill_between(x, lo95, hi95, color=colour, alpha=0.13, lw=0, label=label95)
    ax.fill_between(x, lo68, hi68, color=colour, alpha=0.32, lw=0, label=label68)
    return (hi68 - lo68) / 2


# Sources drawn with filled markers and kept in the legend when faded (not fitted).
FILLED = {SHEA, "oeser2026", "oeser2026_linescan"}


def points(ax, x, y, sigma, both: pd.DataFrame, in_fit: np.ndarray, faded=0.3, style=None):
    """Each study's points with their measurement error; points outside the fit faded."""
    for src, (colour, marker, label) in (style or STYLE).items():
        for fitted in (True, False):
            m = ((both["source"] == src).to_numpy()) & (in_fit == fitted)
            if not m.any():
                continue
            ax.errorbar(
                x[m],
                y[m],
                yerr=sigma[m],
                fmt=marker,
                color=colour,
                mfc=colour if src in FILLED else "none",
                ms=4.5 if src in FILLED else 4,
                elinewidth=0.6,
                capsize=0,
                alpha=1.0 if fitted else faded,
                label=label if fitted or src in FILLED else None,
            )


def melt_panels(both: pd.DataFrame, fits: dict) -> list[dict]:
    """The three panels of fig1: which fit, its title, and which points it uses."""
    out = []
    for key, (variant, melt_src, title) in PANELS.items():
        shea_in = SHEA in melt_src and variant != "baseline"
        out.append(
            {
                "key": key,
                "variant": variant,
                "fit": fits[variant],
                "title": title,
                "in_fit": (both["source"] != SHEA).to_numpy() | shea_in,
                "note": "faded crosses: Shea et al., not in this fit",
            }
        )
    return out


def melt_title(rows: list[dict]) -> str:
    a, b = rows[0], rows[1]
    return (
        f"A melt term takes melt contact out of the lab spread: τ {a['tau_log10']:.2f} → "
        f"{b['tau_log10']:.2f} log, and the solid-contact law at KS20, 1200 °C moves from "
        f"{a['KS20_1200C_vs_dc07']:+.2f} to {b['KS20_1200C_vs_dc07']:+.2f} log above DC07"
    )


def fig_bowties(both, sigma, panels, rng, path=None, title=melt_title, style=None) -> pd.DataFrame:
    form = F.FORMS["dc07_free"]
    n = len(form.params)
    log_fo2, x_fo, p_pa = KS20
    v = B._grid(T_GRID, log_fo2, x_fo, p_pa)
    x = 1e4 / (T_GRID + 273.15)
    dc07 = B._published("dohmen_chakraborty2007", T_GRID, log_fo2, x_fo, p_pa)
    at = {"log10_fO2_Pa": log_fo2, "X_Fo": x_fo, "P_Pa": p_pa, "axis": "[001]"}
    dc07_pts = np.array(
        [
            F.preset_log10_d("dohmen_chakraborty2007", t - 273.15, p_pa, log_fo2, x_fo)
            for t in both["T_K"]
        ]
    )
    x_pts = 1e4 / both["T_K"].to_numpy()
    i1200 = int(np.argmin(np.abs(T_GRID - 1200)))
    fig, axes = plt.subplots(1, 3, figsize=(12, 5.4), sharey=True, layout="constrained")
    rows = []
    for ax, panel in zip(axes, panels, strict=True):
        key, variant, fit, in_fit = panel["key"], panel["variant"], panel["fit"], panel["in_fit"]
        best, solid, melt = curves(form, fit, v, rng)
        half = band(ax, x, solid - dc07[:, None], SOLID, "law, 68%", "law, 95%")
        ax.plot(x, best - dc07, color=SOLID, lw=1.8, label="best fit (solid contact)")
        tau = fit["tau"] / LN10
        new_lab = np.hypot(half, tau)
        ax.plot(x, best - dc07 + new_lab, color=SOLID, lw=0.9, ls="--", label="68%, one new lab")
        ax.plot(x, best - dc07 - new_lab, color=SOLID, lw=0.9, ls="--")
        row = {
            "panel": key,
            "variant": variant,
            "tau_log10": tau,
            "KS20_1200C_vs_dc07": best[i1200] - dc07[i1200],
            "band68_1200C": half[i1200],
            "new_lab68_1200C": new_lab[i1200],
        }
        melt_txt = ""
        if melt is not None:
            band(ax, x, melt - dc07[:, None], MELT, "with melt term, 68%")
            m_best, m_se = float(fit["beta"][n]), math.sqrt(fit["cov"][n, n])
            ax.plot(x, best - dc07 + m_best, color=MELT, lw=1.6, label="best fit + melt term")
            row |= {"melt_log10": m_best, "melt_se": m_se}
            melt_txt = f"\nmelt term {m_best:+.2f} ± {m_se:.2f}"
        moved = F.moved(both, form, fit["beta"][:n], **at)
        points(ax, x_pts, moved - dc07_pts, sigma, both, in_fit, style=style)
        if not in_fit.all():
            ax.text(
                0.98,
                0.97,
                panel["note"],
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7,
                color="0.35",
            )
        ax.axhline(0, color="0.2", lw=0.9, ls=":", label="DC07 (default law)")
        ax.axvline(x[i1200], color="0.6", lw=0.7, ls=":")
        ax.set_title(
            f"({key}) {panel['title']}\nτ {tau:.2f}; at 1200 °C {row['KS20_1200C_vs_dc07']:+.2f}, "
            f"±{half[i1200]:.2f} law, ±{new_lab[i1200]:.2f} new lab{melt_txt}",
            loc="left",
        )
        ax.set_xlabel("10⁴ / T (1/K)")
        lo, hi = ax.get_xlim()
        top = ax.twiny()
        top.set_xlim(lo, hi)
        marks = [c for c in (800, 1000, 1200, 1400, 1600) if lo <= 1e4 / (c + 273.15) <= hi]
        top.set_xticks([1e4 / (c + 273.15) for c in marks], [str(c) for c in marks])
        top.set_xlabel("T (°C)")
        rows.append(row)
    axes[0].set_ylabel("log10 D[001] minus DC07 at the KS20 conditions")
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=6, fontsize=7, frameon=False)
    fig.suptitle(title(rows), fontsize=10)
    fig.savefig(path or OUT / "fig1_bowties.png", dpi=200)
    plt.close(fig)
    return pd.DataFrame(rows)


def _exponent(form: F.Form, fit: dict) -> tuple[float, float]:
    i = form.params.index("m_fO2" if "m_fO2" in form.params else "b")
    return float(fit["beta"][i]), float(math.sqrt(fit["cov"][i, i]))


def exponents(base, both) -> pd.DataFrame:
    """fO2 exponent (1 sigma) of every variant and form, and each study's own."""
    rows = []
    for form_name in ("dc07_free", "mutch_x6"):
        form = F.FORMS[form_name]
        for variant, fit in M.variant_fits(form, base, both).items():
            rows.append((form_name, variant, *_exponent(form, fit)))
    every = FO.load()
    dohmen = every[(every["source"] == "dohmen2007_part1") & (every["log10_fO2_Pa"] > -10)]
    rows.append(("study's own", "Dohmen 2007 (T, axis removed)", *FO._within_study(dohmen, True)))
    petry = every[every["source"] == "petry2004"]
    rows.append(("study's own", "Petry 2004 (T removed)", *FO._within_study(petry, False)))
    return pd.DataFrame(rows, columns=["form", "variant", "exponent", "se"])


def fig_fo2(base, both, sigma, fit, ex: pd.DataFrame, rng) -> None:
    form = F.FORMS["dc07_free"]
    n = len(form.params)
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(11.5, 6.2), layout="constrained", width_ratios=[1.3, 1]
    )
    fo2 = np.linspace(-10, 4, 71)
    v = F.Vars.of(pd.DataFrame({"log10_fO2_Pa": fo2, **AT_1200}))
    ks20, paper = FO.CONDITIONS_FO2["KS20"], FO.CONDITIONS_FO2["Paper"]
    lo_rr, hi_rr = min(ks20, paper) - RR_BELOW_KS20, max(ks20, paper)
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
    b_best, b_solid, _ = curves(form, fit["baseline"], v, rng)
    band(a, fo2, b_solid, BASE, "as the preset, 68%")
    a.plot(fo2, b_best, color=BASE, lw=1.4, label="as the preset")
    best, solid, melt = curves(form, fit["melt_fixed_both"], v, rng)
    band(a, fo2, solid, SOLID, "solid contact (melt term fitted), 68%")
    a.plot(fo2, best, color=SOLID, lw=1.8, label="solid contact")
    band(a, fo2, melt, MELT, "with melt term, 68%")
    a.plot(fo2, best + fit["melt_fixed_both"]["beta"][n], color=MELT, lw=1.6, label="melt contact")
    for name, ls, label in (
        ("dohmen_chakraborty2007", ":", "DC07 (default)"),
        ("mutch2019_tamed", "-.", "Mutch 2019 TaMED"),
    ):
        y = [F.preset_log10_d(name, 1200.0, 1e5, g, 0.9) for g in fo2]
        a.plot(fo2, y, color="0.15", ls=ls, lw=1.0, label=label)
    moved = F.moved(both, form, fit["melt_fixed_both"]["beta"][:n], **AT_1200)
    points(a, both["log10_fO2_Pa"].to_numpy(), moved, sigma, both, np.ones(len(both), bool))
    a.set_xlabel("log10 fO2 (Pa)")
    a.set_ylabel("log10 D[001] (m²/s) at 1200 °C, Fo90, 1 atm")
    a.set_title(
        "(a) points moved in T, Fo, P and axis by the Spandler + Shea melt-term fit", loc="left"
    )
    handles, labels = a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", fontsize=6.5, frameon=False, ncol=5)

    exponent_panel(b, ex)
    b.set_title("(b) fO2 exponent of each fit and study", loc="left")
    base_m = ex[(ex.form == "dc07_free") & (ex.variant == "baseline")].exponent.iloc[0]
    melt_m = ex[(ex.form == "dc07_free") & (ex.variant == "melt_fixed_both")].exponent.iloc[0]
    one_atm = both[both["P_Pa"] < 2e5]
    near = one_atm["log10_fO2_Pa"].between(lo_rr - NEAR, hi_rr + NEAR)
    names = [STYLE[s][2].split(" (")[0] for s in STYLE if s in set(one_atm[near]["source"])]
    melt = fit["melt_fixed_both"]["beta"][n]
    fig.suptitle(
        f"Separating melt contact moves the fO2 exponent only {base_m:.3f} → {melt_m:.3f}: "
        f"the melt studies shift the level ({melt:+.2f} log), not the slope.\n"
        f"1-atm data within {NEAR} log of our fO2: " + ", ".join(names),
        fontsize=10,
    )
    fig.savefig(OUT / "fig2_fo2.png", dpi=200)
    plt.close(fig)


def exponent_panel(b, ex: pd.DataFrame, names=None) -> None:
    """One row per fit or study: the fO2 exponent and its 1 sigma, against DC07 and Mutch."""
    names = {**VARIANT_NAME, **(names or {})}
    colour = {"dc07_free": SOLID, "mutch_x6": "#E69F00", "study's own": "0.3"}
    marker = {"dc07_free": "o", "mutch_x6": "s", "study's own": "D"}
    labels = []
    for i, r in enumerate(ex.itertuples()):
        b.errorbar(
            r.exponent,
            i,
            xerr=r.se,
            fmt=marker[r.form],
            color=colour[r.form],
            ms=5,
            capsize=0,
            label=FORM_LEGEND[r.form] if r.form not in [lab for lab, _ in labels] else None,
        )
        labels.append((r.form, r.variant))
    b.set_yticks(range(len(ex)), [f"{FORM_NAME.get(f, '')}{names.get(v, v)}" for f, v in labels])
    b.invert_yaxis()
    for value, ls, label in (
        (DIFFUSIVITY_PRESETS["dohmen_chakraborty2007"].fO2_exponent, ":", "DC07 1/6"),
        (DIFFUSIVITY_PRESETS["mutch2019_tamed"].fO2_exponent, "-.", "Mutch 2019 TaMED"),
    ):
        b.axvline(value, color="0.15", ls=ls, lw=1.0, label=label)
    b.set_xlabel("fO2 exponent, d log D / d log fO2 (±1 sigma)")
    b.legend(loc="upper right", fontsize=6.5, frameon=False)


def offsets(base: pd.DataFrame) -> pd.DataFrame:
    """Residual against DC07 at each point's own conditions, one row per point."""
    axis = {"[100]": 0, "[010]": 1, "[001]": 2}
    own = base.assign(
        resid=[
            r.log10_D_m2s
            - F.preset_log10_d(
                "dohmen_chakraborty2007",
                r.T_K - 273.15,
                r.P_Pa,
                r.log10_fO2_Pa,
                r.X_Fo,
                axis[r.axis],
            )
            for r in base.itertuples()
        ],
        set="",
    )[["source", "set", "resid"]]
    figs = pd.read_csv(HERE / "data" / "shea2023_figs_extracted.csv").dropna(
        subset=["resid_dc07_shea"]
    )
    figs = figs.assign(resid=figs["resid_dc07_shea"], set="all, Shea Fig 6")
    pinned = pd.read_csv(HERE / "results" / "shea_residuals.csv")
    pinned = pinned[(pinned["species"] == "Fe-Mg") & pinned["axis"].notna()].assign(
        source=SHEA, set="8 axis rows", resid=lambda d: d["resid_law_shea"]
    )
    return pd.concat(
        [own, figs[["source", "set", "resid"]], pinned[["source", "set", "resid"]]],
        ignore_index=True,
    )


ROWS = (  # (source, set, label, melt contact)
    ("chakraborty1997", "", "Chakraborty 1997", False),
    ("dohmen2007_part1", "", "Dohmen et al. 2007", False),
    ("petry2004", "", "Petry et al. 2004", False),
    ("holzapfel2007", "", "Holzapfel et al. 2007 (6-12 GPa)", False),
    ("spandler_oneill2010", "", "Spandler & O'Neill 2010", True),
    ("jurewicz_watson1988", "all, Shea Fig 6", "Jurewicz & Watson 1988 (Shea Fig 6)", True),
    (SHEA, "all, Shea Fig 6", "Shea et al. 2023, all 20 (Fig 6; not fittable)", True),
    (SHEA, "8 axis rows", "Shea et al. 2023, 8 profiles with a known axis (used in fits)", True),
)
JW_STYLE = ("0.55", "s", "Jurewicz & Watson 1988")


def fig_offsets(
    off: pd.DataFrame, rng, rows_spec=ROWS, style=None, path=None, title=None
) -> pd.DataFrame:
    style = {**STYLE, "jurewicz_watson1988": JW_STYLE, **(style or {})}
    height = round(0.55 * len(rows_spec) + 0.2, 2)  # 4.6 in for the 8 original rows
    fig, ax = plt.subplots(figsize=(9.5, height), layout="constrained")
    rows = []
    for i, (src, subset, _, melt) in enumerate(rows_spec):
        r = off[(off["source"] == src) & (off["set"] == subset)]["resid"].to_numpy()
        colour, marker, _ = style[src]
        y = i + rng.uniform(-0.18, 0.18, len(r))
        ax.plot(
            r, y, marker, color=colour, mfc=colour if src in FILLED else "none", ms=4, alpha=0.7
        )
        m, sd = r.mean(), r.std(ddof=1)
        ax.errorbar(m, i, xerr=sd, fmt="|", color="0.1", ms=14, mew=2, elinewidth=1.4, capsize=0)
        ax.annotate(
            f"{m:+.2f} ± {sd:.2f} (n {len(r)})",
            (max(r.max(), m + sd), i),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            fontsize=7.5,
        )
        rows.append({"source": src, "set": subset, "mean": m, "sd": sd, "n": len(r), "melt": melt})
    labels = [label for _, _, label, _ in rows_spec]
    ax.set_yticks(range(len(rows_spec)), labels)
    for tick, (*_, melt) in zip(ax.get_yticklabels(), rows_spec, strict=True):
        tick.set_color(MELT if melt else "0.1")
    ax.invert_yaxis()
    ax.axvline(0, color="0.2", lw=0.9, ls=":")
    first_melt = next(i for i, (*_, melt) in enumerate(rows_spec) if melt)
    ax.axhline(first_melt - 0.5, color="0.7", lw=0.8)
    ax.set_xlim(right=ax.get_xlim()[1] + 0.6)  # room for the mean labels
    ax.set_xlabel("log10 D measured minus DC07 at the experiment's own conditions")
    summary = pd.DataFrame(rows)
    solid, melt = (
        summary[~summary["melt"]],
        summary[summary["melt"] & (summary["set"] != "8 axis rows")],
    )
    lo, hi = melt["mean"].min(), melt["mean"].max()
    text = (
        f"Melt-contact studies sit {lo:+.2f} to {hi:+.2f} log above DC07;\n"
        f"solid-source studies {solid['mean'].min():+.2f} to {solid['mean'].max():+.2f}"
    )
    ax.set_title(title(summary) if title else text, loc="left", fontsize=10)
    fig.savefig(path or OUT / "fig3_offsets.png", dpi=200)
    plt.close(fig)
    return summary


def main() -> None:
    OUT.mkdir(exist_ok=True)
    rng = np.random.default_rng(SEED)
    base, both, sigma = data()
    fits = M.variant_fits(F.FORMS["dc07_free"], base, both)
    table = fig_bowties(both, sigma, melt_panels(both, fits), rng)
    table.to_csv(OUT / "fig1_bowties.csv", index=False)
    ex = exponents(base, both)
    ex.to_csv(OUT / "fig2_fo2_exponents.csv", index=False)
    fig_fo2(base, both, sigma, fits, ex, rng)
    fig_offsets(offsets(base), rng).to_csv(OUT / "fig3_offsets.csv", index=False)


if __name__ == "__main__":
    main()
