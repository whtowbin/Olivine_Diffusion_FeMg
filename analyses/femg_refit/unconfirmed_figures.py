"""What the unconfirmed data suggest, against the verified law (femg_refit2026).

The preset is fitted to the verified data only (fit_verified.py). These figures put
everything else beside it, labelled, so the reader sees what each unconfirmed set would
do; docs/references/data-to-source.md says what would confirm each.

- figures/unconfirmed/u1_law_and_data.png: the verified law at the KS20 conditions
  (log D minus DC07, as the melt-contact bowties), its verified points, the unconfirmed
  points moved by the same law, and as boxes the melt-contact evidence that cannot be
  placed point by point (Jurewicz and Watson; Shea et al.'s full set; the Kilauea Iki
  lava lake, a lower bound).
- figures/unconfirmed/u2_variants.png: the KS20 and SH63 levels and their 1-sigma
  (law, and one new lab) for the verified law and for each unconfirmed addition.
- figures/unconfirmed/u3_fo2.png: the verified law against fO2 at 1200 C with the
  unconfirmed data; Holzapfel's Au-capsule runs carry their assumed fO2 (Ni-NiO) with a
  bar of +-2 log units.

Run: uv run python analyses/femg_refit/unconfirmed_figures.py
Writes the figures and results/verified/scenarios.csv.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "melt_contact"))

import fit_extended as X  # noqa: E402
import fit_models as F  # noqa: E402
import fit_verified as V  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import melt_figures as MF  # noqa: E402
import melt_sensitivity as M  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

OUT = HERE / "figures" / "unconfirmed"
RESULTS = HERE / "results" / "verified"
LN10 = math.log(10)
FORM = V.FORM  # the preset: DC07 form, 6x anisotropy
FREE = F.FORMS["dc07_free"]  # anisotropy fitted (the presets before 2026-10-15)
SEED = 20261014
AU, OESER_LS, SHEA = "holzapfel2007_au", "oeser2026_linescan", "shea2023"
FO2_BAR = 2.0  # log units drawn on the Au runs' assumed fO2
STYLE = {
    **F.STUDY_STYLE,
    "oeser2026": ("#117733", "X", "Oeser et al. 2026, depth profiles"),
    AU: ("#D55E00", "d", "Holzapfel 2007, Au capsules (fO2 assumed)"),
    OESER_LS: ("#117733", "x", "Oeser 2026, linescan runs"),
    SHEA: ("#000000", "P", "Shea et al. 2023 (8 profiles with a known axis)"),
}
# Filled markers, kept in the legend when faded: every unconfirmed set.
FILLED = MF.FILLED | {AU, OESER_LS, M.SPANDLER}
HATCHES = ("//", "..", "xx")  # one per melt-contact box, so the legend tells them apart
VERIFIED = ("chakraborty1997", "dohmen2007_part1", "petry2004", "holzapfel2007")
# Melt-contact evidence without per-point conditions: (label, T range C, mean, sd vs DC07).
BOXES = (
    ("Jurewicz & Watson 1988\n(read from Shea Fig 6)", (1219, 1298), 0.71, 0.19, False),
    ("Shea et al. 2023, all 20\n(read from Fig 6)", (1200, 1290), 0.96, 0.17, False),
    ("Kilauea Iki lava lake\n(5-15x DC07, lower bound)", (1090, 1150), 0.94, 0.24, True),
)


def frames():
    """Verified data, the unconfirmed additions, and every point to draw."""
    ver = V.verified()
    oes_all = X.oeser(False)
    ls = oes_all[oes_all["run"].isin(X.LINESCAN_RUNS)]
    au = V.holzapfel_au_rows()
    t = F.tamed()
    spandler = t[t["source"] == M.SPANDLER]
    shea = M.shea_femg_rows(M.load_shea())
    dp = V.oeser_dp()
    show = pd.concat(
        [
            ver,
            dp,
            au.assign(source=AU),
            ls.assign(source=OESER_LS),
            spandler,
            shea,
        ],
        ignore_index=True,
    )
    sigma = F._y_sigma_ln(show) / LN10
    sigma[show["source"].isin([SHEA]).to_numpy()] = 0.3
    sigma[show["source"].isin(["oeser2026", OESER_LS]).to_numpy()] = 0.0  # not given
    au_rows = show["source"] == AU
    sigma[au_rows.to_numpy()] = np.hypot(0.3, 0.22 * FO2_BAR)  # with the fO2 prior
    return ver, {"au": au, "ls": ls, "dp": dp, "spandler": spandler, "shea": shea}, show, sigma


def scenarios(ver, add) -> dict[str, tuple[F.Form, dict, int]]:
    """Each fit: (form, fit, n). Melt-term fits predict for solid contact."""
    out = {
        "first preset (all data, fitted anisotropy, Spandler a study)": (
            FREE,
            F.fit_random(FREE, F.tamed()),
            135,
        )
    }
    out["verified (the preset)"] = (FORM, F.fit_random(FORM, ver), len(ver))
    for sd in V.FO2_PRIOR_SD:
        fit, n = V.with_au(sd)
        out[f"+ Holzapfel Au, NNO, sd {sd:g} log"] = (FORM, fit, n)
    dp = add["dp"]
    for label, extra in (
        ("+ Oeser depth profiles", dp),
        ("+ Oeser [001] and [100] only", dp[dp["axis"] != "[010]"]),
        ("+ Oeser, all runs incl. linescans", pd.concat([dp, add["ls"]])),
    ):
        d = pd.concat([ver, extra], ignore_index=True)
        out[label] = (FORM, F.fit_random(FORM, d), len(d))
    d = pd.concat([ver, dp], ignore_index=True)
    out["+ Oeser depth profiles, fitted anisotropy"] = (FREE, F.fit_random(FREE, d), len(d))
    d = pd.concat([ver, add["spandler"]], ignore_index=True)
    out["+ Spandler as a study"] = (FORM, F.fit_random(FORM, d), len(d))
    out["+ Spandler, melt term"] = (
        FORM,
        F.fit_random(M.melt_form(FORM, d, (M.SPANDLER,)), d),
        len(d),
    )
    d2 = pd.concat([d, add["shea"]], ignore_index=True)
    out["+ Spandler + Shea (8), melt term"] = (
        FORM,
        F.fit_random(M.melt_form(FORM, d2, (M.SPANDLER, SHEA)), d2),
        len(d2),
    )
    return out


def table(fits) -> pd.DataFrame:
    rows = []
    conds = {k.split(",")[0]: v for k, v in F._conditions().items()}
    for name, (form, fit, n) in fits.items():
        b, cov, tau = fit["beta"], fit["cov"], fit["tau"] / LN10
        row = {
            "scenario": name,
            "n_points": n,
            "tau_log10": tau,
            "m_fO2": b[2],
            "E_kJ_mol": b[1] / 1e3,
        }
        k = len(form.params)
        if len(b) > k:
            row |= {"melt_log10": b[k], "melt_se": math.sqrt(cov[k, k])}
        for label, c in conds.items():
            mu, se = F.predict(form, b[:k], cov[:k, :k], c)
            row |= {
                f"{label}_vs_dc07": mu - F._dc07_preset(c),
                f"{label}_se_law": se,
                f"{label}_se_new_lab": math.hypot(se, tau),
            }
        rows.append(row)
    return pd.DataFrame(rows)


def u1(show, sigma, fit, rng) -> None:
    log_fo2, x_fo, p_pa = MF.KS20
    t = MF.T_GRID
    v = MF.B._grid(t, log_fo2, x_fo, p_pa)
    x = 1e4 / (t + 273.15)
    dc07 = MF.B._published("dohmen_chakraborty2007", t, log_fo2, x_fo, p_pa)
    best, solid, _ = MF.curves(FORM, fit, v, rng)
    fig, ax = plt.subplots(figsize=(10, 6.4), layout="constrained")
    half = MF.band(ax, x, solid - dc07[:, None], MF.SOLID, "verified law, 68%", "verified law, 95%")
    ax.plot(x, best - dc07, color=MF.SOLID, lw=1.8, label="verified law (femg_refit2026)")
    tau = fit["tau"] / LN10
    nl = np.hypot(half, tau)
    ax.plot(x, best - dc07 + nl, color=MF.SOLID, lw=0.9, ls="--", label="68%, one new lab")
    ax.plot(x, best - dc07 - nl, color=MF.SOLID, lw=0.9, ls="--")
    at = {"log10_fO2_Pa": log_fo2, "X_Fo": x_fo, "P_Pa": p_pa, "axis": "[001]"}
    moved = F.moved(show, FORM, fit["beta"], **at)
    dpts = np.array(
        [
            F.preset_log10_d("dohmen_chakraborty2007", tk - 273.15, p_pa, log_fo2, x_fo)
            for tk in show["T_K"]
        ]
    )
    in_fit = show["source"].isin(VERIFIED).to_numpy()
    MF.points(
        ax,
        1e4 / show["T_K"].to_numpy(),
        moved - dpts,
        sigma,
        show,
        in_fit,
        faded=0.45,
        style=STYLE,
        filled=FILLED,
    )
    for (label, (t0, t1), mean, sd, lower_bound), hatch in zip(BOXES, HATCHES, strict=True):
        x0, x1 = 1e4 / (t1 + 273.15), 1e4 / (t0 + 273.15)
        ax.add_patch(
            plt.Rectangle(
                (x0, mean - sd),
                x1 - x0,
                2 * sd,
                facecolor=MF.MELT,
                alpha=0.22,
                hatch=hatch,
                edgecolor=MF.MELT,
                lw=0,
                label=label.replace("\n", " "),
            )
        )
        ax.plot([x0, x1], [mean, mean], color=MF.MELT, lw=1.6)
        if lower_bound:
            ax.annotate(
                "",
                (x1 + 0.05, mean + sd + 0.25),
                (x1 + 0.05, mean - sd),
                arrowprops={"arrowstyle": "->", "color": MF.MELT, "lw": 1.2},
            )
    ax.axhline(0, color="0.2", lw=0.9, ls=":", label="DC07 (default law)")
    i1200 = int(np.argmin(np.abs(t - 1200)))
    ax.set_xlabel("10⁴ / T (1/K)")
    ax.set_ylabel("log10 D[001] minus DC07 at the KS20 conditions")
    lo, hi = ax.get_xlim()
    top = ax.twiny()
    top.set_xlim(lo, hi)
    marks = [c for c in (800, 1000, 1200, 1400, 1600) if lo <= 1e4 / (c + 273.15) <= hi]
    top.set_xticks([1e4 / (c + 273.15) for c in marks], [str(c) for c in marks])
    top.set_xlabel("T (°C)")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=7, frameon=False)
    ax.set_title(
        f"The verified law sits {best[i1200] - dc07[i1200]:+.2f} ± {half[i1200]:.2f} log "
        "above DC07 "
        f"at KS20, 1200 °C (±{nl[i1200]:.2f} for one new lab).\nFaded: unconfirmed data, not "
        "fitted. Pink boxes: melt-contact evidence (mean ± sd), placed by temperature only",
        loc="left",
        fontsize=9.5,
    )
    fig.savefig(OUT / "u1_law_and_data.png", dpi=200)
    plt.close(fig)


def u2(tab: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained", sharey=True)
    names = tab["scenario"].tolist()
    y = np.arange(len(names))
    for ax, cond in zip(axes, ("KS20", "SH63"), strict=True):
        mu, law, nl = tab[f"{cond}_vs_dc07"], tab[f"{cond}_se_law"], tab[f"{cond}_se_new_lab"]
        colours = [
            MF.SOLID if "verified (" in n else ("#882255" if "melt" in n else "0.35") for n in names
        ]
        ax.errorbar(mu, y, xerr=nl, fmt="none", ecolor="0.7", elinewidth=1, capsize=0)
        for yi, m, s, c in zip(y, mu, law, colours, strict=True):
            ax.errorbar(m, yi, xerr=s, fmt="o", color=c, ms=5, elinewidth=2.2, capsize=0)
        ref = float(mu[names.index("verified (the preset)")])
        ax.axvline(ref, color=MF.SOLID, lw=0.8, ls=":")
        ax.axvline(0, color="0.2", lw=0.8, ls=":")
        ax.set_xlabel(f"D[001] at {cond} minus DC07 (log)")
        ax.set_title(f"{cond}", loc="left")
    labels = []
    for r in tab.itertuples():
        extra = (
            f"  [melt {r.melt_log10:+.2f}]"
            if not math.isnan(getattr(r, "melt_log10", float("nan")))
            else ""
        )
        labels.append(f"{r.scenario}{extra}")
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    solid_src = tab["scenario"].str.contains("Holzapfel|Oeser")  # unconfirmed solid-source rows
    ref = tab.loc[tab["scenario"] == "verified (the preset)", "KS20_vs_dc07"].iloc[0]
    shift = (tab.loc[solid_src, "KS20_vs_dc07"] - ref).abs()
    fig.suptitle(
        "Unconfirmed solid-source data would move the law at KS20 by at most "
        f"{shift.max():.2f} log; "
        "melt contact is a separate offset (in brackets).\nThick bars: the law's 1 sigma; "
        "thin: one new lab",
        fontsize=10,
    )
    fig.savefig(OUT / "u2_variants.png", dpi=200)
    plt.close(fig)


def u3(show, sigma, fit, rng) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 6.2), layout="constrained")
    fo2 = np.linspace(-10, 4, 71)
    v = F.Vars.of(pd.DataFrame({"log10_fO2_Pa": fo2, **MF.AT_1200}))
    best, solid, _ = MF.curves(FORM, fit, v, rng)
    MF.band(ax, fo2, solid, MF.SOLID, "verified law, 68%")
    ax.plot(fo2, best, color=MF.SOLID, lw=1.8, label="verified law")
    ax.plot(
        fo2,
        [F.preset_log10_d("dohmen_chakraborty2007", 1200.0, 1e5, g, 0.9) for g in fo2],
        color="0.15",
        ls=":",
        lw=1.0,
        label="DC07 (default)",
    )
    moved = F.moved(show, FORM, fit["beta"], **MF.AT_1200)
    in_fit = show["source"].isin(VERIFIED).to_numpy()
    MF.points(
        ax,
        show["log10_fO2_Pa"].to_numpy(),
        moved,
        sigma,
        show,
        in_fit,
        faded=0.45,
        style=STYLE,
        filled=FILLED,
    )
    au = (show["source"] == AU).to_numpy()
    ax.errorbar(
        show["log10_fO2_Pa"][au],
        moved[au],
        xerr=FO2_BAR,
        fmt="none",
        ecolor="#D55E00",
        alpha=0.35,
        elinewidth=0.8,
        capsize=0,
    )
    ks20, paper = -2.86, -2.08
    ax.axvspan(ks20 - 0.9, paper, color="0.9", lw=0, zorder=0)
    ax.set_xlabel("log10 fO2 (Pa)")
    ax.set_ylabel("log10 D[001] (m²/s) at 1200 °C, Fo90, 1 atm (points moved by the verified law)")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=7, frameon=False)
    ax.set_title(
        f"Verified law: fO2 exponent {fit['beta'][2]:.3f}. Faded: unconfirmed data; Holzapfel's "
        f"Au runs at Ni-NiO\nwith ±{FO2_BAR:g} log bars (their fO2 was not buffered). "
        "Shaded: round "
        "robin and paper fO2",
        loc="left",
        fontsize=9.5,
    )
    fig.savefig(OUT / "u3_fo2.png", dpi=200)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    ver, add, show, sigma = frames()
    fits = scenarios(ver, add)
    tab = table(fits)
    tab.to_csv(RESULTS / "scenarios.csv", index=False)
    preset = fits["verified (the preset)"][1]
    u1(show, sigma, preset, rng)
    u2(tab)
    u3(show, sigma, preset, rng)
    pd.set_option("display.width", 220)
    cols = ["scenario", "n_points", "tau_log10", "m_fO2", "E_kJ_mol", "KS20_vs_dc07", "KS20_se_law"]
    cols += ["SH63_vs_dc07", "melt_log10", "melt_se"]
    print(tab[cols].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
