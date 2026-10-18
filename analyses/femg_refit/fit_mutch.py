"""The Mutch et al. (2019) forms fitted to the preset's data by the preset's method.

    ln D[001] = a + b ln fO2(bar) + c X_Fo + q/T + j P + h P/T   (P in Pa, T in K)

On the verified data alone this form is singular: its two pressure terms need
high-pressure data at more than one temperature. Holzapfel et al.'s Au-capsule runs
(in the preset at Ni-NiO, fO2 sd 2 log) supply four runs at 12 GPa from 1350 to 1550 C,
so the form is identifiable on `fit_verified.preset_data()`. Fitted by
`fit_verified.fit_preset` (REML with a random offset per study and each run's input
errors), as the femg_refit2026 preset:

- mutch_x6: [100] and [010] 6x slower than [001], as DFENS and the preset;
  (the activation volume is reported at 1400 C as V = -R (j T + h); j and h alone are
  nearly collinear, as the high-pressure runs span one pair of temperatures at 12 GPa)
- mutch_aniso: both anisotropy factors fitted;
- mutch_fT: mutch_aniso plus an fO2 x 1/T term (the fO2 exponent varying with T);
- dc07_x6_V, the preset's form (V fitted since 2026-10-15), and dc07_x6 with V fixed at
  7 cm3/mol (the preset's form before), for reference; and the DFENS presets
  mutch2019_tamed and mutch2019_global (Mutch et al. 2021, Data Set S6).

Control: the input-error slopes by central differences (used for the Mutch forms) must
equal the closed form on the preset's form.

figures/presets/p4_mutch_forms.png: (a) log10 D[001] minus DC07 against T at RR1200,
(b) log10 D[001] against P at 1400 C, Fo97, Ni-NiO, with Holzapfel's runs, (c) the levels
at RR1200, RR1102 and the paper conditions.

Run: uv run python analyses/femg_refit/fit_mutch.py
Writes results/verified/mutch_forms.csv.
"""

from __future__ import annotations

import math
from pathlib import Path

import fit_models as F
import fit_verified as V
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forster import DIFFUSIVITY_PRESETS, Conditions, Orientation, engine

HERE = Path(__file__).parent
OUT = HERE / "figures" / "presets"
LN10, LOG6 = math.log(10), math.log10(6)
ALONG_001 = Orientation(90, 90, 0)
# The preset's form fits V (log D0, E, m, n, V): the Mutch form without j. Its previous
# form, V fixed at 7 cm3/mol, is kept for comparison.
FORMS = {
    "dc07_x6_V (the preset)": V.FORM,
    "dc07_x6, V fixed at 7": V.FORM_V7,
    "mutch_x6": F.FORMS["mutch_x6"],
    "mutch_aniso": F.FORMS["mutch_aniso"],
    "mutch_fT": F.FORMS["mutch_fT"],
}
DFENS = ("mutch2019_tamed", "mutch2019_global")
STYLE = {  # colour, line style
    "dc07_x6_V (the preset)": ("#0072B2", "-"),
    "dc07_x6, V fixed at 7": ("#000000", (0, (3, 1, 1, 1))),
    "mutch_x6": ("#009E73", "--"),
    "mutch_aniso": ("#E69F00", "-."),
    "mutch_fT": ("#56B4E9", ":"),
    "mutch2019_tamed": ("#D55E00", (0, (5, 2))),
    "mutch2019_global": ("#CC79A7", (0, (1, 1))),
}
PANEL_B = (1400.0, 0.97)  # Holzapfel's temperature and composition
T_V = 1673.15  # K: the Mutch forms' activation volume is reported at 1400 C
plt.rcParams.update({"pdf.fonttype": 42, "font.size": 9, "axes.titlesize": 9.5})


def check_slopes(df: pd.DataFrame, beta: np.ndarray) -> None:
    """Central differences through a copy of the preset's form (renamed, so the closed
    form is not used) must give the closed-form slopes."""
    copy = F.Form("dc07_x6_V copy", tuple(f"p{i}" for i in range(5)), V.FORM.design, V.FORM.offset)
    exact, numeric = V.input_slopes(df, beta), V.input_slopes(df, beta, copy)
    worst = max(
        float(np.max(np.abs(n - e) / np.maximum(np.abs(e), 1e-30)))
        for e, n in zip(exact, numeric, strict=True)
    )
    print(f"control: input slopes, central differences against closed form: {worst:.1e}")
    assert worst < 1e-6, "central-difference slopes do not match the closed form"


def forster_units(name: str, beta: np.ndarray, cov: np.ndarray) -> dict:
    """The Mutch coefficients as forster's parameters (as diffusivity._mutch2019), +- se."""
    se = np.sqrt(np.diag(cov))
    out = {"log10_D0_001": beta[0] / LN10, "log10_D0_001_se": se[0] / LN10}
    out |= {"fO2_exponent": beta[1], "fO2_exponent_se": se[1]}
    out |= {"fo_coefficient": -beta[2] / LN10, "fo_coefficient_se": se[2] / LN10}
    out |= {"E_kJ_mol": -beta[3] * F.R / 1e3, "E_se_kJ_mol": se[3] * F.R / 1e3}
    out |= {"j_log10_per_GPa": beta[4] / LN10 * 1e9, "j_se": se[4] / LN10 * 1e9}
    out |= {"h_as_V_cm3_mol": -beta[5] * F.R * 1e6, "h_as_V_se": se[5] * F.R * 1e6}
    # The activation volume at Holzapfel's 1400 C, V = -R (j T + h): j and h are nearly
    # collinear (one pair of high-pressure temperatures), their combination is not.
    g = np.zeros(len(beta))
    g[4], g[5] = -F.R * T_V * 1e6, -F.R * 1e6
    out |= {"V_cm3_mol": float(beta @ g), "V_se_cm3_mol": float(math.sqrt(g @ cov @ g))}
    if name != "mutch_x6":  # fitted anisotropy: log10 D[001]/D[axis]
        out |= {"log10_c_over_a": -beta[6] / LN10, "log10_c_over_a_se": se[6] / LN10}
        out |= {"log10_c_over_b": -beta[7] / LN10, "log10_c_over_b_se": se[7] / LN10}
    if name == "mutch_fT":
        out |= {"g_fT": beta[8], "g_fT_z": beta[8] / se[8]}
    return out


def fit_all(df: pd.DataFrame) -> dict[str, dict]:
    return {name: V.fit_preset(df, form) for name, form in FORMS.items()}


def table(fits: dict[str, dict], n: int) -> pd.DataFrame:
    rows = []
    for name, fit in fits.items():
        form = FORMS[name]
        row = {"form": name, "n_points": n, "n_params": len(form.params)}
        row |= {"tau_log10": fit["tau"] / LN10, "sd_within_log10": fit["sd"] / LN10}
        lo, hi = fit["tau_range"]
        row |= {"tau_lo95": lo / LN10, "tau_hi95": hi / LN10}
        if name.startswith("mutch"):
            row |= forster_units(name, fit["beta"], fit["cov"])
        else:
            b, se = fit["beta"], np.sqrt(np.diag(fit["cov"]))
            row |= {"fO2_exponent": b[2], "fO2_exponent_se": se[2]}
            row |= {"E_kJ_mol": b[1] / 1e3, "E_se_kJ_mol": se[1] / 1e3, "V_cm3_mol": 7.0}
            if "V_m3_mol" in form.params:
                row |= {"V_cm3_mol": b[4] * 1e6, "V_se_cm3_mol": se[4] * 1e6}
        for label, c in V.CONDITIONS.items():
            mu, s = F.predict(form, fit["beta"], fit["cov"], c)
            row |= {f"{label}_vs_dc07": mu - F._dc07_preset(c), f"{label}_se_law": s}
            row[f"{label}_se_new_lab"] = math.hypot(s, fit["tau"] / LN10)
        rows.append(row)
    for name in DFENS:
        law = DIFFUSIVITY_PRESETS[name]
        row = {"form": f"{name} (DFENS)", "fO2_exponent": law.fO2_exponent}
        row |= {"E_kJ_mol": law.activation_energy_J_mol[2] / 1e3}
        j = law.pressure_coefficient_per_Pa * LN10  # back to Mutch's j
        row["V_cm3_mol"] = (law.activation_volume_m3_mol - F.R * T_V * j) * 1e6
        for label, c in V.CONDITIONS.items():
            cc, x = _conditions(c)
            row[f"{label}_vs_dc07"] = math.log10(law.axis_diffusivities(cc, x)[2]) - F._dc07_preset(
                c
            )
            row[f"{label}_se_law"] = law.log10_sigma(cc, ALONG_001, x)
        rows.append(row)
    return pd.DataFrame(rows)


def _conditions(c: F.Vars) -> tuple[Conditions, float]:
    t_c = float(c.T[0]) - 273.15
    return Conditions(t_c, float(c.P[0]), 10 ** float(c.log_fo2_pa[0])), float(c.x_fo[0])


def _grid(t_c: np.ndarray, log_fo2: np.ndarray, x_fo: float, p_pa: np.ndarray) -> F.Vars:
    n = max(len(np.atleast_1d(a)) for a in (t_c, log_fo2, p_pa))
    full = [np.broadcast_to(np.asarray(a, float), n) for a in (log_fo2, x_fo, t_c + 273.15)]
    return F.Vars(*full, np.broadcast_to(np.asarray(p_pa, float), n), np.zeros(n), np.zeros(n))


def _curve(form: F.Form, fit: dict, v: F.Vars) -> tuple[np.ndarray, np.ndarray]:
    A = form.design(v)
    mu = (form.offset(v) + A @ fit["beta"]) / LN10
    sd = np.sqrt(np.einsum("ij,jk,ik->i", A, fit["cov"], A)) / LN10
    return mu, sd


def _law_curve(name: str, t_c, log_fo2, x_fo: float, p_pa) -> np.ndarray:
    law = DIFFUSIVITY_PRESETS[name]
    n = max(len(np.atleast_1d(a)) for a in (t_c, log_fo2, p_pa))
    t_c, log_fo2, p_pa = (np.broadcast_to(np.asarray(a, float), n) for a in (t_c, log_fo2, p_pa))
    return np.array(
        [
            math.log10(law.axis_diffusivities(Conditions(t, p, 10**f), x_fo)[2])
            for t, f, p in zip(t_c, log_fo2, p_pa, strict=True)
        ]
    )


def panel_a(ax, fits: dict) -> None:
    ks = V.CONDITIONS["RR1200"]
    log_fo2, x_fo, p_pa = float(ks.log_fo2_pa[0]), float(ks.x_fo[0]), float(ks.P[0])
    t = np.linspace(800, 1450, 66)
    dc = np.array([F.preset_log10_d("dohmen_chakraborty2007", tt, p_pa, log_fo2, x_fo) for tt in t])
    for name, fit in fits.items():
        colour, ls = STYLE[name]
        mu, sd = _curve(FORMS[name], fit, _grid(t, log_fo2, x_fo, p_pa))
        ax.fill_between(t, mu - sd - dc, mu + sd - dc, color=colour, alpha=0.15, lw=0)
        ax.plot(t, mu - dc, color=colour, ls=ls, lw=1.6, label=f"{name} refit, 68%")
    for name in DFENS:
        colour, ls = STYLE[name]
        ax.plot(
            t,
            _law_curve(name, t, log_fo2, x_fo, p_pa) - dc,
            color=colour,
            ls=ls,
            lw=1.3,
            label=f"{name} (DFENS)",
        )
    ax.axhline(0, color="0.2", lw=0.8, ls=":")
    ax.axvline(1200, color="0.6", lw=0.6)
    ax.set_xlabel("T (°C)")
    ax.set_ylabel("log10 D[001] minus DC07, at RR1200 fO2, Fo85, 42 MPa")
    ax.set_title("(a) against T at the RR1200 conditions", loc="left")


def panel_b(ax, fits: dict, df: pd.DataFrame) -> None:
    t_c, x_fo = PANEL_B
    p = np.linspace(1e5, 13e9, 60)
    nno = np.array([math.log10(engine.fo2buffer(t_c + 273.15, pp, 0, "NNO")) for pp in p])
    for name, fit in fits.items():
        colour, ls = STYLE[name]
        mu, sd = _curve(FORMS[name], fit, _grid(np.array(t_c), nno, x_fo, p))
        ax.fill_between(p / 1e9, mu - sd, mu + sd, color=colour, alpha=0.15, lw=0)
        ax.plot(p / 1e9, mu, color=colour, ls=ls, lw=1.6)
    for name in DFENS:
        colour, ls = STYLE[name]
        ax.plot(p / 1e9, _law_curve(name, np.array(t_c), nno, x_fo, p), color=colour, ls=ls, lw=1.3)
    h = df[df["source"] == "holzapfel2007"]
    au = V._is_au(h)
    moved = F.moved(h, V.FORM, fits["dc07_x6_V (the preset)"]["beta"], T_K=t_c + 273.15, X_Fo=x_fo)
    for mask, filled, label in ((~au, False, "Ni-NiO capsules"), (au, True, "Au capsules")):
        off = (h["T_K"] - 273.15 - t_c).abs().to_numpy() > 1
        for m, marker, extra in ((mask & ~off, "D", ""), (mask & off, "s", ", other T, moved")):
            if m.any():
                ax.plot(
                    h["P_Pa"][m] / 1e9,
                    moved[m],
                    marker,
                    color="#D55E00",
                    mfc="#D55E00" if filled else "none",
                    ms=5,
                    ls="",
                    label=f"Holzapfel 2007, {label}{extra}",
                )
    ax.set_xlabel("P (GPa)")
    ax.set_ylabel(f"log10 D[001] (m²/s) at {t_c:.0f} °C, Fo{x_fo * 100:.0f}, Ni-NiO")
    ax.set_title("(b) against P (points at other T moved by the preset)", loc="left")
    ax.legend(loc="lower left", fontsize=7, frameon=False)


def panel_c(ax_row, tab: pd.DataFrame) -> None:
    y = np.arange(len(tab))
    for ax, cond in zip(ax_row, ("RR1200", "RR1102", "Paper"), strict=True):
        for yi, r in enumerate(tab.itertuples()):
            key = r.form.split(" (")[0] if "DFENS" in r.form else r.form
            colour = STYLE[key][0]
            ax.errorbar(
                getattr(r, f"{cond}_vs_dc07"),
                yi,
                xerr=getattr(r, f"{cond}_se_law"),
                fmt="o",
                color=colour,
                ms=5,
                elinewidth=2,
                capsize=0,
            )
        ax.axvline(0, color="0.2", lw=0.8, ls=":")
        ax.set_title(f"(c) {cond}", loc="left")
        ax.set_xlabel("D[001] minus DC07 (log)")
    ax_row[0].set_yticks(y, tab["form"])  # shared: one formatter for the row
    ax_row[0].invert_yaxis()


def figure(fits: dict, tab: pd.DataFrame, df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(12, 9), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1.25, 1])
    a, b = top.subplots(1, 2)
    panel_a(a, fits)
    panel_b(b, fits, df)
    handles, labels = a.get_legend_handles_labels()
    top.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=7, frameon=False)
    panel_c(bottom.subplots(1, 3, sharey=True), tab)
    t = tab.set_index("form")
    k = t["RR1200_vs_dc07"]
    mx6 = t.loc["mutch_x6"]
    dv = t.loc["dc07_x6_V (the preset)"]
    fig.suptitle(
        f"The preset fits V ({dv['V_cm3_mol']:.1f} ± {dv['V_se_cm3_mol']:.1f} cm³/mol): RR1200 "
        f"{k['dc07_x6_V (the preset)']:+.2f}, against {k['dc07_x6, V fixed at 7']:+.2f} with V "
        f"fixed at 7; the Mutch forms add nothing to that ({k['mutch_x6']:+.2f} at 6x, "
        f"{k['mutch_aniso']:+.2f} with fitted anisotropy).\nThe 6x Mutch refit: fO2 exponent "
        f"{mx6['fO2_exponent']:.3f} ± {mx6['fO2_exponent_se']:.3f}, E {mx6['E_kJ_mol']:.0f} ± "
        f"{mx6['E_se_kJ_mol']:.0f} kJ/mol, V at 1400 °C {mx6['V_cm3_mol']:.1f} ± "
        f"{mx6['V_se_cm3_mol']:.1f} cm³/mol; with fO2 x 1/T, {k['mutch_fT']:+.2f}. "
        "Bars and bands: the law's 1 sigma",
        fontsize=10,
    )
    fig.savefig(OUT / "p4_mutch_forms.png", dpi=200)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ver, df = V.verified(), V.preset_data()
    print(
        "mutch_x6 identifiable on the verified data alone:",
        F._identifiable(F.FORMS["mutch_x6"], ver),
        "; with the Au runs:",
        F._identifiable(F.FORMS["mutch_x6"], df),
    )
    fits = fit_all(df)
    check_slopes(df, fits["dc07_x6_V (the preset)"]["beta"])
    tab = table(fits, len(df))
    tab.to_csv(HERE / "results" / "verified" / "mutch_forms.csv", index=False)
    figure(fits, tab, df)
    pd.set_option("display.width", 240)
    cols = [
        "form",
        "tau_log10",
        "tau_lo95",
        "tau_hi95",
        "sd_within_log10",
        "fO2_exponent",
        "fO2_exponent_se",
        "E_kJ_mol",
        "E_se_kJ_mol",
        "V_cm3_mol",
        "V_se_cm3_mol",
        "j_log10_per_GPa",
        "j_se",
        "RR1200_vs_dc07",
        "RR1200_se_law",
        "RR1200_se_new_lab",
        "RR1102_vs_dc07",
        "Paper_vs_dc07",
    ]
    print(tab[[c for c in cols if c in tab]].round(3).to_string(index=False))
    extra = [
        "log10_c_over_a",
        "log10_c_over_a_se",
        "log10_c_over_b",
        "log10_c_over_b_se",
        "g_fT",
        "g_fT_z",
        "fo_coefficient",
        "fo_coefficient_se",
    ]
    print(tab[["form", *[c for c in extra if c in tab]]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
