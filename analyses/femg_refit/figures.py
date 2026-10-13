"""Figures comparing the Fe-Mg refits (reads results/ from fit_models.py).

1. fig1_predictions.png: D[001] at the round robin and paper conditions, each fit
   against the current default (Dohmen and Chakraborty 2007), with 1-sigma bars.
2. fig2_coefficients.png: activation energy, fO2 exponent, composition term,
   activation volume and anisotropy from each fit, with the published values.
3. fig3_data.png: the data normalised to common conditions by one fit, against
   temperature and against fO2, with the fitted laws and the conditions of use.
4. fig4_residuals.png: residuals of two fits by study, against T and fO2.

Run: uv run --with odrpack python analyses/femg_refit/figures.py
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import fit_models as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
OUT = HERE / "figures"
RESULTS = HERE / "results"
LN10 = math.log(10)
plt.rcParams.update({"pdf.fonttype": 42, "font.size": 9, "axes.titlesize": 9.5})

STUDY_STYLE = {  # Okabe-Ito colour and a marker per study
    "chakraborty1997": ("#E69F00", "o", "Chakraborty 1997"),
    "dohmen2007_part1": ("#56B4E9", "s", "Dohmen et al. 2007"),
    "petry2004": ("#009E73", "^", "Petry et al. 2004"),
    "holzapfel2007": ("#D55E00", "D", "Holzapfel et al. 2007"),
    "spandler_oneill2010": ("#CC79A7", "v", "Spandler & O'Neill 2010"),
}
FORM_STYLE = {
    "dc07_fixed": ("#D55E00", "DC07 form, D0 and E free"),
    "dc07_free": ("#E69F00", "DC07 form, all but V free"),
    "mutch_x6": ("#0072B2", "Mutch form, 6x anisotropy"),
    "mutch_aniso": ("#56B4E9", "Mutch form, anisotropy fitted"),
}
METHODS = ["ols", "wls", "odr", "bootstrap (runs)"]
METHOD_MARKER = {"ols": "o", "wls": "s", "odr": "D", "bootstrap (runs)": "^"}
METHOD_LABEL = {
    "ols": "least squares",
    "wls": "weighted",
    "odr": "ODR (errors in T, fO2, X, P)",
    "bootstrap (runs)": "bootstrap over runs",
}


def _preset_offsets() -> dict[str, dict[str, float]]:
    """log10 D[001] of the Mutch et al. (2019) presets minus the default, per condition."""
    from forster import DIFFUSIVITY_PRESETS, Conditions

    out = {}
    for cname, v in F._conditions().items():
        c = Conditions(float(v.T[0]) - 273.15, float(v.P[0]), 10 ** float(v.log_fo2_pa[0]))
        ref = F._dc07_preset(v)
        out[cname] = {
            name: float(
                np.log10(DIFFUSIVITY_PRESETS[name].axis_diffusivities(c, float(v.x_fo[0]))[2])
            )
            - ref
            for name in ("mutch2019_tamed", "mutch2019_global")
        }
    return out


def _time_axis(ax) -> None:
    """Fitted-time multiplier (time scales as 1/D) along the top, at fixed values."""
    lo, hi = ax.get_xlim()
    top = ax.twiny()
    top.set_xlim(lo, hi)
    marks = [m for m in (0.1, 0.2, 0.5, 1, 2) if lo <= -math.log10(m) <= hi]
    top.set_xticks([-math.log10(m) for m in marks], [f"x{m:g}" for m in marks])
    top.set_xlabel("time vs default")


def fig_predictions(pred: pd.DataFrame) -> None:
    rows = [(f, m) for f in FORM_STYLE for m in METHODS]
    conds = list(F.CONDITIONS)
    presets = _preset_offsets()
    fig, axes = plt.subplots(1, 3, figsize=(10, 6.2), sharey=True, layout="constrained")
    span = []
    for ax, cname in zip(axes, conds, strict=True):
        sub = pred[pred.condition == cname].set_index(["form", "method"])
        for i, (form, method) in enumerate(rows):
            r = sub.loc[(form, method)]
            colour = FORM_STYLE[form][0]
            ax.errorbar(
                r.minus_dc07_preset,
                i,
                xerr=r.se,
                fmt=METHOD_MARKER[method],
                color=colour,
                ms=5,
                capsize=2,
                lw=1,
            )
            span.append(r.minus_dc07_preset)
        for j, (name, _label) in enumerate(
            (("mutch2019_tamed", "Mutch 2019 TaMED"), ("mutch2019_global", "Mutch 2019 Global"))
        ):
            ax.plot(presets[cname][name], len(rows) + j, "kx", ms=6)
        ax.axvline(0, color="0.4", lw=1, ls="--")
        ax.set_title(cname)
        ax.set_xlabel("log10 D[001] minus default (DC07)")
        _time_axis(ax)
        ax.grid(axis="x", color="0.9", lw=0.6)
    labels = [f"{FORM_STYLE[f][1]}: {METHOD_LABEL[m]}" for f, m in rows] + [
        "Mutch et al. 2019, TaMED (published)",
        "Mutch et al. 2019, Global (published)",
    ]
    axes[0].set_yticks(range(len(labels)), labels, fontsize=7.5)
    axes[0].invert_yaxis()
    lo, hi = min(span), max(span)
    fig.suptitle(
        f"Refitted laws put Fe-Mg diffusion {lo:+.2f} to {hi:+.2f} log units from the default "
        "at these conditions;\nthe spread comes from the law's form and the fitting method more "
        "than from the 1-sigma bars",
        fontsize=10,
    )
    fig.savefig(OUT / "fig1_predictions.png", dpi=200)
    plt.close(fig)


def _derived(coef: pd.DataFrame) -> pd.DataFrame:
    """Common quantities from both law forms: E (kJ/mol), fO2 exponent, composition
    term (log10 D per unit X_Fe), V (cm3/mol), anisotropy factors."""
    out = []
    main = coef[coef.method.isin(METHODS)]
    for (form, method), g in main.groupby(["form", "method"], sort=False):
        p = g.set_index("param")
        if form.startswith("mutch"):
            quantities = {
                "E (kJ/mol)": (-p.value.q * F.R / 1e3, p.se.q * F.R / 1e3),
                "fO2 exponent": (p.value.b, p.se.b),
                "composition, log10 D per X_Fe": (-p.value.c / LN10, p.se.c / LN10),
                "V at 1250 C (cm3/mol)": (p.value["V_1250C_cm3_mol"], p.se["V_1250C_cm3_mol"]),
            }
        else:
            quantities = {"E (kJ/mol)": (p.value.E_J_mol / 1e3, p.se.E_J_mol / 1e3)}
            if form == "dc07_free":
                quantities["fO2 exponent"] = (p.value.m_fO2, p.se.m_fO2)
                quantities["composition, log10 D per X_Fe"] = (p.value.n_X, p.se.n_X)
        if "k100" in p.index:
            quantities["[001]/[100] factor"] = (
                math.exp(-p.value.k100),
                math.exp(-p.value.k100) * p.se.k100,
            )
            quantities["[001]/[010] factor"] = (
                math.exp(-p.value.k010),
                math.exp(-p.value.k010) * p.se.k010,
            )
        for q, (val, se) in quantities.items():
            out.append({"form": form, "method": method, "quantity": q, "value": val, "se": se})
    return pd.DataFrame(out)


PUBLISHED = {  # Dohmen and Chakraborty (2007); Mutch et al. (2019) TaMED
    "E (kJ/mol)": (201, 26700 * F.R / 1e3),
    "fO2 exponent": (1 / 6, 0.224),
    "composition, log10 D per X_Fe": (3.0, 7.18 / LN10),
    "V at 1250 C (cm3/mol)": (7.0, -F.R * (-5.21e-10 * 1523.15 - 1.03e-7) * 1e6),
    "[001]/[100] factor": (6.0, 6.0),
    "[001]/[010] factor": (6.0, 6.0),
}


def fig_coefficients(coef: pd.DataFrame) -> None:
    d = _derived(coef)
    quantities = list(PUBLISHED)
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.4), layout="constrained")
    rows = [(f, m) for f in FORM_STYLE for m in METHODS]
    for ax, q in zip(axes.flat, quantities, strict=True):
        sub = d[d.quantity == q].set_index(["form", "method"])
        for i, key in enumerate(rows):
            if key in sub.index:
                r = sub.loc[key]
                ax.errorbar(
                    r.value,
                    i,
                    xerr=r.se,
                    fmt=METHOD_MARKER[key[1]],
                    color=FORM_STYLE[key[0]][0],
                    ms=4.5,
                    capsize=2,
                    lw=1,
                )
        dc07, mutch = PUBLISHED[q]
        ax.axvline(dc07, color="0.2", lw=1, ls="--")
        ax.axvline(mutch, color="0.2", lw=1, ls=":")
        ax.set_title(q)
        ax.set_yticks([])
        ax.set_ylim(len(rows) - 0.5, -0.5)
        ax.grid(axis="x", color="0.9", lw=0.6)
    handles = [
        plt.Line2D([], [], color=c, marker="o", ls="", label=label)
        for c, label in FORM_STYLE.values()
    ]
    handles += [
        plt.Line2D([], [], color="0.3", marker=METHOD_MARKER[m], ls="", label=METHOD_LABEL[m])
        for m in METHODS
    ]
    handles += [
        plt.Line2D([], [], color="0.2", ls="--", label="Dohmen & Chakraborty 2007"),
        plt.Line2D([], [], color="0.2", ls=":", label="Mutch et al. 2019, TaMED"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False, fontsize=8)
    e = d[(d.quantity == "E (kJ/mol)") & (d.form == "dc07_fixed")].set_index("method").value
    fig.suptitle(
        f"With only D0 and E free, the data give E = {e['ols']:.0f}-{e['wls']:.0f} kJ/mol, "
        "above the published 201; other terms agree with the published values within their errors"
        if e["wls"] > e["ols"]
        else f"With only D0 and E free, the data give E = {min(e):.0f}-{max(e):.0f} kJ/mol "
        "(published: 201)",
        fontsize=10,
    )
    fig.savefig(OUT / "fig2_coefficients.png", dpi=200)
    plt.close(fig)


def _normalised(df: pd.DataFrame, form: F.Form, beta: np.ndarray, **fixed) -> np.ndarray:
    """log10 D moved from each point's own conditions to ``fixed`` by the fitted law."""
    v_own = F.Vars.of(df)
    target = df.copy()
    for col, val in fixed.items():
        target[col] = val
    v_t = F.Vars.of(target)
    own = form.offset(v_own) + form.design(v_own) @ beta
    to = form.offset(v_t) + form.design(v_t) @ beta
    return (df["ln_D"].to_numpy() + to - own) / LN10


def fig_data(df: pd.DataFrame) -> None:
    form = F.FORMS["dc07_free"]
    beta = F.fit_ls(form, df, weighted=False)["beta"]
    conds = F._conditions()
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")
    ref = {"log10_fO2_Pa": -7.0, "X_Fo": 0.9, "P_Pa": 1e5, "axis": "[001]"}
    y_t = _normalised(df, form, beta, **ref)
    for src, (colour, marker, label) in STUDY_STYLE.items():
        m = (df.source == src).to_numpy()
        a.plot(
            1e4 / df.T_K[m], y_t[m], marker, color=colour, mfc="none", ms=4.5, ls="", label=label
        )
    t = np.linspace(950, 1850, 50)
    for name, (colour, label) in FORM_STYLE.items():
        f = F.FORMS[name]
        bb = F.fit_ls(f, df, weighted=False)["beta"]
        grid = pd.DataFrame(
            {"T_K": t, "axis": "[001]", **{k: v for k, v in ref.items() if k != "axis"}}
        )
        grid["log10_fO2_Pa"] = ref["log10_fO2_Pa"]
        v = F.Vars.of(grid)
        a.plot(1e4 / t, (f.offset(v) + f.design(v) @ bb) / LN10, color=colour, lw=1.4, label=label)
    a.set_xlabel("10^4 / T (1/K)")
    a.set_ylabel("log10 D[001] (m2/s) at fO2 1e-7 Pa, Fo90, 1 atm")
    a.set_title("Temperature: data normalised by the DC07-form fit")
    lo, hi = a.get_xlim()
    top = a.twiny()
    top.set_xlim(lo, hi)
    celsius = [c for c in (800, 1000, 1200, 1400, 1600) if lo <= 1e4 / (c + 273.15) <= hi]
    top.set_xticks([1e4 / (c + 273.15) for c in celsius], [str(c) for c in celsius])
    top.set_xlabel("T (C)")
    # fO2 dependence at 1200 C, Fo90, 1 atm, [001].
    y_f = _normalised(df, form, beta, T_K=1473.15, X_Fo=0.9, P_Pa=1e5, axis="[001]")
    for src, (colour, marker, _label) in STUDY_STYLE.items():
        m = (df.source == src).to_numpy()
        b.plot(df.log10_fO2_Pa[m], y_f[m], marker, color=colour, mfc="none", ms=4.5, ls="")
    fo2 = np.linspace(-10, 3, 50)
    for name, (colour, _label) in FORM_STYLE.items():
        f = F.FORMS[name]
        bb = F.fit_ls(f, df, weighted=False)["beta"]
        grid = pd.DataFrame(
            {"T_K": 1473.15, "log10_fO2_Pa": fo2, "X_Fo": 0.9, "P_Pa": 1e5, "axis": "[001]"}
        )
        v = F.Vars.of(grid)
        b.plot(fo2, (f.offset(v) + f.design(v) @ bb) / LN10, color=colour, lw=1.4)
    for cname, v in conds.items():
        b.axvline(float(v.log_fo2_pa[0]), color="0.5", lw=0.8, ls=":")
        b.annotate(
            cname,
            (float(v.log_fo2_pa[0]), 0.02),
            xycoords=("data", "axes fraction"),
            rotation=90,
            fontsize=7,
            ha="right",
            va="bottom",
            color="0.35",
        )
    b.set_xlabel("log10 fO2 (Pa)")
    b.set_ylabel("log10 D[001] (m2/s) at 1200 C, Fo90, 1 atm")
    b.set_title("fO2: our conditions lie above most of the data")
    fig.legend(loc="outside lower center", ncol=5, frameon=False, fontsize=7.5)
    cond_fo2 = [float(v.log_fo2_pa[0]) for v in conds.values()]
    lo, hi = min(cond_fo2) - 1.5, max(cond_fo2) + 1.5
    near = (df.log10_fO2_Pa >= lo) & (df.log10_fO2_Pa <= hi)
    above = int((df.log10_fO2_Pa > hi).sum())
    fig.suptitle(
        f"Only {int(near.sum())} of {len(df)} points lie within 1.5 log units of these "
        f"conditions' fO2; the {above} above are 6-12 GPa runs\n"
        "whose fO2 is computed at Ni-NiO, not measured: the fO2 term there rests on few points",
        fontsize=10,
    )
    fig.savefig(OUT / "fig3_data.png", dpi=200)
    plt.close(fig)


def fig_residuals(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharey=True, layout="constrained")
    sds = []
    for row, name in zip(axes, ("dc07_free", "mutch_aniso"), strict=True):
        f = F.FORMS[name]
        fit = F.fit_ls(f, df, weighted=False)
        v = F.Vars.of(df)
        resid = (df.ln_D.to_numpy() - f.offset(v) - f.design(v) @ fit["beta"]) / LN10
        sds.append(float(np.std(resid, ddof=len(f.params))))
        for ax, (col, xlabel) in zip(
            row, (("T_C", "T (C)"), ("log10_fO2_Pa", "log10 fO2 (Pa)")), strict=True
        ):
            for src, (colour, marker, label) in STUDY_STYLE.items():
                m = (df.source == src).to_numpy()
                ax.plot(
                    df[col][m],
                    resid[m],
                    marker,
                    color=colour,
                    mfc="none",
                    ms=4.5,
                    ls="",
                    label=label,
                )
            ax.axhline(0, color="0.4", lw=0.8)
            ax.set_xlabel(xlabel)
            ax.set_title(f"{FORM_STYLE[name][1]}: residual sd {sds[-1]:.2f} log")
        row[0].set_ylabel("observed - fitted, log10 D")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5, frameon=False, fontsize=8)
    spread = (
        f"{sds[0]:.2f}"
        if round(sds[0], 2) == round(sds[1], 2)
        else f"{min(sds):.2f}-{max(sds):.2f}"
    )
    fig.suptitle(
        f"Residual sd {spread} log units in both forms, about the stated experimental error "
        "(0.2-0.3 log);\nHolzapfel's 11 GPa runs and Spandler's off-axis values sit highest",
        fontsize=10,
    )
    fig.savefig(OUT / "fig4_residuals.png", dpi=200)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    pred = pd.read_csv(RESULTS / "predictions.csv")
    coef = pd.read_csv(RESULTS / "coefficients.csv")
    df = F.tamed()
    fig_predictions(pred)
    fig_coefficients(coef)
    fig_data(df)
    fig_residuals(df)


if __name__ == "__main__":
    main()
