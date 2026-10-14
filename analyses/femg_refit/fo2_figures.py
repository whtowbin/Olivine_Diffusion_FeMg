"""How the fits treat fO2, and why the studies disagree.

5. fig5_fo2.png: (a) the data at 1200 C, Fo90, 1 atm, [001] against fO2 with every
   law; (b) the fO2 exponent each study's own data give, against the fitted and
   published ones.
6. fig6_studies.png: (a) where each study sits in temperature and fO2 (pressure and
   composition noted); (b) each study's offset from the joint fit, per law form.

Run: uv run --with odrpack python analyses/femg_refit/fo2_figures.py
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
from fit_ols import load

HERE = Path(__file__).parent
OUT = HERE / "figures"
LN10, R = math.log(10), 8.3145
plt.rcParams.update({"pdf.fonttype": 42, "font.size": 9, "axes.titlesize": 9.5})

STUDY_STYLE = {
    "chakraborty1997": ("#E69F00", "o", "Chakraborty 1997"),
    "dohmen2007_part1": ("#56B4E9", "s", "Dohmen et al. 2007"),
    "petry2004": ("#009E73", "^", "Petry et al. 2004"),
    "holzapfel2007": ("#D55E00", "D", "Holzapfel et al. 2007"),
    "spandler_oneill2010": ("#CC79A7", "v", "Spandler & O'Neill 2010"),
}
STUDY_NOTE = {
    "chakraborty1997": "EPMA, Fo25-97 couples, one fO2 (1e-7 Pa)",
    "dohmen2007_part1": "thin films, RBS (nm), Fo86, 1e-12 to 1e-5 Pa",
    "petry2004": "EPMA, Fo90-100 couples, 1e-10 to 1e-4 Pa",
    "holzapfel2007": "6-12 GPa, fO2 not measured (Ni-NiO assumed)",
    "spandler_oneill2010": "olivine against basaltic melt, 1300 C",
}
FORM_STYLE = {
    "dc07_fixed": ("#D55E00", "-", "DC07 form, D0 and E free"),
    "dc07_free": ("#E69F00", "-", "DC07 form, all but V free"),
    "mutch_x6": ("#0072B2", "-", "Mutch form, 6x"),
    "mutch_fT": ("#56B4E9", "-", "Mutch form + fO2 x 1/T"),
}
# Where the conditions' labels go (points), clear of the data, with a leader line.
LABEL_OFFSET = {"KS20": (-45, 28), "SH63": (-70, -10), "Paper": (30, 26)}
CONDITIONS_FO2 = {"SH63": None, "KS20": math.log10(0.00138), "Paper": math.log10(8.3e-3)}


def _move(df, form, beta, cov, **fixed):
    """log10 D moved to ``fixed`` conditions by a fit, and the 1-sigma of that move."""
    v_own = F.Vars.of(df)
    v_t = F.Vars.of(df.assign(**fixed))
    shift = (form.offset(v_t) + form.design(v_t) @ beta) - (
        form.offset(v_own) + form.design(v_own) @ beta
    )
    dg = form.design(v_t) - form.design(v_own)
    sd = np.sqrt(np.clip(np.einsum("ij,jk,ik->i", dg, cov, dg), 0, None)) / LN10
    return df["log10_D_m2s"].to_numpy() + shift / LN10, sd


def _series_slope(sub: pd.DataFrame) -> tuple[float, float]:
    """Slope of log D against log fO2 (with an intercept) and its standard error."""
    X = np.column_stack([np.ones(len(sub)), sub["log10_fO2_Pa"]])
    y = sub["log10_D_m2s"].to_numpy()
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    cov = (r @ r / (len(y) - 2)) * np.linalg.inv(X.T @ X)
    return float(b[1]), float(math.sqrt(cov[1, 1]))


def _within_study(sub: pd.DataFrame, axes: bool) -> tuple[float, float]:
    """fO2 exponent from one study, with 1/T (and axis offsets) removed."""
    cols = [np.ones(len(sub)), sub["log10_fO2_Pa"], 1 / sub["T_K"]]
    if axes:
        cols += [(sub["axis"] == "[100]").astype(float), (sub["axis"] == "[010]").astype(float)]
    X = np.column_stack(cols)
    scale = np.abs(X).max(axis=0)
    y = sub["log10_D_m2s"].to_numpy()
    b, *_ = np.linalg.lstsq(X / scale, y, rcond=None)
    r = y - (X / scale) @ b
    cov = (r @ r / (len(y) - X.shape[1])) * np.linalg.inv((X / scale).T @ (X / scale))
    return float(b[1] / scale[1]), float(math.sqrt(cov[1, 1]) / scale[1])


def fig_fo2(df: pd.DataFrame) -> None:
    every = load()  # including fO2 <= 1e-10 Pa (PED), shown faded
    every = every.assign(ln_D=every["log10_D_m2s"] * LN10)
    every.loc[every["P_Pa"] < 2e5, "P_Pa"] = 1e5
    norm_form = F.FORMS["dc07_free"]
    norm = F.fit_ls(norm_form, df, weighted=False)
    ref = {"T_K": 1473.15, "X_Fo": 0.9, "P_Pa": 1e5, "axis": "[001]"}
    y_n, corr = _move(every, norm_form, norm["beta"], norm["cov"], **ref)
    meas = np.full(len(every), 0.2)
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(10.5, 4.8), layout="constrained", width_ratios=[1.35, 1]
    )
    ks20, paper = CONDITIONS_FO2["KS20"], CONDITIONS_FO2["Paper"]
    a.axvspan(min(ks20, paper) - 0.9, max(ks20, paper), color="0.92", lw=0, zorder=0)
    a.text(
        ks20 - 0.4,
        0.97,
        "round robin and paper fO2",
        transform=a.get_xaxis_transform(),
        fontsize=7,
        color="0.35",
        ha="center",
        va="top",
        rotation=90,
    )
    for src, (colour, marker, label) in STUDY_STYLE.items():
        for tamed, alpha in ((True, 1.0), (False, 0.3)):
            m = ((every["source"] == src) & ((every["log10_fO2_Pa"] > -10) == tamed)).to_numpy()
            if not m.any():
                continue
            x = every["log10_fO2_Pa"][m]
            a.errorbar(
                x,
                y_n[m],
                yerr=np.sqrt(meas[m] ** 2 + corr[m] ** 2),
                fmt="none",
                ecolor=colour,
                alpha=0.2 * alpha,
                elinewidth=2,
            )
            a.plot(
                x,
                y_n[m],
                marker,
                color=colour,
                mfc="none",
                ms=4.5,
                alpha=alpha,
                ls="",
                label=label if tamed else None,
            )
    a.axvline(-10, color="0.5", lw=0.8, ls="--")
    fo2 = np.linspace(-10, 4, 60)
    grid = pd.DataFrame({"log10_fO2_Pa": fo2, **ref})
    for name, (colour, ls, label) in FORM_STYLE.items():
        f = F.FORMS[name]
        beta = F.fit_ls(f, df, weighted=False)["beta"]
        v = F.Vars.of(grid)
        a.plot(
            fo2, (f.offset(v) + f.design(v) @ beta) / LN10, color=colour, ls=ls, lw=1.5, label=label
        )
    f = F.FORMS["mutch_x6"]
    beta = F.fit_odr(f, df)["beta"]
    v = F.Vars.of(grid)
    a.plot(
        fo2,
        (f.offset(v) + f.design(v) @ beta) / LN10,
        color="#0072B2",
        ls="-.",
        lw=1.3,
        label="Mutch form, 6x, ODR",
    )
    from forster import DIFFUSIVITY_PRESETS, Conditions

    for name, ls, label in (
        ("dohmen_chakraborty2007", "--", "DC07 (published)"),
        ("mutch2019_tamed", ":", "Mutch 2019 TaMED (published)"),
    ):
        d = DIFFUSIVITY_PRESETS[name]
        a.plot(
            fo2,
            [np.log10(d.axis_diffusivities(Conditions(1200, 1e5, 10**g), 0.9)[2]) for g in fo2],
            color="0.15",
            ls=ls,
            lw=1.1,
            label=label,
        )
    a.set_xlabel("log10 fO2 (Pa)")
    a.set_ylabel("log10 D[001] (m2/s) at 1200 C, Fo90, 1 atm")
    a.set_title("(a) data at 1200 C (moved by the DC07-form fit); dashed grey: 1e-10 Pa")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), fontsize=6.5, frameon=False, ncol=3)

    # (b) fO2 exponents.
    rows = []
    d1 = every[(every["source"] == "dohmen2007_part1") & (every["log10_fO2_Pa"] > -10)]
    s = d1[(d1["T_C"] == 1080) & (d1["axis"] == "[001]")]
    rows.append(("Dohmen: 1080 C [001] series", *_series_slope(s), "#56B4E9"))
    s = d1[(d1["T_C"] == 1050) & (d1["axis"] == "[100]")]
    rows.append(("Dohmen: 1050 C [100] series", *_series_slope(s), "#56B4E9"))
    rows.append(("Dohmen: all TaMED, T and axis removed", *_within_study(d1, axes=True), "#56B4E9"))
    pe = every[every["source"] == "petry2004"]
    rows.append(("Petry: T removed", *_within_study(pe, axes=False), "#009E73"))
    for name, method, label in (
        ("dc07_free", "ols", "joint fit: DC07 form, OLS"),
        ("dc07_free", "wls", "joint fit: DC07 form, weighted"),
        ("dc07_free", "odr", "joint fit: DC07 form, ODR"),
        ("mutch_x6", "ols", "joint fit: Mutch form, OLS"),
        ("mutch_x6", "odr", "joint fit: Mutch form, ODR"),
    ):
        f = F.FORMS[name]
        fit = F.fit_odr(f, df) if method == "odr" else F.fit_ls(f, df, weighted=(method == "wls"))
        i = f.params.index("m_fO2" if "m_fO2" in f.params else "b")
        rows.append((label, float(fit["beta"][i]), float(math.sqrt(fit["cov"][i, i])), "0.2"))
    f = F.FORMS["mutch_fT"]
    fit = F.fit_ls(f, df, weighted=False)
    ib, ig = f.params.index("b"), f.params.index("g_fT")
    for t_c in (900, 1200, 1400):
        g = np.zeros(len(f.params))
        g[ib], g[ig] = 1.0, 1 / (t_c + 273.15)
        rows.append(
            (
                f"joint fit: Mutch + fO2 x 1/T, at {t_c} C",
                float(fit["beta"] @ g),
                float(math.sqrt(g @ fit["cov"] @ g)),
                "#56B4E9",
            )
        )
    for i, (_label, val, se, colour) in enumerate(rows):
        b.errorbar(val, i, xerr=se, fmt="o", color=colour, ms=4.5, capsize=2, lw=1)
    b.axvline(1 / 6, color="0.15", ls="--", lw=1)
    b.axvline(0.224, color="0.15", ls=":", lw=1)
    b.axvspan(1 / 7, 1 / 4, color="0.9", lw=0, zorder=0)
    b.set_yticks(range(len(rows)), [r[0] for r in rows], fontsize=7)
    b.invert_yaxis()
    b.set_xlabel(
        "fO2 exponent (d log D / d log fO2)\ngrey band: 1/4-1/7 (Dohmen et al. 2007)\n"
        "dashed: 1/6 (DC07); dotted: 0.224 (Mutch)",
        fontsize=8,
    )
    b.set_title("(b) fO2 exponents, per study and fitted")
    own = [r for r in rows if not r[0].startswith("joint")]
    joint = [r for r in rows if r[0].startswith("joint")]
    lsq = [r[1] for r in joint if "ODR" not in r[0] and "1/T" not in r[0]]
    odr = max(r[1] for r in joint if "ODR" in r[0])
    lo_own, hi_own = min(r[1] for r in own), max(r[1] for r in own)
    fig.suptitle(
        f"Studies that varied fO2 measure exponents {lo_own:.2f}-{hi_own:.2f} (Dohmen, mostly "
        "1050-1080 C, lower; Petry, to 1400 C, higher),\nas an exponent rising with T predicts. "
        f"Joint least-squares fits give {min(lsq):.2f}-{max(lsq):.2f}; ODR's {odr:.2f} is above "
        "every direct measurement",
        fontsize=10,
    )
    fig.savefig(OUT / "fig5_fo2.png", dpi=200)
    plt.close(fig)
    return rows


def _study_effects(df: pd.DataFrame, form: F.Form) -> pd.DataFrame:
    """Each study's offset (log D, centred) from the form, with its standard error."""
    v = F.Vars.of(df)
    studies = list(STUDY_STYLE)
    D = np.column_stack([(df["source"] == s).to_numpy(float) for s in studies])
    X = np.column_stack([form.design(v), D[:, 1:]])
    y = df["ln_D"].to_numpy() - form.offset(v)
    scale = np.abs(X).max(axis=0)
    b, *_ = np.linalg.lstsq(X / scale, y, rcond=None)
    r = y - (X / scale) @ b
    cov_s = (r @ r / (len(y) - X.shape[1])) * np.linalg.pinv((X / scale).T @ (X / scale))
    k = len(studies)
    eff = np.concatenate([[0.0], b[-(k - 1) :] / scale[-(k - 1) :]])
    cov_e = np.zeros((k, k))
    cov_e[1:, 1:] = cov_s[-(k - 1) :, -(k - 1) :] / np.outer(scale[-(k - 1) :], scale[-(k - 1) :])
    C = np.eye(k) - 1 / k
    return pd.DataFrame(
        {"study": studies, "effect": C @ eff / LN10, "se": np.sqrt(np.diag(C @ cov_e @ C.T)) / LN10}
    )


def fig_studies(df: pd.DataFrame) -> None:
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(10.5, 4.6), layout="constrained", width_ratios=[1, 1.15]
    )
    for src, (colour, marker, label) in STUDY_STYLE.items():
        m = (df["source"] == src).to_numpy()
        a.plot(
            df["T_C"][m],
            df["log10_fO2_Pa"][m],
            marker,
            color=colour,
            mfc="none",
            ms=5,
            ls="",
            label=f"{label}: {STUDY_NOTE[src]}",
        )
    from forster import read_runs  # noqa: F401  (conditions come from fit_models)

    for name, v in F._conditions().items():
        a.plot(float(v.T[0]) - 273.15, float(v.log_fo2_pa[0]), "*", color="k", ms=11)
        a.annotate(
            name,
            (float(v.T[0]) - 273.15, float(v.log_fo2_pa[0])),
            xytext=LABEL_OFFSET[name.split(",")[0]],
            textcoords="offset points",
            fontsize=7,
            va="center",
            ha="center",
            arrowprops={"arrowstyle": "-", "lw": 0.5, "color": "0.4"},
        )
    a.set_xlabel("T (C)")
    a.set_ylabel("log10 fO2 (Pa)")
    a.set_title("(a) where the TaMED data sit (stars: conditions of use)")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=6.5, frameon=False)
    forms = ("dc07_fixed", "dc07_free", "mutch_aniso")
    colours = {"dc07_fixed": "#D55E00", "dc07_free": "#E69F00", "mutch_aniso": "#0072B2"}
    labels = {
        "dc07_fixed": "DC07 form, D0 and E free",
        "dc07_free": "DC07 form, all but V free",
        "mutch_aniso": "Mutch form, anisotropy fitted",
    }
    effects = {}
    for j, name in enumerate(forms):
        e = _study_effects(df, F.FORMS[name])
        effects[name] = e
        for i, r in enumerate(e.itertuples()):
            b.errorbar(
                r.effect,
                i + (j - 1) * 0.22,
                xerr=r.se,
                fmt="o",
                color=colours[name],
                ms=4.5,
                capsize=2,
                lw=1,
                label=labels[name] if i == 0 else None,
            )
    b.axvline(0, color="0.4", lw=0.8)
    b.set_yticks(range(len(STUDY_STYLE)), [STUDY_STYLE[s][2] for s in STUDY_STYLE], fontsize=8)
    b.invert_yaxis()
    b.set_xlabel("study offset from the joint fit (log D, centred)")
    b.set_title("(b) each study's offset, with an offset per study fitted")
    b.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=7, frameon=False, ncol=2)
    sp = [effects[f].set_index("study").effect["spandler_oneill2010"] for f in forms]
    hz = [effects[f].set_index("study").effect["holzapfel2007"] for f in forms]
    fig.suptitle(
        "Each study sits in its own region of T, fO2 and P, so lab offsets and physical terms "
        f"trade off. Spandler's melt experiment sits {min(sp):+.2f} to {max(sp):+.2f} log high;\n"
        f"Holzapfel's offset swings from {max(hz):+.2f} to {min(hz):+.2f} with the law's form, "
        "because its runs alone are at 6-12 GPa and 1350-1550 C",
        fontsize=10,
    )
    fig.savefig(OUT / "fig6_studies.png", dpi=200)
    plt.close(fig)
    return effects


def main() -> None:
    df = F.tamed()
    rows = fig_fo2(df)
    for label, val, se, _ in rows:
        print(f"  {label:44s} {val:.3f} +- {se:.3f}")
    effects = fig_studies(df)
    for name, e in effects.items():
        print(name, ", ".join(f"{r.study[:8]} {r.effect:+.2f}+-{r.se:.2f}" for r in e.itertuples()))


if __name__ == "__main__":
    main()
