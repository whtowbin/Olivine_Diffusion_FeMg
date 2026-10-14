"""Arrhenius plots with confidence bands ("bowties") for every fit configuration.

One figure per law form x method in figures/bowties/<form>__<method>.png. Each has
two panels: (a) at the centre of the data (fO2 1e-7 Pa, Fo90, 1 atm, [001]) and
(b) at the round robin KS20 conditions (fO2 1.38e-3 Pa, Fo85, 42 MPa, [001]). The
data are moved to each panel's conditions by that fit itself; bands are the
16-84% and 2.5-97.5% quantiles of the predicted log D over parameter draws (the
bootstrap resamples, or draws from the fit's covariance), as in Henry's
odr_bootstrap. A paragraph under each figure says what is particular to that form
and method, with numbers from the fit.

Run: uv run --with odrpack python analyses/femg_refit/bowties.py
"""

from __future__ import annotations

import math
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import fit_models as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
OUT = HERE / "figures" / "bowties"
LN10 = math.log(10)
DRAWS = 4000
SEED = 20261014
plt.rcParams.update({"pdf.fonttype": 42, "font.size": 9, "axes.titlesize": 9.5})

STUDY_STYLE = {
    "chakraborty1997": ("#E69F00", "o", "Chakraborty 1997"),
    "dohmen2007_part1": ("#56B4E9", "s", "Dohmen et al. 2007"),
    "petry2004": ("#009E73", "^", "Petry et al. 2004"),
    "holzapfel2007": ("#D55E00", "D", "Holzapfel et al. 2007"),
    "spandler_oneill2010": ("#CC79A7", "v", "Spandler & O'Neill 2010"),
}
BAND = "#0072B2"

FORM_TEXT = {
    "dc07_fixed": (
        "Dohmen & Chakraborty (2007) point-defect form with the fO2 exponent (1/6), composition "
        "term (3 per unit X_Fe), activation volume (7 cm3/mol) and 6x anisotropy held at the "
        "published values; only log D0 and E are fitted. The model, not the data, sets how D "
        "scales with fO2 and composition, so the band stays narrow away from the data: its width "
        "is conditional on that model being right."
    ),
    "dc07_free": (
        "The same point-defect form with the fO2 exponent, composition coefficient and both "
        "anisotropy factors fitted (V held at 7 cm3/mol). The data now set the fO2 dependence, "
        "and because few points lie near the round robin fO2 the band widens when it is "
        "extrapolated there."
    ),
    "mutch_x6": (
        "The Mutch et al. (2019) regression form: ln D linear in ln fO2, X_Fo, 1/T, P and P/T, "
        "with D[100] = D[010] = D[001]/6 assumed. Pressure enters through j P and h P/T, which "
        "only Holzapfel's 6-12 GPa runs constrain: their combination (the activation volume) is "
        "well determined, each term alone is not."
    ),
    "mutch_aniso": (
        "The Mutch form with the [100] and [010] factors fitted instead of fixed at 6, so the "
        "off-axis data (Dohmen Part I, Spandler & O'Neill) inform the anisotropy rather than "
        "being forced onto [001]."
    ),
    "mutch_fT": (
        "The Mutch form with fitted anisotropy plus an fO2 x 1/T term, so the fO2 exponent "
        "changes with temperature (b + g/T). An F test finds the term significant (p = 0.008); a "
        "negative g weakens the fO2 dependence at low temperature, as Dohmen et al. (2007) "
        "observed."
    ),
}
METHOD_TEXT = {
    "ols": (
        "Ordinary least squares, every point weighted equally (Mutch et al.'s method). Band: draws "
        "from the covariance s^2 (X'X)^-1, which counts every row as independent."
    ),
    "wls": (
        "Least squares weighted by each point's stated (or assumed) log D error, so the most "
        "precise points pull hardest. Band: draws from the weighted covariance."
    ),
    "odr": (
        "Orthogonal distance regression (odrpack) with errors in T, fO2, X_Fo and P as well as "
        "log D. Of these only the fO2 errors move the fit: letting points slide in fO2 steepens "
        "the fitted fO2 dependence. Band: draws from the ODR covariance."
    ),
    "bootstrap (runs)": (
        "Least squares on 2000 resamples of whole experimental runs (several compositions from "
        "one Chakraborty couple move together). Band: quantiles of the resampled fits, as in "
        "odr_bootstrap."
    ),
    "random study": (
        "Least squares with a random offset per study (tau, fitted by REML): points from one "
        "study share their offset, so a study counts for less than its number of points and "
        "the two low, data-rich studies (Chakraborty, Dohmen) pull less. Band: the law averaged "
        "over labs; dashed lines add tau, the 68% range for what one new lab would measure."
    ),
}
METHODS = ("ols", "wls", "odr", "bootstrap (runs)", "random study")

PANELS = {
    "a": ("centre of the data: fO2 1e-7 Pa, Fo90, 1 atm", -7.0, 0.90, 1e5),
    "b": ("round robin KS20: fO2 1.4e-3 Pa, Fo85, 42 MPa", math.log10(0.00138), 0.85, 4.2e7),
}


def _draws(fit: dict, rng: np.random.Generator) -> np.ndarray:
    """Parameter sets: the bootstrap resamples, or draws from the fit's covariance."""
    if "samples" in fit:
        return fit["samples"]
    cov = fit["cov"]
    sd = np.sqrt(np.diag(cov))
    corr = cov / np.outer(sd, sd)
    w, vecs = np.linalg.eigh((corr + corr.T) / 2)
    root = vecs * np.sqrt(np.clip(w, 0, None))  # tolerant of a near-singular matrix
    z = rng.standard_normal((DRAWS, len(sd)))
    return fit["beta"] + (z @ root.T) * sd


def _grid(t_c: np.ndarray, log_fo2: float, x_fo: float, p_pa: float) -> F.Vars:
    n = len(t_c)
    return F.Vars(
        np.full(n, log_fo2),
        np.full(n, x_fo),
        t_c + 273.15,
        np.full(n, p_pa),
        np.zeros(n),
        np.zeros(n),
    )


def _published(name: str, t_c: np.ndarray, log_fo2: float, x_fo: float, p_pa: float) -> np.ndarray:
    from forster import DIFFUSIVITY_PRESETS, Conditions

    d = DIFFUSIVITY_PRESETS[name]
    return np.array(
        [np.log10(d.axis_diffusivities(Conditions(t, p_pa, 10**log_fo2), x_fo)[2]) for t in t_c]
    )


def _moved(df: pd.DataFrame, form: F.Form, beta: np.ndarray, log_fo2, x_fo, p_pa) -> np.ndarray:
    """Each point's log10 D moved to the panel's fO2, composition, P and [001] by the fit."""
    v_own = F.Vars.of(df)
    target = df.assign(log10_fO2_Pa=log_fo2, X_Fo=x_fo, P_Pa=p_pa, axis="[001]")
    v_t = F.Vars.of(target)
    own = form.offset(v_own) + form.design(v_own) @ beta
    to = form.offset(v_t) + form.design(v_t) @ beta
    return (df["ln_D"].to_numpy() + to - own) / LN10


def _change(a: float, b: float) -> str:
    """'2.3x wider', 'about the same', or '0.7x as wide'."""
    ratio = b / a
    if 0.9 <= ratio <= 1.1:
        return "about the same width"
    return f"{ratio:.1f}x wider" if ratio > 1 else f"{ratio:.1f}x as wide"


# Opacity of a point by the 1-sigma uncertainty (log D) of the correction that moved it.
MOVE_BINS = ((0.0, 0.03, 1.0), (0.03, 0.1, 0.55), (0.1, np.inf, 0.28))


def _correction_sd(df: pd.DataFrame, form: F.Form, cov: np.ndarray, log_fo2, x_fo, p_pa):
    """1-sigma (log D) of each point's move to the panel's conditions, from the fit's
    covariance. One error in a coefficient moves all points the same way, so this
    part is shared between points (and with the band), not independent scatter."""
    v_own = F.Vars.of(df)
    target = df.assign(log10_fO2_Pa=log_fo2, X_Fo=x_fo, P_Pa=p_pa, axis="[001]")
    dg = form.design(F.Vars.of(target)) - form.design(v_own)
    return np.sqrt(np.clip(np.einsum("ij,jk,ik->i", dg, cov, dg), 0, None)) / LN10


def _summary(form: F.Form, fit: dict, band_a: float, band_b: float, at_ks20: float) -> str:
    p = dict(zip(form.params, fit["beta"], strict=True))
    se = dict(zip(form.params, np.sqrt(np.diag(fit["cov"])), strict=True))
    bits = []
    if "E_J_mol" in p:
        bits.append(f"E = {p['E_J_mol'] / 1e3:.0f} +- {se['E_J_mol'] / 1e3:.0f} kJ/mol")
    if "q" in p and "g_fT" not in p:
        bits.append(f"E = {-p['q'] * F.R / 1e3:.0f} +- {se['q'] * F.R / 1e3:.0f} kJ/mol")
    if "g_fT" in p:
        # The fO2 x 1/T term moves part of the T dependence: E = -R (q + g ln fO2[bar]).
        e_at = {
            label: -F.R * (p["q"] + p["g_fT"] * (log_pa - 5) * LN10) / 1e3
            for label, log_pa in (("1e-7 Pa", -7.0), ("KS20 fO2", math.log10(0.00138)))
        }
        bits.append(
            "E = "
            + ", ".join(f"{e:.0f} kJ/mol at {k}" for k, e in e_at.items())
            + " (q alone is not an activation energy here)"
        )
    if "m_fO2" in p:
        bits.append(f"fO2 exponent {p['m_fO2']:.3f} +- {se['m_fO2']:.3f}")
    if "b" in p:
        b_eff = p["b"] + p.get("g_fT", 0.0) / 1473.15
        label = "at 1200 C " if "g_fT" in p else ""
        bits.append(f"fO2 exponent {label}{b_eff:.3f}")
    if "k100" in p:
        bits.append(f"anisotropy {math.exp(-p['k100']):.1f}x, {math.exp(-p['k010']):.1f}x")
    if "tau" in fit:
        lo, hi = (t / LN10 for t in fit["tau_range"])
        bits.append(
            f"between-study sd {fit['tau'] / LN10:.2f} log (95% {lo:.2f}-{hi:.2f}) against "
            f"{fit['sd'] / LN10:.2f} within a study"
        )
    return (
        "; ".join(bits)
        + f". At 1200 C, D[001] is {at_ks20:+.2f} log units from the default (Dohmen & Chakraborty"
        f" 2007) at the KS20 conditions. The 68% band at 1200 C is +-{band_a:.2f} log at the data's"
        f" centre and +-{band_b:.2f} at the KS20 fO2 ({_change(band_a, band_b)})."
    )


def figure(form: F.Form, method: str, fit: dict, df: pd.DataFrame, rng: np.random.Generator):
    draws = _draws(fit, rng)
    t = np.linspace(700, 1600, 91)
    y_sigma = F._y_sigma_ln(df) / LN10
    # Fixed positions: constrained layout would widen a column to fit the legend.
    fig = plt.figure(figsize=(10, 7.0))
    axd = {
        "a": fig.add_axes((0.07, 0.38, 0.40, 0.47)),
        "b": fig.add_axes((0.57, 0.38, 0.40, 0.47)),
    }
    widths, at_ks20 = {}, 0.0
    for key, (title, log_fo2, x_fo, p_pa) in PANELS.items():
        ax = axd[key]
        v = _grid(t, log_fo2, x_fo, p_pa)
        curves = (form.offset(v)[:, None] + form.design(v) @ draws.T) / LN10
        best = (form.offset(v) + form.design(v) @ fit["beta"]) / LN10
        x = 1e4 / (t + 273.15)
        lo95, lo68, hi68, hi95 = np.quantile(curves, [0.025, 0.16, 0.84, 0.975], axis=1)
        ax.fill_between(x, lo95, hi95, color=BAND, alpha=0.15, lw=0, label="95% band")
        ax.fill_between(x, lo68, hi68, color=BAND, alpha=0.35, lw=0, label="68% band")
        ax.plot(x, best, color=BAND, lw=1.8, label="best fit")
        if "tau" in fit:
            half = np.hypot((hi68 - lo68) / 2, fit["tau"] / LN10)
            ax.plot(x, best + half, color=BAND, lw=0.9, ls="--", label="68%, one new lab")
            ax.plot(x, best - half, color=BAND, lw=0.9, ls="--")
        for name, ls, label in (
            ("dohmen_chakraborty2007", "--", "Dohmen & Chakraborty 2007"),
            ("mutch2019_tamed", ":", "Mutch et al. 2019, TaMED"),
        ):
            ax.plot(
                x, _published(name, t, log_fo2, x_fo, p_pa), color="0.2", lw=1.1, ls=ls, label=label
            )
        moved = _moved(df, form, fit["beta"], log_fo2, x_fo, p_pa)
        corr = _correction_sd(df, form, fit["cov"], log_fo2, x_fo, p_pa)
        total = np.sqrt(y_sigma**2 + corr**2)
        for src, (colour, marker, label) in STUDY_STYLE.items():
            labelled = False
            for lo_c, hi_c, alpha in MOVE_BINS:
                m = ((df["source"] == src) & (corr >= lo_c) & (corr < hi_c)).to_numpy()
                if not m.any():
                    continue
                x_pts = 1e4 / df["T_K"][m]
                # Light bar: measurement and correction combined; dark bar: measurement only.
                ax.errorbar(
                    x_pts,
                    moved[m],
                    yerr=total[m],
                    fmt="none",
                    ecolor=colour,
                    elinewidth=2.2,
                    alpha=0.18 * alpha / MOVE_BINS[0][2],
                    capsize=0,
                )
                ax.errorbar(
                    x_pts,
                    moved[m],
                    yerr=y_sigma[m],
                    fmt=marker,
                    color=colour,
                    mfc="none",
                    ms=4,
                    elinewidth=0.6,
                    capsize=0,
                    alpha=alpha,
                    label=None if labelled else label,
                )
                labelled = True
        i1200 = int(np.argmin(np.abs(t - 1200)))
        widths[key] = (hi68[i1200] - lo68[i1200]) / 2
        if key == "b":
            at_ks20 = (
                best[i1200]
                - _published("dohmen_chakraborty2007", t[i1200 : i1200 + 1], log_fo2, x_fo, p_pa)[0]
            )
            ax.axvline(x[i1200], color="0.5", lw=0.8, ls=":")
        ax.set_title(f"({key}) {title}")
        ax.set_xlabel("10^4 / T (1/K)")
        ax.set_ylabel("log10 D[001] (m2/s)")
        lo, hi = ax.get_xlim()
        top = ax.twiny()
        top.set_xlim(lo, hi)
        marks = [c for c in (800, 1000, 1200, 1400, 1600) if lo <= 1e4 / (c + 273.15) <= hi]
        top.set_xticks([1e4 / (c + 273.15) for c in marks], [str(c) for c in marks])
        top.set_xlabel("T (C)")
    handles, labels = axd["a"].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.63, 0.27),
        fontsize=6.8,
        frameon=False,
        ncol=2,
        columnspacing=0.8,
        handletextpad=0.4,
    )
    body = " ".join(
        [
            FORM_TEXT[form.name],
            METHOD_TEXT[method],
            _summary(form, fit, widths["a"], widths["b"], at_ks20),
            "Points are moved to each panel's conditions by this fit: dark bars are the "
            "measurement error, light bars add the uncertainty of that move, and fainter points "
            "were moved further (correction 1-sigma over 0.03 or 0.1 log). The move's "
            "uncertainty is shared by all points moved the same way, so it shifts the whole cloud "
            "with the line (the band) rather than adding scatter.",
        ]
    )
    fig.text(0.07, 0.27, textwrap.fill(body, 96), va="top", ha="left", fontsize=7.8)
    fig.suptitle(
        f"{form.name}, {method}: D[001] at 1200 C and the KS20 fO2 is {at_ks20:+.2f} log from the "
        f"default;\nits 68% band there is +-{widths['b']:.2f} log, "
        f"against +-{widths['a']:.2f} at the centre of the data",
        fontsize=10,
        y=0.985,
    )
    name = f"{form.name}__{method.split(' ')[0]}.png"
    fig.savefig(OUT / name, dpi=170)
    plt.close(fig)
    return name, at_ks20, widths["a"], widths["b"]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    df = F.tamed()
    rows = []
    for form in F.FORMS.values():
        fits = {
            "ols": F.fit_ls(form, df, weighted=False),
            "wls": F.fit_ls(form, df, weighted=True),
            "odr": F.fit_odr(form, df),
            "bootstrap (runs)": F.fit_bootstrap(form, df),
            "random study": F.fit_random(form, df),
        }
        for method in METHODS:
            name, d, wa, wb = figure(form, method, fits[method], df, rng)
            rows.append(
                {
                    "figure": name,
                    "form": form.name,
                    "method": method,
                    "minus_default_1200C_KS20": d,
                    "band68_centre": wa,
                    "band68_KS20": wb,
                }
            )
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "index.csv", index=False)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
