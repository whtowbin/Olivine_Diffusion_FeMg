"""The final Fe-Mg presets: their fits, residuals, and the candidate methods.

figures/presets/
- p1_bowties.png: log10 D[001] against 1/T at (a) the centre of the data (fO2 1e-7 Pa,
  Fo90, 1 atm) and (b) the RR1200 conditions (round robin, 1200 C). femg_refit2026 with its 68% and
  95% bands and one new laboratory; mutch2019_tamed and mutch2019_global with their
  68% bands (their DFENS covariance); DC07. Points: the preset's data, moved to each
  panel's conditions by the preset; Holzapfel's Au-capsule runs filled, their bars
  including the fO2 sd of 2 log units.
- p2_residuals.png: the preset's residuals (log10 D minus the preset at each point's
  own conditions and axis) against T, fO2, X_Fo and P, by study.
- p3_candidates.png: D[001] at RR1200, RR1102 and the paper conditions, relative to DC07,
  for the candidate fitting methods, built up step by step. A-C fix the activation
  volume V at 7 cm3/mol: REML (least squares with a random offset per laboratory) on
  the verified data (A); with each run's errors in T (at least 10 C), fO2, X_Fo and P
  (B); plus Holzapfel's Au-capsule runs at Ni-NiO with an fO2 sd of 1 or 2 log (C).
  D fits V. E, the femg_refit2026 preset, adds each value's published D error and a
  random offset per experiment. ODR on the preset's data, V 7, with the stated T errors and
  with a 20 C floor (odr_check.py); the bootstrap of runs (bootstrap_preset.py), by the
  preset's method and by OLS; and the Mutch presets.

Bands are the laws' own linear propagation (ArrheniusDiffusivity.log10_sigma), exact
for these laws (docs/references/law-uncertainty-limits.md).

Run: uv run --with odrpack python analyses/femg_refit/preset_figures.py
Writes the figures and results/verified/candidates.csv.
"""

from __future__ import annotations

import math
from pathlib import Path

import fit_models as F
import fit_verified as V
import matplotlib
import odr_check as O

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from forster import DIFFUSIVITY_PRESETS, Conditions, Orientation

HERE = Path(__file__).parent
OUT = HERE / "figures" / "presets"
LN10 = math.log(10)
ALONG_001 = Orientation(90, 90, 0)
AXIS = {"[100]": 0, "[010]": 1, "[001]": 2}
AU_SD = V.PRESET_AU_FO2_SD
PRESET = DIFFUSIVITY_PRESETS["femg_refit2026"]
LAWS = {  # name: (colour, line style, label)
    "femg_refit2026": ("#0072B2", "-", "femg_refit2026 (the preset)"),
    "mutch2019_tamed": ("#D55E00", "--", "mutch2019_tamed"),
    "mutch2019_global": ("#CC79A7", ":", "mutch2019_global"),
}
PANELS = {
    "a": ("centre of the data: fO2 1e-7 Pa, Fo90, 1 atm", -7.0, 0.90, 1e5),
    "b": ("round robin RR1200: fO2 1.4e-3 Pa, Fo85, 42 MPa", math.log10(0.00138), 0.85, 4.2e7),
}
plt.rcParams.update({"pdf.fonttype": 42, "font.size": 9, "axes.titlesize": 9.5})


def law_curve(name: str, t_c: np.ndarray, log_fo2: float, x_fo: float, p_pa: float):
    """log10 D[001] and its 1 sigma along a temperature grid."""
    law = DIFFUSIVITY_PRESETS[name]
    mu, sd = [], []
    for t in t_c:
        c = Conditions(float(t), p_pa, 10**log_fo2)
        mu.append(math.log10(law.axis_diffusivities(c, x_fo)[2]))
        sd.append(law.log10_sigma(c, ALONG_001, x_fo))
    return np.array(mu), np.array(sd)


def _groups(data: pd.DataFrame):
    """(mask, colour, marker, filled, label) per study, the Au runs apart."""
    au = V._is_au(data)
    for src, (colour, marker, label) in F.STUDY_STYLE.items():
        m = (data["source"] == src).to_numpy()
        if src == "holzapfel2007":
            yield m & ~au, colour, marker, False, label + ", Ni-NiO capsules"
            yield m & au, colour, marker, True, label + f", Au capsules (fO2 ± {AU_SD:g} log)"
        elif m.any():
            yield m, colour, marker, False, label


def _sigma(data: pd.DataFrame, fit: dict) -> np.ndarray:
    """log10 D error of each point; the Au runs' includes their fO2 sd times m."""
    sigma = F._y_sigma_ln(data) / LN10
    return np.where(V._is_au(data), np.hypot(sigma, fit["beta"][2] * AU_SD), sigma)


def p1(data: pd.DataFrame, fit: dict) -> None:
    t = np.linspace(780, 1450, 68)
    x = 1e4 / (t + 273.15)
    tau = PRESET.uncertainty.between_study_log10
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0), layout="constrained", sharey=False)
    sigma_pts = _sigma(data, fit)
    for ax, (key, (title, log_fo2, x_fo, p_pa)) in zip(axes, PANELS.items(), strict=True):
        for name, (colour, ls, label) in LAWS.items():
            mu, sd = law_curve(name, t, log_fo2, x_fo, p_pa)
            if name == "femg_refit2026":
                ax.fill_between(
                    x,
                    mu - 1.96 * sd,
                    mu + 1.96 * sd,
                    color=colour,
                    alpha=0.12,
                    lw=0,
                    label="preset, 95%",
                )
                ax.fill_between(
                    x, mu - sd, mu + sd, color=colour, alpha=0.3, lw=0, label="preset, 68%"
                )
                nl = np.hypot(sd, tau)
                ax.plot(x, mu + nl, color=colour, lw=0.8, ls="--", label="preset, 68% one new lab")
                ax.plot(x, mu - nl, color=colour, lw=0.8, ls="--")
            else:
                ax.fill_between(x, mu - sd, mu + sd, color=colour, alpha=0.15, lw=0)
            ax.plot(x, mu, color=colour, ls=ls, lw=1.6, label=label)
        dc = [F.preset_log10_d("dohmen_chakraborty2007", tt, p_pa, log_fo2, x_fo) for tt in t]
        ax.plot(x, dc, color="0.2", lw=1.0, ls="-.", label="dohmen_chakraborty2007 (default)")
        at = {"log10_fO2_Pa": log_fo2, "X_Fo": x_fo, "P_Pa": p_pa, "axis": "[001]"}
        moved = F.moved(data, V.FORM, fit["beta"], **at)
        for m, colour, marker, filled, label in _groups(data):
            ax.errorbar(
                1e4 / data["T_K"][m],
                moved[m],
                yerr=sigma_pts[m],
                fmt=marker,
                color=colour,
                mfc=colour if filled else "none",
                ms=4,
                elinewidth=0.6,
                capsize=0,
                label=label,
            )
        ax.set_title(f"({key}) {title}", loc="left")
        ax.set_xlabel("10⁴ / T (1/K)")
        ax.set_ylabel("log10 D[001] (m²/s)")
        lo, hi = ax.get_xlim()
        top = ax.twiny()
        top.set_xlim(lo, hi)
        marks = [c for c in (800, 1000, 1200, 1400) if lo <= 1e4 / (c + 273.15) <= hi]
        top.set_xticks([1e4 / (c + 273.15) for c in marks], [str(c) for c in marks])
        top.set_xlabel("T (°C)")
    seen: dict = {}
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels(), strict=True):
            seen.setdefault(lab, h)
    fig.legend(
        list(seen.values()),
        list(seen.keys()),
        loc="outside lower center",
        ncol=4,
        fontsize=7,
        frameon=False,
    )
    ks = PANELS["b"]
    c = Conditions(1200.0, ks[3], 10 ** ks[1])
    rel = {
        n: math.log10(DIFFUSIVITY_PRESETS[n].axis_diffusivities(c, ks[2])[2])
        - F.preset_log10_d("dohmen_chakraborty2007", 1200.0, ks[3], ks[1], ks[2])
        for n in LAWS
    }
    fig.suptitle(
        f"At RR1200, 1200 °C, against DC07: femg_refit2026 {rel['femg_refit2026']:+.2f} "
        f"± {PRESET.log10_sigma(c, ALONG_001, ks[2]):.2f}, mutch2019_tamed "
        f"{rel['mutch2019_tamed']:+.2f}, mutch2019_global {rel['mutch2019_global']:+.2f}",
        fontsize=10,
    )
    fig.savefig(OUT / "p1_bowties.png", dpi=200)
    plt.close(fig)


def within_study_trends(data: pd.DataFrame, resid: np.ndarray) -> list[tuple]:
    """(study, variable, slope, z) of each study's residuals against T, log fO2 and X_Fo."""
    out = []
    for src, idx in data.groupby("source").groups.items():
        sub = data.loc[idx]
        r = resid[np.asarray([data.index.get_loc(i) for i in idx])]
        for name, xv in (
            ("T (C)", sub["T_K"] - 273.15),
            ("log fO2", sub["log10_fO2_Pa"]),
            ("X_Fo", sub["X_Fo"]),
        ):
            xv = np.asarray(xv, float)
            if len(r) < 4 or np.ptp(xv) < 1e-9:
                continue
            X = np.column_stack([np.ones(len(r)), xv])
            b, *_ = np.linalg.lstsq(X, r, rcond=None)
            e = r - X @ b
            se = math.sqrt((e @ e / (len(r) - 2)) / ((xv - xv.mean()) ** 2).sum())
            out.append((src, name, float(b[1]), float(b[1] / se)))
    return out


def p2(data: pd.DataFrame) -> None:
    pred = np.array(
        [
            math.log10(
                PRESET.axis_diffusivities(
                    Conditions(r.T_K - 273.15, r.P_Pa, 10**r.log10_fO2_Pa), r.X_Fo
                )[AXIS[r.axis]]
            )
            for r in data.itertuples()
        ]
    )
    resid = data["log10_D_m2s"].to_numpy() - pred
    xs = (
        ("T (°C)", data["T_K"] - 273.15),
        ("log10 fO2 (Pa)", data["log10_fO2_Pa"]),
        ("X_Fo", data["X_Fo"]),
        ("P (GPa)", data["P_Pa"] / 1e9),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained", sharey=True)
    for ax, (label, xv) in zip(axes.flat, xs, strict=True):
        for m, colour, marker, filled, name in _groups(data):
            ax.plot(
                np.asarray(xv)[m],
                resid[m],
                marker,
                color=colour,
                mfc=colour if filled else "none",
                ms=4.5,
                ls="",
                label=name,
            )
        ax.axhline(0, color="0.2", lw=0.8, ls=":")
        ax.set_xlabel(label)
        ax.set_ylabel("log10 D minus the preset")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=7.5, frameon=False)
    by = pd.Series(resid).groupby(data["source"].to_numpy()).mean()
    trends = within_study_trends(data, resid)
    strong = [t for t in trends if abs(t[3]) >= 2]
    trend_txt = (
        "; ".join(
            f"{F.STUDY_STYLE[s][2]} vs {v}: {b:+.3f} per unit (z {z:+.1f})" for s, v, b, z in strong
        )
        if strong
        else "no within-study trend beyond 2 sigma"
    )
    fig.suptitle(
        f"The preset's residuals (rms {np.sqrt(np.mean(resid**2)):.2f} log).\nStudy means: "
        + ", ".join(f"{F.STUDY_STYLE[k][2]} {v:+.2f}" for k, v in by.items())
        + f".\nWithin studies, beyond 2 sigma: {trend_txt}",
        fontsize=10,
    )
    fig.savefig(OUT / "p2_residuals.png", dpi=200)
    plt.close(fig)


PRESET_ROW = "E  D + published D errors, per-experiment offset (femg_refit2026)"


def bootstrap_rows(n_points: int, conds: list[str]) -> list[dict]:
    """The bootstrap of runs (bootstrap_preset.py): mean and half the 16-84% width."""
    path = HERE / "results" / "verified" / "bootstrap.csv"
    if not path.exists():
        print("no results/verified/bootstrap.csv: run bootstrap_preset.py for its rows")
        return []
    boot = pd.read_csv(path)
    rows = []
    for method, label in (
        ("preset method", "E, bootstrap of runs"),
        ("OLS", "OLS, bootstrap of runs"),
    ):
        q = boot[boot["method"] == method].set_index("quantity")
        row = {"candidate": f"{label} ({int(q['n_draws'].iloc[0])} draws)"}
        row |= {"n_points": n_points, "tau": np.nan, "m": q.loc["m_fO2", "boot_mean"]}
        for c in conds:
            r = q.loc[f"{c}_vs_dc07"]
            row |= {c: r["boot_mean"], f"{c}_se": (r["boot_p84"] - r["boot_p16"]) / 2}
        rows.append(row)
    return rows


def candidates() -> pd.DataFrame:
    ver, both = V.verified(), V.preset_data()
    v7 = V.FORM_V7
    fits = {
        "A  verified data, REML, V 7": (v7, F.fit_random(v7, ver), len(ver)),
        "B  A + errors in T (at least 10 °C), fO2, X_Fo, P": (
            v7,
            V.fit_with_input_errors(ver, form=v7, published=False, run_term=False),
            len(ver),
        ),
        "C  B + Holzapfel Au at NNO, fO2 sd 1": (
            v7,
            V.fit_preset(both, v7, au_fo2_sd=1.0, published=False, run_term=False),
            len(both),
        ),
        "C  B + Holzapfel Au at NNO, fO2 sd 2": (
            v7,
            V.fit_preset(both, v7, published=False, run_term=False),
            len(both),
        ),
        "D  C (sd 2) with V fitted": (
            V.FORM,
            V.fit_preset(both, published=False, run_term=False),
            len(both),
        ),
        PRESET_ROW: (V.FORM, V.fit_preset(both), len(both)),
    }
    conds = {k.split(",")[0]: c for k, c in F._conditions().items()}
    rows = []
    for name, (form, fit, n) in fits.items():
        row = {"candidate": name, "n_points": n, "tau": fit["tau"] / LN10, "m": fit["beta"][2]}
        for label, c in conds.items():
            mu, se = F.predict(form, fit["beta"], fit["cov"], c)
            row |= {f"{label}": mu - F._dc07_preset(c), f"{label}_se": se}
        rows.append(row)
    for floor in (0.0, 20.0):
        beta = O.odr(both, floor, offsets=False)
        tag = "stated T errors" if floor == 0 else f"T errors at least {floor:.0f} °C"
        row = {"candidate": f"ODR, V 7, no study terms, {tag}", "n_points": len(both)}
        row |= {"tau": np.nan, "m": beta[2]}
        for label, c in conds.items():
            mu, _ = F.predict(O.FORM, beta, np.eye(4), c)
            row |= {f"{label}": mu - F._dc07_preset(c), f"{label}_se": np.nan}
        rows.append(row)
    rows += bootstrap_rows(len(both), list(conds))
    for name in ("mutch2019_tamed", "mutch2019_global"):
        law = DIFFUSIVITY_PRESETS[name]
        row = {"candidate": name, "n_points": np.nan, "tau": np.nan, "m": law.fO2_exponent}
        for label, c in conds.items():
            cc = Conditions(float(c.T[0]) - 273.15, float(c.P[0]), 10 ** float(c.log_fo2_pa[0]))
            x = float(c.x_fo[0])
            row |= {
                label: math.log10(law.axis_diffusivities(cc, x)[2]) - F._dc07_preset(c),
                f"{label}_se": law.log10_sigma(cc, ALONG_001, x),
            }
        rows.append(row)
    return pd.DataFrame(rows)


def p3(tab: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained", sharey=True)
    y = np.arange(len(tab))
    for ax, cond in zip(axes, ("RR1200", "RR1102", "Paper"), strict=True):
        for yi, r in enumerate(tab.itertuples()):
            colour = (
                "#0072B2"
                if r.candidate == PRESET_ROW
                else ("#D55E00" if r.candidate.startswith("mutch") else "0.35")
            )
            se = getattr(r, f"{cond}_se")
            ax.errorbar(
                getattr(r, cond),
                yi,
                xerr=None if math.isnan(se) else se,
                fmt="D"
                if r.candidate.startswith("ODR")
                else ("s" if "bootstrap" in r.candidate else "o"),
                mfc="none" if r.candidate.startswith("ODR") else colour,
                color=colour,
                ms=5,
                elinewidth=2,
                capsize=0,
            )
        ax.axvline(0, color="0.2", lw=0.8, ls=":")
        ax.set_title(cond, loc="left")
        ax.set_xlabel("D[001] minus DC07 (log)")
    axes[0].set_yticks(y, tab["candidate"])
    axes[0].invert_yaxis()
    ks = tab.set_index("candidate")["RR1200"]
    odr = ks[ks.index.str.startswith("ODR")]
    fig.suptitle(
        "log10 D[001] relative to Dohmen and Chakraborty (2007) at the round robin "
        "conditions (RR1200, 1200 °C; RR1102, 1102 °C) and the paper's (1250 °C).\n"
        f"At RR1200, step by step: A {ks.iloc[0]:+.2f}, B {ks.iloc[1]:+.2f}, C {ks.iloc[3]:+.2f}, "
        f"D {ks.iloc[4]:+.2f}, E (femg_refit2026, blue) {ks[PRESET_ROW]:+.2f}. ODR without "
        f"laboratory offsets gives {odr.iloc[0]:+.2f} or {odr.iloc[1]:+.2f}, set by the assumed "
        "T error.\nBars: 1 sigma of each fitted equation; squares: bootstrap mean and half its "
        "16-84% range; open diamonds: ODR (no uncertainty shown)",
        fontsize=9.5,
    )
    fig.savefig(OUT / "p3_candidates.png", dpi=200)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = V.preset_data()
    fit = V.fit_preset(data)
    p1(data, fit)
    p2(data)
    tab = candidates()
    tab.to_csv(HERE / "results" / "verified" / "candidates.csv", index=False)
    p3(tab)
    pd.set_option("display.width", 220)
    print(tab.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
