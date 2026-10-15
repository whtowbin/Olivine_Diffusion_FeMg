"""The [010] question: Oeser et al. (2026) against the 6x anisotropy of DC07, DIPRA, DFENS.

figures/anisotropy/
- a1_oeser_ratios.png: within each Oeser depth-profile run, log10 D[001]/D[010] and
  log10 D[001]/D[100] against temperature, with 6x (log 0.78) and the fitted 3.5x.
- a2_residuals.png: Dohmen et al. (2007) and Oeser rows against the 6x law fitted
  without Oeser, by axis, against temperature. Under 6x the three axes should agree.
- a3_options.png: the three ways Oeser can enter the 6x preset (out; [001] and [100]
  only; all rows), and the law with fitted anisotropy, for D along [001] and along
  [010] at KS20 and along [001] at SH63, relative to DC07.

Run: uv run python analyses/femg_refit/anisotropy_figures.py
Writes the figures and results/verified/oeser_options.csv.
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

HERE = Path(__file__).parent
OUT = HERE / "figures" / "anisotropy"
LN10, LOG6 = math.log(10), math.log10(6)
AXIS_STYLE = {"[001]": ("#0072B2", "o"), "[010]": ("#D55E00", "s"), "[100]": ("#009E73", "^")}
plt.rcParams.update({"pdf.fonttype": 42, "font.size": 9, "axes.titlesize": 9.5})


def oeser_runs() -> pd.DataFrame:
    """Axis means of each Oeser depth-profile run and the ratios between axes (log10)."""
    o = pd.read_csv(HERE / "data" / "oeser2026.csv")
    o = o[~o["run"].isin(V.X.LINESCAN_RUNS)]
    m = o.groupby(["run", "T_C", "log10_fO2_Pa", "axis"])["log10_D_FeMg"].mean().unstack("axis")
    m["c_over_b"] = m["[001]"] - m["[010]"]
    m["c_over_a"] = m["[001]"] - m["[100]"]
    return m.reset_index()


def a1(runs: pd.DataFrame, fitted_010: float) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.6), layout="constrained")
    for col, axis, label in (
        ("c_over_a", "[100]", "D[001]/D[100]"),
        ("c_over_b", "[010]", "D[001]/D[010]"),
    ):
        colour, marker = AXIS_STYLE[axis]
        ax.plot(runs["T_C"], runs[col], marker, color=colour, ms=7, ls="", label=label)
        for r in runs.itertuples():
            ax.annotate(
                f"1e{r.log10_fO2_Pa:g}",
                (r.T_C, getattr(r, col)),
                xytext=(5, 4),
                textcoords="offset points",
                va="bottom",
                fontsize=6.5,
                color="0.4",
            )
    ax.axhline(LOG6, color="0.15", ls="--", lw=1.1, label="6x: DC07, DIPRA, DFENS")
    ax.axhline(fitted_010, color="#D55E00", ls=":", lw=1.1, label="[010] fitted with Oeser (3.5x)")
    ax.axhline(0, color="0.6", lw=0.7)
    ax.set_xlabel("T (°C), each point one Oeser depth-profile run (labels: fO2 in Pa)")
    ax.set_ylabel("log10 of the ratio within the run")
    ax.set_xlim(1080, 1340)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", fontsize=7, frameon=False, ncol=4)
    c_b, c_a = 10 ** runs["c_over_b"], 10 ** runs["c_over_a"]
    ax.set_title(
        f"Oeser: [001] is {c_a.min():.1f}-{c_a.max():.1f}x faster than [100] (6x is DC07's), "
        "but only "
        f"{c_b.min():.1f}-{c_b.max():.1f}x\nfaster than [010], in every run from 1100 to 1303 °C",
        loc="left",
    )
    fig.savefig(OUT / "a1_oeser_ratios.png", dpi=200)
    plt.close(fig)


def a2(fit: dict) -> None:
    d = pd.concat(
        [V.verified().query("source == 'dohmen2007_part1'"), V.oeser_dp()], ignore_index=True
    )
    v = F.Vars.of(d)
    resid = (d["ln_D"].to_numpy() - V.FORM.offset(v) - V.FORM.design(v) @ fit["beta"]) / LN10
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.0), layout="constrained", sharey=True)
    for ax, axis in zip(axes, ("[001]", "[100]", "[010]"), strict=True):
        colour, marker = AXIS_STYLE[axis]
        for src, mfc, label in (
            ("dohmen2007_part1", "none", "Dohmen et al. 2007"),
            ("oeser2026", colour, "Oeser et al. 2026"),
        ):
            m = ((d["source"] == src) & (d["axis"] == axis)).to_numpy()
            ax.plot(
                d["T_K"][m] - 273.15,
                resid[m],
                marker,
                color=colour,
                mfc=mfc,
                ms=5,
                ls="",
                label=label,
            )
        ax.axhline(0, color="0.2", lw=0.9, ls=":")
        oe = ((d["source"] == "oeser2026") & (d["axis"] == axis)).to_numpy()
        ax.set_title(f"{axis}: Oeser mean {resid[oe].mean():+.2f}", loc="left")
        ax.set_xlabel("T (°C)")
        ax.legend(loc="upper left", fontsize=7, frameon=False)
    axes[0].set_ylabel("log10 D minus the 6x law fitted without Oeser")
    fig.suptitle(
        "Under 6x the three axes should sit alike: Dohmen's do; Oeser's [010] sits about 0.4 log "
        "above their own [001] and [100]",
        fontsize=10,
    )
    fig.savefig(OUT / "a2_residuals.png", dpi=200)
    plt.close(fig)


def options() -> pd.DataFrame:
    base, oes = V.verified(), V.oeser_dp()
    sets = {
        "(c) 6x, Oeser out (the preset)": (V.FORM, base),
        "(b) 6x, Oeser [001] and [100] only": (
            V.FORM,
            pd.concat([base, oes[oes["axis"] != "[010]"]]),
        ),
        "(a) 6x, all Oeser depth profiles": (V.FORM, pd.concat([base, oes])),
        "fitted anisotropy, all Oeser (previous preset)": (
            F.FORMS["dc07_free"],
            pd.concat([base, oes]),
        ),
    }
    conds = {k.split(",")[0]: c for k, c in F._conditions().items()}
    rows = []
    for name, (form, df) in sets.items():
        df = df.reset_index(drop=True)
        fit = F.fit_random(form, df)
        b, cov, tau = fit["beta"], fit["cov"], fit["tau"] / LN10
        row = {"option": name, "n_points": len(df), "tau": tau, "m": b[2], "E_kJ_mol": b[1] / 1e3}
        for label, axis in (("KS20 [001]", 0), ("KS20 [010]", 1), ("SH63 [001]", 0)):
            c = conds[label.split()[0]]
            v = F.Vars(c.log_fo2_pa, c.x_fo, c.T, c.P, np.zeros(1), np.array([float(axis)]))
            mu, se = F.predict(form, b, cov, v)
            ref = F._dc07_preset(c) - (LOG6 if axis else 0.0)  # DC07 on the same axis
            row |= {
                f"{label} vs dc07": mu - ref,
                f"{label} se": se,
                f"{label} se_new_lab": math.hypot(se, tau),
            }
        rows.append(row)
    return pd.DataFrame(rows)


def a3(tab: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained", sharey=True)
    y = np.arange(len(tab))
    for ax, label in zip(axes, ("KS20 [001]", "KS20 [010]", "SH63 [001]"), strict=True):
        mu, se, nl = tab[f"{label} vs dc07"], tab[f"{label} se"], tab[f"{label} se_new_lab"]
        ax.errorbar(mu, y, xerr=nl, fmt="none", ecolor="0.75", elinewidth=1, capsize=0)
        ax.errorbar(mu, y, xerr=se, fmt="o", color="#0072B2", ms=5, elinewidth=2.2, capsize=0)
        ax.axvline(0, color="0.2", lw=0.8, ls=":")
        ax.set_title(label, loc="left")
        ax.set_xlabel("log10 D minus DC07 on the same axis")
    axes[0].set_yticks(
        y, [f"{o}  (n {n})" for o, n in zip(tab["option"], tab["n_points"], strict=True)]
    )
    axes[0].invert_yaxis()
    span_001 = np.ptp(tab["KS20 [001] vs dc07"])
    span_010 = np.ptp(tab["KS20 [010] vs dc07"])
    fig.suptitle(
        f"Along [001] the options differ by {span_001:.2f} log at KS20; "
        f"along [010] by {span_010:.2f}. "
        "Thick: law 1 sigma; thin: one new lab",
        fontsize=10,
    )
    fig.savefig(OUT / "a3_options.png", dpi=200)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = oeser_runs()
    fitted = F.fit_random(
        F.FORMS["dc07_free"], pd.concat([V.verified(), V.oeser_dp()], ignore_index=True)
    )
    a1(runs, -fitted["beta"][5] / LN10)
    a2(F.fit_random(V.FORM, V.verified()))
    tab = options()
    tab.to_csv(HERE / "results" / "verified" / "oeser_options.csv", index=False)
    a3(tab)
    pd.set_option("display.width", 220)
    print(runs.round(2).to_string(index=False))
    print(tab.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
