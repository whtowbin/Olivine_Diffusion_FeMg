"""Can Oeser et al. (2026)'s [001] data guide the fO2 dependence? Plots to decide.

The preset (fit_verified.FORM, 6x anisotropy; fit_verified.fit_preset) is fitted without
Oeser; each variant here is refitted by the same method. Here Oeser's [001]
depth-profile rows are shown against it, and added to it, to see what they say about the
fO2 exponent near the round robin and paper conditions (1e-4 to 1e-2 Pa).

- figures/oeser_fo2/f1_fo2_series.png: log D[001] against fO2 at 1150 and 1250 C, the
  temperatures at which Oeser varied fO2: Oeser's [001] runs and Dohmen's and Petry's
  points within 30 C, each moved to the panel's temperature by the law fitted without
  Oeser; that law, and the law refitted with Oeser's [001] rows.
- figures/oeser_fo2/f2_exponents.png: the fO2 exponent and the KS20 level for the law
  without Oeser, with Oeser's [001] rows, and with [001] and [100]; beside each study's
  own exponent (temperature, and axis where needed, removed).

Run: uv run python analyses/femg_refit/oeser_fo2_figures.py
"""

from __future__ import annotations

import math
from pathlib import Path

import fit_models as F
import fit_verified as V
import fo2_figures as FO
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
OUT = HERE / "figures" / "oeser_fo2"
LN10 = math.log(10)
PANELS = (1150.0, 1250.0)
NEAR_C = 30.0  # points within this many degrees of a panel are shown on it
STYLE = {
    "dohmen2007_part1": ("#56B4E9", "s", "Dohmen et al. 2007"),
    "petry2004": ("#009E73", "^", "Petry et al. 2004"),
    "chakraborty1997": ("#E69F00", "o", "Chakraborty 1997"),
    "oeser2026": ("#000000", "X", "Oeser et al. 2026, [001]"),
}
plt.rcParams.update({"pdf.fonttype": 42, "font.size": 9, "axes.titlesize": 9.5})


def fits() -> dict:
    base, oes = V.preset_data(), V.oeser_dp()
    sets = {
        "without Oeser (the preset)": base,
        "+ Oeser [001]": pd.concat([base, oes[oes["axis"] == "[001]"]]),
        "+ Oeser [001] and [100]": pd.concat([base, oes[oes["axis"] != "[010]"]]),
    }
    return {k: (V.fit_preset(d.reset_index(drop=True)), len(d)) for k, d in sets.items()}


def f1(fit_map: dict) -> None:
    base_fit = fit_map["without Oeser (the preset)"][0]
    with_fit = fit_map["+ Oeser [001]"][0]
    pts = pd.concat([V.verified(), V.oeser_dp()], ignore_index=True)
    pts = pts[pts["axis"] == "[001]"]
    fo2 = np.linspace(-9, -1, 41)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained", sharey=True)
    for ax, t_c in zip(axes, PANELS, strict=True):
        v = F.Vars.of(
            pd.DataFrame(
                {
                    "log10_fO2_Pa": fo2,
                    "T_K": t_c + 273.15,
                    "X_Fo": 0.915,
                    "P_Pa": 1e5,
                    "axis": "[001]",
                }
            )
        )
        for fit, colour, ls, label in (
            (base_fit, "#0072B2", "-", "6x law without Oeser"),
            (with_fit, "#D55E00", "--", "6x law with Oeser [001]"),
        ):
            ax.plot(
                fo2,
                (V.FORM.offset(v) + V.FORM.design(v) @ fit["beta"]) / LN10,
                color=colour,
                ls=ls,
                lw=1.6,
                label=label,
            )
        near = pts[((pts["T_K"] - 273.15) - t_c).abs() <= NEAR_C]
        moved = F.moved(near, V.FORM, base_fit["beta"], T_K=t_c + 273.15, X_Fo=0.915, P_Pa=1e5)
        for src, (colour, marker, label) in STYLE.items():
            m = (near["source"] == src).to_numpy()
            if m.any():
                filled = colour if src == "oeser2026" else "none"
                ax.plot(
                    near["log10_fO2_Pa"][m],
                    moved[m],
                    marker,
                    color=colour,
                    mfc=filled,
                    ms=6,
                    ls="",
                    label=label,
                )
        ax.axvspan(-3.75, -2.1, color="0.9", lw=0, zorder=0)
        ax.set_title(
            f"{t_c:.0f} °C, [001], Fo91.5, 1 atm (points within {NEAR_C:.0f} °C, moved)", loc="left"
        )
        ax.set_xlabel("log10 fO2 (Pa); shaded: round robin and paper fO2")
    axes[0].set_ylabel("log10 D[001] (m²/s)")
    seen: dict = {}
    for ax in axes:
        for h, lab in zip(*ax.get_legend_handles_labels(), strict=True):
            seen.setdefault(lab, h)
    fig.legend(
        list(seen.values()),
        list(seen.keys()),
        loc="outside lower center",
        ncol=5,
        fontsize=7,
        frameon=False,
    )
    steps = pd.read_csv(HERE / "results" / "oeser2026_fo2_steps.csv")
    s001 = steps[(steps["axis"] == "[001]") & (steps["T_C_low_fO2"].between(1149, 1157))]
    lo = s001[s001["log10_fO2_from"] < -6]["slope"].iloc[0]
    hi = s001[(s001["log10_fO2_from"] == -5.0)]["slope"].iloc[0]
    fig.suptitle(
        f"Oeser's [001] at 1150 °C: slope {lo:.2f} from 1e-7.1 to 1e-5 Pa, {hi:.2f} from 1e-5 to "
        f"1e-3.7 Pa; the law's exponent {base_fit['beta'][2]:.3f} without Oeser, "
        f"{with_fit['beta'][2]:.3f} with",
        fontsize=10,
    )
    fig.savefig(OUT / "f1_fo2_series.png", dpi=200)
    plt.close(fig)


def own_exponents() -> list[tuple[str, float, float]]:
    every = FO.load()
    dohmen = every[(every["source"] == "dohmen2007_part1") & (every["log10_fO2_Pa"] > -10)]
    petry = every[every["source"] == "petry2004"]
    oes = V.oeser_dp()
    return [
        ("Dohmen 2007, own (T, axis removed)", *FO._within_study(dohmen, True)),
        ("Petry 2004, own (T removed)", *FO._within_study(petry, False)),
        (
            "Oeser 2026 [001], own (T removed)",
            *FO._within_study(oes[oes["axis"] == "[001]"], False),
        ),
        ("Oeser 2026 all axes, own (T, axis removed)", *FO._within_study(oes, True)),
    ]


def f2(fit_map: dict) -> pd.DataFrame:
    ks20 = {k.split(",")[0]: c for k, c in F._conditions().items()}["KS20"]
    rows = []
    for name, (fit, n) in fit_map.items():
        mu, se = F.predict(V.FORM, fit["beta"], fit["cov"], ks20)
        rows.append(
            {
                "label": f"law: {name} (n {n})",
                "m": fit["beta"][2],
                "m_se": math.sqrt(fit["cov"][2, 2]),
                "ks20": mu - F._dc07_preset(ks20),
                "ks20_se": se,
            }
        )
    rows += [
        {"label": lab, "m": m, "m_se": se, "ks20": np.nan, "ks20_se": np.nan}
        for lab, m, se in own_exponents()
    ]
    tab = pd.DataFrame(rows)
    fig, (a, b) = plt.subplots(
        1, 2, figsize=(11, 4.2), layout="constrained", sharey=True, width_ratios=[1.3, 1]
    )
    y = np.arange(len(tab))
    colours = ["#0072B2" if lab.startswith("law") else "0.35" for lab in tab["label"]]
    for yi, r, c in zip(y, tab.itertuples(), colours, strict=True):
        a.errorbar(r.m, yi, xerr=r.m_se, fmt="o", color=c, ms=5, capsize=0)
        if not math.isnan(r.ks20):
            b.errorbar(r.ks20, yi, xerr=r.ks20_se, fmt="o", color=c, ms=5, capsize=0)
    a.axvline(1 / 6, color="0.15", ls=":", lw=1, label="DC07 1/6")
    a.legend(loc="lower right", fontsize=7, frameon=False)
    a.set_yticks(y, tab["label"])
    a.invert_yaxis()
    a.set_xlabel("fO2 exponent, d log D / d log fO2 (±1 sigma)")
    b.axvline(0, color="0.15", ls=":", lw=1)
    b.set_xlabel("D[001] at KS20 minus DC07 (log), law 1 sigma")
    m0, m1, k0, k1 = tab.loc[0, "m"], tab.loc[1, "m"], tab.loc[0, "ks20"], tab.loc[1, "ks20"]
    own = tab.set_index("label").loc["Oeser 2026 [001], own (T removed)"]
    fig.suptitle(
        f"Adding Oeser's [001] moves the exponent {m0:.3f} → {m1:.3f} and KS20 {k0:+.2f} → "
        f"{k1:+.2f}; Oeser's own [001] exponent is {own['m']:.2f} ± {own['m_se']:.2f}",
        fontsize=10,
    )
    fig.savefig(OUT / "f2_exponents.png", dpi=200)
    plt.close(fig)
    return tab


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fit_map = fits()
    f1(fit_map)
    tab = f2(fit_map)
    tab.to_csv(HERE / "results" / "verified" / "oeser_fo2_options.csv", index=False)
    print(tab.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
