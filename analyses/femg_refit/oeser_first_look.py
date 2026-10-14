"""First look at Oeser et al. (2026) against DC07 and femg_refit2026, and their fO2 steps.

Not a fit. Two tables:

- results/oeser2026_vs_laws.csv: each profile's log D_Fe-Mg minus DC07 and minus the
  femg_refit2026 preset at its own run's T, fO2 and axis (X_Fo 0.915, 1 atm).
- results/oeser2026_fo2_steps.csv: the fO2 slope between two runs on the same axis at
  (nearly) the same temperature, d log D / d log fO2, from the axis means, with the
  temperature difference removed by the refit's activation energy (209 kJ/mol). These
  are local slopes, not an exponent fitted across the range.

Run: uv run python analyses/femg_refit/oeser_first_look.py
"""

from __future__ import annotations

import math
from itertools import combinations
from pathlib import Path

import pandas as pd

from forster import DIFFUSIVITY_PRESETS, Conditions

HERE = Path(__file__).parent
DATA = HERE / "data" / "oeser2026.csv"
RESULTS = HERE / "results"
AXIS = {"[100]": 0, "[010]": 1, "[001]": 2}
E_REFIT = 209065.7  # J/mol, femg_refit2026 (for the small T differences between runs)
R = 8.3145
SAME_T = 10  # C: runs this close count as one temperature


def law(name: str, row) -> float:
    c = Conditions(T_C=row.T_C, P_Pa=1e5, fO2_Pa=10**row.log10_fO2_Pa)
    return math.log10(DIFFUSIVITY_PRESETS[name].axis_diffusivities(c, row.X_Fo)[AXIS[row.axis]])


def steps(df: pd.DataFrame) -> pd.DataFrame:
    """Local fO2 slopes between run pairs on one axis at nearly one temperature."""
    means = df.groupby(["run", "T_C", "log10_fO2_Pa", "axis"], as_index=False)
    means = means.agg(log10_D=("log10_D_FeMg", "mean"), n=("log10_D_FeMg", "size"))
    rows = []
    for axis, sub in means.groupby("axis"):
        for a, b in combinations(sub.itertuples(), 2):
            if abs(a.T_C - b.T_C) > SAME_T or a.log10_fO2_Pa == b.log10_fO2_Pa:
                continue
            lo, hi = (a, b) if a.log10_fO2_Pa < b.log10_fO2_Pa else (b, a)
            # Move hi to lo's temperature before taking the slope.
            t_shift = (
                -E_REFIT / (math.log(10) * R) * (1 / (lo.T_C + 273.15) - 1 / (hi.T_C + 273.15))
            )
            d_log_d = (hi.log10_D + t_shift) - lo.log10_D
            rows.append(
                {
                    "axis": axis,
                    "T_C_low_fO2": lo.T_C,
                    "T_C_high_fO2": hi.T_C,
                    "log10_fO2_from": lo.log10_fO2_Pa,
                    "log10_fO2_to": hi.log10_fO2_Pa,
                    "runs": f"{lo.run} -> {hi.run}",
                    "slope": d_log_d / (hi.log10_fO2_Pa - lo.log10_fO2_Pa),
                    "n_profiles": f"{lo.n} + {hi.n}",
                }
            )
    return pd.DataFrame(rows).sort_values(["T_C_low_fO2", "log10_fO2_from", "axis"])


def main() -> None:
    df = pd.read_csv(DATA)
    df["minus_dc07"] = [r.log10_D_FeMg - law("dohmen_chakraborty2007", r) for r in df.itertuples()]
    df["minus_refit"] = [r.log10_D_FeMg - law("femg_refit2026", r) for r in df.itertuples()]
    keep = ["table", "run", "T_C", "log10_fO2_Pa", "axis", "profile", "log10_D_FeMg"]
    keep += ["log10_D_FeMg_italic", "technique_Fo", "minus_dc07", "minus_refit"]
    df[keep].to_csv(RESULTS / "oeser2026_vs_laws.csv", index=False)
    st = steps(df)
    st.to_csv(RESULTS / "oeser2026_fo2_steps.csv", index=False)
    pd.set_option("display.width", 160)
    table = df.groupby(["T_C", "log10_fO2_Pa", "axis"])["minus_refit"].mean().unstack("axis")
    print("log D minus femg_refit2026, axis means:\n" + table.round(2).to_string())
    for label, m in (("below 1200 C", df["T_C"] < 1200), ("1200 C and above", df["T_C"] >= 1200)):
        sub = df[m]
        print(
            f"{label}: minus DC07 {sub['minus_dc07'].mean():+.2f}, minus refit "
            f"{sub['minus_refit'].mean():+.2f} (n {len(sub)})"
        )
    print("\nlocal fO2 slopes:\n" + st.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
