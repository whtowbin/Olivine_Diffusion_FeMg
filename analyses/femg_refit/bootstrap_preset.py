"""Bootstrap of the femg_refit2026 preset: does its covariance match a resampling spread?

Runs are resampled with replacement within each study (Holzapfel's Ni-NiO-capsule and
Au-capsule runs as separate strata), whole runs at a time (the rows of one run move
together), and each resample is refitted by the preset's own method
(`fit_verified.fit_preset`: REML with a random offset per study and each run's input
errors). The four studies are kept in every draw: with four, resampling studies would
say little. So the bootstrap measures the spread given these laboratories, and should
be no wider than the law's covariance, which also carries the between-study variance
through the random offsets.

For comparison, the same resampling with ordinary least squares (no study terms, no
input errors), as fit_models.fit_bootstrap did for the full compilation.

Run: uv run python analyses/femg_refit/bootstrap_preset.py
Writes results/verified/bootstrap_draws.csv (every draw) and bootstrap.csv (summary:
the full-data fit, the law's 1 sigma, the bootstrap sd and its 2.5, 16, 84 and 97.5
percentiles for log10 D[001] at RR1200, RR1102 and the paper conditions minus DC07, the fO2
exponent, E and V).
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import fit_models as F
import fit_verified as V
import numpy as np
import pandas as pd

warnings.simplefilter("ignore")
HERE = Path(__file__).parent
RESULTS = HERE / "results" / "verified"
LN10 = math.log(10)
DRAWS = 1000
SEED = 20261015


def strata(df: pd.DataFrame) -> dict[str, dict[str, np.ndarray]]:
    """{stratum: {run: row indices}}; a stratum is a study, Holzapfel split by capsule."""
    label = df["source"].where(~V._is_au(df), "holzapfel2007_au").to_numpy()
    runs = df["run_id"].to_numpy()
    out: dict[str, dict[str, np.ndarray]] = {}
    for s in np.unique(label):
        in_s = np.flatnonzero(label == s)
        out[s] = {r: in_s[runs[in_s] == r] for r in np.unique(runs[in_s])}
    return out


def resample(groups: dict[str, dict[str, np.ndarray]], rng: np.random.Generator) -> np.ndarray:
    rows = []
    for runs in groups.values():
        names = list(runs)
        rows += [runs[r] for r in rng.choice(names, size=len(names), replace=True)]
    return np.concatenate(rows)


def summary_values(beta: np.ndarray) -> dict[str, float]:
    out = {}
    for label, c in V.CONDITIONS.items():
        mu, _ = F.predict(V.FORM, beta, np.eye(len(beta)), c)
        out[f"{label}_vs_dc07"] = mu - F._dc07_preset(c)
    out |= {"m_fO2": beta[2], "E_kJ_mol": beta[1] / 1e3, "V_cm3_mol": beta[4] * 1e6}
    return out


def ols(df: pd.DataFrame) -> np.ndarray:
    return F.fit_ls(V.FORM, df, weighted=False)["beta"]


def run() -> pd.DataFrame:
    df = V.preset_data()
    groups = strata(df)
    rng = np.random.default_rng(SEED)
    rows, failed = [], 0
    for i in range(DRAWS):
        sub = df.iloc[resample(groups, rng)].reset_index(drop=True)
        try:
            fit = V.fit_preset(sub, tau_range=False)
        except (RuntimeError, np.linalg.LinAlgError):
            failed += 1
            continue
        row = {"draw": i, "method": "preset method"} | summary_values(fit["beta"])
        rows.append(row | {"tau_log10": fit["tau"] / LN10})
        rows.append({"draw": i, "method": "OLS"} | summary_values(ols(sub)))
    print(f"{DRAWS} draws, {failed} failed to fit")
    draws = pd.DataFrame(rows)
    draws.to_csv(RESULTS / "bootstrap_draws.csv", index=False)
    return draws


def summarise(draws: pd.DataFrame) -> pd.DataFrame:
    df = V.preset_data()
    full = {"preset method": V.fit_preset(df), "OLS": F.fit_ls(V.FORM, df, weighted=False)}
    rows = []
    for method, fit in full.items():
        point = summary_values(fit["beta"])
        law = {}
        for label, c in V.CONDITIONS.items():
            law[f"{label}_vs_dc07"] = F.predict(V.FORM, fit["beta"], fit["cov"], c)[1]
        se = np.sqrt(np.diag(fit["cov"]))
        law |= {"m_fO2": se[2], "E_kJ_mol": se[1] / 1e3, "V_cm3_mol": se[4] * 1e6}
        sample = draws[draws["method"] == method]
        for quantity, value in point.items():
            x = sample[quantity].to_numpy()
            q = np.percentile(x, [2.5, 16, 84, 97.5])
            rows.append(
                {
                    "method": method,
                    "quantity": quantity,
                    "full_fit": value,
                    "law_sigma": law[quantity],
                    "boot_mean": x.mean(),
                    "boot_sd": x.std(ddof=1),
                    "boot_p2.5": q[0],
                    "boot_p16": q[1],
                    "boot_p84": q[2],
                    "boot_p97.5": q[3],
                    "n_draws": len(x),
                }
            )
    tab = pd.DataFrame(rows)
    tab.to_csv(RESULTS / "bootstrap.csv", index=False)
    return tab


def main() -> None:
    tab = summarise(run())
    pd.set_option("display.width", 220)
    print(tab.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
