"""Reproduce the Fe-Mg regressions of Mutch et al. (2019) by ordinary least squares.

ln D[001] = a + b ln fO2(bar) + c X_Fo + q/T + j P + h P/T   (P in Pa, T in K)

[100] and [010] data are put on [001] by a factor of 6, as Mutch et al. state.
Targets: their Supplementary Table 1 (coefficients) and Table 2 (covariance), as
printed (3 significant figures). Run: uv run python analyses/femg_refit/fit_ols.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from forster import engine

DATA = Path(__file__).parent / "data"
NAMES = ["a", "b", "c", "q", "j", "h"]

# Mutch et al. (2019) Supplementary Tables 1 and 2, as printed.
MUTCH = {
    "Global": (
        [-7.86, 0.187, -7.21, -26600, -4.15e-10, -1.54e-07],
        [
            [4.97e-01, 3.63e-03, -1.32e-01, -3.78e02, -2.77e-11, 2.69e-08],
            [3.63e-03, 4.31e-04, 1.08e-03, 1.02e01, -6.41e-13, -1.99e-10],
            [-1.32e-01, 1.08e-03, 1.49e-01, 5.10e01, -1.46e-13, -4.71e-09],
            [-3.78e02, 1.02e01, 5.10e01, 8.40e05, 1.33e-08, -3.94e-05],
            [-2.77e-11, -6.41e-13, -1.46e-13, 1.33e-08, 2.33e-19, -3.91e-16],
            [2.69e-08, -1.99e-10, -4.71e-09, -3.94e-05, -3.91e-16, 6.61e-13],
        ],
    ),
    "TaMED": (
        [-6.76, 0.224, -7.18, -26700, -5.21e-10, -1.03e-07],
        [
            [7.20e-01, 1.36e-02, -1.37e-01, -3.17e02, -5.11e-11, 3.57e-08],
            [1.36e-02, 8.25e-04, 2.25e-04, 1.18e01, -1.61e-12, 2.07e-10],
            [-1.37e-01, 2.25e-04, 1.34e-01, 4.45e01, 1.76e-12, -5.05e-09],
            [-3.17e02, 1.18e01, 4.45e01, 8.20e05, 8.12e-09, -3.61e-05],
            [-5.11e-11, -1.61e-12, 1.76e-12, 8.12e-09, 2.08e-19, -3.46e-16],
            [3.57e-08, 2.07e-10, -5.05e-09, -3.61e-05, -3.46e-16, 5.83e-13],
        ],
    ),
}


def _fe_mg(frame: pd.DataFrame) -> pd.DataFrame:
    if "species" in frame:
        frame = frame[frame["species"] == "Fe-Mg"]
    return frame.copy()


def load(include_spandler: bool = True) -> pd.DataFrame:
    """Every Fe-Mg diffusion coefficient on a common basis, one row each."""
    parts = []
    for name in ("chakraborty1997_table3", "dohmen2007_part1_table3", "petry2004", "holzapfel2007"):
        parts.append(_fe_mg(pd.read_csv(DATA / f"{name}.csv")))
    spandler = DATA / "spandler_oneill2010.csv"
    if include_spandler and spandler.exists():
        parts.append(_fe_mg(pd.read_csv(spandler)))
    df = pd.concat(parts, ignore_index=True)
    df["T_K"] = df["T_C"] + 273.15
    # Holzapfel et al. give no fO2: NNO at each run's P and T (engine.fo2buffer, Pa).
    missing = df["log10_fO2_Pa"].isna()
    df.loc[missing, "log10_fO2_Pa"] = [
        math.log10(engine.fo2buffer(t, p, 0.0, "NNO"))
        for t, p in zip(df.loc[missing, "T_K"], df.loc[missing, "P_Pa"], strict=True)
    ]
    df["ln_fO2_bar"] = (df["log10_fO2_Pa"] - 5) * math.log(10)
    df["ln_D001"] = df["log10_D_m2s"] * math.log(10) + np.where(
        df["axis"].isin(["[100]", "[010]"]), math.log(6), 0.0
    )
    return df


def design(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack(
        [
            np.ones(len(df)),
            df["ln_fO2_bar"],
            df["X_Fo"],
            1 / df["T_K"],
            df["P_Pa"],
            df["P_Pa"] / df["T_K"],
        ]
    )


def ols(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, float]:
    """Coefficients, their covariance s^2 (X'X)^-1 (as R's vcov(lm)), residual sd."""
    X, y = design(df), df["ln_D001"].to_numpy()
    # The columns span 13 orders of magnitude (P ~ 1e10 Pa, 1/T ~ 1e-3 /K): solve with
    # each column scaled to unit size, or lstsq treats the 1/T direction as zero.
    scale = np.abs(X).max(axis=0)
    Xs = X / scale
    beta_s, *_ = np.linalg.lstsq(Xs, y, rcond=None)
    beta = beta_s / scale
    resid = y - X @ beta
    s2 = float(resid @ resid / (len(y) - X.shape[1]))
    cov = s2 * np.linalg.inv(Xs.T @ Xs) / np.outer(scale, scale)
    return beta, cov, math.sqrt(s2)


def compare(label: str, df: pd.DataFrame, target: str) -> float:
    beta, cov, sd = ols(df)
    ref_beta, ref_cov = (np.array(v) for v in MUTCH[target])
    ref_sd = np.sqrt(np.diag(ref_cov))
    se = np.sqrt(np.diag(cov))
    # Coefficient distance in units of Mutch's standard errors.
    z = (beta - ref_beta) / ref_sd
    corr = cov / np.outer(se, se)
    ref_corr = ref_cov / np.outer(ref_sd, ref_sd)
    print(f"\n{label}: n = {len(df)} ({df['source'].value_counts().to_dict()}), sd {sd:.3f} ln")
    print(f"  {'':3}{'ours':>12}{'Mutch':>12}{'(ours-M)/seM':>14}{'se ours/M':>11}")
    for n, b, r, zz, s, rs in zip(NAMES, beta, ref_beta, z, se, ref_sd, strict=True):
        print(f"  {n:3}{b:12.4g}{r:12.4g}{zz:14.2f}{s / rs:11.2f}")
    print(f"  max |correlation difference| {np.abs(corr - ref_corr).max():.2f}")
    return float(np.abs(z).max())


def main() -> None:
    df = load()
    print(df.groupby("source").agg(n=("ln_D001", "size"), T=("T_C", "min"), T2=("T_C", "max")))
    eleven = df["notes"].fillna("").str.contains("11 GPa runs anomalously", case=False)
    variants = {
        "Global, all": df,
        "Global, without the 11 GPa runs": df[~eleven],
    }
    for label, sub in variants.items():
        compare(label, sub, "Global")
    tamed = {
        "TaMED, fO2 > 1e-10 Pa": df[df["log10_fO2_Pa"] > -10],
        "TaMED, fO2 >= 1e-10 Pa": df[df["log10_fO2_Pa"] >= -10],
        "TaMED, fO2 >= 1e-10 Pa, without 11 GPa": df[(df["log10_fO2_Pa"] >= -10) & ~eleven],
    }
    for label, sub in tamed.items():
        compare(label, sub, "TaMED")


if __name__ == "__main__":
    main()
