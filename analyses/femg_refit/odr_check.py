"""ODR against the preset's method (REML with study offsets plus effective variance).

Both carry the errors in T, fO2, X_Fo and P; ODR minimises the error-scaled orthogonal
distance S(b) = sum r^2 / (sigma_y^2 + sum (d ln D/dx)^2 sigma_x^2) with sigma_y the
stated log D error and no study terms, while the preset's method (fit_verified.fit_preset)
puts the same input variance inside a REML likelihood that also estimates the scatter
within and between studies. On the preset's data (verified plus the Au runs at Ni-NiO,
fO2 sd 2), for a floor on each run's T error of 0 (stated: 2-3 C at 1 atm, 25 C for
Holzapfel), 10 and 20 C:

- REML + effective variance (the preset's method);
- full maximum likelihood with the input variance depending on the coefficients
  (the iterated REML holds it fixed: a check that this does not matter);
- ODR (odrpack), without and with fixed study offsets, and the direct minimiser of S,
  which ODR must equal (a check of the ODR call).

Run: uv run --with odrpack python analyses/femg_refit/odr_check.py
Writes results/verified/odr_check.csv.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import fit_models as F
import fit_verified as V
import numpy as np
import pandas as pd
from odrpack import odr_fit
from scipy import optimize

warnings.simplefilter("ignore")
HERE = Path(__file__).parent
LN10 = math.log(10)
FLOORS = (0.0, 10.0, 20.0)  # C (= K), on each run's T error
KS20 = V.CONDITIONS["KS20"]
# V fixed at 7 cm3/mol: the preset's form when this comparison was made (2026-10-14).
# The comparison is of methods; the form with V fitted is now the preset (fit_verified).
FORM = V.FORM_V7


def _fo2_sd(df: pd.DataFrame) -> np.ndarray:
    return np.where(V._is_au(df), V.PRESET_AU_FO2_SD, np.nan)


def _at_floor(floor: float, fn, *args):
    """Call ``fn`` with fit_verified's T error floor set to ``floor``."""
    keep = V.T_ERROR_FLOOR_K
    V.T_ERROR_FLOOR_K = floor
    try:
        return fn(*args)
    finally:
        V.T_ERROR_FLOOR_K = keep


def odr(df: pd.DataFrame, floor: float, offsets: bool) -> np.ndarray:
    """ODR coefficients (log D0, E, m, n[, study offsets]); weights from the stated errors."""
    v = F.Vars.of(df)
    src = df["source"]
    dummies = pd.get_dummies(src).to_numpy(float)[:, 1:] if offsets else np.empty((len(df), 0))

    def design(vv: F.Vars, d: np.ndarray) -> np.ndarray:
        return np.column_stack([FORM.design(vv), d])

    y = df["ln_D"].to_numpy()
    start = np.linalg.lstsq(design(v, dummies), y - FORM.offset(v), rcond=None)[0]
    scale = np.abs(start) + 1e-300

    def f(x: np.ndarray, b: np.ndarray) -> np.ndarray:
        vv = F.Vars.from_matrix(x[:6])
        return FORM.offset(vv) + design(vv, x[6:].T) @ (b * scale)

    fo2 = np.where(V._is_au(df), V.PRESET_AU_FO2_SD, src.map(F.ERRORS["log10_fO2"]))
    exact = np.full(len(df), 1e-9)  # axis indicators and study dummies
    sx = np.vstack(
        [
            fo2.astype(float),
            src.map(F.ERRORS["X_Fo"]).to_numpy(float),
            np.maximum(src.map(F.ERRORS["T_K"]).to_numpy(float), floor),
            np.maximum(src.map(F.ERRORS["P_frac"]).to_numpy(float) * v.P, 1.0),
            exact,
            exact,
            *([exact] * dummies.shape[1]),
        ]
    )
    sol = odr_fit(
        f,
        np.vstack([v.matrix(), dummies.T]),
        y,
        start / scale,
        weight_x=1 / sx**2,
        weight_y=1 / V.published_sigma_ln(df) ** 2,
        maxit=500,
    )
    return np.asarray(sol.beta) * scale


def min_s(df: pd.DataFrame) -> np.ndarray:
    """Minimise S(b) directly, at fit_verified's current T error floor."""
    v = F.Vars.of(df)
    A, y = FORM.design(v), df["ln_D"].to_numpy() - FORM.offset(v)
    sy2, fo2 = V.published_sigma_ln(df) ** 2, _fo2_sd(df)
    sc = np.abs(np.linalg.lstsq(A, y, rcond=None)[0])

    def s(q: np.ndarray) -> float:
        b = q * sc
        return float(np.sum((y - A @ b) ** 2 / (sy2 + V.input_error_variance(df, b, fo2, FORM))))

    opts = {"xatol": 1e-10, "fatol": 1e-12, "maxiter": 20000}
    return optimize.minimize(s, np.ones(4), method="Nelder-Mead", options=opts).x * sc


def full_ml(df: pd.DataFrame) -> tuple[np.ndarray, float]:
    """ML over (b, s^2, tau^2, r^2) with V(b) = s^2 I + tau^2 ZZ' + r^2 WW' +
    diag(published D variance + input variance(b)): the preset's model, with the input
    variance depending on the coefficients instead of held fixed and iterated."""
    v = F.Vars.of(df)
    A, y = FORM.design(v), df["ln_D"].to_numpy() - FORM.offset(v)
    Z = pd.get_dummies(df["source"]).to_numpy(float)
    W = pd.get_dummies(pd.Series(V._run_ids(df))).to_numpy(float)
    base = V.published_sigma_ln(df) ** 2
    fo2 = _fo2_sd(df)
    start = V.fit_preset(df, FORM)["beta"]
    sc = np.abs(start)

    def nll(q: np.ndarray) -> float:
        b = q[:4] * sc
        var = math.exp(q[4]) * np.eye(len(y)) + math.exp(q[5]) * Z @ Z.T
        var = var + math.exp(q[6]) * W @ W.T + np.diag(base)
        L = np.linalg.cholesky(var + np.diag(V.input_error_variance(df, b, fo2, FORM)))
        r = np.linalg.solve(L, y - A @ b)
        return float(np.log(np.diag(L)).sum() + 0.5 * r @ r)

    q = np.r_[np.ones(4), math.log(0.05), math.log(0.1), math.log(0.1)]
    opts = {"xatol": 1e-9, "fatol": 1e-11, "maxiter": 40000, "maxfev": 40000}
    for _ in range(3):  # restarts, as Nelder-Mead stalls in seven dimensions
        q = optimize.minimize(nll, q, method="Nelder-Mead", options=opts).x
    return q[:4] * sc, math.exp(q[5] / 2) / LN10


def _row(method: str, floor: float, beta: np.ndarray, tau: float = math.nan) -> dict:
    mu, _ = F.predict(FORM, beta[:4], np.eye(4), KS20)
    return {
        "method": method,
        "T_error_floor_C": floor,
        "m_fO2": beta[2],
        "E_kJ_mol": beta[1] / 1e3,
        "n_X": beta[3],
        "KS20_vs_dc07": mu - F._dc07_preset(KS20),
        "tau_log10": tau,
    }


def main() -> pd.DataFrame:
    df = V.preset_data()
    rows = []
    for floor in FLOORS:
        fit = _at_floor(floor, V.fit_preset, df, FORM)
        rows.append(_row("REML + effective variance", floor, fit["beta"], fit["tau"] / LN10))
        b, tau = _at_floor(floor, full_ml, df)
        rows.append(_row("full ML, input variance depending on b", floor, b, tau))
        rows.append(_row("ODR, no study terms", floor, odr(df, floor, False)))
        rows.append(_row("minimum of S (must equal ODR)", floor, _at_floor(floor, min_s, df)))
        row = _row("ODR, fixed study offsets", floor, odr(df, floor, True))
        row["KS20_vs_dc07"] = math.nan  # relative to one reference study: not comparable
        rows.append(row)
    tab = pd.DataFrame(rows)
    tab.to_csv(HERE / "results" / "verified" / "odr_check.csv", index=False)
    pd.set_option("display.width", 200)
    print(tab.round(3).to_string(index=False))
    return tab


if __name__ == "__main__":
    main()
