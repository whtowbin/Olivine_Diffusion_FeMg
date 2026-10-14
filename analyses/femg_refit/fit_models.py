"""Fe-Mg diffusivity refits: four law forms, five ways of fitting them.

Data: the TaMED set (fO2 > 1e-10 Pa) of `fit_ols.load`, each D on its own axis.
Forms (ln D on the measured axis):

- mutch_x6:    a + b ln fO2(bar) + c X_Fo + q/T + j P + h P/T - ln 6 [off-axis]
- mutch_aniso: the same with the [100] and [010] offsets fitted
- dc07_fixed:  Dohmen and Chakraborty (2007): log D0 + (1/6) log(fO2/1e-7 Pa)
               + 3 (0.9 - X_Fo) - (E + (P - 1e5) 7e-6)/(ln10 R T), x6 anisotropy;
               only log D0 and E fitted
- dc07_free:   that form with log D0, E, the fO2 exponent m, the composition
               coefficient n and both anisotropy offsets free (V fixed at 7e-6)
- mutch_fT:    mutch_aniso plus g ln fO2 / T, an fO2 exponent that changes with T
               (b + g/T); the term an F test finds missing (p = 0.008)

Methods: ols (unweighted, as Mutch et al. 2019), wls (weighted by the y errors),
odr (errors in T, fO2, X_Fo and P as well; odrpack, after Henry's odr_bootstrap),
bootstrap (resampling whole runs, OLS each), loso (leave one study out, OLS).

Run: uv run --with odrpack python analyses/femg_refit/fit_models.py
Writes results/ (coefficients, covariances, predictions at the round robin and
paper conditions).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from fit_ols import load

LN10, LN6, R = math.log(10), math.log(6), 8.3145
V_DC07 = 7e-6  # m3/mol, Dohmen and Chakraborty (2007)
RESULTS = Path(__file__).parent / "results"
BOOTSTRAP_DRAWS = 2000
SEED = 20261013
RANDOM_GRID = 400  # tau^2 / sigma^2 values searched by fit_random

# Measurement errors (1 sigma) for ODR and WLS. "stated" = given in the paper;
# "assumed" = our judgement where the paper gives none (see README).
ERRORS = pd.DataFrame(
    [
        # source,             log D, T (K), log fO2, X_Fo,  P (fraction)
        (
            "chakraborty1997",
            None,
            3.0,
            0.1,
            0.01,
            0.0,
        ),  # log D: per-row % (stated); fO2 +-0.1 (stated)
        ("dohmen2007_part1", 0.2, 2.0, 0.1, 0.005, 0.0),  # +-0.2 log, +-2 C, +-0.1 log fO2 (stated)
        ("petry2004", 0.3, 3.0, 0.1, 0.01, 0.0),  # factor-2 reproducibility (stated, for Ni)
        ("holzapfel2007", 0.3, 25.0, 0.5, 0.01, 0.05),  # 0.3 log (stated); fO2 at NNO assumed
        (
            "spandler_oneill2010",
            0.1,
            3.0,
            0.1,
            0.002,
            0.0,
        ),  # SD 0.04 (stated); 0.1 for the angle fit
    ],
    columns=["source", "log10_D", "T_K", "log10_fO2", "X_Fo", "P_frac"],
).set_index("source")


def tamed() -> pd.DataFrame:
    df = load()
    df = df[df["log10_fO2_Pa"] > -10].reset_index(drop=True)
    df["run_id"] = df["source"] + ":" + df["run"].astype(str)
    # 1 atm and 1 bar rows on one value, so pressure terms see only the high-P data.
    df.loc[df["P_Pa"] < 2e5, "P_Pa"] = 1e5
    df["ln_D"] = df["log10_D_m2s"] * LN10  # on the measured axis
    return df


@dataclass(frozen=True)
class Vars:
    """Predictors as arrays (one value per observation)."""

    log_fo2_pa: np.ndarray
    x_fo: np.ndarray
    T: np.ndarray
    P: np.ndarray
    i100: np.ndarray
    i010: np.ndarray

    @classmethod
    def of(cls, df: pd.DataFrame) -> Vars:
        return cls(
            df["log10_fO2_Pa"].to_numpy(float),
            df["X_Fo"].to_numpy(float),
            df["T_K"].to_numpy(float),
            df["P_Pa"].to_numpy(float),
            (df["axis"] == "[100]").to_numpy(float),
            (df["axis"] == "[010]").to_numpy(float),
        )

    @classmethod
    def from_matrix(cls, x: np.ndarray) -> Vars:
        return cls(*x)

    def matrix(self) -> np.ndarray:
        return np.vstack([self.log_fo2_pa, self.x_fo, self.T, self.P, self.i100, self.i010])


@dataclass(frozen=True)
class Form:
    """ln D = offset(v) + design(v) @ beta."""

    name: str
    params: tuple[str, ...]
    design: Callable[[Vars], np.ndarray]
    offset: Callable[[Vars], np.ndarray]


def _ln_fo2_bar(v: Vars) -> np.ndarray:
    return (v.log_fo2_pa - 5) * LN10


def _mutch_design(v: Vars, aniso: bool) -> np.ndarray:
    cols = [np.ones_like(v.T), _ln_fo2_bar(v), v.x_fo, 1 / v.T, v.P, v.P / v.T]
    if aniso:
        cols += [v.i100, v.i010]
    return np.column_stack(cols)


def _off_axis(v: Vars) -> np.ndarray:
    return -LN6 * (v.i100 + v.i010)


def _dc07_fixed_offset(v: Vars) -> np.ndarray:
    return (
        LN10 * ((v.log_fo2_pa + 7) / 6 + 3 * (0.9 - v.x_fo))
        - (v.P - 1e5) * V_DC07 / (R * v.T)
        + _off_axis(v)
    )


FORMS = {
    "mutch_x6": Form(
        "mutch_x6", ("a", "b", "c", "q", "j", "h"), lambda v: _mutch_design(v, False), _off_axis
    ),
    "mutch_aniso": Form(
        "mutch_aniso",
        ("a", "b", "c", "q", "j", "h", "k100", "k010"),
        lambda v: _mutch_design(v, True),
        lambda v: np.zeros_like(v.T),
    ),
    "mutch_fT": Form(
        "mutch_fT",
        ("a", "b", "c", "q", "j", "h", "k100", "k010", "g_fT"),
        lambda v: np.column_stack([_mutch_design(v, True), _ln_fo2_bar(v) / v.T]),
        lambda v: np.zeros_like(v.T),
    ),
    "dc07_fixed": Form(
        "dc07_fixed",
        ("log10_D0", "E_J_mol"),
        lambda v: np.column_stack([np.full_like(v.T, LN10), -1 / (R * v.T)]),
        _dc07_fixed_offset,
    ),
    "dc07_free": Form(
        "dc07_free",
        ("log10_D0", "E_J_mol", "m_fO2", "n_X", "k100", "k010"),
        lambda v: np.column_stack(
            [
                np.full_like(v.T, LN10),
                -1 / (R * v.T),
                LN10 * (v.log_fo2_pa + 7),
                LN10 * (0.9 - v.x_fo),
                v.i100,
                v.i010,
            ]
        ),
        lambda v: -(v.P - 1e5) * V_DC07 / (R * v.T),
    ),
}


def _lsq(A: np.ndarray, y: np.ndarray, w: np.ndarray | None = None):
    """(Weighted) least squares on scaled columns; returns beta, covariance, residual sd."""
    w = np.ones(len(y)) if w is None else w
    sw = np.sqrt(w)
    scale = np.abs(A).max(axis=0)
    As = A / scale * sw[:, None]
    beta_s, *_ = np.linalg.lstsq(As, y * sw, rcond=None)
    beta = beta_s / scale
    resid = (y - A @ beta) * sw
    dof = len(y) - A.shape[1]
    s2 = float(resid @ resid / dof)
    cov = s2 * np.linalg.inv(As.T @ As) / np.outer(scale, scale)
    return beta, cov, math.sqrt(s2)


def _y_sigma_ln(df: pd.DataFrame) -> np.ndarray:
    sigma = df["source"].map(ERRORS["log10_D"]).astype(float) * LN10
    pct = df["D_error_as_printed"].astype(str).str.rstrip("%").str.extract(r"^([\d.]+)$")[0]
    chak = (df["source"] == "chakraborty1997") & pct.notna()
    # Printed analytical % error, at least 10% (the author: true errors 7-98%).
    sigma[chak] = np.log1p(np.maximum(pct[chak].astype(float), 10) / 100)
    return sigma.to_numpy(float)


def fit_ls(form: Form, df: pd.DataFrame, weighted: bool) -> dict:
    v = Vars.of(df)
    y = df["ln_D"].to_numpy() - form.offset(v)
    w = 1 / _y_sigma_ln(df) ** 2 if weighted else None
    beta, cov, sd = _lsq(form.design(v), y, w)
    return {"beta": beta, "cov": cov, "sd": sd, "n": len(df)}


def fit_odr(form: Form, df: pd.DataFrame) -> dict:
    """ODR with errors in log fO2, X_Fo, T and P as well as ln D (odrpack)."""
    from odrpack import odr_fit

    v = Vars.of(df)
    start = fit_ls(form, df, weighted=True)["beta"]
    scale = np.abs(start) + 1e-300  # parameters O(1) for the solver

    def f(x: np.ndarray, b: np.ndarray) -> np.ndarray:
        vv = Vars.from_matrix(x)
        return form.offset(vv) + form.design(vv) @ (b * scale)

    src = df["source"]
    sx = np.vstack(
        [
            src.map(ERRORS["log10_fO2"]).to_numpy(float),
            src.map(ERRORS["X_Fo"]).to_numpy(float),
            src.map(ERRORS["T_K"]).to_numpy(float),
            np.maximum(src.map(ERRORS["P_frac"]).to_numpy(float) * v.P, 1.0),
            np.full(len(df), 1e-9),  # axis indicators: exact
            np.full(len(df), 1e-9),
        ]
    )
    sol = odr_fit(
        f,
        v.matrix(),
        df["ln_D"].to_numpy(),
        start / scale,
        weight_x=1 / sx**2,
        weight_y=1 / _y_sigma_ln(df) ** 2,
        maxit=200,
    )
    beta = np.asarray(sol.beta) * scale
    cov = np.asarray(sol.cov_beta) * np.outer(scale, scale) * float(sol.res_var)
    return {"beta": beta, "cov": cov, "sd": math.sqrt(float(sol.res_var)), "n": len(df)}


def fit_bootstrap(form: Form, df: pd.DataFrame) -> dict:
    """OLS on resamples of whole runs (rows of one run move together)."""
    rng = np.random.default_rng(SEED)
    runs = df["run_id"].unique()
    groups = {r: np.flatnonzero(df["run_id"].to_numpy() == r) for r in runs}
    draws = []
    for _ in range(BOOTSTRAP_DRAWS):
        pick = rng.choice(runs, size=len(runs), replace=True)
        rows = np.concatenate([groups[r] for r in pick])
        sub = df.iloc[rows]
        if not _identifiable(form, sub):
            continue
        draws.append(fit_ls(form, sub, weighted=False)["beta"])
    samples = np.array(draws)
    base = fit_ls(form, df, weighted=False)
    return {
        "beta": base["beta"],
        "cov": np.cov(samples, rowvar=False),
        "sd": base["sd"],
        "n": len(df),
        "samples": samples,
    }


def _reml(A: np.ndarray, y: np.ndarray, Z: np.ndarray, lam: float):
    """GLS with covariance s^2 (I + lam Z Z') on scaled columns, and its REML log
    likelihood (s^2 profiled out). Returns (log lik, beta, s^2, covariance)."""
    n, p = A.shape
    scale = np.abs(A).max(axis=0)
    L = np.linalg.cholesky(np.eye(n) + lam * Z @ Z.T)
    Aw = np.linalg.solve(L, A / scale)
    yw = np.linalg.solve(L, y)
    beta_s, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
    r = yw - Aw @ beta_s
    s2 = float(r @ r) / (n - p)
    info = Aw.T @ Aw
    loglik = -0.5 * (
        2 * np.log(np.diag(L)).sum() + np.linalg.slogdet(info)[1] + (n - p) * math.log(s2)
    )
    cov = s2 * np.linalg.inv(info) / np.outer(scale, scale)
    return loglik, beta_s / scale, s2, cov


def fit_random(form: Form, df: pd.DataFrame) -> dict:
    """Least squares with a random offset per study, u ~ N(0, tau^2), by REML.

    Points of one study share their offset, so they count for less than as many
    independent points. The covariance is that of the law averaged over labs; a
    prediction for one new lab adds tau^2. tau is in ln units; its 95% range is from
    the profile likelihood (five studies, so wide)."""
    from scipy import stats

    v = Vars.of(df)
    A = form.design(v)
    y = df["ln_D"].to_numpy() - form.offset(v)
    studies = pd.get_dummies(df["source"])
    Z = studies.to_numpy(float)
    lams = np.concatenate([[0.0], np.logspace(-4, 3, RANDOM_GRID)])
    fits = [_reml(A, y, Z, lam) for lam in lams]
    loglik = np.array([f[0] for f in fits])
    tau = np.sqrt(lams * np.array([f[2] for f in fits]))
    k = int(loglik.argmax())
    inside = tau[loglik >= loglik[k] - stats.chi2.ppf(0.95, 1) / 2]
    _, beta, s2, cov = fits[k]
    V = np.eye(len(y)) + lams[k] * Z @ Z.T
    effects = lams[k] * Z.T @ np.linalg.solve(V, y - A @ beta)  # shrunken (BLUP)
    return {
        "beta": beta,
        "cov": cov,
        "sd": math.sqrt(s2),
        "n": len(df),
        "tau": float(tau[k]),
        "tau_range": (float(inside.min()), float(inside.max())),
        "study_effects": dict(zip(studies.columns, effects, strict=True)),
    }


def _identifiable(form: Form, sub: pd.DataFrame) -> bool:
    """Full rank, and pressure terms only with high-pressure data present."""
    if "j" in form.params and sub["P_Pa"].max() < 1e8:
        return False
    A = form.design(Vars.of(sub))
    return bool(np.linalg.matrix_rank(A / np.abs(A).max(axis=0)) == A.shape[1])


def fit_loso(form: Form, df: pd.DataFrame) -> dict[str, dict]:
    """OLS without each study in turn; skipped where the rest cannot identify the form."""
    out = {}
    for study in df["source"].unique():
        sub = df[df["source"] != study]
        if not _identifiable(form, sub):
            continue
        out[study] = fit_ls(form, sub, weighted=False)
    return out


CONDITIONS = {
    # T_C, P_Pa, log10 fO2 (Pa), X_Fo: round robin KS20 (1200 C) and SH63 (1102 C),
    # and the Towbin and Plank (2026) xenoliths (1250 C, QFM+0.3).
    "KS20, 1200 C": (1200.0, 4.2e7, math.log10(0.00138), 0.85),
    "SH63, 1102 C": (1102.0, None, None, 0.70),
    "Paper, 1250 C": (1250.0, 5e8, math.log10(8.3e-3), 0.90),
}


def _conditions() -> dict[str, Vars]:
    from forster import read_runs

    runs = {
        r.name: r
        for r in read_runs(
            Path(__file__).parents[2]
            / "Diffusion Round Robin"
            / "Round_Robin_Olivine_Profile_Diffusion_Modeling_Parameters_DB.xlsx"
        )
    }
    sh63 = runs["SH63olv084"].conditions
    out = {}
    for name, (t, p, f, x) in CONDITIONS.items():
        if p is None:
            p, f = sh63.P_Pa, math.log10(sh63.fO2_Pa)
        out[name] = Vars(*(np.array([val]) for val in (f, x, t + 273.15, p, 0.0, 0.0)))
    return out


def predict(form: Form, beta: np.ndarray, cov: np.ndarray, v: Vars) -> tuple[float, float]:
    """log10 D[001] and its 1-sigma from the covariance."""
    g = form.design(v)[0]
    ln_d = float(form.offset(v)[0] + g @ beta)
    return ln_d / LN10, float(math.sqrt(max(g @ cov @ g, 0.0))) / LN10


def _dc07_preset(v: Vars) -> float:
    from forster import DIFFUSIVITY_PRESETS, Conditions

    c = Conditions(
        T_C=float(v.T[0]) - 273.15, P_Pa=float(v.P[0]), fO2_Pa=10 ** float(v.log_fo2_pa[0])
    )
    return float(
        np.log10(
            DIFFUSIVITY_PRESETS["dohmen_chakraborty2007"].axis_diffusivities(c, float(v.x_fo[0]))[2]
        )
    )


def _volumes(form: Form, method: str, fit: dict) -> list[dict]:
    """Activation volume (cm3/mol) of the Mutch forms, V = -R (j T + h), at two T."""
    if "j" not in form.params:
        return []
    rows = []
    ij, ih = form.params.index("j"), form.params.index("h")
    for t_c in (1250.0, 1450.0):
        g = np.zeros(len(form.params))
        g[ij], g[ih] = -R * (t_c + 273.15) * 1e6, -R * 1e6
        value = float(fit["beta"] @ g)
        if "samples" in fit:
            se = float(np.std(fit["samples"] @ g, ddof=1))
        else:
            se = float(math.sqrt(g @ fit["cov"] @ g))
        rows.append(
            {
                "form": form.name,
                "method": method,
                "n": fit["n"],
                "param": f"V_{t_c:.0f}C_cm3_mol",
                "value": value,
                "se": se,
            }
        )
    return rows


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    df = tamed()
    conditions = _conditions()
    coef_rows, pred_rows = [], []
    for form in FORMS.values():
        fits = {
            "ols": fit_ls(form, df, weighted=False),
            "wls": fit_ls(form, df, weighted=True),
            "odr": fit_odr(form, df),
            "bootstrap (runs)": fit_bootstrap(form, df),
            "random study": fit_random(form, df),
        }
        for study, fit in fit_loso(form, df).items():
            fits[f"without {study}"] = fit
        for method, fit in fits.items():
            coef_rows += _volumes(form, method, fit)
            if "tau" in fit:
                lo, hi = fit["tau_range"]
                for p, val in (
                    ("tau_log10", fit["tau"]),
                    ("tau_log10_lo95", lo),
                    ("tau_log10_hi95", hi),
                ):
                    coef_rows.append(
                        {
                            "form": form.name,
                            "method": method,
                            "n": fit["n"],
                            "param": p,
                            "value": val / LN10,
                            "se": np.nan,
                        }
                    )
            se = np.sqrt(np.diag(fit["cov"]))
            for p, b, s in zip(form.params, fit["beta"], se, strict=True):
                coef_rows.append(
                    {
                        "form": form.name,
                        "method": method,
                        "n": fit["n"],
                        "param": p,
                        "value": b,
                        "se": s,
                    }
                )
            pd.DataFrame(fit["cov"], index=form.params, columns=form.params).to_csv(
                RESULTS / f"cov_{form.name}_{method.split(' ')[0]}.csv"
            ) if not method.startswith("without") else None
            for cname, v in conditions.items():
                logd, se_logd = predict(form, fit["beta"], fit["cov"], v)
                if "samples" in fit:
                    se_logd = float(
                        np.std([predict(form, s, fit["cov"], v)[0] for s in fit["samples"]], ddof=1)
                    )
                pred_rows.append(
                    {
                        "form": form.name,
                        "method": method,
                        "condition": cname,
                        "log10_D001": logd,
                        "se": se_logd,
                        # One new lab: the law's spread plus a study offset (random study only).
                        "se_new_lab": math.hypot(se_logd, fit["tau"] / LN10)
                        if "tau" in fit
                        else np.nan,
                        "minus_dc07_preset": logd - _dc07_preset(v),
                    }
                )
    coef = pd.DataFrame(coef_rows)
    pred = pd.DataFrame(pred_rows)
    coef.to_csv(RESULTS / "coefficients.csv", index=False)
    pred.to_csv(RESULTS / "predictions.csv", index=False)
    pd.set_option("display.width", 200)
    print(
        coef[~coef.method.str.startswith("without")]
        .pivot_table(index=["form", "param"], columns="method", values="value", sort=False)
        .to_string(float_format=lambda x: f"{x:.4g}")
    )
    print(
        coef[~coef.method.str.startswith("without")]
        .pivot_table(index=["form", "param"], columns="method", values="se", sort=False)
        .to_string(float_format=lambda x: f"{x:.3g}")
    )
    print(
        pred.pivot_table(
            index=["form", "method"], columns="condition", values="minus_dc07_preset", sort=False
        ).to_string(float_format=lambda x: f"{x:+.3f}")
    )
    print(
        pred.pivot_table(
            index=["form", "method"], columns="condition", values="se", sort=False
        ).to_string(float_format=lambda x: f"{x:.3f}")
    )


if __name__ == "__main__":
    main()
