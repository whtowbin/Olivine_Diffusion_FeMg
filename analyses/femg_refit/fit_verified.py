"""The verified compilation, the fit behind the femg_refit2026 preset, and variants.

Verified: every value the fit uses is stated in the source paper's tables or text.
`docs/references/data-to-source.md` lists what is not, and what would confirm it.

- Chakraborty (1997), Dohmen et al. (2007) Part I (fO2 > 1e-10 Pa) and Petry et al.
  (2004), as `fit_models.tamed()`. The [001] axis of Chakraborty's and Petry's runs is a
  statement for the whole paper, not given run by run (accepted, 2026-10-14; the per-run
  orientations are on the to-source list).
- Holzapfel et al. (2007): the Ni-NiO-capsule runs only, buffered at Ni-NiO. In the
  Au-capsule runs fO2 was not buffered; the authors estimate it as close to or slightly
  above Ni-NiO.
- Not in the preset:
  - Oeser et al. (2026). Their values are tabulated, but their [010] is about 2x slower
    than [001], against 6x in Dohmen et al. (2007) and in this form. Following DC07,
    DIPRA and DFENS, the preset keeps 6x (decided 2026-10-14); Oeser's anisotropy is
    reported as an unconfirmed scenario (`oeser_dp()`; unconfirmed-data.md).
  - Spandler and O'Neill (2010): melt contact (see melt-contact-diffusion.md).
  - Shea et al. (2023), Jurewicz and Watson (1988) and Meissner (2000): they lack values
    or details only their sources can supply.

The form is the Dohmen and Chakraborty (2007) form: log D0, E, the fO2 exponent and the
composition coefficient fitted; D[100] = D[010] = D[001]/6 and V = 7 cm3/mol fixed.

Variants: the Holzapfel Au-capsule runs added at fO2 = Ni-NiO with a prior sd of 0 (as
before), 1, 2 or 3 log units on fO2, carried as extra variance (ln10 m sigma)^2 on
their ln D and refitted until m settles. That likelihood is checked against
fit_models.fit_random with no extra variance before use.

Run: uv run python analyses/femg_refit/fit_verified.py
Writes results/verified/{coefficients,predictions,variants}.csv.
"""

from __future__ import annotations

import math
from pathlib import Path

import fit_extended as X
import fit_models as F
import numpy as np
import pandas as pd
from scipy import optimize

HERE = Path(__file__).parent
OUT = HERE / "results" / "verified"
LN10 = math.log(10)
LN6 = math.log(6)
_FREE = F.FORMS["dc07_free"]
# The preset's form: dc07_free with both anisotropy factors fixed at 6 (DC07, as DIPRA and
# DFENS). Not added to fit_models.FORMS, whose every form fit_models.main fits.
FORM = F.Form(
    "dc07_x6",
    _FREE.params[:4],
    lambda v: _FREE.design(v)[:, :4],
    lambda v: _FREE.offset(v) - LN6 * (v.i100 + v.i010),
)
FO2_PRIOR_SD = (0.0, 1.0, 2.0, 3.0)  # log units, for the Au-capsule runs at Ni-NiO
CONDITIONS = {k.split(",")[0]: v for k, v in F._conditions().items()}


def holzapfel_au() -> set[str]:
    """Runs of Holzapfel et al. (2007) made in Au capsules (fO2 not buffered)."""
    h = pd.read_csv(HERE / "data" / "holzapfel2007.csv")
    return set(h.loc[h["notes"].str.contains("capsule Au"), "run"].astype(str))


def _is_au(df: pd.DataFrame) -> np.ndarray:
    return (
        (df["source"] == "holzapfel2007") & df["run"].astype(str).isin(holzapfel_au())
    ).to_numpy()


def verified() -> pd.DataFrame:
    """The data the femg_refit2026 preset is fitted to."""
    t = F.tamed()
    keep = (t["source"] != "spandler_oneill2010").to_numpy() & ~_is_au(t)
    return t[keep].reset_index(drop=True)


def oeser_dp() -> pd.DataFrame:
    """Oeser et al. (2026) depth-profile runs: verified values, out of the preset (6x)."""
    return X.oeser(True)


def holzapfel_au_rows() -> pd.DataFrame:
    t = F.tamed()
    return t[_is_au(t)]


def fit_extra(form: F.Form, df: pd.DataFrame, extra_ln_var: np.ndarray) -> dict:
    """REML fit with a random offset per study and a known extra variance per point:
    V = s^2 I + tau^2 Z Z' + diag(extra). The variances are found by maximising the
    REML likelihood; with extra = 0 this is fit_models.fit_random's model."""
    v = F.Vars.of(df)
    A = form.design(v)
    scale = np.abs(A).max(axis=0)
    As = A / scale
    y = df["ln_D"].to_numpy() - form.offset(v)
    Z = pd.get_dummies(df["source"]).to_numpy(float)
    n = As.shape[0]

    def parts(log_s2: float, log_t2: float):
        V = math.exp(log_s2) * np.eye(n) + math.exp(log_t2) * Z @ Z.T + np.diag(extra_ln_var)
        L = np.linalg.cholesky(V)
        Aw = np.linalg.solve(L, As)
        yw = np.linalg.solve(L, y)
        info = Aw.T @ Aw
        beta_s = np.linalg.solve(info, Aw.T @ yw)
        r = yw - Aw @ beta_s
        nll = 0.5 * (2 * np.log(np.diag(L)).sum() + np.linalg.slogdet(info)[1] + r @ r)
        return nll, beta_s, info, V

    res = optimize.minimize(
        lambda q: parts(*q)[0],
        x0=np.log([0.25, 0.3]),
        method="Nelder-Mead",
        options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 4000},
    )
    _, beta_s, info, V = parts(*res.x)
    beta = beta_s / scale
    cov = np.linalg.inv(info) / np.outer(scale, scale)
    tau2 = math.exp(res.x[1])
    effects = tau2 * Z.T @ np.linalg.solve(V, y - A @ beta)
    return {
        "beta": beta,
        "cov": cov,
        "sd": math.exp(res.x[0] / 2),
        "tau": math.sqrt(tau2),
        "study_effects": dict(zip(pd.get_dummies(df["source"]).columns, effects, strict=True)),
    }


def check_extra() -> None:
    """With no extra variance, fit_extra must reproduce fit_random (its grid's precision)."""
    df = verified()
    a, b = fit_extra(FORM, df, np.zeros(len(df))), F.fit_random(FORM, df)
    for name, x, y in (("tau", a["tau"], b["tau"]), ("m", a["beta"][2], b["beta"][2])):
        print(
            f"control {name}: fit_extra {x / (LN10 if name == 'tau' else 1):.4f}, "
            f"fit_random {y / (LN10 if name == 'tau' else 1):.4f}"
        )
    assert abs(a["tau"] - b["tau"]) / LN10 < 0.005, "tau does not reproduce fit_random"
    assert abs(a["beta"][2] - b["beta"][2]) < 0.002, "m does not reproduce fit_random"
    assert abs(a["beta"][1] - b["beta"][1]) < 500, "E does not reproduce fit_random"


def with_au(prior_sd: float) -> tuple[dict, int]:
    """The verified data plus the Au runs at Ni-NiO, fO2 prior sd ``prior_sd`` (log)."""
    df = pd.concat([verified(), holzapfel_au_rows()], ignore_index=True)
    au = _is_au(df)
    m = 0.22
    for _ in range(20):  # the extra variance depends on m: iterate until it settles
        fit = fit_extra(FORM, df, np.where(au, (LN10 * m * prior_sd) ** 2, 0.0))
        if abs(fit["beta"][2] - m) < 1e-5:
            break
        m = float(fit["beta"][2])
    return fit, len(df)


def row(name: str, fit: dict, n: int, studies: int, form: F.Form = FORM) -> dict:
    b = fit["beta"]
    se = np.sqrt(np.diag(fit["cov"]))
    out = {"variant": name, "n_points": n, "n_studies": studies}
    out |= {"tau_log10": fit["tau"] / LN10, "sd_within_log10": fit["sd"] / LN10}
    out |= {"log10_D0": b[0], "E_kJ_mol": b[1] / 1e3, "m_fO2": b[2], "n_X": b[3]}
    out |= {"m_fO2_se": se[2], "E_se_kJ_mol": se[1] / 1e3}
    for label, c in CONDITIONS.items():
        mu, s = F.predict(form, b, fit["cov"], c)
        out[f"{label}_vs_dc07"] = mu - F._dc07_preset(c)
        out[f"{label}_se_law"] = s
        out[f"{label}_se_new_lab"] = math.hypot(s, fit["tau"] / LN10)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    check_extra()
    df = verified()
    preset = F.fit_random(FORM, df)
    studies = df["source"].nunique()
    se = np.sqrt(np.diag(preset["cov"]))
    coef = [
        {"param": p, "value": v, "se": s}
        for p, v, s in zip(FORM.params, preset["beta"], se, strict=True)
    ]
    coef += [
        {"param": "tau_log10", "value": preset["tau"] / LN10},
        {"param": "tau_log10_lo95", "value": preset["tau_range"][0] / LN10},
        {"param": "tau_log10_hi95", "value": preset["tau_range"][1] / LN10},
        {"param": "n_points", "value": len(df)},
        {"param": "n_studies", "value": studies},
    ]
    pd.DataFrame(coef).to_csv(OUT / "coefficients.csv", index=False)
    # The preset's predictions, in the columns of results/predictions.csv (read by the tests).
    preds = []
    for name, c in F._conditions().items():
        mu, se_law = F.predict(FORM, preset["beta"], preset["cov"], c)
        preds.append(
            {
                "form": FORM.name,
                "method": "random study",
                "condition": name,
                "log10_D001": mu,
                "se": se_law,
                "se_new_lab": math.hypot(se_law, preset["tau"] / LN10),
                "minus_dc07_preset": mu - F._dc07_preset(c),
            }
        )
    pd.DataFrame(preds).to_csv(OUT / "predictions.csv", index=False)
    old = F.tamed()
    rows = [
        row(
            "previous preset (full compilation, Spandler as a study)",
            F.fit_random(_FREE, old),
            len(old),
            old["source"].nunique(),
            _FREE,
        ),
        row("verified (the preset)", preset, len(df), studies),
    ]
    for sd in FO2_PRIOR_SD:
        fit, n = with_au(sd)
        rows.append(row(f"+ Holzapfel Au runs at NNO, fO2 sd {sd:g} log", fit, n, studies))
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "variants.csv", index=False)
    pd.set_option("display.width", 220)
    print(pd.DataFrame(coef).to_string(index=False))
    show = ["variant", "n_points", "tau_log10", "m_fO2", "E_kJ_mol", "KS20_vs_dc07", "KS20_se_law"]
    show += ["KS20_se_new_lab", "SH63_vs_dc07", "Paper_vs_dc07"]
    print(table[show].round(3).to_string(index=False))
    effects = {k: round(v / LN10, 2) for k, v in preset["study_effects"].items()}
    print("study offsets (verified):", effects)


if __name__ == "__main__":
    main()
