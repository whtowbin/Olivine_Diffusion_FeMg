"""Melt-contact experiments and the Fe-Mg refit: Shea et al. (2023), Spandler and O'Neill (2010).

What happens to the between-study term tau, the fO2 exponent and the level at our
run conditions when the melt-contact experiments get a term of their own instead of
being part of the study spread. Reuses analyses/femg_refit/fit_models.py unchanged.

Variants (each for the dc07_free and mutch_x6 forms, random study offsets by REML):

- baseline:             the 135 TaMED points, as the femg_refit2026 preset
- no_spandler:          without Spandler and O'Neill (2010)
- melt_fixed_spandler:  a fixed log offset for melt contact (Spandler)
- shea_random:          Shea et al. added as a sixth random study
- melt_fixed_both:      a fixed melt offset shared by Spandler and Shea

Shea et al. rows enter only where their Figs 2-4 tie a profile to an axis; see
data/shea2023_table1.csv. Predictions for the melt variants are for solid contact
(melt offset zero).

Run: uv run python analyses/melt_contact/melt_sensitivity.py
Writes results/variants.csv, results/shea_residuals.csv, results/element_offsets.csv.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
REFIT = HERE.parent / "femg_refit"
sys.path.insert(0, str(REFIT))

import fit_models as fm  # noqa: E402  (needs REFIT on the path)

from forster import DIFFUSIVITY_PRESETS, Conditions, engine  # noqa: E402

LN10, R = math.log(10), 8.3145
RESULTS = HERE / "results"
SHEA = HERE / "data" / "shea2023_table1.csv"
SPANDLER = "spandler_oneill2010"
AXES = ("[100]", "[010]", "[001]")
QFM_DELTA = -1.0  # Shea et al. (2023): all runs at QFM-1, 1 atm
SPECIES_D = {"Fe-Mg": "D_Fo", "Ni": "D_Ni", "Mn": "D_Mn"}
SPECIES_T = {"Fe-Mg": "t_model_FeMg_h", "Ni": "t_model_Ni_h", "Mn": "t_model_Mn_h"}
# Laws the residuals are taken against: Shea et al. used DC07 (Fe-Mg), Chakraborty
# 2010 (Mn, the DC07 form) and Petry et al. 2004 / Holzapfel et al. 2007 (Ni).
ELEMENT_LAW = {"Fe-Mg": "dohmen_chakraborty2007", "Ni": "petry2004_ni", "Mn": "girona_costa2013_mn"}


def load_shea() -> pd.DataFrame:
    df = pd.read_csv(SHEA)
    df["T_K"] = df["T_C"] + 273.15
    df["P_Pa"] = 1e5
    df["log10_fO2_Pa"] = [math.log10(engine.fo2buffer(t, 1e5, QFM_DELTA, "QFM")) for t in df["T_K"]]
    df["t_true_h"] = df["dwell_end_h"]
    return df


def shea_femg_rows(shea: pd.DataFrame) -> pd.DataFrame:
    """Shea et al. Fe-Mg rows on a principal axis, in the schema of fit_models.tamed()."""
    sub = shea[shea["axis"].isin(AXES) & shea["D_Fo"].notna()]
    return pd.DataFrame(
        {
            "source": "shea2023",
            "run": sub["run"] + "/" + sub["table_row"].astype(str),
            "T_K": sub["T_K"],
            "P_Pa": sub["P_Pa"],
            "log10_fO2_Pa": sub["log10_fO2_Pa"],
            "X_Fo": sub["X_Fo"],
            "axis": sub["axis"],
            "log10_D_m2s": np.log10(sub["D_Fo"]),
            "ln_D": np.log(sub["D_Fo"]),
            "run_id": "shea2023:" + sub["run"],
        }
    ).reset_index(drop=True)


def law_log10_d(preset: str, row: pd.Series) -> float:
    """log10 D of a forster preset on the row's axis at the row's conditions."""
    c = Conditions(T_C=row["T_K"] - 273.15, P_Pa=row["P_Pa"], fO2_Pa=10 ** row["log10_fO2_Pa"])
    d = DIFFUSIVITY_PRESETS[preset].axis_diffusivities(c, float(row["X_Fo"]))
    return float(np.log10(d[AXES.index(row["axis"])]))


def shea_residuals(shea: pd.DataFrame) -> pd.DataFrame:
    """Each pinned Shea et al. profile against DC07 and the refit, two ways.

    resid_dc07_shea is Shea et al.'s own: log10(modelled time / true time), which is
    log D measured - log D predicted with their conditions. resid_dc07_ours uses our
    transcribed D and assumed conditions (QFM-1, the profile's mid Fo); the gap
    between them checks the transcription and the assumptions. The refit residual
    adds (DC07 - refit) at the row's conditions, which hardly depends on Fo."""
    out = []
    for _, row in shea[shea["X_Fo"].notna()].iterrows():
        for species, dcol in SPECIES_D.items():
            t_model = row[SPECIES_T[species]]
            if pd.isna(t_model) or pd.isna(row[dcol]):
                continue
            rec = {
                "table_row": row["table_row"],
                "run": row["run"],
                "species": species,
                "T_C": row["T_C"],
                "axis": row["axis"],
                "angle_to_c_deg": row["angle_to_c_deg"],
                "X_Fo": row["X_Fo"],
                "log10_fO2_Pa": row["log10_fO2_Pa"],
                "log10_D": math.log10(row[dcol]),
                "resid_law_shea": math.log10(t_model / row["t_true_h"]),
                "law": ELEMENT_LAW[species],
            }
            if row["axis"] in AXES:
                law = law_log10_d(ELEMENT_LAW[species], row)
                rec["resid_law_ours"] = rec["log10_D"] - law
                if species == "Fe-Mg":
                    shift = law - law_log10_d("femg_refit2026", row)
                    rec["resid_refit_ours"] = rec["resid_law_ours"] + shift
                    rec["resid_refit_shea"] = rec["resid_law_shea"] + shift
            out.append(rec)
    df = pd.DataFrame(out)
    df["gap_ours_minus_shea"] = df["resid_law_ours"] - df["resid_law_shea"]
    return df


def melt_form(base: fm.Form, df: pd.DataFrame, sources: tuple[str, ...]) -> fm.Form:
    """base plus a log10 offset for the rows of `sources` (melt contact)."""
    melt = df["source"].isin(sources).to_numpy(float)
    return fm.Form(
        base.name + "+melt",
        (*base.params, "melt_log10"),
        lambda v: np.column_stack([base.design(v), LN10 * melt]),
        base.offset,
    )


def _exponent_and_e(form_name: str, beta: np.ndarray) -> tuple[float, float]:
    """fO2 exponent (d log D / d log fO2) and E (J/mol) of a fit."""
    if form_name == "dc07_free":
        return float(beta[2]), float(beta[1])
    return float(beta[1]), float(-R * beta[3])  # mutch_x6: b, q


def variant_row(form_name: str, variant: str, fit: dict, n_base: int) -> dict:
    """One results row: tau, exponent, E, melt term, study effects, predictions."""
    beta, cov = fit["beta"], fit["cov"]
    melt = len(beta) > n_base
    m, e = _exponent_and_e(form_name, beta)
    rec = {
        "form": form_name,
        "variant": variant,
        "n": fit["n"],
        "tau_log10": fit["tau"] / LN10,
        "tau_lo95": fit["tau_range"][0] / LN10,
        "tau_hi95": fit["tau_range"][1] / LN10,
        "sd_within_log10": fit["sd"] / LN10,
        "fO2_exponent": m,
        "E_kJ_mol": e / 1e3,
        "melt_log10": float(beta[-1]) if melt else np.nan,
        "melt_se": math.sqrt(cov[-1, -1]) if melt else np.nan,
    }
    for study, effect in fit["study_effects"].items():
        rec[f"effect_{study}"] = effect / LN10
    form = fm.FORMS[form_name]
    for name, v in fm._conditions().items():
        mu, se = fm.predict(form, beta[:n_base], cov[:n_base, :n_base], v)
        key = name.split(",")[0]
        rec[f"{key}_vs_dc07"] = mu - fm._dc07_preset(v)
        rec[f"{key}_se_law"] = se
        rec[f"{key}_se_new_lab"] = math.hypot(se, rec["tau_log10"])
    return rec


def check_baseline(fit: dict) -> None:
    """The dc07_free baseline must reproduce the committed refit (the preset's source)."""
    coef = pd.read_csv(REFIT / "results" / "coefficients.csv")
    ref = coef[(coef["form"] == "dc07_free") & (coef["method"] == "random study")]
    ref = ref.set_index("param")["value"]
    got = dict(zip(fm.FORMS["dc07_free"].params, fit["beta"], strict=True))
    got["tau_log10"] = fit["tau"] / LN10
    for param, value in got.items():
        if not math.isclose(value, ref[param], rel_tol=1e-6):
            raise AssertionError(f"baseline {param} = {value} against {ref[param]} committed")


def variant_fits(form: fm.Form, base: pd.DataFrame, both: pd.DataFrame) -> dict[str, dict]:
    """The five variants of `form` (module docstring). ``both`` is ``base`` plus Shea's rows."""
    return {
        "baseline": fm.fit_random(form, base),
        "no_spandler": fm.fit_random(form, base[base["source"] != SPANDLER]),
        "melt_fixed_spandler": fm.fit_random(melt_form(form, base, (SPANDLER,)), base),
        "shea_random": fm.fit_random(form, both),
        "melt_fixed_both": fm.fit_random(melt_form(form, both, (SPANDLER, "shea2023")), both),
    }


def variants(shea_rows: pd.DataFrame) -> pd.DataFrame:
    base = fm.tamed()
    both = pd.concat([base, shea_rows], ignore_index=True)
    rows = []
    for form_name in ("dc07_free", "mutch_x6"):
        form = fm.FORMS[form_name]
        n_base = len(form.params)
        fits = variant_fits(form, base, both)
        if form_name == "dc07_free":
            check_baseline(fits["baseline"])
        rows += [variant_row(form_name, k, fit, n_base) for k, fit in fits.items()]
    return pd.DataFrame(rows)


def element_offsets(shea_resid: pd.DataFrame) -> pd.DataFrame:
    """Mean residual by study and species against the law of that species.

    Asks whether a study's offset is shared by Fe-Mg, Ni and Mn, as one vacancy
    population would make it. Holzapfel et al. give no fO2: NNO, as fit_ols.load."""
    recs = []
    for name in ("petry2004", "holzapfel2007", SPANDLER):
        df = pd.read_csv(REFIT / "data" / f"{name}.csv")
        df["T_K"] = df["T_C"] + 273.15
        missing = df["log10_fO2_Pa"].isna()
        df.loc[missing, "log10_fO2_Pa"] = [
            math.log10(engine.fo2buffer(t, p, 0.0, "NNO"))
            for t, p in zip(df.loc[missing, "T_K"], df.loc[missing, "P_Pa"], strict=True)
        ]
        for _, row in df[df["species"].isin(ELEMENT_LAW)].iterrows():
            law = law_log10_d(ELEMENT_LAW[row["species"]], row)
            recs.append(
                {"source": name, "species": row["species"], "resid": row["log10_D_m2s"] - law}
            )
    for _, row in shea_resid.iterrows():
        recs.append(
            {"source": "shea2023", "species": row["species"], "resid": row["resid_law_shea"]}
        )
    out = pd.DataFrame(recs).groupby(["source", "species"])["resid"].agg(["mean", "std", "count"])
    return out.reset_index()


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    shea = load_shea()
    resid = shea_residuals(shea)
    table = variants(shea_femg_rows(shea))
    offsets = element_offsets(resid)
    resid.to_csv(RESULTS / "shea_residuals.csv", index=False)
    table.to_csv(RESULTS / "variants.csv", index=False)
    offsets.to_csv(RESULTS / "element_offsets.csv", index=False)
    pd.set_option("display.width", 200)
    print(resid.round(2).to_string(index=False))
    cols = ["form", "variant", "n", "tau_log10", "fO2_exponent", "E_kJ_mol", "melt_log10"]
    cols += ["melt_se", "RR1200_vs_dc07", "RR1200_se_law", "RR1200_se_new_lab", "Paper_vs_dc07"]
    print(table[cols].round(3).to_string(index=False))
    print(table.filter(like="effect_").assign(v=table["variant"]).round(2).to_string(index=False))
    print(offsets.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
