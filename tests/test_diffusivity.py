"""Diffusivity equations: the DIPRA default, the presets, and user-defined tables.

The default must give exactly the numbers the engine always gave (the round robin
differential test relies on it); every preset is checked against an independent
transcription of its source, and the general formula against the engine.
"""

import itertools
import math
from dataclasses import fields, replace

import numpy as np
import pandas as pd
import pytest

from forster import (
    DIFFUSIVITY_PRESETS,
    ArrheniusDiffusivity,
    Conditions,
    Grid,
    Orientation,
    RunDefaults,
    dipra,
    engine,
    fit_profile,
    fit_run,
    read_diffusivities,
    read_profiles,
    read_runs,
    summary_row,
    write_template,
)
from forster.diffusivity import DOHMEN_CHAKRABORTY_2007, diffusivity_table, from_table

R = 8.3145
CONDITIONS = [
    Conditions(T_C, P, fO2)
    for T_C, P, fO2 in itertools.product((900.0, 1102.0, 1250.0), (1e5, 4.2e7, 1e9), (1e-6, 1e-3))
]
ORIENTATIONS = [Orientation(90, 90, 0), Orientation(88, 123, 33), Orientation(32.5, 71.4, 64.4)]
FO = np.linspace(0.6, 0.95, 15)


def generic(d):
    """The same equation as a plain ArrheniusDiffusivity (general formula, no engine shortcut)."""
    return ArrheniusDiffusivity(**{f.name: getattr(d, f.name) for f in fields(d)})


def along(d_axes, o):
    a, b, g = np.deg2rad([o.alpha_deg, o.beta_deg, o.gamma_deg])
    return d_axes[0] * np.cos(a) ** 2 + d_axes[1] * np.cos(b) ** 2 + d_axes[2] * np.cos(g) ** 2


def cases():
    return itertools.product(CONDITIONS, ORIENTATIONS)


def test_default_is_dipra_table_1_through_the_engine_exactly():
    assert RunDefaults().diffusivity == "dohmen_chakraborty2007"
    for c, o in cases():
        D, dD = DOHMEN_CHAKRABORTY_2007.functions(c, o)
        old = engine.D_Fo(c.T_K, c.P_Pa, c.fO2_Pa, o.alpha_deg, o.beta_deg, o.gamma_deg)
        old_D, old_dD = dipra.D_Fo_and_derivative(
            c.T_K, c.P_Pa, c.fO2_Pa, o.alpha_deg, o.beta_deg, o.gamma_deg, 201_000
        )
        assert np.array_equal(D(FO), old(FO))
        assert np.array_equal(old_D(FO), old(FO))
        assert np.array_equal(dD(FO), old_dD(FO))


def test_general_formula_reproduces_dipra_table_1():
    """DIPRA Table 1 (Girona and Costa 2013) written out independently."""
    d = generic(DOHMEN_CHAKRABORTY_2007)
    for c, o in cases():
        D001 = (
            10**-9.21
            * (c.fO2_Pa / 1e-7) ** (1 / 6)
            * 10 ** (3 * (0.9 - FO))
            * np.exp(-(201_000 + 7e-6 * (c.P_Pa - 1e5)) / (R * c.T_K))
        )
        np.testing.assert_allclose(
            d.functions(c, o)[0](FO), along((D001 / 6, D001 / 6, D001), o), rtol=1e-12
        )


def test_dipra_ni_mn_ca_presets_match_the_engine_and_table_1():
    ni, mn, ca = (DIFFUSIVITY_PRESETS[f"girona_costa2013_{e}"] for e in ("ni", "mn", "ca"))
    for c, o in cases():
        args = (c.T_K, c.P_Pa, c.fO2_Pa, o.alpha_deg, o.beta_deg, o.gamma_deg)
        np.testing.assert_allclose(ni.functions(c, o)[0](FO), engine.D_Ni(*args)(FO), rtol=1e-12)
        # DIPRA Table 1 gives Mn the Fe-Mg form with E_Mn = 201 kJ/mol.
        np.testing.assert_allclose(mn.functions(c, o)[0](FO), engine.D_Fo(*args)(FO), rtol=1e-12)
        ca_engine = engine.D_Func_Ca(c.T_K, c.fO2_Pa, o.alpha_deg, o.beta_deg, o.gamma_deg)
        np.testing.assert_allclose(ca.functions(c, o)[0](FO), ca_engine, rtol=1e-12)


@pytest.mark.parametrize(
    ("name", "coefficients"),
    [
        ("mutch2019_tamed", (-6.76, 0.224, -7.18, -26700, -5.21e-10, -1.03e-7)),
        ("mutch2019_global", (-7.86, 0.187, -7.21, -26600, -4.15e-10, -1.54e-7)),
    ],
)
def test_mutch_presets_match_the_published_regression(name, coefficients):
    """Mutch et al. (2019) Supplementary Table 1: ln D[001] = a + b ln fO2(bar) + c X_Fo
    + q/T + j P + h P/T, with [100] and [010] six times slower."""
    a, b, c_fo, q, j, h = coefficients
    d = DIFFUSIVITY_PRESETS[name]
    for c, o in cases():
        T, P = c.T_K, c.P_Pa
        D001 = np.exp(a + b * np.log(c.fO2_Pa / 1e5) + c_fo * FO + q / T + j * P + h * P / T)
        np.testing.assert_allclose(
            d.functions(c, o)[0](FO), along((D001 / 6, D001 / 6, D001), o), rtol=1e-12
        )


@pytest.mark.parametrize("name", list(DIFFUSIVITY_PRESETS))
def test_derivative_is_the_derivative_of_D(name):
    d = DIFFUSIVITY_PRESETS[name]
    D, dD = d.functions(CONDITIONS[4], ORIENTATIONS[1])
    h = 1e-6
    np.testing.assert_allclose(dD(FO), (D(FO + h) - D(FO - h)) / (2 * h), rtol=1e-6, atol=1e-40)


def test_shifting_the_activation_energy():
    c, o = CONDITIONS[4], ORIENTATIONS[1]
    shifted = DOHMEN_CHAKRABORTY_2007.shift_activation_energy(8000)
    assert shifted.activation_energy_J_mol == (209_000.0,) * 3
    expected = engine.D_Fo(
        c.T_K, c.P_Pa, c.fO2_Pa, o.alpha_deg, o.beta_deg, o.gamma_deg, EFo=209_000
    )
    assert np.array_equal(shifted.functions(c, o)[0](FO), expected(FO))
    ratio = shifted.functions(c, o)[0](FO) / DOHMEN_CHAKRABORTY_2007.functions(c, o)[0](FO)
    np.testing.assert_allclose(ratio, math.exp(-8000 / (R * c.T_K)), rtol=1e-12)
    # Per-axis energies leave the engine path; the general formula takes over.
    per_axis = replace(DOHMEN_CHAKRABORTY_2007, activation_energy_J_mol=(201e3, 201e3, 205e3))
    np.testing.assert_allclose(
        per_axis.functions(c, o)[0](FO), generic(per_axis).functions(c, o)[0](FO), rtol=1e-15
    )


def test_presets_round_trip_through_a_csv_table(tmp_path):
    path = tmp_path / "diffusivities.csv"
    diffusivity_table(DIFFUSIVITY_PRESETS.values()).to_csv(path, index=False)
    read = read_diffusivities(path)
    assert read.keys() == DIFFUSIVITY_PRESETS.keys()
    for name, d in read.items():
        assert d is DIFFUSIVITY_PRESETS[name]  # a preset row stands for the preset itself


def test_a_custom_row_is_read_with_its_values(tmp_path):
    row = diffusivity_table([DOHMEN_CHAKRABORTY_2007]).assign(name="faster", log10_D0_001=-8.0)
    (custom,) = from_table(row).values()
    assert custom.name == "faster"
    assert custom.log10_D0_m2_s[2] == -8.0
    assert type(custom) is ArrheniusDiffusivity


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"log10_D0_001": -9.0}, "is a preset name"),
        ({"element": "Cr"}, "element 'Cr'"),
        ({"fO2_ref_Pa": float("nan")}, "missing fO2_ref_Pa"),
        ({"fO2_ref_Pa": 0.0}, "must be positive"),
    ],
)
def test_bad_table_rows_are_errors(change, message):
    table = diffusivity_table([DOHMEN_CHAKRABORTY_2007]).assign(**change)
    with pytest.raises(ValueError, match=message):
        from_table(table)


def test_duplicate_names_and_missing_columns_are_errors():
    table = diffusivity_table([DOHMEN_CHAKRABORTY_2007]).assign(name="mine")
    with pytest.raises(ValueError, match="defined twice"):
        from_table(pd.concat([table, table]))
    with pytest.raises(ValueError, match="missing column"):
        from_table(table.drop(columns="fo_ref"))


@pytest.fixture(scope="module")
def example(tmp_path_factory):
    out = tmp_path_factory.mktemp("template")
    write_template(out)
    return out


def test_template_writes_the_presets(example):
    assert read_diffusivities(example / "diffusivities.csv").keys() == DIFFUSIVITY_PRESETS.keys()


def test_fit_rejects_a_diffusivity_for_another_element_or_an_unknown_name(example):
    (run,) = read_runs(example / "parameters.csv")
    profile = read_profiles(example / "profiles.csv")[run.profile_name]
    args = (profile, run.conditions, run.orientation, run.initial, Grid(1.0, 600.0, 40))
    with pytest.raises(ValueError, match="is for Ni"):
        fit_profile(*args, diffusivity="girona_costa2013_ni")
    with pytest.raises(ValueError, match="unknown diffusivity 'dipra'"):
        fit_profile(*args, diffusivity="dipra")


def test_time_scales_inversely_with_the_diffusivity(example):
    """Halving D doubles the fitted time (the profile depends on D t)."""
    (run,) = read_runs(example / "parameters.csv")
    profiles = read_profiles(example / "profiles.csv")
    base = fit_run(run, profiles)
    half = generic(DOHMEN_CHAKRABORTY_2007)
    half = replace(
        half, name="half", log10_D0_m2_s=tuple(v - math.log10(2) for v in half.log10_D0_m2_s)
    )
    grid = replace(run.grid, max_time_days=2.5 * base.time_days)
    slow = fit_run(replace(run, diffusivity=half, grid=grid), profiles)
    assert slow.time_days / base.time_days == pytest.approx(2.0, rel=0.01)
    assert summary_row(slow)["diffusivity"] == "half"


def test_diffusivity_column_names_presets_and_table_rows(tmp_path, example):
    params = pd.read_csv(example / "parameters.csv")
    params = pd.concat([params.assign(run="a"), params.assign(run="b", diffusivity="mine")])
    params = pd.concat([params, params.iloc[:1].assign(run="c", diffusivity="mutch2019_tamed")])
    params.loc[params.run == "a", "diffusivity"] = None
    path = tmp_path / "parameters.csv"
    params.to_csv(path, index=False)
    mine = diffusivity_table([DOHMEN_CHAKRABORTY_2007]).assign(name="mine", log10_D0_001=-9.0)
    runs = {r.name: r for r in read_runs(path, diffusivities=from_table(mine))}
    assert runs["a"].diffusivity is DOHMEN_CHAKRABORTY_2007
    assert runs["b"].diffusivity.name == "mine"
    assert runs["c"].diffusivity is DIFFUSIVITY_PRESETS["mutch2019_tamed"]
    with pytest.raises(ValueError, match="unknown diffusivity 'mine'"):
        read_runs(path)
