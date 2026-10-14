"""Diffusivity-law uncertainty: `LawUncertainty`, `log10_sigma`, and the fitted time's
range with it (`FitResult.time_range_days`, `widen_log10`, summary columns).

Plan: ``docs/plans/2026-10-13-law-uncertainty.md``.
"""

import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forster import (
    DIFFUSIVITY_PRESETS,
    Conditions,
    Grid,
    InitialCondition,
    LawUncertainty,
    Orientation,
    Profile,
    fit_profile,
    fit_run,
    read_profiles,
    read_runs,
    summary_row,
    write_template,
)
from forster.diffusivity import LAW_PARAMETERS, ArrheniusDiffusivity
from forster.model import _law_uncertainty
from forster.uncertainty import widen_log10

PRESET = DIFFUSIVITY_PRESETS["femg_refit2026"]
KS20 = Conditions(T_C=1200, P_Pa=4.2e7, fO2_Pa=0.00138)
ALONG_001 = Orientation(90, 90, 0)
OBLIQUE = Orientation(50, 70, 46.5)
PREDICTIONS = Path(__file__).parents[1] / "analyses/femg_refit/results/predictions.csv"


def _perturbed(law: ArrheniusDiffusivity, name: str, eps: float) -> ArrheniusDiffusivity:
    """``law`` with one `LAW_PARAMETERS` entry moved by ``eps``, by its definition."""
    d0, e = list(law.log10_D0_m2_s), law.activation_energy_J_mol
    if name == "log10_D0_001":
        return replace(law, log10_D0_m2_s=tuple(v + eps for v in d0))
    if name == "activation_energy_J_mol":
        return law.shift_activation_energy(eps)
    if name in ("log10_100_vs_001", "log10_010_vs_001"):
        d0[0 if name == "log10_100_vs_001" else 1] += eps
        return replace(law, log10_D0_m2_s=tuple(d0), activation_energy_J_mol=e)
    return replace(law, **{name: getattr(law, name) + eps})


def _log10_d_along(law, conditions, orientation, x_fo) -> float:
    return math.log10(float(law.functions(conditions, orientation)[0](np.array([x_fo]))[0]))


@pytest.mark.parametrize("orientation", [ALONG_001, OBLIQUE])
def test_log10_sigma_is_the_covariance_through_the_laws_own_gradient(orientation):
    """Every parameter, by finite differences of D along the profile, against g' C g."""
    rng = np.random.default_rng(3)
    scale = np.array([0.1, 5e3, 0.02, 0.2, 1e-6, 1e-11, 0.07, 0.08])
    a = rng.normal(size=(8, 8))
    cov = (a @ a.T / 8) * np.outer(scale, scale)
    law = replace(
        PRESET,
        name="all terms",
        activation_volume_m3_mol=6e-6,
        pressure_coefficient_per_Pa=-2e-11,
        uncertainty=LawUncertainty(LAW_PARAMETERS, tuple(map(tuple, cov))),
    )
    conditions, x_fo = Conditions(T_C=1150, P_Pa=6e8, fO2_Pa=3e-4), 0.82
    base = _log10_d_along(law, conditions, orientation, x_fo)
    g = []
    for name, s in zip(LAW_PARAMETERS, scale, strict=True):
        eps = 1e-4 * s
        up = _log10_d_along(_perturbed(law, name, eps), conditions, orientation, x_fo)
        g.append((up - base) / eps)
    g = np.array(g)
    assert law.log10_sigma(conditions, orientation, x_fo) == pytest.approx(
        math.sqrt(g @ cov @ g), rel=1e-5
    )


def test_a_law_without_covariance_has_no_sigma():
    default = DIFFUSIVITY_PRESETS["dohmen_chakraborty2007"]
    assert default.uncertainty is None
    assert default.log10_sigma(KS20, ALONG_001, 0.85) is None


@pytest.mark.skipif(not PREDICTIONS.exists(), reason="the refit's results are not here")
def test_the_preset_reproduces_the_refit_at_the_round_robin_conditions():
    """`analyses/femg_refit` (dc07_free, random study) at KS20, Fo85, [001]."""
    pred = pd.read_csv(PREDICTIONS)
    row = pred[
        (pred.form == "dc07_free")
        & (pred.method == "random study")
        & (pred.condition == "KS20, 1200 C")
    ].iloc[0]
    log10_d = math.log10(PRESET.axis_diffusivities(KS20, 0.85)[2])
    assert log10_d == pytest.approx(row.log10_D001, abs=1e-9)
    assert PRESET.log10_sigma(KS20, ALONG_001, 0.85) == pytest.approx(row.se, abs=1e-9)
    assert PRESET.uncertainty is not None
    tau = PRESET.uncertainty.between_study_log10
    assert math.hypot(row.se, tau) == pytest.approx(row.se_new_lab, abs=1e-9)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"parameters": ("log10_D0",), "covariance": ((0.01,),)}, "LAW_PARAMETERS"),
        ({"parameters": ("fO2_exponent",), "covariance": ((0.01, 0.0),)}, "1 x 1"),
        (
            {
                "parameters": ("fO2_exponent", "fo_coefficient"),
                "covariance": ((1.0, 0.5), (0.4, 1.0)),
            },
            "symmetric",
        ),
        (
            {
                "parameters": ("fO2_exponent", "fo_coefficient"),
                "covariance": ((1.0, 2.0), (2.0, 1.0)),
            },
            "positive semi-definite",
        ),
        (
            {"parameters": ("fO2_exponent",), "covariance": ((0.01,),), "between_study_log10": -1},
            "zero or more",
        ),
    ],
)
def test_invalid_law_uncertainty_is_an_error(kwargs, message):
    with pytest.raises(ValueError, match=message):
        LawUncertainty(**kwargs)


def test_widen_log10():
    assert widen_log10(10.0, 8.0, 12.5, 0.0) == pytest.approx((8.0, 12.5))
    side = math.hypot(math.log10(1.25), 0.1)
    assert widen_log10(10.0, 8.0, 12.5, 0.1) == pytest.approx((10 * 10**-side, 10 * 10**side))
    low, high = widen_log10(10.0, 0.0, 11.0, 0.2)
    assert low == 0.0
    assert high == pytest.approx(10 * 10 ** math.hypot(math.log10(1.1), 0.2))
    with pytest.raises(ValueError, match="low <= time <= high"):
        widen_log10(10.0, 11.0, 12.0, 0.1)


@pytest.fixture(scope="module")
def template_fits(tmp_path_factory):
    out = tmp_path_factory.mktemp("template")
    write_template(out)
    (run,) = read_runs(out / "parameters.csv")
    profiles = read_profiles(out / "profiles.csv")
    refit = fit_run(replace(run, diffusivity="femg_refit2026"), profiles)
    return run, fit_run(run, profiles), refit


def test_a_fit_with_the_preset_reports_both_ranges(template_fits):
    run, _, fit = template_fits
    x_ref = run.initial.edge_fo + (run.initial.core_fo - run.initial.edge_fo) / 3
    assert fit.law_sigma_log10 == PRESET.log10_sigma(run.conditions, run.orientation, x_ref)
    assert PRESET.uncertainty is not None
    assert fit.between_study_log10 == PRESET.uncertainty.between_study_log10
    own = fit.time_range_days()
    law = fit.time_range_days("law")
    both = fit.time_range_days("law_and_studies")
    assert fit.law_sigma_log10 is not None
    assert law == widen_log10(fit.time_days, own[0], own[1], fit.law_sigma_log10)
    assert both[0] < law[0] < own[0] < fit.time_days < own[1] < law[1] < both[1]
    row = summary_row(fit)
    assert (row["law_low_days"], row["law_high_days"]) == law
    assert (row["all_low_days"], row["all_high_days"]) == both
    assert row["law_sigma_log10"] == fit.law_sigma_log10


def test_a_fit_with_the_default_law_has_no_law_range(template_fits):
    _, default, _ = template_fits
    assert default.law_sigma_log10 is None
    with pytest.raises(ValueError, match="no law uncertainty"):
        default.time_range_days("law")
    with pytest.raises(ValueError, match="include must be"):
        default.time_range_days("everything")
    row = summary_row(default)
    assert all(math.isnan(row[c]) for c in ("law_sigma_log10", "law_low_days", "all_high_days"))


def test_no_law_range_with_growth_stages():
    from forster.boundary import Stage

    initial = InitialCondition((40, 199), edge_fo=0.86, core_fo=0.90, steps_um=(40,))
    stage = Stage(6 * 86400.0, rim_um=0.0, zone_value=0.82)
    assert _law_uncertainty(PRESET, KS20, ALONG_001, initial, ()) != (None, None)
    assert _law_uncertainty(PRESET, KS20, ALONG_001, initial, (stage,)) == (None, None)


@pytest.mark.parametrize(("edge", "core"), [(0.80, 0.90), (0.90, 0.80)])
def test_the_composition_term_moves_time_as_evaluated_a_third_of_the_way_in(edge, core):
    """Fo 0.80 to 0.90, normal and reverse zoning: changing the composition coefficient
    by dn moves log10 t by -dn (fo_ref - X) with X a third of the way from edge to core,
    the approximation `fit_profile` uses (the midpoint misses by up to 0.005 here)."""
    initial = InitialCondition((0, 199), edge_fo=edge, core_fo=core, steps_um=(0,))
    law = replace(PRESET, name="wide", uncertainty=None)
    x = np.arange(0.0, 200.0, 1.0)
    dummy = Profile(x, np.full(len(x), 0.85) + 1e-4 * np.sin(x))
    grid = Grid(1.0, 1800.0, max_time_days=20)
    curves = fit_profile(
        dummy, KS20, ALONG_001, initial, grid, diffusivity=law, stop_early=False
    ).curves
    points = np.arange(2.0, 120.0, 2.0)
    truth = np.interp(points, x, curves[int(10 * 86400 / grid.dt_s)])
    profile = Profile(points, truth + np.random.default_rng(1).normal(0, 0.002, len(points)))
    long = replace(grid, max_time_days=60)
    base = fit_profile(profile, KS20, ALONG_001, initial, long, diffusivity=law).time_days
    x_ref = edge + (core - edge) / 3
    for dn in (-0.3, 0.3):
        moved = replace(law, name=f"n{dn}", fo_coefficient=law.fo_coefficient + dn)
        t = fit_profile(profile, KS20, ALONG_001, initial, long, diffusivity=moved).time_days
        assert math.log10(t / base) == pytest.approx(-dn * (law.fo_ref - x_ref), abs=0.002)
