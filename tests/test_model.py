import dataclasses

import numpy as np
import pytest

from forster import (
    Conditions,
    Grid,
    InitialCondition,
    Orientation,
    Profile,
    engine,
    fit_profile,
    orientation,
)


def test_profile_rejects_percent_fo():
    with pytest.raises(ValueError, match="fraction"):
        Profile(np.arange(5.0), np.full(5, 88.0))


def test_profile_rejects_mismatched_arrays():
    with pytest.raises(ValueError, match="same length"):
        Profile(np.arange(5.0), np.full(4, 0.88))


def test_buffer_conditions_match_engine():
    c = Conditions.from_buffer(1250, 5e8, 0.3, "QFM")
    assert c.fO2_Pa == engine.fo2buffer(1250 + 273.15, 5e8, 0.3, "QFM")
    assert Conditions.from_buffer(1250, 5e8, 0.3, "fmq").fO2_Pa == c.fO2_Pa


def test_unknown_buffer_is_an_error():
    # engine.fo2buffer returns a string for an unknown buffer; the API must not pass that on.
    with pytest.raises(ValueError, match="unknown fO2 buffer"):
        Conditions.from_buffer(1250, 5e8, 0.3, "IW")


def test_orientation_from_euler_matches_orientation_module():
    o = Orientation.from_euler(60, 20, 10, profile_angle_deg=40)
    assert (o.alpha_deg, o.beta_deg, o.gamma_deg) == orientation.vector_direction(40, 60, 20, 10)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"domain_um": (50, 10)}, "increasing"),
        ({"steps_um": ()}, "at least one"),
        ({"edge_fo": 88}, "fractions"),
        ({"step_fo": 0.9}, "exactly two"),
    ],
)
def test_initial_condition_validation(kwargs, message):
    base = {"domain_um": (0, 100), "edge_fo": 0.82, "core_fo": 0.88, "steps_um": (40,)}
    with pytest.raises(ValueError, match=message):
        InitialCondition(**(base | kwargs))


def test_grid_timesteps_match_model_diffusion_expression():
    assert Grid(dx_um=1, dt_s=600, max_time_days=730).timesteps == int(730 * 24 * 60 * 60 / 600)


def test_unknown_scheme_is_an_error():
    profile = Profile(np.arange(0.0, 50.0, 5.0), np.linspace(0.82, 0.88, 10))
    initial = InitialCondition(domain_um=(0, 45), edge_fo=0.82, core_fo=0.88, steps_um=(20,))
    with pytest.raises(ValueError, match="unknown scheme"):
        fit_profile(
            profile,
            Conditions(1200, 4.2e7, 1e-3),
            Orientation(90, 90, 0),
            initial,
            Grid(dx_um=1, dt_s=600, max_time_days=1),
            scheme="crank_nicolson",
        )


@pytest.mark.parametrize("scheme", ["implicit", "central"])
def test_a_lean_fit_equals_a_fit_that_keeps_every_row(scheme, tmp_path):
    """The default keeps the model at the points plus a few full profiles; the rows it
    rebuilds from its checkpoints are bit-identical to a run that kept every row."""
    from forster import fit_run, read_profiles, read_runs, write_template

    write_template(tmp_path)
    (run,) = read_runs(tmp_path / "parameters.csv")
    profiles = read_profiles(tmp_path / "profiles.csv")
    run = dataclasses.replace(run, scheme=scheme)
    lean = fit_run(run, profiles)
    full = fit_run(run, profiles, keep_curves=True)
    assert (lean.time_s, lean.time_minus_s, lean.time_plus_s) == (
        full.time_s,
        full.time_minus_s,
        full.time_plus_s,
    )
    assert (lean.time_minus2_s, lean.time_plus2_s) == (full.time_minus2_s, full.time_plus2_s)
    np.testing.assert_array_equal(lean.sum_chi2, full.sum_chi2)
    assert lean.curve_rows is not None and len(lean.curve_rows) <= 6
    assert full.curve_rows is None and len(full.curves) == lean.n_rows
    for row in lean.curve_rows:
        np.testing.assert_array_equal(lean.curve(row), full.curves[row])
    with pytest.raises(KeyError, match="keep_curves"):
        lean.curve(next(r for r in range(lean.n_rows) if r not in lean.curve_rows))


def test_the_2_sigma_range_contains_the_1_sigma_range(tmp_path):
    from forster import fit_run, read_profiles, read_runs, summary_row, write_template
    from forster.diffusivity import resolve

    write_template(tmp_path)
    (run,) = read_runs(tmp_path / "parameters.csv")
    result = fit_run(
        dataclasses.replace(run, diffusivity=resolve("refit")),
        read_profiles(tmp_path / "profiles.csv"),
    )
    low1, high1 = result.time_range_days()
    low2, high2 = result.time_range_days(sigmas=2)
    assert low2 < low1 < result.time_days < high1 < high2
    law1, law2 = result.time_range_days("law"), result.time_range_days("law", sigmas=2)
    assert law2[0] < law1[0] and law2[1] > law1[1] and law2[0] < low2 and law2[1] > high2
    row = summary_row(result)
    assert (row["low2_days"], row["high2_days"]) == pytest.approx((low2, high2))
    with pytest.raises(ValueError, match="sigmas"):
        result.time_range_days(sigmas=3)
