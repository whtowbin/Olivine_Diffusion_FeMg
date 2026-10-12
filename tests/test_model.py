import numpy as np
import pytest

from olivine_diffusion import (
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
            scheme="implicit",
        )
