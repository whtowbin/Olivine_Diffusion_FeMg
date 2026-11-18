"""The `dipra` scheme's limit on the Fo contrast between adjacent nodes.

Girona and Costa's eq. 5 uses a forward difference in the cross term; with dD/dFo =
-3 ln10 D it makes the effective diffusivity of the forward difference D (1 - 3 ln10 d),
negative once the contrast d between adjacent nodes exceeds 1 / (3 ln 10) = 0.145
(derivation in `forster.dipra`). Measured 2026-11-17: undershoot of the held rim at
contrast 0.16-0.18, -inf / NaN at 0.23. `central` and `implicit` do not have the limit.
"""

import math
import warnings

import numpy as np
import pytest

from forster import (
    DIFFUSIVITY_PRESETS,
    Conditions,
    Grid,
    InitialCondition,
    Orientation,
    Profile,
    dipra,
    fit_profile,
)
from forster.implicit import cell_average, stepper

CONDITIONS = Conditions(T_C=1200, P_Pa=4.2e7, fO2_Pa=0.00138)
ALONG_001 = Orientation(90, 90, 0)
DX_UM = 1.0
X_UM = np.arange(0.0, 60.0, DX_UM)


def fit_dipra(edge_fo: float, core_fo: float):
    initial = InitialCondition((30.0, 59.0), edge_fo=edge_fo, core_fo=core_fo, steps_um=(30.0,))
    profile = Profile(X_UM, np.where(X_UM < 30.0, edge_fo, core_fo))
    grid = Grid(DX_UM, 3600.0, max_time_days=0.05)
    return fit_profile(profile, CONDITIONS, ALONG_001, initial, grid, scheme="dipra")


def test_the_limit_is_one_over_three_ln_ten():
    assert pytest.approx(1 / (3 * math.log(10)), rel=1e-15) == dipra.CONTRAST_LIMIT
    assert pytest.approx(0.1448, abs=5e-5) == dipra.CONTRAST_LIMIT


def test_contrast_counts_the_boundary_values_and_every_adjacent_pair():
    # 0.625 -> 0.625 -> 0.625, boundaries 0.75 and 0.875: the largest jump is 0.875 - 0.625.
    assert dipra.max_contrast(np.array([0.625] * 3), (0.75, 0.875)) == pytest.approx(0.25)
    assert dipra.max_contrast(np.array([0.9, 0.9, 0.7, 0.7]), (0.9, 0.7)) == pytest.approx(0.2)


def test_warns_above_the_limit():
    with pytest.warns(UserWarning, match=r"contrast of 0\.200.*limit 0\.145.*implicit"):
        fit_dipra(0.70, 0.90)


def test_silent_below_the_limit():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fit_dipra(0.80, 0.90)


def test_other_schemes_do_not_warn_at_a_large_contrast():
    initial = InitialCondition((30.0, 59.0), edge_fo=0.70, core_fo=0.90, steps_um=(30.0,))
    profile = Profile(X_UM, np.where(X_UM < 30.0, 0.70, 0.90))
    grid = Grid(DX_UM, 3600.0, max_time_days=0.05)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fit_profile(profile, CONDITIONS, ALONG_001, initial, grid, scheme="implicit")


def test_dipra_undershoots_the_rim_above_the_limit_and_implicit_does_not():
    """Step 0.70 / 0.90 (contrast 0.20) with both ends held: the exact solution stays in
    [0.70, 0.90] (maximum principle). After 5 stable steps (D dt / dx^2 = 0.2) dipra is 0.089
    below the rim (measured 2026-11-17; -inf after 20 steps); implicit stays inside."""
    low, high = 0.70, 0.90
    d_fo, d_derivative = DIFFUSIVITY_PRESETS["dohmen_chakraborty2007"].functions(
        CONDITIONS, ALONG_001
    )
    start = np.where(X_UM < 30.0, low, high)
    dt = 0.2 * (DX_UM * 1e-6) ** 2 / d_fo(np.array([low, high])).max()
    rows = dipra.timestepper(start, d_fo, d_derivative, (low, high), 5, dt, DX_UM * 1e-6)
    assert rows[-1].min() < low - 0.01
    inside = stepper(d_fo, low, high, dt, X_UM)(cell_average(X_UM, [30.0], [low, high]), 5)
    assert inside.min() >= low - 1e-9 and inside.max() <= high + 1e-9
