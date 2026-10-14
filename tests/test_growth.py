"""Instantaneous growth stages (`fit_profile(..., stages=...)`, `forster.boundary`).

The staged model is checked against an independent conservative scheme (D on the
cell faces) run with the same history: a crystal [40, 200] um with its rim held at
Fo 0.86, then at day 6 an overgrowth [0, 40) um of Fo 0.82 with the rim at 0.82.
"""

import numpy as np
import pytest

from forster import (
    DIFFUSIVITY_PRESETS,
    Conditions,
    Grid,
    InitialCondition,
    Orientation,
    Profile,
    fit_profile,
)
from forster.boundary import BoundaryHistory, Stage

DAY = 86400.0
DX_UM = 1.0
X_UM = np.arange(0.0, 200.0, DX_UM)
CONDITIONS = Conditions(T_C=1200, P_Pa=4.2e7, fO2_Pa=0.00138)
ALONG_001 = Orientation(90, 90, 0)
CORE_FO, EARLY_RIM, ZONE = 0.90, 0.86, 0.82
STAGE_DAYS, TOTAL_DAYS = 6.0, 10.0
# Stage 0: the crystal from 40 um, all core Fo (the step sits at the rim).
EARLY = InitialCondition((40, 199), edge_fo=EARLY_RIM, core_fo=CORE_FO, steps_um=(40,))
GROWTH = Stage(STAGE_DAYS * DAY, rim_um=0.0, zone_value=ZONE)


def d_fo(fo):
    return DIFFUSIVITY_PRESETS["dohmen_chakraborty2007"].functions(CONDITIONS, ALONG_001)[0](
        np.asarray(fo)
    )


def flux_form_staged(t_end_s, dx_m, safety=0.05):
    """Fo on [0, 200) um: crystal from 40 um until STAGE_DAYS, then the overgrowth."""
    x = np.arange(0.0, 200.0, dx_m * 1e6)
    first = int(np.abs(x - 40).argmin())
    c = np.full(len(x), CORE_FO)
    h_max = safety * dx_m**2 / d_fo(np.array([ZONE])).max()

    def run(c, lo, rim, t):
        n = int(np.ceil(t / h_max))
        h = t / n
        for _ in range(n):
            full = np.concatenate([[rim], c[lo:], [CORE_FO]])
            face = 0.5 * (d_fo(full[:-1]) + d_fo(full[1:]))
            c[lo:] = c[lo:] + h * np.diff(face * np.diff(full) / dx_m) / dx_m
        return c

    c = run(c, first, EARLY_RIM, STAGE_DAYS * DAY)
    c[:first] = ZONE
    return x, run(c, 0, ZONE, t_end_s - STAGE_DAYS * DAY)


@pytest.fixture(scope="module")
def staged_curves():
    """The model's full staged run (no early stopping), on a profile spanning [0, 199]."""
    dt = 0.2 * (DX_UM * 1e-6) ** 2 / d_fo(np.array([ZONE])).max()
    grid = Grid(dx_um=DX_UM, dt_s=dt, max_time_days=TOTAL_DAYS)
    dummy = Profile(X_UM, np.full(len(X_UM), 0.88) + 1e-4 * np.sin(X_UM), name="dummy")
    result = fit_profile(
        dummy,
        CONDITIONS,
        ALONG_001,
        EARLY,
        grid,
        stop_early=False,
        stages=[GROWTH],
        scheme="central",
    )
    return result, grid


def test_nodes_outside_the_crystal_are_inactive_until_it_grows(staged_curves):
    result, grid = staged_curves
    x = result.model_x_um
    stage_row = round(STAGE_DAYS * DAY / grid.dt_s)
    assert np.isnan(result.curves[0][x < 39.5]).all()
    assert np.isfinite(result.curves[0][x > 40.5]).all()
    assert np.isnan(result.curves[stage_row - 1][x < 39.5]).all()
    assert np.isfinite(result.curves[stage_row]).all()
    assert result.curves[stage_row][x < 39.5] == pytest.approx(ZONE)


def test_staged_run_matches_an_independent_solver(staged_curves):
    result, grid = staged_curves
    row = len(result.curves) - 1
    x_ref, ref = flux_form_staged(row * grid.dt_s, DX_UM * 1e-6)
    model = np.interp(x_ref, result.model_x_um, result.curves[row])
    # Same order as the engine against the Fo reference (1.4e-4 on a 0.1 step).
    assert np.abs(model - ref).max() < 5e-4
    # Control: without the early stage (overgrowth and core in contact from time
    # zero, as a two-zone initial condition assumes) the profile differs far more.
    single = InitialCondition((0, 199), edge_fo=ZONE, core_fo=CORE_FO, steps_um=(40,))
    dummy = Profile(X_UM, np.full(len(X_UM), 0.88) + 1e-4 * np.sin(X_UM))
    unstaged = fit_profile(
        dummy,
        CONDITIONS,
        ALONG_001,
        single,
        Grid(DX_UM, grid.dt_s, (row - round(STAGE_DAYS * DAY / grid.dt_s)) * grid.dt_s / DAY),
        stop_early=False,
        scheme="central",
    )
    late = np.interp(x_ref, unstaged.model_x_um, unstaged.curves[-1])
    assert np.abs(late - ref).max() > 10 * 5e-4


def test_a_stage_that_changes_nothing_leaves_the_run_unchanged():
    initial = InitialCondition((0, 199), edge_fo=0.82, core_fo=0.90, steps_um=(40,))
    nothing = Stage(1.0 * DAY, rim_um=0.0, rim_value=0.82)
    dt = 0.2 * (DX_UM * 1e-6) ** 2 / d_fo(np.array([0.82])).max()
    grid = Grid(DX_UM, dt, max_time_days=4)
    profile = Profile(X_UM, np.full(len(X_UM), 0.88) + 1e-4 * np.sin(X_UM))
    plain = fit_profile(
        profile, CONDITIONS, ALONG_001, initial, grid, stop_early=False, scheme="central"
    )
    staged = fit_profile(
        profile,
        CONDITIONS,
        ALONG_001,
        initial,
        grid,
        stop_early=False,
        stages=[nothing],
        scheme="central",
    )
    assert np.array_equal(plain.curves, staged.curves)


def test_fit_recovers_the_time_of_a_staged_profile():
    x_ref, ref = flux_form_staged(TOTAL_DAYS * DAY, DX_UM * 1e-6)
    points = np.arange(1.0, 199.0, 3.0)
    rng = np.random.default_rng(5)
    profile = Profile(points, np.interp(points, x_ref, ref) + rng.normal(0, 0.001, len(points)))
    grid = Grid(DX_UM, 60.0, max_time_days=40)
    result = fit_profile(profile, CONDITIONS, ALONG_001, EARLY, grid, stages=[GROWTH])
    assert result.time_days == pytest.approx(TOTAL_DAYS, rel=0.05)
    assert result.time_since_last_stage_days == pytest.approx(TOTAL_DAYS - STAGE_DAYS, rel=0.1)
    assert result.last_stage_s == pytest.approx(STAGE_DAYS * DAY, abs=grid.dt_s)
    # Early stopping gives the same fit as the full run.
    full = fit_profile(
        profile, CONDITIONS, ALONG_001, EARLY, grid, stop_early=False, stages=[GROWTH]
    )
    assert (result.indices, result.time_s, result.time_plus_s) == (
        full.indices,
        full.time_s,
        full.time_plus_s,
    )


@pytest.mark.parametrize(
    ("stages", "message"),
    [
        ([Stage(2 * DAY, rim_um=50.0, zone_value=0.8)], "cannot move inward"),
        ([Stage(2 * DAY, rim_um=0.0)], "give zone_value"),
        ([Stage(2 * DAY, rim_um=0.0, zone_value=1.2)], "fractions"),
        (
            [
                Stage(3 * DAY, rim_um=20.0, zone_value=0.8),
                Stage(2 * DAY, rim_um=0.0, zone_value=0.8),
            ],
            "must increase",
        ),
    ],
)
def test_invalid_stages_are_errors(stages, message):
    with pytest.raises(ValueError, match=message):
        BoundaryHistory.from_initial(EARLY, stages)


def test_stages_need_time_after_the_last_one_and_the_correlated_method():
    profile = Profile(X_UM, np.full(len(X_UM), 0.88) + 1e-4 * np.sin(X_UM))
    with pytest.raises(ValueError, match="must extend past the last stage"):
        fit_profile(
            profile,
            CONDITIONS,
            ALONG_001,
            EARLY,
            Grid(DX_UM, 600, max_time_days=5),
            stages=[GROWTH],
        )
    with pytest.raises(ValueError, match="correlated"):
        fit_profile(
            profile,
            CONDITIONS,
            ALONG_001,
            EARLY,
            Grid(DX_UM, 600, max_time_days=20),
            uncertainty="legacy",
            stages=[GROWTH],
        )
