"""forster.grid for the explicit schemes: choosing dx, dt and the longest time, and
checking a given grid (the implicit scheme's rules: tests/test_implicit.py).

The accuracy rule (error of the fitted time ~ 2.4 dx / l, worst case on 35 real
profiles against a converged reference; docs/references/grid-choice.md) is checked
here on a two-step round robin profile against an implicit fine-grid reference.
"""

import math
import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from conftest import ROUND_ROBIN_DATA, ROUND_ROBIN_PARAMS, require_round_robin

from forster import (
    Grid,
    InitialCondition,
    RunDefaults,
    fit_profile,
    fit_run,
    read_profiles,
    read_runs,
    summary_row,
    write_template,
)
from forster.grid import (
    GridRequest,
    actual_dx_um,
    check_grid,
    choose_grid,
    diffusion_length_um,
    fo_range,
    narrowest_gap_um,
    resolving_dt,
    stability_ratio,
    stable_dt,
)
from forster.model import Profile, _stepper


@pytest.fixture(scope="module")
def two_step():
    """KS20-527-8_transect-2: steps at 20 and 110 um, a plateau between."""
    require_round_robin()
    runs = {r.name: r for r in read_runs(ROUND_ROBIN_PARAMS, defaults=RunDefaults(730))}
    run = runs["KS20-527-8_transect-2"]
    m = read_profiles(ROUND_ROBIN_DATA)[run.profile_name]
    return run, Profile(m.distance_um, m.fo, name=run.name)


def test_fo_range_and_narrowest_gap():
    initial = InitialCondition((0, 100), 0.914, 0.90, steps_um=(10, 33), step_fo=0.93)
    assert fo_range(initial) == (0.90, 0.93)  # the plateau lies outside edge..core
    assert narrowest_gap_um(initial) == 10  # rim to the first step


def test_stable_dt_uses_the_fastest_species_on_the_actual_spacing(two_step):
    run, profile = two_step
    dx = actual_dx_um(profile, 1.0, run.initial)
    fe_mg = [(run.diffusivity, fo_range(run.initial))]
    with_ni = [*fe_mg, ("girona_costa2013_ni", fo_range(run.initial))]
    dt = stable_dt(dx, fe_mg, run.conditions, run.orientation)
    assert stability_ratio(dx, dt, fe_mg, run.conditions, run.orientation) == pytest.approx(0.4)
    dt_both = stable_dt(dx, with_ni, run.conditions, run.orientation)
    assert dt_both <= dt
    assert stability_ratio(dx, dt_both, with_ni, run.conditions, run.orientation) == (
        pytest.approx(0.4)
    )


def test_choose_grid_meets_its_tolerance(two_step):
    run, profile = two_step
    args = (profile, run.conditions, run.orientation, run.initial, run.diffusivity)
    final = choose_grid(*args, purpose="final", max_time_days=300, scheme="central")
    scan = choose_grid(*args, purpose="scan", time_days=final.coarse_time_days, scheme="central")
    assert final.estimated_error <= 0.01 + 1e-9
    assert scan.estimated_error <= 0.05 + 1e-9
    assert final.resolution == pytest.approx(240, rel=0.02)
    assert scan.grid.dx_um == pytest.approx(5 * final.grid.dx_um, rel=0.05)
    # dt resolves the fit (the explicit run takes stable substeps within it), unless the
    # stable step is longer.
    sp = [(run.diffusivity, fo_range(run.initial))]
    final_dx = actual_dx_um(profile, final.grid.dx_um, run.initial, "central")
    assert final.grid.dt_s == max(
        resolving_dt(final.coarse_time_days, 0.01),
        stable_dt(final_dx, sp, run.conditions, run.orientation),
    )
    assert final.stability_ratio > 100  # hundreds of stable steps per kept row
    # Against a converged reference: the implicit scheme at l / dx = 40 (an explicit
    # reference would share the explicit first-order bias). The final explicit grid
    # (l / dx = 240) is too slow for a test; the scan grid is checked.
    length = diffusion_length_um(
        final.coarse_time_days,
        [(run.diffusivity, fo_range(run.initial))],
        run.conditions,
        run.orientation,
    )
    ref_grid = Grid(length / 40, final.coarse_time_days * 86400 / 1000, 300)
    fit_args = (profile, run.conditions, run.orientation, run.initial)
    ref = fit_profile(*fit_args, ref_grid, None, run.diffusivity, scheme="implicit")
    coarse = fit_profile(*fit_args, scan.grid, None, run.diffusivity, scheme="central")
    assert coarse.time_days == pytest.approx(ref.time_days, rel=0.05)


def test_check_grid_messages(two_step):
    run, profile = two_step
    args = (profile, run.conditions, run.orientation, run.initial, run.diffusivity)
    sp = [(run.diffusivity, fo_range(run.initial))]
    dx = actual_dx_um(profile, 0.5, run.initial, "central")
    good_dt = stable_dt(dx, sp, run.conditions, run.orientation)
    # A dt above the stable step is split into stable steps by fit_profile: no complaint.
    assert check_grid(Grid(0.5, 10 * good_dt, 100), *args, scheme="central") == []
    assert (
        check_grid(Grid(0.5, good_dt, 100), *args, time_days=71, purpose="scan", scheme="central")
        == []
    )  # l/dx ~ 72; 48 needed for 5%
    coarse = check_grid(Grid(12.0, 1.0, 100), *args, time_days=71, scheme="central")
    assert any("coarse" in m for m in coarse)
    assert any("nodes across the narrowest" in m for m in coarse)
    fine = check_grid(
        Grid(0.05, 0.4 * (0.05e-6) ** 2 / 1e-15, 100),
        *args,
        time_days=71,
        purpose="scan",
        scheme="central",
    )
    assert any("much finer than needed" in m for m in fine)
    slow = check_grid(Grid(0.5, good_dt / 20, 100), *args, scheme="central")
    assert any("below the stable step" in m for m in slow)


def test_grid_request_fills_in_what_is_missing(two_step):
    run, profile = two_step
    args = (profile, run.conditions, run.orientation, run.initial, run.diffusivity)
    given_dx = GridRequest(dx_um=2.0).resolve(*args, time_days=71, scheme="central")
    assert given_dx.grid.dx_um == 2.0
    dx = actual_dx_um(profile, 2.0, run.initial, "central")
    stable = stable_dt(dx, [(run.diffusivity, fo_range(run.initial))], *args[1:3])
    assert given_dx.grid.dt_s == max(stable, resolving_dt(71, 0.01))
    assert given_dx.grid.max_time_days == pytest.approx(3 * 71)
    # A given dt is kept, whatever its stability: the explicit run substeps within it.
    assert GridRequest(dt_s=1e5).resolve(*args, time_days=71, scheme="central").grid.dt_s == 1e5
    # The implicit scheme (the default) keeps a given dt: it is stable at any dt.
    implicit = GridRequest(dx_um=2.0).resolve(*args, time_days=71)
    assert implicit.grid.dt_s == pytest.approx(71 * 86400 * 0.01 / 4)


@pytest.fixture
def blank_grid_template(tmp_path):
    write_template(tmp_path / "t")
    params = pd.read_csv(tmp_path / "t" / "parameters.csv").assign(dx_um=np.nan, dt_s=np.nan)
    path = tmp_path / "params.csv"
    params.to_csv(path, index=False)
    return path, read_profiles(tmp_path / "t" / "profiles.csv")


def test_blank_grid_is_chosen_and_the_summary_refits_identically(tmp_path, blank_grid_template):
    path, profiles = blank_grid_template
    (run,) = read_runs(path)
    assert isinstance(run.grid, GridRequest)
    result = fit_run(run, profiles)
    row = summary_row(result)
    # Chosen from the coarse fit's time; evaluated at the final fit's, so within a hair.
    assert row["grid_error_pct"] <= 1.05
    assert math.isnan(row["grid_stability"])  # the implicit scheme has no limit
    again_path = tmp_path / "again.csv"
    pd.DataFrame([dict(row, profile=run.profile_name)]).to_csv(again_path, index=False)
    (again,) = read_runs(again_path)
    assert isinstance(again.grid, Grid)
    assert fit_run(again, profiles).time_days == result.time_days


def test_a_coarse_user_grid_is_fitted_with_a_warning(blank_grid_template):
    path, profiles = blank_grid_template
    (run,) = read_runs(path)
    coarse = replace(run, grid=Grid(5.0, 600.0, 40))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit_run(coarse, profiles)
    assert any("may be off by up to" in str(w.message) for w in caught)


@pytest.mark.parametrize("scheme", ["central", "dipra"])
@pytest.mark.parametrize("fast", [True, False])
def test_explicit_substeps_match_a_stable_run_row_for_row(two_step, scheme, fast):
    """A dt 3x the stable step runs as 3 stable steps per kept row: the kept rows are
    those of the stable run at every third step (compiled and NumPy steppers)."""
    run, _ = two_step
    diffusivity = run.diffusivity
    D, dD = diffusivity.functions(run.conditions, run.orientation)
    dx_um = 1.0
    fo = np.array(run.initial.fo_values)
    dt = 0.35 * (dx_um * 1e-6) ** 2 / float(D(fo).max())
    x = np.arange(0.0, 300.0, dx_um)
    start = np.where(x < 20, run.initial.edge_fo, run.initial.core_fo)
    bounds = (run.initial.edge_fo, run.initial.core_fo)
    stable = _stepper(scheme, D, dD, bounds, dt, dx_um * 1e-6, fast)(start, 300)
    coarse = _stepper(scheme, D, dD, bounds, 3 * dt, dx_um * 1e-6, fast, substeps=3)(start, 100)
    assert coarse.shape == (101, len(x))
    np.testing.assert_allclose(coarse, stable[::3], rtol=0, atol=1e-13)


def test_fit_with_a_dt_above_the_stable_step_substeps(two_step):
    """fit_profile with dt 3x the stable step: the same best fit as the stable grid, to
    within the coarser time resolution, and a third of the rows kept."""
    run, profile = two_step
    sp = [(run.diffusivity, fo_range(run.initial))]
    dx = actual_dx_um(profile, 1.0, run.initial, "central")
    dt = stable_dt(dx, sp, run.conditions, run.orientation, ratio=0.35)
    args = (profile, run.conditions, run.orientation, run.initial)
    fine = fit_profile(*args, Grid(1.0, dt, 120), None, run.diffusivity, scheme="central")
    coarse = fit_profile(*args, Grid(1.0, 3 * dt, 120), None, run.diffusivity, scheme="central")
    assert abs(coarse.time_days - fine.time_days) <= 1.5 * 3 * dt / 86400
    assert len(coarse.curves) <= len(fine.curves) // 3 + 2


def test_a_step_next_to_the_rim_does_not_ask_for_a_huge_grid(two_step):
    """A step 0.06 um from the rim (round robin refit, 2026-10-15) asked for dx = 0.015
    um; the gap rule now stops at span / MAX_GAP_NODES (explicit) and the coarse fit at
    span / COARSE_MAX_NODES, and check_grid still reports the unresolved gap."""
    import time

    from forster.grid import MAX_GAP_NODES

    run, profile = two_step
    lo, hi = run.initial.domain_um
    near = replace(run.initial, steps_um=(lo + 0.06, run.initial.steps_um[1]))
    args = (profile, run.conditions, run.orientation, near, run.diffusivity)
    t0 = time.perf_counter()
    choice = choose_grid(*args, purpose="scan", scheme="central")
    assert time.perf_counter() - t0 < 60
    assert choice.grid.dx_um >= 0.99 * (hi - lo) / MAX_GAP_NODES  # dx_not_above rounds down
    assert any("narrowest" in m for m in check_grid(choice.grid, *args, scheme="central"))
