"""forster.grid: choosing dx, dt and the longest time, and checking a given grid.

The accuracy rule (error of the fitted time ~ 0.65 dx / l) comes from a convergence
study on six real profiles (docs/references/grid-choice.md); here it is checked on a
two-step round robin profile against a fine-grid reference.
"""

import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from conftest import ROUND_ROBIN_DATA, ROUND_ROBIN_PARAMS

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
    stability_ratio,
    stable_dt,
)
from forster.model import Profile


@pytest.fixture(scope="module")
def two_step():
    """KS20-527-8_transect-2: steps at 20 and 110 um, a plateau between."""
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
    final = choose_grid(*args, purpose="final", max_time_days=300)
    scan = choose_grid(*args, purpose="scan", time_days=final.coarse_time_days)
    assert final.estimated_error <= 0.01 + 1e-9
    assert scan.estimated_error <= 0.05 + 1e-9
    assert scan.grid.dx_um == pytest.approx(5 * final.grid.dx_um, rel=0.05)
    assert final.stability_ratio == pytest.approx(0.4)
    # Against a reference at l / dx = 120 (twice as fine as "final").
    length = diffusion_length_um(
        final.coarse_time_days,
        [(run.diffusivity, fo_range(run.initial))],
        run.conditions,
        run.orientation,
    )
    dx_ref = actual_dx_um(profile, length / 120, run.initial)
    ref_grid = Grid(
        dx_ref,
        stable_dt(
            dx_ref, [(run.diffusivity, fo_range(run.initial))], run.conditions, run.orientation
        ),
        300,
    )
    fit_args = (profile, run.conditions, run.orientation, run.initial)
    ref = fit_profile(*fit_args, ref_grid, None, run.diffusivity)
    chosen = fit_profile(*fit_args, final.grid, None, run.diffusivity)
    coarse = fit_profile(*fit_args, scan.grid, None, run.diffusivity)
    assert chosen.time_days == pytest.approx(ref.time_days, rel=0.02)
    assert coarse.time_days == pytest.approx(ref.time_days, rel=0.08)


def test_check_grid_messages(two_step):
    run, profile = two_step
    args = (profile, run.conditions, run.orientation, run.initial, run.diffusivity)
    sp = [(run.diffusivity, fo_range(run.initial))]
    dx = actual_dx_um(profile, 1.0, run.initial)
    good_dt = stable_dt(dx, sp, run.conditions, run.orientation)
    with pytest.raises(ValueError, match="unstable grid"):
        check_grid(Grid(1.0, 2 * good_dt / 0.4 * 0.5 + good_dt, 100), *args)
    assert (
        check_grid(Grid(1.0, good_dt, 100), *args, time_days=71, purpose="scan") == []
    )  # l/dx ~ 36 for 5%
    coarse = check_grid(Grid(12.0, 1.0, 100), *args, time_days=71)
    assert any("coarse" in m for m in coarse)
    assert any("nodes across the narrowest" in m for m in coarse)
    fine = check_grid(
        Grid(0.05, 0.4 * (0.05e-6) ** 2 / 1e-15, 100), *args, time_days=71, purpose="scan"
    )
    assert any("much finer than needed" in m for m in fine)
    slow = check_grid(Grid(1.0, good_dt / 20, 100), *args)
    assert any("below the stable step" in m for m in slow)


def test_grid_request_fills_in_what_is_missing(two_step):
    run, profile = two_step
    args = (profile, run.conditions, run.orientation, run.initial, run.diffusivity)
    given_dx = GridRequest(dx_um=2.0).resolve(*args, time_days=71)
    assert given_dx.grid.dx_um == 2.0
    assert given_dx.stability_ratio == pytest.approx(0.4)
    assert given_dx.grid.max_time_days == pytest.approx(3 * 71)
    with pytest.raises(ValueError, match="unstable grid"):
        GridRequest(dt_s=1e9).resolve(*args, time_days=71)


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
    assert row["grid_stability"] == pytest.approx(0.4)
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
