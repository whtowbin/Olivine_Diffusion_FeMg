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
    final = choose_grid(*args, purpose="final", max_time_days=300, scheme="central")
    scan = choose_grid(*args, purpose="scan", time_days=final.coarse_time_days, scheme="central")
    assert final.estimated_error <= 0.01 + 1e-9
    assert scan.estimated_error <= 0.05 + 1e-9
    assert final.resolution == pytest.approx(240, rel=0.02)
    assert scan.grid.dx_um == pytest.approx(5 * final.grid.dx_um, rel=0.05)
    assert final.stability_ratio == pytest.approx(0.4)
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
    with pytest.raises(ValueError, match="unstable grid"):
        check_grid(Grid(0.5, 2 * good_dt / 0.4 * 0.5 + good_dt, 100), *args, scheme="central")
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
    assert given_dx.stability_ratio == pytest.approx(0.4)
    assert given_dx.grid.max_time_days == pytest.approx(3 * 71)
    with pytest.raises(ValueError, match="unstable grid"):
        GridRequest(dt_s=1e9).resolve(*args, time_days=71, scheme="central")
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
