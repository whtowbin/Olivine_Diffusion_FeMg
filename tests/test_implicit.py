"""The implicit scheme (`forster.implicit`, `fit_profile(..., scheme="implicit")`).

Checked against closed-form solutions (a fixed rim: erfc; a rim rising linearly in
time: Carslaw and Jaeger 1959, section 2.5), against an independent explicit solver
with the same on-node boundary for growth stages, and on a real profile against a
fine-grid reference.
"""

import itertools
import math
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from conftest import ROUND_ROBIN_DATA, ROUND_ROBIN_PARAMS, require_round_robin, requires_round_robin
from scipy.special import erfc

from forster import (
    DIFFUSIVITY_PRESETS,
    Conditions,
    Grid,
    InitialCondition,
    Orientation,
    Profile,
    RunDefaults,
    fit_profile,
    read_profiles,
    read_runs,
)
from forster.boundary import RimPath, Stage
from forster.grid import (
    STABILITY_RATIO,
    check_grid,
    choose_grid,
    diffusion_length_um,
    fo_range,
)
from forster.implicit import cell_average, node_grid, stepper

DAY = 86400.0
D_CONST = 1e-17  # m^2/s


def constant_d(c):
    return np.full_like(c, D_CONST)


def test_node_grid_puts_a_node_on_every_given_point():
    x = node_grid([0.0, 40.3, 199.0], 1.0)
    assert {0.0, 40.3, 199.0} <= set(x.tolist())
    assert np.diff(x).max() <= 1.0 + 1e-12
    assert len(node_grid([0.0, 10.0], 100.0)) == 3  # never fewer than three nodes


def test_cell_average_keeps_a_step_between_nodes():
    x = np.arange(0.0, 11.0)
    c = cell_average(x, [4.3], [1.0, 0.0])
    assert c[4] == pytest.approx(0.8)  # cell [3.5, 4.5]: 0.8 of it before the step
    assert c[3] == 1.0 and c[5] == 0.0
    # The amount of the step function is kept (trapezoid weights = cell widths).
    widths = np.r_[0.5, np.ones(9), 0.5]
    assert np.sum(c * widths) == pytest.approx(4.3)


def _fixed_rim_error(dx_um, steps):
    t = 2 * DAY
    length = math.sqrt(D_CONST * t) * 1e6
    x = node_grid([0.0, 12 * length], dx_um)
    rows = stepper(constant_d, 0.8, 0.9, t / steps, x)(np.full(len(x), 0.9), steps)
    exact = 0.9 + (0.8 - 0.9) * erfc(x / (2 * length))
    return np.abs(rows[-1] - exact).max() / 0.1, length


def test_fixed_rim_matches_erfc_and_converges_at_second_order():
    _, length = _fixed_rim_error(1.0, 10)
    coarse, _ = _fixed_rim_error(length / 8, 400)
    fine, _ = _fixed_rim_error(length / 16, 400)
    assert coarse < 2e-3
    assert coarse / fine > 3.0  # second order in dx: 4 in the limit
    few, _ = _fixed_rim_error(length / 32, 25)
    many, _ = _fixed_rim_error(length / 32, 50)
    assert few / many > 3.0  # second order in time


def test_time_order_with_an_fo_dependent_d():
    """D frozen at the start of each stage makes TR-BDF2 first order in time when D
    depends on Fo (the constant-D test above sees second order): halving dt halves the
    error against a fine-dt run. Measured 2026-10-14: ratio 2.05."""
    conditions = Conditions(T_C=1200, P_Pa=4.2e7, fO2_Pa=0.00138)
    D, _ = DIFFUSIVITY_PRESETS["dohmen_chakraborty2007"].functions(
        conditions, Orientation(90, 90, 0)
    )
    x = node_grid([0.0, 199.0], 0.5)
    start = cell_average(x, [60.0], [0.80, 0.90])
    t = 2 * DAY

    def final(steps):
        return stepper(D, 0.80, 0.90, t / steps, x)(start, steps)[-1]

    reference = final(1280)
    errors = [np.abs(final(n) - reference).max() for n in (10, 20, 40)]
    assert errors[0] < 2e-4
    for coarse, fine in itertools.pairwise(errors):
        assert 1.7 < coarse / fine < 2.4


def test_a_stage_starting_one_ulp_after_a_step():
    """A stage start a rounding error past a time step counts as on it, as the offset
    of the last stage does; it used to leave the new zone unfilled (NaN)."""
    early = InitialCondition((40, 199), edge_fo=0.86, core_fo=0.90, steps_um=(40,))
    x = np.arange(0.0, 200.0, 1.0)
    dummy = Profile(x, np.full(len(x), 0.88) + 1e-4 * np.sin(x))
    grid = Grid(1.0, 864.0, max_time_days=10)
    on_step = 637 * grid.dt_s

    def curves(start_s):
        stage = Stage(start_s, rim_um=0.0, zone_value=0.82)
        return fit_profile(
            dummy,
            CONDITIONS,
            ALONG_001,
            early,
            grid,
            stop_early=False,
            stages=[stage],
            keep_curves=True,
        ).curves

    late = curves(float(np.nextafter(on_step, np.inf)))
    assert np.isfinite(late[-1]).all()
    assert np.allclose(late[-1], curves(on_step)[-1], rtol=0, atol=1e-12)


def test_rim_rising_linearly_in_time_matches_the_closed_form():
    t_days, v0, v1 = 3.0, 0.80, 0.86
    rim = RimPath((0.0, t_days), (v0, v1))
    length = math.sqrt(D_CONST * t_days * DAY) * 1e6
    x = node_grid([0.0, 12 * length], length / 16)
    rows = stepper(constant_d, rim, v0, t_days * DAY / 200, x)(np.full(len(x), v0), 200)
    eta = x / (2 * length)
    shape = (1 + 2 * eta**2) * erfc(eta) - 2 / math.sqrt(math.pi) * eta * np.exp(-(eta**2))
    exact = v0 + (v1 - v0) * shape
    assert rows[-1][0] == pytest.approx(v1)
    assert np.abs(rows[-1] - exact).max() < 2e-3 * (v1 - v0)


def test_runs_restarted_in_blocks_are_identical():
    x = node_grid([0.0, 50.0], 1.0)
    rim = RimPath((0.0, 1.0, 2.0), (0.8, 0.85, 0.83))
    advance = stepper(constant_d, rim, 0.9, 2000.0, x)
    start = cell_average(x, [12.5], [0.82, 0.9])
    whole = advance(start, 100)
    first = advance(start, 40)
    rest = advance(first[-1], 60, first_row=40)
    assert np.array_equal(whole, np.concatenate([first, rest[1:]]))


# Growth stages: a crystal from 40.3 um (not a multiple of dx) with its rim at 0.86,
# then at 6.37 days (between time steps) an overgrowth [0, 40.3) of 0.82.
CONDITIONS = Conditions(T_C=1200, P_Pa=4.2e7, fO2_Pa=0.00138)
ALONG_001 = Orientation(90, 90, 0)
EARLY = InitialCondition((40.3, 199), edge_fo=0.86, core_fo=0.90, steps_um=(40.3,))
GROWTH = Stage(6.37 * DAY, rim_um=0.0, zone_value=0.82)


def d_fo(fo):
    D, _ = DIFFUSIVITY_PRESETS["dohmen_chakraborty2007"].functions(CONDITIONS, ALONG_001)
    return D(np.asarray(fo))


def on_node_reference(t_end_s, x):
    """Explicit flux form on ``x`` (nodes on 0, 40.3 and 199), rims held on their
    nodes, the stage start hit exactly, the old-rim node averaged over its cell."""
    first = int(np.argmin(np.abs(x - 40.3)))
    c = np.full(len(x), 0.90)
    c[first] = 0.86

    def run(c, lo, rim, t):
        xs = x[lo:]
        h = np.diff(xs) * 1e-6
        w = 0.5 * (h[1:] + h[:-1])
        n = math.ceil(t / (0.2 * h.min() ** 2 / d_fo(np.array([0.82]))[0]))
        for _ in range(n):
            s = c[lo:]
            face = 0.5 * (d_fo(s[1:]) + d_fo(s[:-1]))
            s[1:-1] += (t / n) * np.diff(face * np.diff(s) / h) / w
            s[0], s[-1] = rim, 0.90
        return c

    c = run(c, first, 0.86, GROWTH.start_s)
    outer, inner = x[first] - x[first - 1], x[first + 1] - x[first]
    c[first] = (outer * 0.82 + inner * c[first]) / (outer + inner)
    c[:first] = 0.82
    return run(c, 0, 0.82, t_end_s - GROWTH.start_s)


def test_staged_implicit_run_matches_an_on_node_reference():
    dt = 0.05 * DAY
    grid = Grid(dx_um=1.0, dt_s=dt, max_time_days=10.0)
    x_um = np.arange(0.0, 199.5, 1.0)
    dummy = Profile(x_um, 0.88 + 1e-4 * np.sin(x_um))
    result = fit_profile(
        dummy,
        CONDITIONS,
        ALONG_001,
        EARLY,
        grid,
        scheme="implicit",
        stop_early=False,
        stages=[GROWTH],
        keep_curves=True,
    )
    x = result.model_x_um
    assert 40.3 in x.tolist()
    assert result.last_stage_s == GROWTH.start_s  # exact, not rounded to a step
    stage_row = math.ceil(GROWTH.start_s / dt)
    assert np.isnan(result.curves[stage_row - 1][x < 40]).all()
    assert np.isfinite(result.curves[stage_row]).all()
    row = len(result.curves) - 1
    ref = on_node_reference(row * dt, x)
    # The growth test's ghost-node run reaches 3.4e-4 against its own reference.
    assert np.abs(result.curves[row] - ref).max() < 1e-4


@requires_round_robin
def test_changing_rims_need_the_implicit_scheme_and_delay_early_stopping():
    profiles = read_profiles(ROUND_ROBIN_DATA)
    runs = {r.name: r for r in read_runs(ROUND_ROBIN_PARAMS, defaults=RunDefaults(200))}
    run = runs["KS20-527_2-transect-1"]
    m = profiles[run.profile_name]
    profile = Profile(m.distance_um, m.fo)
    args = (profile, run.conditions, run.orientation, run.initial)
    rim = RimPath((0.0, 150.0), (0.82, 0.80))
    grid = Grid(2.0, 0.05 * DAY, 200)  # early stopping advances in blocks of 500 rows
    with pytest.raises(ValueError, match="implicit"):
        fit_profile(*args, grid, scheme="central", rim_fo=rim)
    with pytest.raises(ValueError, match="correlated"):
        fit_profile(*args, grid, scheme="implicit", uncertainty="legacy")
    late = fit_profile(*args, grid, scheme="implicit", rim_fo=rim)
    assert late.modelled_days >= 150.0  # no early stop while the rim still changes
    fixed = fit_profile(*args, grid, scheme="implicit", rim_fo=0.82)
    assert fixed.modelled_days < 150.0
    assert late.inputs["rim_fo"] == rim


@pytest.fixture(scope="module")
def two_step():
    require_round_robin()
    runs = {r.name: r for r in read_runs(ROUND_ROBIN_PARAMS, defaults=RunDefaults(730))}
    run = runs["KS20-527-8_transect-2"]
    m = read_profiles(ROUND_ROBIN_DATA)[run.profile_name]
    return run, Profile(m.distance_um, m.fo, name=run.name)


def test_chosen_implicit_grid_meets_its_tolerance_on_a_real_profile(two_step):
    run, profile = two_step
    args = (profile, run.conditions, run.orientation, run.initial, run.diffusivity)
    final = choose_grid(*args, purpose="final", scheme="implicit")
    scan = choose_grid(*args, purpose="scan", scheme="implicit", time_days=final.coarse_time_days)
    assert final.resolution == pytest.approx(8, rel=0.02)
    assert scan.resolution == pytest.approx(4, rel=0.05)
    assert final.grid.dt_s == pytest.approx(0.0025 * final.coarse_time_days * DAY)
    fit_args = (profile, run.conditions, run.orientation, run.initial)
    length = diffusion_length_um(
        final.coarse_time_days,
        [(run.diffusivity, fo_range(run.initial))],
        run.conditions,
        run.orientation,
    )
    ref_grid = Grid(length / 40, final.grid.dt_s / 5, final.grid.max_time_days)
    ref = fit_profile(*fit_args, ref_grid, None, run.diffusivity, scheme="implicit")
    chosen = fit_profile(*fit_args, final.grid, None, run.diffusivity, scheme="implicit")
    coarse = fit_profile(*fit_args, scan.grid, None, run.diffusivity, scheme="implicit")
    assert chosen.time_days == pytest.approx(ref.time_days, rel=0.01)
    assert coarse.time_days == pytest.approx(ref.time_days, rel=0.05)
    # Far fewer nodes and steps than even the explicit scan grid (5%; the explicit
    # final grid, l / dx = 240, is too slow for a test), which agrees within 5%.
    explicit = choose_grid(
        *args, purpose="scan", time_days=final.coarse_time_days, scheme="central"
    )
    assert final.grid.dx_um > 5 * explicit.grid.dx_um
    # The explicit run steps at its stable step, below its kept dt (substeps).
    explicit_step = explicit.grid.dt_s * STABILITY_RATIO / explicit.stability_ratio
    assert final.grid.dt_s > 10 * explicit_step
    explicit_fit = fit_profile(*fit_args, explicit.grid, None, run.diffusivity, scheme="central")
    assert explicit_fit.time_days == pytest.approx(ref.time_days, rel=0.05)


def test_check_grid_for_the_implicit_scheme(two_step):
    run, profile = two_step
    args = (profile, run.conditions, run.orientation, run.initial, run.diffusivity)
    # Far beyond the explicit stability limit: fine for the implicit scheme.
    big_dt = Grid(2.0, 0.15 * DAY, 200)
    # The explicit schemes substep within it (fit_profile), so it is no error either.
    assert check_grid(big_dt, *args, scheme="central") == []
    assert check_grid(big_dt, *args, scheme="implicit") == []
    assert check_grid(big_dt, *args, scheme="implicit", time_days=70) == []
    coarse_dt = replace(big_dt, dt_s=10 * DAY)
    assert any(
        "time steps up to the fit" in m
        for m in check_grid(coarse_dt, *args, scheme="implicit", time_days=70)
    )
    coarse_dx = replace(big_dt, dx_um=20.0)
    assert any(
        "may be off by up to" in m
        for m in check_grid(coarse_dx, *args, scheme="implicit", time_days=70)
    )


def test_implicit_run_from_a_table_chooses_its_grid_and_refits_identically(tmp_path):
    from forster import fit_run, scan_run, scan_summary, summary_row, write_template
    from forster.io import uncertainty_columns

    write_template(tmp_path / "t")
    params = pd.read_csv(tmp_path / "t" / "parameters.csv").assign(
        dx_um=np.nan, dt_s=np.nan, scheme="implicit", steps_sd_um=1.0, scan_points=3
    )
    path = tmp_path / "params.csv"
    params.to_csv(path, index=False)
    profiles = read_profiles(tmp_path / "t" / "profiles.csv")
    (run,) = read_runs(path)
    result = fit_run(run, profiles)
    scan = scan_run(run, profiles, time_days=result.time_days)
    row = summary_row(result) | uncertainty_columns(run) | scan_summary(scan)
    assert row["scheme"] == "implicit"
    assert row["grid_resolution"] == pytest.approx(8, rel=0.1)
    assert math.isnan(row["grid_stability"])
    again_path = tmp_path / "again.csv"
    pd.DataFrame([dict(row, profile=run.profile_name)]).to_csv(again_path, index=False)
    (again,) = read_runs(again_path)
    assert fit_run(again, profiles).time_days == result.time_days
    assert scan_run(again, profiles, time_days=result.time_days).time_days == scan.time_days
