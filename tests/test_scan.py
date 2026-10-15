"""forster.scan: fitting over candidate initial conditions.

A profile made with the step 3 um away from the nominal position is fitted with the
nominal step (biased, range misses the true time: docs/references/likelihood-method.md,
section 4) and with a scan of step positions, whose ranges should contain it.
"""

from dataclasses import replace

import numpy as np
import pytest
from conftest import ROUND_ROBIN_DATA, ROUND_ROBIN_PARAMS, require_round_robin

from forster import fit_run, read_profiles, read_runs
from forster.model import Profile
from forster.scan import (
    Uncertainty,
    candidate_grid,
    fit_candidates,
    run_candidates,
    scan_grid,
    scan_initial_conditions,
    step_offsets,
)
from forster.uncertainty import interpolation_matrix, interval_threshold

SHIFT_UM = 3.0
OFFSETS = range(-6, 7)


@pytest.fixture(scope="module")
def setup():
    require_round_robin()
    run = {r.name: r for r in read_runs(ROUND_ROBIN_PARAMS)}["KS20-527_2-transect-1"]
    profiles = read_profiles(ROUND_ROBIN_DATA)
    shifted = replace(run.initial, steps_um=(run.initial.steps_um[0] + SHIFT_UM,))
    truth = fit_run(replace(run, initial=shifted), profiles)
    k = truth.indices[0]
    x = truth.fit_x_um
    assert x is not None
    model = interpolation_matrix(truth.model_x_um, x) @ truth.curves[k]
    return run, x, model, k, truth.dt_s


def test_step_offsets(setup):
    run = setup[0]
    moved = step_offsets(run.initial, [-2, 0, 5])
    assert [c.steps_um for c in moved] == [(39,), (41,), (46,)]
    assert {c.domain_um for c in moved} == {run.initial.domain_um}
    with pytest.raises(ValueError, match="outside the domain"):
        step_offsets(run.initial, [-50])


def test_scan_finds_the_step_and_covers_the_true_time(setup):
    run, x, model, k, dt = setup
    y = model + np.random.default_rng(1).normal(0, 0.002, len(x))
    calls = []
    scan = scan_initial_conditions(
        Profile(x, y),
        run.conditions,
        run.orientation,
        step_offsets(run.initial, OFFSETS),
        run.grid,
        progress=lambda i, n: calls.append((i, n)),
    )
    assert calls == [(i, len(OFFSETS)) for i in range(1, len(OFFSETS) + 1)]
    assert abs(scan.best_initial.steps_um[0] - (run.initial.steps_um[0] + SHIFT_UM)) <= 1
    true_days = k * dt / 86400
    assert scan.profile_days[0] <= true_days <= scan.profile_days[1]
    assert scan.marginal_days[0] <= true_days <= scan.marginal_days[1]
    assert scan.threshold == pytest.approx(interval_threshold(len(x), 2))
    assert scan.candidate_probability.sum() == pytest.approx(1.0)


def test_scan_coverage_with_a_misplaced_step(setup):
    """Over many noise draws, both scan ranges contain the true time far more often
    than the nominal-step range (22% in the validation)."""
    run, x, model, k, _ = setup
    grid = replace(run.grid, max_time_days=3 * k * run.grid.dt_s / 86400)
    runs = run_candidates(
        Profile(x, model),
        run.conditions,
        run.orientation,
        step_offsets(run.initial, OFFSETS),
        grid,
        stop_early=False,
    )
    rng = np.random.default_rng(20261013)
    trials, prof_hits, marg_hits = 150, 0, 0
    for _ in range(trials):
        noisy = replace(runs, fo=model + rng.normal(0, 0.002, len(x)))
        scan = fit_candidates(noisy)
        prof_hits += scan.profile.low <= k <= scan.profile.high
        marg_hits += scan.marginal.low <= k <= scan.marginal.high
    assert prof_hits / trials >= 0.55
    assert marg_hits / trials >= 0.55


def test_candidates_must_share_the_domain(setup):
    run, x, model, _, _ = setup
    other = replace(run.initial, domain_um=(0, 250))
    with pytest.raises(ValueError, match="share the model domain"):
        run_candidates(
            Profile(x, model), run.conditions, run.orientation, [run.initial, other], run.grid
        )


def test_warns_when_the_best_candidate_is_at_the_end_of_the_scan(setup):
    run, x, model, _, _ = setup
    one_sided = step_offsets(run.initial, range(-6, 1))  # truth is at +3
    with pytest.warns(UserWarning, match="last of the scan"):
        scan_initial_conditions(
            Profile(x, model), run.conditions, run.orientation, one_sided, run.grid
        )


# Uncertainties on each parameter -----------------------------------------------------


def test_candidate_grid_gaussian_and_uniform(setup):
    run = setup[0]
    grid = candidate_grid(
        run.initial,
        {"step1_um": Uncertainty(2.0), "edge_fo": Uncertainty(0.004, "uniform")},
        points=5,
    )
    assert grid.names == ("step1_um", "edge_fo")
    np.testing.assert_allclose(grid.nodes["step1_um"], 41 + np.linspace(-5, 5, 5))  # +-2.5 sd
    np.testing.assert_allclose(
        grid.nodes["edge_fo"], run.initial.edge_fo + np.linspace(-0.004, 0.004, 5)
    )
    assert len(grid.candidates) == 25 and grid.dropped == 0
    # Gaussian penalty ((v - v0) / sd)^2, none for the uniform parameter.
    i = next(
        j for j, v in enumerate(grid.values) if v[0] == 46.0 and v[1] == grid.nodes["edge_fo"][0]
    )
    assert grid.penalty[i] == pytest.approx(2.5**2)
    assert grid.penalty.min() == 0.0


def test_candidate_grid_drops_invalid_nodes_and_caps_its_size():
    from forster import InitialCondition

    initial = InitialCondition((0, 100), 0.82, 0.88, steps_um=(20, 25), step_fo=0.85)
    grid = candidate_grid(
        initial, {"step1_um": Uncertainty(4.0), "step2_um": Uncertainty(4.0)}, points=5
    )
    assert all(c.steps_um[0] < c.steps_um[1] for c in grid.candidates)
    assert grid.dropped > 0 and len(grid.candidates) + grid.dropped == 25
    with pytest.raises(ValueError, match="more than 100"):
        candidate_grid(
            initial,
            {"step1_um": Uncertainty(1.0), "core_fo": Uncertainty(0.01)},
            points=11,
            max_candidates=100,
        )
    with pytest.raises(ValueError, match="no parameter"):
        candidate_grid(
            InitialCondition((0, 100), 0.82, 0.88, steps_um=(20,)), {"step_fo": Uncertainty(0.01)}
        )


def test_scan_grid_raises_dt_but_not_dx(setup):
    run = setup[0]
    fast = scan_grid(run.grid, [run.initial], run.conditions, run.orientation)
    assert fast.dx_um == run.grid.dx_um
    assert fast.dt_s >= run.grid.dt_s
    from forster.model import fe_mg_diffusivity

    D, _ = fe_mg_diffusivity("dohmen_chakraborty2007").functions(run.conditions, run.orientation)
    d_max = float(D(np.array([min(run.initial.edge_fo, run.initial.core_fo)]))[0])
    assert d_max * fast.dt_s / (fast.dx_um * 1e-6) ** 2 == pytest.approx(0.4, rel=0.02)


def test_two_parameter_scan_covers_step_and_edge_errors(setup):
    """Data made with the step 3 um and the edge 0.002 off; uniform uncertainties that
    contain them. Measured 69% (profile) and 67% (marginal) over 150 draws."""
    run, x, _, _, _ = setup
    grid_c = candidate_grid(
        run.initial,
        {"step1_um": Uncertainty(6.0, "uniform"), "edge_fo": Uncertainty(0.004, "uniform")},
        points=7,
    )
    truth_initial = replace(run.initial, steps_um=(44.0,), edge_fo=run.initial.edge_fo + 0.002)
    grid = scan_grid(
        replace(run.grid, max_time_days=130),
        (*grid_c.candidates, truth_initial),
        run.conditions,
        run.orientation,
    )
    runs = run_candidates(
        Profile(x, np.full(len(x), 0.85)),
        run.conditions,
        run.orientation,
        (*grid_c.candidates, truth_initial),
        grid,
        stop_early=False,
    )
    k = round(42.2 * 86400 / grid.dt_s)
    truth = runs.at_points[-1][k]
    candidates = replace(
        runs,
        candidates=runs.candidates[:-1],
        at_points=runs.at_points[:-1],
        complete=runs.complete[:-1],
    )
    rng = np.random.default_rng(20261013)
    trials, prof_hits, marg_hits = 100, 0, 0
    for _ in range(trials):
        scan = fit_candidates(
            replace(candidates, fo=truth + rng.normal(0, 0.002, len(x))), prior=grid_c
        )
        prof_hits += scan.profile.low <= k <= scan.profile.high
        marg_hits += scan.marginal.low <= k <= scan.marginal.high
    assert prof_hits / trials >= 0.55
    assert marg_hits / trials >= 0.55


def test_warns_when_a_scanned_parameter_is_best_at_its_limit(setup):
    run, x, model, _, _ = setup  # truth: step 3 um past nominal
    narrow = candidate_grid(run.initial, {"step1_um": Uncertainty(1.0, "uniform")}, points=3)
    with pytest.warns(UserWarning, match="best step1_um"):
        scan_initial_conditions(
            Profile(x, model), run.conditions, run.orientation, narrow, run.grid
        )
