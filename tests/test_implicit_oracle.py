"""The implicit solver (`forster.implicit`) against answers from outside the code.

`test_implicit.py` uses uniform grids, erfc, and the solver's own fine-dt runs. On a uniform
grid the cell widths, the half cell at a free end and the order of the nodes cannot go
wrong visibly, so this file uses graded grids (spacing changing 8- to 16-fold) and
expected values the code cannot influence:

1. the exact similarity solution of a D(Fo) step (`similarity_oracle.py`, a boundary-value
   solve to 1e-10), with the observed order of convergence;
2. a free far end equals the same run on the mirrored domain with both ends held;
3. metamorphic: reverse the grid and swap the ends; scale x by k and t by k^2;
4. the times the rim function is asked for (the documented stage times) and ``t0_s``;
5. the caller's start array is not modified; `node_grid`, `cell_bounds` and `cell_average`
   against their definitions on seeded random inputs.

Mutants of ``implicit.py`` these tests kill, the existing tests do not (mutmut 3.8.1,
2026-11-17): cell widths ``w = h[1:]`` or ``h[:+1]``; the free end's half cell from
``h[0]`` or ``h[-2]``; the mid-stage rim time ``t + dt / 2``; ``t0_s`` ignored; a shifted
end of `cell_bounds`; the start array mutated in place; the ``- 1e-9`` guard of
`node_grid` dropped.
"""

import math

import numpy as np
import pytest
from similarity_oracle import assert_order, mass_change, step_profile

from forster import DIFFUSIVITY_PRESETS, Conditions, Orientation
from forster.implicit import cell_average, cell_bounds, node_grid, stepper

CONDITIONS = Conditions(T_C=1200, P_Pa=4.2e7, fO2_Pa=0.00138)
D_FO, _ = DIFFUSIVITY_PRESETS["dohmen_chakraborty2007"].functions(
    CONDITIONS, Orientation(90, 90, 0)
)
LEFT, RIGHT = 0.80, 0.90  # Fo on either side of the step: D differs by 1.4x across it
GAMMA = 2.0 - math.sqrt(2.0)  # TR-BDF2 (Bank et al. 1985), written out here, not imported


def graded_side(width_um: float, grading: float, h0_um: float) -> np.ndarray:
    """Distances from the step of the nodes on one side of it, the spacing growing by the
    factor ``grading`` over ``width_um`` from about ``h0_um``."""
    # s(xi) = w (g^xi - 1) / (g - 1): the first spacing is w ln(g) / ((g - 1) n); set it to h0
    n = max(2, round(width_um / (h0_um * (grading - 1) / math.log(grading))))
    xi = (np.arange(n) + 0.5) / n
    return width_um * (grading**xi - 1) / (grading - 1)


def graded_grid(h0_um: float, x0_um: float = 20.0):
    """Nodes on [x0 - 16, x0 + 24] um, fine at the step x0 and coarser away from it: 8-fold
    on the left, 16-fold on the right. The step falls inside a cell (the two sides' first
    nodes are not equally far), which `cell_average` handles; the oracle compares nodal values."""
    left = graded_side(16.0, 8.0, h0_um)
    right = graded_side(24.0, 16.0, h0_um)
    return np.concatenate([x0_um - left[::-1], x0_um + right]), x0_um


def run_step(h0_um: float, steps: int = 800, t_s: float = 7200.0):
    x, x0 = graded_grid(h0_um)
    start = cell_average(x, [x0], [LEFT, RIGHT])
    final = stepper(D_FO, LEFT, RIGHT, t_s / steps, x)(start, steps)[-1]
    exact = step_profile(D_FO, LEFT, RIGHT, (x - x0) * 1e-6, 0.0, t_s)
    return x, x0, start, final, exact


def test_graded_grid_is_graded_as_claimed():
    x, x0 = graded_grid(0.4)
    h = np.diff(x)
    assert h.max() / h.min() > 8  # measured 11.9: the spacing grows 8-fold and 16-fold per side
    lo, hi = cell_bounds(x)
    assert ((lo < x0) & (x0 < hi)).sum() == 1  # the step is inside one cell, which gets a mean


def test_step_matches_the_exact_similarity_solution_at_second_order():
    """Exact C(eta) of dC/dt = d/dx(D(C) dC/dx) from a step (Boltzmann; solve_bvp to 1e-10)
    against the solver on graded grids refined 1x, 2x, 4x. D(Fo) is DC07 at 1200 C, 2.2e-16
    to 4.4e-16 m2/s, diffusion length 2 sqrt(D t) = 3 um in 2 h, ends 5 lengths away."""
    h0s = [0.4, 0.2, 0.1]
    errors, masses = [], []
    for h0 in h0s:
        x, _x0, start, final, exact = run_step(h0)
        errors.append(np.abs(final - exact).max())
        lo, hi = cell_bounds(x)
        length_um = 2 * math.sqrt(3.1e-16 * 7200.0) * 1e6  # D at the mean Fo
        masses.append(mass_change(final, start, hi - lo, LEFT, RIGHT, length_um))
    # Measured 2026-11-17: 6.4e-4, 1.5e-4, 3.8e-5 Fo (ratios 4.2, 4.0). Order 2 +- 0.3: the
    # observed orders 2.06, 2.01 plus the scatter of a graded grid. Absolute bounds are
    # 1.5x the measured errors; w = h[1:] or h[:+1] gives 2x or more.
    assert_order(h0s, errors, 2.0, tol=0.3)
    assert errors[0] < 1.0e-3
    assert errors[1] < 2.3e-4
    assert errors[2] < 5.8e-5
    # A conservative scheme moves no mass (the step is the Matano plane x0). Measured:
    # -5e-7, -8e-9, -3e-10 (the start's cell averages are not the exact start, and fall 4x
    # to 25x per refinement); bound 4x the coarsest. The held ends' flux is exp(-25)-small.
    assert max(abs(m) for m in masses) < 2e-6


def test_free_end_equals_the_mirrored_domain_with_both_ends_held():
    """Free far end at L on a graded grid [0, L] = the same grid mirrored to [0, 2L] with
    the rim value held at both ends (no flux across the middle by symmetry). The free end's
    half cell is h[-1] / 2; D depends on Fo, so a wrong cell width shows."""
    t_s, steps, length = 20000.0, 100, 30.0
    xi = np.linspace(0.0, 1.0, 41)
    fine_at_rim = length * (np.exp(2.2 * xi) - 1) / (math.exp(2.2) - 1)  # spacing grows 9-fold
    for grid in (fine_at_rim, length - fine_at_rim[::-1]):  # then fine at the free end
        assert np.diff(grid).max() / np.diff(grid).min() > 8
        step_um = 26.0
        free_start = cell_average(grid, [step_um], [0.80, 0.90])
        free = stepper(D_FO, 0.80, None, t_s / steps, grid)(free_start, steps)
        full_x = np.concatenate([grid, 2 * length - grid[-2::-1]])
        full_start = cell_average(full_x, [step_um, 2 * length - step_um], [0.80, 0.90, 0.80])
        full = stepper(D_FO, 0.80, 0.80, t_s / steps, full_x)(full_start, steps)
        assert abs(free[-1, -1] - 0.90) > 1e-3  # the profile does reach the free end
        # Same tridiagonal system up to the order of rounding: 1e-11 Fo is 1e-8 of the signal.
        np.testing.assert_allclose(free, full[:, : len(grid)], rtol=0, atol=1e-11)


def random_graded_grid(rng, n=60, length_um=30.0):
    h = np.exp(rng.uniform(0.0, math.log(10.0), n))  # spacings within 10-fold, in no order
    return np.concatenate([[0.0], np.cumsum(h)]) * length_um / h.sum()


def test_reversing_the_grid_and_swapping_the_ends_reverses_the_profile():
    rng = np.random.default_rng(20261117)
    x = random_graded_grid(rng)
    start = cell_average(x, [9.0, 17.5], [0.84, 0.90, 0.82])
    t_s, steps = 5e4, 40
    forward = stepper(D_FO, 0.84, 0.82, t_s / steps, x)(start, steps)
    x_rev = x[-1] - x[::-1]
    backward = stepper(D_FO, 0.82, 0.84, t_s / steps, x_rev)(start[::-1].copy(), steps)
    assert np.ptp(forward[-1] - start) > 0.01  # the profile moves
    # Mirror image of the same finite volume system; only rounding differs (measured 1e-16).
    np.testing.assert_allclose(backward[:, ::-1], forward, rtol=0, atol=1e-13)


def test_scaling_x_by_k_and_t_by_k_squared_gives_the_same_profile():
    rng = np.random.default_rng(7)
    x = random_graded_grid(rng)
    start = cell_average(x, [12.0], [0.80, 0.90])
    t_s, steps, k = 3e4, 30, 3.0
    base = stepper(D_FO, 0.80, 0.90, t_s / steps, x)(start, steps)
    scaled = stepper(D_FO, 0.80, 0.90, k**2 * t_s / steps, k * x)(start, steps)
    assert np.ptp(base[-1] - start) > 0.01
    # The equation is invariant under (x, t) -> (kx, k^2 t); only rounding differs (1e-16).
    np.testing.assert_allclose(scaled, base, rtol=0, atol=1e-13)


def test_rim_function_is_asked_for_at_the_documented_stage_times():
    """Row j is at t0_s + j dt; step j asks for the rim at t0 + j dt + gamma dt, then
    t0 + (j + 1) dt, after the row's start value at t0 + j dt. A callable rim is not
    cached or called twice."""
    dt, t0_s, first_row, n = 600.0, 4321.0, 2, 3
    calls = []

    def rim(t):
        calls.append(t)
        return 0.80 + 1e-6 * t  # rises 6e-4 per step, so a wrong time moves the node

    x = np.linspace(0.0, 10.0, 11)
    start = np.full(11, 0.90)
    rows = stepper(D_FO, rim, 0.90, dt, x, t0_s=t0_s)(start, n, first_row=first_row)
    t_row = lambda j: t0_s + (first_row + j) * dt  # noqa: E731 (time of row j)
    expected = [t_row(0)]
    for j in range(n):
        expected += [t_row(j) + GAMMA * dt, t_row(j + 1)]
    np.testing.assert_allclose(calls, expected, rtol=1e-12, atol=0)
    # The held end node carries the rim at each row's own time.
    np.testing.assert_allclose(rows[:, 0], [rim(t_row(j)) for j in range(n + 1)], rtol=1e-13)


def test_start_array_is_not_modified():
    x = np.linspace(0.0, 10.0, 21)
    start = np.linspace(0.85, 0.88, 21)  # end values differ from the rim and core
    kept = start.copy()
    rows = stepper(D_FO, 0.80, 0.90, 100.0, x)(start, 5)
    np.testing.assert_array_equal(start, kept)
    assert rows[0, 0] == 0.80 and rows[0, -1] == 0.90  # the first row has the ends set


def test_node_grid_has_every_given_node_and_exact_multiples_are_not_split_again():
    rng = np.random.default_rng(11)
    for _ in range(300):
        dx = float(rng.uniform(0.05, 3.0))
        start = float(rng.uniform(-50.0, 50.0))
        counts = rng.integers(1, 40, size=3)  # intervals expected in each gap
        fractions = rng.choice([0.0, 0.0, 0.01, 0.5, 0.99], size=3)  # 0: an exact multiple
        edges, intervals = [start], 0
        for k, frac in zip(counts, fractions, strict=True):
            edges.append(edges[-1] + (k + frac) * dx)
            intervals += k + (frac > 0)
        x = node_grid(edges, dx)
        assert len(x) == intervals + 1, (dx, edges)
        assert np.all(np.diff(x) > 0)
        assert np.diff(x).max() <= dx * (1 + 1e-9)
        assert all(np.any(x == e) for e in edges)  # the given nodes are exactly on the grid
    # 3 * 0.1 = 0.30000000000000004 and /0.1 is 3.0000000000000004: still three intervals.
    assert len(node_grid([0.0, 3 * 0.1], 0.1)) == 4


def test_cell_bounds_are_halfway_with_half_cells_at_the_ends():
    rng = np.random.default_rng(3)
    for _ in range(50):
        x = np.sort(rng.uniform(0.0, 100.0, rng.integers(3, 40)))
        lo, hi = cell_bounds(x)
        assert lo[0] == x[0] and hi[-1] == x[-1]
        np.testing.assert_allclose(lo[1:], (x[1:] + x[:-1]) / 2, rtol=1e-14)
        np.testing.assert_array_equal(hi[:-1], lo[1:])
        assert np.all(lo <= x) and np.all(x <= hi)
        assert (hi - lo).sum() == pytest.approx(x[-1] - x[0], rel=1e-12)


def test_cell_average_equals_the_step_function_integrated_over_each_cell():
    rng = np.random.default_rng(5)
    for _ in range(50):
        x = np.sort(rng.uniform(0.0, 100.0, rng.integers(3, 40)))
        breaks = np.sort(rng.uniform(-10.0, 110.0, rng.integers(1, 5)))
        values = rng.uniform(0.7, 0.95, len(breaks) + 1)
        got = cell_average(x, breaks.tolist(), values.tolist())
        lo, hi = cell_bounds(x)
        for g, a, b in zip(got, lo, hi, strict=True):
            # Integrate piece by piece, taking each piece's value at its midpoint.
            cuts = np.concatenate([[a], breaks[(breaks > a) & (breaks < b)], [b]])
            mids = 0.5 * (cuts[1:] + cuts[:-1])
            integral = np.sum(values[np.searchsorted(breaks, mids)] * np.diff(cuts))
            assert g == pytest.approx(integral / (b - a), abs=1e-12)


def test_node_grid_edge_cases():
    assert node_grid([0.0, 1.0, 5.0], 10.0).tolist() == [0.0, 1.0, 5.0]  # three nodes stay as given
    assert node_grid([2.0, 10.0], 100.0).tolist() == [2.0, 6.0, 10.0]  # two ends get a middle node
    with pytest.raises(ValueError, match=r"two distinct ends, got \[1.0, 1.0\]"):
        node_grid([1.0, 1.0], 1.0)


def test_cell_average_needs_one_more_value_than_breaks():
    x = np.linspace(0.0, 10.0, 11)
    with pytest.raises(ValueError, match="one more value than breaks"):
        cell_average(x, [5.0], [0.8])
    assert cell_average(np.arange(11), [5], [1, 2]).dtype == float  # integer input too


def test_the_clock_of_a_callable_rim_starts_at_zero_by_default():
    calls = []
    rim = lambda t: calls.append(t) or 0.8  # noqa: E731
    stepper(D_FO, rim, 0.9, 100.0, np.linspace(0.0, 10.0, 11))(np.full(11, 0.9), 1)
    assert calls[0] == 0.0
