"""Implicit finite differences with the boundary values held on the end nodes.

The explicit schemes (`forster.engine`, `forster.dipra`) hold the rim and core values
one node outside the profile and snap each step of the initial condition to the
nearest node; both make the fitted time first order in dx (about 2.4 dx / l
against a converged reference, ``docs/references/grid-choice.md``). Here

- the grid has nodes exactly on both ends of the model domain and on every growth
  stage's rim (`node_grid`; uniform between those nodes), and the rim and core values
  are held on the end nodes;
- the start is the step function averaged over each node's cell (`cell_average`), so
  a step between nodes is not moved to one;
- time steps are TR-BDF2 (Bank et al. 1985; Hosea and Shampine 1996): a trapezoidal
  stage to t + gamma dt, then BDF2 to t + dt, with gamma = 2 - sqrt(2). It is
  L-stable, so the steps of the initial condition do not ring, and one-step, so a run
  restarted from any row continues exactly. D is evaluated at the start of each stage
  (linearly implicit), and each stage is one tridiagonal solve. That makes it first
  order in time when D depends on Fo (second order only for a constant D): with the
  DC07 D on a 0.80/0.90 step the error halves with dt (9e-5 Fo at 10 steps over 2 days,
  ``test_time_order_with_an_fo_dependent_d``). Its effect on the fitted time is
  measured in the next paragraph.

On six real profiles the fitted time is within 0.7% of the converged one at
l / dx = 5 and 0.16% at l / dx = 8, and within 0.5% with 20 time steps up to the fit
(``docs/references/grid-choice.md``). Both changes are needed: at l / dx = 8, averaging
the start with the ghost-node boundary left errors up to 12%, and the boundary on a
node with snapped steps up to 22%.

The rim value may change in time: a function of the time in seconds since the start
of the run, evaluated at each stage's end. The far end is held (the core value) or, with
``core=None``, free: no flux crosses it, as at a crystal's centre (the profile mirrored
there). Its node then has a half cell, like the held ends.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence

import numpy as np
from scipy.linalg import lapack

from forster.boundary import Rim, rim_at

GAMMA = 2.0 - math.sqrt(2.0)
(_gtsv,) = lapack.get_lapack_funcs(("gtsv",), dtype=np.float64)  # dgtsv


def node_grid(nodes_um: Sequence[float], dx_um: float) -> np.ndarray:
    """Grid through every one of ``nodes_um`` (the domain ends and any stage rims), even
    between consecutive ones with spacing at most ``dx_um``."""
    fixed = np.unique(np.asarray(nodes_um, dtype=float))
    if len(fixed) < 2:
        raise ValueError(f"the model domain needs two distinct ends, got {list(nodes_um)}")
    pieces = []
    for a, b in itertools.pairwise(fixed):
        intervals = max(1, math.ceil((b - a) / dx_um - 1e-9))
        pieces.append(np.linspace(a, b, intervals + 1)[:-1])
    x = np.concatenate([*pieces, fixed[-1:]])
    if len(x) < 3:
        x = np.linspace(fixed[0], fixed[-1], 3)
    return x


def cell_bounds(x_um: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Each node's cell: halfway to its neighbours, half cells at the two ends."""
    mid = 0.5 * (x_um[1:] + x_um[:-1])
    return np.concatenate([x_um[:1], mid]), np.concatenate([mid, x_um[-1:]])


def cell_average(
    x_um: np.ndarray, breaks_um: Sequence[float], values: Sequence[float]
) -> np.ndarray:
    """A step function averaged over each node's cell (`cell_bounds`).

    ``values[i]`` holds between ``breaks_um[i - 1]`` and ``breaks_um[i]``: there is one
    more value than breaks, the first before the first break and the last after the last.
    """
    if len(values) != len(breaks_um) + 1:
        raise ValueError("give one more value than breaks")
    lo, hi = cell_bounds(x_um)
    edges = [-np.inf, *breaks_um, np.inf]
    out = np.zeros_like(x_um, dtype=float)
    for value, a, b in zip(values, edges[:-1], edges[1:], strict=True):
        out += value * np.clip(np.minimum(hi, b) - np.maximum(lo, a), 0.0, None)
    return out / (hi - lo)


def _operator(
    c: np.ndarray,
    D: Callable[[np.ndarray], np.ndarray],
    dt_s: float,
    geometry,
    free_end: bool = False,
):
    """dt * A as its three diagonals (lower and upper of length n - 1, main of n), zero on
    the end rows that are held. With ``free_end`` the last row has no flux through the
    end: dc_n/dt = -F_n-1/2 / (h_n-1 / 2).

    Flux form: dc_i/dt = (F_i+1/2 - F_i-1/2) / w_i with F_i+1/2 = D_face (c_i+1 - c_i) / h_i,
    h the node spacings and w the interior cell widths (m); D_face is the mean of the
    two nodes' D.
    """
    h, w = geometry
    d = D(c)
    conductance = dt_s * 0.5 * (d[1:] + d[:-1]) / h  # per face
    lower = np.zeros(len(c) - 1)  # row i's coefficient of c[i - 1]
    lower[:-1] = conductance[:-1] / w
    upper = np.zeros(len(c) - 1)  # row i's coefficient of c[i + 1]
    upper[1:] = conductance[1:] / w
    main = np.zeros_like(c)
    main[1:-1] = -(conductance[:-1] + conductance[1:]) / w
    if free_end:
        half = 0.5 * h[-1]
        lower[-1] = conductance[-1] / half
        main[-1] = -conductance[-1] / half
    return lower, main, upper


def _apply(lower, main, upper, c: np.ndarray) -> np.ndarray:
    out = main * c
    out[:-1] += upper * c[1:]
    out[1:] += lower * c[:-1]
    return out


def _solve(weight: float, rhs: np.ndarray, lower, main, upper, ends: tuple[float, float | None]):
    """Solve (I - weight * dt A) c = rhs with c held at ``ends`` on the end nodes (a far
    end of None is free: its row of A is the no-flux row).

    LAPACK's tridiagonal solver directly: half the cost of a step through
    `scipy.linalg.solve_banded` (whose checks dominate at these sizes), same result."""
    rhs = rhs.copy()
    rhs[0] = ends[0]
    if ends[1] is not None:
        rhs[-1] = ends[1]
    *_, c, info = _gtsv(-weight * lower, 1.0 - weight * main, -weight * upper, rhs, 1, 1, 1, 1)
    if info != 0:
        raise np.linalg.LinAlgError(f"tridiagonal solve failed (LAPACK gtsv info {info})")
    return c


def tr_bdf2_step(
    c: np.ndarray,
    D: Callable[[np.ndarray], np.ndarray],
    dt_s: float,
    geometry: tuple[np.ndarray, np.ndarray],
    ends_mid: tuple[float, float | None],
    ends: tuple[float, float | None],
) -> np.ndarray:
    """One TR-BDF2 step; ``ends_mid`` and ``ends`` are the held values at t + gamma dt
    and t + dt (a far end of None is free); ``geometry`` = `grid_geometry` of the grid."""
    free = ends[1] is None
    lower, main, upper = _operator(c, D, dt_s, geometry, free)
    half = GAMMA / 2
    mid = _solve(half, c + half * _apply(lower, main, upper, c), lower, main, upper, ends_mid)
    lower, main, upper = _operator(mid, D, dt_s, geometry, free)
    a = 1.0 / (GAMMA * (2.0 - GAMMA))
    b = (1.0 - GAMMA) ** 2 / (GAMMA * (2.0 - GAMMA))
    weight = (1.0 - GAMMA) / (2.0 - GAMMA)
    return _solve(weight, a * mid - b * c, lower, main, upper, ends)


def grid_geometry(x_um: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Node spacings and interior cell widths, in metres."""
    h = np.diff(x_um) * 1e-6
    return h, 0.5 * (h[1:] + h[:-1])


def stepper(
    D: Callable[[np.ndarray], np.ndarray],
    rim: Rim,
    core: float | None,
    dt_s: float,
    x_um: np.ndarray,
    t0_s: float = 0.0,
):
    """``advance(c, n, first_row=0)``: rows ``first_row`` .. ``first_row + n``.

    Row j is at time t0_s + j dt_s (the clock of a callable ``rim``). The first row returned
    is ``c`` with its end values set to the rim and core values at that time. ``core``
    None: the far end is free (no flux), as at a crystal's centre.
    """
    geometry = grid_geometry(np.asarray(x_um, dtype=float))

    def advance(c: np.ndarray, n: int, first_row: int = 0) -> np.ndarray:
        rows = np.empty((n + 1, len(c)))
        c = np.array(c, dtype=float)
        t0 = t0_s + first_row * dt_s
        c[0] = rim_at(rim, t0)
        if core is not None:
            c[-1] = core
        rows[0] = c
        for j in range(n):
            t = t0 + j * dt_s
            ends_mid = (rim_at(rim, t + GAMMA * dt_s), core)
            c = tr_bdf2_step(c, D, dt_s, geometry, ends_mid, (rim_at(rim, t + dt_s), core))
            rows[j + 1] = c
        return rows

    return advance
