# ruff: noqa: E501
"""Exact oracle for nonlinear diffusion from a step: the Boltzmann similarity solution.

A step from `left` (x < x0) to `right` (x > x0) in an infinite medium with D = D(C) has the similarity solution
C(eta), eta = (x - x0) / (2 sqrt(t)), which satisfies -2 eta C' = (D(C) C')'. It is solved with scipy's solve_bvp
to ~1e-10 and used as the reference for numerical schemes (observed convergence order) and for mass checks.

Copied from ~/.claude/skills/scientific-testing/scripts/similarity_oracle.py (2026-11-17), with a ruff E501 waiver for its long lines and `ruff format` applied.

Run `python similarity_oracle.py` to print the constant-D control (the solution must equal the erfc solution).
"""

import math
from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.integrate import solve_bvp
from scipy.special import erfc


def similarity(
    D: Callable[[NDArray], NDArray],
    left: float,
    right: float,
    xi_max: float = 12.0,
    tol: float = 1e-10,
) -> tuple[Callable[[NDArray], NDArray], float]:
    """Exact C(eta) for dC/dt = d/dx(D(C) dC/dx) from a step, eta = (x - x0) / (2 sqrt(t)).

    D is a vectorized callable of C. Returns (C_of_eta, D_ref) with D_ref = D at the mean of left and right.
    Internally the solve uses xi = eta / sqrt(D_ref) on [-xi_max, xi_max]; eta has units of sqrt(D t).
    """
    d_ref = float(np.asarray(D(np.array([(left + right) / 2])))[0])

    def rhs(xi, y):  # y0 = C, y1 = D(C) C' / D_ref (scaled flux)
        d = np.asarray(D(y[0])) / d_ref
        dc = y[1] / d
        return np.vstack([dc, -2 * xi * dc])

    def bc(ya, yb):
        return np.array([ya[0] - left, yb[0] - right])

    xi = np.linspace(-xi_max, xi_max, 2001)
    guess_c = left + (right - left) * 0.5 * erfc(-xi)
    guess = np.vstack([guess_c, (right - left) / math.sqrt(math.pi) * np.exp(-(xi**2))])
    sol = solve_bvp(rhs, bc, xi, guess, tol=tol, max_nodes=200_000)
    if not sol.success:
        raise RuntimeError(sol.message)

    def c_of_eta(eta: ArrayLike) -> NDArray:
        return sol.sol(np.asarray(eta, float) / math.sqrt(d_ref))[0]

    return c_of_eta, d_ref


def step_profile(
    D: Callable[[NDArray], NDArray], left: float, right: float, x: ArrayLike, x0: float, t: float
) -> NDArray:
    """C at positions x after time t for a step at x0; x, x0 in the length units of sqrt(D t)."""
    c_of_eta, _ = similarity(D, left, right)
    return c_of_eta((np.asarray(x, float) - x0) / (2 * math.sqrt(t)))


def observed_order(dxs: ArrayLike, errs: ArrayLike) -> NDArray:
    """Observed order log(e_i / e_{i+1}) / log(dx_i / dx_{i+1}) for each consecutive pair of refinements."""
    dxs = np.asarray(dxs, float)
    errs = np.asarray(errs, float)
    return np.log(errs[:-1] / errs[1:]) / np.log(dxs[:-1] / dxs[1:])


def assert_order(dxs: ArrayLike, errs: ArrayLike, expected: float, tol: float = 0.15) -> None:
    """Raise AssertionError (with a small table) if any observed order is outside expected +- tol."""
    orders = observed_order(dxs, errs)
    if np.all(np.abs(orders - expected) <= tol):
        return
    lines = [
        f"observed order outside {expected} +- {tol}",
        f"{'dx':>12} {'error':>12} {'order':>8}",
    ]
    for i, (dx, err) in enumerate(
        zip(np.asarray(dxs, float), np.asarray(errs, float), strict=False)
    ):
        order = "" if i == 0 else f"{orders[i - 1]:.2f}"
        lines.append(f"{dx:12.4g} {err:12.3e} {order:>8}")
    raise AssertionError("\n".join(lines))


def mass_change(
    c: ArrayLike, c0: ArrayLike, widths: ArrayLike, left: float, right: float, length_scale: float
) -> float:
    """Relative mass change sum((c - c0) * widths) / (|right - left| * length_scale).

    Use length_scale = 2 * sqrt(D_ref * t), the diffusion length, so the result is the mass moved in units of
    the step's own mass scale. A conservative scheme (or the exact profile with the step at the Matano plane x0)
    gives ~0.
    """
    delta = (np.asarray(c, float) - np.asarray(c0, float)) * np.asarray(widths, float)
    return float(np.sum(delta) / (abs(right - left) * length_scale))


if __name__ == "__main__":
    const = 1e-17

    def d_const(c):
        return np.full_like(np.asarray(c, float), const)

    c_of_eta, _ = similarity(d_const, 0.82, 0.88)
    eta = np.linspace(-5e-8, 5e-8, 101)
    exact = 0.82 + 0.06 * 0.5 * erfc(-eta / math.sqrt(const))
    print(
        f"control: similarity vs erfc, constant D: max diff {np.max(np.abs(c_of_eta(eta) - exact)):.1e}"
    )
