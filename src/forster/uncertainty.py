"""Uncertainty on the fitted time from a correlated-noise likelihood on the raw points.

Replaces the kriged-grid reduced chi-squared rule (``engine.Best_fit_Chi2``), which
is kept as ``fit_profile(..., uncertainty="legacy")``. Why, and how the two differ:
``docs/references/fit-uncertainty.md``.

The measured points are compared with each model curve interpolated to their
positions. Their errors are modelled as white analytical noise plus a smooth,
spatially correlated misfit (the part of the profile the model cannot follow):

    C = s^2 (I + rho^2 K_L),   K_L[i, j] = exp(-(x_i - x_j)^2 / (2 L^2))

so the white noise has standard deviation ``s`` and the correlated misfit
``a = rho * s`` over a length ``L``. ``s``, ``rho`` and ``L`` are fitted by maximum
likelihood to the residuals of a first, white-noise best fit, then held fixed.
The generalised chi-squared r^T C^-1 r is evaluated for every time step, its
minimum is the best fit, and the 1-sigma range is where it rises by 1 (a profile
likelihood interval for the one fitted parameter, time).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla

# Grid searched for the noise model. rho = a / s is the ratio of correlated misfit to
# white noise; L runs from the point spacing to half the profile length.
RHO_GRID = np.concatenate([[0.0], np.geomspace(0.1, 30.0, 12)])
N_LENGTHS = 12
CHUNK_ROWS = 20_000  # time steps per block when evaluating chi2 (bounds memory)


@dataclass(frozen=True)
class NoiseModel:
    """Fitted error model for the measured points (Fo fractions and micrometres)."""

    white_sigma: float
    correlated_sigma: float
    correlation_length_um: float

    def covariance(self, x_um: np.ndarray) -> np.ndarray:
        d2 = (x_um[:, None] - x_um[None, :]) ** 2
        cov = self.white_sigma**2 * np.eye(len(x_um))
        if self.correlated_sigma > 0:
            cov += self.correlated_sigma**2 * np.exp(-0.5 * d2 / self.correlation_length_um**2)
        return cov


def interpolation_matrix(grid_x: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Matrix W with ``W @ curve == np.interp(x, grid_x, curve)`` for points inside the grid.

    The measured positions never change during a fit, so interpolating every model
    curve to them is one fixed linear map (two nonzeros per row).
    """
    j = np.clip(np.searchsorted(grid_x, x), 1, len(grid_x) - 1)
    w = (x - grid_x[j - 1]) / (grid_x[j] - grid_x[j - 1])
    W = np.zeros((len(x), len(grid_x)))
    rows = np.arange(len(x))
    W[rows, j - 1] = 1 - w
    W[rows, j] += w
    return W


def chi2_per_step(
    curves: np.ndarray, W: np.ndarray, y: np.ndarray, cov: np.ndarray | None = None
) -> np.ndarray:
    """r_k^T C^-1 r_k for every time step k, where r_k = y - W @ curves[k].

    Expanded as c^T A c - 2 b^T c + y^T C^-1 y with A = W^T C^-1 W (grid x grid), so
    the cost per step depends on the model grid, not on the number of measured
    points. ``cov=None`` means unit white noise (plain sum of squares).
    """
    if cov is None:
        Ci_W, Ci_y = W, y
    else:
        factor = sla.cho_factor(cov, lower=True)
        Ci_W, Ci_y = sla.cho_solve(factor, W), sla.cho_solve(factor, y)
    A = W.T @ Ci_W
    b = W.T @ Ci_y
    const = y @ Ci_y
    out = np.empty(len(curves))
    for start in range(0, len(curves), CHUNK_ROWS):
        block = curves[start : start + CHUNK_ROWS]
        out[start : start + CHUNK_ROWS] = (
            np.einsum("kn,kn->k", block @ A, block) - 2 * block @ b + const
        )
    return out


def fit_noise_model(x_um: np.ndarray, residuals: np.ndarray) -> NoiseModel:
    """Maximum-likelihood ``s``, ``rho`` and ``L`` for residuals at the best fit.

    For C = s^2 B with B = I + rho^2 K_L, the best s^2 is r^T B^-1 r / n in closed
    form, so only (rho, L) is searched on a grid.
    """
    n = len(x_um)
    spacing = float(np.median(np.diff(np.sort(x_um))))
    span = float(x_um.max() - x_um.min())
    lengths = np.geomspace(max(spacing, 1e-3), max(span / 2, 2 * spacing), N_LENGTHS)
    d2 = (x_um[:, None] - x_um[None, :]) ** 2
    # rho = 0 (white noise only) does not depend on L, so it is tried once.
    s2 = max(float(residuals @ residuals) / n, 1e-30)
    best = (0.5 * n * np.log(s2), s2, 0.0, 0.0)
    for L in lengths:
        K = np.exp(-0.5 * d2 / L**2)
        for rho in RHO_GRID[1:]:
            chol = np.linalg.cholesky(np.eye(n) + rho**2 * K)
            z = sla.solve_triangular(chol, residuals, lower=True)
            s2 = max(float(z @ z) / n, 1e-30)
            nll = 0.5 * n * np.log(s2) + np.log(np.diag(chol)).sum()
            if nll < best[0]:
                best = (nll, s2, rho, L)
    _, s2, rho, L = best
    if rho == RHO_GRID[-1] or (rho > 0 and L in (lengths[0], lengths[-1])):
        warnings.warn(
            f"noise model at the edge of its search grid (rho={rho:.3g}, L={L:.3g} um); "
            "the uncertainty may be underestimated",
            stacklevel=3,
        )
    s = float(np.sqrt(s2))
    return NoiseModel(
        white_sigma=s,
        correlated_sigma=float(rho * s),
        correlation_length_um=float(L) if rho > 0 else 0.0,
    )


@dataclass(frozen=True)
class Interval:
    """Best-fit row and the 1-sigma range in time steps (fractional, interpolated)."""

    best: int
    low: float
    high: float
    low_row: int
    high_row: int


def delta_chi2_interval(chi2: np.ndarray, delta: float = 1.0) -> Interval:
    """Rows where chi2 <= min + delta, with the ends interpolated to the crossing.

    Takes the outermost rows under the threshold, as the legacy rule did, so a
    second minimum within delta widens the range rather than being ignored. If
    the range reaches the first or last time step it stops there, with a warning.
    """
    best = int(np.argmin(chi2))
    level = chi2[best] + delta
    inside = np.flatnonzero(chi2 <= level)
    lo_row, hi_row = int(inside[0]), int(inside[-1])

    def crossing(inner: int, outer: int) -> float:
        c_in, c_out = chi2[inner], chi2[outer]
        return inner + (outer - inner) * (level - c_in) / (c_out - c_in)

    if lo_row > 0:
        low = crossing(lo_row, lo_row - 1)
    else:
        low = 0.0
        if best > 0:
            warnings.warn("1-sigma range reaches time zero", stacklevel=2)
    if hi_row < len(chi2) - 1:
        high = crossing(hi_row, hi_row + 1)
    else:
        high = float(hi_row)
        warnings.warn(
            "1-sigma range reaches the last modelled time step; increase max_time_days",
            stacklevel=2,
        )
    return Interval(best=best, low=low, high=high, low_row=lo_row, high_row=hi_row)


@dataclass(frozen=True)
class CorrelatedFit:
    chi2: np.ndarray
    interval: Interval
    noise: NoiseModel
    x_um: np.ndarray
    fo: np.ndarray


def fit_correlated(
    curves: np.ndarray, model_x_um: np.ndarray, x_um: np.ndarray, fo: np.ndarray
) -> CorrelatedFit:
    """Best fit and 1-sigma range in time from the measured points.

    Points outside the model domain or with missing Fo are left out.
    """
    keep = (x_um >= model_x_um[0]) & (x_um <= model_x_um[-1]) & np.isfinite(fo)
    x, y = x_um[keep], fo[keep]
    if len(x) < 3:
        raise ValueError(f"need at least 3 measured points inside the model domain, got {len(x)}")
    W = interpolation_matrix(model_x_um, x)
    first = int(np.argmin(chi2_per_step(curves, W, y)))
    noise = fit_noise_model(x, y - W @ curves[first])
    chi2 = chi2_per_step(curves, W, y, noise.covariance(x))
    return CorrelatedFit(chi2, delta_chi2_interval(chi2), noise, x, y)
