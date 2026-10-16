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
The generalised chi-squared r^T C^-1 r is evaluated for every time step and its
minimum is the best fit. The 1-sigma range is where it rises by
`interval_threshold`: 1 for many points, more for few, because ``s`` is estimated
from the same residuals (the Student-t small-sample correction). This is a profile
likelihood interval for the one fitted parameter, time. `posterior_interval` gives
the Bayesian alternative, with ``s`` integrated out.

Method, validation and references: ``docs/references/likelihood-method.md``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla
from scipy import stats

# Grid searched for the noise model. rho = a / s is the ratio of correlated misfit to
# white noise; L runs from the point spacing to half the profile length.
RHO_GRID = np.concatenate([[0.0], np.geomspace(0.1, 30.0, 12)])
N_LENGTHS = 12
CHUNK_ROWS = 20_000  # time steps per block when evaluating chi2 (bounds memory)
ONE_SIGMA = 0.6827  # probability inside a 1-sigma range
TWO_SIGMA = 0.9545  # and a 2-sigma range
WARN_RHO = 0.5  # warn of a profile-wide misfit only if it is at least half the white noise


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
    # Only a substantial profile-wide misfit predicts a range that is too narrow. In the
    # validation (docs/references/likelihood-method.md), trials with L at the upper end
    # (half the profile) and rho >= 0.5 covered 50-55%, the rest 62-67%; L at the lower
    # end (the point spacing: misfit too short to resolve) is harmless. On pure white
    # noise this fires in about 1% of fits.
    if rho == RHO_GRID[-1] or (rho >= WARN_RHO and lengths[-1] == L):
        warnings.warn(
            f"the misfit spans the profile (correlated noise {rho * np.sqrt(s2) * 100:.2g} Fo "
            f"over L={L:.3g} um, half the profile): check the initial and boundary "
            "conditions; the range may be too narrow",
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


def interval_threshold(n_points: int, n_parameters: int = 1, level: float = ONE_SIGMA) -> float:
    """Rise in chi-squared that bounds the ``level`` range, with the noise estimated from
    the same ``n_points`` residuals and ``n_parameters`` fitted.

    n / (n - p) * F(level; 1, n - p), i.e. n / (n - p) * t(0.5 + level / 2; n - p)^2:
    the first factor corrects the maximum-likelihood noise estimate (which divides by
    n) for the parameters fitted, the second widens the range for the uncertainty of
    that estimate. 1.48 at 6 points, 1.07 at 29, 1.007 at 300; tends to 1.
    """
    dof = n_points - n_parameters
    if dof < 1:
        raise ValueError(
            f"{n_points} points cannot constrain {n_parameters} parameter(s) and the noise"
        )
    return n_points / dof * float(stats.f.ppf(level, 1, dof))


def delta_chi2_interval(
    chi2: np.ndarray, delta: float = 1.0, *, end_is_limit: bool = True
) -> Interval:
    """Rows where chi2 <= min + delta, with the ends interpolated to the crossing.

    Takes the outermost rows under the threshold, as the legacy rule did, so a
    second minimum within delta widens the range rather than being ignored. If
    the range reaches the first or last time step it stops there, with a warning
    (for the last step only if ``end_is_limit``, i.e. it is the longest time modelled).
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
        if end_is_limit:
            warnings.warn(
                "1-sigma range reaches the last modelled time step; increase max_time_days",
                stacklevel=2,
            )
    return Interval(best=best, low=float(low), high=float(high), low_row=lo_row, high_row=hi_row)


def central_range(weights: np.ndarray, level: float = ONE_SIGMA) -> tuple[float, float]:
    """Rows (fractional) holding the central ``level`` of a posterior given by its
    unnormalised ``weights`` per time step (trapezoid rule between steps)."""
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (weights[1:] + weights[:-1]))])
    cdf /= cdf[-1]
    rows = np.arange(len(weights), dtype=float)
    return float(np.interp(0.5 - level / 2, cdf, rows)), float(
        np.interp(0.5 + level / 2, cdf, rows)
    )


def posterior_interval(chi2: np.ndarray, n_points: int, level: float = ONE_SIGMA) -> Interval:
    """Central ``level`` credible range of time with the noise scale integrated out.

    ``chi2`` must be r^T C^-1 r with C = s^2 B for a fixed B (as from `fit_correlated`).
    With a flat prior on time and a 1/s prior on s, the posterior of time is
    proportional to chi2^(-n/2) (Student-t-like). Ends are interpolated between time
    steps; the best row is the posterior mode (the minimum of chi2). ``chi2`` should
    cover all times with appreciable posterior weight: from a fit with
    ``stop_early=False``, or with the run extended well past the range.
    """
    log_post = -0.5 * n_points * np.log(chi2)
    low, high = central_range(np.exp(log_post - log_post.max()), level)
    return Interval(
        best=int(np.argmin(chi2)),
        low=low,
        high=high,
        low_row=int(np.floor(low)),
        high_row=int(np.ceil(high)),
    )


@dataclass(frozen=True)
class CorrelatedFit:
    chi2: np.ndarray
    interval: Interval
    noise: NoiseModel
    x_um: np.ndarray
    fo: np.ndarray
    threshold: float  # rise in chi2 that bounds the 1-sigma range (`interval_threshold`)
    interval2: Interval | None = None  # the 2-sigma range
    threshold2: float = float("nan")


def points_in_domain(
    model_x_um: np.ndarray, x_um: np.ndarray, fo: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The measured points that are fitted: inside the model domain, Fo not missing."""
    keep = (x_um >= model_x_um[0]) & (x_um <= model_x_um[-1]) & np.isfinite(fo)
    x, y = x_um[keep], fo[keep]
    if len(x) < 3:
        raise ValueError(f"need at least 3 measured points inside the model domain, got {len(x)}")
    return x, y


def fit_correlated(
    curves: np.ndarray,
    model_x_um: np.ndarray,
    x_um: np.ndarray,
    fo: np.ndarray,
    *,
    end_is_limit: bool = True,
) -> CorrelatedFit:
    """Best fit and 1-sigma range in time from the measured points.

    Points outside the model domain or with missing Fo are left out. ``end_is_limit``
    says whether the last row of ``curves`` is the longest time that can be modelled
    (False while a run is being extended).
    """
    x, y = points_in_domain(model_x_um, x_um, fo)
    W = interpolation_matrix(model_x_um, x)
    first = int(np.argmin(chi2_per_step(curves, W, y)))
    noise = fit_noise_model(x, y - W @ curves[first])
    chi2 = chi2_per_step(curves, W, y, noise.covariance(x))
    return _correlated_ranges(chi2, noise, x, y, end_is_limit)


def fit_correlated_points(
    at_points: np.ndarray, x_um: np.ndarray, fo: np.ndarray, *, end_is_limit: bool = True
) -> CorrelatedFit:
    """`fit_correlated` from the model at the measured points only (``at_points[k]`` is the
    model at ``x_um`` at time step k), so a run need not keep its full profiles. ``x_um``
    and ``fo`` are the fitted points (`points_in_domain`)."""
    chi2, noise = correlated_chi2(at_points, x_um, fo)
    return _correlated_ranges(chi2, noise, x_um, fo, end_is_limit)


def white_rss(at_points: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Sum of squared residuals at every time step, from the model at the points."""
    r = at_points - y
    return np.einsum("kp,kp->k", r, r)


def correlated_chi2(
    at_points: np.ndarray, x_um: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, NoiseModel]:
    """Generalised chi-squared at every time step from the model at the points: the noise
    model is fitted at the white (least-squares) best row, then r_k^T C^-1 r_k."""
    residuals = at_points - y
    first = int(np.argmin(white_rss(at_points, y)))
    noise = fit_noise_model(x_um, -residuals[first])
    chol = np.linalg.cholesky(noise.covariance(x_um))
    chi2 = np.empty(len(at_points))
    for start in range(0, len(at_points), CHUNK_ROWS):
        z = sla.solve_triangular(chol, residuals[start : start + CHUNK_ROWS].T, lower=True)
        chi2[start : start + CHUNK_ROWS] = np.einsum("pk,pk->k", z, z)
    return chi2, noise


def _correlated_ranges(chi2, noise, x, y, end_is_limit) -> CorrelatedFit:
    """The 1- and 2-sigma ranges of a generalised chi-squared series."""
    threshold = interval_threshold(len(x))
    threshold2 = interval_threshold(len(x), level=TWO_SIGMA)
    span = delta_chi2_interval(chi2, threshold, end_is_limit=end_is_limit)
    with warnings.catch_warnings():  # the 1-sigma range already warns about the run's end
        warnings.simplefilter("ignore")
        span2 = delta_chi2_interval(chi2, threshold2, end_is_limit=end_is_limit)
    return CorrelatedFit(chi2, span, noise, x, y, threshold, span2, threshold2)


def widen_log10(time: float, low: float, high: float, sigma_log10: float) -> tuple[float, float]:
    """A range of time widened by an independent error of ``sigma_log10`` in log10 time.

    Each side's distance from ``time`` in log10 is combined in quadrature with
    ``sigma_log10``, so an asymmetric range stays asymmetric. A low end at zero (the
    range reached time zero) stays at zero. With the best fit at time zero there is no
    distance to combine: the high end is scaled by ``10**sigma_log10`` (the law's error
    moves the bound, as it moves every time).
    """
    if not (0 <= low <= time <= high):
        raise ValueError(f"need 0 <= low <= time <= high, got {low}, {time}, {high}")
    if time == 0:
        return 0.0, float(high * 10**sigma_log10)
    up = np.hypot(np.log10(high / time), sigma_log10)
    new_low = 0.0 if low == 0 else time * 10 ** -float(np.hypot(np.log10(time / low), sigma_log10))
    return float(new_low), float(time * 10**up)
