"""Prototype: FORSTER's near-step check with the growth h fitted continuously.

FORSTER (`forster.uncertainty.near_step_check`) compares, at the residuals r of the white
best fit, the default noise model

    C = s^2 (I + rho^2 K_L),          K_L squared-exponential with length L,

with the near-step model

    C = s^2 (I + rho^2 S K_L S),      S = diag(sqrt(1 + h e)),

where e(x) is the largest Gaussian bump of width one diffusion length l centred on a step
(peak 1). The misfit's standard deviation is a = rho s far from the steps and
a sqrt(1 + h) at a step. s is profiled out in closed form, (rho, L) on FORSTER's grid
(rho in RHO_GRID, 12 lengths), h on NEAR_STEP_H_GRID = (0, 1, 3, 10, 30, 100), and
dAIC = 2 (nll(best h) - nll(h = 0)) + 2 (one extra parameter; < 0 favours the near-step
model).

Here h is instead fitted by a bounded 1-D optimisation in log10 h on [-2, 4], with the
same (rho, L) profile, so nll(h) is FORSTER's at every h. The grid's values are part of
the search, so the continuous dAIC can only be lower than or equal to the grid's.

Each (h, L) uses one eigendecomposition of S K S: for B = I + rho^2 M, M = U diag(lam) U^T,
r^T B^-1 r = sum (U^T r)^2 / (1 + rho^2 lam) and log|B| = sum log(1 + rho^2 lam), so every
rho on the grid costs one projection. This is the same likelihood as FORSTER's Cholesky
loop (checked against `near_step_check` in continuous_h.py).

Nothing here changes forster.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla
from scipy import optimize

from forster.uncertainty import N_LENGTHS, NEAR_STEP_H_GRID, RHO_GRID

LOG10_H_BOUNDS = (-2.0, 4.0)
LR95 = 3.841  # chi-squared(1) 95% point: the profile-likelihood interval for h
ZONE = 2.0  # "near the step": within this many diffusion lengths (the study's zone)
_SCAN = np.unique(
    np.concatenate(
        [
            np.arange(LOG10_H_BOUNDS[0], LOG10_H_BOUNDS[1] + 1e-9, 0.25),
            np.log10([h for h in NEAR_STEP_H_GRID if h > 0]),
        ]
    )
)


@dataclass(frozen=True)
class Fit:
    """One near-step fit at a given h: profile nll and the (s, rho, L) that reach it."""

    nll: float
    h: float
    s: float
    rho: float
    L: float


@dataclass(frozen=True)
class Result:
    """The grid (FORSTER) and continuous fits, the 95% profile-likelihood interval of h,
    and what the continuous fit says about the misfit near and far from the step."""

    daic_grid: float
    h_grid: float
    daic_cont: float
    h_cont: float
    h_lo95: float  # 0 when h = 0 is inside the interval
    h_hi95: float  # inf when the likelihood stays within the interval up to h = 1e4
    s: float  # white noise sd at the continuous fit
    rho: float
    L: float
    a_far: float  # correlated misfit sd far from the steps, rho s
    a_peak: float  # correlated misfit sd at a step, rho s sqrt(1 + h)
    implied_rms_ratio: float  # near/far raw-residual RMS the fitted model implies
    measured_rms_ratio: float  # near/far raw-residual RMS of r itself
    n_near: int
    n_far: int
    cov_cont: np.ndarray
    cov_grid: np.ndarray
    n_evals: int


class NearStep:
    """Profile likelihood of the near-step model in h, for residuals r at points x."""

    def __init__(self, x: np.ndarray, r: np.ndarray, steps_um, length_um: float):
        x = np.asarray(x, float)
        r = np.asarray(r, float)
        self.x, self.r, self.n = x, r, len(x)
        spacing = float(np.median(np.diff(np.sort(x))))
        span = float(x.max() - x.min())
        lengths = np.geomspace(max(spacing, 1e-3), max(span / 2, 2 * spacing), N_LENGTHS)
        d2 = (x[:, None] - x[None, :]) ** 2
        self.lengths = lengths
        self.Ks = [np.exp(-0.5 * d2 / L**2) for L in lengths]
        self.ell = max(float(length_um), 1e-6)
        steps = np.asarray(steps_um, float)
        self.e = np.exp(-0.5 * ((x[:, None] - steps[None, :]) / self.ell) ** 2).max(axis=1)
        self.near = np.abs(x[:, None] - steps[None, :]).min(axis=1) <= ZONE * self.ell
        self.rho2 = RHO_GRID[1:] ** 2
        s2w = max(float(r @ r) / self.n, 1e-30)
        self.white = Fit(0.5 * self.n * np.log(s2w), 0.0, float(np.sqrt(s2w)), 0.0, 0.0)
        self._memo: dict[float, Fit] = {}

    def at(self, h: float) -> Fit:
        """Best (rho, L) on FORSTER's grid at this h (rho = 0, white only, included)."""
        h = float(h)
        if h in self._memo:
            return self._memo[h]
        n, r = self.n, self.r
        sc = np.sqrt(1.0 + h * self.e)
        best = self.white
        for L, K in zip(self.lengths, self.Ks, strict=True):
            lam, U = np.linalg.eigh(sc[:, None] * K * sc[None, :])
            lam = np.clip(lam, 0.0, None)
            q = (U.T @ r) ** 2
            D = 1.0 + self.rho2[:, None] * lam[None, :]
            z2 = np.maximum((q[None, :] / D).sum(axis=1) / n, 1e-30)
            vals = 0.5 * n * np.log(z2) + 0.5 * np.log(D).sum(axis=1)
            i = int(np.argmin(vals))
            if vals[i] < best.nll:
                best = Fit(float(vals[i]), h, float(np.sqrt(z2[i])), float(RHO_GRID[1 + i]), L)
        self._memo[h] = best
        return best

    def nll_log10(self, t: float) -> float:
        return self.at(10.0**t).nll

    def covariance(self, f: Fit) -> np.ndarray:
        if f.rho == 0:
            return f.s**2 * np.eye(self.n)
        sc = np.sqrt(1.0 + f.h * self.e)
        K = np.exp(-0.5 * (self.x[:, None] - self.x[None, :]) ** 2 / f.L**2)
        return f.s**2 * (np.eye(self.n) + f.rho**2 * sc[:, None] * K * sc[None, :])

    def grid(self) -> tuple[float, Fit, Fit]:
        """FORSTER's grid result: dAIC, best fit (smallest h among ties) and h = 0 fit."""
        fits = [self.at(h) for h in NEAR_STEP_H_GRID]
        i = int(np.argmin([f.nll for f in fits]))  # first minimum, as near_step_check
        return 2 * (fits[i].nll - fits[0].nll) + 2, fits[i], fits[0]

    def _crossing(self, t_best: float, target: float, direction: int) -> float:
        """log10 h where nll first rises above ``target``, walking from t_best outward;
        +-inf if it never does inside the bounds."""
        pts = _SCAN[t_best < _SCAN] if direction > 0 else _SCAN[t_best > _SCAN][::-1]
        prev = t_best
        for t in pts:
            if self.nll_log10(t) > target:
                return float(
                    optimize.brentq(
                        lambda u: self.nll_log10(u) - target, *sorted((prev, t)), xtol=1e-3
                    )
                )
            prev = t
        return float(np.inf) * direction

    def fit(self) -> Result:
        daic_grid, fg, f0 = self.grid()
        scan = np.array([self.nll_log10(t) for t in _SCAN])
        i = int(np.argmin(scan))
        lo, hi = _SCAN[max(i - 1, 0)], _SCAN[min(i + 1, len(_SCAN) - 1)]
        res = optimize.minimize_scalar(
            self.nll_log10, bounds=(lo, hi), method="bounded", options={"xatol": 1e-3}
        )
        cands = [self.at(10.0 ** float(res.x)), self.at(10.0 ** _SCAN[i]), fg, f0]
        fc = min(cands, key=lambda f: f.nll)
        daic_cont = 2 * (fc.nll - f0.nll) + 2
        # 95% profile-likelihood interval of h
        target = fc.nll + 0.5 * LR95
        if fc.h == 0:
            h_lo, t_best = 0.0, LOG10_H_BOUNDS[0]
        else:
            t_best = float(np.log10(fc.h))
            if f0.nll <= target:
                h_lo = 0.0
            else:
                t_lo = self._crossing(t_best, target, -1)
                h_lo = 10.0 ** LOG10_H_BOUNDS[0] if np.isinf(t_lo) else 10.0**t_lo
        t_hi = self._crossing(t_best, target, +1)
        h_hi = np.inf if np.isinf(t_hi) else 10.0**t_hi
        # what the fit says about the misfit near and far from the step
        var = fc.s**2 * (1.0 + fc.rho**2 * (1.0 + fc.h * self.e))
        near, far = self.near, ~self.near
        ok = near.sum() >= 3 and far.sum() >= 3
        implied = float(np.sqrt(var[near].mean() / var[far].mean())) if ok else np.nan
        r2 = self.r**2
        measured = float(np.sqrt(r2[near].mean() / r2[far].mean())) if ok else np.nan
        return Result(
            daic_grid=float(daic_grid),
            h_grid=fg.h,
            daic_cont=float(daic_cont),
            h_cont=fc.h,
            h_lo95=float(h_lo),
            h_hi95=float(h_hi),
            s=fc.s,
            rho=fc.rho,
            L=fc.L,
            a_far=fc.rho * fc.s,
            a_peak=fc.rho * fc.s * float(np.sqrt(1.0 + fc.h)),
            implied_rms_ratio=implied,
            measured_rms_ratio=measured,
            n_near=int(near.sum()),
            n_far=int(far.sum()),
            cov_cont=self.covariance(fc),
            cov_grid=self.covariance(fg),
            n_evals=len(self._memo),
        )


def chi2_rows(at: np.ndarray, y: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Generalised chi-squared r_k^T C^-1 r_k at every row k of the model at the points."""
    chol = np.linalg.cholesky(cov)
    z = sla.solve_triangular(chol, (at - y).T, lower=True)
    return np.einsum("pk,pk->k", z, z)
