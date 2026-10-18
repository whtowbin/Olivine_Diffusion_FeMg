"""Shared helpers for the whitening-validity study (2026-10-15).

- Covariance kernels: squared-exponential (forster's), exponential (Matern 1/2) and
  Matern 3/2, each as C = s^2 (I + rho^2 K_L).
- Maximum-likelihood noise-model fits for any of them, on the same (rho, L) grid as
  `forster.uncertainty.fit_noise_model`, then refined continuously (same optimiser and
  bounds for every kernel, so their maximised likelihoods are comparable).
- Whiteness tests of whitened residuals: Ljung-Box, runs test, Shapiro-Wilk.

Nothing here changes forster; the default method is always called from the package.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.linalg as sla
from scipy import optimize, stats

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(REPO / "analyses" / "ic_sampling"))

from forster.uncertainty import N_LENGTHS, RHO_GRID  # noqa: E402

KERNELS = ("white", "sqexp", "exp", "matern32")
KERNEL_LABELS = {
    "white": "white only",
    "sqexp": "squared-exponential (current)",
    "exp": "exponential (Matern 1/2)",
    "matern32": "Matern 3/2",
}
N_PARAMS = {"white": 1, "sqexp": 3, "exp": 3, "matern32": 3}

# Round robin rows (other laboratories' profiles) are written only to the private
# submodule; the CSVs here keep the paper and example rows.
PRIVATE = REPO / "Diffusion Round Robin" / "outputs" / "whitening_validity"


def write_split(df, name: str, float_format: str) -> None:
    """Write every row to PRIVATE/name and the rows that are not round robin to HERE/name."""
    PRIVATE.mkdir(parents=True, exist_ok=True)
    df.to_csv(PRIVATE / name, index=False, float_format=float_format)
    df[df.source != "round_robin"].to_csv(HERE / name, index=False, float_format=float_format)


def kernel_matrix(kind: str, x: np.ndarray, length: float) -> np.ndarray:
    """Correlation matrix K_L (unit diagonal) for points x (um)."""
    d = np.abs(x[:, None] - x[None, :])
    if kind == "sqexp":
        return np.exp(-0.5 * (d / length) ** 2)
    if kind == "exp":
        return np.exp(-d / length)
    if kind == "matern32":
        a = np.sqrt(3.0) * d / length
        return (1.0 + a) * np.exp(-a)
    raise ValueError(kind)


def length_grid(x: np.ndarray) -> np.ndarray:
    """The L grid of `forster.uncertainty.fit_noise_model`."""
    spacing = float(np.median(np.diff(np.sort(x))))
    span = float(x.max() - x.min())
    return np.geomspace(max(spacing, 1e-3), max(span / 2, 2 * spacing), N_LENGTHS)


def _nll(r: np.ndarray, K: np.ndarray, rho: float) -> tuple[float, float]:
    """Profile negative log-likelihood (s^2 at its closed-form optimum, constants dropped)
    and that s^2, for C = s^2 (I + rho^2 K)."""
    n = len(r)
    chol = np.linalg.cholesky(np.eye(n) + rho**2 * K)
    z = sla.solve_triangular(chol, r, lower=True)
    s2 = max(float(z @ z) / n, 1e-30)
    return 0.5 * n * np.log(s2) + float(np.log(np.diag(chol)).sum()), s2


@dataclass(frozen=True)
class KernelFit:
    kind: str
    white_sigma: float
    correlated_sigma: float
    length_um: float
    loglik: float  # maximised log-likelihood, with all constants
    n_params: int

    @property
    def aic(self) -> float:
        return 2 * self.n_params - 2 * self.loglik

    def covariance(self, x: np.ndarray) -> np.ndarray:
        cov = self.white_sigma**2 * np.eye(len(x))
        if self.correlated_sigma > 0:
            cov += self.correlated_sigma**2 * kernel_matrix(self.kind, x, self.length_um)
        return cov


def _full_loglik(n: int, nll: float) -> float:
    # -2 log L = n log(2 pi s^2) + log|B| + r^T C^-1 r, with r^T C^-1 r = n at the ML s.
    return -(nll + 0.5 * n * np.log(2 * np.pi) + 0.5 * n)


def fit_kernel(kind: str, x: np.ndarray, r: np.ndarray, refine: bool = True) -> KernelFit:
    """Maximum-likelihood white + ``kind`` noise model for residuals r at points x.

    Grid search as forster does (rho in RHO_GRID, 12 lengths), then, if ``refine``,
    L-BFGS-B on (log rho, log L) from the best grid point with rho in [1e-3, 30] and L
    within the grid's range. rho = 0 (white only) is always a candidate.
    """
    n = len(r)
    s2w = max(float(r @ r) / n, 1e-30)
    white_nll = 0.5 * n * np.log(s2w)
    if kind == "white":
        return KernelFit("white", float(np.sqrt(s2w)), 0.0, 0.0, _full_loglik(n, white_nll), 1)
    lengths = length_grid(x)
    best = (white_nll, s2w, 0.0, 0.0)
    for L in lengths:
        K = kernel_matrix(kind, x, L)
        for rho in RHO_GRID[1:]:
            nll, s2 = _nll(r, K, rho)
            if nll < best[0]:
                best = (nll, s2, rho, L)
    if refine:
        rho0 = best[2] if best[2] > 0 else 0.1
        L0 = best[3] if best[3] > 0 else float(np.sqrt(lengths[0] * lengths[-1]))
        lo = np.array([np.log(1e-3), np.log(lengths[0])])
        hi = np.array([np.log(RHO_GRID[-1]), np.log(lengths[-1])])

        def f(p):
            return _nll(r, kernel_matrix(kind, x, float(np.exp(p[1]))), float(np.exp(p[0])))[0]

        res = optimize.minimize(
            f,
            np.clip([np.log(rho0), np.log(L0)], lo, hi),
            method="L-BFGS-B",
            bounds=list(zip(lo, hi, strict=True)),
        )
        if res.fun < best[0]:
            rho, L = float(np.exp(res.x[0])), float(np.exp(res.x[1]))
            nll, s2 = _nll(r, kernel_matrix(kind, x, L), rho)
            best = (nll, s2, rho, L)
    nll, s2, rho, L = best
    s = float(np.sqrt(s2))
    return KernelFit(kind, s, float(rho * s), float(L) if rho > 0 else 0.0, _full_loglik(n, nll), 3)


# ----------------------------------------------------------------------------- whiteness


def ljung_box(z: np.ndarray, lags: int = 5) -> np.ndarray:
    """Ljung-Box p-values for lags 1..``lags`` (no fitted ARMA terms, dof = lag)."""
    n = len(z)
    zc = z - z.mean()
    denom = float(zc @ zc)
    acf = np.array([float(zc[:-k] @ zc[k:]) / denom for k in range(1, lags + 1)])
    q = n * (n + 2) * np.cumsum(acf**2 / (n - np.arange(1, lags + 1)))
    return stats.chi2.sf(q, np.arange(1, lags + 1))


def runs_test(z: np.ndarray) -> float:
    """Wald-Wolfowitz runs test on the signs of z (about zero), two-sided p (normal
    approximation)."""
    s = z > 0
    n1, n2 = int(s.sum()), int((~s).sum())
    n = n1 + n2
    if n1 == 0 or n2 == 0:
        return 0.0
    runs = 1 + int(np.count_nonzero(s[1:] != s[:-1]))
    mean = 2 * n1 * n2 / n + 1
    var = 2 * n1 * n2 * (2 * n1 * n2 - n) / (n**2 * (n - 1))
    return float(2 * stats.norm.sf(abs(runs - mean) / np.sqrt(var)))


def whiteness(z: np.ndarray) -> dict[str, float]:
    """Tests that whitened residuals z (ordered by distance) are iid N(0, 1)."""
    lb = ljung_box(z, 5)
    return {
        "lag1_z": float(np.corrcoef(z[:-1], z[1:])[0, 1]),
        "lb_p_lag1": float(lb[0]),
        "lb_p_lag5": float(lb[-1]),
        "lb_p_min": float(lb.min()),
        "runs_p": runs_test(z),
        "shapiro_p": float(stats.shapiro(z).pvalue),
        "var_z": float(np.var(z)),
        "max_abs_z": float(np.max(np.abs(z))),
        "kurtosis_z": float(stats.kurtosis(z)),  # excess kurtosis: 0 for a normal
    }


def whiten(cov: np.ndarray, r: np.ndarray) -> np.ndarray:
    return sla.solve_triangular(np.linalg.cholesky(cov), r, lower=True)
