"""Prototype of a hierarchical (population) fit of segment Fo. Not part of FORSTER.

The model, for one role (say the core) in one sample of N crystals:

    population:  mu (the sample's typical core Fo), tau (spread between crystals)
    crystal i:   fo_i ~ Normal(mu, tau)          its own true core Fo
    data i:      profile_i | fo_i, t_i, other starting-profile values of crystal i

Each crystal keeps its own time t_i and its own steps and other segments; only the core
Fo values are tied together, through mu and tau, which are fitted too. With tau large
the crystals are independent (today's fits); with tau -> 0 they share one core value.

Two ways to fit it:

1. `random_effects` (two-stage, cheap): summarise each crystal's own fit of fo_i as a
   Gaussian (value, error), then fit mu and tau by restricted maximum likelihood and
   shrink each crystal toward mu. This is the normal-normal model of meta-analysis. It is
   accurate when each crystal's likelihood for fo_i is close to Gaussian, which the
   starting-profile fits suggest for measured plateaus (posterior sd 0.05-0.2 mol%) but
   not for a barely measured plateau whose posterior is skewed or cut by the prior.
   `time_shift` then estimates how each crystal's time moves, from the fit's
   sensitivity dE[t]/dfo.

2. `joint_log_posterior` (full): the log posterior of all crystals' starting-profile
   parameters plus (mu, log tau), using each crystal's `forster.sampling.ICLikelihood`
   with its time marginalised. One evaluation runs one diffusion model per crystal, so
   sampling it (nested sampling or HMC over N x d + 2 parameters) costs about N times a
   single fit times the extra dimensions; for WHT01 (9 crystals, ~4 parameters each)
   that is a 38-dimensional problem. Use it only where (1) is not good enough.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy import optimize


def _reml(log_tau: float, y: np.ndarray, e: np.ndarray) -> float:
    v = e**2 + math.exp(log_tau) ** 2
    w = 1 / v
    mu = (w * y).sum() / w.sum()
    return 0.5 * (np.log(v).sum() + math.log(w.sum()) + (w * (y - mu) ** 2).sum())


def random_effects(values: Sequence[float], errors: Sequence[float]) -> dict:
    """Normal-normal hierarchical fit of crystal values with known errors.

    Returns mu (population mean) and its standard error, tau (between-crystal sd, by
    restricted maximum likelihood), the shrunk value and sd of each crystal, its
    shrinkage weight (0: own data only, 1: population mean only) and the predictive
    sd for a new crystal of the sample (the prior a new, unmeasured crystal would get).
    """
    y, e = np.asarray(values, float), np.asarray(errors, float)
    if len(y) < 2:
        raise ValueError("need at least two crystals")
    scale = max(float(y.std()), float(e.max()))
    res = optimize.minimize_scalar(
        _reml, bounds=(math.log(scale * 1e-4), math.log(scale * 10)), args=(y, e), method="bounded"
    )
    tau = math.exp(res.x)
    if _reml(math.log(scale * 1e-6), y, e) <= res.fun:  # boundary: no spread beyond errors
        tau = 0.0
    v = e**2 + tau**2
    w = 1 / v
    mu = float((w * y).sum() / w.sum())
    mu_se = float(1 / math.sqrt(w.sum()))
    if tau > 0:
        shrink = e**2 / (e**2 + tau**2)
        post_sd = np.sqrt(1 / (1 / e**2 + 1 / tau**2))
    else:
        shrink, post_sd = np.ones_like(y), np.full_like(y, mu_se)
    return {
        "mu": mu,
        "mu_se": mu_se,
        "tau": tau,
        "shrunk": (1 - shrink) * y + shrink * mu,
        "shrunk_sd": post_sd,
        "shrinkage": shrink,
        "predictive_sd": math.sqrt(tau**2 + mu_se**2),
    }


def time_shift(own: np.ndarray, shrunk: np.ndarray, sensitivity_days_per_fo: np.ndarray):
    """First-order change of each crystal's time when its segment Fo moves from its own
    estimate to the shrunk one (`forster.sampling.ICPosterior.sensitivity`)."""
    return np.asarray(sensitivity_days_per_fo) * (np.asarray(shrunk) - np.asarray(own))


def joint_log_posterior(
    theta: Sequence[np.ndarray],
    mu: float,
    log_tau: float,
    likes: Sequence,
    shared: str,
    tau_scale: float = 0.01,
) -> float:
    """Log posterior of a hierarchical fit over several crystals (full version).

    ``likes``: one `forster.sampling.ICLikelihood` per crystal (its noise model already
    fitted, e.g. from the crystal's own fit: ``like.use_noise(post.noise)``), each with
    its own parameter vector ``theta[i]`` in ``like.names`` order; ``shared`` names the
    parameter tied across crystals (e.g. "c_fo" for a core labelled c in every crystal;
    labels must mean the same zone in every crystal). Each crystal's other parameters
    keep their own priors (``like.log_prior`` without the shared one); the shared one
    gets Normal(mu, tau); mu has a flat prior and tau a half-Cauchy of scale
    ``tau_scale`` (1 mol% Fo).
    """
    tau = math.exp(log_tau)
    lp = math.log(2 / (math.pi * tau_scale * (1 + (tau / tau_scale) ** 2))) + log_tau
    for th, like in zip(theta, likes, strict=True):
        k = like.names.index(shared)
        z = (th - like.nominal) / like.spread
        if np.any(th < like.lo) or np.any(th > like.hi):
            return -math.inf
        own = -0.5 * float(np.sum(np.where(like.normal, z**2, 0.0)) - z[k] ** 2 * like.normal[k])
        pop = -0.5 * ((th[k] - mu) / tau) ** 2 - math.log(tau)
        ll = like.log_marginal(th)
        if not math.isfinite(ll):
            return -math.inf
        lp += own + pop + ll
    return lp
