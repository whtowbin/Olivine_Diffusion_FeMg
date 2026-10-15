"""Ways to sample the IC parameters, all on one `likelihood.Problem`.

Each returns the 16/50/84th percentiles of the marginal posterior of time (days) and the
posterior mass in the last 5% of the time axis (plus method-specific extras).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from scipy import optimize, stats
from scipy.special import logsumexp

from forster.sampling import _log_t_density
from forster.scan import candidate_grid

RNG_SEED = 1


def method_grid(P, pts):
    """Grid over every IC parameter, ``pts`` nodes each (`forster.scan.candidate_grid`)."""
    cg = candidate_grid(P.run.initial, P.unc, pts, max_candidates=10**7)
    L = P.loglike_rows(cg.values)
    return P.summarise(L + cg.log_prior[:, None])


def method_laplace_is(P, n_samples, seed=RNG_SEED, verbose=True):
    """Nelder-Mead to the mode of p(theta), finite-difference Hessian, then importance
    sampling from a Student-t (df 5, covariance x 1.5^2) on scrambled Sobol points.
    Returns (q, tail, ESS); the ESS is of the weights marginalised over time."""
    scale = P.spread

    def to_theta(z):
        return P.nominal + z * scale

    def negpost(z):
        th = to_theta(z)[None]
        lp = P.log_prior(th)[0]
        if not np.isfinite(lp):
            return 1e12
        return -(lp + logsumexp(P.loglike_rows(th)[0]))

    d = len(scale)
    res = optimize.minimize(
        negpost,
        np.zeros(d),
        method="Nelder-Mead",
        options={
            "xatol": 0.01,
            "fatol": 0.005,
            "maxfev": 600,
            "initial_simplex": np.vstack([np.zeros(d), 0.5 * np.eye(d)]),
        },
    )
    z0 = res.x
    # finite-difference Hessian in scaled units
    h = 0.15
    f0 = res.fun
    H = np.zeros((d, d))
    for i in range(d):
        for j in range(i, d):
            ei, ej = np.eye(d)[i] * h, np.eye(d)[j] * h
            if i == j:
                H[i, i] = (negpost(z0 + ei) - 2 * f0 + negpost(z0 - ei)) / h**2
            else:
                H[i, j] = H[j, i] = (
                    negpost(z0 + ei + ej)
                    - negpost(z0 + ei - ej)
                    - negpost(z0 - ei + ej)
                    + negpost(z0 - ei - ej)
                ) / (4 * h**2)
    try:
        cov = np.linalg.inv(H)
        np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        cov = np.diag(1 / np.clip(np.diag(H), 1e-3, None))
    if verbose:
        print(
            f"    [laplace] nfev {res.nfev} z0 {np.round(z0, 2)} "
            f"sd {np.round(np.sqrt(np.diag(cov)), 3)}",
            flush=True,
        )
    cov *= 1.5**2
    df = 5
    sob = stats.qmc.Sobol(d + 1, scramble=True, seed=seed).random(n_samples)
    zn = stats.norm.ppf(np.clip(sob[:, :d], 1e-12, 1 - 1e-12))
    w = stats.chi2.ppf(np.clip(sob[:, d], 1e-12, 1 - 1e-12), df)
    Lc = np.linalg.cholesky(cov)
    z = z0 + (zn @ Lc.T) / np.sqrt(w / df)[:, None]
    logq = _log_t_density(z, z0, Lc, df)
    th = to_theta(z)
    lp = P.log_prior(th)
    ok = np.isfinite(lp)
    L = P.loglike_rows(th[ok])
    logw = lp[ok] - logq[ok]
    W = L + logw[:, None]
    ws = logsumexp(L, axis=1) + logw
    ws = np.exp(ws - ws.max())
    ess = ws.sum() ** 2 / (ws**2).sum()
    q, tail = P.summarise(W)
    return q, tail, ess


def method_emcee(P, steps, walkers=32, seed=RNG_SEED):
    import emcee

    def logp(th):
        lp = P.log_prior(th)
        out = lp.copy()
        ok = np.isfinite(lp)
        if ok.any():
            out[ok] += logsumexp(P.loglike_rows(th[ok]), axis=1)
        return out

    rng = np.random.default_rng(seed)
    d = len(P.nominal)
    p0 = P.nominal + 0.3 * P.spread * rng.standard_normal((walkers, d))
    s = emcee.EnsembleSampler(walkers, d, logp, vectorize=True)
    s.random_state = np.random.RandomState(seed)
    s.run_mcmc(p0, steps, progress=False)
    chain = s.get_chain(discard=steps // 2, flat=True)
    pick = chain[
        np.random.default_rng(seed).choice(len(chain), min(1000, len(chain)), replace=False)
    ]
    L = P.loglike_rows(pick)
    L = L - logsumexp(L, axis=1, keepdims=True)  # time conditional per sample
    return P.summarise(L)


def method_dynesty(P, nlive, seed=RNG_SEED):
    """Static nested sampling (dlogz 0.1); the time posterior from 1000 equal-weight
    resamples, each run again (the method sweep's version)."""
    import dynesty

    d = len(P.nominal)

    def ll(th):
        return float(logsumexp(P.loglike_rows(th[None])[0]))

    s = dynesty.NestedSampler(
        ll, P.prior_transform, d, nlive=nlive, rstate=np.random.default_rng(seed)
    )
    s.run_nested(dlogz=0.1, print_progress=False)
    res = s.results
    samples = res.samples_equal()
    pick = samples[
        np.random.default_rng(seed).choice(len(samples), min(1000, len(samples)), replace=False)
    ]
    L = P.loglike_rows(pick)
    L = L - logsumexp(L, axis=1, keepdims=True)
    return P.summarise(L), float(res.logz[-1])


# dynesty needs a finite log-likelihood; an invalid IC (zero prior) gets this
LOGL_INVALID = -1e300


@dataclass(frozen=True)
class NestedResult:
    q: np.ndarray  # p16, p50, p84 of time, days
    tail: float  # posterior mass in the last 5% of the time axis
    logz: float
    logz_err: float
    ncall: int  # likelihood calls (each valid one is one forward run)
    niter: int
    n_samples: int  # dead + final live points carrying weight


def method_dynesty_weighted(P, nlive, seed=RNG_SEED, dlogz=0.01):
    """Static nested sampling; the time posterior from every weighted sample.

    Each likelihood call keeps p(t | theta) as a dynesty blob, so the time marginal is
    sum_i w_i p(t | theta_i) over all dead and final live points with their nested
    sampling weights: no resampling and no second run per sample.
    """
    import dynesty

    d = len(P.nominal)
    flat = np.full(P.rows, -np.log(P.rows), dtype=np.float32)

    def ll(th):
        rows = P.loglike_rows(th[None])[0]
        lz = logsumexp(rows)
        if not np.isfinite(lz):
            return LOGL_INVALID, flat
        return float(lz), (rows - lz).astype(np.float32)

    s = dynesty.NestedSampler(
        ll, P.prior_transform, d, nlive=nlive, blob=True, rstate=np.random.default_rng(seed)
    )
    s.run_nested(dlogz=dlogz, print_progress=False)
    res = s.results
    q, tail = P.summarise(res.logwt[:, None] + res["blob"].astype(np.float64))
    return NestedResult(
        q=q,
        tail=float(tail),
        logz=float(res.logz[-1]),
        logz_err=float(res.logzerr[-1]),
        ncall=int(np.sum(res.ncall)),
        niter=int(res.niter),
        n_samples=len(res.logwt),
    )


def timed(P, fn, *a, **kw):
    """(result, forward runs, wall seconds)."""
    P.n_runs = 0
    t0 = time.perf_counter()
    out = fn(P, *a, **kw)
    return out, P.n_runs, time.perf_counter() - t0
