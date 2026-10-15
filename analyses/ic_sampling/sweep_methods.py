"""Compare ways to combine initial-condition uncertainty with the time fit.

Every method uses the same likelihood: one noise model (fitted once at a 9-point grid's
best fit), chi2 at every time step from one forward run per initial condition, flat
prior on time, posterior ~ chi2^(-n/2) * prior. So differences are sampling error only.
Output: 16/50/84th percentiles of the marginal posterior of time (days), forward runs
used, wall seconds.
"""

import sys
import time
import warnings
from dataclasses import replace

import numpy as np
from scipy import optimize, stats
from scipy.special import logsumexp

sys.path.insert(0, "tests")
from conftest import ROUND_ROBIN_DATA, ROUND_ROBIN_PARAMS

from forster import Uncertainty, fit_run, read_profiles, read_runs
from forster.scan import (
    NORMAL_SPAN,
    _whitened_chi2,
    _with_values,
    candidate_grid,
    fit_candidates,
    parameter_names,
    run_candidates,
)

warnings.filterwarnings("ignore")
RNG_SEED = 1


class Problem:
    def __init__(self, run, profiles, unc):
        self.run, self.unc = run, unc
        self.profile = profiles[run.profile_name]
        nominal = fit_run(run, profiles)
        req = run.scan_grid
        req = replace(req, max_time_days=None)
        self.grid = req.resolve(
            self.profile,
            run.conditions,
            run.orientation,
            run.initial,
            run.diffusivity,
            purpose="scan",
            time_days=nominal.time_days,
            scheme=run.scheme,
        ).grid
        # longer run so wide priors do not truncate the time posterior
        self.grid = replace(self.grid, max_time_days=self.grid.max_time_days * 2)
        self.names = tuple(n for n in parameter_names(run.initial) if n in unc)
        self.nominal = np.array([_value(run.initial, n) for n in self.names])
        self.spread = np.array([unc[n].spread for n in self.names])
        self.kinds = [unc[n].kind for n in self.names]
        half = np.where([k == "normal" for k in self.kinds], NORMAL_SPAN, 1.0) * self.spread
        self.lo, self.hi = self.nominal - half, self.nominal + half
        self.n_runs = 0
        # one noise model for every method
        cg = candidate_grid(
            run.initial, unc, 9 if len(self.names) <= 3 else 5, max_candidates=10**6
        )
        r = self._runs(list(cg.candidates))
        s = fit_candidates(r, prior=cg)
        self.chol = np.linalg.cholesky(s.noise.covariance(r.x_um))
        self.y = r.fo
        self.n = len(self.y)
        self.days_per_row = self.grid.dt_s / 86400
        self.n_runs = 0

    def _runs(self, initials):
        self.n_runs += len(initials)
        return run_candidates(
            self.profile,
            self.run.conditions,
            self.run.orientation,
            initials,
            self.grid,
            self.run.diffusivity,
            self.run.scheme,
            stop_early=False,
        )

    def log_prior(self, theta):
        theta = np.atleast_2d(theta)
        inside = np.all((theta >= self.lo) & (theta <= self.hi), axis=1)
        lp = np.zeros(len(theta))
        for j, k in enumerate(self.kinds):
            if k == "normal":
                lp += -0.5 * ((theta[:, j] - self.nominal[j]) / self.spread[j]) ** 2
        return np.where(inside, lp, -np.inf)

    def loglike_rows(self, theta):
        """(len(theta), rows) log-likelihood; -inf rows for invalid initial conditions."""
        theta = np.atleast_2d(theta)
        rows = self.grid.timesteps + 1
        out = np.full((len(theta), rows), -np.inf)
        initials, idx = [], []
        for i, th in enumerate(theta):
            try:
                initials.append(_with_values(self.run.initial, dict(zip(self.names, th))))
                idx.append(i)
            except ValueError:
                pass
        if initials:
            r = self._runs(initials)
            for i, m in zip(idx, r.at_points):
                c = _whitened_chi2(m, self.y, self.chol)
                out[i, : len(c)] = -0.5 * self.n * np.log(c)
        return out

    def summarise(self, log_w_rows):
        """Percentiles of time from per-(sample,row) log weights."""
        p = np.exp(logsumexp(log_w_rows, axis=0) - logsumexp(log_w_rows))
        cdf = np.cumsum(p)
        rows = np.arange(len(p))
        q = np.interp([0.16, 0.5, 0.84], cdf, rows) * self.days_per_row
        tail = p[-len(p) // 20 :].sum()
        return q, tail


def _value(initial, name):
    if name == "step1_um":
        return initial.steps_um[0]
    if name == "step2_um":
        return initial.steps_um[1]
    return getattr(initial, name)


def method_grid(P, pts):
    cg = candidate_grid(P.run.initial, P.unc, pts, max_candidates=10**7)
    L = P.loglike_rows(cg.values)
    return P.summarise(L + cg.log_prior[:, None])


def method_laplace_is(P, n_samples, seed=RNG_SEED):
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
    logq = stats.multivariate_t(loc=z0, shape=cov, df=df).logpdf(z)
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
    import dynesty

    d = len(P.nominal)

    def ptform(u):
        th = np.empty(d)
        for j, k in enumerate(P.kinds):
            if k == "normal":
                a = stats.norm.cdf(-NORMAL_SPAN)
                th[j] = P.nominal[j] + P.spread[j] * stats.norm.ppf(a + u[j] * (1 - 2 * a))
            else:
                th[j] = P.lo[j] + u[j] * (P.hi[j] - P.lo[j])
        return th

    def ll(th):
        return float(logsumexp(P.loglike_rows(th[None])[0]))

    s = dynesty.NestedSampler(ll, ptform, d, nlive=nlive, rstate=np.random.default_rng(seed))
    s.run_nested(dlogz=0.1, print_progress=False)
    res = s.results
    samples = res.samples_equal()
    pick = samples[
        np.random.default_rng(seed).choice(len(samples), min(1000, len(samples)), replace=False)
    ]
    L = P.loglike_rows(pick)
    L = L - logsumexp(L, axis=1, keepdims=True)
    return P.summarise(L), float(res.logz[-1])


def timed(P, fn, *a):
    P.n_runs = 0
    t0 = time.perf_counter()
    out = fn(P, *a)
    return out, P.n_runs, time.perf_counter() - t0


def main():
    runs = {r.name: r for r in read_runs(ROUND_ROBIN_PARAMS)}
    profiles = read_profiles(ROUND_ROBIN_DATA)
    which = sys.argv[1:] or ["all"]
    cases = []
    for name in ("KS20-527_2-transect-1", "KS20-527_5-transect-2"):
        for label, unc in (
            (
                "gauss",
                {
                    "step1_um": Uncertainty(3.0),
                    "edge_fo": Uncertainty(0.003),
                    "core_fo": Uncertainty(0.003),
                },
            ),
            (
                "wide",
                {
                    "step1_um": Uncertainty(10.0, "uniform"),
                    "edge_fo": Uncertainty(0.01, "uniform"),
                    "core_fo": Uncertainty(0.01, "uniform"),
                },
            ),
        ):
            cases.append((name, label, unc, 3))
    for label, unc in (
        (
            "gauss",
            {
                "step1_um": Uncertainty(3.0),
                "step2_um": Uncertainty(3.0),
                "edge_fo": Uncertainty(0.003),
                "core_fo": Uncertainty(0.003),
                "step_fo": Uncertainty(0.003),
            },
        ),
        (
            "wide",
            {
                "step1_um": Uncertainty(10.0, "uniform"),
                "step2_um": Uncertainty(10.0, "uniform"),
                "edge_fo": Uncertainty(0.01, "uniform"),
                "core_fo": Uncertainty(0.01, "uniform"),
                "step_fo": Uncertainty(0.01, "uniform"),
            },
        ),
    ):
        cases.append(("KS20-527-8_transect-2", label, unc, 5))

    for name, label, unc, d in cases:
        if which != ["all"] and f"{name}:{label}" not in which and name not in which:
            continue
        P = Problem(runs[name], profiles, unc)
        print(
            f"\n== {name} [{label}, {d} params]  grid {P.grid.timesteps} steps, "
            f"dt {P.grid.dt_s:.0f}s",
            flush=True,
        )

        def show(tag, out, n, sec, extra=""):
            q, tail = out[0], out[1]
            print(
                f"  {tag:24s} p16/50/84 = {q[0]:7.2f} {q[1]:7.2f} {q[2]:7.2f} d"
                f"  runs {n:7d}  {sec:6.1f}s  tail {tail:.1e} {extra}",
                flush=True,
            )

        ref_pts = 31 if d == 3 else 11
        grid_pts = (5, 9, 13, 17) if d == 3 else (3, 5, 7)
        out, n, sec = timed(P, method_grid, ref_pts)
        show(f"REF grid {ref_pts}^{d}", out, n, sec)
        for pts in grid_pts:
            out, n, sec = timed(P, method_grid, pts)
            show(f"grid {pts}^{d}", out, n, sec)
        for ns in (256, 1024, 4096):
            (q, tail, ess), n, sec = timed(P, method_laplace_is, ns)
            show(f"laplace+IS N={ns}", (q, tail), n, sec, f"ESS {ess:.0f}")
        for steps in (250, 1000):
            out, n, sec = timed(P, method_emcee, steps)
            show(f"emcee 32x{steps}", out, n, sec)
        for nlive in (100, 400):
            (out, logz), n, sec = timed(P, method_dynesty, nlive)
            show(f"dynesty nlive={nlive}", out, n, sec, f"logZ {logz:.2f}")


if __name__ == "__main__":
    main()
