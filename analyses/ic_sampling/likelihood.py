"""The likelihood shared by every initial-condition (IC) sampling method in this directory.

For uncertain IC parameters theta (step positions, edge/core/plateau Fo) and time t:

- one forward run per theta gives chi2(theta, t) at every time step of one fixed grid;
- chi2 is generalised with one fixed noise model (white sigma, correlated sigma, length);
- the posterior is p(theta, t) ~ chi2^(-n/2) * prior(theta), with a flat prior on time;
- prior(theta): per parameter a Gaussian truncated at +-NORMAL_SPAN sigma, or a uniform;
  ICs with steps outside the domain or out of order, or Fo outside (0, 1], have zero
  prior (the rule `forster.scan.candidate_grid` uses to drop nodes).

`Problem` holds all of it. Pass ``noise`` to evaluate exactly the same likelihood as an
earlier run (for example a reference posterior in references.csv); without it the noise
model is fitted as the method sweep did, at the best node of a coarse grid
(9 points per parameter for up to 3 parameters, 5 for more).
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping
from dataclasses import replace

import numpy as np
from scipy import stats
from scipy.special import logsumexp

from forster import Uncertainty, fit_run
from forster.model import InitialCondition
from forster.scan import (
    NORMAL_SPAN,
    _whitened_chi2,
    _with_values,
    candidate_grid,
    fit_candidates,
    parameter_names,
    run_candidates,
)
from forster.uncertainty import NoiseModel

# max time of the problem's grid = TIME_FACTOR x the scan grid's (3 x the nominal fit)
TIME_FACTOR = 2.0


def ic_value(initial: InitialCondition, name: str) -> float:
    """Nominal value of one IC parameter (names as `forster.scan.parameter_names`)."""
    if name == "step1_um":
        return initial.steps_um[0]
    if name == "step2_um":
        return initial.steps_um[1]
    return getattr(initial, name)


def is_valid(candidate: InitialCondition, domain_um: tuple[float, float]) -> bool:
    """Steps inside ``domain_um`` and in order, every Fo in (0, 1]."""
    lo, hi = domain_um
    steps = candidate.steps_um
    return (
        all(lo <= s <= hi for s in steps)
        and all(b > a for a, b in itertools.pairwise(steps))
        and all(0 < f <= 1 for f in candidate.fo_values)
    )


def coarse_noise_model(problem: Problem) -> NoiseModel:
    """Noise model at the best node of a coarse grid (9 points per parameter up to 3
    parameters, 5 for more), as `forster.scan.fit_candidates` fits it."""
    cg = candidate_grid(
        problem.run.initial,
        problem.unc,
        9 if len(problem.names) <= 3 else 5,
        max_candidates=10**6,
    )
    runs = problem._runs(list(cg.candidates))
    return fit_candidates(runs, prior=cg).noise


class Problem:
    """Likelihood and prior of one (run, uncertainties) case on one fixed time grid.

    ``noise``: the noise model to use (None: `coarse_noise_model`). ``time_factor``:
    the grid runs to ``time_factor`` x the scan grid's max time (3 x the nominal fit).
    ``n_runs`` counts forward runs since it was last reset.
    """

    def __init__(
        self,
        run,
        profiles,
        unc: Mapping[str, Uncertainty],
        *,
        noise: NoiseModel | None = None,
        time_factor: float = TIME_FACTOR,
    ):
        self.run, self.unc = run, dict(unc)
        self.profile = profiles[run.profile_name]
        self.nominal_days = fit_run(run, profiles).time_days
        req = replace(run.scan_grid, max_time_days=None)
        self.grid = req.resolve(
            self.profile,
            run.conditions,
            run.orientation,
            run.initial,
            run.diffusivity,
            purpose="scan",
            time_days=self.nominal_days,
            scheme=run.scheme,
        ).grid
        # longer run so wide priors do not truncate the time posterior
        self.time_factor = time_factor
        self.grid = replace(self.grid, max_time_days=self.grid.max_time_days * time_factor)
        self.names = tuple(n for n in parameter_names(run.initial) if n in self.unc)
        self.nominal = np.array([ic_value(run.initial, n) for n in self.names])
        self.spread = np.array([self.unc[n].spread for n in self.names])
        self.kinds = [self.unc[n].kind for n in self.names]
        half = np.where([k == "normal" for k in self.kinds], NORMAL_SPAN, 1.0) * self.spread
        self.lo, self.hi = self.nominal - half, self.nominal + half
        self.n_runs = 0
        self.noise = coarse_noise_model(self) if noise is None else noise
        probe = self._runs([run.initial])
        self.n_runs = 0
        self.x_um, self.y = probe.x_um, probe.fo
        self.n = len(self.y)
        self.chol = np.linalg.cholesky(self.noise.covariance(self.x_um))
        self.days_per_row = self.grid.dt_s / 86400
        self.rows = self.grid.timesteps + 1

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
        """Log prior up to a constant: 0 or the Gaussian term inside the box, else -inf.
        IC validity (steps, Fo) is applied in `loglike_rows`."""
        theta = np.atleast_2d(theta)
        inside = np.all((theta >= self.lo) & (theta <= self.hi), axis=1)
        lp = np.zeros(len(theta))
        for j, k in enumerate(self.kinds):
            if k == "normal":
                lp += -0.5 * ((theta[:, j] - self.nominal[j]) / self.spread[j]) ** 2
        return np.where(inside, lp, -np.inf)

    def prior_transform(self, u):
        """Unit cube -> theta, for nested sampling; the same prior as `log_prior`."""
        th = np.empty(len(self.names))
        a = stats.norm.cdf(-NORMAL_SPAN)
        for j, k in enumerate(self.kinds):
            if k == "normal":
                th[j] = self.nominal[j] + self.spread[j] * stats.norm.ppf(a + u[j] * (1 - 2 * a))
            else:
                th[j] = self.lo[j] + u[j] * (self.hi[j] - self.lo[j])
        return th

    def initial_for(self, th) -> InitialCondition | None:
        """The IC at ``th``, or None if it is invalid (zero prior)."""
        try:
            values = dict(zip(self.names, map(float, th), strict=True))
            cand = _with_values(self.run.initial, values)
        except ValueError:
            return None
        return cand if is_valid(cand, self.run.initial.domain_um) else None

    def loglike_rows(self, theta):
        """(len(theta), rows) log-likelihood -n/2 log chi2; -inf rows for invalid ICs."""
        theta = np.atleast_2d(theta)
        out = np.full((len(theta), self.rows), -np.inf)
        initials, idx = [], []
        for i, th in enumerate(theta):
            cand = self.initial_for(th)
            if cand is not None:
                initials.append(cand)
                idx.append(i)
        if initials:
            r = self._runs(initials)
            for i, m in zip(idx, r.at_points, strict=True):
                c = _whitened_chi2(m, self.y, self.chol)
                out[i, : len(c)] = -0.5 * self.n * np.log(c)
        return out

    def summarise(self, log_w_rows):
        """Percentiles 16/50/84 of time (days) and the posterior mass in the last 5% of
        the time axis, from per-(sample, row) log weights."""
        p = np.exp(logsumexp(log_w_rows, axis=0) - logsumexp(log_w_rows))
        cdf = np.cumsum(p)
        rows = np.arange(len(p))
        q = np.interp([0.16, 0.5, 0.84], cdf, rows) * self.days_per_row
        tail = p[-len(p) // 20 :].sum()
        return q, tail
