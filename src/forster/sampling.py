"""Initial-condition uncertainty by optimisation and sampling: `fit_initial_condition`.

The grid scan (`forster.scan`) fits every node of a grid over the uncertain
initial-condition parameters. On round robin profiles a grid needed about 15 points
per parameter to converge (its default of 5 put the time up to 19% off), because the
data pin the parameters much more tightly than their priors do. This module spends
its model runs where the posterior is instead
(``analyses/ic_sampling/sweep_methods.py``):

1. **Best fit.** Nelder-Mead over the uncertain parameters (in units of their prior
   spread) on minus the log posterior with time marginalised: one model run per
   evaluation gives chi-squared at every time step, so time is never sampled. The
   first search scores with white noise; the noise model
   (`forster.uncertainty.fit_noise_model`) is then fitted at that data-driven optimum,
   and refitted at each new optimum until it is stable (`NOISE_REFITS`). Fitted at the
   prior's centre instead, a prior centred away from the data inflated the correlated
   term 4-18x and left the time unconstrained.
2. **Laplace proposal and importance sampling.** A multivariate t (5 degrees of
   freedom, the Laplace covariance from a finite-difference Hessian at the optimum,
   inflated 1.5x) proposes ``n_samples`` initial conditions from a scrambled Sobol
   sequence (seeded, so a seed reproduces the run). Each is weighted by prior x
   likelihood / proposal, which corrects the proposal to the exact posterior up to
   sampling error: within 0.5% of a dense reference grid wherever the data pin the
   initial condition.
3. **Nested sampling fallback.** When the effective sample size is below
   ``ess_fallback`` of the samples (a parameter flat across a wide prior, or several
   modes; the proposal then misses part of the posterior) the posterior is sampled with
   dynesty instead (static nested sampling, seeded), which was within 1-3% in every
   case tried.

Both give the marginal posterior of time (integrated over the initial condition), the
posterior of each parameter, the split of the variance of time into the part from
the fit at a fixed initial condition and the part from the choice of initial
condition (law of total variance), each parameter's sensitivity dE[t]/dvalue, and the
evidence. Priors are as in `forster.scan.candidate_grid`: Gaussian truncated at
`forster.scan.NORMAL_SPAN` sigma, or uniform over +- the half-width.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
import scipy.linalg as sla
from scipy import optimize, special, stats

from forster import notes
from forster.diffusivity import DEFAULT_DIFFUSIVITY, Diffusivity
from forster.model import (
    DEFAULT_SCHEME,
    SECONDS_PER_DAY,
    Conditions,
    Grid,
    InitialCondition,
    Orientation,
    Profile,
    _diffusion_model,
    _explicit_substeps,
    check_curve_memory,
    fe_mg_diffusivity,
    initial_grid,
    measured_points,
    near_step_at_best_initial,
)
from forster.scan import (
    NORMAL_SPAN,
    Uncertainty,
    _nominal,
    _with_values,
    is_valid,
    parameter_names,
    resolve_names,
    warn_prior_conflict,
)
from forster.uncertainty import (
    ONE_SIGMA,
    NoiseModel,
    central_range,
    fit_noise_model,
    interpolation_matrix,
    points_in_domain,
)

METHODS = ("auto", "laplace", "nested")
DEFAULT_SAMPLES = 1024
DEFAULT_NLIVE = 200
# Below this effective sample size per sample, "auto" falls back to nested sampling. On
# 35 real profiles x 2 priors against a nested-sampling reference (nlive 1000;
# analyses/ic_sampling/references.csv), every case Laplace kept at a threshold of
# 0.15-0.40 was within 0.81% on p16/50/84; the six cases more than 2% off all had 0.11
# or less, and the next case above 0.25 had 0.26 (0.6% off).
ESS_FALLBACK = 0.25
# Those profiles had 2-5 uncertain parameters, mostly 3-4. The proposal's ESS fraction
# falls with the number of parameters even when the posterior is exactly its Laplace
# Gaussian (`ideal_ess_fraction`: 0.64 at 2, 0.48 at 4, 0.34 at 8, 0.25 at 15), so a joint
# fit of many parameters scales the threshold by ideal(d) / ideal(ESS_REFERENCE_DIMENSION)
# (`sample_posterior`'s ``scaled``). Not yet checked against nested-sampling references.
ESS_REFERENCE_DIMENSION = 4
MAX_SAMPLE_FACTOR = 4  # scaled: importance samples doubled up to this many x n_samples
PROPOSAL_DF = 5
PROPOSAL_INFLATION = 1.5
HESSIAN_STEP = 0.15  # finite-difference step, in prior spreads
SIMPLEX_STEP = 0.5  # Nelder-Mead start simplex, in prior spreads
TAIL_FRACTION = 0.05  # warn if more than TAIL_MASS of time lies in this last part of the run
TAIL_MASS = 1e-3
_OUTSIDE = 1e12  # objective outside the prior's support


class Cancelled(Exception):
    """Raised inside `fit_initial_condition` when its ``cancel`` hook returns True."""


def model_at_points(
    scheme: str,
    diffusivity: Diffusivity,
    conditions: Conditions,
    orientation: Orientation,
    initial: InitialCondition,
    grid: Grid,
    model_x: np.ndarray,
    dx_um: float,
    W: np.ndarray,
) -> np.ndarray:
    """One model run from ``initial`` on ``model_x``, read at the points of the
    interpolation matrix ``W``: (time steps + 1, points)."""
    substeps = 1
    if scheme != "implicit":
        substeps = _explicit_substeps(
            diffusivity, conditions, orientation, initial, None, None, grid, dx_um
        )
    start, advance = _diffusion_model(
        scheme,
        conditions,
        orientation,
        initial,
        grid,
        diffusivity,
        model_x,
        dx_um,
        substeps=substeps,
    )
    return advance(start, grid.timesteps) @ W.T


class PriorLikelihood:
    """The prior over a parameter vector and the likelihood interface the optimiser and
    samplers below use (`ICLikelihood` for one profile, `forster.joint.JointLikelihood`
    for a crystal's profiles).

    Each parameter has a nominal value and a Gaussian (sd ``spread``, truncated at
    `NORMAL_SPAN` sigma) or uniform (half-width ``spread``) prior. A subclass sets
    ``names``, ``grid`` and ``rows`` (time steps + 1), calls `_set_prior` and gives
    ``log_like_rows(theta)``: the log likelihood at every time step, or None for
    invalid values.
    """

    names: tuple[str, ...]
    grid: Grid
    rows: int

    def _set_prior(self, nominal: np.ndarray, spread: np.ndarray, normal: np.ndarray) -> None:
        self.nominal, self.spread, self.normal = nominal, spread, normal
        half = np.where(self.normal, NORMAL_SPAN, 1.0) * self.spread
        self.lo, self.hi = self.nominal - half, self.nominal + half
        # log of the normalised prior density's constant (truncated Gaussian or uniform)
        mass = 1 - 2 * stats.norm.cdf(-NORMAL_SPAN)
        self.log_norm = float(
            np.sum(
                np.where(
                    self.normal,
                    -np.log(self.spread * math.sqrt(2 * math.pi) * mass),
                    -np.log(2 * self.spread),
                )
            )
        )

    def log_prior(self, theta: np.ndarray) -> float:
        """Normalised log prior density (-inf outside the support)."""
        if np.any(theta < self.lo) or np.any(theta > self.hi):
            return -math.inf
        z = (theta - self.nominal) / self.spread
        return self.log_norm - 0.5 * float(np.sum(np.where(self.normal, z**2, 0.0)))

    def log_like_rows(self, theta: np.ndarray) -> np.ndarray | None:
        """The log likelihood at every time step, or None for invalid values."""
        raise NotImplementedError

    def log_marginal(self, theta: np.ndarray) -> float:
        """Log likelihood with time marginalised (flat prior over the run's time steps)."""
        rows = self.log_like_rows(theta)
        return -math.inf if rows is None else float(special.logsumexp(rows) - math.log(self.rows))


class ICLikelihood(PriorLikelihood):
    """Log prior and per-time log likelihood of the uncertain initial-condition
    parameters, for one profile, grid and noise model.

    ``log_like_rows(theta)`` runs the model once from the initial condition with
    values ``theta`` (in the order of ``names``) and gives, for every time step,
    -n/2 log chi2 (the noise scale integrated out, as `forster.scan.fit_candidates`).
    """

    def __init__(
        self,
        profile: Profile,
        conditions: Conditions,
        orientation: Orientation,
        initial: InitialCondition,
        uncertainties: Mapping[str, Uncertainty],
        grid: Grid,
        diffusivity: Diffusivity | str = DEFAULT_DIFFUSIVITY,
        scheme: str = DEFAULT_SCHEME,
    ) -> None:
        uncertainties = resolve_names(initial, uncertainties)
        self.names = tuple(n for n in parameter_names(initial) if n in uncertainties)
        if not self.names:
            raise ValueError("no uncertain parameters given")
        self.initial, self.conditions, self.orientation = initial, conditions, orientation
        self.grid, self.scheme = grid, scheme
        self.diffusivity = fe_mg_diffusivity(diffusivity)
        self._set_prior(
            np.array([_nominal(initial, n) for n in self.names]),
            np.array([uncertainties[n].spread for n in self.names]),
            np.array([uncertainties[n].kind == "normal" for n in self.names]),
        )
        measured_x, measured_fo = measured_points(profile, initial)
        step_x, self.dx_um, (low, high) = initial_grid(scheme, measured_x, grid, initial)
        self.model_x = step_x[low : high + 1]
        check_curve_memory(grid.timesteps + 1, len(self.model_x))
        self.x, self.y = points_in_domain(self.model_x, measured_x, measured_fo)
        self.W = interpolation_matrix(self.model_x, self.x)
        self.rows = grid.timesteps + 1
        self.runs = 0
        # Called before every model run; True stops the fit (`Cancelled`).
        self.cancel: Callable[[], bool] | None = None
        self.chol: np.ndarray | None = None

    def candidate(self, theta: np.ndarray) -> InitialCondition | None:
        """The initial condition with values ``theta``, or None if it is invalid."""
        try:
            c = _with_values(self.initial, dict(zip(self.names, map(float, theta), strict=True)))
        except ValueError:
            return None
        return c if is_valid(c, self.initial.domain_um) else None

    def at_points(self, initial: InitialCondition) -> np.ndarray:
        """The model at the measured points, (time steps + 1, points)."""
        if self.cancel is not None and self.cancel():
            raise Cancelled("the initial-condition fit was cancelled")
        self.runs += 1
        return model_at_points(
            self.scheme,
            self.diffusivity,
            self.conditions,
            self.orientation,
            initial,
            self.grid,
            self.model_x,
            self.dx_um,
            self.W,
        )

    def fit_noise(self, initial: InitialCondition) -> NoiseModel:
        """Noise model at ``initial``'s white-noise best fit; it is used from then on."""
        m = self.at_points(initial)
        best = int(np.argmin(np.sum((m - self.y) ** 2, axis=1)))
        return self.use_noise(fit_noise_model(self.x, self.y - m[best]))

    def use_noise(self, noise: NoiseModel) -> NoiseModel:
        """Use ``noise`` from now on (for example the one an earlier fit ended with)."""
        self.chol = np.linalg.cholesky(noise.covariance(self.x))
        return noise

    def log_like_rows(self, theta: np.ndarray) -> np.ndarray | None:
        """-n/2 log chi2 at every time step, or None for an invalid initial condition."""
        if self.chol is None:
            raise RuntimeError("fit the noise model first (fit_noise)")
        initial = self.candidate(theta)
        if initial is None:
            return None
        m = self.at_points(initial)
        z = sla.solve_triangular(self.chol, (m - self.y).T, lower=True)
        return -0.5 * len(self.y) * np.log(np.sum(z**2, axis=0))


@dataclass(frozen=True)
class ICPosterior:
    """Posterior of time and of the uncertain initial-condition parameters."""

    method: str  # "laplace" or "nested": the one whose samples were used
    names: tuple[str, ...]
    best_values: dict[str, float]  # the optimum
    best_initial: InitialCondition
    best_days: float  # time of the best fit at the optimum
    median_days: float
    low_days: float  # central 68% of the marginal posterior of time
    high_days: float
    ic_share: float  # fraction of the variance of time from the choice of initial condition
    mean: dict[str, float]  # posterior mean and sd of each parameter
    sd: dict[str, float]
    sensitivity: dict[str, float]  # dE[t | values] / dvalue, days per unit
    correlation: dict[str, float]  # correlation of each parameter with E[t | values]
    ess: float  # effective sample size of the importance samples
    n_samples: int
    log_evidence: float
    runs: int  # model runs used
    seed: int
    noise: NoiseModel
    grid: Grid
    days: np.ndarray  # time axis of `density`
    density: np.ndarray  # marginal posterior of time, per day
    # `forster.uncertainty.near_step_check` at the best initial condition and time (below
    # -6: the misfit is still concentrated near the step)
    near_step_delta_aic: float = math.nan
    # each parameter's best value, (value - nominal) / spread (prior sd or half-width)
    prior_offsets: dict[str, float] = field(default_factory=dict)

    @property
    def time_days(self) -> float:
        return self.best_days

    @property
    def range_days(self) -> tuple[float, float]:
        return self.low_days, self.high_days


def _optimise(like: PriorLikelihood, z0: np.ndarray, max_evals: int) -> tuple[np.ndarray, float]:
    """Nelder-Mead on minus the log posterior, in prior spreads from the nominal values."""

    def objective(z: np.ndarray) -> float:
        theta = like.nominal + z * like.spread
        lp = like.log_prior(theta)
        if not math.isfinite(lp):
            return _OUTSIDE
        lm = like.log_marginal(theta)
        return _OUTSIDE if not math.isfinite(lm) else -(lp + lm)

    d = len(z0)
    simplex = np.vstack([z0, z0 + SIMPLEX_STEP * np.eye(d)])
    result = optimize.minimize(
        objective,
        z0,
        method="Nelder-Mead",
        options={"initial_simplex": simplex, "xatol": 0.01, "fatol": 0.005, "maxfev": max_evals},
    )
    return np.asarray(result.x), float(result.fun)


def _fd_hessian(
    f: Callable[[np.ndarray], float], z0: np.ndarray, f0: float, steps: np.ndarray, h: float
) -> np.ndarray:
    """Finite-difference Hessian of ``f`` at ``z0`` along the step vectors ``steps[i]``
    (each of length ``h`` in the coordinates the Hessian is wanted in)."""
    d = len(steps)
    H = np.zeros((d, d))
    for i in range(d):
        H[i, i] = (f(z0 + steps[i]) - 2 * f0 + f(z0 - steps[i])) / h**2
        for j in range(i + 1, d):
            H[i, j] = H[j, i] = (
                f(z0 + steps[i] + steps[j])
                - f(z0 + steps[i] - steps[j])
                - f(z0 - steps[i] + steps[j])
                + f(z0 - steps[i] - steps[j])
            ) / (4 * h**2)
    return H


def _laplace_covariance(
    like: PriorLikelihood, z0: np.ndarray, f0: float, whiten: bool = False
) -> np.ndarray:
    """Inverse finite-difference Hessian of minus the log posterior at ``z0`` (prior
    spreads); the diagonal alone, or the prior, where that is not positive definite.

    ``whiten``: a second pass along the first covariance's Cholesky directions, so every
    step is `HESSIAN_STEP` posterior sd rather than prior sd. A parameter the data pin
    far inside its prior is then probed within the quadratic part of the posterior; on a
    crystal's joint fit, where each profile's Fo offset and the shared segment Fo are
    degenerate but for their priors, the first pass made that direction 2.5x too narrow
    and the importance samples' ESS fell from 340 to 64 of 1024 (forster.joint)."""

    def f(z: np.ndarray) -> float:
        theta = like.nominal + z * like.spread
        lp = like.log_prior(theta)
        lm = like.log_marginal(theta) if math.isfinite(lp) else -math.inf
        return -(lp + lm) if math.isfinite(lm) else f0 + 50.0

    d, h = len(z0), HESSIAN_STEP
    H = _fd_hessian(f, z0, f0, np.eye(d) * h, h)
    try:
        cov = np.linalg.inv(H)
        np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        return np.diag(1 / np.clip(np.diag(H), 1.0, None))
    if not whiten:
        return cov
    L = np.linalg.cholesky(cov)
    L_inv = np.linalg.inv(L)
    H_white = _fd_hessian(f, z0, f0, h * L.T, h)  # the Hessian in w, z = z0 + L w
    try:
        refined = np.linalg.inv(L_inv.T @ H_white @ L_inv)
        np.linalg.cholesky(refined)
        return refined
    except np.linalg.LinAlgError:
        return cov


@dataclass
class _Samples:
    theta: np.ndarray  # (samples, parameters)
    log_w: np.ndarray  # log importance weight (prior / proposal); posterior draws: see _nested
    rows: np.ndarray  # (samples, time steps) log likelihood


def _log_t_density(z: np.ndarray, loc: np.ndarray, chol: np.ndarray, df: float) -> np.ndarray:
    """Log density of a multivariate t with scale ``chol @ chol.T``, from the Cholesky
    factor itself (scipy's multivariate_t refits it and can reject a nearly singular
    matrix that factorised: 11 of 70 validation cases with wide priors)."""
    d = len(loc)
    m = np.sum(sla.solve_triangular(chol, (z - loc).T, lower=True) ** 2, axis=0)
    return (
        special.gammaln((df + d) / 2)
        - special.gammaln(df / 2)
        - d / 2 * math.log(df * math.pi)
        - float(np.sum(np.log(np.diag(chol))))
        - (df + d) / 2 * np.log1p(m / df)
    )


def _importance(
    like: PriorLikelihood, z0: np.ndarray, n: int, seed: int, whiten: bool = False
) -> tuple[_Samples, float]:
    f0 = -_log_post(like, z0)
    cov = _laplace_covariance(like, z0, f0, whiten) * PROPOSAL_INFLATION**2
    d = len(z0)
    u = stats.qmc.Sobol(d + 1, scramble=True, seed=seed).random(n)
    u = np.clip(u, 1e-12, 1 - 1e-12)
    chi = stats.chi2.ppf(u[:, d], PROPOSAL_DF)
    chol = np.linalg.cholesky(cov)
    z = z0 + (stats.norm.ppf(u[:, :d]) @ chol.T) / np.sqrt(chi / PROPOSAL_DF)[:, None]
    log_q = _log_t_density(z, z0, chol, PROPOSAL_DF) - float(np.sum(np.log(like.spread)))
    thetas, log_ws, rows = [], [], []
    for zi, lq in zip(z, log_q, strict=True):
        theta = like.nominal + zi * like.spread
        lp = like.log_prior(theta)
        r = like.log_like_rows(theta) if math.isfinite(lp) else None
        if r is not None:
            thetas.append(theta)
            log_ws.append(lp - lq)
            rows.append(r)
    if not rows:
        raise ValueError("no valid initial condition among the proposals")
    s = _Samples(np.array(thetas), np.array(log_ws), np.array(rows))
    marginal = s.log_w + special.logsumexp(s.rows, axis=1)
    w = np.exp(marginal - marginal.max())
    return s, float(w.sum() ** 2 / (w**2).sum())


def _log_post(like: PriorLikelihood, z: np.ndarray) -> float:
    theta = like.nominal + z * like.spread
    lp = like.log_prior(theta)
    return lp + like.log_marginal(theta) if math.isfinite(lp) else -math.inf


def _nested(like: PriorLikelihood, nlive: int, seed: int) -> tuple[_Samples, float]:
    import dynesty  # imported here: only the fallback needs it

    d = len(like.names)
    a = stats.norm.cdf(-NORMAL_SPAN)

    def transform(u: np.ndarray) -> np.ndarray:
        gauss = like.nominal + like.spread * stats.norm.ppf(a + u * (1 - 2 * a))
        return np.where(like.normal, gauss, like.lo + u * (like.hi - like.lo))

    def loglike(theta: np.ndarray) -> float:
        value = like.log_marginal(theta)
        return value if math.isfinite(value) else -1e300

    # Keyword arguments through a dict: NestedSampler builds the sampler in __new__, and
    # a type checker reading Sampler.__init__ would reject nlive and rstate.
    options: dict = {"nlive": nlive, "rstate": np.random.default_rng(seed)}
    sampler = dynesty.NestedSampler(loglike, transform, d, **options)
    sampler.run_nested(dlogz=0.1, print_progress=False)
    result = sampler.results
    draws = result.samples_equal(rstate=np.random.default_rng(seed))
    pick = np.random.default_rng(seed).choice(len(draws), min(1000, len(draws)), replace=False)
    thetas, rows = [], []
    for theta in draws[pick]:
        r = like.log_like_rows(theta)
        if r is not None:
            thetas.append(theta)
            rows.append(r)
    rows_array = np.array(rows)
    # Posterior draws: each weighs the same, so the weight cancels its marginal likelihood
    # (`_summarise` multiplies by it, as importance samples need).
    s = _Samples(np.array(thetas), -special.logsumexp(rows_array, axis=1), rows_array)
    return s, float(result.logz[-1])


def _summarise(like: PriorLikelihood, s: _Samples) -> dict:
    """Time posterior, parameter posterior, variance split and sensitivities."""
    days_per_row = like.grid.dt_s / SECONDS_PER_DAY
    joint = s.log_w[:, None] + s.rows
    density = np.exp(special.logsumexp(joint, axis=0) - special.logsumexp(joint))
    low, high = central_range(density, ONE_SIGMA)
    cdf = np.cumsum(density)
    median = float(np.interp(0.5, cdf, np.arange(like.rows)))
    t = np.arange(like.rows) * days_per_row
    cond = np.exp(s.rows - special.logsumexp(s.rows, axis=1, keepdims=True))
    e_t = cond @ t
    var_t = cond @ t**2 - e_t**2
    marginal = s.log_w + special.logsumexp(s.rows, axis=1)
    omega = np.exp(marginal - special.logsumexp(marginal))
    mean_t = float(omega @ e_t)
    between = float(omega @ (e_t - mean_t) ** 2)
    within = float(omega @ var_t)
    mean = omega @ s.theta
    centred = s.theta - mean
    sd = np.sqrt(omega @ centred**2)
    cov_t = omega @ (centred * (e_t - mean_t)[:, None])
    sd_e = math.sqrt(between) if between > 0 else math.nan
    slopes = np.full(len(like.names), math.nan)
    try:
        X = np.column_stack([np.ones(len(omega)), centred])
        sw = np.sqrt(omega)[:, None]
        slopes = np.linalg.lstsq(X * sw, (e_t - mean_t) * sw[:, 0], rcond=None)[0][1:]
    except np.linalg.LinAlgError:
        pass
    tail = float(density[-max(1, int(TAIL_FRACTION * like.rows)) :].sum())
    return {
        "density": density / days_per_row,
        "days": t,
        "low_days": low * days_per_row,
        "high_days": high * days_per_row,
        "median_days": median * days_per_row,
        "ic_share": between / (between + within) if between + within > 0 else 0.0,
        "mean": dict(zip(like.names, map(float, mean), strict=True)),
        "sd": dict(zip(like.names, map(float, sd), strict=True)),
        "sensitivity": dict(zip(like.names, map(float, slopes), strict=True)),
        "correlation": {
            n: float(c / (sdi * sd_e)) if sdi > 0 and sd_e > 0 else math.nan
            for n, c, sdi in zip(like.names, cov_t, sd, strict=True)
        },
        "tail": tail,
    }


# The noise model is refitted at each new best fit until it stops changing (relative
# change in its standard deviations below NOISE_TOLERANCE), at most NOISE_REFITS times.
NOISE_REFITS = 4
NOISE_TOLERANCE = 0.05


def _noise_changed(a: NoiseModel, b: NoiseModel) -> bool:
    """Whether two noise models differ by more than `NOISE_TOLERANCE` in their total sd at
    zero lag or in the correlated part."""
    total_a = math.hypot(a.white_sigma, a.correlated_sigma)
    total_b = math.hypot(b.white_sigma, b.correlated_sigma)
    return (
        abs(total_a - total_b) > NOISE_TOLERANCE * total_a
        or abs(a.correlated_sigma - b.correlated_sigma) > NOISE_TOLERANCE * total_a
    )


def at_prior_edge(like: PriorLikelihood, z: np.ndarray) -> list[str]:
    """The parameters whose value ``z`` (in prior spreads) is at the edge of the prior."""
    return [
        n
        for n, zi, k in zip(like.names, z, like.normal, strict=True)
        if abs(zi) >= (NORMAL_SPAN if k else 1.0) - 1e-3
    ]


@functools.cache
def ideal_ess_fraction(d: int, n: int = 100_000) -> float:
    """ESS per sample of the importance proposal (multivariate t, `PROPOSAL_DF` degrees of
    freedom, scale inflated `PROPOSAL_INFLATION` x) when the posterior is exactly the
    Gaussian it was built from, in ``d`` dimensions: the best the sampler can do
    (seeded Monte Carlo of ``n`` draws; 0.78 at 1, 0.48 at 4, 0.34 at 8)."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal((n, d)) / np.sqrt(rng.chisquare(PROPOSAL_DF, n) / PROPOSAL_DF)[:, None]
    m = np.sum(x**2, axis=1)  # squared distance in the proposal's scale
    x *= PROPOSAL_INFLATION
    log_q = -d * math.log(PROPOSAL_INFLATION) - (PROPOSAL_DF + d) / 2 * np.log1p(m / PROPOSAL_DF)
    log_w = -0.5 * np.sum(x**2, axis=1) - log_q  # up to a constant
    w = np.exp(log_w - log_w.max())
    return float(w.sum() ** 2 / (w**2).sum() / n)


def sample_posterior(
    like: PriorLikelihood,
    z: np.ndarray,
    method: str,
    n_samples: int,
    nlive: int,
    seed: int,
    ess_fallback: float,
    say: Callable[[str], None],
    whiten: bool = False,
    scaled: bool = False,
) -> tuple[_Samples, str, float, float]:
    """Samples of the posterior around the best fit ``z`` (prior spreads), the method
    used ("laplace" or "nested"), the importance samples' effective sample size and the
    log evidence: importance sampling from the Laplace proposal, nested sampling for
    "nested" or when "auto" finds the effective sample size below ``ess_fallback`` x
    ``n_samples``. ``whiten``: refine the Laplace covariance (`_laplace_covariance`).

    ``scaled`` (for many parameters): the fallback threshold per sample is
    ``ess_fallback`` x ideal(d) / ideal(`ESS_REFERENCE_DIMENSION`) (`ideal_ess_fraction`),
    and while the samples pass it but their ESS is below ``ess_fallback`` x
    ``n_samples`` (the single-profile fit's standard), their number is doubled, up to
    `MAX_SAMPLE_FACTOR` x ``n_samples``."""
    used, ess, log_z = method, math.nan, math.nan
    samples = None
    if method in ("auto", "laplace"):
        say("importance sampling")
        n, fraction = n_samples, ess_fallback
        if scaled:
            d = len(z)
            fraction *= ideal_ess_fraction(d) / ideal_ess_fraction(ESS_REFERENCE_DIMENSION)
        samples, ess = _importance(like, z, n, seed, whiten)
        while (
            scaled
            and fraction * n <= ess < ess_fallback * n_samples
            and n < MAX_SAMPLE_FACTOR * n_samples
        ):
            n *= 2
            samples, ess = _importance(like, z, n, seed, whiten)
        joint = samples.log_w + special.logsumexp(samples.rows, axis=1) - math.log(like.rows)
        log_z = float(special.logsumexp(joint) - math.log(n))
        used = "laplace"
        if method == "auto" and ess < fraction * n:
            samples = None
    if samples is None:
        say("nested sampling")
        samples, log_z = _nested(like, nlive, seed)
        used = "nested"
    return samples, used, ess, log_z


def _best_fit(
    like: ICLikelihood, initial: InitialCondition, d: int
) -> tuple[np.ndarray, InitialCondition, NoiseModel]:
    """The best-fit parameters (in prior units), that initial condition, and the noise
    model fitted at it.

    The first search scores with white noise, so the first noise model is fitted at a
    data-driven best fit, not at the prior's centre: fitted at an initial condition the data
    reject, the correlated term absorbs that misfit (4-18x larger on the paper profiles,
    docs/plans/2026-10-16-population-priors) and leaves the time unconstrained. The noise
    model is then refitted at each new best fit until it is stable."""
    white = like.fit_noise(initial)
    like.use_noise(NoiseModel(white.white_sigma, 0.0, 1.0))
    z, _ = _optimise(like, np.zeros(d), 200 * d)
    noise, best = white, initial
    for _ in range(NOISE_REFITS):
        best = like.candidate(like.nominal + z * like.spread)
        if best is None:  # the optimiser keeps to valid values; guard against the impossible
            raise ValueError("the best-fit initial condition is invalid")
        refit = like.fit_noise(best)
        z, _ = _optimise(like, z, 100 * d)
        stable = not _noise_changed(noise, refit)
        noise = refit
        if stable:
            break
    return z, best, noise


def fit_initial_condition(
    profile: Profile,
    conditions: Conditions,
    orientation: Orientation,
    initial: InitialCondition,
    uncertainties: Mapping[str, Uncertainty],
    grid: Grid,
    diffusivity: Diffusivity | str = DEFAULT_DIFFUSIVITY,
    scheme: str = DEFAULT_SCHEME,
    *,
    method: str = "auto",
    n_samples: int = DEFAULT_SAMPLES,
    nlive: int = DEFAULT_NLIVE,
    seed: int = 0,
    ess_fallback: float = ESS_FALLBACK,
    progress: Callable[[str], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> ICPosterior:
    """Best-fit initial condition and the posterior of time over its uncertain parameters.

    ``uncertainties`` as for `forster.scan.candidate_grid` (``step1_um``, ``step2_um``, ...,
    ``a_fo``, ``b_fo``, ... or the old names
    ``edge_fo``, ``core_fo``, ``step_fo``). ``method``: "auto" (importance sampling
    from the Laplace proposal, nested sampling if its effective sample size is below
    ``ess_fallback`` x ``n_samples``), "laplace" or "nested". The same ``seed`` gives
    the same result. Every run uses ``grid`` (a scan grid: `forster.grid.choose_grid`
    with purpose "scan" is enough for the 5% the ranges need). ``progress(stage)`` is
    called at each stage. ``cancel()`` is called before every model run (in the
    optimiser, the Hessian and both samplers) and raises `Cancelled` when it returns
    True; otherwise it leaves the result unchanged. See the module docstring.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; use one of {', '.join(METHODS)}")
    like = ICLikelihood(
        profile, conditions, orientation, initial, uncertainties, grid, diffusivity, scheme
    )
    like.cancel = cancel
    say = progress or (lambda stage: None)
    d = len(like.names)
    say("best fit")
    z, best, noise = _best_fit(like, initial, d)
    best_theta = like.nominal + z * like.spread
    best = like.candidate(best_theta) or best
    best_rows = like.log_like_rows(best_theta)
    best_row = int(np.argmax(best_rows)) if best_rows is not None else 0
    at_edge = at_prior_edge(like, z)
    if at_edge:
        notes.warn(
            f"the best {', '.join(at_edge)} is at the edge of its prior; the data favour "
            "values beyond it, so the range may be cut short",
            stacklevel=2,
        )
    offsets = {n: float(zi) for n, zi in zip(like.names, z, strict=True)}
    kinds = {n: "normal" if k else "uniform" for n, k in zip(like.names, like.normal, strict=True)}
    warn_prior_conflict(offsets, kinds)
    samples, used, ess, log_z = sample_posterior(
        like, z, method, n_samples, nlive, seed, ess_fallback, say
    )
    summary = _summarise(like, samples)
    best_days = best_row * grid.dt_s / SECONDS_PER_DAY
    near = near_step_at_best_initial(
        like.x,
        like.y,
        like.at_points(best)[best_row],
        best,
        best_days,
        like.diffusivity,
        like.conditions,
        like.orientation,
    )
    if summary["tail"] > TAIL_MASS:
        notes.warn(
            f"{summary['tail']:.1%} of the posterior of time lies in the last "
            f"{TAIL_FRACTION:.0%} of the run; give a longer scan_max_time_days",
            stacklevel=2,
        )
    return ICPosterior(
        method=used,
        names=like.names,
        best_values=dict(zip(like.names, map(float, best_theta), strict=True)),
        best_initial=best,
        best_days=best_days,
        median_days=summary["median_days"],
        low_days=summary["low_days"],
        high_days=summary["high_days"],
        ic_share=summary["ic_share"],
        mean=summary["mean"],
        sd=summary["sd"],
        sensitivity=summary["sensitivity"],
        correlation=summary["correlation"],
        ess=ess,
        n_samples=len(samples.theta),
        log_evidence=log_z,
        runs=like.runs,
        seed=seed,
        noise=noise,
        grid=grid,
        days=summary["days"],
        density=summary["density"],
        near_step_delta_aic=near.delta_aic,
        prior_offsets=offsets,
    )
