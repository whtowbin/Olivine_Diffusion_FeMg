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
   noise model (`forster.uncertainty.fit_noise_model`) is fitted at the nominal
   initial condition, refitted at the optimum, and the optimum refined once.
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

import math
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import scipy.linalg as sla
from scipy import optimize, special, stats

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
    near_step_at_best_initial,
    scheme_grid,
)
from forster.scan import (
    NORMAL_SPAN,
    Uncertainty,
    _nominal,
    _with_values,
    is_valid,
    parameter_names,
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
PROPOSAL_DF = 5
PROPOSAL_INFLATION = 1.5
HESSIAN_STEP = 0.15  # finite-difference step, in prior spreads
SIMPLEX_STEP = 0.5  # Nelder-Mead start simplex, in prior spreads
TAIL_FRACTION = 0.05  # warn if more than TAIL_MASS of time lies in this last part of the run
TAIL_MASS = 1e-3
_OUTSIDE = 1e12  # objective outside the prior's support


class ICLikelihood:
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
        unknown = sorted(set(uncertainties) - set(parameter_names(initial)))
        if unknown:
            raise ValueError(
                f"no parameter {unknown} in this initial condition; use {parameter_names(initial)}"
            )
        self.names = tuple(n for n in parameter_names(initial) if n in uncertainties)
        if not self.names:
            raise ValueError("no uncertain parameters given")
        self.initial, self.conditions, self.orientation = initial, conditions, orientation
        self.grid, self.scheme = grid, scheme
        self.diffusivity = fe_mg_diffusivity(diffusivity)
        self.nominal = np.array([_nominal(initial, n) for n in self.names])
        self.spread = np.array([uncertainties[n].spread for n in self.names])
        self.normal = np.array([uncertainties[n].kind == "normal" for n in self.names])
        half = np.where(self.normal, NORMAL_SPAN, 1.0) * self.spread
        self.lo, self.hi = self.nominal - half, self.nominal + half
        step_x, self.dx_um, (low, high) = scheme_grid(
            scheme, profile.distance_um, grid, initial.domain_um
        )
        self.model_x = step_x[low : high + 1]
        check_curve_memory(grid.timesteps + 1, len(self.model_x))
        self.x, self.y = points_in_domain(self.model_x, profile.distance_um, profile.fo)
        self.W = interpolation_matrix(self.model_x, self.x)
        self.rows = grid.timesteps + 1
        self.runs = 0
        self.chol: np.ndarray | None = None
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

    def candidate(self, theta: np.ndarray) -> InitialCondition | None:
        """The initial condition with values ``theta``, or None if it is invalid."""
        try:
            c = _with_values(self.initial, dict(zip(self.names, map(float, theta), strict=True)))
        except ValueError:
            return None
        return c if is_valid(c, self.initial.domain_um) else None

    def log_prior(self, theta: np.ndarray) -> float:
        """Normalised log prior density (-inf outside the support)."""
        if np.any(theta < self.lo) or np.any(theta > self.hi):
            return -math.inf
        z = (theta - self.nominal) / self.spread
        return self.log_norm - 0.5 * float(np.sum(np.where(self.normal, z**2, 0.0)))

    def at_points(self, initial: InitialCondition) -> np.ndarray:
        """The model at the measured points, (time steps + 1, points)."""
        substeps = 1
        if self.scheme != "implicit":
            substeps = _explicit_substeps(
                self.diffusivity,
                self.conditions,
                self.orientation,
                initial,
                None,
                None,
                self.grid,
                self.dx_um,
            )
        start, advance = _diffusion_model(
            self.scheme,
            self.conditions,
            self.orientation,
            initial,
            self.grid,
            self.diffusivity,
            self.model_x,
            self.dx_um,
            substeps=substeps,
        )
        self.runs += 1
        return advance(start, self.grid.timesteps) @ self.W.T

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

    def log_marginal(self, theta: np.ndarray) -> float:
        """Log likelihood with time marginalised (flat prior over the run's time steps)."""
        rows = self.log_like_rows(theta)
        return -math.inf if rows is None else float(special.logsumexp(rows) - math.log(self.rows))


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

    @property
    def time_days(self) -> float:
        return self.best_days

    @property
    def range_days(self) -> tuple[float, float]:
        return self.low_days, self.high_days


def _optimise(like: ICLikelihood, z0: np.ndarray, max_evals: int) -> tuple[np.ndarray, float]:
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


def _laplace_covariance(like: ICLikelihood, z0: np.ndarray, f0: float) -> np.ndarray:
    """Inverse finite-difference Hessian of minus the log posterior at ``z0`` (prior
    spreads); the diagonal alone, or the prior, where that is not positive definite."""

    def f(z: np.ndarray) -> float:
        theta = like.nominal + z * like.spread
        lp = like.log_prior(theta)
        lm = like.log_marginal(theta) if math.isfinite(lp) else -math.inf
        return -(lp + lm) if math.isfinite(lm) else f0 + 50.0

    d, h = len(z0), HESSIAN_STEP
    eye = np.eye(d) * h
    H = np.zeros((d, d))
    for i in range(d):
        H[i, i] = (f(z0 + eye[i]) - 2 * f0 + f(z0 - eye[i])) / h**2
        for j in range(i + 1, d):
            H[i, j] = H[j, i] = (
                f(z0 + eye[i] + eye[j])
                - f(z0 + eye[i] - eye[j])
                - f(z0 - eye[i] + eye[j])
                + f(z0 - eye[i] - eye[j])
            ) / (4 * h**2)
    try:
        cov = np.linalg.inv(H)
        np.linalg.cholesky(cov)
        return cov
    except np.linalg.LinAlgError:
        return np.diag(1 / np.clip(np.diag(H), 1.0, None))


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


def _importance(like: ICLikelihood, z0: np.ndarray, n: int, seed: int) -> tuple[_Samples, float]:
    cov = _laplace_covariance(like, z0, -_log_post(like, z0)) * PROPOSAL_INFLATION**2
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


def _log_post(like: ICLikelihood, z: np.ndarray) -> float:
    theta = like.nominal + z * like.spread
    lp = like.log_prior(theta)
    return lp + like.log_marginal(theta) if math.isfinite(lp) else -math.inf


def _nested(like: ICLikelihood, nlive: int, seed: int) -> tuple[_Samples, float]:
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


def _summarise(like: ICLikelihood, s: _Samples) -> dict:
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
) -> ICPosterior:
    """Best-fit initial condition and the posterior of time over its uncertain parameters.

    ``uncertainties`` as for `forster.scan.candidate_grid` (``step1_um``, ``step2_um``,
    ``edge_fo``, ``core_fo``, ``step_fo``). ``method``: "auto" (importance sampling
    from the Laplace proposal, nested sampling if its effective sample size is below
    ``ess_fallback`` x ``n_samples``), "laplace" or "nested". The same ``seed`` gives
    the same result. Every run uses ``grid`` (a scan grid: `forster.grid.choose_grid`
    with purpose "scan" is enough for the 5% the ranges need). ``progress(stage)`` is
    called at each stage. See the module docstring.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; use one of {', '.join(METHODS)}")
    like = ICLikelihood(
        profile, conditions, orientation, initial, uncertainties, grid, diffusivity, scheme
    )
    say = progress or (lambda stage: None)
    d = len(like.names)
    say("best fit")
    like.fit_noise(initial)
    z, _ = _optimise(like, np.zeros(d), 200 * d)
    best = like.candidate(like.nominal + z * like.spread)
    if best is None:  # the optimiser keeps to valid values; guard against the impossible
        raise ValueError("the best-fit initial condition is invalid")
    noise = like.fit_noise(best)
    z, _ = _optimise(like, z, 100 * d)
    best_theta = like.nominal + z * like.spread
    best = like.candidate(best_theta) or best
    best_rows = like.log_like_rows(best_theta)
    best_row = int(np.argmax(best_rows)) if best_rows is not None else 0
    at_edge = [
        n
        for n, zi, k in zip(like.names, z, like.normal, strict=True)
        if abs(zi) >= (NORMAL_SPAN if k else 1.0) - 1e-3
    ]
    if at_edge:
        warnings.warn(
            f"the best {', '.join(at_edge)} is at the edge of its prior; the data favour "
            "values beyond it, so the range may be cut short",
            stacklevel=2,
        )
    used, ess, log_z = method, math.nan, math.nan
    samples = None
    if method in ("auto", "laplace"):
        say("importance sampling")
        samples, ess = _importance(like, z, n_samples, seed)
        joint = samples.log_w + special.logsumexp(samples.rows, axis=1) - math.log(like.rows)
        log_z = float(special.logsumexp(joint) - math.log(n_samples))
        used = "laplace"
        if method == "auto" and ess < ess_fallback * n_samples:
            samples = None
    if samples is None:
        say("nested sampling")
        samples, log_z = _nested(like, nlive, seed)
        used = "nested"
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
        warnings.warn(
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
    )
