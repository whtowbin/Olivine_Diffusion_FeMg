"""Uncertainty from the initial condition: fit over a set of candidate initial conditions.

The correlated-noise likelihood (`forster.uncertainty`) gives the range of time for
one initial condition. A wrong step position or edge composition biases the time
without widening that range (``docs/references/likelihood-method.md``, section 4).
`scan_initial_conditions` fits several candidates, for example the step at a range of
positions from `step_offsets`, and combines them:

- **profile likelihood**: at each time, the best candidate's chi-squared (the lower
  envelope of all candidates' curves); the range of time is where that envelope
  rises by `interval_threshold` for 1 + ``n_free`` fitted parameters;
- **marginal posterior**: with flat priors on time and on the candidates and the
  noise scale integrated out (prior 1/s), the posterior of each (candidate, time) is
  proportional to chi2^(-n/2); summed over candidates it gives the central 68% range
  of time and each candidate's probability.

Each candidate costs one model run, and each run gives chi-squared at every time
step, so time is never sampled. One noise model, fitted at the overall best fit, is
used for every candidate, so their chi-squared values are comparable.

Uncertainties on the initial condition (`Uncertainty`, per parameter: each step
position, and the edge, core and plateau Fo) build the candidates with
`candidate_grid`: a grid over each uncertain parameter, Gaussian (sigma; nodes over
+-2.5 sigma, weighted by the prior density, and a penalty ((v - v0)/sigma)^2 added to
chi-squared for the profile range) or uniform (+- half-width; nodes over the whole
range, flat weights, no penalty). Edge and core Fo are the boundary values held
fixed during diffusion, so scanning them changes the boundary conditions too.

All candidates run on one numerical grid. By default its time step is set so that
D dt / dx^2 = `SCAN_STABILITY` (0.4; the explicit scheme needs 0.5 or less) for the
largest diffusivity over every candidate's compositions. The grid spacing is not
changed here, as coarsening it biases the result
(``docs/references/likelihood-method.md``, section 7); `forster.grid.choose_grid`
chooses it for the accuracy wanted.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace

import numpy as np
import scipy.linalg as sla

from forster import notes
from forster.diffusivity import DEFAULT_DIFFUSIVITY, Diffusivity
from forster.grid import (
    STABILITY_RATIO,
    actual_dx_um,
    stable_dt,
)
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
    _run_past_best_fit,
    check_curve_memory,
    fe_mg_diffusivity,
    initial_grid,
    measured_points,
    near_step_at_best_initial,
)
from forster.uncertainty import (
    ONE_SIGMA,
    Interval,
    NoiseModel,
    central_range,
    delta_chi2_interval,
    fit_noise_model,
    interpolation_matrix,
    interval_threshold,
    points_in_domain,
)

# A candidate's run is extended while chi2 at its end is within this of the overall
# minimum: its posterior weight there is then above about exp(-TAIL_DELTA / 2) = 1e-3.
TAIL_DELTA = 14.0
UNCERTAINTY_KINDS = ("normal", "uniform")
NORMAL_SPAN = 2.5  # Gaussian parameters are scanned over +-NORMAL_SPAN sigma
# Nodes per uncertain parameter when not given: finer for one or two parameters.
DEFAULT_POINTS = {1: 13, 2: 9}
DEFAULT_POINTS_MANY = 5
MAX_CANDIDATES = 2000  # refuse larger grids unless the caller raises the limit
# Scan time step: raised until D_max dt / dx^2 reaches this (the explicit scheme is
# unstable above 0.5). Raising dt to 0.4 changed the scan's best fit by under 0.2% on the
# round robin and template profiles; coarsening dx instead moved it by up to 12%.
SCAN_STABILITY = STABILITY_RATIO


@dataclass(frozen=True)
class Uncertainty:
    """Uncertainty on one initial-condition parameter: a Gaussian ``spread`` (sigma) or a
    uniform one (half-width), in the parameter's units (um for steps, Fo fraction)."""

    spread: float
    kind: str = "normal"

    def __post_init__(self) -> None:
        if self.kind not in UNCERTAINTY_KINDS:
            raise ValueError(f"uncertainty kind {self.kind!r}; use one of {UNCERTAINTY_KINDS}")
        if not (math.isfinite(self.spread) and self.spread > 0):
            raise ValueError(f"an uncertainty needs a positive spread, got {self.spread}")


# A fitted value this many prior sd from its nominal (a Gaussian prior), short of the
# prior's edge, suggests the prior and the data disagree: the prior then pulls the
# starting profile, and the time with it (docs/plans/2026-10-16-population-priors).
PRIOR_CONFLICT_SD = 2.0


def warn_prior_conflict(offsets: Mapping[str, float], kinds: Mapping[str, str]) -> None:
    """Warn for Gaussian-prior parameters fitted `PRIOR_CONFLICT_SD` or more prior sd from
    their nominal value but inside the prior (its edge has its own warning)."""
    conflict = {
        n: z
        for n, z in offsets.items()
        if kinds.get(n) == "normal" and PRIOR_CONFLICT_SD <= abs(z) < NORMAL_SPAN - 1e-3
    }
    if conflict:
        listed = ", ".join(f"{n} {z:+.1f} sd" for n, z in conflict.items())
        notes.warn(
            f"the data place {listed} from the nominal value: the prior and the data disagree, "
            "and the prior pulls the fit. Check the prior (one built from other crystals may "
            "not suit this one) or the starting profile",
            stacklevel=3,
        )


def parameter_names(initial: InitialCondition) -> tuple[str, ...]:
    """Names of ``initial``'s parameters, as `candidate_grid` and the sampler take them:
    ``step1_um``, ``step2_um``, ... and ``<segment>_fo`` for each segment (``a_fo``,
    ``b_fo``, ... by default). The order is the steps, the first segment, the last, then
    the middle ones (the order of the old edge, core and plateau names, so the fits of
    earlier runs are unchanged)."""
    steps = tuple(f"step{i + 1}_um" for i in range(len(initial.steps_um)))
    fo = [f"{n}_fo" for n in initial.segment_names]
    return (*steps, fo[0], fo[-1], *fo[1:-1])


def _aliases(initial: InitialCondition) -> dict[str, str]:
    """The half-profile names: edge_fo (first segment), core_fo (last) and, with three
    segments, step_fo (the middle one); a segment's own name wins over an alias."""
    fo = [f"{n}_fo" for n in initial.segment_names]
    out = {"edge_fo": fo[0], "core_fo": fo[-1]}
    if len(fo) == 3:
        out["step_fo"] = fo[1]
    return {k: v for k, v in out.items() if k not in fo}


def resolve_names(
    initial: InitialCondition, uncertainties: Mapping[str, Uncertainty]
) -> dict[str, Uncertainty]:
    """``uncertainties`` keyed by `parameter_names`, the old edge_fo / core_fo / step_fo
    names read as their segments. Raises for a name ``initial`` does not have."""
    names, aliases = parameter_names(initial), _aliases(initial)
    out: dict[str, Uncertainty] = {}
    unknown = []
    for key, u in uncertainties.items():
        name = key if key in names else aliases.get(key)
        if name is None:
            unknown.append(key)
        elif name in out:
            raise ValueError(f"{key!r} and another name both give {name!r}")
        else:
            out[name] = u
    if unknown:
        raise ValueError(
            f"no parameter {sorted(unknown)} in this initial condition; use {list(names)}"
        )
    return out


def _nominal(initial: InitialCondition, name: str) -> float:
    if name.startswith("step") and name.endswith("_um"):
        return float(initial.steps_um[int(name[4:-3]) - 1])
    return float(initial.segment_fo[initial.segment_names.index(name[:-3])])


def is_valid(candidate: InitialCondition, domain_um: tuple[float, float]) -> bool:
    """Steps inside ``domain_um`` and in order, every Fo in (0, 1]."""
    lo, hi = domain_um
    steps = candidate.steps_um
    return (
        all(lo <= s <= hi for s in steps)
        and all(b > a for a, b in itertools.pairwise(steps))
        and all(0 < f <= 1 for f in candidate.fo_values)
    )


def _with_values(initial: InitialCondition, values: Mapping[str, float]) -> InitialCondition:
    steps, fo = list(initial.steps_um), list(initial.segment_fo)
    for name, v in values.items():
        if name.startswith("step") and name.endswith("_um"):
            steps[int(name[4:-3]) - 1] = v
        else:
            fo[initial.segment_names.index(name[:-3])] = v
    return replace(initial, steps_um=tuple(steps), segment_fo=tuple(fo))


@dataclass(frozen=True)
class CandidateGrid:
    """Candidates on a grid over the uncertain parameters, with their prior."""

    candidates: tuple[InitialCondition, ...]
    names: tuple[str, ...]
    values: np.ndarray  # (candidates, parameters)
    nodes: dict[str, np.ndarray]  # grid values of each parameter
    uncertainties: dict[str, Uncertainty]
    log_prior: np.ndarray  # per candidate, up to a constant
    penalty: np.ndarray  # -2 log prior relative to the nominal values (0 for uniform)
    dropped: int  # nodes left out as invalid (steps out of order or domain, Fo outside 0-1)


def default_points(n_parameters: int) -> int:
    return DEFAULT_POINTS.get(n_parameters, DEFAULT_POINTS_MANY)


def candidate_grid(
    initial: InitialCondition,
    uncertainties: Mapping[str, Uncertainty],
    points: int | None = None,
    *,
    max_candidates: int = MAX_CANDIDATES,
) -> CandidateGrid:
    """A grid of candidates around ``initial`` over the uncertain parameters.

    ``points`` nodes per parameter (default 13 for one parameter, 9 for two, 5 for
    more). Gaussian: nodes over +-2.5 sigma, prior density as weight, penalty
    ((v - v0)/sigma)^2. Uniform: nodes over +- half-width, flat weight, no penalty.
    Nodes that make steps leave the domain or cross, or Fo leave (0, 1], are dropped
    and counted. Raises if the grid would exceed ``max_candidates``.
    """
    uncertainties = resolve_names(initial, uncertainties)
    names = tuple(n for n in parameter_names(initial) if n in uncertainties)
    if not names:
        raise ValueError("no uncertain parameters given")
    points = points or default_points(len(names))
    if points < 2:
        raise ValueError("a scan needs at least 2 points per parameter")
    total = points ** len(names)
    if total > max_candidates:
        raise ValueError(
            f"{len(names)} uncertain parameters x {points} points = {total} candidates, more "
            f"than {max_candidates}; use fewer points or parameters, or raise max_candidates"
        )
    nodes = {}
    for name in names:
        u, v0 = uncertainties[name], _nominal(initial, name)
        half = NORMAL_SPAN * u.spread if u.kind == "normal" else u.spread
        nodes[name] = np.linspace(v0 - half, v0 + half, points)
    candidates, values, log_prior, penalty = [], [], [], []
    dropped = 0
    for combo in itertools.product(*(nodes[n] for n in names)):
        vals = dict(zip(names, (float(v) for v in combo), strict=True))
        cand = _with_values(initial, vals)
        if not is_valid(cand, initial.domain_um):
            dropped += 1
            continue
        pen = 0.0
        for name in names:
            u = uncertainties[name]
            if u.kind == "normal":
                pen += ((vals[name] - _nominal(initial, name)) / u.spread) ** 2
        candidates.append(cand)
        values.append(combo)
        penalty.append(pen)
        log_prior.append(-0.5 * pen)
    if not candidates:
        raise ValueError("every node of the scan is invalid; check the uncertainties")
    return CandidateGrid(
        tuple(candidates),
        names,
        np.array(values, dtype=float),
        nodes,
        {n: uncertainties[n] for n in names},
        np.array(log_prior),
        np.array(penalty),
        dropped,
    )


def _fo_span(candidates: Sequence[InitialCondition]) -> tuple[float, float]:
    """Smallest and largest Fo over every candidate's edge, core and plateau."""
    values = [fo for c in candidates for fo in c.fo_values]
    return min(values), max(values)


def scan_grid(
    grid: Grid,
    candidates: Sequence[InitialCondition],
    conditions: Conditions,
    orientation: Orientation,
    diffusivity: Diffusivity | str = DEFAULT_DIFFUSIVITY,
    stability: float = SCAN_STABILITY,
    profile: Profile | None = None,
    scheme: str = "central",
) -> Grid:
    """``grid`` with dt raised so D_max dt / dx^2 = ``stability``, D_max over every
    candidate's compositions (edge, core, plateau); a larger dt (set to resolve the fit;
    the run takes stable substeps within it) is kept. dx and max time unchanged. One grid
    for every candidate, so their chi-squared values stay comparable. With ``profile``
    the spacing `fit_profile` will actually use with the explicit ``scheme`` is taken
    (`forster.grid.actual_dx_um`)."""
    lo, hi = _fo_span(candidates)
    dx = grid.dx_um if profile is None else actual_dx_um(profile, grid.dx_um, candidates[0], scheme)
    stable = stable_dt(dx, [(diffusivity, (lo, hi))], conditions, orientation, stability)
    return replace(grid, dt_s=max(stable, grid.dt_s))


def step_offsets(initial: InitialCondition, offsets_um: Sequence[float]) -> list[InitialCondition]:
    """Copies of ``initial`` with every step moved by each offset (um; + is away from
    the edge). The domain, compositions and plateau are unchanged."""
    out = []
    for offset in offsets_um:
        steps = tuple(s + offset for s in initial.steps_um)
        if not all(initial.domain_um[0] <= s <= initial.domain_um[1] for s in steps):
            raise ValueError(
                f"offset {offset} um moves a step outside the domain {initial.domain_um}"
            )
        out.append(replace(initial, steps_um=steps))
    return out


@dataclass(frozen=True)
class CandidateRuns:
    """Each candidate's model at the measured points: ``at_points[i]`` is (time steps, n)."""

    candidates: tuple[InitialCondition, ...]
    at_points: list[np.ndarray]
    complete: list[bool]  # whether each run reached grid.max_time_days
    model_x_um: np.ndarray
    x_um: np.ndarray
    fo: np.ndarray
    grid: Grid  # as run: the requested spacing, the time step used

    @property
    def dt_s(self) -> float:
        return self.grid.dt_s


@dataclass(frozen=True)
class ScanResult:
    """Best candidate and time, and the ranges of time from both ways of combining them."""

    runs: CandidateRuns
    chi2: list[np.ndarray]  # per candidate, against time step
    noise: NoiseModel
    threshold: float
    best_candidate: int
    best_row: int
    profile: Interval  # profile-likelihood range, rows of the time axis
    marginal: Interval  # central 68% of the marginal posterior of time
    candidate_probability: np.ndarray
    prior: CandidateGrid | None = None  # when built with `candidate_grid`
    # `forster.uncertainty.near_step_check` at the best candidate and time (NaN when not
    # run, as from `fit_candidates` alone)
    near_step_delta_aic: float = math.nan
    # each scanned value at the best candidate, (value - nominal) / spread (sd or half-width)
    prior_offsets: dict[str, float] = field(default_factory=dict)

    @property
    def best_initial(self) -> InitialCondition:
        return self.runs.candidates[self.best_candidate]

    @property
    def best_values(self) -> dict[str, float]:
        """The best candidate's values of the scanned parameters (empty without a grid)."""
        if self.prior is None:
            return {}
        return dict(zip(self.prior.names, self.prior.values[self.best_candidate], strict=True))

    def _days(self, row: float) -> float:
        return float(row) * self.runs.dt_s / SECONDS_PER_DAY

    @property
    def time_days(self) -> float:
        return self._days(self.best_row)

    @property
    def profile_days(self) -> tuple[float, float]:
        return self._days(self.profile.low), self._days(self.profile.high)

    @property
    def marginal_days(self) -> tuple[float, float]:
        return self._days(self.marginal.low), self._days(self.marginal.high)


def run_candidates(
    profile: Profile,
    conditions: Conditions,
    orientation: Orientation,
    candidates: Sequence[InitialCondition],
    grid: Grid,
    diffusivity: Diffusivity | str = DEFAULT_DIFFUSIVITY,
    scheme: str = DEFAULT_SCHEME,
    *,
    stop_early: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> CandidateRuns:
    """One model run per candidate, kept as the model at the measured points.

    All candidates must share the model domain. With ``stop_early`` each run stops
    once its own fit is well behind it, as in `fit_profile`. ``progress(i, n)`` is
    called after each run.
    """
    if not candidates:
        raise ValueError("no candidate initial conditions")
    domains = {c.domain_um for c in candidates}
    if len(domains) != 1:
        raise ValueError(f"candidates must share the model domain, got {sorted(domains)}")
    diffusivity = fe_mg_diffusivity(diffusivity)
    if len({c.centre_um for c in candidates}) != 1:
        raise ValueError("candidates must share the crystal centre (centre_um)")
    measured_x, measured_fo = measured_points(profile, candidates[0])
    step_x, dx_um, (low, high) = initial_grid(scheme, measured_x, grid, candidates[0])
    model_x = step_x[low : high + 1]
    check_curve_memory(grid.timesteps + 1, len(model_x))
    x, y = points_in_domain(model_x, measured_x, measured_fo)
    W = interpolation_matrix(model_x, x)
    at_points, complete, advances, lasts = [], [], [], []
    explicit = scheme != "implicit"
    for i, initial in enumerate(candidates):
        substeps = (
            _explicit_substeps(
                diffusivity, conditions, orientation, initial, None, None, grid, dx_um
            )
            if explicit
            else 1
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
        if stop_early:
            curves = _run_past_best_fit(start, advance, grid.timesteps, W, y)
        else:
            curves = advance(start, grid.timesteps)
        at_points.append(curves @ W.T)
        complete.append(len(curves) - 1 >= grid.timesteps)
        advances.append(advance)
        lasts.append(curves[-1])
        if progress is not None:
            progress(i + 1, len(candidates))
    runs = CandidateRuns(tuple(candidates), at_points, complete, model_x, x, y, grid)
    return _extend_tails(runs, advances, lasts, W, grid.timesteps)


def _whitened_chi2(at_points: np.ndarray, y: np.ndarray, chol: np.ndarray) -> np.ndarray:
    z = sla.solve_triangular(chol, (at_points - y).T, lower=True)
    return np.sum(z**2, axis=0)


def _extend_tails(runs, advances, lasts, W, total_steps):
    """Extend runs whose end is still within TAIL_DELTA of the best chi-squared."""
    while True:
        rss = [np.sum((m - runs.fo) ** 2, axis=1) for m in runs.at_points]
        s2 = min(float(r.min()) for r in rss) / len(runs.fo)
        overall = min(float(r.min()) for r in rss) / s2
        grew = False
        for i, r in enumerate(rss):
            if runs.complete[i] or r[-1] / s2 - overall > TAIL_DELTA:
                continue
            n_have = len(r) - 1
            extra = min(n_have, total_steps - n_have)
            rows = advances[i](lasts[i], extra, first_row=n_have)[1:]
            runs.at_points[i] = np.concatenate([runs.at_points[i], rows @ W.T])
            lasts[i] = rows[-1]
            runs.complete[i] = n_have + extra >= total_steps
            grew = True
        if not grew:
            return runs


def _prior_offsets(prior: CandidateGrid, best: int) -> dict[str, float]:
    """The best candidate's values in prior units from the nominal (the grid's centre),
    with `warn_prior_conflict`."""
    offsets = {}
    for j, name in enumerate(prior.names):
        u, grid_nodes = prior.uncertainties[name], prior.nodes[name]
        offsets[name] = (float(prior.values[best, j]) - float(grid_nodes.mean())) / u.spread
    warn_prior_conflict(offsets, {n: u.kind for n, u in prior.uncertainties.items()})
    return offsets


def fit_candidates(
    runs: CandidateRuns,
    *,
    n_free: int = 1,
    prior: CandidateGrid | None = None,
    level: float = ONE_SIGMA,
) -> ScanResult:
    """Combine candidate runs: noise model at the overall best fit, then both ranges.

    ``n_free`` is the number of initial-condition parameters the candidates vary
    (1 for a scan of step positions), which sets the small-sample threshold; with a
    ``prior`` (from `candidate_grid`, in the same order as the runs) it is the number
    of scanned parameters, the profile range adds the prior penalty to chi-squared,
    and the marginal weights each candidate by its prior.
    """
    if prior is not None:
        if len(prior.candidates) != len(runs.at_points):
            raise ValueError("the prior has a different number of candidates than the runs")
        n_free = len(prior.names)
    x, y = runs.x_um, runs.fo
    rss = [np.sum((m - y) ** 2, axis=1) for m in runs.at_points]
    c0 = int(np.argmin([r.min() for r in rss]))
    t0 = int(np.argmin(rss[c0]))
    noise = fit_noise_model(x, y - runs.at_points[c0][t0])
    chol = np.linalg.cholesky(noise.covariance(x))
    chi2 = [_whitened_chi2(m, y, chol) for m in runs.at_points]
    n_steps = max(len(c) for c in chi2)
    table = np.full((len(chi2), n_steps), np.inf)
    for i, c in enumerate(chi2):
        table[i, : len(c)] = c
    penalty = np.zeros(len(chi2)) if prior is None else prior.penalty
    log_prior = np.zeros(len(chi2)) if prior is None else prior.log_prior
    penalised = table + penalty[:, None]
    best_candidate, best_row = np.unravel_index(int(np.argmin(penalised)), table.shape)
    if prior is None:
        if len(chi2) >= 3 and best_candidate in (0, len(chi2) - 1):
            notes.warn(
                f"the best candidate is the {'first' if best_candidate == 0 else 'last'} of "
                "the scan; extend the scan past it, or the ranges may be cut short",
                stacklevel=2,
            )
    else:
        for j, name in enumerate(prior.names):
            v, grid_nodes = prior.values[best_candidate, j], prior.nodes[name]
            if v in (grid_nodes[0], grid_nodes[-1]):
                u = prior.uncertainties[name]
                limit = f"{NORMAL_SPAN:g} sigma" if u.kind == "normal" else "the range given"
                notes.warn(
                    f"the best {name} ({v:.4g}) is at {limit} from its nominal value; the "
                    "data favour values beyond the scan, so the ranges may be cut short",
                    stacklevel=2,
                )
    threshold = interval_threshold(len(x), 1 + n_free, level)
    envelope = penalised.min(axis=0)
    profile = delta_chi2_interval(envelope, threshold, end_is_limit=all(runs.complete))
    # -inf where a candidate did not run
    log_post = -0.5 * len(x) * np.log(table) + log_prior[:, None]
    weights = np.exp(log_post - log_post.max())
    low, high = central_range(weights.sum(axis=0), level)
    marginal = Interval(int(best_row), low, high, int(np.floor(low)), int(np.ceil(high)))
    probability = weights.sum(axis=1) / weights.sum()
    offsets = _prior_offsets(prior, int(best_candidate)) if prior is not None else {}
    return ScanResult(
        runs,
        chi2,
        noise,
        threshold,
        int(best_candidate),
        int(best_row),
        profile,
        marginal,
        probability,
        prior,
        prior_offsets=offsets,
    )


def scan_initial_conditions(
    profile: Profile,
    conditions: Conditions,
    orientation: Orientation,
    candidates: Sequence[InitialCondition] | CandidateGrid,
    grid: Grid,
    diffusivity: Diffusivity | str = DEFAULT_DIFFUSIVITY,
    scheme: str = DEFAULT_SCHEME,
    *,
    n_free: int = 1,
    stability: float | None = SCAN_STABILITY,
    stop_early: bool = True,
    progress: Callable[[int, int], None] | None = None,
) -> ScanResult:
    """Fit ``profile`` with every candidate initial condition and combine the fits.

    ``candidates`` is a `CandidateGrid` (from `candidate_grid`: uncertainties on each
    step and concentration) or a plain list (flat prior; ``n_free`` parameters).
    Example, the step Gaussian with sigma 2 um and the edge uniform within 0.003::

        candidates = candidate_grid(initial, {"step1_um": Uncertainty(2.0),
                                              "edge_fo": Uncertainty(0.003, "uniform")})
        scan = scan_initial_conditions(profile, conditions, orientation, candidates, grid)
        scan.time_days, scan.profile_days, scan.marginal_days, scan.best_values

    The runs use `scan_grid` (dt raised to ``stability``; None keeps ``grid``; the
    implicit scheme always keeps ``grid``). A plain
    list should be evenly spaced. To plot the best fit, fit ``scan.best_initial`` with
    `fit_profile`.
    """
    if isinstance(candidates, CandidateGrid):
        prior, initials = candidates, candidates.candidates
    else:
        prior, initials = None, tuple(candidates)
    # The implicit scheme is stable at any dt; an explicit run takes stable substeps
    # within a dt above its limit (`forster.model._explicit_substeps`).
    if scheme != "implicit" and stability is not None:
        grid = scan_grid(
            grid, initials, conditions, orientation, diffusivity, stability, profile, scheme
        )
    runs = run_candidates(
        profile,
        conditions,
        orientation,
        initials,
        grid,
        diffusivity,
        scheme,
        stop_early=stop_early,
        progress=progress,
    )
    scan = fit_candidates(runs, n_free=n_free, prior=prior)
    near = near_step_at_best_initial(
        runs.x_um,
        runs.fo,
        runs.at_points[scan.best_candidate][scan.best_row],
        scan.best_initial,
        scan.time_days,
        fe_mg_diffusivity(diffusivity),
        conditions,
        orientation,
    )
    return replace(scan, near_step_delta_aic=near.delta_aic)
