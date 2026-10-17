"""Joint fit of one crystal's profiles: `fit_crystal`.

Profiles measured along different directions of one crystal share one history: the
time, the conditions (identical T, P and fO2 are required) and the compositions, each
segment's Fo matched by segment label. Each profile keeps its own step positions, its
own orientation (so its own D) and its own noise model
(docs/plans/2026-10-16-joint-fits.md, Part 1 as changed by "Henry's answers").

**Likelihood.** Every profile runs its own model on one shared time step, so the time
axis is common, and the joint log likelihood at each time step is the sum over profiles
of -n_i/2 log chi2_i(t): each profile's generalised chi-squared under its own noise
model, its noise scale integrated out (as `forster.sampling.ICLikelihood`). The noise
models are fitted per profile at the white-noise best fit of the joint model and
refitted at each new best fit until stable, as `fit_initial_condition` does.

**Parameters** (the union over the profiles, each with a prior):
- per profile: its uncertain steps (``<run>:step1_um``, ...); segments it fits on its
  own (``own_segments``, e.g. ``<run>:c_fo`` for its core); an Fo offset
  ``<run>:fo_offset`` (measurement drift between sessions: the profile's measured Fo
  is the model's plus the offset; Gaussian prior, default sd 0.001 = 0.1 mol%) and an
  edge position offset ``<run>:edge1_offset_um`` (where the true edge 1 lies in the
  profile's measured distances: the model is read at measured distance minus the
  offset; Gaussian prior, default sd 2 um). An offset sd of 0 leaves that offset out.
- shared: each other segment label's Fo (``<label>_fo``), with the prior the runs give
  it; runs that disagree on a shared label's nominal Fo or prior are an error.

**Fit.** The optimiser, Laplace importance sampling and nested-sampling fallback are
`forster.sampling`'s, run on `JointLikelihood` (one model run per profile per
evaluation), with two options single-profile fits leave off: the Laplace covariance
refined along its own directions (``whiten``: each profile's Fo offset and the shared
segment Fo are degenerate but for their priors, which the prior-spread steps of the
first pass mis-measure), and a fallback threshold scaled by the number of parameters
(``scaled``; see `forster.sampling.sample_posterior`). The result's ranges are the
central 68.27% and 95.45% of the marginal posterior of time (over every free parameter
and each profile's noise scale), as `ICPosterior`'s; with one profile and the offsets
off, the best values are that profile's `fit_initial_condition`'s exactly and the
ranges agree within sampling error.

A profile's edge offset is told apart from its step positions only by the edge itself:
with free steps and the first step several diffusion lengths from edge 1, moving the
steps and the edge together hardly changes the fit, so the offset stays near its prior
and the posterior is less Gaussian (the importance sampler's ESS halves; with Fo
offsets off as well, the fit may fall back to nested sampling, minutes).

Before the joint fit each profile is fitted alone (`forster.io.fit_run`); profiles
whose 2-sigma ranges do not overlap are reported in a warning.

Not built yet: growth stages and a held edge1_fo (a run with either is an error);
tables, the command line and the app.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace

import numpy as np
import scipy.linalg as sla
from scipy import special

from forster import notes
from forster.io import Run, _profile, fit_run
from forster.model import (
    SECONDS_PER_DAY,
    FitResult,
    Grid,
    InitialCondition,
    Profile,
    check_curve_memory,
    fe_mg_diffusivity,
    initial_grid,
    measured_points,
)
from forster.sampling import (
    DEFAULT_NLIVE,
    DEFAULT_SAMPLES,
    ESS_FALLBACK,
    METHODS,
    NOISE_REFITS,
    TAIL_FRACTION,
    TAIL_MASS,
    Cancelled,
    PriorLikelihood,
    _noise_changed,
    _optimise,
    _Samples,
    _summarise,
    at_prior_edge,
    model_at_points,
    sample_posterior,
)
from forster.scan import (
    Uncertainty,
    _nominal,
    _with_values,
    is_valid,
    parameter_names,
    warn_prior_conflict,
)
from forster.uncertainty import (
    ONE_SIGMA,
    TWO_SIGMA,
    NoiseModel,
    central_range,
    delta_chi2_interval,
    fit_noise_model,
    interpolation_matrix,
    interval_threshold,
    points_in_domain,
    white_rss,
)

# Inter-session accuracy of Fo (0.1 mol%; to confirm, Henry 2026-10-16) and how far the
# first point may lie from the true edge (a few um).
DEFAULT_FO_OFFSET_SD = 0.001
# The edge position offset is off by default: with free steps it is barely identifiable
# (moving a step and the edge together hardly changes the fit) and it pushes the fit to
# nested sampling. Turn it on per crystal (e.g. this value) when the steps are well pinned.
DEFAULT_EDGE_OFFSET_SD_UM = 2.0
FO_TOLERANCE = 1e-9  # shared segments' nominal Fo must agree to this


@dataclass
class _Part:
    """One profile of the crystal: its model grid, fitted points and parameter slots."""

    run: Run
    grid: Grid
    model_x: np.ndarray
    dx_um: float
    x: np.ndarray
    y: np.ndarray
    W: np.ndarray
    slots: tuple[tuple[str, int], ...]  # (its own parameter name, index in theta)
    fo_offset: int | None  # index of its Fo offset in theta (None: off)
    edge_offset: int | None  # and of its edge offset
    chol: np.ndarray | None = None

    def candidate(self, theta: np.ndarray) -> InitialCondition | None:
        """Its starting profile with values ``theta``, or None if invalid."""
        initial = self.run.initial
        if not self.slots:
            return initial
        try:
            c = _with_values(initial, {n: float(theta[i]) for n, i in self.slots})
        except ValueError:
            return None
        return c if is_valid(c, initial.domain_um) else None

    def matrix(self, theta: np.ndarray) -> np.ndarray:
        """Model grid -> its points, shifted by its edge offset (clipped to the grid)."""
        if self.edge_offset is None:
            return self.W
        x = np.clip(self.x - float(theta[self.edge_offset]), self.model_x[0], self.model_x[-1])
        return interpolation_matrix(self.model_x, x)

    def predict(self, initial: InitialCondition, theta: np.ndarray) -> np.ndarray:
        """The model at its points, with its offsets: (time steps + 1, points)."""
        run = self.run
        m = model_at_points(
            run.scheme,
            fe_mg_diffusivity(run.diffusivity),
            run.conditions,
            run.orientation,
            initial,
            self.grid,
            self.model_x,
            self.dx_um,
            self.matrix(theta),
        )
        return m if self.fo_offset is None else m + theta[self.fo_offset]

    def chi2(self, m: np.ndarray) -> np.ndarray:
        """Generalised chi-squared at every time step under its noise model."""
        if self.chol is None:
            raise RuntimeError("fit the noise models first (fit_noise)")
        z = sla.solve_triangular(self.chol, (m - self.y).T, lower=True)
        return np.sum(z**2, axis=0)

    def log_like(self, chi2: np.ndarray) -> np.ndarray:
        """-n/2 log chi2: its noise scale integrated out."""
        return -0.5 * len(self.y) * np.log(chi2)


class JointLikelihood(PriorLikelihood):
    """Log prior and per-time joint log likelihood of a crystal's parameters (see the
    module docstring), with the interface `forster.sampling`'s optimiser and samplers
    use."""

    def __init__(
        self,
        parts: Sequence[_Part],
        names: Sequence[str],
        nominal: np.ndarray,
        spread: np.ndarray,
        normal: np.ndarray,
    ) -> None:
        self.parts = tuple(parts)
        self.names = tuple(names)
        self.grid = self.parts[0].grid  # the common time step and length
        self.rows = self.grid.timesteps + 1
        self._set_prior(nominal, spread, normal)
        self.runs = 0
        self.cancel: Callable[[], bool] | None = None
        self.noise: tuple[NoiseModel, ...] = ()

    def candidates(self, theta: np.ndarray) -> list[InitialCondition] | None:
        """Each profile's starting profile with values ``theta``, or None if any is invalid."""
        out = []
        for part in self.parts:
            c = part.candidate(theta)
            if c is None:
                return None
            out.append(c)
        return out

    def predictions(self, theta: np.ndarray) -> list[np.ndarray] | None:
        """Each profile's model at its points (one model run each), or None."""
        initials = self.candidates(theta)
        if initials is None:
            return None
        out = []
        for part, initial in zip(self.parts, initials, strict=True):
            if self.cancel is not None and self.cancel():
                raise Cancelled("the joint fit was cancelled")
            self.runs += 1
            out.append(part.predict(initial, theta))
        return out

    def fit_noise(self, theta: np.ndarray) -> tuple[NoiseModel, ...]:
        """Each profile's noise model at the white-noise best time of the joint model with
        values ``theta`` (the time that maximises the sum of -n_i/2 log rss_i); used from
        then on."""
        m = self.predictions(theta)
        if m is None:
            raise ValueError("the joint fit's starting profiles are invalid")
        score = sum(
            len(p.y) * np.log(white_rss(mi, p.y)) for p, mi in zip(self.parts, m, strict=True)
        )
        best = int(np.argmin(score))
        return self.use_noise(
            [fit_noise_model(p.x, p.y - mi[best]) for p, mi in zip(self.parts, m, strict=True)]
        )

    def use_noise(self, noise: Sequence[NoiseModel]) -> tuple[NoiseModel, ...]:
        """Use one noise model per profile from now on."""
        for part, n in zip(self.parts, noise, strict=True):
            part.chol = np.linalg.cholesky(n.covariance(part.x))
        self.noise = tuple(noise)
        return self.noise

    def profile_chi2(self, theta: np.ndarray) -> list[np.ndarray] | None:
        """Each profile's generalised chi-squared at every time step, or None."""
        m = self.predictions(theta)
        if m is None:
            return None
        return [p.chi2(mi) for p, mi in zip(self.parts, m, strict=True)]

    def log_like_rows(self, theta: np.ndarray) -> np.ndarray | None:
        """The joint -sum n_i/2 log chi2_i at every time step, or None if invalid."""
        chi2 = self.profile_chi2(theta)
        if chi2 is None:
            return None
        total = np.zeros(self.rows)
        for p, c in zip(self.parts, chi2, strict=True):
            total = total + p.log_like(c)
        return total


@dataclass(frozen=True)
class CrystalFit:
    """Joint posterior of time and of the shared and per-profile parameters of one
    crystal's profiles (`fit_crystal`).

    ``best_days`` is the joint best time at the best-fit parameters; ``low_days`` to
    ``high_days`` and ``low2_days`` to ``high2_days`` hold the central 68.27% and 95.45%
    of the marginal posterior of time (``density`` per day on ``days``), as
    `ICPosterior`'s ranges. ``threshold_range_days`` and ``threshold_range2_days`` hold
    the ranges at the best-fit starting profiles by the rule of `fit_profile` and
    `forster.elements.fit_joint`: where the sum of the profiles' generalised chi-squared
    rises by the small-sample threshold for all the crystal's points
    (`forster.uncertainty.interval_threshold`). They leave out the starting profiles'
    uncertainty, like the individual fits' ranges they compare with; for one profile with
    a fixed starting profile they are `fit_profile`'s range (the central range of that
    skewed posterior sits about 4% later on the template example). Parameters are
    named as in the module docstring; ``best_values`` is the optimum, ``mean`` and ``sd``
    the posterior's. ``segment_fo`` / ``segment_fo_sd``: each shared segment label's
    best Fo and posterior sd (0 when fixed); ``profile_fo``: every segment of each
    profile, shared or its own. ``steps_um``, ``fo_offset``, ``edge1_offset_um``: each
    profile's best values (0 for an offset left out). ``profile_days``: each profile's
    own best time at the joint best starting profile; ``individual``: each profile's fit
    alone (`fit_run`), and ``agree`` whether their 2-sigma ranges overlap pairwise.
    ``method`` is "laplace" or "nested", or "fixed" when nothing is free.
    """

    crystal: str | None
    runs: tuple[str, ...]
    method: str
    names: tuple[str, ...]
    best_values: dict[str, float]
    best_initials: dict[str, InitialCondition]
    best_days: float
    median_days: float
    low_days: float
    high_days: float
    low2_days: float
    high2_days: float
    threshold_range_days: tuple[float, float]
    threshold_range2_days: tuple[float, float]
    days: np.ndarray
    density: np.ndarray
    mean: dict[str, float]
    sd: dict[str, float]
    segment_fo: dict[str, float]
    segment_fo_sd: dict[str, float]
    profile_fo: dict[str, dict[str, float]]
    steps_um: dict[str, tuple[float, ...]]
    fo_offset: dict[str, float]
    edge1_offset_um: dict[str, float]
    profile_days: dict[str, float]
    individual: dict[str, FitResult]
    agree: bool
    noise: dict[str, NoiseModel]
    n_points: dict[str, int]
    ic_share: float
    sensitivity: dict[str, float]
    prior_offsets: dict[str, float]
    ess: float
    n_samples: int
    log_evidence: float
    model_runs: int
    seed: int
    grids: dict[str, Grid]

    @property
    def time_days(self) -> float:
        return self.best_days

    @property
    def range_days(self) -> tuple[float, float]:
        return self.low_days, self.high_days

    @property
    def range2_days(self) -> tuple[float, float]:
        return self.low2_days, self.high2_days


def _per_run(
    value: float | Mapping[str, float], runs: Sequence[Run], what: str
) -> dict[str, float]:
    """One offset prior sd per run, from one value or a mapping (missing runs: default)."""
    default = DEFAULT_FO_OFFSET_SD if what == "fo_offset_sd" else 0.0
    names = [r.name for r in runs]
    if isinstance(value, Mapping):
        unknown = sorted(set(value) - set(names))
        if unknown:
            raise ValueError(f"{what}: no run {unknown} in this crystal (runs: {names})")
        out = {n: float(value.get(n, default)) for n in names}
    else:
        out = dict.fromkeys(names, float(value))
    bad = {n: v for n, v in out.items() if not (math.isfinite(v) and v >= 0)}
    if bad:
        raise ValueError(f"{what} must be 0 (off) or positive, got {bad}")
    return out


def _own_segments(
    runs: Sequence[Run], own: Mapping[str, Sequence[str]] | None
) -> dict[str, tuple[str, ...]]:
    """Each run's own segment labels, checked against its segments."""
    own = own or {}
    by_name = {r.name: r for r in runs}
    unknown = sorted(set(own) - set(by_name))
    if unknown:
        raise ValueError(f"own_segments: no run {unknown} in this crystal (runs: {list(by_name)})")
    out = {}
    for name, labels in own.items():
        labels = (labels,) if isinstance(labels, str) else tuple(labels)
        missing = [x for x in labels if x not in by_name[name].initial.segment_names]
        if missing:
            raise ValueError(
                f"own_segments: run {name!r} has no segment {missing} "
                f"(its segments: {list(by_name[name].initial.segment_names)})"
            )
        out[name] = labels
    return out


def _same_conditions(a: Run, b: Run) -> bool:
    pairs = (
        (a.conditions.T_C, b.conditions.T_C),
        (a.conditions.P_Pa, b.conditions.P_Pa),
        (a.conditions.fO2_Pa, b.conditions.fO2_Pa),
    )
    return all(math.isclose(x, y, rel_tol=1e-9, abs_tol=0.0) for x, y in pairs)


def _conditions_text(run: Run) -> str:
    c = run.conditions
    return f"{run.name} ({c.T_C:g} C, {c.P_Pa:g} Pa, fO2 {c.fO2_Pa:.4g} Pa)"


def _check_runs(runs: Sequence[Run]) -> str | None:
    """Raise for runs that cannot be fitted jointly; return their crystal name, if given."""
    if not runs:
        raise ValueError("a joint fit needs at least one run")
    names = [r.name for r in runs]
    if len(set(names)) != len(names):
        raise ValueError(f"each run can appear once in a joint fit, got {names}")
    for r in runs:
        if r.stages:
            raise ValueError(
                f"run {r.name!r} has growth stages: the joint fit does not model growth "
                "stages yet (docs/plans/2026-10-16-joint-fits.md)"
            )
        if r.rim_fo is not None:
            raise ValueError(
                f"run {r.name!r} holds edge1_fo: the joint fit, like the starting-profile fit, "
                "holds edge 1 at the first segment's Fo"
            )
    first = runs[0]
    differ = [r for r in runs[1:] if not _same_conditions(r, first)]
    if differ:
        raise ValueError(
            "the profiles of one crystal need identical T, P and fO2: "
            + ", ".join(_conditions_text(r) for r in (first, *differ))
        )
    laws = {r.diffusivity.name for r in runs}
    if len(laws) > 1:
        listed = ", ".join(f"{r.name} ({r.diffusivity.name})" for r in runs)
        raise ValueError(f"the profiles of one crystal need one diffusivity, got {listed}")
    crystals = {str(r.metadata["crystal"]) for r in runs if "crystal" in r.metadata}
    if len(crystals) > 1:
        raise ValueError(f"these runs belong to different crystals: {sorted(crystals)}")
    return crystals.pop() if crystals else None


def _segment(run: Run, label: str) -> tuple[float, Uncertainty | None, str]:
    """A run's nominal Fo for a segment label, its prior (None: fixed) and its role."""
    names = run.initial.segment_names
    i = names.index(label)
    role = "edge" if i == 0 else "core" if i == len(names) - 1 else "middle"
    return run.initial.segment_fo[i], run.uncertainties.get(f"{label}_fo"), role


def _prior_text(u: Uncertainty | None) -> str:
    if u is None:
        return "fixed"
    return f"{'sd' if u.kind == 'normal' else 'half-width'} {u.spread:g}"


def _check_shared(runs: Sequence[Run], own: Mapping[str, tuple[str, ...]]) -> None:
    """Runs sharing a segment label must agree on its nominal Fo and prior (an error) and
    should use it for the same zone (a warning)."""
    users: dict[str, list[Run]] = {}
    for r in runs:
        for label in r.initial.segment_names:
            if label not in own.get(r.name, ()):
                users.setdefault(label, []).append(r)
    for label, rs in users.items():
        seen = [_segment(r, label) for r in rs]
        fo0, u0, _ = seen[0]
        if any(abs(fo - fo0) > FO_TOLERANCE or u != u0 for fo, u, _ in seen[1:]):
            listed = "; ".join(
                f"{r.name}: Fo {fo:.6g}, {_prior_text(u)}"
                for r, (fo, u, _) in zip(rs, seen, strict=True)
            )
            raise ValueError(
                f"segment {label!r} is shared by runs that disagree on its nominal Fo or prior "
                f"({listed}); give them the same, or let a profile fit its own with own_segments"
            )
        roles = {role for _, _, role in seen}
        if len(roles) > 1:
            listed = ", ".join(f"{r.name} ({s[2]})" for r, s in zip(rs, seen, strict=True))
            notes.warn(
                f"segment {label!r} is shared by profiles that use it for different zones: "
                f"{listed}. Shared labels are matched by name; rename the segments so each "
                "label names one zone",
                stacklevel=3,
            )


@dataclass
class _Layout:
    names: list[str]
    nominal: list[float]
    spread: list[float]
    normal: list[bool]
    index: dict[str, int]

    def add(self, name: str, value: float, u: Uncertainty) -> int:
        if name not in self.index:
            self.index[name] = len(self.names)
            self.names.append(name)
            self.nominal.append(value)
            self.spread.append(u.spread)
            self.normal.append(u.kind == "normal")
        return self.index[name]


def _global_name(run: Run, name: str, own: Mapping[str, tuple[str, ...]]) -> str:
    """A run's parameter name in the joint fit: its steps and own segments prefixed."""
    if name.startswith("step") or name[:-3] in own.get(run.name, ()):
        return f"{run.name}:{name}"
    return name


def _grids(
    runs: Sequence[Run],
    profiles: Mapping[str, Profile],
    fits: Mapping[str, FitResult],
    grid: Grid | None,
) -> dict[str, Grid]:
    """Each profile's grid on one time step: its scan grid (as `forster.io.scan_run`, sized
    by its individual fit's time), with the smallest time step and the longest run."""
    if grid is not None:
        return {r.name: grid for r in runs}
    chosen = {}
    for run in runs:
        request = run.scan_grid
        if not request.is_complete():
            max_time = run.grid.max_time_days if isinstance(run.grid, Grid) else None
            request = replace(request, max_time_days=request.max_time_days or max_time)
        chosen[run.name] = request.resolve(
            _profile(run, profiles),
            run.conditions,
            run.orientation,
            run.initial,
            run.diffusivity,
            purpose="scan",
            time_days=fits[run.name].time_days,
            scheme=run.scheme,
        ).grid
    dt = min(g.dt_s for g in chosen.values())
    max_time = max(g.max_time_days for g in chosen.values())
    return {n: Grid(g.dx_um, dt, max_time) for n, g in chosen.items()}


def _likelihood(
    runs: Sequence[Run],
    profiles: Mapping[str, Profile],
    grids: Mapping[str, Grid],
    own: Mapping[str, tuple[str, ...]],
    fo_sd: Mapping[str, float],
    edge_sd: Mapping[str, float],
) -> JointLikelihood:
    """The joint likelihood: parameters in each run's order (its steps, then its segments
    in `parameter_names` order, each shared label at its first use), then the offsets."""
    layout = _Layout([], [], [], [], {})
    slots = {}
    for run in runs:
        local = []
        for n in parameter_names(run.initial):
            u = run.uncertainties.get(n)
            if u is not None:
                i = layout.add(_global_name(run, n, own), _nominal(run.initial, n), u)
                local.append((n, i))
        slots[run.name] = tuple(local)
    offsets = {}
    for run in runs:
        fo = fo_sd[run.name]
        edge = edge_sd[run.name]
        offsets[run.name] = (
            layout.add(f"{run.name}:fo_offset", 0.0, Uncertainty(fo)) if fo > 0 else None,
            layout.add(f"{run.name}:edge1_offset_um", 0.0, Uncertainty(edge)) if edge > 0 else None,
        )
    parts = []
    for run in runs:
        grid = grids[run.name]
        measured_x, measured_fo = measured_points(_profile(run, profiles), run.initial)
        step_x, dx_um, (low, high) = initial_grid(run.scheme, measured_x, grid, run.initial)
        model_x = step_x[low : high + 1]
        check_curve_memory(grid.timesteps + 1, len(model_x))
        x, y = points_in_domain(model_x, measured_x, measured_fo)
        W = interpolation_matrix(model_x, x)
        parts.append(_Part(run, grid, model_x, dx_um, x, y, W, slots[run.name], *offsets[run.name]))
    return JointLikelihood(
        parts,
        layout.names,
        np.array(layout.nominal, dtype=float),
        np.array(layout.spread, dtype=float),
        np.array(layout.normal, dtype=bool),
    )


def _individual(
    runs: Sequence[Run], profiles: Mapping[str, Profile], fits: Mapping[str, FitResult] | None
) -> dict[str, FitResult]:
    """Each profile's fit alone: from ``fits`` where given, else `fit_run`."""
    fits = fits or {}
    return {r.name: fits[r.name] if r.name in fits else fit_run(r, profiles) for r in runs}


def _range2(fit: FitResult) -> tuple[float, float]:
    try:
        return fit.time_range_days(sigmas=2)
    except ValueError:  # a legacy fit has no 2-sigma range
        return fit.time_range_days()


def _check_agreement(fits: Mapping[str, FitResult], crystal: str | None) -> bool:
    """Whether every pair of individual fits has overlapping 2-sigma ranges; warns if not."""
    ranges = {n: _range2(f) for n, f in fits.items()}
    apart = [
        (a, b)
        for a, b in itertools.combinations(ranges, 2)
        if ranges[a][1] < ranges[b][0] or ranges[b][1] < ranges[a][0]
    ]
    if apart:
        listed = ", ".join(
            f"{n} {fits[n].time_days:.3g} d ({lo:.3g}-{hi:.3g})" for n, (lo, hi) in ranges.items()
        )
        pairs = "; ".join(f"{a} and {b}" for a, b in apart)
        notes.warn(
            f"the profiles of crystal {crystal or '(unnamed)'} do not agree on one time: the "
            f"2-sigma ranges of {pairs} do not overlap (individual fits: {listed}). Check the "
            "orientations, whether a profile is off the crystal's centre, or a 3D effect near "
            "a corner, before trusting the joint fit",
            stacklevel=3,
        )
    return not apart


def _best_fit(like: JointLikelihood) -> tuple[np.ndarray, tuple[NoiseModel, ...]]:
    """`forster.sampling._best_fit` for a crystal: the first search scores each profile
    with white noise, then the noise models are fitted at that data-driven best fit and
    refitted at each new one until none changes."""
    d = len(like.names)
    white = like.fit_noise(like.nominal)
    if d == 0:
        return np.zeros(0), white
    like.use_noise([NoiseModel(w.white_sigma, 0.0, 1.0) for w in white])
    z, _ = _optimise(like, np.zeros(d), 200 * d)
    noise = white
    for _ in range(NOISE_REFITS):
        theta = like.nominal + z * like.spread
        if like.candidates(theta) is None:  # the optimiser keeps to valid values
            raise ValueError("the best-fit starting profiles are invalid")
        refit = like.fit_noise(theta)
        z, _ = _optimise(like, z, 100 * d)
        stable = not any(_noise_changed(a, b) for a, b in zip(noise, refit, strict=True))
        noise = refit
        if stable:
            break
    return z, noise


def _fixed_samples(like: JointLikelihood) -> tuple[_Samples, float]:
    """Nothing free: the posterior of time is the joint likelihood itself."""
    rows = like.log_like_rows(np.zeros(0))
    if rows is None:
        raise ValueError("the joint fit's starting profiles are invalid")
    log_z = float(special.logsumexp(rows) - math.log(like.rows))
    return _Samples(np.zeros((1, 0)), np.zeros(1), rows[None, :]), log_z


def _warn_best(like: JointLikelihood, z: np.ndarray) -> dict[str, float]:
    """The best values' offsets in prior units; warns at a prior's edge or in conflict."""
    at_edge = at_prior_edge(like, z)
    if at_edge:
        notes.warn(
            f"the best {', '.join(at_edge)} is at the edge of its prior; the data favour "
            "values beyond it, so the range may be cut short",
            stacklevel=3,
        )
    offsets = {n: float(zi) for n, zi in zip(like.names, z, strict=True)}
    kinds = {n: "normal" if k else "uniform" for n, k in zip(like.names, like.normal, strict=True)}
    warn_prior_conflict(offsets, kinds)
    return offsets


def _profile_values(
    like: JointLikelihood,
    theta: np.ndarray,
    initials: Sequence[InitialCondition],
    sd: Mapping[str, float],
) -> dict[str, dict]:
    """Per-profile and shared outputs at the best fit."""
    out: dict[str, dict] = {
        "segment_fo": {},
        "segment_fo_sd": {},
        "profile_fo": {},
        "steps_um": {},
        "fo_offset": {},
        "edge1_offset_um": {},
    }
    for part, initial in zip(like.parts, initials, strict=True):
        name = part.run.name
        out["steps_um"][name] = tuple(initial.steps_um)
        out["profile_fo"][name] = dict(zip(initial.segment_names, initial.segment_fo, strict=True))
        out["fo_offset"][name] = 0.0 if part.fo_offset is None else float(theta[part.fo_offset])
        out["edge1_offset_um"][name] = (
            0.0 if part.edge_offset is None else float(theta[part.edge_offset])
        )
        for label, fo in out["profile_fo"][name].items():
            if f"{name}:{label}_fo" not in like.names and label not in out["segment_fo"]:
                out["segment_fo"][label] = float(fo)
                out["segment_fo_sd"][label] = sd.get(f"{label}_fo", 0.0)
    return out


def _time_ranges(density: np.ndarray, chi2: np.ndarray, n_points: int, dt_days: float) -> dict:
    """The central 95.45% of the posterior of time, and the 1- and 2-sigma ranges where
    the summed chi-squared ``chi2`` rises by the small-sample threshold for ``n_points``."""
    low2, high2 = central_range(density, TWO_SIGMA)
    spans = []
    with notes.quiet():  # the tail warning covers a range that reaches the run's end
        for level in (ONE_SIGMA, TWO_SIGMA):
            span = delta_chi2_interval(chi2, interval_threshold(n_points, level=level))
            spans.append((span.low * dt_days, span.high * dt_days))
    return {
        "low2_days": low2 * dt_days,
        "high2_days": high2 * dt_days,
        "threshold_range_days": spans[0],
        "threshold_range2_days": spans[1],
    }


def _with_reference(
    fo_sd: dict[str, float],
    runs: Sequence[Run],
    profiles: Mapping[str, Profile],
    reference: str | None,
) -> dict[str, float]:
    """``fo_sd`` with the reference profile's Fo offset off: "auto" picks the run whose
    profile has the most points, a run name that run, None none."""
    if reference is None or len(runs) < 2 or not any(fo_sd.values()):
        return fo_sd
    if reference == "auto":
        reference = max(runs, key=lambda r: len(profiles[r.profile_name].distance_um)).name
    if reference not in fo_sd:
        raise ValueError(f"fo_reference: no run {reference!r} in this crystal ({list(fo_sd)})")
    return {**fo_sd, reference: 0.0}


def fit_crystal(
    runs: Sequence[Run],
    profiles: Mapping[str, Profile],
    *,
    own_segments: Mapping[str, Sequence[str]] | None = None,
    fo_offset_sd: float | Mapping[str, float] = DEFAULT_FO_OFFSET_SD,
    edge_offset_sd_um: float | Mapping[str, float] = 0.0,
    fo_reference: str | None = "auto",
    fits: Mapping[str, FitResult] | None = None,
    grid: Grid | None = None,
    method: str = "auto",
    n_samples: int = DEFAULT_SAMPLES,
    nlive: int = DEFAULT_NLIVE,
    seed: int = 0,
    ess_fallback: float = ESS_FALLBACK,
    progress: Callable[[str], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> CrystalFit:
    """Fit one time to the profiles of one crystal (see the module docstring).

    ``runs`` are the crystal's runs (template layout: each with its starting profile and
    the priors of its uncertain parameters in ``run.uncertainties``); ``profiles`` the
    measured profiles by name. ``own_segments`` maps a run to the segment labels it fits
    on its own instead of sharing. ``fo_offset_sd`` (Fo fraction) and
    ``edge_offset_sd_um``: the Gaussian prior sd of each profile's Fo offset and edge
    offset, one value for all or one per run (0 leaves it out); the edge offset is off
    by default (`DEFAULT_EDGE_OFFSET_SD_UM` is a value to turn it on with). With every
    profile carrying an Fo offset the data fix only their differences, so one profile is
    the reference with no offset: ``fo_reference`` "auto" (default) picks the profile with
    the most points, a run name picks that one, None keeps every offset free; a mapping
    given for ``fo_offset_sd`` is used as given. ``fits``: the individual
    fits where already done (each missing one is fitted with `fit_run`). ``grid``: one
    grid for every profile; by default each profile's scan grid (`forster.io.scan_run`)
    with the smallest time step and longest run among them. ``method``, ``n_samples``,
    ``nlive``, ``seed``, ``ess_fallback``, ``progress`` and ``cancel`` as for
    `forster.sampling.fit_initial_condition`; the same seed gives the same result.
    """
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; use one of {', '.join(METHODS)}")
    runs = tuple(runs)
    crystal = _check_runs(runs)
    own = _own_segments(runs, own_segments)
    _check_shared(runs, own)
    fo_sd = _per_run(fo_offset_sd, runs, "fo_offset_sd")
    if not isinstance(fo_offset_sd, Mapping):
        fo_sd = _with_reference(fo_sd, runs, profiles, fo_reference)
    edge_sd = _per_run(edge_offset_sd_um, runs, "edge_offset_sd_um")
    say = progress or (lambda stage: None)
    say("individual fits")
    individual = _individual(runs, profiles, fits)
    agree = _check_agreement(individual, crystal)
    like = _likelihood(
        runs, profiles, _grids(runs, profiles, individual, grid), own, fo_sd, edge_sd
    )
    like.cancel = cancel
    say("best fit")
    z, noise = _best_fit(like)
    best_theta = like.nominal + z * like.spread
    offsets = _warn_best(like, z)
    if len(like.names):
        samples, used, ess, log_z = sample_posterior(
            like, z, method, n_samples, nlive, seed, ess_fallback, say, whiten=True, scaled=True
        )
    else:
        (samples, log_z), used, ess = _fixed_samples(like), "fixed", math.nan
    summary = _summarise(like, samples)
    if summary["tail"] > TAIL_MASS:
        notes.warn(
            f"{summary['tail']:.1%} of the posterior of time lies in the last "
            f"{TAIL_FRACTION:.0%} of the run; give a longer scan_max_time_days",
            stacklevel=2,
        )
    chi2 = like.profile_chi2(best_theta)
    initials = like.candidates(best_theta)
    if chi2 is None or initials is None:  # the optimiser keeps to valid values
        raise ValueError("the best-fit starting profiles are invalid")
    joint_rows = np.zeros(like.rows)
    for p, c in zip(like.parts, chi2, strict=True):
        joint_rows = joint_rows + p.log_like(c)  # as log_like_rows
    best_row = int(np.argmax(joint_rows))
    names = [p.run.name for p in like.parts]
    dt_days = like.grid.dt_s / SECONDS_PER_DAY
    n_points = sum(len(p.y) for p in like.parts)
    return CrystalFit(
        crystal=crystal,
        runs=tuple(names),
        method=used,
        names=like.names,
        best_values=dict(zip(like.names, map(float, best_theta), strict=True)),
        best_initials=dict(zip(names, initials, strict=True)),
        best_days=best_row * like.grid.dt_s / SECONDS_PER_DAY,  # as fit_initial_condition
        median_days=summary["median_days"],
        low_days=summary["low_days"],
        high_days=summary["high_days"],
        **_time_ranges(summary["density"], np.sum(chi2, axis=0), n_points, dt_days),
        days=summary["days"],
        density=summary["density"],
        mean=summary["mean"],
        sd=summary["sd"],
        **_profile_values(like, best_theta, initials, summary["sd"]),
        profile_days={n: int(np.argmin(c)) * dt_days for n, c in zip(names, chi2, strict=True)},
        individual=individual,
        agree=agree,
        noise=dict(zip(names, noise, strict=True)),
        n_points={p.run.name: len(p.y) for p in like.parts},
        ic_share=summary["ic_share"],
        sensitivity=summary["sensitivity"],
        prior_offsets=offsets,
        ess=ess,
        n_samples=len(samples.theta),
        log_evidence=log_z,
        model_runs=like.runs,
        seed=seed,
        grids={p.run.name: p.grid for p in like.parts},
    )
