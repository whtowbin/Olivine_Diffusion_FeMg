"""Ni, Mn and Ca diffusion alongside Fo, and fits of their profiles.

Ni and Mn diffusivities depend on X_Fo, so they are modelled with the Fo profile
evolving alongside them, on the same grid and time step:

    dC/dt = d/dx(D_el(X_Fo) dC/dx),   dX_Fo/dt = d/dx(D_FeMg(X_Fo) dX_Fo/dx)

Each time step advances Fo with the same stepper `fit_profile` uses, so the Fo rows
are identical to its curves, and advances each element with that stepper given the
current Fo vector (the cross term then uses dD/dx = dD/dX_Fo dX_Fo/dx). Ca's
diffusivity (DIPRA Table 1) does not depend on Fo; when no modelled element depends
on Fo, Fo is not advanced. The equation is linear in each element's concentration,
so any unit (wt% oxide, ppm) can be fitted as measured.

`fit_element` fits one element's profile for its own time, a check against the Fo
time; `fit_joint` fits one time to Fo and the elements together. Both use the
correlated-noise likelihood of `forster.uncertainty` on the measured points, each
species with its own noise model. Plan and decisions:
``docs/plans/2026-10-13-minor-elements.md``.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from forster import dipra, engine
from forster.diffusivity import DEFAULT_DIFFUSIVITY as DEFAULT_FO_DIFFUSIVITY
from forster.diffusivity import Diffusivity, resolve
from forster.model import (
    SCHEMES,
    SECONDS_PER_DAY,
    Conditions,
    FitResult,
    Grid,
    InitialCondition,
    Orientation,
    Profile,
    _run_past_best_fit,
    fe_mg_diffusivity,
    model_grid,
)
from forster.uncertainty import (
    CorrelatedFit,
    NoiseModel,
    chi2_per_step,
    delta_chi2_interval,
    fit_correlated,
    fit_noise_model,
    interpolation_matrix,
    interval_threshold,
    points_in_domain,
)

ELEMENTS = ("Ni", "Mn", "Ca")
DEFAULT_DIFFUSIVITY = {
    "Ni": "girona_costa2013_ni",
    "Mn": "girona_costa2013_mn",
    "Ca": "girona_costa2013_ca",
}
STABILITY_LIMIT = 0.5  # the explicit schemes need D dt / dx^2 at or below this

Advance = Callable[[np.ndarray, int], np.ndarray]
Step = Callable[..., np.ndarray]


@dataclass(frozen=True)
class ElementProfile:
    """A measured Ni, Mn or Ca profile, in any unit (wt% oxide, ppm).

    Missing analyses may be NaN; at least 3 values must be finite.
    """

    distance_um: np.ndarray
    values: np.ndarray
    element: str
    name: str = "profile"

    def __post_init__(self) -> None:
        if self.element not in ELEMENTS:
            raise ValueError(f"element {self.element!r}; use one of {', '.join(ELEMENTS)}")
        distance = np.asarray(self.distance_um, dtype=float)
        values = np.asarray(self.values, dtype=float)
        if distance.shape != values.shape or distance.ndim != 1:
            raise ValueError(
                f"{self.name} {self.element}: distance_um and values must be 1-D arrays of the "
                f"same length, got shapes {distance.shape} and {values.shape}"
            )
        if np.isfinite(values).sum() < 3:
            raise ValueError(f"{self.name} {self.element}: need at least 3 measured values")
        object.__setattr__(self, "distance_um", distance)
        object.__setattr__(self, "values", values)


@dataclass(frozen=True)
class ElementInitial:
    """Step-function starting profile of an element, in the profile's units.

    ``edge`` is held at the start of the model domain and ``core`` at its end.
    ``steps_um`` defaults to the Fo initial condition's step positions; ``step``
    is the plateau between two steps (required with two steps, as for Fo).
    """

    edge: float
    core: float
    step: float | None = None
    steps_um: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        values = [self.edge, self.core] + ([self.step] if self.step is not None else [])
        if not all(np.isfinite(v) for v in values):
            raise ValueError(f"element initial concentrations must be finite, got {values}")

    def steps_for(self, fo_initial: InitialCondition) -> tuple[float, ...]:
        """Step positions: this element's own, or the Fo initial condition's."""
        steps = self.steps_um if self.steps_um is not None else fo_initial.steps_um
        expected = 2 if self.step is not None else 1
        if len(steps) != expected:
            source = "given" if self.steps_um is not None else "taken from the Fo initial condition"
            raise ValueError(
                f"{len(steps)} step position(s) {source}: use one step, or two with a "
                "plateau value (step)"
            )
        return tuple(steps)

    def start(self, model_x_um: np.ndarray, fo_initial: InitialCondition) -> np.ndarray:
        intervals = [self.edge, self.core]
        if self.step is not None:
            intervals[1:1] = [self.step]
        return engine.step_condition(model_x_um, list(self.steps_for(fo_initial)), intervals)


def element_diffusivity(element: str, diffusivity: Diffusivity | str | None = None) -> Diffusivity:
    """The equation for ``element``: the DIPRA preset by default, checked for the element."""
    if element not in ELEMENTS:
        raise ValueError(f"element {element!r}; use one of {', '.join(ELEMENTS)}")
    resolved = resolve(diffusivity if diffusivity is not None else DEFAULT_DIFFUSIVITY[element])
    if resolved.element != element:
        raise ValueError(f"diffusivity {resolved.name!r} is for {resolved.element}, not {element}")
    return resolved


@dataclass(frozen=True)
class _Species:
    name: str
    start: np.ndarray
    bounds: tuple[float, float]
    D: Callable[[np.ndarray], np.ndarray]
    dD: Callable[[np.ndarray], np.ndarray]


def _depends_on_fo(D: Callable[[np.ndarray], np.ndarray]) -> bool:
    values = D(np.array([0.6, 0.95]))
    return bool(values[0] != values[1])


def _check_stable(species: _Species, fo_values: Sequence[float], dt_s: float, dx_m: float):
    """D dt / dx^2 at or below `STABILITY_LIMIT` over the Fo values the run passes through.

    D is monotonic in Fo for every equation here, so the extremes of the initial Fo
    values (between which Fo stays) bound it.
    """
    d_max = float(np.max(species.D(np.asarray(fo_values, dtype=float))))
    ratio = d_max * dt_s / dx_m**2
    if ratio > STABILITY_LIMIT:
        raise ValueError(
            f"{species.name}: D dt / dx^2 = {ratio:.2f} exceeds {STABILITY_LIMIT}, so the "
            f"explicit scheme is unstable; use dt_s <= {STABILITY_LIMIT * dx_m**2 / d_max:.4g} "
            "or a larger dx_um"
        )


def _stepper(scheme: str, dt_s: float, dx_m: float) -> Step:
    """One time step of a species given the Fo vector, with the scheme's own stepper."""
    if scheme == "central":
        kernel_1, kernel_2, delta = engine.diffusion_kernel(dt=dt_s, dx=dx_m)

        def central(c, fo, D, dD, bounds_c, bounds_fo):
            return engine.diffusion_step(c, fo, D, kernel_1, kernel_2, delta, bounds_c, bounds_fo)

        return central

    def dipra_step(c, fo, D, dD, bounds_c, bounds_fo):
        return dipra.diffusion_step(c, fo, D, dD, bounds_c, bounds_fo, dt_s, dx_m)

    return dipra_step


def coupled_model(
    conditions: Conditions,
    orientation: Orientation,
    fo_initial: InitialCondition,
    elements: Sequence[tuple[str, ElementInitial, Diffusivity | str | None]],
    grid: Grid,
    model_x_um: np.ndarray,
    dx_um: float,
    *,
    fo_diffusivity: Diffusivity | str = DEFAULT_FO_DIFFUSIVITY,
    scheme: str = "central",
) -> tuple[np.ndarray, Advance]:
    """Packed start [Fo, element 1, ...] and a function advancing it by n time steps.

    Each block of ``len(model_x_um)`` values is one species, Fo first. ``advance(state,
    n)`` returns n + 1 rows, the first being ``state``; it restarts exactly where it
    stopped, so running in blocks gives the same numbers as one run.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}; use one of {', '.join(SCHEMES)}")
    dx_m = dx_um * 1e-6
    fo_d = fe_mg_diffusivity(fo_diffusivity)
    fo_intervals = [fo_initial.edge_fo, fo_initial.core_fo]
    if fo_initial.step_fo is not None:
        fo_intervals[1:1] = [fo_initial.step_fo]
    fo = _Species(
        "Fo",
        engine.step_condition(model_x_um, list(fo_initial.steps_um), fo_intervals),
        (fo_initial.edge_fo, fo_initial.core_fo),
        *fo_d.functions(conditions, orientation),
    )
    species = [fo]
    for element, initial, diffusivity in elements:
        d = element_diffusivity(element, diffusivity)
        start = initial.start(model_x_um, fo_initial)
        species.append(
            _Species(
                element, start, (initial.edge, initial.core), *d.functions(conditions, orientation)
            )
        )
    move_fo = not elements or any(_depends_on_fo(s.D) for s in species[1:])
    for s in species if move_fo else species[1:]:
        _check_stable(s, fo_intervals, grid.dt_s, dx_m)
    return np.concatenate([s.start for s in species]), _advance(
        species, move_fo, _stepper(scheme, grid.dt_s, dx_m)
    )


def _advance(species: list[_Species], move_fo: bool, step: Step) -> Advance:
    n_x = len(species[0].start)
    fo, others = species[0], species[1:]

    def advance(state: np.ndarray, n: int) -> np.ndarray:
        rows = np.empty((n + 1, len(state)))
        rows[0] = state
        x_fo = state[:n_x]
        cs = [state[(i + 1) * n_x : (i + 2) * n_x] for i in range(len(others))]
        for k in range(1, n + 1):
            # Elements step with the Fo vector of the start of the step, as Fo itself does.
            cs = [
                step(c, x_fo, s.D, s.dD, s.bounds, fo.bounds)
                for c, s in zip(cs, others, strict=True)
            ]
            if move_fo:
                x_fo = step(x_fo, x_fo, fo.D, fo.dD, fo.bounds, fo.bounds)
            rows[k, :n_x] = x_fo
            for i, c in enumerate(cs):
                rows[k, (i + 1) * n_x : (i + 2) * n_x] = c
        return rows

    return advance


def _fit_extending(
    packed: np.ndarray,
    advance: Advance,
    total_steps: int,
    fit: Callable[[np.ndarray, bool], Any],
) -> tuple[np.ndarray, Any]:
    """``fit(packed, complete)``, extending the run while the 1-sigma range reaches its end."""
    while True:
        complete = len(packed) - 1 >= total_steps
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = fit(packed, complete)
        if complete or result.interval.high_row < len(packed) - 1:
            for w in caught:
                warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
            return packed, result
        extra = min(len(packed) - 1, total_steps - (len(packed) - 1))
        packed = np.concatenate([packed, advance(packed[-1], extra)[1:]])


def _times(interval, dt_s: float) -> tuple[float, float, float]:
    return (
        interval.best * dt_s,
        (interval.low - interval.best) * dt_s,
        (interval.high - interval.best) * dt_s,
    )


def fit_element(
    profile: ElementProfile,
    conditions: Conditions,
    orientation: Orientation,
    fo_initial: InitialCondition,
    initial: ElementInitial,
    grid: Grid,
    *,
    diffusivity: Diffusivity | str | None = None,
    fo_diffusivity: Diffusivity | str = DEFAULT_FO_DIFFUSIVITY,
    scheme: str = "central",
    stop_early: bool = True,
) -> FitResult:
    """Fit an element's profile for its own diffusion time.

    The element diffuses from ``initial`` while Fo diffuses from ``fo_initial``
    (whose measured profile is not needed here). The best fit and 1-sigma range come
    from the element's measured points by the correlated method of `fit_profile`,
    with the same early stopping. ``diffusivity`` defaults to the element's DIPRA
    preset (`DEFAULT_DIFFUSIVITY`). The result's ``curves`` are the element's; its
    Fo-named fields (``data_fo``, ``fit_fo``) hold the element's values, and
    ``element`` says which element.
    """
    d = element_diffusivity(profile.element, diffusivity)
    x, y = profile.distance_um, profile.values
    step_x, dx_um, (low, high) = model_grid(x, grid, fo_initial.domain_um)
    model_x = step_x[low : high + 1]
    start, advance = coupled_model(
        conditions,
        orientation,
        fo_initial,
        [(profile.element, initial, d)],
        grid,
        model_x,
        dx_um,
        fo_diffusivity=fo_diffusivity,
        scheme=scheme,
    )
    n_x = len(model_x)
    columns = slice(n_x, 2 * n_x)
    if stop_early:
        px, py = points_in_domain(model_x, x, y)
        W = np.zeros((len(px), 2 * n_x))
        W[:, columns] = interpolation_matrix(model_x, px)
        packed = _run_past_best_fit(start, advance, grid.timesteps, W, py)
    else:
        packed = advance(start, grid.timesteps)

    def fit_columns(rows: np.ndarray, complete: bool) -> CorrelatedFit:
        return fit_correlated(rows[:, columns], model_x, x, y, end_is_limit=complete)

    packed, fit = _fit_extending(packed, advance, grid.timesteps, fit_columns)
    time = _times(fit.interval, grid.dt_s)
    empty = np.empty(0)
    return FitResult(
        name=profile.name,
        time_s=float(time[0]),
        time_minus_s=float(time[1]),
        time_plus_s=float(time[2]),
        indices=(fit.interval.best, fit.interval.low_row, fit.interval.high_row),
        sum_chi2=fit.chi2,
        model_x_um=model_x,
        curves=packed[:, columns],
        data_x_um=x,
        data_fo=y,
        interp_x_um=empty,
        interp_fo=empty,
        interp_fo_sigma=empty,
        dt_s=grid.dt_s,
        inputs={
            "conditions": conditions,
            "orientation": orientation,
            "initial": fo_initial,
            "element_initial": initial,
            "grid": grid,
            "diffusivity": d,
            "fo_diffusivity": fe_mg_diffusivity(fo_diffusivity),
            "scheme": scheme,
            "uncertainty": "correlated",
            "stop_early": stop_early,
        },
        uncertainty="correlated",
        fit_x_um=fit.x_um,
        fit_fo=fit.fo,
        noise=fit.noise,
        element=profile.element,
    )


@dataclass(frozen=True)
class _Observed:
    """A species with measured points: its block of the packed state and its data."""

    name: str
    columns: slice
    x_um: np.ndarray
    values: np.ndarray
    W: np.ndarray  # model block -> measured points


@dataclass(frozen=True)
class JointFit:
    """One diffusion time fitted to Fo and elements together.

    ``chi2`` is the sum over species of each one's correlated-noise chi-squared
    (``parts``), each with its own ``noise`` model fitted at its own first-pass best
    fit. The 1-sigma range is where the sum rises by ``threshold``
    (`interval_threshold` of the total number of points). ``curves`` holds each
    species' model on ``model_x_um`` at every time step (row ``i`` at ``i * dt_s``).
    """

    time_s: float
    time_minus_s: float
    time_plus_s: float
    indices: tuple[int, int, int]
    chi2: np.ndarray
    parts: dict[str, np.ndarray]
    noise: dict[str, NoiseModel]
    n_points: dict[str, int]
    threshold: float
    model_x_um: np.ndarray
    curves: dict[str, np.ndarray]
    data: dict[str, tuple[np.ndarray, np.ndarray]]
    dt_s: float
    inputs: dict[str, Any]

    @property
    def time_days(self) -> float:
        return self.time_s / SECONDS_PER_DAY

    @property
    def low_days(self) -> float:
        return (self.time_s + self.time_minus_s) / SECONDS_PER_DAY

    @property
    def high_days(self) -> float:
        return (self.time_s + self.time_plus_s) / SECONDS_PER_DAY


@dataclass(frozen=True)
class _JointChi2:
    chi2: np.ndarray
    parts: dict[str, np.ndarray]
    noise: dict[str, NoiseModel]
    interval: Any
    threshold: float


def _joint_chi2(packed: np.ndarray, observed: list[_Observed], complete: bool) -> _JointChi2:
    """Each species' correlated chi-squared (as `fit_correlated` computes it) and the sum."""
    parts, noise = {}, {}
    for o in observed:
        curves = packed[:, o.columns]
        first = int(np.argmin(chi2_per_step(curves, o.W, o.values)))
        noise[o.name] = fit_noise_model(o.x_um, o.values - o.W @ curves[first])
        parts[o.name] = chi2_per_step(curves, o.W, o.values, noise[o.name].covariance(o.x_um))
    total = np.sum(list(parts.values()), axis=0)
    threshold = interval_threshold(sum(len(o.x_um) for o in observed))
    span = delta_chi2_interval(total, threshold, end_is_limit=complete)
    return _JointChi2(total, parts, noise, span, threshold)


def _stop_rows(observed: list[_Observed], width: int) -> tuple[np.ndarray, np.ndarray]:
    """W and y for the early-stopping rule over all species.

    With more than one species each is divided by its data range, so no species
    dominates the sum of squares by its units alone.
    """
    blocks_W, blocks_y = [], []
    for o in observed:
        scale = 1.0 if len(observed) == 1 else float(np.ptp(o.values)) or 1.0
        W = np.zeros((len(o.x_um), width))
        W[:, o.columns] = o.W / scale
        blocks_W.append(W)
        blocks_y.append(o.values / scale)
    return np.vstack(blocks_W), np.concatenate(blocks_y)


def fit_joint(
    fo_profile: Profile | None,
    element_profiles: Sequence[ElementProfile],
    conditions: Conditions,
    orientation: Orientation,
    fo_initial: InitialCondition,
    element_initials: Mapping[str, ElementInitial],
    grid: Grid,
    *,
    diffusivities: Mapping[str, Diffusivity | str] | None = None,
    fo_diffusivity: Diffusivity | str = DEFAULT_FO_DIFFUSIVITY,
    scheme: str = "central",
    stop_early: bool = True,
) -> JointFit:
    """Fit one diffusion time to Fo and element profiles together.

    ``fo_profile`` may be None to fit the elements only (Fo still diffuses from
    ``fo_initial``). ``element_initials`` and ``diffusivities`` are keyed by element.
    With Fo alone the result equals `fit_profile`'s correlated fit.

    Open point: the threshold treats the summed chi-squared like one species with
    all the points, although each species' noise is estimated separately; see
    ``docs/plans/2026-10-13-minor-elements.md``.
    """
    names = [p.element for p in element_profiles]
    if len(set(names)) != len(names):
        raise ValueError(f"each element can appear once in a joint fit, got {names}")
    missing = [n for n in names if n not in element_initials]
    if missing:
        raise ValueError(f"no initial condition for {', '.join(missing)}")
    if fo_profile is None and not element_profiles:
        raise ValueError("nothing to fit: give an Fo profile, element profiles, or both")
    diffusivities = diffusivities or {}
    all_x = [p.distance_um for p in element_profiles]
    if fo_profile is not None:
        all_x.insert(0, fo_profile.distance_um)
    step_x, dx_um, (low, high) = model_grid(np.concatenate(all_x), grid, fo_initial.domain_um)
    model_x = step_x[low : high + 1]
    start, advance = coupled_model(
        conditions,
        orientation,
        fo_initial,
        [(n, element_initials[n], diffusivities.get(n)) for n in names],
        grid,
        model_x,
        dx_um,
        fo_diffusivity=fo_diffusivity,
        scheme=scheme,
    )
    observed = _observed(fo_profile, element_profiles, model_x)
    if stop_early:
        W, y = _stop_rows(observed, len(start))
        packed = _run_past_best_fit(start, advance, grid.timesteps, W, y)
    else:
        packed = advance(start, grid.timesteps)
    packed, joint = _fit_extending(
        packed, advance, grid.timesteps, lambda rows, done: _joint_chi2(rows, observed, done)
    )
    n_x = len(model_x)
    blocks = ["Fo", *names]
    time = _times(joint.interval, grid.dt_s)
    return JointFit(
        time_s=float(time[0]),
        time_minus_s=float(time[1]),
        time_plus_s=float(time[2]),
        indices=(joint.interval.best, joint.interval.low_row, joint.interval.high_row),
        chi2=joint.chi2,
        parts=joint.parts,
        noise=joint.noise,
        n_points={o.name: len(o.x_um) for o in observed},
        threshold=joint.threshold,
        model_x_um=model_x,
        curves={b: packed[:, i * n_x : (i + 1) * n_x] for i, b in enumerate(blocks)},
        data={o.name: (o.x_um, o.values) for o in observed},
        dt_s=grid.dt_s,
        inputs={
            "conditions": conditions,
            "orientation": orientation,
            "initial": fo_initial,
            "element_initials": dict(element_initials),
            "grid": grid,
            "diffusivities": {n: element_diffusivity(n, diffusivities.get(n)) for n in names},
            "fo_diffusivity": fe_mg_diffusivity(fo_diffusivity),
            "scheme": scheme,
            "stop_early": stop_early,
        },
    )


def _observed(
    fo_profile: Profile | None, element_profiles: Sequence[ElementProfile], model_x: np.ndarray
) -> list[_Observed]:
    """The species with data, in packed-state order (Fo first, then the elements)."""
    n_x = len(model_x)
    sources: list[tuple[str, int, np.ndarray, np.ndarray]] = []
    if fo_profile is not None:
        sources.append(("Fo", 0, fo_profile.distance_um, fo_profile.fo))
    for i, p in enumerate(element_profiles, 1):
        sources.append((p.element, i, p.distance_um, p.values))
    observed = []
    for name, block, x, y in sources:
        px, py = points_in_domain(model_x, x, y)
        observed.append(
            _Observed(
                name,
                slice(block * n_x, (block + 1) * n_x),
                px,
                py,
                interpolation_matrix(model_x, px),
            )
        )
    return observed
