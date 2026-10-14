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
import scipy.linalg as sla
from scipy import stats

from forster import dipra, engine
from forster.diffusivity import DEFAULT_DIFFUSIVITY as DEFAULT_FO_DIFFUSIVITY
from forster.diffusivity import Diffusivity, resolve
from forster.grid import STABILITY_RATIO, UNSTABLE_RATIO, stability_ratio, stable_dt
from forster.model import (
    EXPLICIT_SCHEMES,
    SCHEMES,
    SECONDS_PER_DAY,
    Advance,
    Conditions,
    FitResult,
    Grid,
    InitialCondition,
    Orientation,
    Profile,
    _fit_extending,
    _run_past_best_fit,
    fe_mg_diffusivity,
    model_grid,
)
from forster.uncertainty import (
    CHUNK_ROWS,
    NoiseModel,
    chi2_per_step,
    delta_chi2_interval,
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

# advance(state, n, first_row=0): ``first_row`` is the row ``state`` is at; the element
# steppers do not depend on it (it matches `forster.model`'s advance functions).
ELEMENT_SCHEMES = EXPLICIT_SCHEMES  # explicit steppers that take a separate Fo vector
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
    diffusivity: Diffusivity


def _depends_on_fo(D: Callable[[np.ndarray], np.ndarray]) -> bool:
    values = D(np.array([0.6, 0.95]))
    return bool(values[0] != values[1])


def _check_stable(
    species: _Species,
    fo_range: tuple[float, float],
    conditions: Conditions,
    orientation: Orientation,
    dt_s: float,
    dx_um: float,
) -> None:
    """D dt / dx^2 (`forster.grid.stability_ratio`) at or below `UNSTABLE_RATIO` over the
    Fo range the run passes through (Fo stays between its initial extremes)."""
    pair = [(species.diffusivity, fo_range)]
    ratio = stability_ratio(dx_um, dt_s, pair, conditions, orientation)
    if ratio > UNSTABLE_RATIO:
        limit = stable_dt(dx_um, pair, conditions, orientation, ratio=UNSTABLE_RATIO)
        chosen = stable_dt(dx_um, pair, conditions, orientation)
        raise ValueError(
            f"{species.name}: D dt / dx^2 = {ratio:.2f} exceeds {UNSTABLE_RATIO}, so the "
            f"explicit scheme is unstable; use dt_s <= {limit:.4g} or a larger dx_um "
            f"(forster.grid.stable_dt gives {chosen:.4g} s at ratio {STABILITY_RATIO})"
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
    advance_fo: bool | None = None,
) -> tuple[np.ndarray, Advance]:
    """Packed start [Fo, element 1, ...] and a function advancing it by n time steps.

    Each block of ``len(model_x_um)`` values is one species, Fo first. ``advance(state,
    n)`` returns n + 1 rows, the first being ``state``; it restarts exactly where it
    stopped, so running in blocks gives the same numbers as one run.

    ``advance_fo``: whether Fo diffuses. By default it does unless every element's
    diffusivity is independent of Fo (Ca); pass True when Fo itself is fitted.
    """
    if scheme not in ELEMENT_SCHEMES:
        known = "is not supported for elements yet" if scheme in SCHEMES else "is unknown"
        raise ValueError(
            f"scheme {scheme!r} {known}; element fits use {' or '.join(ELEMENT_SCHEMES)}"
        )
    dx_m = dx_um * 1e-6
    fo_d = fe_mg_diffusivity(fo_diffusivity)
    fo_intervals = list(fo_initial.fo_values)
    fo = _Species(
        "Fo",
        engine.step_condition(model_x_um, list(fo_initial.steps_um), fo_intervals),
        (fo_initial.edge_fo, fo_initial.core_fo),
        *fo_d.functions(conditions, orientation),
        fo_d,
    )
    species = [fo]
    for element, initial, diffusivity in elements:
        d = element_diffusivity(element, diffusivity)
        start = initial.start(model_x_um, fo_initial)
        species.append(
            _Species(
                element,
                start,
                (initial.edge, initial.core),
                *d.functions(conditions, orientation),
                d,
            )
        )
    move_fo = advance_fo
    if move_fo is None:
        move_fo = not elements or any(_depends_on_fo(s.D) for s in species[1:])
    fo_range = (min(fo_intervals), max(fo_intervals))
    for s in species if move_fo else species[1:]:
        _check_stable(s, fo_range, conditions, orientation, grid.dt_s, dx_um)
    return np.concatenate([s.start for s in species]), _advance(
        species, move_fo, _stepper(scheme, grid.dt_s, dx_m)
    )


def _advance(species: list[_Species], move_fo: bool, step: Step) -> Advance:
    n_x = len(species[0].start)
    fo, others = species[0], species[1:]

    def advance(state: np.ndarray, n: int, first_row: int = 0) -> np.ndarray:
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


def _times(interval, dt_s: float) -> tuple[float, float, float]:
    return (
        interval.best * dt_s,
        (interval.low - interval.best) * dt_s,
        (interval.high - interval.best) * dt_s,
    )


@dataclass(frozen=True)
class FittedInitial:
    """Fit an element's starting concentrations instead of giving them.

    The edge and core concentrations (and the plateau, with two steps) are fitted by
    generalised least squares at every time step: for a given Fo history the
    element's equation is linear in them, so the model at any concentrations is a
    sum of one or two unit runs plus a constant. The time's 1-sigma range then
    allows for them (they count as fitted parameters in `interval_threshold`).
    ``steps_um`` defaults to the Fo initial condition's step positions.
    """

    steps_um: tuple[float, ...] | None = None

    def steps_for(self, fo_initial: InitialCondition) -> tuple[float, ...]:
        steps = tuple(self.steps_um if self.steps_um is not None else fo_initial.steps_um)
        if len(steps) not in (1, 2):
            raise ValueError(f"{len(steps)} step positions: use one, or two with a plateau")
        return steps

    def basis(self, fo_initial: InitialCondition) -> list[ElementInitial]:
        """Unit starting profiles: 1 before the first step (and 1 on the plateau)."""
        steps = self.steps_for(fo_initial)
        if len(steps) == 1:
            return [ElementInitial(1.0, 0.0, steps_um=steps)]
        return [
            ElementInitial(1.0, 0.0, step=0.0, steps_um=steps),
            ElementInitial(0.0, 0.0, step=1.0, steps_um=steps),
        ]


Initial = ElementInitial | FittedInitial


@dataclass(frozen=True)
class _SpeciesFit:
    chi2: np.ndarray  # correlated chi-squared at every time step
    noise: NoiseModel
    # Fitted concentrations at every time step: [core, edge - core(, plateau - core)].
    coefficients: np.ndarray | None


@dataclass(frozen=True)
class _Observed:
    """A species with measured points, and how its model is read from the packed state.

    One block holds its model (given concentrations), or its unit runs (fitted).
    """

    name: str
    blocks: tuple[slice, ...]
    fitted: bool
    x_um: np.ndarray
    values: np.ndarray
    W: np.ndarray  # model grid -> measured points

    @property
    def n_linear(self) -> int:
        return 1 + len(self.blocks) if self.fitted else 0

    def _design(self, rows: np.ndarray) -> np.ndarray:
        ones = np.ones((len(rows), len(self.x_um)))
        return np.stack([ones, *(rows[:, b] @ self.W.T for b in self.blocks)], axis=2)

    def _profiled(self, rows: np.ndarray, L: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        """Least-squares concentrations at each row and the (whitened) sum of squares."""
        out_chi2, out_coef = [], []
        for start in range(0, len(rows), CHUNK_ROWS):
            A = self._design(rows[start : start + CHUNK_ROWS])
            y = self.values
            if L is not None:
                t, n, p = A.shape
                flat = A.transpose(1, 0, 2).reshape(n, t * p)
                A = sla.solve_triangular(L, flat, lower=True).reshape(n, t, p).transpose(1, 0, 2)
                y = sla.solve_triangular(L, y, lower=True)
            G = np.einsum("tni,tnj->tij", A, A)
            h = np.einsum("tni,n->ti", A, y)
            coef = np.einsum("tij,tj->ti", np.linalg.pinv(G), h)
            r = y[None, :] - np.einsum("tnp,tp->tn", A, coef)
            out_chi2.append((r**2).sum(axis=1))
            out_coef.append(coef)
        return np.concatenate(out_chi2), np.concatenate(out_coef)

    def rss(self, rows: np.ndarray) -> np.ndarray:
        """Plain sum of squared residuals at every row (concentrations fitted if free)."""
        if not self.fitted:
            return chi2_per_step(rows[:, self.blocks[0]], self.W, self.values)
        return self._profiled(rows, None)[0]

    def fit(self, rows: np.ndarray) -> _SpeciesFit:
        """As `fit_correlated`: a white first pass, the noise model at its best row, then
        the correlated chi-squared at every row."""
        if not self.fitted:
            curves = rows[:, self.blocks[0]]
            first = int(np.argmin(chi2_per_step(curves, self.W, self.values)))
            noise = fit_noise_model(self.x_um, self.values - self.W @ curves[first])
            cov = noise.covariance(self.x_um)
            return _SpeciesFit(chi2_per_step(curves, self.W, self.values, cov), noise, None)
        white, coef = self._profiled(rows, None)
        first = int(np.argmin(white))
        predicted = self._design(rows[first : first + 1])[0] @ coef[first]
        noise = fit_noise_model(self.x_um, self.values - predicted)
        L = np.linalg.cholesky(noise.covariance(self.x_um))
        chi2, coef = self._profiled(rows, L)
        return _SpeciesFit(chi2, noise, coef)

    def curves(self, rows: np.ndarray, fit: _SpeciesFit, best: int) -> np.ndarray:
        """The model on the grid at every row; fitted concentrations held at the best fit's."""
        if not self.fitted or fit.coefficients is None:
            return rows[:, self.blocks[0]]
        c = fit.coefficients[best]
        return c[0] + sum(c[k + 1] * rows[:, b] for k, b in enumerate(self.blocks))

    def initial_at(self, fit: _SpeciesFit, best: int, steps: tuple[float, ...]) -> ElementInitial:
        """The fitted starting concentrations at the best fit."""
        assert fit.coefficients is not None
        c = fit.coefficients[best]
        step = float(c[0] + c[2]) if len(self.blocks) == 2 else None
        return ElementInitial(float(c[0] + c[1]), float(c[0]), step=step, steps_um=steps)


@dataclass(frozen=True)
class _Setup:
    model_x: np.ndarray
    start: np.ndarray
    advance: Advance
    observed: list[_Observed]
    blocks: dict[str, tuple[slice, ...]]  # every modelled species, Fo included
    initials: dict[str, Initial]
    diffusivities: dict[str, Diffusivity]


def _setup(
    fo_profile: Profile | None,
    element_profiles: Sequence[ElementProfile],
    initials: Mapping[str, Initial],
    conditions: Conditions,
    orientation: Orientation,
    fo_initial: InitialCondition,
    grid: Grid,
    diffusivities: Mapping[str, Diffusivity | str],
    fo_diffusivity: Diffusivity | str,
    scheme: str,
) -> _Setup:
    names = [p.element for p in element_profiles]
    if len(set(names)) != len(names):
        raise ValueError(f"each element can appear once in a joint fit, got {names}")
    if fo_profile is None and not element_profiles:
        raise ValueError("nothing to fit: give an Fo profile, element profiles, or both")
    resolved = {n: initials.get(n, FittedInitial()) for n in names}
    ds = {n: element_diffusivity(n, diffusivities.get(n)) for n in names}
    all_x = [p.distance_um for p in element_profiles]
    if fo_profile is not None:
        all_x.insert(0, fo_profile.distance_um)
    step_x, dx_um, (low, high) = model_grid(np.concatenate(all_x), grid, fo_initial.domain_um)
    model_x = step_x[low : high + 1]
    n_x = len(model_x)
    runs: list[tuple[str, ElementInitial, Diffusivity]] = []
    blocks: dict[str, tuple[slice, ...]] = {"Fo": (slice(0, n_x),)}
    for n in names:
        init = resolved[n]
        unit = init.basis(fo_initial) if isinstance(init, FittedInitial) else [init]
        first = 1 + len(runs)
        blocks[n] = tuple(slice((first + k) * n_x, (first + k + 1) * n_x) for k in range(len(unit)))
        runs.extend((n, u, ds[n]) for u in unit)
    start, advance = coupled_model(
        conditions,
        orientation,
        fo_initial,
        runs,
        grid,
        model_x,
        dx_um,
        fo_diffusivity=fo_diffusivity,
        scheme=scheme,
        advance_fo=True if fo_profile is not None else None,
    )
    sources = [(p.element, p.distance_um, p.values) for p in element_profiles]
    if fo_profile is not None:
        sources.insert(0, ("Fo", fo_profile.distance_um, fo_profile.fo))
    observed = []
    for name, x, y in sources:
        px, py = points_in_domain(model_x, x, y)
        fitted = isinstance(resolved.get(name), FittedInitial)
        W = interpolation_matrix(model_x, px)
        observed.append(_Observed(name, blocks[name], fitted, px, py, W))
    return _Setup(model_x, start, advance, observed, blocks, dict(resolved), ds)


@dataclass(frozen=True)
class _JointChi2:
    chi2: np.ndarray
    fits: dict[str, _SpeciesFit]
    interval: Any
    threshold: float


def _joint_chi2(rows: np.ndarray, observed: list[_Observed], complete: bool) -> _JointChi2:
    fits = {o.name: o.fit(rows) for o in observed}
    total = np.sum([f.chi2 for f in fits.values()], axis=0)
    n_points = sum(len(o.x_um) for o in observed)
    threshold = interval_threshold(n_points, 1 + sum(o.n_linear for o in observed))
    span = delta_chi2_interval(total, threshold, end_is_limit=complete)
    return _JointChi2(total, fits, span, threshold)


def _run(setup: _Setup, grid: Grid, stop_early: bool) -> tuple[np.ndarray, _JointChi2]:
    observed = setup.observed
    if stop_early:
        # Each species divided by its data range, so none dominates by its units alone.
        scales = [1.0 if len(observed) == 1 else float(np.ptp(o.values)) or 1.0 for o in observed]

        def rss(rows: np.ndarray) -> np.ndarray:
            return np.sum(
                [o.rss(rows) / s**2 for o, s in zip(observed, scales, strict=True)], axis=0
            )

        packed = _run_past_best_fit(setup.start, setup.advance, grid.timesteps, None, None, rss=rss)
    else:
        packed = setup.advance(setup.start, grid.timesteps)
    return _fit_extending(
        packed,
        setup.advance,
        grid.timesteps,
        lambda rows, complete: _joint_chi2(rows, observed, complete),
    )


def _warn_if_unconstrained(name: str, best: int) -> None:
    if best == 0:
        warnings.warn(
            f"{name}: the best fit is at time zero, so the profile does not constrain a "
            "diffusion time (zoning within the noise, or the initial condition is wrong)",
            stacklevel=3,
        )


def fit_element(
    profile: ElementProfile,
    conditions: Conditions,
    orientation: Orientation,
    fo_initial: InitialCondition,
    initial: Initial | None,
    grid: Grid,
    *,
    diffusivity: Diffusivity | str | None = None,
    fo_diffusivity: Diffusivity | str = DEFAULT_FO_DIFFUSIVITY,
    scheme: str = "central",
    stop_early: bool = True,
) -> FitResult:
    """Fit an element's profile for its own diffusion time.

    The element diffuses from ``initial`` while Fo diffuses from ``fo_initial``
    (whose measured profile is not needed here). ``initial`` is an `ElementInitial`
    (given concentrations) or a `FittedInitial` (concentrations fitted at every time
    step); None means `FittedInitial()`. The best fit and 1-sigma range come from
    the element's points by the correlated method of `fit_profile`, with the same
    early stopping. ``diffusivity`` defaults to the element's DIPRA preset.

    The result's Fo-named fields (``data_fo``, ``fit_fo``) hold the element's values;
    ``element`` names it. ``inputs["element_initial"]`` is the starting profile used,
    with fitted concentrations at the best fit; ``curves`` hold those concentrations
    at every row. Warns if the best fit is at time zero.
    """
    element = profile.element
    setup = _setup(
        None,
        [profile],
        {element: initial if initial is not None else FittedInitial()},
        conditions,
        orientation,
        fo_initial,
        grid,
        {element: diffusivity} if diffusivity is not None else {},
        fo_diffusivity,
        scheme,
    )
    packed, joint = _run(setup, grid, stop_early)
    (observed,) = setup.observed
    fit, span = joint.fits[element], joint.interval
    _warn_if_unconstrained(f"{profile.name} {element}", span.best)
    init = setup.initials[element]
    if isinstance(init, FittedInitial):
        init = observed.initial_at(fit, span.best, init.steps_for(fo_initial))
    time = _times(span, grid.dt_s)
    empty = np.empty(0)
    return FitResult(
        name=profile.name,
        time_s=float(time[0]),
        time_minus_s=float(time[1]),
        time_plus_s=float(time[2]),
        indices=(span.best, span.low_row, span.high_row),
        sum_chi2=fit.chi2,
        model_x_um=setup.model_x,
        curves=observed.curves(packed, fit, span.best),
        data_x_um=profile.distance_um,
        data_fo=profile.values,
        interp_x_um=empty,
        interp_fo=empty,
        interp_fo_sigma=empty,
        dt_s=grid.dt_s,
        inputs={
            "conditions": conditions,
            "orientation": orientation,
            "initial": fo_initial,
            "element_initial": init,
            "concentrations_fitted": observed.fitted,
            "grid": grid,
            "diffusivity": setup.diffusivities[element],
            "fo_diffusivity": fe_mg_diffusivity(fo_diffusivity),
            "scheme": scheme,
            "uncertainty": "correlated",
            "stop_early": stop_early,
        },
        uncertainty="correlated",
        fit_x_um=observed.x_um,
        fit_fo=observed.values,
        noise=fit.noise,
        element=element,
    )


@dataclass(frozen=True)
class JointFit:
    """One diffusion time fitted to Fo and elements together.

    ``chi2`` is the sum over species of each one's correlated-noise chi-squared
    (``parts``), each with its own ``noise`` model fitted at its own first-pass best
    fit. The 1-sigma range is where the sum rises by ``threshold``
    (`interval_threshold` of all the points, counting fitted concentrations as
    parameters).

    Whether the species agree on one time: ``species_days`` is each one's own best
    time on the same run, ``delta_chi2`` how much each one's chi-squared rises from
    its own minimum to the joint best fit, and ``consistency_p`` the probability of a
    total rise (``consistency_chi2``) at least this large if they share one time
    (chi-squared with species - 1 degrees of freedom; asymptotic). A small value
    means the joint range should not be used as it stands.

    ``curves`` holds each species' model on ``model_x_um`` at every time step (row
    ``i`` at ``i * dt_s``), Fo included; fitted concentrations are held at the joint
    best fit's, which ``initials`` reports.
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
    species_days: dict[str, float]
    delta_chi2: dict[str, float]
    consistency_chi2: float
    consistency_p: float
    model_x_um: np.ndarray
    curves: dict[str, np.ndarray]
    initials: dict[str, ElementInitial]
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


def _consistency(
    fits: dict[str, _SpeciesFit], best: int, dt_s: float
) -> tuple[dict[str, float], dict[str, float], float, float]:
    own = {n: int(np.argmin(f.chi2)) for n, f in fits.items()}
    days = {n: own[n] * dt_s / SECONDS_PER_DAY for n in fits}
    rise = {n: float(f.chi2[best] - f.chi2[own[n]]) for n, f in fits.items()}
    total = float(sum(rise.values()))
    dof = len(fits) - 1
    p = float(stats.chi2.sf(total, dof)) if dof > 0 else float("nan")
    if dof > 0 and p < 0.05:
        worst = max(rise, key=lambda n: rise[n])
        warnings.warn(
            f"the species do not agree on one time (consistency p = {p:.2g}); "
            f"{worst} moves most (chi-squared +{rise[worst]:.1f}); own best times "
            + ", ".join(f"{n} {d:.3g} d" for n, d in days.items()),
            stacklevel=3,
        )
    return days, rise, total, p


def fit_joint(
    fo_profile: Profile | None,
    element_profiles: Sequence[ElementProfile],
    conditions: Conditions,
    orientation: Orientation,
    fo_initial: InitialCondition,
    element_initials: Mapping[str, Initial] | None,
    grid: Grid,
    *,
    diffusivities: Mapping[str, Diffusivity | str] | None = None,
    fo_diffusivity: Diffusivity | str = DEFAULT_FO_DIFFUSIVITY,
    scheme: str = "central",
    stop_early: bool = True,
) -> JointFit:
    """Fit one diffusion time to Fo and element profiles together.

    ``fo_profile`` may be None to fit the elements only (Fo still diffuses from
    ``fo_initial``). ``element_initials`` and ``diffusivities`` are keyed by element;
    an element without an initial condition has its concentrations fitted
    (`FittedInitial`). With Fo alone the result equals `fit_profile`'s correlated
    fit. Check ``consistency_p`` before using the range (it warns below 0.05).

    Open point: the threshold treats the summed chi-squared like one species with
    all the points, although each species' noise is estimated separately; see
    ``docs/plans/2026-10-13-minor-elements.md``.
    """
    setup = _setup(
        fo_profile,
        element_profiles,
        element_initials or {},
        conditions,
        orientation,
        fo_initial,
        grid,
        diffusivities or {},
        fo_diffusivity,
        scheme,
    )
    packed, joint = _run(setup, grid, stop_early)
    best = joint.interval.best
    days, rise, total, p = _consistency(joint.fits, best, grid.dt_s)
    by_name = {o.name: o for o in setup.observed}
    curves = {"Fo": packed[:, setup.blocks["Fo"][0]]}
    initials = {}
    for p_el in element_profiles:
        o, f = by_name[p_el.element], joint.fits[p_el.element]
        curves[p_el.element] = o.curves(packed, f, best)
        init = setup.initials[p_el.element]
        initials[p_el.element] = (
            o.initial_at(f, best, init.steps_for(fo_initial))
            if isinstance(init, FittedInitial)
            else init
        )
    time = _times(joint.interval, grid.dt_s)
    return JointFit(
        time_s=float(time[0]),
        time_minus_s=float(time[1]),
        time_plus_s=float(time[2]),
        indices=(best, joint.interval.low_row, joint.interval.high_row),
        chi2=joint.chi2,
        parts={n: f.chi2 for n, f in joint.fits.items()},
        noise={n: f.noise for n, f in joint.fits.items()},
        n_points={o.name: len(o.x_um) for o in setup.observed},
        threshold=joint.threshold,
        species_days=days,
        delta_chi2=rise,
        consistency_chi2=total,
        consistency_p=p,
        model_x_um=setup.model_x,
        curves=curves,
        initials=initials,
        data={o.name: (o.x_um, o.values) for o in setup.observed},
        dt_s=grid.dt_s,
        inputs={
            "conditions": conditions,
            "orientation": orientation,
            "initial": fo_initial,
            "element_initials": dict(setup.initials),
            "grid": grid,
            "diffusivities": setup.diffusivities,
            "fo_diffusivity": fe_mg_diffusivity(fo_diffusivity),
            "scheme": scheme,
            "stop_early": stop_early,
        },
    )
