"""Choosing the numerical grid (dx, dt, longest time) for a profile, and checking a given one.

The explicit schemes need D_max dt / dx^2 <= 0.5 to be stable; `stable_dt` gives the
time step at ``STABILITY_RATIO`` (0.4) of that, for the largest diffusivity of every
species over the compositions it passes through. Raising dt to that ratio changed
fitted times by under 0.2% (``docs/references/likelihood-method.md``, section 7).

The grid spacing sets the accuracy. On six real profiles (round robin and Towbin and
Plank 2026), the error in the fitted time fell roughly as 1/k with k = l / dx, l the
diffusion length sqrt(D t) at the fit: worst case about 0.65 / k (5.6% at k = 12, 3%
at k = 20, 1.3% at k = 30-40, against k = 60). The schemes are first order in dx here
because the boundary values sit one node outside the domain and the initial steps
snap to the nearest node. `choose_grid` therefore sets dx = l * tolerance / 0.65,
and never more than a quarter of the narrowest gap between steps (or between a step
and an end of the domain), so the initial condition is resolved:

- ``purpose="scan"``: tolerance 5%, for parameter scans (k = 13);
- ``purpose="final"``: tolerance 1%, for the final fit (k = 65).

l comes from a quick fit on a coarse grid, using the slowest species' diffusivity at
the middle of its composition range (for several species, each one's own best time
would be more exact; the slowest D with the Fo time is the default). `check_grid`
reports what is wrong or wasteful about a grid a user gives.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from forster.diffusivity import DEFAULT_DIFFUSIVITY, Diffusivity, resolve
from forster.model import (
    SECONDS_PER_DAY,
    Conditions,
    FitResult,
    Grid,
    InitialCondition,
    Orientation,
    Profile,
    fe_mg_diffusivity,
    fit_profile,
    model_grid,
)

STABILITY_RATIO = 0.4  # D_max dt / dx^2 used for chosen grids
UNSTABLE_RATIO = 0.5  # the explicit schemes diverge above this
ERROR_COEFFICIENT = 0.65  # fitted-time error ~ ERROR_COEFFICIENT * dx / l (worst case measured)
TOLERANCES = {"scan": 0.05, "final": 0.01}
MIN_NODES_PER_GAP = 4  # nodes across the narrowest feature of the initial condition
MAX_TIME_FACTOR = 3.0  # chosen max time = this x the coarse best fit
COARSE_POINTS = 150  # nodes of the coarse grid that estimates the time
COARSE_MAX_DAYS = 3650.0  # coarse fit horizon when none is given (early stopping ends it)
COARSE_MIN_RESOLUTION = 8.0  # refine the coarse fit once if l / dx is below this
COST_FACTOR = 4.0  # warn of wasted time when a grid is this much finer than needed

Species = tuple[Diffusivity | str, tuple[float, float]]


def fo_range(initial: InitialCondition) -> tuple[float, float]:
    """Smallest and largest Fo of the initial condition (edge, core and any plateau)."""
    values = [initial.edge_fo, initial.core_fo] + (
        [initial.step_fo] if initial.step_fo is not None else []
    )
    return min(values), max(values)


def _diffusivities(
    species: Sequence[Species], conditions: Conditions, orientation: Orientation
) -> list[tuple[float, float]]:
    """(largest, middle) diffusivity of each species over its Fo range, m^2/s."""
    out = []
    for diffusivity, (lo, hi) in species:
        D, _ = resolve(diffusivity).functions(conditions, orientation)
        fo = np.linspace(lo, hi, 101)
        values = np.broadcast_to(np.asarray(D(fo), dtype=float), fo.shape)
        out.append((float(values.max()), float(D(np.array([(lo + hi) / 2]))[0])))
    return out


def stable_dt(
    dx_um: float,
    species: Sequence[Species],
    conditions: Conditions,
    orientation: Orientation,
    ratio: float = STABILITY_RATIO,
) -> float:
    """Time step (s) with D_max dt / dx^2 = ``ratio``, D_max over every species and its
    Fo range. ``dx_um`` should be the grid's actual spacing (from `model_grid`)."""
    d_max = max(d for d, _ in _diffusivities(species, conditions, orientation))
    return ratio * (dx_um * 1e-6) ** 2 / d_max


def stability_ratio(
    dx_um: float,
    dt_s: float,
    species: Sequence[Species],
    conditions: Conditions,
    orientation: Orientation,
) -> float:
    """D_max dt / dx^2 for a grid: stable up to 0.5."""
    d_max = max(d for d, _ in _diffusivities(species, conditions, orientation))
    return d_max * dt_s / (dx_um * 1e-6) ** 2


def diffusion_length_um(
    time_days: float,
    species: Sequence[Species],
    conditions: Conditions,
    orientation: Orientation,
) -> float:
    """sqrt(D t) with the slowest species' D at the middle of its Fo range (um)."""
    d_slow = min(d_mid for _, d_mid in _diffusivities(species, conditions, orientation))
    return math.sqrt(d_slow * time_days * SECONDS_PER_DAY) * 1e6


def narrowest_gap_um(initial: InitialCondition) -> float:
    """Smallest distance between steps, or between a step and an end of the domain."""
    points = sorted([initial.domain_um[0], *initial.steps_um, initial.domain_um[1]])
    gaps = [b - a for a, b in itertools.pairwise(points) if b > a]
    return min(gaps) if gaps else initial.domain_um[1] - initial.domain_um[0]


def actual_dx_um(profile: Profile, dx_um: float, initial: InitialCondition) -> float:
    """The spacing `fit_profile` will really use for a requested ``dx_um``."""
    _, dx, _ = model_grid(profile.distance_um, Grid(dx_um, 1.0), initial.domain_um)
    return dx


def dx_not_above(
    profile: Profile, target_um: float, initial: InitialCondition
) -> tuple[float, float]:
    """A ``dx_um`` to ask for, and the spacing `fit_profile` will then use, no coarser than
    ``target_um``. The grid's node count is rounded, so the two differ slightly; keep the
    requested value in the Grid so a refit builds the same grid."""
    request = target_um
    for _ in range(20):
        actual = actual_dx_um(profile, request, initial)
        if actual <= target_um:
            return request, actual
        request *= 0.999 * target_um / actual
    return request, actual_dx_um(profile, request, initial)


def estimated_error(dx_um: float, length_um: float) -> float:
    """Worst-case relative error of the fitted time from the grid spacing (fraction)."""
    return ERROR_COEFFICIENT * dx_um / length_um


@dataclass(frozen=True)
class GridChoice:
    """A chosen grid and why: diffusion length, resolution and estimated error."""

    grid: Grid
    purpose: str
    tolerance: float
    coarse_time_days: float
    diffusion_length_um: float
    resolution: float  # l / dx
    estimated_error: float  # fraction of the fitted time
    stability_ratio: float


def _species_or_default(
    species: Sequence[Species] | None, diffusivity: Diffusivity | str, initial: InitialCondition
) -> list[Species]:
    return list(species) if species else [(diffusivity, fo_range(initial))]


def _coarse_time_days(
    profile: Profile,
    conditions: Conditions,
    orientation: Orientation,
    initial: InitialCondition,
    diffusivity: Diffusivity | str,
    species: Sequence[Species],
    max_time_days: float,
    scheme: str,
) -> float:
    """Best-fit time from a quick fit on a coarse grid, refined once if too coarse."""
    span = initial.domain_um[1] - initial.domain_um[0]
    dx = min(span / COARSE_POINTS, narrowest_gap_um(initial) / MIN_NODES_PER_GAP)
    for _ in range(2):
        actual = actual_dx_um(profile, dx, initial)
        grid = Grid(dx, stable_dt(actual, species, conditions, orientation), max_time_days)
        result = fit_profile(
            profile,
            conditions,
            orientation,
            initial,
            grid,
            None,
            fe_mg_diffusivity(diffusivity),
            scheme,
        )
        time = max(result.time_days, grid.dt_s / SECONDS_PER_DAY)
        length = diffusion_length_um(time, species, conditions, orientation)
        if length / actual >= COARSE_MIN_RESOLUTION:
            break
        dx = length / COARSE_MIN_RESOLUTION
    return time


def choose_grid(
    profile: Profile,
    conditions: Conditions,
    orientation: Orientation,
    initial: InitialCondition,
    diffusivity: Diffusivity | str = DEFAULT_DIFFUSIVITY,
    *,
    purpose: str = "final",
    tolerance: float | None = None,
    species: Sequence[Species] | None = None,
    time_days: float | None = None,
    max_time_days: float | None = None,
    scheme: str = "central",
) -> GridChoice:
    """A grid for fitting ``profile`` to within ``tolerance`` (default by ``purpose``).

    ``time_days``: the fitted time if already known (for example the nominal fit when
    choosing a scan's grid); otherwise a coarse fit estimates it. ``species``: for
    coupled element runs, every (diffusivity, Fo range) modelled; dt satisfies the
    fastest, dx resolves the slowest. ``max_time_days``: kept if given, otherwise
    3 x the estimated time.
    """
    if purpose not in TOLERANCES:
        raise ValueError(f"purpose {purpose!r}; use one of {', '.join(TOLERANCES)}")
    tolerance = tolerance or TOLERANCES[purpose]
    species = _species_or_default(species, diffusivity, initial)
    if time_days is None:
        time_days = _coarse_time_days(
            profile,
            conditions,
            orientation,
            initial,
            diffusivity,
            species,
            max_time_days or COARSE_MAX_DAYS,
            scheme,
        )
    length = diffusion_length_um(time_days, species, conditions, orientation)
    target = min(
        length * tolerance / ERROR_COEFFICIENT, narrowest_gap_um(initial) / MIN_NODES_PER_GAP
    )
    request, dx = dx_not_above(profile, target, initial)
    dt = stable_dt(dx, species, conditions, orientation)
    grid = Grid(request, dt, max_time_days or MAX_TIME_FACTOR * time_days)
    return GridChoice(
        grid,
        purpose,
        tolerance,
        time_days,
        length,
        length / dx,
        estimated_error(dx, length),
        stability_ratio(dx, dt, species, conditions, orientation),
    )


def check_grid(
    grid: Grid,
    profile: Profile,
    conditions: Conditions,
    orientation: Orientation,
    initial: InitialCondition,
    diffusivity: Diffusivity | str = DEFAULT_DIFFUSIVITY,
    *,
    time_days: float | None = None,
    purpose: str = "final",
    tolerance: float | None = None,
    species: Sequence[Species] | None = None,
) -> list[str]:
    """Problems with a given grid, as messages; raises if it is unstable.

    Stability is checked on the spacing actually used. With ``time_days`` (a fitted
    time), also the accuracy: estimated error above ``tolerance``, a grid much finer
    than needed (wasted time), and too few nodes across the initial condition.
    """
    tolerance = tolerance or TOLERANCES[purpose]
    species = _species_or_default(species, diffusivity, initial)
    dx = actual_dx_um(profile, grid.dx_um, initial)
    ratio = stability_ratio(dx, grid.dt_s, species, conditions, orientation)
    if ratio > UNSTABLE_RATIO:
        good = stable_dt(dx, species, conditions, orientation)
        raise ValueError(
            f"unstable grid: D_max dt / dx^2 = {ratio:.2f} > {UNSTABLE_RATIO} (dx {dx:.4g} um, "
            f"dt {grid.dt_s:.4g} s); use dt <= {good:.4g} s or a larger dx"
        )
    messages = []
    gap = narrowest_gap_um(initial)
    if gap / dx < MIN_NODES_PER_GAP:
        messages.append(
            f"dx {dx:.4g} um leaves {gap / dx:.1f} nodes across the narrowest part of the "
            f"initial condition ({gap:.4g} um); use dx <= {gap / MIN_NODES_PER_GAP:.4g} um"
        )
    if time_days is not None:
        length = diffusion_length_um(time_days, species, conditions, orientation)
        error = estimated_error(dx, length)
        needed = length * tolerance / ERROR_COEFFICIENT
        if error > tolerance:
            messages.append(
                f"dx {dx:.4g} um is coarse for a diffusion length of {length:.3g} um: the "
                f"fitted time may be off by up to {error:.0%} (aim for {tolerance:.0%}: "
                f"dx <= {needed:.3g} um)"
            )
        elif dx * COST_FACTOR < needed:
            messages.append(
                f"dx {dx:.4g} um is much finer than needed for {tolerance:.0%} "
                f"({needed:.3g} um would do); the fit is about "
                f"{(needed / dx) ** 3:.0f}x slower than necessary"
            )
    good_dt = stable_dt(dx, species, conditions, orientation)
    if grid.dt_s * COST_FACTOR < good_dt:
        messages.append(
            f"dt {grid.dt_s:.4g} s is {good_dt / grid.dt_s:.0f}x below the stable step "
            f"({good_dt:.4g} s): slower, no more accurate"
        )
    return messages


@dataclass(frozen=True)
class GridRequest:
    """A grid with any of its parts left to `choose_grid` (None)."""

    dx_um: float | None = None
    dt_s: float | None = None
    max_time_days: float | None = None

    def is_complete(self) -> bool:
        return None not in (self.dx_um, self.dt_s, self.max_time_days)

    def resolve(
        self,
        profile: Profile,
        conditions: Conditions,
        orientation: Orientation,
        initial: InitialCondition,
        diffusivity: Diffusivity | str = DEFAULT_DIFFUSIVITY,
        *,
        purpose: str = "final",
        time_days: float | None = None,
        species: Sequence[Species] | None = None,
        scheme: str = "central",
    ) -> GridChoice:
        """The grid with the missing parts chosen; given parts are kept (and a given
        dt that would be unstable on the chosen dx raises)."""
        choice = choose_grid(
            profile,
            conditions,
            orientation,
            initial,
            diffusivity,
            purpose=purpose,
            species=species,
            time_days=time_days,
            max_time_days=self.max_time_days,
            scheme=scheme,
        )
        grid = choice.grid
        if self.dx_um is not None:
            dx = actual_dx_um(profile, self.dx_um, initial)
            sp = _species_or_default(species, diffusivity, initial)
            grid = replace(grid, dx_um=self.dx_um, dt_s=stable_dt(dx, sp, conditions, orientation))
        if self.dt_s is not None:
            grid = replace(grid, dt_s=self.dt_s)
        check_grid(grid, profile, conditions, orientation, initial, diffusivity, species=species)
        length = choice.diffusion_length_um
        dx = actual_dx_um(profile, grid.dx_um, initial)
        sp = _species_or_default(species, diffusivity, initial)
        return replace(
            choice,
            grid=grid,
            resolution=length / dx,
            estimated_error=estimated_error(dx, length),
            stability_ratio=stability_ratio(dx, grid.dt_s, sp, conditions, orientation),
        )


def grid_diagnostics(result: FitResult) -> dict[str, float]:
    """How well a fit's grid resolved it: l / dx, the estimated error of the fitted time
    (percent, worst case) and the stability ratio D_max dt / dx^2."""
    inputs = result.inputs
    dx = float(result.model_x_um[1] - result.model_x_um[0])
    species = [(inputs["diffusivity"], fo_range(inputs["initial"]))]
    conditions, orientation = inputs["conditions"], inputs["orientation"]
    time = max(result.time_days, result.dt_s / SECONDS_PER_DAY)
    length = diffusion_length_um(time, species, conditions, orientation)
    return {
        "grid_resolution": length / dx,
        "grid_error_pct": 100 * estimated_error(dx, length),
        "grid_stability": stability_ratio(dx, result.dt_s, species, conditions, orientation),
    }
