"""Fit a diffusion model to one olivine Fo profile.

`fit_profile` is the user-facing form of `model_diffusion` from the round robin
script (``Round_Robin_Diffusion_Modeling Part 2.py``), taking its inputs as arguments
instead of looking them up by name in global spreadsheets. With the legacy
uncertainty method (whose default scheme is "central") it makes the same engine calls
in the same order (those of ``engine.Diffusion_call``); ``tests/test_differential.py``
checks that both give identical results. The default scheme is "implicit" (since
2026-10-13).

Units are in the names: micrometres (``_um``), seconds (``_s``), days (``_days``),
pascals (``_Pa``), degrees Celsius (``_C``). Fo is a fraction (0.88, not 88).
"""

from __future__ import annotations

import math
import os
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from forster import compiled, dipra, engine, implicit, orientation
from forster.boundary import (
    SECONDS_PER_DAY,
    BoundaryHistory,
    Rim,
    Stage,
    is_fixed,
    rim_at,
    settles_at_s,
)
from forster.compiled import Advance
from forster.diffusivity import (
    DEFAULT_DIFFUSIVITY,
    ArrheniusDiffusivity,
    Diffusivity,
    resolve,
)
from forster.uncertainty import (
    NoiseModel,
    chi2_per_step,
    fit_correlated,
    interpolation_matrix,
    points_in_domain,
    widen_log10,
)

FO2_BUFFERS = ("QFM", "FMQ", "NNO")
SCHEMES = ("central", "dipra", "implicit")  # finite-difference schemes, see fit_profile
EXPLICIT_SCHEMES = ("central", "dipra")
DEFAULT_SCHEME = "implicit"  # since 2026-10-13; "central" before
STEP_TOLERANCE = 1e-9  # implicit stages: a start this many steps past a step is on it


def default_scheme(uncertainty: str) -> str:
    """The scheme used when none is given: "implicit", or "central" with the legacy
    uncertainty method (which needs an explicit scheme and reproduces the old scripts)."""
    return "central" if uncertainty == "legacy" else DEFAULT_SCHEME


# How the time uncertainty is estimated, see fit_profile and docs/references/fit-uncertainty.md
UNCERTAINTY_METHODS = ("correlated", "legacy")
# Early stopping (correlated method): stop once the run is STOP_TIME_FACTOR times past the
# best fit so far and the sum of squared residuals is STOP_RSS_RATIO times its minimum and
# rising. If the 1-sigma range still reaches the end, the run is extended.
STOP_TIME_FACTOR = 2.0
STOP_RSS_RATIO = 4.0
STOP_BLOCKS = 100  # the run is advanced in about this many blocks of max_time_days


@dataclass(frozen=True)
class Profile:
    """A measured profile: distance from the crystal edge and Fo at each point."""

    distance_um: np.ndarray
    fo: np.ndarray
    name: str = "profile"

    def __post_init__(self) -> None:
        distance = np.asarray(self.distance_um, dtype=float)
        fo = np.asarray(self.fo, dtype=float)
        if distance.shape != fo.shape or distance.ndim != 1:
            raise ValueError(
                f"{self.name}: distance_um and fo must be 1-D arrays of the same length, "
                f"got shapes {distance.shape} and {fo.shape}"
            )
        if len(distance) < 3:
            raise ValueError(f"{self.name}: need at least 3 points, got {len(distance)}")
        if np.nanmax(fo) > 1.0:
            raise ValueError(
                f"{self.name}: Fo must be a fraction (0-1); got values up to {np.nanmax(fo)}"
            )
        object.__setattr__(self, "distance_um", distance)
        object.__setattr__(self, "fo", fo)


@dataclass(frozen=True)
class Orientation:
    """Angles (degrees) between the profile and the olivine [100], [010] and [001] axes."""

    alpha_deg: float
    beta_deg: float
    gamma_deg: float

    @classmethod
    def from_euler(
        cls, phi1_deg: float, Phi_deg: float, phi2_deg: float, profile_angle_deg: float
    ) -> Orientation:
        """From EBSD Euler angles (Bunge convention) and the profile direction.

        ``profile_angle_deg`` is the angle of the profile from the positive x axis
        (east) of the EBSD map.
        """
        alpha, beta, gamma = orientation.vector_direction(
            profile_angle_deg, phi1_deg, Phi_deg, phi2_deg
        )
        return cls(alpha, beta, gamma)


@dataclass(frozen=True)
class Conditions:
    """Temperature, pressure and oxygen fugacity during diffusion."""

    T_C: float
    P_Pa: float
    fO2_Pa: float

    @property
    def T_K(self) -> float:
        return self.T_C + 273.15

    @classmethod
    def from_buffer(
        cls, T_C: float, P_Pa: float, delta_log: float, buffer: str = "QFM"
    ) -> Conditions:
        """fO2 given as log units relative to a buffer (Frost 1991): QFM (= FMQ) or NNO."""
        if buffer.upper() not in FO2_BUFFERS:
            raise ValueError(f"unknown fO2 buffer {buffer!r}; use one of {', '.join(FO2_BUFFERS)}")
        fO2 = engine.fo2buffer(T_C + 273.15, P_Pa, delta_log, buffer.upper())
        return cls(T_C, P_Pa, float(fO2))


@dataclass(frozen=True)
class InitialCondition:
    """Step-function starting profile.

    The model runs between ``domain_um[0]`` and ``domain_um[1]`` (the round robin
    tables call these edge_x1 and edge_x2). Fo is held at ``edge_fo`` at the first
    end and ``core_fo`` at the second. ``steps_um`` are the positions of the steps;
    with one step the profile is ``edge_fo`` before it and ``core_fo`` after it.
    ``step_fo`` adds one intermediate plateau between two steps.
    """

    domain_um: tuple[float, float]
    edge_fo: float
    core_fo: float
    steps_um: tuple[float, ...]
    step_fo: float | None = None

    def __post_init__(self) -> None:
        low, high = self.domain_um
        if not low < high:
            raise ValueError(f"domain_um must be increasing, got {self.domain_um}")
        if not self.steps_um:
            raise ValueError("steps_um needs at least one step position")
        for fo in (self.edge_fo, self.core_fo, self.step_fo):
            if fo is not None and not 0.0 < fo <= 1.0:
                raise ValueError(f"Fo values must be fractions (0-1); got {fo}")
        if self.step_fo is not None and len(self.steps_um) != 2:
            raise ValueError("step_fo needs exactly two step positions (one on each side of it)")
        if self.step_fo is None and len(self.steps_um) != 1:
            raise ValueError(
                f"{len(self.steps_um)} step positions given: use one step, or two with "
                "step_fo for the plateau between them"
            )

    @property
    def fo_values(self) -> tuple[float, ...]:
        """Fo of each part of the step function, from the edge: edge, [plateau], core."""
        if self.step_fo is None:
            return (self.edge_fo, self.core_fo)
        return (self.edge_fo, self.step_fo, self.core_fo)


@dataclass(frozen=True)
class Grid:
    """Numerical settings: grid spacing, time step and the longest time to model."""

    dx_um: float
    dt_s: float
    max_time_days: float = 200.0

    @property
    def timesteps(self) -> int:
        # Same expression as model_diffusion, so float rounding matches it exactly.
        return int(self.max_time_days * 24 * 60 * 60 / self.dt_s)


@dataclass(frozen=True)
class KrigingSettings:
    """How the measured profile is interpolated onto the model grid.

    The kriged profile is always computed for plotting. Only the legacy
    uncertainty method fits to it: there the kriging variance, plus
    ``uniform_uncertainty`` (Fo fraction), is used as the data uncertainty. (It is
    used as a standard deviation although PyKrige returns a variance; kept that way
    so legacy results reproduce.) Parameters are passed to PyKrige.
    """

    variogram_model: str = "linear"
    variogram_parameters: dict[str, float] = field(
        default_factory=lambda: {"slope": 1e-4, "nugget": 2e-4}
    )
    uniform_uncertainty: float = 1e-5


@dataclass(frozen=True)
class FitResult:
    """Best-fit diffusion time with its uncertainty, and the curves behind it.

    ``curves`` holds the model profile at every time step (row ``i`` is at
    ``i * dt_s``); ``indices`` are the rows of the best fit and of the low and high
    ends of the uncertainty range.

    ``uncertainty`` names the method. "correlated": ``sum_chi2`` is the generalised
    chi-squared of the measured points (``fit_x_um``, ``fit_fo``) under the fitted
    ``noise`` model, and the range ends are where it crosses its minimum + 1,
    interpolated between time steps. "legacy": ``sum_chi2`` is the chi-squared of
    the kriged grid (``fit_x_um`` is the model grid) and the range is the old
    reduced chi-squared rule; ``noise`` is None.
    """

    name: str
    time_s: float
    time_minus_s: float
    time_plus_s: float
    indices: tuple[int, int, int]
    sum_chi2: np.ndarray
    model_x_um: np.ndarray
    curves: np.ndarray
    data_x_um: np.ndarray
    data_fo: np.ndarray
    interp_x_um: np.ndarray
    interp_fo: np.ndarray
    interp_fo_sigma: np.ndarray
    dt_s: float
    inputs: dict[str, Any]
    uncertainty: str = "legacy"
    fit_x_um: np.ndarray | None = None
    fit_fo: np.ndarray | None = None
    noise: NoiseModel | None = None
    # "Fe-Mg" for Fo fits; "Ni", "Mn" or "Ca" for `forster.elements.fit_element`, whose
    # curves and data (in the Fo-named fields) are that element's, in its own units.
    element: str = "Fe-Mg"
    # Start of the last growth stage (seconds after the first); 0 without stages.
    last_stage_s: float = 0.0
    # 1-sigma of log10 D from the diffusivity law's covariance, and the spread between
    # laboratories (`forster.diffusivity.LawUncertainty`), at the run's conditions; None
    # when the law has no covariance or the fit has growth stages (`time_range_days`).
    law_sigma_log10: float | None = None
    between_study_log10: float | None = None

    @property
    def time_days(self) -> float:
        return self.time_s / SECONDS_PER_DAY

    @property
    def time_since_last_stage_days(self) -> float:
        """The best fit measured from the start of the last growth stage (`fit_profile`'s
        ``stages``); equal to ``time_days`` without stages."""
        return (self.time_s - self.last_stage_s) / SECONDS_PER_DAY

    @property
    def time_minus_days(self) -> float:
        """Distance from the best fit to the low end of the range (negative or zero)."""
        return self.time_minus_s / SECONDS_PER_DAY

    @property
    def time_plus_days(self) -> float:
        return self.time_plus_s / SECONDS_PER_DAY

    def time_range_days(self, include: str = "fit") -> tuple[float, float]:
        """Low and high ends of the 1-sigma range of time (days).

        - "fit": from the profile alone (``time_minus_days``, ``time_plus_days``). For
          comparing crystals fitted with the same law, whose error is common to all.
        - "law": also the diffusivity law's calibration uncertainty.
        - "law_and_studies": also the spread between laboratories; for an absolute
          time, if the crystal may sit off the laboratories' average as one more
          experiment would.

        An Fo profile depends on D t, so an error in log10 D is the same error, with the
        sign changed, in log10 t; it is combined with the fit's range on each side in
        log10 (`forster.uncertainty.widen_log10`). Where that holds and where it does
        not: ``docs/references/law-uncertainty-limits.md``.
        """
        low = self.time_days + self.time_minus_days
        high = self.time_days + self.time_plus_days
        if include == "fit":
            return low, high
        if include not in ("law", "law_and_studies"):
            raise ValueError(f"include must be 'fit', 'law' or 'law_and_studies', not {include!r}")
        if self.law_sigma_log10 is None:
            raise ValueError(
                "no law uncertainty for this fit: its diffusivity has no covariance "
                "(see forster.diffusivity.LawUncertainty), or the fit has growth stages "
                "or a rim value that changes in time (docs/references/law-uncertainty-limits.md)"
            )
        sigma = self.law_sigma_log10
        if include == "law_and_studies":
            sigma = math.hypot(sigma, self.between_study_log10 or 0.0)
        return widen_log10(self.time_days, low, high, sigma)

    @property
    def reduced_chi2(self) -> np.ndarray:
        """``sum_chi2`` over (points fitted - 1). A rough guide only: for a nonlinear model
        the degrees of freedom are not known (Andrae et al. 2010). Under the
        correlated method it is close to 1 at the best fit by construction."""
        n = len(self.fit_x_um) if self.fit_x_um is not None else len(self.model_x_um)
        return self.sum_chi2 / (n - 1)

    @property
    def modelled_days(self) -> float:
        """Time the model was run to: ``max_time_days`` unless the run stopped early."""
        return (len(self.curves) - 1) * self.dt_s / SECONDS_PER_DAY

    @property
    def best_fit(self) -> np.ndarray:
        return self.curves[self.indices[0]]

    @property
    def initial(self) -> np.ndarray:
        return self.curves[0]


# advance(c, n, first_row=0): rows first_row .. first_row + n, the first being c.


def fe_mg_diffusivity(diffusivity: Diffusivity | str) -> Diffusivity:
    """The equation object for a name or object, checked to be for Fe-Mg."""
    diffusivity = resolve(diffusivity)
    if diffusivity.element != "Fe-Mg":
        raise ValueError(
            f"diffusivity {diffusivity.name!r} is for {diffusivity.element}; fit_profile models "
            "Fe-Mg (Fo) profiles and needs an Fe-Mg equation"
        )
    return diffusivity


def model_grid(
    x_um: np.ndarray, grid: Grid, domain_um: tuple[float, float]
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Even grid from 0 to the last measured point (as in model_diffusion), its spacing,
    and the grid rows nearest the two ends of the model domain."""
    num_x = int(round(abs(x_um.max() - x_um.min()) / grid.dx_um, 0))
    step_x, dx_um = np.linspace(0, x_um.max(), num_x, endpoint=True, retstep=True)
    low_idx = int(np.abs(step_x - domain_um[0]).argmin())
    high_idx = int(np.abs(step_x - domain_um[1]).argmin())
    return step_x, float(dx_um), (low_idx, high_idx)


# Largest array of model profiles (time steps x nodes, float64) a fit may allocate. A
# grid mistake once asked for tens of GB and froze the machine in swap; a fit now
# refuses instead. FORSTER_MAX_CURVES_GB in the environment raises or lowers it.
MAX_CURVES_GB = 4.0


def check_curve_memory(rows: int, nodes: int) -> None:
    """Raise ValueError if ``rows`` x ``nodes`` model profiles exceed the memory limit."""
    limit = float(os.environ.get("FORSTER_MAX_CURVES_GB", MAX_CURVES_GB))
    gb = rows * nodes * 8 / 1e9
    if gb > limit:
        raise ValueError(
            f"this grid would hold {rows:,} time steps x {nodes:,} nodes = {gb:.1f} GB of "
            f"model profiles (limit {limit:g} GB, set FORSTER_MAX_CURVES_GB to change it); "
            "use a larger dx or dt, or a shorter max_time_days"
        )


def scheme_grid(
    scheme: str,
    x_um: np.ndarray,
    grid: Grid,
    domain_um: tuple[float, float],
    rims_um: Sequence[float] = (),
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """`model_grid` for the explicit schemes. For "implicit", a grid with nodes on the
    domain's rim, on its core end (or the last measured point, if nearer) and on every
    stage rim in ``rims_um``, even in between with spacing at most ``grid.dx_um``
    (`forster.implicit.node_grid`); the spacing returned is the largest."""
    if scheme != "implicit":
        return model_grid(x_um, grid, domain_um)
    end = min(domain_um[1], float(x_um.max()))
    x = implicit.node_grid([domain_um[0], *rims_um, end], grid.dx_um)
    return x, float(np.diff(x).max()), (0, len(x) - 1)


def _diffusion_model(
    scheme: str,
    conditions: Conditions,
    orientation: Orientation,
    initial: InitialCondition,
    grid: Grid,
    diffusivity: Diffusivity,
    model_x: np.ndarray,
    dx_um: float,
    rim: Rim | None = None,
    fast: bool = True,
    substeps: int = 1,
) -> tuple[np.ndarray, Advance]:
    """The step-function start and a function advancing a profile by n time steps.

    ``advance(c, n, first_row=0)`` returns n + 1 rows, the first being ``c`` (row
    ``first_row`` of the run). The time steppers restart exactly where they stopped,
    so running in blocks gives the same numbers as one run. The calls are those of
    ``engine.Diffusion_call``, with the stepper chosen by ``scheme``: "central" is the
    engine's, "dipra" is Girona and Costa (2013) eq. 5, "implicit" is
    `forster.implicit` (cell-averaged start, values held on the end nodes). ``rim`` is
    the value held at the rim (default ``initial.edge_fo``); a function of time needs
    "implicit". ``fast``: use the compiled steppers when available (`forster.compiled`);
    False keeps NumPy (bit-identical to earlier versions). ``substeps`` (explicit
    schemes): each time step of ``grid.dt_s`` is that many stable steps.
    """
    D, dD_dfo = diffusivity.functions(conditions, orientation)
    rim = initial.edge_fo if rim is None else rim
    if scheme == "implicit":
        start = implicit.cell_average(model_x, initial.steps_um, list(initial.fo_values))
        return start, _implicit_stepper(D, rim, initial.core_fo, grid.dt_s, model_x, fast=fast)
    if not isinstance(rim, int | float):
        raise ValueError('a rim value that changes in time needs scheme="implicit"')
    start = engine.step_condition(model_x, list(initial.steps_um), list(initial.fo_values))
    bounds = (float(rim), initial.core_fo)
    return start, _stepper(scheme, D, dD_dfo, bounds, grid.dt_s, dx_um * 1e-6, fast, substeps)


def _implicit_stepper(
    D: Callable[[np.ndarray], np.ndarray],
    rim: Rim,
    core: float,
    dt_s: float,
    model_x: np.ndarray,
    t0_s: float = 0.0,
    fast: bool = True,
) -> Advance:
    """The implicit stepper: compiled when available and ``fast``, else NumPy."""
    advance = fast and compiled.implicit_stepper(D, rim, core, dt_s, model_x, t0_s=t0_s)
    return advance or implicit.stepper(D, rim, core, dt_s, model_x, t0_s=t0_s)


def _stepper(
    scheme: str,
    D: Callable[[np.ndarray], np.ndarray],
    dD_dfo: Callable[[np.ndarray], np.ndarray],
    bounds: tuple[float, float],
    dt_s: float,
    dx_m: float,
    fast: bool = True,
    substeps: int = 1,
) -> Advance:
    """``advance(c, n, first_row=0)``: n time steps with Fo held at ``bounds`` beyond
    both ends (``first_row`` is not needed: the bounds are fixed). ``fast``: compiled
    when available. Each time step is ``substeps`` steps of ``dt_s / substeps``, and
    only the rows at whole time steps are kept."""
    advance = fast and compiled.explicit_stepper(scheme, D, bounds, dt_s, dx_m, substeps)
    if advance:
        return advance
    step_s = dt_s / substeps

    def steps(c: np.ndarray, n: int) -> np.ndarray:
        if scheme == "central":
            return engine.timestepper(
                vector_c_in=c,
                vector_Fo_in=c,
                diffusivity_function=D,
                bounds_c=bounds,
                timesteps=n,
                dx=dx_m,
                dt=step_s,
            )
        return dipra.timestepper(c, D, dD_dfo, bounds, n, dt=step_s, dx=dx_m)

    def advance(c: np.ndarray, n: int, first_row: int = 0) -> np.ndarray:
        if substeps == 1:
            return steps(c, n)
        rows = [np.asarray(c, dtype=float)]
        for _ in range(n):
            rows.append(steps(rows[-1], substeps)[-1])
        return np.array(rows)

    return advance


def _check_staged_grid(
    last_start_row: int, grid: Grid, history: BoundaryHistory, firsts: Sequence[int]
) -> None:
    """The run must reach past the last stage, and the grid start at the outer rim."""
    if last_start_row >= grid.timesteps:
        raise ValueError(
            f"max_time_days ({grid.max_time_days}) must extend past the last stage "
            f"({history.stages[-1].start_s / SECONDS_PER_DAY:.4g} days)"
        )
    if firsts[-1] != 0:
        raise ValueError("the model grid must start at the outermost rim")


def _staged_prefix(
    scheme: str,
    conditions: Conditions,
    orientation: Orientation,
    initial: InitialCondition,
    grid: Grid,
    diffusivity: Diffusivity,
    model_x: np.ndarray,
    dx_um: float,
    history: BoundaryHistory,
) -> tuple[np.ndarray, np.ndarray, Advance, int]:
    """Run every growth stage but the last.

    Returns the rows from time zero up to the start of the last stage (NaN where the
    crystal had not grown yet), the state at that start (new zone filled in), the
    advance function of the last stage, and the row at which it starts. Each stage
    runs the same stepper on the crystal it has, with its rim value beyond the rim.
    """
    D, dD_dfo = diffusivity.functions(conditions, orientation)
    dx_m = dx_um * 1e-6
    stages = history.stages
    firsts, starts = history.first_nodes(model_x), history.start_rows(grid.dt_s)
    _check_staged_grid(starts[-1], grid, history, firsts)
    full_start, _ = _diffusion_model(
        scheme, conditions, orientation, initial, grid, diffusivity, model_x, dx_um
    )
    state = np.full(len(model_x), np.nan)
    state[firsts[0] :] = full_start[firsts[0] :]
    chunks = []
    for k in range(len(stages) - 1):
        bounds = (stages[k].held_value, initial.core_fo)
        advance = _stepper(scheme, D, dD_dfo, bounds, grid.dt_s, dx_m)
        rows = np.full((starts[k + 1] - starts[k] + 1, len(model_x)), np.nan)
        rows[:, firsts[k] :] = advance(
            state[firsts[k] :], starts[k + 1] - starts[k], first_row=starts[k]
        )
        chunks.append(rows[:-1])
        state = rows[-1].copy()
        if firsts[k + 1] < firsts[k]:
            state[firsts[k + 1] : firsts[k]] = stages[k + 1].zone_value
    last = _stepper(scheme, D, dD_dfo, (stages[-1].held_value, initial.core_fo), grid.dt_s, dx_m)
    return np.concatenate(chunks), state, last, starts[-1]


def _grow(state: np.ndarray, model_x: np.ndarray, old: int, new: int, zone_value: float) -> None:
    """Fill the new zone, nodes ``new`` .. ``old``; the node on the old rim takes the
    average over its cell (outer part new zone, inner part old crystal)."""
    if new == old:
        return
    state[new:old] = zone_value
    if old < len(model_x) - 1:
        outer, inner = model_x[old] - model_x[old - 1], model_x[old + 1] - model_x[old]
        state[old] = (outer * zone_value + inner * state[old]) / (outer + inner)


def _implicit_staged_prefix(
    conditions: Conditions,
    orientation: Orientation,
    initial: InitialCondition,
    grid: Grid,
    diffusivity: Diffusivity,
    model_x: np.ndarray,
    history: BoundaryHistory,
) -> tuple[np.ndarray, np.ndarray, Advance, int]:
    """`_staged_prefix` for the implicit scheme, with every stage starting at its exact
    time: a time step that crosses a stage start is split there. ``model_x`` has a node
    on every stage rim (`scheme_grid`). The last stage starts from the first row at or
    after its start time."""
    D, _ = diffusivity.functions(conditions, orientation)
    stages, dt = history.stages, grid.dt_s
    firsts = history.first_nodes(model_x)
    # A start within STEP_TOLERANCE steps after a step counts as on it, here and in
    # the loop below (floating-point products such as 6.37 * 86400 can sit one ulp off).
    offset = math.ceil(stages[-1].start_s / dt - STEP_TOLERANCE)
    _check_staged_grid(offset, grid, history, firsts)
    state = np.full(len(model_x), np.nan)
    state[firsts[0] :] = implicit.cell_average(
        model_x[firsts[0] :], initial.steps_um, list(initial.fo_values)
    )
    state[firsts[0]] = rim_at(stages[0].rim, 0.0)
    state[-1] = initial.core_fo
    rows = np.full((offset, len(model_x)), np.nan)
    t, k = 0.0, 0

    def advance_to(t_end: float) -> None:
        nonlocal t
        if t_end > t:
            rim, active = stages[k].rim, slice(firsts[k], None)
            geometry = implicit.grid_geometry(model_x[active])
            mid = (rim_at(rim, t + implicit.GAMMA * (t_end - t)), initial.core_fo)
            end = (rim_at(rim, t_end), initial.core_fo)
            state[active] = implicit.tr_bdf2_step(state[active], D, t_end - t, geometry, mid, end)
            t = t_end

    for j in range(offset + 1):
        while k + 1 < len(stages) and stages[k + 1].start_s / dt - STEP_TOLERANCE <= j:
            advance_to(stages[k + 1].start_s)
            k += 1
            _grow(state, model_x, firsts[k - 1], firsts[k], stages[k].zone_value or 0.0)
            state[firsts[k]] = rim_at(stages[k].rim, t)
        advance_to(j * dt)
        if j < offset:
            rows[j] = state
    rim, core = stages[-1].rim, initial.core_fo
    last = _implicit_stepper(D, rim, core, dt, model_x, t0_s=offset * dt)
    return rows, state.copy(), last, offset


def _run_past_best_fit(
    start: np.ndarray,
    advance: Advance,
    total_steps: int,
    W: np.ndarray | None,
    y: np.ndarray | None,
    *,
    rss: Callable[[np.ndarray], np.ndarray] | None = None,
    min_steps: int = 0,
) -> np.ndarray:
    """Profiles from time zero until the fit is well behind (or ``total_steps``).

    Advances in blocks and stops when the run is ``STOP_TIME_FACTOR`` times past the
    lowest sum of squared residuals so far, the sum is ``STOP_RSS_RATIO`` times that
    lowest value, and it is still rising. The sum is that of ``y - W @ row``, or
    ``rss(rows)`` when given (`forster.elements`, where concentrations are fitted).
    It never stops before ``min_steps`` (a rim value still changing could bring a
    second, better minimum).
    """
    if rss is None:
        if W is None or y is None:
            raise ValueError("give W and y, or rss")
        weights, data = W, y

        def rss(rows: np.ndarray) -> np.ndarray:
            return chi2_per_step(rows, weights, data)

    block = max(500, -(-total_steps // STOP_BLOCKS))
    chunks = [start[None, :]]
    rss_min = float(rss(start[None, :])[0])
    k_min = done = 0
    c = start
    while done < total_steps:
        rows = advance(c, min(block, total_steps - done), first_row=done)[1:]
        chunks.append(rows)
        c = rows[-1]
        sums = rss(rows)
        i = int(np.argmin(sums))
        if sums[i] < rss_min:
            rss_min, k_min = float(sums[i]), done + 1 + i
        done += len(rows)
        past = done >= max(STOP_TIME_FACTOR * max(k_min, 1), min_steps)
        if past and sums[-1] >= STOP_RSS_RATIO * rss_min and sums[-1] >= sums[0]:
            break
    return np.concatenate(chunks)


def _fit_extending(
    rows: np.ndarray,
    advance: Advance,
    total_steps: int,
    fit: Callable[[np.ndarray, bool], Any],
) -> tuple[np.ndarray, Any]:
    """``fit(rows, complete)``, extending the run while the 1-sigma range reaches its end.
    Warnings are raised only for the fit that is kept."""
    while True:
        complete = len(rows) - 1 >= total_steps
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = fit(rows, complete)
        if complete or result.interval.high_row < len(rows) - 1:
            for w in caught:
                warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
            return rows, result
        extra = min(len(rows) - 1, total_steps - (len(rows) - 1))
        rows = np.concatenate([rows, advance(rows[-1], extra, first_row=len(rows) - 1)[1:]])


def _correlated_run(
    start: np.ndarray,
    advance: Advance,
    total_steps: int,
    model_x: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    stop_early: bool,
    min_steps: int = 0,
):
    """Model rows from ``start`` and their correlated fit (`fit_correlated`)."""
    if stop_early:
        px, py = points_in_domain(model_x, x, y)
        W = interpolation_matrix(model_x, px)
        curves = _run_past_best_fit(start, advance, total_steps, W, py, min_steps=min_steps)
    else:
        curves = advance(start, total_steps)
    return _fit_extending(
        curves,
        advance,
        total_steps,
        lambda rows, complete: fit_correlated(rows, model_x, x, y, end_is_limit=complete),
    )


def _explicit_substeps(
    diffusivity: Diffusivity,
    conditions: Conditions,
    orientation: Orientation,
    initial: InitialCondition,
    history: BoundaryHistory | None,
    rim_fo: Rim | None,
    grid: Grid,
    dx_um: float,
) -> int:
    """Explicit schemes: the stable steps per time step of ``grid.dt_s``, from D_max dt /
    dx^2 with D_max over every Fo the run holds (initial condition, each stage's zone and
    rim, a fixed ``rim_fo``). 1 when that ratio is within the stability limit (then the
    run is exactly as before); otherwise enough steps to bring it to `STABILITY_RATIO`,
    so ``grid.dt_s`` only sets which rows are kept (the fit's time resolution). Runs
    with growth stages need a stable ``grid.dt_s`` (their stage starts fall on its
    steps) and raise instead."""
    from forster.grid import STABILITY_RATIO, UNSTABLE_RATIO

    values = list(initial.fo_values)
    values += [float(rim_fo)] if isinstance(rim_fo, int | float) else []
    for stage in history.stages if history is not None else ():
        values += [stage.zone_value] if stage.zone_value is not None else []
        values += [stage.held_value]
    D, _ = diffusivity.functions(conditions, orientation)
    ratio = float(np.max(D(np.array(values, dtype=float)))) * grid.dt_s / (dx_um * 1e-6) ** 2
    if ratio <= UNSTABLE_RATIO:
        return 1
    if history is None:
        return math.ceil(ratio / STABILITY_RATIO)
    raise ValueError(
        f"unstable grid for growth stages: D_max dt / dx^2 = {ratio:.2f} > {UNSTABLE_RATIO} "
        f"over Fo {min(values):.3f}-{max(values):.3f} (dx {dx_um:.4g} um, dt "
        f'{grid.dt_s:.4g} s); use a smaller dt or scheme="implicit"'
    )


def _check_options(
    scheme: str, uncertainty: str, stages: Sequence[Stage], rim_fo: Rim | None
) -> None:
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}; use one of {', '.join(SCHEMES)}")
    if uncertainty not in UNCERTAINTY_METHODS:
        raise ValueError(
            f"unknown uncertainty method {uncertainty!r}; use one of "
            f"{', '.join(UNCERTAINTY_METHODS)}"
        )
    if uncertainty == "legacy" and stages:
        raise ValueError("growth stages need the correlated uncertainty method")
    if uncertainty == "legacy" and scheme == "implicit":
        raise ValueError("the implicit scheme needs the correlated uncertainty method")
    if rim_fo is not None and not is_fixed(rim_fo) and scheme != "implicit":
        raise ValueError('a rim value that changes in time needs scheme="implicit"')


def fit_profile(
    profile: Profile,
    conditions: Conditions,
    orientation: Orientation,
    initial: InitialCondition,
    grid: Grid,
    kriging: KrigingSettings | None = None,
    diffusivity: Diffusivity | str = DEFAULT_DIFFUSIVITY,
    scheme: str | None = None,
    uncertainty: str = "correlated",
    stop_early: bool = True,
    stages: Sequence[Stage] = (),
    rim_fo: Rim | None = None,
) -> FitResult:
    """Fit Fe-Mg diffusion from a step-function start to a measured Fo profile.

    The profile is interpolated by kriging onto an even grid from 0 to its largest
    distance, the model runs forward from ``initial`` for ``grid.max_time_days``,
    and the time step with the smallest chi-squared is the best fit.

    ``scheme`` picks the finite-difference scheme: "implicit" (the default since
    2026-10-13; "central" with the legacy uncertainty method), "central" (the engine's,
    central differences in the cross term) or "dipra" (Girona and Costa 2013,
    eq. 5: forward differences and the analytic dD/dX_Fo). Both solve the same
    equation and converge to the same answer as ``grid.dx_um`` shrinks; running
    both is a check on the discretization. Both hold the rim and core values one
    node outside the profile and move each step of ``initial`` to the nearest node,
    which makes the fitted time first order in dx (``docs/references/grid-choice.md``).
    "implicit" (`forster.implicit`, since 2026-10-13) holds them on the end nodes
    (the rim exactly at ``domain_um[0]``), averages the start over each node's cell
    instead of moving its steps, and steps in time with TR-BDF2. On 35 real profiles
    its chosen grid (dx = l / 8, l = sqrt(D t); 400 steps up to the fit) gave times
    within 0.3% of a converged reference, where the explicit grid then chosen
    (dx = l / 65; l / 240 since) was off by up to 3.6%. The same inputs put the rim
    one node apart in the two kinds of scheme, so at coarse grids they give slightly
    different fits; both converge to the same answer. Correlated uncertainty only.

    ``diffusivity`` is the Fe-Mg diffusivity equation: a preset name from
    `forster.diffusivity.DIFFUSIVITY_PRESETS` or an equation object (an
    `ArrheniusDiffusivity`, or anything with the `Diffusivity` attributes). The
    default, "dohmen_chakraborty2007", is the equation DIPRA uses (Girona and Costa
    2013, Table 1). ``docs/references/diffusivity-literature.md`` compares the
    presets and newer work.

    ``uncertainty`` picks how the best fit and its 1-sigma range are found:

    - "correlated" (default since 2026-10-12): each model curve is interpolated to
      the measured points, whose errors are modelled as white noise plus a smooth
      correlated misfit fitted by maximum likelihood; the range is where that
      chi-squared rises by `forster.uncertainty.interval_threshold` (1 for many
      points, more for few: the Student-t small-sample correction, since
      2026-10-13). See `forster.uncertainty`.
    - "legacy": the round robin and paper-era rule, ``engine.Best_fit_Chi2`` on
      the kriged grid (reduced chi-squared within 1 of its minimum, which is
      chi-squared within N - 1 for N grid points). Gives results identical to the
      old scripts.

    With "correlated" and ``stop_early`` (the default) the model stops once the fit
    is well behind it (see `STOP_TIME_FACTOR`, `STOP_RSS_RATIO`) instead of running to
    ``grid.max_time_days``, and is extended if the 1-sigma range reaches the end;
    ``FitResult.curves`` then ends at ``FitResult.modelled_days``. The fit is the same
    as with the full run. The legacy method always runs to ``max_time_days``.

    ``stages`` (`forster.boundary.Stage`) are instantaneous growth steps after the
    start: at each stage's start the rim moves outward and the new zone appears, then
    diffuses with the older crystal. ``initial`` is the crystal before the first of
    them. Only times after the last stage are fitted (the measured profile includes
    every zone); ``time_days`` counts from the start, ``time_since_last_stage_days``
    from the last stage. Correlated uncertainty only. With "implicit" each stage starts
    at its exact time and the grid has a node on every stage rim.

    ``rim_fo`` is the Fo held at the rim (default ``initial.edge_fo``, which also
    remains the start profile's edge value): a number for a fixed rim, or for a rim
    that changes in time (needs "implicit") a `forster.boundary.RimPath` or any
    function of the time in seconds since the start. Early stopping waits until a
    `RimPath` has stopped changing; with any other function the model runs to
    ``max_time_days``. With ``stages``, each stage's own ``rim_value`` may change in
    time the same way.

    ``docs/references/likelihood-method.md`` explains the method and its validation,
    ``docs/references/fit-uncertainty.md`` the change from earlier versions.
    """
    scheme = scheme or default_scheme(uncertainty)
    _check_options(scheme, uncertainty, stages, rim_fo)
    diffusivity = fe_mg_diffusivity(diffusivity)
    kriging = kriging or KrigingSettings()
    history = BoundaryHistory.from_initial(initial, stages, rim_fo) if stages else None
    domain, rims = initial.domain_um, ()
    if history is not None:
        domain = (history.outer_rim_um, initial.domain_um[1])
        rims = tuple(s.rim_um for s in history.stages)
    x, y = profile.distance_um, profile.fo
    step_x, dx_um, (low_idx, high_idx) = scheme_grid(scheme, x, grid, domain, rims)
    model_x = step_x[low_idx : high_idx + 1]
    check_curve_memory(grid.timesteps + 1, len(model_x))
    substeps = 1
    if scheme in EXPLICIT_SCHEMES:
        substeps = _explicit_substeps(
            diffusivity, conditions, orientation, initial, history, rim_fo, grid, dx_um
        )
    # Only the legacy method fits a kriged profile; the correlated method fits the
    # measured points, so it skips kriging (and leaves the kriged fields empty).
    interp_x = interp_fo = interp_sigma = np.empty(0)
    if uncertainty == "legacy":
        interp_x, interp_fo, interp_sigma = engine.Krige_Interpolate(
            x,
            y,
            step_x,
            variogram_model=kriging.variogram_model,
            variogram_parameters=kriging.variogram_parameters,
        )
    data_interp = interp_fo[low_idx : high_idx + 1]
    sigma_interp = interp_sigma[low_idx : high_idx + 1] + kriging.uniform_uncertainty

    # The legacy method keeps the NumPy steppers: it reproduces the old scripts exactly.
    prefix, offset, last_stage_s = None, 0, 0.0
    if history is None:
        start, advance = _diffusion_model(
            scheme,
            conditions,
            orientation,
            initial,
            grid,
            diffusivity,
            model_x,
            dx_um,
            rim_fo,
            fast=uncertainty != "legacy",
            substeps=substeps,
        )
    elif scheme == "implicit":
        prefix, start, advance, offset = _implicit_staged_prefix(
            conditions, orientation, initial, grid, diffusivity, model_x, history
        )
        last_stage_s = history.stages[-1].start_s
    else:
        prefix, start, advance, offset = _staged_prefix(
            scheme, conditions, orientation, initial, grid, diffusivity, model_x, dx_um, history
        )
        last_stage_s = offset * grid.dt_s
    last_rim = (
        history.stages[-1].rim
        if history is not None
        else initial.edge_fo
        if rim_fo is None
        else rim_fo
    )
    settle_s = settles_at_s(last_rim)
    min_steps = 0 if settle_s is None else max(0, math.ceil(settle_s / grid.dt_s) - offset)
    noise = None
    if uncertainty == "legacy":
        curves = advance(start, grid.timesteps)
        time, indices, sum_chi2 = engine.Best_fit_Chi2(
            curves, data_interp, sigma_interp, grid.dt_s, scale_error=False
        )
        fit_x, fit_fo = model_x, data_interp
    else:
        curves, fit = _correlated_run(
            start,
            advance,
            grid.timesteps - offset,
            model_x,
            x,
            y,
            stop_early and settle_s is not None,
            min_steps,
        )
        span = fit.interval
        time = (
            (offset + span.best) * grid.dt_s,
            (span.low - span.best) * grid.dt_s,
            (span.high - span.best) * grid.dt_s,
        )
        indices = (offset + span.best, offset + span.low_row, offset + span.high_row)
        sum_chi2, noise, fit_x, fit_fo = fit.chi2, fit.noise, fit.x_um, fit.fo
        if prefix is not None:
            curves = np.concatenate([prefix, curves])
            sum_chi2 = np.concatenate([np.full(offset, np.nan), sum_chi2])
    law_sigma, between = _law_uncertainty(
        diffusivity, conditions, orientation, initial, stages, rim_fo
    )
    return FitResult(
        name=profile.name,
        time_s=float(time[0]),
        time_minus_s=float(time[1]),
        time_plus_s=float(time[2]),
        indices=(int(indices[0]), int(indices[1]), int(indices[2])),
        sum_chi2=sum_chi2,
        model_x_um=model_x,
        curves=curves,
        data_x_um=x,
        data_fo=y,
        interp_x_um=interp_x,
        interp_fo=interp_fo,
        interp_fo_sigma=interp_sigma + kriging.uniform_uncertainty,
        dt_s=grid.dt_s,
        inputs={
            "conditions": conditions,
            "orientation": orientation,
            "initial": initial,
            "grid": grid,
            "kriging": kriging,
            "diffusivity": diffusivity,
            "scheme": scheme,
            "uncertainty": uncertainty,
            "stop_early": stop_early,
            "stages": tuple(stages),
            "rim_fo": rim_fo,
        },
        uncertainty=uncertainty,
        fit_x_um=fit_x,
        fit_fo=fit_fo,
        noise=noise,
        last_stage_s=last_stage_s,
        law_sigma_log10=law_sigma,
        between_study_log10=between,
    )


def _law_uncertainty(
    diffusivity: Diffusivity,
    conditions: Conditions,
    orientation: Orientation,
    initial: InitialCondition,
    stages: Sequence[Stage],
    rim_fo: Rim | None = None,
) -> tuple[float | None, float | None]:
    """The law's 1-sigma in log10 D and the between-study spread, for `FitResult`.

    Evaluated a third of the way from the edge Fo to the core Fo. The composition term
    changes D's shape as well as its level; on fits over Fo 0.80-0.90 (either way
    round) and 0.85-0.88 a change dn in its coefficient moved log10 time by
    -dn (fo_ref - X) with X there, to 0.0011 log for dn = 0.3 (the midpoint was off
    by up to 0.005). The choice hardly matters for `femg_refit2026`: its sigma varies by
    0.0005 log over X = 0.80-0.90. None with growth stages or a rim value that changes
    in time: their times are fixed in seconds, so the fitted time no longer scales as
    1/D (measured: a stage or rim change halfway through moved log10 t by 0.6 of the
    change in log10 D; ``docs/references/law-uncertainty-limits.md``).
    """
    if not isinstance(diffusivity, ArrheniusDiffusivity) or diffusivity.uncertainty is None:
        return None, None
    if stages or not (rim_fo is None or is_fixed(rim_fo)):
        return None, None
    x_fo = initial.edge_fo + (initial.core_fo - initial.edge_fo) / 3
    sigma = diffusivity.log10_sigma(conditions, orientation, x_fo)
    return sigma, diffusivity.uncertainty.between_study_log10
