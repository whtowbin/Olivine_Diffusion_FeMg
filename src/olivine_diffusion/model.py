"""Fit a diffusion model to one olivine Fo profile.

`fit_profile` is the user-facing form of `model_diffusion` from the round robin
script (``Round_Robin_Diffusion_Modeling Part 2.py``). With the default scheme it
makes the same engine calls in the same order (those of ``engine.Diffusion_call``),
but takes its inputs as arguments instead of looking them up by name in global
spreadsheets. ``tests/test_differential.py`` checks that both give identical
results.

Units are in the names: micrometres (``_um``), seconds (``_s``), days (``_days``),
pascals (``_Pa``), degrees Celsius (``_C``). Fo is a fraction (0.88, not 88).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from olivine_diffusion import dipra, engine, orientation

SECONDS_PER_DAY = 24 * 60 * 60
FE_MG_ACTIVATION_ENERGY_J_MOL = 201_000.0  # Dohmen and Chakraborty (2007)
FO2_BUFFERS = ("QFM", "FMQ", "NNO")
SCHEMES = ("central", "dipra")  # finite-difference schemes, see fit_profile


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

    The kriging variance, plus ``uniform_uncertainty`` (Fo fraction), is used as the
    data uncertainty in the chi-squared fit. Parameters are passed to PyKrige.
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

    @property
    def time_days(self) -> float:
        return self.time_s / SECONDS_PER_DAY

    @property
    def time_minus_days(self) -> float:
        """Distance from the best fit to the low end of the range (negative or zero)."""
        return self.time_minus_s / SECONDS_PER_DAY

    @property
    def time_plus_days(self) -> float:
        return self.time_plus_s / SECONDS_PER_DAY

    @property
    def reduced_chi2(self) -> np.ndarray:
        return self.sum_chi2 / (len(self.model_x_um) - 1)

    @property
    def best_fit(self) -> np.ndarray:
        return self.curves[self.indices[0]]

    @property
    def initial(self) -> np.ndarray:
        return self.curves[0]


def _run_diffusion(
    scheme: str,
    conditions: Conditions,
    orientation: Orientation,
    initial: InitialCondition,
    grid: Grid,
    activation_energy_J_mol: float,
    model_x: np.ndarray,
    dx_um: float,
) -> np.ndarray:
    """Model Fo profiles at every time step, starting from the step-function profile.

    The same calls as ``engine.Diffusion_call``, with the time stepper chosen by
    ``scheme``: "central" is the engine's, "dipra" is Girona and Costa (2013) eq. 5.
    """
    intervals = [initial.edge_fo, initial.core_fo]
    if initial.step_fo is not None:
        intervals[1:1] = [initial.step_fo]
    start = engine.step_condition(model_x, list(initial.steps_um), intervals)
    bounds = (initial.edge_fo, initial.core_fo)
    dx_m = dx_um * 1e-6
    if scheme == "central":
        diffusivity = engine.D_Fo(
            T=conditions.T_K,
            P=conditions.P_Pa,
            fO2=conditions.fO2_Pa,
            alpha=orientation.alpha_deg,
            beta=orientation.beta_deg,
            gamma=orientation.gamma_deg,
            EFo=activation_energy_J_mol,
        )
        return engine.timestepper(
            vector_c_in=start,
            vector_Fo_in=start,
            diffusivity_function=diffusivity,
            bounds_c=bounds,
            timesteps=grid.timesteps,
            dx=dx_m,
            dt=grid.dt_s,
        )
    diffusivity, derivative = dipra.D_Fo_and_derivative(
        conditions.T_K,
        conditions.P_Pa,
        conditions.fO2_Pa,
        orientation.alpha_deg,
        orientation.beta_deg,
        orientation.gamma_deg,
        activation_energy_J_mol,
    )
    return dipra.timestepper(
        start, diffusivity, derivative, bounds, grid.timesteps, dt=grid.dt_s, dx=dx_m
    )


def fit_profile(
    profile: Profile,
    conditions: Conditions,
    orientation: Orientation,
    initial: InitialCondition,
    grid: Grid,
    kriging: KrigingSettings | None = None,
    activation_energy_J_mol: float = FE_MG_ACTIVATION_ENERGY_J_MOL,
    scheme: str = "central",
) -> FitResult:
    """Fit Fe-Mg diffusion from a step-function start to a measured Fo profile.

    The profile is interpolated by kriging onto an even grid from 0 to its largest
    distance, the model runs forward from ``initial`` for ``grid.max_time_days``,
    and the time step with the smallest chi-squared is the best fit.

    ``scheme`` picks the finite-difference scheme: "central" (the engine's,
    central differences in the cross term) or "dipra" (Girona and Costa 2013,
    eq. 5: forward differences and the analytic dD/dX_Fo). Both solve the same
    equation and converge to the same answer as ``grid.dx_um`` shrinks; running
    both is a check on the discretization.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}; use one of {', '.join(SCHEMES)}")
    kriging = kriging or KrigingSettings()
    x, y = profile.distance_um, profile.fo

    # Even grid from 0 to the last measured point (as in model_diffusion).
    num_x = int(round(abs(x.max() - x.min()) / grid.dx_um, 0))
    step_x, dx_um = np.linspace(0, x.max(), num_x, endpoint=True, retstep=True)
    interp_x, interp_fo, interp_sigma = engine.Krige_Interpolate(
        x,
        y,
        step_x,
        variogram_model=kriging.variogram_model,
        variogram_parameters=kriging.variogram_parameters,
    )

    low_idx = int(np.abs(step_x - initial.domain_um[0]).argmin())
    high_idx = int(np.abs(step_x - initial.domain_um[1]).argmin())
    model_x = step_x[low_idx : high_idx + 1]
    data_interp = interp_fo[low_idx : high_idx + 1]
    sigma_interp = interp_sigma[low_idx : high_idx + 1] + kriging.uniform_uncertainty

    curves = _run_diffusion(
        scheme, conditions, orientation, initial, grid, activation_energy_J_mol, model_x, dx_um
    )
    time, indices, sum_chi2 = engine.Best_fit_Chi2(
        curves, data_interp, sigma_interp, grid.dt_s, scale_error=False
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
            "activation_energy_J_mol": activation_energy_J_mol,
            "scheme": scheme,
        },
    )
