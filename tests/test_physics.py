"""The engine against solutions it does not share code with.

- Constant diffusivity: the error-function solution for a step in an infinite
  medium, C = mean + half-step * erf((x - x0) / (2 sqrt(D t))) (Crank 1975, The
  Mathematics of Diffusion, chapter 2).
- Diffusivity that depends on Fo: an independent conservative scheme that puts D on
  the cell faces, dC/dt = d/dx(D dC/dx). The engine expands the same equation as
  D d2C/dx2 + dD/dx dC/dx; before 2026-10-11 its cross term was twice too large,
  which these tests catch (8.6e-3 Fo off, against 1.4e-4 after the fix).
"""

import numpy as np
import pytest
from scipy.special import erf

from forster import (
    Conditions,
    Grid,
    InitialCondition,
    KrigingSettings,
    Orientation,
    Profile,
    dipra,
    engine,
    fit_profile,
)

DX_UM = 1.0
DX_M = DX_UM * 1e-6
X_UM = np.arange(0, 300, DX_UM)
STEP_UM = 150
EDGE_FO, CORE_FO = 0.80, 0.90
# Round robin conditions (1200 C, 42 MPa, fO2 1.38e-3 Pa); profile along [001].
CONDITIONS = Conditions(T_C=1200, P_Pa=4.2e7, fO2_Pa=0.00138)
D_FO = engine.D_Fo(CONDITIONS.T_K, CONDITIONS.P_Pa, CONDITIONS.fO2_Pa, 90, 90, 0)


def step_profile(x_um):
    return np.where(x_um < STEP_UM, EDGE_FO, CORE_FO)


def flux_form(c0, diffusivity, t_end_s, dx_m, safety=0.05):
    """Explicit conservative scheme with face diffusivities and fixed end values."""
    d_max = diffusivity(np.array([c0.min(), c0.max()])).max()
    n = int(np.ceil(t_end_s / (safety * dx_m**2 / d_max)))
    dt = t_end_s / n
    c = c0.copy()
    for _ in range(n):
        full = np.concatenate([[c0[0]], c, [c0[-1]]])
        d_face = 0.5 * (diffusivity(full[:-1]) + diffusivity(full[1:]))
        c = c + dt * np.diff(d_face * np.diff(full) / dx_m) / dx_m
    return c


def engine_run(c0, diffusivity, steps, dt_s):
    return engine.timestepper(
        vector_c_in=c0,
        vector_Fo_in=c0,
        diffusivity_function=diffusivity,
        bounds_c=(c0[0], c0[-1]),
        timesteps=steps,
        dt=dt_s,
        dx=DX_M,
    )


def test_constant_diffusivity_matches_erf_solution():
    d0 = 1e-16
    dt = 0.2 * DX_M**2 / d0
    steps = 2000
    curves = engine_run(step_profile(X_UM), lambda c: np.full_like(c, d0, dtype=float), steps, dt)
    t = steps * dt
    exact = (EDGE_FO + CORE_FO) / 2 + (CORE_FO - EDGE_FO) / 2 * erf(
        (X_UM - (STEP_UM - 0.5)) * 1e-6 / (2 * np.sqrt(d0 * t))
    )
    assert np.abs(curves[-1] - exact).max() < 1e-5


def test_fo_dependent_diffusivity_matches_conservative_scheme():
    dt = 0.2 * DX_M**2 / D_FO(np.array([EDGE_FO])).item()
    steps = 3000
    c0 = step_profile(X_UM)
    engine_out = engine_run(c0, D_FO, steps, dt)[-1]
    reference = flux_form(c0, D_FO, steps * dt, DX_M)
    # The reference's own grid error is about 3.5e-4 Fo (dx 1 vs 0.5 um).
    assert np.abs(engine_out - reference).max() < 5e-4


@pytest.mark.parametrize("scheme", ["central", "dipra"])
def test_fit_recovers_the_time_of_a_known_profile(scheme):
    """A profile made by the conservative scheme at a known time is fitted back."""
    dt = 0.2 * DX_M**2 / D_FO(np.array([EDGE_FO])).item()
    true_steps = 1500
    truth = flux_form(step_profile(X_UM), D_FO, true_steps * dt, DX_M)
    sampled = slice(0, None, 3)  # a measurement every 3 um
    result = fit_profile(
        Profile(X_UM[sampled], truth[sampled], name="synthetic"),
        CONDITIONS,
        Orientation(90, 90, 0),
        InitialCondition(domain_um=(0, 297), edge_fo=EDGE_FO, core_fo=CORE_FO, steps_um=(STEP_UM,)),
        Grid(dx_um=DX_UM, dt_s=dt, max_time_days=2 * true_steps * dt / 86400),
        KrigingSettings(variogram_parameters={"slope": 1e-4, "nugget": 1e-7}),
        scheme=scheme,
        # The profile is noiseless, so the default (correlated) range only reflects the
        # ~1e-4 Fo grid error and is +-0.6%: it rightly excludes the 1.5% discretization
        # bias. This test is about the solver; the uncertainty method is tested for
        # coverage in test_uncertainty.py.
        uncertainty="legacy",
    )
    # Measured at dx = 1 um: central 1.017, dipra 1.022 (dipra's forward-difference cross
    # term is first order); at dx = 0.5 um both are within 0.7%. The bound allows that grid
    # error; the range check below is the stricter requirement.
    assert abs(result.indices[0] / true_steps - 1) < 0.03
    assert result.indices[1] <= true_steps <= result.indices[2]


def dipra_run(c0, steps, dt_s, dx_m):
    diffusivity, derivative = dipra.D_Fo_and_derivative(
        CONDITIONS.T_K, CONDITIONS.P_Pa, CONDITIONS.fO2_Pa, 90, 90, 0, 201_000
    )
    return dipra.timestepper(c0, diffusivity, derivative, (c0[0], c0[-1]), steps, dt=dt_s, dx=dx_m)


def test_dipra_derivative_is_the_derivative_of_D():
    diffusivity, derivative = dipra.D_Fo_and_derivative(
        CONDITIONS.T_K, CONDITIONS.P_Pa, CONDITIONS.fO2_Pa, 90, 90, 0, 201_000
    )
    fo = np.linspace(0.6, 0.95, 8)
    h = 1e-6
    numerical = (diffusivity(fo + h) - diffusivity(fo - h)) / (2 * h)
    np.testing.assert_allclose(derivative(fo), numerical, rtol=1e-6)


def test_dipra_scheme_matches_conservative_scheme():
    """Girona and Costa (2013) eq. 5, run as published, against the flux-form reference."""
    dt = 0.2 * DX_M**2 / D_FO(np.array([EDGE_FO])).item()
    steps = 3000
    c0 = step_profile(X_UM)
    out = dipra_run(c0, steps, dt, DX_M)[-1]
    assert np.abs(out - flux_form(c0, D_FO, steps * dt, DX_M)).max() < 5e-4


def test_central_and_dipra_schemes_agree_on_a_fine_grid():
    """Same equation, different cross-term differences: they converge to the same answer."""
    dx_um = 0.5
    dx_m = dx_um * 1e-6
    x = np.arange(0, 300, dx_um)
    c0 = step_profile(x)
    dt = 0.2 * dx_m**2 / D_FO(np.array([EDGE_FO])).item()
    steps = round(3000 * (DX_M / dx_m) ** 2)  # same physical time as the dx = 1 um tests
    central = engine.timestepper(
        vector_c_in=c0,
        vector_Fo_in=c0,
        diffusivity_function=D_FO,
        bounds_c=(c0[0], c0[-1]),
        timesteps=steps,
        dt=dt,
        dx=dx_m,
    )[-1]
    assert np.abs(central - dipra_run(c0, steps, dt, dx_m)[-1]).max() < 5e-4
