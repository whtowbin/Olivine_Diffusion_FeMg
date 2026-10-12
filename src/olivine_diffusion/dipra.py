"""DIPRA's finite-difference scheme, so it can be run beside the engine's.

Girona and Costa (2013), DIPRA: A user-friendly program to model multi-element
diffusion in olivine with applications to timescales of magmatic processes,
Geochem. Geophys. Geosyst. 14, 422-431, eq. 5 (Auxiliary Material 1, eq. 8):

    C[k, j+1] = C[k, j] + dt/dx**2 * {
        (C[k+1] - C[k]) * (X_Fo[k+1] - X_Fo[k]) * dD/dX_Fo at X_Fo[k]
        + (C[k+1] - 2 C[k] + C[k-1]) * D at X_Fo[k] }

Both schemes solve dC/dt = D d2C/dx2 + (dD/dx)(dC/dx). They differ only in the
cross term: DIPRA uses forward differences and the analytic dD/dX_Fo; the engine
(`engine.diffusion_step`) uses central differences of C and of D. Both converge to
the same solution as the grid is refined (``tests/test_physics.py``).
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np

from olivine_diffusion import engine

# D_Fo is proportional to 10**(3 * (0.9 - X_Fo)) (Dohmen and Chakraborty 2007;
# DIPRA Table 1), so dD/dX_Fo = -3 ln(10) D.
_FO_EXPONENT = 3.0


def D_Fo_and_derivative(
    T_K: float,
    P_Pa: float,
    fO2_Pa: float,
    alpha_deg: float,
    beta_deg: float,
    gamma_deg: float,
    activation_energy_J_mol: float,
) -> tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """Fe-Mg diffusivity along the profile and its analytic derivative with respect to X_Fo."""
    diffusivity = engine.D_Fo(
        T_K, P_Pa, fO2_Pa, alpha_deg, beta_deg, gamma_deg, EFo=activation_energy_J_mol
    )

    def derivative(x_fo: np.ndarray) -> np.ndarray:
        return -_FO_EXPONENT * math.log(10) * diffusivity(x_fo)

    return diffusivity, derivative


def diffusion_step(
    c: np.ndarray,
    x_fo: np.ndarray,
    diffusivity: Callable[[np.ndarray], np.ndarray],
    derivative: Callable[[np.ndarray], np.ndarray],
    bounds_c: tuple[float, float],
    bounds_fo: tuple[float, float],
    dt_s: float,
    dx_m: float,
) -> np.ndarray:
    """One DIPRA time step (eq. 5) with fixed concentrations at both ends."""
    full_c = np.concatenate([[bounds_c[0]], c, [bounds_c[1]]])
    full_fo = np.concatenate([[bounds_fo[0]], x_fo, [bounds_fo[1]]])
    c_k, c_next, c_prev = full_c[1:-1], full_c[2:], full_c[:-2]
    fo_k, fo_next = full_fo[1:-1], full_fo[2:]
    cross = (c_next - c_k) * (fo_next - fo_k) * derivative(fo_k)
    laplacian = (c_next - 2 * c_k + c_prev) * diffusivity(fo_k)
    return c + dt_s / dx_m**2 * (cross + laplacian)


def timestepper(
    vector_c_in: np.ndarray,
    diffusivity: Callable[[np.ndarray], np.ndarray],
    derivative: Callable[[np.ndarray], np.ndarray],
    bounds_c: tuple[float, float],
    timesteps: int,
    dt: float,
    dx: float,
) -> np.ndarray:
    """Fo diffusion with DIPRA's scheme; same arguments and output as `engine.timestepper`.

    Returns an array with the profile at every step (row ``i`` is at ``i * dt``).
    """
    results = np.zeros((timesteps + 1, len(vector_c_in)))
    results[0] = vector_c_in
    c = np.asarray(vector_c_in, dtype=float)
    for n in range(timesteps):
        c = diffusion_step(c, c, diffusivity, derivative, bounds_c, bounds_c, dt, dx)
        results[n + 1] = c
    return results
