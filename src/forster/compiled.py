"""Compiled (Numba) time steppers, used when Numba is installed (``pip install
forster[fast]``) and the diffusivity is a `forster.diffusivity.LogLinearDiffusivity`
(every built-in law). Otherwise, or with ``FORSTER_NO_NUMBA=1`` in the environment, the
NumPy steppers run. The legacy uncertainty method always uses NumPy, so it reproduces
the old scripts bit for bit.

Per time step (150 nodes): central 15 -> 0.9 us, dipra 20 -> 0.9 us, implicit 22 -> 3 us.
Results agree with NumPy to rounding (1e-16 explicit, 1e-14 implicit in Fo).
"""

from __future__ import annotations

import os
from collections.abc import Callable

import numpy as np

from forster.boundary import Rim, is_fixed, rim_at
from forster.diffusivity import LogLinearDiffusivity
from forster.implicit import grid_geometry

try:
    if os.environ.get("FORSTER_NO_NUMBA") == "1":
        raise ImportError("disabled by FORSTER_NO_NUMBA")
    from forster import _kernels
except ImportError:
    _kernels = None

AVAILABLE = _kernels is not None
Advance = Callable[..., np.ndarray]


def _parameters(D: object) -> tuple[float, float, float] | None:
    if _kernels is None or not isinstance(D, LogLinearDiffusivity):
        return None
    return D.at_ref, D.fo_coefficient, D.fo_ref


def explicit_stepper(
    scheme: str, D: object, bounds: tuple[float, float], dt_s: float, dx_m: float
) -> Advance | None:
    """``advance(c, n, first_row=0)`` for "central" or "dipra", or None if it cannot be
    compiled (no Numba, or D not log-linear)."""
    params = _parameters(D)
    if params is None or _kernels is None:
        return None
    steps = _kernels.central_steps if scheme == "central" else _kernels.dipra_steps
    lo, hi = float(bounds[0]), float(bounds[1])

    def advance(c: np.ndarray, n: int, first_row: int = 0) -> np.ndarray:
        return steps(np.asarray(c, dtype=float), *params, lo, hi, dt_s, dx_m, n)

    return advance


def implicit_stepper(
    D: object, rim: Rim, core: float, dt_s: float, x_um: np.ndarray, t0_s: float = 0.0
) -> Advance | None:
    """The compiled `forster.implicit.stepper`, or None if it cannot be compiled."""
    params = _parameters(D)
    if params is None or _kernels is None:
        return None
    kernels = _kernels
    h, w = grid_geometry(np.asarray(x_um, dtype=float))
    gamma = kernels.GAMMA

    def advance(c: np.ndarray, n: int, first_row: int = 0) -> np.ndarray:
        t = t0_s + (first_row + np.arange(n)) * dt_s
        if is_fixed(rim):
            rim_mid = rim_end = np.full(n, rim_at(rim, 0.0))
        else:
            rim_mid = np.array([rim_at(rim, s) for s in t + gamma * dt_s])
            rim_end = np.array([rim_at(rim, s) for s in t + dt_s])
        start = np.array(c, dtype=float)
        start[0], start[-1] = rim_at(rim, t0_s + first_row * dt_s), core
        return kernels.tr_bdf2_steps(start, *params, rim_mid, rim_end, core, dt_s, h, w, n)

    return advance
