"""Numba versions of the three time steppers (imported by `forster.compiled` only when
Numba is installed). D(X) = a * 10**(k * (ref - X)), the form of every built-in law.

Each matches its NumPy original to rounding (1e-16 explicit, 1e-14 implicit):
``engine.timestepper`` ("central"), ``dipra.timestepper`` ("dipra") and
``implicit.tr_bdf2_step`` (TR-BDF2 with the ends held on the end nodes).
"""

import math

import numpy as np
from numba import njit

GAMMA = 2.0 - math.sqrt(2.0)


@njit(cache=True)
def _d(a, k, ref, x):
    return a * 10.0 ** (k * (ref - x))


@njit(cache=True)
def central_steps(c0, a, k, ref, lo, hi, dt, dx, n):
    """`engine.timestepper`: lo and hi held one node outside, central cross term."""
    m = c0.size
    out = np.empty((n + 1, m))
    out[0] = c0
    c = c0.copy()
    new = np.empty(m)
    d = np.empty(m + 2)
    delta = dt / dx**2
    d[0] = _d(a, k, ref, lo)
    d[m + 1] = _d(a, k, ref, hi)
    for t in range(n):
        for i in range(m):
            d[i + 1] = _d(a, k, ref, c[i])
        for i in range(m):
            cm = c[i - 1] if i > 0 else lo
            cp = c[i + 1] if i < m - 1 else hi
            lap = (cp - 2.0 * c[i] + cm) * d[i + 1]
            new[i] = c[i] + delta * (lap + (cp - cm) * (d[i + 2] - d[i]) / 4)
        c, new = new, c
        out[t + 1] = c
    return out


@njit(cache=True)
def dipra_steps(c0, a, k, ref, lo, hi, dt, dx, n):
    """`dipra.timestepper`: forward cross term, dD/dX = -k ln(10) D."""
    m = c0.size
    out = np.empty((n + 1, m))
    out[0] = c0
    c = c0.copy()
    new = np.empty(m)
    slope = -k * math.log(10.0)
    delta = dt / dx**2
    for t in range(n):
        for i in range(m):
            cm = c[i - 1] if i > 0 else lo
            cp = c[i + 1] if i < m - 1 else hi
            di = _d(a, k, ref, c[i])
            new[i] = c[i] + delta * (
                (cp - c[i]) * (cp - c[i]) * slope * di + (cp - 2 * c[i] + cm) * di
            )
        c, new = new, c
        out[t + 1] = c
    return out


@njit(cache=True)
def _stage(c, a, k, ref, dt, h, w, weight, rhs, end0, end1, explicit_part, d, cond, cp, dp, y):
    """y solving (I - weight dt A(c)) y = rhs + explicit_part dt A(c) c with the end
    values held (Thomas algorithm); d, cond, cp, dp: workspace."""
    m = c.size
    for i in range(m):
        d[i] = _d(a, k, ref, c[i])
    for i in range(m - 1):
        cond[i] = dt * 0.5 * (d[i] + d[i + 1]) / h[i]
    cp[0] = 0.0
    dp[0] = end0
    for i in range(1, m - 1):
        lo = cond[i - 1] / w[i - 1]
        up = cond[i] / w[i - 1]
        b = rhs[i]
        if explicit_part != 0.0:
            b += explicit_part * (lo * c[i - 1] - (lo + up) * c[i] + up * c[i + 1])
        low = -weight * lo
        den = 1.0 + weight * (lo + up) - low * cp[i - 1]
        cp[i] = -weight * up / den
        dp[i] = (b - low * dp[i - 1]) / den
    y[m - 1] = end1
    for i in range(m - 2, -1, -1):
        y[i] = dp[i] - cp[i] * y[i + 1]


@njit(cache=True)
def tr_bdf2_steps(c0, a, k, ref, rim_mid, rim_end, core, dt, h, w, n):
    """n TR-BDF2 steps; rim_mid[j] and rim_end[j] are the rim at t_j + gamma dt and t_j + dt."""
    m = c0.size
    out = np.empty((n + 1, m))
    c = c0.copy()
    out[0] = c
    aa = 1.0 / (GAMMA * (2.0 - GAMMA))
    bb = (1.0 - GAMMA) ** 2 / (GAMMA * (2.0 - GAMMA))
    weight = (1.0 - GAMMA) / (2.0 - GAMMA)
    d = np.empty(m)
    cond = np.empty(m - 1)
    cp = np.empty(m)
    dp = np.empty(m)
    mid = np.empty(m)
    rhs = np.empty(m)
    for t in range(n):
        _stage(
            c, a, k, ref, dt, h, w, GAMMA / 2, c, rim_mid[t], core, GAMMA / 2, d, cond, cp, dp, mid
        )
        for i in range(m):
            rhs[i] = aa * mid[i] - bb * c[i]
        _stage(mid, a, k, ref, dt, h, w, weight, rhs, rim_end[t], core, 0.0, d, cond, cp, dp, c)
        out[t + 1] = c
    return out
