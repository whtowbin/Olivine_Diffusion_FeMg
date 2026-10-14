"""Numba kernels for the three schemes vs the current NumPy code: per-step cost and agreement.

D(X) = a * 10**(k * (ref - X)), the form of every built-in Fe-Mg law. Run from the repo
root with: env -u VIRTUAL_ENV uv run --with numba python <this file>
"""

import math
import time

import numpy as np
from numba import njit

from forster import DIFFUSIVITY_PRESETS, Conditions, Orientation, dipra, engine, implicit

GAMMA = 2.0 - math.sqrt(2.0)


@njit(cache=True)
def _d(a, k, ref, x):
    return a * 10.0 ** (k * (ref - x))


@njit(cache=True)
def central_steps(c0, a, k, ref, lo, hi, dt, dx, n):
    """engine.timestepper: ghost values lo/hi one node outside, central cross term."""
    m = c0.size
    out = np.empty((n + 1, m))
    out[0] = c0
    c = c0.copy()
    new = np.empty(m)
    d = np.empty(m + 2)
    delta = dt / dx**2
    for t in range(n):
        d[0] = _d(a, k, ref, lo)
        d[m + 1] = _d(a, k, ref, hi)
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
    """dipra.timestepper: forward cross term, analytic dD/dX = -k ln10 D."""
    m = c0.size
    out = np.empty((n + 1, m))
    out[0] = c0
    c = c0.copy()
    new = np.empty(m)
    s = -k * math.log(10.0)
    delta = dt / dx**2
    for t in range(n):
        for i in range(m):
            cm = c[i - 1] if i > 0 else lo
            cp = c[i + 1] if i < m - 1 else hi
            di = _d(a, k, ref, c[i])
            new[i] = c[i] + delta * ((cp - c[i]) * (cp - c[i]) * s * di + (cp - 2 * c[i] + cm) * di)
        c, new = new, c
        out[t + 1] = c
    return out


@njit(cache=True)
def _stage(c, a, k, ref, dt, h, w, weight, rhs, end0, end1, explicit_part, d, cond, cp, dp, y):
    """Solve (I - weight*dt*A(c)) y = rhs + explicit_part*dt*A(c) c, ends held (Thomas).
    d, cond, cp, dp, y: workspace."""
    m = c.size
    for i in range(m):
        d[i] = _d(a, k, ref, c[i])
    for i in range(m - 1):
        cond[i] = dt * 0.5 * (d[i] + d[i + 1]) / h[i]
    # Row 0 held: main 1, upper 0.
    cp[0] = 0.0
    dp[0] = end0
    for i in range(1, m - 1):
        lo_ = cond[i - 1] / w[i - 1]
        up_ = cond[i] / w[i - 1]
        b = rhs[i]
        if explicit_part != 0.0:
            b += explicit_part * (lo_ * c[i - 1] - (lo_ + up_) * c[i] + up_ * c[i + 1])
        low = -weight * lo_
        den = 1.0 + weight * (lo_ + up_) - low * cp[i - 1]
        cp[i] = -weight * up_ / den
        dp[i] = (b - low * dp[i - 1]) / den
    y[m - 1] = end1
    for i in range(m - 2, -1, -1):
        y[i] = dp[i] - cp[i] * y[i + 1]
    return y


@njit(cache=True)
def trbdf2_steps(c0, a, k, ref, rim_mid, rim_end, core, dt, h, w, n):
    m = c0.size
    out = np.empty((n + 1, m))
    c = c0.copy()
    out[0] = c
    g = GAMMA
    aa = 1.0 / (g * (2.0 - g))
    bb = (1.0 - g) ** 2 / (g * (2.0 - g))
    wt = (1.0 - g) / (2.0 - g)
    d = np.empty(m); cond = np.empty(m - 1); cp = np.empty(m); dp = np.empty(m)
    mid = np.empty(m); rhs = np.empty(m)
    for t in range(n):
        _stage(c, a, k, ref, dt, h, w, g / 2, c, rim_mid[t], core, g / 2, d, cond, cp, dp, mid)
        for i in range(m):
            rhs[i] = aa * mid[i] - bb * c[i]
        _stage(mid, a, k, ref, dt, h, w, wt, rhs, rim_end[t], core, 0.0, d, cond, cp, dp, c)
        out[t + 1] = c
    return out


def per_step(f, n):
    f()  # warm
    best = min((lambda: (t0 := time.perf_counter(), f(), time.perf_counter() - t0)[2])() for _ in range(3))
    return 1e6 * best / n


if __name__ == "__main__":
    t0 = time.perf_counter()
    law = DIFFUSIVITY_PRESETS["dohmen_chakraborty2007"]
    cond, orient = Conditions(1200, 4.2e7, 1.4e-3), Orientation(90, 90, 0)
    D, dD = law.functions(cond, orient)
    ref = 0.9
    a = float(D(np.array([ref]))[0])
    k = law.fo_coefficient
    print(f"law check: max rel |D - a 10^(k(ref-X))| = "
          f"{np.max(np.abs(D(np.linspace(0.6, 1, 50)) / (a * 10 ** (k * (ref - np.linspace(0.6, 1, 50)))) - 1)):.1e}")
    for m in (75, 150, 600):
        x = np.linspace(0.0, 150.0, m)
        dx = (x[1] - x[0]) * 1e-6
        dt = 0.4 * dx**2 / float(D(np.array([0.82]))[0])
        c0 = np.where(x < 20, 0.82, 0.88).astype(float)
        nsteps = 3000
        ce = engine.timestepper(c0, c0, D, (0.82, 0.88), nsteps, dt=dt, dx=dx)
        cn = central_steps(c0, a, k, ref, 0.82, 0.88, dt, dx, nsteps)
        de = dipra.timestepper(c0, D, dD, (0.82, 0.88), nsteps, dt, dx)
        dn = dipra_steps(c0, a, k, ref, 0.82, 0.88, dt, dx, nsteps)
        ci = implicit.cell_average(x, [20.0], [0.82, 0.88])
        dti = 50 * dt
        adv = implicit.stepper(D, 0.82, 0.88, dti, x)
        ie = adv(ci, nsteps // 10)
        hh = np.diff(x) * 1e-6
        ww = 0.5 * (hh[1:] + hh[:-1])
        rim = np.full(nsteps // 10, 0.82)
        ci0 = ci.copy(); ci0[0], ci0[-1] = 0.82, 0.88
        inn = trbdf2_steps(ci0, a, k, ref, rim, rim, 0.88, dti, hh, ww, nsteps // 10)
        print(f"n={m}: central numpy {per_step(lambda: engine.timestepper(c0, c0, D, (0.82, 0.88), nsteps, dt=dt, dx=dx), nsteps):.1f} "
              f"numba {per_step(lambda: central_steps(c0, a, k, ref, 0.82, 0.88, dt, dx, nsteps), nsteps):.2f} us (diff {np.abs(ce - cn).max():.1e}); "
              f"dipra numpy {per_step(lambda: dipra.timestepper(c0, D, dD, (0.82, 0.88), nsteps, dt, dx), nsteps):.1f} "
              f"numba {per_step(lambda: dipra_steps(c0, a, k, ref, 0.82, 0.88, dt, dx, nsteps), nsteps):.2f} us (diff {np.abs(de - dn).max():.1e}); "
              f"implicit numpy {per_step(lambda: adv(ci, nsteps // 10), nsteps // 10):.1f} "
              f"numba {per_step(lambda: trbdf2_steps(ci0, a, k, ref, rim, rim, 0.88, dti, hh, ww, nsteps // 10), nsteps // 10):.2f} us (diff {np.abs(ie - inn).max():.1e})",
              flush=True)
    print(f"total incl. compile/cache load: {time.perf_counter() - t0:.1f} s")
