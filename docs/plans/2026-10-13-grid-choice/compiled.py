"""Explicit step as a compiled loop (Numba) vs the numpy/convolve engine; and implicit."""
import sys, time, warnings
import numpy as np
from numba import njit
sys.path.insert(0, "tests"); warnings.simplefilter("ignore")
from conftest import ROUND_ROBIN_DATA, ROUND_ROBIN_PARAMS
from forster import RunDefaults, read_profiles, read_runs, fit_run, engine
from forster.model import fe_mg_diffusivity

run = {r.name: r for r in read_runs(ROUND_ROBIN_PARAMS, defaults=RunDefaults(max_time_days=730))}["KS20-527_2-transect-1"]
ref = fit_run(run, read_profiles(ROUND_ROBIN_DATA), stop_early=False)
k = ref.indices[0]; c0 = ref.curves[0]; dx = (ref.model_x_um[1] - ref.model_x_um[0]) * 1e-6; dt = ref.dt_s
D, _ = fe_mg_diffusivity(run.diffusivity).functions(run.conditions, run.orientation)
D_ref = float(D(np.array([0.9]))[0])  # D(X) = D_ref * 10**(3*(0.9 - X)) for this law

@njit(cache=True)
def explicit_loop(c0, D_ref, dt, dx, nsteps, lo, hi):
    n = c0.size
    out = np.empty((nsteps + 1, n)); out[0] = c0
    c = c0.copy(); new = c0.copy(); d = np.empty(n)
    r = dt / dx**2
    for t in range(nsteps):
        # Engine convention: the boundary values sit in ghost nodes just outside the
        # array, and every array node evolves.
        d_lo = D_ref * 10.0 ** (3.0 * (0.9 - lo)); d_hi = D_ref * 10.0 ** (3.0 * (0.9 - hi))
        for i in range(n):
            d[i] = D_ref * 10.0 ** (3.0 * (0.9 - c[i]))
        for i in range(n):
            cm = c[i - 1] if i > 0 else lo
            cp = c[i + 1] if i < n - 1 else hi
            dm = d[i - 1] if i > 0 else d_lo
            dp = d[i + 1] if i < n - 1 else d_hi
            new[i] = c[i] + r * (d[i] * (cp - 2.0 * c[i] + cm) + 0.25 * (dp - dm) * (cp - cm))
        c, new = new, c
        out[t + 1] = c
    return out

explicit_loop(c0, D_ref, dt, dx, 10, c0[0], c0[-1])  # compile
for label, f in (("numpy/convolve engine", lambda: engine.timestepper(c0, c0, D, (c0[0], c0[-1]), k, dt=dt, dx=dx)),
                 ("compiled loop (Numba)", lambda: explicit_loop(c0, D_ref, dt, dx, k, c0[0], c0[-1]))):
    best = min((lambda: (t0 := time.perf_counter(), f(), time.perf_counter() - t0)[2])() for _ in range(3))
    print(f"{label:24s}: {best:.4f} s for {k} steps ({1e6 * best / k:.2f} us/step)")
a = engine.timestepper(c0, c0, D, (c0[0], c0[-1]), k, dt=dt, dx=dx)
b = explicit_loop(c0, D_ref, dt, dx, k, c0[0], c0[-1])
print(f"max |engine - compiled| over all steps: {np.abs(a - b).max():.2e} (Fo fraction)")
