"""Where does fit time go, and what would non-uniform or implicit grids save?"""
import sys, time, warnings
import numpy as np
import scipy.linalg as sla
sys.path.insert(0, "tests"); warnings.simplefilter("ignore")
from conftest import ROUND_ROBIN_DATA, ROUND_ROBIN_PARAMS
from forster import RunDefaults, read_profiles, read_runs, fit_run
from forster import engine
from forster.model import fe_mg_diffusivity

# 1. Explicit step cost vs number of grid points
run = {r.name: r for r in read_runs(ROUND_ROBIN_PARAMS, defaults=RunDefaults(max_time_days=730))}["KS20-527_2-transect-1"]
D, _ = fe_mg_diffusivity(run.diffusivity).functions(run.conditions, run.orientation)
print("1. explicit (engine) cost per time step vs grid points")
for N in (50, 100, 150, 300, 600, 1200, 4800):
    x = np.linspace(0, 300, N); c = np.where(x < 41, 0.82, 0.885)
    steps = 2000
    t0 = time.perf_counter()
    engine.timestepper(c, c, D, (0.82, 0.885), steps, dt=1.0, dx=(x[1]-x[0])*1e-6)
    print(f"   N={N:5d}: {1e6*(time.perf_counter()-t0)/steps:6.1f} us/step")

# 2. Semi-implicit backward Euler (D lagged one step, flux form, tridiagonal), uniform grid
def implicit(c0, D, bounds, dt, dx, nsteps):
    c = c0.copy(); out = [c.copy()]
    n = len(c)
    for _ in range(nsteps):
        d = D(c); dface = 0.5 * (d[1:] + d[:-1]); r = dt / dx**2
        ab = np.zeros((3, n)); ab[1] = 1.0
        ab[1, 1:-1] += r * (dface[:-1] + dface[1:])
        ab[0, 2:] = -r * dface[1:]
        ab[2, :-2] = -r * dface[:-1]
        rhs = c.copy(); rhs[0], rhs[-1] = bounds   # fixed ends
        ab[0, 1] = 0.0; ab[2, -2] = 0.0
        c = sla.solve_banded((1, 1), ab, rhs); out.append(c.copy())
    return np.array(out)

profiles = read_profiles(ROUND_ROBIN_DATA)
ref = fit_run(run, profiles, stop_early=False)       # explicit, dx 1 um, dt 600 s
k = ref.indices[0]; target_t = k * ref.dt_s
x = ref.model_x_um; c0 = ref.curves[0]; dx = (x[1]-x[0])*1e-6
ref_profile = ref.curves[k]
print(f"\n2. KS20-527_2-transect-1 at its best fit ({target_t/86400:.1f} d; explicit: {k} steps of 600 s)")
t0 = time.perf_counter(); engine.timestepper(c0, c0, D, (c0[0], c0[-1]), k, dt=600.0, dx=dx); te = time.perf_counter() - t0
print(f"   explicit to the best fit: {te:.2f} s")
for nsteps in (10000, 2000, 500, 200, 100, 50):
    dt = target_t / nsteps
    t0 = time.perf_counter(); out = implicit(c0, D, (c0[0], c0[-1]), dt, dx, nsteps); ti = time.perf_counter() - t0
    err = np.abs(out[-1] - ref_profile).max()
    print(f"   implicit {nsteps:6d} steps (dt {dt:7.0f} s): {ti:5.2f} s, max |profile - explicit| = {err*100:.4f} Fo")

print("\n3. Is the 0.065 mol% gap spatial? explicit vs implicit at dx 1 and 0.5 um; implicit dt convergence")
for dxu in (1.0, 0.5):
    xg = np.arange(0, 300 + dxu / 2, dxu); cg = np.where(xg < 41, 0.82, 0.885); cg[0], cg[-1] = 0.82, 0.885
    dtx = 0.4 * (dxu * 1e-6) ** 2 / float(D(np.array([0.82]))[0])
    nx = int(round(target_t / dtx))
    ex = engine.timestepper(cg, cg, D, (0.82, 0.885), nx, dt=target_t / nx, dx=dxu * 1e-6)[-1]
    im_fine = implicit(cg, D, (0.82, 0.885), target_t / 20000, dxu * 1e-6, 20000)[-1]
    im_50 = implicit(cg, D, (0.82, 0.885), target_t / 50, dxu * 1e-6, 50)[-1]
    im_200 = implicit(cg, D, (0.82, 0.885), target_t / 200, dxu * 1e-6, 200)[-1]
    print(f"   dx {dxu} um: |explicit - implicit(20000 steps)| = {np.abs(ex-im_fine).max()*100:.4f} mol%; "
          f"implicit 200 vs 20000 steps: {np.abs(im_200-im_fine).max()*100:.4f}; 50 vs 20000: {np.abs(im_50-im_fine).max()*100:.4f}")
