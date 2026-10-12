"""Fixed engine, old engine and DIPRA eq. 5 (Girona and Costa 2013) against a converged flux-form reference."""
import sys, numpy as np
sys.path.insert(0, sys.argv[1])
import engine_old as OLD
from olivine_diffusion import engine as NEW

T, P, fO2 = 1200 + 273.15, 4.2e7, 0.00138          # round robin conditions, profile along [001]
D = NEW.D_Fo(T, P, fO2, 90, 90, 0)
dDdX = lambda X: -3 * np.log(10) * D(X)            # analytic, as DIPRA evaluates it (Table 1 law)
EDGE, CORE, L_UM, STEP_UM = 0.80, 0.90, 300.0, 150.0
T_END = 15.8 * 86400                               # same case as the earlier probe

def grid(dx_um):
    x = np.arange(0, L_UM, dx_um); return x, np.where(x < STEP_UM, EDGE, CORE)

def nsteps(dx_m, safety):
    n = int(np.ceil(T_END / (safety * dx_m**2 / D(np.array([EDGE])).item()))); return n, T_END / n

def engine_scheme(mod, dx_um):
    x, c0 = grid(dx_um); dx = dx_um * 1e-6; n, dt = nsteps(dx, 0.2)
    out = mod.timestepper(vector_c_in=c0, vector_Fo_in=c0, diffusivity_function=D,
                          bounds_c=(EDGE, CORE), timesteps=n, dt=dt, dx=dx)[-1]
    return x, out

def dipra_eq5(dx_um):
    x, c = grid(dx_um); dx = dx_um * 1e-6; n, dt = nsteps(dx, 0.2)
    for _ in range(n):
        full = np.concatenate([[EDGE], c, [CORE]])            # fixed boundary values
        ck, cp, cm = full[1:-1], full[2:], full[:-2]
        # C_k^{j+1} = C_k + dt/dx^2 { (C_{k+1}-C_k)(X_{k+1}-X_k) dD/dX|_k + (C_{k+1}-2C_k+C_{k-1}) D|_k }
        c = ck + dt / dx**2 * ((cp - ck) * (cp - ck) * dDdX(ck) + (cp - 2 * ck + cm) * D(ck))
    return x, c

def flux_form(dx_um, safety=0.05):
    x, c = grid(dx_um); dx = dx_um * 1e-6; n, dt = nsteps(dx, safety)
    for _ in range(n):
        full = np.concatenate([[EDGE], c, [CORE]])
        Df = 0.5 * (D(full[:-1]) + D(full[1:]))
        c = c + dt * np.diff(Df * np.diff(full) / dx) / dx
    return x, c

xr, ref = flux_form(0.25)                                   # converged reference
def err(xc):
    x, c = xc; return np.abs(c - np.interp(x, xr, ref)).max()
print(f"reference: flux form, dx 0.25 um; its change from dx 0.5: {err(flux_form(0.5)):.1e} Fo")
print(f"{'scheme':38s} {'dx 1.0 um':>11s} {'dx 0.5 um':>11s}")
for name, f in [("engine as written (cross term / 2)", lambda d: engine_scheme(OLD, d)),
                ("engine fixed (cross term / 4)", lambda d: engine_scheme(NEW, d)),
                ("DIPRA eq. 5 (forward cross term)", dipra_eq5)]:
    print(f"{name:38s} {err(f(1.0)):11.1e} {err(f(0.5)):11.1e}")
x1, a = engine_scheme(NEW, 0.5); x2, b = dipra_eq5(0.5)
print(f"fixed engine vs DIPRA eq. 5 directly, dx 0.5 um: {np.abs(a - b).max():.1e} Fo")
