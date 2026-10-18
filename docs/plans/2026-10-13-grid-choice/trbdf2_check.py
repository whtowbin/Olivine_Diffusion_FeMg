"""Implicit scheme with the boundary on a node: convergence of the fitted time in dx and dt.

Fitted time = parabolic minimum of the white-noise residual sum over rows (so the time
step does not quantise it). Six real profiles; reference = implicit, k = l/dx = 160,
BDF2 with 4000 steps to the fit. Compared with the explicit engine on the same metric.
Run from the repo root.
"""

import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.linalg as sla

sys.path.insert(0, "tests")
warnings.simplefilter("ignore")
from conftest import PAPER_DATA, PAPER_PARAMS, ROUND_ROBIN_DATA, ROUND_ROBIN_PARAMS  # noqa: E402

from forster import RunDefaults, fit_profile, read_profiles, read_runs  # noqa: E402
from forster.model import Profile, _diffusion_model, fe_mg_diffusivity, model_grid  # noqa: E402
from forster.uncertainty import chi2_per_step, interpolation_matrix, points_in_domain  # noqa: E402

# The round robin rows of the reference tables are private: read the full copy in the
# round robin submodule when it is checked out, else the public table (paper rows only).
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
PRIVATE = ROOT / "Diffusion Round Robin" / "outputs" / "public_tree_moved" / HERE.relative_to(ROOT)


def reference_table(name):
    return pd.read_csv(PRIVATE / name if (PRIVATE / name).exists() else HERE / name)


rr = {r.name: r for r in read_runs(ROUND_ROBIN_PARAMS, defaults=RunDefaults(max_time_days=730))}
pp = {r.name: r for r in read_runs(PAPER_PARAMS)}
rrp, ppp = read_profiles(ROUND_ROBIN_DATA), read_profiles(PAPER_DATA, sheet="Olivine")
cases = [(rr[n], rrp) for n in ("KS20-527_2-transect-1", "KS20-527-8_transect-2", "SH63olv066-1")]
cases += [
    (pp[n], ppp)
    for n in ("AZ18 WHT01_Ol_63_prof_1", "AZ18 WHT01_bigol1_overgrowth_prof_1", "Xenocryst_ol6_prof_1")
]


def parabolic_min(s: np.ndarray, dt: float) -> float:
    i = int(np.argmin(s))
    if 0 < i < len(s) - 1:
        a, b, c = s[i - 1], s[i], s[i + 1]
        den = a - 2 * b + c
        if den > 0:
            return (i + 0.5 * (a - c) / den) * dt
    return i * dt


def cell_average(x: np.ndarray, initial) -> np.ndarray:
    """Average of the step-function start over each node's cell (half cells at the ends)."""
    breaks = list(initial.steps_um)
    values = [initial.edge_fo, initial.core_fo]
    if initial.step_fo is not None:
        values[1:1] = [initial.step_fo]
    h = x[1] - x[0]
    lo = np.maximum(x - h / 2, x[0])
    hi = np.minimum(x + h / 2, x[-1])
    edges = [-np.inf, *breaks, np.inf]
    out = np.zeros_like(x)
    for v, a, b in zip(values, edges[:-1], edges[1:], strict=True):
        out += v * np.clip(np.minimum(hi, b) - np.maximum(lo, a), 0, None)
    return out / (hi - lo)


def snapped(x: np.ndarray, initial) -> np.ndarray:
    values = [initial.edge_fo, initial.core_fo]
    if initial.step_fo is not None:
        values[1:1] = [initial.step_fo]
    out = np.full_like(x, values[-1])
    for s, v in zip(reversed(initial.steps_um), reversed(values[:-1]), strict=True):
        out[: int(np.abs(x - s).argmin())] = v
    return out


def implicit(c0, D, rim, core, dt, dx_m, nsteps, method="bdf2"):
    """Rows 0..nsteps; c[0] = rim and c[-1] = core held on the end nodes; D lagged/extrapolated."""
    n = len(c0)
    r = dt / dx_m**2
    rows = np.empty((nsteps + 1, n))
    rows[0] = c0
    prev, c = None, c0.copy()
    c[0], c[-1] = rim, core
    for k in range(nsteps):
        bdf2 = method == "bdf2" and prev is not None
        star = 2 * c - prev if bdf2 else c
        d = D(np.clip(star, 1e-3, 1.0))
        f = 0.5 * (d[1:] + d[:-1])  # face diffusivities
        a0 = 1.5 if bdf2 else 1.0
        ab = np.zeros((3, n))
        ab[1] = a0
        ab[1, 1:-1] += r * (f[:-1] + f[1:])
        ab[0, 2:] = -r * f[1:]
        ab[2, :-2] = -r * f[:-1]
        ab[1, 0] = ab[1, -1] = 1.0
        ab[0, 1] = ab[2, -2] = 0.0
        rhs = (2 * c - 0.5 * prev) if bdf2 else c.copy()
        rhs[0], rhs[-1] = rim, core
        prev, c = c, sla.solve_banded((1, 1), ab, rhs)
        rows[k + 1] = c
    return rows



G = 2 - np.sqrt(2)

def operator(c, D, r):
    d = D(np.clip(c, 1e-3, 1.0)); f = 0.5 * (d[1:] + d[:-1]); n = len(c)
    main = np.zeros(n); main[1:-1] = -(f[:-1] + f[1:]); up = np.zeros(n); up[2:] = f[1:]; lo = np.zeros(n); lo[:-2] = f[:-1]
    return r * up, r * main, r * lo   # banded A*dt pieces (zero rows at the ends)

def solve(a, c_rhs, up, main, lo, rim, core):
    n = len(c_rhs); ab = np.zeros((3, n)); ab[1] = 1 - a * main; ab[0] = -a * up; ab[2] = -a * lo
    ab[1, 0] = ab[1, -1] = 1; ab[0, 1] = ab[2, -2] = 0
    rhs = c_rhs.copy(); rhs[0], rhs[-1] = rim, core
    return sla.solve_banded((1, 1), ab, rhs)

def apply(up, main, lo, c):
    out = main * c; out[:-1] += up[1:] * c[1:]; out[1:] += lo[:-1] * c[:-1]; return out

def trbdf2(c0, D, rim, core, dt, dx_m, nsteps, lag="stage"):
    r = dt / dx_m**2; rows = np.empty((nsteps + 1, len(c0))); rows[0] = c0
    c = c0.copy(); c[0], c[-1] = rim, core
    for k in range(nsteps):
        up, main, lo = operator(c, D, r)
        cs = solve(G / 2, c + G / 2 * apply(up, main, lo, c), up, main, lo, rim, core)
        if lag == "stage":
            up, main, lo = operator(cs, D, r)
        w = (1 - G) / (2 - G)
        c = solve(w, cs / (G * (2 - G)) - (1 - G) ** 2 / (G * (2 - G)) * c, up, main, lo, rim, core)
        rows[k + 1] = c
    return rows

for run, profiles in cases:
    m = profiles[run.profile_name]; ini = run.initial
    D, _ = fe_mg_diffusivity(run.diffusivity).functions(run.conditions, run.orientation)
    fos = [ini.edge_fo, ini.core_fo] + ([ini.step_fo] if ini.step_fo else [])
    d_mid = float(D(np.array([np.mean(fos)]))[0])
    ref = reference_table("implicit_study.csv").query("run == @run.name").ref_s.iloc[0]
    ell = np.sqrt(d_mid * ref) * 1e6
    lo_, hi = ini.domain_um[0], min(ini.domain_um[1], m.distance_um.max())
    px, py = points_in_domain(np.array([lo_, hi]), m.distance_um, m.fo)
    x = np.linspace(lo_, hi, int(round((hi - lo_) / (ell / 60))) + 1); c0 = cell_average(x, ini)
    W = interpolation_matrix(x, px); out = []
    for n in (5, 10, 20, 50):
        dt = ref / n
        e = {}
        for name, f in (("bdf2", lambda: implicit(c0, D, ini.edge_fo, ini.core_fo, dt, (x[1]-x[0])*1e-6, int(2.5*n), "bdf2")),
                        ("trbdf2", lambda: trbdf2(c0, D, ini.edge_fo, ini.core_fo, dt, (x[1]-x[0])*1e-6, int(2.5*n))),
                        ("trbdf2_lag", lambda: trbdf2(c0, D, ini.edge_fo, ini.core_fo, dt, (x[1]-x[0])*1e-6, int(2.5*n), "step"))):
            e[name] = 100 * (parabolic_min(chi2_per_step(f(), W, py), dt) / ref - 1)
        out.append(f"n={n}: " + " ".join(f"{k} {v:+.2f}%" for k, v in e.items()))
    print(run.name, " | ".join(out), flush=True)
