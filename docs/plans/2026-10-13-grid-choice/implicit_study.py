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

import numpy as np
import pandas as pd
import scipy.linalg as sla

sys.path.insert(0, "tests")
warnings.simplefilter("ignore")
from conftest import PAPER_DATA, PAPER_PARAMS, ROUND_ROBIN_DATA, ROUND_ROBIN_PARAMS  # noqa: E402

from forster import RunDefaults, fit_profile, read_profiles, read_runs  # noqa: E402
from forster.model import Profile, _diffusion_model, fe_mg_diffusivity, model_grid  # noqa: E402
from forster.uncertainty import chi2_per_step, interpolation_matrix, points_in_domain  # noqa: E402

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


rows = []
for run, profiles in cases:
    m = profiles[run.profile_name]
    prof = Profile(m.distance_um, m.fo, name=run.name)
    ini = run.initial
    D, _ = fe_mg_diffusivity(run.diffusivity).functions(run.conditions, run.orientation)
    fos = [ini.edge_fo, ini.core_fo] + ([ini.step_fo] if ini.step_fo else [])
    d_mid = float(D(np.array([np.mean(fos)]))[0])
    d_max = float(D(np.array([min(fos)]))[0])
    base = fit_profile(prof, run.conditions, run.orientation, ini, run.grid, run.kriging, run.diffusivity)
    t0 = base.time_s
    ell = np.sqrt(d_mid * t0) * 1e6
    lo, hi = ini.domain_um[0], min(ini.domain_um[1], m.distance_um.max())
    px, py = points_in_domain(np.array([lo, hi]), m.distance_um, m.fo)

    def fit_implicit(k, n_to_fit, method="bdf2", start="average"):
        npts = max(3, int(round((hi - lo) / (ell / k))) + 1)
        x = np.linspace(lo, hi, npts)
        c0 = (cell_average if start == "average" else snapped)(x, ini)
        dt = t0 / n_to_fit
        tt = time.perf_counter()
        out = implicit(c0, D, ini.edge_fo, ini.core_fo, dt, (x[1] - x[0]) * 1e-6, int(2.5 * n_to_fit), method)
        sec = time.perf_counter() - tt
        s = chi2_per_step(out, interpolation_matrix(x, px), py)
        return parabolic_min(s, dt), sec, len(x)

    def fit_explicit(k):
        dx = ell / k
        grid = replace(run.grid, dx_um=dx, dt_s=0.4 * (dx * 1e-6) ** 2 / d_max, max_time_days=2.5 * t0 / 86400)
        step_x, dxu, (li, hi_) = model_grid(m.distance_um, grid, ini.domain_um)
        mx = step_x[li : hi_ + 1]
        start, advance = _diffusion_model("central", run.conditions, run.orientation, ini, grid, fe_mg_diffusivity(run.diffusivity), mx, dxu)
        tt = time.perf_counter()
        out = advance(start, grid.timesteps)
        sec = time.perf_counter() - tt
        qx, qy = points_in_domain(mx, m.distance_um, m.fo)
        s = chi2_per_step(out, interpolation_matrix(mx, qx), qy)
        return parabolic_min(s, grid.dt_s), sec, len(mx)

    ref, _, _ = fit_implicit(160, 4000)
    print(f"\n{run.name}: l = {ell:.1f} um, reference {ref / 86400:.3f} d", flush=True)
    for k in (3, 5, 8, 12, 20, 30, 60):
        for scheme, start in (("implicit", "average"), ("implicit", "snapped"), ("explicit", "snapped")):
            if scheme == "explicit" and k < 8:
                continue
            t, sec, nx = fit_explicit(k) if scheme == "explicit" else fit_implicit(k, 2000, start=start)
            rows.append(dict(run=run.name, study="dx", scheme=scheme, start=start, k=k, n=2000, time_s=t, ref_s=ref, seconds=sec, nodes=nx))
            print(f"  dx k={k:3d} {scheme:8s} {start:8s}: {100 * (t / ref - 1):+7.2f}%  {sec:6.2f} s", flush=True)
    for n in (5, 10, 20, 50, 100, 200):
        for method in ("be", "bdf2"):
            t, sec, nx = fit_implicit(60, n, method)
            rows.append(dict(run=run.name, study="dt", scheme=method, start="average", k=60, n=n, time_s=t, ref_s=ref, seconds=sec, nodes=nx))
            print(f"  dt n={n:4d} {method:5s}: {100 * (t / ref - 1):+7.2f}%  {sec:6.3f} s", flush=True)

pd.DataFrame(rows).to_csv(__file__.replace(".py", ".csv"), index=False)
