"""Convergence of fitted time with grid spacing dx = l/k, l = sqrt(D t); dt at D_max dt/dx^2 = 0.4."""
import sys, time, warnings
from dataclasses import replace
import numpy as np, pandas as pd
sys.path.insert(0, "tests"); warnings.simplefilter("ignore")
from conftest import PAPER_DATA, PAPER_PARAMS, ROUND_ROBIN_DATA, ROUND_ROBIN_PARAMS

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _private import write_rows  # noqa: E402  (round robin rows to the submodule)
from forster import RunDefaults, fit_profile, read_profiles, read_runs
from forster.model import Profile, fe_mg_diffusivity

rr = {r.name: r for r in read_runs(ROUND_ROBIN_PARAMS, defaults=RunDefaults(max_time_days=730))}
pp = {r.name: r for r in read_runs(PAPER_PARAMS)}
rrp, ppp = read_profiles(ROUND_ROBIN_DATA), read_profiles(PAPER_DATA, sheet="Olivine")
cases = [(rr[n], rrp) for n in ("KS20-527_2-transect-1", "KS20-527-8_transect-2", "SH63olv066-1")] + \
        [(pp[n], ppp) for n in ("AZ18 WHT01_Ol_63_prof_1", "AZ18 WHT01_bigol1_overgrowth_prof_1", "Xenocryst_ol6_prof_1")]
KS = (1, 1.5, 2, 3, 5, 8, 12, 20, 30)
rows = []
for run, profiles in cases:
    m = profiles[run.profile_name]; prof = Profile(m.distance_um, m.fo, name=run.name)
    D, _ = fe_mg_diffusivity(run.diffusivity).functions(run.conditions, run.orientation)
    fos = [run.initial.edge_fo, run.initial.core_fo] + ([run.initial.step_fo] if run.initial.step_fo else [])
    d_mid = float(D(np.array([np.mean(fos)]))[0]); d_max = float(D(np.array([min(fos)]))[0])
    base = fit_profile(prof, run.conditions, run.orientation, run.initial, run.grid, run.kriging, run.diffusivity)
    t0 = base.time_s
    ell = np.sqrt(d_mid * t0) * 1e6
    print(f"\n{run.name}: native dx {run.grid.dx_um} um -> {base.time_days:.3f} d; l = {ell:.2f} um (l/dx_native = {ell/run.grid.dx_um:.1f})", flush=True)
    for k in KS:
        dx = ell / k
        grid = replace(run.grid, dx_um=dx, dt_s=0.4 * (dx * 1e-6) ** 2 / d_max, max_time_days=3 * t0 / 86400)
        tt = time.perf_counter()
        try:
            r = fit_profile(prof, run.conditions, run.orientation, run.initial, grid, run.kriging, run.diffusivity)
        except Exception as e:
            print(f"  k={k}: failed {e}"); continue
        el = time.perf_counter() - tt
        rows.append(dict(run=run.name, k=k, dx_um=dx, ell_um=ell, time_days=r.time_days,
                         low_days=r.time_days + r.time_minus_days, high_days=r.time_days + r.time_plus_days, seconds=el))
        print(f"  k={k:4}: dx {dx:7.3f} um  t {r.time_days:9.4f} d  range {r.time_days + r.time_minus_days:8.3f}-{r.time_days + r.time_plus_days:8.3f}  {el:6.2f} s", flush=True)
write_rows(rows, Path(sys.argv[1]).name)
