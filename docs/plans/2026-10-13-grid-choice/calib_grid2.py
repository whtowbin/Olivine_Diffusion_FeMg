"""Separate step-snapping from resolution: steps on nodes (aligned) vs not, k up to 60."""
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
rows = []
for run, profiles in cases:
    m = profiles[run.profile_name]
    D, _ = fe_mg_diffusivity(run.diffusivity).functions(run.conditions, run.orientation)
    fos = [run.initial.edge_fo, run.initial.core_fo] + ([run.initial.step_fo] if run.initial.step_fo else [])
    d_mid, d_max = float(D(np.array([np.mean(fos)]))[0]), float(D(np.array([min(fos)]))[0])
    base = fit_profile(Profile(m.distance_um, m.fo), run.conditions, run.orientation, run.initial, run.grid, run.kriging, run.diffusivity)
    ell = np.sqrt(d_mid * base.time_s) * 1e6
    xmax = m.distance_um.max()
    print(f"\n{run.name}: l = {ell:.2f} um, steps {run.initial.steps_um}, profile to {xmax:.1f} um", flush=True)
    for k in (12, 20, 30, 40, 60):
        for aligned in (False, True):
            dx = ell / k
            if aligned:
                # Grid runs 0..xmax with n intervals; choose n so the first step is on a node.
                n = max(1, round(xmax / dx)); h = xmax / n
                s = run.initial.steps_um[0]
                if s > 0:
                    j = max(1, round(s / h)); h = s / j; n = round(xmax / h)
                    dx = xmax / n  # node spacing (forster's grid = linspace(0, xmax, n))
                    # shift so step falls on a node exactly: need s/h integer; recompute
                    # (approximate: alignment within h * |xmax/h - n|)
            grid = replace(run.grid, dx_um=dx, dt_s=0.4 * (dx * 1e-6) ** 2 / d_max, max_time_days=3 * base.time_s / 86400)
            tt = time.perf_counter()
            r = fit_profile(Profile(m.distance_um, m.fo), run.conditions, run.orientation, run.initial, grid, run.kriging, run.diffusivity)
            step_x = r.model_x_um
            offset = min(abs(step_x - run.initial.steps_um[0]).min(), 99)
            rows.append(dict(run=run.name, k=k, aligned=aligned, dx=dx, step_offset_um=offset, time_days=r.time_days, seconds=time.perf_counter() - tt))
            print(f"  k={k:3d} {'aligned' if aligned else 'as is  '} dx {dx:6.3f}  nearest node to step {offset:5.3f} um  t {r.time_days:9.4f} d  {rows[-1]['seconds']:5.2f} s", flush=True)
write_rows(rows, Path(sys.argv[1]).name)
