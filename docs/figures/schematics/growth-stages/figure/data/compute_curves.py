"""Curves for the growth-stages schematic, computed with FORSTER's own model.

The case is the one in the repo's growth-stage plan and tests (tests/test_growth.py,
docs/plans/2026-10-13-growth-stages.md): a crystal [40, 200] um of Fo 0.90 whose rim is
held at Fo 0.86 for 6 days (stage 0), then an overgrowth [0, 40) um of Fo 0.82 with the
rim held at 0.82 (stage 1), observed at 10 days, along [001], Dohmen & Chakraborty 2007.
The "measured" profile is synthetic (independent flux-form solver, 66 points, noise
0.001, seed 5), as in the test. Illustrative only.

Run read-only from the repo:
  cd <repo> && PYTHONDONTWRITEBYTECODE=1 uv run --frozen python <this file> <out.json>
"""

import json
import sys

import numpy as np

from forster import (
    DIFFUSIVITY_PRESETS,
    Conditions,
    Grid,
    InitialCondition,
    Orientation,
    Profile,
    fit_profile,
)
from forster.boundary import Stage

DAY = 86400.0
DX_UM = 1.0
X_UM = np.arange(0.0, 200.0, DX_UM)
CONDITIONS = Conditions(T_C=1200, P_Pa=4.2e7, fO2_Pa=0.00138)
ALONG_001 = Orientation(90, 90, 0)
CORE_FO, EARLY_RIM, ZONE = 0.90, 0.86, 0.82
STAGE_DAYS, TOTAL_DAYS, END_DAYS = 6.0, 10.0, 16.0
EARLY = InitialCondition((40, 199), edge_fo=EARLY_RIM, core_fo=CORE_FO, steps_um=(40,))
GROWTH = Stage(STAGE_DAYS * DAY, rim_um=0.0, zone_value=ZONE)
SINGLE = InitialCondition((0, 199), edge_fo=ZONE, core_fo=CORE_FO, steps_um=(40,))


def d_fo(fo):
    return DIFFUSIVITY_PRESETS["dohmen_chakraborty2007"].functions(CONDITIONS, ALONG_001)[0](
        np.asarray(fo)
    )


def flux_form_staged(t_end_s, dx_m, safety=0.05):
    """Independent solver (copied from tests/test_growth.py)."""
    x = np.arange(0.0, 200.0, dx_m * 1e6)
    first = int(np.abs(x - 40).argmin())
    c = np.full(len(x), CORE_FO)
    h_max = safety * dx_m**2 / d_fo(np.array([ZONE])).max()

    def run(c, lo, rim, t):
        n = int(np.ceil(t / h_max))
        h = t / n
        for _ in range(n):
            full = np.concatenate([[rim], c[lo:], [CORE_FO]])
            face = 0.5 * (d_fo(full[:-1]) + d_fo(full[1:]))
            c[lo:] = c[lo:] + h * np.diff(face * np.diff(full) / dx_m) / dx_m
        return c

    c = run(c, first, EARLY_RIM, STAGE_DAYS * DAY)
    c[:first] = ZONE
    return x, run(c, 0, ZONE, t_end_s - STAGE_DAYS * DAY)


def clean(a):
    return [None if not np.isfinite(v) else round(float(v), 6) for v in np.asarray(a)]


def main(out_path):
    # 1. Snapshots of the staged history (full run, every row kept).
    dt = 0.2 * (DX_UM * 1e-6) ** 2 / d_fo(np.array([ZONE])).max()
    grid = Grid(dx_um=DX_UM, dt_s=dt, max_time_days=TOTAL_DAYS)
    dummy = Profile(X_UM, np.full(len(X_UM), 0.88) + 1e-4 * np.sin(X_UM), name="dummy")
    run = fit_profile(
        dummy,
        CONDITIONS,
        ALONG_001,
        EARLY,
        grid,
        stop_early=False,
        stages=[GROWTH],
        scheme="central",
        keep_curves=True,
    )
    x = run.model_x_um
    stage_row = round(STAGE_DAYS * DAY / dt)
    last = len(run.curves) - 1
    snaps = {
        "t0": run.curves[0],
        "before_stage1": run.curves[stage_row - 1],
        "after_stage1": run.curves[stage_row],
        "end": run.curves[last],
    }
    snap_days = {
        "t0": 0.0,
        "before_stage1": (stage_row - 1) * dt / DAY,
        "after_stage1": stage_row * dt / DAY,
        "end": last * dt / DAY,
    }
    # Fo along the profile at a few more times (for the crystal colour maps).
    more = {f"day{d:g}": run.curves[min(last, round(d * DAY / dt))] for d in (2, 4, 8)}

    # 2. Synthetic measured profile, as in test_fit_recovers_the_time_of_a_staged_profile.
    x_ref, ref = flux_form_staged(TOTAL_DAYS * DAY, DX_UM * 1e-6)
    points = np.arange(1.0, 199.0, 3.0)
    rng = np.random.default_rng(5)
    fo_meas = np.interp(points, x_ref, ref) + rng.normal(0, 0.001, len(points))
    profile = Profile(points, fo_meas)

    # 3. Fits: staged (stages given) and single-stage (both zones from time zero).
    fit_grid = Grid(DX_UM, 60.0, max_time_days=END_DAYS)
    staged = fit_profile(
        profile, CONDITIONS, ALONG_001, EARLY, fit_grid, stop_early=False, stages=[GROWTH]
    )
    single = fit_profile(profile, CONDITIONS, ALONG_001, SINGLE, fit_grid, stop_early=False)

    def misfit(res):
        t = np.arange(len(res.sum_chi2)) * res.dt_s / DAY
        step = max(1, len(t) // 400)
        return {"t_days": clean(t[::step]), "chi2": clean(res.sum_chi2[::step])}

    def summary(res):
        return {
            "time_days": res.time_days,
            "minus_days": res.time_minus_s / DAY,
            "plus_days": res.time_plus_s / DAY,
            "since_last_days": res.time_since_last_stage_days,
            "chi2_min": float(np.nanmin(res.sum_chi2)),
            "best_curve": clean(res.curve(res.indices[0])),
            "model_x_um": clean(res.model_x_um),
            "n_points": len(res.fit_x_um) if res.fit_x_um is not None else None,
        }

    out = {
        "note": "Illustrative staged case from FORSTER's growth-stage tests; computed with "
        "forster.fit_profile (Dohmen & Chakraborty 2007, along [001]).",
        "case": {
            "core_fo": CORE_FO,
            "early_rim_fo": EARLY_RIM,
            "zone_fo": ZONE,
            "stage0_rim_um": 40.0,
            "stage1_rim_um": 0.0,
            "stage1_start_days": STAGE_DAYS,
            "observed_days": TOTAL_DAYS,
            "T_C": 1200,
            "P_Pa": 4.2e7,
            "fO2_Pa": 0.00138,
            "dt_snap_s": dt,
        },
        "x_um": clean(x),
        "snapshots": {k: clean(v) for k, v in {**snaps, **more}.items()},
        "snapshot_days": snap_days,
        "measured": {"x_um": clean(points), "fo": clean(fo_meas)},
        "staged_fit": {**summary(staged), "misfit": misfit(staged)},
        "single_fit": {**summary(single), "misfit": misfit(single)},
    }
    with open(out_path, "w") as f:
        json.dump(out, f)
    for k in ("staged_fit", "single_fit"):
        s = out[k]
        print(
            k,
            {
                q: s[q]
                for q in (
                    "time_days",
                    "minus_days",
                    "plus_days",
                    "since_last_days",
                    "chi2_min",
                    "n_points",
                )
            },
        )
    print("snapshot days", snap_days, "rows", len(run.curves), "dt", dt)


if __name__ == "__main__":
    main(sys.argv[1])
