"""Curves for the growth-stages schematic, version 2, computed with FORSTER's own model.

Same idea as ../../../growth-stages (instantaneous growth stages, each new zone then
diffusing with everything grown before), but with two growth stages, so that its zones
match profile 1 of the joint-fit schematic ("near c, high D"): an edge, then a rim zone,
a mantle and a core, with steps at the joint-fit profile's positions. The geometry is
read from ../../../joint-fit/figure/data/profiles.json, never typed in:

- step positions: p1["steps"] (distance from the true rim, um);
- profile length: p1["x_end"] (um);
- levels: meta["edge"] and meta["segments"] (rim, mantle, core; arbitrary units), mapped
  linearly to Fo with the edge at FO_EDGE = 0.80 and the core at FO_CORE = 0.90.

Only the zone geometry is reused (steps and levels), not the joint-fit curve, its
diffusivity or its time. History (1200 C, along [001], Dohmen & Chakraborty 2007):

- stage 0: the core crystal (from the inner step inward), its edge held at the mantle Fo
  (the melt the mantle will grow from);
- stage 1 at STAGE1_DAYS: the mantle grows (outer step to inner step) at the mantle Fo,
  rim held at the mantle Fo;
- stage 2 at STAGE2_DAYS: the rim zone grows (rim to outer step) at the rim-zone Fo,
  with the edge then held at the edge Fo (Stage's rim_value differs from zone_value);
- observed at OBS_DAYS.

The "measured" profile is synthetic: an independent flux-form solver (not FORSTER), on
the joint-fit profile's spacing (2 um), with seeded noise 0.001 Fo. Illustrative only.

Run read-only from the repo:
  cd <repo> && PYTHONDONTWRITEBYTECODE=1 uv run --frozen python <this file> <out.json>
"""

import json
import sys
import warnings
from pathlib import Path

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
from forster.model import near_step_at
from forster.uncertainty import NEAR_STEP_AIC

DAY = 86400.0

# ---- zone geometry from joint-fit profile 1 ------------------------------------------------------
JOINT = Path(__file__).resolve().parents[3] / "joint-fit" / "figure" / "data" / "profiles.json"
_J = json.loads(JOINT.read_text())
_P1, _META = _J["p1"], _J["meta"]
STEP_OUTER_UM, STEP_INNER_UM = (float(s) for s in _P1["steps"])  # rim zone|mantle, mantle|core
PROFILE_UM = float(_P1["x_end"])
SPACING_UM = float(_META["spacing"])
FO_EDGE, FO_CORE = 0.80, 0.90
_lo, _hi = float(_META["edge"]), float(_META["segments"]["core"])


def to_fo(level):
    return FO_EDGE + (float(level) - _lo) / (_hi - _lo) * (FO_CORE - FO_EDGE)


RIM_FO = to_fo(_META["segments"]["rim"])
MANTLE_FO = to_fo(_META["segments"]["mantle"])
assert abs(to_fo(_META["segments"]["core"]) - FO_CORE) < 1e-12

# ---- the illustrative history (choices, not data) ------------------------------------------------
STAGE1_DAYS, STAGE2_DAYS, OBS_DAYS, END_DAYS = 0.5, 1.25, 1.4, 2.0
DX_UM = 0.5
REF_DX_UM = 0.1  # independent solver's spacing
DOMAIN_END_UM = 120.0  # far end held at the core Fo; beyond the reach of diffusion here
X_UM = np.arange(0.0, DOMAIN_END_UM + 1e-9, DX_UM)
CONDITIONS = Conditions(T_C=1200, P_Pa=4.2e7, fO2_Pa=0.00138)
ALONG_001 = Orientation(90, 90, 0)
LAW = DIFFUSIVITY_PRESETS["dohmen_chakraborty2007"]

CORE = InitialCondition(
    (STEP_INNER_UM, DOMAIN_END_UM), edge_fo=MANTLE_FO, core_fo=FO_CORE, steps_um=(STEP_INNER_UM,)
)
MANTLE = Stage(STAGE1_DAYS * DAY, rim_um=STEP_OUTER_UM, zone_value=MANTLE_FO)
RIMZONE = Stage(STAGE2_DAYS * DAY, rim_um=0.0, rim_value=FO_EDGE, zone_value=RIM_FO)
STAGES = [MANTLE, RIMZONE]
# Comparison: every zone present from time zero, edge held at the edge Fo (a zero-width
# first segment carries the edge value, as stage 0 of the original figure does).
SINGLE = InitialCondition(
    (0.0, DOMAIN_END_UM),
    segment_fo=(FO_EDGE, RIM_FO, MANTLE_FO, FO_CORE),
    steps_um=(0.0, STEP_OUTER_UM, STEP_INNER_UM),
)


def d_fo(fo):
    return LAW.functions(CONDITIONS, ALONG_001)[0](np.asarray(fo))


def flux_form_staged(t_end_s, dx_m, safety=0.2):
    """Independent solver: explicit flux form, D averaged on cell faces, three stages.
    Each rim is a node held at the rim value (the rim lies at its own distance, to
    within dx / 2), the far end a node held at the core Fo."""
    x = np.arange(0.0, DOMAIN_END_UM + 1e-9, dx_m * 1e6)
    inner = int(np.abs(x - STEP_INNER_UM).argmin())
    outer = int(np.abs(x - STEP_OUTER_UM).argmin())
    c = np.full(len(x), FO_CORE)
    h_max = safety * dx_m**2 / d_fo(np.array([FO_EDGE])).max()

    def run(c, lo, rim, t):
        c[lo] = rim
        n = int(np.ceil(t / h_max))
        h = t / n
        for _ in range(n):
            u = c[lo:]
            face = 0.5 * (d_fo(u[:-1]) + d_fo(u[1:]))
            u[1:-1] = u[1:-1] + h * np.diff(face * np.diff(u) / dx_m) / dx_m
        return c

    c = run(c, inner, MANTLE_FO, STAGE1_DAYS * DAY)
    c[outer:inner] = MANTLE_FO
    c = run(c, outer, MANTLE_FO, (STAGE2_DAYS - STAGE1_DAYS) * DAY)
    c[:outer] = RIM_FO
    return x, run(c, 0, FO_EDGE, t_end_s - STAGE2_DAYS * DAY)


def clean(a):
    return [None if not np.isfinite(v) else round(float(v), 6) for v in np.asarray(a)]


def diffusion_length(days, fo_a, fo_b):
    """sqrt(D t) (um) with D at the middle of the step's two Fo values."""
    return float(np.sqrt(d_fo(np.array([0.5 * (fo_a + fo_b)]))[0] * days * DAY) * 1e6)


def main(out_path):
    # 1. Snapshots of the staged history (full run, every row kept).
    dt = 0.2 * (DX_UM * 1e-6) ** 2 / d_fo(np.array([FO_EDGE])).max()
    grid = Grid(dx_um=DX_UM, dt_s=dt, max_time_days=OBS_DAYS)
    dummy = Profile(X_UM, np.full(len(X_UM), 0.85) + 1e-4 * np.sin(X_UM), name="dummy")
    run = fit_profile(
        dummy,
        CONDITIONS,
        ALONG_001,
        CORE,
        grid,
        stop_early=False,
        stages=STAGES,
        keep_curves=True,
    )
    x = run.model_x_um

    # the first row in which each new zone exists (its first time step already taken)
    def first_row(um):
        node = int(np.abs(x - um).argmin()) + 1
        return next(r for r, c in enumerate(run.curves) if np.isfinite(c[node]))

    row1, row2 = first_row(STEP_OUTER_UM), first_row(0.0)
    last = len(run.curves) - 1
    rows = {
        "t0": 0,
        "before_stage1": row1 - 1,
        "after_stage1": row1,
        "before_stage2": row2 - 1,
        "after_stage2": row2,
        "end": last,
    }
    snaps = {k: run.curves[r] for k, r in rows.items()}
    snap_days = {k: r * dt / DAY for k, r in rows.items()}

    # 2. Synthetic measured profile: independent solver, joint-fit spacing, seeded noise.
    x_ref, ref = flux_form_staged(OBS_DAYS * DAY, REF_DX_UM * 1e-6)
    points = np.arange(0.0, PROFILE_UM + 1e-9, SPACING_UM)
    rng = np.random.default_rng(5)
    fo_meas = np.interp(points, x_ref, ref) + rng.normal(0, 0.001, len(points))
    profile = Profile(points, fo_meas)
    # agreement of the two solvers at the observation time (a check, not drawn)
    solver_gap = float(np.nanmax(np.abs(np.interp(x, x_ref, ref) - snaps["end"])))

    # 3. Fits: staged (stages given) and single-stage (every zone from time zero).
    fit_grid = Grid(DX_UM, 30.0, max_time_days=END_DAYS)
    # FORSTER's own warnings on each fit are kept (they are part of what a user sees).
    with warnings.catch_warnings(record=True) as w_staged:
        warnings.simplefilter("always")
        staged = fit_profile(
            profile, CONDITIONS, ALONG_001, CORE, fit_grid, stop_early=False, stages=STAGES
        )
    with warnings.catch_warnings(record=True) as w_single:
        warnings.simplefilter("always")
        single = fit_profile(profile, CONDITIONS, ALONG_001, SINGLE, fit_grid, stop_early=False)
    fit_warnings = {
        "staged": [str(w.message) for w in w_staged],
        "single": [str(w.message) for w in w_single],
    }

    # 4. Can the misfit tell the two histories apart? FORSTER's near-step check runs on the
    # single-stage fit; it is skipped for staged runs, so run it here on the staged
    # residuals at the same steps, sized by the time since the last stage.
    def at_points(res):
        return np.interp(points, res.model_x_um, res.curve(res.indices[0]))

    staged_check = near_step_at(
        points,
        fo_meas,
        at_points(staged),
        SINGLE,
        staged.time_since_last_stage_days,
        LAW,
        CONDITIONS,
        ALONG_001,
    )
    near_step = {
        "warn_below": NEAR_STEP_AIC,
        "single_delta_aic": single.near_step_delta_aic,
        "staged_delta_aic": staged_check.delta_aic,
        "single_rms": float(np.sqrt(np.mean((fo_meas - at_points(single)) ** 2))),
        "staged_rms": float(np.sqrt(np.mean((fo_meas - at_points(staged)) ** 2))),
        "noise_sd": 0.001,
    }

    # 4b. How typical is the drawn noise draw? The same two checks over other seeds.
    sweep = []
    for seed in range(10):
        fo_s = np.interp(points, x_ref, ref) + np.random.default_rng(seed).normal(
            0, 0.001, len(points)
        )
        prof_s = Profile(points, fo_s)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            st = fit_profile(
                prof_s, CONDITIONS, ALONG_001, CORE, fit_grid, stop_early=False, stages=STAGES
            )
            si = fit_profile(prof_s, CONDITIONS, ALONG_001, SINGLE, fit_grid, stop_early=False)
        chk = near_step_at(
            points,
            fo_s,
            np.interp(points, st.model_x_um, st.curve(st.indices[0])),
            SINGLE,
            st.time_since_last_stage_days,
            LAW,
            CONDITIONS,
            ALONG_001,
        )
        sweep.append(
            {
                "seed": seed,
                "staged_days": st.time_days,
                "staged_since_last_days": st.time_since_last_stage_days,
                "single_days": si.time_days,
                "single_delta_aic": si.near_step_delta_aic,
                "staged_delta_aic": chk.delta_aic,
                "single_rms": float(
                    np.sqrt(
                        np.mean(
                            (fo_s - np.interp(points, si.model_x_um, si.curve(si.indices[0]))) ** 2
                        )
                    )
                ),
            }
        )
    near_step["seed_sweep"] = sweep
    near_step["single_flagged_in"] = sum(r["single_delta_aic"] < NEAR_STEP_AIC for r in sweep)
    near_step["staged_flagged_in"] = sum(r["staged_delta_aic"] < NEAR_STEP_AIC for r in sweep)
    near_step["sweep_draws"] = len(sweep)

    # 5. Diffusion length sqrt(D t) of each step at the observation time.
    since2 = OBS_DAYS - STAGE2_DAYS
    lengths = {
        "inner_step_um": diffusion_length(OBS_DAYS, MANTLE_FO, FO_CORE),
        "outer_step_um": diffusion_length(since2, RIM_FO, MANTLE_FO),
        "edge_um": diffusion_length(since2, FO_EDGE, RIM_FO),
        "inner_age_days": OBS_DAYS,
        "outer_age_days": since2,
        "definition": "sqrt(D t), D (Dohmen & Chakraborty 2007, along [001], 1200 C) at the "
        "mean Fo of the step's two sides, t = time since the step formed",
    }

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
        "note": "Illustrative staged case with the zone geometry of joint-fit profile 1; "
        "computed with forster.fit_profile (Dohmen & Chakraborty 2007, along [001]).",
        "case": {
            "core_fo": FO_CORE,
            "mantle_fo": MANTLE_FO,
            "rim_fo": RIM_FO,
            "edge_fo": FO_EDGE,
            "step_outer_um": STEP_OUTER_UM,
            "step_inner_um": STEP_INNER_UM,
            "profile_um": PROFILE_UM,
            "spacing_um": SPACING_UM,
            "stage1_start_days": STAGE1_DAYS,
            "stage2_start_days": STAGE2_DAYS,
            "observed_days": OBS_DAYS,
            "fit_max_days": END_DAYS,
            "T_C": 1200,
            "P_Pa": 4.2e7,
            "fO2_Pa": 0.00138,
            "dx_um": DX_UM,
            "dt_snap_s": dt,
            "solver_gap_fo": solver_gap,
        },
        "x_um": clean(x),
        "snapshots": {k: clean(v) for k, v in snaps.items()},
        "snapshot_days": snap_days,
        "measured": {"x_um": clean(points), "fo": clean(fo_meas)},
        "staged_fit": {**summary(staged), "misfit": misfit(staged)},
        "single_fit": {**summary(single), "misfit": misfit(single)},
        "near_step": near_step,
        "fit_warnings": fit_warnings,
        "diffusion_lengths": lengths,
    }
    with open(out_path, "w") as f:
        json.dump(out, f)
    print("Fo: edge", FO_EDGE, "rim", RIM_FO, "mantle", MANTLE_FO, "core", FO_CORE)
    print("steps", STEP_OUTER_UM, STEP_INNER_UM, "profile", PROFILE_UM, "n", len(points))
    print("near_step", {k: v for k, v in near_step.items() if k != "seed_sweep"})
    print("warnings", json.dumps(fit_warnings, indent=1))
    print("lengths", lengths)
    print("solver gap (Fo)", solver_gap)
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
