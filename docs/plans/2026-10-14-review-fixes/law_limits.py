"""Measurements behind docs/references/law-uncertainty-limits.md.

1. D t scaling: a synthetic Fo profile (made with `femg_refit2026` at the round robin
   RR1200 conditions, 6 days, noise 5e-4) fitted with the law and with D x 10^delta. Where
   the profile depends on D t only, log10 t moves by exactly -delta.
2. Linear propagation against a Monte Carlo: log10 D along the profile at Fo 0.85 from
   4000 draws of the preset's covariance, against `log10_sigma` (g' C g).

Writes dt_scaling.csv and linear_vs_mc.csv next to this file. Run from the repo root:
    uv run python docs/plans/2026-10-14-review-fixes/law_limits.py
"""

import math
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from forster import (
    DIFFUSIVITY_PRESETS,
    Conditions,
    Grid,
    InitialCondition,
    Orientation,
    Profile,
    fit_profile,
)
from forster.boundary import RimPath, Stage

HERE = Path(__file__).parent
DAY = 86400.0
LAW = DIFFUSIVITY_PRESETS["femg_refit2026"]
RR1200 = Conditions(T_C=1200, P_Pa=4.2e7, fO2_Pa=0.00138)
ALONG_001 = Orientation(90, 90, 0)
T_TRUE, DT = 6.0, 0.01 * DAY
POINTS = np.arange(1.0, 199.0, 2.0)


def shifted(delta):
    return replace(
        LAW, name=f"shift{delta}", log10_D0_m2_s=tuple(v + delta for v in LAW.log10_D0_m2_s)
    )


def synthetic(initial, **kw):
    x = np.arange(0.0, 200.0)
    dummy = Profile(x, np.full(len(x), 0.88) + 1e-4 * np.sin(x))
    run = fit_profile(
        dummy, RR1200, ALONG_001, initial, Grid(0.5, DT, T_TRUE), None, LAW, stop_early=False, **kw
    )
    fo = np.interp(POINTS, run.model_x_um, run.curves[round(T_TRUE * DAY / DT)])
    return Profile(POINTS, fo + np.random.default_rng(2).normal(0, 5e-4, len(POINTS)))


CASES = {
    "fixed rim (control)": (InitialCondition((0, 199), 0.82, 0.90, steps_um=(40,)), {}),
    "fixed rim_fo, not the edge Fo": (
        InitialCondition((0, 199), 0.86, 0.90, steps_um=(40,)),
        {"rim_fo": 0.82},
    ),
    "RimPath 0.86 to 0.82 over days 0-3": (
        InitialCondition((0, 199), 0.86, 0.90, steps_um=(40,)),
        {"rim_fo": RimPath((0.0, 3.0), (0.86, 0.82))},
    ),
    "RimPath 0.86 to 0.82 over days 0-0.5": (
        InitialCondition((0, 199), 0.86, 0.90, steps_um=(40,)),
        {"rim_fo": RimPath((0.0, 0.5), (0.86, 0.82))},
    ),
    "growth stage (Fo 0.82 zone) at day 3": (
        InitialCondition((40, 199), 0.86, 0.90, steps_um=(40,)),
        {"stages": [Stage(3.0 * DAY, rim_um=0.0, zone_value=0.82)]},
    ),
}


def dt_scaling() -> pd.DataFrame:
    rows = []
    grid = Grid(0.5, DT, 40)
    for label, (initial, kw) in CASES.items():
        profile = synthetic(initial, **kw)
        base = fit_profile(profile, RR1200, ALONG_001, initial, grid, None, LAW, **kw).time_days
        for delta in (0.122, 0.301):
            t = fit_profile(
                profile, RR1200, ALONG_001, initial, grid, None, shifted(delta), **kw
            ).time_days
            shift = math.log10(t / base)
            rows.append(
                {
                    "case": label,
                    "delta_log10_D": delta,
                    "t_days": base,
                    "t_shifted_days": t,
                    "shift_log10_t": shift,
                    "fraction_of_expected": -shift / delta,
                }
            )
    return pd.DataFrame(rows)


def linear_vs_mc(n=4000) -> pd.DataFrame:
    u = LAW.uncertainty
    assert u is not None
    index = {name: i for i, name in enumerate(u.parameters)}
    draws = np.random.default_rng(0).multivariate_normal(
        np.zeros(len(u.parameters)), np.array(u.covariance), n
    )
    d0 = LAW.log10_D0_m2_s

    def law(d):
        def draw(name):  # a parameter the covariance leaves out (fixed anisotropy) is exact
            return d[index[name]] if name in index else 0.0

        base = d0[2] + d[index["log10_D0_001"]]
        return replace(
            LAW,
            log10_D0_m2_s=(
                base + d0[0] - d0[2] + draw("log10_100_vs_001"),
                base + d0[1] - d0[2] + draw("log10_010_vs_001"),
                base,
            ),
            activation_energy_J_mol=tuple(
                e + d[index["activation_energy_J_mol"]] for e in LAW.activation_energy_J_mol
            ),
            fO2_exponent=LAW.fO2_exponent + d[index["fO2_exponent"]],
            fo_coefficient=LAW.fo_coefficient + d[index["fo_coefficient"]],
            activation_volume_m3_mol=LAW.activation_volume_m3_mol
            + draw("activation_volume_m3_mol"),
            uncertainty=None,
        )

    def log_d(lw, c, o):
        return math.log10(lw.functions(c, o)[0](np.array([0.85]))[0])

    rows = []
    conditions = {
        "1200 C, 0.042 GPa, RR1200 fO2": RR1200,
        "900 C, 0.1 GPa, 1e-9 Pa": Conditions(900, 1e8, 1e-9),
        "1300 C, 1 atm, 1e-2 Pa": Conditions(1300, 1e5, 1e-2),
    }
    orientations = {
        "[001]": Orientation(90, 90, 0),
        "[100]": Orientation(0, 90, 90),
        "oblique (50, 70, 46.5)": Orientation(50, 70, 46.5),
        "equal angles (54.7)": Orientation(54.7, 54.7, 54.7),
    }
    for c_label, c in conditions.items():
        for o_label, o in orientations.items():
            centre = log_d(LAW, c, o)
            mc = np.array([log_d(law(d), c, o) for d in draws]) - centre
            lo, hi = np.percentile(mc, [16, 84])
            rows.append(
                {
                    "conditions": c_label,
                    "orientation": o_label,
                    "linear_sigma": LAW.log10_sigma(c, o, 0.85),
                    "mc_sd": mc.std(),
                    "mc_half_16_84": (hi - lo) / 2,
                    "mc_mean": mc.mean(),
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    scaling = dt_scaling()
    scaling.to_csv(HERE / "dt_scaling.csv", index=False)
    print(scaling.round(4).to_string(index=False))
    mc = linear_vs_mc()
    mc.to_csv(HERE / "linear_vs_mc.csv", index=False)
    print(mc.round(4).to_string(index=False))
