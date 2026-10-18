"""The interval rules and the step scan on a few real profiles.

For each profile: the legacy rule; the correlated likelihood with a rise of 1 (the
default before 2026-10-13), with the small-sample threshold (the default since), and
the Bayesian posterior with the noise scale integrated out; and a scan of the step
position over +-6 um in 1 um steps (`forster.scan`), with its profile-likelihood and
marginal ranges. Times in days.

Writes real_methods.csv next to this file (paper rows only; the full table, with the
round robin rows, goes to the private round robin submodule). Run from the repo root:
    uv run python docs/plans/2026-10-13-likelihood-validation/compare_real.py
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import replace
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "tests"))
from conftest import (  # noqa: E402
    PAPER_DATA,
    PAPER_PARAMS,
    ROUND_ROBIN_DATA,
    ROUND_ROBIN_PARAMS,
)

from forster import RunDefaults, fit_run, read_profiles, read_runs  # noqa: E402
from forster.model import Profile  # noqa: E402
from forster.scan import scan_initial_conditions, step_offsets  # noqa: E402
from forster.uncertainty import delta_chi2_interval, posterior_interval  # noqa: E402

PRIVATE = (
    HERE.parents[2]
    / "Diffusion Round Robin"
    / "outputs"
    / "public_tree_moved"
    / HERE.relative_to(HERE.parents[2])
)


def write_table(table, name, **kwargs):
    """The round robin rows are private (other labs' data): the full table goes to the
    round robin submodule, the public copy next to this file keeps the paper rows."""
    PRIVATE.mkdir(parents=True, exist_ok=True)
    table.to_csv(PRIVATE / name, index=False, **kwargs)
    table[table["set"] != "round robin"].to_csv(HERE / name, index=False, **kwargs)


DAY = 86400.0
PICKS = {
    "round robin": ["KS20-527_2-transect-1", "KS20-527_2-transect-3", "SH63olv066-1"],
    "Towbin and Plank 2026": [
        "AZ18 WHT01_Ol_63_prof_1",
        "AZ18 WHT01_Ol_192_prof_1",
        "Xenocryst_ol6_prof_1",
    ],
}
SETS = {
    "round robin": (
        read_runs(ROUND_ROBIN_PARAMS, defaults=RunDefaults(max_time_days=730)),
        read_profiles(ROUND_ROBIN_DATA),
    ),
    "Towbin and Plank 2026": (
        read_runs(PAPER_PARAMS),
        read_profiles(PAPER_DATA, sheet="Olivine"),
    ),
}

rows = []
for label, (runs, profiles) in SETS.items():
    by_name = {r.name: r for r in runs}
    for name in PICKS[label]:
        run = replace(by_name[name], uncertainty="correlated")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = fit_run(run, profiles, stop_early=False)  # posterior needs the whole curve
            legacy = fit_run(replace(run, uncertainty="legacy"), profiles)
            one = delta_chi2_interval(res.sum_chi2, 1.0, end_is_limit=False)
            post = posterior_interval(res.sum_chi2, len(res.fit_x_um))
            measured = profiles[run.profile_name]
            # +-6 um, but never past the domain ends.
            lo, hi = run.initial.domain_um
            offsets = [
                d for d in range(-6, 7) if all(lo <= s + d <= hi for s in run.initial.steps_um)
            ]
            scan = scan_initial_conditions(
                Profile(measured.distance_um, measured.fo),
                run.conditions,
                run.orientation,
                step_offsets(run.initial, offsets),
                run.grid,
                run.diffusivity,
            )
        dt = res.dt_s / DAY
        common = {"set": label, "run": run.name, "n_points": len(res.fit_x_um)}
        ranges = {
            "legacy": (
                legacy.time_days,
                legacy.time_days + legacy.time_minus_days,
                legacy.time_days + legacy.time_plus_days,
            ),
            "delta chi2 = 1": (one.best * dt, one.low * dt, one.high * dt),
            "small-sample (default)": (
                res.time_days,
                res.time_days + res.time_minus_days,
                res.time_days + res.time_plus_days,
            ),
            "Bayesian posterior": (post.best * dt, post.low * dt, post.high * dt),
            "step scan, profile": (scan.time_days, *scan.profile_days),
            "step scan, marginal": (scan.time_days, *scan.marginal_days),
        }
        for method, (best, low, high) in ranges.items():
            rows.append(
                {**common, "method": method, "best_days": best, "low_days": low, "high_days": high}
            )
        rows[-1]["best_steps_um"] = ", ".join(f"{s:g}" for s in scan.best_initial.steps_um)
        rows[-1]["nominal_steps_um"] = ", ".join(f"{s:g}" for s in run.initial.steps_um)
        print(f"{name}: best step {scan.best_initial.steps_um} (nominal {run.initial.steps_um})")

table = pd.DataFrame(rows)
write_table(table, "real_methods.csv", float_format="%.4g")
pd.set_option("display.width", 200)
print(table.round(2).to_string(index=False))
