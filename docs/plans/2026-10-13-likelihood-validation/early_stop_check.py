"""Early stopping gives the same fit as the full run, on every real profile, and how much faster.

For each of the 17 round robin and 18 Towbin and Plank (2026) runs, fits with
``stop_early=True`` (the default) and ``False`` and compares the best fit, the 1-sigma
range and the noise model; reports the time modelled and the wall time of each.

Writes early_stop.csv next to this file (paper rows only; the full table, with the
round robin rows, goes to the private round robin submodule). Run from the repo root:
    uv run python docs/plans/2026-10-13-likelihood-validation/early_stop_check.py
"""

from __future__ import annotations

import sys
import time
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

from forster import RunDefaults, fit_profile, read_profiles, read_runs  # noqa: E402
from forster.model import Profile  # noqa: E402

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


def fit(run, profiles, stop_early):
    measured = profiles[run.profile_name]
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        r = fit_profile(
            Profile(measured.distance_um, measured.fo, name=run.name),
            run.conditions,
            run.orientation,
            run.initial,
            run.grid,
            run.kriging,
            run.diffusivity,
            run.scheme,
            run.uncertainty,
            stop_early=stop_early,
        )
    notes = sorted({str(w.message)[:60] for w in caught if issubclass(w.category, UserWarning)})
    return r, time.perf_counter() - t0, notes


rows = []
for label, (runs, profiles) in SETS.items():
    for run in runs:
        run = replace(run, uncertainty="correlated")
        full, t_full, w_full = fit(run, profiles, False)
        early, t_early, w_early = fit(run, profiles, True)
        rows.append(
            {
                "set": label,
                "run": run.name,
                "two_steps": run.initial.step_fo is not None,
                "same_best": early.indices[0] == full.indices[0],
                "low_diff_s": early.time_minus_s - full.time_minus_s,
                "high_diff_s": early.time_plus_s - full.time_plus_s,
                "same_noise": early.noise == full.noise,
                "same_warnings": w_early == w_full,
                "time_days": full.time_days,
                "modelled_days_full": full.modelled_days,
                "modelled_days_early": early.modelled_days,
                "seconds_full": t_full,
                "seconds_early": t_early,
            }
        )
        print(
            f"{run.name:40s} same={rows[-1]['same_best'] and rows[-1]['same_noise']} "
            f"{t_full:5.2f}s -> {t_early:5.2f}s ({full.modelled_days:.0f} -> "
            f"{early.modelled_days:.1f} d)",
            flush=True,
        )
        del full, early

table = pd.DataFrame(rows)
write_table(table, "early_stop.csv", float_format="%.6g")
print(
    f"identical best fit and noise model: {int((table.same_best & table.same_noise).sum())} of "
    f"{len(table)}; max |range end difference| "
    f"{max(table.low_diff_s.abs().max(), table.high_diff_s.abs().max()):.3g} s; "
    f"same warnings: {int(table.same_warnings.sum())}; total time "
    f"{table.seconds_full.sum():.1f} s -> {table.seconds_early.sum():.1f} s"
)
