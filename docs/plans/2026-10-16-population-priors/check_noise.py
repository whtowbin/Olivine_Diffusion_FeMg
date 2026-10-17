"""Is the WHT06 blow-up the prior or the noise model? Refit with the noise model fixed.

`forster.sampling.fit_initial_condition` fits its noise model (white + correlated misfit)
first at the prior's centre, then again at the optimum found under that first noise
model. In the population-prior fits that came out worse (the WHT06 blow-ups, Ol_63 and
Ol_180), the noise model grew 1.3-11 times over the wide-prior fit's. This script refits those
cases with the noise model held at the wide-prior fit's (by replacing
`ICLikelihood.fit_noise` in this process only; FORSTER itself is unchanged). If the
time posterior then stays narrow, the blow-up came from the noise model being fitted at
the prior's centre, not from the prior alone.

    uv run python docs/plans/2026-10-16-population-priors/check_noise.py

Needs fits_cache.json from analyse.py (read only); its own fits go to noise_cache.json.
Writes noise_check.csv. A few minutes.
"""

from __future__ import annotations

import sys
import warnings
from dataclasses import replace
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import analyse as A  # noqa: E402

import forster  # noqa: E402
from forster.model import Grid  # noqa: E402
from forster.sampling import ICLikelihood  # noqa: E402
from forster.scan import _with_values  # noqa: E402

CASES = [  # (case, run, segment)
    ("full", "AZ18 WHT06_ol_overgrowth1_prof_1", "b"),
    ("full", "AZ18 WHT06_ol48_prof 1", "b"),
    ("truncated", "AZ18 WHT01_Ol_63_prof_1", "b"),
    ("truncated", "AZ18 WHT06_ol48_prof 1", "b"),
    ("full", "AZ18 WHT01_Ol_63_prof_1", "b"),
    ("full", "AZ18 WHT01_Ol_180_prof_1", "b"),
    ("truncated", "AZ18 WHT01_Ol_180_prof_1", "b"),
]


def main() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runs = {r.name: r for r in forster.read_runs(A.PARAMS)}
        profiles = forster.read_profiles(A.DATA, sheet="Olivine")
    cache = A.Cache(A.CACHE)  # read only
    own = A.Cache(HERE / "noise_cache.json")
    refits = pd.read_csv(HERE / "refits.csv")
    original = ICLikelihood.fit_noise
    rows = []
    for case, name, segment in CASES:
        run = runs[name]
        prof = profiles[run.profile_name]
        pick = (refits["run"] == name) & (refits["case"] == case) & (refits["role"] == "core")
        r = refits[pick].iloc[0]
        if case == "full":
            ic0, wide_key = run.initial, f"wide|{name}"
            grid = Grid(*cache.data[wide_key]["grid"])
        else:
            ic0 = replace(run.initial, domain_um=(run.initial.domain_um[0], r["truncated_end_um"]))
            wide_key = f"trunc_wide|{name}"
            grid = Grid(*cache.data[f"grid_trunc|{name}"]["grid"])
        w = cache.data[wide_key]
        unc = A.wide_uncertainties(ic0)
        # the wide fit's final noise model: refitted at its best values, as it ended
        like = ICLikelihood(
            prof, run.conditions, run.orientation, ic0, unc, grid, run.diffusivity, run.scheme
        )
        fixed = like.fit_noise(_with_values(ic0, w["best_values"]))
        print(name, case, "noise refit", fixed, "cached", w["noise_white"], w["noise_corr"])
        ic, u = A.with_prior(ic0, unc, segment, r["prior_mean"], r["prior_sd"])

        def make(run=run, prof=prof, ic=ic, u=u, grid=grid, fixed=fixed):
            ICLikelihood.fit_noise = lambda self, initial: self.use_noise(fixed)
            try:
                return A.run_fit(run, prof, ic, u, grid)[0]
            finally:
                ICLikelihood.fit_noise = original

        rec = own.get(f"fixednoise_pop|{case}|{name}", make)
        rows.append(
            {"case": case, "run": name, "noise_white": fixed.white_sigma}
            | {"noise_corr": fixed.correlated_sigma, "noise_L_um": fixed.correlation_length_um}
            | {f"wide_{k}": v for k, v in A.summarise_fit(w, segment).items()}
            | {
                f"pop_{k}": r[f"pop_{k}"]
                for k in ("best_days", "low_days", "high_days", "seg_mean")
            }
            | {f"fixed_{k}": v for k, v in A.summarise_fit(rec, segment).items()}
        )
    pd.DataFrame(rows).to_csv(HERE / "noise_check.csv", index=False, float_format="%.5g")


if __name__ == "__main__":
    main()
