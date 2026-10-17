"""Does the joint fit's nested-sampling fallback rule pick the right sampler on real crystals?

`forster.fit_crystal` samples the posterior of a crystal's starting-profile parameters by
Laplace importance sampling and falls back to nested sampling when the importance
samples' effective sample size (ESS) is too low (`forster.sampling.sample_posterior`,
``scaled=True``). This script measures, for each case (one crystal with one prior and
offset setting):

1. ``fit_crystal(method="auto")`` with the defaults: the method it chose, ESS, wall time;
2. Laplace importance sampling alone at 1024, 2048 and 4096 samples, seeds 0-2, on the
   same likelihood and best fit (`forster.sampling._importance`, called directly:
   ``fit_crystal(method="laplace")`` still doubles the samples under the scaled rule, so
   it cannot give a fixed sample count);
3. a nested-sampling reference (`forster.sampling._nested`, nlive 1000, seeds 0 and 1:
   two seeds measure the reference's own noise; with the edge offsets on, one run at
   nlive 500, `NESTED_PLAN`).

The likelihood and best fit are rebuilt with `forster.joint`'s own helpers, as
`fit_crystal` builds them, and checked against the auto fit's best values.

Results are cached per case in ``cache_<set>.json`` (``--extra`` rows: 7 more Laplace seeds
and fit_crystal's default nested sampling, nlive 200, in ``extra_<set>.json``);
``diagnostics.py`` adds two other diagnostics and ``rules.py`` scores the fallback rules.
Run from the repository root:

    uv run python -u docs/plans/2026-10-16-joint-calibration/calibrate.py [--set paper|stress]

``--set paper``: the two paper crystals with two profiles each (public data);
``--set stress``: a pseudo-crystal of 4 profiles from different crystals of one
xenolith, a sampler stress test at more parameters, not a geological result.
One process; nested sampling one run at a time.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np

import forster
from forster import joint
from forster.sampling import (
    DEFAULT_NLIVE,
    ESS_FALLBACK,
    _importance,
    _nested,
    _summarise,
    ideal_ess_fraction,
)
from forster.scan import Uncertainty

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PAPER = REPO / "analyses" / "xenolith_archive"
PARAMS = PAPER / "Olivine_Profile_Diffusion_Modeling_Parameters_DB.xlsx"
DATA = PAPER / "Xenolith_MeltVein_EMPA_Master.xlsx"

SAMPLE_COUNTS = (1024, 2048, 4096)
SEEDS = (0, 1, 2)
# Nested-sampling references: (nlive, seed) per setting. With the edge offsets on, one
# nlive-1000 run took 2 h (GCB1A), so those cases get one nlive-500 run; the two seeds
# elsewhere measure the reference's own noise.
NESTED_PLAN = {"edge": ((500, 0),)}
NESTED_DEFAULT = ((1000, 0), (1000, 1))
# --extra: more Laplace seeds (Monte Carlo noise vs bias) and the fallback's own nested
# sampling at fit_crystal's default nlive (is falling back more accurate?).
EXTRA_SEEDS = tuple(range(3, 10))
FALLBACK_SEEDS = (0, 1, 2)
THIN_UM = 2.0  # a step this close to the edge marks the held edge value: kept fixed

# Crystals with two or more profiles among the paper runs (same name stem, same T, P,
# fO2; checked by eye: same zoning shape, prof_2 of GCB1A about 0.3 mol% higher).
PAPER_CRYSTALS = {
    "GCB1A R31_MP_Ol5": ("GCB1A R31_MP_Ol5_prof_1", "GCB1A R31_MP_Ol5_prof_2"),
    "Xenocryst_ol6": ("Xenocryst_ol6_prof_1", "Xenocryst_ol6_prof_2"),
}
# Not one crystal: profiles of different olivines of xenolith AZ18 WHT01 with the same two
# zones (rim about Fo 91.5, core about Fo 89.7), joined to reach more parameters.
STRESS_CRYSTALS = {
    "WHT01 pseudo-4": (
        "AZ18 WHT01_bigol2_overgrowth_prof_1",
        "AZ18 WHT01_Ol_63_prof_1",
        "AZ18 WHT01_Ol_192_prof_1",
        "AZ18 WHT01_Ol_193_prof_1",
    ),
}

# The paper table gives every run 200 days of model time; Xenocryst_ol6's times are 0.1-0.2
# days, so its scan grid (time step sized for those times) would hold 134,000 time steps and
# each model run take 0.2 s (the best fit alone 260 s). Its scan runs to 2 days instead
# (10x the longest individual time; fit_crystal warns if the posterior reaches the end).
SCAN_MAX_DAYS = {"Xenocryst_ol6": 2.0}

# Prior and offset settings: (step sd um, segment Fo sd, fit_crystal keyword arguments).
SETTINGS = {
    "base": (3.0, 0.005, {}),
    "tight": (2.0, 0.003, {}),
    "edge": (3.0, 0.005, {"edge_offset_sd_um": joint.DEFAULT_EDGE_OFFSET_SD_UM}),
    "no-ref": (3.0, 0.005, {"fo_reference": None}),
}
STRESS_SETTINGS = ("base", "no-ref")  # the edge offsets make nested sampling take hours


def crystal_runs(names, runs, step_sd, fo_sd, crystal):
    """The crystal's runs with moderate priors: each step more than THIN_UM inside the
    edge Gaussian (sd step_sd), each segment Fo Gaussian (sd fo_sd); shared labels get
    the mean nominal Fo of the runs (fit_crystal needs them equal)."""
    chosen = [runs[n] for n in names]
    labels = chosen[0].initial.segment_names
    if any(r.initial.segment_names != labels for r in chosen):
        raise ValueError(f"{crystal}: segment labels differ")
    mean_fo = np.mean([r.initial.segment_fo for r in chosen], axis=0)
    out = []
    for r in chosen:
        initial = replace(r.initial, segment_fo=tuple(float(f) for f in mean_fo))
        unc = {f"{n}_fo": Uncertainty(fo_sd) for n in labels}
        for i, s in enumerate(initial.steps_um):
            if s - initial.domain_um[0] > THIN_UM:
                unc[f"step{i + 1}_um"] = Uncertainty(step_sd)
        scan_grid = r.scan_grid
        if crystal in SCAN_MAX_DAYS:
            scan_grid = replace(scan_grid, max_time_days=SCAN_MAX_DAYS[crystal])
        out.append(
            replace(
                r,
                initial=initial,
                scan_grid=scan_grid,
                uncertainties=unc,
                metadata={**dict(r.metadata), "crystal": crystal},
            )
        )
    return out


def rebuild(runs, profiles, individual, kwargs):
    """The joint likelihood and best fit exactly as `fit_crystal` builds them."""
    fo_sd = joint._per_run(joint.DEFAULT_FO_OFFSET_SD, runs, "fo_offset_sd")
    fo_sd = joint._with_reference(fo_sd, runs, profiles, kwargs.get("fo_reference", "auto"))
    edge_sd = joint._per_run(kwargs.get("edge_offset_sd_um", 0.0), runs, "edge_offset_sd_um")
    grids = joint._grids(runs, profiles, individual, None)
    like = joint._likelihood(runs, profiles, grids, {}, fo_sd, edge_sd)
    z, _ = joint._best_fit(like)
    return like, z


def shared_labels(names):
    return [n for n in names if ":" not in n and n.endswith("_fo")]


def summary_row(like, samples, ess, n, method, seed, seconds):
    s = _summarise(like, samples)
    row = {
        "method": method,
        "n": n,
        "seed": seed,
        "ess": ess,
        "ess_fraction": ess / n if n else math.nan,
        "p16_days": s["low_days"],
        "p50_days": s["median_days"],
        "p84_days": s["high_days"],
        "seconds": seconds,
    }
    for label in shared_labels(like.names):
        row[f"{label}_mean"] = s["mean"][label]
        row[f"{label}_sd"] = s["sd"][label]
    return row


def run_case(case, names, setting, runs, profiles, individual_cache):
    step_sd, fo_sd, kwargs = SETTINGS[setting]
    crystal = case.split(" | ")[0]
    rs = crystal_runs(names, runs, step_sd, fo_sd, crystal)
    key = (crystal, step_sd, fo_sd)
    out = {"case": case, "crystal": crystal, "setting": setting, "step_sd_um": step_sd}
    out["fo_sd"] = fo_sd
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t0 = time.perf_counter()
        fit = forster.fit_crystal(
            rs, profiles, fits=individual_cache.get(key), method="auto", **kwargs
        )
        auto_s = time.perf_counter() - t0
    individual_cache[key] = fit.individual
    out["warnings"] = sorted({str(w.message)[:160] for w in caught})
    d = len(fit.names)
    out["d"] = d
    out["names"] = list(fit.names)
    out["ideal_fraction"] = ideal_ess_fraction(d)
    out["individual_days"] = {n: f.time_days for n, f in fit.individual.items()}
    out["agree"] = fit.agree
    labels = shared_labels(fit.names)
    auto = {
        "method": f"auto->{fit.method}",
        "n": fit.n_samples,
        "seed": 0,
        "ess": fit.ess,
        "ess_fraction": fit.ess / fit.n_samples,
        "p16_days": fit.low_days,
        "p50_days": fit.median_days,
        "p84_days": fit.high_days,
        "seconds": auto_s,
    }
    for label in labels:
        auto[f"{label}_mean"] = fit.mean[label]
        auto[f"{label}_sd"] = fit.sd[label]
    rows = [auto]
    print(f"  auto: {fit.method}, d={d}, ESS {fit.ess:.0f}/{fit.n_samples}, {auto_s:.1f} s")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        like, z = rebuild(rs, profiles, fit.individual, kwargs)
        out["best_fit_seconds"] = time.perf_counter() - t0
    best = like.nominal + z * like.spread
    out["best_matches_auto"] = bool(
        np.allclose(best, [fit.best_values[n] for n in like.names], rtol=0, atol=1e-12)
    )
    for n in SAMPLE_COUNTS:
        for seed in SEEDS:
            t0 = time.perf_counter()
            samples, ess = _importance(like, z, n, seed, whiten=True)
            rows.append(
                summary_row(like, samples, ess, n, "laplace", seed, time.perf_counter() - t0)
            )
            print(f"  laplace n={n} seed={seed}: ESS {ess:.0f} ({ess / n:.3f})")
    out["rows"] = rows
    return out, like, z


def add_nested(c, names, runs, profiles, save, like=None):
    """Run the case's nested-sampling references not yet in its rows, saving after each."""
    plan = NESTED_PLAN.get(c["setting"], NESTED_DEFAULT)
    if c["crystal"] in STRESS_CRYSTALS:  # 30-60 min a run at 9-10 parameters: one seed
        plan = plan[:1]
    done = {(r.get("nlive"), r["seed"]) for r in c["rows"] if r["method"] == "nested"}
    for nlive, seed in plan:
        if (nlive, seed) in done:
            continue
        if like is None:
            step_sd, fo_sd, kwargs = SETTINGS[c["setting"]]
            rs = crystal_runs(names, runs, step_sd, fo_sd, c["crystal"])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                individual = {r.name: forster.fit_run(r, profiles) for r in rs}
                like, _ = rebuild(rs, profiles, individual, kwargs)
            if list(like.names) != c["names"]:
                raise RuntimeError(f"{c['case']}: parameters changed")
        t0 = time.perf_counter()
        runs_before = like.runs
        samples, _ = _nested(like, nlive, seed)
        sec = time.perf_counter() - t0
        row = summary_row(like, samples, math.nan, 0, "nested", seed, sec)
        row["nlive"] = nlive
        row["model_runs"] = like.runs - runs_before
        c["rows"].append(row)
        save()
        print(f"  nested nlive {nlive} seed={seed}: {sec:.0f} s", flush=True)


def extra_case(c, names, runs, profiles):
    """Rows for --extra, on the case's likelihood rebuilt as in run_case."""
    step_sd, fo_sd, kwargs = SETTINGS[c["setting"]]
    rs = crystal_runs(names, runs, step_sd, fo_sd, c["crystal"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        individual = {r.name: forster.fit_run(r, profiles) for r in rs}
        like, z = rebuild(rs, profiles, individual, kwargs)
    if list(like.names) != c["names"]:
        raise RuntimeError(f"{c['case']}: parameters changed")
    rows = []
    for n in SAMPLE_COUNTS:
        for seed in EXTRA_SEEDS:
            t0 = time.perf_counter()
            samples, ess = _importance(like, z, n, seed, whiten=True)
            rows.append(
                summary_row(like, samples, ess, n, "laplace", seed, time.perf_counter() - t0)
            )
        print(f"  laplace n={n} seeds {EXTRA_SEEDS[0]}-{EXTRA_SEEDS[-1]}", flush=True)
    # one seed with the edge offsets on: hours per run
    seeds = FALLBACK_SEEDS[:1] if c["setting"] == "edge" else FALLBACK_SEEDS
    if c["crystal"] in STRESS_CRYSTALS:  # left out to keep the stress set within an hour
        seeds = ()
    for seed in seeds:
        t0 = time.perf_counter()
        samples, _ = _nested(like, DEFAULT_NLIVE, seed)
        sec = time.perf_counter() - t0
        rows.append(summary_row(like, samples, math.nan, 0, f"nested{DEFAULT_NLIVE}", seed, sec))
        print(f"  nested nlive {DEFAULT_NLIVE} seed={seed}: {sec:.0f} s", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", choices=("paper", "stress"), default="paper")
    parser.add_argument("--only", help="run only cases whose name contains this")
    parser.add_argument("--extra", action="store_true", help="add the --extra rows to cached cases")
    args = parser.parse_args()
    out_dir = HERE
    cache_path = out_dir / f"cache_{args.set}.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runs = {r.name: r for r in forster.read_runs(PARAMS)}
        profiles = forster.read_profiles(DATA, sheet="Olivine")
    crystals = PAPER_CRYSTALS if args.set == "paper" else STRESS_CRYSTALS
    settings = SETTINGS if args.set == "paper" else STRESS_SETTINGS
    if args.extra:  # its own file, so it can run beside the main pass
        extra_path = out_dir / f"extra_{args.set}.json"
        extra = json.loads(extra_path.read_text()) if extra_path.exists() else {}
        for case, c in cache.items():
            if case in extra or (args.only and args.only not in case):
                continue
            print(case, "(extra)", flush=True)
            extra[case] = extra_case(c, crystals[c["crystal"]], runs, profiles)
            extra_path.write_text(json.dumps(extra, indent=1))
        return
    individual_cache: dict = {}
    # the edge-offset cases last: their nested references take hours
    pairs = sorted(((c, s) for c in crystals for s in settings), key=lambda p: p[1] == "edge")
    for crystal, setting in pairs:
        names = crystals[crystal]
        case = f"{crystal} | {setting}"
        if args.only and args.only not in case:
            continue
        print(case, flush=True)

        def save():
            cache_path.write_text(json.dumps(cache, indent=1))

        like = None
        if case not in cache:
            cache[case], like, _ = run_case(case, names, setting, runs, profiles, individual_cache)
            save()
        add_nested(cache[case], names, runs, profiles, save, like)
    print(f"ESS_FALLBACK {ESS_FALLBACK}; cached {len(cache)} cases in {cache_path}")


if __name__ == "__main__":
    sys.exit(main())
