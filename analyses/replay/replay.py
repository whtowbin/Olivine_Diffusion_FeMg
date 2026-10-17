"""Differential replay: every run read from an existing table (the template example with
both starting-profile fit methods, the Monte Carlo and a summary round trip; every paper
run; every round robin run when the private submodule is checked out), fitted and
summarised, the outputs written to JSON (full curves as SHA-256 prefixes).

Run it on the old code and on the new, plus the old against itself (the control: it must
show 0 differences), then compare with compare.py. Write the JSON OUTSIDE the repository
(it holds round robin results): e.g. the session scratchpad.

usage (from the repository root): uv run python analyses/replay/replay.py OUT.json
"""
import hashlib, json, math, sys, tempfile, warnings
from pathlib import Path
sys.path.insert(0, "tests")
import numpy as np, pandas as pd
import forster
from forster import read_profiles, read_runs, write_template, fit_run, scan_run, scan_summary
from forster.io import summary_row, mc_run, mc_summary

def h(a):
    a = np.ascontiguousarray(np.asarray(a, dtype=float))
    return hashlib.sha256(a.tobytes()).hexdigest()[:16]

def clean(v):
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, (np.floating,)):
        return repr(float(v))
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (tuple, list)):
        return [clean(x) for x in v]
    if isinstance(v, dict):
        return {k: clean(x) for k, x in v.items()}
    return v if isinstance(v, (int, str, bool, type(None))) else repr(v)

def fit_record(r):
    rec = {"time_s": r.time_s, "minus": r.time_minus_s, "plus": r.time_plus_s,
           "minus2": r.time_minus2_s, "plus2": r.time_plus2_s, "indices": list(r.indices),
           "chi2": h(np.nan_to_num(r.sum_chi2, nan=-1)), "curves": h(r.curves),
           "model_x": h(r.model_x_um), "near_step": r.near_step_delta_aic,
           "core_end": getattr(r, "core_end_lengths", None)}
    if r.noise is not None:
        rec |= {"s": r.noise.white_sigma, "a": r.noise.correlated_sigma, "L": r.noise.correlation_length_um}
    row = summary_row(r)
    rec["summary"] = {k: v for k, v in row.items() if k not in ("forster_version",)}
    return rec

out = {}
warnings.simplefilter("ignore")
# template example: fit, sampler scan, grid scan, Monte Carlo
with tempfile.TemporaryDirectory() as d:
    d = Path(d)
    write_template(d)
    params = pd.read_csv(d / "parameters.csv")
    profiles = read_profiles(d / "profiles.csv")
    (run,) = read_runs(d / "parameters.csv")
    res = fit_run(run, profiles)
    out["template"] = fit_record(res)
    p2 = params.assign(steps_sd_um=2.0, edge_fo_sd=0.002, mc_T_sd_C=10, mc_samples=300)
    p2.to_csv(d / "p2.csv", index=False)
    (r2,) = read_runs(d / "p2.csv")
    post = scan_run(r2, profiles, time_days=res.time_days)
    out["template_scan"] = clean(scan_summary(post)) | {"best_initial": repr(post.best_initial)}
    out["template_mc"] = clean(mc_summary(mc_run(r2, fit_run(r2, profiles), post)))
    p2.assign(scan_points=3).to_csv(d / "p3.csv", index=False)
    (r3,) = read_runs(d / "p3.csv")
    g = scan_run(r3, profiles, time_days=res.time_days)
    out["template_grid"] = clean(scan_summary(g)) | {"best_initial": repr(g.best_initial)}
    # round trip: summary row -> table -> read -> refit
    row = summary_row(res)
    pd.DataFrame([row | {"run": run.name, "profile": run.profile_name}]).to_csv(d / "rt.csv", index=False)
    (rt,) = read_runs(d / "rt.csv")
    out["template_roundtrip"] = fit_record(fit_run(rt, profiles))
from conftest import PAPER_DATA, PAPER_PARAMS, HAS_ROUND_ROBIN, ROUND_ROBIN_DATA, ROUND_ROBIN_PARAMS
pruns = read_runs(PAPER_PARAMS)
pprof = read_profiles(PAPER_DATA, sheet="Olivine")
for r in pruns:
    try:
        out["paper:" + r.name] = fit_record(fit_run(r, pprof))
    except Exception as e:
        out["paper:" + r.name] = {"error": repr(e)}
    out["paper_run:" + r.name] = repr(r.initial) + " | " + repr(r.uncertainties) + " | rim=" + repr(r.rim_fo)
if HAS_ROUND_ROBIN:
    from test_differential import BATCH_DEFAULTS  # noqa
    rruns = read_runs(ROUND_ROBIN_PARAMS)
    rprof = read_profiles(ROUND_ROBIN_DATA)
    for r in rruns:
        out["rr_run:" + r.name] = repr(r.initial) + " | " + repr(r.uncertainties)
        try:
            out["rr:" + r.name] = fit_record(fit_run(r, rprof))
        except Exception as e:
            out["rr:" + r.name] = {"error": repr(e)}
Path(sys.argv[1]).write_text(json.dumps(clean(out), indent=1, sort_keys=True))
print("records", len(out))
