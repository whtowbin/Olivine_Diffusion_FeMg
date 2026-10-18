"""Reference posteriors of the fitted time, marginalised over uncertain IC parameters.

The yardstick for IC-uncertainty methods. For every real profile (the 35 runs of the
grid-choice study: 17 round robin + 18 paper-table runs, each with its own scheme) and
two priors over every IC parameter it has:

- "gauss": steps sigma 3 um, Fo sigma 0.003, Gaussian truncated at +-2.5 sigma;
- "wide": steps +-10 um, Fo +-0.01, uniform;

a parameter is dropped (and recorded) if its range, with the others at their nominal
values, would make the IC invalid (a step outside the domain or past the other step).

Likelihood (`likelihood.Problem`): one noise model per (profile, prior), fitted at the
best node of a coarse grid; posterior ~ chi2^(-n/2) x prior, flat prior on time.

Reference: dynesty static nested sampling, nlive 1000, dlogz 0.01, rstate
default_rng(1); the time posterior from every weighted sample (`method_dynesty_weighted`).
If more than 1e-3 of the posterior is in the last 5% of the time axis, the max time is
doubled (same noise model) and the reference rerun. Candidates on the same likelihood:
Laplace + Sobol IS (N 1024), dynesty nlive 200, grid of 17 points per parameter (up to 3
parameters). Seed check: nlive 1000 with seeds 2 and 3 on 3 profiles, both priors.

    uv run --with dynesty python analyses/ic_sampling/references.py          # run/resume
    uv run --with dynesty python analyses/ic_sampling/references.py --report # summary

Resumable: (profile, prior) cases and seed checks already in references.csv with status
"ok" are skipped. Round robin rows are written only to the private submodule
(`PRIVATE_CSV`); references.csv here holds the paper rows. At most 4 worker processes,
one BLAS thread each.
"""

import argparse
import csv
import os
import sys
import time
import traceback
import warnings
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests"))
from conftest import PAPER_DATA, PAPER_PARAMS, ROUND_ROBIN_DATA, ROUND_ROBIN_PARAMS
from likelihood import Problem, ic_value, is_valid
from methods import (
    method_dynesty_weighted,
    method_grid,
    method_laplace_is,
    timed,
)

from forster import RunDefaults, Uncertainty, read_profiles, read_runs
from forster.scan import NORMAL_SPAN, _with_values, parameter_names
from forster.uncertainty import NoiseModel

HERE = Path(__file__).resolve().parent
CSV_PATH = HERE / "references.csv"
# Round robin rows (other laboratories' profiles) go only to the private submodule; the
# public references.csv holds the paper rows. Resuming reads both.
PRIVATE_CSV = (
    HERE.parents[1] / "Diffusion Round Robin" / "outputs" / "ic_sampling" / "references.csv"
)
DEFAULT_CSV = CSV_PATH
BLAS_THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
SEED = 1
NLIVE_REF = 1000
DLOGZ = 0.01
TAIL_MAX = 1e-3
MAX_TIME_FACTOR = 32.0  # stop doubling the max time here (recorded as status "tail")
IS_N = 1024
NLIVE_CANDIDATE = 200
GRID_POINTS = 17
GRID_MAX_PARAMS = 3
MAX_WORKERS = 4
REF = f"dynesty_nlive{NLIVE_REF}"
SEED_CHECK_PROFILES = (
    "KS20-527_2-transect-1",  # 3 parameters
    "GCB1A R31_MP_Ol5_prof_1",  # 4 (step1 at the domain edge is dropped)
    "KS20-527-8_transect-2",  # 5
)
SEED_CHECK_SEEDS = (2, 3)
PRIORS = {
    "gauss": (Uncertainty(3.0), Uncertainty(0.003)),
    "wide": (Uncertainty(10.0, "uniform"), Uncertainty(0.01, "uniform")),
}
FIELDS = [
    "profile",
    "source",
    "prior",
    "method",
    "seed",
    "status",
    "error",
    "n_params",
    "params",
    "dropped",
    "scheme",
    "nominal_days",
    "time_factor",
    "lengthened",
    "max_time_days",
    "timesteps",
    "dt_s",
    "n_points",
    "white_sigma",
    "correlated_sigma",
    "correlation_length_um",
    "p16",
    "p50",
    "p84",
    "tail",
    "logz",
    "logz_err",
    "ncall",
    "runs",
    "wall_s",
    "ess",
    "ess_frac",
    "n_samples",
    "err16_pct",
    "err50_pct",
    "err84_pct",
    "case_wall_s",
]

_DATA = {}


def load():
    """{run name: (run, profiles, source)} for the round robin and paper runs."""
    if not _DATA:
        warnings.filterwarnings("ignore")
        sets = (
            (
                "round_robin",
                read_runs(ROUND_ROBIN_PARAMS, defaults=RunDefaults(max_time_days=730)),
                read_profiles(ROUND_ROBIN_DATA),
            ),
            ("paper", read_runs(PAPER_PARAMS), read_profiles(PAPER_DATA, sheet="Olivine")),
        )
        for source, runs, profiles in sets:
            for run in runs:
                if run.name in _DATA:  # de-duplicate by run name
                    continue
                _DATA[run.name] = (run, profiles, source)
    return _DATA


def uncertainties_for(initial, prior):
    """Uncertainties for every IC parameter, minus those whose range alone (others at
    nominal) reaches an invalid IC. Returns (kept, dropped)."""
    step_u, fo_u = PRIORS[prior]
    kept, dropped = {}, []
    for name in parameter_names(initial):
        u = step_u if name.endswith("_um") else fo_u
        half = NORMAL_SPAN * u.spread if u.kind == "normal" else u.spread
        v0 = ic_value(initial, name)
        ends = [_with_values(initial, {name: v0 + s * half}) for s in (-1, 1)]
        if all(is_valid(c, initial.domain_um) for c in ends):
            kept[name] = u
        else:
            dropped.append(name)
    return kept, dropped


def _q(q):
    return {"p16": q[0], "p50": q[1], "p84": q[2]}


def _err(q, ref):
    return {f"err{k}_pct": 100 * (q[i] - ref[i]) / ref[i] for i, k in enumerate((16, 50, 84))}


def _case_info(name, prior, P, dropped, source):
    return {
        "profile": name,
        "source": source,
        "prior": prior,
        "status": "ok",
        "error": "",
        "n_params": len(P.names),
        "params": " ".join(P.names),
        "dropped": " ".join(dropped),
        "scheme": P.run.scheme,
        "nominal_days": P.nominal_days,
        "time_factor": P.time_factor,
        "max_time_days": P.grid.max_time_days,
        "timesteps": P.grid.timesteps,
        "dt_s": P.grid.dt_s,
        "n_points": P.n,
        "white_sigma": P.noise.white_sigma,
        "correlated_sigma": P.noise.correlated_sigma,
        "correlation_length_um": P.noise.correlation_length_um,
    }


def _nested_row(info, res, runs, sec, seed, method, ref_q=None):
    row = {**info, "method": method, "seed": seed, **_q(res.q), "tail": res.tail}
    row.update(
        logz=res.logz,
        logz_err=res.logz_err,
        ncall=res.ncall,
        runs=runs,
        wall_s=sec,
        n_samples=res.n_samples,
    )
    if ref_q is not None:
        row.update(_err(res.q, ref_q))
    return row


def _candidates(n_params):
    names = [f"laplace_is_N{IS_N}", f"dynesty_nlive{NLIVE_CANDIDATE}"]
    return names + ([f"grid_{GRID_POINTS}"] if n_params <= GRID_MAX_PARAMS else [])


def _candidate_row(P, method, info, ref_q):
    """Run one candidate method on ``P``; its CSV row with errors against ``ref_q``."""
    if method.startswith("laplace_is"):
        (q, tail, ess), runs, sec = timed(P, method_laplace_is, IS_N, seed=SEED, verbose=False)
        extra = {"seed": SEED, "ess": ess, "ess_frac": ess / IS_N}
    elif method.startswith("dynesty"):
        res, runs, sec = timed(P, method_dynesty_weighted, NLIVE_CANDIDATE, seed=SEED, dlogz=DLOGZ)
        return _nested_row(info, res, runs, sec, SEED, method, ref_q)
    else:
        (q, tail), runs, sec = timed(P, method_grid, GRID_POINTS)
        extra = {"seed": ""}
    return {
        **info,
        "method": method,
        **extra,
        **_q(q),
        "tail": tail,
        "runs": runs,
        "wall_s": sec,
        **_err(q, ref_q),
    }


def run_case(name, prior):
    """Reference and candidate methods for one (profile, prior). Returns CSV rows."""
    t_case = time.perf_counter()
    print(f"start {name} {prior}", flush=True)
    warnings.filterwarnings("ignore")
    run, profiles, source = load()[name]
    unc, dropped = uncertainties_for(run.initial, prior)
    P = Problem(run, profiles, unc)
    noise = P.noise
    lengthened = []
    while True:
        ref, runs, sec = timed(P, method_dynesty_weighted, NLIVE_REF, seed=SEED, dlogz=DLOGZ)
        if ref.tail <= TAIL_MAX or P.time_factor >= MAX_TIME_FACTOR:
            break
        lengthened.append(f"x{P.time_factor:g}:tail={ref.tail:.2e}")
        print(f"{name} {prior}: tail {ref.tail:.2e}, lengthening", flush=True)
        P = Problem(run, profiles, unc, noise=noise, time_factor=2 * P.time_factor)
    info = _case_info(name, prior, P, dropped, source)
    info["lengthened"] = " ".join(lengthened)
    if ref.tail > TAIL_MAX:
        info["status"] = "tail"
    rows = [_nested_row(info, ref, runs, sec, SEED, REF)]
    for method in _candidates(len(P.names)):
        try:
            rows.append(_candidate_row(P, method, info, ref.q))
        except Exception as exc:  # a failed candidate is recorded, the case goes on
            msg = "".join(traceback.format_exception_only(exc)).strip()[:300]
            print(f"{name} {prior} {method} failed: {msg}", flush=True)
            rows.append({**info, "method": method, "seed": SEED, "status": "error", "error": msg})
    case_wall = time.perf_counter() - t_case
    for r in rows:
        r["case_wall_s"] = case_wall
    return rows


def run_seed_check(name, prior, seed, ref_row):
    """nlive 1000 with another seed, on exactly the reference's likelihood."""
    t_case = time.perf_counter()
    warnings.filterwarnings("ignore")
    run, profiles, source = load()[name]
    unc, dropped = uncertainties_for(run.initial, prior)
    noise = NoiseModel(
        float(ref_row["white_sigma"]),
        float(ref_row["correlated_sigma"]),
        float(ref_row["correlation_length_um"]),
    )
    P = Problem(run, profiles, unc, noise=noise, time_factor=float(ref_row["time_factor"]))
    res, runs, sec = timed(P, method_dynesty_weighted, NLIVE_REF, seed=seed, dlogz=DLOGZ)
    info = _case_info(name, prior, P, dropped, source)
    info["lengthened"] = ""
    ref_q = [float(ref_row[k]) for k in ("p16", "p50", "p84")]
    row = _nested_row(info, res, runs, sec, seed, REF, ref_q)
    row["case_wall_s"] = time.perf_counter() - t_case
    return [row]


def _error_row(task, exc):
    name, prior = task[1], task[2]
    seed = task[3] if task[0] == "seed" else SEED
    return [
        {
            "profile": name,
            "prior": prior,
            "method": REF if task[0] == "seed" else "case",
            "seed": seed,
            "status": "error",
            "error": "".join(traceback.format_exception_only(exc)).strip()[:300],
        }
    ]


def _execute(task):
    try:
        if task[0] == "case":
            return run_case(task[1], task[2])
        return run_seed_check(task[1], task[2], task[3], task[4])
    except Exception as exc:  # recorded as a failure row, the run goes on
        traceback.print_exc()
        return _error_row(task, exc)


def read_csv():
    paths = [CSV_PATH] + ([PRIVATE_CSV] if CSV_PATH == DEFAULT_CSV else [])
    frames = [pd.read_csv(p, keep_default_na=False, na_values=[""]) for p in paths if p.exists()]
    if not frames:
        return pd.DataFrame(columns=FIELDS)
    df = pd.concat(frames, ignore_index=True)
    # seeds as text ("1", "" for the grid), however pandas parsed the column
    df["seed"] = df["seed"].map(lambda v: "" if pd.isna(v) else str(int(float(v))))
    return df


def _ref_row(df, name, prior):
    m = (
        (df.profile == name)
        & (df.prior == prior)
        & (df.method == REF)
        & (df.seed.astype(str) == str(SEED))
        & (df.status != "error")
    )
    return df[m].iloc[-1].to_dict() if m.any() else None


def run_all(workers, only=None, seed_check=True):
    data = load()
    names = [n for n in data if only is None or n in only]
    df = read_csv()
    ok = df[df.status != "error"] if len(df) else df
    done_cases = {(r.profile, r.prior) for r in ok.itertuples() if r.method == REF}
    done_cases &= {
        (r.profile, r.prior)
        for r in ok.itertuples()
        if str(r.seed) == str(SEED) and r.method == REF
    }
    done_seeds = {
        (r.profile, r.prior, int(r.seed))
        for r in ok.itertuples()
        if r.method == REF and str(r.seed) not in ("", str(SEED), "nan")
    }
    # seed-check profiles first, so their checks can start early; then larger problems
    order = sorted(
        names, key=lambda n: (n not in SEED_CHECK_PROFILES, -len(data[n][0].initial.steps_um))
    )
    tasks = [("case", n, p) for n in order for p in PRIORS if (n, p) not in done_cases]

    def seed_tasks(name, prior, ref_row):
        if not seed_check or name not in SEED_CHECK_PROFILES or ref_row is None:
            return []
        return [
            ("seed", name, prior, s, ref_row)
            for s in SEED_CHECK_SEEDS
            if (name, prior, s) not in done_seeds
        ]

    for n in SEED_CHECK_PROFILES:
        for p in PRIORS:
            if n in names and (n, p) in done_cases:
                tasks += seed_tasks(n, p, _ref_row(df, n, p))
    print(f"{len(names)} profiles; {len(tasks)} tasks to run", flush=True)
    # one BLAS thread per worker: spawned workers read these when they import numpy
    for var in BLAS_THREAD_VARS:
        os.environ[var] = "1"
    t0 = time.perf_counter()
    split = CSV_PATH == DEFAULT_CSV
    if split:
        PRIVATE_CSV.parent.mkdir(parents=True, exist_ok=True)
    private_path = PRIVATE_CSV if split else CSV_PATH
    new_file = not CSV_PATH.exists()
    new_private = not private_path.exists()
    with (
        open(CSV_PATH, "a", newline="") as fh,
        open(private_path, "a", newline="") as fh_private,
        ProcessPoolExecutor(max_workers=min(workers, MAX_WORKERS)) as pool,
    ):
        public = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        private = csv.DictWriter(fh_private, fieldnames=FIELDS, extrasaction="ignore")
        if new_file:
            public.writeheader()
        if split and new_private:
            private.writeheader()
        data = load()
        pending = {pool.submit(_execute, t): t for t in tasks}
        while pending:
            finished, _ = wait(pending, return_when=FIRST_COMPLETED)
            for fut in finished:
                task = pending.pop(fut)
                rows = fut.result()
                for r in rows:
                    rr = data.get(r["profile"], (None, None, ""))[2] == "round_robin"
                    (private if split and rr else public).writerow(r)
                fh.flush()
                fh_private.flush()
                el = time.perf_counter() - t0
                for r in rows:
                    print(
                        f"[{el:7.0f}s] {r['profile']:40s} {r['prior']:5s} {r['method']:18s} "
                        f"seed {r.get('seed', '')!s:2s} {r['status']:5s} "
                        + (
                            f"p16/50/84 {r['p16']:.4g} {r['p50']:.4g} {r['p84']:.4g} "
                            f"tail {r['tail']:.1e} runs {r['runs']} {r['wall_s']:.0f}s"
                            if r["status"] != "error"
                            else r["error"]
                        ),
                        flush=True,
                    )
                if task[0] == "case" and rows[0]["status"] != "error":
                    for t in seed_tasks(task[1], task[2], rows[0]):
                        pending[pool.submit(_execute, t)] = t
    print(f"elapsed {time.perf_counter() - t0:.0f} s", flush=True)


def _abs_err_table(df):
    errs = ["err16_pct", "err50_pct", "err84_pct"]
    a = df.assign(**{e: df[e].abs() for e in errs})
    g = a.groupby(["method", "prior", "n_params"])
    out = g[errs].median().add_prefix("med_|").add_suffix("|")
    out = out.join(g[errs].max().add_prefix("max_|").add_suffix("|"))
    out.insert(0, "cases", g.size())
    out["median_runs"] = g["runs"].median()
    out["median_wall_s"] = g["wall_s"].median()
    return out


def report():
    df = read_csv()
    pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 200)
    pd.set_option("display.float_format", lambda v: f"{v:.3g}")
    errs = df[df.status == "error"]
    good = df[df.status != "error"].drop_duplicates(
        ["profile", "prior", "method", "seed"], keep="last"
    )
    main = good[good.seed.astype(str).isin(["", str(SEED), "nan"])]
    ref = main[main.method == REF]
    print(
        f"profiles: {good.profile.nunique()}  cases (profile x prior) with a reference: {len(ref)}"
    )
    print(f"failure rows: {len(errs)}")
    if len(errs):
        print(errs[["profile", "prior", "method", "seed", "error"]].to_string(index=False))
    print("\nreference status:", ref.status.value_counts().to_dict())
    print(
        "lengthened:",
        ref[ref.lengthened.notna() & (ref.lengthened != "")][
            ["profile", "prior", "lengthened", "tail"]
        ].to_string(index=False),
    )
    print("max reference tail:", ref["tail"].max())
    print("\nparameters per case:", ref.groupby(["prior", "n_params"]).size().to_dict())
    dr = ref[ref.dropped.notna() & (ref.dropped != "")]
    print("dropped parameters:")
    print(dr[["profile", "prior", "dropped"]].to_string(index=False))
    print(
        "\nreference: median ncall, wall s, logZ err:",
        ref.ncall.median(),
        ref.wall_s.median(),
        ref.logz_err.median(),
    )
    # candidates are scored only where the reference converged (status ok, tail <= 1e-3)
    ok_cases = set(zip(ref.profile[ref.status == "ok"], ref.prior[ref.status == "ok"], strict=True))
    in_ok = [(p, q) in ok_cases for p, q in zip(main.profile, main.prior, strict=True)]
    cand = main[(main.method != REF) & in_ok]
    bad_ref = main[(main.method != REF) & [not v for v in in_ok]]
    print("\ncandidates on cases whose reference did not converge (excluded below):")
    print(
        bad_ref[["profile", "prior", "method", "p16", "p50", "p84", "err50_pct"]].to_string(
            index=False
        )
    )
    lap_fail = errs[errs.method == f"laplace_is_N{IS_N}"]
    print(
        f"\nLaplace IS failures (no result): {len(lap_fail)}",
        lap_fail.prior.value_counts().to_dict(),
    )
    print("\n|error| vs reference, % (by method, prior, number of parameters)")
    print(_abs_err_table(cand).to_string())
    print("\n|error| vs reference, % (by method, all cases)")
    print(_abs_err_table(cand.assign(prior="all", n_params=0)).to_string())

    sc = good[(good.method == REF) & good.profile.isin(SEED_CHECK_PROFILES)]
    print("\nseed check: nlive 1000, seeds 1-3 (p16/50/84 days; spread = (max - min) / mean %)")
    rows = []
    for (p, pr), g in sc.groupby(["profile", "prior"]):
        line = {"profile": p, "prior": pr, "seeds": len(g)}
        for k in ("p16", "p50", "p84"):
            v = g[k].to_numpy(float)
            line[f"{k}_range%"] = 100 * (v.max() - v.min()) / v.mean()
            line[f"{k}_sd%"] = 100 * v.std(ddof=1) / v.mean() if len(v) > 1 else np.nan
        line["logz_range"] = g.logz.max() - g.logz.min()
        line["logz_err"] = g.logz_err.mean()
        rows.append(line)
    print(pd.DataFrame(rows).to_string(index=False))

    lap = cand[cand.method == f"laplace_is_N{IS_N}"].copy()
    lap["max_err"] = lap[["err16_pct", "err50_pct", "err84_pct"]].abs().max(axis=1)
    print("\nLaplace IS ESS/N distribution:")
    print(lap.ess_frac.describe(percentiles=[0.05, 0.1, 0.25, 0.5, 0.75]).to_string())
    print("by prior:", lap.groupby("prior").ess_frac.median().to_dict())
    print("Spearman(ESS/N, max |err|):", lap[["ess_frac", "max_err"]].corr("spearman").iloc[0, 1])
    print("\nLaplace IS cases > 2% off (max over p16/50/84):")
    bad = lap[lap.max_err > 2].sort_values("ess_frac")
    print(
        bad[
            [
                "profile",
                "prior",
                "n_params",
                "ess",
                "ess_frac",
                "err16_pct",
                "err50_pct",
                "err84_pct",
            ]
        ].to_string(index=False)
    )
    print("\nLaplace IS, lowest ESS/N cases:")
    print(
        lap.nsmallest(12, "ess_frac")[
            ["profile", "prior", "n_params", "ess_frac", "max_err"]
        ].to_string(index=False)
    )
    print("\nthreshold on ESS/N (fall back to nested sampling below it):")
    thr = []
    for t in (0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5):
        keep = lap[lap.ess_frac >= t]
        thr.append(
            {
                "threshold": t,
                "fall back": int((lap.ess_frac < t).sum()),
                "kept": len(keep),
                "kept >2% off": int((keep.max_err > 2).sum()),
                "kept >1% off": int((keep.max_err > 1).sum()),
                "kept worst %": keep.max_err.max() if len(keep) else np.nan,
            }
        )
    print(pd.DataFrame(thr).to_string(index=False))
    cases = good.drop_duplicates(["profile", "prior", "seed", "case_wall_s"])
    print(f"\nsum of case wall times (worker seconds): {cases.case_wall_s.sum():.0f} s")
    print(
        "sum of method wall times by method (s):",
        good.groupby("method").wall_s.sum().round(0).to_dict(),
    )


def main():
    global CSV_PATH
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--report", action="store_true", help="summarise references.csv and exit")
    ap.add_argument("--workers", type=int, default=MAX_WORKERS)
    ap.add_argument("--only", nargs="*", help="run only these run names")
    ap.add_argument("--no-seed-check", action="store_true")
    ap.add_argument("--csv", type=Path, default=CSV_PATH, help="results file (for tests)")
    args = ap.parse_args()
    CSV_PATH = args.csv
    if args.report:
        report()
        return
    run_all(args.workers, args.only, not args.no_seed_check)


if __name__ == "__main__":
    main()
