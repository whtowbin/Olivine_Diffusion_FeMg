"""Population priors for segment Fo: does the crystal population pin a crystal's start?

For every Towbin and Plank (2026) paper run this script

1. fits the starting profile with moderately wide priors (`wide_uncertainties`: every
   segment Fo a Gaussian of sd 1 mol% Fo, every step more than 2 um inside the edge a
   Gaussian of sd 3 um), and measures each segment's plateau in the data: the points
   where the best-fit model still sits within 0.1 mol% of the segment's starting Fo,
   i.e. where diffusion has not reached (`plateau_points`);
2. labels each segment by role (`roles`): edge (a first segment at most 2 um wide: the
   Fo held at the rim), rim (the outermost segment with width), plateau (any segment
   between rim and core), core (the last segment);
3. gathers one value per crystal and role (the measured plateau mean where at least 3
   points sit on it, else the fitted value) and compares the spread between crystals of
   a sample with the within-crystal errors (`spread_table`);
4. builds leave-one-out population priors (`loo_priors`): for each crystal, the mean and
   spread of the other crystals of the same group, and refits the crystal with that
   prior on the one segment in place of the wide one;
5. repeats 4 on profiles cut short 5 um past the first point of the measured core
   plateau (a core the measurements barely reached), against each crystal's own
   full-profile fit as the reference;
6. fits WHT06_ol_overgrowth2 (core not measured) with a core prior from overgrowth1, or
   at Henry's value, with the WHT01 spread between crystals;
7. repeats with nested sampling every fit whose best time falls outside its own range
   (sampler_check.csv). check_noise.py then holds the noise model fixed for the fits
   that came out worse.

Fits are cached in fits_cache.json (delete an entry, or the file, to refit). CSVs go
beside this file; figures.py draws the figures and hierarchical.py holds the
hierarchical-fit prototype. Run from the repository root:

    uv run python docs/plans/2026-10-16-population-priors/analyse.py

Public paper profiles only (analyses/xenolith_archive). About 20-60 min the first time
(more on a loaded machine), one process, one core per fit.
"""

from __future__ import annotations

import json
import math
import re
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import forster
from forster.model import Grid
from forster.sampling import ICLikelihood, fit_initial_condition
from forster.scan import Uncertainty

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
from hierarchical import random_effects  # noqa: E402

PAPER = REPO / "analyses" / "xenolith_archive"
PARAMS = PAPER / "Olivine_Profile_Diffusion_Modeling_Parameters_DB.xlsx"
DATA = PAPER / "Xenolith_MeltVein_EMPA_Master.xlsx"
CACHE = HERE / "fits_cache.json"

WIDE_FO_SD = 0.01  # 1 mol% Fo
WIDE_STEP_SD_UM = 3.0
THIN_UM = 2.0  # a first segment this narrow is the held edge value, not a zone
PLATEAU_TOL = 0.001  # model within 0.1 mol% of the segment's starting Fo: undiffused
MIN_PLATEAU_POINTS = 3
MIN_OTHERS = 3  # a leave-one-out prior needs at least this many other crystals
SD_FLOOR = 0.001  # population prior sd never below 0.1 mol% (inter-session EPMA accuracy)
TRUNC_UM = 5.0  # truncated-core test: model ends this far past the core plateau's first point
SAMPLES = ("AZ18 WHT01", "AZ18 WHT06", "Xenocryst", "GCB1A")
DAY = 86400.0


# ----------------------------------------------------------------------------- naming


def sample_of(name: str) -> str:
    for s in SAMPLES:
        if name.startswith(s):
            return s
    raise ValueError(f"no sample for {name!r}")


def crystal_of(name: str) -> str:
    return re.sub(r"[_ ]prof[_ ]?\d+$", "", name)


def roles(initial) -> list[str]:
    bounds = (initial.domain_um[0], *initial.steps_um, initial.domain_um[1])
    out, have_rim = [], False
    n = len(initial.segment_fo)
    for k in range(n):
        width = bounds[k + 1] - bounds[k]
        if k == n - 1:
            out.append("core")
        elif k == 0 and width <= THIN_UM:
            out.append("edge")
        elif not have_rim:
            out.append("rim")
            have_rim = True
        else:
            out.append("plateau")
    return out


def zoning(initial) -> str:
    """'reverse' when the outer zone (rim, else edge) is richer in Fo than the core."""
    r = roles(initial)
    outer = r.index("rim") if "rim" in r else 0
    return "reverse" if initial.segment_fo[outer] > initial.segment_fo[-1] else "normal"


def group_of(sample: str, zone: str, role: str) -> str:
    """Population a segment is drawn from: the sample for cores; sample and zoning
    direction for the outer zones (reverse and normal rims are different growth)."""
    return sample if role == "core" else f"{sample} ({zone})"


# ----------------------------------------------------------------------------- fitting


def wide_uncertainties(initial) -> dict[str, Uncertainty]:
    unc = {f"{n}_fo": Uncertainty(WIDE_FO_SD) for n in initial.segment_names}
    for i, s in enumerate(initial.steps_um):
        if s - initial.domain_um[0] > THIN_UM:
            unc[f"step{i + 1}_um"] = Uncertainty(WIDE_STEP_SD_UM)
    return unc


def scan_grid(run, profile, initial) -> Grid:
    """The grid `forster.io.scan_run` would use (purpose "scan", no time known)."""
    request = run.scan_grid
    max_time = run.grid.max_time_days if isinstance(run.grid, Grid) else None
    request = replace(request, max_time_days=request.max_time_days or max_time)
    return request.resolve(
        profile,
        run.conditions,
        run.orientation,
        initial,
        run.diffusivity,
        purpose="scan",
        time_days=None,
        scheme=run.scheme,
    ).grid


def plateau_points(run, profile, post, unc, grid) -> list[dict]:
    """Per segment: the measured points the best fit leaves within PLATEAU_TOL of the
    segment's starting Fo, their mean, sd and standard error."""
    best = post.best_initial
    like = ICLikelihood(
        profile, run.conditions, run.orientation, best, unc, grid, run.diffusivity, run.scheme
    )
    row = round(post.best_days * DAY / grid.dt_s)
    model = like.at_points(best)[row]
    seg = np.searchsorted(np.asarray(best.steps_um), like.x, side="right")
    out = []
    for k, fo in enumerate(best.segment_fo):
        on = (seg == k) & (np.abs(model - fo) < PLATEAU_TOL)
        y = like.y[on]
        n = int(on.sum())
        out.append(
            {
                "n": n,
                "mean": float(y.mean()) if n else math.nan,
                "sd": float(y.std(ddof=1)) if n > 1 else math.nan,
                "se": float(y.std(ddof=1) / math.sqrt(n)) if n > 1 else math.nan,
                "x_from": float(like.x[on].min()) if n else math.nan,
                "x_to": float(like.x[on].max()) if n else math.nan,
            }
        )
    return out


def run_fit(run, profile, initial, unc, grid, method="auto") -> dict:
    t0 = time.time()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        post = fit_initial_condition(
            profile,
            run.conditions,
            run.orientation,
            initial,
            unc,
            grid,
            run.diffusivity,
            run.scheme,
            method=method,
            seed=0,
        )
    rec = {
        "method": post.method,
        "best_days": post.best_days,
        "median_days": post.median_days,
        "low_days": post.low_days,
        "high_days": post.high_days,
        "ic_share": post.ic_share,
        "mean": post.mean,
        "sd": post.sd,
        "best_values": post.best_values,
        "sensitivity": {k: v for k, v in post.sensitivity.items() if math.isfinite(v)},
        "prior_centre": {
            n: (
                initial.steps_um[int(n[4:-3]) - 1]
                if n.startswith("step")
                else initial.segment_fo[initial.segment_names.index(n[:-3])]
            )
            for n in unc
        },
        "prior_sd": {n: u.spread for n, u in unc.items()},
        "noise_white": post.noise.white_sigma,
        "noise_corr": post.noise.correlated_sigma,
        "near_step_daic": post.near_step_delta_aic,
        "grid": [grid.dx_um, grid.dt_s, grid.max_time_days],
        "domain": list(initial.domain_um),
        "ess": post.ess,
        "seconds": time.time() - t0,
        "warnings": sorted({str(w.message) for w in caught}),
    }
    return rec, post


class Cache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.data = json.loads(path.read_text()) if path.exists() else {}

    def get(self, key, make):
        if key not in self.data:
            print(f"fitting {key}", flush=True)
            self.data[key] = make()
            self.path.write_text(json.dumps(self.data, indent=1, default=float))
        return self.data[key]


# ----------------------------------------------------------------------------- tables


def segment_rows(runs, cache) -> pd.DataFrame:
    rows = []
    for run in runs:
        ic = run.initial
        rec = cache.data[f"wide|{run.name}"]
        r, zone = roles(ic), zoning(ic)
        bounds = (ic.domain_um[0], *ic.steps_um, ic.domain_um[1])
        for k, (name, fo) in enumerate(zip(ic.segment_names, ic.segment_fo, strict=True)):
            p = rec["plateau"][k]
            key = f"{name}_fo"
            rows.append(
                {
                    "run": run.name,
                    "crystal": crystal_of(run.name),
                    "sample": sample_of(run.name),
                    "zoning": zone,
                    "segment": name,
                    "role": r[k],
                    "group": group_of(sample_of(run.name), zone, r[k]),
                    "from_um": bounds[k],
                    "to_um": bounds[k + 1],
                    "nominal_fo": fo,
                    "measured_n": p["n"],
                    "measured_fo": p["mean"],
                    "measured_sd": p["sd"],
                    "measured_se": p["se"],
                    "measured_from_um": p["x_from"],
                    "measured_to_um": p["x_to"],
                    "fitted_best_fo": rec["best_values"][key],
                    "fitted_mean_fo": rec["mean"][key],
                    "fitted_sd_fo": rec["sd"][key],
                    "point_sd_noise": rec["noise_white"],
                }
            )
    return pd.DataFrame(rows)


def crystal_values(seg: pd.DataFrame) -> pd.DataFrame:
    """One value per crystal and role: measured plateau mean (+- its standard error)
    where at least MIN_PLATEAU_POINTS points sit on it, else the fitted posterior mean
    (+- posterior sd); profiles of one crystal combined by inverse-variance weights."""
    s = seg.copy()
    measured = s["measured_n"] >= MIN_PLATEAU_POINTS
    s["value"] = np.where(measured, s["measured_fo"], s["fitted_mean_fo"])
    s["error"] = np.where(measured, s["measured_se"], s["fitted_sd_fo"])
    s["source"] = np.where(measured, "measured", "fitted")
    out = []
    for (crystal, role), g in s.groupby(["crystal", "role"], sort=False):
        w = 1 / g["error"] ** 2
        value = float((w * g["value"]).sum() / w.sum())
        out.append(
            {
                "crystal": crystal,
                "sample": g["sample"].iat[0],
                "zoning": g["zoning"].iat[0],
                "role": role,
                "group": g["group"].iat[0],
                "profiles": len(g),
                "value": value,
                "error": float(1 / math.sqrt(w.sum())),
                "source": "+".join(sorted(set(g["source"]))),
                "profile_values": "; ".join(f"{v:.4f}" for v in g["value"]),
                "profile_errors": "; ".join(f"{e:.4f}" for e in g["error"]),
            }
        )
    return pd.DataFrame(out)


def spread_table(cv: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (group, role), g in cv.groupby(["group", "role"], sort=False):
        y, e = g["value"].to_numpy(), g["error"].to_numpy()
        row = {
            "group": group,
            "role": role,
            "crystals": len(g),
            "mean_fo": y.mean(),
            "min_fo": y.min(),
            "max_fo": y.max(),
        }
        if len(g) >= 2:
            w = 1 / e**2
            ybar = (w * y).sum() / w.sum()
            q = float((w * (y - ybar) ** 2).sum())
            re_fit = random_effects(y, e)
            row.update(
                {
                    "between_sd": y.std(ddof=1),
                    "rms_within_error": math.sqrt((e**2).mean()),
                    "ratio": y.std(ddof=1) / math.sqrt((e**2).mean()),
                    "cochran_q": q,
                    "p_same_value": float(stats.chi2.sf(q, len(g) - 1)),
                    "i2": max(0.0, (q - (len(g) - 1)) / q) if q > 0 else 0.0,
                    "tau_ml": re_fit["tau"],
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def loo_priors(cv: pd.DataFrame) -> pd.DataFrame:
    """For each crystal in a group of at least MIN_OTHERS + 1: the Gaussian predictive
    prior from the other crystals (mean, sd x sqrt(1 + 1/m), floored at SD_FLOOR), the
    crystal's own value in units of that sd, and the log predictive density of its own
    value under the Gaussian and under a kernel density (Scott bandwidth)."""
    rows = []
    for (group, role), g in cv.groupby(["group", "role"], sort=False):
        if len(g) < MIN_OTHERS + 1:
            continue
        for i, own in g.iterrows():
            others = g.drop(index=i)["value"].to_numpy()
            m = len(others)
            sd = max(others.std(ddof=1) * math.sqrt(1 + 1 / m), SD_FLOOR)
            kde = stats.gaussian_kde(others)
            rows.append(
                {
                    "crystal": own["crystal"],
                    "group": group,
                    "role": role,
                    "others": m,
                    "prior_mean": others.mean(),
                    "prior_sd": sd,
                    "own_value": own["value"],
                    "own_error": own["error"],
                    "own_source": own["source"],
                    "z_own": (own["value"] - others.mean()) / sd,
                    "logpdf_gauss": stats.norm.logpdf(own["value"], others.mean(), sd),
                    "logpdf_kde": float(np.log(kde(own["value"])[0] + 1e-300)),
                }
            )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- refits


def with_prior(initial, unc, segment, mean, sd):
    k = initial.segment_names.index(segment)
    fo = list(initial.segment_fo)
    fo[k] = float(mean)
    new = replace(initial, segment_fo=tuple(fo))
    u = dict(unc)
    u[f"{segment}_fo"] = Uncertainty(float(sd))
    return new, u


def summarise_fit(rec, segment) -> dict:
    key = f"{segment}_fo"
    return {
        "best_days": rec["best_days"],
        "low_days": rec["low_days"],
        "high_days": rec["high_days"],
        "range_days": rec["high_days"] - rec["low_days"],
        "ic_share": rec["ic_share"],
        "seg_mean": rec["mean"][key],
        "seg_sd": rec["sd"][key],
        "method": rec["method"],
        "edge_warning": any("edge of its prior" in w for w in rec["warnings"]),
    }


def main() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runs = forster.read_runs(PARAMS)
        profiles = forster.read_profiles(DATA, sheet="Olivine")
    cache = Cache(CACHE)
    by_name = {r.name: r for r in runs}
    grids = {}

    # 1. wide-prior fits and measured plateaus
    for run in runs:
        prof = profiles[run.profile_name]
        unc = wide_uncertainties(run.initial)

        def make(run=run, prof=prof, unc=unc):
            grid = scan_grid(run, prof, run.initial)
            rec, post = run_fit(run, prof, run.initial, unc, grid)
            rec["plateau"] = plateau_points(run, prof, post, unc, grid)
            return rec

        rec = cache.get(f"wide|{run.name}", make)
        grids[run.name] = Grid(*rec["grid"])

    seg = segment_rows(runs, cache)
    seg.to_csv(HERE / "segments.csv", index=False, float_format="%.5g")
    cv = crystal_values(seg)
    cv.to_csv(HERE / "crystal_values.csv", index=False, float_format="%.5g")
    spread = spread_table(cv)
    spread.to_csv(HERE / "spread.csv", index=False, float_format="%.4g")
    priors = loo_priors(cv)
    priors.to_csv(HERE / "loo_priors.csv", index=False, float_format="%.5g")
    hier = hierarchical_table(seg, cv, cache)
    hier.to_csv(HERE / "hierarchical.csv", index=False, float_format="%.5g")

    # 2. leave-one-out refits on the full profiles
    out, jobs = [], []
    for _, p in priors.iterrows():
        for _, s in seg[(seg["crystal"] == p["crystal"]) & (seg["role"] == p["role"])].iterrows():
            run = by_name[s["run"]]
            prof = profiles[run.profile_name]
            unc = wide_uncertainties(run.initial)
            ic, u = with_prior(run.initial, unc, s["segment"], p["prior_mean"], p["prior_sd"])

            def make(run=run, prof=prof, ic=ic, u=u):
                return run_fit(run, prof, ic, u, grids[run.name])[0]

            pop = cache.get(f"loo_{p['role']}|{run.name}", make)
            wide = cache.data[f"wide|{run.name}"]
            row = {"run": run.name, "role": p["role"], "segment": s["segment"], "case": "full"}
            row |= {k: p[k] for k in ("prior_mean", "prior_sd")}
            row |= {f"wide_{k}": v for k, v in summarise_fit(wide, s["segment"]).items()}
            row |= {f"pop_{k}": v for k, v in summarise_fit(pop, s["segment"]).items()}
            out.append(row)
            jobs.append((row, run, prof, run.initial, unc, ic, u, grids[run.name]))

    # 3. truncated-core test: crystals with a measured core plateau and a LOO core prior
    core_priors = priors[priors["role"] == "core"].set_index("crystal")
    for _, s in seg[(seg["role"] == "core") & (seg["measured_n"] >= 8)].iterrows():
        if s["crystal"] not in core_priors.index:
            continue
        run = by_name[s["run"]]
        prof = profiles[run.profile_name]
        ic0 = run.initial
        end = s["measured_from_um"] + TRUNC_UM
        if end >= ic0.domain_um[1]:
            continue
        ic_t = replace(ic0, domain_um=(ic0.domain_um[0], end))
        unc = wide_uncertainties(ic_t)
        p = core_priors.loc[s["crystal"]]
        ic_p, u_p = with_prior(ic_t, unc, s["segment"], p["prior_mean"], p["prior_sd"])
        grid_key = f"grid_trunc|{run.name}"
        if grid_key not in cache.data:
            cache.data[grid_key] = {"grid": list(map(float, astuple(scan_grid(run, prof, ic_t))))}
        g = Grid(*cache.data[grid_key]["grid"])

        def make_w(run=run, prof=prof, ic=ic_t, u=unc, g=g):
            return run_fit(run, prof, ic, u, g)[0]

        def make_p(run=run, prof=prof, ic=ic_p, u=u_p, g=g):
            return run_fit(run, prof, ic, u, g)[0]

        tw = cache.get(f"trunc_wide|{run.name}", make_w)
        tp = cache.get(f"trunc_loo_core|{run.name}", make_p)
        full = cache.data[f"wide|{run.name}"]
        row = {"run": run.name, "role": "core", "segment": s["segment"], "case": "truncated"}
        row |= {"prior_mean": p["prior_mean"], "prior_sd": p["prior_sd"]}
        row |= {"truncated_end_um": end, "full_end_um": ic0.domain_um[1]}
        row |= {f"full_{k}": v for k, v in summarise_fit(full, s["segment"]).items()}
        row |= {f"wide_{k}": v for k, v in summarise_fit(tw, s["segment"]).items()}
        row |= {f"pop_{k}": v for k, v in summarise_fit(tp, s["segment"]).items()}
        out.append(row)
        jobs.append((row, run, prof, ic_t, unc, ic_p, u_p, g))

    # 4. WHT06_ol_overgrowth2, whose core plateau is not measured (Henry set its core from
    # other crystals): core prior from the one crystal of the same kind (overgrowth1),
    # with the between-crystal spread borrowed from the WHT01 cores (a shared spread, as
    # a hierarchical fit across samples would give it); and Henry's own core value with
    # the same spread.
    og1 = cv[(cv["crystal"] == "AZ18 WHT06_ol_overgrowth1") & (cv["role"] == "core")].iloc[0]
    tau = hier[(hier["group"] == "AZ18 WHT01") & (hier["role"] == "core")]["predictive_sd"]
    run = by_name["AZ18 WHT06_ol_overgrowth2_prof_1"]
    prof = profiles[run.profile_name]
    unc = wide_uncertainties(run.initial)
    for label, centre in (("og1", og1["value"]), ("henry", run.initial.segment_fo[-1])):
        ic, u = with_prior(run.initial, unc, "b", centre, float(tau.iat[0]))

        def make(run=run, prof=prof, ic=ic, u=u):
            return run_fit(run, prof, ic, u, grids[run.name])[0]

        rec = cache.get(f"borrowed_{label}|{run.name}", make)
        row = {"run": run.name, "role": "core", "segment": "b", "case": f"borrowed ({label})"}
        row |= {"prior_mean": centre, "prior_sd": float(tau.iat[0])}
        row |= {
            f"wide_{k}": v for k, v in summarise_fit(cache.data[f"wide|{run.name}"], "b").items()
        }
        row |= {f"pop_{k}": v for k, v in summarise_fit(rec, "b").items()}
        out.append(row)
        jobs.append((row, run, prof, run.initial, unc, ic, u, grids[run.name]))

    # 5. sampler check: wherever a best-fit time falls outside its own 68% range, refit
    # both priors with nested sampling, which can find several modes
    checks = []
    for row, run, prof, ic_w, u_w, ic_p, u_p, g in jobs:
        odd = any(
            not row[f"{w}_low_days"] <= row[f"{w}_best_days"] <= row[f"{w}_high_days"]
            for w in ("wide", "pop")
        )
        if not odd:
            continue
        tag = f"{row['case']}|{row['role']}|{run.name}"
        for which, ic, u in (("wide", ic_w, u_w), ("pop", ic_p, u_p)):

            def make(run=run, prof=prof, ic=ic, u=u, g=g):
                return run_fit(run, prof, ic, u, g, method="nested")[0]

            rec = cache.get(f"nested_{which}|{tag}", make)
            checks.append(
                {"case": row["case"], "role": row["role"], "run": run.name, "prior": which}
                | {
                    f"auto_{k}": row[f"{which}_{k}"]
                    for k in ("method", "best_days", "low_days", "high_days")
                }
                | {f"nested_{k}": v for k, v in summarise_fit(rec, row["segment"]).items()}
            )
    pd.DataFrame(checks).to_csv(HERE / "sampler_check.csv", index=False, float_format="%.5g")

    refits = pd.DataFrame(out)
    refits["time_change_pct"] = 100 * (refits["pop_best_days"] / refits["wide_best_days"] - 1)
    refits["range_ratio"] = refits["pop_range_days"] / refits["wide_range_days"]
    refits["seg_sd_ratio"] = refits["pop_seg_sd"] / refits["wide_seg_sd"]
    refits.to_csv(HERE / "refits.csv", index=False, float_format="%.5g")
    print(
        f"{len(cache.data)} cached fits; wrote segments, crystal_values, spread, loo_priors, refits"
    )


def hierarchical_table(seg: pd.DataFrame, cv: pd.DataFrame, cache: Cache) -> pd.DataFrame:
    """Two-stage hierarchical fit (`hierarchical.random_effects`) of every group with at
    least 3 crystals, and the first-order change of each run's time when its segment Fo
    moves to the shrunk value (sensitivity from the wide fit)."""
    rows = []
    for (group, role), g in cv.groupby(["group", "role"], sort=False):
        if len(g) < 3:
            continue
        fit = random_effects(g["value"], g["error"])
        for j, (_, c) in enumerate(g.iterrows()):
            for _, s in seg[(seg["crystal"] == c["crystal"]) & (seg["role"] == role)].iterrows():
                rec = cache.data[f"wide|{s['run']}"]
                key = f"{s['segment']}_fo"
                sens = rec["sensitivity"].get(key, math.nan)
                shift = sens * (fit["shrunk"][j] - c["value"])
                rows.append(
                    {
                        "group": group,
                        "role": role,
                        "run": s["run"],
                        "segment": s["segment"],
                        "mu": fit["mu"],
                        "tau": fit["tau"],
                        "predictive_sd": fit["predictive_sd"],
                        "own_value": c["value"],
                        "own_error": c["error"],
                        "shrunk": fit["shrunk"][j],
                        "shrunk_sd": fit["shrunk_sd"][j],
                        "shrinkage": fit["shrinkage"][j],
                        "sensitivity_days_per_molpct": sens / 100,
                        "best_days": rec["best_days"],
                        "time_shift_days": shift,
                        "time_shift_pct": 100 * shift / rec["best_days"],
                    }
                )
    return pd.DataFrame(rows)


def astuple(grid: Grid) -> tuple[float, float, float]:
    return (grid.dx_um, grid.dt_s, grid.max_time_days)


if __name__ == "__main__":
    main()
