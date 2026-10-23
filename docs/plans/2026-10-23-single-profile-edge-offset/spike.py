"""Step 0 of docs/plans/2026-10-23-single-profile-edge-offset.md: can one profile pin
its rim (edge) position offset when its starting profile is fitted?

Measurement only: nothing in src/ or tests/ changes. Every fit is
`forster.fit_crystal` with ONE run, the Fo offset off (fo_offset_sd=0), the edge offset
off (edge_offset_sd_um=0) or on (2 um Gaussian prior sd), method="laplace": Laplace
importance sampling with the joint fit's sample-count step rule (1024 samples, 4096 if
the ESS fraction is below ideal(d)), no nested fallback. One fit at a time, one process.

Sign convention: reported distance = true distance + delta. The code reads the model at
(reported - offset), so a correct fit gives offset = delta.

Cases (synthetic, A): one-step profile from the forward model, points every 2.5 um,
the first analysis 1-5 um inside the true rim, white plus smooth correlated noise.
  i    steps free (sd 2 um, nominal = the step where the reported data show it),
       segment Fo free (sd 0.003)
  ii   steps pinned at the true distance from the rim (what a fixed step means in the
       code: the offset moves the data against rim and step together), segment Fo free
  iii  steps free, edge 1 (first segment) Fo fixed at its true value, core Fo free
  iv   (extra) steps pinned in REPORTED distance (as a step read off a P2O5 traverse
       on the same distance scale): the model's step is (reported step - offset), so
       the offset moves the rim against the step. Done with a spike-only patch of
       joint._Part.candidate; not what the code does today.
Paper (B): the 18 Towbin and Plank (2026) runs, every step sd 2 um, every segment Fo sd
0.003.

Usage (as run for the README):
  uv run python spike.py synthetic --seeds 20     # cases i-iv, 800 fits
  uv run python spike.py paper                    # case i, 36 fits
  uv run python spike.py paper --cases ii,iv      # 72 fits
  uv run python spike.py report                   # tables (stdout) and both figures
Results are appended to results.csv (rows already there are skipped).
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import tempfile
import time
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

import forster.sampling as S
from forster import (
    Grid,
    InitialCondition,
    Profile,
    Uncertainty,
    fit_run,
    read_profiles,
    read_runs,
    write_template,
)
from forster import joint as J
from forster.model import _diffusion_model, scheme_grid
from forster.scan import is_valid, parameter_names

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "tests"))
from conftest import PAPER_DATA, PAPER_PARAMS  # noqa: E402

RESULTS = HERE / "results.csv"
PRIOR_SD = 2.0  # edge offset prior sd, um (Henry, 2026-10-23)
STEP_SD = 2.0
FO_SD = 0.003
FALLBACK = S.ESS_FALLBACK  # the single-profile fit's rule: nested below ESS/N = 0.25

# Synthetic crystal: the packaged example's conditions, orientation and law, with its
# one-step starting profile (edge 0.9146 -> core 0.8985 at 24 um from the rim).
TRUE_STEP = 24.0
TRUE_FO = (0.9146, 0.8985)
DOMAIN_END = 80.0
SPACING = 2.5
LAST_TRUE_X = 72.0
DELTAS = (-4.0, -2.0, 0.0, 2.0, 4.0)

# ----------------------------------------------------------------------------- records
# The sampler's first importance draw (1024 samples) shows whether the single-profile
# ESS rule would fall back; the joint fit only returns the last draw's ESS.
_draws: list[tuple[int, float]] = []
_orig_importance = S._importance


def _recording_importance(like, z0, n, seed, whiten=False):
    s, ess = _orig_importance(like, z0, n, seed, whiten)
    _draws.append((n, ess))
    return s, ess


S._importance = _recording_importance

# Case iv: steps pinned in reported distance move with the edge offset (spike only).
STEPS_IN_REPORTED = False
_orig_candidate = J._Part.candidate


def _candidate(self, theta):
    c = _orig_candidate(self, theta)
    if c is None or not STEPS_IN_REPORTED or self.edge_offset is None:
        return c
    off = float(theta[self.edge_offset])
    try:
        moved = replace(c, steps_um=tuple(s - off for s in c.steps_um))
    except ValueError:
        return None
    return moved if is_valid(moved, c.domain_um) else None


J._Part.candidate = _candidate


FIELDS = [
    "source",
    "case",
    "run",
    "delta_um",
    "seed",
    "offset",
    "n_points",
    "first_true_um",
    "n_params",
    "true_days",
    "best_days",
    "median_days",
    "low_days",
    "high_days",
    "offset_best_um",
    "offset_sd_um",
    "sd_ratio",
    "offset_bias_um",
    "step1_best_um",
    "edge_fo_best",
    "core_fo_best",
    "method",
    "n_samples",
    "ess_final",
    "ess_frac_final",
    "ess_first",
    "ess_frac_first",
    "ideal_frac",
    "would_fall_back",
    "seconds",
    "model_runs",
    "error",
]


def _done() -> set[tuple]:
    if not RESULTS.exists():
        return set()
    with RESULTS.open() as f:
        return {
            (r["source"], r["case"], r["run"], r["delta_um"], r["seed"], r["offset"])
            for r in csv.DictReader(f)
        }


def _write(row: dict) -> None:
    new = not RESULTS.exists()
    with RESULTS.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def one_fit(run, profiles, fits, offset_on: bool, seed: int = 0) -> dict:
    """One single-profile fit; the row of what step 0 records."""
    _draws.clear()
    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = J.fit_crystal(
            [run],
            profiles,
            fits=fits,
            fo_offset_sd=0.0,
            edge_offset_sd_um=PRIOR_SD if offset_on else 0.0,
            method="laplace",
            seed=seed,
        )
    seconds = time.perf_counter() - t0
    name = run.name
    key = f"{name}:edge1_offset_um"
    d = len(fit.names)
    ess_first = _draws[0][1] if _draws else math.nan
    n_first = _draws[0][0] if _draws else 0
    initial = fit.best_initials[name]
    row = {
        "n_points": fit.n_points[name],
        "n_params": d,
        "best_days": fit.best_days,
        "median_days": fit.median_days,
        "low_days": fit.low_days,
        "high_days": fit.high_days,
        "step1_best_um": initial.steps_um[0] if initial.steps_um else "",
        "edge_fo_best": initial.segment_fo[0],
        "core_fo_best": initial.segment_fo[-1],
        "method": fit.method,
        "n_samples": fit.n_samples,
        "ess_final": fit.ess,
        "ess_frac_final": fit.ess / fit.n_samples if fit.n_samples else math.nan,
        "ess_first": ess_first,
        "ess_frac_first": ess_first / n_first if n_first else math.nan,
        "ideal_frac": S.ideal_ess_fraction(d) if d else math.nan,
        "would_fall_back": bool(n_first and ess_first < FALLBACK * n_first),
        "seconds": seconds,
        "model_runs": fit.model_runs,
    }
    if offset_on:
        row["offset_best_um"] = fit.edge1_offset_um[name]
        row["offset_sd_um"] = fit.sd[key]
        row["sd_ratio"] = fit.sd[key] / PRIOR_SD
    return row


def _say(label: str, row: dict) -> None:
    g = row.get
    nan = math.nan
    print(
        f"{label} {row['offset']:>3}: {g('best_days', nan):.3f} d "
        f"[{g('low_days', nan):.3f}-{g('high_days', nan):.3f}] "
        f"off {g('offset_best_um', nan):.2f}+-{g('offset_sd_um', nan):.2f} "
        f"ess1 {g('ess_frac_first', nan):.2f} {g('seconds', nan):.1f}s {g('error', '')}",
        flush=True,
    )


# ----------------------------------------------------------------------------- synthetic
def template_run():
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        write_template(out)
        (run,) = read_runs(out / "parameters.csv")
        return run, read_profiles(out / "profiles.csv")


def truth_and_noise(base, base_profiles):
    """The true time and noise model: the packaged example's own fit (its best time and
    the noise model fitted to its residuals)."""
    fit = fit_run(base, base_profiles)
    return fit.time_days, fit.noise


def synthetic_profile(base, true_days, noise, seed, delta):
    """Fo at points every SPACING um from a first analysis 1-5 um inside the true rim,
    after true_days (a fine implicit run), plus white and squared-exponential noise
    (the example's fitted noise model). Distances reported as true + delta; points
    reported at negative distance are left out (as the fit would)."""
    rng = np.random.default_rng(1000 + seed)
    first = float(rng.uniform(1.0, 5.0))
    x = np.arange(first, LAST_TRUE_X + 1e-9, SPACING)
    initial = InitialCondition((0.0, DOMAIN_END), steps_um=(TRUE_STEP,), segment_fo=TRUE_FO)
    grid = Grid(0.25, true_days * 86400 / 400, 2 * true_days)
    nodes, dx, (low, high) = scheme_grid("implicit", x, grid, initial.domain_um)
    model_x = nodes[low : high + 1]
    start, advance = _diffusion_model(
        "implicit", base.conditions, base.orientation, initial, grid, base.diffusivity, model_x, dx
    )
    fo = np.interp(x, model_x, advance(start, 400)[400])
    fo = fo + rng.multivariate_normal(np.zeros(len(x)), noise.covariance(x))
    reported = x + delta
    keep = reported >= 0
    return Profile(reported[keep], fo[keep], name="syn"), first


def synthetic_run(base, case, delta):
    """The run a user would set up from the reported data, for one case."""
    nominal_step = TRUE_STEP + delta  # where the reported data show the step
    if case == "ii":
        step, priors = TRUE_STEP, {"a_fo": Uncertainty(FO_SD), "b_fo": Uncertainty(FO_SD)}
    elif case == "iv":
        step, priors = nominal_step, {"a_fo": Uncertainty(FO_SD), "b_fo": Uncertainty(FO_SD)}
    elif case == "iii":
        step, priors = nominal_step, {"step1_um": Uncertainty(STEP_SD), "b_fo": Uncertainty(FO_SD)}
    else:
        step = nominal_step
        priors = {
            "step1_um": Uncertainty(STEP_SD),
            "a_fo": Uncertainty(FO_SD),
            "b_fo": Uncertainty(FO_SD),
        }
    initial = InitialCondition((0.0, DOMAIN_END), steps_um=(step,), segment_fo=TRUE_FO)
    return replace(
        base,
        name="syn",
        profile_name="syn",
        initial=initial,
        uncertainties=priors,
        metadata={},
        scan_seed=0,
    )


def run_synthetic(seeds, cases, deltas, offsets=(False, True)):
    global STEPS_IN_REPORTED
    base, base_profiles = template_run()
    true_days, noise = truth_and_noise(base, base_profiles)
    print(f"truth: {true_days:.3f} d; noise {noise}", flush=True)
    done = _done()
    for seed in seeds:
        for delta in deltas:
            profile, first = synthetic_profile(base, true_days, noise, seed, delta)
            profiles = {"syn": profile}
            for case in cases:
                run = synthetic_run(base, case, delta)
                fits = None
                for on in offsets:
                    key = ("synthetic", case, "syn", f"{delta:g}", str(seed), "on" if on else "off")
                    if key in done:
                        continue
                    if fits is None:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            fits = {"syn": fit_run(run, profiles)}
                    STEPS_IN_REPORTED = case == "iv"
                    row = {
                        "source": "synthetic",
                        "case": case,
                        "run": "syn",
                        "delta_um": f"{delta:g}",
                        "seed": seed,
                        "offset": "on" if on else "off",
                        "first_true_um": first,
                        "true_days": true_days,
                    }
                    try:
                        row |= one_fit(run, profiles, fits, on)
                        if on:
                            row["offset_bias_um"] = row["offset_best_um"] - delta
                    except Exception as e:  # record and go on
                        row["error"] = f"{type(e).__name__}: {e}"[:200]
                    finally:
                        STEPS_IN_REPORTED = False
                    _write(row)
                    _say(f"{case:>3} d={delta:+g} s={seed}", row)


# ----------------------------------------------------------------------------- paper
def paper_run(run, case="i"):
    """Case i: every step sd 2 um and segment Fo sd 0.003. Cases ii and iv: the steps
    fixed at the table's values (ii: relative to the rim, as the code does; iv: in
    reported distance, moving with the offset), segment Fo sd 0.003."""
    priors = {}
    for n in parameter_names(run.initial):
        if n.startswith("step"):
            if case == "i":
                priors[n] = Uncertainty(STEP_SD)
        else:
            priors[n] = Uncertainty(FO_SD)
    return replace(run, uncertainties=priors)


def run_paper(names=None, cases=("i",)):
    global STEPS_IN_REPORTED
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        runs = read_runs(PAPER_PARAMS)
        profiles = read_profiles(PAPER_DATA, sheet="Olivine")
    done = _done()
    for base in runs:
        if names and base.name not in names:
            continue
        for case in cases:
            run = paper_run(base, case)
            fits = None
            for on in (False, True):
                key = ("paper", case, run.name, "", "0", "on" if on else "off")
                if key in done:
                    continue
                row = {
                    "source": "paper",
                    "case": case,
                    "run": run.name,
                    "delta_um": "",
                    "seed": 0,
                    "offset": "on" if on else "off",
                }
                STEPS_IN_REPORTED = case == "iv"
                try:
                    if fits is None:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            fits = {run.name: fit_run(run, profiles)}
                    row |= one_fit(run, profiles, fits, on)
                except Exception as e:
                    row["error"] = f"{type(e).__name__}: {e}"[:200]
                finally:
                    STEPS_IN_REPORTED = False
                _write(row)
                _say(f"{case:>3} {run.name}", row)


# ----------------------------------------------------------------------------- report
CASE_LABELS = {
    "i": "i. steps free, Fo free",
    "ii": "ii. steps pinned (to the rim), Fo free",
    "iii": "iii. steps free, edge Fo fixed",
    "iv": "iv. steps pinned (in reported distance), Fo free",
    "paper": "Paper profiles (steps free, Fo free)",
}


def paired() -> pd.DataFrame:
    """One row per data set and case: the offset-on fit next to the offset-off fit."""
    df = pd.read_csv(RESULTS)
    df = df[df.error.isna()]
    keys = ["source", "case", "run", "delta_um", "seed"]
    off = df[df.offset == "off"].set_index(keys)
    on = df[df.offset == "on"].set_index(keys)
    j = on.join(off, lsuffix="", rsuffix="_off", how="inner").reset_index()
    width = j.high_days - j.low_days
    width_off = j.high_days_off - j.low_days_off
    j["time_change_pct"] = 100 * (j.best_days / j.best_days_off - 1)
    j["median_change_pct"] = 100 * (j.median_days / j.median_days_off - 1)
    j["widening_pct"] = 100 * (width / width_off - 1)
    j["half_width_off_pct"] = 100 * width_off / 2 / j.best_days_off
    j["change_over_half_width"] = j.time_change_pct.abs() / j.half_width_off_pct
    j["err_off_pct"] = 100 * (j.best_days_off / j.true_days - 1)
    j["err_on_pct"] = 100 * (j.best_days / j.true_days - 1)
    j["fall_back_on"] = j.would_fall_back.astype(str) == "True"
    j["fall_back_off"] = j.would_fall_back_off.astype(str) == "True"
    return j


def _mr(x: pd.Series, fmt: str) -> str:
    x = x.dropna()
    if not len(x):
        return ""
    return f"{format(x.median(), fmt)} [{format(x.min(), fmt)}, {format(x.max(), fmt)}]"


def case_table(j: pd.DataFrame, case: str) -> str:
    d = j[(j.source == "synthetic") & (j.case == case)].copy()
    d["delta"] = d.delta_um.astype(float)
    head = (
        "| delta (um) | n | sd ratio | offset bias (um) | time change % | range widening % "
        "| abs change / half-range | ESS fraction (on) | would fall back, off / on "
        "| s per fit, off / on |\n|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    groups = [(f"{k:+g}", g) for k, g in d.groupby("delta")] + [("all", d)]
    for label, g in groups:
        rows.append(
            f"| {label} | {len(g)} | {_mr(g.sd_ratio, '.2f')} | {_mr(g.offset_bias_um, '+.2f')} "
            f"| {_mr(g.time_change_pct, '+.1f')} | {_mr(g.widening_pct, '+.0f')} "
            f"| {_mr(g.change_over_half_width, '.2f')} | {_mr(g.ess_frac_first, '.2f')} "
            f"| {g.fall_back_off.mean():.0%} / {g.fall_back_on.mean():.0%} "
            f"| {g.seconds_off.median():.1f} / {g.seconds.median():.1f} |"
        )
    return head + "\n".join(rows)


def truth_table(j: pd.DataFrame) -> str:
    """Error of the best time against the truth, offset off and on (synthetic)."""
    d = j[j.source == "synthetic"].copy()
    d["delta"] = d.delta_um.astype(float)
    head = (
        "| case | delta (um) | time error off % | time error on % "
        "| abs error off % | abs error on % |\n|---|---|---|---|---|---|\n"
    )
    rows = []
    for (case, delta), g in d.groupby(["case", "delta"]):
        rows.append(
            f"| {case} | {delta:+g} | {_mr(g.err_off_pct, '+.0f')} | {_mr(g.err_on_pct, '+.0f')} "
            f"| {g.err_off_pct.abs().median():.0f} | {g.err_on_pct.abs().median():.0f} |"
        )
    return head + "\n".join(rows)


def paper_table(j: pd.DataFrame, case: str) -> str:
    d = j[(j.source == "paper") & (j.case == case)]
    head = (
        "| profile | points | params (on) | best time off (d) | best time on (d) | time change % "
        "| range off (d) | range on (d) | widening % | offset (um) | sd ratio | ESS fraction "
        "off / on | ESS on | would fall back off / on | s per fit off / on |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    rows = []
    for _, r in d.iterrows():
        rows.append(
            f"| {r.run} | {r.n_points} | {r.n_params} | {r.best_days_off:.3g} | {r.best_days:.3g} "
            f"| {r.time_change_pct:+.1f} | {r.low_days_off:.3g}-{r.high_days_off:.3g} "
            f"| {r.low_days:.3g}-{r.high_days:.3g} | {r.widening_pct:+.0f} "
            f"| {r.offset_best_um:+.2f} | {r.sd_ratio:.2f} "
            f"| {r.ess_frac_first_off:.2f} / {r.ess_frac_first:.2f} | {r.ess_final:.0f} "
            f"| {'yes' if r.fall_back_off else 'no'} / {'yes' if r.fall_back_on else 'no'} "
            f"| {r.seconds_off:.1f} / {r.seconds:.1f} |"
        )
    s = d
    rows.append(
        f"| **median** | | | | | {s.time_change_pct.median():+.1f} | | | "
        f"{s.widening_pct.median():+.0f} | | {s.sd_ratio.median():.2f} "
        f"| {s.ess_frac_first_off.median():.2f} / {s.ess_frac_first.median():.2f} "
        f"| {s.ess_final.median():.0f} "
        f"| {s.fall_back_off.mean():.0%} / {s.fall_back_on.mean():.0%} "
        f"| {s.seconds_off.median():.1f} / {s.seconds.median():.1f} |"
    )
    return head + "\n".join(rows)


def report() -> None:
    j = paired()
    out = ["## Synthetic cases (median [min, max] over seeds)"]
    for case in ("i", "ii", "iii", "iv"):
        out += [f"\n### {CASE_LABELS[case]}\n", case_table(j, case)]
    out += ["\n## Time error against the truth\n", truth_table(j)]
    for case in ("i", "ii", "iv"):
        out += [f"\n## Paper profiles, {CASE_LABELS[case]}\n", paper_table(j, case)]
    errors = pd.read_csv(RESULTS)
    errors = errors[errors.error.notna()]
    if len(errors):
        out += ["\n## Errors\n"] + [
            f"- {r.run} {r.case} {r.offset}: {r.error}" for _, r in errors.iterrows()
        ]
    df = pd.read_csv(RESULTS)
    out.append(f"\nTotal fit wall time: {df.seconds.sum():.0f} s over {len(df)} fits")
    print("\n".join(out))
    figures(j)


DELTA_COLORS = {-4.0: "#2166ac", -2.0: "#67a9cf", 0.0: "#555555", 2.0: "#ef8a62", 4.0: "#b2182b"}
MIN_ESS = 50  # below this the offset sd from importance samples is not trusted
SHORT = {
    "i": "i. steps free,\nFo free",
    "ii": "ii. steps fixed\nto the rim",
    "iii": "iii. steps free,\nedge Fo fixed",
    "iv": "iv. steps fixed\nin reported x",
}


def figures(j: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"pdf.fonttype": 42, "font.size": 8})
    rng = np.random.default_rng(0)
    # Figure 1: the identifiability measure by case.
    fig, ax = plt.subplots(figsize=(7.5, 4.2), layout="constrained")
    order = [("synthetic", c) for c in ("i", "ii", "iii", "iv")] + [
        ("paper", c) for c in ("i", "ii", "iv")
    ]
    ticks = []
    for k, (source, case) in enumerate(order):
        d = j[(j.source == source) & (j.case == case)]
        xs = k + rng.uniform(-0.25, 0.25, len(d))
        if source == "paper":
            weak = (d.ess_final < MIN_ESS).to_numpy()
            ax.scatter(
                xs[~weak],
                d.sd_ratio[~weak],
                s=14,
                color="0.3",
                marker="s",
                zorder=3,
                label="paper profile" if case == "i" else None,
            )
            ax.scatter(
                xs[weak],
                d.sd_ratio[weak],
                s=14,
                facecolor="none",
                edgecolor="0.3",
                marker="s",
                zorder=3,
                label=f"paper profile, ESS below {MIN_ESS}" if case == "iv" else None,
            )
        else:
            for delta, color in DELTA_COLORS.items():
                m = d.delta_um.astype(float).to_numpy() == delta
                ax.scatter(
                    xs[m],
                    d.sd_ratio[m],
                    s=8,
                    color=color,
                    alpha=0.75,
                    zorder=3,
                    label=f"synthetic, true offset {delta:+g} um" if k == 0 else None,
                )
        med = d.sd_ratio.median()
        ax.plot([k - 0.32, k + 0.32], [med, med], color="black", lw=1.5, zorder=4)
        n_pin = int((d.sd_ratio <= 0.5).sum())
        ticks.append(f"{SHORT[case]}\n{n_pin}/{len(d)} at \u2264 0.5\nmedian {med:.2f}")
    ax.axvline(3.5, color="0.7", lw=0.8)
    ax.text(1.5, 1.47, "Synthetic profile (20 seeds x 5 true offsets)", ha="center", va="top")
    ax.text(5.0, 1.47, "18 published profiles", ha="center", va="top")
    ax.axhline(0.5, color="#009E73", lw=1, ls="--", zorder=1)
    ax.axhline(1.0, color="0.5", lw=1, ls=":", zorder=1)
    for y, text, color in ((0.5, "pinned by\nthe data", "#007a5a"), (1.0, "prior\nonly", "0.35")):
        ax.annotate(
            text,
            (1.0, y),
            xycoords=("axes fraction", "data"),
            xytext=(4, 0),
            textcoords="offset points",
            color=color,
            fontsize=7,
            va="center",
        )
    ax.set_xticks(range(len(order)), ticks, fontsize=6.5)
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_ylim(0, 1.5)
    ax.set_ylabel("Posterior sd of rim offset / prior sd (2 um)")
    syn, pap = j[j.source == "synthetic"], j[j.source == "paper"]
    ax.set_title(
        f"Few fits pin the rim offset: {int((syn.sd_ratio <= 0.5).sum())}/{len(syn)} synthetic "
        f"and {int((pap.sd_ratio <= 0.5).sum())}/{len(pap)} paper fits at \u2264 0.5 "
        "(black bars: medians)",
        fontsize=8.5,
    )
    fig.legend(loc="outside lower center", ncol=4, fontsize=6.5, frameon=False)
    fig.savefig(HERE / "fig1_sd_ratio.png", dpi=250)
    plt.close(fig)

    # Figure 2: fitted against true offset (synthetic).
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.4), layout="constrained", sharey=True)
    for ax, case in zip(axes, ["i", "ii", "iii", "iv"], strict=True):
        d = j[(j.source == "synthetic") & (j.case == case)]
        delta = d.delta_um.astype(float).to_numpy()
        ax.scatter(
            delta + rng.uniform(-0.35, 0.35, len(d)),
            d.offset_best_um,
            s=6,
            color="0.55",
            alpha=0.7,
            zorder=2,
        )
        med = d.groupby(delta).offset_best_um.median()
        ax.plot(med.index, med.to_numpy(), "o-", color="#D55E00", ms=3.5, lw=1.2, zorder=3)
        ax.plot([-4.5, 4.5], [-4.5, 4.5], color="black", lw=0.8, ls="--", zorder=1)
        ax.axhline(0, color="0.8", lw=0.6, zorder=0)
        slope = np.polyfit(delta, d.offset_best_um, 1)[0]
        label = SHORT[case].replace(",\n", ", ").replace("\n", " ")
        ax.set_title(f"{label}\nslope {slope:.2f}", fontsize=7)
        ax.set_xlabel("True offset (um)")
        ax.set_xticks([-4, -2, 0, 2, 4])
    axes[0].set_ylabel("Fitted offset (um)")
    fig.suptitle(
        "Fitted rim offset against the true one (dashed: perfect recovery; "
        "orange: median over seeds)",
        fontsize=8,
    )
    fig.savefig(HERE / "fig2_fitted_vs_true.png", dpi=250)
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("what", choices=("synthetic", "paper", "probe", "report"))
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--cases", default="")
    p.add_argument("--deltas", default=",".join(f"{d:g}" for d in DELTAS))
    p.add_argument("--runs", default="")
    a = p.parse_args()
    t0 = time.perf_counter()
    cases = (
        a.cases.split(",")
        if a.cases
        else (["i", "ii", "iii", "iv"] if a.what != "paper" else ["i"])
    )
    deltas = [float(d) for d in a.deltas.split(",")]
    if a.what == "probe":
        run_synthetic([0], cases, deltas)
    elif a.what == "synthetic":
        run_synthetic(range(a.seeds), cases, deltas)
    elif a.what == "report":
        report()
    else:
        run_paper([n for n in a.runs.split("|") if n], cases)
    print(f"total {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
