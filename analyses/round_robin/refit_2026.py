"""Refit the 17 olivine diffusion round robin runs with the current FORSTER (October 2026).

The round robin samples are other laboratories' and private: this script reads them from
the git submodule ``Diffusion Round Robin/`` and writes every result, figure and table
into ``Diffusion Round Robin/outputs/refit_2026/`` only. The script itself holds no
results and can be public.

For each run of the round robin parameter table:

1. **Nominal fit**: the table's initial condition, the default method (correlated
   uncertainty, implicit scheme), dx and dt chosen by `forster.choose_grid` (a blank
   `GridRequest`), the Classic DIPRA law (``dipra``, Dohmen and Chakraborty 2007).
2. **Initial-condition uncertainty** (`forster.scan_run`, scan_method "auto", seed 0):
   Gaussian priors, step positions sigma 3 um, edge, core and plateau Fo sigma 0.003. A
   step at an end of the model domain (0 um) is the rim boundary, not a zone boundary,
   and is held fixed. Then a fit at the best-fit initial condition on the final grid.
3. **Monte Carlo** (`forster.monte_carlo`) on that posterior over the conditions, with
   the brief's uncertainties on T, P and log fO2 (taken as 1-sigma). They are read from
   ``outputs/refit_2026/brief_settings.json`` in the private submodule, like everything
   else about the samples; the tables give fO2 in Pa, so it is drawn on its own (not
   held to a buffer). ``dipra`` has no covariance, so the law is exact here. A
   second Monte Carlo with T sigma 10 C, fO2 sigma 0.3 log and P fixed is kept for
   comparison (``mc_alt_`` columns).
4. **Law comparison**: the nominal fit under each of the four Fe-Mg choices.

Then for SH63olv066-1 to -4, whose fits got worse after the 2026-10-11 cross-term fix,
the initial condition is fitted with wide uniform priors (variant "wide": the inner step
+-10 um, edge, core and plateau Fo +-0.01, the rim step at 0 um fixed; "wide_rim": the
same with the rim step free over 0-10 um; "wider_fo": Fo +-0.03, because under +-0.01 the
plateau Fo ends at the edge of its prior in all four runs), with the misfit before and
after: the noise model of the correlated method, chi-squared against the brief's Fo
uncertainty (`brief_settings.json`), and the old kriged-grid reduced chi-squared (legacy method, the
batch settings of ``Round_Robin_Diffusion_Modeling Part 2.py``) that the restructure
plan's table reports.

Results are cached per run in ``outputs/refit_2026/cache/`` (ignored by git), so a run
that stops can be restarted and only the missing runs are fitted; ``--force`` refits.
Figures are redrawn from the cache every time.

    OMP_NUM_THREADS=1 uv run python analyses/round_robin/refit_2026.py [--workers 4] [--force]

(one BLAS thread per worker process: the pool supplies the parallelism). ``--run NAME
...`` refits and redraws only those runs and leaves the tables as they are.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import subprocess
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

import forster
from forster import (
    ConditionUncertainty,
    Grid,
    GridRequest,
    KrigingSettings,
    Profile,
    Uncertainty,
    fit_profile,
    fit_run,
    fit_tables,
    monte_carlo,
    plot_fit,
    plot_law_comparison,
    plot_law_times,
    plot_time_distributions,
    read_profiles,
    read_runs,
    scan_run,
    scan_summary,
    summary_row,
)
from forster.diffusivity import FE_MG_CHOICES, resolve
from forster.grid import narrowest_gap_um

ROOT = Path(__file__).resolve().parents[2]
RR = ROOT / "Diffusion Round Robin"
PARAMS = RR / "Round_Robin_Olivine_Profile_Diffusion_Modeling_Parameters_DB.xlsx"
DATA = RR / "Diffusion Round Robin Data Reorg.xlsx"
OUT = RR / "outputs" / "refit_2026"
CACHE = OUT / "cache"
SUBMITTED = OUT / "submitted_results.csv"  # transcribed from "Round Robin_Sample Times version 2"
FIX_EFFECT = ROOT / "docs" / "plans" / "2026-10-11-engine-fix" / "round_robin_fix_effect.csv"

LAW = "dipra"
LAWS = tuple(FE_MG_CHOICES)  # dipra, refit, dfens_tamed, dfens_global
STEP_SD_UM = 3.0
FO_SD = 0.003
SCAN_METHOD = "auto"
SEED = 0
MC_DRAWS = 4000
# The brief's uncertainties (conditions, taken as 1-sigma, and Fo), kept in the private
# submodule with the rest of the round robin material.
BRIEF = json.loads((RR / "outputs" / "refit_2026" / "brief_settings.json").read_text())
MC_BRIEF = ConditionUncertainty(
    T_sd_C=BRIEF["T_sd_C"], P_sd_Pa=BRIEF["P_sd_Pa"], fO2_sd_log=BRIEF["fO2_sd_log"]
)
MC_ALT = ConditionUncertainty(T_sd_C=10.0, P_sd_Pa=0.0, fO2_sd_log=0.3, fO2_buffer=None)

SH63_RUNS = ("SH63olv066-1", "SH63olv066-2", "SH63olv066-3", "SH63olv066-4")
WIDE_STEP_UM = 10.0
WIDE_FO = 0.01
# variant: (uniform half-width of the steps in um, of the Fo values, rim step free)
VARIANTS = {
    "wide": (WIDE_STEP_UM, WIDE_FO, False),
    "wide_rim": (WIDE_STEP_UM, WIDE_FO, True),
    "wider_fo": (WIDE_STEP_UM, 0.03, False),
}
FO_SIGMA_BRIEF = BRIEF["fo_sigma"]  # the brief's Fo uncertainty
# The legacy (kriged-grid) fit of the restructure plan's table: the batch loop's kriging,
# the table's grid (dx 1 um, dt 120 s), the explicit central scheme. 40 days covers the
# SH63 minimum (about 10 days) without the batch loop's 2-year run.
BATCH_KRIGING = KrigingSettings(
    variogram_model="gaussian",
    variogram_parameters={"sill": (10 / 100) ** 2, "range": 25, "nugget": (0.2 / 100) ** 2},
    uniform_uncertainty=0.001,
)
LEGACY_DAYS = 40.0

CACHE_VERSION = 3
SETTINGS = {
    "cache_version": CACHE_VERSION,
    "forster_version": version("forster"),
    "law": LAW,
    "step_sd_um": STEP_SD_UM,
    "fo_sd": FO_SD,
    "scan_method": SCAN_METHOD,
    "seed": SEED,
    "mc_draws": MC_DRAWS,
    "mc_brief": MC_BRIEF,
    "mc_alt": MC_ALT,
    "variants": VARIANTS,
}
DPI = 150


# --------------------------------------------------------------------------- inputs


def load() -> tuple[dict[str, forster.io.Run], dict[str, Profile]]:
    runs = {r.name: r for r in read_runs(PARAMS)}
    return runs, read_profiles(DATA)


def profile_of(run, profiles) -> Profile:
    p = profiles[run.profile_name]
    return Profile(p.distance_um, p.fo, name=run.name)


def nominal_run(run, law: str = LAW):
    """The run as refitted: default method, implicit scheme, grid left to choose_grid."""
    return replace(
        run,
        grid=GridRequest(),
        diffusivity=resolve(law),
        scheme="implicit",
        uncertainty="correlated",
    )


def priors(initial, step: Uncertainty, fo: Uncertainty, *, free_rim: bool = False) -> dict:
    """Uncertainties on every step and Fo value; a step on an end of the domain is the
    rim (or core) boundary and stays fixed unless ``free_rim``."""
    lo, hi = initial.domain_um
    out = {
        f"step{i + 1}_um": step
        for i, s in enumerate(initial.steps_um)
        if free_rim or not (math.isclose(s, lo) or math.isclose(s, hi))
    }
    out["edge_fo"] = fo
    out["core_fo"] = fo
    if initial.step_fo is not None:
        out["step_fo"] = fo
    return out


class Caught:
    """Collect the warnings of each stage, labelled, without printing them."""

    def __init__(self) -> None:
        self.notes: list[str] = []

    def run(self, label: str, fn, *args, **kwargs):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = fn(*args, **kwargs)
        for w in caught:
            if issubclass(w.category, UserWarning | RuntimeWarning):
                note = f"{label}: {w.message}"
                if note not in self.notes:
                    self.notes.append(note)
        return out


# --------------------------------------------------------------------------- cache


def cache_path(task: str, name: str) -> Path:
    return CACHE / f"{task}_{name}.pkl"


# The cache holds this script's own pickles (fit results with NumPy arrays), written
# into the private output folder and read back only by this script on this machine.
def cached(task: str, name: str) -> dict | None:
    path = cache_path(task, name)
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            data = pickle.load(f)
    except Exception:  # a truncated or stale file is refitted
        return None
    return data if data.get("settings") == SETTINGS else None


def store(task: str, name: str, data: dict) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = cache_path(task, name).with_suffix(".tmp")
    with tmp.open("wb") as f:
        pickle.dump({**data, "settings": SETTINGS}, f)
    tmp.replace(cache_path(task, name))


# --------------------------------------------------------------------------- per run


def chi2_brief(result) -> float:
    """Chi-squared per degree of freedom at the best fit against the brief's Fo
    uncertainty (independent points): comparable between initial conditions."""
    pts = fit_tables(result)["points"]
    return float(np.sum(pts.residual_fo**2) / FO_SIGMA_BRIEF**2 / (len(pts) - 1))


def legacy_fit(run, profiles, initial):
    """The old kriged-grid fit (legacy method) at ``initial``, as in the restructure plan."""
    profile = profile_of(run, profiles)
    for dt in (120.0, 60.0, 30.0):
        try:
            return fit_profile(
                profile,
                run.conditions,
                run.orientation,
                initial,
                Grid(dx_um=1.0, dt_s=dt, max_time_days=LEGACY_DAYS),
                BATCH_KRIGING,
                resolve(LAW),
                "central",
                "legacy",
            )
        except ValueError as error:  # unstable on this grid for the candidate's Fo
            if "unstable" not in str(error).lower() and "stab" not in str(error).lower():
                raise
    raise ValueError("legacy fit unstable at dt 30 s")


def best_ic_fit(log: Caught, run, profiles, table_fit, initial):
    """The fit at a fitted initial condition. choose_grid keeps dx at or below a quarter
    of the narrowest gap between steps (or a step and a domain end), already in its
    coarse first fit, so a step that lands within a fraction of a micrometre of the rim
    gives a vanishingly fine grid (a rim step at 0.06 um: dx 0.015 um, and tens of GB).
    When the narrowest gap is under the table fit's dx, the table fit's whole grid is
    given instead (the implicit scheme averages a step over its cell; the fit checks the
    grid and warns if it is coarse), and the note says so."""
    grid = GridRequest()
    table_grid = table_fit.inputs["grid"]
    if narrowest_gap_um(initial) < table_grid.dx_um:
        grid = table_grid
        log.notes.append(
            f"best-IC fit: the table fit's grid kept (dx {table_grid.dx_um:.3g} um; narrowest "
            f"gap {narrowest_gap_um(initial):.3g} um)"
        )
    return log.run("best-IC fit", fit_run, replace(run, initial=initial, grid=grid), profiles)


def process_run(name: str) -> dict:
    """Nominal fit, initial-condition posterior, Monte Carlo and law comparison."""
    t0 = time.perf_counter()
    runs, profiles = load()
    run = nominal_run(runs[name])
    log = Caught()
    nominal = log.run("fit", fit_run, run, profiles)
    scan_run_def = replace(
        run,
        uncertainties=priors(run.initial, Uncertainty(STEP_SD_UM), Uncertainty(FO_SD)),
        scan_method=SCAN_METHOD,
        scan_seed=SEED,
    )
    post = log.run("scan", scan_run, scan_run_def, profiles, time_days=nominal.time_days)
    best = best_ic_fit(log, run, profiles, nominal, post.best_initial)
    mc = log.run("mc", monte_carlo, nominal, MC_BRIEF, posterior=post, n=MC_DRAWS, seed=SEED)
    mc_alt = log.run("mc_alt", monte_carlo, nominal, MC_ALT, posterior=post, n=MC_DRAWS, seed=SEED)
    laws = {LAW: nominal}
    for law in LAWS:
        if law != LAW:
            laws[law] = log.run(f"law {law}", fit_run, nominal_run(runs[name], law), profiles)
    data = {
        "name": name,
        "nominal": nominal,
        "post": post,
        "best": best,
        "mc": mc,
        "mc_alt": mc_alt,
        "laws": laws,
        "warnings": log.notes,
        "seconds": time.perf_counter() - t0,
    }
    store("run", name, data)
    return data


def process_sh63(name: str, variant: str) -> dict:
    """Wide uniform priors on the initial condition of one SH63olv066 run."""
    t0 = time.perf_counter()
    runs, profiles = load()
    run = nominal_run(runs[name])
    log = Caught()
    table_fit = log.run("fit", fit_run, run, profiles)
    wide = replace(
        run,
        uncertainties=priors(
            run.initial,
            Uncertainty(VARIANTS[variant][0], "uniform"),
            Uncertainty(VARIANTS[variant][1], "uniform"),
            free_rim=VARIANTS[variant][2],
        ),
        scan_method=SCAN_METHOD,
        scan_seed=SEED,
    )
    post = log.run("scan", scan_run, wide, profiles, time_days=table_fit.time_days)
    best = best_ic_fit(log, run, profiles, table_fit, post.best_initial)
    mc = log.run("mc", monte_carlo, best, MC_BRIEF, posterior=post, n=MC_DRAWS, seed=SEED)
    legacy_table = log.run("legacy table IC", legacy_fit, run, profiles, run.initial)
    legacy_best = log.run("legacy best IC", legacy_fit, run, profiles, post.best_initial)
    data = {
        "name": name,
        "variant": variant,
        "table_fit": table_fit,
        "post": post,
        "best": best,
        "mc": mc,
        "legacy_table": summary_row(legacy_table),
        "legacy_best": summary_row(legacy_best),
        "warnings": log.notes,
        "seconds": time.perf_counter() - t0,
    }
    store(variant, name, data)
    return data


def task(job: tuple[str, str]) -> tuple[str, str, float]:
    kind, name = job
    t0 = time.perf_counter()
    if kind == "run":
        process_run(name)
    else:
        process_sh63(name, kind)
    return kind, name, time.perf_counter() - t0


# --------------------------------------------------------------------------- tables


def fmt_values(values: dict[str, float]) -> str:
    return "; ".join(f"{k}={v:.5g}" for k, v in values.items())


def prior_position(post, run_initial, spreads: dict[str, Uncertainty]) -> dict[str, float]:
    """Best value minus nominal over the prior's half-width (Gaussian: 2.5 sigma, the
    truncation; uniform: the half-width): 1 is the edge of the prior."""
    from forster.scan import NORMAL_SPAN, _nominal

    out = {}
    for k, v in post.best_values.items():
        u = spreads[k]
        half = u.spread * (NORMAL_SPAN if u.kind == "normal" else 1.0)
        out[k] = (v - _nominal(run_initial, k)) / half
    return out


def old_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    submitted = pd.read_csv(SUBMITTED).set_index("run") if SUBMITTED.exists() else pd.DataFrame()
    fix = pd.read_csv(FIX_EFFECT).set_index("run") if FIX_EFFECT.exists() else pd.DataFrame()
    return submitted, fix


def run_row(data: dict, run, submitted: pd.DataFrame, fix: pd.DataFrame) -> dict[str, Any]:
    nominal, post, best, mc, alt = (data[k] for k in ("nominal", "post", "best", "mc", "mc_alt"))
    s = summary_row(nominal)
    name = data["name"]
    spreads = priors(run.initial, Uncertainty(STEP_SD_UM), Uncertainty(FO_SD))
    row: dict[str, Any] = {"run": name, "profile": run.profile_name}
    if name in submitted.index:
        sub = submitted.loc[name]
        row.update(
            submitted_days=sub.submitted_best_days,
            submitted_low_days=sub.submitted_best_days - sub.submitted_minus_days,
            submitted_high_days=sub.submitted_best_days + sub.submitted_plus_days,
            submitted_reduced_chi2=sub.submitted_reduced_chi2,
        )
    if name in fix.index:
        f = fix.loc[name]
        lo, hi = (float(v) for v in str(f["new range (d)"]).split("-"))
        row.update(
            refit_20261011_days=f["new best (d)"],
            refit_20261011_low_days=lo,
            refit_20261011_high_days=hi,
            refit_20261011_reduced_chi2=f["new min red chi2"],
        )
    row.update(
        time_days=s["time_days"],
        low_days=s["low_days"],
        high_days=s["high_days"],
        white_sigma_fo=s["white_sigma_fo"],
        correlated_sigma_fo=s["correlated_sigma_fo"],
        correlation_length_um=s["correlation_length_um"],
        reduced_chi2=s["reduced_chi2"],
        chi2_brief_sigma=chi2_brief(nominal),
        points_fitted=s["points_fitted"],
    )
    ps = scan_summary(post)
    row.update(
        scan_priors=fmt_values({k: u.spread for k, u in spreads.items()}),
        scan_sampler=ps["scan_sampler"],
        scan_ess=ps["scan_ess"],
        scan_model_runs=ps["scan_candidates"],
        scan_best=ps["scan_best"],
        scan_best_prior_position=fmt_values(prior_position(post, run.initial, spreads)),
        scan_best_days=ps["scan_best_days"],
        scan_median_days=ps["scan_median_days"],
        scan_low_days=ps["scan_low_days"],
        scan_high_days=ps["scan_high_days"],
        scan_ic_share=ps["scan_ic_share"],
        scan_sensitivity=ps["scan_sensitivity"],
        scan_posterior=fmt_values(post.mean) + " (sd " + fmt_values(post.sd) + ")",
        best_ic_time_days=best.time_days,
        best_ic_low_days=best.time_days + best.time_minus_days,
        best_ic_high_days=best.time_days + best.time_plus_days,
        best_ic_white_sigma_fo=best.noise.white_sigma,
        best_ic_correlated_sigma_fo=best.noise.correlated_sigma,
        best_ic_correlation_length_um=best.noise.correlation_length_um,
        best_ic_chi2_brief_sigma=chi2_brief(best),
    )
    for prefix, dist in (("mc", mc), ("mc_alt", alt)):
        p = dist.percentiles_days
        row.update(
            {
                f"{prefix}_p05_days": p[5],
                f"{prefix}_p16_days": p[16],
                f"{prefix}_p50_days": p[50],
                f"{prefix}_p84_days": p[84],
                f"{prefix}_p95_days": p[95],
                f"{prefix}_sd_log10": "; ".join(
                    f"{k}={v:.3g}" for k, v in dist.sd_log10_by_source.items()
                ),
            }
        )
    row.update(
        mc_settings="T sd 30 C, P sd 10 MPa, log fO2 sd 0.2 (brief, as 1-sigma; fO2 on its own)",
        mc_alt_settings="T sd 10 C, P fixed, log fO2 sd 0.3 (fO2 on its own)",
        T_C=s["T_C"],
        P_Pa=s["P_Pa"],
        fO2_Pa=s["fO2_Pa"],
        alpha_deg=s["alpha_deg"],
        beta_deg=s["beta_deg"],
        gamma_deg=s["gamma_deg"],
        domain_start_um=s["domain_start_um"],
        domain_end_um=s["domain_end_um"],
        steps_um=s["steps_um"],
        edge_fo=s["edge_fo"],
        core_fo=s["core_fo"],
        step_fo=s["step_fo"],
        diffusivity=s["diffusivity"],
        diffusivity_label=s["diffusivity_label"],
        scheme=s["scheme"],
        uncertainty=s["uncertainty"],
        dx_um=s["dx_um"],
        dt_s=s["dt_s"],
        max_time_days=s["max_time_days"],
        grid_resolution=s["grid_resolution"],
        grid_error_pct=s["grid_error_pct"],
        scan_dx_um=ps["scan_dx_um"],
        scan_dt_s=ps["scan_dt_s"],
        scan_max_time_days=ps["scan_max_time_days"],
        warnings=" | ".join(data["warnings"]),
        seconds=data["seconds"],
    )
    return row


def law_rows(data: dict) -> list[dict[str, Any]]:
    rows = []
    for law, result in data["laws"].items():
        s = summary_row(result)
        rows.append(
            {
                "run": data["name"],
                "law": law,
                "diffusivity": s["diffusivity"],
                "diffusivity_label": s["diffusivity_label"],
                "time_days": s["time_days"],
                "low_days": s["low_days"],
                "high_days": s["high_days"],
                "law_sigma_log10": s["law_sigma_log10"],
                "law_low_days": s["law_low_days"],
                "law_high_days": s["law_high_days"],
                "between_study_log10": s["between_study_log10"],
                "all_low_days": s["all_low_days"],
                "all_high_days": s["all_high_days"],
                "ratio_to_dipra": s["time_days"] / data["nominal"].time_days,
                "white_sigma_fo": s["white_sigma_fo"],
                "correlated_sigma_fo": s["correlated_sigma_fo"],
                "dx_um": s["dx_um"],
                "dt_s": s["dt_s"],
            }
        )
    return rows


def sh63_row(data: dict, run) -> dict[str, Any]:
    post, best, table = data["post"], data["best"], data["table_fit"]
    spreads = priors(
        run.initial,
        Uncertainty(VARIANTS[data["variant"]][0], "uniform"),
        Uncertainty(VARIANTS[data["variant"]][1], "uniform"),
        free_rim=VARIANTS[data["variant"]][2],
    )
    nominal = {k: v for k, v in zip(("step1_um", "step2_um"), run.initial.steps_um, strict=False)}
    nominal.update(edge_fo=run.initial.edge_fo, core_fo=run.initial.core_fo)
    nominal["step_fo"] = run.initial.step_fo
    ps = scan_summary(post)
    b = post.best_initial
    p = data["mc"].percentiles_days
    return {
        "run": data["name"],
        "variant": data["variant"],
        "priors_uniform_half_width": fmt_values({k: u.spread for k, u in spreads.items()}),
        "table_steps_um": "; ".join(f"{s:g}" for s in run.initial.steps_um),
        "best_steps_um": "; ".join(f"{s:.4g}" for s in b.steps_um),
        "table_edge_fo": run.initial.edge_fo,
        "best_edge_fo": b.edge_fo,
        "table_step_fo": run.initial.step_fo,
        "best_step_fo": b.step_fo,
        "table_core_fo": run.initial.core_fo,
        "best_core_fo": b.core_fo,
        "best_prior_position": fmt_values(prior_position(post, run.initial, spreads)),
        "table_time_days": table.time_days,
        "table_low_days": table.time_days + table.time_minus_days,
        "table_high_days": table.time_days + table.time_plus_days,
        "best_time_days": best.time_days,
        "best_low_days": best.time_days + best.time_minus_days,
        "best_high_days": best.time_days + best.time_plus_days,
        "time_ratio": best.time_days / table.time_days,
        "scan_sampler": ps["scan_sampler"],
        "scan_ess": ps["scan_ess"],
        "scan_model_runs": ps["scan_candidates"],
        "scan_median_days": ps["scan_median_days"],
        "scan_low_days": ps["scan_low_days"],
        "scan_high_days": ps["scan_high_days"],
        "scan_ic_share": ps["scan_ic_share"],
        "scan_posterior": fmt_values(post.mean) + " (sd " + fmt_values(post.sd) + ")",
        "scan_log_evidence": ps["scan_log_evidence"],
        "mc_p05_days": p[5],
        "mc_p16_days": p[16],
        "mc_p50_days": p[50],
        "mc_p84_days": p[84],
        "mc_p95_days": p[95],
        "table_white_sigma_fo": table.noise.white_sigma,
        "best_white_sigma_fo": best.noise.white_sigma,
        "table_correlated_sigma_fo": table.noise.correlated_sigma,
        "best_correlated_sigma_fo": best.noise.correlated_sigma,
        "table_correlation_length_um": table.noise.correlation_length_um,
        "best_correlation_length_um": best.noise.correlation_length_um,
        "table_reduced_chi2": summary_row(table)["reduced_chi2"],
        "best_reduced_chi2": summary_row(best)["reduced_chi2"],
        "table_chi2_brief_sigma": chi2_brief(table),
        "best_chi2_brief_sigma": chi2_brief(best),
        "table_legacy_days": data["legacy_table"]["time_days"],
        "best_legacy_days": data["legacy_best"]["time_days"],
        "table_legacy_reduced_chi2": data["legacy_table"]["reduced_chi2"],
        "best_legacy_reduced_chi2": data["legacy_best"]["reduced_chi2"],
        "legacy_dt_s": data["legacy_best"]["dt_s"],
        "warnings": " | ".join(data["warnings"]),
        "seconds": data["seconds"],
    }


# --------------------------------------------------------------------------- figures


def save(fig: Figure, path: Path) -> None:
    fig.savefig(path, dpi=DPI)


def fit_figure(data: dict) -> None:
    """The fit at the table's initial condition, and at the best-fit initial condition
    under the sigma 3 um / 0.003 priors."""
    fig = Figure(figsize=(8, 8), layout="constrained")
    top, bottom = fig.subplots(2, 1)
    plot_fit(data["nominal"], top)
    plot_fit(data["best"], bottom)
    top.set_title("Table initial condition\n" + top.get_title(), fontsize="medium")
    bottom.set_title(
        "Best-fit initial condition (priors: steps sd 3 µm, Fo sd 0.003)\n"
        + fmt_values(data["post"].best_values),
        fontsize="medium",
    )
    save(fig, OUT / f"{data['name']}_fit.png")


def time_figure(data: dict) -> None:
    fig = Figure(figsize=(7, 3.8), layout="constrained")
    plot_time_distributions(data["nominal"], data["post"], data["mc"], fig.subplots())
    save(fig, OUT / f"{data['name']}_time.png")


def law_figure(data: dict) -> None:
    results = [data["laws"][law] for law in LAWS if law in data["laws"]]
    fig = Figure(figsize=(7, 8.5), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=(2.2, 1))
    plot_law_comparison(results, top.subplots())
    plot_law_times(results, bottom.subplots())
    save(fig, OUT / f"{data['name']}_laws.png")


def residuals(result) -> tuple[np.ndarray, np.ndarray]:
    pts = fit_tables(result)["points"]
    return pts.distance_um.to_numpy(), pts.residual_fo.to_numpy() * 100


def sh63_figure(data: dict) -> None:
    """Before and after the wide-prior initial-condition fit, with the residuals."""
    fig = Figure(figsize=(8, 11), layout="constrained")
    a, b, c = fig.subplots(3, 1, height_ratios=(1, 1, 0.6))
    plot_fit(data["table_fit"], a)
    plot_fit(data["best"], b)
    a.set_title("Table initial condition\n" + a.get_title(), fontsize="medium")
    b.set_title(
        f"Best-fit initial condition, wide priors ({data['variant']})\n" + b.get_title(),
        fontsize="medium",
    )
    for result, label, color in (
        (data["table_fit"], "table initial condition", "0.55"),
        (data["best"], "best-fit initial condition", "C0"),
    ):
        x, r = residuals(result)
        c.plot(x, r, ".", ms=3, color=color, label=label)
    c.axhline(0, color="k", lw=0.8)
    c.axhspan(-FO_SIGMA_BRIEF * 100, FO_SIGMA_BRIEF * 100, color="0.9", zorder=0)
    c.set_xlabel("Distance from the crystal edge (µm)")
    c.set_ylabel("Data - best fit (Fo mol%)")
    c.set_title(
        f"Residuals (grey band: the brief's ±{100 * FO_SIGMA_BRIEF:g} mol% Fo uncertainty)",
        fontsize="medium",
    )
    c.legend(loc="upper left", bbox_to_anchor=(1.01, 1), frameon=False, fontsize="small")
    save(fig, OUT / f"{data['name']}_ic_{data['variant']}.png")


def sh63_overview(datas: list[dict], variant: str) -> None:
    """The four SH63olv066 runs' initial conditions, table against best fit, over the data."""
    datas = [d for d in datas if d["variant"] == variant]
    if not datas:
        return
    fig = Figure(figsize=(8, 5), layout="constrained")
    ax = fig.subplots()
    first = datas[0]["table_fit"]
    ax.plot(first.data_x_um, first.data_fo * 100, ".", ms=3, color="0.6", label="Measured")
    for i, d in enumerate(datas):
        color = f"C{i}"
        t, b = d["table_fit"], d["best"]
        ax.plot(t.model_x_um, t.initial * 100, "--", color=color, lw=1)
        ax.plot(
            b.model_x_um,
            b.initial * 100,
            "-",
            color=color,
            lw=1.6,
            label=f"{d['name']}: {t.time_days:.1f} d (table, dashed) → {b.time_days:.1f} d",
        )
        ax.plot(b.model_x_um, b.best_fit * 100, ":", color=color, lw=1.2)
    ax.set_xlabel("Distance from the crystal edge (µm)")
    ax.set_ylabel("Fo (mol%)")
    ax.set_xlim(0, 60)
    ax.set_title(
        f"SH63olv066 initial conditions: table (dashed) and best fit (solid, {variant}); "
        "dotted: best-fit model",
        fontsize="medium",
    )
    ax.legend(loc="upper right", frameon=False, fontsize="small")
    save(fig, OUT / f"SH63olv066_ic_{variant}_overview.png")


# --------------------------------------------------------------------------- main


def source_note() -> str:
    """Where forster was imported from, and the repository commit."""
    path = Path(forster.__file__).resolve()
    try:
        commit = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    if path.is_relative_to(ROOT / "src"):
        dirty = subprocess.run(
            ["git", "-C", str(ROOT), "status", "--porcelain", "src"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        return f"{commit}{' + uncommitted changes in src/' if dirty else ''}"
    return os.environ.get("FORSTER_SOURCE_COMMIT", f"imported from {path.parent}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workers", type=int, default=4, help="processes (at most 4)")
    parser.add_argument("--force", action="store_true", help="refit even if cached")
    parser.add_argument("--run", nargs="*", help="only these runs")
    args = parser.parse_args(argv)
    t_start = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / ".gitignore").write_text("cache/\n")
    runs, _ = load()
    names = [n for n in runs if not args.run or n in args.run]
    jobs = [("run", n) for n in names]
    jobs += [(v, n) for v in VARIANTS for n in SH63_RUNS if n in names]
    todo = [j for j in jobs if args.force or cached(j[0] if j[0] != "run" else "run", j[1]) is None]
    print(f"{len(jobs)} jobs, {len(todo)} to fit ({len(jobs) - len(todo)} cached)", flush=True)
    failed = []
    if todo:
        with ProcessPoolExecutor(max_workers=max(1, min(4, args.workers))) as pool:
            futures = {pool.submit(task, j): j for j in todo}
            for future in as_completed(futures):
                kind, name = futures[future]
                try:
                    _, _, seconds = future.result()
                except Exception as error:  # report and keep the others
                    print(f"  FAILED {kind} {name}: {type(error).__name__}: {error}", flush=True)
                    failed.append((kind, name, f"{type(error).__name__}: {error}"))
                else:
                    print(f"  done {kind} {name} ({seconds:.1f} s)", flush=True)
    submitted, fix = old_tables()
    rows, laws, sh63 = [], [], []
    sh63_datas = []
    for kind, name in jobs:
        data = cached(kind, name)
        if data is None:
            continue
        if kind == "run":
            rows.append(run_row(data, runs[name], submitted, fix))
            laws += law_rows(data)
            fit_figure(data)
            time_figure(data)
            law_figure(data)
        else:
            sh63.append(sh63_row(data, runs[name]))
            sh63_datas.append(data)
            sh63_figure(data)
    for variant in VARIANTS:
        sh63_overview(sh63_datas, variant)
    note = source_note()
    for table in (rows, laws, sh63):
        for row in table:
            row["forster_version"] = SETTINGS["forster_version"]
            row["forster_source"] = note
    if rows and not args.run:
        pd.DataFrame(rows).to_csv(OUT / "summary.csv", index=False)
        pd.DataFrame(laws).to_csv(OUT / "laws.csv", index=False)
    if sh63 and not args.run:
        pd.DataFrame(sh63).to_csv(OUT / "sh63olv066_initial_conditions.csv", index=False)
    print(f"{len(rows)} runs, {len(sh63)} SH63 refits; outputs in {OUT}")
    print(f"total {time.perf_counter() - t_start:.1f} s")
    for kind, name, error in failed:
        print(f"failed: {kind} {name}: {error}", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
