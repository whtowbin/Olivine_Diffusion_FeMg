"""Parameter and profile tables to fill in, with a worked example and a column guide.

`write_template` writes ``profiles.csv`` and ``parameters.csv`` holding one
published profile (AZ18 WHT01_Ol_63_prof_1 from Towbin and Plank 2026, a
melt-vein olivine), so the template fits as it is; ``diffusivities.csv`` with the
preset diffusivity equations, to copy and edit; and ``COLUMNS.md``, generated
from `PROFILE_COLUMNS`, `PARAMETER_COLUMNS` and `DIFFUSIVITY_COLUMNS`. The tables are read by
`forster.io.read_profiles` and `forster.io.read_runs` (the "template" layout).
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import NamedTuple

from forster.diffusivity import (
    DIFFUSIVITY_PRESETS,
    FE_MG_CHOICES,
    diffusivity_table,
)


class Column(NamedTuple):
    name: str
    unit: str
    need: str  # "required", "optional" or "one form": see the guide's note
    description: str


PROFILE_COLUMNS = (
    Column(
        "profile",
        "",
        "required",
        "Name of the profile. The `profile` column in parameters.csv refers to it.",
    ),
    Column(
        "distance_um", "µm", "required", "Distance of the analysis point from the crystal edge."
    ),
    Column(
        "fo", "fraction", "required", "Forsterite content as a fraction, 0-1 (0.895, not 89.5)."
    ),
    Column(
        "Marked_bad",
        "",
        "optional",
        "`bad` leaves the point out of the fits. These are the points masked in the desktop app.",
    ),
)

PARAMETER_COLUMNS = (
    Column(
        "run",
        "",
        "required",
        "Name of the model run. It must be unique and names the output files.",
    ),
    Column("profile", "", "required", "Name of the profile in profiles.csv to fit."),
    Column("T_C", "°C", "required", "Temperature."),
    Column("P_Pa", "Pa", "required", "Pressure (5e8 is 0.5 GPa)."),
    Column(
        "fO2_Pa", "Pa", "one form", "Oxygen fugacity. Alternatively, give the next two columns."
    ),
    Column("fO2_buffer", "", "one form", "Buffer: QFM (or FMQ) or NNO (Frost 1991)."),
    Column("fO2_delta_log", "log units", "one form", "Offset of fO2 from that buffer."),
    Column("alpha_deg", "degrees", "one form", "Angle between the profile and [100]."),
    Column("beta_deg", "degrees", "one form", "Angle between the profile and [010]."),
    Column("gamma_deg", "degrees", "one form", "Angle between the profile and [001]."),
    Column("phi1_deg", "degrees", "one form", "EBSD Euler angle phi1 (Bunge)."),
    Column("Phi_deg", "degrees", "one form", "EBSD Euler angle Phi (Bunge)."),
    Column("phi2_deg", "degrees", "one form", "EBSD Euler angle phi2 (Bunge)."),
    Column(
        "profile_angle_deg",
        "degrees",
        "one form",
        "Direction of the profile on the EBSD map, measured from the positive x axis (east).",
    ),
    Column(
        "domain_start_um",
        "µm",
        "required",
        "Start of the model domain, edge 1, where Fo is held (at edge1_fo).",
    ),
    Column(
        "domain_end_um",
        "µm",
        "required",
        "End of the model domain, the far end. Fo is held there at the last segment's Fo "
        "(the core), unless centre_um is given.",
    ),
    Column(
        "steps_um",
        "µm",
        "required",
        "Positions of the steps in the starting profile, in order: one number, or several in "
        'quotes ("10, 33"). The steps divide the profile into segments.',
    ),
    Column(
        "segment_fo",
        "fraction",
        "required",
        "Starting Fo of each segment, from edge 1 inward. Give one more value than steps_um "
        '("0.91, 0.89" for one step, "0.91, 0.93, 0.89" for two). Replaces edge_fo, core_fo '
        "and step_fo, which are still read until the release.",
    ),
    Column(
        "segment_names",
        "",
        "optional",
        'Labels for the segments, from edge 1 inward ("rim, mantle, core"). Empty: a, b, '
        "c, ... Joint fits match segments by label, and uncertainty results name them "
        "(a_fo, ...).",
    ),
    Column(
        "edge1_fo",
        "fraction",
        "optional",
        "Fo held at edge 1 (domain_start_um) during diffusion, when it differs from the "
        "first segment's Fo. The first segment's Fo then stays the starting value. Empty: "
        "the first segment's Fo. With growth stages, it holds until the first stage. It "
        "cannot be combined with the initial-condition scan. Replaces rim_fo.",
    ),
    Column(
        "centre_um",
        "µm",
        "optional",
        "Distance of the crystal's center, for modeling the whole crystal. The model runs "
        "to the center (beyond the measured points if needed) and mirrors the profile "
        "there, so no Fo crosses it and the core is free to change. Empty: a half profile, "
        "with the far end held at the core's Fo. Needs the implicit scheme and no growth "
        "stages.",
    ),
    Column(
        "stage_days",
        "days",
        "optional",
        "Growth stages: the start of each stage, in days after the start of the run, one "
        'entry per stage ("2, 5" for two). At its start a stage moves the rim out to '
        "stage_rim_um, where a new zone of Fo stage_zone_fo appears and diffuses with the "
        "older crystal. The domain, step and Fo columns above describe the crystal before "
        "the first stage. Only times after the last stage are fitted, and time_days counts "
        "from the start. Needs dx_um and dt_s, the correlated uncertainty method, and no "
        "initial-condition scan (no _sd or _range columns).",
    ),
    Column(
        "stage_rim_um",
        "µm",
        "optional",
        "Rim position from each stage on, one entry per stage. The crystal grows "
        "outward, so each value is at most the previous rim's (domain_start_um before "
        "the first stage). Keeping the previous rim changes only the rim's Fo "
        "(stage_rim_fo). Required with stage_days.",
    ),
    Column(
        "stage_zone_fo",
        "fraction",
        "optional",
        "Fo of the zone each stage adds between the old and the new rim, one entry per "
        "stage. Every stage that moves the rim needs one. Leave an entry empty "
        '("0.91, ") for a stage that only changes the rim\'s Fo.',
    ),
    Column(
        "stage_rim_fo",
        "fraction",
        "optional",
        "Fo held at the rim from each stage on, one entry per stage. Empty, or an empty "
        "entry: that stage's stage_zone_fo.",
    ),
    Column(
        "dx_um",
        "µm",
        "optional",
        "Grid spacing of the final fit. Leave it empty to have it chosen for about 1% "
        "accuracy in the fitted time. A value you give is kept, with a warning if it is "
        "coarse, wastefully fine, or too coarse for the steps.",
    ),
    Column(
        "dt_s",
        "s",
        "optional",
        "Time step of the final fit, which sets how finely the fitted time is resolved. "
        "Leave it empty to have it chosen as 1/400 of the fitted time. The explicit "
        "schemes (central, dipra) split a dt above their stability limit (D dt / dx² = "
        "0.5) into stable steps and keep only the rows at whole time steps. Growth stages "
        "need a stable dt, so they do not use the splitting.",
    ),
    Column(
        "max_time_days",
        "days",
        "optional",
        "Longest time modeled. Default 200; with dx_um or dt_s empty, 3x a quick estimate "
        "of the time.",
    ),
    Column(
        "scheme",
        "",
        "optional",
        "implicit (default since 2026-10-13): rim on the end node, steps not snapped to "
        "nodes, any time step. central: the engine's explicit scheme, used before. dipra: "
        "DIPRA's forward-difference scheme (Girona and Costa 2013), a numerical scheme and "
        "not the diffusion law. With uncertainty = legacy the default is central.",
    ),
    Column(
        "diffusivity",
        "",
        "optional",
        "Fe-Mg diffusivity equation (the diffusion law). Choices: "
        + "; ".join(
            f"{alias}: {label}, preset {name}" for alias, (name, label) in FE_MG_CHOICES.items()
        )
        + ". dipra is the default. Every choice but dipra carries a covariance, which adds "
        "the law_ range to summary.csv. refit also carries the spread between laboratories "
        "(the all_ range). Any other preset name also works, as does a name from "
        "diffusivities.csv (then run forster fit with --diffusivities).",
    ),
    Column(
        "uncertainty",
        "",
        "optional",
        "correlated (default) or legacy (the round robin rule, which reproduces old results).",
    ),
    Column(
        "steps_sd_um",
        "µm",
        "optional",
        "Uncertainty on the step position(s), as a Gaussian 1 sigma: one value, or one per "
        'step ("2, 5"). 0 keeps a step fixed. Alternatively, give steps_range_um.',
    ),
    Column(
        "steps_range_um",
        "µm",
        "optional",
        "Uncertainty on the step position(s) as a uniform ± half-width, in the same form "
        "as steps_sd_um.",
    ),
    Column(
        "segment_fo_sd",
        "fraction",
        "optional",
        "Uncertainty on each segment's Fo, as a Gaussian 1 sigma: one value for every "
        'segment, or one per segment ("0.002, , 0.003"). Empty or 0 keeps a segment '
        "fixed. Replaces edge_fo_sd, core_fo_sd and step_fo_sd.",
    ),
    Column(
        "segment_fo_range",
        "fraction",
        "optional",
        "Uncertainty on each segment's Fo as a uniform ± half-width, in the same form as "
        "segment_fo_sd. A segment may take one of the two, not both.",
    ),
    Column(
        "scan_method",
        "",
        "optional",
        "How the uncertain starting-profile parameters are fitted. auto (default) finds "
        "the best fit, then uses importance sampling from a Laplace proposal, or nested "
        "sampling when that has too few effective samples. The other choices are laplace, "
        "nested, and grid (every node of a grid, as before 2026-10-15). Blank with "
        "scan_points given means grid.",
    ),
    Column(
        "scan_seed",
        "",
        "optional",
        "Seed of the sampler (default 0). The same seed gives the same result.",
    ),
    Column(
        "scan_points",
        "",
        "optional",
        "For scan_method grid, the points per uncertain parameter (default 13 for one, 9 "
        "for two, 5 for more).",
    ),
    Column(
        "scan_dx_um",
        "µm",
        "optional",
        "Grid spacing of the scan's runs. Empty: chosen for about 5% accuracy, which is "
        "coarser and faster than the final fit.",
    ),
    Column(
        "scan_dt_s",
        "s",
        "optional",
        "Time step of the scan's runs. Empty: 1/80 of the nominal fit's time. For "
        "central or dipra it is raised, if needed, to the step at which the fastest "
        "candidate reaches D dt / dx² = 0.4.",
    ),
    Column(
        "scan_max_time_days",
        "days",
        "optional",
        "Longest time the scan's runs model. Empty: max_time_days if given, capped at 10x "
        "the fitted time, otherwise 3x the fitted time. A warning says if the posterior of "
        "time reaches the end.",
    ),
    Column(
        "mc_T_sd_C",
        "°C",
        "optional",
        "Monte Carlo over the conditions and the diffusion law (forster.rescale), run when "
        "any mc_ spread column is given. This column is the 1 sigma of temperature.",
    ),
    Column("mc_P_sd_Pa", "Pa", "optional", "Monte Carlo: 1 sigma of pressure."),
    Column(
        "mc_fO2_sd_log",
        "log10",
        "optional",
        "Monte Carlo: 1 sigma of log10 fO2 (relative to mc_fO2_buffer when given).",
    ),
    Column(
        "mc_fO2_buffer",
        "",
        "optional",
        "Monte Carlo: QFM or NNO, to hold fO2 relative to that buffer so it follows each "
        "draw's T and P. Empty: the row's fO2_buffer when fO2 is given that way, "
        "otherwise fO2 is drawn on its own.",
    ),
    Column(
        "mc_law",
        "",
        "optional",
        "Monte Carlo: yes (default) draws the diffusion law's parameters from its "
        "covariance, or no. Laws without a covariance, such as dipra, are exact.",
    ),
    Column(
        "mc_between_study",
        "",
        "optional",
        "Monte Carlo: yes adds the spread between laboratories (for an absolute time). Default no.",
    ),
    Column(
        "mc_samples",
        "",
        "optional",
        "Monte Carlo draws (default 4000; 200 refits for runs with growth stages).",
    ),
    Column("mc_seed", "", "optional", "Monte Carlo seed (default 0)."),
    Column(
        "crystal",
        "",
        "optional",
        "Name of the crystal this profile was measured on. Runs with the same crystal will "
        "be fitted jointly, with a shared time and compositions and their own steps. For "
        "now the name is only carried through. The joint fit exists from Python "
        "(forster.fit_crystal), and tables and the CLI are planned "
        "(docs/plans/2026-10-16-joint-fits.md).",
    ),
    Column(
        "sample",
        "",
        "optional",
        "Name of the sample (rock) the crystal comes from. Crystals of one sample form the "
        "population for priors on segment Fo. Carried through for now.",
    ),
    Column("notes", "", "optional", "Free text for your own records."),
)

# Columns of summary.csv besides the parameter columns above (which it repeats, with
# fO2 in Pa and the orientation as angles, so it reads back as a parameter table).
SUMMARY_COLUMNS = (
    Column(
        "diffusivity_label",
        "",
        "",
        "Display name of the diffusivity equation. The diffusivity column holds its preset name.",
    ),
    Column("time_days", "days", "", "Best-fit time."),
    Column(
        "time_since_last_stage_days",
        "days",
        "",
        "Best-fit time since the start of the last growth stage (stage_days). Without "
        "stages it equals time_days.",
    ),
    Column("low_days", "days", "", "Low end of the 1-sigma range."),
    Column("high_days", "days", "", "High end of the 1-sigma range."),
    Column(
        "low2_days",
        "days",
        "",
        "Low end of the 2-sigma range (95.45%). Empty for the legacy method.",
    ),
    Column("high2_days", "days", "", "High end of the 2-sigma range."),
    Column(
        "law_sigma_log10",
        "log10",
        "",
        "1 sigma of log10 D from the diffusion law's covariance at the run's conditions "
        "(also the 1 sigma of log10 time, with the sign changed). Empty for laws without a "
        "covariance (the default) and for fits with growth stages or a rim value that "
        "changes in time (docs/references/law-uncertainty-limits.md).",
    ),
    Column(
        "between_study_log10",
        "log10",
        "",
        "Spread between laboratories in the law's calibration data (a random study offset).",
    ),
    Column(
        "law_low_days",
        "days",
        "",
        "Low end of the 1-sigma range with the law's uncertainty added. To compare crystals "
        "fitted with the same law, use low_days and high_days, because the law's error is "
        "common to all of them. For a time on its own, use these.",
    ),
    Column("law_high_days", "days", "", "High end of that range."),
    Column(
        "all_low_days",
        "days",
        "",
        "Low end of the range with the spread between laboratories added as well. Use it "
        "for an absolute time, if the crystal may sit off the laboratories' average as one "
        "more experiment would.",
    ),
    Column("all_high_days", "days", "", "High end of that range."),
    Column(
        "reduced_chi2",
        "",
        "",
        "Chi-squared at the best fit over (points fitted - 1). It is a diagnostic only. For "
        "the correlated method it is about 1 by construction, because the noise model is "
        "fitted to the residuals. For legacy it is taken over the kriged grid points, so "
        "the two are not comparable. Judge correlated fits by the three noise columns.",
    ),
    Column(
        "points_fitted",
        "",
        "",
        "Measured points inside the model domain (legacy: kriged grid points).",
    ),
    Column(
        "white_sigma_fo",
        "fraction",
        "",
        "Fitted scatter of the points (correlated method). A value much larger than your "
        "analytical precision points to outliers or an underestimated precision.",
    ),
    Column(
        "correlated_sigma_fo",
        "fraction",
        "",
        "Fitted smooth misfit between model and data. If it is large next to "
        "white_sigma_fo, the model misses the shape of the profile, so check the starting "
        "profile.",
    ),
    Column(
        "correlation_length_um",
        "µm",
        "",
        "Length scale of that misfit. Near half the profile length, the misfit spans the "
        "whole profile.",
    ),
    Column(
        "near_step_daic",
        "",
        "",
        "AIC change when the misfit may be larger near the steps (correlated method, no "
        "growth stages). Below -6 a warning says the misfit is concentrated near the step "
        "and the 1-sigma range is probably too narrow "
        "(docs/references/noise-models-guide.md).",
    ),
    Column(
        "near_step_misfit_sigma_fo",
        "fraction",
        "",
        "Standard deviation of the near-step model's misfit at the step, which shows how "
        "large the misfit is there (0.001 = 0.1 mol% Fo).",
    ),
    Column(
        "near_step_points",
        "",
        "",
        "Measured points within 2 diffusion lengths of a step (-1: check not run). With "
        "fewer than about 13 the check often misses a near-step misfit, so a silent check "
        "is weak evidence. Quote the 2-sigma range then.",
    ),
    Column(
        "core_end_diffusion_lengths",
        "",
        "",
        "Distance from the last step to the model's core end (the last measured point), in "
        "diffusion lengths at the best fit. The core Fo is held there. Below 3 a warning "
        "says the data probably stop at or before the plateau, which moves the fit "
        "(docs/guide/practical-advice.md).",
    ),
    Column(
        "points_outside_domain",
        "",
        "",
        "Measured points before domain_start_um or past domain_end_um, which are not "
        "fitted. This is often deliberate, for example to leave out an earlier stage. If "
        "the points continue a flat plateau, extending the domain to include them makes the "
        "near-step check more reliable.",
    ),
    Column(
        "warnings",
        "",
        "",
        "Warnings from the fit, such as a range that reaches the last time step.",
    ),
    Column("forster_version", "", "", "FORSTER version that made the fit."),
    Column(
        "scan_best_days",
        "days",
        "",
        "With uncertainties: best-fit time over all scanned initial conditions. With the "
        "sampled methods, the time of the best-fit initial condition.",
    ),
    Column(
        "scan_sampler",
        "",
        "",
        "Method that produced the scan columns: grid, laplace or nested. auto runs laplace "
        "and falls back to nested.",
    ),
    Column(
        "scan_median_days",
        "days",
        "",
        "Sampled methods: median of the posterior of time over the initial-condition parameters.",
    ),
    Column(
        "scan_low_days",
        "days",
        "",
        "Low end of the 1-sigma range including the initial-condition uncertainty. For "
        "grid it is the profile likelihood, and for the sampled methods the central 68% of "
        "the posterior of time.",
    ),
    Column("scan_high_days", "days", "", "High end of that range."),
    Column(
        "scan_marginal_low_days",
        "days",
        "",
        "Low end of the Bayesian (marginal) range. With Gaussian uncertainties it is "
        "pulled toward the nominal initial condition.",
    ),
    Column("scan_marginal_high_days", "days", "", "High end of the Bayesian range."),
    Column(
        "scan_best",
        "",
        "",
        "Best values of the scanned parameters (step1_um, a_fo, ...). The nominal values "
        "stay in the parameter columns.",
    ),
    Column("scan_candidates", "", "", "Initial conditions fitted (one model run each)."),
    Column(
        "scan_ic_share",
        "",
        "",
        "Sampled methods: fraction of the variance of time that comes from the choice of "
        "initial condition. The rest is the fit at a fixed initial condition.",
    ),
    Column(
        "scan_sensitivity",
        "",
        "",
        "Sampled methods: change of the expected time per unit of each parameter, from the "
        "posterior samples (days per µm for steps, days per unit Fo fraction).",
    ),
    Column(
        "scan_ess",
        "",
        "",
        "Sampled methods: effective sample size of the importance samples (out of 1024). "
        "Below 256, auto uses nested sampling.",
    ),
    Column(
        "scan_log_evidence",
        "",
        "",
        "Sampled methods: log evidence. Use it to compare initial-condition models (one "
        "step or two) fitted to the same profile with the same grid.",
    ),
    Column(
        "scan_prior_offsets",
        "",
        "",
        "Distance of each fitted starting-profile value from its nominal value, in prior "
        "units (sd for _sd, half-width for _range). For example, a_fo=+2.30 is 2.3 sd "
        "above. From 2 sd (short of the prior's edge at 2.5) a warning says the prior and "
        "the data disagree.",
    ),
    Column(
        "scan_near_step_daic",
        "",
        "",
        "near_step_daic at the best-fit initial condition. Fitting the initial condition "
        "usually clears a near-step warning. If it is still below -6, the misfit is "
        "concentrated near the step, so quote the 2-sigma range.",
    ),
    Column(
        "mc_method",
        "",
        "",
        "Monte Carlo method: rescale (t = t_fit D_fit / D_draw, exact for a fixed rim) or "
        "refit (for growth stages, every draw is refitted).",
    ),
    Column("mc_draws", "", "", "Monte Carlo draws made."),
    Column(
        "mc_median_days",
        "days",
        "",
        "Monte Carlo median time, combining the fit (or the initial-condition posterior), "
        "the conditions and the diffusion law.",
    ),
    Column("mc_low_days", "days", "", "Monte Carlo 16th percentile."),
    Column("mc_high_days", "days", "", "Monte Carlo 84th percentile."),
    Column("mc_p05_days", "days", "", "Monte Carlo 5th percentile."),
    Column("mc_p95_days", "days", "", "Monte Carlo 95th percentile."),
    Column(
        "mc_sd_log10",
        "log10",
        "",
        "Monte Carlo: standard deviation of log10 time from each source drawn alone (fit, "
        "T, P, fO2, law, studies). It shows which source matters most.",
    ),
    Column(
        "scan_dropped",
        "",
        "",
        "Grid points left out as invalid (steps out of order or outside the domain, Fo "
        "outside 0-1).",
    ),
    Column(
        "grid_resolution",
        "",
        "",
        "Diffusion length sqrt(D t) at the fit over the grid spacing. For 1% accuracy in "
        "the fitted time it should be about 5 (implicit) or 240 (central, dipra).",
    ),
    Column(
        "grid_error_pct",
        "%",
        "",
        "Estimated worst-case error of the fitted time from the grid spacing: 0.2 x "
        "(dx / l)² for implicit, 2.4 x dx / l for central and dipra, where l is the "
        "diffusion length (measured on 35 real profiles).",
    ),
    Column(
        "grid_stability",
        "",
        "",
        "D dt / dx² of the final fit's time step. For central and dipra, above 0.5 each "
        "step is split into stable steps. Empty for implicit.",
    ),
)

# Columns of diffusivities.csv; the equation is in forster.diffusivity.
DIFFUSIVITY_COLUMNS = (
    Column("name", "", "required", "Name to use in the `diffusivity` column of parameters.csv."),
    Column("element", "", "required", "Fe-Mg for fits. Ni, Mn and Ca are listed for reference."),
    Column("log10_D0_100", "log10 m²/s", "required", "Pre-factor along [100]."),
    Column("log10_D0_010", "log10 m²/s", "required", "Pre-factor along [010]."),
    Column("log10_D0_001", "log10 m²/s", "required", "Pre-factor along [001]."),
    Column("E_100_J_mol", "J/mol", "required", "Activation energy along [100]."),
    Column("E_010_J_mol", "J/mol", "required", "Activation energy along [010]."),
    Column("E_001_J_mol", "J/mol", "required", "Activation energy along [001]."),
    Column("fO2_exponent", "", "required", "Exponent n of (fO2 / fO2_ref_Pa)."),
    Column("fO2_ref_Pa", "Pa", "required", "Reference fO2 (1e5 if the equation uses bar)."),
    Column(
        "fo_coefficient",
        "log10 per unit Fo",
        "required",
        "The coefficient m in m (fo_ref - X_Fo). Use 0 for none.",
    ),
    Column("fo_ref", "fraction", "required", "Reference Fo of the composition term."),
    Column("activation_volume_m3_mol", "m³/mol", "required", "Activation volume (0 for none)."),
    Column("P_ref_Pa", "Pa", "required", "Reference pressure of the pressure terms."),
    Column(
        "pressure_coefficient_per_Pa",
        "log10 per Pa",
        "required",
        "Pressure term outside the exponential (0 for none).",
    ),
    Column("source", "", "optional", "Where the equation comes from."),
)

DIFFUSIVITY_EQUATION = (
    "log10 D_i = log10_D0_i + fO2_exponent log10(fO2 / fO2_ref_Pa) "
    "+ fo_coefficient (fo_ref - X_Fo) "
    "- (E_i + activation_volume_m3_mol (P - P_ref_Pa)) / (ln(10) R T) "
    "+ pressure_coefficient_per_Pa (P - P_ref_Pa)"
)

FILES = {"profiles.csv": "example_profiles.csv", "parameters.csv": "example_parameters.csv"}


def column_table(columns: tuple[Column, ...]) -> str:
    """Markdown table of ``columns``; the "Needed" column only for input tables."""
    if any(c.need for c in columns):
        rows = [f"| `{c.name}` | {c.unit} | {c.need} | {c.description} |" for c in columns]
        return "\n".join(["| Column | Unit | Needed | Meaning |", "|---|---|---|---|", *rows])
    rows = [f"| `{c.name}` | {c.unit} | {c.description} |" for c in columns]
    return "\n".join(["| Column | Unit | Meaning |", "|---|---|---|", *rows])


def column_guide() -> str:
    """Markdown description of every template column (also in the documentation site,
    docs/site/columns.md)."""
    return "\n\n".join(
        [
            "# Columns of the FORSTER templates",
            "## profiles.csv\n\nOne row per analysis point.",
            column_table(PROFILE_COLUMNS),
            "## parameters.csv\n\nOne row per model run. Fo is a fraction throughout.",
            column_table(PARAMETER_COLUMNS),
            "*one form*: give fO2 either as `fO2_Pa` or as `fO2_buffer` with "
            "`fO2_delta_log`, and the orientation either as `alpha_deg`, `beta_deg`, "
            "`gamma_deg` or as the four EBSD columns. Leave the other form empty.",
            "The example row is AZ18 WHT01_Ol_63_prof_1 from Towbin and Plank (2026), "
            "J. Petrology 67(9), egag077. Replace it with your own rows.",
            "**Uncertain initial conditions.** Give an uncertainty on any step position or "
            "concentration, and `forster fit` also fits those parameters. Use `_sd` for a "
            "Gaussian 1 sigma or `_range` for a uniform ± half-width, but not both for one "
            "parameter. Gaussian priors are cut at ±2.5 sigma. FORSTER finds the best-fit "
            "initial condition, then the posterior of time over them, by importance "
            "sampling from a Laplace approximation or, when that has too few effective "
            "samples, nested sampling (`scan_method` auto, the default; "
            "`forster.sampling`). It reports the range of time including that "
            "uncertainty, the best values, and how much of the range comes from the "
            "initial condition (`scan_*` columns in summary.csv), and writes "
            "`<run>_scan.csv` with the posterior of time. Edge and core Fo are the boundary "
            "values held during diffusion, so their uncertainty also covers the boundary "
            "conditions. `scan_method` grid fits every node of a grid instead "
            "(`scan_points` values per parameter, one row per candidate in "
            "`<run>_scan.csv`). Grids over 2000 candidates are refused "
            "(`--max-candidates`), and a grid needs about 15 points per parameter to "
            "converge.",
            "**Grids.** dx_um and dt_s may be left empty. `forster fit` then makes a quick "
            "fit on a coarse grid and takes the diffusion length l = sqrt(D t) at its time. "
            "For the implicit scheme (the default) it sets dx = l / 8 for the final fit and "
            "l / 4 for scans, and dt = 1/400 and 1/80 of the time (within about 0.3% and "
            "5%). For central and dipra it sets dx = l / 240 and l / 48, never more than a "
            "quarter of the narrowest gap between steps, and dt as the stable step "
            "D dt / dx² = 0.4. Values you give are kept and checked. summary.csv records the "
            "grid actually used, so a refit reproduces it.",
            "## diffusivities.csv\n\nOptional. One row per diffusivity equation, for each "
            "crystal axis i ([100], [010], [001]):\n\n"
            f"`{DIFFUSIVITY_EQUATION}`\n\n"
            "with T in K, P and fO2 in Pa and R = 8.3145 J/mol/K. Along the profile, "
            "D = sum of D_i cos²(angle to axis i). The table `forster template` writes "
            "holds the presets. Copy a row, give it a new name and change the values. "
            "A row with a preset's name must keep the preset's values.",
            column_table(DIFFUSIVITY_COLUMNS),
            "## summary.csv (written by forster fit)\n\nOne row per fitted run: `run`, "
            "`profile`, the results below, and every parameter column above.",
            column_table(SUMMARY_COLUMNS),
            "`forster fit` also writes `diffusivities.csv` to the output folder with every "
            "equation the runs used, so summary.csv refits with `forster fit profiles.csv "
            "summary.csv --diffusivities diffusivities.csv`.",
        ]
    )


def write_template(out_dir: str | Path, *, overwrite: bool = False) -> list[Path]:
    """Write ``profiles.csv``, ``parameters.csv``, ``diffusivities.csv`` and ``COLUMNS.md``.

    Refuses (FileExistsError, nothing written) if any of them exists, unless
    ``overwrite``.
    """
    out_dir = Path(out_dir)
    data = resources.files("forster").joinpath("data")
    contents = {name: data.joinpath(source).read_text() for name, source in FILES.items()}
    contents["diffusivities.csv"] = diffusivity_table(DIFFUSIVITY_PRESETS.values()).to_csv(
        index=False
    )
    contents["COLUMNS.md"] = column_guide() + "\n"
    targets = [out_dir / name for name in contents]
    existing = [str(p) for p in targets if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"would overwrite {', '.join(existing)}; use --force to replace")
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in targets:
        path.write_text(contents[path.name], encoding="utf-8")
    return targets
