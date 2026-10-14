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

from forster.diffusivity import DEFAULT_DIFFUSIVITY, DIFFUSIVITY_PRESETS, diffusivity_table


class Column(NamedTuple):
    name: str
    unit: str
    need: str  # "required", "optional" or "one form": see the guide's note
    description: str


PROFILE_COLUMNS = (
    Column("profile", "", "required", "Profile name; `profile` in parameters.csv refers to it."),
    Column("distance_um", "µm", "required", "Distance of the analysis from the crystal edge."),
    Column("fo", "fraction", "required", "Forsterite content, 0-1 (0.895, not 89.5)."),
)

PARAMETER_COLUMNS = (
    Column("run", "", "required", "Name of the model run (unique); names the output files."),
    Column("profile", "", "required", "Which profile in profiles.csv to fit."),
    Column("T_C", "°C", "required", "Temperature."),
    Column("P_Pa", "Pa", "required", "Pressure (5e8 is 0.5 GPa)."),
    Column("fO2_Pa", "Pa", "one form", "Oxygen fugacity. Or give the next two columns."),
    Column("fO2_buffer", "", "one form", "QFM (or FMQ) or NNO (Frost 1991)."),
    Column("fO2_delta_log", "log units", "one form", "fO2 relative to that buffer."),
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
        "Direction of the profile on the EBSD map, from the positive x axis (east).",
    ),
    Column(
        "domain_start_um",
        "µm",
        "required",
        "Start of the model domain; Fo is held at edge_fo here.",
    ),
    Column(
        "domain_end_um", "µm", "required", "End of the model domain; Fo is held at core_fo here."
    ),
    Column(
        "steps_um",
        "µm",
        "required",
        'Step position(s) of the initial profile: one number, or two in quotes ("10, 33").',
    ),
    Column("edge_fo", "fraction", "required", "Initial Fo before the first step (edge side)."),
    Column("core_fo", "fraction", "required", "Initial Fo after the last step (core side)."),
    Column(
        "step_fo",
        "fraction",
        "optional",
        "Initial Fo of the plateau between two steps; required with two steps.",
    ),
    Column(
        "dx_um",
        "µm",
        "optional",
        "Grid spacing of the final fit. Leave empty to have it chosen for about 1% "
        "accuracy in the fitted time; a value you give is kept, with a warning if it is "
        "coarse, wastefully fine, or too coarse for the initial steps.",
    ),
    Column(
        "dt_s",
        "s",
        "optional",
        "Time step of the final fit. Leave empty to have it chosen: 1/400 of the fitted "
        "time (implicit), or the stable step D dt / dx² = 0.4 (central, dipra; above 0.5 "
        "is unstable and refused).",
    ),
    Column(
        "max_time_days",
        "days",
        "optional",
        "Longest time modelled (default 200; with dx_um or dt_s empty, 3x a quick estimate "
        "of the time).",
    ),
    Column(
        "scheme",
        "",
        "optional",
        "implicit (default since 2026-10-13: rim on the end node, steps not snapped to "
        "nodes, any time step), central (the engine's explicit scheme, used before) or "
        "dipra (Girona and Costa 2013). With uncertainty = legacy the default is central.",
    ),
    Column(
        "diffusivity",
        "",
        "optional",
        f"Fe-Mg diffusivity equation: a preset ({DEFAULT_DIFFUSIVITY}, the default and "
        "DIPRA's, or mutch2019_tamed, mutch2019_global) or a name from diffusivities.csv "
        "(then run forster fit with --diffusivities).",
    ),
    Column(
        "uncertainty",
        "",
        "optional",
        "correlated (default) or legacy (the round robin rule; reproduces old results).",
    ),
    Column(
        "steps_sd_um",
        "µm",
        "optional",
        "Uncertainty (1 sigma, Gaussian) on the step position(s): one value, or one per step "
        '("2, 5"); 0 keeps a step fixed. Or give steps_range_um.',
    ),
    Column(
        "steps_range_um",
        "µm",
        "optional",
        "Uncertainty (± half-width, uniform) on the step position(s), as steps_sd_um.",
    ),
    Column("edge_fo_sd", "fraction", "optional", "1-sigma uncertainty on edge_fo (Gaussian)."),
    Column("edge_fo_range", "fraction", "optional", "± half-width on edge_fo (uniform)."),
    Column("core_fo_sd", "fraction", "optional", "1-sigma uncertainty on core_fo (Gaussian)."),
    Column("core_fo_range", "fraction", "optional", "± half-width on core_fo (uniform)."),
    Column("step_fo_sd", "fraction", "optional", "1-sigma uncertainty on step_fo (Gaussian)."),
    Column("step_fo_range", "fraction", "optional", "± half-width on step_fo (uniform)."),
    Column(
        "scan_points",
        "",
        "optional",
        "Grid points per uncertain parameter (default 13 for one, 9 for two, 5 for more).",
    ),
    Column(
        "scan_dx_um",
        "µm",
        "optional",
        "Grid spacing of the initial-condition scan's runs. Empty: chosen for about 5% "
        "accuracy (coarser and faster than the final fit).",
    ),
    Column(
        "scan_dt_s",
        "s",
        "optional",
        "Time step of the scan's runs. Empty: D dt / dx² = 0.4 for every candidate.",
    ),
    Column(
        "scan_max_time_days",
        "days",
        "optional",
        "Longest time the scan's runs model. Empty: max_time_days if given, else 3x the "
        "fitted time.",
    ),
    Column("notes", "", "optional", "Free text for your own records."),
)

# Columns of summary.csv besides the parameter columns above (which it repeats, with
# fO2 in Pa and the orientation as angles, so it reads back as a parameter table).
SUMMARY_COLUMNS = (
    Column("time_days", "days", "", "Best-fit time."),
    Column("low_days", "days", "", "Low end of the 1-sigma range."),
    Column("high_days", "days", "", "High end of the 1-sigma range."),
    Column(
        "reduced_chi2",
        "",
        "",
        "Chi-squared at the best fit over (points fitted - 1). Diagnostic only: about 1 by "
        "construction for the correlated method, whose noise model is fitted to the "
        "residuals; for legacy it is over the kriged grid points, so the two are not "
        "comparable. Judge correlated fits by the three noise columns.",
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
        "Fitted scatter of the points (correlated method). Much larger than your "
        "analytical precision: outliers, or the precision is underestimated.",
    ),
    Column(
        "correlated_sigma_fo",
        "fraction",
        "",
        "Fitted smooth misfit between model and data. Large next to white_sigma_fo: the "
        "model misses the shape of the profile; check the initial condition.",
    ),
    Column(
        "correlation_length_um",
        "µm",
        "",
        "Length scale of that misfit; near half the profile length, the misfit is profile-wide.",
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
        "With uncertainties: best-fit time over all scanned initial conditions.",
    ),
    Column(
        "scan_low_days",
        "days",
        "",
        "Low end of the 1-sigma range including the initial-condition uncertainty (profile "
        "likelihood; use this one).",
    ),
    Column("scan_high_days", "days", "", "High end of that range."),
    Column(
        "scan_marginal_low_days",
        "days",
        "",
        "Low end of the Bayesian (marginal) range; with Gaussian uncertainties it is pulled "
        "toward the nominal initial condition.",
    ),
    Column("scan_marginal_high_days", "days", "", "High end of the Bayesian range."),
    Column(
        "scan_best",
        "",
        "",
        "Best values of the scanned parameters (step1_um, edge_fo, ...); the nominal "
        "values stay in the parameter columns.",
    ),
    Column("scan_candidates", "", "", "Initial conditions fitted (one model run each)."),
    Column(
        "scan_dropped",
        "",
        "",
        "Grid points left out as invalid (steps out of order or domain, Fo outside 0-1).",
    ),
    Column(
        "grid_resolution",
        "",
        "",
        "Diffusion length sqrt(D t) at the fit over the grid spacing: for 1% accuracy in "
        "the fitted time about 5 (implicit) or 240 (central, dipra).",
    ),
    Column(
        "grid_error_pct",
        "%",
        "",
        "Estimated worst-case error of the fitted time from the grid spacing (implicit: "
        "0.2 x (dx / l)²; central, dipra: 2.4 x dx / l; l the diffusion length; measured "
        "on 35 real profiles).",
    ),
    Column(
        "grid_stability",
        "",
        "",
        "D dt / dx² of the final fit (central, dipra: stable up to 0.5; empty for implicit).",
    ),
)

# Columns of diffusivities.csv; the equation is in forster.diffusivity.
DIFFUSIVITY_COLUMNS = (
    Column("name", "", "required", "Name to use in the `diffusivity` column of parameters.csv."),
    Column("element", "", "required", "Fe-Mg for fits (Ni, Mn and Ca are listed for reference)."),
    Column("log10_D0_100", "log10 m²/s", "required", "Pre-factor along [100]."),
    Column("log10_D0_010", "log10 m²/s", "required", "Pre-factor along [010]."),
    Column("log10_D0_001", "log10 m²/s", "required", "Pre-factor along [001]."),
    Column("E_100_J_mol", "J/mol", "required", "Activation energy along [100]."),
    Column("E_010_J_mol", "J/mol", "required", "Activation energy along [010]."),
    Column("E_001_J_mol", "J/mol", "required", "Activation energy along [001]."),
    Column("fO2_exponent", "", "required", "Exponent n of (fO2 / fO2_ref_Pa)."),
    Column("fO2_ref_Pa", "Pa", "required", "Reference fO2 (1e5 if the equation uses bar)."),
    Column(
        "fo_coefficient", "log10 per unit Fo", "required", "m in m (fo_ref - X_Fo); 0 for none."
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
    """Markdown description of every template column (also in README.md)."""
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
            "concentration (`_sd` for a Gaussian 1 sigma, or `_range` for a uniform ± "
            "half-width; not both for one parameter) and `forster fit` also fits a grid of "
            "initial conditions around the nominal one: Gaussian parameters over ±2.5 "
            "sigma, uniform ones over the whole range. It reports the range of time "
            "including that uncertainty (`scan_*` columns in summary.csv) and writes "
            "`<run>_scan.csv`, one row per candidate. Edge and core Fo are the boundary "
            "values held during diffusion, so their uncertainty also covers the boundary "
            "conditions. The grid has `scan_points` values per parameter; the number of "
            "model runs is their product, printed before the scan, and grids over 2000 "
            "candidates are refused (`--max-candidates`). The scan's runs share one grid "
            "with a larger time step (D dt / dx² = 0.4), which changed fitted times by "
            "under 0.2% in tests.",
            "**Grids.** dx_um and dt_s may be left empty: `forster fit` then makes a quick "
            "fit on a coarse grid, takes the diffusion length l = sqrt(D t) at its time, "
            "and, for the implicit scheme (the default), sets dx = l / 8 for the final fit "
            "and l / 4 for scans and dt = 1/400 and 1/80 of the time (within about 0.3% and "
            "5%). For central and dipra: dx = l / 240 and l / 48, never more than a quarter "
            "of the narrowest gap between steps, and dt the stable step D dt / dx² = 0.4. "
            "Values you give are kept and checked. summary.csv records the grid actually "
            "used, so a refit reproduces it.",
            "## diffusivities.csv\n\nOptional. One row per diffusivity equation, for each "
            "crystal axis i ([100], [010], [001]):\n\n"
            f"`{DIFFUSIVITY_EQUATION}`\n\n"
            "with T in K, P and fO2 in Pa and R = 8.3145 J/mol/K. Along the profile, "
            "D = sum of D_i cos²(angle to axis i). The table `forster template` writes "
            "holds the presets; copy a row, give it a new name and change the values. "
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
