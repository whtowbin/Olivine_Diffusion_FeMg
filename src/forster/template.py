"""Parameter and profile tables to fill in, with a worked example and a column guide.

`write_template` writes ``profiles.csv`` and ``parameters.csv`` holding one
published profile (AZ18 WHT01_Ol_63_prof_1 from Towbin and Plank 2026, a
melt-vein olivine), so the template fits as it is, and ``COLUMNS.md``, generated
from `PROFILE_COLUMNS` and `PARAMETER_COLUMNS`. The tables are read by
`forster.io.read_profiles` and `forster.io.read_runs` (the "template" layout).
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import NamedTuple


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
        "required",
        "Grid spacing. Keep D dt / dx² below 0.5: the explicit scheme is unstable above it, "
        "and this is not checked.",
    ),
    Column("dt_s", "s", "required", "Time step."),
    Column("max_time_days", "days", "optional", "Longest time modelled (default 200)."),
    Column("scheme", "", "optional", "central (default) or dipra (Girona and Costa 2013)."),
    Column(
        "uncertainty",
        "",
        "optional",
        "correlated (default) or legacy (the round robin rule; reproduces old results).",
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
        "activation_energy_J_mol",
        "J/mol",
        "",
        "Fe-Mg activation energy used (not read back; set it in Python with RunDefaults).",
    ),
    Column(
        "warnings",
        "",
        "",
        "Warnings from the fit, such as a range that reaches the last time step.",
    ),
    Column("forster_version", "", "", "FORSTER version that made the fit."),
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
            "## summary.csv (written by forster fit)\n\nOne row per fitted run: `run`, "
            "`profile`, the results below, and every parameter column above.",
            column_table(SUMMARY_COLUMNS),
        ]
    )


def write_template(out_dir: str | Path, *, overwrite: bool = False) -> list[Path]:
    """Write ``profiles.csv``, ``parameters.csv`` and ``COLUMNS.md`` into ``out_dir``.

    Refuses (FileExistsError, nothing written) if any of them exists, unless
    ``overwrite``.
    """
    out_dir = Path(out_dir)
    data = resources.files("forster").joinpath("data")
    contents = {name: data.joinpath(source).read_text() for name, source in FILES.items()}
    contents["COLUMNS.md"] = column_guide() + "\n"
    targets = [out_dir / name for name in contents]
    existing = [str(p) for p in targets if p.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"would overwrite {', '.join(existing)}; use --force to replace")
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in targets:
        path.write_text(contents[path.name], encoding="utf-8")
    return targets
