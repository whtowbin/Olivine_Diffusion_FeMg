"""Read measured profiles and model parameters from spreadsheets (CSV or Excel), and
turn fit results into tables.

Three parameter-table layouts are recognised from their columns:

- template (written by ``forster template``; the column guide is
  `forster.template.PARAMETER_COLUMNS`): the Python API's names with units, Fo as
  a fraction, ``steps_um`` and ``step_fo`` for the initial condition. Orientation
  is ``alpha_deg``, ``beta_deg``, ``gamma_deg`` or EBSD Euler angles ``phi1_deg``,
  ``Phi_deg``, ``phi2_deg`` with ``profile_angle_deg``; fO2 is ``fO2_Pa`` or
  ``fO2_buffer`` with ``fO2_delta_log``;
- round robin: ``Sample`` (run name), ``File_Name`` (profile), ``alpha``,
  ``beta``, ``gamma`` (degrees to [100], [010], [001]), ``T_C``, ``P_Pa``,
  ``fO2_Pa``, ``dx_um``, ``dt_s``, optional ``max_model_time_days``;
- paper (Towbin and Plank 2026): ``File_Name`` (run and profile), EBSD Euler
  angles ``phi1``, ``Phi``, ``phi2`` with ``prof_angle``, ``T`` (degrees C),
  ``P`` (Pa), ``FO2_dQFM`` (log units relative to QFM), ``dx`` (um), ``dt`` (s).

The last two give the initial condition as ``edge_x1``, ``edge_x2`` (model domain, um),
``inflection_x`` (step positions, um; one number or text such as ``"0, 31"``),
``edge_c``, ``center_c`` and ``inflection_c`` (Fo in percent). An optional
``scheme`` column picks the finite-difference scheme per run ("central" or
"dipra"), and an optional ``uncertainty`` column the uncertainty method
("correlated" or "legacy"); see `fit_profile`.

Profile tables have one row per analysis: ``profile``, ``distance_um`` and ``fo``
(fraction) in the template layout, or ``Profile_Name``, ``Distance µm`` and
``Fo#`` (percent) in the lab layout.
"""

from __future__ import annotations

import ast
import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from forster.model import (
    FE_MG_ACTIVATION_ENERGY_J_MOL,
    SCHEMES,
    UNCERTAINTY_METHODS,
    Conditions,
    FitResult,
    Grid,
    InitialCondition,
    KrigingSettings,
    Orientation,
    Profile,
    fit_profile,
)

ROUND_ROBIN_COLUMNS = {
    "Sample",
    "File_Name",
    "alpha",
    "beta",
    "gamma",
    "T_C",
    "P_Pa",
    "fO2_Pa",
    "dx_um",
    "dt_s",
}
PAPER_COLUMNS = {"File_Name", "phi1", "Phi", "phi2", "prof_angle", "T", "P", "FO2_dQFM", "dx", "dt"}
INITIAL_COLUMNS = {"edge_x1", "edge_x2", "inflection_x", "inflection_c", "edge_c", "center_c"}
TEMPLATE_COLUMNS = {
    "run",
    "profile",
    "T_C",
    "P_Pa",
    "domain_start_um",
    "domain_end_um",
    "steps_um",
    "edge_fo",
    "core_fo",
    "dx_um",
    "dt_s",
}
# Template rows give each of these in one of two forms (see _pick_form).
ORIENTATION_FORMS = (
    ("alpha_deg", "beta_deg", "gamma_deg"),
    ("phi1_deg", "Phi_deg", "phi2_deg", "profile_angle_deg"),
)
FO2_FORMS = (("fO2_Pa",), ("fO2_buffer", "fO2_delta_log"))
METADATA_COLUMNS = ("Descriptive_Name", "Category", "Uncert_Category", "Notes", "notes")
MIN_PROFILE_POINTS = 3  # Profile needs at least this many points


class ProfileLayout(NamedTuple):
    name_column: str
    distance_column: str
    fo_column: str
    fo_in_percent: bool


TEMPLATE_PROFILE_LAYOUT = ProfileLayout("profile", "distance_um", "fo", fo_in_percent=False)
LAB_PROFILE_LAYOUT = ProfileLayout("Profile_Name", "Distance µm", "Fo#", fo_in_percent=True)


@dataclass(frozen=True)
class RunDefaults:
    """Settings a parameter table does not give, applied to every run."""

    max_time_days: float = 200.0
    kriging: KrigingSettings = field(default_factory=KrigingSettings)
    activation_energy_J_mol: float = FE_MG_ACTIVATION_ENERGY_J_MOL
    scheme: str = "central"
    uncertainty: str = "correlated"


@dataclass(frozen=True)
class Run:
    """One model run: a profile plus everything needed to fit it."""

    name: str
    profile_name: str
    conditions: Conditions
    orientation: Orientation
    initial: InitialCondition
    grid: Grid
    kriging: KrigingSettings
    activation_energy_J_mol: float
    scheme: str = "central"
    uncertainty: str = "correlated"
    metadata: Mapping[str, Any] = field(default_factory=dict)


def read_table(path: str | Path, sheet: str | int = 0) -> pd.DataFrame:
    """A CSV file, or one sheet of an Excel workbook (the first by default)."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        # pandas' default float parser can be 1 ulp off; round_trip reads back exactly
        # what was written, so a table written by FORSTER refits identically.
        # pandas documents "round_trip" but its type stubs leave it out.
        return pd.read_csv(path, float_precision="round_trip")  # ty: ignore[no-matching-overload]
    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    raise ValueError(f"{path}: expected a .csv or .xlsx file")


def _usable_rows(df: pd.DataFrame) -> pd.Series:
    """Rows not flagged as bad (``Marked_bad == "bad"``) or ignored (``Ignore == "yes"``)."""
    keep = pd.Series(True, index=df.index)
    if "Marked_bad" in df.columns:
        keep &= df["Marked_bad"] != "bad"
    if "Ignore" in df.columns:
        keep &= df["Ignore"] != "yes"
    return keep


def read_profiles(
    path: str | Path,
    *,
    sheet: str | int = 0,
    name_column: str | None = None,
    distance_column: str | None = None,
    fo_column: str | None = None,
    fo_in_percent: bool | None = None,
) -> dict[str, Profile]:
    """Profiles keyed by name, sorted by distance, with flagged rows dropped.

    The table has one row per analysis point: a profile name, the distance from
    the crystal edge (um) and Fo. Arguments left as None come from the layout:
    the template layout (``profile``, ``distance_um``, ``fo`` as a fraction) when
    the table has those columns, otherwise the lab layout (``Profile_Name``,
    ``Distance µm``, ``Fo#`` in percent). Names with fewer than
    `MIN_PROFILE_POINTS` usable points (single analyses, averages) are skipped
    with a warning that lists them.
    """
    df = read_table(path, sheet)
    layout = (
        TEMPLATE_PROFILE_LAYOUT
        if set(TEMPLATE_PROFILE_LAYOUT[:3]) <= set(df.columns)
        else LAB_PROFILE_LAYOUT
    )
    name_column = name_column or layout.name_column
    distance_column = distance_column or layout.distance_column
    fo_column = fo_column or layout.fo_column
    if fo_in_percent is None:
        fo_in_percent = layout.fo_in_percent
    missing = {name_column, distance_column, fo_column} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing column(s) {sorted(missing)}")
    keep = _usable_rows(df)
    profiles = {}
    too_short = []
    for name in df.loc[keep, name_column].dropna().unique():
        rows = df.loc[(df[name_column] == name) & keep].sort_values(distance_column)
        if len(rows) < MIN_PROFILE_POINTS:
            too_short.append(str(name))
            continue
        fo = rows[fo_column].to_numpy()
        if fo_in_percent:
            fo = fo / 100
        profiles[str(name)] = Profile(rows[distance_column].to_numpy(), fo, name=str(name))
    if too_short:
        warnings.warn(
            f"{path}: skipped {len(too_short)} name(s) with fewer than {MIN_PROFILE_POINTS} "
            f"points: {too_short}",
            stacklevel=2,
        )
    return profiles


def _steps(value: Any) -> tuple[float, ...]:
    """Step positions: a number, or text such as ``"0, 31"``."""
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            raise ValueError(
                f"step positions {value!r}: give one number, or two separated by a comma "
                'such as "10, 33"'
            ) from None
    if isinstance(value, (tuple, list)):
        return tuple(value)
    return (value,)


def _plateau_fraction(value: Any) -> float | None:
    """Intermediate plateau Fo, percent to fraction; None when the cell is empty."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, str):
        parsed = ast.literal_eval(value)
        values = parsed if isinstance(parsed, (tuple, list)) else [parsed]
        if len(values) != 1:
            raise ValueError(f"inflection_c {value!r}: only one intermediate plateau is supported")
        value = values[0]
    return value / 100


def _initial(row: pd.Series) -> InitialCondition:
    return InitialCondition(
        domain_um=(row["edge_x1"], row["edge_x2"]),
        edge_fo=row["edge_c"] / 100,
        core_fo=row["center_c"] / 100,
        steps_um=_steps(row["inflection_x"]),
        step_fo=_plateau_fraction(row["inflection_c"]),
    )


def _given(row: pd.Series, column: str) -> bool:
    return column in row.index and not pd.isna(row[column])


def _max_time(row: pd.Series, defaults: RunDefaults, column: str = "max_model_time_days") -> float:
    if not _given(row, column):
        return defaults.max_time_days
    return float(row[column])


def _scheme(row: pd.Series, defaults: RunDefaults) -> str:
    """Optional ``scheme`` column ("central" or "dipra"); the default otherwise."""
    value = row.get("scheme")
    if value is None or pd.isna(value):
        return defaults.scheme
    value = str(value).strip().lower()
    if value not in SCHEMES:
        raise ValueError(
            f"scheme {value!r} in the parameter table; use one of {', '.join(SCHEMES)}"
        )
    return value


def _uncertainty(row: pd.Series, defaults: RunDefaults) -> str:
    """Optional ``uncertainty`` column ("correlated" or "legacy"); the default otherwise."""
    value = row.get("uncertainty")
    if value is None or pd.isna(value):
        return defaults.uncertainty
    value = str(value).strip().lower()
    if value not in UNCERTAINTY_METHODS:
        raise ValueError(
            f"uncertainty {value!r} in the parameter table; use one of "
            f"{', '.join(UNCERTAINTY_METHODS)}"
        )
    return value


def _round_robin_run(row: pd.Series, defaults: RunDefaults) -> Run:
    return Run(
        name=str(row["Sample"]),
        profile_name=str(row["File_Name"]),
        conditions=Conditions(T_C=row["T_C"], P_Pa=row["P_Pa"], fO2_Pa=row["fO2_Pa"]),
        orientation=Orientation(row["alpha"], row["beta"], row["gamma"]),
        initial=_initial(row),
        grid=Grid(dx_um=row["dx_um"], dt_s=row["dt_s"], max_time_days=_max_time(row, defaults)),
        kriging=defaults.kriging,
        activation_energy_J_mol=defaults.activation_energy_J_mol,
        scheme=_scheme(row, defaults),
        uncertainty=_uncertainty(row, defaults),
        metadata={c: row[c] for c in METADATA_COLUMNS if c in row.index and not pd.isna(row[c])},
    )


def _paper_run(row: pd.Series, defaults: RunDefaults) -> Run:
    name = str(row["File_Name"])
    return Run(
        name=name,
        profile_name=name,
        conditions=Conditions.from_buffer(row["T"], row["P"], row["FO2_dQFM"], "QFM"),
        orientation=Orientation.from_euler(row["phi1"], row["Phi"], row["phi2"], row["prof_angle"]),
        initial=_initial(row),
        grid=Grid(dx_um=row["dx"], dt_s=row["dt"], max_time_days=_max_time(row, defaults)),
        kriging=defaults.kriging,
        activation_energy_J_mol=defaults.activation_energy_J_mol,
        scheme=_scheme(row, defaults),
        uncertainty=_uncertainty(row, defaults),
        metadata={c: row[c] for c in METADATA_COLUMNS if c in row.index and not pd.isna(row[c])},
    )


def _pick_form(row: pd.Series, forms: tuple[tuple[str, ...], ...], what: str) -> int:
    """Which of two column groups a template row fills in: 0 or 1.

    Exactly one group must be complete and the other empty.
    """
    started = [any(_given(row, c) for c in form) for form in forms]
    options = " or ".join(f"({', '.join(form)})" for form in forms)
    if all(started):
        raise ValueError(f"run {row['run']!r}: give {what} as {options}, not both")
    for i, form in enumerate(forms):
        if started[i]:
            missing = [c for c in form if not _given(row, c)]
            if missing:
                raise ValueError(f"run {row['run']!r}: {what} is missing {', '.join(missing)}")
            return i
    raise ValueError(f"run {row['run']!r}: give {what} as {options}")


def _template_conditions(row: pd.Series) -> Conditions:
    if _pick_form(row, FO2_FORMS, "fO2") == 0:
        return Conditions(T_C=row["T_C"], P_Pa=row["P_Pa"], fO2_Pa=row["fO2_Pa"])
    return Conditions.from_buffer(
        row["T_C"], row["P_Pa"], row["fO2_delta_log"], str(row["fO2_buffer"]).strip()
    )


def _template_orientation(row: pd.Series) -> Orientation:
    if _pick_form(row, ORIENTATION_FORMS, "the orientation") == 0:
        return Orientation(row["alpha_deg"], row["beta_deg"], row["gamma_deg"])
    return Orientation.from_euler(
        row["phi1_deg"], row["Phi_deg"], row["phi2_deg"], row["profile_angle_deg"]
    )


def _template_run(row: pd.Series, defaults: RunDefaults) -> Run:
    return Run(
        name=str(row["run"]),
        profile_name=str(row["profile"]),
        conditions=_template_conditions(row),
        orientation=_template_orientation(row),
        initial=InitialCondition(
            domain_um=(row["domain_start_um"], row["domain_end_um"]),
            edge_fo=row["edge_fo"],
            core_fo=row["core_fo"],
            steps_um=_steps(row["steps_um"]),
            step_fo=float(row["step_fo"]) if _given(row, "step_fo") else None,
        ),
        grid=Grid(
            dx_um=row["dx_um"],
            dt_s=row["dt_s"],
            max_time_days=_max_time(row, defaults, "max_time_days"),
        ),
        kriging=defaults.kriging,
        activation_energy_J_mol=defaults.activation_energy_J_mol,
        scheme=_scheme(row, defaults),
        uncertainty=_uncertainty(row, defaults),
        metadata={c: row[c] for c in METADATA_COLUMNS if _given(row, c)},
    )


# (name, columns that identify it, run-name column, row reader), tried in order.
LAYOUTS = (
    ("Template layout", TEMPLATE_COLUMNS, "run", _template_run),
    ("Round robin layout", ROUND_ROBIN_COLUMNS | INITIAL_COLUMNS, "Sample", _round_robin_run),
    ("paper layout", PAPER_COLUMNS | INITIAL_COLUMNS, "File_Name", _paper_run),
)


def read_runs(
    path: str | Path, *, sheet: str | int = 0, defaults: RunDefaults | None = None
) -> list[Run]:
    """One `Run` per named row of a parameter table, in table order."""
    defaults = defaults or RunDefaults()
    df = read_table(path, sheet)
    columns = set(df.columns)
    layout = next((layout for layout in LAYOUTS if columns >= layout[1]), None)
    if layout is None:
        needs = "; ".join(
            f"{label} needs {sorted(required - columns)}" for label, required, _, _ in LAYOUTS
        )
        raise ValueError(
            f"{path}: not a recognised parameter table (see `forster template`). {needs}"
        )
    _, _, name_column, make_run = layout
    named = df[df[name_column].notna()]
    duplicates = named[name_column][named[name_column].duplicated()].unique()
    if len(duplicates):
        raise ValueError(f"{path}: run names must be unique; repeated: {list(duplicates)}")
    return [make_run(row, defaults) for _, row in named.iterrows()]


def _steps_text(steps: tuple[float, ...]) -> str:
    """``"10, 33"``; exact, so the text reads back as the same numbers."""
    return ", ".join(repr(float(s)).removesuffix(".0") for s in steps)


def summary_row(result: FitResult) -> dict[str, Any]:
    """The fitted time and every input of one fit, as one table row.

    Input columns use the template layout's names, so a table of these rows (with a
    ``profile`` column added) reads back with `read_runs` as the same runs. fO2 is
    given in Pa and the orientation as angles to the crystal axes, whatever form
    the parameter table used.
    """
    inputs = result.inputs
    conditions: Conditions = inputs["conditions"]
    orientation: Orientation = inputs["orientation"]
    initial: InitialCondition = inputs["initial"]
    grid: Grid = inputs["grid"]
    noise = result.noise
    n_fitted = len(result.fit_x_um) if result.fit_x_um is not None else len(result.model_x_um)
    return {
        "run": result.name,
        "time_days": result.time_days,
        "low_days": result.time_days + result.time_minus_days,
        "high_days": result.time_days + result.time_plus_days,
        "reduced_chi2": float(result.reduced_chi2[result.indices[0]]),
        "points_fitted": n_fitted,
        "uncertainty": result.uncertainty,
        "scheme": inputs["scheme"],
        "T_C": conditions.T_C,
        "P_Pa": conditions.P_Pa,
        "fO2_Pa": conditions.fO2_Pa,
        "alpha_deg": orientation.alpha_deg,
        "beta_deg": orientation.beta_deg,
        "gamma_deg": orientation.gamma_deg,
        "domain_start_um": initial.domain_um[0],
        "domain_end_um": initial.domain_um[1],
        "steps_um": _steps_text(initial.steps_um),
        "edge_fo": initial.edge_fo,
        "core_fo": initial.core_fo,
        "step_fo": initial.step_fo,
        "dx_um": grid.dx_um,
        "dt_s": grid.dt_s,
        "max_time_days": grid.max_time_days,
        "activation_energy_J_mol": inputs["activation_energy_J_mol"],
        "white_sigma_fo": noise.white_sigma if noise else math.nan,
        "correlated_sigma_fo": noise.correlated_sigma if noise else math.nan,
        "correlation_length_um": noise.correlation_length_um if noise else math.nan,
    }


def fit_tables(result: FitResult) -> dict[str, pd.DataFrame]:
    """The curves and data behind a fit, keyed by a short name for the file.

    - "curves": the model on its grid at time zero, the best fit, and the time
      steps at the low and high ends of the 1-sigma range;
    - "kriged": the kriged profile and its uncertainty (fitted only by the legacy
      method);
    - "points": the measured points that were fitted, with the best fit
      interpolated to them and the residual (legacy: the kriged grid points).
    """
    _, low, high = result.indices
    x = result.fit_x_um if result.fit_x_um is not None else result.model_x_um
    fo = result.fit_fo if result.fit_fo is not None else result.interp_fo
    at_points = np.interp(x, result.model_x_um, result.best_fit)
    return {
        "curves": pd.DataFrame(
            {
                "distance_um": result.model_x_um,
                "initial_fo": result.initial,
                "best_fit_fo": result.best_fit,
                "minus_1sigma_fo": result.curves[low],
                "plus_1sigma_fo": result.curves[high],
            }
        ),
        "kriged": pd.DataFrame(
            {
                "distance_um": result.interp_x_um,
                "fo": result.interp_fo,
                "sigma_fo": result.interp_fo_sigma,
            }
        ),
        "points": pd.DataFrame(
            {"distance_um": x, "fo": fo, "best_fit_fo": at_points, "residual_fo": fo - at_points}
        ),
    }


def fit_run(run: Run, profiles: Mapping[str, Profile]) -> FitResult:
    """Fit one run against its measured profile."""
    if run.profile_name not in profiles:
        raise KeyError(
            f"run {run.name!r} needs profile {run.profile_name!r}, which is not in the profile "
            f"table (available: {sorted(profiles)})"
        )
    measured = profiles[run.profile_name]
    return fit_profile(
        Profile(measured.distance_um, measured.fo, name=run.name),
        run.conditions,
        run.orientation,
        run.initial,
        run.grid,
        run.kriging,
        run.activation_energy_J_mol,
        run.scheme,
        run.uncertainty,
    )
