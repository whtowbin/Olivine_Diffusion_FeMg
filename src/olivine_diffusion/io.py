"""Read measured profiles and model parameters from spreadsheets (CSV or Excel).

Two parameter-table layouts are recognised from their columns:

- round robin: ``Sample`` (run name), ``File_Name`` (profile), ``alpha``,
  ``beta``, ``gamma`` (degrees to [100], [010], [001]), ``T_C``, ``P_Pa``,
  ``fO2_Pa``, ``dx_um``, ``dt_s``, optional ``max_model_time_days``;
- paper (Towbin and Plank 2026): ``File_Name`` (run and profile), EBSD Euler
  angles ``phi1``, ``Phi``, ``phi2`` with ``prof_angle``, ``T`` (degrees C),
  ``P`` (Pa), ``FO2_dQFM`` (log units relative to QFM), ``dx`` (um), ``dt`` (s).

Both give the initial condition as ``edge_x1``, ``edge_x2`` (model domain, um),
``inflection_x`` (step positions, um; one number or text such as ``"0, 31"``),
``edge_c``, ``center_c`` and ``inflection_c`` (Fo in percent). An optional
``scheme`` column picks the finite-difference scheme per run ("central" or
"dipra"; see `fit_profile`).
"""

from __future__ import annotations

import ast
import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from olivine_diffusion.model import (
    FE_MG_ACTIVATION_ENERGY_J_MOL,
    SCHEMES,
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
METADATA_COLUMNS = ("Descriptive_Name", "Category", "Uncert_Category", "Notes")
MIN_PROFILE_POINTS = 3  # Profile needs at least this many points


@dataclass(frozen=True)
class RunDefaults:
    """Settings a parameter table does not give, applied to every run."""

    max_time_days: float = 200.0
    kriging: KrigingSettings = field(default_factory=KrigingSettings)
    activation_energy_J_mol: float = FE_MG_ACTIVATION_ENERGY_J_MOL
    scheme: str = "central"


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
    metadata: Mapping[str, Any] = field(default_factory=dict)


def read_table(path: str | Path, sheet: str | int = 0) -> pd.DataFrame:
    """A CSV file, or one sheet of an Excel workbook (the first by default)."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
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
    name_column: str = "Profile_Name",
    distance_column: str = "Distance µm",
    fo_column: str = "Fo#",
    fo_in_percent: bool = True,
) -> dict[str, Profile]:
    """Profiles keyed by name, sorted by distance, with flagged rows dropped.

    The table has one row per analysis point: a profile name, the distance from
    the crystal edge (um) and Fo (percent by default). Names with fewer than
    `MIN_PROFILE_POINTS` usable points (single analyses, averages) are skipped
    with a warning that lists them.
    """
    df = read_table(path, sheet)
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
        value = ast.literal_eval(value)
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


def _max_time(row: pd.Series, defaults: RunDefaults) -> float:
    value = row.get("max_model_time_days")
    if value is None or pd.isna(value):
        return defaults.max_time_days
    return float(value)


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
        metadata={c: row[c] for c in METADATA_COLUMNS if c in row.index and not pd.isna(row[c])},
    )


def read_runs(
    path: str | Path, *, sheet: str | int = 0, defaults: RunDefaults | None = None
) -> list[Run]:
    """One `Run` per named row of a parameter table, in table order."""
    defaults = defaults or RunDefaults()
    df = read_table(path, sheet)
    columns = set(df.columns)
    if INITIAL_COLUMNS - columns:
        raise ValueError(f"{path}: missing column(s) {sorted(INITIAL_COLUMNS - columns)}")
    if columns >= ROUND_ROBIN_COLUMNS:
        name_column, make_run = "Sample", _round_robin_run
    elif columns >= PAPER_COLUMNS:
        name_column, make_run = "File_Name", _paper_run
    else:
        raise ValueError(
            f"{path}: not a recognised parameter table. Round robin layout needs "
            f"{sorted(ROUND_ROBIN_COLUMNS - columns)}; paper layout needs "
            f"{sorted(PAPER_COLUMNS - columns)}"
        )
    named = df[df[name_column].notna()]
    duplicates = named[name_column][named[name_column].duplicated()].unique()
    if len(duplicates):
        raise ValueError(f"{path}: run names must be unique; repeated: {list(duplicates)}")
    return [make_run(row, defaults) for _, row in named.iterrows()]


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
    )
