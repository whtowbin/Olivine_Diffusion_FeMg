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
"dipra"), an optional ``uncertainty`` column the uncertainty method
("correlated" or "legacy"), and an optional ``diffusivity`` column the
diffusivity equation (a preset, or a row of a diffusivities table read with
`read_diffusivities`); see `fit_profile`.

Profile tables have one row per analysis: ``profile``, ``distance_um`` and ``fo``
(fraction) in the template layout, or ``Profile_Name``, ``Distance µm`` and
``Fo#`` (percent) in the lab layout.
"""

from __future__ import annotations

import ast
import math
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from forster.diffusivity import (
    DEFAULT_DIFFUSIVITY,
    ArrheniusDiffusivity,
    Diffusivity,
    from_table,
    resolve,
)
from forster.grid import (
    STABILITY_RATIO,
    GridRequest,
    check_grid,
    grid_diagnostics,
)
from forster.model import (
    DEFAULT_SCHEME,
    SCHEMES,
    UNCERTAINTY_METHODS,
    Conditions,
    FitResult,
    Grid,
    InitialCondition,
    KrigingSettings,
    Orientation,
    Profile,
    default_scheme,
    fit_profile,
)
from forster.scan import (
    MAX_CANDIDATES,
    ScanResult,
    Uncertainty,
    candidate_grid,
    scan_initial_conditions,
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
}
# Template rows give each of these in one of two forms (see _pick_form).
ORIENTATION_FORMS = (
    ("alpha_deg", "beta_deg", "gamma_deg"),
    ("phi1_deg", "Phi_deg", "phi2_deg", "profile_angle_deg"),
)
FO2_FORMS = (("fO2_Pa",), ("fO2_buffer", "fO2_delta_log"))
# Initial-condition uncertainties (template layout): per parameter a Gaussian sigma
# (``_sd``) or a uniform half-width (``_range``), not both. Steps take one value for all
# steps or a list, one per step ("2, 5"); 0 or empty leaves a parameter fixed.
UNCERTAINTY_COLUMNS = (
    ("steps", "steps_sd_um", "steps_range_um"),
    ("edge_fo", "edge_fo_sd", "edge_fo_range"),
    ("core_fo", "core_fo_sd", "core_fo_range"),
    ("step_fo", "step_fo_sd", "step_fo_range"),
)
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
    diffusivity: Diffusivity | str = DEFAULT_DIFFUSIVITY
    scheme: str | None = None  # None: `forster.model.default_scheme` of the uncertainty
    uncertainty: str = "correlated"


@dataclass(frozen=True)
class Run:
    """One model run: a profile plus everything needed to fit it."""

    name: str
    profile_name: str
    conditions: Conditions
    orientation: Orientation
    initial: InitialCondition
    grid: Grid | GridRequest  # a GridRequest has parts left to `forster.grid.choose_grid`
    kriging: KrigingSettings
    diffusivity: Diffusivity
    scheme: str = DEFAULT_SCHEME
    uncertainty: str = "correlated"
    metadata: Mapping[str, Any] = field(default_factory=dict)
    # Uncertain initial-condition parameters (template layout): scanned by `scan_run`.
    uncertainties: Mapping[str, Uncertainty] = field(default_factory=dict)
    scan_points: int | None = None
    # The scan's grid: parts left as None are chosen by `forster.grid.choose_grid`.
    scan_grid: GridRequest = field(default_factory=GridRequest)


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
    """Optional ``scheme`` column ("implicit", "central" or "dipra"); otherwise the
    default's, or `default_scheme` of the run's uncertainty method."""
    value = row.get("scheme")
    if value is None or pd.isna(value):
        return defaults.scheme or default_scheme(_uncertainty(row, defaults))
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


def _diffusivity(
    row: pd.Series, defaults: RunDefaults, custom: Mapping[str, Diffusivity]
) -> Diffusivity:
    """Optional ``diffusivity`` column: a preset name or a name from the diffusivities
    table; the default otherwise."""
    value = row.get("diffusivity")
    if value is None or pd.isna(value):
        return resolve(defaults.diffusivity, custom)
    return resolve(str(value).strip(), custom)


def read_diffusivities(
    path: str | Path, *, sheet: str | int = 0
) -> dict[str, ArrheniusDiffusivity]:
    """Diffusivity equations from a table (see `forster.diffusivity.TABLE_COLUMNS`), by name."""
    return from_table(read_table(path, sheet), origin=str(path))


def _round_robin_run(
    row: pd.Series, defaults: RunDefaults, custom: Mapping[str, Diffusivity]
) -> Run:
    return Run(
        name=str(row["Sample"]),
        profile_name=str(row["File_Name"]),
        conditions=Conditions(T_C=row["T_C"], P_Pa=row["P_Pa"], fO2_Pa=row["fO2_Pa"]),
        orientation=Orientation(row["alpha"], row["beta"], row["gamma"]),
        initial=_initial(row),
        grid=Grid(dx_um=row["dx_um"], dt_s=row["dt_s"], max_time_days=_max_time(row, defaults)),
        kriging=defaults.kriging,
        diffusivity=_diffusivity(row, defaults, custom),
        scheme=_scheme(row, defaults),
        uncertainty=_uncertainty(row, defaults),
        metadata={c: row[c] for c in METADATA_COLUMNS if c in row.index and not pd.isna(row[c])},
    )


def _paper_run(row: pd.Series, defaults: RunDefaults, custom: Mapping[str, Diffusivity]) -> Run:
    name = str(row["File_Name"])
    return Run(
        name=name,
        profile_name=name,
        conditions=Conditions.from_buffer(row["T"], row["P"], row["FO2_dQFM"], "QFM"),
        orientation=Orientation.from_euler(row["phi1"], row["Phi"], row["phi2"], row["prof_angle"]),
        initial=_initial(row),
        grid=Grid(dx_um=row["dx"], dt_s=row["dt"], max_time_days=_max_time(row, defaults)),
        kriging=defaults.kriging,
        diffusivity=_diffusivity(row, defaults, custom),
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


def _uncertainty_source(row: pd.Series, sd_col: str, range_col: str) -> tuple[str, str] | None:
    """The column a row gives for one parameter's uncertainty and its kind, or None."""
    given = [c for c in (sd_col, range_col) if _given(row, c)]
    if not given:
        return None
    if len(given) == 2:
        raise ValueError(f"run {row['run']!r}: give {sd_col} or {range_col}, not both")
    column = given[0]
    return column, "normal" if column == sd_col else "uniform"


def _step_uncertainties(
    row: pd.Series, column: str, kind: str, initial: InitialCondition
) -> dict[str, Uncertainty]:
    """One spread for every step, or one per step; 0 leaves a step fixed."""
    spreads = [float(v) for v in _steps(row[column])]
    n = len(initial.steps_um)
    if len(spreads) == 1:
        spreads *= n
    if len(spreads) != n:
        raise ValueError(f"run {row['run']!r}: {column} has {len(spreads)} values for {n} steps")
    return {f"step{i + 1}_um": Uncertainty(s, kind) for i, s in enumerate(spreads) if s}


def _value_uncertainty(
    row: pd.Series, what: str, column: str, kind: str, initial: InitialCondition
) -> dict[str, Uncertainty]:
    """A concentration's uncertainty (edge, core or plateau Fo); 0 leaves it fixed."""
    if what == "step_fo" and initial.step_fo is None:
        raise ValueError(f"run {row['run']!r}: {column} needs a plateau (step_fo)")
    spread = float(row[column])
    return {what: Uncertainty(spread, kind)} if spread else {}


def _template_uncertainties(row: pd.Series, initial: InitialCondition) -> dict[str, Uncertainty]:
    """The ``_sd`` / ``_range`` columns of a template row, by `forster.scan` parameter name."""
    out: dict[str, Uncertainty] = {}
    for what, sd_col, range_col in UNCERTAINTY_COLUMNS:
        source = _uncertainty_source(row, sd_col, range_col)
        if source is None:
            continue
        column, kind = source
        if what == "steps":
            out.update(_step_uncertainties(row, column, kind, initial))
        else:
            out.update(_value_uncertainty(row, what, column, kind, initial))
    return out


def _template_run(row: pd.Series, defaults: RunDefaults, custom: Mapping[str, Diffusivity]) -> Run:
    initial = InitialCondition(
        domain_um=(row["domain_start_um"], row["domain_end_um"]),
        edge_fo=row["edge_fo"],
        core_fo=row["core_fo"],
        steps_um=_steps(row["steps_um"]),
        step_fo=float(row["step_fo"]) if _given(row, "step_fo") else None,
    )
    return Run(
        name=str(row["run"]),
        profile_name=str(row["profile"]),
        conditions=_template_conditions(row),
        orientation=_template_orientation(row),
        initial=initial,
        grid=_template_grid(row, defaults),
        kriging=defaults.kriging,
        diffusivity=_diffusivity(row, defaults, custom),
        scheme=_scheme(row, defaults),
        uncertainty=_uncertainty(row, defaults),
        metadata={c: row[c] for c in METADATA_COLUMNS if _given(row, c)},
        uncertainties=_template_uncertainties(row, initial),
        scan_points=int(row["scan_points"]) if _given(row, "scan_points") else None,
        scan_grid=GridRequest(
            dx_um=_optional_float(row, "scan_dx_um"),
            dt_s=_optional_float(row, "scan_dt_s"),
            max_time_days=_optional_float(row, "scan_max_time_days"),
        ),
    )


def _optional_float(row: pd.Series, column: str) -> float | None:
    return float(row[column]) if _given(row, column) else None


def _template_grid(row: pd.Series, defaults: RunDefaults) -> Grid | GridRequest:
    """The grid a template row gives; blank dx_um or dt_s are chosen at fit time."""
    dx, dt = _optional_float(row, "dx_um"), _optional_float(row, "dt_s")
    if dx is not None and dt is not None:
        return Grid(dx_um=dx, dt_s=dt, max_time_days=_max_time(row, defaults, "max_time_days"))
    return GridRequest(dx_um=dx, dt_s=dt, max_time_days=_optional_float(row, "max_time_days"))


# (name, columns that identify it, run-name column, row reader), tried in order.
LAYOUTS = (
    ("Template layout", TEMPLATE_COLUMNS, "run", _template_run),
    ("Round robin layout", ROUND_ROBIN_COLUMNS | INITIAL_COLUMNS, "Sample", _round_robin_run),
    ("paper layout", PAPER_COLUMNS | INITIAL_COLUMNS, "File_Name", _paper_run),
)


def read_runs(
    path: str | Path,
    *,
    sheet: str | int = 0,
    defaults: RunDefaults | None = None,
    diffusivities: Mapping[str, Diffusivity] | None = None,
) -> list[Run]:
    """One `Run` per named row of a parameter table, in table order.

    A ``diffusivity`` column names a preset or an equation in ``diffusivities``
    (from `read_diffusivities`).
    """
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
    custom = dict(diffusivities or {})
    return [make_run(row, defaults, custom) for _, row in named.iterrows()]


def _steps_text(steps: tuple[float, ...]) -> str:
    """``"10, 33"``; exact, so the text reads back as the same numbers."""
    return ", ".join(repr(float(s)).removesuffix(".0") for s in steps)


def _law_columns(result: FitResult) -> dict[str, float]:
    """The ranges with the diffusivity law's uncertainty (NaN without one)."""
    if result.law_sigma_log10 is None:
        names = ("law_low_days", "law_high_days", "all_low_days", "all_high_days")
        return {
            "law_sigma_log10": math.nan,
            "between_study_log10": math.nan,
            **dict.fromkeys(names, math.nan),
        }
    law = result.time_range_days("law")
    both = result.time_range_days("law_and_studies")
    return {
        "law_sigma_log10": result.law_sigma_log10,
        "between_study_log10": result.between_study_log10 or 0.0,
        "law_low_days": law[0],
        "law_high_days": law[1],
        "all_low_days": both[0],
        "all_high_days": both[1],
    }


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
        **_law_columns(result),
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
        "diffusivity": inputs["diffusivity"].name,
        **grid_diagnostics(result),
        "white_sigma_fo": noise.white_sigma if noise else math.nan,
        "correlated_sigma_fo": noise.correlated_sigma if noise else math.nan,
        "correlation_length_um": noise.correlation_length_um if noise else math.nan,
    }


def fit_tables(result: FitResult) -> dict[str, pd.DataFrame]:
    """The curves and data behind a fit, keyed by a short name for the file.

    - "curves": the model on its grid at time zero, the best fit, and the time
      steps at the low and high ends of the 1-sigma range;
    - "kriged": the kriged profile and its uncertainty (legacy method only: the
      correlated method fits the measured points and does not krige);
    - "points": the measured points that were fitted, with the best fit
      interpolated to them and the residual (legacy: the kriged grid points).
    """
    _, low, high = result.indices
    x = result.fit_x_um if result.fit_x_um is not None else result.model_x_um
    fo = result.fit_fo if result.fit_fo is not None else result.interp_fo
    at_points = np.interp(x, result.model_x_um, result.best_fit)
    tables = {
        "curves": pd.DataFrame(
            {
                "distance_um": result.model_x_um,
                "initial_fo": result.initial,
                "best_fit_fo": result.best_fit,
                "minus_1sigma_fo": result.curves[low],
                "plus_1sigma_fo": result.curves[high],
            }
        ),
        "points": pd.DataFrame(
            {"distance_um": x, "fo": fo, "best_fit_fo": at_points, "residual_fo": fo - at_points}
        ),
    }
    if len(result.interp_x_um):
        tables["kriged"] = pd.DataFrame(
            {
                "distance_um": result.interp_x_um,
                "fo": result.interp_fo,
                "sigma_fo": result.interp_fo_sigma,
            }
        )
    return tables


def _profile(run: Run, profiles: Mapping[str, Profile]) -> Profile:
    if run.profile_name not in profiles:
        raise KeyError(
            f"run {run.name!r} needs profile {run.profile_name!r}, which is not in the profile "
            f"table (available: {sorted(profiles)})"
        )
    measured = profiles[run.profile_name]
    return Profile(measured.distance_um, measured.fo, name=run.name)


def fit_run(run: Run, profiles: Mapping[str, Profile], *, stop_early: bool = True) -> FitResult:
    """Fit one run against its measured profile (``stop_early``: see `fit_profile`).

    A grid with parts left blank (`GridRequest`) is completed by `choose_grid` for a
    final fit (1%); a grid given in full is checked (`check_grid`): unstable raises,
    and after the fit any accuracy or cost concerns are issued as warnings.
    """
    profile = _profile(run, profiles)
    args = (profile, run.conditions, run.orientation, run.initial)
    if isinstance(run.grid, GridRequest):
        grid = run.grid.resolve(*args, run.diffusivity, scheme=run.scheme).grid
        given = False
    else:
        grid = run.grid
        check_grid(grid, *args, run.diffusivity, scheme=run.scheme)  # raises if unstable
        given = True
    result = fit_profile(
        *args,
        grid,
        run.kriging,
        run.diffusivity,
        run.scheme,
        run.uncertainty,
        stop_early=stop_early,
    )
    if given:
        for message in check_grid(
            grid, *args, run.diffusivity, time_days=result.time_days, scheme=run.scheme
        ):
            warnings.warn(f"run {run.name!r}: {message}", stacklevel=2)
    return result


def _steps_column(run: Run) -> dict[str, Any]:
    """The steps' uncertainty column: all steps must share one kind."""
    steps = [run.uncertainties.get(f"step{i + 1}_um") for i in range(len(run.initial.steps_um))]
    kinds = {u.kind for u in steps if u is not None}
    if len(kinds) > 1:
        raise ValueError(f"run {run.name!r}: steps mix Gaussian and uniform uncertainties")
    if not kinds:
        return {}
    _, sd_col, range_col = UNCERTAINTY_COLUMNS[0]
    column = sd_col if kinds == {"normal"} else range_col
    return {column: _steps_text(tuple(u.spread if u else 0.0 for u in steps))}


def uncertainty_columns(run: Run) -> dict[str, Any]:
    """A run's uncertainty columns, as `read_runs` reads them (for summary.csv to refit)."""
    out = _steps_column(run)
    for what, sd_col, range_col in UNCERTAINTY_COLUMNS[1:]:
        u = run.uncertainties.get(what)
        if u is not None:
            out[sd_col if u.kind == "normal" else range_col] = u.spread
    if run.scan_points is not None:
        out["scan_points"] = run.scan_points
    return out


def scan_run(
    run: Run,
    profiles: Mapping[str, Profile],
    *,
    time_days: float | None = None,
    max_candidates: int = MAX_CANDIDATES,
    progress: Callable[[int, int], None] | None = None,
) -> ScanResult:
    """Fit a run over its uncertain initial-condition parameters (`forster.scan`).

    The scan's grid is ``run.scan_grid`` with blank parts chosen for a 5% scan
    (`choose_grid`, purpose "scan"), using ``time_days`` (the nominal fit's time, if
    known) to size it; otherwise a coarse fit estimates the time.
    """
    if not run.uncertainties:
        raise ValueError(f"run {run.name!r} has no initial-condition uncertainties to scan")
    profile = _profile(run, profiles)
    candidates = candidate_grid(
        run.initial, run.uncertainties, run.scan_points, max_candidates=max_candidates
    )
    request = run.scan_grid
    if not request.is_complete():
        max_time = run.grid.max_time_days if isinstance(run.grid, Grid) else None
        request = replace(request, max_time_days=request.max_time_days or max_time)
    grid = request.resolve(
        profile,
        run.conditions,
        run.orientation,
        run.initial,
        run.diffusivity,
        purpose="scan",
        time_days=time_days,
        scheme=run.scheme,
    ).grid
    return scan_initial_conditions(
        profile,
        run.conditions,
        run.orientation,
        candidates,
        grid,
        run.diffusivity,
        run.scheme,
        # A scan dt the user gave is kept (checked for stability over every candidate);
        # otherwise dt is set at the stability ratio for every candidate's compositions.
        stability=None if run.scan_grid.dt_s is not None else STABILITY_RATIO,
        progress=progress,
    )


def scan_size(run: Run, *, max_candidates: int = MAX_CANDIDATES) -> int:
    """Number of candidates (model runs) `scan_run` would make for ``run``."""
    if not run.uncertainties:
        return 0
    return len(
        candidate_grid(
            run.initial, run.uncertainties, run.scan_points, max_candidates=max_candidates
        ).candidates
    )


def scan_summary(scan: ScanResult) -> dict[str, Any]:
    """The scan's results as summary.csv columns (see SUMMARY_COLUMNS in forster.template)."""
    best = scan.best_values
    return {
        "scan_best_days": scan.time_days,
        "scan_low_days": scan.profile_days[0],
        "scan_high_days": scan.profile_days[1],
        "scan_marginal_low_days": scan.marginal_days[0],
        "scan_marginal_high_days": scan.marginal_days[1],
        "scan_best": "; ".join(f"{k}={v:.6g}" for k, v in best.items()),
        "scan_candidates": len(scan.runs.candidates),
        "scan_dropped": scan.prior.dropped if scan.prior is not None else 0,
        # The grid as requested, so a refit rebuilds the same one.
        "scan_dx_um": scan.runs.grid.dx_um,
        "scan_dt_s": scan.runs.grid.dt_s,
        "scan_max_time_days": scan.runs.grid.max_time_days,
    }


def scan_table(scan: ScanResult) -> pd.DataFrame:
    """One row per candidate: its parameter values, best time, chi-squared and probability."""
    rows = []
    for i, chi2 in enumerate(scan.chi2):
        k = int(np.argmin(chi2))
        values = (
            dict(zip(scan.prior.names, scan.prior.values[i], strict=True))
            if scan.prior is not None
            else {"steps_um": _steps_text(scan.runs.candidates[i].steps_um)}
        )
        penalty = float(scan.prior.penalty[i]) if scan.prior is not None else 0.0
        rows.append(
            {
                **values,
                "best_days": k * scan.runs.dt_s / 86400,
                "min_chi2": float(chi2[k]),
                "prior_penalty": penalty,
                "probability": float(scan.candidate_probability[i]),
            }
        )
    return pd.DataFrame(rows)
