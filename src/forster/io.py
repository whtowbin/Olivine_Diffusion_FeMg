"""Read measured profiles and model parameters from spreadsheets (CSV or Excel), and
turn fit results into tables.

Three parameter-table layouts are recognised from their columns:

- template (written by ``forster template``; the column guide is
  `forster.template.PARAMETER_COLUMNS`): the Python API's names with units, Fo as
  a fraction, ``steps_um`` and ``segment_fo`` for the initial condition (the old
  ``edge_fo``, ``core_fo``, ``step_fo`` and ``rim_fo`` are read until the release, with
  a `RenamedColumnWarning`), optional ``segment_names`` and ``centre_um``. Orientation
  is ``alpha_deg``, ``beta_deg``, ``gamma_deg`` or EBSD Euler angles ``phi1_deg``,
  ``Phi_deg``, ``phi2_deg`` with ``profile_angle_deg``; fO2 is ``fO2_Pa`` or
  ``fO2_buffer`` with ``fO2_delta_log``. Optional ``edge1_fo`` (a fixed edge value other
  than the first segment's Fo) and growth stages (`STAGE_COLUMNS`, list-valued like
  ``steps_um``, one entry per `forster.boundary.Stage`) give `fit_profile`'s ``rim_fo``
  and ``stages``;
- round robin: ``Sample`` (run name), ``File_Name`` (profile), ``alpha``,
  ``beta``, ``gamma`` (degrees to [100], [010], [001]), ``T_C``, ``P_Pa``,
  ``fO2_Pa``, ``dx_um``, ``dt_s``, optional ``max_model_time_days``;
- paper (Towbin and Plank 2026): ``File_Name`` (run and profile), EBSD Euler
  angles ``phi1``, ``Phi``, ``phi2`` with ``prof_angle``, ``T`` (degrees C),
  ``P`` (Pa), ``FO2_dQFM`` (log units relative to QFM), ``dx`` (um), ``dt`` (s).

The last two give the initial condition as ``edge_x1``, ``edge_x2`` (model domain, um),
``inflection_x`` (step positions, um; one number or text such as ``"0, 31"``),
``edge_c``, ``center_c`` and ``inflection_c`` (Fo in percent). An optional
``scheme`` column picks the finite-difference scheme per run ("implicit", the
default, "central" or "dipra"), an optional ``uncertainty`` column the uncertainty method
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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from forster import notes
from forster.boundary import BoundaryHistory, Rim, Stage
from forster.diffusivity import (
    DEFAULT_DIFFUSIVITY,
    ArrheniusDiffusivity,
    Diffusivity,
    from_table,
    label,
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
    SECONDS_PER_DAY,
    UNCERTAINTY_METHODS,
    Conditions,
    FitResult,
    Grid,
    InitialCondition,
    KrigingSettings,
    Orientation,
    Profile,
    default_scheme,
    default_segment_names,
    fit_profile,
)
from forster.rescale import ConditionUncertainty, TimeDistribution, monte_carlo
from forster.sampling import METHODS as SAMPLING_METHODS
from forster.sampling import Cancelled, ICPosterior, fit_initial_condition
from forster.scan import (
    MAX_CANDIDATES,
    ScanResult,
    Uncertainty,
    candidate_grid,
    resolve_names,
    scan_initial_conditions,
)
from forster.template import PARAMETER_COLUMNS, SUMMARY_COLUMNS

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
    ("segments", "segment_fo_sd", "segment_fo_range"),
)
# Old half-profile names, read until the release (with a RenamedColumnWarning): the Fo
# of the first, last and middle segment and their uncertainties, and the held edge value.
OLD_FO_COLUMNS = ("edge_fo", "core_fo", "step_fo")
OLD_UNCERTAINTY_COLUMNS = (
    ("edge_fo", "edge_fo_sd", "edge_fo_range"),
    ("core_fo", "core_fo_sd", "core_fo_range"),
    ("step_fo", "step_fo_sd", "step_fo_range"),
)
RENAMED_COLUMNS = {
    "edge_fo": "segment_fo (first value)",
    "core_fo": "segment_fo (last value)",
    "step_fo": "segment_fo (middle value)",
    "rim_fo": "edge1_fo",
    **{c: "segment_fo_sd" for _, c, _ in OLD_UNCERTAINTY_COLUMNS},
    **{c: "segment_fo_range" for _, _, c in OLD_UNCERTAINTY_COLUMNS},
}


class RenamedColumnWarning(UserWarning):
    """A parameter table uses an old column name; it is read, until the release."""


# Growth stages (template layout): list-valued like steps_um, one entry per stage, as
# `forster.boundary.Stage` (start in days, new rim position, new zone's Fo, rim Fo).
# The first two are needed for every stage; an empty entry in the others is None.
STAGE_COLUMNS = ("stage_days", "stage_rim_um", "stage_zone_fo", "stage_rim_fo")
METADATA_COLUMNS = (
    "crystal",  # profiles of one crystal (for joint fits; carried, not yet used)
    "sample",  # crystals of one sample (for population priors; carried, not yet used)
    "Descriptive_Name",
    "Category",
    "Uncert_Category",
    "Notes",
    "notes",
)
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
    # Growth stages after ``initial`` and the fixed Fo held at edge 1 (None: the first
    # segment's Fo; the table's edge1_fo); `fit_profile`'s ``stages`` and ``rim_fo``.
    stages: tuple[Stage, ...] = ()
    rim_fo: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    # Uncertain initial-condition parameters (template layout): scanned by `scan_run`.
    uncertainties: Mapping[str, Uncertainty] = field(default_factory=dict)
    scan_points: int | None = None
    # The scan's grid: parts left as None are chosen by `forster.grid.choose_grid`.
    scan_grid: GridRequest = field(default_factory=GridRequest)
    # How the uncertain parameters are fitted: "auto", "laplace" or "nested"
    # (`forster.sampling`), or "grid" (`forster.scan`); and the sampler's seed.
    scan_method: str = "auto"
    scan_seed: int = 0
    # Monte Carlo over T, P, fO2 and the law (`forster.rescale`); None: not run.
    mc: ConditionUncertainty | None = None
    mc_samples: int | None = None
    mc_seed: int = 0
    # How fO2 and the orientation were given, kept so `run_row` (and a GUI) writes them
    # back the same way: (buffer, log10 offset), and (phi1, Phi, phi2, profile angle) in
    # degrees. Display only: ``conditions`` and ``orientation`` hold the values used, so
    # these are left out of equality.
    fO2_form: tuple[str, float] | None = field(default=None, compare=False)
    euler_form: tuple[float, float, float, float] | None = field(default=None, compare=False)

    def __post_init__(self) -> None:
        # Uncertainties by `forster.scan.parameter_names` (a_fo, ...), the old edge_fo,
        # core_fo and step_fo read as their segments, so equal runs compare equal.
        if self.uncertainties:
            object.__setattr__(
                self, "uncertainties", resolve_names(self.initial, self.uncertainties)
            )


def read_table(path: str | Path, sheet: str | int = 0) -> pd.DataFrame:
    """A CSV file, or one sheet of an Excel workbook (the first by default)."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        # pandas' default float parser can be 1 ulp off; round_trip reads back exactly
        # what was written, so a table written by FORSTER refits identically.
        # pandas documents "round_trip" but its type stubs leave it out.
        return pd.read_csv(path, float_precision="round_trip")  # ty: ignore[no-matching-overload]
    if path.suffix.lower() in {".xlsx", ".xlsm"}:
        return pd.read_excel(path, sheet_name=sheet, engine="openpyxl")
    # openpyxl cannot read the old binary .xls format: save it as .xlsx or .csv.
    raise ValueError(f"{path}: expected a .csv or .xlsx file")


def _usable_rows(df: pd.DataFrame) -> pd.Series:
    """Rows not flagged as bad (``Marked_bad`` "bad") or ignored (``Ignore`` "yes"), in any
    case and with spaces around them."""
    keep = pd.Series(True, index=df.index)
    for column, flag in (("Marked_bad", "bad"), ("Ignore", "yes")):
        if column in df.columns:
            keep &= df[column].astype(str).str.strip().str.lower() != flag
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
    names = df.loc[keep, name_column].dropna().unique()
    # A row with no distance or Fo cannot be fitted; a name left with too few points
    # is skipped below like any other.
    blank = keep & df[name_column].notna() & df[[distance_column, fo_column]].isna().any(axis=1)
    keep &= ~blank
    partly_blank = sorted(
        str(n) for n in df.loc[blank, name_column].unique() if (keep & (df[name_column] == n)).any()
    )
    if partly_blank:
        notes.warn(f"{path}: dropped rows with no distance or Fo from {partly_blank}", stacklevel=2)
    profiles = {}
    too_short = []
    for name in names:
        rows = df.loc[(df[name_column] == name) & keep].sort_values(distance_column)
        if len(rows) < MIN_PROFILE_POINTS:
            too_short.append(str(name))
            continue
        fo = rows[fo_column].to_numpy()
        if fo_in_percent:
            fo = fo / 100
        profiles[str(name)] = Profile(rows[distance_column].to_numpy(), fo, name=str(name))
    if too_short:
        notes.warn(
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
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            raise ValueError(f"inflection_c {value!r}: expected a number (Fo in percent)") from None
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


SCAN_METHODS = (*SAMPLING_METHODS, "grid")
# Any of these given turns the Monte Carlo on (`_template_mc`).
MC_SPREADS = ("mc_T_sd_C", "mc_P_sd_Pa", "mc_fO2_sd_log", "mc_law", "mc_between_study")
_TRUE, _FALSE = ("yes", "true", "1", "y"), ("no", "false", "0", "n")


def _flag(row: pd.Series, column: str, default: bool) -> bool:
    if not _given(row, column):
        return default
    value = str(row[column]).strip().lower()
    if value.endswith(".0"):
        value = value[:-2]
    if value not in _TRUE + _FALSE:
        raise ValueError(f"{column} {row[column]!r} in the parameter table; use yes or no")
    return value in _TRUE


def _template_mc(row: pd.Series) -> ConditionUncertainty | None:
    """The Monte Carlo's uncertainties, or None if no mc_ spread column is given.
    fO2 follows ``mc_fO2_buffer``, or the row's own ``fO2_buffer`` when fO2 is given
    relative to one."""
    if not any(_given(row, c) for c in MC_SPREADS):
        return None
    buffer = row["mc_fO2_buffer"] if _given(row, "mc_fO2_buffer") else None
    if buffer is None and _given(row, "fO2_buffer") and not _given(row, "fO2_Pa"):
        buffer = row["fO2_buffer"]
    return ConditionUncertainty(
        T_sd_C=_optional_float(row, "mc_T_sd_C") or 0.0,
        P_sd_Pa=_optional_float(row, "mc_P_sd_Pa") or 0.0,
        fO2_sd_log=_optional_float(row, "mc_fO2_sd_log") or 0.0,
        fO2_buffer=None if buffer is None else str(buffer).strip().upper(),
        law=_flag(row, "mc_law", True),
        between_study=_flag(row, "mc_between_study", False),
    )


def _scan_method(row: pd.Series) -> str:
    """Optional ``scan_method``; blank means "grid" when ``scan_points`` is given (as in
    tables written before the samplers), "auto" otherwise."""
    value = row.get("scan_method")
    if value is None or pd.isna(value):
        return "grid" if _given(row, "scan_points") else "auto"
    value = str(value).strip().lower()
    if value not in SCAN_METHODS:
        raise ValueError(
            f"scan_method {value!r} in the parameter table; use one of {', '.join(SCAN_METHODS)}"
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


def _old_value_uncertainty(
    row: pd.Series, what: str, column: str, kind: str, initial: InitialCondition
) -> dict[str, Uncertainty]:
    """An old-named concentration uncertainty (edge, core or plateau Fo) as its segment's
    parameter; 0 leaves it fixed."""
    if what == "step_fo" and initial.step_fo is None:
        raise ValueError(f"run {row['run']!r}: {column} needs a plateau (step_fo)")
    names = initial.segment_names
    segment = {"edge_fo": names[0], "core_fo": names[-1], "step_fo": names[1 % len(names)]}
    spread = float(row[column])
    return {f"{segment[what]}_fo": Uncertainty(spread, kind)} if spread else {}


def _segment_uncertainties(row: pd.Series, initial: InitialCondition) -> dict[str, Uncertainty]:
    """``segment_fo_sd`` / ``segment_fo_range``: one value for every segment, or one per
    segment; an empty entry or 0 leaves a segment fixed, and a segment takes one column."""
    names = initial.segment_names
    out: dict[str, Uncertainty] = {}
    for column, kind in (("segment_fo_sd", "normal"), ("segment_fo_range", "uniform")):
        values = _number_list(row, column)
        if values is None:
            continue
        if len(values) == 1:
            values = values * len(names)
        if len(values) != len(names):
            raise ValueError(
                f"run {row['run']!r}: {column} has {len(values)} values for {len(names)} segments"
            )
        for name, spread in zip(names, values, strict=True):
            if not spread:
                continue
            if f"{name}_fo" in out:
                raise ValueError(
                    f"run {row['run']!r}: segment {name} has both segment_fo_sd and "
                    "segment_fo_range"
                )
            out[f"{name}_fo"] = Uncertainty(spread, kind)
    return out


def _template_uncertainties(row: pd.Series, initial: InitialCondition) -> dict[str, Uncertainty]:
    """The ``_sd`` / ``_range`` columns of a template row, by `forster.scan` parameter name."""
    out: dict[str, Uncertainty] = {}
    source = _uncertainty_source(row, "steps_sd_um", "steps_range_um")
    if source is not None:
        out.update(_step_uncertainties(row, *source, initial))
    out.update(_segment_uncertainties(row, initial))
    for what, sd_col, range_col in OLD_UNCERTAINTY_COLUMNS:
        source = _uncertainty_source(row, sd_col, range_col)
        if source is None:
            continue
        for name, u in _old_value_uncertainty(row, what, *source, initial).items():
            if name in out:
                raise ValueError(
                    f"run {row['run']!r}: {source[0]} repeats the uncertainty of {name} "
                    "(give segment_fo_sd or segment_fo_range only)"
                )
            out[name] = u
    return out


def _number_list(row: pd.Series, column: str) -> tuple[float | None, ...] | None:
    """A list-valued cell, numbers separated by commas as in ``steps_um`` (``"2, 5.5"``);
    an empty entry (``"0.91, "``) is None. None when the cell is empty."""
    if not _given(row, column):
        return None
    value = row[column]
    entries = value.strip().strip("()[]").split(",") if isinstance(value, str) else [value]
    try:
        values = tuple(None if str(e).strip() == "" else float(e) for e in entries)
    except (TypeError, ValueError):
        values = (math.nan,)
    if not all(v is None or math.isfinite(v) for v in values):
        raise ValueError(
            f"run {row['run']!r}: {column} {value!r}: give numbers separated by commas, "
            'such as "2, 5.5"'
        )
    return values


def _every_stage(row: pd.Series, column: str, values: tuple[float | None, ...]) -> list[float]:
    present = [v for v in values if v is not None]
    if len(present) != len(values):
        raise ValueError(f"run {row['run']!r}: {column} needs a value for every stage")
    return present


def _template_stages(row: pd.Series) -> tuple[Stage, ...]:
    """The growth stages of a template row (`STAGE_COLUMNS`); () when none are given."""
    lists = {c: _number_list(row, c) for c in STAGE_COLUMNS}
    given = {c: v for c, v in lists.items() if v is not None}
    if not given:
        return ()
    name = row["run"]
    missing = [c for c in STAGE_COLUMNS[:2] if c not in given]
    if missing:
        raise ValueError(
            f"run {name!r}: growth stages need {' and '.join(missing)} (one entry per stage)"
        )
    n = len(given["stage_days"])
    for column, values in given.items():
        if len(values) != n:
            raise ValueError(
                f"run {name!r}: {column} has {len(values)} entries for {n} stages (stage_days)"
            )
    days = _every_stage(row, "stage_days", given["stage_days"])
    rims = _every_stage(row, "stage_rim_um", given["stage_rim_um"])
    if any(b <= a for a, b in zip([0.0, *days], days, strict=False)):
        raise ValueError(
            f"run {name!r}: stage_days must increase and be after 0 (the start); got "
            f"{_steps_text(tuple(days))}"
        )
    zones = given.get("stage_zone_fo", (None,) * n)
    held = given.get("stage_rim_fo", (None,) * n)
    return tuple(
        Stage(d * SECONDS_PER_DAY, r, rim_value=h, zone_value=z)
        for d, r, z, h in zip(days, rims, zones, held, strict=True)
    )


def _template_rim_fo(row: pd.Series) -> float | None:
    """The Fo held at edge 1: ``edge1_fo`` (or the old ``rim_fo``); None when not given."""
    given = [c for c in ("edge1_fo", "rim_fo") if _given(row, c)]
    if len(given) == 2:
        raise ValueError(f"run {row['run']!r}: give edge1_fo or the old rim_fo, not both")
    if not given:
        return None
    column = given[0]
    value = float(row[column])
    if not (math.isfinite(value) and 0.0 < value <= 1.0):
        raise ValueError(f"run {row['run']!r}: {column} {value} must be a fraction (0-1)")
    return value


def _text_list(row: pd.Series, column: str) -> tuple[str, ...] | None:
    """A list of labels separated by commas (``"rim, mantle, core"``); None when empty."""
    if not _given(row, column):
        return None
    return tuple(e.strip() for e in str(row[column]).strip().strip("()[]").split(","))


def _template_initial(row: pd.Series) -> InitialCondition:
    """The starting profile of a template row: ``segment_fo`` (or the old ``edge_fo``,
    ``core_fo``, ``step_fo``), ``steps_um``, ``segment_names`` and ``centre_um``."""
    name = row["run"]
    old = [c for c in OLD_FO_COLUMNS if _given(row, c)]
    segments = _number_list(row, "segment_fo")
    if segments is not None and old:
        raise ValueError(f"run {name!r}: give segment_fo or the old {', '.join(old)}, not both")
    kwargs: dict[str, Any] = {}
    if segments is not None:
        if any(v is None for v in segments):
            raise ValueError(f"run {name!r}: segment_fo needs a value for every segment")
        kwargs["segment_fo"] = segments
    elif {"edge_fo", "core_fo"} <= set(old):
        kwargs |= {"edge_fo": row["edge_fo"], "core_fo": row["core_fo"]}
        if _given(row, "step_fo"):
            kwargs["step_fo"] = float(row["step_fo"])
    else:
        raise ValueError(f"run {name!r}: give segment_fo (one Fo per segment, from edge 1)")
    try:
        return InitialCondition(
            domain_um=(row["domain_start_um"], row["domain_end_um"]),
            steps_um=_steps(row["steps_um"]),
            segment_names=_text_list(row, "segment_names"),
            centre_um=_optional_float(row, "centre_um"),
            **kwargs,
        )
    except ValueError as error:
        raise ValueError(f"run {name!r}: {error}") from None


def _check_history(
    row: pd.Series, initial: InitialCondition, stages: tuple[Stage, ...], rim_fo: float | None
) -> None:
    """The stages' order, rims and values (`BoundaryHistory.check`), naming the run."""
    if not stages:
        return
    try:
        BoundaryHistory.from_initial(initial, stages, rim_fo)
    except ValueError as error:
        raise ValueError(f"run {row['run']!r}: {error}") from None


def _template_run(row: pd.Series, defaults: RunDefaults, custom: Mapping[str, Diffusivity]) -> Run:
    initial = _template_initial(row)
    stages, rim_fo = _template_stages(row), _template_rim_fo(row)
    _check_history(row, initial, stages, rim_fo)
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
        stages=stages,
        rim_fo=rim_fo,
        metadata={c: row[c] for c in METADATA_COLUMNS if _given(row, c)},
        uncertainties=_template_uncertainties(row, initial),
        scan_points=int(row["scan_points"]) if _given(row, "scan_points") else None,
        scan_grid=GridRequest(
            dx_um=_optional_float(row, "scan_dx_um"),
            dt_s=_optional_float(row, "scan_dt_s"),
            max_time_days=_optional_float(row, "scan_max_time_days"),
        ),
        scan_method=_scan_method(row),
        scan_seed=int(row["scan_seed"]) if _given(row, "scan_seed") else 0,
        mc=_template_mc(row),
        mc_samples=int(row["mc_samples"]) if _given(row, "mc_samples") else None,
        mc_seed=int(row["mc_seed"]) if _given(row, "mc_seed") else 0,
        fO2_form=_fo2_form(row),
        euler_form=_euler_form(row),
    )


def _fo2_form(row: pd.Series) -> tuple[str, float] | None:
    if _pick_form(row, FO2_FORMS, "fO2") == 0:
        return None
    return str(row["fO2_buffer"]).strip().upper(), float(row["fO2_delta_log"])


def _euler_form(row: pd.Series) -> tuple[float, float, float, float] | None:
    if _pick_form(row, ORIENTATION_FORMS, "the orientation") == 0:
        return None
    columns = ORIENTATION_FORMS[1]
    phi1, big_phi, phi2, angle = (float(row[c]) for c in columns)
    return phi1, big_phi, phi2, angle


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


def _warn_unknown_columns(path: str | Path, columns: set[str]) -> None:
    """Warn of columns a template (or summary) table does not use: usually a misspelt
    optional column, which would otherwise be ignored without a word."""
    known = (
        {c.name for c in (*PARAMETER_COLUMNS, *SUMMARY_COLUMNS)}
        | set(METADATA_COLUMNS)
        | set(RENAMED_COLUMNS)
    )
    unknown = sorted(str(c) for c in columns - known if not str(c).startswith("Unnamed:"))
    if unknown:
        notes.warn(
            f"{path}: column(s) not used by FORSTER (check the spelling): {unknown}",
            stacklevel=3,
        )


def _warn_renamed_columns(path: str | Path, columns: set[str]) -> None:
    """One `RenamedColumnWarning` naming every old column a table uses."""
    old = [c for c in RENAMED_COLUMNS if c in columns]
    if old:
        renames = "; ".join(f"{c} -> {RENAMED_COLUMNS[c]}" for c in old)
        notes.warn(
            f"{path}: old column names, read until the release: {renames}",
            RenamedColumnWarning,
            stacklevel=3,
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
            f"{path}: not a recognized parameter table (see `forster template`). {needs}"
        )
    label, _, name_column, make_run = layout
    if label == "Template layout":
        _warn_unknown_columns(path, columns)
        _warn_renamed_columns(path, columns)
    named = df[df[name_column].notna()]
    duplicates = named[name_column][named[name_column].duplicated()].unique()
    if len(duplicates):
        raise ValueError(f"{path}: run names must be unique; repeated: {list(duplicates)}")
    custom = dict(diffusivities or {})
    return [make_run(row, defaults, custom) for _, row in named.iterrows()]


def _steps_text(steps: tuple[float, ...]) -> str:
    """``"10, 33"``; exact, so the text reads back as the same numbers."""
    return ", ".join(repr(float(s)).removesuffix(".0") for s in steps)


def _list_text(values: Sequence[float | None]) -> str | None:
    """A list-valued cell as `_number_list` reads it: ``"0.91, "`` for (0.91, None);
    None (an empty cell) when every value is None."""
    if all(v is None for v in values):
        return None
    return ", ".join("" if v is None else _steps_text((v,)) for v in values)


def _fixed(value: Rim | None) -> float | None:
    if value is None or isinstance(value, int | float):
        return value
    raise ValueError(
        "a rim value that changes in time (RimPath or a function) has no parameter-table "
        "form, so summary_row cannot write it"
    )


def _initial_columns(initial: InitialCondition) -> dict[str, Any]:
    """``steps_um``, ``segment_fo`` and, when not the defaults, ``segment_names`` and
    ``centre_um``, as `read_runs` reads them."""
    out: dict[str, Any] = {
        "steps_um": _steps_text(initial.steps_um),
        "segment_fo": _list_text(list(initial.segment_fo)),
    }
    if initial.segment_names != default_segment_names(len(initial.segment_fo)):
        out["segment_names"] = ", ".join(initial.segment_names)
    if initial.centre_um is not None:
        out["centre_um"] = initial.centre_um
    return out


def _boundary_columns(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """``edge1_fo`` and the growth-stage columns (`STAGE_COLUMNS`), as `read_runs` reads them."""
    stages: tuple[Stage, ...] = inputs.get("stages", ())
    return {
        "edge1_fo": _fixed(inputs.get("rim_fo")),
        "stage_days": _list_text([s.start_s / SECONDS_PER_DAY for s in stages]),
        "stage_rim_um": _list_text([s.rim_um for s in stages]),
        "stage_zone_fo": _list_text([s.zone_value for s in stages]),
        "stage_rim_fo": _list_text([_fixed(s.rim_value) for s in stages]),
    }


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
        "time_since_last_stage_days": result.time_since_last_stage_days,
        "low_days": result.time_days + result.time_minus_days,
        "high_days": result.time_days + result.time_plus_days,
        "low2_days": result.time_days + result.time_minus2_s / SECONDS_PER_DAY,
        "high2_days": result.time_days + result.time_plus2_s / SECONDS_PER_DAY,
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
        **_initial_columns(initial),
        **_boundary_columns(inputs),
        "dx_um": grid.dx_um,
        "dt_s": grid.dt_s,
        "max_time_days": grid.max_time_days,
        "diffusivity": inputs["diffusivity"].name,
        "diffusivity_label": label(inputs["diffusivity"]),
        **grid_diagnostics(result),
        "white_sigma_fo": noise.white_sigma if noise else math.nan,
        "correlated_sigma_fo": noise.correlated_sigma if noise else math.nan,
        "correlation_length_um": noise.correlation_length_um if noise else math.nan,
        "near_step_daic": (
            result.near_step_delta_aic if result.near_step_delta_aic is not None else math.nan
        ),
        "core_end_diffusion_lengths": (
            result.core_end_lengths if result.core_end_lengths is not None else math.nan
        ),
        "near_step_misfit_sigma_fo": _or_nan(result.near_step_misfit_sd),
        "near_step_points": result.near_step_points if result.near_step_points is not None else -1,
        "points_outside_domain": _points_outside(result.data_x_um, initial.domain_um),
    }


def _or_nan(value: float | None) -> float:
    return math.nan if value is None else value


def _points_outside(x_um: np.ndarray, domain_um: tuple[float, float]) -> int:
    """Measured points before the domain's start or past its end: not fitted. A note, not
    a warning (leaving points out can be deliberate)."""
    return int(np.sum((x_um < domain_um[0]) | (x_um > domain_um[1])))


def fit_tables(result: FitResult) -> dict[str, pd.DataFrame]:
    """The curves and data behind a fit, keyed by a short name for the file.

    - "curves": the model on its grid at time zero, the best fit, and the time
      steps at the low and high ends of the 1- and 2-sigma ranges (2-sigma columns
      only where the fit has that range: not legacy);
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
                "minus_1sigma_fo": result.curve(low),
                "plus_1sigma_fo": result.curve(high),
                **(
                    {
                        "minus_2sigma_fo": result.curve(result.indices2[0]),
                        "plus_2sigma_fo": result.curve(result.indices2[1]),
                    }
                    if result.indices2 is not None
                    else {}
                ),
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


def fit_run(
    run: Run,
    profiles: Mapping[str, Profile],
    *,
    stop_early: bool = True,
    keep_curves: bool = False,
    check_near_step: bool = True,
) -> FitResult:
    """Fit one run against its measured profile (``stop_early``, ``keep_curves``: see
    `fit_profile`).

    A grid with parts left blank (`GridRequest`) is completed by `choose_grid` for a
    final fit (1%); a grid given in full is checked (`check_grid`): unstable raises,
    and after the fit any accuracy or cost concerns are issued as warnings.

    ``run.stages`` and ``run.rim_fo`` go to `fit_profile`. The grid choice and checks
    model the crystal before any stage, with ``edge_fo`` at the rim, so a run with
    stages needs its grid given in full (dx_um and dt_s); its accuracy checks use the
    time since the last stage, all the youngest zone boundaries have had to diffuse.
    """
    profile = _profile(run, profiles)
    args = (profile, run.conditions, run.orientation, run.initial)
    if run.stages and isinstance(run.grid, GridRequest):
        raise ValueError(
            f"run {run.name!r}: growth stages need dx_um and dt_s (the grid choice does not "
            "model the stages)"
        )
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
        stages=run.stages,
        rim_fo=run.rim_fo,
        keep_curves=keep_curves,
        check_near_step=check_near_step,
    )
    if given:
        since = max(result.time_since_last_stage_days, grid.dt_s / SECONDS_PER_DAY)
        for message in check_grid(grid, *args, run.diffusivity, time_days=since, scheme=run.scheme):
            notes.warn(f"run {run.name!r}: {message}", stacklevel=2)
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


def _segment_columns(run: Run) -> dict[str, Any]:
    """``segment_fo_sd`` and / or ``segment_fo_range``, one entry per segment."""
    names = run.initial.segment_names
    given = resolve_names(run.initial, run.uncertainties)
    out: dict[str, Any] = {}
    for column, kind in (("segment_fo_sd", "normal"), ("segment_fo_range", "uniform")):
        spreads = []
        for name in names:
            u = given.get(f"{name}_fo")
            spreads.append(u.spread if u is not None and u.kind == kind else None)
        text = _list_text(spreads)
        if text is not None:
            out[column] = text
    return out


def uncertainty_columns(run: Run) -> dict[str, Any]:
    """A run's uncertainty columns, as `read_runs` reads them (for summary.csv to refit)."""
    out = _steps_column(run) | _segment_columns(run)
    if run.scan_points is not None:
        out["scan_points"] = run.scan_points
    if run.uncertainties:
        out["scan_method"] = run.scan_method
        out["scan_seed"] = run.scan_seed
    if run.mc is not None:
        out.update(
            mc_T_sd_C=run.mc.T_sd_C,
            mc_P_sd_Pa=run.mc.P_sd_Pa,
            mc_fO2_sd_log=run.mc.fO2_sd_log,
            mc_fO2_buffer=run.mc.fO2_buffer,
            mc_law="yes" if run.mc.law else "no",
            mc_between_study="yes" if run.mc.between_study else "no",
            mc_seed=run.mc_seed,
        )
        if run.mc_samples is not None:
            out["mc_samples"] = run.mc_samples
    return out


def run_row(run: Run) -> dict[str, Any]:
    """A run as one row of a template parameter table, without fitting it: `read_runs`
    reads the row back as the same run (fO2 and the orientation in the form they were
    given, `Run.fO2_form` and `Run.euler_form`; blank grid parts left blank)."""
    c, o, initial = run.conditions, run.orientation, run.initial
    fo2: dict[str, Any] = {"fO2_Pa": c.fO2_Pa}
    if run.fO2_form is not None:
        fo2 = {"fO2_buffer": run.fO2_form[0], "fO2_delta_log": run.fO2_form[1]}
    angles: dict[str, Any] = {
        "alpha_deg": o.alpha_deg,
        "beta_deg": o.beta_deg,
        "gamma_deg": o.gamma_deg,
    }
    if run.euler_form is not None:
        angles = dict(zip(ORIENTATION_FORMS[1], run.euler_form, strict=True))
    grid = run.grid
    return {
        "run": run.name,
        "profile": run.profile_name,
        "T_C": c.T_C,
        "P_Pa": c.P_Pa,
        **fo2,
        **angles,
        "domain_start_um": initial.domain_um[0],
        "domain_end_um": initial.domain_um[1],
        **_initial_columns(initial),
        **_boundary_columns({"stages": run.stages, "rim_fo": run.rim_fo}),
        "dx_um": grid.dx_um,
        "dt_s": grid.dt_s,
        "max_time_days": grid.max_time_days,
        "scheme": run.scheme,
        "diffusivity": run.diffusivity.name,
        "uncertainty": run.uncertainty,
        **uncertainty_columns(run),
        "scan_dx_um": run.scan_grid.dx_um,
        "scan_dt_s": run.scan_grid.dt_s,
        "scan_max_time_days": run.scan_grid.max_time_days,
        **dict(run.metadata),
    }


def mc_run(
    run: Run, result: FitResult, posterior: ScanResult | ICPosterior | None = None
) -> TimeDistribution:
    """The run's Monte Carlo (`forster.rescale.monte_carlo`) on its fit, drawing the fit's
    time from an initial-condition posterior when one is given (a sampled scan; a grid
    scan's is not used)."""
    if run.mc is None:
        raise ValueError(f"run {run.name!r} has no mc_ columns")
    return monte_carlo(
        result,
        run.mc,
        posterior=posterior if isinstance(posterior, ICPosterior) else None,
        n=run.mc_samples,
        seed=run.mc_seed,
    )


def mc_summary(dist: TimeDistribution) -> dict[str, Any]:
    """The Monte Carlo as summary.csv columns."""
    p = dist.percentiles_days
    return {
        "mc_method": dist.method,
        "mc_draws": len(dist.days),
        "mc_median_days": p[50],
        "mc_low_days": p[16],
        "mc_high_days": p[84],
        "mc_p05_days": p[5],
        "mc_p95_days": p[95],
        "mc_sd_log10": "; ".join(f"{k}={v:.3g}" for k, v in dist.sd_log10_by_source.items()),
    }


def scan_run(
    run: Run,
    profiles: Mapping[str, Profile],
    *,
    time_days: float | None = None,
    max_candidates: int = MAX_CANDIDATES,
    progress: Callable[[int, int], None] | None = None,
    stage: Callable[[str], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> ScanResult | ICPosterior:
    """Fit a run over its uncertain initial-condition parameters.

    ``run.scan_method`` "grid" fits every node of a grid (`forster.scan`; ``progress(i,
    n)`` after each run); "auto", "laplace" or "nested" find the best fit and sample the
    posterior (`forster.sampling.fit_initial_condition`, seeded by ``run.scan_seed``).
    Either runs on ``run.scan_grid`` with blank parts chosen for a 5% scan
    (`choose_grid`, purpose "scan"), using ``time_days`` (the nominal fit's time, if
    known) to size it; otherwise a coarse fit estimates the time. The scan does not
    model growth stages or a ``rim_fo``: a run with either raises.

    ``stage(name)`` reports the sampler's stages; ``cancel()``, checked before every
    model run, stops the scan with `forster.sampling.Cancelled` when it returns True.
    """
    if not run.uncertainties:
        raise ValueError(f"run {run.name!r} has no initial-condition uncertainties to scan")
    _check_scannable(run)
    profile = _profile(run, profiles)
    candidates = None
    if run.scan_method == "grid":  # built first: a grid too large fails before any run
        candidates = candidate_grid(
            run.initial, run.uncertainties, run.scan_points, max_candidates=max_candidates
        )
    request = scan_request(run, time_days)
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
    if candidates is None:
        return fit_initial_condition(
            profile,
            run.conditions,
            run.orientation,
            run.initial,
            run.uncertainties,
            grid,
            run.diffusivity,
            run.scheme,
            method=run.scan_method,
            seed=run.scan_seed,
            progress=stage,
            cancel=cancel,
        )
    if cancel is not None:
        progress = _cancellable(progress, cancel)
    return scan_initial_conditions(
        profile,
        run.conditions,
        run.orientation,
        candidates,
        grid,
        run.diffusivity,
        run.scheme,
        # A scan dt the user gave is kept (an explicit run substeps within it if it is
        # above the stable step); otherwise dt is raised to the stability ratio for
        # every candidate's compositions.
        stability=None if run.scan_grid.dt_s is not None else STABILITY_RATIO,
        progress=progress,
    )


def _cancellable(
    progress: Callable[[int, int], None] | None, cancel: Callable[[], bool]
) -> Callable[[int, int], None]:
    """A grid scan's ``progress`` that also stops the scan when ``cancel()`` is True."""

    def report(i: int, n: int) -> None:
        if cancel():
            raise Cancelled("the initial-condition scan was canceled")
        if progress is not None:
            progress(i, n)

    return report


# The starting-profile fit's runs model at most this many times the nominal fit's time
# when scan_max_time_days is blank: the run's own max_time_days (200 days in the paper
# table) gave Xenocryst_ol6 (0.23 days) a scan of 134,000 time steps. The fit warns when
# its posterior of time reaches the end of the run.
SCAN_TIME_FACTOR = 10.0


def scan_request(run: Run, time_days: float | None = None) -> GridRequest:
    """The starting-profile fit's grid request: ``scan_max_time_days`` if given, else the
    run's ``max_time_days`` capped at `SCAN_TIME_FACTOR` x ``time_days`` (the nominal
    fit's time, when known)."""
    request = run.scan_grid
    if request.is_complete() or request.max_time_days is not None:
        return request
    max_time = run.grid.max_time_days if isinstance(run.grid, Grid) else None
    if max_time is not None and time_days is not None and time_days > 0:
        max_time = min(max_time, SCAN_TIME_FACTOR * time_days)
    return replace(request, max_time_days=max_time)


def _check_scannable(run: Run) -> None:
    """`forster.scan` runs each candidate from its initial condition alone."""
    given = [
        what for what, used in (("growth stages", run.stages), ("edge1_fo", run.rim_fo)) if used
    ]
    if given:
        raise ValueError(
            f"run {run.name!r}: the initial-condition scan does not model "
            f"{' or '.join(given)}; remove the _sd and _range columns to fit without a scan"
        )


def scan_size(run: Run, *, max_candidates: int = MAX_CANDIDATES) -> int:
    """Number of candidates (model runs) a grid scan of ``run`` would make; 0 for a run
    without uncertainties or fitted by sampling (`forster.sampling`). Raises, like
    `scan_run`, for a run the scan cannot model."""
    if not run.uncertainties:
        return 0
    _check_scannable(run)
    if run.scan_method != "grid":
        return 0
    return len(
        candidate_grid(
            run.initial, run.uncertainties, run.scan_points, max_candidates=max_candidates
        ).candidates
    )


def _posterior_summary(post: ICPosterior) -> dict[str, Any]:
    return {
        "scan_sampler": post.method,
        "scan_best_days": post.best_days,
        "scan_median_days": post.median_days,
        "scan_low_days": post.low_days,
        "scan_high_days": post.high_days,
        "scan_marginal_low_days": post.low_days,
        "scan_marginal_high_days": post.high_days,
        "scan_best": "; ".join(f"{k}={v:.6g}" for k, v in post.best_values.items()),
        "scan_candidates": post.runs,
        "scan_dropped": 0,
        "scan_ic_share": post.ic_share,
        "scan_sensitivity": "; ".join(
            f"{k}={v:.4g}" for k, v in post.sensitivity.items() if math.isfinite(v)
        ),
        "scan_ess": post.ess,
        "scan_log_evidence": post.log_evidence,
        "scan_near_step_daic": post.near_step_delta_aic,
        "scan_prior_offsets": _offsets_text(post.prior_offsets),
        "scan_dx_um": post.grid.dx_um,
        "scan_dt_s": post.grid.dt_s,
        "scan_max_time_days": post.grid.max_time_days,
    }


def _offsets_text(offsets: Mapping[str, float]) -> str:
    return "; ".join(f"{k}={v:+.2f}" for k, v in offsets.items())


def scan_summary(scan: ScanResult | ICPosterior) -> dict[str, Any]:
    """The scan's results as summary.csv columns (see SUMMARY_COLUMNS in forster.template)."""
    if isinstance(scan, ICPosterior):
        return _posterior_summary(scan)
    best = scan.best_values
    return {
        "scan_sampler": "grid",
        "scan_best_days": scan.time_days,
        "scan_low_days": scan.profile_days[0],
        "scan_high_days": scan.profile_days[1],
        "scan_marginal_low_days": scan.marginal_days[0],
        "scan_marginal_high_days": scan.marginal_days[1],
        "scan_best": "; ".join(f"{k}={v:.6g}" for k, v in best.items()),
        "scan_candidates": len(scan.runs.candidates),
        "scan_dropped": scan.prior.dropped if scan.prior is not None else 0,
        "scan_near_step_daic": scan.near_step_delta_aic,
        "scan_prior_offsets": _offsets_text(scan.prior_offsets),
        # The grid as requested, so a refit rebuilds the same one.
        "scan_dx_um": scan.runs.grid.dx_um,
        "scan_dt_s": scan.runs.grid.dt_s,
        "scan_max_time_days": scan.runs.grid.max_time_days,
    }


def scan_table(scan: ScanResult | ICPosterior) -> pd.DataFrame:
    """Grid: one row per candidate (its parameter values, best time, chi-squared and
    probability). Sampled: the marginal posterior of time (days, density per day and
    cumulative probability)."""
    if isinstance(scan, ICPosterior):
        step = scan.days[1] - scan.days[0] if len(scan.days) > 1 else 1.0
        return pd.DataFrame(
            {
                "days": scan.days,
                "density_per_day": scan.density,
                "cumulative": np.cumsum(scan.density) * step,
            }
        )
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
                "best_days": k * scan.runs.dt_s / SECONDS_PER_DAY,
                "min_chi2": float(chi2[k]),
                "prior_penalty": penalty,
                "probability": float(scan.candidate_probability[i]),
            }
        )
    return pd.DataFrame(rows)
