"""Profile import for the app, in plain Python (no Qt): a column mapping (column ->
role and unit) guessed from the table and confirmed in the dialog, then
`profiles_from_table`, which turns the table and the mapping into profiles.

Roles: ``profile`` (the profile's name; without one the whole table is one profile),
``distance`` (from the rim; um, mm or nm) and ``fo`` (mol% or a fraction, detected from
the values: above 1.5 anywhere means mol%); every other column is ignored. Rows flagged
as the table reader flags them (``Marked_bad`` "bad", ``Ignore`` "yes") come in as
masked points: kept in the project, left out of the fits. The distances can be flipped
and shifted so the rim is at 0 (`forster.project.moved_x`).

This is the interim mapping (distance, Fo, name); a fuller importer (oxides, elements,
units, saved templates) is planned as ``forster.imports``, and the dialog is meant to
move onto it: keep the logic in `profiles_from_table`.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from forster.io import MIN_PROFILE_POINTS, _usable_rows, read_table
from forster.model import Profile
from forster.project import Project, moved_x

ROLES = ("ignore", "profile", "distance", "fo")
ROLE_TITLES = {
    "ignore": "(ignore)",
    "profile": "Profile name",
    "distance": "Distance",
    "fo": "Fo",
}
UNITS = {"distance": ("um", "mm", "nm"), "fo": ("mol%", "fraction")}
TO_UM = {"um": 1.0, "mm": 1000.0, "nm": 1e-3}
PERCENT_ABOVE = 1.5  # Fo above this anywhere: the column is in mol%
# Header patterns for the guess, tried in order (case-insensitive, on the whole name).
GUESSES = {
    "profile": (r"profile(_?name)?", r"name", r"sample", r"file_?name", r"spot_?name"),
    "distance": (r"distance.*", r"dist.*", r"x(_?um)?", r"position.*"),
    "fo": (r"fo#?", r"fo_?(mol)?%?", r"fo.*", r"forsterite.*", r"x_?fo"),
}


@dataclass(frozen=True)
class Column:
    """A column's role (`ROLES`) and unit (`UNITS`; None for ``profile``, ``ignore``)."""

    role: str = "ignore"
    unit: str | None = None


@dataclass
class Imported:
    """What `profiles_from_table` read: the profiles, the flagged points as masked points
    (per profile: distances, Fo as a fraction), and notes on what was left out."""

    profiles: dict[str, Profile] = field(default_factory=dict)
    masked: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def read_columns(path: str | Path, sheet: str | int = 0) -> pd.DataFrame:
    """The table to map (CSV, or one sheet of an .xlsx)."""
    return read_table(path, sheet)


def fo_unit(values: pd.Series) -> str:
    """mol% when any value is above `PERCENT_ABOVE`, else a fraction."""
    numbers = pd.to_numeric(values, errors="coerce")
    return "mol%" if bool((numbers > PERCENT_ABOVE).any()) else "fraction"


def guess_mapping(df: pd.DataFrame) -> dict[str, Column]:
    """A first mapping for the user to confirm: the first column whose header matches
    each role's patterns (`GUESSES`), Fo's unit from its values; the rest ignored."""
    mapping = {str(c): Column() for c in df.columns}
    taken: set[str] = set()
    for role, patterns in GUESSES.items():
        for pattern in patterns:
            match = next(
                (
                    c
                    for c in mapping
                    if c not in taken and re.fullmatch(pattern, c.strip(), re.IGNORECASE)
                ),
                None,
            )
            if match is not None:
                taken.add(match)
                unit = {"distance": "um", "fo": fo_unit(df[match])}.get(role)
                mapping[match] = Column(role, unit)
                break
    return mapping


def role_columns(mapping: Mapping[str, Column]) -> dict[str, str]:
    """role -> column for the roles that are given; each role at most once."""
    out: dict[str, str] = {}
    for column, spec in mapping.items():
        if spec.role == "ignore":
            continue
        if spec.role not in ROLES:
            raise ValueError(f"column {column!r}: unknown role {spec.role!r}")
        if spec.role in out:
            raise ValueError(
                f"{ROLE_TITLES[spec.role]} is given by both {out[spec.role]!r} and {column!r}"
            )
        out[spec.role] = column
    missing = [ROLE_TITLES[r] for r in ("distance", "fo") if r not in out]
    if missing:
        raise ValueError(f"choose a column for {' and '.join(missing)}")
    return out


def rim_offset(x: np.ndarray, flip: bool) -> float:
    """The offset that puts the rim at 0: the smallest distance, or the largest when the
    rim is at the far end (``flip``)."""
    x = np.asarray(x, dtype=float)
    return float(np.nanmax(x) if flip else np.nanmin(x))


def profiles_from_table(
    df: pd.DataFrame,
    mapping: Mapping[str, Column],
    *,
    name: str = "profile",
    flip: bool = False,
    offset_um: float = 0.0,
) -> Imported:
    """Profiles from a table and a column mapping. Without a ``profile`` column every row
    belongs to one profile called ``name``. Distances are converted to um, then flipped
    or shifted (`forster.project.moved_x`, applied after the unit), and sorted; Fo is
    stored as a fraction. Rows with no distance or Fo are left out, and a profile with
    fewer than `forster.io.MIN_PROFILE_POINTS` fitted points is skipped, both with a
    note."""
    roles = role_columns(mapping)
    out = Imported()
    distance = pd.to_numeric(df[roles["distance"]], errors="coerce")
    distance = distance * TO_UM[mapping[roles["distance"]].unit or "um"]
    fo = pd.to_numeric(df[roles["fo"]], errors="coerce")
    unit = mapping[roles["fo"]].unit or fo_unit(df[roles["fo"]])
    fo = fo / 100 if unit == "mol%" else fo
    if "profile" in roles:
        names = df[roles["profile"]]
        given = names.notna() & (names.astype(str).str.strip() != "")
        names = names.astype(str).str.strip()
    else:
        names = pd.Series(name, index=df.index)
        given = pd.Series(True, index=df.index)
    blank = given & (distance.isna() | fo.isna())
    if blank.any():
        out.notes.append(f"left out {int(blank.sum())} rows with no distance or Fo")
    rows = given & ~blank
    if (fo[rows] > 1).any():
        raise ValueError(f"Fo above 100 mol% in column {roles['fo']!r}: check its unit")
    x = pd.Series(moved_x(distance.to_numpy(dtype=float), flip, offset_um), index=df.index)
    usable = _usable_rows(df)
    too_short = []
    for profile in pd.unique(names[rows]):
        mine = rows & (names == profile)
        keep, flagged = mine & usable, mine & ~usable
        if keep.sum() < MIN_PROFILE_POINTS:
            too_short.append(str(profile))
            continue
        order = np.argsort(x[keep].to_numpy(), kind="stable")
        out.profiles[str(profile)] = Profile(
            x[keep].to_numpy()[order], fo[keep].to_numpy(dtype=float)[order], name=str(profile)
        )
        if flagged.any():
            order = np.argsort(x[flagged].to_numpy(), kind="stable")
            out.masked[str(profile)] = (
                x[flagged].to_numpy()[order],
                fo[flagged].to_numpy(dtype=float)[order],
            )
    if too_short:
        out.notes.append(
            f"skipped {len(too_short)} profile(s) with fewer than {MIN_PROFILE_POINTS} "
            f"points: {too_short}"
        )
    if not out.profiles:
        raise ValueError("no profile with enough points to fit in this table")
    return out


def add_imported(project: Project, imported: Imported) -> list[str]:
    """Add the profiles (and their masked points) to the project; their names. A name
    already in the project is an error, and then nothing is added."""
    names = project.add_profiles(imported.profiles)
    for profile, points in imported.masked.items():
        project.masked[profile] = points
    return names


__all__ = [
    "ROLES",
    "Column",
    "Imported",
    "add_imported",
    "fo_unit",
    "guess_mapping",
    "profiles_from_table",
    "read_columns",
    "rim_offset",
    "role_columns",
]
