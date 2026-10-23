"""Profile import for the app, in plain Python (no Qt), on top of `forster.imports`.

The column mapping is a `forster.imports.ImportTemplate` (guessed by
`forster.imports.guess`, confirmed in the dialog) and the import is
`forster.imports.apply`: units converted, Fo from an Fo column or in moles from MgO and
FeO (or Mg and Fe), points with an analytical total outside the template's window kept
and flagged. On top of that, `import_table` adds what only the app does:

- the distances can be flipped and shifted so the rim is at 0
  (`forster.project.moved_x`, applied after the unit conversion);
- points in an exclusion column (``Marked_bad`` "bad", ``Ignore`` "yes", ...) come in as
  masked points: kept in the project, left out of the fits;
- the flagged points are listed with their moved distances, for the dialog to show.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from forster.imports import (
    POSITION_UNITS,
    ImportResult,
    ImportTemplate,
    TemplateMismatch,
    apply,
    claim_names,
)
from forster.io import read_table
from forster.model import Profile
from forster.project import Project, moved_x


@dataclass
class Imported:
    """What `import_table` read: the profiles (distances moved), the points an exclusion
    column marks as masked points (per profile: distances, Fo as a fraction), the
    flagged points (total outside the window; kept in the profiles), notes, and the
    core importer's result."""

    profiles: dict[str, Profile]
    result: ImportResult
    masked: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    flagged: pd.DataFrame = field(default_factory=pd.DataFrame)
    notes: list[str] = field(default_factory=list)


def read_columns(path: str | Path, sheet: str | int = 0) -> pd.DataFrame:
    """The table to map (CSV, or one sheet of an .xlsx)."""
    return read_table(path, sheet)


def rim_offset(x: np.ndarray, flip: bool) -> float:
    """The offset that puts the rim at 0: the smallest distance, or the largest when the
    rim is at the far end (``flip``)."""
    x = np.asarray(x, dtype=float)
    return float(np.nanmax(x) if flip else np.nanmin(x))


def position_um(df: pd.DataFrame, template: ImportTemplate) -> np.ndarray:
    """The position column's distances in µm (for "Rim at 0"); a ValueError when the
    template has no usable position column."""
    found = template.by_role("position")
    if len(found) != 1 or found[0].unit not in POSITION_UNITS:
        raise ValueError("choose one distance column, with its unit, first")
    values = pd.to_numeric(df[found[0].column], errors="coerce").to_numpy(dtype=float)
    return values * POSITION_UNITS[str(found[0].unit)]


# Shown whenever distances are shifted to start at the first analysis.
RIM_ADVICE = (
    "each profile's first point was put at 0; for the best fits, give distances measured "
    "from the crystal rim yourself (from an image of the crystal): the first analysis "
    "usually sits a few µm inside the rim"
)


def _moved(points: pd.DataFrame, flip: bool, offsets: dict[str, float]) -> pd.DataFrame:
    if points.empty:
        return points
    out = points.copy()
    x = out["distance_um"].to_numpy(dtype=float)
    shift = out["profile"].astype(str).map(offsets).fillna(0.0).to_numpy(dtype=float)
    out["distance_um"] = moved_x(x, flip, shift)
    return out.sort_values(["profile", "distance_um"], kind="stable")


def _first_points(result, flip: bool) -> dict[str, float]:
    """Each profile's own "rim at 0" offset, from all its points (masked ones too, so
    none lands below 0)."""
    excluded = result.excluded
    offsets = {}
    for key, p in result.profiles.items():
        x = [np.asarray(p.distance_um, dtype=float)]
        if len(excluded):
            x.append(excluded.loc[excluded["profile"] == key, "distance_um"].to_numpy(float))
        offsets[str(key)] = rim_offset(np.concatenate(x), flip)
    return offsets


def import_table(
    df: pd.DataFrame,
    template: ImportTemplate,
    *,
    name: str = "profile",
    flip: bool = False,
    offset_um: float | None = 0.0,
) -> Imported:
    """Profiles from a table and a confirmed import template (`forster.imports.apply`),
    with the distances flipped or shifted (after the unit) and sorted. ``offset_um=None``
    shifts each profile by its own first point (flipped: its last), so each starts at 0.
    Without a profile column every row belongs to one profile called ``name``. A ValueError (a
    `forster.imports.TemplateMismatch` among them) when the template is incomplete, does
    not match the columns, or no profile has enough points."""
    result = apply(df, template, name=name)
    if not result.profiles:
        raise ValueError("no profile with enough points to fit in this table")
    if offset_um is None:
        offsets = _first_points(result, flip)
    else:
        offsets = {str(key): offset_um for key in result.profiles}
    profiles = {}
    for key, p in result.profiles.items():
        x = moved_x(p.distance_um, flip, offsets[str(key)])
        order = np.argsort(x, kind="stable")
        profiles[key] = Profile(x[order], p.fo[order], name=key)
    excluded = _moved(result.excluded, flip, offsets)
    masked = {
        str(key): (rows["distance_um"].to_numpy(float), rows["fo"].to_numpy(float))
        for key, rows in (excluded.groupby("profile", sort=False) if len(excluded) else ())
        if str(key) in profiles
    }
    flagged = _moved(result.flagged, flip, offsets)
    flagged = flagged[flagged["profile"].isin(list(profiles))] if len(flagged) else flagged
    notes = list(result.notes)
    if offset_um is None:
        notes.append(RIM_ADVICE)
    if masked:
        count = sum(len(x) for x, _ in masked.values())
        notes.append(f"{count} point(s) marked in an exclusion column come in masked")
    used = len(result.points) + len(result.excluded)
    if used < len(df):
        notes.append(f"left out {len(df) - used} row(s) with no distance, Fo or name")
    return Imported(profiles, result, masked, flagged, notes)


def import_files(
    paths: list[Path],
    template: ImportTemplate,
    *,
    flip: bool = False,
    offset_um: float | None = 0.0,
    tables: dict[Path, pd.DataFrame] | None = None,
) -> tuple[Imported, dict[Path, str]]:
    """`import_table` for each file with one template (the mapping confirmed for the
    first file), a profile without a name column named after its file, the same flip and
    offset for every file (``offset_um=None``: each profile starts at its own first
    point). The profiles of all files that matched, and ``{file: why}``
    for those that did not (columns the template does not map, an unreadable file, no
    profile with enough points). A ValueError when no file can be imported, or when two
    files give the same profile name (naming both). ``tables`` caches the files read (the
    dialog calls this on every change of the mapping)."""
    tables = {} if tables is None else tables
    owners: dict[str, Path] = {}
    parts: list[Imported] = []
    problems: dict[Path, str] = {}
    for path in paths:
        try:
            if path not in tables:
                tables[path] = read_columns(path)
            part = import_table(
                tables[path], template, name=path.stem, flip=flip, offset_um=offset_um
            )
        except (TemplateMismatch, OSError, ValueError) as exc:
            problems[path] = str(exc)
            continue
        claim_names(owners, part.profiles, path)
        part.notes = [f"{path.name}: {note}" for note in part.notes[1:] if note != RIM_ADVICE]
        parts.append(part)
    if not parts:
        raise ValueError(
            "none of the files can be imported with this mapping:\n- "
            + "\n- ".join(f"{p.name}: {why}" for p, why in problems.items())
        )
    merged = Imported(
        {k: v for part in parts for k, v in part.profiles.items()},
        parts[0].result,
        {k: v for part in parts for k, v in part.masked.items()},
        pd.concat([part.flagged for part in parts if len(part.flagged)])
        if any(len(part.flagged) for part in parts)
        else parts[0].flagged,
        ([RIM_ADVICE] if offset_um is None else []) + [n for part in parts for n in part.notes],
    )
    return merged, problems


def add_imported(project: Project, imported: Imported) -> list[str]:
    """Add the profiles (and their masked points) to the project; their names. A name
    already in the project is an error, and then nothing is added."""
    names = project.add_profiles(imported.profiles)
    for profile, points in imported.masked.items():
        project.masked[profile] = points
    return names


__all__ = [
    "RIM_ADVICE",
    "Imported",
    "add_imported",
    "import_files",
    "import_table",
    "position_um",
    "read_columns",
    "rim_offset",
]
