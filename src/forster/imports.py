"""Import measured profiles from any spreadsheet layout.

The user says which column is which quantity (an `ImportTemplate`: one `ColumnMap` per
column, with its role and unit); `guess` proposes one from the header names and the
values, to be checked and confirmed, never applied silently. `apply` then converts the
units, computes Fo from MgO and FeO when there is no Fo column, flags points whose
analytical total lies outside a window (kept, not dropped), and returns the profiles.
A confirmed mapping saved as an import template (`ImportTemplate.save`, a JSON file,
not the `forster template` parameter tables) imports files with the same columns
again; a file with columns the template does not know, or without ones it needs, stops
with a message (`check_columns`).

Fo from oxides: Fo = (MgO / 40.3044) / (MgO / 40.3044 + FeO / 71.8444), in moles, with
all Fe as Fe2+ (FeO is total iron; Henry, 2026-10-16). Elements in wt% or ppm are
converted to moles the same way (Mg 24.305, Fe 55.845 g/mol).
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from forster.model import Profile

ROLES = (
    "profile",  # profile name (several profiles in one table)
    "position",  # distance along the profile from the edge
    "fo",  # Fo, as a fraction or in percent
    "oxide",  # an oxide concentration (quantity = MgO, FeO, ...)
    "element",  # an element concentration (quantity = Mg, Fe, Ni, ...)
    "total",  # analytical total, wt%
    "stage_x",  # stage coordinates, carried through unconverted
    "stage_y",
    "exclude",  # rows marked here are left out (Marked_bad "bad", Ignore "yes", ...)
    "ignore",
)
POSITION_UNITS = {"um": 1.0, "mm": 1000.0}
FO_UNITS = ("fraction", "percent")
CONCENTRATION_UNITS = {"wt%": 1.0, "ppm": 1e-4}  # to wt%
OXIDE_MASS = {
    "MgO": 40.3044,
    "FeO": 71.8444,
    "SiO2": 60.0843,
    "NiO": 74.6928,
    "MnO": 70.9374,
    "CaO": 56.0774,
    "Al2O3": 101.9613,
    "Cr2O3": 151.9904,
    "TiO2": 79.8658,
    "Na2O": 61.9789,
    "K2O": 94.1960,
    "P2O5": 141.9445,
    "CoO": 74.9326,
    "ZnO": 81.3794,
}
ELEMENT_MASS = {
    "Mg": 24.305,
    "Fe": 55.845,
    "Si": 28.0855,
    "Ni": 58.6934,
    "Mn": 54.938,
    "Ca": 40.078,
    "Al": 26.9815,
    "Cr": 51.9961,
    "Ti": 47.867,
    "Na": 22.9898,
    "K": 39.0983,
    "P": 30.9738,
    "Co": 58.9332,
    "Zn": 65.38,
    "Sc": 44.9559,
    "V": 50.9415,
}
TOTAL_WINDOW = (98.5, 101.5)  # wt%: a total outside it may mean a bad point
EXCLUDE_VALUES = {"bad", "yes", "y", "x", "true", "1", "exclude", "excluded", "ignore"}
FORMAT = "forster-import"
VERSION = 1
MIN_PROFILE_POINTS = 3


class TemplateMismatch(ValueError):
    """A file's columns do not match an import template: stop and ask the user."""


@dataclass(frozen=True)
class ColumnMap:
    """What one column holds: its role, the quantity (for an oxide or element) and the
    unit; ``note`` says why `guess` chose it."""

    column: str
    role: str
    quantity: str | None = None
    unit: str | None = None
    note: str = ""

    def problems(self) -> list[str]:
        """What is missing or wrong in this mapping, for the user to fix."""
        out = []
        if self.role not in ROLES:
            out.append(f"{self.column!r}: unknown role {self.role!r}; use one of {ROLES}")
        if self.role == "position" and self.unit not in POSITION_UNITS:
            out.append(f"{self.column!r}: give the position's unit ({', '.join(POSITION_UNITS)})")
        if self.role == "fo" and self.unit not in FO_UNITS:
            out.append(f"{self.column!r}: give Fo as fraction or percent")
        if self.role in ("oxide", "element"):
            masses = OXIDE_MASS if self.role == "oxide" else ELEMENT_MASS
            if self.quantity not in masses:
                out.append(f"{self.column!r}: name the {self.role} (one of {', '.join(masses)})")
            if self.unit not in CONCENTRATION_UNITS:
                out.append(f"{self.column!r}: give the unit ({', '.join(CONCENTRATION_UNITS)})")
        return out


@dataclass(frozen=True)
class ImportTemplate:
    """A confirmed column mapping, reusable for files with the same columns."""

    columns: tuple[ColumnMap, ...]
    total_window: tuple[float, float] = TOTAL_WINDOW
    name: str = ""
    notes: tuple[str, ...] = field(default=(), compare=False)  # from `guess`

    def by_role(self, role: str) -> list[ColumnMap]:
        return [c for c in self.columns if c.role == role]

    def concentration(self, role: str, quantity: str) -> ColumnMap | None:
        found = [c for c in self.columns if c.role == role and c.quantity == quantity]
        return found[0] if found else None

    def problems(self) -> list[str]:
        """Everything that stops this mapping from being applied."""
        out = [p for c in self.columns for p in c.problems()]
        for role in ("profile", "position", "fo", "total", "stage_x", "stage_y"):
            if len(self.by_role(role)) > 1:
                names = [c.column for c in self.by_role(role)]
                out.append(f"more than one {role} column: {names}; keep one, ignore the rest")
        if not self.by_role("position"):
            out.append("no position column (distance along the profile)")
        if not self.by_role("fo") and not _fo_source(self):
            out.append("no Fo: map an Fo column, or MgO and FeO (or Mg and Fe) columns")
        lo, hi = self.total_window
        if not lo < hi:
            out.append(f"the total window must be increasing, got {self.total_window}")
        names = [c.column for c in self.columns]
        if len(set(names)) != len(names):
            out.append("a column is mapped twice")
        return out

    def check(self) -> None:
        problems = self.problems()
        if problems:
            raise ValueError("the import mapping is not complete:\n- " + "\n- ".join(problems))

    def to_dict(self) -> dict:
        return {
            "format": FORMAT,
            "version": VERSION,
            "name": self.name,
            "total_window": list(self.total_window),
            "columns": [
                {k: v for k, v in asdict(c).items() if v not in (None, "")} for c in self.columns
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> ImportTemplate:
        if data.get("format") != FORMAT:
            raise ValueError(f"not a FORSTER import template (format {data.get('format')!r})")
        if data.get("version", 0) > VERSION:
            raise ValueError(f"import template version {data['version']} is newer than {VERSION}")
        columns = tuple(ColumnMap(**c) for c in data["columns"])
        window = tuple(data.get("total_window", TOTAL_WINDOW))
        return cls(columns, (float(window[0]), float(window[1])), data.get("name", ""))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=1, ensure_ascii=False) + "\n")
        return path

    @classmethod
    def load(cls, path: str | Path) -> ImportTemplate:
        template = cls.from_dict(json.loads(Path(path).read_text()))
        return replace(template, name=template.name or Path(path).name)


def _fo_source(template: ImportTemplate) -> str | None:
    """Where Fo comes from without an Fo column: "oxide" (MgO and FeO), "element"
    (Mg and Fe), or None."""
    for role, (mg, fe) in (("oxide", ("MgO", "FeO")), ("element", ("Mg", "Fe"))):
        if template.concentration(role, mg) and template.concentration(role, fe):
            return role
    return None


# ------------------------------------------------------------------------------ guess


def _key(header: str) -> str:
    """A header reduced for matching: lower case, units and punctuation dropped."""
    text = re.sub(r"\(.*?\)|\[.*?\]", " ", str(header).replace("µ", "u").replace("μ", "u"))
    return re.sub(r"[^a-z0-9#]+", "_", text.lower()).strip("_")


def _unit_in(header: str) -> str | None:
    text = str(header).lower()
    if re.search(r"\bppm\b", text):
        return "ppm"
    if "wt" in text or "%" in text:
        return "wt%"
    return None


PROFILE_KEYS = (
    "profile_name",
    "profile",
    "traverse",
    "transect",
    "line",
    "grain_name",
    "sample",
    "name",
    "label",
)
POSITION_KEYS = (
    "distance_um",
    "distance",
    "distance_from_edge",
    "dist",
    "position",
    "x_distance_from_edge",
    "edge_distance_x_mm",
    "distance_mm",
)
FO_KEYS = ("fo", "fo#", "fo_mol", "fo_mol%", "forsterite", "fo_number")
TOTAL_KEYS = ("total", "totals", "sum", "total_wt")
STAGE_KEYS = {"stage_x": ("x", "stage_x", "x_stage"), "stage_y": ("y", "stage_y", "y_stage")}
EXCLUDE_KEYS = ("marked_bad", "ignore", "exclude", "bad", "omit")


def _position_unit(header: str) -> str:
    text = str(header).lower()
    return "mm" if re.search(r"(\(|_|\s|^)mm(\)|\s|$|_)", text) else "um"


def _oxide_name(header: str) -> str | None:
    text = re.sub(r"\(.*?\)|wt\.?%?|_wt|\s", "", str(header))
    text = text.rstrip("*").removesuffix("t").removesuffix("T") if text[:3] == "FeO" else text
    for name in OXIDE_MASS:
        if text.lower() == name.lower():
            return name
    return None


def _element_name(header: str) -> str | None:
    text = re.sub(r"\(.*?\)|ppm|wt\.?%?|_|\s", "", str(header))
    return text if text in ELEMENT_MASS else None


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _guess_column(header: str, values: pd.Series) -> ColumnMap:
    """The likeliest role of one column on its own (`guess` resolves clashes)."""
    key, numbers = _key(header), _numeric(values)
    numeric = numbers.notna().sum() >= max(1, values.notna().sum() // 2)
    if key in EXCLUDE_KEYS:
        return ColumnMap(header, "exclude", note="rows marked here are left out")
    if not numeric and key in PROFILE_KEYS:
        return ColumnMap(header, "profile", note="text column with a name-like header")
    if numeric and key in POSITION_KEYS:
        return ColumnMap(header, "position", unit=_position_unit(header), note="distance header")
    if numeric and key in FO_KEYS:
        unit = "percent" if numbers.max() > 1.5 else "fraction"
        return ColumnMap(
            header, "fo", unit=unit, note=f"Fo header; values up to {numbers.max():.3g}"
        )
    if numeric and key in TOTAL_KEYS:
        return ColumnMap(header, "total", note="total header")
    for role, keys in STAGE_KEYS.items():
        if numeric and key in keys:
            return ColumnMap(header, role, note="stage coordinate (not converted to distance)")
    oxide = _oxide_name(header) if numeric else None
    if oxide:
        return ColumnMap(header, "oxide", oxide, _unit_in(header) or "wt%", "oxide header")
    element = _element_name(header) if numeric else None
    if element:
        unit = _unit_in(header)
        note = "element header" + ("" if unit else "; give its unit (wt% or ppm)")
        return ColumnMap(header, "element", element, unit, note)
    return ColumnMap(header, "ignore", note="not recognized")


def guess(df: pd.DataFrame, name: str = "") -> ImportTemplate:
    """A proposed mapping for ``df``'s columns, for the user to check: one `ColumnMap` per
    column (``note`` says why), and in ``notes`` the choices made between candidates."""
    maps = [_guess_column(str(c), df[c]) for c in df.columns]
    notes: list[str] = []
    for role, keys in (("profile", PROFILE_KEYS), ("position", POSITION_KEYS), ("fo", FO_KEYS)):
        maps = _keep_best(maps, role, keys, notes)
    if not [m for m in maps if m.role == "position"]:
        notes.append("no position column recognized: map one by hand")
    return ImportTemplate(tuple(maps), name=name, notes=tuple(notes))


def _keep_best(maps: list[ColumnMap], role: str, keys: tuple[str, ...], notes: list[str]):
    """Keep one column of ``role``: the one whose header comes first in ``keys``."""
    found = [m for m in maps if m.role == role]
    if len(found) <= 1:
        return maps
    found.sort(key=lambda m: keys.index(_key(m.column)) if _key(m.column) in keys else len(keys))
    best = found[0]
    others = [m.column for m in found[1:]]
    notes.append(f"{role}: chose {best.column!r} over {others}; change it if that is wrong")
    return [
        m if m.role != role or m is best else replace(m, role="ignore", note=f"another {role}")
        for m in maps
    ]


# ------------------------------------------------------------------------------ apply


def check_columns(columns, template: ImportTemplate) -> None:
    """Raise `TemplateMismatch` when ``columns`` has columns the template does not map,
    or lacks columns the template needs (those not ignored)."""
    have = {str(c) for c in columns}
    known = {c.column for c in template.columns}
    unknown = sorted(have - known)
    missing = sorted(
        c.column for c in template.columns if c.role != "ignore" and c.column not in have
    )
    if unknown or missing:
        parts = []
        if unknown:
            parts.append(f"columns the template does not know: {unknown}")
        if missing:
            parts.append(f"columns the template needs but the file lacks: {missing}")
        label = f"import template {template.name!r}" if template.name else "the import template"
        raise TemplateMismatch(
            f"the file does not match {label}: "
            + "; ".join(parts)
            + ". Map them (or mark them ignore) and save the template again."
        )


@dataclass(frozen=True)
class ImportResult:
    """Profiles keyed by name, the points as a tidy table (profile, distance_um, fo, and
    every mapped concentration, total and stage coordinate, with ``total_ok``), the
    template used, and notes for the user (flagged totals, skipped profiles).
    ``excluded`` holds the points left out by an exclusion column, in the same layout
    (the app keeps them as masked points)."""

    profiles: dict[str, Profile]
    points: pd.DataFrame
    template: ImportTemplate
    notes: tuple[str, ...]
    excluded: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def flagged(self) -> pd.DataFrame:
        """Points whose analytical total lies outside the template's window (kept)."""
        return self.points[self.points["total_ok"].eq(False)]


def _concentration_wt(df: pd.DataFrame, cmap: ColumnMap) -> pd.Series:
    return _numeric(df[cmap.column]) * CONCENTRATION_UNITS[str(cmap.unit)]


def _fo(df: pd.DataFrame, template: ImportTemplate) -> pd.Series:
    """Fo as a fraction: from the Fo column, or in moles from MgO and FeO (all Fe as
    Fe2+) or from Mg and Fe."""
    fo_maps = template.by_role("fo")
    if fo_maps:
        values = _numeric(df[fo_maps[0].column])
        return values / 100 if fo_maps[0].unit == "percent" else values
    role = _fo_source(template)
    masses = OXIDE_MASS if role == "oxide" else ELEMENT_MASS
    mg_name, fe_name = ("MgO", "FeO") if role == "oxide" else ("Mg", "Fe")
    mg_map, fe_map = (
        template.concentration(str(role), mg_name),
        template.concentration(str(role), fe_name),
    )
    assert mg_map is not None and fe_map is not None
    mg = _concentration_wt(df, mg_map) / masses[mg_name]
    fe = _concentration_wt(df, fe_map) / masses[fe_name]
    return mg / (mg + fe)


def _total(df: pd.DataFrame, template: ImportTemplate) -> tuple[pd.Series, str]:
    """The analytical total (wt%) and where it came from: the total column, or the sum of
    the oxide columns when SiO2, MgO and FeO are all mapped; else NaN."""
    total_maps = template.by_role("total")
    if total_maps:
        return _numeric(df[total_maps[0].column]), f"column {total_maps[0].column!r}"
    oxides = template.by_role("oxide")
    if {"SiO2", "MgO", "FeO"} <= {c.quantity for c in oxides}:
        summed = sum(
            (_concentration_wt(df, c).fillna(0.0) for c in oxides), pd.Series(0.0, df.index)
        )
        return summed, "sum of the oxide columns"
    return pd.Series(np.nan, index=df.index), "none"


def _excluded(df: pd.DataFrame, template: ImportTemplate) -> pd.Series:
    out = pd.Series(False, index=df.index)
    for cmap in template.by_role("exclude"):
        out |= df[cmap.column].astype(str).str.strip().str.lower().isin(EXCLUDE_VALUES)
    return out


def _points(
    df: pd.DataFrame, template: ImportTemplate, default_name: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The tidy table of every usable point, and of the points an exclusion column
    leaves out."""
    position = template.by_role("position")[0]
    profile = template.by_role("profile")
    out = pd.DataFrame(index=df.index)
    out["profile"] = df[profile[0].column].astype(str) if profile else default_name
    out["distance_um"] = _numeric(df[position.column]) * POSITION_UNITS[str(position.unit)]
    out["fo"] = _fo(df, template)
    for cmap in template.by_role("oxide") + template.by_role("element"):
        out[f"{cmap.quantity}_wt%"] = _concentration_wt(df, cmap)
    total, _ = _total(df, template)
    out["total_wt%"] = total
    lo, hi = template.total_window
    out["total_ok"] = total.between(lo, hi).where(total.notna(), other=np.nan)
    for role in ("stage_x", "stage_y"):
        for cmap in template.by_role(role):
            out[role] = df[cmap.column]
    if profile:
        out = out[df[profile[0].column].notna()]
    usable = out["distance_um"].notna() & out["fo"].notna()
    excluded = _excluded(df, template).loc[out.index]
    return out[usable & ~excluded], out[usable & excluded]


def apply(
    df: pd.DataFrame, template: ImportTemplate, *, source: str = "", name: str = ""
) -> ImportResult:
    """Import ``df`` with ``template`` (checked first; see `check_columns`). Without a
    profile column every row belongs to one profile, called ``name`` (default: the
    stem of ``source``, else "profile")."""
    template.check()
    check_columns(df.columns, template)
    default_name = name or (Path(source).stem if source else "profile")
    points, excluded = _points(df, template, default_name)
    notes: list[str] = []
    profiles: dict[str, Profile] = {}
    short = []
    for key, rows in points.groupby("profile", sort=False):
        rows = rows.sort_values("distance_um")
        if len(rows) < MIN_PROFILE_POINTS:
            short.append(str(key))
            continue
        profiles[str(key)] = Profile(
            rows["distance_um"].to_numpy(float), rows["fo"].to_numpy(float), name=str(key)
        )
    if short:
        notes.append(
            f"skipped {len(short)} name(s) with fewer than {MIN_PROFILE_POINTS} points: {short}"
        )
    _, total_from = _total(df, template)
    flagged = points[points["total_ok"].eq(False)]
    if len(flagged):
        lo, hi = template.total_window
        by_profile = flagged.groupby("profile").size().to_dict()
        notes.append(
            f"{len(flagged)} point(s) with an analytical total outside {lo:g}-{hi:g} wt% "
            f"({total_from}), kept and flagged (total_ok False): {by_profile}"
        )
    used = template.name or "the confirmed mapping"
    notes.insert(0, f"imported with {used}: {len(profiles)} profile(s), {len(points)} point(s)")
    return ImportResult(
        profiles,
        points.reset_index(drop=True),
        template,
        tuple(notes),
        excluded.reset_index(drop=True),
    )


def read_with_template(
    path: str | Path, template: ImportTemplate, *, sheet: str | int = 0
) -> ImportResult:
    """`apply` to a CSV or Excel file."""
    from forster.io import read_table

    return apply(read_table(path, sheet), template, source=str(path))


def template_table(points: pd.DataFrame) -> pd.DataFrame:
    """The points as a FORSTER profiles table (`forster.io.read_profiles` reads it back:
    profile, distance_um, fo as a fraction, then the other columns)."""
    first = ["profile", "distance_um", "fo"]
    return points[first + [c for c in points.columns if c not in first]]
