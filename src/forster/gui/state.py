"""The GUI's state for one run, in plain Python (no Qt): the segment table, the other
fields of the side panel, their validation, and the mapping to and from a `Run`.

Henry's side table (docs/plans/2026-10-16-geometry-naming.md, decision 3) has one row
per segment: its name, start and end position and Fo, and the Fo's uncertainty (a
Gaussian sd or a uniform half-width). The starts and ends are the model domain's ends
and the steps, so editing one segment's end moves the next one's start. Each step
position has its own uncertainty (one kind, sd or range, for all steps, as the table
columns ``steps_sd_um`` / ``steps_range_um`` allow).

Every edit goes through `forster.project.Project.update_run`, which checks the edited
run by writing it as a table row and reading it back: the GUI accepts exactly what the
command line does. A refused edit leaves the run unchanged and its message is kept in
`RunEditor.errors` under the field's key, for the window to show next to that field.

Fo is shown in mol% and stored as a fraction; distances in um.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from forster.diffusivity import FE_MG_CHOICES, label, resolve
from forster.grid import GridRequest
from forster.io import Run, RunDefaults, scan_summary, uncertainty_columns
from forster.model import (
    FO2_BUFFERS,
    Conditions,
    InitialCondition,
    Orientation,
    default_segment_names,
)
from forster.plotting import SIGMA
from forster.project import Project
from forster.sampling import METHODS as SAMPLING_METHODS
from forster.scan import Uncertainty, resolve_names

KINDS = {"sd": "normal", "range": "uniform"}  # the table's word -> `Uncertainty.kind`
WORDS = {v: k for k, v in KINDS.items()}
SCAN_METHODS = (*SAMPLING_METHODS, "grid")
SEGMENT_COLUMNS = ("name", "start", "end", "fo", "spread", "kind")
SEGMENT_TITLES = {
    "name": "Segment",
    "start": "Start (µm)",
    "end": "End (µm)",
    "fo": "Fo (mol%)",
    "spread": "± Fo (mol%)",
    "kind": "sd or range",
}
# What each offered Fe-Mg equation changes, for the Diffusion equation tab (README,
# parameter column ``diffusivity``).
DIFFUSIVITY_NOTES = {
    "dipra": "The default. No covariance, so no range from the equation itself.",
    "refit": "This project's refit of the verified experiments. Carries a covariance "
    "and the spread between laboratories (the law_ and all_ ranges).",
    "dfens_tamed": "The Mutch et al. (2019) regression used by DFENS. Carries a "
    "covariance (the law_ range).",
    "dfens_global": "The Mutch et al. (2019) global regression used by DFENS. Carries a "
    "covariance (the law_ range).",
}
STEP_SNAP_UM = 0.1  # drags snap to 0.1 um and 0.01 mol%: finer than the data
FO_SNAP_PCT = 0.01


@dataclass(frozen=True)
class SegmentRow:
    """One row of the segment table: Fo and its spread as fractions."""

    name: str
    start_um: float
    end_um: float
    fo: float
    spread: float | None = None
    kind: str = "sd"  # "sd" or "range"


def pct(fo: float | None) -> str:
    """A fraction as mol% text ("" for None)."""
    return "" if fo is None else f"{fo * 100:.6g}"


def um(x: float | None) -> str:
    return "" if x is None else f"{x:.6g}"


def number(text: str, what: str, *, optional: bool = False) -> float | None:
    """Parse a typed number; blank is None when ``optional``, else an error."""
    text = text.strip()
    if not text:
        if optional:
            return None
        raise ValueError(f"{what}: give a number")
    try:
        value = float(text)
    except ValueError:
        raise ValueError(f"{what}: {text!r} is not a number") from None
    if not math.isfinite(value):
        raise ValueError(f"{what}: {text!r} is not a finite number")
    return value


def fraction(text: str, what: str, *, optional: bool = False) -> float | None:
    """mol% text to a fraction, rounded so 91.46 is stored as 0.9146."""
    value = number(text, what, optional=optional)
    return None if value is None else round(value / 100, 10)


def fo_fraction(text: str, what: str) -> float:
    """An Fo typed in mol% (0-100) as a fraction."""
    value = number(text, what)
    assert value is not None
    if not 0 < value <= 100:
        raise ValueError(f"{what}: Fo is in mol%, above 0 and at most 100; got {text.strip()}")
    return round(value / 100, 10)


def rows_of(run: Run) -> list[SegmentRow]:
    """The run's segment table."""
    ic = run.initial
    edges = (ic.domain_um[0], *ic.steps_um, ic.domain_um[1])
    given = resolve_names(ic, run.uncertainties)
    rows = []
    for i, (name, fo) in enumerate(zip(ic.segment_names, ic.segment_fo, strict=True)):
        u = given.get(f"{name}_fo")
        rows.append(
            SegmentRow(
                name,
                float(edges[i]),
                float(edges[i + 1]),
                float(fo),
                u.spread if u else None,
                WORDS[u.kind] if u else "sd",
            )
        )
    return rows


def step_spreads_of(run: Run) -> tuple[list[float | None], str]:
    """Each step's uncertainty and their shared kind ("sd" or "range")."""
    n = len(run.initial.steps_um)
    found = [run.uncertainties.get(f"step{i + 1}_um") for i in range(n)]
    kinds = [u.kind for u in found if u is not None]
    return [u.spread if u else None for u in found], WORDS[kinds[0]] if kinds else "sd"


def initial_of(rows: list[SegmentRow], centre_um: float | None) -> InitialCondition:
    """The starting profile a segment table gives (InitialCondition checks it)."""
    for left, right in itertools.pairwise(rows):
        if left.end_um != right.start_um:
            raise ValueError(f"segment {right.name} must start where {left.name} ends")
    return InitialCondition(
        (rows[0].start_um, rows[-1].end_um),
        steps_um=[r.end_um for r in rows[:-1]],
        segment_fo=[r.fo for r in rows],
        segment_names=[r.name for r in rows],
        centre_um=centre_um,
    )


def uncertainties_of(
    rows: list[SegmentRow], step_spreads: list[float | None], steps_kind: str
) -> dict[str, Uncertainty]:
    """The run's ``uncertainties`` from the table: blank or 0 leaves a value fixed."""
    out = {
        f"step{i + 1}_um": Uncertainty(s, KINDS[steps_kind])
        for i, s in enumerate(step_spreads)
        if s
    }
    out |= {f"{r.name}_fo": Uncertainty(r.spread, KINDS[r.kind]) for r in rows if r.spread}
    return out


def _relabel(rows: list[SegmentRow], had_defaults: bool) -> list[SegmentRow]:
    """Default names a, b, c ... again after a split or merge, if they were defaults."""
    if not had_defaults:
        return rows
    names = default_segment_names(len(rows))
    return [replace(r, name=n) for r, n in zip(rows, names, strict=True)]


def _free_name(rows: list[SegmentRow]) -> str:
    taken = {r.name for r in rows}
    names = (*default_segment_names(26), *default_segment_names(27 * 26))
    return next(n for n in names if n not in taken)


class RunEditor:
    """The selected run as the side panel edits it.

    Each setter returns True when the edit was accepted (the run in the project is
    updated and should be refitted) and False when it was refused; its message is then
    in ``errors[key]``. Keys: ``segment:<row>:<column>``, ``step:<i>``, ``steps_kind``,
    ``edge1``, ``centre``, ``drag``, the condition and orientation field names,
    ``diffusivity``, ``scan_method`` and ``initial``.
    """

    def __init__(self, project: Project, name: str) -> None:
        self.project, self.name = project, name
        self.errors: dict[str, str] = {}
        self.reload()

    @property
    def run(self) -> Run:
        return self.project.run(self.name)

    def reload(self) -> None:
        """Read the table back from the run (after a refused edit, or another change)."""
        run = self.run
        self.rows = rows_of(run)
        self.step_spreads, self.steps_kind = step_spreads_of(run)

    # -- committing -------------------------------------------------------------------

    def _commit(self, key: str, **changes: Any) -> bool:
        try:
            self.project.update_run(self.name, **changes)
        except (ValueError, KeyError) as error:
            self.errors[key] = str(error.args[0] if isinstance(error, KeyError) else error)
            self.reload()
            return False
        self.errors.pop(key, None)
        self.reload()
        return True

    def _commit_table(
        self,
        key: str,
        rows: list[SegmentRow],
        step_spreads: list[float | None] | None = None,
        steps_kind: str | None = None,
        centre_um: float | str | None = "keep",
    ) -> bool:
        """The run with this segment table (and step uncertainties), or the error."""
        spreads = self.step_spreads if step_spreads is None else step_spreads
        kind = steps_kind or self.steps_kind
        centre = self.run.initial.centre_um if centre_um == "keep" else centre_um
        assert not isinstance(centre, str)
        try:
            initial = initial_of(rows, centre)
            uncertainties = uncertainties_of(rows, spreads, kind)
        except ValueError as error:
            self.errors[key] = str(error)
            self.reload()
            return False
        return self._commit(key, initial=initial, uncertainties=uncertainties)

    # -- the segment table ------------------------------------------------------------

    def set_cell(self, row: int, column: str, text: str) -> bool:
        """A typed value in the segment table."""
        key = f"segment:{row}:{column}"
        rows = list(self.rows)
        title = f"segment {rows[row].name}, {SEGMENT_TITLES[column]}"
        try:
            rows = self._edited(rows, row, column, text, title)
        except ValueError as error:
            self.errors[key] = str(error)
            return False
        return self._commit_table(key, rows)

    def _edited(
        self, rows: list[SegmentRow], row: int, column: str, text: str, title: str
    ) -> list[SegmentRow]:
        r = rows[row]
        if column == "name":
            rows[row] = replace(r, name=text.strip())
        elif column == "fo":
            rows[row] = replace(r, fo=fo_fraction(text, title))
        elif column == "spread":
            rows[row] = replace(r, spread=fraction(text, title, optional=True) or None)
        elif column == "kind":
            word = text.strip().lower()
            if word not in KINDS:
                raise ValueError(f"{title}: give sd or range")
            rows[row] = replace(r, kind=word)
        elif column in ("start", "end"):
            value = number(text, title)
            assert value is not None
            rows = self._moved(rows, row if column == "start" else row + 1, value)
        else:
            raise ValueError(f"no column {column!r}")
        return rows

    @staticmethod
    def _moved(rows: list[SegmentRow], edge: int, x_um: float) -> list[SegmentRow]:
        """Move edge ``edge`` (0: the domain's start, n: its end, else step ``edge``):
        the segments on both sides of it follow."""
        rows = list(rows)
        if edge > 0:
            rows[edge - 1] = replace(rows[edge - 1], end_um=x_um)
        if edge < len(rows):
            rows[edge] = replace(rows[edge], start_um=x_um)
        return rows

    def add_segment(self, row: int) -> bool:
        """Split a segment in two at its middle (the new one takes its Fo)."""
        rows = list(self.rows)
        defaults = [r.name for r in rows] == list(default_segment_names(len(rows)))
        r = rows[row]
        middle = round((r.start_um + r.end_um) / 2, 1)
        new = replace(r, name=_free_name(rows), start_um=middle, spread=None)
        rows[row : row + 1] = [replace(r, end_um=middle), new]
        return self._commit_table("segments", _relabel(rows, defaults), self._spreads_with(row))

    def remove_segment(self, row: int) -> bool:
        """Remove a segment: the next one (or, for the last, the one before) takes its
        place. The step between them goes, with its uncertainty."""
        rows = list(self.rows)
        if len(rows) <= 2:
            self.errors["segments"] = "a starting profile needs at least two segments (one step)"
            return False
        defaults = [r.name for r in rows] == list(default_segment_names(len(rows)))
        spreads = list(self.step_spreads)
        r = rows.pop(row)
        if row < len(rows):
            rows[row] = replace(rows[row], start_um=r.start_um)
            del spreads[row]  # the step at its end goes
        else:
            rows[row - 1] = replace(rows[row - 1], end_um=r.end_um)
            del spreads[row - 1]
        return self._commit_table("segments", _relabel(rows, defaults), spreads)

    def _spreads_with(self, row: int) -> list[float | None]:
        """Step uncertainties with a new, fixed step after segment ``row``."""
        spreads = list(self.step_spreads)
        spreads.insert(row, None)
        return spreads

    # -- drags on the plot --------------------------------------------------------------

    def dragged_rows(self, kind: str, index: int, value: float) -> list[SegmentRow]:
        """The table as a drag would leave it (shown while dragging, not committed):
        ``kind`` "step" (index: the step), "start", "end" (domain ends, in um) or "fo"
        (index: the segment; value in mol%)."""
        if kind == "fo":
            rows = list(self.rows)
            rows[index] = replace(rows[index], fo=round(value / 100, 10))
            return rows
        edge = {"start": 0, "end": len(self.rows)}.get(kind, index + 1)
        return self._moved(self.rows, edge, value)

    def drag(self, kind: str, index: int, value: float) -> bool:
        """Commit a released drag."""
        if kind == "fo" and not 0 < value <= 100:
            self.errors["drag"] = f"Fo is in mol%, above 0 and at most 100; got {value:g}"
            self.reload()
            return False
        return self._commit_table("drag", self.dragged_rows(kind, index, value))

    @staticmethod
    def snap(kind: str, value: float) -> float:
        step = FO_SNAP_PCT if kind == "fo" else STEP_SNAP_UM
        return round(round(value / step) * step, 6)

    # -- edges and the crystal centre -----------------------------------------------------

    @property
    def edge1_text(self) -> str:
        return pct(self.run.rim_fo)

    @property
    def centre_text(self) -> str:
        return um(self.run.initial.centre_um)

    def set_edge1(self, text: str) -> bool:
        """The Fo held at edge 1 (blank: segment a's Fo)."""
        try:
            value = fraction(text, "Edge 1 Fo", optional=True)
        except ValueError as error:
            self.errors["edge1"] = str(error)
            return False
        return self._commit("edge1", rim_fo=value)

    def set_centre(self, text: str) -> bool:
        """The crystal centre (blank: a half profile, its far end held)."""
        try:
            value = number(text, "Crystal centre", optional=True)
        except ValueError as error:
            self.errors["centre"] = str(error)
            return False
        return self._commit_table("centre", self.rows, centre_um=value)

    # -- step uncertainties -------------------------------------------------------------

    def set_step_spread(self, i: int, text: str) -> bool:
        try:
            value = number(text, f"step {i + 1} ± (µm)", optional=True)
        except ValueError as error:
            self.errors[f"step:{i}"] = str(error)
            return False
        spreads = list(self.step_spreads)
        spreads[i] = value or None
        return self._commit_table(f"step:{i}", self.rows, spreads)

    def set_steps_kind(self, word: str) -> bool:
        if word not in KINDS:
            self.errors["steps_kind"] = "give sd or range"
            return False
        return self._commit_table("steps_kind", self.rows, steps_kind=word)

    def set_scan_method(self, method: str) -> bool:
        return self._commit("scan_method", scan_method=method)

    def apply_initial(self, initial: InitialCondition) -> bool:
        """Use a starting profile (the starting-profile fit's best), rounded to 0.1 um and
        1e-4 Fo so the table stays readable; names and uncertainties are kept."""
        rounded = replace(
            initial,
            steps_um=tuple(round(s, 1) for s in initial.steps_um),
            segment_fo=tuple(round(f, 4) for f in initial.segment_fo),
        )
        return self._commit("initial", initial=rounded)

    # -- conditions ---------------------------------------------------------------------

    def conditions_text(self) -> dict[str, str]:
        run = self.run
        c = run.conditions
        out = {
            "T_C": f"{c.T_C:.6g}",
            "P_MPa": f"{c.P_Pa / 1e6:.6g}",
            "fO2_Pa": f"{c.fO2_Pa:.4g}",
            "fO2_mode": "buffer" if run.fO2_form else "Pa",
            "fO2_buffer": run.fO2_form[0] if run.fO2_form else "QFM",
            "fO2_delta_log": f"{run.fO2_form[1]:.6g}" if run.fO2_form else "0",
        }
        return out

    def set_conditions(self, fields: dict[str, str], edited: str) -> bool:
        """T (deg C), P (MPa) and fO2 (``fO2_mode`` "buffer": ``fO2_buffer`` and
        ``fO2_delta_log``; "Pa": ``fO2_Pa``), all given; ``edited`` names the field the
        user changed, under which a refusal is kept."""
        try:
            T = number(fields["T_C"], "T (°C)")
            P = number(fields["P_MPa"], "P (MPa)")
            assert T is not None and P is not None
            if fields["fO2_mode"] == "buffer":
                buffer = fields["fO2_buffer"].strip().upper()
                if buffer not in FO2_BUFFERS:
                    raise ValueError(f"fO2 buffer: use one of {', '.join(FO2_BUFFERS)}")
                delta = number(fields["fO2_delta_log"], "fO2 offset (log units)")
                assert delta is not None
                conditions = Conditions.from_buffer(T, P * 1e6, delta, buffer)
                return self._commit(edited, conditions=conditions, fO2_form=(buffer, float(delta)))
            fo2 = number(fields["fO2_Pa"], "fO2 (Pa)")
            assert fo2 is not None
            if fo2 <= 0:
                raise ValueError("fO2 (Pa) must be positive")
        except ValueError as error:
            self.errors[edited] = str(error)
            return False
        return self._commit(edited, conditions=Conditions(T, P * 1e6, fo2), fO2_form=None)

    # -- orientation --------------------------------------------------------------------

    def orientation_text(self) -> dict[str, str]:
        run = self.run
        o = run.orientation
        out = {
            "mode": "euler" if run.euler_form else "angles",
            "alpha_deg": f"{o.alpha_deg:.6g}",
            "beta_deg": f"{o.beta_deg:.6g}",
            "gamma_deg": f"{o.gamma_deg:.6g}",
        }
        names = ("phi1_deg", "Phi_deg", "phi2_deg", "profile_angle_deg")
        values = run.euler_form or ("",) * 4
        out |= {n: (f"{v:.6g}" if v != "" else "") for n, v in zip(names, values, strict=True)}
        return out

    def set_orientation(self, fields: dict[str, str], edited: str) -> bool:
        """``mode`` "angles" (alpha, beta, gamma: degrees to [100], [010], [001]) or
        "euler" (EBSD phi1, Phi, phi2 and the profile's angle on the map)."""
        try:
            if fields["mode"] == "angles":
                a, b, g = (
                    number(fields[k], k.removesuffix("_deg"))
                    for k in ("alpha_deg", "beta_deg", "gamma_deg")
                )
                assert a is not None and b is not None and g is not None
                return self._commit(edited, orientation=Orientation(a, b, g), euler_form=None)
            names = ("phi1_deg", "Phi_deg", "phi2_deg", "profile_angle_deg")
            values = [number(fields[k], k.removesuffix("_deg")) for k in names]
        except ValueError as error:
            self.errors[edited] = str(error)
            return False
        p1, big, p2, angle = (float(v or 0.0) for v in values)
        return self._commit(edited, euler_form=(p1, big, p2, angle))

    # -- diffusion equation -------------------------------------------------------------

    def diffusivity_choices(self) -> list[tuple[str, str]]:
        """(key, label) of the offered Fe-Mg equations, plus the run's own if it uses
        another one (a project equation or preset)."""
        choices = [(alias, text) for alias, (_, text) in FE_MG_CHOICES.items()]
        current = self.diffusivity_key
        if current not in FE_MG_CHOICES:
            choices.append((current, label(self.run.diffusivity)))
        return choices

    @property
    def diffusivity_key(self) -> str:
        name = self.run.diffusivity.name
        return next((a for a, (n, _) in FE_MG_CHOICES.items() if n == name), name)

    def set_diffusivity(self, key: str) -> bool:
        try:
            equation = resolve(key, self.project.diffusivities)
        except ValueError as error:
            self.errors["diffusivity"] = str(error)
            return False
        return self._commit("diffusivity", diffusivity=equation)


# -- runs and results across the project ------------------------------------------------


def new_run(project: Project, name: str, profile_name: str, like: str | None = None) -> Run:
    """A run on ``profile_name`` with the settings of run ``like`` (or defaults: 1200
    deg C, 100 MPa, QFM, the default equation), its starting profile spanning the
    measured points with one step in the middle and the end points' Fo."""
    profile = project.profiles[profile_name]
    x, fo = np.asarray(profile.distance_um), np.asarray(profile.fo)
    lo, hi = float(np.nanmin(x)), float(np.nanmax(x))
    order = np.argsort(x)
    edge, core = float(fo[order[0]]), float(fo[order[-1]])
    initial = InitialCondition(
        (lo, hi),
        steps_um=[round((lo + hi) / 2, 1)],
        segment_fo=[round(min(edge, 1.0), 4), round(min(core, 1.0), 4)],
    )
    if like is not None:
        base = project.run(like)
        return replace(
            base,
            name=name,
            profile_name=profile_name,
            initial=initial,
            uncertainties={},
            stages=(),
            rim_fo=None,
            metadata={},
        )
    defaults = RunDefaults()
    return Run(
        name=name,
        profile_name=profile_name,
        conditions=Conditions.from_buffer(1200.0, 1e8, 0.0, "QFM"),
        orientation=Orientation(90.0, 90.0, 0.0),
        initial=initial,
        grid=GridRequest(max_time_days=defaults.max_time_days),
        kriging=defaults.kriging,
        diffusivity=resolve(defaults.diffusivity),
        fO2_form=("QFM", 0.0),
    )


RESULT_COLUMNS = (
    "run",
    "profile",
    "equation",
    "days",
    "low_1s",
    "high_1s",
    "low_2s",
    "high_2s",
    "status",
    "warnings",
)
RESULT_TITLES = {
    "run": "Run",
    "profile": "Profile",
    "equation": "Diffusion equation",
    "days": "Days",
    "low_1s": f"1{SIGMA} low",
    "high_1s": f"1{SIGMA} high",
    "low_2s": f"2{SIGMA} low",
    "high_2s": f"2{SIGMA} high",
    "status": "Status",
    "warnings": "Warnings",
}


def result_rows(project: Project) -> list[dict[str, Any]]:
    """One row per run for the results table, from the cache (no fitting)."""
    rows = []
    for run in project.runs:
        summary = project.summary(run.name)
        status = "fitted" if summary else "changed" if project.is_stale(run.name) else "not fitted"
        row: dict[str, Any] = {
            "run": run.name,
            "profile": run.profile_name,
            "equation": label(run.diffusivity),
            "status": status,
            "warnings": len(project.fit_warnings(run.name) or ()),
        }
        if summary:
            row |= {
                "days": summary["time_days"],
                "low_1s": summary["low_days"],
                "high_1s": summary["high_days"],
                "low_2s": summary["low2_days"],
                "high_2s": summary["high2_days"],
            }
        rows.append(row)
    return rows


def summary_table(project: Project) -> pd.DataFrame:
    """summary.csv for the fitted runs, as ``forster fit`` writes it (with the cached
    starting-profile fit's columns where there is one)."""
    rows = []
    for run in project.runs:
        summary = project.summary(run.name)
        if summary is None:
            continue
        row = dict(summary)
        row = {"run": row.pop("run"), "profile": run.profile_name, **row}
        row |= uncertainty_columns(run)
        scan = project.scan(run.name) if run.uncertainties else None
        if scan is not None:
            row |= scan.summary
        row["warnings"] = "; ".join(project.fit_warnings(run.name) or ())
        rows.append(row)
    return pd.DataFrame(rows)


def scan_lines(summary: dict[str, Any]) -> list[str]:
    """A starting-profile fit's summary as short lines for the Uncertainties tab."""
    lines = [
        f"Best fit {summary['scan_best_days']:.3g} days; 68% range "
        f"{summary['scan_low_days']:.3g}-{summary['scan_high_days']:.3g} days "
        f"({summary['scan_sampler']})",
        f"Best starting profile: {summary['scan_best']}",
    ]
    share = summary.get("scan_ic_share")
    if share is not None and math.isfinite(share):
        lines.append(f"{share:.0%} of the spread in time comes from the starting profile")
    return lines


__all__ = [
    "SCAN_METHODS",
    "RunEditor",
    "SegmentRow",
    "new_run",
    "result_rows",
    "scan_lines",
    "scan_summary",
    "summary_table",
]
