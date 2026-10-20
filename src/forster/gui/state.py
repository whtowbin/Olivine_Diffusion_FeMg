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

from forster.boundary import SECONDS_PER_DAY, Stage
from forster.diffusivity import FE_MG_CHOICES, label, resolve
from forster.grid import GridRequest, grid_diagnostics
from forster.io import Run, RunDefaults, scan_summary, uncertainty_columns
from forster.model import (
    FO2_BUFFERS,
    SCHEMES,
    Conditions,
    FitResult,
    Grid,
    InitialCondition,
    Orientation,
    default_segment_names,
)
from forster.plotting import SIGMA
from forster.project import FitTask, Project, default_initial
from forster.rescale import ConditionUncertainty, PosteriorOfTime, TimeDistribution
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
            value = number(text, "Crystal center", optional=True)
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
        presets = {name for name, _ in FE_MG_CHOICES.values()}
        for name, equation in self.project.diffusivities.items():
            if equation.element == "Fe-Mg" and name not in presets:
                choices.append((name, label(equation)))
        current = self.diffusivity_key
        if current not in dict(choices):
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

    def diffusivity_source(self, key: str) -> str:
        """Where an offered equation comes from (its ``source``), for the tooltip."""
        try:
            equation = resolve(key, self.project.diffusivities)
        except ValueError:
            return ""
        return str(getattr(equation, "source", "") or "No source recorded.")

    # -- the profile's points: masking and moving ----------------------------------------

    def points(self) -> pd.DataFrame:
        """The run's profile, masked points too (`Project.points`)."""
        return self.project.points(self.run.profile_name)

    def set_point_masked(self, index: int, masked: bool) -> bool:
        """Mask point ``index`` (a row of `points`) out of the fits, or back in. This
        changes the profile, so every run on it."""
        flags = self.points().masked.to_numpy().copy()
        flags[index] = masked
        try:
            self.project.set_masked(self.run.profile_name, flags)
        except ValueError as error:
            self.errors["mask"] = str(error)
            return False
        self.errors.pop("mask", None)
        return True

    def nearest_point(self, x_um: float, fo_pct: float, x_scale: float, fo_scale: float) -> int:
        """The row of `points` nearest a place on the plot, distances measured in units
        of ``x_scale`` um and ``fo_scale`` mol% (the plot's pixels per unit)."""
        table = self.points()
        d = np.hypot((table.distance_um - x_um) * x_scale, (table.fo * 100 - fo_pct) * fo_scale)
        return int(d.to_numpy().argmin())

    def rim_offset_text(self, flip: bool) -> str:
        """The offset that puts the rim at 0 (the first point, or the last when flipped)."""
        x = self.points().distance_um.to_numpy()
        return f"{float(x.max() if flip else x.min()):.12g}"  # exact enough to land on 0

    def move_profile(self, flip: bool, offset_text: str) -> bool:
        """Flip and / or shift the run's profile (`Project.move_profile`): runs on it move
        with a shift; a flip gives them new starting profiles."""
        try:
            offset = number(offset_text, "Offset (µm)", optional=True) or 0.0
            if not flip and not offset:
                raise ValueError("give an offset, or flip the profile")
            self.project.move_profile(self.run.profile_name, flip=flip, offset_um=offset)
        except ValueError as error:
            self.errors["move"] = str(error)
            return False
        self.errors.pop("move", None)
        self.reload()
        return True

    # -- the Monte Carlo ----------------------------------------------------------------

    def mc_buffers(self) -> list[str]:
        """The choices for holding fO2 to a buffer in the Monte Carlo: "" (not held: log
        fO2 drawn on its own) and the buffers. A run whose fO2 is given relative to a
        buffer is always held (the table reads a blank mc_fO2_buffer as its own)."""
        buffers = [b for b in FO2_BUFFERS if b != "FMQ"]
        return buffers if self.run.fO2_form else ["", *buffers]

    def mc_text(self) -> dict[str, str]:
        """The Monte Carlo fields (``on`` and the flags "yes" / "no"; P in MPa; blank
        samples: the default)."""
        run = self.run
        mc = run.mc or ConditionUncertainty(fO2_buffer=run.fO2_form[0] if run.fO2_form else None)
        buffer = mc.fO2_buffer or ""
        return {
            "on": "yes" if run.mc is not None else "no",
            "T_sd_C": f"{mc.T_sd_C:.6g}",
            "P_sd_MPa": f"{mc.P_sd_Pa / 1e6:.6g}",
            "fO2_sd_log": f"{mc.fO2_sd_log:.6g}",
            "fO2_buffer": "QFM" if buffer == "FMQ" else buffer,
            "law": "yes" if mc.law else "no",
            "between_study": "yes" if mc.between_study else "no",
            "samples": "" if run.mc_samples is None else str(run.mc_samples),
            "seed": str(run.mc_seed),
        }

    def set_mc(self, fields: dict[str, str], edited: str) -> bool:
        """The Monte Carlo from its fields (`mc_text`); ``on`` "no" turns it off."""
        if fields["on"] != "yes":
            return self._commit(edited, mc=None, mc_samples=None, mc_seed=0)
        try:
            spreads = [
                number(fields[k], what) or 0.0
                for k, what in (
                    ("T_sd_C", "T sd (°C)"),
                    ("P_sd_MPa", "P sd (MPa)"),
                    ("fO2_sd_log", "fO2 sd (log units)"),
                )
            ]
            samples = number(fields["samples"], "Draws", optional=True)
            seed = number(fields["seed"] or "0", "Seed")
            for value, what in ((samples, "Draws"), (seed, "Seed")):
                if value is not None and (value != int(value) or value < 0):
                    raise ValueError(f"{what}: give a whole number, 0 or more")
            if samples is not None and samples < 10:
                raise ValueError("Draws: give at least 10 (blank: the default)")
            mc = ConditionUncertainty(
                T_sd_C=spreads[0],
                P_sd_Pa=spreads[1] * 1e6,
                fO2_sd_log=spreads[2],
                fO2_buffer=fields["fO2_buffer"].strip().upper() or None,
                law=fields["law"] == "yes",
                between_study=fields["between_study"] == "yes",
            )
        except ValueError as error:
            self.errors[edited] = str(error)
            return False
        return self._commit(
            edited,
            mc=mc,
            mc_samples=None if samples is None else int(samples),
            mc_seed=int(seed or 0),
        )

    # -- advanced: scheme, grid and growth stages ------------------------------------------

    def grid_text(self) -> dict[str, str]:
        """``scheme`` and the grid's ``dx_um``, ``dt_s``, ``max_time_days`` (blank: chosen
        at fit time, `forster.grid.choose_grid`)."""
        grid = self.run.grid
        return {
            "scheme": self.run.scheme,
            "dx_um": um(grid.dx_um),
            "dt_s": um(grid.dt_s),
            "max_time_days": um(grid.max_time_days),
        }

    def set_scheme(self, scheme: str) -> bool:
        if scheme not in SCHEMES:
            self.errors["scheme"] = f"scheme: use one of {', '.join(SCHEMES)}"
            return False
        return self._commit("scheme", scheme=scheme)

    def set_grid(self, fields: dict[str, str], edited: str) -> bool:
        """dx (um), dt (s) and the longest time (days); a blank part is chosen at fit
        time. With dx and dt both given the grid is fixed, and a blank longest time is the
        table's default (`forster.io.RunDefaults`)."""
        try:
            dx, dt, longest = (
                number(fields[k], what, optional=True)
                for k, what in (
                    ("dx_um", "dx (µm)"),
                    ("dt_s", "dt (s)"),
                    ("max_time_days", "Longest time (days)"),
                )
            )
            for value, what in ((dx, "dx"), (dt, "dt"), (longest, "Longest time")):
                if value is not None and value <= 0:
                    raise ValueError(f"{what} must be positive")
        except ValueError as error:
            self.errors[edited] = str(error)
            return False
        if dx is not None and dt is not None:
            grid = Grid(dx, dt, longest or RunDefaults().max_time_days)
        else:
            grid = GridRequest(dx, dt, longest)
        return self._commit(edited, grid=grid)

    def grid_lines(self) -> list[str]:
        """What the cached fit's grid was and how well it resolved the fit
        (`forster.grid.grid_diagnostics`)."""
        result = self.project.result(self.name)
        if result is None:
            return ["Fit the run to see the grid it used."]
        grid: Grid = result.inputs["grid"]
        d = grid_diagnostics(result)
        lines = [
            f"Used: dx {grid.dx_um:.3g} µm ({len(result.model_x_um)} nodes), "
            f"dt {grid.dt_s:.4g} s, up to {grid.max_time_days:.4g} days",
            f"Diffusion length / dx = {d['grid_resolution']:.3g}; error of the time at most "
            f"about {d['grid_error_pct']:.2g}% (worst case measured)",
        ]
        if math.isfinite(d["grid_stability"]):
            lines.append(
                f"Stability ratio D·dt/dx² = {d['grid_stability']:.2g} (above 0.5 each step "
                "is split into stable substeps)"
            )
        return lines

    def set_stage_cell(self, row: int, column: str, text: str) -> bool:
        """A typed value in the growth-stage table (`STAGE_FIELDS`)."""
        key = f"stage:{row}:{column}"
        rows = stage_rows(self.run)
        title = f"stage {row + 1}, {STAGE_TITLES[column]}"
        try:
            if column in ("zone_fo", "rim_fo"):
                value = fraction(text, title, optional=True)
                if value is not None and not 0 < value <= 1:
                    raise ValueError(f"{title}: Fo is in mol%, above 0 and at most 100")
            else:
                value = number(text, title)
            rows[row] = replace(rows[row], **{column: value})
            stages = stages_of(rows)
        except ValueError as error:
            self.errors[key] = str(error)
            return False
        return self._commit(key, stages=stages)

    def add_stage(self) -> bool:
        """A new last stage a day after the one before, the rim where it was and its
        value that of edge 1 (edit from there)."""
        rows = stage_rows(self.run)
        days = rows[-1].days + 1 if rows else 1.0
        rim = rows[-1].rim_um if rows else self.run.initial.domain_um[0]
        held = self.run.rim_fo or self.run.initial.segment_fo[0]
        rows.append(StageRow(days, rim, None, held))
        return self._commit("stages", stages=stages_of(rows))

    def remove_stage(self, row: int) -> bool:
        rows = stage_rows(self.run)
        if not 0 <= row < len(rows):
            return False
        del rows[row]
        return self._commit("stages", stages=stages_of(rows))


# -- growth stages ----------------------------------------------------------------------

STAGE_FIELDS = ("days", "rim_um", "zone_fo", "rim_fo")
STAGE_TITLES = {
    "days": "Starts (days)",
    "rim_um": "Rim at (µm)",
    "zone_fo": "New zone Fo (mol%)",
    "rim_fo": "Rim Fo (mol%)",
}


@dataclass(frozen=True)
class StageRow:
    """One growth stage as the table shows it (the parameter table's stage_days,
    stage_rim_um, stage_zone_fo and stage_rim_fo); Fo as fractions, None when blank."""

    days: float
    rim_um: float
    zone_fo: float | None = None
    rim_fo: float | None = None


def stage_rows(run: Run) -> list[StageRow]:
    """The run's growth stages (fixed rim values only, as the table writes them)."""
    rows = []
    for s in run.stages:
        rim = s.rim_value
        if rim is not None and not isinstance(rim, int | float):
            raise ValueError("a rim value that changes in time has no table form")
        rows.append(
            StageRow(
                s.start_s / SECONDS_PER_DAY,
                s.rim_um,
                s.zone_value,
                None if rim is None else float(rim),
            )
        )
    return rows


def stages_of(rows: list[StageRow]) -> tuple[Stage, ...]:
    """Stages from the table (the run's edit checks them as the table reader does)."""
    return tuple(
        Stage(r.days * SECONDS_PER_DAY, r.rim_um, rim_value=r.rim_fo, zone_value=r.zone_fo)
        for r in rows
    )


def stage_text(row: StageRow) -> tuple[str, str, str, str]:
    return um(row.days), um(row.rim_um), pct(row.zone_fo), pct(row.rim_fo)


# -- copying settings between runs ------------------------------------------------------

# What "copy settings to other runs" offers: a group's title and the Run fields it copies.
COPY_GROUPS: dict[str, tuple[str, tuple[str, ...]]] = {
    "conditions": ("Conditions (T, P, fO2)", ("conditions",)),
    "orientation": ("Orientation", ("orientation",)),
    "equation": ("Diffusion equation", ("diffusivity",)),
    "uncertainties": (
        "Starting-profile uncertainties and sampler",
        ("uncertainties", "scan_method", "scan_seed", "scan_points", "scan_grid"),
    ),
    "mc": ("Monte Carlo settings", ("mc", "mc_samples", "mc_seed")),
    "grid": ("Grid and scheme", ("grid", "scheme")),
}


def copy_settings(project: Project, source: str, targets: list[str], groups: list[str]) -> None:
    """Copy the chosen groups of settings (`COPY_GROUPS`) from one run to others
    (`Project.copy_settings`: each target is checked, and none changes if one fails)."""
    unknown = [g for g in groups if g not in COPY_GROUPS]
    if unknown:
        raise ValueError(f"no settings group {unknown[0]!r}")
    if not groups or not targets:
        raise ValueError("choose at least one group of settings and one run")
    fields = [f for g in groups for f in COPY_GROUPS[g][1]]
    project.copy_settings(source, [t for t in targets if t != source], fields)


# -- views: time distributions and the law comparison --------------------------------------

TIME_VIEWS = {
    "fit": "The fit alone",
    "start": "With the starting profile",
    "all": "With conditions and law",
}


TimeView = tuple[FitResult, PosteriorOfTime | None, TimeDistribution | None]


def time_view(project: Project, name: str, mode: str) -> TimeView | str:
    """What `plot_time_distributions` draws for ``mode`` (`TIME_VIEWS`): (fit, starting-
    profile posterior or None, Monte Carlo or None), or why it cannot yet."""
    result = project.result(name)
    if result is None:
        return "Fit the run first."
    if result.uncertainty != "correlated":
        return "The distribution of time needs the correlated uncertainty method."
    posterior = mc = None
    if mode in ("start", "all"):
        scan = project.scan(name) if project.run(name).uncertainties else None
        if scan is not None and scan.days is not None and scan.density is not None:
            posterior = PosteriorOfTime(scan.days, scan.density)
        elif mode == "start":
            return (
                "Run 'Fit starting profile' (Uncertainties tab) with a sampler (not grid) "
                "to see the time with the starting profile's uncertainty."
            )
    if mode == "all":
        record = project.mc(name)
        if record is None:
            return "Run the Monte Carlo (Uncertainties tab) to see the time with the conditions."
        mc = record.distribution
    return result, posterior, mc


def app_refusal(run: Run) -> str | None:
    """Why the app will not fit ``run`` (None if it will). The app never krigs (Henry,
    2026-10-16): the legacy uncertainty method fits a kriged profile, so a table that
    asks for it is fitted by the command line, not silently with another method."""
    if run.uncertainty == "legacy":
        return (
            "uses the legacy uncertainty method, which needs kriging and is not "
            "available in the app; fit it with `forster fit`, or set its uncertainty "
            "to correlated"
        )
    return None


def law_tasks(project: Project, name: str) -> list[FitTask]:
    """The run fitted under its own equation and each other offered Fe-Mg equation, as
    tasks (`Project.variant_task`), the run's own first."""
    run = project.run(name)
    tasks = [project.variant_task(name)]
    for _, (preset, _label) in FE_MG_CHOICES.items():
        if preset != run.diffusivity.name:
            tasks.append(project.variant_task(name, diffusivity=resolve(preset)))
    return tasks


# -- runs and results across the project ------------------------------------------------


def new_run(project: Project, name: str, profile_name: str, like: str | None = None) -> Run:
    """A run on ``profile_name`` with the settings of run ``like`` (or defaults: 1200
    deg C, 100 MPa, QFM, the default equation), its starting profile spanning the
    measured points with one step in the middle and the end points' Fo. On another
    profile than ``like``'s, a longest model time left to be chosen is chosen afresh
    (another profile, another time), unless ``like`` fixes the whole grid."""
    initial = default_initial(project.profiles[profile_name])
    if like is not None:
        base = project.run(like)
        grid = base.grid
        if isinstance(grid, GridRequest) and base.profile_name != profile_name:
            grid = replace(grid, max_time_days=None)
        return replace(
            base,
            name=name,
            profile_name=profile_name,
            grid=grid,
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
    "mc_median",
    "mc_low",
    "mc_high",
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
    "mc_median": "MC median",
    "mc_low": "MC 16%",
    "mc_high": "MC 84%",
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
            mc = project.mc(run.name)
            if mc is not None:
                row |= {
                    "mc_median": mc.summary["mc_median_days"],
                    "mc_low": mc.summary["mc_low_days"],
                    "mc_high": mc.summary["mc_high_days"],
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
        mc = project.mc(run.name)
        if mc is not None:
            row |= mc.summary
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
    "COPY_GROUPS",
    "SCAN_METHODS",
    "TIME_VIEWS",
    "RunEditor",
    "SegmentRow",
    "StageRow",
    "copy_settings",
    "law_tasks",
    "new_run",
    "result_rows",
    "scan_lines",
    "scan_summary",
    "stage_rows",
    "stages_of",
    "summary_table",
    "time_view",
]
