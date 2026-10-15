"""The prototype's state and plot, without NiceGUI: one `forster.project.Project`.

The page (`gui_web.app`) and the self-test (`gui_web.selftest`) both drive this
module, so the self-test exercises the same code paths as the window. Validation,
files and the result cache all stay in `forster.project`; this module only turns
plot edits into `Project.update_run` calls and results into a Plotly figure.
"""

from __future__ import annotations

import copy
import os
import time
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Any

from forster import FitResult, InitialCondition, Uncertainty
from forster.io import scan_run, scan_summary
from forster.project import Project

# "Fit initial condition" scans these (σ of a normal prior around the current value).
SCAN_UNCERTAINTIES = {
    "step1_um": Uncertainty(2.0),
    "edge_fo": Uncertainty(0.002),
    "core_fo": Uncertainty(0.002),
}
# Measurement only (FORSTER_GUI_WIDE_SCAN=1): wide priors and nested sampling, ~4 s.
WIDE_SCAN = {
    "step1_um": Uncertainty(8.0),
    "edge_fo": Uncertainty(0.008),
    "core_fo": Uncertainty(0.008),
}

# Shape order in the figure: the drag handles.
STEP, EDGE, CORE = 0, 1, 2
MODEL, CONTEXT, DATA = "#1f5fa8", "#7a7a7a", "#c0392b"


def snapshot(project: Project) -> Project:
    """A copy whose runs stay fixed while a worker thread fits it.

    `Project.fit` computes the cache key after fitting, from the run as it is then; an
    edit made during the fit would file the old fit under the new settings. Fitting a
    snapshot keeps the key and the fit consistent. The cache folder is shared, so the
    snapshot's fit is found by the real project afterwards (`Session.adopt_fit`).
    """
    return replace(project, runs=list(project.runs), state=copy.deepcopy(project.state))


def fit_in_thread(snap: Project, name: str) -> tuple[FitResult, float]:
    """Fit (or load from the cache) in a worker thread; returns the result and seconds."""
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # grid advice: the command line reports it
        result = snap.fit(name)
    return result, time.perf_counter() - start


def scan_in_thread(snap: Project, name: str) -> tuple[Any, float]:
    """`forster.io.scan_run` in a worker thread; returns the posterior and seconds."""
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scan = scan_run(snap.run(name), snap.profiles)
    return scan, time.perf_counter() - start


class Session:
    """The open project and the selected run."""

    def __init__(self) -> None:
        self.project: Project | None = None
        self.selected: str | None = None

    # -- project ---------------------------------------------------------------------

    def open(self, folder: str | Path) -> Project:
        self._adopt(Project.open(Path(folder).expanduser()))
        assert self.project is not None
        return self.project

    def new(self, folder: str | Path) -> Project:
        self._adopt(Project.new(Path(folder).expanduser()))
        assert self.project is not None
        return self.project

    def _adopt(self, project: Project) -> None:
        self.project = project
        names = project.run_names
        chosen = project.state.get("selected_run")
        self.selected = chosen if chosen in names else (names[0] if names else None)

    def save(self) -> None:
        self._need().save()

    def close(self) -> None:
        self.project, self.selected = None, None

    def _need(self) -> Project:
        if self.project is None:
            raise RuntimeError("no project is open")
        return self.project

    def select(self, name: str) -> None:
        project = self._need()
        project.run(name)  # KeyError for an unknown run
        self.selected = name
        project.state["selected_run"] = name

    # -- the starting profile -------------------------------------------------------

    @property
    def initial(self) -> InitialCondition:
        assert self.selected is not None
        return self._need().run(self.selected).initial

    def set_initial(
        self, *, step: float | None = None, edge: float | None = None, core: float | None = None
    ) -> InitialCondition:
        """Move the step or the edge or core Fo. Raises ValueError (from
        `InitialCondition` or `Project.update_run`) and changes nothing if refused."""
        old = self.initial
        if len(old.steps_um) != 1:
            raise ValueError("this prototype edits one-step starting profiles only")
        new = InitialCondition(
            old.domain_um,
            old.edge_fo if edge is None else round(float(edge), 5),
            old.core_fo if core is None else round(float(core), 5),
            (old.steps_um[0] if step is None else round(float(step), 3),),
        )
        assert self.selected is not None
        return self._need().update_run(self.selected, initial=new).initial

    def edit_from_relayout(self, event: dict[str, Any]) -> dict[str, float]:
        """The handle moves in a Plotly relayout event: {"step"|"edge"|"core": value}.

        A dragged line reports both ends; an end that was dragged alone (resizing the
        line) is taken as the new position."""
        old = self.initial
        edits: dict[str, float] = {}
        for key, index, axis, value in (
            ("step", STEP, "x", old.steps_um[0]),
            ("edge", EDGE, "y", old.edge_fo),
            ("core", CORE, "y", old.core_fo),
        ):
            a = event.get(f"shapes[{index}].{axis}0")
            b = event.get(f"shapes[{index}].{axis}1")
            if a is None and b is None:
                continue
            ends = [float(v) for v in (a, b) if v is not None]
            moved = max(ends, key=lambda v: abs(v - value))
            if abs(moved - value) > 1e-9:
                edits[key] = moved
        return edits

    # -- scan --------------------------------------------------------------------------

    def prepare_scan(self) -> None:
        """Give the run the scan's uncertainties (through `update_run`)."""
        assert self.selected is not None
        wide = os.environ.get("FORSTER_GUI_WIDE_SCAN") == "1"
        changes: dict[str, Any] = {"uncertainties": WIDE_SCAN if wide else SCAN_UNCERTAINTIES}
        if wide:
            changes["scan_method"] = "nested"
        self._need().update_run(self.selected, **changes)

    def apply_scan(self, scan: Any) -> tuple[dict[str, Any], InitialCondition]:
        """Move the starting profile to the scan's best values; the scan's summary."""
        best = scan.best_values
        initial = self.set_initial(
            step=best.get("step1_um"), edge=best.get("edge_fo"), core=best.get("core_fo")
        )
        return scan_summary(scan), initial

    # -- results -----------------------------------------------------------------------

    def adopt_fit(self, name: str, snap: Project) -> FitResult | None:
        """After a snapshot's fit: the fit for the run's current settings, recorded in the
        project (a cache hit), or None if the run changed while the snapshot was fitted."""
        project = self.project
        if project is None or name not in project.run_names:
            return None
        if project.result_key(name) != snap.result_key(name):
            return None
        return project.fit(name)  # from the cache the snapshot just wrote


def time_text(result: FitResult) -> str:
    low, high = result.time_range_days()
    return f"{result.time_days:.2f} days (1σ {low:.2f}–{high:.2f})"


def figure(
    project: Project, name: str, result: FitResult | None, *, seq: int, fresh: bool
) -> dict[str, Any]:
    """The Plotly figure: measured points, the starting profile with its three drag
    handles (step position, edge Fo, core Fo), and the fit with its 1σ band.

    ``fresh`` says the fit belongs to the run's current settings (the client-side latency
    probe waits for a fresh figure); ``seq`` numbers the figures sent."""
    run = project.run(name)
    profile = project.profiles[run.profile_name]
    ini = run.initial
    (x0, x1), step = ini.domain_um, ini.steps_um[0]
    data: list[dict[str, Any]] = []
    if result is not None:
        _, low, high = result.indices
        x = result.model_x_um.tolist()
        lo, hi = result.time_range_days()
        data += [
            {
                "x": x,
                "y": result.curves[low].tolist(),
                "mode": "lines",
                "line": {"width": 0},
                "hoverinfo": "skip",
                "showlegend": False,
            },
            {
                "x": x,
                "y": result.curves[high].tolist(),
                "mode": "lines",
                "line": {"width": 0},
                "fill": "tonexty",
                "fillcolor": "rgba(31,95,168,0.22)",
                "name": f"1σ range, {lo:.2f}–{hi:.2f} d",
                "hoverinfo": "skip",
            },
            {
                "x": x,
                "y": result.best_fit.tolist(),
                "mode": "lines",
                "line": {"color": MODEL, "width": 2.5},
                "name": f"Best fit, {result.time_days:.2f} d",
            },
        ]
    data += [
        {
            "x": [x0, step, step, x1],
            "y": [ini.edge_fo, ini.edge_fo, ini.core_fo, ini.core_fo],
            "mode": "lines",
            "line": {"color": CONTEXT, "dash": "dash", "width": 1.2},
            "name": "Starting profile",
            "hoverinfo": "skip",
        },
        {
            "x": profile.distance_um.tolist(),
            "y": profile.fo.tolist(),
            "mode": "markers",
            "marker": {"color": "white", "size": 7, "line": {"color": DATA, "width": 1.5}},
            "name": "Measured",
        },
    ]
    handle = {"color": "#e67e22", "width": 3}
    shapes = [
        {"type": "line", "xref": "x", "yref": "paper", "x0": step, "x1": step, "y0": 0, "y1": 1},
        {"type": "line", "x0": x0, "x1": step, "y0": ini.edge_fo, "y1": ini.edge_fo},
        {"type": "line", "x0": step, "x1": x1, "y0": ini.core_fo, "y1": ini.core_fo},
    ]
    for shape in shapes:
        shape["line"] = handle
    fos = [*profile.fo.tolist(), ini.edge_fo, ini.core_fo]
    pad = 0.15 * (max(fos) - min(fos)) or 0.005
    return {
        "data": data,
        "layout": {
            "shapes": shapes,
            "meta": {"seq": seq, "fresh": fresh},
            "uirevision": name,  # keep the user's zoom across updates of one run
            "margin": {"l": 60, "r": 20, "t": 30, "b": 50},
            "xaxis": {"title": {"text": "Distance from the crystal edge (µm)"}},
            "yaxis": {
                "title": {"text": "Fo (fraction)"},
                "range": [min(fos) - pad, max(fos) + pad],
            },
            "legend": {"x": 1, "xanchor": "right", "y": 1, "bgcolor": "rgba(255,255,255,0.7)"},
            "title": {"text": name, "font": {"size": 14}},
            "template": "plotly_white",
        },
        "config": {"edits": {"shapePosition": True}, "displaylogo": False, "responsive": True},
    }
