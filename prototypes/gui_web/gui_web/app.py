"""The NiceGUI page: run list, the plot with drag handles, number fields, fit and scan.

One `Session` (one open project) serves the app; in a native window there is one client.
Fits and scans run in worker threads (`nicegui.run.io_bound`); the Numba kernels
release the GIL, so the event loop keeps serving the page meanwhile.

Instrumentation (always on, cheap): the status line shows the server-side drag-to-redraw
time. With FORSTER_GUI_METRICS=<file.jsonl> the app also appends JSON records: "drag"
(server side), "client_drag" (mouse release to the redrawn fit, measured in the page),
"scan" (event-loop and client lateness during a scan) and "ready" (cold start).
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

from nicegui import app, background_tasks, run, ui

from gui_web.session import Session, figure, fit_in_thread, scan_in_thread, snapshot, time_text

SESSION = Session()
T0 = float(os.environ.get("FORSTER_GUI_T0") or time.time())  # launch time, for cold start
QUIT_AFTER: float | None = None  # set by main: quit this many seconds after the page is up


def record(kind: str, **values: Any) -> None:
    path = os.environ.get("FORSTER_GUI_METRICS")
    if path:
        with open(path, "a") as f:
            f.write(json.dumps({"kind": kind, "t": time.time(), **values}) + "\n")


# Client-side probes, installed once per page. Drag latency: a shape drag's relayout
# stores the release time and the figure number shown; the next "fresh" figure (a fit
# of the current settings) drawn after it reports the elapsed time. Responsiveness: a
# 50 ms interval records its own lateness, and a ping every 100 ms measures the round
# trip through the server.
PROBES = """
window.__fg = {late: [], rtt: [], drag: null};
(() => {
  let next = performance.now() + 50;
  setInterval(() => {
    const now = performance.now();
    window.__fg.late.push(now - next);
    next = now + 50;
  }, 50);
  setInterval(() => emitEvent('fg_ping', performance.now()), 100);
})();
window.__fgPong = (t) => window.__fg.rtt.push(performance.now() - t);
"""
RELAYOUT_JS = """(e) => {
  const shapes = Object.keys(e).filter(k => k.startsWith('shapes['));
  if (!shapes.length) return;
  const gd = getHtmlElement(%d);
  window.__fg.drag = {t0: performance.now(), seq: gd.layout.meta?.seq ?? -1};
  emit(e);
}"""
AFTERPLOT_JS = """() => {
  const d = window.__fg.drag;
  const meta = getHtmlElement(%d).layout.meta || {};
  if (d && meta.fresh && meta.seq > d.seq) {
    window.__fg.drag = null;
    const ms = performance.now() - d.t0;
    (window.__fg.dragMs ||= []).push(ms);
    emit({ms: ms, seq: meta.seq});
  }
}"""


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0}
    ordered = sorted(values)
    return {
        "n": len(values),
        "median_ms": round(statistics.median(ordered), 2),
        "p99_ms": round(ordered[min(len(ordered) - 1, int(0.99 * len(ordered)))], 2),
        "max_ms": round(ordered[-1], 2),
    }


class Heartbeat:
    """An asyncio timer that records how late it wakes: the event loop's lag."""

    def __init__(self, period_s: float = 0.005) -> None:
        self.period, self.samples, self.task = period_s, [], None

    async def _beat(self) -> None:
        while True:
            due = time.perf_counter() + self.period
            await asyncio.sleep(self.period)
            self.samples.append((time.perf_counter() - due) * 1000)

    def start(self) -> None:
        self.samples = []
        self.task = asyncio.create_task(self._beat())

    def stop(self) -> list[float]:
        if self.task:
            self.task.cancel()
        return self.samples


class Page:
    """The widgets of one client and the workers that update them."""

    def __init__(self) -> None:
        self.seq = 0
        self.fitting = False
        self.refit_again = False
        self.drag_started: float | None = None  # perf_counter at the relayout event
        self.scan_generation = 0
        self.native = app.native.main_window is not None

    # -- layout ------------------------------------------------------------------------

    def build(self) -> None:
        ui.add_head_html("<style>body{background:#f6f7f9}</style>")
        with ui.header().classes("items-center bg-white text-black shadow-sm q-py-xs"):
            ui.label("FORSTER").classes("text-h6 text-weight-bold")
            ui.label("NiceGUI prototype").classes("text-caption text-grey-7")
            ui.space()
            self.folder = ui.input("Project folder", value=self._default_folder()).props(
                "dense outlined clearable"
            )
            self.folder.classes("w-96")
            ui.button("Open", icon="folder_open", on_click=self.open).props("flat dense")
            ui.button("New from example", icon="add", on_click=self.new).props("flat dense")
            self.save_button = ui.button("Save", icon="save", on_click=self.save)
            self.save_button.props("flat dense")
            ui.button("Close", icon="close", on_click=self.close).props("flat dense")
        with ui.row().classes("w-full no-wrap items-stretch gap-3 q-pa-sm"):
            with ui.card().classes("w-56 q-pa-sm"):
                ui.label("Runs").classes("text-subtitle2")
                self.runs = ui.column().classes("w-full gap-0")
            with ui.card().classes("col q-pa-xs"):
                self.plot = ui.plotly({"data": [], "layout": {}})
                self.plot.classes("w-full h-[calc(100vh-190px)] min-h-[420px]")
                self.status = ui.label("Open a project folder, or start one from the example.")
                self.status.classes("text-caption text-grey-8 q-px-sm")
            with ui.card().classes("w-72 q-pa-sm gap-2"):
                ui.label("Starting profile").classes("text-subtitle2")
                self.step = self._field("Step position (µm)", "%.2f", 0.5)
                self.edge = self._field("Edge Fo (fraction)", "%.4f", 0.0005)
                self.core = self._field("Core Fo (fraction)", "%.4f", 0.0005)
                ui.separator()
                ui.label("Diffusion time").classes("text-subtitle2")
                self.time = ui.label("–").classes("text-body1 text-weight-medium")
                self.time_note = ui.label("").classes("text-caption text-grey-7")
                ui.separator()
                ui.label("Starting-profile uncertainty").classes("text-subtitle2")
                ui.label(
                    "Fits the step (σ 2 µm prior) and the edge and core Fo (σ 0.002) with "
                    "the time, and moves the lines to the best values."
                ).classes("text-caption text-grey-7")
                with ui.row().classes("items-center gap-2"):
                    self.scan_button = ui.button("Fit initial condition", on_click=self.scan)
                    self.spinner = ui.spinner("dots", size="lg", color="primary")
                    self.cancel_button = ui.button("Cancel", on_click=self.cancel_scan)
                    self.cancel_button.props("flat color=negative")
                self.spinner.set_visibility(False)
                self.cancel_button.set_visibility(False)
                self.scan_text = ui.label("").classes("text-caption whitespace-pre-line")
        self.plot.on("plotly_relayout", self.on_relayout, js_handler=RELAYOUT_JS % self.plot.id)
        self.plot.on(
            "plotly_afterplot", self.on_client_redraw, js_handler=AFTERPLOT_JS % self.plot.id
        )
        ui.on("fg_ping", lambda e: ui.run_javascript(f"window.__fgPong({e.args})"))
        ui.run_javascript(PROBES)
        self.show_project()

    def _field(self, label: str, fmt: str, step: float) -> ui.number:
        field = ui.number(label, format=fmt, step=step).props("dense outlined")
        field.classes("w-full")
        field.on("keydown.enter", lambda: self.commit_field(field))
        field.on("blur", lambda: self.commit_field(field))
        return field

    @staticmethod
    def _default_folder() -> str:
        if SESSION.project is not None:
            return str(SESSION.project.folder)
        return str(Path.home() / "FORSTER projects" / "example")

    # -- project actions -----------------------------------------------------------------

    async def _pick_folder(self) -> str | None:
        """A native folder dialog in a native window; the path field otherwise."""
        if self.native:
            import webview

            chosen = await app.native.main_window.create_file_dialog(webview.FileDialog.FOLDER)
            if chosen:
                self.folder.value = chosen[0]
        return self.folder.value or None

    async def open(self) -> None:
        folder = await self._pick_folder()
        if folder is None:
            return
        try:
            SESSION.open(folder)
        except (OSError, ValueError) as error:
            ui.notify(str(error), type="negative", multi_line=True)
            return
        self.show_project()

    async def new(self) -> None:
        folder = self.folder.value
        if not folder:
            ui.notify("Type a new (or empty) folder in the path field", type="warning")
            return
        try:
            SESSION.new(folder)
        except (OSError, ValueError) as error:
            ui.notify(str(error), type="negative", multi_line=True)
            return
        self.show_project()

    def save(self) -> None:
        if SESSION.project is None:
            return
        SESSION.save()
        ui.notify(f"Saved {SESSION.project.folder}", type="positive")
        self._title()

    def close(self) -> None:
        SESSION.close()
        self.scan_generation += 1
        self.show_project()

    def _title(self) -> None:
        dirty = SESSION.project is not None and SESSION.project.dirty
        self.save_button.text = "Save •" if dirty else "Save"

    # -- showing a run -----------------------------------------------------------------

    def show_project(self) -> None:
        self.runs.clear()
        project = SESSION.project
        if project is None:
            self.plot.update_figure({"data": [], "layout": {"template": "plotly_white"}})
            self.time.text, self.time_note.text, self.scan_text.text = "–", "", ""
            self.status.text = "Open a project folder, or start one from the example."
            return
        with self.runs:
            for name in project.run_names:
                item = ui.item(name, on_click=lambda n=name: self.select(n)).classes("rounded")
                if name == SESSION.selected:
                    item.classes("bg-blue-1 text-weight-medium")
        self._title()
        self.status.text = (
            "Drag the orange lines (step position, edge Fo, core Fo); the fit redraws on release."
        )
        if SESSION.selected is not None:
            self.show_run(cached_only=False)

    def select(self, name: str) -> None:
        SESSION.select(name)
        self.show_project()

    def show_run(self, *, cached_only: bool) -> None:
        """Draw the selected run with its cached fit if there is one, else refit."""
        project, name = SESSION.project, SESSION.selected
        assert project is not None and name is not None
        ini = SESSION.initial
        self.step.value, self.edge.value, self.core.value = (
            ini.steps_um[0],
            ini.edge_fo,
            ini.core_fo,
        )
        for field in (self.step, self.edge, self.core):
            self._clear_error(field)
        cached = project.result(name)
        self.draw(cached)
        if cached is not None:
            self.time_note.text = "from the result cache (not refitted)"
        elif not cached_only:
            self.time.text = "fitting…"
            self.time_note.text = "stale: settings changed" if project.is_stale(name) else ""
            background_tasks.create(self.refit())

    def draw(self, result: Any) -> None:
        project, name = SESSION.project, SESSION.selected
        assert project is not None and name is not None
        fresh = result is not None  # every caller passes the fit of the current settings
        self.seq += 1
        self.plot.update_figure(figure(project, name, result, seq=self.seq, fresh=fresh))
        if result is not None:
            self.time.text = time_text(result)

    # -- edits ---------------------------------------------------------------------------

    def _clear_error(self, field: ui.number) -> None:
        field.props.pop("error", None)
        field.props.pop("error-message", None)

    def _apply(self, field: ui.number | None, **edit: float) -> bool:
        try:
            SESSION.set_initial(**edit)
        except ValueError as error:
            if field is not None:
                field.props["error"] = True
                field.props["error-message"] = str(error)
            else:
                ui.notify(str(error), type="negative", multi_line=True)
            project, name = SESSION.project, SESSION.selected
            assert project is not None and name is not None
            self.draw(project.result(name))  # put the lines back; the field keeps its text
            return False
        if field is not None:
            self._clear_error(field)
        return True

    async def commit_field(self, field: ui.number) -> None:
        if SESSION.project is None or field.value is None:
            return
        key = {id(self.step): "step", id(self.edge): "edge", id(self.core): "core"}[id(field)]
        ini = SESSION.initial
        current = {"step": ini.steps_um[0], "edge": ini.edge_fo, "core": ini.core_fo}[key]
        if abs(float(field.value) - current) < 1e-12:
            return
        if self._apply(field, **{key: float(field.value)}):
            self.drag_started = time.perf_counter()
            self.after_edit()

    async def on_relayout(self, event: Any) -> None:
        if SESSION.project is None or SESSION.selected is None:
            return
        self.drag_started = time.perf_counter()
        edits = SESSION.edit_from_relayout(event.args)
        if not edits:
            self.show_run(cached_only=True)  # a handle slid along itself: redraw it in place
            return
        if self._apply(None, **edits):
            self.after_edit()

    def after_edit(self) -> None:
        ini = SESSION.initial
        self.step.value, self.edge.value, self.core.value = (
            ini.steps_um[0],
            ini.edge_fo,
            ini.core_fo,
        )
        self._title()
        project, name = SESSION.project, SESSION.selected
        assert project is not None and name is not None
        cached = project.result(name)
        if cached is not None:  # e.g. dragged back to settings fitted before
            self.draw(cached)
            self._report_drag(0.0, cached=True)
            return
        self.draw(None)  # the lines where they were dropped; the old fit is gone
        self.time.text, self.time_note.text = "fitting…", ""
        background_tasks.create(self.refit())

    def on_client_redraw(self, event: Any) -> None:
        record("client_drag", ms=event.args["ms"], seq=event.args["seq"])
        self.status.text += f"; client {event.args['ms']:.0f} ms"

    def _report_drag(self, fit_s: float, *, cached: bool) -> None:
        if self.drag_started is None:
            return
        total = (time.perf_counter() - self.drag_started) * 1000
        self.drag_started = None
        source = "cache" if cached else f"fit {fit_s * 1000:.0f} ms"
        self.status.text = f"Redrawn {total:.0f} ms after release ({source})"
        record("drag", server_ms=round(total, 2), fit_ms=round(fit_s * 1000, 2), cached=cached)

    # -- workers ---------------------------------------------------------------------------

    async def refit(self) -> None:
        """Fit the selected run in a worker thread; a request arriving during a fit is
        served once that fit ends (only the latest settings are fitted)."""
        if self.fitting:
            self.refit_again = True
            return
        self.fitting = True
        try:
            while True:
                self.refit_again = False
                project, name = SESSION.project, SESSION.selected
                if project is None or name is None:
                    return
                snap = snapshot(project)
                done = await run.io_bound(fit_in_thread, snap, name)
                if done is None or SESSION.project is not project:
                    return
                _, seconds = done
                result = SESSION.adopt_fit(name, snap)
                if result is not None and name == SESSION.selected and not self.refit_again:
                    self.draw(result)
                    self.time_note.text = f"fitted in {seconds * 1000:.0f} ms"
                    self._report_drag(seconds, cached=False)
                if not self.refit_again:
                    return
        except Exception as error:  # show it; the next edit retries
            self.time.text = "fit failed"
            self.time_note.text = str(error)
        finally:
            self.fitting = False

    async def scan(self) -> None:
        project, name = SESSION.project, SESSION.selected
        if project is None or name is None:
            return
        try:
            SESSION.prepare_scan()
        except ValueError as error:
            ui.notify(str(error), type="negative", multi_line=True)
            return
        self._title()  # the run changed (its uncertainties)
        self.scan_generation += 1
        generation = self.scan_generation
        self._busy(True)
        self.scan_text.text = "Fitting the starting profile…"
        beat = Heartbeat()
        beat.start()
        measuring = bool(os.environ.get("FORSTER_GUI_METRICS"))
        if measuring:
            ui.run_javascript("window.__fg.late = []; window.__fg.rtt = [];")
        try:
            done = await run.io_bound(scan_in_thread, snapshot(project), name)
        except Exception as error:
            done = None
            if generation == self.scan_generation:
                self.scan_text.text = f"Scan failed: {error}"
        late = beat.stop()
        if generation != self.scan_generation or done is None:
            return  # cancelled (or the project closed): the result is dropped
        self._busy(False)
        scan, seconds = done
        if measuring:
            client = await ui.run_javascript("({late: window.__fg.late, rtt: window.__fg.rtt})")
            record(
                "scan",
                seconds=round(seconds, 3),
                method=scan.method if hasattr(scan, "method") else "grid",
                loop_lateness=_stats(late),
                client_timer_lateness=_stats(client["late"]),
                client_ping_rtt=_stats(client["rtt"]),
            )
        try:
            summary, _ = SESSION.apply_scan(scan)
        except ValueError as error:
            self.scan_text.text = f"Best values refused: {error}"
            return
        self.scan_text.text = (
            f"{summary['scan_sampler']} in {seconds:.1f} s: "
            f"best {summary['scan_best_days']:.2f} d, "
            f"68% range {summary['scan_low_days']:.2f}–{summary['scan_high_days']:.2f} d\n"
            f"best: {summary['scan_best']}\n"
            f"event loop late by ≤ {max(late, default=0):.0f} ms during the scan"
        )
        self.show_project()

    def cancel_scan(self) -> None:
        self.scan_generation += 1  # the worker finishes in the background; its result is dropped
        self._busy(False)
        self.scan_text.text = "Cancelled (the result will be discarded when it arrives)."

    def _busy(self, busy: bool) -> None:
        self.spinner.set_visibility(busy)
        self.cancel_button.set_visibility(busy)
        self.scan_button.set_enabled(not busy)


@ui.page("/")
async def index() -> None:
    page = Page()
    page.build()
    await ui.context.client.connected()
    ready = time.time() - T0
    record("ready", seconds=round(ready, 3), native=page.native)
    print(f"ready {ready:.3f} s after launch", flush=True)
    if QUIT_AFTER is not None:
        await asyncio.sleep(QUIT_AFTER)  # let the window paint (and fit) before closing
        app.shutdown()
