"""The main window: run list, profile plot with draggable starting profile, number
fields, refit on release, and the initial-condition posterior ("Fit initial condition").

All state lives in `forster.project.Project`; this module only shows it and forwards
edits (``update_run``), so the GUI accepts exactly what the command line does.
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from PySide6.QtCore import QThreadPool
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from forster.io import scan_run, scan_summary
from forster.model import FitResult, InitialCondition
from forster.project import Project
from forster.scan import Uncertainty

from .plot import HANDLES, ProfileCanvas
from .workers import Job

# Set on a run before its first "Fit initial condition" if it has none: step 2 µm,
# edge and core Fo 0.002 (1 sigma, Gaussian).
SCAN_PRIORS = {
    "step1_um": Uncertainty(2.0),
    "edge_fo": Uncertainty(0.002),
    "core_fo": Uncertainty(0.002),
}
FIELD_LABELS = {
    "step": "Step position (µm)",
    "edge": "Edge Fo (mol%)",
    "core": "Core Fo (mol%)",
}
SIGMA = "\N{GREEK SMALL LETTER SIGMA}"


class ErrorLabel(QLabel):
    """A refusal shown under its field; hidden while there is none."""

    def __init__(self) -> None:
        super().__init__()
        self.setWordWrap(True)
        self.setStyleSheet("color: #B00020;")
        self.setVisible(False)

    def setText(self, text: str) -> None:
        super().setText(text)
        self.setVisible(bool(text))

    def clear(self) -> None:
        self.setText("")


def snapshot(project: Project) -> Project:
    """A copy a worker can fit while the GUI keeps editing the original: it shares the
    folder (and so the result cache) but not the run list or the session state."""
    return Project(
        project.folder,
        dict(project.profiles),
        list(project.runs),
        dict(project.diffusivities),
        {},
    )


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.project: Project | None = None
        self.current: str | None = None
        self.ask = True  # False in the self-test: no dialogs
        self._serial = 0  # bumped per opened project: drops answers for a closed one
        # Fits run one at a time (the cache folder is written without locking); the
        # newest request waits in _pending and older ones are dropped.
        self.fit_pool = QThreadPool(self)
        self.fit_pool.setMaxThreadCount(1)
        self.scan_pool = QThreadPool(self)
        self._jobs: set[Job] = set()  # keeps each job's signals alive until it answers
        self._fit_running = False
        self._pending: tuple[str, float | None] | None = None
        self._scan_token: object | None = None
        self._scan_best: tuple[str, InitialCondition] | None = None
        # Instrumentation read by the self-test and the measurements.
        self.fits_started = 0
        self.scans_dropped = 0
        self.latencies: list[dict[str, float]] = []
        self.last_error = ""
        self._build()
        self._show_project()

    # -- layout ---------------------------------------------------------------------

    def _build(self) -> None:
        self.resize(1280, 760)
        bar = self.addToolBar("Project")
        bar.setMovable(False)
        for text, slot, key in (
            ("Open folder…", self._choose_folder, QKeySequence.StandardKey.Open),
            ("New from example…", self._choose_new_folder, QKeySequence.StandardKey.New),
            ("Save", self.save, QKeySequence.StandardKey.Save),
            ("Close project", self.close_project, QKeySequence.StandardKey.Close),
        ):
            action = QAction(text, self)
            action.setShortcut(key)
            action.triggered.connect(slot)
            bar.addAction(action)

        self.run_list = QListWidget()
        self.run_list.currentItemChanged.connect(self._run_selected)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(QLabel("<b>Runs</b>"))
        left_layout.addWidget(self.run_list)

        self.canvas = ProfileCanvas()
        self.canvas.moved.connect(self._handle_moved)
        self.canvas.released.connect(self._commit)
        toolbar = NavigationToolbar2QT(self.canvas, self)
        self.canvas.toolbar = toolbar
        centre = QWidget()
        centre_layout = QVBoxLayout(centre)
        centre_layout.addWidget(toolbar)
        centre_layout.addWidget(self.canvas, 1)

        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(centre)
        splitter.addWidget(self._build_side())
        splitter.setSizes([210, 760, 310])
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)
        self.statusBar().showMessage("Ready")

    def _build_side(self) -> QWidget:
        side = QWidget()
        layout = QVBoxLayout(side)

        start = QGroupBox("Starting profile")
        form = QFormLayout(start)
        form.addRow(QLabel("Drag the orange lines on the plot, or type a value."))
        self.inputs: dict[str, QLineEdit] = {}
        self.errors: dict[str, ErrorLabel] = {}
        for handle in HANDLES:
            edit = QLineEdit()
            edit.editingFinished.connect(lambda h=handle: self._typed(h))
            error = ErrorLabel()
            self.inputs[handle], self.errors[handle] = edit, error
            form.addRow(FIELD_LABELS[handle], edit)
            form.addRow(error)
        layout.addWidget(start)

        fit = QGroupBox("Best fit")
        fit_layout = QVBoxLayout(fit)
        self.fit_label = QLabel("-")
        self.fit_label.setWordWrap(True)
        self.timing_label = QLabel()
        self.timing_label.setStyleSheet("color: gray;")
        fit_layout.addWidget(self.fit_label)
        fit_layout.addWidget(self.timing_label)
        layout.addWidget(fit)

        scan = QGroupBox("Starting-profile uncertainty")
        scan_layout = QVBoxLayout(scan)
        self.scan_button = QPushButton("Fit initial condition")
        self.scan_button.setToolTip(
            "Fits the step position and the edge and core Fo with the time, and samples "
            "their posterior (forster.io.scan_run). Uses the run's uncertainties; a run "
            f"without any gets step {SIGMA} 2 µm and edge and core {SIGMA} 0.002."
        )
        self.scan_button.clicked.connect(self.start_scan)
        self.busy = QProgressBar()
        self.busy.setRange(0, 0)  # indeterminate: scan_run reports no progress here
        self.busy.setVisible(False)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_scan)
        self.cancel_button.setVisible(False)
        self.scan_label = QLabel()
        self.scan_label.setWordWrap(True)
        self.apply_button = QPushButton("Use the best starting profile")
        self.apply_button.clicked.connect(self.apply_scan)
        self.apply_button.setEnabled(False)
        row = QHBoxLayout()
        row.addWidget(self.busy, 1)
        row.addWidget(self.cancel_button)
        for widget in (self.scan_button, self.scan_label, self.apply_button):
            scan_layout.addWidget(widget)
        scan_layout.insertLayout(1, row)
        layout.addWidget(scan)
        layout.addStretch(1)
        return side

    # -- project folder -------------------------------------------------------------

    def _choose_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Open a FORSTER project folder")
        if folder:
            self.open_project(folder)

    def _choose_new_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Choose an empty folder for the new project (the worked example)"
        )
        if folder:
            self.new_project(folder)

    def new_project(self, folder: str | Path) -> bool:
        if not self._may_discard():
            return False
        try:
            project = Project.new(folder)
        except (FileExistsError, OSError, ValueError) as exc:
            return self._refuse("Cannot start a project there", exc)
        self._set_project(project)
        return True

    def open_project(self, folder: str | Path) -> bool:
        if not self._may_discard():
            return False
        try:
            project = Project.open(folder)
        except (FileNotFoundError, OSError, ValueError) as exc:
            return self._refuse("Cannot open that folder", exc)
        self._set_project(project)
        return True

    def save(self) -> None:
        if self.project is None:
            return
        try:
            self.project.save()
        except OSError as exc:
            self._refuse("Save failed", exc)
            return
        self._update_title()
        self.statusBar().showMessage(f"Saved {self.project.folder}", 5000)

    def close_project(self) -> bool:
        if not self._may_discard():
            return False
        self._set_project(None)
        return True

    def _set_project(self, project: Project | None) -> None:
        self._serial += 1
        self.project, self.current = project, None
        self._pending, self._scan_token, self._scan_best = None, None, None
        self._scan_idle("")
        self._show_project()

    def _show_project(self) -> None:
        self.run_list.blockSignals(True)
        self.run_list.clear()
        self.run_list.blockSignals(False)
        self._update_title()
        if self.project is None:
            self.canvas.show_empty("Open a project folder, or start one from the example.")
            self._fill_inputs(None)
            self.fit_label.setText("-")
            self.timing_label.clear()
            self.scan_button.setEnabled(False)
            return
        for name in self.project.run_names:
            self.run_list.addItem(QListWidgetItem(name))
        self._label_runs()
        names = self.project.run_names
        selected = self.project.state.get("selected_run")
        if names:
            self.run_list.setCurrentRow(names.index(selected) if selected in names else 0)
        else:
            self.canvas.show_empty("This project has no runs.")

    def _may_discard(self) -> bool:
        """True if there is nothing unsaved, or the user saves or discards it."""
        if self.project is None or not self.project.dirty or not self.ask:
            return True
        answer = QMessageBox.question(
            self,
            "Unsaved changes",
            f"Save the changes to {self.project.folder.name}?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Save:
            self.save()
            return not self.project.dirty
        return answer == QMessageBox.StandardButton.Discard

    def _refuse(self, title: str, exc: Exception) -> bool:
        self.last_error = str(exc)
        if self.ask:
            QMessageBox.warning(self, title, str(exc))
        self.statusBar().showMessage(f"{title}: {exc}")
        return False

    def _update_title(self) -> None:
        if self.project is None:
            self.setWindowTitle("FORSTER (Qt prototype)")
            return
        mark = " *" if self.project.dirty else ""
        self.setWindowTitle(f"FORSTER: {self.project.folder.name}{mark}")

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._may_discard():
            event.accept()
        else:
            event.ignore()

    # -- runs -----------------------------------------------------------------------

    def _label_runs(self) -> None:
        """Mark each run in the list: fitted (cached), stale, or nothing yet."""
        assert self.project is not None
        for row, name in enumerate(self.project.run_names):
            if self.project.summary(name) is not None:
                tag = "  ✓"
            elif self.project.is_stale(name):
                tag = "  (stale)"
            else:
                tag = ""
            item = self.run_list.item(row)
            if item is not None:
                item.setText(name + tag)

    def _run_selected(self, item: QListWidgetItem | None, _previous: Any = None) -> None:
        if self.project is None or item is None:
            return
        name = self.project.run_names[self.run_list.row(item)]
        self.current = name
        if self.project.state.get("selected_run") != name:
            self.project.state["selected_run"] = name
        for error in self.errors.values():
            error.clear()
        run = self.project.run(name)
        self._fill_inputs(run.initial)
        self._scan_best = None
        self.apply_button.setEnabled(False)
        self.scan_button.setEnabled(self._scan_token is None)
        result = self.project.result(name)
        if result is not None:
            self._show_result(name, result)
        else:
            self._draw(name, None, "fitting…")
            self._request_fit(name)

    def _draggable(self, initial: InitialCondition) -> bool:
        return len(initial.steps_um) == 1

    def _draw(self, name: str, result: FitResult | None, note: str = "") -> None:
        assert self.project is not None
        run = self.project.run(name)
        initial = run.initial if self._draggable(run.initial) else None
        self.canvas.show_run(name, self.project.profiles[run.profile_name], initial, result, note)

    # -- editing the starting profile -----------------------------------------------

    def _fill_inputs(self, initial: InitialCondition | None) -> None:
        values = {}
        if initial is not None and self._draggable(initial):
            values = {
                "step": f"{initial.steps_um[0]:g}",
                "edge": f"{initial.edge_fo * 100:.2f}",
                "core": f"{initial.core_fo * 100:.2f}",
            }
        for handle, edit in self.inputs.items():
            edit.setText(values.get(handle, ""))
            edit.setEnabled(handle in values)
        if initial is not None and not values:
            self.errors["step"].setText("This prototype drags one step only.")

    def _handle_moved(self, handle: str, value: float) -> None:
        self.inputs[handle].setText(f"{value:g}" if handle == "step" else f"{value:.2f}")

    def _typed(self, handle: str) -> None:
        if self.project is None or self.current is None:
            return
        text = self.inputs[handle].text().strip()
        try:
            value = float(text)
        except ValueError:
            self.errors[handle].setText(f"{text!r} is not a number")
            return
        if value == self.canvas.values.get(handle):
            self.errors[handle].clear()
            return
        self.canvas.set_values(**{handle: value})
        self._commit(handle, value, time.perf_counter())

    def _commit(self, handle: str, value: float, started_at: float | None = None) -> None:
        """One edited handle (a drag or a typed value) as a new starting profile."""
        if self.project is None or self.current is None:
            return
        initial = self.project.run(self.current).initial
        try:
            if handle == "step":
                new = replace(initial, steps_um=(value,))
            else:  # the plot and the fields show mol%; the core takes fractions
                new = replace(initial, **{f"{handle}_fo": round(value / 100, 6)})
        except ValueError as exc:  # InitialCondition's own checks
            self._refused(handle, initial, exc)
            return
        self._apply_initial(new, started_at, handle)

    def _apply_initial(
        self, new: InitialCondition, started_at: float | None, handle: str = "step"
    ) -> None:
        """Send a starting profile through `Project.update_run`; show its refusal next to
        the field and put the handles back, or refit."""
        assert self.project is not None and self.current is not None
        name = self.current
        initial = self.project.run(name).initial
        try:
            self.project.update_run(name, initial=new)
        except ValueError as exc:
            self._refused(handle, initial, exc)
            return
        for error in self.errors.values():
            error.clear()
        self._fill_inputs(new)
        self.canvas.set_values(
            step=float(new.steps_um[0]), edge=new.edge_fo * 100, core=new.core_fo * 100
        )
        self._update_title()
        self._label_runs()
        self._scan_best = None
        self.apply_button.setEnabled(False)
        self._request_fit(name, started_at)

    def _refused(self, handle: str, initial: InitialCondition, exc: ValueError) -> None:
        self.errors[handle].setText(str(exc))
        self._fill_inputs(initial)
        self.canvas.set_values(
            step=float(initial.steps_um[0]), edge=initial.edge_fo * 100, core=initial.core_fo * 100
        )

    # -- fitting --------------------------------------------------------------------

    def _request_fit(self, name: str, started_at: float | None = None) -> None:
        assert self.project is not None
        cached = self.project.result(name)  # e.g. a drag back to a fitted position
        if cached is not None:
            if name == self.current:
                self._show_result(name, cached, started_at)
            return
        self._pending = (name, started_at)
        if not self._fit_running:
            self._start_fit()

    def _start_fit(self) -> None:
        assert self.project is not None and self._pending is not None
        name, started_at = self._pending
        self._pending = None
        copy = snapshot(self.project)

        def work() -> tuple[str, FitResult]:
            # The key is taken from the same snapshot that is fitted, so an edit made
            # meanwhile in the GUI cannot file this result under its new settings.
            return copy.result_key(name), copy.fit(name)

        job = Job(work, (name, started_at, self._serial))
        job.signals.done.connect(self._fit_done)
        job.signals.failed.connect(self._fit_failed)
        self._jobs.add(job)
        self._fit_running = True
        self.fits_started += 1
        self.statusBar().showMessage(f"Fitting {name}…")
        self.fit_pool.start(job)

    def _finish_job(self, tag: object) -> None:
        self._jobs = {j for j in self._jobs if j.tag != tag}

    def _fit_done(self, tag: tuple, answer: tuple[str, FitResult], seconds: float) -> None:
        self._finish_job(tag)
        self._fit_running = False
        name, started_at, serial = tag
        key, result = answer
        project = self.project
        current_settings = (
            project is not None
            and serial == self._serial
            and name in project.run_names
            and key == project.result_key(name)
        )
        if current_settings:
            assert project is not None
            project.state.setdefault("results", {})[name] = key
            self._label_runs()
            if name == self.current:
                self._show_result(name, result, started_at, seconds)
            self.statusBar().showMessage(f"Fitted {name} in {seconds * 1000:.0f} ms", 4000)
        if self._pending is not None and serial == self._serial:
            self._start_fit()

    def _fit_failed(self, tag: tuple, message: str) -> None:
        self._finish_job(tag)
        self._fit_running = False
        name, _, serial = tag
        self.last_error = message
        if serial == self._serial and name == self.current:
            self._draw(name, None, "fit failed")
            self.fit_label.setText(f"Fit failed: {message}")
        if self._pending is not None and serial == self._serial:
            self._start_fit()

    def _show_result(
        self,
        name: str,
        result: FitResult,
        started_at: float | None = None,
        fit_seconds: float | None = None,
    ) -> None:
        drawing = time.perf_counter()
        self._draw(name, result)
        done = time.perf_counter()
        low, high = result.time_range_days()
        self.fit_label.setText(
            f"<b>{result.time_days:.2f} days</b><br>1{SIGMA} range {low:.2f}-{high:.2f} days "
            "(from the profile alone)"
        )
        if started_at is None:
            self.timing_label.setText(
                "From the result cache"
                if fit_seconds is None
                else f"Fit {fit_seconds * 1000:.0f} ms, draw {(done - drawing) * 1000:.0f} ms"
            )
            return
        sample = {
            "total_ms": (done - started_at) * 1000,
            "fit_ms": (fit_seconds or 0.0) * 1000,
            "draw_ms": (done - drawing) * 1000,
        }
        self.latencies.append(sample)
        self.timing_label.setText(
            "Release to redraw {total_ms:.0f} ms (fit {fit_ms:.0f}, draw {draw_ms:.0f})".format(
                **sample
            )
        )

    # -- the initial-condition posterior --------------------------------------------

    def start_scan(self) -> None:
        if self.project is None or self.current is None or self._scan_token is not None:
            return
        name = self.current
        run = self.project.run(name)
        if not run.uncertainties:
            try:
                run = self.project.update_run(name, uncertainties=SCAN_PRIORS)
            except ValueError as exc:
                self.scan_label.setText(str(exc))
                return
            # The uncertainties are columns of parameters.csv, so they are part of the
            # result key: the nominal fit is redone once (tens of ms).
            self._update_title()
            self._label_runs()
            self._request_fit(name)
        cached = self.project.summary(name)
        time_days = cached["time_days"] if cached else None
        profiles = dict(self.project.profiles)
        token = object()
        self._scan_token = token
        job = Job(lambda: scan_run(run, profiles, time_days=time_days), (name, token))
        job.signals.done.connect(self._scan_done)
        job.signals.failed.connect(self._scan_failed)
        self._jobs.add(job)
        self.scan_button.setEnabled(False)
        self.busy.setVisible(True)
        self.cancel_button.setVisible(True)
        self.apply_button.setEnabled(False)
        self.scan_label.setText(f"Sampling the starting profile of {name}…")
        self.scan_pool.start(job)

    def cancel_scan(self) -> None:
        """Forget the running scan: scan_run cannot be interrupted, so it finishes on its
        pool thread and `_scan_done` drops its answer (the token no longer matches)."""
        if self._scan_token is None:
            return
        self._scan_token = None
        self._scan_idle("Cancelled. The scan finishes in the background; its result is dropped.")

    def _scan_idle(self, message: str) -> None:
        self.busy.setVisible(False)
        self.cancel_button.setVisible(False)
        self.scan_button.setEnabled(self.current is not None)
        self.scan_label.setText(message)

    def _scan_done(self, tag: tuple, answer: Any, seconds: float) -> None:
        self._finish_job(tag)
        name, token = tag
        if token is not self._scan_token:
            self.scans_dropped += 1
            return
        self._scan_token = None
        s = scan_summary(answer)
        method = s.get("scan_sampler", "grid")
        share = s.get("scan_ic_share")
        lines = [
            f"<b>{name}</b> ({method}, {seconds:.1f} s)",
            f"Best fit {s['scan_best_days']:.2f} days; 68% of the posterior "
            f"{s['scan_low_days']:.2f}-{s['scan_high_days']:.2f} days",
            f"Best starting profile: {s['scan_best']}",
        ]
        if share is not None:
            lines.append(f"Share of the time's variance from the starting profile: {share:.0%}")
        self._scan_idle("<br>".join(lines))
        self._scan_best = (name, answer.best_initial)
        self.apply_button.setEnabled(name == self.current)

    def _scan_failed(self, tag: tuple, message: str) -> None:
        self._finish_job(tag)
        if tag[1] is not self._scan_token:
            self.scans_dropped += 1
            return
        self._scan_token = None
        self.last_error = message
        self._scan_idle(f"Failed: {message}")

    def apply_scan(self) -> None:
        if self.project is None or self._scan_best is None:
            return
        name, initial = self._scan_best
        if name != self.current:
            return
        rounded = replace(
            initial,
            steps_um=tuple(round(s, 1) for s in initial.steps_um),
            edge_fo=round(initial.edge_fo, 4),
            core_fo=round(initial.core_fo, 4),
        )
        self._apply_initial(rounded, time.perf_counter())
