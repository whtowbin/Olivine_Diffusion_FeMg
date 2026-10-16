"""The main window: run list (left), the plot with its warnings and the results table
(centre), the side panel (right); fits and starting-profile fits in worker threads.

All state lives in `forster.project.Project` (the folder, the runs, the result cache) and
`forster.gui.state.RunEditor` (the selected run's side panel); this module shows them and
forwards edits, so the app accepts exactly what the command line does.
"""

from __future__ import annotations

import os
import sys
import threading
from collections.abc import Sequence
from html import escape
from pathlib import Path
from typing import Any

from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from PySide6.QtCore import QEvent, QThreadPool, QTimer
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from forster.model import FitResult
from forster.project import FitOutcome, FitTask, Project, ScanRecord, ScanTask

from .panels import ERROR_COLOR, SidePanel
from .plot import ProfileCanvas
from .state import (
    RESULT_COLUMNS,
    RESULT_TITLES,
    RunEditor,
    new_run,
    result_rows,
    scan_lines,
    summary_table,
)
from .style import palette
from .workers import Job

DEBOUNCE_MS = 120  # a refit waits this long after the last edit
AUTOSAVE_MS = 60_000
STAGES = {"best fit": 1, "importance sampling": 2, "nested sampling": 2}


def is_dark(app: QApplication) -> bool:
    """The system theme, read from the palette (it follows the OS on every platform)."""
    return app.palette().color(QPalette.ColorRole.Window).lightness() < 128


class MainWindow(QMainWindow):
    def __init__(self, ask: bool = True) -> None:
        super().__init__()
        self.ask = ask  # False in tests: no dialogs
        self.project: Project | None = None
        self.editor: RunEditor | None = None
        self.serial = 0  # bumped per opened project: drops answers for a closed one
        # One fit at a time (the newest request waits); one starting-profile fit.
        self.fit_pool, self.scan_pool = QThreadPool(self), QThreadPool(self)
        self.fit_pool.setMaxThreadCount(1)
        self.scan_pool.setMaxThreadCount(1)
        self.jobs: set[Job] = set()
        self.fit_job: Job | None = None
        self.scan_job: Job | None = None
        self.scan_cancel = threading.Event()
        self.wanted: list[str] = []  # runs to fit, the selected one first
        self.debounce = QTimer(self, singleShot=True, interval=DEBOUNCE_MS)
        self.debounce.timeout.connect(self._start_fit)
        self.autosave_timer = QTimer(self, interval=AUTOSAVE_MS)
        self.autosave_timer.timeout.connect(self.autosave)
        self.autosave_timer.start()
        self.scan_record: ScanRecord | None = None
        self.fit_errors: dict[str, str] = {}  # the last failed fit's message, per run
        self.fits_started = 0
        self.last_error = ""
        self._build()
        self.apply_theme()
        self._show_project()

    # -- layout ---------------------------------------------------------------------

    def _build(self) -> None:
        self.resize(1520, 880)
        self._build_toolbar()
        splitter = QSplitter()
        splitter.addWidget(self._build_runs())
        splitter.addWidget(self._build_centre())
        self.panel = SidePanel()
        self.panel.edited.connect(self._edited)
        self.panel.uncertainties.fit_button.clicked.connect(self.start_scan)
        self.panel.uncertainties.cancel_button.clicked.connect(self.cancel_scan)
        self.panel.uncertainties.apply_button.clicked.connect(self.apply_scan)
        splitter.addWidget(self.panel)
        splitter.setSizes([210, 760, 550])
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)
        self.statusBar().showMessage("Ready")

    def _build_toolbar(self) -> None:
        bar = self.addToolBar("Project")
        bar.setMovable(False)
        for text, slot, key in (
            ("New from example…", self._choose_new, QKeySequence.StandardKey.New),
            ("Open…", self._choose_open, QKeySequence.StandardKey.Open),
            ("Save", self.save, QKeySequence.StandardKey.Save),
            ("Save as…", self._choose_save_as, QKeySequence.StandardKey.SaveAs),
            ("Export summary CSV…", self._choose_export, None),
            ("Fit all", self.fit_all, None),
        ):
            action = QAction(text, self)
            if key is not None:
                action.setShortcut(key)
            action.triggered.connect(slot)
            bar.addAction(action)

    def _build_runs(self) -> QWidget:
        left = QWidget()
        layout = QVBoxLayout(left)
        layout.addWidget(QLabel("<b>Runs</b>"))
        self.run_list = QListWidget()
        self.run_list.currentRowChanged.connect(self._run_selected)
        layout.addWidget(self.run_list, 1)
        row1, row2 = QHBoxLayout(), QHBoxLayout()
        self.run_buttons: dict[str, QPushButton] = {}
        for text, slot, row in (
            ("Add", self._choose_add, row1),
            ("Duplicate", self.duplicate_run, row1),
            ("Rename", self._choose_rename, row2),
            ("Delete", self.delete_run, row2),
        ):
            button = QPushButton(text)
            button.clicked.connect(slot)
            row.addWidget(button)
            self.run_buttons[text] = button
        layout.addLayout(row1)
        layout.addLayout(row2)
        return left

    def _build_centre(self) -> QWidget:
        self.centre_tabs = QTabWidget()
        plot_page = QWidget()
        layout = QVBoxLayout(plot_page)
        self.canvas = ProfileCanvas()
        self.canvas.moved.connect(self._handle_moved)
        self.canvas.released.connect(self._handle_released)
        toolbar = NavigationToolbar2QT(self.canvas, self)
        self.canvas.toolbar = toolbar
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas, 1)
        self.notes = QLabel()
        self.notes.setWordWrap(True)
        self.notes.setVisible(False)
        layout.addWidget(self.notes)
        self.centre_tabs.addTab(plot_page, "Profile")
        results_page = QWidget()
        results_layout = QVBoxLayout(results_page)
        self.results = QTableWidget(0, len(RESULT_COLUMNS))
        self.results.setHorizontalHeaderLabels([RESULT_TITLES[c] for c in RESULT_COLUMNS])
        header = self.results.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        self.results.verticalHeader().setVisible(False)
        self.results.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        results_layout.addWidget(self.results)
        self.centre_tabs.addTab(results_page, "Results")
        self.centre_tabs.currentChanged.connect(lambda _: self._fill_results())
        return self.centre_tabs

    # -- theme ----------------------------------------------------------------------

    def apply_theme(self) -> None:
        app = QApplication.instance()
        dark = isinstance(app, QApplication) and is_dark(app)
        self.canvas.set_palette(palette(dark))
        self._redraw()

    def changeEvent(self, event: QEvent) -> None:
        if event.type() in (QEvent.Type.ApplicationPaletteChange, QEvent.Type.PaletteChange):
            self.apply_theme()
        super().changeEvent(event)

    # -- the project folder -----------------------------------------------------------

    def _choose_new(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self, "Choose an empty folder for the new project (the worked example)"
        )
        if folder:
            self.new_project(folder)

    def _choose_open(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Open a FORSTER project folder")
        if folder:
            self.open_project(folder)

    def _choose_save_as(self) -> None:
        if self.project is None:
            return
        folder = QFileDialog.getExistingDirectory(self, "Save the project into a folder")
        if folder:
            self.save(folder)

    def _choose_export(self) -> None:
        if self.project is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export summary CSV", str(self.project.folder / "summary.csv"), "CSV (*.csv)"
        )
        if path:
            self.export_summary(path)

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
            if (
                project.has_autosave
                and self.ask
                and self._confirm(
                    "Restore autosave",
                    "This project has unsaved changes from an earlier session. Restore them?",
                )
            ):
                project = project.restore_autosave()
        except (FileNotFoundError, OSError, ValueError) as exc:
            return self._refuse("Cannot open that folder", exc)
        self._set_project(project)
        return True

    def save(self, folder: str | Path | None = None) -> bool:
        if self.project is None:
            return False
        try:
            self.project.save(folder)
        except (OSError, ValueError) as exc:
            return self._refuse("Save failed", exc)
        self._update_title()
        self.statusBar().showMessage(f"Saved {self.project.folder}", 5000)
        return True

    def autosave(self) -> None:
        if self.project is not None and self.project.dirty:
            try:
                self.project.autosave()
            except (OSError, ValueError) as exc:
                self.statusBar().showMessage(f"Autosave failed: {exc}")

    def export_summary(self, path: str | Path) -> bool:
        if self.project is None:
            return False
        table = summary_table(self.project)
        if table.empty:
            return self._refuse("Nothing to export", ValueError("no run has been fitted yet"))
        table.to_csv(path, index=False)
        self.statusBar().showMessage(f"Wrote {len(table)} runs to {path}", 5000)
        return True

    def _set_project(self, project: Project | None) -> None:
        self.serial += 1
        self.scan_cancel.set()
        self.project, self.editor, self.wanted = project, None, []
        self.fit_errors = {}
        self._show_project()

    def _show_project(self) -> None:
        self._update_title()
        blocked = self.run_list.blockSignals(True)
        self.run_list.clear()
        self.run_list.blockSignals(blocked)
        for button in self.run_buttons.values():
            button.setEnabled(self.project is not None)
        if self.project is None:
            self.panel.load(None)
            self.canvas.show_message("Open a project folder, or start one from the example.")
            self._show_notes([])
            return
        self.run_list.addItems(self._run_labels())
        names = self.project.run_names
        selected = self.project.selected_run
        if names:
            self.run_list.setCurrentRow(names.index(selected) if selected else 0)
        else:
            self.panel.load(None)
            self.canvas.show_message("This project has no runs: Add one.")
        self._fill_results()

    def _run_labels(self) -> list[str]:
        assert self.project is not None
        labels = []
        for name in self.project.run_names:
            if self.project.summary(name) is not None:
                labels.append(f"{name}  ✓")
            elif self.project.is_stale(name):
                labels.append(f"{name}  (changed)")
            else:
                labels.append(name)
        return labels

    def _relabel_runs(self) -> None:
        for row, text in enumerate(self._run_labels()):
            item = self.run_list.item(row)
            if item is not None:
                item.setText(text)

    def _may_discard(self) -> bool:
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
            return self.save()
        return answer == QMessageBox.StandardButton.Discard

    def _confirm(self, title: str, text: str) -> bool:
        answer = QMessageBox.question(self, title, text)
        return answer == QMessageBox.StandardButton.Yes

    def _refuse(self, title: str, exc: Exception) -> bool:
        self.last_error = str(exc)
        if self.ask:
            QMessageBox.warning(self, title, str(exc))
        self.statusBar().showMessage(f"{title}: {exc}")
        return False

    def _update_title(self) -> None:
        if self.project is None:
            self.setWindowTitle("FORSTER")
            return
        mark = " *" if self.project.dirty else ""
        self.setWindowTitle(f"FORSTER: {self.project.folder.name}{mark}")

    def closeEvent(self, event: QCloseEvent) -> None:
        if not self._may_discard():
            event.ignore()
            return
        self.scan_cancel.set()
        self.autosave_timer.stop()
        self.fit_pool.waitForDone(30_000)
        self.scan_pool.waitForDone(30_000)
        event.accept()

    # -- runs -----------------------------------------------------------------------

    @property
    def current(self) -> str | None:
        return self.editor.name if self.editor is not None else None

    def _run_selected(self, row: int) -> None:
        if self.project is None or not 0 <= row < len(self.project.run_names):
            return
        name = self.project.run_names[row]
        self.project.selected_run = name
        self.editor = RunEditor(self.project, name)
        self.panel.load(self.editor)
        self._show_scan(self.project.scan(name) if self.project.run(name).uncertainties else None)
        self._redraw()
        if self.project.result(name) is None:
            self.request_fit(name)

    def _choose_add(self) -> None:
        if self.project is None or not self.project.profiles:
            return
        profiles = list(self.project.profiles)
        profile, ok = QInputDialog.getItem(self, "Add a run", "Profile", profiles, 0, False)
        if not ok:
            return
        name, ok = QInputDialog.getText(self, "Add a run", "Run name", text=profile)
        if ok:
            self.add_run(name, profile)

    def add_run(self, name: str, profile: str) -> bool:
        """A new run on ``profile`` with the selected run's settings (or defaults)."""
        assert self.project is not None
        try:
            run = new_run(self.project, name, profile, like=self.current)
            self.project.validate(run)
            self.project.add_run(run)
        except (ValueError, KeyError) as exc:
            return self._refuse("Cannot add that run", exc)
        self._after_run_change(name)
        return True

    def duplicate_run(self) -> None:
        if self.project is None or self.current is None:
            return
        copy = self.project.duplicate_run(self.current)
        self._after_run_change(copy.name)

    def _choose_rename(self) -> None:
        if self.current is None:
            return
        name, ok = QInputDialog.getText(self, "Rename run", "New name", text=self.current)
        if ok and name != self.current:
            self.rename_run(name)

    def rename_run(self, new_name: str) -> bool:
        if self.project is None or self.current is None:
            return False
        try:
            self.project.rename_run(self.current, new_name)
        except ValueError as exc:
            return self._refuse("Cannot rename", exc)
        self._after_run_change(new_name)
        return True

    def delete_run(self) -> None:
        if self.project is None or self.current is None:
            return
        name = self.current
        if self.ask and not self._confirm("Delete run", f"Delete the run {name!r}?"):
            return
        self.project.remove_run(name)
        names = self.project.run_names
        self._after_run_change(names[0] if names else None)

    def _after_run_change(self, select: str | None) -> None:
        assert self.project is not None
        self.project.selected_run = select
        self.editor = None
        self._show_project()

    # -- edits and drags --------------------------------------------------------------

    def _handle_moved(self, kind: str, index: int, value: float) -> None:
        if self.editor is None:
            return
        rows = self.editor.dragged_rows(kind, index, self.editor.snap(kind, value))
        self.canvas.set_rows(rows)
        self.panel.segments.show_rows(rows)

    def _handle_released(self, kind: str, index: int, value: float) -> None:
        if self.editor is None:
            return
        accepted = self.editor.drag(kind, index, self.editor.snap(kind, value))
        self.panel.load(self.editor)
        if accepted:
            self._edited()
        else:
            self.canvas.set_rows(self.editor.rows)
            self.panel.setCurrentWidget(self.panel.segments)

    def _edited(self) -> None:
        """An accepted edit of the selected run: redraw the starting profile and refit."""
        if self.project is None or self.current is None:
            return
        self._update_title()
        self._relabel_runs()
        self._show_scan(None)
        self._redraw()
        self.request_fit(self.current)

    # -- drawing --------------------------------------------------------------------

    def _redraw(self, result: FitResult | None = None) -> None:
        """Draw the selected run: its fit (``result``, else the cached one), the
        starting profile from the side panel, and the fit's warnings or error."""
        if self.project is None or self.editor is None:
            return
        name = self.editor.name
        run = self.project.run(name)
        result = result or self.project.result(name)
        error = self.fit_errors.get(name, "") if result is None else ""
        note = "the fit failed (see below)" if error else "fitting…" if result is None else ""
        self.canvas.show_run(
            name, self.project.profiles[run.profile_name], self.editor.rows, result, note
        )
        notes = list(self.project.fit_warnings(name) or ()) if result is not None else []
        self._show_notes(notes, error)

    def _show_notes(self, warnings: list[str], error: str = "") -> None:
        p = self.canvas.palette
        parts = (
            [f'<span style="color:{ERROR_COLOR}"><b>Fit failed:</b> {escape(error)}</span>']
            if error
            else []
        )
        parts += [
            f'<span style="color:{p.warning}"><b>Warning:</b> {escape(w)}</span>' for w in warnings
        ]
        self.notes.setText("<br>".join(parts))
        self.notes.setVisible(bool(parts))

    def _fill_results(self) -> None:
        if self.project is None:
            self.results.setRowCount(0)
            return
        rows = result_rows(self.project)
        self.results.setRowCount(len(rows))
        for i, row in enumerate(rows):
            for j, column in enumerate(RESULT_COLUMNS):
                value = row.get(column, "")
                text = f"{value:.4g}" if isinstance(value, float) else str(value)
                self.results.setItem(i, j, QTableWidgetItem(text))

    # -- fitting --------------------------------------------------------------------

    def request_fit(self, name: str) -> None:
        """Fit the run after a short pause (edits in a burst give one fit)."""
        self.wanted = [name, *(n for n in self.wanted if n != name)]
        self.debounce.start()

    def fit_all(self) -> None:
        if self.project is None:
            return
        todo = [n for n in self.project.run_names if self.project.summary(n) is None]
        self.wanted += [n for n in todo if n not in self.wanted]
        self.debounce.start()

    def _start_fit(self) -> None:
        if self.fit_job is not None or self.project is None:
            return
        while self.wanted:
            name = self.wanted.pop(0)
            if name in self.project.run_names and self.project.result(name) is None:
                break
        else:
            return
        task = self.project.fit_task(name)
        job = Job(lambda _job: task.run_collecting(), (self.serial, task))
        job.signals.done.connect(self._fit_done)
        job.signals.failed.connect(self._fit_failed)
        self.jobs.add(job)
        self.fit_job = job
        self.fits_started += 1
        self.statusBar().showMessage(f"Fitting {name}…")
        self.fit_pool.start(job)

    def _finish(self, job_tag: Any) -> None:
        self.jobs = {j for j in self.jobs if j.tag is not job_tag}

    def _fit_done(self, tag: tuple[int, FitTask], outcome: FitOutcome, seconds: float) -> None:
        self._finish(tag)
        self.fit_job = None
        serial, task = tag
        if serial == self.serial and self.project is not None:
            self.fit_errors.pop(task.name, None)
            current = self.project.store(task, outcome.result, outcome.warnings)
            if current:
                self._relabel_runs()
                self._fill_results()
                if task.name == self.current:
                    self._redraw(outcome.result)
                self.statusBar().showMessage(f"Fitted {task.name} in {seconds * 1000:.0f} ms", 4000)
        self._start_fit()

    def _fit_failed(self, tag: tuple[int, FitTask], message: str) -> None:
        self._finish(tag)
        self.fit_job = None
        serial, task = tag
        if serial == self.serial:
            self.last_error = self.fit_errors[task.name] = message
            self.statusBar().showMessage(f"Fit of {task.name} failed: {message}")
            if task.name == self.current:
                self._redraw()
        self._start_fit()

    # -- the starting-profile fit -------------------------------------------------------

    def start_scan(self) -> bool:
        tab = self.panel.uncertainties
        if self.project is None or self.editor is None or self.scan_job is not None:
            return False
        name = self.editor.name
        if not self.project.run(name).uncertainties:
            tab.error.show_errors(
                ["Give a ± for at least one step (here) or segment Fo (Segments tab) first."]
            )
            return False
        cached = self.project.scan(name)
        if cached is not None:
            self._show_scan(cached)
            return True
        task = self.project.scan_task(name)
        self.scan_cancel = cancel = threading.Event()
        runs = [0]

        def work(job: Job) -> ScanRecord:
            def stopped() -> bool:
                runs[0] += 1
                if runs[0] % 20 == 0:
                    job.report(f"{runs[0]} model runs")
                return cancel.is_set()

            return task(stage=job.report, cancel=stopped)

        job = Job(work, (self.serial, task))
        job.signals.done.connect(self._scan_done)
        job.signals.failed.connect(self._scan_failed)
        job.signals.cancelled.connect(self._scan_cancelled)
        job.signals.progress.connect(self._scan_progress)
        self.jobs.add(job)
        self.scan_job = job
        tab.error.show_errors([])
        self._scan_busy(True, f"Fitting the starting profile of {name}…")
        self.scan_pool.start(job)
        return True

    def cancel_scan(self) -> None:
        self.scan_cancel.set()

    def _scan_busy(self, busy: bool, message: str) -> None:
        tab = self.panel.uncertainties
        tab.fit_button.setEnabled(not busy)
        tab.cancel_button.setVisible(busy)
        tab.progress.setVisible(busy)
        tab.progress.setValue(0)
        tab.result.setText(message)
        if busy:
            tab.apply_button.setEnabled(False)

    def _scan_progress(self, tag: tuple[int, ScanTask], text: str) -> None:
        tab = self.panel.uncertainties
        if text in STAGES:
            tab.progress.setValue(STAGES[text])
            tab.progress.setFormat(f"{text} (%v of 3)")
        else:
            tab.result.setText(f"Fitting the starting profile of {tag[1].name}: {text}")

    def _scan_done(self, tag: tuple[int, ScanTask], record: ScanRecord, seconds: float) -> None:
        self._finish(tag)
        self.scan_job = None
        serial, task = tag
        self._scan_busy(False, "")
        if serial != self.serial or self.project is None:
            return
        self.project.store_scan(task, record)
        if task.name == self.current:
            self._show_scan(record, seconds)
        self._fill_results()

    def _scan_failed(self, tag: tuple[int, ScanTask], message: str) -> None:
        self._finish(tag)
        self.scan_job = None
        self.last_error = message
        self._scan_busy(False, "")
        self.panel.uncertainties.error.show_errors([f"Starting-profile fit failed: {message}"])

    def _scan_cancelled(self, tag: tuple[int, ScanTask]) -> None:
        self._finish(tag)
        self.scan_job = None
        self._scan_busy(False, "Cancelled.")

    def _show_scan(self, record: ScanRecord | None, seconds: float | None = None) -> None:
        tab = self.panel.uncertainties
        self.scan_record = record
        tab.apply_button.setEnabled(record is not None)
        if record is None:
            if self.scan_job is None:
                tab.result.setText("")
            return
        lines = scan_lines(record.summary)
        if seconds is not None:
            lines.append(f"({seconds:.1f} s)")
        lines += [f"Warning: {w}" for w in record.warnings]
        tab.result.setText("\n".join(lines))

    def apply_scan(self) -> None:
        record = self.scan_record
        if self.editor is None or record is None:
            return
        accepted = self.editor.apply_initial(record.best_initial)
        self.panel.load(self.editor)
        if accepted:
            self._edited()


def run_app(folder: str | Path | None = None, argv: Sequence[str] | None = None) -> int:
    """Start Qt, show the window and run until it is closed."""
    os.environ.setdefault("QT_API", "pyside6")
    app = QApplication.instance()
    if not isinstance(app, QApplication):
        app = QApplication(list(argv) if argv is not None else sys.argv[:1])
    app.setApplicationName("FORSTER")
    window = MainWindow()
    window.show()
    if folder is not None:
        path = Path(folder)
        has_tables = (path / "parameters.csv").exists()
        if has_tables:
            window.open_project(path)
        elif not path.exists() or not any(path.iterdir()):
            window.new_project(path)
        else:
            window.open_project(path)  # shows why it is not a project
    return app.exec()
