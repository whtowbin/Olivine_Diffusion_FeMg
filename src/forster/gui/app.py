"""The main window: run list (left); the plot, the time-distribution and law-comparison
views and the results table (centre); the side panel (right). Fits, starting-profile
fits, Monte Carlos and the law comparison's fits run in worker threads.

All state lives in `forster.project.Project` (the folder, the runs, the result cache) and
`forster.gui.state.RunEditor` (the selected run's side panel); this module shows them and
forwards edits, so the app accepts exactly what the command line does. Each run's view
(the centre and side tabs, the plot's zoom, the time view's choice) is kept in
``project.json`` through `Project.view`.
"""

from __future__ import annotations

import os
import sys
import threading
from collections.abc import Callable, Sequence
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT
from matplotlib.figure import Figure
from PySide6.QtCore import QEvent, QThreadPool, QTimer
from PySide6.QtGui import QAction, QCloseEvent, QKeySequence, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QListWidget,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from forster.imports import ImportTemplate, list_profile_files
from forster.model import FitResult
from forster.plotting import SIGMA, plot_fit
from forster.project import FitOutcome, FitTask, McRecord, McTask, Project, ScanRecord, ScanTask

from .dialogs import CopySettingsDialog, ImportDialog
from .importing import Imported, add_imported, import_files, import_table, read_columns
from .panels import ERROR_COLOR, SidePanel
from .plot import ProfileCanvas
from .state import (
    RESULT_COLUMNS,
    RESULT_TITLES,
    RunEditor,
    app_refusal,
    copy_settings,
    law_tasks,
    new_run,
    result_rows,
    scan_lines,
    summary_table,
    time_view,
)
from .style import palette
from .views import LawPage, TimePage
from .workers import Job

DEBOUNCE_MS = 120  # a refit waits this long after the last edit
AUTOSAVE_MS = 60_000
STAGES = {"best fit": 1, "importance sampling": 2, "nested sampling": 2}
EXPORT_DPI = 200  # as forster fit writes its figures
LAW_FITS_KEPT = 24  # law-comparison fits kept in memory (four per run and settings)
PROFILE, TIME, LAWS, RESULTS = "Profile", "Time distribution", "Law comparison", "Results"


def is_dark(app: QApplication) -> bool:
    """The system theme, read from the palette (it follows the OS on every platform)."""
    return app.palette().color(QPalette.ColorRole.Window).lightness() < 128


def mc_lines(record: McRecord) -> list[str]:
    """A Monte Carlo's summary as short lines for the Uncertainties tab."""
    s = record.summary
    how = "every draw refitted" if s["mc_method"] == "refit" else "rescaled"
    lines = [
        f"Median {s['mc_median_days']:.3g} days; 68%: {s['mc_low_days']:.3g}-"
        f"{s['mc_high_days']:.3g}; 90%: {s['mc_p05_days']:.3g}-{s['mc_p95_days']:.3g} "
        f"({s['mc_draws']} draws, {how})",
    ]
    if s.get("mc_sd_log10"):
        lines.append(f"Spread of log10 time by source: {s['mc_sd_log10']}")
    return lines + [f"Warning: {w}" for w in record.warnings]


class MainWindow(QMainWindow):
    def __init__(self, ask: bool = True) -> None:
        super().__init__()
        self.ask = ask  # False in tests: no dialogs
        self.project: Project | None = None
        self.editor: RunEditor | None = None
        self.serial = 0  # bumped per opened project: drops answers for a closed one
        # One job of each kind at a time (the newest fit request waits).
        self.fit_pool, self.scan_pool = QThreadPool(self), QThreadPool(self)
        self.mc_pool, self.law_pool = QThreadPool(self), QThreadPool(self)
        for pool in (self.fit_pool, self.scan_pool, self.mc_pool, self.law_pool):
            pool.setMaxThreadCount(1)
        self.jobs: set[Job] = set()
        self.fit_job: Job | None = None
        self.scan_job: Job | None = None
        self.mc_job: Job | None = None
        self.law_job: Job | None = None
        self.scan_cancel = threading.Event()
        self.mc_cancel = threading.Event()
        self.law_fits: dict[str, FitResult] = {}
        self.wanted: list[str] = []  # runs to fit, the selected one first
        self.debounce = QTimer(self, singleShot=True, interval=DEBOUNCE_MS)
        self.debounce.timeout.connect(self._start_fit)
        self.autosave_timer = QTimer(self, interval=AUTOSAVE_MS)
        self.autosave_timer.timeout.connect(self.autosave)
        self.autosave_timer.start()
        self.scan_record: ScanRecord | None = None
        self.mc_record: McRecord | None = None
        self.fit_errors: dict[str, str] = {}  # the last failed fit's message, per run
        self.fits_started = 0
        self.last_error = ""
        self.last_import: Imported | None = None  # with its flagged points and notes
        self._build()
        self.apply_theme()
        self._show_project()

    # -- layout ---------------------------------------------------------------------

    def _build(self) -> None:
        self.resize(1560, 900)
        self._build_toolbar()
        splitter = QSplitter()
        splitter.addWidget(self._build_runs())
        splitter.addWidget(self._build_centre())
        self.panel = SidePanel()
        self.panel.edited.connect(self._edited)
        tab = self.panel.uncertainties
        tab.fit_button.clicked.connect(self.start_scan)
        tab.cancel_button.clicked.connect(self.cancel_scan)
        tab.apply_button.clicked.connect(self.apply_scan)
        tab.mc_button.clicked.connect(self.start_mc)
        tab.mc_cancel.clicked.connect(self.cancel_mc)
        self.panel.profile.import_requested.connect(self._choose_import)
        splitter.addWidget(self.panel)
        splitter.setSizes([190, 720, 650])
        splitter.setStretchFactor(1, 1)
        self.setCentralWidget(splitter)
        self.statusBar().showMessage("Ready")

    def _build_toolbar(self) -> None:
        bar = self.addToolBar("Project")
        bar.setMovable(False)
        self.actions_: dict[str, QAction] = {}
        for text, slot, key, tip in (
            ("New from example…", self._choose_new, QKeySequence.StandardKey.New, None),
            ("Open…", self._choose_open, QKeySequence.StandardKey.Open, None),
            ("Save", self.save, QKeySequence.StandardKey.Save, None),
            ("Save as…", self._choose_save_as, QKeySequence.StandardKey.SaveAs, None),
            (
                "Import profiles…",
                self._choose_import,
                None,
                "One or several CSV or Excel files, with a column mapping",
            ),
            (
                "Import folder…",
                self._choose_import_folder,
                None,
                "Every CSV or Excel file in a folder, with one column mapping",
            ),
            ("Copy settings…", self._choose_copy, None, "Copy the selected run's settings"),
            ("Fit all", self.fit_all, None, "Fit every run not fitted yet"),
        ):
            action = QAction(text, self)
            if key is not None:
                action.setShortcut(key)
            if tip:
                action.setToolTip(tip)
            action.triggered.connect(slot)
            bar.addAction(action)
            self.actions_[text] = action
        menu = QMenu(self)
        for text, slot in (
            ("Plot as PNG or PDF…", self._choose_export_plot),
            ("Fit figure as forster fit draws it…", self._choose_export_fit_figure),
            ("Summary CSV…", self._choose_export),
            ("Project as zip…", self._choose_export_zip),
        ):
            menu.addAction(text).triggered.connect(slot)
        button = QToolButton()
        button.setText("Export")
        button.setMenu(menu)
        button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        bar.addWidget(button)

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
        copy = QPushButton("Copy settings to other runs…")
        copy.clicked.connect(self._choose_copy)
        layout.addWidget(copy)
        self.run_buttons["Copy"] = copy
        return left

    def _build_centre(self) -> QWidget:
        self.centre_tabs = QTabWidget()
        plot_page = QWidget()
        layout = QVBoxLayout(plot_page)
        self.canvas = ProfileCanvas()
        self.canvas.moved.connect(self._handle_moved)
        self.canvas.released.connect(self._handle_released)
        self.canvas.mask_clicked.connect(self._mask_clicked)
        toolbar = NavigationToolbar2QT(self.canvas, self)
        self.canvas.toolbar = toolbar
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas, 1)
        self.notes = QLabel()
        self.notes.setWordWrap(True)
        self.notes.setVisible(False)
        layout.addWidget(self.notes)
        self.centre_tabs.addTab(plot_page, PROFILE)
        self.time_page = TimePage()
        self.time_page.mode_changed.connect(lambda _: self._refresh_views())
        self.centre_tabs.addTab(self.time_page, TIME)
        self.law_page = LawPage()
        self.centre_tabs.addTab(self.law_page, LAWS)
        results_page = QWidget()
        results_layout = QVBoxLayout(results_page)
        self.results = QTableWidget(0, len(RESULT_COLUMNS))
        self.results.setHorizontalHeaderLabels([RESULT_TITLES[c] for c in RESULT_COLUMNS])
        header = self.results.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        self.results.verticalHeader().setVisible(False)
        self.results.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.results.setToolTip(
            f"Days: the best fit; 1{SIGMA} and 2{SIGMA}: its ranges from the profile alone; "
            "MC: the Monte Carlo's median and central 68% with the conditions and the equation"
        )
        results_layout.addWidget(self.results)
        self.centre_tabs.addTab(results_page, RESULTS)
        self.centre_tabs.currentChanged.connect(lambda _: self._centre_changed())
        return self.centre_tabs

    @property
    def centre_title(self) -> str:
        return self.centre_tabs.tabText(self.centre_tabs.currentIndex())

    def show_centre(self, title: str) -> None:
        for i in range(self.centre_tabs.count()):
            if self.centre_tabs.tabText(i) == title:
                self.centre_tabs.setCurrentIndex(i)

    def _centre_changed(self) -> None:
        self._fill_results()
        self._refresh_views()

    # -- theme ----------------------------------------------------------------------

    def apply_theme(self) -> None:
        app = QApplication.instance()
        dark = isinstance(app, QApplication) and is_dark(app)
        self.canvas.set_palette(palette(dark))
        self.time_page.set_palette(palette(dark))
        self.law_page.set_palette(palette(dark))
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

    def _save_path(self, title: str, name: str, filters: str) -> str:
        if self.project is None:
            return ""
        path, _ = QFileDialog.getSaveFileName(self, title, str(self.project.folder / name), filters)
        return path

    def _choose_export(self) -> None:
        path = self._save_path("Export summary CSV", "summary.csv", "CSV (*.csv)")
        if path:
            self.export_summary(path)

    def _choose_export_plot(self) -> None:
        stem = (self.current or "plot").replace("/", "_")
        path = self._save_path("Export the plot", f"{stem}.png", "PNG (*.png);;PDF (*.pdf)")
        if path:
            self.export_plot(path)

    def _choose_export_fit_figure(self) -> None:
        stem = (self.current or "fit").replace("/", "_")
        path = self._save_path(
            "Export the fit figure", f"{stem}_fit.png", "PNG (*.png);;PDF (*.pdf)"
        )
        if path:
            self.export_fit_figure(path)

    def _choose_export_zip(self) -> None:
        if self.project is None:
            return
        path = self._save_path(
            "Export the project", f"{self.project.folder.name}.zip", "Zip (*.zip)"
        )
        if path:
            self.export_zip(path)

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
        self._remember_view()
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

    def export_plot(self, path: str | Path) -> bool:
        """The figure of the centre tab shown (the plot, or a view) as PNG or PDF (by
        the file's extension)."""
        figure = {
            PROFILE: self.canvas.figure if self.canvas.run_name else None,
            TIME: self.time_page.figure if self.time_page.drawn else None,
            LAWS: self.law_page.figure if self.law_page.drawn else None,
        }.get(self.centre_title)
        if figure is None:
            return self._refuse("Nothing to export", ValueError("show a plot or a view first"))
        return self._save_figure(figure, path)

    def export_fit_figure(self, path: str | Path) -> bool:
        """The selected run's fit as ``forster fit`` draws it (`plot_fit`)."""
        result = self.project.result(self.current) if self.project and self.current else None
        if result is None:
            return self._refuse("Nothing to export", ValueError("fit the run first"))
        figure = Figure(figsize=(7, 4), layout="constrained")
        plot_fit(result, figure.subplots())
        return self._save_figure(figure, path)

    def _save_figure(self, figure: Figure, path: str | Path) -> bool:
        path = Path(path)
        if path.suffix.lower() not in (".png", ".pdf"):
            return self._refuse("Cannot export", ValueError("give a .png or .pdf file name"))
        try:
            figure.savefig(path, dpi=EXPORT_DPI, facecolor=figure.get_facecolor())
        except OSError as exc:
            return self._refuse("Export failed", exc)
        self.statusBar().showMessage(f"Wrote {path}", 5000)
        return True

    def export_zip(self, path: str | Path) -> bool:
        if self.project is None:
            return False
        self._remember_view()
        try:
            written = self.project.export_zip(path)
        except (OSError, ValueError) as exc:
            return self._refuse("Export failed", exc)
        self.statusBar().showMessage(f"Wrote {written}", 5000)
        return True

    def _set_project(self, project: Project | None) -> None:
        self.serial += 1
        self.scan_cancel.set()
        self.mc_cancel.set()
        self.project, self.editor, self.wanted = project, None, []
        self.fit_errors, self.law_fits = {}, {}
        self.law_page.key = None
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
        self.last_error = str(exc.args[0]) if isinstance(exc, KeyError) else str(exc)
        if self.ask:
            QMessageBox.warning(self, title, self.last_error)
        self.statusBar().showMessage(f"{title}: {self.last_error}")
        return False

    def _update_title(self) -> None:
        if self.project is None:
            self.setWindowTitle("FORSTER")
            return
        mark = " *" if self.project.dirty else ""
        self.setWindowTitle(f"FORSTER: {self.project.folder.name}{mark}")

    def closeEvent(self, event: QCloseEvent) -> None:
        self._remember_view()
        if not self._may_discard():
            event.ignore()
            return
        self.scan_cancel.set()
        self.mc_cancel.set()
        self.autosave_timer.stop()
        for pool in (self.fit_pool, self.scan_pool, self.mc_pool, self.law_pool):
            pool.waitForDone(30_000)
        event.accept()

    # -- per-run views ----------------------------------------------------------------

    def _remember_view(self) -> None:
        """Keep the selected run's view in project.json: the tabs, the plot's zoom and
        the time view's choice."""
        if self.project is None or self.current is None:
            return
        view = self.project.view(self.current)
        view["centre_tab"] = self.centre_title
        view["side_tab"] = self.panel.current_title
        view["time_view"] = self.time_page.current
        limits = self.canvas.user_limits() if self.canvas.run_name == self.current else None
        if limits is None:
            view.pop("xlim", None)
            view.pop("ylim", None)
        else:
            view["xlim"], view["ylim"] = list(limits[0]), list(limits[1])

    def _restore_view(self, name: str) -> None:
        assert self.project is not None
        view = self.project.view(name)
        if "side_tab" in view:
            self.panel.show_title(str(view["side_tab"]))
        if "time_view" in view:
            self.time_page.set_mode(str(view["time_view"]))
        if "xlim" in view and "ylim" in view:
            xlim, ylim = view["xlim"], view["ylim"]
            self.canvas.set_user_limits(
                (float(xlim[0]), float(xlim[1])), (float(ylim[0]), float(ylim[1]))
            )
        if "centre_tab" in view:
            self.show_centre(str(view["centre_tab"]))

    # -- runs -----------------------------------------------------------------------

    @property
    def current(self) -> str | None:
        return self.editor.name if self.editor is not None else None

    def _run_selected(self, row: int) -> None:
        if self.project is None or not 0 <= row < len(self.project.run_names):
            return
        self._remember_view()
        name = self.project.run_names[row]
        self.project.selected_run = name
        self.editor = RunEditor(self.project, name)
        self.panel.load(self.editor)
        self._show_scan(self.project.scan(name) if self.project.run(name).uncertainties else None)
        self._show_mc(self.project.mc(name))
        self._redraw()
        self._restore_view(name)
        self._refresh_views()
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
            self._add(name, profile)
        except (ValueError, KeyError) as exc:
            return self._refuse("Cannot add that run", exc)
        self._after_run_change(name)
        return True

    def _add(self, name: str, profile: str) -> None:
        assert self.project is not None
        run = new_run(self.project, name, profile, like=self.current)
        self.project.validate(run)
        self.project.add_run(run)

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
        self.editor = None
        names = self.project.run_names
        self._after_run_change(names[0] if names else None)

    def _after_run_change(self, select: str | None) -> None:
        assert self.project is not None
        self._remember_view()
        self.project.selected_run = select
        self.editor = None
        self._show_project()

    # -- importing profiles and copying settings ---------------------------------------

    def _choose_import(self) -> None:
        if self.project is None:
            return
        names, _ = QFileDialog.getOpenFileNames(
            self, "Import profiles", str(self.project.folder), "Tables (*.csv *.xlsx *.xlsm)"
        )
        self._import_paths([Path(n) for n in names])

    def _choose_import_folder(self) -> None:
        if self.project is None:
            return
        folder = QFileDialog.getExistingDirectory(
            self, "Import every table in a folder", str(self.project.folder)
        )
        if not folder:
            return
        try:
            paths = list_profile_files(folder)
        except (OSError, ValueError) as exc:
            self._refuse("Cannot read that folder", exc)
            return
        if not paths:
            self._refuse("Nothing to import", ValueError("no .csv, .xlsx or .xlsm file there"))
            return
        self._import_paths(paths)

    def _import_paths(self, paths: list[Path]) -> None:
        """Import the files: the mapping dialog opens once, for the first file, and its
        mapping, flip and offset apply to all of them."""
        if self.project is None or not paths:
            return
        try:
            table = read_columns(paths[0])
        except (OSError, ValueError) as exc:
            self._refuse("Cannot read that file", exc)
            return
        dialog = ImportDialog(paths[0], table, self, paths=paths)
        try:
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            name, flip, offset = dialog.options()
            template, add_runs = dialog.template(), dialog.add_runs.isChecked()
        finally:
            # Parented to the window, a closed dialog stayed alive (about 3 MB per
            # import; analyses/gui_memory, 2026-10-20).
            dialog.deleteLater()
        if len(paths) == 1:
            self.import_table(
                table, template, name=name, flip=flip, offset_um=offset, add_runs=add_runs
            )
        else:
            self.import_paths(paths, template, flip=flip, offset_um=offset, add_runs=add_runs)

    def import_paths(
        self,
        paths: list[Path],
        template: ImportTemplate,
        *,
        flip: bool = False,
        offset_um: float | None = 0.0,
        add_runs: bool = True,
    ) -> bool:
        """`import_table` for several files with one template
        (`forster.gui.importing.import_files`); the status line says how many files and
        profiles came in and names the files that did not match."""
        if self.project is None:
            return False
        try:
            imported, problems = import_files(paths, template, flip=flip, offset_um=offset_um)
        except ValueError as exc:
            return self._refuse("Cannot import", exc)
        return self._add_imported(
            imported, add_runs, files=len(paths) - len(problems), problems=problems
        )

    def import_table(
        self,
        table: pd.DataFrame,
        template: ImportTemplate,
        *,
        name: str = "profile",
        flip: bool = False,
        offset_um: float | None = 0.0,
        add_runs: bool = True,
    ) -> bool:
        """Add the profiles a table and an import template give
        (`forster.gui.importing.import_table`: flagged totals kept, points in exclusion
        columns masked) and, with ``add_runs``, a run on each (named after it, with the
        selected run's settings); the first new run is selected."""
        if self.project is None:
            return False
        try:
            imported = import_table(table, template, name=name, flip=flip, offset_um=offset_um)
        except ValueError as exc:
            return self._refuse("Cannot import", exc)
        return self._add_imported(imported, add_runs)

    def _add_imported(
        self,
        imported: Imported,
        add_runs: bool,
        *,
        files: int = 1,
        problems: dict[Path, str] | None = None,
    ) -> bool:
        """Add the imported profiles to the project, a run on each with ``add_runs``;
        the first new run is selected."""
        assert self.project is not None
        try:
            names = add_imported(self.project, imported)
        except ValueError as exc:
            return self._refuse("Cannot import", exc)
        self.last_import = imported
        added = []
        for profile in names if add_runs else ():
            run_name = profile
            while run_name in self.project.run_names:
                run_name += " (new)"
            try:
                self._add(run_name, profile)
            except ValueError as exc:
                self._refuse(f"Imported {profile}, but cannot add a run on it", exc)
                continue
            added.append(run_name)
        text = f"Imported {len(names)} profile(s)"
        if files > 1 or problems:
            text += f" from {files} file(s)"
        notes = list(imported.notes)
        if problems:
            notes.append(
                f"{len(problems)} file(s) did not match and were not imported: "
                + "; ".join(f"{p.name} ({why})" for p, why in problems.items())
            )
        self.statusBar().showMessage(text + (f"; {'; '.join(notes)}" if notes else ""), 8000)
        self._after_run_change(added[0] if added else self.current)
        return True

    def _choose_copy(self) -> None:
        if self.project is None or self.current is None:
            return
        dialog = CopySettingsDialog(self.current, self.project.run_names, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.copy_settings_to(*dialog.choice())

    def copy_settings_to(self, groups: list[str], targets: list[str]) -> bool:
        """Copy groups of the selected run's settings to other runs and refit them."""
        if self.project is None or self.current is None:
            return False
        try:
            copy_settings(self.project, self.current, targets, groups)
        except (ValueError, KeyError) as exc:
            return self._refuse("Cannot copy those settings", exc)
        self._update_title()
        self._relabel_runs()
        self._fill_results()
        for name in targets:
            if name != self.current:
                self.request_fit(name)
        if self.current is not None:  # keep the selected run first in the queue
            self.wanted = [self.current, *(n for n in self.wanted if n != self.current)]
        self.statusBar().showMessage(f"Copied settings to {len(targets)} run(s)", 5000)
        return True

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
            self.panel.show_tab(self.panel.segments)

    def _mask_clicked(self, x_um: float, fo_pct: float, x_scale: float, fo_scale: float) -> None:
        """Shift-click on the plot: mask the nearest point, or unmask it."""
        if self.editor is None:
            return
        i = self.editor.nearest_point(x_um, fo_pct, x_scale, fo_scale)
        masked = bool(self.editor.points().masked.iloc[i])
        accepted = self.editor.set_point_masked(i, not masked)
        self.panel.load(self.editor)
        if accepted:
            self._edited()
        else:
            self.panel.show_tab(self.panel.profile)

    def _edited(self) -> None:
        """An accepted edit of the selected run: redraw the starting profile and refit."""
        if self.project is None or self.current is None:
            return
        self._update_title()
        self._relabel_runs()
        self._show_scan(None)
        self._show_mc(self.project.mc(self.current))
        self._redraw()
        self._refresh_views()
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
            name,
            self.project.profiles[run.profile_name],
            self.editor.rows,
            result,
            note,
            self.project.masked.get(run.profile_name),
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

    # -- the views --------------------------------------------------------------------

    def _refresh_views(self) -> None:
        """Redraw the view shown, if it is one (the others wait until shown)."""
        if self.project is None or self.current is None:
            return
        if self.centre_title == TIME:
            self.time_page.draw(time_view(self.project, self.current, self.time_page.current))
        elif self.centre_title == LAWS:
            self._refresh_laws()

    def _refresh_laws(self) -> None:
        """The law comparison: fits already made are drawn at once; the others are made
        in a worker first."""
        assert self.project is not None and self.current is not None
        name = self.current
        refusal = app_refusal(self.project.run(name))
        if refusal is not None:
            self.law_page.key = None
            self.law_page.show_message(f"{name} {refusal}.")
            return
        tasks = law_tasks(self.project, name)
        key = tuple(t.key for t in tasks)
        if self.law_page.key == key and self.law_page.drawn:
            return
        own = self.project.result(name)
        if own is not None:
            self.law_fits[tasks[0].key] = own
        missing = [t for t in tasks if t.key not in self.law_fits]
        if not missing:
            self.law_page.draw([self.law_fits[t.key] for t in tasks], key)
            return
        self.law_page.key = None
        self.law_page.show_message(f"Fitting {name} under {len(missing)} equation(s)…")
        if self.law_job is not None:
            return  # its answer redraws this view, and starts the next if needed
        job = Job(
            lambda _job: [(t.key, t.run_collecting().result) for t in missing], (self.serial, name)
        )
        job.signals.done.connect(self._laws_done)
        job.signals.failed.connect(self._laws_failed)
        self.jobs.add(job)
        self.law_job = job
        self.law_pool.start(job)

    def _laws_done(
        self, tag: tuple[int, str], fits: list[tuple[str, FitResult]], _s: float
    ) -> None:
        self._finish(tag)
        self.law_job = None
        if tag[0] != self.serial:
            return
        for key, result in fits:
            self.law_fits[key] = result
        while len(self.law_fits) > LAW_FITS_KEPT:
            self.law_fits.pop(next(iter(self.law_fits)))
        self._refresh_views()

    def _laws_failed(self, tag: tuple[int, str], message: str) -> None:
        self._finish(tag)
        self.law_job = None
        self.last_error = message
        self.law_page.show_message(f"A fit for the law comparison failed: {message}")

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
        refusal = app_refusal(self.project.run(name))
        if refusal is not None:
            self.last_error = self.fit_errors[name] = refusal
            self.statusBar().showMessage(f"Not fitted: {name} {refusal}")
            if name == self.current:
                self._redraw()
            self._start_fit()
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
                    self.panel.advanced.show_diagnostics()
                    self._refresh_views()
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
        refusal = app_refusal(self.project.run(name))
        if refusal is not None:
            tab.error.show_errors([f"{name} {refusal}."])
            return False
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

        job = self._start_job(work, (self.serial, task), self.scan_pool, self._scan_done)
        job.signals.failed.connect(self._scan_failed)
        job.signals.cancelled.connect(self._scan_cancelled)
        job.signals.progress.connect(self._scan_progress)
        self.scan_job = job
        tab.error.show_errors([])
        self._scan_busy(True, f"Fitting the starting profile of {name}…")
        return True

    def _start_job(
        self, work: Callable[[Job], Any], tag: object, pool: QThreadPool, done: Callable
    ) -> Job:
        job = Job(work, tag)
        job.signals.done.connect(done)
        self.jobs.add(job)
        pool.start(job)
        return job

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
            self._show_mc(self.project.mc(task.name))  # a new posterior: a new Monte Carlo
            self._refresh_views()
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
        self._scan_busy(False, "Canceled.")

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

    # -- the Monte Carlo ----------------------------------------------------------------

    def start_mc(self) -> bool:
        """Run the selected run's Monte Carlo in a worker (or show the cached one)."""
        tab = self.panel.uncertainties
        if self.project is None or self.editor is None or self.mc_job is not None:
            return False
        name = self.editor.name
        refusal = app_refusal(self.project.run(name))
        if refusal is not None:
            tab.mc_error.show_errors([f"{name} {refusal}."])
            return False
        try:
            task = self.project.mc_task(name)
        except ValueError as exc:
            tab.mc_error.show_errors([str(exc)])
            return False
        cached = self.project.mc(name)
        if cached is not None:
            self._show_mc(cached)
            return True
        self.mc_cancel = cancel = threading.Event()

        def work(job: Job) -> McRecord:
            return task(progress=lambda done, n: job.report(f"{done}/{n}"), cancel=cancel.is_set)

        job = self._start_job(work, (self.serial, task), self.mc_pool, self._mc_done)
        job.signals.failed.connect(self._mc_failed)
        job.signals.cancelled.connect(self._mc_cancelled)
        job.signals.progress.connect(self._mc_progress)
        self.mc_job = job
        tab.mc_error.show_errors([])
        self._mc_busy(True, f"Monte Carlo for {name}…")
        return True

    def cancel_mc(self) -> None:
        self.mc_cancel.set()

    def _mc_busy(self, busy: bool, message: str) -> None:
        tab = self.panel.uncertainties
        tab.mc_button.setEnabled(not busy and tab.mc_on.isChecked())
        tab.mc_cancel.setVisible(busy)
        tab.mc_progress.setVisible(busy)
        tab.mc_progress.setRange(0, 0)  # busy until a refitted draw reports
        tab.mc_result.setText(message)

    def _mc_progress(self, _tag: tuple[int, McTask], text: str) -> None:
        done, n = (int(v) for v in text.split("/"))
        bar = self.panel.uncertainties.mc_progress
        bar.setRange(0, n)
        bar.setValue(done)

    def _mc_done(self, tag: tuple[int, McTask], record: McRecord, seconds: float) -> None:
        self._finish(tag)
        self.mc_job = None
        serial, task = tag
        self._mc_busy(False, "")
        if serial != self.serial or self.project is None:
            return
        self.project.store_mc(task, record)
        if task.name == self.current:
            self._show_mc(record, seconds)
            self._refresh_views()
        self._fill_results()

    def _mc_failed(self, tag: tuple[int, McTask], message: str) -> None:
        self._finish(tag)
        self.mc_job = None
        self.last_error = message
        self._mc_busy(False, "")
        self.panel.uncertainties.mc_error.show_errors([f"Monte Carlo failed: {message}"])

    def _mc_cancelled(self, tag: tuple[int, McTask]) -> None:
        self._finish(tag)
        self.mc_job = None
        self._mc_busy(False, "Canceled.")

    def _show_mc(self, record: McRecord | None, seconds: float | None = None) -> None:
        self.mc_record = record
        if self.mc_job is not None:
            return
        lines = mc_lines(record) if record is not None else []
        if record is not None and seconds is not None:
            lines.append(f"({seconds:.1f} s)")
        self.panel.uncertainties.mc_result.setText("\n".join(lines))


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
