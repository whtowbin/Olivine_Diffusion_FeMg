"""The desktop app, headless (``QT_QPA_PLATFORM=offscreen``): open the template example,
fit, edit a segment in the side table and by dragging on the plot, refit, fit the
starting profile, save, reopen and see the same state from the cache. Skipped without
PySide6 (the ``gui`` extra)."""

import os
import time
import zipfile
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEvent, QEventLoop, QPoint, Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton

from forster.gui.app import MainWindow
from forster.gui.dialogs import ImportDialog
from forster.gui.importing import read_columns
from forster.gui.quit_dialog import QuitDialog
from forster.gui.state import RESULT_COLUMNS, RunEditor
from forster.imports import guess

EXAMPLE = "AZ18 WHT01_Ol_63_prof_1"


@pytest.fixture(scope="module")
def app():
    instance = QApplication.instance()
    return instance if isinstance(instance, QApplication) else QApplication([])


def wait(app, condition, what, timeout=180.0):
    start = time.perf_counter()
    while not condition():
        if time.perf_counter() - start > timeout:
            raise AssertionError(f"timed out waiting for {what}")
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 20)
        time.sleep(0.002)


def settled(window):
    return window.fit_job is None and not window.wanted and not window.debounce.isActive()


def point(canvas, x, y):
    """Data coordinates to canvas widget pixels (matplotlib counts from the bottom)."""
    dx, dy = canvas.ax.transData.transform((x, y))
    ratio = canvas.device_pixel_ratio
    return QPoint(round(dx / ratio), round((canvas.figure.bbox.height - dy) / ratio))


def drag_step(canvas, x0, dx):
    low, high = canvas.ax.get_ylim()
    y = (low + high) / 2
    left, none = Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier
    QTest.mousePress(canvas, left, none, point(canvas, x0, y))
    for i in range(1, 7):
        QTest.mouseMove(canvas, point(canvas, x0 + dx * i / 6, y))
    QTest.mouseRelease(canvas, left, none, point(canvas, x0 + dx, y))


def test_open_fit_edit_refit_save_reopen(app, tmp_path):
    window = MainWindow(ask=False)
    window.show()
    folder = tmp_path / "example"
    try:
        assert window.new_project(folder)
        project = window.project
        assert project is not None and window.current == EXAMPLE
        wait(app, lambda: project.result(EXAMPLE) is not None and settled(window), "first fit")
        first = project.result(EXAMPLE)
        assert first is not None and first.time_days > 0
        assert EXAMPLE in window.canvas.ax.get_title()
        assert [t.get_text() for t in window.canvas.labels] == ["a", "b"]

        # Edit segment b's Fo in the side table: the run changes and is refitted.
        table = window.panel.segments.table
        item = table.item(1, 3)
        assert item is not None and item.text() == "89.85"
        item.setText("89.9")
        assert project.run(EXAMPLE).initial.segment_fo == (0.9146, 0.899)
        wait(app, lambda: project.result(EXAMPLE) is not None and settled(window), "refit")
        refit = project.result(EXAMPLE)
        assert refit is not None and refit.time_s != first.time_s
        assert window.windowTitle().endswith("*")

        # A refused value is shown next to the table and changes nothing.
        before = project.run(EXAMPLE)
        item = table.item(0, 3)
        assert item is not None
        item.setText("120")
        assert project.run(EXAMPLE) == before
        assert "at most 100" in window.panel.segments.error.text()
        assert window.panel.segments.error.isVisibleTo(window.panel.segments)

        # Drag the step 3 um into the crystal on the plot.
        app.processEvents()
        drag_step(window.canvas, 24.0, 3.0)
        step = project.run(EXAMPLE).initial.steps_um[0]
        assert abs(step - 27.0) < 0.6
        wait(app, lambda: project.result(EXAMPLE) is not None and settled(window), "drag refit")

        # The starting-profile fit: priors on the step and segment b, then use its best.
        editor = window.editor
        assert editor is not None
        assert editor.set_step_spread(0, "2") and editor.set_cell(1, "spread", "0.2")
        window.panel.load(editor)
        window._edited()
        wait(app, lambda: project.result(EXAMPLE) is not None and settled(window), "prior refit")
        assert window.start_scan()
        wait(app, lambda: window.scan_job is None, "starting-profile fit", timeout=300)
        record = window.scan_record
        assert record is not None and "Best fit" in window.panel.uncertainties.result.text()
        assert window.panel.uncertainties.apply_button.isEnabled()

        assert window.save()
        saved_run, saved = project.run(EXAMPLE), project.result(EXAMPLE)
        assert saved is not None
    finally:
        window.close()

    again = MainWindow(ask=False)
    again.show()
    try:
        assert again.open_project(folder)
        app.processEvents()
        reopened = again.project
        assert reopened is not None and again.current == EXAMPLE
        assert reopened.run(EXAMPLE) == saved_run
        cached = reopened.result(EXAMPLE)
        assert cached is not None and cached.time_s == saved.time_s
        assert again.fits_started == 0  # shown from the cache
        assert again.editor is not None
        assert again.editor.rows == RunEditor(reopened, EXAMPLE).rows
        shown = again.panel.segments.table.item(1, 3)
        assert shown is not None and shown.text() == "89.9"
        assert again.scan_record is not None  # the starting-profile fit, cached too
    finally:
        again.close()


MGO, FEO = 40.3044, 71.8444  # g/mol


def lab_fo():
    x = np.arange(10, 70, 2.0)
    return x, 0.90 - 0.025 * np.exp(-(x - 10) / 8)


def lab_csv(path, **extra):
    """Two profiles of microprobe analyses: MgO, FeO and SiO2 in wt% (Fo of A as
    `lab_fo`, B 1 mol% lower), distances from 10 um, a Total column with one low total
    (A at 24 um), and ``extra`` columns."""
    x, fo = lab_fo()
    rows = []
    for name, f in (("A", fo), ("B", fo - 0.01)):
        mgo = np.full(len(x), 48.0)
        feo = mgo / MGO * (1 - f) / f * FEO  # Fo = (MgO/40.3)/(MgO/40.3 + FeO/71.8)
        total = np.full(len(x), 100.1)
        if name == "A":
            total[7] = 97.0
        rows.append(
            pd.DataFrame(
                {
                    "Sample": name,
                    "Distance (um)": x,
                    "SiO2": total - mgo - feo,
                    "MgO": mgo,
                    "FeO": feo,
                    "Total": total,
                    **extra,
                }
            )
        )
    pd.concat(rows).to_csv(path, index=False)
    return path


def click(canvas, x, y, modifier):
    QTest.mouseClick(canvas, Qt.MouseButton.LeftButton, modifier, point(canvas, x, y))


def test_import_mask_monte_carlo_views_export_and_reopen(app, tmp_path):
    window = MainWindow(ask=False)
    window.show()
    folder = tmp_path / "example"
    try:
        assert window.new_project(folder)
        project = window.project
        assert project is not None
        wait(app, lambda: project.result(EXAMPLE) is not None and settled(window), "first fit")

        # Import microprobe analyses through the dialog: the mapping is
        # forster.imports.guess (oxides with their units, the Total column), Fo comes
        # from MgO and FeO, and the low total is flagged and kept; the rim moved to 0.
        path = lab_csv(tmp_path / "lab.csv")
        table = read_columns(path)
        dialog = ImportDialog(path, table, window)
        ok = dialog.buttons.button(dialog.buttons.StandardButton.Ok)
        template = dialog.template()
        assert template.columns == guess(table).columns and template.problems() == []
        holds = {c.column: (c.role, c.quantity, c.unit) for c in template.columns}
        assert holds["FeO"] == ("oxide", "FeO", "wt%") and holds["Total"][0] == "total"
        assert dialog.rows["FeO"].role.toolTip() == "FeO: oxide header"
        assert "Would add 2 profile(s)" in dialog.summary.text() and ok.isEnabled()
        dialog.offset.setText("10")
        dialog.update_summary()
        assert "0-58 µm" in dialog.summary.text()
        flagged = dialog.flagged.text()
        assert "outside 98.5-101.5 wt%" in flagged and "A: 14 µm" in flagged
        assert "total 97.00 wt%" in flagged and dialog.flagged.isVisibleTo(dialog)
        # The window is typed in the dialog; a mapping with no Fo source disables OK.
        dialog.total_low.setText("99")
        dialog.update_summary()
        assert ok.isEnabled() and dialog.template().total_window == (99.0, 101.5)
        dialog.total_low.setText("98.5")
        mgo = dialog.rows["MgO"]
        mgo.role.setCurrentIndex(mgo.role.findData("ignore"))
        assert not ok.isEnabled() and "no Fo" in dialog.error.text()
        mgo.role.setCurrentIndex(mgo.role.findData("oxide"))
        assert not ok.isEnabled() and "name the oxide" in dialog.error.text()
        mgo.quantity.setCurrentIndex(mgo.quantity.findData("MgO"))
        assert ok.isEnabled()
        # Save the mapping as an import template and load it back.
        saved = tmp_path / "lab.import.json"
        assert dialog.save_template(saved) and saved.exists()
        assert dialog.load_template(saved)
        assert "Loaded the import template lab.import.json" in dialog.notes.text()
        assert dialog.template().columns == template.columns and ok.isEnabled()
        _, flip, offset = dialog.options()
        chosen = dialog.template()
        dialog.close()
        # A file with an extra column: the template is refused with the mismatch message.
        other = lab_csv(tmp_path / "other.csv", **{"Beam current": 20.0})
        refused = ImportDialog(other, read_columns(other), window)
        assert not refused.load_template(saved)
        assert "does not know: ['Beam current']" in refused.notes.text()
        assert refused.template_name == ""  # the guess is still shown
        refused.close()

        assert window.import_table(table, chosen, flip=flip, offset_um=offset)
        assert window.current == "A" and project.run_names[-2:] == ["A", "B"]
        a = project.profiles["A"]
        assert a.distance_um[0] == 0.0 and len(a.fo) == 30  # the flagged point kept
        np.testing.assert_allclose(a.fo, lab_fo()[1], rtol=1e-12)  # Fo from MgO and FeO
        assert window.last_import is not None and len(window.last_import.flagged) == 1
        assert "kept and flagged" in window.statusBar().currentMessage()
        wait(app, lambda: project.result("A") is not None and settled(window), "fit of A")
        first = project.result("A")
        assert first is not None

        # Mask a point by shift-clicking it on the plot, then another in the table.
        x, fo = project.profiles["A"].distance_um[5], project.profiles["A"].fo[5] * 100
        app.processEvents()
        click(window.canvas, x, fo, Qt.KeyboardModifier.ShiftModifier)
        assert project.points("A").masked.sum() == 1
        wait(app, lambda: project.result("A") is not None and settled(window), "refit")
        refit = project.result("A")
        assert refit is not None and len(refit.data_fo) == len(first.data_fo) - 1
        legend = window.canvas.legend  # under the panels, so it covers no data
        assert legend is not None and "Masked (left out)" in [t.get_text() for t in legend.texts]
        assert window.canvas.ax.get_legend() is None
        check = window.panel.profile.table.item(6, 2)
        assert check is not None
        check.setCheckState(Qt.CheckState.Unchecked)
        assert project.points("A").masked.sum() == 2
        wait(app, lambda: project.result("A") is not None and settled(window), "refit 2")

        # A short Monte Carlo (200 draws) from the Uncertainties tab.
        u = window.panel.uncertainties
        u.mc_on.setChecked(True)
        assert project.run("A").mc is not None
        for key, text in (("T_sd_C", "15"), ("samples", "200")):
            u.mc_fields[key].setText(text)
            u.mc_fields[key].editingFinished.emit()
        mc = project.run("A").mc
        assert mc is not None and mc.T_sd_C == 15.0 and project.run("A").mc_samples == 200
        assert window.start_mc()
        wait(app, lambda: window.mc_job is None, "Monte Carlo")
        assert "Median" in u.mc_result.text() and "200 draws" in u.mc_result.text()
        record = project.mc("A")
        assert record is not None
        window.show_centre("Results")
        cell = window.results.item(
            project.run_names.index("A"), list(RESULT_COLUMNS).index("mc_median")
        )
        assert cell is not None
        assert float(cell.text()) == pytest.approx(record.summary["mc_median_days"], rel=1e-3)

        # Each view: the time distribution (with the Monte Carlo) and the law comparison.
        window.time_page.set_mode("all")
        window.show_centre("Time distribution")
        assert window.time_page.drawn
        legend = window.time_page.figure.axes[0].get_legend()
        assert legend is not None
        assert any("conditions and equation" in t.get_text() for t in legend.texts)
        window.show_centre("Law comparison")
        wait(app, lambda: window.law_job is None and window.law_page.drawn, "law fits")
        assert len(window.law_page.figure.subfigs) == 2

        # Export the view and the plot (PNG, PDF), the CLI's fit figure and the zip.
        assert window.export_plot(tmp_path / "laws.png")
        window.show_centre("Profile")
        assert window.export_plot(tmp_path / "profile.pdf")
        assert window.export_fit_figure(tmp_path / "fit.png")
        assert not window.export_plot(tmp_path / "plot.txt")
        assert window.export_zip(tmp_path / "shared.zip")
        for name in ("laws.png", "profile.pdf", "fit.png"):
            assert (tmp_path / name).stat().st_size > 10_000
        assert (tmp_path / "profile.pdf").read_bytes().startswith(b"%PDF")
        with zipfile.ZipFile(tmp_path / "shared.zip") as shared:
            profiles = shared.read("example/profiles.csv").decode()
        assert "Marked_bad" in profiles and profiles.count(",bad") == 2

        # Copy the Monte Carlo settings to the example run.
        assert window.copy_settings_to(["mc"], [EXAMPLE])
        assert project.run(EXAMPLE).mc == project.run("A").mc

        # The view to come back to: zoomed in, on the time-distribution tab.
        window.canvas.set_user_limits((0.0, 20.0), (86.0, 90.0))
        window.panel.show_title("Advanced")
        window.show_centre("Time distribution")
        assert window.save()
    finally:
        window.close()

    again = MainWindow(ask=False)
    again.show()
    try:
        assert again.open_project(folder)
        app.processEvents()
        reopened = again.project
        assert reopened is not None and again.current == "A"
        assert reopened.points("A").masked.sum() == 2
        assert again.fits_started == 0 and again.mc_record is not None
        assert again.centre_title == "Time distribution" and again.time_page.drawn
        assert again.panel.current_title == "Advanced"
        assert again.time_page.current == "all"
        limits = again.canvas.user_limits()
        assert limits is not None and limits[0] == pytest.approx((0.0, 20.0))
    finally:
        again.close()


def test_the_installers_self_test_passes_in_a_checkout(app, tmp_path):
    from forster.gui.selftest import self_test

    assert self_test(tmp_path) == 0
    assert (tmp_path / "project.zip").exists() and (tmp_path / "laws.png").exists()


def test_the_import_dialog_is_freed_after_each_import(app, tmp_path, monkeypatch):
    """A dialog parented to the window stayed alive after closing (about 3 MB per
    import, analyses/gui_memory): opening it 20 times must leave none behind."""
    window = MainWindow(ask=False)
    window.show()
    try:
        assert window.new_project(tmp_path / "example")
        path = lab_csv(tmp_path / "lab.csv")
        monkeypatch.setattr(
            "forster.gui.app.QFileDialog.getOpenFileNames", lambda *a, **k: ([str(path)], "")
        )
        monkeypatch.setattr(ImportDialog, "exec", lambda self: 0)  # the user cancels
        for _ in range(20):
            window._choose_import()
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()
        assert window.findChildren(ImportDialog) == []
    finally:
        window.close()


def test_importing_several_files_asks_for_the_mapping_once(app, tmp_path, monkeypatch):
    """Three files (one with another layout) through the real dialog: the dialog opens
    once for the first file, the profiles are named after the files, and the status
    line names the file that did not match."""
    window = MainWindow(ask=False)
    window.show()
    try:
        assert window.new_project(tmp_path / "example")
        project = window.project
        assert project is not None
        x, fo = lab_fo()
        files = []
        for stem, shift in (("g1", 0.0), ("g2", 0.01)):
            path = tmp_path / f"{stem}.csv"
            pd.DataFrame({"Distance (um)": x, "Fo": fo - shift}).to_csv(path, index=False)
            files.append(str(path))
        odd = tmp_path / "odd.csv"
        pd.DataFrame({"Distance (um)": x, "Fo": fo, "Other": 1}).to_csv(odd, index=False)
        files.append(str(odd))
        monkeypatch.setattr(
            "forster.gui.app.QFileDialog.getOpenFileNames", lambda *a, **k: (files, "")
        )
        shown = []

        def accept(self):
            shown.append(self.windowTitle())
            assert "2 more" in self.windowTitle() and not self.name.isEnabled()
            assert list(self.mismatched) == [odd]
            assert "2 of 3 files import" in self.summary.text()
            self.add_runs.setChecked(False)  # no fits: the profiles alone are enough here
            return 1

        monkeypatch.setattr(ImportDialog, "exec", accept)
        window._choose_import()
        assert len(shown) == 1
        message = window.statusBar().currentMessage()
        assert "2 profile(s) from 2 file(s)" in message and "odd.csv" in message
        assert window.last_import is not None
        assert {"g1", "g2"} <= set(window.last_import.profiles)
        assert {"g1", "g2"} <= set(project.profiles)
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()
        assert window.findChildren(ImportDialog) == []
    finally:
        window.close()


def test_the_app_refuses_a_legacy_run_and_never_krigs(app, tmp_path, monkeypatch):
    """The legacy uncertainty method fits a kriged profile; the app never loads pykrige
    (Henry, 2026-10-16), so a table asking for it is refused with a pointer to the CLI."""

    def no_kriging(*args, **kwargs):
        raise AssertionError("the app reached kriging")

    monkeypatch.setattr("forster.engine._kriging_model", no_kriging)
    window = MainWindow(ask=False)
    window.show()
    try:
        assert window.new_project(tmp_path / "example")
        project = window.project
        assert project is not None
        wait(app, lambda: project.result(EXAMPLE) is not None and settled(window), "first fit")
        project.update_run(EXAMPLE, uncertainty="legacy", scheme="central")
        window._after_run_change(EXAMPLE)
        wait(app, lambda: EXAMPLE in window.fit_errors and settled(window), "refusal")
        assert "forster fit" in window.fit_errors[EXAMPLE]
        assert project.result(EXAMPLE) is None
        assert not window.start_scan()
        window.show_centre("Law comparison")
        assert window.law_job is None and not window.law_page.drawn
    finally:
        window.close()


def test_the_import_dialog_starts_each_profile_at_its_first_point_with_advice(app, tmp_path):
    path = lab_csv(tmp_path / "lab.csv")
    dialog = ImportDialog(path, read_columns(path), None)
    try:
        assert not dialog.rim_advice.isVisibleTo(dialog) and dialog.offset.isEnabled()
        assert dialog.options()[2] == 0.0
        dialog.each_start.setChecked(True)
        assert dialog.rim_advice.isVisibleTo(dialog) and not dialog.offset.isEnabled()
        assert "measured from the crystal rim" in dialog.rim_advice.text()
        assert dialog.options()[2] is None
    finally:
        dialog.deleteLater()


def test_the_cite_dialog_shows_the_runs_equation_copies_it_and_is_freed(app, tmp_path):
    from PySide6.QtCore import QTimer
    from PySide6.QtGui import QGuiApplication

    from forster import cite
    from forster.gui.dialogs import CiteDialog

    window = MainWindow(ask=False)
    window.show()
    try:
        assert window.new_project(tmp_path / "example")
        seen = {}

        def press_copy():
            dialog = window.findChildren(CiteDialog)[0]
            seen["text"] = dialog.text.toPlainText()
            QGuiApplication.clipboard().clear()
            dialog.copy_button.click()
            seen["clipboard"] = QGuiApplication.clipboard().text()
            dialog.reject()

        QTimer.singleShot(200, press_copy)
        window.actions_["Cite FORSTER…"].trigger()  # the real Help menu action
        assert [a.text() for a in window.menuBar().actions()] == ["Help"]
        assert window.editor is not None
        expected = cite(window.editor.run.diffusivity)
        assert seen["text"] == expected
        assert "Dohmen and Chakraborty" in expected
        assert seen["clipboard"] == expected
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()
        assert window.findChildren(CiteDialog) == []
    finally:
        window.close()


def record_dialogs(monkeypatch, opened, answers=None):
    """Every file, input and app dialog the window can open records its name in
    ``opened`` and answers from ``answers`` (by name), else as if cancelled."""
    from forster.gui.dialogs import CiteDialog, CopySettingsDialog

    answers = answers or {}

    def fake(name, cancelled):
        def ask(*args, **kwargs):
            opened.append(name)
            answer = answers.get(name, cancelled)
            return answer(*args) if callable(answer) else answer

        return ask

    for name, cancelled in (
        ("getOpenFileNames", ([], "")),
        ("getExistingDirectory", ""),
        ("getSaveFileName", ("", "")),
    ):
        monkeypatch.setattr(f"forster.gui.app.QFileDialog.{name}", fake(name, cancelled))
    for name, cancelled in (("getItem", ("", False)), ("getText", ("", False))):
        monkeypatch.setattr(f"forster.gui.app.QInputDialog.{name}", fake(name, cancelled))
    for dialog in (ImportDialog, CiteDialog, CopySettingsDialog):
        monkeypatch.setattr(dialog, "exec", fake(dialog.__name__, 0))


@pytest.mark.parametrize("state", ["no project", "empty project", "example project"])
def test_every_enabled_button_does_something_visible(app, tmp_path, monkeypatch, state):
    """The way a user meets the app: press every toolbar, menu and run button. An
    enabled one must open a dialog, write to the status line or change the project;
    one that cannot act yet must be greyed out. (Import, Fit all, Save and Export did
    nothing at all with no project open: every earlier test opened the example first.)"""
    opened: list[str] = []
    record_dialogs(monkeypatch, opened)
    window = MainWindow(ask=False)
    window.show()
    try:
        if state != "no project":
            assert window.new_project(tmp_path / "p", example=state == "example project")
        app.processEvents()
        buttons: list[tuple[str, Callable[[], object], Callable[[], bool]]] = [
            (text, action.trigger, action.isEnabled) for text, action in window.actions_.items()
        ]
        buttons += [
            (f"Export > {action.text()}", action.trigger, window.export_button.isEnabled)
            for action in window.export_button.menu().actions()
        ]
        # Delete last, so the run is still there for the others.
        buttons += [
            (text, button.click, button.isEnabled)
            for text, button in sorted(window.run_buttons.items(), key=lambda t: t[0] == "Delete")
        ]
        silent, greyed = [], []
        for text, press, enabled in buttons:
            if not enabled():
                greyed.append(text)
                continue
            opened.clear()
            window.statusBar().clearMessage()
            before = (window.project, window.project and list(window.project.run_names))
            press()
            app.processEvents()
            after = (window.project, window.project and list(window.project.run_names))
            if not opened and not window.statusBar().currentMessage() and before == after:
                silent.append(text)
        assert silent == [], f"{state}: these buttons did nothing: {silent}"
        always = ["New project…", "Open…", "Import profiles…", "Import folder…"]
        assert not set(always) & set(greyed)
        if state == "no project":
            assert {"Save", "Fit all", "Export > Summary CSV…", "Add"} <= set(greyed)
        if state == "example project":
            assert greyed == []
    finally:
        window.close()


def test_cold_start_import_excel_fit_save_reopen(app, tmp_path, monkeypatch):
    """A new user's path, through the real buttons: no project open, Import folder on
    two Excel workbooks, map the columns, name a folder for the new project; then Import
    profiles on an old-format .xls; fit, Save, reopen. Opening a folder of spreadsheets
    points to Import."""
    data = tmp_path / "probe data"
    data.mkdir()
    for grain in ("1", "2"):
        table = pd.read_csv(lab_csv(tmp_path / f"{grain}.csv"))
        table["Sample"] += grain  # A1, B1 in grain 1.xlsx, ...
        table.to_excel(data / f"grain {grain}.xlsx", index=False)
    extra = Path(__file__).parent / "data" / "probe_export.xls"  # profile X1, old format
    folder = tmp_path / "my project"
    mapped = []

    def accept(dialog):
        ok = dialog.buttons.button(dialog.buttons.StandardButton.Ok)
        assert ok.isEnabled(), dialog.error.text()  # the guessed mapping is usable
        mapped.append(dialog.path.name)
        return 1

    opened: list[str] = []
    record_dialogs(
        monkeypatch,
        opened,
        {
            "getExistingDirectory": str(data),
            "getSaveFileName": (str(folder), ""),
            "getOpenFileNames": ([str(extra)], ""),
            "ImportDialog": accept,
        },
    )
    window = MainWindow(ask=False)
    window.show()
    try:
        assert not window.open_project(data)
        assert "Import profiles or Import folder" in window.last_error
        assert window.project is None and not window.actions_["Save"].isEnabled()

        window.actions_["Import folder…"].trigger()
        assert opened == ["getExistingDirectory", "ImportDialog", "getSaveFileName"]
        assert mapped == ["grain 1.xlsx"]
        project = window.project
        assert project is not None and project.folder == folder
        assert set(project.profiles) == {"A1", "B1", "A2", "B2"}
        assert "from 2 file(s)" in window.statusBar().currentMessage()
        assert window.actions_["Save"].isEnabled() and window.run_buttons["Delete"].isEnabled()

        opened.clear()
        window.actions_["Import profiles…"].trigger()
        assert opened == ["getOpenFileNames", "ImportDialog"]  # into the open project
        assert "X1" in project.profiles

        wait(app, lambda: project.result(window.current) is not None and settled(window), "fit")
        fitted = window.current
        window.actions_["Save"].trigger()
        assert (folder / "parameters.csv").exists() and not project.dirty
    finally:
        window.close()

    again = MainWindow(ask=False)
    try:
        assert again.open_project(folder)
        assert again.project is not None and "X1" in again.project.profiles
        assert again.project.result(fitted) is not None  # from the cache
    finally:
        again.close()


def test_rename_a_run_then_switch_save_and_reopen(app, tmp_path):
    """Renaming looked the view up under the old name: afterwards switching runs, Save
    and closing all raised KeyError."""
    window = MainWindow(ask=False)
    window.show()
    folder = tmp_path / "example"
    try:
        assert window.new_project(folder)
        project = window.project
        assert project is not None
        window.duplicate_run()
        copy = window.current
        assert copy is not None and copy != EXAMPLE
        window.show_centre("Time distribution")
        assert window.rename_run("renamed")
        assert window.current == "renamed"
        window._run_selected(project.run_names.index(EXAMPLE))
        assert window.current == EXAMPLE
        window._run_selected(project.run_names.index("renamed"))
        assert window.save()
        assert project.view("renamed")["centre_tab"] == "Time distribution"
        wait(app, lambda: settled(window), "fits")
    finally:
        window.close()
    again = MainWindow(ask=False)
    try:
        assert again.open_project(folder)
        assert again.project is not None
        assert "renamed" in again.project.run_names and copy not in again.project.run_names
        assert again.project.view("renamed")["centre_tab"] == "Time distribution"
    finally:
        again.close()


def test_save_as_refuses_a_folder_that_holds_files(app, tmp_path):
    window = MainWindow(ask=False)
    window.show()
    try:
        assert window.new_project(tmp_path / "example")
        other = tmp_path / "other"
        other.mkdir()
        (other / "notes.txt").write_text("keep me", encoding="utf-8")
        assert not window.save(other)
        assert "already holds files" in window.last_error
        assert (other / "notes.txt").read_text(encoding="utf-8") == "keep me"
        assert not (other / "parameters.csv").exists()
        assert window.save()  # its own folder is fine
        assert window.save(tmp_path / "fresh")  # so is a new one
        assert (tmp_path / "fresh" / "parameters.csv").exists()
    finally:
        window.close()


def test_a_scan_or_monte_carlo_for_changed_settings_is_not_shown(app, tmp_path, monkeypatch):
    window = MainWindow(ask=False)
    window.show()
    try:
        assert window.new_project(tmp_path / "example")
        project = window.project
        assert project is not None
        name = window.current
        assert name is not None
        shown = []
        monkeypatch.setattr(window, "_show_scan", lambda *a: shown.append("scan"))
        monkeypatch.setattr(window, "_show_mc", lambda *a: shown.append("mc"))
        monkeypatch.setattr(project, "store_scan", lambda task, record: False)
        monkeypatch.setattr(project, "store_mc", lambda task, record: False)
        window._scan_done((window.serial, SimpleNamespace(name=name)), None, 1.0)  # ty: ignore[invalid-argument-type]
        assert "fit again" in window.panel.uncertainties.result.text()
        window._mc_done((window.serial, SimpleNamespace(name=name)), None, 1.0)  # ty: ignore[invalid-argument-type]
        assert "fit again" in window.panel.uncertainties.mc_result.text()
        assert shown == []
        assert not window.panel.uncertainties.apply_button.isEnabled()
        wait(app, lambda: settled(window), "fit")
    finally:
        window.close()


def test_the_monte_carlo_button_stays_off_while_one_runs(app, tmp_path):
    window = MainWindow(ask=False)
    window.show()
    try:
        assert window.new_project(tmp_path / "example")
        tab = window.panel.uncertainties
        tab.mc_on.setChecked(True)
        window._mc_busy(True, "")
        assert not tab.mc_button.isEnabled()
        tab._refresh_mc()  # any panel reload
        assert not tab.mc_button.isEnabled()
        window._mc_busy(False, "")
        tab._refresh_mc()
        assert tab.mc_button.isEnabled() == tab.mc_on.isChecked()
        wait(app, lambda: settled(window), "fit")
    finally:
        window.close()


def test_a_law_job_from_a_closed_project_does_not_freeze_the_view(app, tmp_path):
    window = MainWindow(ask=False)
    window.show()
    try:
        assert window.new_project(tmp_path / "example")
        old_serial = window.serial
        window.law_job = object()  # ty: ignore[invalid-assignment]  # a job from the old project
        assert window.new_project(tmp_path / "second")
        assert window.serial != old_serial
        window.law_page.show_message("Fitting…")
        window._laws_done((old_serial, "x"), [], 0.0)
        assert window.law_job is None
        window._laws_failed((old_serial, "x"), "boom")
        assert window.law_job is None
        assert "boom" not in window.last_error
        wait(app, lambda: settled(window), "fit")
    finally:
        window.close()


def test_the_copy_settings_dialog_is_freed_and_no_run_leaves_no_stale_views(
    app, tmp_path, monkeypatch
):
    from forster.gui.dialogs import CopySettingsDialog

    window = MainWindow(ask=False)
    window.show()
    try:
        assert window.new_project(tmp_path / "example")
        monkeypatch.setattr(CopySettingsDialog, "exec", lambda self: 0)
        for _ in range(5):
            window._choose_copy()
        QApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        app.processEvents()
        assert window.findChildren(CopySettingsDialog) == []
        project = window.project
        assert project is not None
        wait(app, lambda: project.result(window.current or "") is not None, "fit")
        window.show_centre("Time distribution")
        window.time_page.figure.text(0.5, 0.5, "old run")
        window.time_page.drawn = True
        window.delete_run()
        assert window.current is None
        assert not window.time_page.drawn and not window.law_page.drawn
        assert not window.export_plot(tmp_path / "stale.png")
        assert not (tmp_path / "stale.png").exists()
        wait(app, lambda: settled(window), "fit")
    finally:
        window.close()


def test_the_data_tab_import_button_opens_the_import(app, tmp_path, monkeypatch):
    window = MainWindow(ask=False)
    window.show()
    try:
        assert window.new_project(tmp_path / "example")
        asked = []
        monkeypatch.setattr(
            "forster.gui.app.QFileDialog.getOpenFileNames",
            lambda *a, **k: asked.append(True) or ([], ""),
        )
        QTest.mouseClick(window.panel.profile.import_button, Qt.MouseButton.LeftButton)
        assert asked == [True]
        wait(app, lambda: settled(window), "fit")
    finally:
        window.close()


def test_the_import_dialog_keeps_the_notes_of_merged_files(app, tmp_path, monkeypatch):
    """The first note of one file is its count (the summary says it); merged files carry
    no count line, so dropping the first note lost a real one."""
    window = MainWindow(ask=False)
    window.show()
    try:
        assert window.new_project(tmp_path / "example")
        files = [str(lab_csv(tmp_path / f"{stem}.csv")) for stem in ("a", "b")]
        # Same profile names in two files are refused; name them apart.
        for stem in ("a", "b"):
            table = pd.read_csv(tmp_path / f"{stem}.csv")
            table["Sample"] = table["Sample"] + stem
            table.to_csv(tmp_path / f"{stem}.csv", index=False)
        monkeypatch.setattr(
            "forster.gui.app.QFileDialog.getOpenFileNames", lambda *a, **k: (files, "")
        )
        seen = []

        def look(self):
            seen.append((self.n_files, self.preview().notes, self.notes.text()))
            return 0

        monkeypatch.setattr(ImportDialog, "exec", look)
        window._choose_import()
        n_files, notes, shown = seen[0]
        assert n_files == 2 and notes
        for note in notes:
            assert note in shown
        wait(app, lambda: settled(window), "fit")
    finally:
        window.close()


def test_hint_text_follows_a_switch_to_dark_mode(app):
    """The gray hints use ``palette(placeholder-text)`` in a style sheet, which Qt resolves
    once: after a light-to-dark switch with the window open they kept the light gray,
    unreadable on the dark background (seen in the dark-mode screenshot)."""
    from PySide6.QtGui import QColor, QPalette
    from PySide6.QtWidgets import QLabel

    role, hint = QPalette.ColorRole, "palette(placeholder-text)"
    before = app.palette()
    light, dark = QPalette(before), QPalette(before)
    light.setColor(role.PlaceholderText, QColor("#808080"))
    dark.setColor(role.Window, QColor("#1f1f1f"))
    dark.setColor(role.WindowText, QColor("#e8e8e8"))
    dark.setColor(role.PlaceholderText, QColor("#a0a0a0"))
    app.setPalette(light)
    window = MainWindow(ask=False)
    window.show()
    try:
        notes = [w for w in window.findChildren(QLabel) if hint in w.styleSheet()]
        assert notes, "no hint labels found"
        app.processEvents()
        assert notes[0].palette().color(role.WindowText).name() == "#808080"
        app.setPalette(dark)
        app.processEvents()
        assert notes[0].palette().color(role.WindowText).name() == "#a0a0a0"
    finally:
        window.close()
        app.setPalette(before)


def answer_quit(window, action, seen, before=None):
    """Click a button of the quit dialog once it is open; ``seen`` collects its text."""

    def click():
        dialog = window.findChild(QuitDialog)
        try:
            assert dialog is not None, "no quit dialog"
            seen.append(dialog.text.text())
            if before is not None:
                before(dialog)
            dialog.findChild(QPushButton, f"quit-{action}").click()
        except Exception as exc:  # a failed assertion must not leave exec() waiting
            seen.append(f"ERROR {exc!r}")
            if dialog is not None:
                dialog.reject()

    QTimer.singleShot(50, click)


def window_with_project(tmp_path):
    window = MainWindow(ask=True)
    window.show()
    assert window.new_project(tmp_path / "example")
    return window


def start_scan_with_priors(app, window):
    """Give the example a prior on the step and a segment, save, and start its scan."""
    wait(app, lambda: settled(window), "fit")
    editor = window.editor
    assert editor.set_step_spread(0, "2") and editor.set_cell(1, "spread", "0.2")
    window.panel.load(editor)
    window._edited()
    wait(app, lambda: settled(window), "refit")
    assert window.save()
    assert window.start_scan()


def test_closing_an_idle_clean_project_asks_nothing(app, tmp_path):
    window = window_with_project(tmp_path)
    wait(app, lambda: settled(window), "fit")
    assert window.save() and not window.project.dirty
    opened = []
    original = QuitDialog.exec
    QuitDialog.exec = lambda self: opened.append(self) or 0
    try:
        assert window.close()
    finally:
        QuitDialog.exec = original
    assert not window.isVisible() and opened == []


def test_closing_with_unsaved_changes_offers_cancel_discard_and_save(app, tmp_path):
    window = window_with_project(tmp_path)
    wait(app, lambda: settled(window), "fit")
    project = window.project
    project.dirty = True
    folder = project.folder
    seen = []

    answer_quit(window, "cancel", seen)
    assert not window.close()
    assert window.isVisible() and project.dirty
    assert "Unsaved changes" in seen[0] and "example" in seen[0] and "Still running" not in seen[0]

    answer_quit(window, "quit", seen)
    assert window.close()
    assert not window.isVisible() and project.dirty and len(seen) == 2
    assert not any(s.startswith("ERROR") for s in seen)

    window = MainWindow(ask=True)
    window.show()
    assert window.open_project(folder)
    reopened = window.project
    assert reopened is not None
    reopened.dirty = True
    answer_quit(window, "save", seen)
    assert window.close()
    assert not window.isVisible() and not reopened.dirty
    assert (folder / "parameters.csv").is_file() and (folder / "project.json").is_file()
    assert not any(s.startswith("ERROR") for s in seen)


def test_closing_during_a_scan_lists_it_and_stop_and_quit_cancels_it(app, tmp_path):
    window = window_with_project(tmp_path)
    start_scan_with_priors(app, window)
    seen = []

    answer_quit(window, "cancel", seen)
    assert not window.close()
    assert window.isVisible() and window.scan_job is not None and not window.scan_cancel.is_set()
    assert "Still running" in seen[0] and f"Starting-profile scan of {EXAMPLE}" in seen[0]
    assert "Unsaved" not in seen[0]

    # Unsaved changes appear in the same dialog; the buttons follow.
    window.project.dirty = True
    labels = []

    def buttons(dialog):
        labels.extend(b.text() for b in dialog.box.buttons())

    answer_quit(window, "quit", seen, before=buttons)
    assert window.close()
    assert not window.isVisible() and window.scan_cancel.is_set()
    assert sorted(labels) == ["Cancel", "Quit without saving", "Save and quit"]
    assert "Still running" in seen[1] and "Unsaved changes" in seen[1]
    assert not any(s.startswith("ERROR") for s in seen)
    assert window.scan_pool.waitForDone(1000)  # the pool was waited for


def test_the_quit_dialog_drops_a_job_that_finishes_while_it_is_open(app, tmp_path):
    window = window_with_project(tmp_path)
    start_scan_with_priors(app, window)
    seen = []

    def stop_scan(dialog):
        window.cancel_scan()
        wait(app, lambda: window.scan_job is None, "scan canceled")
        dialog.refresh()
        seen.append(dialog.text.text())

    answer_quit(window, "quit", seen, before=stop_scan)
    assert window.close()
    assert "Nothing is running" in seen[-1] and not any(s.startswith("ERROR") for s in seen)
