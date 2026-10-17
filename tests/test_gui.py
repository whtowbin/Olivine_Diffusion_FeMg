"""The desktop app, headless (``QT_QPA_PLATFORM=offscreen``): open the template example,
fit, edit a segment in the side table and by dragging on the plot, refit, fit the
starting profile, save, reopen and see the same state from the cache. Skipped without
PySide6 (the ``gui`` extra)."""

import os
import time
import zipfile

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from forster.gui.app import MainWindow
from forster.gui.dialogs import ImportDialog
from forster.gui.importing import read_columns
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
        assert any("conditions and law" in t.get_text() for t in legend.texts)
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
