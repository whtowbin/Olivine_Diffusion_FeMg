"""The desktop app, headless (``QT_QPA_PLATFORM=offscreen``): open the template example,
fit, edit a segment in the side table and by dragging on the plot, refit, fit the
starting profile, save, reopen and see the same state from the cache. Skipped without
PySide6 (the ``gui`` extra)."""

import os
import time

import pytest

pytest.importorskip("PySide6")
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QEventLoop, QPoint, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from forster.gui.app import MainWindow
from forster.gui.state import RunEditor

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
