"""Retake the desktop app's documentation screenshots (docs/site/images/app-*.png).

Run from the repository root:

    uv run --extra gui python docs/site/make_app_screenshots.py

It drives the real window offscreen (QT_QPA_PLATFORM=offscreen): the worked example
plus two synthetic profiles, "OL7 prof 1" and "OL7 prof 2" (made-up microprobe numbers,
not real crystals; nothing from any private data set), in a temporary project folder.
Each shot is taken after every worker has finished. The same run gives the same images
apart from the font rendering of the machine.
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PySide6.QtCore import QEventLoop, Qt
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import QApplication, QSplitter, QWidget
from scipy.special import erfc

from forster.gui.app import MainWindow
from forster.gui.dialogs import CopySettingsDialog, ImportDialog
from forster.gui.importing import read_columns
from forster.imports import guess

IMAGES = Path(__file__).resolve().parent / "images"
EXAMPLE = "AZ18 WHT01_Ol_63_prof_1"
SIZE = (1560, 900)
MGO, FEO = 40.3044, 71.8444  # g/mol


def pump(app: QApplication, condition, what: str, timeout: float = 300.0) -> None:
    """Run the event loop until ``condition()`` holds (workers report through it)."""
    start = time.perf_counter()
    while not condition():
        if time.perf_counter() - start > timeout:
            raise RuntimeError(f"timed out waiting for {what}")
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 20)
        time.sleep(0.002)


def probe_table(path: Path) -> pd.DataFrame:
    """Two synthetic microprobe profiles (MgO, FeO, SiO2 in wt%, a Total column, one low
    total in the first profile): Fo falls from the rim inward along an erfc curve."""
    rng = np.random.default_rng(7)
    x = np.arange(0.0, 60.0, 2.0)
    rows = []
    for name, step, width, rim in (
        ("OL7 prof 1", 22.0, 7.0, 0.905),
        ("OL7 prof 2", 30.0, 10.0, 0.915),
    ):
        fo = 0.885 + (rim - 0.885) * 0.5 * erfc((x - step) / width) * 2 / 1.0
        fo = np.minimum(fo, rim) + rng.normal(0, 0.0004, x.size)
        mgo = np.full(x.size, 48.0)
        feo = mgo / MGO * (1 - fo) / fo * FEO
        total = np.full(x.size, 100.1)
        if name == "OL7 prof 1":
            total[9] = 97.0
        rows.append(
            pd.DataFrame(
                {
                    "Sample": name,
                    "Distance (um)": x,
                    "SiO2": total - mgo - feo,
                    "MgO": mgo,
                    "FeO": feo,
                    "Total": total,
                }
            )
        )
    table = pd.concat(rows, ignore_index=True)
    table.to_csv(path, index=False)
    return table


def set_theme(app: QApplication, dark: bool) -> None:
    """Force the light or the dark look (the window follows the application palette)."""
    app.setStyle("Fusion")
    if not dark:
        app.setPalette(app.style().standardPalette())
        return
    colors = {
        QPalette.ColorRole.Window: "#1f1f1f",
        QPalette.ColorRole.WindowText: "#e8e8e8",
        QPalette.ColorRole.Base: "#2b2b2b",
        QPalette.ColorRole.AlternateBase: "#333333",
        QPalette.ColorRole.Text: "#e8e8e8",
        QPalette.ColorRole.Button: "#333333",
        QPalette.ColorRole.ButtonText: "#e8e8e8",
        QPalette.ColorRole.ToolTipBase: "#333333",
        QPalette.ColorRole.ToolTipText: "#e8e8e8",
        QPalette.ColorRole.PlaceholderText: "#7a7a7a",
        QPalette.ColorRole.Highlight: "#2a82da",
        QPalette.ColorRole.HighlightedText: "#ffffff",
    }
    palette = QPalette()
    for role, color in colors.items():
        palette.setColor(role, color)
    app.setPalette(palette)


def shoot(app: QApplication, widget: QWidget, name: str) -> None:
    app.processEvents()
    pixmap = widget.grab()
    assert pixmap.save(str(IMAGES / name), "PNG"), name
    print(f"{name}: {pixmap.width()}x{pixmap.height()}")


def settled(window: MainWindow) -> bool:
    return window.fit_job is None and not window.wanted and not window.debounce.isActive()


def main() -> None:
    app = QApplication.instance() or QApplication([])
    assert isinstance(app, QApplication)
    set_theme(app, dark=False)
    IMAGES.mkdir(exist_ok=True)

    def idle(what: str) -> None:
        pump(app, lambda: settled(window), what)
        window.statusBar().showMessage(f"Fitted {window.current}")
        app.processEvents()

    with tempfile.TemporaryDirectory() as tmp:
        folder = Path(tmp)
        table = probe_table(folder / "probe-session.csv")
        window = MainWindow(ask=False)
        window.resize(*SIZE)
        window.show()
        try:
            assert window.new_project(folder / "example")
            project = window.project
            assert project is not None and window.current == EXAMPLE
            idle("first fit")

            # Two synthetic profiles through the import (Fo from MgO and FeO).
            template = guess(table)
            assert window.import_table(table, template, offset_um=0.0)
            window.fit_all()
            pump(app, lambda: all(project.result(n) is not None for n in project.run_names), "fits")
            idle("fits")
            if window.current != EXAMPLE:
                row = project.run_names.index(EXAMPLE)
                window.run_list.setCurrentRow(row)
                idle("example selected")
            assert window.current == EXAMPLE
            window.show_centre("Profile")

            # Segments tab, then a refused edit (an Fo of 120 mol%).
            window.panel.show_title("Segments")
            shoot(app, window, "app-segments.png")
            seg = window.panel.segments.table
            item = seg.item(0, 3)
            assert item is not None
            item.setText("120")
            assert window.panel.segments.error.text()
            shoot(app, window, "app-refused-edit.png")
            assert window.editor is not None
            window.editor.errors.clear()  # the refusal changed nothing; drop the message
            window.panel.load(window.editor)  # clears the red cell and the message

            # Data tab with two masked points.
            window.panel.show_title("Data")
            data = window.panel.profile.table
            for row in (7, 8):
                check = data.item(row, 2)
                assert check is not None
                check.setCheckState(Qt.CheckState.Unchecked)
            idle("masked refit")
            assert project.points(EXAMPLE).masked.sum() == 2
            shoot(app, window, "app-data-masked.png")

            # The import dialog for the microprobe table (not applied again).
            dialog = ImportDialog(
                folder / "probe-session.csv", read_columns(folder / "probe-session.csv"), window
            )
            dialog.resize(840, 980)
            dialog.each_start.setChecked(True)
            dialog.show()
            shoot(app, dialog, "app-import-dialog.png")
            dialog.close()
            dialog.deleteLater()

            # Monte Carlo on the example (the hero image), then the starting-profile fit.
            window.panel.show_title("Uncertainties")
            u = window.panel.uncertainties
            u.mc_on.setChecked(True)
            for key, text in (("T_sd_C", "15"), ("P_sd_MPa", "50"), ("fO2_sd_log", "0.3")):
                u.mc_fields[key].setText(text)
                u.mc_fields[key].editingFinished.emit()
            window.start_mc()
            pump(app, lambda: window.mc_job is None, "Monte Carlo")
            idle("after Monte Carlo")
            assert "Median" in u.mc_result.text()
            shoot(app, window, "app-monte-carlo.png")

            editor = window.editor
            assert editor is not None
            assert editor.set_step_spread(0, "2")
            window.panel.load(editor)
            window._edited()
            idle("prior refit")
            assert window.start_scan()
            pump(app, lambda: window.scan_job is None, "starting-profile fit")
            assert "Best fit" in u.result.text()
            idle("after scan")
            shoot(app, window, "app-starting-profile-fit.png")

            # The views that use both: the Monte Carlo is run again if the prior dropped it.
            if window.mc_record is None:
                window.start_mc()
                pump(app, lambda: window.mc_job is None, "Monte Carlo again")
            window.time_page.set_mode("all")
            window.show_centre("Time distribution")
            idle("time view")
            shoot(app, window, "app-time-distribution.png")

            window.show_centre("Law comparison")
            pump(app, lambda: window.law_job is None and window.law_page.drawn, "law fits")
            shoot(app, window, "app-law-comparison.png")

            window.show_centre("Profile")
            window.panel.show_title("Advanced")
            idle("advanced")
            shoot(app, window, "app-advanced.png")

            window.show_centre("Results")
            split = window.centralWidget()
            assert isinstance(split, QSplitter)
            split.setSizes([190, 1370, 0])  # settings panel folded away: every column shows
            shoot(app, window, "app-results.png")
            split.setSizes([190, 720, 650])

            # Copy settings dialog: conditions and Monte Carlo settings to the OL7 runs.
            copy = CopySettingsDialog(EXAMPLE, project.run_names, window)
            copy.groups["conditions"].setChecked(True)
            copy.groups["mc"].setChecked(True)
            copy.show()
            shoot(app, copy, "app-copy-settings.png")
            copy.close()
            copy.deleteLater()

            # Dark mode: the profile with the Segments tab, then the time distribution.
            set_theme(app, dark=True)
            window.show_centre("Profile")
            window.panel.show_title("Segments")
            idle("dark")
            shoot(app, window, "app-dark.png")
            window.show_centre("Time distribution")
            idle("dark time view")
            shoot(app, window, "app-dark-time-distribution.png")
        finally:
            window.close()
            set_theme(app, dark=False)


if __name__ == "__main__":
    main()
