"""A self-test of the app as installed, for the installers built in CI: it drives the
real window headless (``QT_QPA_PLATFORM=offscreen``) through a new project, a fit, a
starting-profile fit, a Monte Carlo, the law comparison, exports, save and reopen, and
checks the files a bundle tends to lose (forster's data and metadata, scipy's Sobol
direction numbers). ``startup_probe`` quits once the window has painted, to time a
cold start.

Run by the bundle's launcher (``packaging/gui/run_gui.py --self-test DIR``); also
importable in a development checkout.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path

from PySide6.QtCore import QEventLoop, QTimer
from PySide6.QtWidgets import QApplication

from forster.project import Project

from .app import MainWindow

EXAMPLE = "AZ18 WHT01_Ol_63_prof_1"


class SelfTestError(AssertionError):
    pass


def check(condition: bool, what: str) -> None:
    if not condition:
        raise SelfTestError(what)
    print(f"  ok  {what}", flush=True)


def wait_until(predicate: Callable[[], bool], timeout: float, what: str) -> float:
    """Run the event loop until ``predicate()``; seconds waited."""
    app = QApplication.instance()
    assert app is not None
    start = time.perf_counter()
    while not predicate():
        if time.perf_counter() - start > timeout:
            raise SelfTestError(f"timed out after {timeout:.0f} s waiting for {what}")
        app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 10)
        time.sleep(0.001)
    return time.perf_counter() - start


def _settled(window: MainWindow) -> bool:
    return window.fit_job is None and not window.wanted and not window.debounce.isActive()


def check_bundle_files() -> None:
    """What PyInstaller misses unless told: forster's data and metadata, and scipy's
    Sobol direction numbers (without them the sampler prints an error and carries on)."""
    import scipy.stats

    check(bool(version("forster")), f"forster metadata (version {version('forster')})")
    data = files("forster") / "data"
    check(any(p.name.endswith(".csv") for p in data.iterdir()), "forster/data tables")
    sobol = Path(scipy.stats.__file__).parent / "_sobol_direction_numbers.npz"
    check(sobol.exists(), "scipy's Sobol direction numbers")


def self_test(folder: Path) -> int:
    start = time.perf_counter()
    check_bundle_files()
    window = MainWindow(ask=False)
    window.show()
    project_dir = folder / "example-project"
    try:
        check(window.new_project(project_dir), f"new project from the example ({project_dir})")
        project = window.project
        assert project is not None
        waited = wait_until(
            lambda: project.result(EXAMPLE) is not None and _settled(window), 300, "the first fit"
        )
        result = project.result(EXAMPLE)
        check(result is not None and result.time_days > 0, f"fitted ({waited:.1f} s)")
        editor = window.editor
        assert editor is not None
        check(editor.set_step_spread(0, "2"), "a prior on the step")
        window.panel.load(editor)
        window._edited()
        wait_until(lambda: project.result(EXAMPLE) is not None and _settled(window), 120, "refit")
        check(window.start_scan(), "starting-profile fit started")
        wait_until(lambda: window.scan_job is None, 600, "the starting-profile fit")
        check(window.scan_record is not None, "starting-profile fit done")
        fields = {**editor.mc_text(), "on": "yes", "T_sd_C": "10", "samples": "200"}
        check(editor.set_mc(fields, "T_sd_C"), "Monte Carlo settings")
        window.panel.load(editor)
        check(window.start_mc(), "Monte Carlo started")
        wait_until(lambda: window.mc_job is None, 300, "the Monte Carlo")
        check(project.mc(EXAMPLE) is not None, "Monte Carlo done")
        window.show_centre("Law comparison")
        wait_until(lambda: window.law_job is None and window.law_page.drawn, 300, "law fits")
        check(window.export_plot(folder / "laws.png"), "law comparison exported (PNG)")
        window.show_centre("Profile")
        check(window.export_plot(folder / "profile.pdf"), "plot exported (PDF)")
        check(window.export_zip(folder / "project.zip"), "project exported (zip)")
        check(window.save(), "saved")
    finally:
        window.close()
    again = Project.open(project_dir)
    check(again.result(EXAMPLE) is not None and again.mc(EXAMPLE) is not None, "reopened")
    print(f"self-test passed in {time.perf_counter() - start:.1f} s", flush=True)
    return 0


def startup_probe(window: MainWindow) -> None:
    """Quit as soon as the window has been painted (to time a cold start)."""
    app = QApplication.instance()
    assert app is not None
    QTimer.singleShot(0, app.quit)
    window.repaint()
