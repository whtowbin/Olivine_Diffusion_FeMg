"""Headless self-test and measurements, driving the real window with synthetic mouse and
keyboard events (run with ``QT_QPA_PLATFORM=offscreen``).

``self_test``: new project from the example, fit, drag the step, a refused edit, the
initial-condition posterior (and a cancelled one), save, close, reopen, and the cached
result shown without a refit. ``measure``: drag-to-redraw latency, and how late a 5 ms
timer runs while a long scan holds a worker thread.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QEvent, QEventLoop, QObject, QPoint, Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from forster.scan import Uncertainty

from .app import MainWindow


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


def settle(ms: int = 50) -> None:
    QTest.qWait(ms)


def _widget_point(window: MainWindow, x: float, y: float) -> QPoint:
    """Data coordinates to canvas widget pixels (matplotlib counts from the bottom, in
    physical pixels)."""
    canvas = window.canvas
    disp_x, disp_y = canvas.ax.transData.transform((x, y))
    ratio = canvas.device_pixel_ratio
    height = canvas.figure.bbox.height
    return QPoint(round(disp_x / ratio), round((height - disp_y) / ratio))


def drag(window: MainWindow, handle: str, delta: float, steps: int = 6) -> None:
    """Press on a handle, move it by ``delta`` (µm or mol%) and release."""
    canvas = window.canvas
    values, (left, right) = canvas.values, canvas.domain
    low, high = canvas.ax.get_ylim()
    if handle == "step":
        x0, y0, dx, dy = values["step"], (low + high) / 2, delta, 0.0
    elif handle == "edge":
        x0, y0, dx, dy = (left + values["step"]) / 2, values["edge"], 0.0, delta
    else:
        x0, y0, dx, dy = (values["step"] + right) / 2, values["core"], 0.0, delta
    left_button, none = Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier
    QTest.mousePress(canvas, left_button, none, _widget_point(window, x0, y0))
    for i in range(1, steps + 1):
        f = i / steps
        QTest.mouseMove(canvas, _widget_point(window, x0 + f * dx, y0 + f * dy))
    QTest.mouseRelease(canvas, left_button, none, _widget_point(window, x0 + dx, y0 + dy))


def type_into(window: MainWindow, handle: str, text: str) -> None:
    edit = window.inputs[handle]
    edit.setFocus()
    edit.selectAll()
    QTest.keyClicks(edit, text)
    QTest.keyClick(edit, Qt.Key.Key_Return)


def grab(window: MainWindow, folder: Path | None, name: str) -> None:
    if folder is None:
        return
    folder.mkdir(parents=True, exist_ok=True)
    settle(100)
    window.grab().save(str(folder / f"{name}.png"))
    print(f"  screenshot {folder / name}.png", flush=True)


def _fit_settled(window: MainWindow) -> bool:
    return not window._fit_running and window._pending is None


def self_test(folder: Path, screenshots: Path | None = None) -> int:
    start = time.perf_counter()
    window = MainWindow()
    window.ask = False
    window.show()
    project_dir = folder / "example-project"
    try:
        print("New project from the example", flush=True)
        check(window.new_project(project_dir), f"Project.new({project_dir})")
        project = window.project
        assert project is not None
        name = window.current
        check(name == project.run_names[0], f"first run selected ({name})")
        assert name is not None
        waited = wait_until(lambda: project.result(name) is not None, 120, "the first fit")
        wait_until(lambda: _fit_settled(window), 30, "the fit queue")
        first = project.result(name)
        assert first is not None
        check(first.time_days > 0, f"fitted: {first.time_days:.2f} days ({waited:.1f} s)")
        grab(window, screenshots, "1-fitted")

        print("Drag the step 4 µm into the crystal", flush=True)
        step_before = project.run(name).initial.steps_um[0]
        n_latencies = len(window.latencies)
        drag(window, "step", 4.0)
        wait_until(lambda: len(window.latencies) > n_latencies, 60, "the refit after the drag")
        step_after = project.run(name).initial.steps_um[0]
        check(abs(step_after - step_before - 4.0) < 0.5, f"step {step_before} -> {step_after} µm")
        refit = project.result(name)
        check(refit is not None and refit.time_days != first.time_days, "refitted after release")
        latency = window.latencies[-1]["total_ms"]
        check(latency > 0, f"release to redraw {latency:.0f} ms")
        check(project.dirty and window.windowTitle().endswith("*"), "marked unsaved")

        print("Type a refused value", flush=True)
        before = project.run(name)
        type_into(window, "edge", "120")
        error = window.errors["edge"].text()
        check("fractions" in error, f"error shown next to the field: {error!r}")
        check(project.run(name) == before, "the run is unchanged")
        check(window.inputs["edge"].text() == f"{before.initial.edge_fo * 100:.2f}", "field reset")
        grab(window, screenshots, "2-refused-edit")
        type_into(window, "core", f"{before.initial.core_fo * 100 + 0.1:.2f}")
        check(window.errors["edge"].text() == "", "an accepted value clears the error")
        wait_until(lambda: project.result(name) is not None, 60, "the refit after typing")

        print("Fit initial condition, cancelled", flush=True)
        window.start_scan()
        check(window.busy.isVisible() and window.cancel_button.isVisible(), "busy indicator")
        window.cancel_scan()
        check(window.scan_button.isEnabled(), "cancel returns the button at once")
        wait_until(lambda: window.scans_dropped == 1, 120, "the cancelled scan to finish")
        check(window._scan_best is None, "the cancelled scan's answer was dropped")

        print("Fit initial condition", flush=True)
        wait_until(lambda: _fit_settled(window), 30, "the fit queue")
        window.start_scan()
        grab(window, screenshots, "3-scanning")
        wait_until(lambda: window._scan_token is None, 300, "the scan")
        check(window._scan_best is not None, f"posterior: {window.scan_label.text()[:80]}…")
        check(set(project.run(name).uncertainties) == {"step1_um", "edge_fo", "core_fo"}, "priors")
        window.apply_scan()
        wait_until(lambda: _fit_settled(window), 60, "the refit at the best starting profile")
        wait_until(lambda: project.result(name) is not None, 60, "the refit result")
        grab(window, screenshots, "4-posterior-applied")

        print("Save, close, reopen", flush=True)
        saved_run, saved = project.run(name), project.result(name)
        assert saved is not None
        window.save()
        check(not project.dirty and (project_dir / "parameters.csv").exists(), "saved")
        check(window.close_project() and window.project is None, "closed")
        fits = window.fits_started
        check(window.open_project(project_dir), "reopened")
        again = window.project
        assert again is not None
        settle(200)
        check(window.current == name, "the selected run is restored")
        check(again.run(name) == saved_run, "the run is the same")
        cached = again.result(name)
        check(cached is not None and cached.time_days == saved.time_days, "cached result")
        check(window.fits_started == fits, "no refit on reopen")
        check("cache" in window.timing_label.text(), "shown from the cache")
        grab(window, screenshots, "5-reopened")
    except SelfTestError as exc:
        print(f"SELF-TEST FAILED: {exc}", flush=True)
        if window.last_error:
            print(f"  last error: {window.last_error}", flush=True)
        return 1
    finally:
        window.ask = False
        window.close()
    print(f"SELF-TEST PASSED in {time.perf_counter() - start:.1f} s", flush=True)
    return 0


class Heartbeat:
    """A 5 ms timer on the GUI thread; how late each beat is shows how long the event
    loop was blocked."""

    def __init__(self, interval_ms: int = 5) -> None:
        self.interval = interval_ms
        self.timer = QTimer()
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self._beat)
        self.late_ms: list[float] = []
        self._last: float | None = None

    def _beat(self) -> None:
        now = time.perf_counter()
        if self._last is not None:
            self.late_ms.append((now - self._last) * 1000 - self.interval)
        self._last = now

    def start(self) -> None:
        self.late_ms, self._last = [], None
        self.timer.start(self.interval)

    def stop(self) -> dict[str, float]:
        self.timer.stop()
        late = sorted(self.late_ms)
        return {
            "beats": len(late),
            "median_late_ms": statistics.median(late),
            "p99_late_ms": late[int(0.99 * (len(late) - 1))],
            "max_late_ms": late[-1],
        }


def measure(folder: Path, out: Path | None, drags: int = 12) -> int:
    """Drag-to-redraw latency and responsiveness during a long scan; prints JSON."""
    window = MainWindow()
    window.ask = False
    window.show()
    window.new_project(folder / "measure-project")
    project = window.project
    assert project is not None and window.current is not None
    name = window.current
    t_first = wait_until(lambda: project.result(name) is not None, 120, "the first fit")
    wait_until(lambda: _fit_settled(window), 30, "the fit queue")
    results: dict = {"first_fit_s": t_first, "python": sys.version.split()[0]}

    window.latencies.clear()
    for i in range(drags):
        n = len(window.latencies)
        # +2.3, -1.1, ...: every release lands on a position not fitted before.
        drag(window, "step", 2.3 if i % 2 == 0 else -1.1)
        wait_until(lambda n=n: len(window.latencies) > n, 60, "a refit")
        settle(30)
    fresh = [s for s in window.latencies if s["fit_ms"] > 0]
    results["drag"] = {
        "n": len(window.latencies),
        "n_refit": len(fresh),
        "median_total_ms": statistics.median(s["total_ms"] for s in fresh),
        "max_total_ms": max(s["total_ms"] for s in fresh),
        "median_fit_ms": statistics.median(s["fit_ms"] for s in fresh),
        "median_draw_ms": statistics.median(s["draw_ms"] for s in fresh),
        "samples": window.latencies,
    }

    beat = Heartbeat()
    beat.start()
    wait_until(lambda: len(beat.late_ms) >= 600, 30, "600 idle heartbeats")
    results["heartbeat_idle"] = beat.stop()

    # A long scan: wide priors, nested sampling.
    wide = {
        "step1_um": Uncertainty(8.0),
        "edge_fo": Uncertainty(0.01),
        "core_fo": Uncertainty(0.01),
    }
    project.update_run(name, uncertainties=wide, scan_method="nested")
    window._request_fit(name)
    wait_until(lambda: _fit_settled(window), 60, "the refit after the priors")
    beat.start()
    started = time.perf_counter()
    window.start_scan()
    wait_until(lambda: window._scan_token is None, 600, "the wide scan")
    results["scan_s"] = time.perf_counter() - started
    results["heartbeat_during_scan"] = beat.stop()
    results["scan_label"] = window.scan_label.text()
    window.close()
    text = json.dumps(results, indent=2, default=float)
    print(text)
    if out is not None:
        out.write_text(text + "\n")
    return 0


class _FirstPaint(QObject):
    """Quits the application after the plot canvas has painted once."""

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if event.type() == QEvent.Type.Paint:
            QTimer.singleShot(0, self._quit)  # after this paint has finished
        return False

    def _quit(self) -> None:
        print("first window painted", flush=True)
        QApplication.quit()


def startup_probe(window: MainWindow) -> None:
    """Quit as soon as the window's plot has first painted (for timing a cold start)."""
    probe = _FirstPaint(window)
    window.canvas.installEventFilter(probe)
