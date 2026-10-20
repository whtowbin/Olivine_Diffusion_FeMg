"""Memory, threads, figures and modules of the FORSTER desktop app across a session.

Run (one heavy process at a time; each sequence is a fresh subprocess):
    uv run --with psutil python analyses/gui_memory/measure.py            # 3 repeats + scaling
    uv run --with psutil python analyses/gui_memory/measure.py --repeats 1

Measure only: nothing under src/ is touched. A real offscreen MainWindow, no mocks.
Children print one JSON object on their last line; the parent takes medians and writes
results.json next to this file (the README table is built from it).
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

HERE = Path(__file__).parent
RSS_LIMIT_MB = 4096
START = time.perf_counter()
EXAMPLE = "AZ18 WHT01_Ol_63_prof_1"
CENTRE_VIEWS = ("Profile", "Results", "Time distribution", "Law comparison")


# -- child ------------------------------------------------------------------------------


def snapshot(label, window=None):
    import psutil

    proc = psutil.Process()
    rss = proc.memory_info().rss / 2**20
    if rss > RSS_LIMIT_MB:
        print(json.dumps({"abort": f"RSS {rss:.0f} MB at {label}"}))
        os._exit(3)
    from matplotlib import pyplot
    from matplotlib.figure import Figure

    gc.collect()
    pools = 0
    if window is not None:
        pools = sum(
            p.activeThreadCount()
            for p in (window.fit_pool, window.scan_pool, window.mc_pool, window.law_pool)
        )
    print(f"[{time.strftime('%X')}] {label}", file=sys.stderr, flush=True)
    row = {
        "label": label,
        "rss_mb": round(proc.memory_info().rss / 2**20, 1),
        "threads_py": threading.active_count(),
        "threads_os": proc.num_threads(),
        "pool_threads": pools,
        "pyplot_figs": len(pyplot.get_fignums()),
        "live_figures": sum(1 for o in gc.get_objects() if isinstance(o, Figure)),
        "modules": len(sys.modules),
        "pykrige": "pykrige" in sys.modules,
        "elapsed_s": round(time.perf_counter() - START, 1),
    }
    return row


def lab_csv(path):
    import numpy as np
    import pandas as pd

    mgo_w, feo_w = 40.3044, 71.8444
    x = np.arange(10, 70, 2.0)
    fo = 0.90 - 0.025 * np.exp(-(x - 10) / 8)
    rows = []
    for name, f in (("A", fo), ("B", fo - 0.01)):
        mgo = np.full(len(x), 48.0)
        feo = mgo / mgo_w * (1 - f) / f * feo_w
        total = np.full(len(x), 100.1)
        rows.append(
            pd.DataFrame(
                {"Sample": name, "Distance (um)": x, "SiO2": total - mgo - feo,
                 "MgO": mgo, "FeO": feo, "Total": total}
            )
        )
    pd.concat(rows).to_csv(path, index=False)
    return path


def child_sequence(workdir):
    import importlib

    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    rows = [snapshot("0 interpreter start (psutil, threading only)")]
    t0 = time.perf_counter()
    importlib.import_module("forster.gui.app")
    rows.append(snapshot("1 after import forster.gui.app"))
    rows[-1]["seconds"] = round(time.perf_counter() - t0, 2)

    from PySide6.QtCore import QEventLoop, QPoint, Qt
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QApplication, QDialog

    from forster.gui.app import MainWindow
    from forster.gui.dialogs import ImportDialog
    from forster.gui.importing import read_columns
    from forster.imports import guess

    app = QApplication([])

    def wait(condition, what, timeout=400.0):
        start = time.perf_counter()
        while not condition():
            if time.perf_counter() - start > timeout:
                raise AssertionError(f"timed out waiting for {what}")
            app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 20)
            time.sleep(0.002)

    def settled(w):
        return w.fit_job is None and not w.wanted and not w.debounce.isActive()

    def drag_step(canvas, x0, dx):
        def point(x, y):
            px, py = canvas.ax.transData.transform((x, y))
            r = canvas.device_pixel_ratio
            return QPoint(round(px / r), round((canvas.figure.bbox.height - py) / r))

        low, high = canvas.ax.get_ylim()
        y = (low + high) / 2
        left, none = Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier
        QTest.mousePress(canvas, left, none, point(x0, y))
        for i in range(1, 7):
            QTest.mouseMove(canvas, point(x0 + dx * i / 6, y))
        QTest.mouseRelease(canvas, left, none, point(x0 + dx, y))

    window = MainWindow(ask=False)
    window.show()
    assert window.new_project(Path(workdir) / "example")
    project = window.project
    wait(lambda: project.result(EXAMPLE) is not None and settled(window), "first fit")
    app.processEvents()
    rows.append(snapshot("2 window open, first fit settled", window))

    drag_step(window.canvas, 24.0, 3.0)
    wait(lambda: project.result(EXAMPLE) is not None and settled(window), "drag refit")
    rows.append(snapshot("3 after refit following a drag", window))

    editor = window.editor
    assert editor.set_step_spread(0, "2") and editor.set_cell(1, "spread", "0.2")
    window.panel.load(editor)
    window._edited()
    wait(lambda: project.result(EXAMPLE) is not None and settled(window), "prior refit")
    assert window.start_scan()
    wait(lambda: window.scan_job is None, "starting-profile fit", timeout=600)
    rows.append(snapshot("4 after starting-profile fit (scan)", window))

    u = window.panel.uncertainties
    u.mc_on.setChecked(True)
    for key, text in (("T_sd_C", "15"), ("samples", "200")):
        u.mc_fields[key].setText(text)
        u.mc_fields[key].editingFinished.emit()
    assert window.start_mc()
    wait(lambda: window.mc_job is None, "Monte Carlo", timeout=600)
    rows.append(snapshot("5 after Monte Carlo (200 draws)", window))

    window.show_centre("Law comparison")
    wait(lambda: window.law_job is None and window.law_page.drawn, "law fits", timeout=600)
    rows.append(snapshot("6 after Law comparison drawn", window))

    window.time_page.set_mode("all")
    window.show_centre("Time distribution")
    assert window.time_page.drawn
    app.processEvents()
    rows.append(snapshot("7 after Time distribution drawn", window))

    # Second profile through the import path (not in the numbered list), then fit it.
    path = lab_csv(Path(workdir) / "lab.csv")
    table = read_columns(path)
    window.show_centre("Profile")
    assert window.import_table(table, guess(table), flip=False, offset_um=10.0)
    wait(lambda: project.result("A") is not None and settled(window), "fit of A", timeout=600)
    rows.append(snapshot("7b after importing 2 profiles (A, B) and fitting A", window))

    for _ in range(20):
        dialog = ImportDialog(path, table, window)  # parented to the window, as the app does
        dialog.show()
        app.processEvents()
        dialog.close()
        app.processEvents()
    rows.append(snapshot("8 after opening/closing ImportDialog x20", window))
    rows[-1]["dialog_children"] = len(window.findChildren(QDialog))

    for _ in range(20):
        for title in window.panel.titles:
            window.panel.show_title(title)
            app.processEvents()
        for title in CENTRE_VIEWS:
            window.show_centre(title)
            app.processEvents()
            if title == "Law comparison":
                wait(lambda: window.law_job is None, "law fits", timeout=600)
    window.show_centre("Profile")
    rows.append(snapshot("9 after switching all centre views and side tabs x20", window))

    window.close()
    app.processEvents()
    gc.collect()
    rows.append(snapshot("10 after window.close() + processEvents + gc", window))
    window.deleteLater()
    app.sendPostedEvents(None, 0)
    app.processEvents()
    del window, project, editor, u
    gc.collect()
    rows.append(snapshot("10b after deleteLater + del references + gc"))
    time.sleep(32)  # QThreadPool expiry is 30 s: do idle worker threads go away?
    rows.append(snapshot("10c 32 s later (idle pool threads expired?)"))
    return rows


def child_scale(workdir, n):
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PySide6.QtCore import QEventLoop
    from PySide6.QtWidgets import QApplication

    from forster.gui.app import MainWindow

    app = QApplication([])

    def wait(condition, what, timeout=900.0):
        start = time.perf_counter()
        while not condition():
            if time.perf_counter() - start > timeout:
                raise AssertionError(f"timed out waiting for {what}")
            app.processEvents(QEventLoop.ProcessEventsFlag.AllEvents, 20)
            time.sleep(0.002)

    window = MainWindow(ask=False)
    window.show()
    assert window.new_project(Path(workdir) / "example")
    project = window.project
    wait(lambda: project.result(EXAMPLE) is not None and window.fit_job is None
         and not window.wanted and not window.debounce.isActive(), "first fit")
    for _ in range(n - 1):
        window.duplicate_run()
    names = list(project.run_names)
    wait(lambda: all(project.result(r) is not None for r in names)
         and window.fit_job is None and not window.wanted, "all fits", timeout=900)
    app.processEvents()
    row = snapshot(f"{n} runs, all fitted", window)
    row["runs"] = len(project.run_names)
    window.close()
    return [row]


# -- parent -----------------------------------------------------------------------------


def run_child(*args):
    with tempfile.TemporaryDirectory() as tmp:
        env = {**os.environ, "QT_QPA_PLATFORM": "offscreen"}
        proc = subprocess.run(
            [sys.executable, str(Path(__file__)), "--child", *args, "--workdir", tmp],
            capture_output=True, text=True, env=env,
        )
    last = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    if proc.returncode != 0 or not last.startswith(("[", "{")):
        raise SystemExit(f"child {args} failed ({proc.returncode}):\n{proc.stdout}\n{proc.stderr[-3000:]}")
    data = json.loads(last)
    if isinstance(data, dict):
        raise SystemExit(f"child aborted: {data}")
    return data


def median_rows(runs):
    out = []
    for i, first in enumerate(runs[0]):
        row = dict(first)
        for key, value in first.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            row[key] = round(statistics.median(r[i][key] for r in runs), 1)
        row["pykrige"] = any(r[i]["pykrige"] for r in runs)
        row["rss_all"] = [r[i]["rss_mb"] for r in runs]
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--child", choices=["sequence", "scale"])
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--workdir")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--scales", default="1,5,10")
    a = ap.parse_args()
    if a.child == "sequence":
        print(json.dumps(child_sequence(a.workdir)))
        return
    if a.child == "scale":
        print(json.dumps(child_scale(a.workdir, a.n)))
        return
    runs = [run_child("sequence") for _ in range(a.repeats)]
    scale = {}
    for n in map(int, a.scales.split(",")):
        scale[n] = median_rows([run_child("scale", "--n", str(n)) for _ in range(a.repeats)])
    result = {"sequence": median_rows(runs), "scale": scale}
    (HERE / "results.json").write_text(json.dumps(result, indent=1))
    prev = None
    for r in result["sequence"]:
        d = "" if prev is None else f"{r['rss_mb'] - prev:+.1f}"
        print(f"{r['label']:<62} {r['rss_mb']:>8} {d:>8} thr={r['threads_py']}/{r['threads_os']} "
              f"pool={r['pool_threads']} figs={r['pyplot_figs']}/{r['live_figures']} "
              f"mods={r['modules']} krige={r['pykrige']}")
        prev = r["rss_mb"]
    for n, rs in scale.items():
        r = rs[0]
        print(f"scale {n}: rss={r['rss_mb']} threads={r['threads_py']} figs={r['live_figures']} mods={r['modules']}")


if __name__ == "__main__":
    main()
