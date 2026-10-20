"""Launcher of the FORSTER desktop app in a PyInstaller bundle (and
``python packaging/gui/run_gui.py`` in a checkout).

    run_gui.py [FOLDER]            # the app, on a project folder if given
    run_gui.py --self-test DIR     # headless self-test (forster.gui.selftest), exit code
    run_gui.py --startup-probe     # quit once the window has painted (cold start timing)
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import traceback
from pathlib import Path


def frozen_setup() -> None:
    """In a bundle, keep matplotlib's font cache between launches.

    PyInstaller's matplotlib runtime hook points MPLCONFIGDIR at a new temporary folder
    on every launch (meant for one-file builds, whose paths change), so the font cache
    is rebuilt at every start (about 15 s). This is a one-folder build, so a fixed
    folder is safe. Numba needs nothing: a frozen app caches its kernels in Numba's
    user-wide folder, so only the first fit after installing is slower."""
    if not getattr(sys, "frozen", False):
        return
    if sys.platform == "darwin":
        root = Path.home() / "Library" / "Caches"
    elif sys.platform == "win32":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home()))
    else:
        root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    os.environ["MPLCONFIGDIR"] = str(root / "forster-gui" / "matplotlib")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="forster-gui", description=__doc__)
    parser.add_argument("folder", nargs="?", help="project folder to open")
    parser.add_argument("--self-test", metavar="DIR", type=Path)
    parser.add_argument("--startup-probe", action="store_true")
    args = parser.parse_args(argv)
    frozen_setup()
    if sys.stdout is None:  # a windowed Windows build has no console: log to a file
        if args.self_test:
            args.self_test.mkdir(parents=True, exist_ok=True)
        path = args.self_test / "forster_gui.log" if args.self_test else os.devnull
        log = open(path, "w", encoding="utf-8")  # noqa: SIM115 (open for the whole run)
        sys.stdout = sys.stderr = log
    os.environ.setdefault("QT_API", "pyside6")

    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv[:1])
    app.setApplicationName("FORSTER")

    from forster.gui import selftest
    from forster.gui.app import MainWindow

    if args.self_test is not None:
        args.self_test.mkdir(parents=True, exist_ok=True)
        try:
            return selftest.self_test(args.self_test)
        except Exception:
            # Printed and returned as a failure: an unhandled exception in a windowed
            # Windows build opens an error dialog that no one closes on a headless runner,
            # so the self-test hung until its 20-minute timeout (2026-10-20).
            traceback.print_exc()
            return 1
    window = MainWindow()
    window.show()
    if args.folder:
        window.open_project(args.folder)
    if args.startup_probe:
        selftest.startup_probe(window)
    return app.exec()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
