"""``python -m gui_qt [folder]``; ``--self-test DIR``, ``--measure DIR``,
``--startup-probe``. Headless: set ``QT_QPA_PLATFORM=offscreen``."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path


def _frozen_setup() -> None:
    """In a PyInstaller bundle, keep matplotlib's font cache between launches.

    PyInstaller's matplotlib runtime hook points MPLCONFIGDIR at a new temporary folder
    on every launch (meant for one-file builds, whose paths change), so the font cache
    is rebuilt at every start. This is a one-folder build, so a fixed folder is safe.
    Numba needs nothing: in a frozen app it caches the kernels in its user-wide folder,
    keyed to the executable."""
    if not getattr(sys, "frozen", False):
        return
    if sys.platform == "darwin":
        root = Path.home() / "Library" / "Caches"
    elif sys.platform == "win32":
        root = Path(os.environ.get("LOCALAPPDATA", Path.home()))
    else:
        root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    os.environ["MPLCONFIGDIR"] = str(root / "forster-gui-qt" / "matplotlib")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gui_qt", description=__doc__)
    parser.add_argument("folder", nargs="?", help="project folder to open")
    parser.add_argument("--self-test", metavar="DIR", type=Path)
    parser.add_argument("--screenshots", metavar="DIR", type=Path)
    parser.add_argument("--measure", metavar="DIR", type=Path)
    parser.add_argument("--measure-out", metavar="JSON", type=Path)
    parser.add_argument("--startup-probe", action="store_true")
    args = parser.parse_args(argv)
    _frozen_setup()
    if sys.stdout is None:  # a windowed Windows build has no console: log to a file
        logdir = args.self_test or args.measure
        if logdir:
            logdir.mkdir(parents=True, exist_ok=True)
        path = logdir / "gui_qt.log" if logdir else os.devnull
        log = open(path, "w", encoding="utf-8")  # noqa: SIM115 (open for the whole run)
        sys.stdout = sys.stderr = log

    from PySide6.QtWidgets import QApplication

    app = QApplication(sys.argv[:1])
    app.setApplicationName("FORSTER (Qt prototype)")

    from . import selftest
    from .app import MainWindow

    if args.self_test is not None:
        args.self_test.mkdir(parents=True, exist_ok=True)
        folder = Path(tempfile.mkdtemp(dir=args.self_test, prefix="selftest-"))
        return selftest.self_test(folder, args.screenshots)
    if args.measure is not None:
        args.measure.mkdir(parents=True, exist_ok=True)
        folder = Path(tempfile.mkdtemp(dir=args.measure, prefix="measure-"))
        return selftest.measure(folder, args.measure_out)

    window = MainWindow()
    window.show()
    if args.folder:
        window.open_project(args.folder)
    if args.startup_probe:
        selftest.startup_probe(window)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
