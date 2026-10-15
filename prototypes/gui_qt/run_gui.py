"""Launcher for PyInstaller (and ``python prototypes/gui_qt/run_gui.py``)."""

import multiprocessing
import sys

from gui_qt.__main__ import main

if __name__ == "__main__":
    multiprocessing.freeze_support()
    sys.exit(main())
