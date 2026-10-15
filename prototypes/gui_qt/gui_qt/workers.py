"""Background jobs: a function run on a QThreadPool, its answer sent back as a signal.

The signals object is created in the GUI thread, so a signal emitted from the worker
reaches its slots through the GUI thread's event loop (a queued connection): slots
never run on the worker. The Numba kernels release the GIL, so the window keeps
painting while a fit runs.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QRunnable, Signal


class JobSignals(QObject):
    done = Signal(object, object, float)  # tag, answer, seconds in the worker
    failed = Signal(object, str)  # tag, message


class Job(QRunnable):
    """Run ``fn()`` on a pool thread; emit ``done(tag, answer, seconds)`` or
    ``failed(tag, message)``."""

    def __init__(self, fn: Callable[[], Any], tag: object) -> None:
        super().__init__()
        self.fn, self.tag = fn, tag
        self.signals = JobSignals()

    def run(self) -> None:
        start = time.perf_counter()
        try:
            answer = self.fn()
        except Exception as exc:  # shown to the user, never raised in the pool thread
            traceback.print_exc()
            self.signals.failed.emit(self.tag, f"{type(exc).__name__}: {exc}")
            return
        self.signals.done.emit(self.tag, answer, time.perf_counter() - start)
