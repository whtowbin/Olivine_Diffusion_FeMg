"""Background jobs: a function run on a QThreadPool, its answer sent back as a signal.

The signals object is created in the GUI thread, so a signal emitted from the worker
reaches its slots through the GUI thread's event loop (a queued connection): slots never
run on the worker. The Numba kernels release the GIL, so the window keeps painting while
a fit runs. Fits collect their warnings as data (`forster.notes`), never through
``warnings.catch_warnings``, which is not thread-safe.
"""

from __future__ import annotations

import time
import traceback
from collections.abc import Callable
from typing import Any

from PySide6.QtCore import QObject, QRunnable, Signal

from forster.sampling import Cancelled


class JobSignals(QObject):
    done = Signal(object, object, float)  # tag, answer, seconds in the worker
    failed = Signal(object, str)  # tag, message
    cancelled = Signal(object)  # tag
    progress = Signal(object, str)  # tag, a stage or a count, from the worker


class Job(QRunnable):
    """Run ``fn(job)`` on a pool thread; emit ``done``, ``failed`` or ``cancelled``.
    ``fn`` may call ``job.report(text)`` to send progress."""

    def __init__(self, fn: Callable[[Job], Any], tag: object) -> None:
        super().__init__()
        self.setAutoDelete(False)  # the window keeps it until it answers
        self.fn, self.tag = fn, tag
        self.signals = JobSignals()

    def report(self, text: str) -> None:
        self.signals.progress.emit(self.tag, text)

    def run(self) -> None:
        start = time.perf_counter()
        try:
            answer = self.fn(self)
        except Cancelled:
            self.signals.cancelled.emit(self.tag)
            return
        except Exception as exc:  # shown to the user, never raised in the pool thread
            traceback.print_exc()
            self.signals.failed.emit(self.tag, str(exc) or type(exc).__name__)
            return
        self.signals.done.emit(self.tag, answer, time.perf_counter() - start)
