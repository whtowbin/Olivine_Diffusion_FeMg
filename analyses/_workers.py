"""Process pools whose workers cannot outlive their parent.

Copied from H_Diffusion_Reaction's ``diffreact.workers``; the analysis scripts import it after
putting ``analyses/`` on ``sys.path``. FORSTER itself starts no processes.

A parent that is SIGKILLed (a timed-out or stopped background job, an out-of-memory kill) never
runs ``finally`` or ``__exit__``, and macOS has no ``PR_SET_PDEATHSIG``, so a worker busy in a
long fit keeps running with PPID 1. One such worker spun at 100% CPU for 2.6 days (2026-10-28, in
H_Diffusion_Reaction). Each worker made here starts a watchdog thread that exits the process once
its parent pid changes.

    with process_pool(n, "fork") as pool: ...   # multiprocessing.Pool; __exit__ terminates
    with process_executor(n) as ex: ...          # concurrent.futures.ProcessPoolExecutor

The start method is the caller's choice and changes nothing else; ``None`` keeps the platform
default. A plain ``ProcessPoolExecutor`` block that raises (Ctrl-C, SIGTERM turned into
SystemExit) still waits for every queued and running job; ``process_executor`` cancels the queue
and terminates the workers.
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import multiprocessing.pool
import os
import threading
import time
from collections.abc import Generator
from concurrent.futures import ProcessPoolExecutor

POLL_S = 2.0


def die_with_parent(parent_pid: int) -> None:
    """Pool initializer: exit this worker (``os._exit``, no cleanup) once ``parent_pid`` is no
    longer its parent."""

    def watch() -> None:
        while os.getppid() == parent_pid:
            time.sleep(POLL_S)
        os._exit(1)

    threading.Thread(target=watch, daemon=True, name="die-with-parent").start()


def process_pool(
    processes: int | None = None, context: str | None = None
) -> multiprocessing.pool.Pool:
    """``multiprocessing`` Pool whose workers exit if this process dies."""
    return mp.get_context(context).Pool(
        processes, initializer=die_with_parent, initargs=(os.getpid(),)
    )


def stop_executor(ex: ProcessPoolExecutor) -> None:
    """Cancel the queued jobs and terminate the workers, without waiting for running jobs."""
    # read the workers first: shutdown() sets _processes to None
    # (Python 3.14's terminate_workers() is the public form)
    procs = list((getattr(ex, "_processes", None) or {}).values())
    ex.shutdown(wait=False, cancel_futures=True)
    for proc in procs:
        proc.terminate()


@contextlib.contextmanager
def process_executor(
    max_workers: int | None = None, context: str | None = None
) -> Generator[ProcessPoolExecutor, None, None]:
    """``ProcessPoolExecutor`` whose workers exit if this process dies; an exception in the
    block stops it at once."""
    ex = ProcessPoolExecutor(
        max_workers,
        mp_context=mp.get_context(context),
        initializer=die_with_parent,
        initargs=(os.getpid(),),
    )
    try:
        yield ex
    except BaseException:
        stop_executor(ex)
        raise
    ex.shutdown(wait=True)
