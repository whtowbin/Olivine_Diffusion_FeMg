"""`analyses/_workers.py` with real processes: workers exit when their parent is SIGKILLed (an
unguarded control must survive), and an exception in a `process_executor` block stops it at once."""

import contextlib
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

ANALYSES = Path(__file__).resolve().parents[1] / "analyses"
sys.path.insert(0, str(ANALYSES))
from _workers import POLL_S, process_executor  # noqa: E402  # ty: ignore[unresolved-import]

PARENT = textwrap.dedent("""
    import multiprocessing as mp, os, signal, sys, time
    from concurrent.futures import ProcessPoolExecutor

    sys.path.insert(0, sys.argv[4])
    import _workers as W

    if __name__ == "__main__":
        kind, ctx, guarded = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
        if kind == "pool":
            pool = W.process_pool(2, ctx) if guarded else mp.get_context(ctx).Pool(2)
            for _ in range(2):
                pool.apply_async(time.sleep, (60,))
        else:
            cm = W.process_executor(2, ctx)  # keep a reference: a collected generator closes it
            plain = ProcessPoolExecutor(2, mp_context=mp.get_context(ctx))
            ex = cm.__enter__() if guarded else plain
            for _ in range(2):
                ex.submit(time.sleep, 60)
        time.sleep(3.0)  # workers start and pick up the 60 s jobs
        print(" ".join(str(p.pid) for p in mp.active_children()), flush=True)
        os.kill(os.getpid(), signal.SIGKILL)
""")


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _wait_gone(pids: list[int], timeout_s: float) -> bool:
    end = time.time() + timeout_s
    while time.time() < end:
        if not any(_alive(p) for p in pids):
            return True
        time.sleep(0.2)
    return False


@contextlib.contextmanager
def _orphan_workers(tmp_path: Path, kind: str, context: str, guarded: bool):
    """Yield the worker pids of a parent that SIGKILLed itself; whatever outlives the test is
    killed with its process group."""
    script = tmp_path / "parent.py"
    script.write_text(PARENT)
    args = [sys.executable, str(script), kind, context, "1" if guarded else "0", str(ANALYSES)]
    # own session, so the cleanup reaches every descendant; read one line, not to EOF:
    # surviving workers keep the pipe open
    with subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        cwd=tmp_path,
        start_new_session=True,
    ) as proc:
        assert proc.stdout is not None
        try:
            pids = [int(v) for v in proc.stdout.readline().split()]
            proc.wait(timeout=60)
            assert len(pids) == 2
            yield pids
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.stdout.close()


@pytest.mark.parametrize("kind", ["pool", "executor"])
@pytest.mark.parametrize("context", ["fork", "spawn"])
def test_workers_exit_when_parent_is_killed(tmp_path, kind, context):
    with _orphan_workers(tmp_path, kind, context, guarded=True) as pids:
        assert _wait_gone(pids, 4 * POLL_S)


def test_control_unguarded_workers_survive(tmp_path):
    """Without the watchdog the busy workers outlive the parent, so the test above can see a
    leak."""
    with _orphan_workers(tmp_path, "executor", "spawn", guarded=False) as pids:
        assert not _wait_gone(pids, 5.0)


def test_executor_error_cancels_queue_and_stops_workers():
    """An exception in the block (Ctrl-C, SystemExit from SIGTERM) waits for neither queued nor
    running jobs."""
    t0 = time.perf_counter()
    with pytest.raises(KeyboardInterrupt), process_executor(2) as ex:
        futs = [ex.submit(time.sleep, 30) for _ in range(6)]
        time.sleep(1.5)  # two running, the rest queued
        pids = list(ex._processes)
        raise KeyboardInterrupt
    assert time.perf_counter() - t0 < 10  # a plain executor waits 3 x 30 s here
    assert not any(f.done() and not f.cancelled() and f.exception() is None for f in futs)
    assert _wait_gone(pids, 5.0)


def test_executor_normal_exit_waits_for_results():
    with process_executor(2) as ex:
        futs = [ex.submit(pow, 2, k) for k in range(5)]
    assert [f.result() for f in futs] == [1, 2, 4, 8, 16]
