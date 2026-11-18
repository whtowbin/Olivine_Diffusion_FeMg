"""`analyses/_workers.py` with real processes: workers exit when their parent is SIGKILLed (an
unguarded control must survive), an exception in a `process_executor` block cancels the queue
and stops the workers at once, a normal exit waits for every job, and the analysis scripts that
import it still start."""

import contextlib
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest
from conftest import HAS_ROUND_ROBIN

REPO = Path(__file__).resolve().parents[1]
ANALYSES = REPO / "analyses"
sys.path.insert(0, str(ANALYSES))
from _workers import POLL_S, process_executor  # noqa: E402  # ty: ignore[unresolved-import]

# Each job touches its flag file when it starts, so the parent kills itself as soon as both
# workers are busy instead of after a fixed sleep.
PARENT = textwrap.dedent("""
    import multiprocessing as mp, os, signal, sys, time
    from concurrent.futures import ProcessPoolExecutor
    from pathlib import Path

    sys.path.insert(0, sys.argv[4])
    import _workers as W


    def job(flag):
        Path(flag).touch()
        time.sleep(60)


    if __name__ == "__main__":
        kind, ctx, guarded = sys.argv[1], sys.argv[2], sys.argv[3] == "1"
        flags = [Path(sys.argv[5]) / f"started-{i}" for i in range(2)]
        if kind == "pool":
            pool = W.process_pool(2, ctx) if guarded else mp.get_context(ctx).Pool(2)
            for f in flags:
                pool.apply_async(job, (str(f),))
        else:
            cm = W.process_executor(2, ctx)  # keep a reference: a collected generator closes it
            plain = ProcessPoolExecutor(2, mp_context=mp.get_context(ctx))
            ex = cm.__enter__() if guarded else plain
            for f in flags:
                ex.submit(job, str(f))
        end = time.time() + 30
        while not all(f.exists() for f in flags) and time.time() < end:
            time.sleep(0.05)
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
    """Yield the worker pids of a parent that SIGKILLed itself once both workers were busy;
    whatever outlives the test is killed with its process group."""
    script = tmp_path / "parent.py"
    script.write_text(PARENT)
    args = [
        sys.executable,
        str(script),
        kind,
        context,
        "1" if guarded else "0",
        str(ANALYSES),
        str(tmp_path),
    ]
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
            assert all((tmp_path / f"started-{i}").exists() for i in range(2)), "jobs never ran"
            yield pids
        finally:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.stdout.close()


@pytest.mark.slow
@pytest.mark.parametrize("kind", ["pool", "executor"])
@pytest.mark.parametrize("context", ["fork", "spawn"])
def test_workers_exit_when_parent_is_killed(tmp_path, kind, context):
    # the watchdog sees the new parent within one poll (POLL_S); 4 polls allow for a loaded host
    with _orphan_workers(tmp_path, kind, context, guarded=True) as pids:
        assert _wait_gone(pids, 4 * POLL_S)


@pytest.mark.slow
def test_control_unguarded_workers_survive(tmp_path):
    """Without the watchdog the busy workers outlive the parent, so the test above can see a
    leak. The control watches for 2 polls: a watchdog would have fired after one."""
    with _orphan_workers(tmp_path, "executor", "spawn", guarded=False) as pids:
        assert not _wait_gone(pids, 2 * POLL_S)


def test_executor_error_cancels_queue_and_stops_workers():
    """An exception in the block (Ctrl-C, SystemExit from SIGTERM) waits for neither queued nor
    running jobs, and the queued ones are cancelled rather than failed."""
    t0 = time.perf_counter()
    with pytest.raises(KeyboardInterrupt), process_executor(2) as ex:
        futs = [ex.submit(time.sleep, 30) for _ in range(6)]
        time.sleep(1.5)  # two running, one in the call queue, three still pending
        pids = list(ex._processes)
        raise KeyboardInterrupt
    assert time.perf_counter() - t0 < 10  # a plain executor waits 3 x 30 s here
    # the executor's manager thread settles the futures after the block has left (~0.5 s)
    end = time.time() + 5.0
    while not all(f.done() for f in futs) and time.time() < end:
        time.sleep(0.05)
    assert all(f.done() for f in futs)
    assert not any(f.done() and not f.cancelled() and f.exception() is None for f in futs)
    # 6 jobs - 2 running - 1 already handed to the call queue (EXTRA_QUEUED_CALLS) = 3 that only
    # cancel_futures=True cancels; without it they fail with BrokenProcessPool instead
    assert sum(f.cancelled() for f in futs) >= 3
    assert _wait_gone(pids, 5.0)


def test_executor_normal_exit_waits_for_results():
    """Leaving the block normally waits for every job: all are done before result() is asked."""
    with process_executor(2) as ex:
        futs = [ex.submit(time.sleep, 0.3) for _ in range(4)]  # ~0.6 s with 2 workers
        squares = [ex.submit(pow, 2, k) for k in range(5)]
    assert all(f.done() for f in futs + squares)
    assert [f.result() for f in squares] == [1, 2, 4, 8, 16]


SCRIPTS = [
    pytest.param(ANALYSES / "ic_sampling" / "references.py", id="ic_sampling/references"),
    pytest.param(
        ANALYSES / "round_robin" / "refit_2026.py",
        id="round_robin/refit_2026",
        # reads the round robin settings at import
        marks=pytest.mark.skipif(not HAS_ROUND_ROBIN, reason="needs the private round robin data"),
    ),
]


@pytest.mark.parametrize("script", SCRIPTS)
def test_analysis_scripts_that_use_workers_start(script):
    """The scripts put analyses/ on sys.path to import _workers; a moved file or a wrong
    parents[...] would only show at the start of a long run."""
    out = subprocess.run(
        [sys.executable, str(script), "--help"],
        capture_output=True,
        text=True,
        cwd=REPO,
        timeout=120,
    )
    assert out.returncode == 0, out.stderr
    assert "usage" in out.stdout.lower()
