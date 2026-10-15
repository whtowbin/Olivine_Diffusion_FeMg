"""Headless self-test: the app's own code paths on the worked example, then the server.

1. Through `gui_web.app.SESSION` (what the page uses): a new project from the example,
   a fit in a worker-style call on a snapshot, a starting-profile edit and its refit, a
   refused edit, save, close, reopen, and the cached result back without a refit.
2. The NiceGUI server started headless on a free port, the page requested over HTTP
   (the page function runs and must render the run list and buttons) and one of the
   page's static scripts, then shut down. No window opens.

Exits 0 when every check passes; prints one line per check and the wall time.
"""

from __future__ import annotations

import asyncio
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any, TextIO

from gui_web.security import COOKIE, HOST, free_port, protect

CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))
    print(f"{'ok  ' if ok else 'FAIL'} {name}{': ' + detail if detail else ''}", flush=True)


def report_numba_cache() -> None:
    """Where Numba keeps its compiled kernels, and whether this run loaded or compiled
    them (a bundle that cannot cache recompiles, ~1 s, at every launch)."""
    from forster import _kernels

    hits = misses = 0
    for name in ("central_steps", "dipra_steps", "tr_bdf2_steps"):
        stats = getattr(_kernels, name).stats
        hits += sum(stats.cache_hits.values())
        misses += sum(stats.cache_misses.values())
    path = _kernels.central_steps.stats.cache_path
    print(f"info numba cache {path}: {hits} loaded, {misses} compiled", flush=True)


def logic(folder: Path) -> None:
    from forster import compiled

    from gui_web.app import SESSION
    from gui_web.session import fit_in_thread, snapshot

    check("numba kernels available", compiled.AVAILABLE)
    project = SESSION.new(folder)
    name = SESSION.selected
    check("new project from the example", name is not None, f"runs {project.run_names}")
    assert name is not None
    snap = snapshot(project)
    result, seconds = fit_in_thread(snap, name)
    adopted = SESSION.adopt_fit(name, snap)
    check(
        "fit (snapshot, then adopted from the cache)",
        adopted is not None and adopted.time_days == result.time_days,
        f"{result.time_days:.3f} d in {seconds:.2f} s",
    )
    report_numba_cache()
    step = SESSION.initial.steps_um[0]
    SESSION.set_initial(step=step + 2.0)
    check("edit drops the cached result", project.result(name) is None)
    edge_event = {"shapes[1].y0": 0.913, "shapes[1].y1": 0.913}
    check("drag event parsed", SESSION.edit_from_relayout(edge_event) == {"edge": 0.913})
    snap = snapshot(project)
    moved, seconds = fit_in_thread(snap, name)
    moved_adopted = SESSION.adopt_fit(name, snap)
    check(
        "refit after the edit",
        moved_adopted is not None,
        f"{moved.time_days:.3f} d in {seconds * 1000:.0f} ms",
    )
    try:
        SESSION.set_initial(edge=1.5)
        refused = ""
    except ValueError as error:
        refused = str(error)
    check("bad edit refused with a message", bool(refused), refused)
    check("refused edit changed nothing", SESSION.initial.steps_um[0] == step + 2.0)
    SESSION.save()
    SESSION.close()
    reopened = SESSION.open(folder)
    start = time.perf_counter()
    cached = reopened.result(name)
    load_ms = (time.perf_counter() - start) * 1000
    check(
        "reopened: same run, cached result without refitting",
        cached is not None
        and cached.time_days == moved.time_days
        and not reopened.is_stale(name)
        and reopened.run(name).initial.steps_um[0] == step + 2.0
        and SESSION.selected == name,
        f"{cached.time_days if cached else None} d, loaded in {load_ms:.0f} ms",
    )


async def _websocket_opens(url: str, cookie: str | None) -> bool:
    """Whether a websocket upgrade to ``url`` is accepted (with the cookie, if given)."""
    from websockets.asyncio.client import connect

    headers = {"Cookie": cookie} if cookie else {}
    try:
        async with connect(url, additional_headers=headers, open_timeout=10):
            return True
    except Exception:  # InvalidStatus (403) for a refused upgrade
        return False


async def _security_probe(client: Any, base: str, token: str) -> str:
    """Requests without the launch token are refused, over HTTP and the websocket; the
    page with it is served and sets the cookie. Returns the page's HTML."""
    ws = base.replace("http", "ws") + "/_nicegui_ws/socket.io/?EIO=4&transport=websocket"
    polling = base + "/_nicegui_ws/socket.io/?EIO=4&transport=polling"
    bare = await client.get(base + "/")
    wrong = await client.get(base + "/?token=wrong")
    check("page without the token refused", bare.status_code == 403, str(bare.status_code))
    check("page with a wrong token refused", wrong.status_code == 403, str(wrong.status_code))
    poll = await client.get(polling)
    check("socket.io without the token refused", poll.status_code == 403, str(poll.status_code))
    check("websocket without the token refused", not await _websocket_opens(ws, None))
    page = await client.get(f"{base}/?token={token}")
    check("page with the token served", page.status_code == 200, f"{len(page.text)} bytes")
    cookie = f"{COOKIE}={client.cookies.get(COOKIE)}"
    check("page sets the token cookie", client.cookies.get(COOKIE) == token)
    poll = await client.get(polling)  # the client now sends the cookie
    check("socket.io with the cookie accepted", poll.status_code == 200, str(poll.status_code))
    check("websocket with the cookie accepted", await _websocket_opens(ws, cookie))
    return page.text


def server() -> None:
    import httpx
    from nicegui import app, background_tasks, ui

    import gui_web.app  # noqa: F401  (registers the page)

    port = free_port()
    token = protect(app)
    base = f"http://{HOST}:{port}"

    async def probe() -> None:
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                for _ in range(100):  # the server listens once startup has finished
                    try:
                        await client.get(base + "/")
                        break
                    except httpx.ConnectError:
                        await asyncio.sleep(0.1)
                else:
                    raise RuntimeError("the server never answered")
                text = await _security_probe(client, base, token)
                check(
                    "page renders the run list and buttons",
                    "Fit initial condition" in text and "AZ18 WHT01_Ol_63_prof_1" in text,
                )
                scripts = re.findall(r'/_nicegui/[^"\']+?\.js', text)
                asset = await client.get(base + scripts[0]) if scripts else None
                check(
                    "static script served (with the cookie)",
                    asset is not None and asset.status_code == 200,
                    scripts[0] if scripts else "no script in the page",
                )
                if scripts:
                    async with httpx.AsyncClient() as stranger:  # no cookie
                        bare = await stranger.get(base + scripts[0])
                    check("static script without the token refused", bare.status_code == 403)
        except Exception:
            check("server probe", False, traceback.format_exc())
        finally:
            app.shutdown()

    app.on_startup(lambda: background_tasks.create(probe()))
    ui.run(
        host=HOST,
        port=port,
        native=False,
        show=False,
        reload=False,
        show_welcome_message=False,
        uvicorn_logging_level="error",
    )


class Tee:
    """Write to the console and to selftest.log (a windowed bundle has no console)."""

    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:  # uvicorn asks, to decide on colours
        return False


def self_test(tmp: Path) -> int:
    start = time.perf_counter()
    tmp.mkdir(parents=True, exist_ok=True)
    with open(tmp / "selftest.log", "w", encoding="utf-8") as log:
        console = sys.stdout
        sys.stdout = Tee(console, log)
        try:
            return _self_test(tmp, start)
        finally:
            sys.stdout = console


def _self_test(tmp: Path, start: float) -> int:
    folder = tmp / f"selftest-{int(time.time() * 1000)}"
    try:
        logic(folder)
        import webview  # bundled? (does not start a GUI backend)

        check("pywebview importable", True, getattr(webview, "__version__", ""))
        server()
    except Exception:
        check("self-test", False, traceback.format_exc())
    failed = [name for name, ok, _ in CHECKS if not ok]
    wall = time.perf_counter() - start
    print(f"self-test {'FAILED' if failed else 'passed'}: {len(CHECKS)} checks, {wall:.1f} s wall")
    sys.stdout.flush()
    return 1 if failed else 0
