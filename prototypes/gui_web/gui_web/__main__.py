"""`python -m gui_web` (from prototypes/gui_web), or the packaged app.

gui_web                      native window
gui_web --browser            in the default browser instead
gui_web --open FOLDER        open a project at start
gui_web --self-test TMPDIR   headless self-test, exit code 0 when it passes
gui_web --new FOLDER         a new project from the worked example, opened
gui_web --exit-when-ready    quit once the page is up (cold-start timing)
gui_web --quit-after S       quit S seconds after the page is up (CI screenshots)
"""

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    # First: in a bundle, the native window's spawned process re-runs this executable.
    multiprocessing.freeze_support()
    # A --windowed bundle on Windows has no console: sys.stdout is None, which breaks
    # print and uvicorn's log setup (the self-test also writes selftest.log).
    for stream in ("stdout", "stderr"):
        if getattr(sys, stream) is None:
            setattr(sys, stream, open(os.devnull, "w"))  # noqa: SIM115 (lives as long as the app)
    parser = argparse.ArgumentParser(prog="gui_web", description="FORSTER NiceGUI prototype")
    parser.add_argument("--self-test", type=Path, metavar="TMPDIR")
    parser.add_argument("--browser", action="store_true", help="serve to a browser tab")
    parser.add_argument("--no-show", action="store_true", help="with --browser: no tab")
    parser.add_argument("--port", type=int)
    parser.add_argument("--open", type=Path, metavar="FOLDER")
    parser.add_argument("--new", type=Path, metavar="FOLDER")
    parser.add_argument("--exit-when-ready", action="store_true")
    parser.add_argument("--quit-after", type=float, metavar="SECONDS")
    args = parser.parse_args(argv)

    if args.self_test is not None:
        from gui_web.selftest import self_test

        return self_test(args.self_test)

    from nicegui import app, ui

    from gui_web import app as gui
    from gui_web.security import HOST, free_port, protect

    if args.open is not None:
        gui.SESSION.open(args.open)
    if args.new is not None:
        gui.SESSION.new(args.new)
    gui.QUIT_AFTER = 0.5 if args.exit_when_ready else args.quit_after
    native = not args.browser
    # Loopback only, a random port, and a per-launch token in the URL (gui_web.security).
    port = args.port or free_port()
    token = protect(app)
    path = f"/?token={token}"
    if native:
        app.native.window_args["url"] = f"http://{HOST}:{port}{path}"
        extra: dict = {"window_size": (1440, 880)}
    else:
        extra = {"show": path if not args.no_show else False}
        print(f"FORSTER at http://{HOST}:{port}{path}", flush=True)
    ui.run(
        title="FORSTER",
        native=native,
        host=HOST,
        port=port,
        reload=False,
        show_welcome_message=False,
        **extra,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
