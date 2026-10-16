"""FORSTER's desktop app (Qt, PySide6): ``forster gui [FOLDER]``.

Only this package imports Qt, and only when the app starts (`main`), so ``import
forster`` never needs PySide6. `forster.gui.state` holds the app's logic in plain Python
(tested without Qt); the window shows it and forwards edits to
`forster.project.Project`, so the app accepts exactly what the command line does.
Install with ``pip install "forster[gui]"`` (PySide6-Essentials).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


def main(folder: str | Path | None = None, argv: Sequence[str] | None = None) -> int:
    """Open the window (on a project folder, if given) and run until it is closed."""
    try:
        from forster.gui.app import run_app
    except ImportError as error:
        if "PySide6" not in str(error) and "Qt" not in str(error):
            raise
        raise SystemExit(
            'forster gui needs PySide6: pip install "forster[gui]" (or uv sync --extra gui)'
        ) from error
    return run_app(folder, argv)
