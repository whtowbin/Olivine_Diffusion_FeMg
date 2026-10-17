---
name: forster-gui
description: Use when changing or testing FORSTER's desktop app (src/forster/gui/, PySide6 / Qt), its worker threads, its tests in tests/test_gui*.py, or its installers (packaging/gui/, .github/workflows/gui-installers.yml).
---

# FORSTER desktop app

Design and decisions: `docs/plans/2026-10-15-gui.md` (layout, what each milestone built,
what is left). Starting-profile names it edits: `docs/plans/2026-10-16-geometry-naming.md`.

## Structure
- `gui/state.py` and other non-Qt modules hold the logic (segment table <-> `Run`,
  validation through `Project.update_run`, so the app accepts exactly what the table
  reader accepts). Keep new logic there and test it without Qt.
- `gui/app.py` (window), `panels.py` (side tabs), `plot.py` (the interactive plot,
  drawn by the app, not by `forster.plotting`), `workers.py`, `style.py` (light/dark).
- The core it sits on: `forster.project.Project` (folder of tables + `project.json`,
  npz/JSON cache, never pickles), `forster.imports` (column mapping), `forster.notes`.

## Threads: the Job pattern
- Work runs as `workers.Job` (a `QRunnable`) on a `QThreadPool`; each pool has
  `setMaxThreadCount(1)` (one fit, scan, Monte Carlo or law job at a time).
- `JobSignals` is created in the GUI thread, so `done(tag, answer, seconds)`,
  `failed(tag, message)`, `cancelled(tag)` and `progress(tag, text)` reach their slots
  through the GUI event loop. **Never touch a widget from the worker function.**
- Cancel: the window holds a `threading.Event`; the job passes `cancel=event.is_set` into
  `scan_run` / `fit_initial_condition` / `monte_carlo`, which raise `sampling.Cancelled`
  between model runs; `Job.run` turns that into `cancelled`.
- `closeEvent` sets the cancel events, stops the autosave timer and `waitForDone` on every
  pool before accepting. Keep that when adding pools or windows.

## Warnings are data
`warnings.catch_warnings` is not thread-safe. FORSTER's own warnings go through
`forster.notes.warn`; inside `with notes.collect() as found:` they land in that thread's
list (a ContextVar) instead of the global warning state. Workers collect and return
them; the window shows them under the plot. Outside `collect()` they are ordinary
`warnings.warn` calls, so the CLI and `pytest.warns` are unchanged. New warnings in the
core must use `notes.warn`, or the app will not see them.

## Tests
- Headless Qt with `QT_QPA_PLATFORM=offscreen` and `PySide6.QtTest.QTest`
  (`tests/test_gui.py`); no pytest-qt. Skip with `pytest.importorskip("PySide6")`.
- Drive the real widgets (synthetic mouse events on the canvas, typing into table cells)
  rather than mocking the controller.
- Screenshots for review go to `docs/plans/gui-screenshots/`; look at them.

## Keep it light (Henry, 2026-10-16; open work in TODO_local.md "GUI: keep it light")
- Measure first (memory with a window open, after a fit, a scan, a Monte Carlo) before
  optimising.
- Build tabs and dialogs, and import their modules, when first opened; reuse one figure
  per view; on closing anything: cancel and wait for its worker, disconnect signals,
  `deleteLater()`, close its matplotlib figures, drop its arrays and caches.
- **No kriging in the app:** it must never load pykrige (only the legacy uncertainty
  method krigs; the command line and API keep it). pykrige is imported lazily; the
  installer spec should exclude it (not done yet).

## Installers
Built only in GitHub CI (`gui-installers.yml`, manual or `v*` tags), never locally (the
machine's memory is tight). Packaging lessons are in the GUI plan (scipy's Sobol data
file, `forster/data`, package metadata, a fixed matplotlib font-cache folder); each
bundle runs `forster.gui.selftest`.
