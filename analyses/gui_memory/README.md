# GUI memory, threads, figures and modules

Measured 2026-10-20 with `measure.py` (a script, not a test): a real offscreen
`MainWindow` (`QT_QPA_PLATFORM=offscreen`), Numba on, the `forster template` example
project (`MainWindow.new_project`). Each sequence ran in a fresh subprocess; 3 repeats,
median shown (`results.json` has the per-repeat RSS). Nothing under `src/` was changed.
Run: `uv run --with psutil python analyses/gui_memory/measure.py`.

## Checkpoints (median of 3)

RSS in MB. Threads: Python `threading.active_count()` / OS threads (psutil). Pool threads:
`activeThreadCount()` summed over the window's 4 `QThreadPool`s at that moment. Figures:
`pyplot.get_fignums()` / live `Figure` objects found by `gc`. 7b is an extra checkpoint
(second profile imported with `forster.imports.guess`, then fitted).

| Checkpoint | RSS MB | delta | Threads py / OS | Pool active | Figures pyplot / live | Modules | pykrige |
|---|---|---|---|---|---|---|---|
| 0 interpreter start (psutil, threading only) | 63.6 |  | 1 / 1 | 0 | 0 / 0 | 425 | n |
| 1 after import forster.gui.app | 222.0 | +158.4 | 1 / 3 | 0 | 0 / 0 | 1625 | n |
| 2 window open, first fit settled | 315.5 | +93.5 | 1 / 10 | 0 | 0 / 3 | 1716 | n |
| 3 after refit following a drag | 317.9 | +2.4 | 1 / 10 | 0 | 0 / 3 | 1716 | n |
| 4 after starting-profile fit (scan) | 332.6 | +14.7 | 1 / 16 | 0 | 0 / 3 | 1716 | n |
| 5 after Monte Carlo (200 draws) | 332.8 | +0.2 | 1 / 17 | 0 | 0 / 3 | 1716 | n |
| 6 after Law comparison drawn | 335.6 | +2.8 | 1 / 18 | 0 | 0 / 3 | 1716 | n |
| 7 after Time distribution drawn | 339.3 | +3.7 | 1 / 18 | 0 | 0 / 3 | 1716 | n |
| 7b after importing 2 profiles (A, B) and fitting A | 353.4 | +14.1 | 1 / 18 | 0 | 0 / 3 | 1716 | n |
| 8 after opening/closing ImportDialog x20 | 419.6 | +66.2 | 1 / 18 | 0 | 0 / 3 | 1716 | n |
| 9 after switching all centre views and side tabs x20 | 440.0 | +20.4 | 1 / 18 | 0 | 0 / 3 | 1716 | n |
| 10 after window.close() + processEvents + gc | 439.8 | -0.2 | 1 / 14 | 0 | 0 / 3 | 1716 | n |
| 10b after deleteLater + del references + gc | 439.8 | +0.0 | 1 / 14 | 0 | 0 / 3 | 1716 | n |
| 10c 32 s later (idle pool threads expired?) | 439.0 | -0.8 | 1 / 3 | 0 | 0 / 3 | 1716 | n |

Repeat spread is small (RSS within 1 to 8 MB between repeats, up to 10 MB from step 4 on in
one repeat). One repeat fell to 344 MB at 10c; the other two stayed at 439-441 MB, so the
late drop is not reliable.

## Scaling with runs in the project

Duplicate (`MainWindow.duplicate_run`) up to N runs, all fitted, window open, one window per
process (median of 3 each):

| Runs | RSS MB | live Figures | Modules |
|---|---|---|---|
| 1 | 372.3 | 3 | 1715 |
| 5 | 371.5 | 3 | 1715 |
| 10 | 373.5 | 3 | 1715 |

Runs are cheap: 9 extra duplicated runs add about 1 MB. (Duplicates carry the same profile
and parameters; a project with ten different long profiles was not measured.) The single-run
RSS here (372 MB) is above checkpoint 2 (315 MB) because this variant also waits for the
`Results` table refresh and has not been run in the same state as the sequence; treat the
slope, not the offset.
The second imported profile (7b) cost +14 MB: its own fit plus the Fo/oxide tables.

## What grew

- Import of `forster.gui.app`: +158 MB, 1,200 modules (425 to 1,625), 0.8 s (this matches the
  baseline of 232 MB / 1,586 modules, measured here without the interpreter's own 64 MB).
- Opening the window and the first fit: +94 MB (Qt widgets, three matplotlib canvases, the
  first Numba-compiled fit, 91 more modules, OS threads 3 to 10).
- Starting-profile fit +15 MB and 6 OS threads (the scan's sampler/pool threads).
- Monte Carlo, Law comparison, Time distribution and a drag refit: +0.2 to +3.7 MB each. The
  law view's four refits are cheap.
- 20 x ImportDialog open and close: +66 MB (about 3.3 MB each), and
  `window.findChildren(QDialog)` returned 20 afterwards: every dialog stays alive as a child
  of the window (see leaks).
- 20 x switching all centre views and side tabs: +20 MB (about 1 MB a loop, still creeping).

## Does closing return memory?

No. After `close()` + `processEvents` + `gc.collect()` RSS stays at 440 MB (-0.2), and also
after `deleteLater()` and dropping the references. It is 218 MB above checkpoint 1 and 125
MB above checkpoint 2. Python's allocator and Qt do not hand pages back, so some of this is
expected for a process, but the retained objects below are real.
Threads: Python threads are always 1 (the pool workers are Qt threads, not `threading`).
OS threads fall from 18 to 14 at close and to 3 after 32 s, once `QThreadPool`'s idle
workers expire (30 s); `waitForDone` returns before they exit. Not a leak.
Figures: 0 in pyplot all the time (the canvases do not register), but 3 live `Figure`
objects (profile plot, time distribution, law comparison) survive `close()`,
`deleteLater()` and `gc.collect()` with their canvases still valid Qt objects.

## Kriging (pykrige)

pykrige is not imported at module import and not in any of the 14 checkpoints:
`'pykrige' in sys.modules` is False throughout the whole session.

How it could be reached, with file:line:
- It is imported lazily at `src/forster/engine.py:482` (`from pykrige import UniversalKriging`
  inside `_kriging_model`), called only from `engine.Krige_Interpolate`
  (`src/forster/engine.py:501`).
- That is called only when `uncertainty == "legacy"`, `src/forster/model.py:1148-1149`
  (the correlated method skips kriging, comment at `model.py:1147`).
- The GUI passes `kriging=defaults.kriging` into new runs at `src/forster/gui/state.py:912`,
  but those are only settings; they do not import or call pykrige. Nothing under
  `src/forster/gui/` mentions pykrige, "legacy" or an uncertainty-method control (the only
  `uncertainty` use is `state.py:847`, which declines to draw the time distribution for a
  non-correlated fit). No menu, combo or tab offers the method.
- The back door is a project file: `io._uncertainty` (`src/forster/io.py:442-453`) reads an
  `uncertainty` column from the parameter table, and `project.py:1019` stores it. A run
  with `uncertainty=legacy` (the table default scheme then becomes `central`,
  `model.py:65-67`; with `implicit` the fit is refused, `model.py:1022`) opened in the
  app and fitted does load pykrige. Checked by setting
  `project.update_run(name, uncertainty="legacy", scheme="central")` in a running offscreen
  window and letting the app fit it: `pykrige` in `sys.modules` went False to True, and the
  fit succeeded (so the app also then draws a kriged result).
- Answer: a user cannot trigger it from the app's own controls, but can by opening a project
  whose parameter table (or a CLI-made project) sets `uncertainty` to `legacy`. A guard at
  load or in the worker (refuse or convert such runs) would close it; not done here.
- pykrige is a hard dependency in `pyproject.toml:28`, and `packaging/gui/requirements.txt`
  has no entry for it (grep found no match), so how it reaches the bundle depends on the
  install used to build it.
- PyInstaller spec `packaging/gui/forster_gui.spec:29-40`: the `excludes` list has tkinter,
  IPython, jupyter_client, ipykernel, marimo, pytest, seaborn, sklearn, selenium,
  great_tables, PyQt5, PyQt6. pykrige is neither excluded nor listed in `hiddenimports`;
  since `engine.py:482` is a function-level import, PyInstaller's static analysis still
  finds it, so it is bundled unless the build environment lacks it. Add `"pykrige"` to
  `excludes` (the skill notes this is not done yet).

## Biggest consumers and leaks

1. The import of `forster.gui.app` and its 1,200 modules: 158 MB of the 440 MB. The window
   plus the first fit adds 94 MB more. Together 252 MB is paid before the user does
   anything; lazy imports (scipy.stats, pandas, matplotlib backends, dialogs) are the lever
   on this, not lazy tabs.
2. `ImportDialog` leak: 3.3 MB per open, 66 MB after 20, and 20 dialogs still children of
   the window. `_choose_import` (`app.py:678`) builds the dialog parented to the window and
   never calls `deleteLater()`. This is the clearest, fixable leak.
3. Nothing returns on close: RSS 440 MB after the window is gone, and 3 `Figure` objects stay
   alive. View switching also creeps by about 1 MB a loop (20 MB over 20 loops). Per-run
   cost is about 0.1 MB, so lazy tabs would not save memory per run; the tabs are fixed cost
   (3 canvases, 7 side tabs, about 3 MB for the time and law views).

Caveats: the whole measure.py run took about 25 minutes of wall time although each sequence's own
elapsed time is 5 s (plus a 32 s idle wait); where the rest goes (process start-up or exit) was not
investigated; the
duplicate-run scaling uses identical runs; the 4 GB abort never triggered (peak 442 MB).
