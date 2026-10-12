# Restructure Olivine_Diffusion_FeMg so others can run the models

Status (2026-10-12): phase 1 done and committed locally (999a40f engine fix, 9108a99 move, de3bfd3 package, d3ce564 rename to `forster`). Fit-uncertainty work committed as 8a46492 (session github-drafts-68), phase 2 as dec3f59 (see "Phase 2 tasks"). Phases 3-4 not started. `master` is 8 commits ahead of GitHub (not pushed).

## Next session: start here

- Start Claude Code in this repo (`~/Projects/Olivine_Diffusion_FeMg`) so `.claude/settings.json` (ruff format hook) applies. Run `prek install` once if it has not been run.
- Built so far: `src/forster/` with `model.py` (`fit_profile` and dataclasses), `io.py` (`read_profiles`, `read_runs`, `fit_run`; round robin and paper table layouts), `dipra.py` (DIPRA scheme, `scheme="dipra"`), `engine.py` and `orientation.py` (research code moved, cross-term fix applied). 51 tests pass on Python 3.11 and 3.13 (`uv run pytest -n 8`, about 45 s). The real API is in `src/forster/model.py`; the "Python API" sketch below is the earlier proposal and differs in detail (for example `Orientation` is separate from `Conditions`, and `InitialCondition` takes `domain_um`, `steps_um`, `step_fo`).
- Next: phase 3 (tkinter GUI, including the Monte Carlo sampling options) and phase 4 (move the round robin and archive scripts under `analyses/`).
- Open questions for Henry: the citations marked [confirm] in "Science and citations" (DIPRA Table 1 resolved several), and the source of the ±1σ rule (reduced chi-squared within 1 of its minimum).
- Known follow-ups: NumPy deprecation warning in `orientation.py` (float of a 1-element array; will become an error in a future NumPy); `engine.timestepper` keeps every time step in memory (about 250 MB for a 2-year round robin run), which the GUI will want to avoid; the "Before release" items in `CHANGELOG.md` (cross-term fix notes, round robin refits, paper addendum).
- Untracked and not mine: the Quarto render of this doc (`.html`, `_files/`).

## Goal

A user with their own olivine profile can install one package and fit it three ways: from Python, from the command line with two filled-in CSV templates, or in a desktop GUI where they set the initial condition, edit the physical and numerical parameters, run the fit and export the results. The fits must stay identical to the current code, apart from the documented cross-term fix (see "Engine fix").

## Where the code is now

- `Fe_Mg_Diffusion_Convolution_Streamlined.py` (805 lines): the engine. Diffusivity laws (Fe-Mg, Ni, Ca), fO2 buffers, the finite-difference stepper, kriging interpolation, chi-squared and R² fit scores, `Diffusion_call`. Importable on its own.
- `EBSD_Profile_Rotation.py`: Euler angles and profile direction to angles between the profile and the crystal axes. Importable.
- `Round_Robin_Diffusion_Modeling Part 2.py` (1,172 lines): defines `model_diffusion()`, the function a user actually needs, but it looks each profile up by name in two global spreadsheets. Plotting, saving, the P-T-fO2 Monte Carlo and the Great Tables output live in the same script.
- Two parameter-table dialects exist: the paper tables give Euler angles, profile angle and fO2 as ΔQFM; the round robin tables give alpha, beta, gamma and fO2 in Pa directly.

## Approaches

1. **New layers on an unchanged engine (recommended).** Move the engine and EBSD modules into a package as they are, and build the user-facing API, file I/O, CLI and GUI on top. The physics code does not change, so the fits are preserved by construction, and the differential test below proves the new `fit_profile` matches `model_diffusion`. Clean names are re-exported from the package (`fit_profile`, `D_Fo`, ...) without renaming the engine internals.
2. **Split the engine into modules** (diffusivity, solver, fitting) and tidy names (`Best_fit_Chi2_old`, `VectorMaker`). Cleaner to read, but it touches code behind an unpublished paper for no change in what users can do. Could follow later, guarded by the same differential test.
3. **Keep the scripts and add a driver script** that reads a user's spreadsheet. Least work, but `model_diffusion` stays inside a research script and a GUI would have to import from it.

## Proposed layout (approach 1)

```
src/forster/
    engine.py          # Fe_Mg_Diffusion_Convolution_Streamlined.py, unchanged
    orientation.py     # EBSD_Profile_Rotation.py, unchanged except the stereonet plot moves to plotting
    model.py           # fit_profile(): model_diffusion without the spreadsheet lookup
    io.py              # read profile and parameter tables (CSV or xlsx, both dialects); write results
    uncertainty.py     # P-T-fO2 Monte Carlo rescaling (from Part 2 and the paper script)
    plotting.py        # fit plot, stereonet
    cli.py             # forster fit | template | gui
    gui/               # tkinter app
examples/              # one public profile + parameter row (AZ18, from the J. Petrology data) and a marimo notebook
analyses/round_robin/  # current Part 1, Part 2, Load and Plot scripts, importing the package
analyses/xenolith_paper_archive/   # "old Xenolith Code/"
tests/
```

### Python API

```python
from forster import Profile, Conditions, InitialCondition, Grid, fit_profile

result = fit_profile(
    Profile(distance_um, fo),                       # Fo as a fraction
    Conditions(T_C=1250, P_Pa=5e8, fO2_dQFM=0.3,    # or fO2_Pa=...
               alpha=..., beta=..., gamma=...),     # or euler=(phi1, Phi, phi2), profile_angle=...
    InitialCondition(edge_x_um=(5, 180), edge_fo=0.86, core_fo=0.90, steps_um=[42]),
    Grid(dx_um=1, dt_s=60, max_time_days=200),
)
result.time_days, result.minus_1sigma_days, result.plus_1sigma_days
result.plot(); result.to_csv("out/")
```

The dataclasses take the same quantities as today's parameter columns, with units in the names.

### CLI

```
forster template out/          # writes profiles.csv, parameters.csv with units and an example row
forster fit profiles.csv parameters.csv --out results/
forster gui
```

`fit` writes, for each run: the fit figure, the best-fit and ±1σ curves, the interpolated data, and one summary table (time, uncertainties, reduced chi-squared, conditions).

### GUI (tkinter)

- ttk with the sv-ttk theme (or ttkbootstrap; both tested), matplotlib embedded with `FigureCanvasTkAgg`.
- Left: load a profile (CSV/xlsx), then a form for T, P, fO2, orientation, grid and kriging settings, with units.
- Right: the profile plot. Edge and step positions are vertical lines the user drags; edge and core Fo are horizontal lines. The initial condition redraws as they move.
- Fit runs in a worker thread so the window stays responsive, then shows the best fit, ±1σ curves, the time with uncertainties and reduced chi-squared.
- Save parameters to the template format (so a GUI session can be rerun with the CLI) and export results.
- Optional extra: `pip install forster[gui]` pulls in the theme only; tkinter itself ships with Python.

Measured on this Mac: tkinter fails inside any uv-created virtual environment (uv's Python 3.13.11 cannot find `init.tcl`). Setting `TCL_LIBRARY` and `TK_LIBRARY` from `sys.base_prefix` before importing tkinter fixes it; the GUI entry point will do that. Homebrew and Linux system Pythons need Tk installed separately (`brew install python-tk`, `apt install python3-tk`); the README will say so.

## Testing

- **Differential replay (the behaviour-preservation proof):** the 17 round-robin runs go through a verbatim copy of `model_diffusion` (tests/reference) and through `read_runs` + `read_profiles` + `fit_run`; times, best-fit and ±1σ indices, the chi-squared series, the interpolated data and a SHA-256 of every model curve must be identical. Fits are deterministic, so any difference is a bug. A reference-against-itself run on three profiles is the control. The 18 paper profiles cannot be an exact test against the paper's numbers: the engine has changed since the paper (universal instead of ordinary kriging, a new ±1σ rule, time = index × dt instead of (index + 1) × dt, and the cross-term fix below). They test the paper-layout reader instead: Euler angles and the QFM buffer must convert exactly as the paper script did, and one paper run is fitted end to end.
- **Physics checks:** constant diffusivity against the error-function solution; Fo-dependent diffusivity against an independent conservative (flux-form) scheme; and a fit that must recover the time of a profile made by that scheme.
- **I/O:** both parameter dialects; CSV reads like Excel; clear errors for an unrecognised table, duplicate run names and a missing profile; template round trip (phase 2).
- **CLI:** run `forster fit` as a real subprocess on the example files in a temporary directory and check the outputs.
- **GUI:** the controller (state, validation, line positions to `InitialCondition`) is plain Python and unit tested; one smoke test builds the window hidden, runs a fit on the example and closes it.
- Ruff and ty clean on new code; complexipy at most 25. `engine.py`, `orientation.py` and the verbatim reference copy are excluded so their diffs stay reviewable; tidying them is a separate change.
- Standard setup: per-project ruff PostToolUse hook (`.claude/settings.json`) and prek hooks (ruff, format, complexipy, ty, no-test-deletions guard, trivy secret scan, tests at pre-push). `prek install` must be run once per clone.

## Engine fix: the cross term in `diffusion_step` (2026-10-11)

Found while writing the physics tests for phase 1. Henry chose to fix it in phase 1, in its own commit before the package move.

### The equation

With a diffusivity that depends on composition, Fe-Mg diffusion along the profile is

$$\frac{\partial C}{\partial t} = \frac{\partial}{\partial x}\left(D(C)\,\frac{\partial C}{\partial x}\right) = D\,\frac{\partial^2 C}{\partial x^2} + \frac{\partial D}{\partial x}\,\frac{\partial C}{\partial x}$$

The engine solves the expanded form on the right. The second term (the cross term) only matters where $D$ changes along the profile. For Fe-Mg it does, strongly: $D_{Fo} \propto 10^{3(0.9 - X_{Fo})}$, so $D$ doubles for every 0.1 drop in Fo.

### What the code did

Each step updates point $i$ from its neighbours. In `diffusion_step`, `delta` is $\Delta t / \Delta x^2$ and

- `Diffusion` is $D_i\,(C_{i+1} - 2C_i + C_{i-1})$,
- `Diff_C` is $C_{i+1} - C_{i-1}$,
- `Diff_D` is $D_{i+1} - D_{i-1}$.

`Diff_C` and `Diff_D` are central differences: each spans two grid spacings, so

$$\frac{\partial C}{\partial x} \approx \frac{C_{i+1} - C_{i-1}}{2\Delta x}, \qquad \frac{\partial D}{\partial x} \approx \frac{D_{i+1} - D_{i-1}}{2\Delta x}, \qquad \frac{\partial D}{\partial x}\frac{\partial C}{\partial x} \approx \frac{\texttt{Diff\_C}\cdot\texttt{Diff\_D}}{4\,\Delta x^2}$$

The correct update is therefore

$$C_i^{n+1} = C_i^n + \frac{\Delta t}{\Delta x^2}\left[D_i\,(C_{i+1} - 2C_i + C_{i-1}) + \frac{1}{4}\,(C_{i+1} - C_{i-1})(D_{i+1} - D_{i-1})\right]$$

The code had $\tfrac{1}{2}$ in place of $\tfrac{1}{4}$, so the cross term was twice as large as it should be. The comment next to it says the intent was a central difference (as opposed to DIPRA's forward difference, Girona and Costa 2013); a forward difference spans one grid spacing and needs no factor, a central difference needs $\tfrac{1}{2}$ on each of the two derivatives, which is $\tfrac{1}{4}$ on their product. The line was the same in the paper version (commit 7c6bf5f).

### The fix

One character: `(Diff_C * Diff_D) / 2` becomes `(Diff_C * Diff_D) / 4`, with a comment explaining the factor. The Ni and Mn time steppers call the same `diffusion_step` and get the fix too; the Ca step uses a constant $D$ and has no cross term.

### How it was checked

Against solutions that share no code with the engine (`tests/test_physics.py`):

| Check | Engine as written | After the fix |
|---|---|---|
| Constant $D$ against the error-function solution for a step (the cross term is zero here, so this tests the rest of the scheme) | 9×10⁻⁷ Fo | same |
| Fo-dependent $D$ against an independent conservative scheme with $D$ on the cell faces, $\partial_t C = \partial_x(D\,\partial_x C)$ (Fo 0.80 to 0.90 step, round robin conditions, 15.8 days) | 8.6×10⁻³ Fo | 1.4×10⁻⁴ Fo |
| Grid error of that reference itself (dx 1 µm against 0.5 µm) | | 3.5×10⁻⁴ Fo |

After the fix the engine agrees with the reference to within the reference's own grid error.

### Checked against the published DIPRA discretization

DIPRA (Girona and Costa 2013, G³ 14, 422-431, eq. 5; Auxiliary Material 1, eq. 8) writes the same update as

$$C_k^{j+1} = C_k^j + \frac{\Delta t}{\Delta x^2}\left[(C_{k+1} - C_k)(X_{Fo,k+1} - X_{Fo,k})\,\frac{dD}{dX_{Fo}}\Big|_k + (C_{k+1} - 2C_k + C_{k-1})\,D_k\right]$$

Its cross term uses forward differences (each over one $\Delta x$) and the analytic $dD/dX_{Fo}$, so it needs no factor. This engine uses central differences (each over $2\Delta x$) and differences of $D$, so it needs $\tfrac{1}{4}$. The old $\tfrac{1}{2}$ matches neither.

DIPRA's eq. 5 was implemented exactly as published and run with both versions of the engine against the flux-form reference at $\Delta x = 0.25$ µm (which changes by 1.8×10⁻⁴ Fo from its $\Delta x = 0.5$ µm run). Same Fo 0.80 to 0.90 step, round robin conditions, 15.8 days. Largest difference from the reference, in Fo:

| Scheme | $\Delta x$ = 1.0 µm | $\Delta x$ = 0.5 µm |
|---|---|---|
| Engine as written (÷2) | 8.1×10⁻³ | 8.6×10⁻³ |
| Engine fixed (÷4) | 6.7×10⁻⁴ | 2.5×10⁻⁴ |
| DIPRA eq. 5 | 3.7×10⁻⁴ | 9.5×10⁻⁵ |

The fixed engine and DIPRA both converge on the reference as the grid is refined and agree with each other to 2.1×10⁻⁴ Fo at $\Delta x$ = 0.5 µm. The old engine does not converge: it settles on a different answer, as expected for a wrong term in the equation rather than a discretization error.

### Effect on fitted times

A profile made by the reference scheme at a known time, fitted back with each engine (idealised single steps about 30 µm wide):

| Case | $D$ ratio across the step | Fitted / true time, as written | After the fix |
|---|---|---|---|
| Paper-like rim, Fo 0.92 to 0.895, 1250 °C | 1.19 | 0.987 | 1.000 |
| KS20-like, Fo 0.82 to 0.885, 1200 °C | 1.57 | 0.917 | 1.001 |
| SH63-like, Fo 0.67 to 0.745, 1200 °C | 1.68 | 0.892 | 1.001 |

The 17 round robin runs, refitted with the batch settings of `Round_Robin_Diffusion_Modeling Part 2.py` (gaussian kriging, 2-year model length, uniform uncertainty 0.001):

| Run | Best fit before (days) | Best fit after (days) | After / before | Range after (days) | Min. reduced χ², before → after |
|---|---|---|---|---|---|
| KS20-527-8_transect-1 | 65.0 | 75.0 | 1.16 | 65.6-85.7 | 1.50 → 1.61 |
| KS20-527-8_transect-2 | 60.8 | 71.8 | 1.18 | 61.5-83.3 | 1.43 → 1.59 |
| KS20-527-8_transect-3 | 29.5 | 35.3 | 1.20 | 25.7-46.2 | 2.46 → 1.72 |
| KS20-527-8_transect-4 | 57.4 | 65.6 | 1.14 | 55.5-76.9 | 1.33 → 1.56 |
| KS20-527-8_transect-5 | 44.5 | 50.4 | 1.13 | 41.1-61.1 | 1.62 → 1.94 |
| KS20-527_2-transect-1 | 29.9 | 42.8 | 1.43 | 30.5-56.3 | 2.40 → 3.40 |
| KS20-527_2-transect-2 | 27.7 | 39.5 | 1.43 | 27.3-53.1 | 2.35 → 3.20 |
| KS20-527_2-transect-3 | 21.6 | 27.6 | 1.28 | 18.1-40.2 | 2.99 → 2.38 |
| KS20-527_2-transect-4 | 18.8 | 22.2 | 1.18 | 15.4-31.5 | 7.23 → 3.39 |
| KS20-527_5-transect-1 | 289.3 | 365.1 | 1.26 | 306.5-429.1 | 1.98 → 3.29 |
| KS20-527_5-transect-2 | 185.2 | 245.2 | 1.32 | 185.6-310.4 | 1.21 → 2.11 |
| KS20-527_5-transect-3 | 146.3 | 179.5 | 1.23 | 128.7-242.0 | 1.85 → 1.26 |
| SH63olv066-1 | 7.7 | 9.7 | 1.26 | 8.6-10.9 | 12.7 → 23.4 |
| SH63olv066-2 | 8.4 | 10.6 | 1.26 | 9.4-11.9 | 6.16 → 17.5 |
| SH63olv066-3 | 9.1 | 11.6 | 1.27 | 10.2-13.0 | 3.51 → 15.1 |
| SH63olv066-4 | 8.2 | 10.2 | 1.25 | 9.0-11.7 | 4.37 → 11.4 |
| SH63olv084 | 611.7 | 441.1 | 0.72 | 407.8-476.5 | 5.17 → 7.80 |

Most times get 13-43% longer. SH63olv084 gets 28% shorter: its Fo falls from the edge into the core (73 to 62.6), so the gradient runs the other way relative to $D$ and the error had the opposite sign. The real profiles move more than the idealised steps because their steps are sharper and their Fo contrasts larger.

### What follows

- The paper version in Xenolith_Melt_Vein_Modeling stays as published; it exists to reproduce the paper. Effect on the paper, measured by running its script as published and with the fix (same random seed, so identical Monte Carlo draws; the published copy reproduces the committed paper fits exactly). Medians of the P-T-corrected times, days, with the 5th-95th percentile range:

  | Group | Paper | With the fix | Change |
  |---|---|---|---|
  | Xenolith WHT01, melt veins | 17.9 (5.5-44.9) | 17.6 (5.4-45.3) | -2% |
  | Xenolith GCB1, melt veins | 12.8 (6.1-28.1) | 13.5 (6.5-29.4) | +5% |
  | Xenolith WHT06 | 5.6 (0.8-39.4) | 5.4 (0.8-33.1) | -3% |
  | WHT01 contact overgrowth (exterior) | 5.9 (2.3-15.2) | 7.5 (3.2-17.9) | +28% |
  | Xenocrysts | 0.79 (0.14-5.5) | 0.86 (0.17-5.7) | +9% |

  The melt-vein timescales (the paper's 1-3 weeks) move by 2-5%; single melt-vein profiles move -9% to +5%, except AZ18 WHT06_ol_overgrowth2 (18.3 to 15.4 days, -16%). The exterior contact overgrowths move most (profile 180: 7.0 to 8.3 days; profile 181: 3.3 to 4.6 days) because their Fo contrast is 7-10 units; their median of 7.5 days is still about one week, and the melt-vein and exterior timescales remain similar. Any ascent rate computed from the exterior times would come down by up to about 20% [check how the paper derives 0.05-0.15 m/s]. Per-profile numbers: `docs/plans/2026-10-11-engine-fix/paper_fix_effect_profiles.csv` and `paper_fix_effect_summary.csv`; round robin: `round_robin_fix_effect.csv`; DIPRA comparison script: `dipra_check.py`. The Xenolith_Melt_Vein_Modeling README should get a short note with these numbers. Henry (2026-10-11): note these effects to return to later in an addendum to the paper.
- The round robin results need rerunning. The reduced χ² of the SH63olv066 fits gets clearly worse (3.5-12.7 before, 11.4-23.4 after), which suggests their step positions and plateau Fo were tuned while fitting with the old engine; they should be revisited before the new times are final. If the round robin results have been sent to the organisers, they need a correction.

## Phases (each ends with tests passing and a commit you approve)

1. Package skeleton, engine and orientation moved, `model.py`, `io.py`, differential replay and physics tests. Last step, as its own commit once the skeleton is committed: rename the working package `olivine-diffusion` / `src/olivine_diffusion` to `forster` / `src/forster` (pyproject name and package path, imports in tests, `uv lock`), then rerun the full test suite.
2. CLI, templates, example data, README with install and the three ways to run.
3. GUI.
4. Move the round robin and archive scripts under `analyses/` and point them at the package (checked by rerunning them).

### Phase 2 tasks (written 2026-10-12)

Builds on the uncommitted fit-uncertainty work from session github-drafts-68 (`forster/uncertainty.py`, `uncertainty=` in `fit_profile`, default "correlated"). That work should be committed first, on its own; phase 2 then commits on top.

Decisions made while planning (Henry to confirm at the phase commit):

- **A third parameter-table layout, "template", using the Python API's names and units** (`run`, `profile`, `T_C`, `P_Pa`, `domain_start_um`, `steps_um`, `edge_fo`, ...). Fo is a fraction everywhere, as in the API. Orientation is given either as `alpha_deg`, `beta_deg`, `gamma_deg` or as EBSD Euler angles `phi1_deg`, `Phi_deg`, `phi2_deg` with `profile_angle_deg`; fO2 either as `fO2_Pa` or as `fO2_buffer` (QFM, NNO) with `fO2_delta_log`. A row that gives both forms, or neither completely, is an error. The round robin and paper layouts keep working unchanged. Profile tables: `profile`, `distance_um`, `fo` (fraction), detected from the columns; the lab layout (`Profile_Name`, `Distance µm`, `Fo#` in percent) stays the fallback.
- **The example is AZ18 WHT01_Ol_63_prof_1** (Towbin and Plank 2026; 30 points, one step, a melt-vein olivine): one step keeps the first example simple. It ships inside the package (`src/forster/data/`), so `forster template DIR` writes it after a pip install and the template is a working example. `max_time_days` is 40 (best fit about 5 days), so it fits in under a second. No separate `examples/` folder or marimo notebook in phase 2.
- `summary.csv` uses the template column names, so it can be fed back to `forster fit` as a parameter table (every run in it reruns as it was fitted, with fO2 in Pa and the orientation as angles).
- `forster gui` is added in phase 3, not stubbed now.
- The P-T-fO2 Monte Carlo cannot be `forster/uncertainty.py` as the layout above planned: that name now holds the fit uncertainty. Use `forster/rescale.py` in phase 3.
- `InitialCondition` now rejects two steps without `step_fo` and more than two steps (the engine crashed with an IndexError). Every row of the round robin and paper tables already satisfies this.

All five tasks done 2026-10-12: 87 tests pass on Python 3.13 and 3.11; ruff, ty and complexipy clean; the wheel holds the example CSVs and the `forster` script. Also found and fixed while testing: pandas' default CSV float parser reads some values 1 ulp off, so `read_table` now uses `float_precision="round_trip"` (without it a summary did not refit identically).

Tasks:

1. `io.py`: template layout in `read_runs` (`_template_run`, `_pick_form`), layout detection by column set with an error naming all three layouts; `read_profiles` detects the template profile columns; a clear error for unparsable `steps_um`; `summary_row(result)`, `fit_tables(result)` (curves, kriged data, measured points with residuals). `model.py`: the step-count check. Tests (`tests/test_template.py`): Euler and angle forms give the same `Orientation` as `Orientation.from_euler`; buffer and Pa forms; both forms and neither form are errors; the packaged example matches the paper row and profile it was taken from (same `Run` inputs and the same fit); a summary row read back as a parameter table gives the same runs; two steps without `step_fo` is an error.
2. `plotting.py`: `plot_fit(result, ax=None)`: measured points, initial condition, best fit and the 1σ band, time in the title; kriged curve only for the legacy method (the only one that fits to it). Checked with figcheck and by looking at the example's PNG.
3. `template.py` and `src/forster/data/`: `write_template(out_dir, overwrite=False)` writes `profiles.csv`, `parameters.csv` and `COLUMNS.md` (the column guide, generated from one table in code); refuses to overwrite.
4. `cli.py`, `__main__.py`, `[project.scripts] forster`: `forster template DIR [--force]`, `forster fit PROFILES PARAMETERS [--out DIR] [--run NAME ...] [--profiles-sheet] [--parameters-sheet]`, `forster --version`. `fit` writes `<run>_fit.png`, `<run>_curves.csv`, `<run>_kriged.csv`, `<run>_points.csv` and `summary.csv`, keeps going past a run that fails and exits 1 if any did; fit warnings go to the summary's `warnings` column. Tests (`tests/test_cli.py`, real subprocesses on the installed `forster` script): template then fit on the example, outputs present and the summary time equal to `fit_run` in-process; template refuses to overwrite; a run with a missing profile fails alone with exit code 1; `--run` selects and an unknown name is an error.
5. `README.md` (install, the three ways to run, the column guide, citing, the tkinter note held for phase 3) and `readme` in `pyproject.toml`; a test that every template column is documented in the README. CHANGELOG "Added" entries.

Verify once at the end: `uv run pytest -n 8`, `uv run ruff check`, `uv run ruff format --check`, `uv run ty check`, complexipy, and `uv build` with the example CSVs inside the wheel.

**Before the first release (flagged 2026-10-11):** come back to the cross-term fix (see "Engine fix" above and `CHANGELOG.md`). The release notes must say that results from versions before the fix are biased where D varies along the profile, and by how much; the round robin results must be refitted (including the SH63olv066 initial conditions) and any submitted round robin results corrected; and the Xenolith_Melt_Vein_Modeling README needs its note on the size of the effect on the paper.

## Decisions (Henry, 2026-10-11)

1. Name: **FORSTER** (`pip install forster`, `import forster`); details below.
2. Python `>=3.11`, recommended install through uv. Measured: a 20,000-step fit takes 0.247 s (median of 5) on 3.11.15 with numpy 2.4.6 and 0.270 s on 3.13.11 with numpy 2.5.4, both recovering a planted answer; dependencies resolve for 3.11. The only 3.12-only syntax is in the archived `kriging_parameter_estimation.py`, which stays out of the package. Tests run on 3.11 and 3.13.
3. Example data: one AZ18 profile from the published J. Petrology data.
4. Monte Carlo P-T-fO2 rescaling goes into the GUI and CLI, with the sampling parameters exposed: mean and 1σ for T, P, log fO2 and the Fe-Mg activation energy (default 201 ± 8 kJ/mol), number of samples, and a random seed. The seed is new: the current code has none, so rescaled times move between runs (by up to about 15% for a pooled median in the paper data).
5. Publish to PyPI and keep the GitHub install.
6. Document and cite the design decisions and every diffusion law; ask Henry when a source is uncertain.

### Name: FORSTER

**F**itting **O**livine **R**esidence & **S**torage **T**imescales from **E**lemental **R**elaxation. Free on PyPI as of 2026-10-11. The expansion names no element, so it still fits after Ni, Mn and Ca are added.

It honours Adolarius Jacob Forster (1739–1806), the London mineral dealer after whom Armand Lévy named forsterite in 1824. A check of his biography found nothing problematic about him personally. One caveat for any "named after" note: the South American crude platinum he supplied to Wollaston and Tennant (the source of palladium, rhodium, iridium and osmium) came from Chocó placers worked largely by enslaved Africans, as did most platinum of that era. Primary source to read before writing that note: Mineralogical Magazine 38, 545 ("Jacob Forster (1739–1806) and his connections with forsterite and palladium").

Searches for "Forster" collide with Förster resonance energy transfer (FRET), so the README and paper titles should say "FORSTER olivine diffusion".

Other candidates considered: PyOZONE (Olivine ZONing Estimator), olivine-time, olivine-clock, TOLD, `zonechron`, `fozone`.

## Science and citations (to confirm before they go into the docs)

Each entry is what the code implements; the source is what I believe it follows. Items marked [confirm] need Henry's check.

| Component | Code | Source |
|---|---|---|
| Fe-Mg diffusivity | `D_Fo`: 10^-9.21 (fO2/10^-7 Pa)^(1/6) 10^(3(X_Fe - 0.1)) exp(-(201 kJ/mol + (P - 10^5 Pa) 7×10^-6 m³/mol)/RT) along [001]; D[100] = D[010] = D[001]/6 | Dohmen and Chakraborty (2007), Phys. Chem. Minerals 34, 409-430 [confirm]; identical to DIPRA Table 1 (Girona and Costa 2013) |
| Activation energy uncertainty | 201 ± 8 kJ/mol in the Monte Carlo | "Dohmen et al. 2007" in the code: Dohmen, Becker and Chakraborty (2007), Phys. Chem. Minerals 34, 389-407? [confirm which paper] |
| Ni diffusivity | `D_Ni`: 3.84×10^-9 (fO2/10^-6 Pa)^(1/4.25) 10^(1.5(X_Fe - 0.1)) exp(-(220 kJ/mol + PV)/RT), same anisotropy | Petry, Chakraborty and Palme (2004), Geochim. Cosmochim. Acta 68, 4179-4188 [confirm]; identical to DIPRA Table 1 |
| Ca diffusivity | `D_Func_Ca`: separate D along [100], [010], [001] (prefactors 16.59, 34.67, 95.49 ×10^-12; E = 193, 201, 207 kJ/mol), (fO2/10^-7 Pa)^0.3 | Coogan, Hain, Stahl and Chakraborty (2005), Geochim. Cosmochim. Acta 69, 3683-3694 [confirm]; identical to DIPRA Table 1 |
| Mn diffusivity | `D_Fo` also serves Mn | Same law as Fe-Mg with E_Mn = 201 kJ/mol, as in DIPRA Table 1 (Girona and Costa 2013); DIPRA cites Dohmen and Chakraborty (2007a, b), Chakraborty (2010), Dohmen et al. (2007) and Holzapfel et al. (2007) for the Fo dependence. Resolved from DIPRA; the primary Mn source is still to pick |
| Diffusivity along the profile | D_profile = D[100] cos²α + D[010] cos²β + D[001] cos²γ | Costa and Chakraborty (2004), Earth Planet. Sci. Lett. 227, 517-530 (DIPRA eq. 4 cites it for this formula) |
| fO2 buffers | QFM and NNO expressions | Frost (1991), Rev. Mineral. 25 (cited in the code; adapted from Dan Rasmussen's code) |
| Crystal orientation | Bunge Euler angles, row-vector rotation convention | Britton et al. (2016), Mater. Charact. 117, 113-126 (cited in the code) |
| Numerical scheme | explicit finite differences, composition-dependent D, central differences for both first derivatives in the cross term (÷4 since 2026-10-11) | own implementation; same equation as DIPRA (Girona and Costa 2013, Geochem. Geophys. Geosyst. 14, 422-431, eq. 5), which uses forward differences and the analytic dD/dX_Fo for the cross term. Checked against it numerically (see "Engine fix"). FTCS sources still to cite, with a log of the search: `docs/references/numerical-methods.md` |
| Profile interpolation | universal kriging with `exact_values=False` (PyKrige; the paper version used ordinary kriging), kriging variance plus a uniform term as the data uncertainty | PyKrige software [confirm preferred citation] |
| Best fit and uncertainty | minimum chi-squared; ±1σ from the times where reduced chi-squared is within 1 of its minimum | Andrae et al. (2010), arXiv:1012.3754 is cited in comments for chi-squared pitfalls [confirm the source for the Δ reduced chi-squared < 1 criterion; it is not DIPRA's, which counts model points outside the data error bars (Girona and Costa 2013, Auxiliary Material 2)] |
| Method overview | diffusion chronometry from zoned crystals | Costa, Dohmen and Chakraborty (2008), Rev. Mineral. Geochem. 69, 545-594 [confirm, or name your preferred review] |
| P-T-fO2 rescaling | Monte Carlo over conditions, times rescaled by the mean diffusivity | Towbin and Plank (2026), J. Petrology 67(9), egag077 |
