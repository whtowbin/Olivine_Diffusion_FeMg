# Review fixes (2026-10-14 repo review)

**Goal:** fix the defects the 2026-10-14 review confirmed, document where the D t
shortcut behind the law range holds and where it does not, and remove the dead and
duplicated code it found, with no change to any default result.

**Approach.** Three groups, committed separately. (A) Behaviour fixes, each with a
test that fails before the fix. (B) Documentation: the law-range limits note and the
docs that contradict the code. (C) Refactors that must not change any number,
proved by a differential replay (the same seeded inputs through the code at the end
of group A and through the refactored code; every `FitResult` array compared exactly;
a baseline-against-itself control), and for `analyses/femg_refit` a byte-for-byte diff
of every result table.

**Out of scope:** a law range for element fits, scans, growth stages or changing rims
(the note says what each would need); a predictor-corrector to make TR-BDF2 second
order with Fo-dependent D (the time error is 1e-3 relative at 10 steps, below the
grid's); dropping the legacy method or pykrige (Henry's call); column-value
validation of `Conditions` / `Orientation` (a separate design question).

## Measurements behind the decisions (2026-10-14)

Taken with `femg_refit2026` as first exported; it was refitted to the verified data
in 1278566 and `law_limits.py` rerun (current numbers in the limits note; the D t
fractions are unchanged).

- D t scaling (synthetic profile, `femg_refit2026`, RR1200 conditions (the round robin 1200 C conditions), fit with D and
  with D x 10^delta; expected shift -delta in log10 t):

  | case | delta 0.122 | delta 0.301 |
  |---|---|---|
  | fixed rim (control) | -0.1213 | -0.3003 |
  | fixed `rim_fo` != edge Fo | -0.1223 | -0.3018 |
  | `RimPath` 0.86 -> 0.82 over days 0-3 of 6 | -0.0803 | -0.1714 |
  | `RimPath` 0.86 -> 0.82 over days 0-0.5 of 6 | -0.1166 | -0.2847 |
  | growth stage at day 3 of 6 | -0.0723 | -0.1477 |

- Linear sigma (g' C g) against a Monte Carlo over the preset's covariance (4000
  draws), 3 conditions x 4 orientations: agree to 1% (sd 0.1212 vs 0.1228 at the
  worst, oblique); the 16-84% half-width agrees to 3%.
- TR-BDF2 with the DC07 D on a 0.80/0.90 step: max error vs a fine-dt reference
  9.0e-5, 4.4e-5, 2.1e-5, 1.0e-5 for dt = 0.2 ... 0.025 d (ratio 2.05): first order.

## Group A: behaviour fixes

1. **Law range at time zero** (`uncertainty.widen_log10`): `time == 0` returns
   `(0.0, high * 10**sigma)` (the law's error scales the bound; nothing to combine in
   quadrature with), `high == 0` gives `(0.0, 0.0)`. Test
   `test_widen_log10_at_time_zero`; `test_summary_row_of_a_time_zero_fit_with_the_preset`
   (flat profile, `femg_refit2026`: `summary_row` no longer raises).
2. **No law range with a changing rim** (`model._law_uncertainty` takes `rim_fo`;
   None unless `rim_fo is None or is_fixed(rim_fo)`; error text of
   `time_range_days` names both causes). Test `test_no_law_range_with_a_changing_rim`
   (RimPath -> None; fixed `rim_fo` -> a sigma).
3. **Stage start on a step** (`model._implicit_staged_prefix`): the loop compares
   `start_s <= j * dt * (1 + 1e-9)`, matching `offset`'s tolerance. Test
   `test_a_stage_starting_one_ulp_after_a_step` (start `np.nextafter(637 * dt, inf)`,
   implicit: finite curves, equal to the start at `637 * dt`).
4. **Explicit stability over every held Fo** (`model.fit_profile`): with an explicit
   scheme, D_max over the initial condition, every stage's zone and held rim value and a
   fixed `rim_fo`; `ValueError` above `grid.UNSTABLE_RATIO`. Test
   `test_explicit_fit_refuses_an_unstable_stage` (zone 0.75, dt stable for 0.85-0.90).
5. **Coarse fit and scans size dt on their own scheme's spacing**
   (`grid._coarse_time_days`, `scan.scan_grid`, the explicit check in `scan`):
   `actual_dx_um(..., scheme)`. Changes dt (larger, still stable) for explicit scans and
   the coarse estimate; the suite's tolerances must hold. `scan`'s 0.4 and 0.5 become
   `grid.STABILITY_RATIO` / `grid.UNSTABLE_RATIO`.
6. **CLI batch** (`cli.fit_command`): any exception in one run is reported
   (`failed: <Type>: <message>`) and the batch continues; `summary.csv` and
   `diffusivities.csv` are written for the runs that fitted. `KeyError` messages print
   without the repr quotes. Test `test_a_run_that_raises_does_not_lose_the_summary`
   (a run whose diffusivity raises a non-ValueError, through the real CLI subprocess).
7. **Input handling** (`io`): `Marked_bad` / `Ignore` matched stripped and
   case-insensitive; `_plateau_fraction` turns a `SyntaxError`/`ValueError` into
   `ValueError("inflection_c ...")`; `read_table` refuses `.xls` (openpyxl cannot read
   it) with the .csv/.xlsx message; `read_runs` warns of columns a template table
   does not know (the misspelt-optional-column case); `read_profiles` drops rows with no
   distance or Fo (added during the work: the old reader passed NaN distances into 6
   of the 37 paper-workbook profiles, among them the measured transect
   `AZ18 WHT06_ol47_not_onMap_prof 1`, 1 of 39 points; no paper run fits it). Tests
   `test_flags_are_case_insensitive`, `test_bad_inflection_c_is_a_value_error`,
   `test_xls_is_refused`, `test_unknown_template_column_warns`,
   `test_rows_without_distance_or_fo_are_dropped_with_a_warning`.
8. **TR-BDF2 order**: docstrings (`implicit.py`, `_kernels.py`) say first order in time
   when D depends on Fo (D frozen at each stage's start), second order for constant D;
   test `test_time_order_with_an_fo_dependent_d` pins the measured ratio (1.7-2.4).

Verify: `uv run pytest -q -n auto`, with and without `FORSTER_NO_NUMBA=1`; ruff, ty.

## Group B: documentation

9. **`docs/references/law-uncertainty-limits.md`**: what the law range is, the D t
   argument, where it holds (measured), where it fails (changing rims, stages,
   elements, scans, time zero, model-form error, extrapolation), the linear vs Monte
   Carlo check, the composition-term approximation, the independence assumptions.
   Linked from README, the plan `2026-10-13-law-uncertainty.md`,
   `diffusivity-uncertainty-sources.md` and the `time_range_days` docstring.
10. **Docs that contradict the code**: `model.py` module docstring (default scheme),
    `fit_profile` docstring (explicit grid l / 240), `implicit.py` (2.4 dx / l),
    `io.py` (scheme column), `boundary.py` and `growth-stages.md` (implicit built),
    README (implicit default; example grid), plan status lines (`femg-refit`,
    `restructure`), `fit_models.py` header (five forms, six methods),
    `grid-choice.md:139`.
11. **Preset pasted once**: `export_preset.main` asserts
    `preset(fit) == DIFFUSIVITY_PRESETS["femg_refit2026"]` and imports its source text.

## Group C: refactors (no number changes)

12. Dead code: `orientation.py` (`Sample2Crystal`, `Crystal2Sample2`,
    `Sample2Crystal2`, `vector_direction2`, `trend_plunge`, `plot_olivine_stereonet`),
    `engine.py` (`VectorMaker`, `boundary_cond`, `diffusion_step_Ca`,
    `timestepper_Ni_Mn`, `Best_fit_R2`, commented-out blocks), unused imports;
    `mplstereonet` out of the dependencies if nothing else imports it.
13. `src` duplication: `InitialCondition.fo_values` (edge, [step], core) used by
    `fo_range`, `_intervals`, elements and scan; one extend-while-the-range-reaches-the-end
    loop; `posterior_interval` reused by `scan.fit_candidates`; `diffusivity.from_row`
    over `_SCALARS`; one cos^2 helper; `SECONDS_PER_DAY` once; `ELEMENT_SCHEMES` =
    `EXPLICIT_SCHEMES`; one "compiled, else NumPy" helper; `Advance` once.
14. `analyses/femg_refit` duplication: one `STUDY_STYLE`, `FORM_STYLE`, `METHODS`,
    conditions, `R`, preset log D, move/correction helper, `fit_all`, in `fit_models`.

Verify 12-13: replay script `docs/plans/2026-10-14-review-fixes/replay.py` (35 real
profiles x central/dipra/implicit x Numba on/off, staged and RimPath fits, element
and joint fits, one scan), group-A commit vs working tree, every array `array_equal`,
plus a control. Verify 14: rerun every script; `results/*.csv` and
`figures/bowties/index.csv` byte-identical to a run of the old scripts.

## Status

- [x] Group A (2026-10-14): every new test failed on the old code except the time-order
  one, which pins existing behaviour; suite 219 passed (210 + 9 skipped without Numba).
- [x] Group B (2026-10-14): `docs/references/law-uncertainty-limits.md` with
  `law_limits.py` (writes `dt_scaling.csv`, `linear_vs_mc.csv`); export_preset checked
  both ways (equal: exit 0; a 1e-15 change to the shipped preset: exit 1).
- [x] Group C, tasks 12-13 (2026-10-14): replay of 161 cases (9510 arrays, SHA-256 of
  each) at 0e8a1c4 against the refactored tree: identical with Numba and without; the
  baseline against itself: identical; Numba against NumPy at the baseline: 588 arrays
  differ (the compiled steppers' rounding), so the comparison does see changes.
  Decisions: kept the Mn and Ni presets written out (each reads as its source; `replace`
  on the DC07 preset would make a subclass); kept `from_row`'s explicit keywords (ty
  cannot check a `**` dict); kept `_round_robin_run` / `_paper_run` (a shared helper
  needs nine parameters to save six lines); kept `scan._extend_tails` (a different loop:
  several runs); guarded the hand-written preset list and `DIFFUSIVITY_COLUMNS` with
  tests instead of generating them; `mplstereonet` moved to the analysis group (the
  archived `old Xenolith Code` copy still imports it).
- [x] Group C, task 14 (2026-10-14): `fit_models` gains `STUDY_STYLE`, `METHODS`,
  `fit_all`, `moved`, `move_sd`, `preset_log10_d` (`_dc07_preset` and every name the
  melt-contact analysis imports are unchanged). Every script rerun: all `results/*.csv`
  and `figures/bowties/index.csv` unchanged in git, every script's stdout and all 31
  PNGs byte-identical to a run before the change (two runs before it were identical to
  each other). `fo2_figures` keeps its own arithmetic for the moved value: the shared one
  (from ln D) changed 861 of 2 million pixels by up to 2/255. Kept per figure:
  `FORM_STYLE` (the figures show different forms and legend lengths on purpose), the
  `Vars` builders and `fit_ols.ols` (`fit_ols` is imported by `fit_models`).
  Found, not changed: the committed fig1, fig2, fig5 and fig6 PNGs are not what the
  committed code draws (fig5: 861 pixels; same bytes on every rerun since); regenerating
  them is a one-line commit for Henry to approve.
