# Choosing the numerical grid

How `forster` picks the grid spacing dx, the time step dt and the longest
modelled time, what it tells you about a grid you give it, and the
measurements behind both. Code: `src/forster/grid.py`. Scripts and data:
`docs/plans/2026-10-13-grid-choice/`.

## What happens

The default scheme is `implicit` (since 2026-10-13; section "The implicit
scheme" below). Leave `dx_um` and `dt_s` blank in `parameters.csv` (or pass a
`GridRequest()` from Python) and the grid is chosen for you:

1. A quick fit on a coarse grid with the explicit scheme (150 nodes, or a
   quarter of the narrowest gap if that is finer) estimates the time, and from
   it the diffusion length l = sqrt(D t), with D the diffusivity at the middle
   of the composition range.
2. dx is set so the fitted time is within a tolerance. Implicit: dx = l / 8 for
   the final fit and l / 4 for scans (the error is about 0.2 (dx / l)²; the start
   is averaged over each node's cell, so narrow zones need no extra nodes).
   Central and dipra: dx = l × tolerance / 2.4, and never more than a quarter of
   the narrowest gap between steps of the initial condition (or between a step
   and an end of the profile).
3. dt: implicit, t × tolerance / 4, that is 400 steps up to the fit (80 for
   scans), because the scheme is stable at any dt and the best fit is a whole
   time step. Central and dipra: D_max dt / dx² = 0.4, with D_max the largest
   diffusivity over the compositions the profile passes through (they diverge
   above 0.5).
4. The longest time is 3× the coarse fit's time. Early stopping ends the run
   sooner and extends it if the 1σ range reaches the end.

| Purpose | Tolerance | l / dx (implicit) | l / dx (central, dipra) | Used for |
|---|---|---|---|---|
| `scan` | 5% | 4 | 48 | initial-condition and parameter scans (`scan_dx_um`, `scan_dt_s`) |
| `final` | 1% | 8 | 240 | the reported fit (`dx_um`, `dt_s`) |

**Correction (2026-10-13, applied).** The explicit rule was first dx = l ×
tolerance / 0.65 (l / dx = 65 for 1%, 13 for 5%). Against a converged
reference, that final grid is off by a median 1.1% and up to 3.6% on the 35
real profiles, above 1% on 18 and too long on 28: its calibration (below) used
an explicit k = 60 reference, which carries the same first-order bias. The
coefficient is now 2.4, which makes explicit final grids about 50× slower than
before; this is why the implicit scheme became the default.

Any part you give is kept and the rest chosen: a `dx_um` alone gets the dt
of step 3 (for central and dipra, the stable dt on that spacing), a `dt_s` alone
gets the dx for the tolerance. Give both and the grid is used as it is, after a
check. For central and dipra:

| Check | Result |
|---|---|
| D_max dt / dx² above 0.5 | error: `unstable grid` (the fit would diverge) |
| fewer than 4 nodes across the narrowest gap | warning: the initial condition is not resolved |
| l / dx below what the tolerance needs | warning: the time `may be off by up to` the estimated error |
| l / dx over 4× what is needed | warning: `much finer than needed` (slow, no gain) |
| dt below a quarter of the stable step | warning: dt is `below the stable step` (slow, no gain) |

For implicit there is no stability limit or gap check; dx is checked the same
way against 0.2 (dx / l)², and dt against step 3: above it, a warning that it
leaves too few `time steps up to the fit`; below a quarter of it, that it is
smaller than needed.

summary.csv records, for every fit, `grid_resolution` (l / dx at the fitted
time), `grid_error_pct` (the estimated error of the time: 0.2 (dx / l)² implicit,
2.4 dx / l explicit) and `grid_stability` (D_max dt / dx²; empty for implicit). A summary row refits on exactly the same
grid: the chosen dx and dt are written back.

From Python: `choose_grid(profile, conditions, orientation, initial,
diffusivity, purpose=...)`, `check_grid(grid, ...)`, `stable_dt(dx_um,
species, conditions, orientation)`; `species` is a list of
`(diffusivity, (fo_min, fo_max))`, so several elements share the smallest
stable step.

## Measurements

### Accuracy against resolution

Six real profiles (three round robin, three from Towbin and Plank 2026),
each fitted on grids with dx = l / k, dt at ratio 0.4, against a reference
at k = 60. Error is |t_k / t_60 − 1|, in percent (`calib_grid2.py`, steps
left where the grid puts them):

| Profile | k = 12 | 20 | 30 | 40 |
|---|---|---|---|---|
| AZ18 WHT01_Ol_63_prof_1 | 1.5 | 0.3 | 0.1 | 0.6 |
| AZ18 WHT01_bigol1_overgrowth_prof_1 | 1.3 | 0.8 | 0.1 | 0.1 |
| KS20-527-8_transect-2 | 6.5 | 3.9 | 1.8 | 2.0 |
| KS20-527_2-transect-1 | 3.5 | 8.1 | 6.5 | 5.7 |
| SH63olv066-1 | 0.5 | 1.4 | 0.4 | 0.3 |
| Xenocryst_ol6_prof_1 | 4.5 | 2.5 | 1.8 | 0.5 |

Below k = 8 the fits are unreliable (`calib_grid.py`): at k = 1–3 the
worst error is 34–196% on five of the six profiles (3% on SH63olv066-1), and
at k = 5–8 it is up to 24%.

The error falls roughly as 1/k, not 1/k² as the scheme's interior order
would suggest, for two reasons: the boundary value sits on a ghost node one
dx outside the profile, and each step of the initial condition snaps to the
nearest node. The worst cases above gave the rule error ≈ 0.65 / k
(5.4% at k = 12, 1% at k = 65). That was too small: these errors are measured
against an explicit k = 60 reference with the same first-order bias. Against a
converged reference the coefficient is about 2.4 (see the correction above and
"All 35 real profiles" below).

**Step snapping.** Repeating the study with the grid shifted so the first
step falls exactly on a node (`aligned` in `calib_grid2.csv`):

| Profile | snapped, k = 20 / 40 | on a node, k = 20 / 40 |
|---|---|---|
| KS20-527_2-transect-1 | 8.1 / 5.7 | 1.8 / 1.4 |
| KS20-527-8_transect-2 | 3.9 / 2.0 | 2.9 / 1.0 |
| others | within 0.6 points of each other | |

KS20-527_2-transect-1's large and non-monotonic errors are snapping: its
fitted time moves by a few percent as the step jumps between nodes. The
grid rule does not remove this; a scheme that puts steps on nodes would.

### Cost

One fit at k = 12 takes 0.02–0.03 s on five of the profiles (0.5 s on
SH63olv066-1, the longest); at k = 60, 0.5–2.8 s. The `final` rule
(k = 65) is therefore about 25× the cost of the `scan` rule, which is why
scans use the coarser grid.

### The time step

Raising dt to ratio 0.4 changed fitted times by under 0.2% on the round
robin profiles (`docs/references/likelihood-method.md`, section 7); coarsening
dx by the same speed-up biased them by up to 12%. The existing parameter
tables use ratios of 0.007–0.17, so they are stable but take 2–60× more
steps than needed.

### Existing grids

The parameter tables use dx = 1 µm (round robin) and 0.5 µm (paper). For the
six profiles above that is k = 29 and 36 (KS20-527_2-transect-1,
KS20-527-8_transect-2), 22 and 18 (Ol_63, bigol1), 6 (SH63olv066-1) and 5
(Xenocryst_ol6_prof_1, whose time is about 7% off its converged value). The
first four met the scan tolerance by the original 0.65 / k rule, but not the final
one; by the corrected 2.4 / k rule none meets even the scan tolerance (k = 48).

## The implicit scheme

`scheme = implicit` (`src/forster/implicit.py`, 2026-10-13) changes three things
at once: the rim and core values are held on the end nodes (the grid has a node
exactly on each end of the domain and on every growth stage's rim, even spacing
in between); the start is the step function averaged over each node's cell; and
time steps are TR-BDF2 (a trapezoidal stage, then BDF2), which is L-stable,
second order and one-step, so a run continued from any row is identical to one
run. D is evaluated at the start of each stage, and each stage is one
tridiagonal solve.

### Space: the boundary and the start must both change

Fitted time against a fine reference (implicit, l / dx = 160), on the six
profiles above, with the explicit stable dt so only dx matters
(`explicit_control.py`). Worst case over the six, in percent:

| l / dx | ghost rim, snapped steps (explicit) | ghost rim, averaged | rim on node, snapped | rim on node, averaged |
|---|---|---|---|---|
| 5 | 37 | 19 | 25 | 0.68 |
| 8 | 29 | 12 | 22 | 0.16 |
| 12 | 11 | 7.7 | 8.1 | 0.12 |
| 20 | 11 | 4.6 | 8.6 | 0.03 |
| 30 | 2.6 | 3.1 | 2.1 | 0.01 |

Either change alone helps somewhat but leaves errors of 2–25%: averaging the
start keeps the first-order error of the ghost rim, and a rim on a node keeps
the snapping noise. The largest errors are on the round robin profiles
(KS20-527_2-transect-1 and KS20-527-8_transect-2). Together the
error falls as about 0.2 (dx / l)², which `forster.grid` uses.

### Time

At l / dx = 60, against 4000 steps up to the fit (`implicit_study.py`,
`trbdf2_check.py`), worst case over the six profiles:

| steps up to the fit | backward Euler | BDF2 | TR-BDF2 |
|---|---|---|---|
| 10 | 5.2% | 1.6% | 1.1% |
| 20 | 2.6% | 0.54% | 0.49% |
| 50 | 1.0% | 0.15% | 0.17% |

TR-BDF2 was chosen over BDF2 (as accurate) because it keeps no history. The
larger part of the time error in a fit is that the best fit is a whole step:
with 100 steps up to the fit the chosen grids were up to 1.0% off on three
profiles, with 400 steps 0.3%.

### All 35 real profiles

Each fitted with the chosen `final` grid in both schemes, against an implicit
reference with dx / 3 and dt / 2 (`implicit_sweep.py`, `implicit_sweep2.csv`):

| | explicit (central) | implicit |
|---|---|---|
| error of the fitted time, median | 1.1% | 0.13% |
| error, worst | 3.6% | 0.28% |
| profiles above 1% | 18 of 35 | 0 |
| median grid | 623 nodes × 30 000 rows | 78 nodes × 1000 rows |
| time per fit, median (whole `fit_run`) | 0.9 s | 0.13 s |
| all 35 | 46 s | 11 s |

The explicit errors are mostly positive (28 of 35): its rim one node outside
the crystal makes diffusion look slower. Halving the explicit dx moves its time
halfway to the implicit value each time, as a first-order error should, and
the extrapolated limits agree with the implicit reference to about 1%.

The implicit fit is 1.3–17× faster end to end (median 6.6×) although its model
run is about 220× less work: what remains is the coarse explicit fit that
estimates the time, and kriging the profile for plots. Building the kriging
model (pykrige's leave-one-out statistics, up to 0.45 s for 300 points) was 89%
of an implicit fit before it was cached per profile (`engine._kriging_model`;
results identical).

### Refitting the existing tables

The round robin and paper parameter tables give their own grids (dx = 1 and
0.5 µm, dt = 600 and 60 s). Refitting them with both schemes on those grids,
against the same converged references (`old_tables.py`, `old_tables.csv`):

| | central (old default) | implicit (new default) |
|---|---|---|
| error of the fitted time, median | 2.6% | 0.05% |
| error, worst | 8.7% | 0.48% |
| all 35 fits | 20 s | 40 s |

So switching the default moves these fits by a median 2.7% and up to 8.1%
(28 of 35 shorter), all toward the converged value. Their time steps were set
for the explicit stability limit and are 10–100× smaller than the implicit
scheme needs, which is why it is slower here; with `dx_um` and `dt_s` left
empty the grids are chosen and the fits take a median 0.13 s. An implicit step
costs about 22 µs for 150 nodes (explicit: 9 µs), after calling LAPACK's
tridiagonal solver directly instead of `scipy.linalg.solve_banded`, whose checks
were half the cost (identical results).

### Boundary values that change

The rim value can be a number or change in time: `fit_profile(...,
scheme="implicit", rim_fo=...)` takes a `forster.boundary.RimPath(times_days,
values)` (linear between the points, constant outside them) or any function
of the time in seconds, and so does each growth stage's `rim_value`. Growth
stages start at their exact times (a time step that crosses a stage start is
split there), and the node on an old rim takes the average over its cell when
the new zone appears. Early stopping waits until a `RimPath` has stopped
changing; with another function the model runs to `max_time_days`. Tests
(`tests/test_implicit.py`): a fixed rim against erfc and a rim rising linearly
in time against its closed form (Carslaw and Jaeger 1959, section 2.5), both
second order; a staged run against an explicit on-node solver to 1e-4 Fo.

## Other ways to make grids cheaper

Measured on the same profiles (`gridbench.py`, `compiled.py`; the latter needs
`uv run --with numba`):

- **Non-uniform or adaptive grids:** at most 10–20% fewer steps with an
  explicit scheme, because the smallest cell still sets dt. Not worth the
  complexity.
- **A compiled loop** (Numba; done 2026-10-13, `forster.compiled`, optional
  `fast` extra): per step for 150 nodes, central 15 → 0.9 µs, dipra 20 → 0.9 µs,
  implicit 22 → 3 µs. End to end on the 35 profiles, default fits are about 5×
  faster (12 s → 2.2 s for all 35), fits on the tables' own grids 5–7× and a
  125-candidate scan 5–20× (timings on a loaded machine; the ratios held across
  four runs). What remains of a default fit is mostly the noise-model fit.
