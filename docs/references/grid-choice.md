# Choosing the numerical grid

How `forster` picks the grid spacing dx, the time step dt and the longest
modelled time, what it tells you about a grid you give it, and the
measurements behind both. Code: `src/forster/grid.py`. Scripts and data:
`docs/plans/2026-10-13-grid-choice/`.

## What happens

Leave `dx_um` and `dt_s` blank in `parameters.csv` (or pass a
`GridRequest()` from Python) and the grid is chosen for you:

1. A quick fit on a coarse grid (150 nodes, or a quarter of the narrowest gap
   if that is finer) estimates the time, and from it the diffusion length
   l = sqrt(D t), with D the diffusivity at the middle of the composition range.
2. dx is set so the fitted time is within a tolerance: dx = l × tolerance / 0.65,
   and never more than a quarter of the narrowest gap between steps of the
   initial condition (or between a step and an end of the profile).
3. dt is set at D_max dt / dx² = 0.4, with D_max the largest diffusivity over
   the compositions the profile passes through (the explicit schemes diverge
   above 0.5).
4. The longest time is 3× the coarse fit's time. Early stopping ends the run
   sooner and extends it if the 1σ range reaches the end.

| Purpose | Tolerance | l / dx | Used for |
|---|---|---|---|
| `scan` | 5% | 13 | initial-condition and parameter scans (`scan_dx_um`, `scan_dt_s`) |
| `final` | 1% | 65 | the reported fit (`dx_um`, `dt_s`) |

Any part you give is kept and the rest chosen: a `dx_um` alone gets the
stable dt for it, a `dt_s` alone gets the dx for the tolerance. Give both and
the grid is used as it is, after a check:

| Check | Result |
|---|---|
| D_max dt / dx² above 0.5 | error: `unstable grid` (the fit would diverge) |
| fewer than 4 nodes across the narrowest gap | warning: the initial condition is not resolved |
| l / dx below what the tolerance needs | warning: the time `may be off by up to` the estimated error |
| l / dx over 4× what is needed | warning: `much finer than needed` (slow, no gain) |
| dt below a quarter of the stable step | warning: dt is `below the stable step` (slow, no gain) |

summary.csv records, for every fit, `grid_resolution` (l / dx at the fitted
time), `grid_error_pct` (0.65 dx / l, the estimated error of the time) and
`grid_stability` (D_max dt / dx²). A summary row refits on exactly the same
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
nearest node. The worst cases above give the rule error ≈ 0.65 / k
(5.4% at k = 12, 1% at k = 65). It is approximate: KS20-527-8_transect-2
reaches 6.5% at k = 12.

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
first four meet the scan tolerance by the 0.65 / k rule, but not the final one.

## What would make grids cheaper

Measured on the same profiles (`gridbench.py`, `compiled.py`; the latter needs
`uv run --with numba`):

- **Non-uniform or adaptive grids:** at most 10–20% fewer steps with an
  explicit scheme, because the smallest cell still sets dt. Not worth the
  complexity.
- **An implicit scheme** (backward Euler, D lagged one step, tridiagonal
  solve): 50–200 time steps reach the same fit, against thousands for the
  explicit schemes; about 25× fewer steps, with a time-discretisation error of
  0.002–0.008 mol% in Fo. This is the real lever, and it would also put the
  boundary on a node (removing the ghost-node offset, 0.066 mol% at
  dx = 1 µm).
- **A compiled loop** (Numba): 8–10× faster per step. Per-step cost is
  overhead (25–30 µs for 50–300 nodes), so the gain is the same at any grid
  size.
