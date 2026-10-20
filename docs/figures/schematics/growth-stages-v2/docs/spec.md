# Spec: growth stages, version 2

## Structure
As `../growth-stages`: a pipeline in time (small multiples of one template) over a shared
literal time axis. Now four snapshot panels (stage 0, stage 1, stage 2, measured), each a
crystal section above a Fo-against-distance plot with the same axes, then a full-width
misfit-against-time panel with three stage bars and circled letters a-d tying each
snapshot to its moment. Only the history changes from panel to panel.

**Size:** 183 mm × 116 mm (four 44.5 mm columns, 66 mm high; time panel 48.5 mm).
**Hero:** none; route flat. Crystal sections are a generic outline filled by the computed
Fo profile (concentric, as in growth-stages).

## Zone geometry (from joint-fit profile 1, read by compute_curves.py)
| Item | joint-fit value | here |
|---|---|---|
| outer step (rim zone / mantle) | p1 steps[0] = 15.16 | 15.16 µm from the rim |
| inner step (mantle / core) | p1 steps[1] = 36.04 | 36.04 µm |
| profile length | p1 x_end = 76 | 76 µm, points every 2 µm (meta spacing) |
| levels edge, rim, mantle, core | 0, 0.9, 2.0, 3.0 | Fo 0.800, 0.830, 0.867, 0.900 (linear, edge 0.80, core 0.90) |

## Panels
| Panel | Job | Position |
|---|---|---|
| a | Stage 0: core crystal (from 36 µm inward), its edge held at the mantle Fo, 0 to 0.5 d | top, column 1 |
| b | Stage 1: mantle grows (15-36 µm) at the mantle Fo; the inner gradient spreads into it, 0.5 to 1.25 d | column 2 |
| c | Stage 2 at 1.25 d: rim zone (0-15 µm, Fo 0.83) appears as sharp steps, edge held at 0.80, on a 1.25-d-old inner gradient | column 3 |
| d | Measured profile at 1.4 d (synthetic) with the staged fit; outer steps 0.15 d old, inner step 1.4 d | column 4 |
| e | Misfit against time since stage 0: no staged fit before the last stage; both reported times; comparison fit with all zones from t = 0, its time and what the near-step check and residuals say | bottom, full width |

## Conventions (unchanged from growth-stages)
- Time left to right; distance from the final rim, rim on the left.
- Staged model blue; comparison grey dashed; data open grey circles; not-yet-grown grey;
  a new zone tinted with its own Fo colour.
- Same axes and crystal size in a-d; no ticks on the crystal; computed curves only.
- The stage bars in e take the Fo colour of the zone each stage grows.

## Negative constraints
- No claim that the near-step check flags the comparison (it does not on this profile).
- No gradual growth; no fitted stage times; no real-sample numbers or habit.
