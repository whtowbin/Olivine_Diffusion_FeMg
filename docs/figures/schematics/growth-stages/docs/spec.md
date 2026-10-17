# Spec: growth stages

## Structure
Skeleton: a **pipeline in time** (small multiples of one template, P3/P8) over a
**shared literal time axis**. Three snapshot panels (stage 0, stage 1 appears,
measured) use one template (crystal section above, Fo-vs-distance profile below, same
axes), so only the history changes. A full-width time panel below carries the fit and
the two reported times; small letter markers on its axis (a, b, c) tie each snapshot to
its moment. Why: the claim is a sequence of events (causal order = reading order) and a
statement about time (what the fit reports), so time must be a literal axis.

## Concepts (scored: 5-s takeaway / true to note / signal state drawn / comparison / explains)
- **A. Snapshots + time axis** (kept). a-c snapshots, d misfit vs time with the
  not-yet-grown interval greyed, both reported times as brackets, and the single-stage
  misfit as the controlled comparison. Scores: 4 / 5 / 5 / 4 / 5.
- **B. Space-time Fo map.** Distance (x) vs time (y) heat map with the domain widening
  at 6 d, slices as profiles. Literal and compact, but a 2D colour map of Fo is slower to
  read for a new reader and the reported times are less obvious. 3 / 5 / 4 / 3 / 4.
- **C. One overlaid profile plot** with curves at several times and a crystal strip.
  Densest, but the zones' different ages and the time-axis claims are lost in one plot.
  2 / 4 / 3 / 2 / 3.

**Takeaway (one sentence):** a multi-step profile records several growth events;
modelling them as stages avoids treating every zone as if it existed from time zero.
**Size:** 183 mm × 118 mm
**Hero:** none, the subject is a model history and its fit (data-like panels); route flat.
Crystal sections are flat generic outlines filled by the computed Fo map.

## Panels
| Panel | Job | Position |
|---|---|---|
| a | Stage 0: the starting crystal diffuses against its rim value (0 to 6 d) | top left, 59 × 68 mm |
| b | Stage 1: the rim moves outward; a new Fo 0.82 zone appears on an already diffused crystal | top middle |
| c | Measured profile at 10 d with the staged fit; inner gradient older (10 d) than outer (4 d) | top right |
| d | Misfit vs time since stage 0: undefined before stage 1; both reported times; single-stage comparison | bottom, full width, 46 mm |

## Components
| Panel | Component | Label | Look | Source |
|---|---|---|---|---|
| a-c | crystal section | "olivine section" (a only), "final rim" (a, ghost) | flat elongate 8-sided outline, interior filled by concentric bands coloured by Fo(distance from rim) | compute (Fo from model snapshot) + draw (generic outline; habit unknown) |
| a-c | traverse | "profile" (a) | thin ink line from left face inward, with zoom wedge to the plot x axis | draw, L.zoomLines-style lines |
| a-c | profile plot | x "distance from rim (µm)", y "Fo" | axes 0-150 µm, Fo 0.81-0.91 | compute (FORSTER curves) |
| a | t = 0 profile | "t = 0" | dashed ink step at 40 µm | compute |
| a | 6 d profile | "6 d" | solid ink | compute |
| a | not-yet-grown band | "not yet grown" | light grey hatch x < 40 | from model (inactive nodes) |
| b | new zone | "new zone" | amber fill x < 40 | compute |
| b | profile right after stage 1 | "6 d" | solid ink | compute |
| c | measured points | "measured" | small open grey circles | compute (synthetic, as the test) |
| c | staged fit | "staged model" | protagonist blue solid | compute |
| c | gradient ages | "10 d of diffusion", "4 d of diffusion" | brackets above curve | compute (stage times) |
| a-c | Fo colour key | "Fo" ramp 0.82-0.90 | small bar in panel a | compute |
| d | time axis | "time since stage 0 (days)" | 0-16 d | compute |
| d | stage bar | "stage 0", "stage 1" | two pale bands above plot | facts |
| d | not-yet-grown interval | "no outer zone yet: cannot fit" | grey band 0-6 d | facts (plan, Fitting) |
| d | staged misfit | "staged model" | blue solid, log scale | compute |
| d | single-stage misfit | "both zones from t = 0" | grey dashed | compute |
| d | best fits | "10.1 d", "6.7 d" | filled / open dots at minima | compute |
| d | reported times | "time since first stage", "time since last stage" | brackets under axis | facts + compute |
| d | snapshot markers | "a", "b", "c" | small circled letters on the time axis | links |

## Relations
| From | To | Kind | Drawn as |
|---|---|---|---|
| crystal traverse | profile x axis | zoom | thin grey converging lines |
| panel a/b/c | time axis in d | maps to (moment) | circled letters at 0-6 d, 6 d, 10.1 d |
| stage 1 start | new zone in b | causes | vertical event line in d at 6 d |

## Conventions
- Time left to right. Distance from the final rim, rim on the left, core to the right.
- One Fo colour ramp: amber (0.82) to olive green (0.90); low Fo is the lighter tone.
- Staged model = blue (protagonist) everywhere; single-stage = grey dashed; data grey.
- Same axes, scales and crystal size in a-c.

## Negative constraints
- No claim that the misfit separates the two histories (equal minima drawn as equal).
- No gradual growth (zones appear instantaneously; sharp step in b).
- No real-sample numbers, face indices or habit; no code identifiers.
