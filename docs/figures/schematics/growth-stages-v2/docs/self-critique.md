# Self-critique (standard depth: faithfulness pass + own cold read)

## Faithfulness: every count, polarity, direction, order, connection
| Item drawn | Source says | OK? |
|---|---|---|
| Three events: stage 0, stage 1 at 0.5 d, stage 2 at 1.25 d | compute_curves.py (CORE, MANTLE, RIMZONE) | yes |
| Steps at 15.2 and 36.0 µm, profile 76 µm, levels 0.80/0.83/0.867/0.90 | joint-fit profiles.json p1, read by the script | yes |
| Stage 0 edge held at the mantle Fo | CORE edge_fo = MANTLE_FO | yes; a shows the blue curve pinned at 0.867 at 36 µm |
| Stage 1 adds no step (zone Fo = held Fo) | MANTLE zone_value = rim value | yes; b shows a flat extension, then spreading |
| Stage 2 zone 0.83, edge then held at 0.80 | RIMZONE rim_value 0.80, zone_value 0.83 | yes; c shows two sharp steps |
| Rims only move outward (36 -> 15 -> 0 µm) | boundary.py check | yes |
| Inner step older and smoother than outer | ages 1.4 d vs 0.15 d; sqrt(Dt) 5.5 vs 2.0 µm | yes, visible in d (curve and crystal) |
| No staged fit before the last stage | staged chi2 undefined before 1.25 d | yes (grey band ends at 1.25 d) |
| Reported times 1.39 d and 0.14 d | staged fit (1.392, 0.142) | yes |
| Comparison minimum 0.21 d, misfit as low | computed (0.207 d, chi2 38.1 vs 39.0) | yes; drawn at its computed height |
| "no near-step warning" | near_step single_delta_aic -3.2 > -6 | yes; the label switches automatically if the data change |
| "residuals 4.3 × noise" | single_rms 0.0043 / 0.001 | yes |
| Markers a-d | stage starts 0, 0.5, 1.25 d; d at the best fit | yes; caption says so |
| Crystal coloured concentrically | simplification | flagged in caption |

## Fixes made after looking at the renders
- "final rim" collided with the colour key's "Fo": the callout moved below the crystal.
- "not yet grown" in b was wider than its 15 µm band: set on three lines.
- The "0.5 d" label in b sat on the curves: moved above-left with a leader to the dashed
  curve where it lies above the blue one.
- "edge held at 0.80" sat on the amber band: label moved right, outside the band.
- The comparison's callout crossed its own "0.21 d" label and then the stage-1 line:
  merged into one three-line label above the dot, left of the 0.5 d line.
- "staged model" in e sat on the blue curve: moved below it.
- Panel e's heading touched marker a: the stage bars and plot moved down 1.5 mm; the
  empty band between the rows was closed (height 122 -> 116 mm).

## Data-side checks
- The first independent solver held the rim one node outside the profile (as the one in
  growth-stages does), which at these short times put the edge 0.5 µm too far out and
  biased the staged time since the last stage by 13% (0.26 d for 0.30 d). The solver now
  holds each rim on a node at its own position on a 0.1 µm grid; it agrees with FORSTER's
  implicit model to 9e-5 Fo, and the staged fit recovers 1.39 d (true 1.4) and 0.14 d
  (true 0.15).
- The "after stage" rows are found as the first row in which the new zone exists, not
  from round(start / dt) (with the implicit scheme that row can still be before the stage: it was at a 2 d stage start).
- The near-step result was checked over 10 noise draws (in curves.json): comparison
  flagged in 2, staged in 0; the comparison's residuals 3.9-4.5 times the noise in all.

## Gate WARNs kept, with reasons
- Small text / words per cm²: as growth-stages; 5.5-6 pt at 183 mm is inside the journal
  5-7 pt; four data panels with ticks carry most words.
- Caption-like labels ("rim zone not grown: no staged fit", the comparison's three-line
  label, the two reported times): key ideas 4 and 5, kept short.
- "Text over busy art" for the circled a-d markers: they sit on white discs (false
  positive, as in growth-stages).
