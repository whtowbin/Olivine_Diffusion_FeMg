# Self-critique (standard depth: faithfulness pass + own cold read)

## Faithfulness (P28): every count, polarity, direction, order, connection
| Item drawn | Source says | OK? |
|---|---|---|
| Two events: stage 0 then one later stage at 6 d | test case: stage 0 + GROWTH at 6 d | yes |
| Stage 0 rim at 40 µm, held at Fo 0.86; core 0.90 | EARLY in tests/test_growth.py | yes |
| Rim moves outward (40 -> 0 µm), never inward | boundary.py: rim_um at or below previous rim | yes |
| New zone Fo 0.82, rim held at 0.82 after stage 1 | GROWTH zone_value, rim default = zone value | yes |
| New zone appears as a sharp, undiffused step | instantaneous growth, zone filled at zone_value | yes |
| Distance measured from the final rim, rim on the left | plan: rim-relative coordinates | yes |
| No staged fit before 6 d | plan, Fitting; staged chi2 is undefined before 6 d in the run | yes |
| Two reported times 10.1 d and 4.1 d | computed staged fit (10.07, 4.07) | yes |
| Single-stage minimum 6.7 d, misfit as low | computed (6.74 d, chi2 64.9 vs 66.0) | yes; drawn with equal height, not as a worse fit |
| Inner gradient 10 d, outer step 4 d | stage times | yes; bracket extents are approximate regions |
| Crystal coloured concentrically from one profile | not in sources (simplification) | flagged in caption: diffusion is anisotropic |
| Crystal outline | habit unknown | generic octagon; no faces named |

## Rubric (1-5)
| Panel | Reads | Takeaway | Faithful | Correct | Links | Economy | Craft |
|---|---|---|---|---|---|---|---|
| a | 4 | 4 | 5 | 5 | 4 | 4 | 4 |
| b | 4 | 5 | 5 | 5 | 4 | 4 | 4 |
| c | 4 | 4 | 5 | 4 | 4 | 4 | 4 |
| d | 4 | 5 | 5 | 5 | 5 | 4 | 4 |
Below 5, why: text is 5.5-6 pt at 183 mm (gate wants larger; inside Nature's 5-7 pt);
the dashed zoom lines from the traverse cross the crystal; the concentric colour map is a
simplification (caption says so); 131 words (0.61 words/cm², above the 0.2-0.45 smell
test), most of them tick labels and axis titles of three data panels.

## Fixes made after the first render
- "best fit" label sat on the axis and the drop line ran through tick label "10": callout
  above-right; drop line broken around the tick labels.
- "staged model" and single-stage labels sat on their curves: moved clear.
- "6.7 d" was struck through by the axis: moved above its dot.
- "profile" label sat on the crystal with a leader through it: moved outside.
- Grey "no fit" band label could be read as applying to both models: now "outer zone not
  grown: no staged fit".
- Caption-like labels shortened ("stage 1: new rim zone", "since first stage: 10.1 d").
- Text raised from 5/5.5 pt to 5.5/6 pt; y label "Fo" only in panel a.

## Gate WARNs kept, with reasons
- Small text / words per cm²: a methods figure with three computed data panels; ticks
  carry most words. Kept within Nature's 5-7 pt.
- Caption-like labels: the two reported times and the no-fit band are key ideas 3 and 4
  verbatim; kept at 4-6 words.
- "Text over busy art" for the circled a, b, c markers: they sit on their own white
  discs (false positive).
