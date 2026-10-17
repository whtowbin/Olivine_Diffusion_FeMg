# Spec card: zoned-olivine concept

## Structure
Skeleton: **a setup with its signal, then a controlled comparison.** Panel a is the setup (the
section of a zoned crystal in melt, two traverses along different axes). Panel b is the signal
read along those traverses: the same step relaxing over the same times, drawn on one template
for the a-axis and the c-axis, so the only change is D (P3). Panel c generalises the direction
(any traverse, angles to the three axes) and closes the loop to the clock. The traverse colours
link a to b (P15); a zoom wedge links the c-axis traverse to its plot (P9).

## Concepts (scored)

| Concept | Panels | Sketch | 5-s takeaway | True to note | Signal state (P1) | Comparison (P3) | Explain (P4) | Score |
|---|---|---|---|---|---|---|---|---|
| A. Section + time series + anisotropy pair | a section with two traverses; b two matched plots (along a, along c), each t0..t3; c triad with general direction and the D relation | traverse colours carry into the plots; zoom wedge | high: wider curves along c read at once | yes | yes, the profile is the state | yes, matched pair | explains | **best** |
| B. Profiles drawn on the traverses | one big section with profile glyphs drawn along each traverse line | compact, but curves on a tilted/vertical line are hard to read, and time series would clutter | medium | yes | yes | weak (two orientations of plot) | explains | 2nd |
| C. 3D crystal + one plot | oblique 3D generic crystal with axes; one plot of t0..t3 | 3D habit would claim a morphology that is unknown; anisotropy only in text | low | risk (habit) | yes | no | impresses | 3rd |

Kept: **A**.

**Takeaway (one sentence):** The relaxed width of a growth step, read with the right D for that direction, is a clock.
**Size:** width 183 mm × height 72 mm
**Hero:** zoned olivine section in melt, drawn flat with the illustrated finish (L.finish shaded body, route flat). A polished section is flat by nature, and a 3D habit would claim a morphology the note does not give.

## Panels
| Panel | Job (one line) | Position / share of the figure |
|---|---|---|
| a | the crystal in melt, its zoning, two microprobe traverses from the rim, the axes | left, ~36 % |
| b | the same step relaxing at the same times along a and along c: width records D t, and c is ~6× faster | centre, ~42 % (two matched plots) |
| c | any direction: angles to a, b, c set D along the profile; width + D → time | right, ~22 % |

## Components
| Panel | Component | Label (exact text) | Look | Source |
|---|---|---|---|---|
| a | melt field | "melt" | pale warm grey field with faint texture, rounded panel-sized blob | draw (L.path + L.finish off) |
| a | crystal outline + rim zone | "olivine", "rim zone" | olivine green (lighter, lower-Fo rim zone band; deeper green core), illustrated finish | draw (L.finish, L.smooth on a generic polygon) |
| a | core | "core" | deeper olivine green | draw |
| a | traverse along a | "along a [100]" colour A | spot dots (L.particles) on a line from the rim inward | draw |
| a | traverse along c | "along c [001]" colour C | spot dots | draw |
| a | axis triad | "a [100]", "c [001]", "b [010]" (⊙) | small arrows, ink | draw (L.arrow, L.circle) |
| a | fast/slow glyph | "~6× faster" next to c arrow | label only | draw |
| b | two plots, same axes | "along a [100]", "along c [001]"; axes "distance from rim", "Fo" | thin axes, no ticks; t0 dashed ink step, t1..t3 sequential ramp of the direction's hue | compute (profiles.py) |
| b | time labels | "t_0" … "t_3" | small labels at curves | compute positions |
| b | width bracket | "width ~ √(*D t*)" (a), "√6 × wider" (c) | bracket above the curves at t3 between the 16 % and 84 % points | compute |
| b | plateau labels | "rim", "core" | small grey | draw |
| c | triad in oblique view | "a" "b" "c" | three L.arrow axes through a hand-made oblique map (L.axes3 rotates its labels; plain upright letters wanted) | draw |
| c | profile direction | "profile", angles "α" "β" "γ" | ink arrow; angle arcs are projected 3D great-circle arcs; dashed projection onto the a-c plane as a depth cue | draw (computed arcs) |
| c | relation | "*D*_{profile} = *D*_a cos²α + *D*_b cos²β + *D*_c cos²γ" | 6 pt, symbols coloured | draw |
| c | clock line | "time ~ width² / *D*_{profile}" with a clock glyph | 6 pt | draw |

## Relations
| From | To | Kind | Drawn as |
|---|---|---|---|
| traverse c (panel a) | plot "along c" (panel b) | maps to | shared colour and title; each traverse is drawn to the plot's distance scale (traverse length = 2.6 × rim-zone width, as the plot's x range is 2.6 × the step position). Zoom lines were dropped: they would cross the crystal |
| traverse a (panel a) | plot "along a" (panel b) | maps to | shared colour |

## Conventions
- Distance runs left to right from the rim (0) inward in every plot; t0 is the dashed ink step, later times run light → dark in the direction's hue, with a time arrow toward the wider curves.
- Right-handed frame everywhere: a right, c up, b into the page (circled cross in a; receding up-right in c).
- a-axis colour = blue (Wong #0072B2), c-axis colour = vermillion (#D55E00); olivine green for the crystal only.
- Same x scale and same times in both plots of panel b.

## Negative constraints (must not appear)
- No face indices, no habit names, no numeric ticks, no real Fo, distance or time values.
- No b-axis profile drawn as different from a.
- No code identifiers.

## Look-up (step 4)
Web search for zoned olivine BSE images and traverse schematics returned text only (no image
saved to docs/refs). What a reader recognises: a polished olivine section in glass, a bright
(lower Fo, Fe-richer) rim band in BSE around a darker core, analyses as a line of spots
from the rim inward, traverses chosen parallel to an axis. Drawn in colour (green olivine)
rather than BSE grey, since colour carries the direction key.
