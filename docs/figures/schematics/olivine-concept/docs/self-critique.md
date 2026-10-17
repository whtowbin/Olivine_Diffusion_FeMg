# Self-critique (own pass, depth=standard)

## Faithfulness (P28), checked against docs/facts.md
| Choice the drawing makes | Check | Result |
|---|---|---|
| Number of traverses: 2 (along a, along c) | the brief asks for one traverse plus the direction; two give the controlled comparison of panel b | kept, deliberate |
| Spot count: 9 per traverse | unknown in the sources | generic, no claim; said in caption |
| Traverse starts at the rim, runs inward | N, R distance_um | correct |
| Zoning sense: core high Fo, rim zone low | not fixed by FORSTER | drawn consistently in a and b; the caption says either sense works |
| Rim value held | N "Edges" | computed curves meet the rim value at distance 0 (the c curve at t3 shows it) |
| D_c = 6 D_a, D_b = D_a | R diffusivity.py:45 | curves computed with D ratio 6, widths ratio √6 asserted in profiles.py |
| Same times in both plots | design (controlled comparison) | same D_a·t values in the script |
| Same step position along a and c | not true in general (N joint fits) | held equal only to isolate D; the caption says so |
| Axis frame handedness | right-handed a × b = c | **error found and fixed**: the first draft had b out of the page with a right and c up (left-handed); now b is into the page in a and recedes into the page in c |
| Profile in panel c vs section in a | a traverse lies in its section | **error found and fixed**: draft 2 drew the a-c section plane in c with a profile leaving it; the plane was removed and panel c now shows a general direction |
| Width bracket | spans 16-84 % of the t3 curve = 2.8 √(Dt) | labelled "~ √(D t)", a proportionality, not an equality |
| time ~ width²/D | N (time scales as 1/D at fixed D t) | correct |

## Rubric (style.md §4), 1-5
| Panel | Reads | Takeaway | Faithful | Correct | Links | Economy | Craft |
|---|---|---|---|---|---|---|---|
| a | 5 | 4 | 5 | 5 | 4 | 5 | 4 |
| b | 5 | 5 | 5 | 5 | 5 | 4 | 4 |
| c | 4 | 4 | 5 | 5 | 4 | 4 | 4 |

Below 5, and why:
- a takeaway 4: the section alone does not say "faster along c"; panel b carries it. Craft 4: the core is a large, plain area. I added no texture on purpose, since in BSE texture would read as extra zoning.
- a/c links 4: panels are linked by colour and axis labels only. No zoom wedge: it would cross the crystal.
- b economy 4: the titles "slow" and "~6× faster" overlap with the bracket label "√6 × wider". I kept both: one gives D, the other what it does to the profile.
- b craft 4: on the a-axis plot the t1 curve is almost a step at print size. That is the physics at that time (√(D_a t1) is small).
- c reads 4: the angle arcs are small at 6 pt (the α arc crosses the b axis). They read at 100 % but not on the thumbnail.

## Gate warnings, kept with reasons
- "text is small for the figure (6 pt at 183 mm)": the skill's own house style sets 5-7 pt for a Nature-style journal figure (style.md §1). Three panels need double-column width. Labels read at print size (checked at 300 dpi).
- "caption-like label: D_a cos²α + …": this is the panel's governing relation, set where it acts (P19). It is the one relation on the figure.
