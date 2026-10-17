# Self-critique (standard depth: faithfulness pass of step 9, then rubric)

## Faithfulness (every count, direction, shared-vs-own choice against facts.md)
| Claim the drawing makes | Check | Result |
|---|---|---|
| two traverses, roughly orthogonal, one near c, one near a | 15° from c and 15° from a, orthogonal | OK |
| crossing point marked in a, and on both profiles in b | same ring glyph; distance from crystal.json | OK |
| distance runs from each profile's own rim | plots start at each rim; first spot at the edge offset | OK |
| profile 1 wider than profile 2 | D1/D2 = 4.24 from the angle formula with D_c = 6 D_a; width ratio 2.06 | OK |
| shared: time, T/P/fO2, segment Fo by label, edge Fo | black bracket, grey levels, black edge dot | OK |
| edge "by default" | not visible on the figure; caption says it can be freed | OK (caption) |
| own: steps, zone widths, D, drift, edge offset | coloured triangles, titles, vermillion bracket, coloured edge brackets | OK; drift drawn on profile 2 only (profile 1's drift is zero here); caption says so |
| drift applies to all segments of a profile | profile 2 curve and data shifted by the same amount at every plateau and at the rim | OK |
| joint misfit = sum | panel c black curve is p1 + p2 (computed) | OK |
| "pins the time and edge value better than either alone" | computed: true for time in this seed (0.80 of the better); edge ≈ the fast profile's (0.97); over 12 seeds the joint is sometimes wider | The figure claims less than the brief's sentence: label says "joint (sum)" only; caption gives the numbers |
| planned feature | panel c heading "Joint fit (planned)" + caption | OK |
| no habit, no face labels, no numeric ticks | checked | OK |

## Rubric (1-5)
1. Reads at final size: 4. 6 pt labels at 183 mm are inside Nature's 5-7 pt; the gate prefers larger. Thumbnail shows the three-part structure.
2. Takeaway: 4. Shared levels across two different-width curves read fast; panel c needs the caption for "1σ".
3. Faithful: 5 (see table).
4. Correct: 5. Computed curves; the 2D map uses the same D t as the profiles.
5. Links: 4. Profile colours carry a to b to c; ramp shared by a's colour bar and b's Fo strip; crossing glyph repeated.
6. Economy: 3. 89 words, 0.74 words/cm² (smell test 0.20-0.45). Every label names a thing; the axis labels "distance from rim" twice could become one.
7. Craft: 4. "edge offset" label sits close to its bracket (gate WARN); the 2D map's anisotropic blur is too subtle to see at print size.

## Fixed during the build
- empty-label callouts drew stray leaders across panel b: removed.
- Fo strip on b's axis ran dark-to-pale upward the wrong way: reversed.
- zone labels in a crowded: spread.
- panel c curve labels collided with curves: moved to the rows of the range bars.
- spot outlines 0.18-0.2 pt below the 0.25 pt floor: raised.
