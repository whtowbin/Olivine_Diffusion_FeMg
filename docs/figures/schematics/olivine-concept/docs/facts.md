# Facts and geometry record

Sources: N = research note `FORSTER_FACTS.md` (scratchpad/forster-schematics). R = FORSTER repo,
read only: `README.md` (parameter tables, diffusivity notes), `src/forster/diffusivity.py`.

## Geometry record (the crystal in panel a), written before drawing

The object is a **polished section** through one zoned olivine crystal, seen from above, as
an electron microprobe analyses it. It is drawn flat (2D) because the analyses lie on that
flat surface.

| Item | Record | Source |
|---|---|---|
| Section plane | contains the a [100] and c [001] axes, so b [010] is normal to it (drawn pointing into the page) | chosen for the schematic (a common choice for two orthogonal traverses); not from a real crystal |
| Crystal outline | generic convex polygon with blunted corners, close to equant; **no face labels** | habit **unknown** (N: "do not claim a specific habit or face indices") |
| Order of materials, outside in | melt (glass) → crystal rim (edge 1, in contact with melt) → rim zone (lower Fo) → step → core (higher Fo) | N "Edges": edge 1 is the crystal rim in contact with melt; one step between two segments |
| Zoning sense | normal: core Fo higher than rim zone Fo | chosen for the schematic; the sense is **not fixed** by FORSTER (R README example `segment_fo` "0.91, 0.89" is the reverse sense). Caption says either sense works |
| Rim zone width | drawn equal on all sides; equal along a and c in panel b only to isolate D | real zone widths differ by direction (N joint fits: "olivine grows faster on some faces") |
| Traverse 1 | line of spot analyses from the rim inward, parallel to a [100] | N "Inputs": profile from microprobe spot analyses, distance from crystal edge |
| Traverse 2 | line of spot analyses from the rim inward, parallel to c [001] | as above |
| Distance origin | 0 at the crystal rim (edge 1), increasing inward | R README `distance_um`: "Distance of the analysis from the crystal edge"; N "Distances are measured from the profile's own edge" |
| Edge condition | rim Fo held fixed (in contact with melt) | N "Edges"; R `domain_start_um`: "edge 1, where Fo is held" |
| Far end of a half profile | core value held | N "Edges" (default) |
| Spot count and spacing | **unknown** (drawn as about 8 evenly spaced dots, generic) | not given |
| Axis triad | a [100], b [010], c [001]; a to the right, c up, b into the page (circled cross), so the frame is right-handed (a × b = c) | orientation in the section is the schematic's choice |
| General profile direction (panel c) | a unit vector at angles α, β, γ to a, b, c (direction cosines 0.30, 0.55, 0.78; chosen to keep the angles legible, not from data); oblique view: a right, c up, b receding up-right (into the page, as in panel a); dashed lines are its projection onto the a-c plane, a depth cue only | R README `alpha_deg`, `beta_deg`, `gamma_deg`: angles between the profile and [100], [010], [001] |

## Diffusion facts

| Fact | Value | Source |
|---|---|---|
| D along c [001] relative to a [100] and b [010] | about 6× faster (exactly 6 in the Fe-Mg laws' anisotropy term) | N; R `diffusivity.py:45` "[001] is 6x faster than [100] and [010] for Fe-Mg"; R README: "[100] and [010] six times slower than [001]" |
| D along the profile | D[100]·cos²α + D[010]·cos²β + D[001]·cos²γ | N; R README: "along the profile, D is the sum of D_i cos²(angle to axis i)" |
| What a profile fixes | D × t (time scales as 1/D) | N; R README: "An Fo profile depends on D t" |
| D depends on | T, P, fO2, Fo and direction | N |
| Fo | molar Mg/(Mg + Fe) | N |

## Computed curves (panel b), illustrative

Script: `figure/data/profiles.py` → `figure/data/profiles.json`. Model: a single step at
distance x0 from the rim, rim held at the rim-zone Fo (Dirichlet), far side semi-infinite:
Fo(x, t) = Fo_rim + ΔFo/2·[erfc((x0 − x)/(2√(Dt))) − erfc((x0 + x)/(2√(Dt)))] (method of
images). Units are arbitrary: x0 = 1, ΔFo = 1, D along a = 1, D along c = 6, three times
with D_a·t = 0.003, 0.015, 0.05 over distances 0 to 2.6 (chosen only so the curves are distinct; not real
values). The same three times are used in both directions. No numbers appear on the figure.

## On-screen strings (each with its source)

| String on figure | Source |
|---|---|
| "a" "b" "c" (panel letters) | layout |
| "Zoned olivine in melt" | N (zoned olivine, edge 1 in contact with melt) |
| "Step relaxes with time" | N (step-shaped starting profile run forward in time) |
| "Direction sets D" | N, R (D along the profile set by its direction) |
| "melt" | N "Edges" |
| "core" | N segment naming example "rim, mantle1, core" |
| "rim zone" | N, R segment naming example "rim, mantle, core"; the outer, lower-Fo segment here |
| "microprobe spots" | N (electron microprobe spot analyses) |
| "a [100]" "b [010]" "c [001]" | N, R |
| "a" "b" "c" (axis tips in panel c) | N, R |
| "along a [100]" "along c [001]" | N, R (profile orientation) |
| "along a [100]: slow" "along c [001]: ~6× faster" | N; R diffusivity.py:45 "[001] is 6x faster than [100] and [010] for Fe-Mg" |
| "Fo" | N |
| "distance from rim →" | R README distance_um "Distance of the analysis from the crystal edge" |
| "t0" | N (t0 step → later times); the starting profile |
| "time" | arrow in the direction later profiles widen (computed curves) |
| "width ~ √(D t)" | diffusion length of an erf profile (computed: the bracket spans 16-84 % of the t3 curve, 2.8 √(Dt)); N "a profile fixes D × t" |
| "√6 × wider" | follows from D_c = 6 D_a at equal t (widths scale as √D); asserted in profiles.py |
| "profile" | N, R (profile direction from EBSD) |
| "α" "β" "γ" | R README alpha_deg, beta_deg, gamma_deg |
| "Dprofile =" and "Da cos²α + Db cos²β + Dc cos²γ" | N; R README "along the profile, D is the sum of D_i cos²(angle to axis i)" |
| "time ~ width² / Dprofile" | N "A profile fixes D × t, so the time scales as 1/D"; with width ~ √(D t) |
| "illustrative" | brief (curves illustrative) |

## Unknown (kept generic on the figure)
- Crystal habit, face indices, and the real section orientation.
- Real Fo values, step position, spot spacing, D values, times (none plotted).
- Whether the rim zone is equally wide on every face (drawn equal; caption says it is not in general).
