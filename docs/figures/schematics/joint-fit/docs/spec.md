# Spec: joint fit of two profiles on one crystal

## Structure
**Setup with its signal, then the comparison it enables** (a setup → signal → result pipeline, left to
right): the crystal with its two traverses (setup), the two Fo profiles they give (signal, a controlled
comparison on one Fo axis and one distance scale), and the joint misfit (result). One colour rule runs
through all three panels: **black/grey = shared by the crystal, blue / vermillion = own to profile 1 / 2.**
That rule carries "shared vs per profile", so no separate ledger is needed.

## Concepts (design mode, standard depth: two written, one kept)

**A. Section → profiles side by side on a shared Fo axis → joint misfit.** a: computed Fo map of a zoned
section, two traverses of spots, crossing marked. b: the two profiles side by side, same Fo axis and same
distance scale, so shared segment levels run as continuous grey lines across both plots while the
fronts differ in width; per-profile marks (steps, edge offset, drift) in profile colour. c: misfit vs
time for each profile alone and for their sum, plus the 1σ ranges of time and edge Fo.
- takeaway in 5 s: yes (one grey level set across two differently wide coloured curves; black curve sharpest);
- true to the note: yes (sum of misfits, shared/own split exactly as the plan);
- signal state drawn (P1): yes, the Fo map and the profiles are computed states;
- controlled comparison (P3): the two profiles on one template, only D and steps differ; joint vs alone on one axis (P22 baseline);
- explains (P4): explanatory.

**B. Marginal plots on the crystal.** The crystal in the middle, profile 1 plotted above it along its
traverse, profile 2 rotated beside it along its traverse (P24 alignment), shared parameters in a box.
- takeaway: weaker: a rotated plot cannot share an Fo axis with the other, so "shared compositions" is lost;
- the 15° tilt breaks the alignment unless the axes glyph is tilted instead;
- no room for the misfit result.

**Kept: A.**

**Takeaway (one sentence):** Profiles on one crystal share one history, so fitting them together gives
one time and one edge Fo constrained by both, while each profile keeps its own geometry.
**Size:** 183 mm × 84 mm (journal, double column)
**Hero:** the zoned olivine section (panel a), drawn flat (route=flat): a computed 2D Fo map (raster,
make_crystal.py) clipped to a vector outline, vector zone boundaries, spots and labels. No Blender:
a thin section is a flat object and its colour is data.

## Panels
| Panel | Job (one line) | Position / share |
|---|---|---|
| a | the crystal: two traverses near c and near a, crossing in the core; zones wider along c; fronts blurred more along c | left, 56 mm |
| b | the two profiles: same compositions (grey levels), different widths, steps and offsets (colour) | middle, 80 mm |
| c | joint fit: misfit of each alone and of the sum vs time; 1σ ranges of time and edge Fo | right, 43 mm |

## Components
| Panel | Component | Label (exact text) | Look | Source |
|---|---|---|---|---|
| a | Fo map of the section | none (colour bar "Fo") | raster, olive ramp, pale = high Fo | compute: make_crystal.py (2D anisotropic diffusion, D_c = 6 D_a) |
| a | outline | none | 0.6 pt dark olive line | draw: L.polygon from crystal.json (generic, habit unknown) |
| a | zone boundaries (starting steps) | "core", "mantle", "rim zone" with leaders | thin dashed grey (starting position) | draw from crystal.json |
| a | traverse 1, traverse 2 | "Profile 1", "Profile 2" | rows of spots, blue / vermillion | compute positions (make_crystal.py), draw L.circle |
| a | crossing point | "crossing point" | black ring | crystal.json |
| a | axes glyph | "c [001]", "a [100]", "fast", "slow" | two grey arrows | draw (L.arrow); orientation illustrative |
| a | Fo colour bar | "Fo", "low", "high" | strip with the ramp | draw from crystal.json ramp |
| b | two plot frames | "Profile 1", "Profile 2", axes "Fo", "distance from rim" | L.plot-like frames drawn by hand, no numeric ticks | draw |
| b | starting profile | (key) "starting profile" | thin step line, profile colour, dashed | compute (profiles.json) |
| b | model profile at time t | (key) "model at time t" | solid curve, profile colour | compute (erf solution) |
| b | data | (key) "microprobe spots" | dots, profile colour | compute (synthetic, seeded noise) |
| b | shared segment levels | "core", "mantle", "rim zone" | grey dashed lines across both plots | profiles.json levels |
| b | shared edge Fo | "edge Fo" | black dot at the rim of each plot on one dotted level | profiles.json |
| b | steps (per profile) | "steps" | profile-coloured triangles on the x axis | profiles.json |
| b | drift offset (profile 2) | "drift offset" | vermillion bracket core plateau vs grey level | profiles.json (enlarged) |
| b | edge offset (each) | "edge offset" | bracket between rim and first spot | profiles.json |
| b | shared conditions | "one time t; same T, P, fO₂" | black bracket over both plots | note |
| b | colour key | "shared by the crystal", "own to each profile" | swatches | — |
| c | misfit vs time | "profile 1 alone", "profile 2 alone", "joint (sum)" | blue, vermillion, black curves | compute (profiles.json misfit) |
| c | 1σ ranges | "time", "edge Fo" rows | bars | compute |
| c | true values | "true value" | dotted grey line | synthetic truth |

## Relations
| From | To | Kind | Drawn as |
|---|---|---|---|
| traverse 1 (a) | Profile 1 plot (b) | maps to | same blue colour + label "Profile 1" (no connector, P9 when-not: peers) |
| traverse 2 (a) | Profile 2 plot (b) | maps to | same vermillion |
| crossing point (a) | crossing marks on both plots (b) | correspondence | the same ring glyph on each plot's x axis |
| two misfits | joint misfit (c) | sum | "joint (sum)" label; black curve |

## Conventions
- distance runs left → right from each profile's own rim (edge 1) inward; no shared origin.
- pale = high Fo (Mg-rich core), dark olive = low Fo; the same ramp in a's colour bar and b's Fo axis strip.
- black/grey = shared by the crystal; blue = profile 1 (near c, fast); vermillion = profile 2 (near a, slow).
- no numeric ticks anywhere: all curves are illustrative.

## Negative constraints (must not appear)
- face indices, a named habit, scale bars with numbers, numeric Fo or time ticks;
- one origin or shared step positions across the two profiles;
- anything that implies the joint fit already exists (panel c heading says "planned");
- code names (InitialCondition, edge1_offset_um, own_segments).

## Look-up (step 4)
Searched for zoned olivine phenocrysts with microprobe traverses (BSE images with traverse lines,
normally zoned crystals with Fo falling toward the rim). Recognisable: euhedral to subhedral convex
outline with bevelled ends, concentric zoning, a line of closely spaced spots from rim to core. No
image embedded or traced.
