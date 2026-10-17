# Facts and geometry record

Sources: [N] = FORSTER_FACTS.md (scratchpad research note); [J] = docs/plans/2026-10-16-joint-fits.md;
[G] = docs/plans/2026-10-16-geometry-naming.md (both in the FORSTER repo, read only);
[C] = my illustrative choice (no source; drawn as a schematic, stated as illustrative in the caption);
[S] = computed by figure/data/make_crystal.py and make_profiles.py from [N] physics plus [C] choices.

## Geometry record (the "device": a zoned olivine section with two traverses)

| Item | Record | Source |
|---|---|---|
| Crystal habit, outline, face indices | **unknown.** Drawn as a generic convex, bevelled, elongate outline with no face labels. The elongation is not a claim about habit. | [N] unknown list; [C] |
| Section plane | **unknown in general.** Drawn as if it contains a [100] and c [001], with b [010] normal to it, so that the angle formula has two terms. Illustrative. | [C] |
| Zoning | three segments from the rim inward: rim zone, mantle, core, each one Fo; normal zoning (Fo highest in the core). Segments are labelled and matched across profiles by label. | [N], [G] decision 3; normal zoning [C] |
| Edge (edge 1) | the crystal rim, in contact with melt, held at the edge Fo; the edge Fo may differ from the rim-zone Fo (drawn lower). | [N], [G] decision 3 |
| Zone widths | differ by direction (growth is anisotropic; a section may miss the centre): zones are wider along c than along a in the drawing. | [J] "What Henry has decided"; which direction is wider [C] |
| Traverse 1 ("Profile 1") | a straight line of microprobe spots from the rim inward, 15 degrees from c [001] (the fast direction). | [N] (c about 6x faster); 15 degrees [C] |
| Traverse 2 ("Profile 2") | a straight line of spots from another rim inward, 15 degrees from a [100] (a slow direction); orthogonal to traverse 1. | "roughly orthogonal" [J]/[N]; 15 degrees [C] |
| Spot spacing | the same on both traverses. | [C] |
| Crossing point | where the two traverses cross, inside the core, a little off the crystal centre. Its distance along each profile is a planned input; the model should agree there. | [N], [G] "Intersection"; position [C] |
| Distances | each profile measured from its own edge; no shared origin. | [N], [G] decision 4 |
| Diffusion in the map | 2D anisotropic diffusion at one time with D along c = 6 x D along a, Fo held at the edge value at the outline. The map is illustrative of the same D x t as the profiles. | [N] physics; [S] |
| What sits in front | spots on top of the Fo map; the outline on top of the map; labels outside the crystal with leaders. | [C] |

## Physics and fit facts used

| Fact | Source |
|---|---|
| Fo = Mg/(Mg+Fe), molar | [N] |
| D along c [001] about 6 times D along a [100] or b [010] | [N] |
| D_profile = D[100] cos²α + D[010] cos²β + D[001] cos²γ | [N] |
| A profile fixes D x t; the profile width scales as sqrt(D t) | [N] |
| Shared on one crystal: time; T, P, fO2 (identical, a mismatch is an error); each segment's Fo, matched by label; the edge value by default | [J] decided + answers 1, 2 |
| Per profile: step positions and zone widths; origin; orientation and so D; an Fo drift offset on all its segments (prior ~0.1 mol%, "to confirm"); an edge position offset (prior a few µm) | [J] answer 1, [N] |
| The edge value can be freed per profile (own segments) | [J] answer 1 |
| Joint misfit = sum over profiles, each with its own noise model | [J] Part 1 "Likelihood", [N] |
| Quick individual fits first; their times are compared as a check | [J], [N] |
| On a slow axis with few points on the slope the data may not reach the edge value; the faster axis pins the shared edge value | [J] answer 1(b), [N] |
| The joint fit is planned, being built now (nothing built yet as of 2026-10-16) | [J] status line, [N] |

## Illustrative numbers (in the scripts only; none appear on the figure as values)

All [C] unless marked [S]: segment Fo relative to the edge: rim zone 0.9, mantle 2.0, core 3.0 mol%;
point scatter 0.10 mol%; spot spacing 2 length units; profile-1 width sqrt(4 D1 t) = 4 units;
drift offset drawn on profile 2 = 0.30 mol% (enlarged three times from the plan's ~0.1 mol%);
edge offsets drawn 1.5 and 1.0 units ("a few µm" in the plan; units here are arbitrary);
D1/D[100] = 5.67, D2/D[100] = 1.33, so D1/D2 = 4.24 and the width ratio is 2.06 [S].
Step positions and crossing distances come from the drawn crystal geometry [S] (make_crystal.py).

## Panel c computation [S]

Synthetic data from the profiles above (seeded noise), fitted three ways: profile 1 alone, profile 2
alone, both jointly. Fitted: time, edge Fo, and the three segment Fo (shared in the joint fit). Held at
their true values: step positions, drift offset, edge offset. Misfit = n ln chi² per profile (noise
scale integrated out), joint = sum; 1σ range where the misfit is 1 above its minimum. Results for the
drawn seed and the 12-seed sweep are in figure/data/profiles.json and are quoted in the caption.

## On-screen strings (each sourced above)

| String | Source |
|---|---|
| "Zoned olivine section" | [N] (zoned olivine, Fo profiles across it) |
| "Profile 1", "Profile 2" | [N] profiles; numbering [C] |
| "crossing point" | [N] "mark where two profiles cross", [G] Intersection |
| "rim zone", "mantle", "core" | segment labels, [J] answer 4 ("rim, mantle, core"), [G] decision 3 |
| "c [001], fast", "a [100], slow" | [N] (c about 6x faster than a or b) |
| "Fo", "low", "high" | [N] Fo definition |
| "Fo profiles from one crystal" | [N], [J] |
| "edge Fo" | [N] edge value; [J] answer 1 |
| "distance from rim" | [N] "distances are measured from the profile's own edge" |
| "Profile 1: near c, high D", "Profile 2: near a, low D" | [N] D by direction, D_profile formula |
| "steps" | [N] step positions, per profile [J] |
| "edge offset" | [J] answer 1(c), [N] |
| "model at time t" | [N] model run forward in time |
| "starting profile" | [N] |
| "drift offset" | [J] answer 1(a), [N] |
| "one time t; same T, P, fO2" | [J] decided + answer 2 |
| "shared by the crystal", "own to each profile" | [J] (shared vs per profile) |
| "Joint fit (planned)" | [J] status line |
| "misfit", "joint (sum)" | [J] Part 1 likelihood (sum over profiles) |
| "profile 1", "profile 2" | [C] row labels |
| "1σ", "1σ ranges", "time", "true value" | [N] fit step 3 (1σ from the misfit curve); "true value" = the synthetic truth [S] |
