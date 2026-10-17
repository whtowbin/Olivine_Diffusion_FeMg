# Brief: joint fit of two profiles on one crystal

- **Mode:** design (research note FORSTER_FACTS.md + docs/plans/2026-10-16-joint-fits.md, geometry-naming.md).
- **Format:** journal, double column, 183 mm wide (FORSTER methods paper).
- **Reader:** petrologists and volcanologists who know diffusion chronometry but not FORSTER's code.
- **Style:** own choice. Clean flat Nature-style schematic: neutral greys, olive-green zone tints for the crystal (pale = Mg-rich core, darker = Fe-richer rim, as olivine zoning is usually pictured), one hue per profile (blue = profile near the fast c axis, vermillion = profile near a slow axis), black for the joint (shared) result. Plots without numeric ticks, because the curves are illustrative.

## Takeaway (one sentence)
Profiles measured on one crystal share one history, so fitting them together pins the time and the edge value better than either profile alone, while each profile keeps its own geometry.

## Key ideas (the user's criteria first, verbatim)
1. "A zoned olivine section with two roughly orthogonal microprobe traverses (one closer to the fast c direction, one closer to a slow direction), their intersection point marked."
2. "Beside it the two Fo profiles, which have different widths because D differs by direction."
3. "Show what is shared (one time; T, P, fO2; each segment's Fo matched by label; the edge value by default)"
4. "and what is per profile (step positions and zone widths, orientation and so D, an Fo drift offset of ~0.1 mol%, an edge position offset of a few µm)."
5. The joint misfit is the sum of the two profiles' misfits, so its minimum is sharper: a narrower time range than either alone; the fast profile pins the shared edge Fo that the slow profile alone barely constrains.

## Must not imply
- A specific olivine habit, face indices, or that the section plane is a known crystallographic plane.
- Real numbers for a real crystal (no numeric ticks; curves illustrative, computed).
- That profiles share step positions, zone widths, an origin or one D.
- That the joint fit is already available (it is planned, being built now).
- That the drift and edge offsets are to scale (they are drawn enlarged).
- That the two profiles always agree; the individual fits are a check.

## Scope
One figure for the joint-fit section of the methods paper. Growth stages, population priors and the Monte Carlo over T, P, fO2 are not shown (other figures / text).
