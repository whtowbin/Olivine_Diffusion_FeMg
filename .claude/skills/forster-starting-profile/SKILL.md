---
name: forster-starting-profile
description: Use when creating, reading, writing or fitting FORSTER starting profiles (InitialCondition, steps_um, segment_fo, segment_names, edge1_fo, centre_um), their parameter-table columns, or the starting-profile fit's parameter names (a_fo, step1_um).
---

# FORSTER starting profiles: the decided names

Full decisions (Henry, 2026-10-16): `docs/plans/2026-10-16-geometry-naming.md`.

## The model
- A starting profile is **segments** of constant Fo between **steps**:
  `InitialCondition(domain_um, steps_um=(...), segment_fo=(...), segment_names=None, centre_um=None)`.
  One more Fo than steps; labels default to a, b, c, ... from edge 1 inward and may be
  renamed (letters and digits, not "step...").
- **Edge 1** is `domain_um[0]`, where Fo is held: the first segment's Fo, or a different
  held value (`edge1_fo` in tables, `rim_fo` in `fit_profile` / `Run`, a `RimPath` if it
  changes in time).
- **The far end** of a half profile is held at the last segment's Fo (the core). With
  `centre_um` the crystal is mirrored at its centre: the model runs to `centre_um` (beyond
  the data if needed) with no flux there, so the core is free to change (implicit scheme
  only, no growth stages yet). Points past `domain_um[1]` are not fitted.
- Profiles of one crystal never share an origin; only segment Fo is matched, by label.

## Old names (still work)
- Python: `InitialCondition(domain, edge_fo, core_fo, steps_um, step_fo)`; the properties
  `edge_fo` (first segment), `core_fo` (last), `step_fo` (middle of three, else None).
- Tables: `edge_fo`, `core_fo`, `step_fo`, `rim_fo` and their `_sd` / `_range` are read
  with a `RenamedColumnWarning` until the release; writers emit the new names.
- Uncertainty keys `edge_fo` / `core_fo` / `step_fo` are resolved to segments by
  `forster.scan.resolve_names`; `Run` stores the resolved keys.

## Starting-profile fit parameters
- Names: `step1_um`, `step2_um`, ... and `<segment>_fo` (`a_fo`, ...).
- **Order matters:** steps, then the first segment, the last, then the middle ones. That is
  the old edge, core, plateau order, which keeps the optimiser's path and the grid's
  candidate order, and so earlier results, identical. Do not "tidy" it into label order.
- Tables: `steps_sd_um` / `steps_range_um` (one value or one per step) and
  `segment_fo_sd` / `segment_fo_range` (one value for all, or one per segment, empty
  entries fixed; a segment takes one of the two).

## Not built yet
`origin` (zero at the centre), `edge2_fo` and growth stages at edge 2, edge offsets, and
the joint-fit columns (`crystal` and `sample` are carried as metadata only); see
`docs/plans/2026-10-16-joint-fits.md`.
