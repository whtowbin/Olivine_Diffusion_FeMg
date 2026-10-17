# FORSTER facts for schematic figures (shared by all four figures)

Source: the FORSTER repo at /Users/henrytowbin/Projects/Olivine_Diffusion_FeMg (README.md,
.claude/skills/forster-*/SKILL.md, docs/plans/2026-10-13-growth-stages.md,
docs/plans/2026-10-16-geometry-naming.md, docs/plans/2026-10-16-joint-fits.md). Read those
files for anything not covered here. If something is not in them, treat it as **unknown**:
keep that part of the drawing generic and list it in docs/facts.md as unknown. Do not invent
numbers.

Audience: readers of a geoscience methods paper (petrologists, volcanologists), the
FORSTER methods paper. Labels must make sense to a reader new to the project: no bare code
names (write "starting profile", not `InitialCondition`; "segment" is fine once defined).

## What FORSTER is
FORSTER (Fitting Olivine Residence & Storage Timescales from Elemental Relaxation) fits
Fe-Mg diffusion times to forsterite (Fo) profiles measured across zoned olivine crystals.
Inputs: a measured profile (Fo vs distance from the crystal edge, from electron microprobe
spot analyses), temperature T, pressure P, oxygen fugacity fO2, the orientation of the
profile in the crystal, and a step-shaped starting profile. It runs the diffusion model
forward in time and reports the time whose model profile best matches the data, with a 1σ
range.

- Fo = molar Mg / (Mg + Fe) (as a fraction or ×100). All Fe is Fe2+ in olivine.
- Fe-Mg interdiffusion coefficient D depends on T, P, fO2, Fo and direction. Diffusion
  along the c-axis [001] is about 6 times faster than along a [100] or b [010].
- The D along a profile combines the three axes by the profile's direction:
  D_profile = D[100]·cos²α + D[010]·cos²β + D[001]·cos²γ, with α, β, γ the angles between
  the profile and the a, b, c axes (from EBSD orientation data).
- Diffusivity choices ("laws"): Dohmen & Chakraborty 2007 (default), FORSTER's 2026 refit
  of the experiments, and two Mutch et al. 2019 regressions. A profile fixes D × t, so the
  time scales as 1/D.

## The model
- One-dimensional diffusion along the profile, solved by finite differences (implicit
  scheme by default).
- **Starting profile** (initial condition): step-shaped. The regions between steps are
  **segments**, labelled a, b, c, ... from edge 1 inward (renamable, e.g. rim, mantle1,
  core). Each segment has one Fo. Step positions are distances from the edge.
- **Edges:** edge 1 is the crystal rim, held at the edge Fo (in contact with the melt); the
  edge value can differ from the Fo of the segment next to it. For a half profile the far
  end is held at the core value (default). If the user gives a crystal centre, the profile
  is mirrored there (no flux across the centre) and the core is free to change.
- Distances are measured from the profile's own edge.

## The fit
1. Run the model once forward in time, storing the model at the measured points at every
   time step.
2. Compare model and data at every time: a misfit (chi-squared) curve versus time. The
   noise model has a white (point-to-point) part and a smooth correlated misfit part.
3. Best-fit time = minimum misfit; 1σ and 2σ ranges from the misfit curve.
4. Uncertain starting profile: step positions and segment Fo can carry a prior (sd or
   range). FORSTER then fits them too (optimise, then importance sampling around the best
   fit; nested sampling if needed) and reports a time range that includes them.
5. Monte Carlo over T, P, fO2 and the diffusivity law's parameters gives the final time
   distribution (median, 68% and 90% ranges).
Outputs: best-fit time in days with its range; fitted profile plot; summary table.

## Growth stages (instantaneous growth)
A crystal grows in instantaneous steps: at each later stage's start time, the rim moves
outward and a new zone (with its own Fo) appears at the rim; it then diffuses together
with everything grown before. Stage 0 is the starting profile. Only times after the last
stage can fit the measured profile (the measured profile includes every zone). FORSTER
reports both the time since the first stage and the time since the last stage. Stage start
times are fixed inputs.

## Joint fit of several profiles on one crystal (planned; being built now)
Profiles along different crystallographic directions of one crystal (usually orthogonal)
share one history.
- Shared: time; T, P, fO2 (must be identical); each segment's Fo, matched by segment label;
  the edge value by default.
- Not shared: step positions and zone widths (olivine grows faster on some faces than
  others, and a section may miss the centre), the origin (each profile is measured from
  its own edge), and orientation, so each profile has its own D.
- Per-profile corrections, each with a prior: an Fo offset on all of a profile's segments
  for drift between microprobe sessions (~0.1 mol%), and an edge position offset of a few
  µm (the first point may be a few µm off the true rim).
- The user may mark where two profiles cross (a distance along each); the model should
  agree there.
- Workflow: quick individual fits of each profile first (their times are compared as a
  check), then the joint fit; the joint misfit is the sum over profiles, each with its own
  noise model.
- On a slow axis with few points on the slope the data may not appear to reach the edge
  value; the faster axis pins the shared edge value.

## Things that are unknown here (keep generic)
- Real crystal habit/morphology: draw a generic schematic olivine outline, do not claim a
  specific habit or face indices.
- Real numbers for a specific crystal: any profile drawn is illustrative; say so in the
  caption.
