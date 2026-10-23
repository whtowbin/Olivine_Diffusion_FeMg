# Fitting the rim position on single profiles (design, 2026-10-23)

Status: **closed after step 0 (2026-10-23): one profile cannot pin the rim offset, so it
is not fitted on single profiles (Henry's decision 1). Step 1 is not built.** Results:
[2026-10-23-single-profile-edge-offset/README.md](2026-10-23-single-profile-edge-offset/README.md)
(908 fits). With free steps the posterior sd of the offset stays at 0.83 of its 2 µm prior
(synthetic, 0 of 100 fits at or below 0.5) and 0.70 on the 18 published profiles (4 of
18, the four sharpest, with poor importance sampling), and the time does not move beyond
fit noise. With free steps the data fix the step's reported position (offset plus step
distance), which puts a floor near 1/√2 = 0.71 on the ratio; only information from the rim
itself could beat it, and the model holds edge 1 at the first segment's Fo. Fixing edge
1's Fo helps a little (0.66). Steps held relative to the rim pin it on the paper profiles
(0.37), but there the offset just plays the part of a free step. The joint fit keeps its
edge offset (off by default), where several profiles share a time.

Where a rim offset can be learned (Henry, 2026-10-23): when something is shared, either
several profiles of one grain fitted together (the joint fit, which has it), or single
profiles across a population of grains whose offsets share one prior (a hierarchical fit,
with the population priors in TODO_local.md).

## The aim

Measured distances give a good first estimate of each point's distance from the crystal
rim, but the rim may need to move a few µm for the model to fit (Henry, 2026-10-23).
Today only joint fits (`forster.fit_crystal`, Python) can fit that shift, as an edge
offset per profile with a Gaussian prior, off by default. The single-profile
starting-profile fit, the one behind the parameter table, `forster fit` and the app,
fits the step positions and the segments' Fo but not the rim position.

## What already exists, and what was measured

- **Code.** `joint._Part.matrix` shifts the measured points by the offset and rebuilds
  the interpolation matrix from the model grid to the points (clipped to the grid). The
  model itself is unchanged. `sampling.ICLikelihood` compares the model with the points
  through the same kind of matrix (`self.W`), so the single-profile version is the same
  idea: one more parameter, `edge_offset_um`, that changes the matrix instead of the
  starting profile.
- **Measurements (joint-fit calibration, `docs/plans/2026-10-16-joint-calibration/`).**
  With free steps, the edge offset is barely identifiable: a step and the edge moving
  together hardly change the fit. It hardly moves the time either (Xenocryst: Laplace
  within 1.1% with the offsets on). But it makes the posterior less bell-shaped: the
  importance-sampling efficiency dropped to 1-28% (median 5%), the ESS rule then falls
  back to nested sampling, and nested sampling with the offsets on took 25 minutes to
  2 hours per crystal (GCB1A). That is why the joint fit has it off by default and uses
  a sample-count step rule instead of the ESS rule.

So the code is the small part. The questions are whether one profile carries enough
information to fit the offset, what it does to the time and its range, and how to keep
the fit fast.

## Where the information about the rim comes from

The model holds edge 1's Fo at the rim (x = 0). Shifting the points by δ moves the data
against that boundary and against the steps. With the steps free, the steps follow the
data, so only the points close to the rim see the shift: the offset is identifiable
roughly when the data reach within a diffusion length or so of the rim, and when edge
1's Fo is pinned (given, or tightly priored). With the steps pinned (for example by a
P2O5 traverse) the offset moves the rim relative to the step, which changes the
overgrowth width and so the time.

## Step 0: measure on single profiles first (proposed now)

A spike, no production code. Use `fit_crystal` with one profile, which already supports
the offset, on:

1. synthetic profiles from the forward model with known rim offsets (0, ±2, ±4 µm),
   realistic noise (the noise models of `docs/plans/2026-10-15-whitening-validity/`),
   the first analysis 1-5 µm inside the rim, with the steps free and with the steps
   fixed;
2. the published profiles of Towbin and Plank (2026) (public), offset prior sd 2 µm.

Record per case: the offset's posterior sd against its prior sd (fitted, or only
carried as uncertainty?), bias against the true offset, the change in best time and in
the 1σ range, the ESS fraction, whether the ESS rule would fall back, and wall time
with the joint fit's step rule. Within the memory budget: one fit at a time, method
`laplace` with the step rule (no nested fallback), so the spike runs in minutes.

**Decision rule.**
- Posterior sd at most about half the prior sd on typical profiles, and the time moves
  by more than its fit noise: worth fitting. Build step 1.
- Posterior sd near the prior sd: the data do not fit the offset. It then only widens
  the range (honestly) and costs time. Offer it as "rim position uncertainty" (same
  code, documented as such), or not at all.
- Identifiable only with pinned steps: offer it, and say so in the docs and as a note
  when it is used with free steps.

## Step 1: build it (if step 0 says so)

- `sampling.ICLikelihood`: an optional `edge_offset_um` parameter. `candidate` ignores
  it; `at_points` uses the shifted matrix. Share one helper with `joint._Part.matrix`
  (`uncertainty.shifted_matrix(model_x, x, offset)`) so the two cannot drift apart.
- Sampler: give the single-profile fit the joint fit's step rule (1024 samples, 4096 if
  the ESS is low, nested only below an ESS of 20) when the offset is on, so an offset
  that barely moves the time does not trigger a long nested run. Or adopt the step rule
  for all single-profile fits if step 0's numbers support it (a separate decision; it
  would change existing results, so it needs a replay).
- Names and table columns, following `steps_sd_um` / `steps_range_um`:
  `edge_offset_sd_um` (Gaussian) or `edge_offset_range_um` (uniform), default off.
  Summary: the fitted offset and its sd (`scan_edge_offset_um`, `scan_edge_offset_sd_um`)
  next to the other starting-profile results, and its share of the time's variance.
  The existing prior-edge warning covers an offset at the edge of its prior.
- `method="grid"`: refuse the offset with a clear message (the grid scan runs a fixed
  candidate set; an offset would need a matrix per candidate). The default `auto`
  supports it.
- App: a ± field for the rim position on the starting-profile panel, the fitted offset
  shown with the other fitted values, and the points drawn shifted by it. "Use the best
  starting profile" can also apply the offset to the profile's distances (see question
  2).
- Docs: setup.md (the rim position), uncertainty.md (what the offset can and cannot
  learn), results.md (the new columns); the reminder to measure distances from the rim
  stays.
- Tests: recovery of a known offset on a synthetic profile with pinned steps; the prior
  returned when the data carry no information; joint fits unchanged by the shared
  helper; runs without the new columns unchanged (differential replay with a control).

## Henry's answers (2026-10-23)

1. If one profile cannot pin the offset, do not fit it (no "rim position uncertainty"
   mode).
2. Report a fitted offset, but do not apply it to the profile's distances.
3. Default prior when turned on: 2 µm Gaussian sd (beam size and placement).

Step 0 approved and started.

## Questions for Henry (they change the work)

1. If step 0 shows the offset is not identifiable with free steps, do you still want it
   as a "rim position uncertainty" that widens the range, or only when steps are fixed?
2. Should a fitted offset stay a nuisance parameter (reported, data unchanged), or should
   the app offer to apply it to the profile's distances, as it does for the best
   starting profile?
3. A default prior when turned on: 2 µm (as the joint fit's suggested value), or tied to
   the analysis spacing or beam size?
