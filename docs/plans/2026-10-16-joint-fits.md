# Joint fits of one crystal's profiles, and priors from the crystal population

Status (2026-10-16): **plan for Henry to check.** Nothing built. The names are settled in
[2026-10-16-geometry-naming.md](2026-10-16-geometry-naming.md); the evidence for
population priors is in [2026-10-16-population-priors/](2026-10-16-population-priors/README.md).

## What Henry has decided

- Profiles along different axes of one crystal share one history: **time, conditions, and
  the compositions** (segment Fo, matched by segment label; edge values). They do **not**
  share step positions or zone widths (growth is anisotropic and a section may miss the
  centre), nor an origin (they do not lie on one line). Each keeps its own orientation, so
  its own D.
- Workflow: quick individual fits first, then a joint fit when the user marks profiles as
  one crystal (`crystal` column: a GUI check box or a typed group name, which wins over a
  file-name rule that only suggests groups).
- A profile may fit its own Fo for a segment on request (`own_segments`), e.g. its core.
- Crystals of a sample form a population (`sample` column); histograms of their segment
  Fo give priors.

## Part 1: joint fit of one crystal

**Likelihood.** Each profile i keeps its own noise model and model run. For a shared
time t the joint log likelihood is the sum over profiles of -n_i/2 log chi2_i(t) (the
noise scale of each integrated out, as today). All profiles of a crystal run on one time
step, so the time axis is shared; the best fit and the range come from the summed curve
with the small-sample threshold for the total number of points, as in the minor-element
joint fit (`forster.elements.fit_joint`).

**Parameters of the starting-profile fit.** The union over the crystal's profiles:
- per profile: its step positions (`<run>:step1_um`, ...) and, with `own_segments`, its own
  segment Fo;
- shared: each segment label's Fo (`core_fo`, `rim_fo`, ...), with one prior per label. A
  label's prior comes from the rows that use it; rows that disagree on a label's nominal
  or prior are an error naming them.
The optimiser, Laplace importance sampling and nested fallback work as now, on the larger
parameter vector (one model run per profile per evaluation). The noise models are
refitted per profile until stable, as in the fixed single-profile fit.

**Checks before the joint fit.** The individual fits' times are compared: if they
disagree beyond their ranges, the app and summary say so (wrong orientation, a profile
off the centre, a 3D effect near a corner) before the joint fit is trusted.

**Outputs.** summary.csv: one row per crystal (`crystal`, the joint time and ranges, each
shared segment's fitted Fo, each profile's steps and its individual time), plus the
per-profile rows as now. The app: a crystal view with each profile and its fit.

**Tests.** A crystal of two synthetic profiles made at one time with shared compositions
and different steps and D: the joint fit recovers the time with a range no wider than
the narrower individual one; a joint fit of one profile equals its single fit exactly;
inconsistent priors are an error.

**Cost.** Each evaluation runs every profile (milliseconds each); a 3-profile crystal with
2 steps each and 3 shared segments has 9 parameters: the Laplace path stays in seconds,
the nested fallback tens of seconds.

## Part 2: priors from the crystal population

From the study: crystals of one sample differ in core Fo by more than their errors (WHT01
0.11 mol%; WHT06 by crystal kind, 0.9-2.5 mol%), while rims agree; a population prior
helps poorly constrained rims and short or missing cores, and biases outlier crystals if
applied to well-measured cores.

**Build (two-stage, cheap):**
1. Collect each crystal's fitted segment Fo and its posterior sd, by `sample` and segment
   label. Labels must name the zone (rim, mantle, core), not the default a, b, c, which
   pair different zones in different crystals; crystal kinds within a sample (e.g. WHT06's
   two kinds) get their own label or sample.
2. Estimate the population mean and spread per (sample or kind, label) by restricted
   maximum likelihood, the spread shared across samples where a sample is too small.
3. Offer the prior per segment, never as a default: only where the crystal's own posterior
   sd exceeds the population spread. The prior for a crystal leaves that crystal out.
4. The conflict check is already built: the starting-profile fit reports each value's
   offset in prior sd and warns from 2 sd (`scan_prior_offsets`).

**Outputs.** A population table per sample (label, crystals, mean, spread, each crystal's
value and shrinkage); in the app, a histogram per label beside the segment table with the
crystal's own value, and an "apply population prior" check box per segment.

## Order

1. `crystal`, `sample` and `own_segments` columns read and written (carried as metadata; no
   behaviour yet).
2. Joint fit of one crystal from Python, with the tests above; then tables and CLI
   (`forster fit` adds the crystal rows); then the app's crystal view.
3. Population table and per-segment priors from Python; then CLI and app.

## Questions for Henry

1. **Edge values in a joint fit:** shared like the segment Fo (one rim value per crystal),
   or per profile? (Proposed: shared, with `own_segments` able to free them.)
2. **Conditions:** require identical T, P and fO2 on a crystal's rows (proposed), or take
   them from the first row?
3. **Growth stages in joint fits:** share the number of stages and their zone Fo, with
   each profile's own stage rims (consistent with steps per profile)? Proposed for later,
   after the stage-free joint fit works.
4. **Population labels:** is renaming segments to zone names (rim, mantle, core) acceptable
   as the way to match zones across crystals, or do you want a separate zone-role column?
