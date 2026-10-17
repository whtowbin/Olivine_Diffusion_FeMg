# How the fit works

This page explains, in plain terms, how FORSTER finds the best-fit time and its 1σ
and 2σ ranges, what those ranges assume, and what each of its checks and warnings
looks for. The full method, with its derivation, sources and validation, is in
[likelihood-method.md](https://github.com/whtowbin/Olivine_Diffusion_FeMg/blob/master/docs/references/likelihood-method.md)
on GitHub.

## 1. The model

FORSTER solves the diffusion equation for Fe-Mg exchange along the profile, starting
from your step-shaped starting profile, with a diffusivity D that depends on Fo,
temperature, pressure, oxygen fugacity and the profile's direction in the crystal
([Setting up a run](setup.md)). Because D depends on Fo, it changes along the profile
and the smoothed steps are not symmetric. Edge 1 is held at its Fo; the far end at the
core's Fo (or mirrored at the crystal's centre with `centre_um`).

The model is run **once**, forward in time, and its profile at every time step is
compared with the measured points. Time is therefore never guessed: every candidate
time is tried. The model stops once the fit is well behind it (twice the best time so
far, with the misfit four times its minimum and still rising), which gives the same
answer as running to `max_time_days`, several times faster.

## 2. Comparing the model with the data

At each time, the **residuals** are the measured Fo minus the model Fo at each
measured point. The misfit is measured by **chi-squared** (χ²), the sum of the squared
residuals, each divided by the expected size of the error on that point. The best fit
is the time with the smallest χ².

The question is what "the expected size of the error" should be. Plain χ² assumes each
point's error is independent of its neighbours' and of known size. Real profiles break
both assumptions:

- **The model never fits perfectly.** A starting profile that is not quite the right
  shape, a beam that averages over a few microns, a profile not quite perpendicular to
  the zoning: all leave a smooth misfit, so neighbouring residuals share a sign.
  Counting such points as independent overstates the evidence: ten analyses on the
  same bump of misfit carry one piece of information about it, not ten.
- **The analytical error is rarely known well**, and is often underestimated.

## 3. The noise model

FORSTER's **noise model** is its description of the residuals. It has two parts, both
fitted to your residuals:

- **White noise** with standard deviation *s*: scatter that is independent from point
  to point, like analytical precision. Reported as `white_sigma_fo`.
- **A smooth correlated misfit** with amplitude *a* (`correlated_sigma_fo`) and length
  *L* (`correlation_length_um`): points closer together than about L share the same
  misfit. *a* = 0 means white noise alone was enough.

The three numbers are found by **maximum likelihood**: the values under which the
observed residuals are most probable, with a penalty that stops the noise from being
made arbitrarily large to absorb a bad fit. They are fitted at a first best fit made
with white noise only, then χ² is recomputed for every time with the fitted noise
model.

**Whitening** is what the noise model does to the residuals: it transforms them so
that, if the noise model is right, they become independent and all of the same size
(like draws from a standard normal distribution). χ² is computed from the whitened
residuals. In effect, points on the same stretch of smooth misfit are counted together
as one piece of information, and the smooth part of the profile carries roughly
(profile length ÷ L) independent pieces of information rather than one per point.
Whitened residuals that still show runs of the same sign, or values well beyond ±2,
mean the noise model has missed something; FORSTER's checks (section 5) look for the
common cases.

The figures below use synthetic profiles of known age (30 points 2.5 µm apart, a
4 mol% step, a true diffusion length of 10 µm, 0.1 mol% white noise), each with one
kind of misfit added. Rows: the data and the fit with its 1σ and 2σ bands; the raw
residuals (green: the smooth misfit the noise model sees; yellow: within two diffusion
lengths of the step); the whitened residuals; and the fitted time divided by the true
time.

![Misfit of the same size everywhere](images/noise-gallery-stationary.png)

*Misfit of the same size along the whole profile. White noise only: the residuals
look the same before and after whitening, and the 1σ range covers the true time 69%
of the time (target 68%). A smooth misfit shorter than the diffusion length: the raw
residuals run in same-sign stretches, the whitened ones do not, and coverage stays
right (67%). A smooth misfit about as long as the diffusion length: it looks like a
slightly older or younger profile, the model cannot tell the two apart, and coverage
falls to 57%.*

![Misfit near the step, outliers and noise growing toward the rim](images/noise-gallery-other.png)

*Misfit concentrated near the step (left) lowers 1σ coverage to 61%; one or two
outliers (middle) widen the range but leave coverage right; scatter that grows toward
the rim (right) lowers it to 63%.*

## 4. The ranges

The **1σ range** is the range of times where χ² stays within a threshold of its
minimum; it should contain the true time 68% of the time. With many points the
threshold is 1 (the textbook rule). With few points it is a little larger, because
the noise was estimated from the same few residuals and could have come out too small:

| Points | Threshold | Range wider than with 1 by |
|---|---|---|
| 6 | 1.48 | ×1.22 |
| 10 | 1.25 | ×1.12 |
| 15 | 1.15 | ×1.07 |
| 29 | 1.07 | ×1.04 |
| 100 | 1.02 | ×1.01 |

The **2σ range** (`low2_days`-`high2_days`) is the same with the threshold for 95.45%;
it should contain the true time 95% of the time. **Coverage** is how often a range
actually contains the true time, measured on thousands of synthetic profiles of known
age fitted exactly as FORSTER fits real data:

| Kind of misfit | 1σ covers | 2σ covers |
|---|---|---|
| White noise only | 69% | 96% |
| Smooth, shorter than the diffusion length | 67% | 92% |
| Smooth, about as long as the diffusion length | 57% | 87% |
| Concentrated near the step | 61% (45-61% in other tests) | 89% |
| One or two outliers | 70% | 97% |
| Scatter 4× larger at the rim | 63% | 95% |

So the 1σ range is right in the common cases, and too narrow in three that the
diagnostics can flag: then quote the 2σ range ([Reading the
results](results.md#warnings)). A range that assumed independent errors (as the
`legacy` method and most earlier software do) covered only 18-40% when real smooth
misfit was present.

!!! note "Best fit and median"
    The distribution of time is usually skewed toward longer times, so its median
    (shown in the time distribution figure) sits slightly later than the best fit
    (5.31 against 5.14 days for the worked example). Both are correct; quote the best
    fit with its range, or the median with its central range, consistently.

### What the range does not include

The fit's range comes from the scatter and misfit of the data around the model. It
assumes the inputs are right:

- **The starting profile.** A step 3 µm out of place biased the time by 18%, and the
  single-fit range covered the true time only about 22% of the time. Fit the starting
  profile to include it.
- **Temperature, pressure, oxygen fugacity and the diffusion law.** These scale D,
  and so the time, for every point at once; the fit cannot see them.

[Uncertainty beyond the fit](uncertainty.md) adds both.

## 5. Checks and warnings

After each fit FORSTER runs a few checks. Each one is reported as a number in
`summary.csv` and, when it fires, as a warning (in the `warnings` column, printed by
`forster fit`, and shown under the plot in the app).

**Misfit concentrated near the step** (`near_step_daic`). The noise model assumes one
misfit size along the whole profile. In real profiles the misfit is often larger near
the steps, where the information about time is, and the default range is then too
narrow. FORSTER compares the default noise model with one whose misfit may be larger
within about one diffusion length of each step, using the **AIC** (Akaike information
criterion, a score that charges for the extra parameter; lower is better). The change,
ΔAIC, is reported as `near_step_daic`: about +2 means no sign of near-step misfit;
below -6 FORSTER warns. In tests it fired in at most 0.2% of fits whose noise model was
right, and in 76-94% of fits with strong near-step misfit on profiles of about 30
points or more. It needs points near the step: `near_step_points` counts those within
two diffusion lengths, and with fewer than about 13 a silent check is weak evidence.
Near-step misfit usually means the starting profile near the step is slightly off;
fitting the starting profile cleared the warning on every profile tested.

![Three published profiles where the near-step check fires](images/noise-near-step-profiles.png)

*The three published profiles (Towbin and Plank 2026) where the near-step check
fires, fitted with the default noise model (blue) and with the near-step model
(orange). The residuals are largest in the yellow near-step zones. The near-step best
fit moves by -12% to +58%, in two of three cases outside the default 1σ range but in
all three inside the default 2σ range.*

**The profile ends too close to the step** (`core_end_diffusion_lengths`). The model
holds the core's Fo at the last measured point. If the data stop before the profile
has flattened, that held value steepens the modelled step and moves the time. FORSTER
reports the distance from the last step to the end of the model in diffusion lengths
and warns below 3 (real plateaus start about 2 diffusion lengths past the step).
Measure further into the core, or model the whole crystal with `centre_um` if the
profile reaches its centre.

**An unresolved diffusion front.** When the diffusion length at the best fit is shorter
than the spacing between points, only one or two points sit on the smoothed step, and
the near-step check sees almost none. FORSTER warns: check that the fit has not
collapsed onto the step, and treat the range with care.

**The misfit spans the profile.** When the fitted misfit length L reaches half the
profile and the smooth misfit is at least half the white noise, the residuals drift
across the whole profile: usually a plateau or core level that sits off the model's, or
a step in the wrong place, not noise. Check the starting profile and its held values.

**The range reaches the end of the model** (or time zero). Increase `max_time_days`; for
the starting-profile fit, `scan_max_time_days`.

**After fitting the starting profile** ([Uncertainty beyond the
fit](uncertainty.md#fitting-the-starting-profile)):

- **Still concentrated near the step** (`scan_near_step_daic` below -6): even the
  best-fit starting profile leaves misfit at the step. The starting profile may need a
  shape FORSTER does not have (a graded rim, growth while diffusing). Quote the 2σ
  range.
- **At the edge of the prior**: a fitted value reached the limit you allowed (2.5
  standard deviations for an `_sd` uncertainty, the end of a `_range`). The data want
  something outside it; rethink the starting profile rather than simply widening the
  limit.
- **Prior conflict**: a fitted value lies 2 or more standard deviations from your
  nominal value (`scan_prior_offsets`). The prior and the data disagree, and the prior
  is pulling the fit; check where the prior came from.

What to do about each warning, in a table: [Reading the results](results.md#warnings).
