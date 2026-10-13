# Instantaneous growth stages

Status (2026-10-13): first version built (Fo, fixed stage times, Python): `forster/boundary.py`, `fit_profile(..., stages=...)`, `FitResult.time_since_last_stage_days`, `tests/test_growth.py`. Henry's choices: report both the time since the first stage and since the last; Fo first, stage times fixed.

## Goal

Model a crystal that grows in instantaneous steps: each later zone appears at a
given time and then diffuses with everything grown before it, instead of all zones
existing from time zero as the step-function initial condition assumes now.
Gradual growth at a rate (AUTODIFF; Couperthwaite et al. 2021, Kahl et al. 2023) is
out of scope until those papers are read.

## Model

Coordinates are rim-relative, as measured: x = 0 is the crystal edge at the end.
Stage 0 is the `InitialCondition` as now (the crystal between ``domain_um[0]`` and
``domain_um[1]``, Fo held at ``edge_fo`` at the rim and ``core_fo`` at the core
end). Each later `Stage` starts at ``start_s`` and:

- moves the rim outward to ``rim_um`` (≤ the previous rim), filling the new zone
  between the two rims with ``zone_value``; with ``rim_um`` equal to the previous
  rim it only changes the rim value (a melt that changes composition);
- holds ``rim_value`` (default: ``zone_value``) at the new rim from then on.

The model grid covers the final crystal [last rim, core end]. Nodes outside the
crystal at a given time are inactive (NaN in the stored curves). The explicit
steppers run on the active slice with the stage's rim value as the boundary, so
with no stages every number is identical to now.

Shared with session github-drafts-68 (its implicit scheme will extend it):
`forster/boundary.py` with `Stage(start_s, rim_um, rim_value=None, zone_value=None)`
and `BoundaryHistory(stages)` (active node range and rim value at a time).

## Fitting

The measured profile includes every zone, so only times after the last stage can
fit it. The fit runs all stages up to the last start, then the usual early-stopping
and extending run, and considers only rows at or after the last stage.

## Choices for Henry

1. What the fitted time means: (a) the time since the first stage (stage start
   times are fixed inputs, so this is the last stage start plus the time after it);
   or (b) the time since the last growth stage, with the earlier stage durations as
   inputs. The fit is the same; only what is reported and what the stage times are
   measured from differ. Proposed: report both, with stage start times measured from
   the first stage.
2. Scope of the first version: Fo only (proposed), with Ni/Mn/Ca next. For elements
   the new zone and rim values are more concentrations, so with fitted
   concentrations each stage adds linear terms (unit runs), not a scan.
3. Stage start times: fixed inputs in the first version. Scanning them (or the
   zone widths) with `forster.scan` comes after.

## Tasks (after the choices)

1. `src/forster/boundary.py`: `Stage`, `BoundaryHistory` (validation: start times
   increasing, rims not moving inward, values in (0, 1] for Fo).
2. `model.py`: `fit_profile(..., stages=())`; `_diffusion_model` and the run
   helpers apply the stages; legacy uncertainty with stages is an error.
3. `tests/test_growth.py`: no stages and a stage that changes nothing give results
   identical to now; a stage matches an independent flux-form solver run with the
   same staged history; a fit recovers the time of a staged synthetic profile; the
   bias of fitting a staged profile with the single-stage model is measured and
   reported (not asserted).
4. Python API first; table columns and CLI with the element tables (stage 2).

## Results (2026-10-13)

Test case: a crystal [40, 200] um with its rim at Fo 0.86 for 6 days, then an
overgrowth [0, 40) um of Fo 0.82 (rim 0.82), observed at 10 days; round robin
conditions, along [001], dx 1 um.

- The staged model matches an independent flux-form solver run with the same history
  to 3.4e-4 Fo (test bound 5e-4; the stage start is rounded to the nearest time step).
  Without the early stage (a two-zone initial condition at time zero) the profile is
  1.3e-2 off.
- A stage that changes nothing leaves every row identical; with no stages the code
  path is the old one.
- Fitting the staged profile (66 points, noise 0.001) with the stages given recovers
  10.1 days total and 4.1 days since the overgrowth (true 10 and 4); early stopping
  gives the same fit.
- Fitting the same profile with the single-stage model gives 6.8 days (6.2-7.6),
  a range that contains neither the total nor the time since growth, with the same
  goodness of fit (chi-squared 65.2 against 66.0). The fit statistic cannot tell a
  staged history from a single one, so the history has to come from petrography,
  other elements or zoning that only growth explains.

## Next

- Elements with stages: each stage's new-zone and rim values are more fitted
  concentrations (unit runs), as `FittedInitial` already does.
- Scanning stage start times with `forster.scan`.
- Table columns and CLI with the element tables.
- Gradual growth after reading the AUTODIFF and DFENS multi-step papers.
