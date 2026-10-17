---
name: forster-fitting
description: Use when changing or explaining how FORSTER fits a time and its uncertainty: fit_profile and fit_run, the correlated noise model and whitening, the near-step and short-plateau warnings, the starting-profile fit (forster.sampling, forster.scan), the Monte Carlo (forster.rescale), diffusivity choices, or growth stages.
---

# FORSTER fitting: what a session must not rediscover

Read before changing anything here: `docs/references/likelihood-method.md` (the method),
`docs/references/noise-model-findings.md` (what was tested and the fixes),
`docs/plans/noise-model-log.md` (the running log; add an entry when this changes).

## The pipeline
1. `fit_profile` runs the model once (implicit TR-BDF2 by default), keeping the model
   at the measured points for every time step and full profiles only for a few rows.
2. The noise model (white s, plus a smooth misfit a over length L) is fitted at the
   white best fit; chi-squared is whitened with it; the 1σ and 2σ ranges come from a
   small-sample threshold. Plateau points carry no time information but count in the
   noise estimate.
3. Checks after each fit: the near-step check (`uncertainty.near_step_check`, warns
   below dAIC -6; reports the misfit sd at the step and the points within 2 diffusion
   lengths: with fewer than ~13 a silent check is weak), the short-plateau warning
   (the data end within 3 diffusion lengths of the last step), and the unresolved-front
   note (diffusion length below the point spacing).
4. The starting-profile fit (`sampling.fit_initial_condition`, default) optimises the
   uncertain steps and segment Fo with time marginalised, then Laplace importance
   sampling, nested sampling if the effective sample size is low. The grid method
   (`scan`) is kept for old results.
5. The Monte Carlo (`rescale.monte_carlo`) rescales by D for a fixed rim, refits with
   growth stages.

## Rules learned the hard way
- **Fit the noise model at a data-driven fit, never at the prior's centre.** The first
  search scores with white noise, then the noise model is refitted until stable. Fitted
  at the prior's centre, an informative prior off the data inflated the correlated term
  4-18x and spread the time over the whole run (fixed 2026-10-16).
- **Misfit near the step usually means the starting profile is slightly off.** Fitting it
  clears the warning; do not add the near-step AIC to the search objective (it rewards a
  wrong core Fo) and do not refit the noise per candidate (7x cost, same optimum).
- **Plateau length does not bias the time once the data reach the plateau;** it narrows
  the range when misfit sits near the step. Data that stop on the slope move the time
  (the far end is held there).
- **The time is very sensitive to core Fo** (20-176 days per mol% on short plateaus):
  a 0.1 mol% bias in a prior or a held value moves times by 5-20%.
- A single-draw test of a 1σ range fails one draw in three by design: test coverage over
  many draws, or a 95% range for one draw.

## Diffusivity choices
`forster.diffusivity.FE_MG_CHOICES`: `dipra` (Dohmen & Chakraborty 2007, the default),
`refit` (FORSTER's 2026 refit), `dfens_tamed` and `dfens_global` (Mutch et al. 2019). A
profile fixes D x t, so the time scales as 1/D at the profile's conditions; laws also
differ in their Fo dependence and anisotropy, so `--compare-laws` refits rather than
rescales. Sources and confirmations: `docs/references/diffusivity-literature.md`,
`docs/references/citations-to-confirm.md`.

## Growth stages
`forster.boundary.Stage`: the rim moves out at each stage's start and a new zone appears.
Only times after the last stage are fitted. Need the implicit scheme for exact stage
times; the starting-profile fit and `centre_um` do not model stages yet. Plan:
`docs/plans/2026-10-13-growth-stages.md`.

## Tolerance and joint fits
- Accuracy targets for samplers and numerical checks: about 5% on p16/50/84 is enough
  (Henry): the diffusivity laws are uncertain by ~0.1 log in D (~25% in time). Aim for
  the best fit and basic parameter uncertainty; estimate whole-job time before long runs.
- Joint fits of one crystal: `forster.fit_crystal` (`src/forster/joint.py`;
  `docs/plans/2026-10-16-joint-fits.md`). Their sampler uses a sample-count step, not an
  ESS threshold: the ESS did not predict the error on real crystals
  (`docs/plans/2026-10-16-joint-calibration/`). With many parameters, do not trust an
  ESS rule tuned on single profiles.
