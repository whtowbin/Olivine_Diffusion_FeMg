# Reading the results

`forster fit` (and the app's Export) writes `summary.csv`, one row per run. This page
groups its columns by what they are for, says which time to quote, and goes through
FORSTER's warnings and what to do about each. Every column is listed in the
[Column reference](columns.md#summarycsv).

## The files

| File | Written when | Holds |
|---|---|---|
| `summary.csv` | always | One row per run: results and every input. |
| `diffusivities.csv` | always | The diffusion equations the runs used. |
| `<run>_fit.png` | always | Data, starting profile, best fit, 1σ and 2σ bands. |
| `<run>_curves.csv` | always | Model profiles at time zero, the best fit and the ends of both ranges. |
| `<run>_points.csv` | always | Fitted points, the best fit at each, and the residuals. |
| `<run>_scan.csv` | starting profile fitted | The posterior of time over the starting profile. |
| `<run>_mc.csv` | Monte Carlo run | The Monte Carlo draws of time. |
| `<run>_time.png` | either of the two above | The time distributions on one axis. |
| `<run>_laws.png` | `--compare-laws` | The run under each of the four built-in laws. |
| `<run>_kriged.csv` | `uncertainty = legacy` | The kriged profile the legacy method fits. |

## summary.csv, by purpose

### The time and its ranges

| Column | Meaning |
|---|---|
| `time_days` | Best-fit time. |
| `low_days`, `high_days` | 1σ range (68%) from the profile alone, at a fixed starting profile, conditions and law. |
| `low2_days`, `high2_days` | 2σ range (95%). |
| `time_since_last_stage_days` | With growth stages: time since the last stage began (equal to `time_days` without). |
| `law_low_days`, `law_high_days` | 1σ range with the diffusion law's calibration uncertainty added (laws with a covariance only). |
| `all_low_days`, `all_high_days` | With the spread between laboratories as well (`refit`). |
| `law_sigma_log10`, `between_study_log10` | Those two uncertainties, as 1σ in log10 D. |

### With the starting profile fitted (`scan_` columns)

| Column | Meaning |
|---|---|
| `scan_best_days` | Time at the best-fit starting profile. |
| `scan_median_days`, `scan_low_days`, `scan_high_days` | Median and 68% range including the starting profile's uncertainty. |
| `scan_best` | Best-fit step positions and segment Fo. |
| `scan_prior_offsets` | How far each fitted value moved from its nominal value, in prior units. |
| `scan_ic_share` | Fraction of the variance of time that comes from the starting profile. |
| `scan_sensitivity` | Days per µm of each step and per unit Fo of each segment. |
| `scan_near_step_daic` | The near-step check at the best-fit starting profile. |
| `scan_sampler`, `scan_ess`, `scan_candidates`, `scan_log_evidence`, `scan_dropped` | How it was sampled ([Uncertainty beyond the fit](uncertainty.md#what-it-reports)). |
| `scan_marginal_low_days`, `scan_marginal_high_days` | The fully Bayesian range; equal to `scan_low_days`-`scan_high_days` for the sampled methods, different for `grid`. |

### With the Monte Carlo (`mc_` columns)

| Column | Meaning |
|---|---|
| `mc_median_days` | Median time with the fit (or the starting-profile posterior), the conditions and the law together. |
| `mc_low_days`, `mc_high_days` | 16th and 84th percentiles (68%). |
| `mc_p05_days`, `mc_p95_days` | 5th and 95th percentiles (90%). |
| `mc_sd_log10` | Spread of log10 time from each source alone (fit, law, studies, T, P, fO2): which input matters most. |
| `mc_method`, `mc_draws` | `rescale` (exact, no model runs) or `refit` (growth stages), and the number of draws. |

### How well the model fits (the noise model)

| Column | Meaning | How to read it |
|---|---|---|
| `white_sigma_fo` | Fitted point-to-point scatter (Fo fraction; 0.001 = 0.1 mol%). | Should be close to your analytical precision. The published profiles give 0.05-0.18 mol%. Much larger: outliers or unmodelled scatter. |
| `correlated_sigma_fo` | Size of the smooth misfit the model does not follow. | 0: white noise was enough. Larger than `white_sigma_fo`: the misfit dominates; check the starting profile. |
| `correlation_length_um` | Length of that misfit. | Compare with the diffusion length (about half the width of the smoothed step). About equal, or longer, or near half the profile: see the table below. |
| `reduced_chi2` | χ² over (points - 1). | About 1 by construction for the correlated method; judge fits by the three columns above instead. |
| `points_fitted`, `points_outside_domain` | Points inside and outside the model domain. | Points outside are not fitted; if they continue a flat plateau, extend the domain. |

### Checks

| Column | Meaning | Warns when |
|---|---|---|
| `near_step_daic` | AIC change when the misfit may be larger near the steps. | below -6 |
| `near_step_misfit_sigma_fo` | The near-step model's misfit size at the step. | |
| `near_step_points` | Points within two diffusion lengths of a step (-1: check not run). | below about 13 the check is weak (no warning) |
| `core_end_diffusion_lengths` | Distance from the last step to the end of the model, in diffusion lengths. | below 3 |
| `warnings` | The text of every warning for this run. | |

### The grid

| Column | Meaning |
|---|---|
| `grid_resolution` | Diffusion length over the grid spacing (about 8 by default for the implicit scheme). |
| `grid_error_pct` | Estimated worst-case error of the time from the grid spacing. Below 1% is fine. |
| `grid_stability` | For the explicit schemes only: D dt / dx² of the time step. |

### Inputs and provenance

Every column of the parameter table is repeated (fO2 in Pa, orientation as the three
angles), plus `diffusivity_label` (the law's display name) and `forster_version`. The
grid actually used (`dx_um`, `dt_s`) is recorded, so `summary.csv` can be given back to
`forster fit` as a parameter table to repeat the fits exactly.

## Which time to quote

| Situation | Quote |
|---|---|
| A single fit, no warnings | `time_days` with `low_days`-`high_days` (1σ). |
| Comparing crystals fitted with the same law and conditions | the 1σ ranges from the profiles alone (`low_days`-`high_days`, or `scan_low_days`-`scan_high_days`): the law's error is shared and cancels. |
| The starting profile was fitted | `scan_best_days` (or `scan_median_days`) with `scan_low_days`-`scan_high_days`. |
| A time on its own, with T, P, fO2 and the law | the Monte Carlo: `mc_median_days`, `mc_low_days`-`mc_high_days` (68%) or `mc_p05_days`-`mc_p95_days` (90%). |
| A warning says the 1σ range is probably too narrow | the 2σ range, `low2_days`-`high2_days` (see below). |

Always state the diffusion law (`diffusivity_label`) and the conditions with the time.

## Warnings

| What you see | Likely cause | What to do |
|---|---|---|
| No warning; `correlated_sigma_fo` ≈ 0; `near_step_daic` ≈ +2 | White noise only. | Nothing. Quote the 1σ range. |
| `correlated_sigma_fo` > 0 with `correlation_length_um` well below the diffusion length | A smooth misfit the noise model handles. | Nothing. Quote the 1σ range. |
| `correlation_length_um` about the diffusion length (no warning) | A misfit the size of the diffusion front, which trades off with time (the 1σ range covers about 57%). | Quote the 2σ range. Check the starting profile near the step. |
| "the misfit spans the profile", or a misfit length several times the diffusion length | A wrong starting or boundary value: a core, edge or plateau Fo off the model's, or a step in the wrong place. | Check them against the data. Give them uncertainties so FORSTER fits them, then quote `scan_low_days`-`scan_high_days`. |
| "the misfit is concentrated near the step" (`near_step_daic` below -6) | The starting profile near the step is not quite right. The 1σ range is probably too narrow (45-61% coverage). | Fit the starting profile first: give the step positions and the segment Fo uncertainties (`steps_sd_um`, `segment_fo_sd`). Check the fitted values are plausible. If `scan_near_step_daic` is still below -6, quote the 2σ range. |
| "even at the best-fit initial condition the misfit is concentrated near the step" | The starting profile may need a shape FORSTER does not have: a step that is not sharp, a graded rim, growth while diffusing. | Quote the 2σ range. Consider growth stages or another starting profile. |
| `near_step_points` below about 13 (no warning) | Too few points near the step for the check to see near-step misfit. | Treat a silent check as weak evidence; quote the 2σ range. Next time, put more analyses near the step. |
| "the profile ends ... diffusion lengths past the last step" (`core_end_diffusion_lengths` below 3) | The data stop at or before the plateau, where the held core Fo moves the fit. | Measure further into the core. Never trim plateau points. If the profile reaches a small crystal's centre, model the whole crystal (`centre_um`). |
| "the diffusion length at the best fit ... is shorter than the point spacing" | Only a point or two sit on the smoothed step. | Check that the fit has not collapsed onto the step; treat the range with care; measure more closely spaced points. |
| "the best ... is at the edge of its prior" | The data want a starting profile outside what you allowed. | Rethink the starting profile (another step, a rim zone) rather than simply widening the prior. The time is not reliable as it stands. |
| "the data place ... sd from the nominal value: the prior and the data disagree" | A prior that does not suit this crystal (one built from other crystals, say), pulling the fit. | Check where the prior came from, or the starting profile. |
| "1-sigma range reaches the last modelled time step" | The model did not run long enough. | Increase `max_time_days`. |
| "... of the posterior of time lies in the last 5% of the run" | The starting-profile fit's runs were too short. | Give a longer `scan_max_time_days`. |
| One or two isolated large residuals, `white_sigma_fo` above your precision | Outliers. | Check those analyses; remove only those with a known fault (mask them), and refit. Do not remove points to narrow the range. |
| Scatter visibly larger near the rim | Noise growing toward the rim (the 1σ range covers about 63%). | Quote the 2σ range. |
| "the profiles of crystal ... do not agree on one time" (joint fits) | Profiles of one crystal that need different times. | Check the orientations, whether a profile is off the crystal's centre, or a 3D effect near a corner, before trusting the joint fit. |

Two things are not worth worrying about: the exact shape of the misfit (smooth or
jagged) costs at most 5 points of coverage, and heavy-tailed noise or isolated outliers
leave coverage at 69-74%.

The evidence behind this table is in
[noise-models-guide.md](https://github.com/whtowbin/Olivine_Diffusion_FeMg/blob/master/docs/references/noise-models-guide.md)
and
[practical-advice.md](https://github.com/whtowbin/Olivine_Diffusion_FeMg/blob/master/docs/guide/practical-advice.md)
on GitHub.
