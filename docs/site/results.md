# Reading the results

`forster fit` writes a file called `summary.csv` with one row per run, and Export in the
app writes the same file. This page groups its columns by purpose, says which time to
quote, and explains what to do about each of FORSTER's warnings. The
[Column reference](columns.md#summarycsv) lists every column.

## The files

| File | Written when | Holds |
|---|---|---|
| `summary.csv` | always | One row per run: results and every input. |
| `diffusivities.csv` | always | The diffusion equations the runs used. |
| `<run>_fit.png` | always | Data, starting profile, best fit, 1σ and 2σ bands. |
| `<run>_curves.csv` | always | Model profiles at time zero, at the best fit and at the ends of both ranges. |
| `<run>_points.csv` | always | Fitted points, the best fit at each point, and the residuals. |
| `<run>_scan.csv` | starting profile fitted | The posterior of time over the starting profile. |
| `<run>_mc.csv` | Monte Carlo run | The Monte Carlo draws of time. |
| `<run>_time.png` | either of the two above | The time distributions on one axis. |
| `<run>_laws.png` | `--compare-laws` | The run with each of the four built-in diffusion laws. |
| `<run>_kriged.csv` | `uncertainty = legacy` | The kriged profile that the legacy method fits. |

## summary.csv, by purpose

### The time and its ranges

| Column | Meaning |
|---|---|
| `time_days` | Best-fit time. |
| `low_days`, `high_days` | 1σ range (68%) from the profile alone, with the starting profile, conditions and diffusion law held fixed. |
| `low2_days`, `high2_days` | 2σ range (95%). |
| `time_since_last_stage_days` | With growth stages, the time since the last stage began. Without them it equals `time_days`. |
| `law_low_days`, `law_high_days` | 1σ range including the calibration uncertainty of the diffusion law (only for laws with a covariance). |
| `all_low_days`, `all_high_days` | 1σ range that also includes the spread between laboratories. That spread is known only for the `refit` equation; for the others this range equals the law range. |
| `law_sigma_log10`, `between_study_log10` | Those two uncertainties (calibration, and spread between laboratories) as 1σ in log10 D. |

### With the starting profile fitted (`scan_` columns)

| Column | Meaning |
|---|---|
| `scan_best_days` | Time at the best-fit starting profile. |
| `scan_median_days`, `scan_low_days`, `scan_high_days` | Median time and 68% range, including the uncertainty of the starting profile. |
| `scan_best` | Best-fit step positions and segment Fo. |
| `scan_prior_offsets` | How far each fitted value moved from its nominal value, in prior units. |
| `scan_ic_share` | Fraction of the variance of time that comes from the starting profile. |
| `scan_sensitivity` | Days per µm of each step and per unit Fo of each segment. |
| `scan_near_step_daic` | The near-step check at the best-fit starting profile. |
| `scan_sampler`, `scan_ess`, `scan_candidates`, `scan_log_evidence`, `scan_dropped` | How the starting profile was sampled ([Uncertainty beyond the fit](uncertainty.md#what-it-reports)). |
| `scan_marginal_low_days`, `scan_marginal_high_days` | The fully Bayesian range. It equals `scan_low_days`-`scan_high_days` for the sampled methods and differs for `grid`. |

### With the Monte Carlo (`mc_` columns)

| Column | Meaning |
|---|---|
| `mc_median_days` | Median time with the uncertainties of the fit (or of the fitted starting profile), the conditions and the diffusion law combined. |
| `mc_low_days`, `mc_high_days` | 16th and 84th percentiles (68%). |
| `mc_p05_days`, `mc_p95_days` | 5th and 95th percentiles (90%). |
| `mc_sd_log10` | Spread of log10 time from each source on its own (fit, law, studies, T, P, fO2). It shows which input matters most. |
| `mc_method`, `mc_draws` | `rescale` (exact, no model runs) or `refit` (growth stages), and the number of draws. |

### How well the model fits (the noise model)

| Column | Meaning | How to read it |
|---|---|---|
| `white_sigma_fo` | Fitted scatter from point to point, with Fo as a fraction (0.001 = 0.1 mol%). | Should be close to your analytical precision. The published profiles give 0.05-0.18 mol%. A much larger value points to outliers or scatter the model does not account for. |
| `correlated_sigma_fo` | Size of the smooth misfit, the part of the data the model does not follow. | 0 means white noise alone was enough. If it is larger than `white_sigma_fo`, the misfit dominates and the starting profile needs checking. |
| `correlation_length_um` | Length of that misfit along the profile. | Compare it with the diffusion length (about half the width of the smoothed step). If the two are about equal, if the misfit is longer, or if it is near half the profile, see the warnings table below. |
| `reduced_chi2` | χ² over (points - 1). | About 1 by construction for the correlated method, so judge the fit by the three columns above instead. |
| `points_fitted`, `points_outside_domain` | Points inside and outside the model domain. | Points outside the domain are not fitted. If they continue a flat plateau, extend the domain. |

### Checks

| Column | Meaning | Warns when |
|---|---|---|
| `near_step_daic` | Change in AIC when the misfit is allowed to be larger near the steps. | below -6 |
| `near_step_misfit_sigma_fo` | Size of the misfit at the step in that near-step model. | |
| `near_step_points` | Points within two diffusion lengths of a step (-1: check not run). | below about 13 the check is weak (no warning) |
| `core_end_diffusion_lengths` | Distance from the last step to the end of the model, in diffusion lengths. | below 3 |
| `warnings` | The text of every warning for this run. | |

### The grid

| Column | Meaning |
|---|---|
| `grid_resolution` | Diffusion length over the grid spacing (about 8 by default for the implicit scheme). |
| `grid_error_pct` | Estimated worst-case error in the time from the grid spacing. Below 1% is fine. |
| `grid_stability` | D dt / dx² for the time step (explicit schemes only). |

### Inputs and provenance

`summary.csv` repeats every column of the parameter table, with fO2 in Pa and the
orientation as its three angles. It adds `diffusivity_label` (the display name of the
diffusion law) and `forster_version`, and it records the grid actually used (`dx_um`,
`dt_s`). You can therefore give `summary.csv` back to `forster fit` as a parameter table
and repeat the fits exactly.

## Which time to quote

| Situation | Quote |
|---|---|
| A single fit, no warnings | `time_days` with `low_days`-`high_days` (1σ). |
| Comparing crystals fitted with the same law and conditions | the 1σ ranges from the profiles alone (`low_days`-`high_days`, or `scan_low_days`-`scan_high_days`). The law's error is common to all the crystals and cancels. |
| The starting profile was fitted | `scan_best_days` (or `scan_median_days`) with `scan_low_days`-`scan_high_days`. |
| A time that stands on its own, with the uncertainty in T, P, fO2 and the law | the Monte Carlo: `mc_median_days`, `mc_low_days`-`mc_high_days` (68%) or `mc_p05_days`-`mc_p95_days` (90%). |
| A warning says the 1σ range is probably too narrow | the 2σ range, `low2_days`-`high2_days` (see below). |

Always report the diffusion law (`diffusivity_label`) and the conditions with the time.

## Warnings

| What you see | Likely cause | What to do |
|---|---|---|
| No warning; `correlated_sigma_fo` ≈ 0; `near_step_daic` ≈ +2 | White noise only. | Nothing. Quote the 1σ range. |
| `correlated_sigma_fo` > 0 with `correlation_length_um` well below the diffusion length | A smooth misfit that the noise model handles. | Nothing. Quote the 1σ range. |
| `correlation_length_um` about the diffusion length (no warning) | A misfit as wide as the diffusion front, which trades off against time. The 1σ range then covers about 57%. | Quote the 2σ range. Check the starting profile near the step. |
| "the misfit spans the profile", or a misfit length several times the diffusion length | A wrong starting or boundary value. The core, edge or plateau Fo in the model differs from the data, or a step is in the wrong place. | Check these values against the data. Give them uncertainties so that FORSTER fits them, then quote `scan_low_days`-`scan_high_days`. |
| "the misfit is concentrated near the step" (`near_step_daic` below -6) | The starting profile near the step is not quite right. The 1σ range is probably too narrow (45-61% coverage). | Fit the starting profile first by giving uncertainties to the step positions and the segment Fo (`steps_sd_um`, `segment_fo_sd`). Check that the fitted values are plausible. If `scan_near_step_daic` is still below -6, quote the 2σ range. |
| "even at the best-fit initial condition the misfit is concentrated near the step" | The starting profile may need a shape FORSTER does not offer, such as a step that is not sharp, a graded rim, or growth during diffusion. | Quote the 2σ range. Consider growth stages or another starting profile. |
| `near_step_points` below about 13 (no warning) | Too few points near the step for the check to see a near-step misfit. | A silent check is weak evidence here, so quote the 2σ range. On the next traverse, place more analyses near the step. |
| "the profile ends ... diffusion lengths past the last step" (`core_end_diffusion_lengths` below 3) | The data stop at or before the core plateau, so the core Fo held at the model's far end moves the fit. | Measure further into the core, and never trim plateau points. If the traverse reaches the center of a small crystal, model the whole crystal (`centre_um`). |
| "the diffusion length at the best fit ... is shorter than the point spacing" | Only one or two points lie on the smoothed step. | Check that the fit has not collapsed onto the step, and treat the range with care. Measure points more closely spaced. |
| "the best ... is at the edge of its prior" | The data favor a starting profile outside the range you allowed. | Rethink the starting profile (another step, a rim zone) instead of only widening the prior. As it stands, the time is not reliable. |
| "the data place ... sd from the nominal value: the prior and the data disagree" | The prior does not suit this crystal (it was built from other crystals, for example) and pulls the fit. | Check where the prior came from, and check the starting profile. |
| "1-sigma range reaches the last modeled time step" | The model did not run long enough. | Increase `max_time_days`. |
| "... of the posterior of time lies in the last 5% of the run" | The model runs in the starting-profile fit were too short. | Give a longer `scan_max_time_days`. |
| One or two isolated large residuals, `white_sigma_fo` above your precision | Outliers. | Check those analyses. Remove only those with a known fault (mask them) and refit. Do not remove points to narrow the range. |
| Scatter visibly larger near the rim | Noise that grows toward the rim. The 1σ range then covers about 63%. | Quote the 2σ range. |
| "the profiles of crystal ... do not agree on one time" (joint fits) | The profiles of one crystal need different times. | Before trusting the joint fit, check the orientations, whether a profile runs off the crystal's center, and whether a corner causes a 3D effect. |

Two things matter less than they might seem. The exact shape of the misfit (smooth or
jagged) costs at most 5 percentage points of coverage. Heavy-tailed noise or isolated
outliers leave coverage at 69-74%.

The evidence behind this table is in
[noise-models-guide.md](https://github.com/whtowbin/Olivine_Diffusion_FeMg/blob/master/docs/references/noise-models-guide.md)
and
[practical-advice.md](https://github.com/whtowbin/Olivine_Diffusion_FeMg/blob/master/docs/guide/practical-advice.md)
on GitHub.
