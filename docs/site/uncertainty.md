# Uncertainty beyond the fit

The fit's own range ([How the fit works](how-it-works.md)) treats the starting
profile, the conditions and the diffusion law as exact. This page adds their
uncertainty in three steps that build on each other:

1. **Fitting the starting profile**: uncertain step positions and segment Fo are
   fitted together with the time.
2. **The Monte Carlo over T, P, fO2 and the law**: the conditions and the law's
   parameters are drawn at random, and each draw is turned into a time.
3. **Law ranges**: a quick version of the law part of the Monte Carlo, reported for
   every fit whose law carries a covariance.

A few terms used below:

- What you know about a parameter before looking at this profile is its **prior**.
  You give it as a nominal value and an uncertainty: either a Gaussian 1σ (`_sd`
  columns) or a uniform ± half-width (`_range` columns, where every value in the range
  is equally likely).
- What you know after fitting, the prior combined with what the profile says, is the
  **posterior**. FORSTER reports the posterior of time averaged over every starting
  profile the posterior allows, each weighted by how probable it is. This is the
  posterior of time *marginalized* over the starting profile.

## Fitting the starting profile

### Giving the uncertainties

| Column | Meaning |
|---|---|
| `steps_sd_um` | 1σ (Gaussian) on the step positions: one value for all, or one per step (`"2, 5"`); 0 keeps a step fixed. |
| `steps_range_um` | ± half-width (uniform) on the step positions, instead of `steps_sd_um`. |
| `segment_fo_sd` | 1σ on each segment's Fo: one value for all, or one per segment (`"0.002, , 0.003"`; an empty entry keeps that segment fixed). |
| `segment_fo_range` | ± half-width on each segment's Fo, instead; a segment takes one of the two. |

For example, a step known to ±2 µm (1σ) and an edge Fo anywhere between 0.817 and
0.823 (a uniform ±0.003 around 0.82), with the core fixed:

```text
steps_um,segment_fo,steps_sd_um,segment_fo_range
41,"0.82, 0.90",2,"0.003, "
```

Gaussian priors are cut at ±2.5σ, and the fit never goes beyond that. In the app,
enter the steps' ± in the Uncertainties tab and each segment's ± in the Segments tab,
then press **Fit starting profile**.

The Fo of the first and last segments are also the values held at edge 1 and at the
far end during diffusion, so their uncertainty covers those boundary conditions too.

**What to give an uncertainty.** Anything you are unsure of, and especially:

- **the core Fo**, unless the plateau is visibly sloped. A core Fo 0.1-0.2 mol% off
  biased times by 5-21% with no warning, and fitting the core Fo removed the bias.
- **the steps and plateau levels when the near-step warning fires**. Fitting the
  starting profile cleared the warning on every tested profile with near-step misfit
  (WHT06_ol48 only just, at -5.4 against the threshold of -6). On the published
  profiles below it moved the time by -40%, +6% and -52% (the last is not reliable,
  see below). The starting profile matters more than the noise model.

### What the fit does

1. **Best fit.** An optimizer moves the uncertain step positions and segment Fo to
   find the starting profile that, with its best time, fits the data best. The prior
   enters as a penalty on the misfit. Each try costs one model run. That run gives the
   misfit at every time at once, so time is handled exactly rather than sampled. The
   noise model is fitted at the data's best fit, not at your nominal values, and
   refitted until it is stable.
2. **Sampling the posterior.** Around the best fit, FORSTER approximates the posterior
   with a bell-shaped curve. This is a *Laplace approximation*: a multivariate
   Gaussian taken from the curvature of the misfit at its minimum, made a little wider
   and with heavier tails to be safe. FORSTER draws 1024 starting profiles from it and
   weights each one by how much more or less probable it really is than the
   approximation says (*importance sampling*). The weighted draws give the exact
   posterior, up to sampling error.
3. **Checking the sample.** If a few draws carry most of the weight, the approximation
   was poor, for example when the data hardly constrain a parameter with a wide prior,
   or when there are two separate good fits. The **effective
   sample size** (ESS, `scan_ess`) measures this. It is the number of equally weighted
   draws that the weighted sample is worth. When the ESS is below 256 (a quarter of
   the draws), the `auto` method switches to **nested sampling** with the dynesty
   package. Nested sampling explores the whole prior. It is slower but reliable.

Most profiles need about 1,000 model runs and take under a second. Wide uniform priors
can need up to about 15,000 runs. `scan_seed` fixes the random draws, so refitting
from a summary gives the same numbers.

| `scan_method` | Does |
|---|---|
| `auto` (default) | Laplace importance sampling, with nested sampling when the ESS is too low. |
| `laplace` | Importance sampling only. |
| `nested` | Nested sampling only. |
| `grid` | Fits every node of a regular grid over the parameters (the method before 2026-10-15, kept for old results). It needs about 15 points per parameter to converge (`scan_points`), so it is slow for more than two parameters. |

`scan_dx_um`, `scan_dt_s` and `scan_max_time_days` set the grid of these model runs.
Left empty, FORSTER chooses a grid coarser than the final fit's, for about 5%
accuracy.

### What it reports

| Column | Meaning |
|---|---|
| `scan_best_days` | The time at the best-fit starting profile. |
| `scan_median_days`, `scan_low_days`, `scan_high_days` | Median and central 68% of the posterior of time, including the starting profile's uncertainty. **This is the range to quote** once the starting profile is fitted. |
| `scan_best` | The best-fit values (`step1_um`, `a_fo`, `b_fo`, ...; segments are named by their labels). |
| `scan_ic_share` | The fraction of the variance of time that comes from the starting profile (the rest is the fit at a fixed starting profile). |
| `scan_sensitivity` | How much the expected time changes per µm of each step and per unit Fo of each segment. |
| `scan_prior_offsets` | Each fitted value's distance from its nominal value in prior units (`a_fo=+2.30` is 2.3σ above). |
| `scan_near_step_daic` | The near-step check, redone at the best-fit starting profile. |
| `scan_ess`, `scan_sampler`, `scan_candidates` | How the sample was made, and how many model runs it took. |
| `scan_log_evidence` | How probable the data are under this starting-profile model as a whole; compare one-step and two-step starting profiles fitted to the same profile (higher is better). |

`<run>_scan.csv` holds the posterior of time (days, probability density, cumulative).

Take the worked example, with the step known to ±2 µm and both segments' Fo to ±0.002
([Quick start](quickstart.md#4-add-uncertainties-optional)). The best step moves from
24 to 21.3 µm. The range including the starting profile is 5.25-8.20 days, against
4.24-6.21 days for the fit alone, and 79% of the variance of time comes from the
starting profile.

### Look at the fitted starting profile before quoting the time

![Three published profiles with their starting profiles fitted](images/starting-profile-fit-examples.png)

*Three published profiles (Towbin and Plank 2026) with their nominal (blue, dashed)
and fitted (green, dotted) starting profiles, the best fit from each, and their
residuals. **WHT06_ol43**: the first step moves from 2 to 6.7 µm and the plateau rises
0.24 mol%. The residuals become white and the time falls 40%. **Xenocryst_ol6**: the
first step runs to the edge of its allowed range, past the first data point, and the
time halves. The data ask for a rim zone at least 6 µm wide, which a sharp step cannot
represent, so this time is not reliable. **WHT06_ol48**: the time changes little
(+6%). The warning only just clears and the residual bump near the step remains, a
sign that a single sharp step does not explain it.*

- A value **at the edge of its prior** (FORSTER warns) means the data want something
  outside what you allowed. Rethink the starting profile rather than widen the prior.
- A value **2σ or more from nominal** (FORSTER warns of a prior conflict) means the
  prior is pulling against the data. Check where the prior came from.
- Steps placed where there are no data, between the rim and the first point, are
  constrained only by the shape of the nearby points.
- If the near-step warning persists (`scan_near_step_daic` below -6), quote the 2σ
  range of the fit and consider a starting profile with another shape.

## The Monte Carlo over T, P, fO2 and the law

FORSTER can draw the uncertain inputs many times at random and look at the spread of
the resulting times. This is a **Monte Carlo**, and `forster fit` runs one when you
give any of these columns:

| Column | Default | Meaning |
|---|---|---|
| `mc_T_sd_C` | | 1σ of temperature (°C). |
| `mc_P_sd_Pa` | | 1σ of pressure (Pa). |
| `mc_fO2_sd_log` | | 1σ of log10 fO2. |
| `mc_fO2_buffer` | the run's buffer | QFM or NNO: draw fO2 relative to that buffer, so it rises and falls with each drawn temperature (as it does in nature for a buffered magma). Empty, with fO2 given in Pa: fO2 is drawn on its own. |
| `mc_law` | yes | Draw the diffusion law's parameters from its published covariance (the correlations between its fitted values). Laws without one, such as `dipra`, are taken as exact. |
| `mc_between_study` | no | Add the spread between laboratories (`refit` only): for an absolute time, if the crystal may sit off the laboratories' average as one more experiment would. |
| `mc_samples` | 4000 | Number of draws. |
| `mc_seed` | 0 | Fixes the draws. |

**How it works.** The fit fixes the product D × t: the same profile at twice the
diffusivity would take half the time. For each draw FORSTER computes D at the drawn
conditions with the drawn law and rescales the time: t = t_fit × D_fit / D_draw. The
fitted time is drawn too, from the fit's own distribution, or from the
starting-profile posterior when the starting profile was fitted. The result therefore
combines all the sources. Rescaling is exact for a held rim and needs no new
model runs, so it takes milliseconds. Runs with growth stages are refitted for each
draw instead (200 draws by default).

**What it reports.** FORSTER reports the median `mc_median_days`, the 68% range
`mc_low_days`-`mc_high_days` (16th-84th percentiles) and the 90% range
`mc_p05_days`-`mc_p95_days`. The draws are saved in `<run>_mc.csv`. `mc_sd_log10`
gives the spread of log10 time from each source drawn on its own, which tells you
which input matters most. For the worked example with ±20 °C, ±0.1 GPa and ±0.3 log
units:

```text
mc_median_days 6.48   mc_low_days 4.41   mc_high_days 9.64   (90%: 3.49-12.38)
mc_sd_log10    fit=0.0978; law=0; studies=0; T=0.128; P=0.0119; fO2=0.0496
```

Temperature (0.128 log units) contributes more than the fit and the starting profile
together (0.098). Pressure hardly matters. The law contributes 0 because the default
law has no covariance (use `refit` or a DFENS law to include it).

`<run>_time.png` and the app's Time distribution view show all three distributions on
one axis ([Quick start](quickstart.md#4-add-uncertainties-optional)).

## Law ranges

Every fit whose law carries a covariance (all built-in laws except `dipra`) also
reports these columns, with no extra model runs:

| Column | Meaning |
|---|---|
| `law_sigma_log10` | 1σ of log10 D from the law's covariance at the run's conditions (and so of log10 time). |
| `law_low_days`, `law_high_days` | The 1σ range of time with the law's uncertainty added to the fit's. |
| `between_study_log10` | The spread between laboratories in the law's calibration data (`refit`: 0.07 log units). |
| `all_low_days`, `all_high_days` | With that spread added as well. |

**Which range to quote.**

| You want to... | Quote |
|---|---|
| compare crystals fitted with the same law and conditions (relative times) | `low_days`-`high_days`: the law's error is common to all of them and cancels. |
| give a time on its own | `law_low_days`-`law_high_days`, or better the Monte Carlo range with your T, P and fO2 uncertainties. |
| give an absolute time that allows for the crystal sitting off the laboratories' average | `all_low_days`-`all_high_days`, or the Monte Carlo with `mc_between_study`. |

The law ranges are not available for runs with growth stages or with a rim value that
changes in time. In those runs the time no longer scales as 1/D, and the Monte Carlo
refits them instead. The law range also assumes that the form of the law is right.
The larger differences between laws ([Setting up a run](setup.md#the-diffusion-law))
are a separate, systematic question.
[law-uncertainty-limits.md](https://github.com/whtowbin/Olivine_Diffusion_FeMg/blob/master/docs/references/law-uncertainty-limits.md)
on GitHub explains what the law range assumes and where it was tested.
