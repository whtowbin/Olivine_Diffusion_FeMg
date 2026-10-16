# FORSTER olivine diffusion

**F**itting **O**livine **R**esidence & **S**torage **T**imescales from **E**lemental **R**elaxation.

FORSTER fits Fe-Mg diffusion times to forsterite (Fo) profiles measured across zoned
olivine crystals. You give it a measured profile, the temperature, pressure and
oxygen fugacity, the orientation of the profile in the crystal, and a step-shaped
starting profile. It runs the diffusion model forward in time and reports the time
whose model profile best matches the data, with a 1σ range.

It is a development version (0.1.0.dev0): results from this version differ from
earlier versions of this code (see [CHANGELOG.md](CHANGELOG.md)).

## Install

FORSTER needs Python 3.11 or newer. With [uv](https://docs.astral.sh/uv/):

```bash
uv tool install git+https://github.com/whtowbin/Olivine_Diffusion_FeMg   # the `forster` command
uv add git+https://github.com/whtowbin/Olivine_Diffusion_FeMg            # or as a library in a uv project
```

or with pip: `pip install git+https://github.com/whtowbin/Olivine_Diffusion_FeMg`.
A PyPI release (`pip install forster`) will follow.

Optional, about 5× faster fits and scans: add the `fast` extra, which installs
[Numba](https://numba.pydata.org/) to compile the time steppers
(`uv tool install "forster[fast] @ git+https://github.com/whtowbin/Olivine_Diffusion_FeMg"`,
or `pip install "forster[fast] @ git+..."`). Results are the same to rounding; without
Numba the same code runs in NumPy. `FORSTER_NO_NUMBA=1` turns it off.

## Three ways to run it

### 1. Command line, with two filled-in tables

```bash
forster template my_fits/      # writes profiles.csv, parameters.csv, diffusivities.csv and COLUMNS.md
forster fit my_fits/profiles.csv my_fits/parameters.csv --out my_fits/results
```

The template already holds a worked example, one published profile (AZ18
WHT01_Ol_63_prof_1 from Towbin and Plank 2026), so the second command works straight
away and shows what the output looks like. Then replace the example rows with your
own: one row per analysis point in `profiles.csv`, one row per model run in
`parameters.csv`. The columns are listed below and in `COLUMNS.md`. Excel files work
too (`--profiles-sheet` and `--parameters-sheet` pick the sheet). To use your own
diffusivity equation, add a row to `diffusivities.csv`, name it in the `diffusivity`
column of `parameters.csv`, and add `--diffusivities my_fits/diffusivities.csv`.

**Choosing the diffusivity law.** The `diffusivity` column takes one of four Fe-Mg
choices: `dipra` (Classic DIPRA: Dohmen and Chakraborty 2007, the default), `refit`
(this project's refit of the experiments, `femg_refit2026`), `dfens_tamed` or
`dfens_global` (the Mutch et al. 2019 regressions used by DFENS). `--diffusivity NAME`
fits every run with one of them, whatever the column says, and `--compare-laws` fits
every run once with each of the four (rows named `<run>_dipra`, `<run>_refit`, ...), so
the effect of the law on the time sits next to the fit's own range.

For each run, `forster fit` writes into the output folder:

- `<run>_fit.png`: the measured points, the initial profile, the best fit and the 1σ range;
- `<run>_curves.csv`: the model profile at time zero, at the best fit, and at the two ends of the 1σ range;
- `<run>_points.csv`: the measured points that were fitted, the best fit at each one and the residual;
- `<run>_kriged.csv`: the kriged (interpolated) profile, for `uncertainty = legacy` only
  (the default method fits the measured points and does not krige);
- `summary.csv`: one row per run with the time (`time_days`), its 1σ range (`low_days`, `high_days`), the fit statistics and every input;
- `diffusivities.csv`: the diffusivity equations the runs used.

**Uncertain initial conditions.** Add an uncertainty to any step position or
concentration in `parameters.csv` and `forster fit` also fits those parameters: it finds
the best-fit initial condition and the posterior of time over the uncertain values
(importance sampling from a Laplace approximation, or nested sampling where that is
not enough; about 1,000 model runs and under a second for most profiles, up to about
15,000 runs for wide uniform priors). For example, a step known to ±2 µm (1σ) and an
edge Fo between 0.817 and 0.823:

```
steps_um,segment_fo,steps_sd_um,segment_fo_range
41,"0.82, 0.90",2,"0.003, "
```

The summary then has `scan_low_days` and `scan_high_days` (the range including the
initial condition), `scan_best` (the best step and edge), `scan_ic_share` (how much of
the variance of time comes from the initial condition) and `scan_sensitivity` (days per
µm of step, per unit Fo), and `<run>_scan.csv` holds the posterior of time. `_sd`
columns are Gaussian 1σ, `_range` columns are uniform ± half-widths. `scan_seed` fixes
the sampler, so a summary refits to the same numbers. `scan_method = grid` (or giving
`scan_points`) fits every node of a grid as before instead; it needs about 15 points
per parameter to converge (`analyses/ic_sampling/`).

**Temperature, pressure, fO2 and the law.** A fitted time is a fitted D·t, so it is
only as good as the conditions and the diffusion equation. Give any of `mc_T_sd_C`,
`mc_P_sd_Pa`, `mc_fO2_sd_log` (1σ each) and `forster fit` draws them, and the
equation's parameters from its covariance (`mc_law`, on by default), and reports the
time from all of it together: `mc_median_days`, `mc_low_days`/`mc_high_days` (68%) and
`mc_p05_days`/`mc_p95_days` (90%), with `mc_sd_log10` saying how much each source
contributes. When fO2 is given relative to a buffer it stays relative to it (`mc_fO2_buffer`),
so it rises and falls with each drawn temperature. Times rescale exactly for a fixed rim
(no extra model runs); runs with growth stages are refitted per draw. The draws are in
`<run>_mc.csv`, and `<run>_time.png` shows the fit alone, with the initial condition and
with everything, on one axis. `mc_seed` makes it repeatable.

`summary.csv` uses the parameter-table column names, so it can be passed back to
`forster fit` as a parameter table to rerun the same fits (with `--diffusivities`
pointing at the output folder's `diffusivities.csv` if a run used your own equation). A run that fails (a
missing profile, say) is reported and skipped; the others are still fitted, and the
command exits with status 1. `--run NAME ...` fits only the named runs.

### 2. Python

The same tables from Python:

```python
import forster

profiles = forster.read_profiles("my_fits/profiles.csv")
runs = forster.read_runs("my_fits/parameters.csv")
result = forster.fit_run(runs[0], profiles)
print(result.time_days, result.time_minus_days, result.time_plus_days)
forster.plot_fit(result)
```

or without tables, from arrays:

```python
from forster import (
    Conditions, InitialCondition, Orientation, Profile, choose_grid, fit_profile,
)

profile = Profile(distance_um, fo, name="my profile")  # Fo as a fraction
conditions = Conditions.from_buffer(T_C=1250, P_Pa=5e8, delta_log=0.3, buffer="QFM")
orientation = Orientation.from_euler(77.0, 61.2, 334.2, profile_angle_deg=-80)
initial = InitialCondition(domain_um=(2.5, 72), steps_um=(24,), segment_fo=(0.9146, 0.8985))
grid = choose_grid(profile, conditions, orientation, initial).grid
result = fit_profile(profile, conditions, orientation, initial, grid)
```

`choose_grid` picks dx, dt and the longest modelled time for the implicit scheme
(`docs/references/grid-choice.md`); give your own with
`forster.Grid(dx_um, dt_s, max_time_days)`.

`Conditions(T_C, P_Pa, fO2_Pa)` and `Orientation(alpha_deg, beta_deg, gamma_deg)`
take the conditions and angles directly. `fit_profile(..., diffusivity=...)` takes
a choice (`dipra`, `refit`, `dfens_tamed`, `dfens_global`; `forster.diffusivity.FE_MG_CHOICES`),
a preset name (`forster.DIFFUSIVITY_PRESETS`), a row of a table read with
`forster.read_diffusivities`, or an `ArrheniusDiffusivity` built in Python. `forster.summary_row(result)` and
`forster.fit_tables(result)` give the tables that `forster fit` writes.

#### Ni, Mn and Ca

`forster.fit_element` fits a Ni, Mn or Ca profile for its own time (a check against
the Fo time), and `forster.fit_joint` fits one time to Fo and elements together and
says whether they agree (`consistency_p`). Ni and Mn diffuse with the Fo profile
alongside them, since their diffusivity depends on Fo; Ca does not need it. The
element's edge, core and plateau concentrations are fitted unless given:

```python
from forster import ElementProfile, fit_element, fit_joint

ni = ElementProfile(distance_um, nio_wt_pct, "Ni", name="my profile")   # any unit
result = fit_element(ni, conditions, orientation, fo_initial, None, grid)
joint = fit_joint(fo_profile, [ni], conditions, orientation, fo_initial, None, grid)
print(result.time_days, joint.time_days, joint.consistency_p, joint.species_days)
```

Python only for now (explicit schemes); `docs/plans/2026-10-13-minor-elements.md`.

#### Growth stages

A zone that grew partway through can be given as a stage: at its start the rim moves
outward and the new zone appears, then diffuses with the older crystal.
`initial` is the crystal before the first stage.

```python
from forster import Stage, fit_profile

overgrowth = Stage(start_s=6 * 86400, rim_um=0.0, zone_value=0.82)
result = fit_profile(profile, conditions, orientation, initial, grid, stages=[overgrowth])
print(result.time_days, result.time_since_last_stage_days)
```

With `scheme="implicit"` stages start at their exact times and the rim value can
change in time (see "How the fit works"). A staged and a single-stage history can
fit a profile equally well with different times, so the history has to come from
other evidence; `docs/plans/2026-10-13-growth-stages.md`.

### 3. Desktop app

A desktop app (`forster gui`) is planned for the next version: load a profile,
place the initial steps by dragging them on the plot, edit the parameters, fit and
export. It uses tkinter, which ships with most Python installs; Homebrew and Linux
system Pythons need it installed separately (`brew install python-tk`,
`apt install python3-tk`).

## Table columns

`forster template` writes this guide as `COLUMNS.md`.

### profiles.csv

One row per analysis point.

| Column | Unit | Needed | Meaning |
|---|---|---|---|
| `profile` |  | required | Profile name; `profile` in parameters.csv refers to it. |
| `distance_um` | µm | required | Distance of the analysis from the crystal edge. |
| `fo` | fraction | required | Forsterite content, 0-1 (0.895, not 89.5). |

### parameters.csv

One row per model run. Fo is a fraction throughout.

| Column | Unit | Needed | Meaning |
|---|---|---|---|
| `run` |  | required | Name of the model run (unique); names the output files. |
| `profile` |  | required | Which profile in profiles.csv to fit. |
| `T_C` | °C | required | Temperature. |
| `P_Pa` | Pa | required | Pressure (5e8 is 0.5 GPa). |
| `fO2_Pa` | Pa | one form | Oxygen fugacity. Or give the next two columns. |
| `fO2_buffer` |  | one form | QFM (or FMQ) or NNO (Frost 1991). |
| `fO2_delta_log` | log units | one form | fO2 relative to that buffer. |
| `alpha_deg` | degrees | one form | Angle between the profile and [100]. |
| `beta_deg` | degrees | one form | Angle between the profile and [010]. |
| `gamma_deg` | degrees | one form | Angle between the profile and [001]. |
| `phi1_deg` | degrees | one form | EBSD Euler angle phi1 (Bunge). |
| `Phi_deg` | degrees | one form | EBSD Euler angle Phi (Bunge). |
| `phi2_deg` | degrees | one form | EBSD Euler angle phi2 (Bunge). |
| `profile_angle_deg` | degrees | one form | Direction of the profile on the EBSD map, from the positive x axis (east). |
| `domain_start_um` | µm | required | Start of the model domain: edge 1, where Fo is held (at edge1_fo). |
| `domain_end_um` | µm | required | End of the model domain: the far end, where Fo is held at the last segment's Fo (the core), unless centre_um is given. |
| `steps_um` | µm | required | Step positions of the initial profile, in order: one number, or several in quotes ("10, 33"). They divide the profile into segments. |
| `segment_fo` | fraction | required | Initial Fo of each segment, from edge 1 inward: one more value than steps_um ("0.91, 0.89" for one step; "0.91, 0.93, 0.89" for two). Replaces edge_fo, core_fo and step_fo, which are still read until the release. |
| `segment_names` |  | optional | Labels of the segments, from edge 1 inward ("rim, mantle, core"). Empty: a, b, c, ... Segments are matched by label in joint fits; uncertainty results name them (a_fo, ...). |
| `edge1_fo` | fraction | optional | Fo held at edge 1 (domain_start_um) during diffusion, when it differs from the first segment's Fo, which stays the start profile's value. Empty: the first segment's Fo. With growth stages it holds until the first stage. No initial-condition scan with edge1_fo. Replaces rim_fo. |
| `centre_um` | µm | optional | Distance of the crystal's centre, to model the whole crystal: the model runs to the centre (beyond the measured points if needed) and mirrors there, so no Fo crosses it and the core is free to change. Empty: a half profile, its far end held at the core's Fo. Needs the implicit scheme and no growth stages. |
| `stage_days` | days | optional | Growth stages: when each stage starts, in days after the start of the run, one entry per stage ("2, 5" for two). At its start a stage moves the rim out to stage_rim_um, where a new zone of Fo stage_zone_fo appears and diffuses with the older crystal; the domain, step and Fo columns above describe the crystal before the first stage. Only times after the last stage are fitted; time_days counts from the start. Needs dx_um and dt_s, the correlated uncertainty method, and no initial-condition scan (no _sd or _range columns). |
| `stage_rim_um` | µm | optional | Rim position from each stage on, one entry per stage: at or below the previous rim (domain_start_um before the first stage). The previous rim itself changes only the rim's Fo (stage_rim_fo). Required with stage_days. |
| `stage_zone_fo` | fraction | optional | Fo of the zone each stage adds between the old and the new rim, one entry per stage; needed by every stage that moves the rim. Leave an entry empty ("0.91, ") for a stage that only changes the rim's Fo. |
| `stage_rim_fo` | fraction | optional | Fo held at the rim from each stage on, one entry per stage. Empty, or an empty entry: that stage's stage_zone_fo. |
| `dx_um` | µm | optional | Grid spacing of the final fit. Leave empty to have it chosen for about 1% accuracy in the fitted time; a value you give is kept, with a warning if it is coarse, wastefully fine, or too coarse for the initial steps. |
| `dt_s` | s | optional | Time step of the final fit, which sets how finely the fitted time is resolved. Leave empty to have it chosen: 1/400 of the fitted time. The explicit schemes (central, dipra) split a dt above their stability limit (D dt / dx² = 0.5) into stable steps and keep only the rows at whole time steps (not with growth stages, which need a stable dt). |
| `max_time_days` | days | optional | Longest time modelled (default 200; with dx_um or dt_s empty, 3x a quick estimate of the time). |
| `scheme` |  | optional | implicit (default since 2026-10-13: rim on the end node, steps not snapped to nodes, any time step), central (the engine's explicit scheme, used before) or dipra (DIPRA's forward-difference scheme, Girona and Costa 2013; a numerical scheme, not the diffusivity law). With uncertainty = legacy the default is central. |
| `diffusivity` |  | optional | Fe-Mg diffusivity equation. One of dipra: Classic DIPRA (Dohmen & Chakraborty 2007), preset dohmen_chakraborty2007; refit: FORSTER refit (2026), preset femg_refit2026; dfens_tamed: DFENS TaMED (Mutch et al. 2019), preset mutch2019_tamed; dfens_global: DFENS Global (Mutch et al. 2019), preset mutch2019_global. dipra is the default. Every choice but dipra carries a covariance, which adds the law_ range to summary.csv; refit also carries the spread between laboratories (the all_ range). Any other preset name, or a name from diffusivities.csv (then run forster fit with --diffusivities), also works. |
| `uncertainty` |  | optional | correlated (default) or legacy (the round robin rule; reproduces old results). |
| `steps_sd_um` | µm | optional | Uncertainty (1 sigma, Gaussian) on the step position(s): one value, or one per step ("2, 5"); 0 keeps a step fixed. Or give steps_range_um. |
| `steps_range_um` | µm | optional | Uncertainty (± half-width, uniform) on the step position(s), as steps_sd_um. |
| `segment_fo_sd` | fraction | optional | Uncertainty (1 sigma, Gaussian) on each segment's Fo, as segment_fo: one value for every segment, or one per segment ("0.002, , 0.003"); empty or 0 keeps a segment fixed. Replaces edge_fo_sd, core_fo_sd and step_fo_sd. |
| `segment_fo_range` | fraction | optional | Uncertainty (± half-width, uniform) on each segment's Fo, as segment_fo_sd; a segment may take one of the two, not both. |
| `scan_method` |  | optional | How the uncertain initial-condition parameters are fitted: auto (default: best fit, then importance sampling from a Laplace proposal, or nested sampling when that has too few effective samples), laplace, nested, or grid (every node of a grid, as before 2026-10-15). Blank with scan_points given means grid. |
| `scan_seed` |  | optional | Seed of the sampler (default 0); the same seed gives the same result. |
| `scan_points` |  | optional | scan_method grid: points per uncertain parameter (default 13 for one, 9 for two, 5 for more). |
| `scan_dx_um` | µm | optional | Grid spacing of the initial-condition scan's runs. Empty: chosen for about 5% accuracy (coarser and faster than the final fit). |
| `scan_dt_s` | s | optional | Time step of the scan's runs. Empty: 1/80 of the nominal fit's time, and for central or dipra at least D dt / dx² = 0.4 for every candidate. |
| `scan_max_time_days` | days | optional | Longest time the scan's runs model. Empty: max_time_days if given, else 3x the fitted time. |
| `mc_T_sd_C` | °C | optional | Monte Carlo over the conditions and the law (forster.rescale; run when any mc_ spread column is given): 1-sigma of temperature. |
| `mc_P_sd_Pa` | Pa | optional | Monte Carlo: 1-sigma of pressure. |
| `mc_fO2_sd_log` | log10 | optional | Monte Carlo: 1-sigma of log10 fO2 (relative to mc_fO2_buffer when given). |
| `mc_fO2_buffer` |  | optional | Monte Carlo: QFM or NNO to hold fO2 relative to that buffer, so it follows each draw's T and P. Empty: the row's fO2_buffer when fO2 is given that way, else fO2 is drawn on its own. |
| `mc_law` |  | optional | Monte Carlo: yes (default) to draw the law's parameters from its covariance (laws without one, such as dipra, are exact), or no. |
| `mc_between_study` |  | optional | Monte Carlo: yes to add the spread between laboratories (for an absolute time); default no. |
| `mc_samples` |  | optional | Monte Carlo draws (default 4000; 200 refits for runs with growth stages). |
| `mc_seed` |  | optional | Monte Carlo seed (default 0). |
| `notes` |  | optional | Free text for your own records. |

*one form*: give fO2 either as `fO2_Pa` or as `fO2_buffer` with `fO2_delta_log`, and
the orientation either as `alpha_deg`, `beta_deg`, `gamma_deg` or as the four EBSD
columns. Leave the other form empty.

### diffusivities.csv

Optional: your own diffusivity equations, one row per equation, named in the `diffusivity` column of parameters.csv and passed with `--diffusivities`. For each crystal axis i ([100], [010], [001]):

`log10 D_i = log10_D0_i + fO2_exponent log10(fO2 / fO2_ref_Pa) + fo_coefficient (fo_ref - X_Fo) - (E_i + activation_volume_m3_mol (P - P_ref_Pa)) / (ln(10) R T) + pressure_coefficient_per_Pa (P - P_ref_Pa)`

with T in K, P and fO2 in Pa and R = 8.3145 J/mol/K; along the profile, D is the sum of D_i cos²(angle to axis i). The table `forster template` writes holds the presets; copy a row, rename it and change the values. A row with a preset's name must keep the preset's values.

| Column | Unit | Needed | Meaning |
|---|---|---|---|
| `name` |  | required | Name to use in the `diffusivity` column of parameters.csv. |
| `element` |  | required | Fe-Mg for fits (Ni, Mn and Ca are listed for reference). |
| `log10_D0_100` | log10 m²/s | required | Pre-factor along [100]. |
| `log10_D0_010` | log10 m²/s | required | Pre-factor along [010]. |
| `log10_D0_001` | log10 m²/s | required | Pre-factor along [001]. |
| `E_100_J_mol` | J/mol | required | Activation energy along [100]. |
| `E_010_J_mol` | J/mol | required | Activation energy along [010]. |
| `E_001_J_mol` | J/mol | required | Activation energy along [001]. |
| `fO2_exponent` |  | required | Exponent n of (fO2 / fO2_ref_Pa). |
| `fO2_ref_Pa` | Pa | required | Reference fO2 (1e5 if the equation uses bar). |
| `fo_coefficient` | log10 per unit Fo | required | m in m (fo_ref - X_Fo); 0 for none. |
| `fo_ref` | fraction | required | Reference Fo of the composition term. |
| `activation_volume_m3_mol` | m³/mol | required | Activation volume (0 for none). |
| `P_ref_Pa` | Pa | required | Reference pressure of the pressure terms. |
| `pressure_coefficient_per_Pa` | log10 per Pa | required | Pressure term outside the exponential (0 for none). |
| `source` |  | optional | Where the equation comes from. |

### summary.csv

Written by `forster fit`: one row per fitted run with `run`, `profile`, the results below, and every parameter column above (fO2 in Pa, orientation as angles).

| Column | Unit | Meaning |
|---|---|---|
| `diffusivity_label` |  | Display name of the diffusivity equation (the diffusivity column holds its preset name). |
| `time_days` | days | Best-fit time. |
| `time_since_last_stage_days` | days | Best-fit time since the start of the last growth stage (stage_days); equal to time_days without stages. |
| `low_days` | days | Low end of the 1-sigma range. |
| `high_days` | days | High end of the 1-sigma range. |
| `low2_days` | days | Low end of the 2-sigma range (95.45%; correlated method, empty for legacy). |
| `high2_days` | days | High end of the 2-sigma range. |
| `law_sigma_log10` | log10 | 1-sigma of log10 D from the diffusivity law's covariance at the run's conditions (and of log10 time, with the sign changed). Empty for laws without one (the default) and for fits with growth stages or a rim value that changes in time (docs/references/law-uncertainty-limits.md). |
| `between_study_log10` | log10 | Spread between laboratories of the law's calibration data (a random study offset). |
| `law_low_days` | days | Low end of the 1-sigma range with the law's uncertainty added. Use low_days and high_days to compare crystals fitted with the same law (its error is common to all of them); these for a time on its own. |
| `law_high_days` | days | High end of that range. |
| `all_low_days` | days | Low end with the spread between laboratories added as well: for an absolute time, if the crystal may sit off the laboratories' average as one more experiment would. |
| `all_high_days` | days | High end of that range. |
| `reduced_chi2` |  | Chi-squared at the best fit over (points fitted - 1). Diagnostic only: about 1 by construction for the correlated method, whose noise model is fitted to the residuals; for legacy it is over the kriged grid points, so the two are not comparable. Judge correlated fits by the three noise columns. |
| `points_fitted` |  | Measured points inside the model domain (legacy: kriged grid points). |
| `white_sigma_fo` | fraction | Fitted scatter of the points (correlated method). Much larger than your analytical precision: outliers, or the precision is underestimated. |
| `correlated_sigma_fo` | fraction | Fitted smooth misfit between model and data. Large next to white_sigma_fo: the model misses the shape of the profile; check the initial condition. |
| `correlation_length_um` | µm | Length scale of that misfit; near half the profile length, the misfit is profile-wide. |
| `near_step_daic` |  | AIC change when the misfit may be larger near the steps (correlated method, no growth stages); below -6 a warning says the misfit is concentrated near the step and the 1-sigma range is probably too narrow (docs/references/noise-models-guide.md). |
| `core_end_diffusion_lengths` |  | Distance from the last step to the model's core end (the last measured point), in diffusion lengths at the best fit. The core Fo is held there; below 3 a warning says the data probably stop at or before the plateau, which moves the fit (docs/guide/practical-advice.md). |
| `points_outside_domain` |  | Measured points before domain_start_um or past domain_end_um, which are not fitted. Often deliberate (an earlier stage left out); if they continue a flat plateau, extending the domain to include them makes the near-step check more reliable. |
| `warnings` |  | Warnings from the fit, such as a range that reaches the last time step. |
| `forster_version` |  | FORSTER version that made the fit. |
| `scan_best_days` | days | With uncertainties: best-fit time over all scanned initial conditions (sampled: the time of the best-fit initial condition). |
| `scan_sampler` |  | Method that produced the scan columns: grid, laplace or nested (auto runs laplace and falls back to nested). |
| `scan_median_days` | days | Sampled: median of the posterior of time over the initial-condition parameters. |
| `scan_low_days` | days | Low end of the 1-sigma range including the initial-condition uncertainty (grid: profile likelihood; sampled: central 68% of the posterior of time). |
| `scan_high_days` | days | High end of that range. |
| `scan_marginal_low_days` | days | Low end of the Bayesian (marginal) range; with Gaussian uncertainties it is pulled toward the nominal initial condition. |
| `scan_marginal_high_days` | days | High end of the Bayesian range. |
| `scan_best` |  | Best values of the scanned parameters (step1_um, a_fo, ...); the nominal values stay in the parameter columns. |
| `scan_candidates` |  | Initial conditions fitted (one model run each). |
| `scan_ic_share` |  | Sampled: fraction of the variance of time that comes from the choice of initial condition (the rest is the fit at a fixed initial condition). |
| `scan_sensitivity` |  | Sampled: change of the expected time per unit of each parameter (days per um for steps, days per unit Fo fraction), from the posterior samples. |
| `scan_ess` |  | Sampled: effective sample size of the importance samples (out of 1024); below 256, auto uses nested sampling. |
| `scan_log_evidence` |  | Sampled: log evidence; compare initial-condition models (one step or two) fitted to the same profile with the same grid. |
| `scan_near_step_daic` |  | near_step_daic at the best-fit initial condition. Fitting the initial condition usually clears a near-step warning; below -6 the misfit is still concentrated near the step, so quote the 2-sigma range. |
| `mc_method` |  | Monte Carlo: rescale (t = t_fit D_fit / D_draw, exact for a fixed rim) or refit (growth stages: every draw refitted). |
| `mc_draws` |  | Monte Carlo draws made. |
| `mc_median_days` | days | Monte Carlo median time: the fit (or the initial-condition posterior), the conditions and the law together. |
| `mc_low_days` | days | Monte Carlo 16th percentile. |
| `mc_high_days` | days | Monte Carlo 84th percentile. |
| `mc_p05_days` | days | Monte Carlo 5th percentile. |
| `mc_p95_days` | days | Monte Carlo 95th percentile. |
| `mc_sd_log10` | log10 | Monte Carlo: sd of log10 time from each source drawn alone (fit, T, P, fO2, law, studies), to see which matters most. |
| `scan_dropped` |  | Grid points left out as invalid (steps out of order or domain, Fo outside 0-1). |
| `grid_resolution` |  | Diffusion length sqrt(D t) at the fit over the grid spacing: for 1% accuracy in the fitted time about 5 (implicit) or 240 (central, dipra). |
| `grid_error_pct` | % | Estimated worst-case error of the fitted time from the grid spacing (implicit: 0.2 x (dx / l)²; central, dipra: 2.4 x dx / l; l the diffusion length; measured on 35 real profiles). |
| `grid_stability` |  | D dt / dx² of the final fit's time step (central, dipra: above 0.5 each step is split into stable steps; empty for implicit). |

Parameter tables in the two older layouts of this project, the round robin
spreadsheets and the tables of Towbin and Plank (2026), are read too.

## How the fit works

- The model is finite differences (implicit by default, explicit on request; see
  `scheme` below) for Fe-Mg interdiffusion with a diffusivity that depends on Fo, oxygen fugacity, temperature and pressure, and
  on the angles between the profile and the crystal axes. The starting profile is
  segments of constant Fo between steps (`segment_fo`, labelled a, b, c, ... or by
  `segment_names`). Edge 1 is held at the first segment's Fo (or `edge1_fo`); the far
  end at the last segment's (the core), or, with `centre_um`, the crystal is mirrored at
  its centre so no Fo crosses it and the core is free to change.
- Every time step is compared with the measured points. The best fit is the time
  with the smallest chi-squared. The 1σ range is where chi-squared rises by about 1
  (a little more for profiles of few points, which corrects for the noise being
  estimated from them), with the errors on the points modelled as white noise plus
  a smooth correlated misfit fitted to the residuals. The model stops once the fit
  is well behind it rather than running to `max_time_days`, which gives the same
  result faster. The `legacy` option keeps the earlier rule, which compared the
  model with a kriged interpolation of the data.
- The grid can be left to FORSTER: with `dx_um` and `dt_s` empty, a quick coarse fit
  gives the diffusion length l = sqrt(D t). With the default implicit scheme the final
  fit uses dx = l / 8 and dt = 1/400 of the fitted time (within 0.3% of a converged
  value on all 35 real profiles), scans dx = l / 4 and 1/80. With `central` or `dipra`:
  dx = l / 240 for 1% and l / 48 for scans, with the same dt; within each time step
  they take as many stable steps (D dt / dx² = 0.4) as they need and keep only the
  rows at whole time steps. A grid you give is kept and checked: coarse or wastefully
  fine ones get a warning, and an explicit dt above the stability limit is split into
  stable steps (refused only with growth stages). `grid_resolution`, `grid_error_pct` and
  `grid_stability` in summary.csv say how well each fit was resolved.
  [docs/references/grid-choice.md](docs/references/grid-choice.md) gives the
  measurements behind these numbers.
- The range does not include uncertainty in the initial condition. From Python,
  `forster.scan.scan_initial_conditions` fits a set of candidates (for example the
  step at a range of positions, `forster.scan.step_offsets`) and gives the range of
  time over all of them.
  [docs/references/likelihood-method.md](docs/references/likelihood-method.md)
  explains the method and how it was validated, and
  [docs/references/fit-uncertainty.md](docs/references/fit-uncertainty.md) how it
  differs from earlier versions.
- The range does not include the uncertainty of the diffusion equation itself either,
  unless the equation carries one (a covariance of its fitted parameters).
  `femg_refit2026` does: our refit of the verified
  Fe-Mg data (four studies, every value stated in the sources' tables or text, plus
  Holzapfel et al.'s Au-capsule runs with their unbuffered fO2 taken at Ni-NiO ± 2 log;
  132 points; [docs/references/data-to-source.md](docs/references/data-to-source.md)) in
  the Dohmen and Chakraborty (2007) form with [100] and [010] six times slower than
  [001] and the activation volume fitted. The fit weights each value by its published
  error and the errors in its temperature, fO2, composition and pressure, and lets each
  laboratory and each experiment sit off the average by its own offset
  ([analyses/femg_refit](analyses/femg_refit/README.md)). The equation, its fitted values and
  their correlations: [docs/references/femg-refit2026.md](docs/references/femg-refit2026.md). An Fo
  profile depends on D t, so an error in log D is the same error in log time, and
  no extra model runs are needed. Fits with it report three ranges:
  `low_days`/`high_days` (the profile alone; use these to compare crystals fitted
  with the same equation, whose error is common to all of them), `law_low_days`/
  `law_high_days` (plus the uncertainty of the fitted equation, about ±0.13 log in D at
  1200 °C), and `all_low_days`/`all_high_days` (plus the spread between laboratories,
  0.07 log; for an absolute time, if the crystal may sit off the laboratories' average as one
  more experiment would). From Python: `result.time_range_days("fit" | "law" |
  "law_and_studies")`. Not available for fits with growth stages, a rim value that
  changes in time or element fits, where the time no longer scales as 1/D; the
  initial-condition scan's range is not widened yet. What the range assumes, where it
  was measured to hold and where not:
  [docs/references/law-uncertainty-limits.md](docs/references/law-uncertainty-limits.md).
- The default diffusivity, `dohmen_chakraborty2007`, is the Fe-Mg equation DIPRA
  uses (Dohmen and Chakraborty 2007; Girona and Costa 2013, Table 1). The presets
  `mutch2019_tamed` and `mutch2019_global` are the regressions of Mutch et al.
  (2019) with their covariance at full precision (Mutch et al. 2021, DFENS Data Set
  S6), so they also give the `law_` range; `mutch2019_ni` and `mutch2019_mn` are the
  same paper's Ni and Mn regressions (for Ni, [001] 10.7x faster as in DFENS). DIPRA's Ni, Mn and Ca equations are listed for reference but cannot be
  used to fit Fo profiles. For Ni, `petry2004_ni` is Petry et al. (2004)'s own
  constant-fO2 law; DIPRA's `girona_costa2013_ni` (the default for Ni fits) pairs
  its pre-factor with the activation energy of a different fit, which makes Ni 0.14
  log units slower at 1200 °C. Choose it with `fit_element(..., diffusivity="petry2004_ni")`.
  [docs/references/diffusivity-literature.md](docs/references/diffusivity-literature.md)
  compares them with newer experiments.
- The default scheme (since 2026-10-13) is `implicit`: the rim and core values are
  held on the end nodes of the grid, the starting steps are averaged over each node's
  cell instead of moved to the nearest node, and time steps are TR-BDF2, stable at any
  step size. Its fitted time is second order in the grid spacing: on all 35 real
  profiles the chosen grid gave times within 0.3% of a converged value, in a median
  0.13 s per fit.
- `scheme = central` is the explicit scheme used before (the default until
  2026-10-13; still the default with `uncertainty = legacy`, so old results
  reproduce exactly). It holds the rim one node outside the profile and moves steps
  to nodes, which makes its fitted time first order in dx: the grid then chosen
  (dx = l / 65) was off by a median 1.1% and up to 3.6%, mostly too long. Refitting
  old parameter tables without a `scheme` column now uses `implicit`: for the round
  robin and paper tables (their own grids) times move by a median 2.7% and up to
  8.1%, mostly shorter, because on those grids `central` itself was a median 2.6%
  (up to 8.7%) off the converged value and `implicit` 0.05%.
- `scheme = dipra` swaps in the finite-difference scheme of DIPRA (Girona and Costa
  2013, G³ 14, 422-431), explicit like `central`; fitting a profile with both is a
  check on the grid spacing.
- From Python, `fit_profile(..., rim_fo=...)` takes a rim value that changes in time
  (`forster.boundary.RimPath`, or any function of time; implicit only), and growth
  stages start at their exact times.

The diffusivity law matters as much as the fit: `analyses/femg_refit/` refits the
Fe-Mg law to the published experiments, and at the round robin and paper fO2 the
refits differ from the default by up to 0.85 log units in D, mostly through the fO2
exponent (`docs/plans/2026-10-13-femg-refit.md`).

## Development

```
git clone https://github.com/whtowbin/Olivine_Diffusion_FeMg.git
cd Olivine_Diffusion_FeMg
uv sync
uv run pytest -n 8
```

Part of the test suite (the differential replay against the original scripts, and
several fits) uses the olivine diffusion round robin profiles. Those are other
laboratories' samples, kept in a private repository until the round robin is published
and checked out as the git submodule `Diffusion Round Robin`. With access, run
`git submodule update --init` first; without it those tests skip and the rest run.
CI (`.github/workflows/tests.yml`) reads the submodule with a read-only deploy key.

## Citing

Towbin, W. H., and Plank, T. (2026). Journal of Petrology 67(9), egag077.

## License

MIT.
