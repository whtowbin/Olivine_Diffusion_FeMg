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

For each run, `forster fit` writes into the output folder:

- `<run>_fit.png`: the measured points, the initial profile, the best fit and the 1σ range;
- `<run>_curves.csv`: the model profile at time zero, at the best fit, and at the two ends of the 1σ range;
- `<run>_points.csv`: the measured points that were fitted, the best fit at each one and the residual;
- `<run>_kriged.csv`: the kriged (interpolated) profile;
- `summary.csv`: one row per run with the time (`time_days`), its 1σ range (`low_days`, `high_days`), the fit statistics and every input;
- `diffusivities.csv`: the diffusivity equations the runs used.

**Uncertain initial conditions.** Add an uncertainty to any step position or
concentration in `parameters.csv` and `forster fit` also fits a grid of initial
conditions around the nominal one, and reports the range of time including that
uncertainty. For example, a step known to ±2 µm (1σ) and an edge Fo between 0.817 and
0.823:

```
steps_um,edge_fo,steps_sd_um,edge_fo_range
41,0.82,2,0.003
```

The summary then has `scan_low_days` and `scan_high_days` (the range including the
initial condition), `scan_best` (the best step and edge), and `<run>_scan.csv` lists
every candidate with its probability. `_sd` columns are Gaussian 1σ, `_range`
columns are uniform ± half-widths. The number of model runs (the product of
`scan_points` over the uncertain parameters) is printed before the scan.

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
    Conditions, Grid, InitialCondition, Orientation, Profile, fit_profile,
)

result = fit_profile(
    Profile(distance_um, fo, name="my profile"),        # Fo as a fraction
    Conditions.from_buffer(T_C=1250, P_Pa=5e8, delta_log=0.3, buffer="QFM"),
    Orientation.from_euler(77.0, 61.2, 334.2, profile_angle_deg=-80),
    InitialCondition(domain_um=(2.5, 72), edge_fo=0.9146, core_fo=0.8985, steps_um=(24,)),
    Grid(dx_um=0.5, dt_s=60, max_time_days=40),
)
```

`Conditions(T_C, P_Pa, fO2_Pa)` and `Orientation(alpha_deg, beta_deg, gamma_deg)`
take the conditions and angles directly. `fit_profile(..., diffusivity=...)` takes
a preset name (`forster.DIFFUSIVITY_PRESETS`), a row of a table read with
`forster.read_diffusivities`, or an `ArrheniusDiffusivity` built in Python. `forster.summary_row(result)` and
`forster.fit_tables(result)` give the tables that `forster fit` writes.

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
| `domain_start_um` | µm | required | Start of the model domain; Fo is held at edge_fo here. |
| `domain_end_um` | µm | required | End of the model domain; Fo is held at core_fo here. |
| `steps_um` | µm | required | Step position(s) of the initial profile: one number, or two in quotes ("10, 33"). |
| `edge_fo` | fraction | required | Initial Fo before the first step (edge side). |
| `core_fo` | fraction | required | Initial Fo after the last step (core side). |
| `step_fo` | fraction | optional | Initial Fo of the plateau between two steps; required with two steps. |
| `dx_um` | µm | required | Grid spacing. Keep D dt / dx² below 0.5: the explicit scheme is unstable above it, and this is not checked. |
| `dt_s` | s | required | Time step. |
| `max_time_days` | days | optional | Longest time modelled (default 200). |
| `scheme` |  | optional | central (default) or dipra (Girona and Costa 2013). |
| `diffusivity` |  | optional | Fe-Mg diffusivity equation: a preset (dohmen_chakraborty2007, the default and DIPRA's, or mutch2019_tamed, mutch2019_global) or a name from diffusivities.csv (then run forster fit with --diffusivities). |
| `uncertainty` |  | optional | correlated (default) or legacy (the round robin rule; reproduces old results). |
| `steps_sd_um` | µm | optional | Uncertainty (1 sigma, Gaussian) on the step position(s): one value, or one per step ("2, 5"); 0 keeps a step fixed. Or give steps_range_um. |
| `steps_range_um` | µm | optional | Uncertainty (± half-width, uniform) on the step position(s), as steps_sd_um. |
| `edge_fo_sd` | fraction | optional | 1-sigma uncertainty on edge_fo (Gaussian). |
| `edge_fo_range` | fraction | optional | ± half-width on edge_fo (uniform). |
| `core_fo_sd` | fraction | optional | 1-sigma uncertainty on core_fo (Gaussian). |
| `core_fo_range` | fraction | optional | ± half-width on core_fo (uniform). |
| `step_fo_sd` | fraction | optional | 1-sigma uncertainty on step_fo (Gaussian). |
| `step_fo_range` | fraction | optional | ± half-width on step_fo (uniform). |
| `scan_points` |  | optional | Grid points per uncertain parameter (default 13 for one, 9 for two, 5 for more). |
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
| `time_days` | days | Best-fit time. |
| `low_days` | days | Low end of the 1-sigma range. |
| `high_days` | days | High end of the 1-sigma range. |
| `reduced_chi2` |  | Chi-squared at the best fit over (points fitted - 1). Diagnostic only: about 1 by construction for the correlated method, whose noise model is fitted to the residuals; for legacy it is over the kriged grid points, so the two are not comparable. Judge correlated fits by the three noise columns. |
| `points_fitted` |  | Measured points inside the model domain (legacy: kriged grid points). |
| `white_sigma_fo` | fraction | Fitted scatter of the points (correlated method). Much larger than your analytical precision: outliers, or the precision is underestimated. |
| `correlated_sigma_fo` | fraction | Fitted smooth misfit between model and data. Large next to white_sigma_fo: the model misses the shape of the profile; check the initial condition. |
| `correlation_length_um` | µm | Length scale of that misfit; near half the profile length, the misfit is profile-wide. |
| `warnings` |  | Warnings from the fit, such as a range that reaches the last time step. |
| `forster_version` |  | FORSTER version that made the fit. |
| `scan_best_days` | days | With uncertainties: best-fit time over all scanned initial conditions. |
| `scan_low_days` | days | Low end of the 1-sigma range including the initial-condition uncertainty (profile likelihood; use this one). |
| `scan_high_days` | days | High end of that range. |
| `scan_marginal_low_days` | days | Low end of the Bayesian (marginal) range; with Gaussian uncertainties it is pulled toward the nominal initial condition. |
| `scan_marginal_high_days` | days | High end of the Bayesian range. |
| `scan_best` |  | Best values of the scanned parameters (step1_um, edge_fo, ...); the nominal values stay in the parameter columns. |
| `scan_candidates` |  | Initial conditions fitted (one model run each). |
| `scan_dropped` |  | Grid points left out as invalid (steps out of order or domain, Fo outside 0-1). |
| `scan_dt_s` | s | Time step of the scan's runs: dt raised until D dt / dx² = 0.4 (never dx). |

Parameter tables in the two older layouts of this project, the round robin
spreadsheets and the tables of Towbin and Plank (2026), are read too.

## How the fit works

- The model is explicit finite differences for Fe-Mg interdiffusion with a
  diffusivity that depends on Fo, oxygen fugacity, temperature and pressure, and
  on the angles between the profile and the crystal axes. The ends of the model
  domain are held at `edge_fo` and `core_fo`.
- Every time step is compared with the measured points. The best fit is the time
  with the smallest chi-squared. The 1σ range is where chi-squared rises by about 1
  (a little more for profiles of few points, which corrects for the noise being
  estimated from them), with the errors on the points modelled as white noise plus
  a smooth correlated misfit fitted to the residuals. The model stops once the fit
  is well behind it rather than running to `max_time_days`, which gives the same
  result faster. The `legacy` option keeps the earlier rule, which compared the
  model with a kriged interpolation of the data.
- The range does not include uncertainty in the initial condition. From Python,
  `forster.scan.scan_initial_conditions` fits a set of candidates (for example the
  step at a range of positions, `forster.scan.step_offsets`) and gives the range of
  time over all of them.
  [docs/references/likelihood-method.md](docs/references/likelihood-method.md)
  explains the method and how it was validated, and
  [docs/references/fit-uncertainty.md](docs/references/fit-uncertainty.md) how it
  differs from earlier versions.
- The default diffusivity, `dohmen_chakraborty2007`, is the Fe-Mg equation DIPRA
  uses (Dohmen and Chakraborty 2007; Girona and Costa 2013, Table 1). The presets
  `mutch2019_tamed` and `mutch2019_global` are the regressions of Mutch et al.
  (2019); DIPRA's Ni, Mn and Ca equations are listed for reference but cannot be
  used to fit Fo profiles.
  [docs/references/diffusivity-literature.md](docs/references/diffusivity-literature.md)
  compares them with newer experiments.
- `scheme = dipra` swaps in the finite-difference scheme of DIPRA (Girona and Costa
  2013, G³ 14, 422-431). Both schemes solve the same equation; fitting a profile with
  both is a check on the grid spacing.

## Citing

Towbin, W. H., and Plank, T. (2026). Journal of Petrology 67(9), egag077.

## License

MIT.
