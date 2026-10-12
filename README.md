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
forster template my_fits/      # writes profiles.csv, parameters.csv and COLUMNS.md
forster fit my_fits/profiles.csv my_fits/parameters.csv --out my_fits/results
```

The template already holds a worked example, one published profile (AZ18
WHT01_Ol_63_prof_1 from Towbin and Plank 2026), so the second command works straight
away and shows what the output looks like. Then replace the example rows with your
own: one row per analysis point in `profiles.csv`, one row per model run in
`parameters.csv`. The columns are listed below and in `COLUMNS.md`. Excel files work
too (`--profiles-sheet` and `--parameters-sheet` pick the sheet).

For each run, `forster fit` writes into the output folder:

- `<run>_fit.png`: the measured points, the initial profile, the best fit and the 1σ range;
- `<run>_curves.csv`: the model profile at time zero, at the best fit, and at the two ends of the 1σ range;
- `<run>_points.csv`: the measured points that were fitted, the best fit at each one and the residual;
- `<run>_kriged.csv`: the kriged (interpolated) profile;
- `summary.csv`: one row per run with the time (`time_days`), its 1σ range (`low_days`, `high_days`), the fit statistics and every input.

`summary.csv` uses the parameter-table column names, so it can be passed back to
`forster fit` as a parameter table to rerun the same fits. A run that fails (a
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
take the conditions and angles directly. `forster.summary_row(result)` and
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
| `uncertainty` |  | optional | correlated (default) or legacy (the round robin rule; reproduces old results). |
| `notes` |  | optional | Free text for your own records. |

*one form*: give fO2 either as `fO2_Pa` or as `fO2_buffer` with `fO2_delta_log`, and
the orientation either as `alpha_deg`, `beta_deg`, `gamma_deg` or as the four EBSD
columns. Leave the other form empty.

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
| `activation_energy_J_mol` | J/mol | Fe-Mg activation energy used (not read back; set it in Python with RunDefaults). |
| `warnings` |  | Warnings from the fit, such as a range that reaches the last time step. |
| `forster_version` |  | FORSTER version that made the fit. |

Parameter tables in the two older layouts of this project, the round robin
spreadsheets and the tables of Towbin and Plank (2026), are read too.

## How the fit works

- The model is explicit finite differences for Fe-Mg interdiffusion with a
  diffusivity that depends on Fo, oxygen fugacity, temperature and pressure, and
  on the angles between the profile and the crystal axes. The ends of the model
  domain are held at `edge_fo` and `core_fo`.
- Every time step is compared with the measured points. The best fit is the time
  with the smallest chi-squared. The 1σ range is where chi-squared rises by 1,
  with the errors on the points modelled as white noise plus a smooth correlated
  misfit fitted to the residuals. The `legacy` option keeps the earlier rule, which
  compared the model with a kriged interpolation of the data.
  [docs/references/fit-uncertainty.md](docs/references/fit-uncertainty.md) explains
  the difference.
- `scheme = dipra` swaps in the finite-difference scheme of DIPRA (Girona and Costa
  2013, G³ 14, 422-431). Both schemes solve the same equation; fitting a profile with
  both is a check on the grid spacing.

## Citing

Towbin, W. H., and Plank, T. (2026). Journal of Petrology 67(9), egag077.

## License

MIT.
