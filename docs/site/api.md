# Python API

Everything the command line and the desktop app do can also be done from Python.
`import forster` gives you the names below. Each heading shows the module that defines
the name, but the short name works too: `forster.fit_profile` and
`forster.model.fit_profile` are the same function. This page is generated from the
docstrings in the code.

Units follow the tables: distances in µm, Fo as a fraction (0-1), temperature in °C,
pressure and fO2 in Pa, times in seconds inside the model and in days in results
(`time_days`). Angles are in degrees.

!!! tip "Where to start"
    - From tables: [`read_profiles`](#forster.io.read_profiles),
      [`read_runs`](#forster.io.read_runs), then [`fit_run`](#forster.io.fit_run)
      (the [Quick start](quickstart.md#the-same-from-python) shows it).
    - From arrays: [`Profile`](#forster.model.Profile),
      [`Conditions`](#forster.model.Conditions),
      [`Orientation`](#forster.model.Orientation),
      [`InitialCondition`](#forster.model.InitialCondition),
      [`choose_grid`](#forster.grid.choose_grid), then
      [`fit_profile`](#forster.model.fit_profile).

## Reading tables and fitting runs

::: forster.io.read_profiles

::: forster.io.read_runs

::: forster.io.read_diffusivities

::: forster.io.fit_run

::: forster.io.summary_row

::: forster.io.fit_tables

::: forster.template.write_template

## The model's inputs

::: forster.model.Profile

::: forster.model.Conditions

::: forster.model.Orientation

::: forster.model.InitialCondition

::: forster.model.Grid

::: forster.grid.choose_grid

::: forster.grid.GridRequest

::: forster.boundary.Stage

## Fitting a profile

::: forster.model.fit_profile

::: forster.model.FitResult

::: forster.uncertainty.NoiseModel

## Fitting the starting profile

::: forster.scan.Uncertainty

::: forster.sampling.fit_initial_condition

::: forster.sampling.ICPosterior

## Monte Carlo over the conditions and the law

::: forster.rescale.ConditionUncertainty

::: forster.rescale.monte_carlo

::: forster.rescale.TimeDistribution

## Joint fit of one crystal

::: forster.joint.fit_crystal

::: forster.joint.CrystalFit

## Ni, Mn and Ca

::: forster.elements.ElementProfile

::: forster.elements.fit_element

::: forster.elements.fit_joint

## Diffusivity equations

::: forster.diffusivity.ArrheniusDiffusivity

## Plotting

::: forster.plotting.plot_fit

::: forster.plotting.plot_law_comparison

::: forster.plotting.plot_law_times

::: forster.plotting.plot_time_distributions

## Importing spreadsheets: `forster.imports`

`forster import` and the app's import dialog use these.

::: forster.imports.guess

::: forster.imports.ColumnMap

::: forster.imports.ImportTemplate

::: forster.imports.apply

::: forster.imports.read_with_template

::: forster.imports.ImportResult

## Project folders: `forster.project.Project`

The project folder used by the desktop app (see [Advanced](advanced.md#project-folders)).

::: forster.project.Project
    options:
      members:
        - open
        - save
        - export_zip
        - run_names
        - run
        - add_run
        - update_run
        - copy_settings
        - add_profiles
        - points
        - set_masked
