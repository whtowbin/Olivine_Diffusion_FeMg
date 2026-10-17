# The desktop app

The app does everything `forster fit` does, with the profile on screen: you draw the
starting profile with the mouse, see the fit update as you go, and look at the
uncertainty from the starting profile, the conditions and the diffusion law in the
same window. It needs the `gui` extra ([Installation](installation.md)).

```bash
forster gui my_project/
```

`my_project/` is a **project folder**. An empty or new folder starts from the worked
example; a folder that already holds FORSTER tables (`profiles.csv`,
`parameters.csv`) opens them. The app saves to the same tables the command line
reads, so `forster fit my_project/profiles.csv my_project/parameters.csv` reruns a
saved project unchanged.

!!! note "About the screenshots"
    The screenshots come from the app's test runs. "OL7 prof 1" and "OL7 prof 2" are
    synthetic test profiles, not real crystals. Three screenshots (the Segments tab,
    a refused edit and the starting-profile fit) were taken on the first build, which
    had fewer tabs; the tabs they show work the same way now.

## The window

![The main window with a fitted run](images/app-monte-carlo.png)

The window has three parts:

- **Left: the runs.** One run is one fit of one profile with one set of inputs (a row
  of `parameters.csv`). A tick means it has been fitted. **Add** makes a new run on a
  profile, starting from the selected run's settings; **Duplicate**, **Rename** and
  **Delete** do what they say; **Copy settings to other runs...** is described
  [below](#copying-settings-between-runs).
- **Centre: the views** of the selected run: **Profile**, **Time distribution**,
  **Law comparison** and **Results**.
- **Right: the run's settings**, in tabs: **Segments**, **Data**, **Conditions**,
  **Orientation**, **Equation**, **Uncertainties** and **Advanced**.

Across the top: **New from example**, **Open**, **Save**, **Save as**, **Import
profiles**, **Copy settings**, **Fit all** (fit every run) and **Export**. Every field
has a tooltip giving its unit and the table column it is saved in.

## The Profile view and the Segments tab

![The Segments tab](images/app-segments.png)

The Profile view shows the measured points, the best fit with its 1σ and 2σ bands
(the model profiles at the two ends of each range), and the residuals (measured minus
model) underneath. The time and both ranges are in the title.

The **starting profile** is drawn as orange lines. It is made of **segments**: stretches
of constant Fo, labelled a, b, c, ... from **edge 1** (the crystal edge where the
profile starts) inward. Segments meet at **steps** (dashed lines). You can:

- drag a step sideways, or a segment's Fo line up and down;
- drag the grey lines at the two ends of the **model domain** (the part of the
  profile that is modelled; points outside it are drawn in grey and not fitted);
- type the same values into the **Segments** tab: each segment's start, end and Fo in
  mol%, and an uncertainty on its Fo (a standard deviation, *sd*, or a ± *range*; see
  [Uncertainty beyond the fit](uncertainty.md)).

The run is refitted when you let go (the example refits in about 35 ms). **Split
segment** and **Remove segment** add or remove a step. Below the table, **Edge 1 Fo**
holds edge 1 at a different Fo from segment a's during diffusion (for example a rim
in contact with a melt of different composition), and **Crystal centre** models a
whole crystal mirrored at its centre instead of a half profile. [Setting up a
run](setup.md#the-starting-profile) explains both.

FORSTER's warnings (a profile that ends too close to the step, misfit concentrated
near the step, ...) appear under the plot. [Reading the results](results.md#warnings)
says what each one means and what to do.

### Values are checked as you type

![A refused edit](images/app-refused-edit.png)

Every edit is checked by writing the run as a table row and reading it back, so the
app accepts exactly what `forster fit` accepts. A refused value turns red and the
reason is shown under the table (here, an Fo of 120 mol%).

## The Data tab: masking points and moving the rim

![The Data tab with two masked points](images/app-data-masked.png)

The **Data** tab lists the profile's points. Untick a point (or shift-click it on the
plot) to **mask** it: it stays in the project, drawn as a grey cross, but is left out
of every fit of this profile. Masked points are saved in `profiles.csv` with
`Marked_bad` = `bad`, which `forster fit` also skips. Mask a point only for a known
fault (a crack, an inclusion, a poor total, beam overlap with the rim), never to
narrow the range ([Preparing data](data.md#collecting-profiles)).

**Move the distances so the rim is at 0** shifts the distances, or flips them when the
rim is at the far end of the traverse. The runs on that profile move with it (domain
and steps). **Import profiles...** opens the import dialog.

## Importing profiles

![The import dialog](images/app-import-dialog.png)

Choose a CSV or Excel file, then say what each column holds: the profile name (none:
the whole file is one profile), the distance (µm or mm), Fo (a fraction or percent),
oxides and elements (wt% or ppm), the analytical total, stage coordinates, exclusion
columns, or nothing. The dialog proposes a mapping from the headers and values (hover
over a column to see why) and keeps **OK** disabled until it is complete. Without an
Fo column, Fo is computed from MgO and FeO in moles. Points whose total lies outside
the window (98.5-101.5 wt% by default) are kept and listed as flagged. The dialog can
flip and shift the distances so the rim is at 0, add a run on each new profile, and
save or load the mapping as an **import template**. It uses the same mapping as
`forster import`; [Preparing data](data.md#importing-other-spreadsheet-layouts)
explains it in full.

## Conditions, Orientation and Equation

- **Conditions**: temperature (°C), pressure, and oxygen fugacity either relative to
  a buffer (QFM or NNO, in log units) or in Pa.
- **Orientation**: the angles between the profile and the crystal axes [100], [010]
  and [001], or EBSD Euler angles with the profile's direction on the map.
- **Equation**: the Fe-Mg diffusion law: one of the four built-in choices, or one of
  the project's own equations from `diffusivities.csv`. The tooltip gives each one's
  source.

[Setting up a run](setup.md) explains what each choice changes.

## The Uncertainties tab

### Fitting the starting profile

![Fitting the starting profile](images/app-starting-profile-fit.png)

Give a ± to the step positions here and to the segments' Fo in the Segments tab, then
press **Fit starting profile**: FORSTER fits the step positions and segment Fo
together with the time, and reports the best fit, the 68% range including the
starting profile's uncertainty, the best starting profile, and how much of the spread
in time comes from the starting profile. **Use the best starting profile** copies the
fitted values into the run. *Method* chooses the sampler (`auto` is right almost
always). Progress is shown and the fit can be cancelled. In the screenshot a run with
two steps, the first known to ±2 µm, gives 4.17 days (68% range 3.07-7.72 days), with
85% of the spread coming from the starting profile; the warning below the plot says
the profile ends only 2.2 diffusion lengths past the second step.

### Monte Carlo over the conditions and the equation

![The Monte Carlo panel](images/app-monte-carlo.png)

Below it, the **Monte Carlo** draws temperature, pressure, fO2 and the diffusion law's
parameters and turns each draw into a time. Give the 1σ of T (°C), P (MPa) and fO2
(log units); choose whether fO2 stays relative to a buffer as T changes, whether to
draw the law's parameters (from its published covariance), and whether to add the
spread between laboratories. The result line gives the median time, the 68% and 90%
ranges, and how much each source contributes. [Uncertainty beyond the
fit](uncertainty.md#the-monte-carlo-over-t-p-fo2-and-the-law) explains the numbers.

## Time distribution

![The Time distribution view](images/app-time-distribution.png)

The distribution of time from the fit alone, with the starting profile's uncertainty
(after **Fit starting profile**), and with the conditions and the law (after a Monte
Carlo), on a log time axis. This is the figure `forster fit` writes as `_time.png`.
*Show* picks which distributions to draw.

## Law comparison

![The Law comparison view](images/app-law-comparison.png)

The run refitted under each of the four built-in Fe-Mg diffusion laws, as `forster
fit --compare-laws` draws it. The profiles look identical because the fit fixes
D × t; the times differ because the laws give different D. Thick bars: the 1σ range
from the profile alone; thin bars: with the law's own calibration uncertainty.

## Advanced

![The Advanced tab](images/app-advanced.png)

- **Scheme and grid**: the numerical scheme (implicit by default) and the grid's
  spacing (dx), time step (dt) and longest modelled time. Leave dx and dt blank to
  let FORSTER choose them; the line underneath says what the last fit used and how
  well it resolved the profile (here: the diffusion length is 8 grid spacings, so
  the time is within about 0.3% of a converged value).
- **Growth stages**: a table of stages, each with its start time, the new rim
  position, the new zone's Fo and the rim's Fo from then on ([Advanced](advanced.md#growth-stages)).

## Results

![The Results view](images/app-results.png)

Every run's time, 1σ and 2σ ranges, Monte Carlo median and 68% range, status and
number of warnings. **Export** writes the same `summary.csv` as `forster fit` for
the fitted runs.

## Copying settings between runs

![Copy settings](images/app-copy-settings.png)

Choose which groups of settings to copy from the selected run (conditions,
orientation, diffusion equation, starting-profile uncertainties and sampler, Monte
Carlo settings, grid and scheme) and to which runs. Each target is checked as the
table reader would check it; if one cannot take the settings, none is changed.

## Dark mode

![Dark mode](images/app-dark.png)

The app follows the system's light or dark setting; the plots follow it too.

![The Time distribution view in dark mode](images/app-dark-time-distribution.png)

## Saving, caching and exporting

- **Save** writes the tables (`profiles.csv`, `parameters.csv`, `diffusivities.csv`)
  and `project.json`, which holds what only the app needs (each run's view: tabs and
  zoom).
- Fits, starting-profile fits and Monte Carlos are cached in the folder
  (`.forster/cache`), so a reopened project shows its results at once.
- The project is autosaved every minute (`.forster/autosave`); when you open a
  project with unsaved changes from an earlier session, the app offers to restore
  them.
- **Export** writes the plot or view shown as PNG or PDF, the fit figure as `forster
  fit` draws it, `summary.csv`, or the whole project as a zip to share.
