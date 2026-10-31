# Changelog

## 0.1.0 (unreleased)

FORSTER fits Fe-Mg diffusion times to forsterite profiles in zoned olivine, from a command line, a desktop app or Python. This first release grew out of the scripts behind Towbin and Plank (2026, J. Petrology 67(9), egag077). Documentation: https://forster.readthedocs.io. The development log, [docs/development-log.md](docs/development-log.md), holds the measurements behind these changes.

### If you used the earlier scripts

Results from FORSTER differ from those of the paper-era scripts for four reasons. None changes the conclusions of Towbin and Plank (2026); the first is a correction, the others are changes of method.

1.  **A coding error in the diffusion model is fixed (2026-10-11).** The model solves dC/dt = D d²C/dx² + (dD/dx)(dC/dx). The second term, which matters only where the diffusivity changes across a step, was twice too large: two central differences were divided by 2Δx² instead of 4Δx². Fitted times were biased low where D varies strongly across the step. For the profiles of Towbin and Plank (2026), run with the published script, the same diffusion equation and the same random draws:

    -   melt-vein medians move by 2-5% (samples in Towbin and Plank 2026: WHT01 17.9 to 17.6 days, GCB1 12.8 to 13.5 days, WHT06 5.6 to 5.4 days);
    -   the exterior contact-overgrowth median moves from 5.9 to 7.5 days (+28%), because their Fo contrast is the largest (7-10 units);
    -   the xenocryst median moves by +9% (0.79 to 0.86 days).

    Every shift lies well inside the published 5th-95th percentile ranges, which are dominated by the uncertainty of the pressure-temperature path (e.g. WHT01 melt veins 5.5-44.9 days before, 5.4-45.3 after; contact overgrowths 2.3-15.2 before, 3.2-17.9 after). Per-profile numbers: `docs/plans/2026-10-11-engine-fix/`.

2.  **The range on a fitted time is computed differently.** The paper-era scripts compared the model with a kriged profile on the model grid and drew the range where grid chi-squared rose by sqrt(2(N-1)) (until March 2026; the ranges of individual fits in Towbin and Plank 2026). FORSTER compares the model with the measured points, allows for smooth misfit between model and data (a correlated noise model), and draws 1σ and 2σ ranges from it; tests on profiles of known age give the stated coverage. Individual-fit ranges therefore change; the paper's reported ranges, from the Monte Carlo over the pressure-temperature path, are dominated by that path and are not affected in substance. The old rule (from March 2026: reduced chi-squared within 1 of its minimum) is kept as `uncertainty="legacy"` to reproduce earlier numbers. Details: `docs/references/fit-uncertainty.md`.

3.  **How time is counted and computed.** The model's profile at row k of the time grid is reported as time k × Δt, where row 0 is the starting profile; the paper-era code reported (k + 1) × Δt, one time step later. This matters only when Δt is a sizable fraction of the fitted time (short times on a coarse grid). The default solver is now an implicit scheme (TR-BDF2), which stays stable at any time step, so the time step no longer has to obey the explicit schemes' stability limit (D Δt / Δx² ≤ 0.5). FORSTER chooses the grid spacing and time step automatically, aiming for a grid error of about 1% in the fitted time. The paper-era explicit schemes remain available (`scheme` = `central` or `dipra`).

4.  **New choices.**

    -   Four named Fe-Mg diffusion equations, set in the `diffusivity` column. They share one form but differ in the overall level of D and in how D depends on fO2 and temperature, which can change a fitted time by a factor of two (for the worked example: 5.14, 3.05, 2.45 and 3.57 days):
        -   `dipra` (the default, as before): Dohmen and Chakraborty (2007) with their erratum, as tabulated in DIPRA (Girona and Costa 2013). Published without a covariance, so it has no calibration uncertainty.
        -   `refit`: FORSTER's 2026 refit of the same form to 132 diffusion coefficients from four experimental studies, with an offset per laboratory. Carries a covariance and the spread between laboratories (`docs/references/femg-refit2026.md`).
        -   `dfens_tamed` and `dfens_global`: the regressions of Mutch et al. (2019) for the transition-metal-extrinsic diffusion mechanism (TaMED, which applies at natural fO2) and for all their compiled data, with the covariance of Mutch et al. (2021, DFENS).

        Full references: `forster cite` and the documentation's citing page.
    -   Fitting the starting profile (step positions and segment Fo) with its uncertainty; a Monte Carlo over temperature, pressure, oxygen fugacity and the diffusion equation; growth stages.
    -   Parameter-table columns renamed for starting profiles of any number of segments (`segment_fo`, `edge1_fo`; the old `edge_fo`, `core_fo`, `step_fo` and `rim_fo` are still read, with a warning; they will be removed in a later version).

### Highlights

-   How to cite: `forster cite`, the app's Help menu and CITATION.cff.
-   Command line (`forster template`, `forster fit`, `forster import`) and a desktop app (`forster gui`) on the same project tables.
-   Import a whole folder of per-profile spreadsheets at once, from the command line, Python or the app, with one column mapping. CSV (including files saved by Excel on Windows), .xlsx, .xlsm and the old .xls are read.
-   Desktop app installers, built and self-tested in CI (the self-test starts the built app).
-   The app asks before quitting while a fit, starting-profile scan, Monte Carlo or law comparison is running or the project has unsaved changes: one dialog lists what is running and what is unsaved ("Save and quit", "Quit without saving" or "Stop and quit", "Cancel").
-   Checks that warn when a range is probably too narrow (misfit concentrated near a step, a profile that ends before its plateau) or a prior conflicts with the data.
-   Joint fits of several profiles of one crystal, and Ni, Mn and Ca fits (Python).
-   Automatic grids estimate the time in the fit's own scheme, which is faster.

### Other changes to know about

-   Inputs that would give a wrong answer are refused with a message: steps out of order or outside the model domain, profile angles whose squared cosines do not sum to 1, non-finite conditions, and fO2_Pa of 0 or less (fO2 is in Pa, not log units).
-   Warnings, with no number changed, when fO2 is below the lowest value the diffusion equation's source states (1e-10 Pa for `dipra` and `dfens_tamed`) and when `T_C` is above 1600 (probably kelvin).
-   The app refuses runs set to the legacy (kriging) uncertainty method; use the command line or Python for those.

### Known limits

-   Joint fits, Ni/Mn/Ca fits and a rim value that changes in time can so far be run only from Python (`import forster`), not from the parameter tables, the command line or the desktop app.
-   The desktop installers are built in CI but unsigned.
-   No Monte Carlo over an ascent (pressure-temperature) path yet: the Monte Carlo draws constant conditions. Planned for an early release.

### Citing

References for FORSTER, the method and each diffusion equation: `forster cite`, the app's Help menu, or the [documentation's citing page](https://forster.readthedocs.io/en/latest/changelog/#citing).
