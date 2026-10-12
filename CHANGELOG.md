# Changelog

## Unreleased

### Added

- `forster` command (2026-10-12). `forster template DIR` writes `profiles.csv` and `parameters.csv` to fill in, holding one published profile as a worked example (AZ18 WHT01_Ol_63_prof_1, Towbin and Plank 2026), and `COLUMNS.md`. `forster fit PROFILES PARAMETERS --out DIR` fits every run and writes, per run, a figure (`_fit.png`), the model curves, the fitted points with residuals and the kriged profile, plus `summary.csv` (time, 1σ range and every input; it reads back as a parameter table). A failing run is reported and skipped (exit status 1). `--run` picks runs; `--profiles-sheet` and `--parameters-sheet` read Excel.
- Template parameter layout: the Python API's names with units, Fo as a fraction, orientation as angles to the crystal axes or as EBSD Euler angles with the profile direction, fO2 in Pa or relative to a buffer. The round robin and paper layouts are still read. Profile tables in the template layout (`profile`, `distance_um`, `fo`) are detected from their columns.
- `forster.plot_fit(result, ax=None)`, `forster.summary_row(result)`, `forster.fit_tables(result)`, `forster.write_template(dir)`.
- README with installation and the three ways to run (the desktop app follows in the next phase).

- DIPRA's finite-difference scheme (Girona and Costa 2013, eq. 5) as an alternative to the engine's: `fit_profile(..., scheme="dipra")`, `RunDefaults(scheme="dipra")`, or a `scheme` column in a parameter table. Both schemes solve the same equation and differ only in how the cross term is differenced (DIPRA: forward differences and the analytic dD/dX_Fo; engine: central differences of C and D), so fitting a profile with both is a check on the discretization. Default stays "central".

### Changed

- **New default for the uncertainty on the fitted time (2026-10-12).** The best fit and its 1-sigma range now come from the measured points rather than the kriged profile: each model curve is interpolated to the analyses, their errors are modelled as white noise plus a smooth correlated misfit (fitted by maximum likelihood, `FitResult.noise`), and the range is where that chi-squared rises by 1, interpolated between time steps (`forster/uncertainty.py`). Select with `fit_profile(..., uncertainty=...)`, `RunDefaults(uncertainty=...)` or an `uncertainty` column: "correlated" (new default) or "legacy".
  - "legacy" is the rule used from March 2026 for the round robin (`engine.Best_fit_Chi2`: reduced chi-squared of the kriged grid within 1 of its minimum, i.e. chi-squared within N - 1 for N grid points) and reproduces those results exactly. Before March 2026 (paper era) the range was where grid chi-squared reached min + sqrt(2(N-1)) (`engine.Best_fit_Chi2_old`), with time reported as (row + 1) x dt.
  - Why: the old rules counted 150-300 correlated kriged points as independent data and drew the range by rule of thumb (reduced chi-squared, which Andrae et al. 2010 show is undefined for nonlinear models); the kriging variance was also used as a standard deviation, so it had almost no effect. On synthetic profiles with a known time, the legacy range contained it 99.7% of the time when the model is exact and 62% with smooth misfit (target 68%); the new range 71% and 66%.
  - Effect on the 17 round robin profiles: median 1-sigma width (as a share of the best fit) 11% before March 2026, 34% legacy, 24% new (9-36% by profile); best-fit times move by -14% to +8%. Cost: 0.04-0.07 s per fit against 1.9-8.1 s for the model run; faster than the legacy grid chi-squared.
  - `FitResult` gains `uncertainty`, `fit_x_um`, `fit_fo` (the points fitted) and `noise`; `reduced_chi2` now divides by the number of points fitted.
  - Details, coverage tests and the version comparison: `docs/references/fit-uncertainty.md`, `docs/plans/2026-10-12-fit-uncertainty/`.

### Fixed

- CSV tables are read with exact float parsing (`float_precision="round_trip"`). pandas' default parser returned some values 1 ulp off, so a table written by FORSTER did not refit identically.
- Two step positions without `step_fo` (or three or more steps) are now an error naming the problem; the engine used to fail with an IndexError.

- Cross term of the diffusion equation was doubled (`diffusion_step`, found 2026-10-11). The engine solves dC/dt = D d²C/dx² + (dD/dx)(dC/dx). The cross term was built from two central differences, each spanning two grid spacings, so their product needs dividing by 4Δx²; the code divided by 2Δx². The fix changes `/ 2` to `/ 4`. It affects every Fe-Mg, Mn and Ni model where D varies along the profile, and only those (D depends on Fo). Checked against an independent flux-form solver and against DIPRA's published scheme (Girona and Costa 2013, eq. 5), which converge to the same answer as the fixed engine while the old engine does not (`tests/test_physics.py`; details in `docs/plans/2026-10-11-restructure.md`, "Engine fix").
  - Effect on fitted times: idealised single steps came out 1.3% too short for Fo 0.92 to 0.895 and 8-11% too short for Fo 0.82 to 0.885 and 0.67 to 0.745. The 17 round robin fits move by -28% to +43% (most become 13-43% longer). For Towbin and Plank (2026) the melt-vein medians move by 2-5% (1-3 weeks unchanged) and the exterior contact-overgrowth median by +28% (5.9 to 7.5 days).

### Before release

- [ ] Paper addendum, to return to later: the cross-term fix changes the timescales in Towbin and Plank (2026, J. Petrology 67(9), egag077). Melt-vein medians move by 2-5% (WHT01 17.9 to 17.6 days, GCB1 12.8 to 13.5, WHT06 5.6 to 5.4); the exterior contact-overgrowth median moves from 5.9 to 7.5 days (+28%); xenocrysts +9%. Check the ascent rates derived from these times. Numbers per profile: `docs/plans/2026-10-11-engine-fix/`.

- [ ] Uncertainty method: state in the release notes which version (pre-March 2026, legacy, correlated) produced each published or submitted range, and refit the round robin with "correlated". The round robin diffusivity Monte Carlo (`Round_Robin_Diffusion_Modeling Part 2.py`) has separate problems: temperature in degrees C passed where kelvin is expected, 33.33/66.66% quantiles labelled +-1 sigma, the 1200 C samples used for SH63 (1102 C), and the activation energy varied with the pre-factor held fixed. For samples 1-3 its "+-1 sigma" time multiplier was x0.66 to x1.51; corrected for the first two it is x0.48 to x2.10. See `docs/references/fit-uncertainty.md`.

- [ ] Come back to the cross-term fix: state the size of the bias for older results in the release notes, refit the round robin profiles (revisit the SH63olv066 initial conditions, whose fits get worse with the fixed engine), correct any submitted round robin results, and add the note on the paper results to the Xenolith_Melt_Vein_Modeling README.
