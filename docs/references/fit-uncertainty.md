# Uncertainty on the fitted diffusion time

How the 1-sigma range on a fitted time is found, how that has changed between
versions of this code, and why. Started 2026-10-12.

The method itself, step by step, with a worked example, validation on synthetic
and real profiles, and figures: [likelihood-method.md](likelihood-method.md).

## Summary

Every version fits time the same way: run the model forward, keep the profile at
every time step, and take the time step whose profile best matches the data. The
versions differ in what the profile is compared with and in how the range is
drawn.

| Version | Used | Compared with | Range on time | Code |
|---|---|---|---|---|
| v1 | until March 2026 (paper era, Towbin and Plank 2026) | kriged profile on the model grid (N points) | where grid chi-squared reaches min + sqrt(2(N-1)) | `engine.Best_fit_Chi2_old` |
| v2 | March 2026 (commit 7dcd080, 2026-03-26) to 2026-10-11; round robin results | same | where reduced chi-squared reaches min + 1, i.e. grid chi-squared reaches min + (N-1) | `engine.Best_fit_Chi2`, now `fit_profile(uncertainty="legacy")` |
| v3 | default from 2026-10-12 | the measured points themselves | where a correlated-noise chi-squared reaches min + 1 | `forster.uncertainty`, `fit_profile(uncertainty="correlated")` |

v1 also reported time as (row + 1) x dt; v2 and v3 use row x dt.

v2 is kept as `uncertainty="legacy"` and gives results identical to the round
robin script (`tests/test_differential.py`). Use it to reproduce earlier numbers;
use v3 for new work.

The three versions were compared on the 17 round robin profiles. v1 gave the
narrowest ranges and v2 the widest, with v3 in between. v3 also moves some
best-fit times, because it weights the measured points rather than the kriged
curve. The round robin data belong to other labs, so the per-profile table
(`round_robin_versions.csv`, made by
`docs/plans/2026-10-12-fit-uncertainty/compare_versions.py`) is kept in the
private round robin repository.

## What was wrong with v1 and v2

1. **The data were kriged pseudo-points.** Chi-squared was summed over the model
   grid (150 to 300 points), not over the measured points (20 to 300). Grid points
   between two measurements are interpolations of the same few analyses, so they
   are strongly correlated, but chi-squared counts each one as an independent
   measurement. The amount of evidence is overstated by roughly the ratio of grid
   points to measurements. A textbook delta chi-squared = 1 on that grid covers the
   true time only 16% of the time (coverage tests below).
2. **The v1 and v2 thresholds compensate for that by rule of thumb, not by
   statistics.** sqrt(2(N-1)) (v1) is the standard deviation of the
   chi-squared distribution, a measure of goodness of fit, not a confidence
   interval on a parameter. Reduced chi-squared within 1 of its minimum (v2) means
   chi-squared within N - 1, and so depends on the grid spacing. Andrae et al.
   (2010) explain why reduced chi-squared should not be used at all for nonlinear
   models: the number of degrees of freedom is unknown. In the coverage tests, v2
   was far too wide when the model is exact (99.7% instead of 68%) and about
   right only by coincidence when the model misfits.
   Where they came from (Henry, 2026-10-21): the aim was the Numerical Recipes rule
   (Press et al. 2007, §15.6), but applied to reduced chi-squared, while the book's
   Δχ² = 1 applies to chi-squared itself, scaled to the chi-squared distribution. The
   Wikipedia articles "Reduced chi-squared statistic", "Chi-squared test" and
   "Goodness of fit" were the other sources consulted.
3. **The kriging variance was used as a standard deviation.** PyKrige returns the
   variance (`sigmasq`). With the variogram used for the round robin fits it
   is small next to `uniform_uncert` (0.001) and `sigma_min`
   (1e-4), so every grid point had almost the same weight, about 0.11 Fo. The
   intended down-weighting of gaps between analyses did not happen. Legacy mode
   keeps this behaviour so old results reproduce. The intent (Henry, 2026-10-21) was
   that points far from a measurement, with a larger kriging uncertainty, would count
   for less in chi-squared. Even with the standard deviation used correctly, that
   would not have removed the overcounting of point 1: a weight scales each point
   independently, while kriged points near one analysis are near-copies of it and
   still count as separate evidence. Correlated points need the full covariance
   between them, which is what v3's noise model supplies on the measured points.
4. **Hand-set kriging parameters entered the answer.** The variogram, the nugget,
   `uniform_uncert` and the grid spacing all changed the range. They did not
   change the best fit much (v1 and v2 share it; v3 moves some best fits).

What was not wrong: varying chi-squared over a grid of times is a sound way to
fit one parameter (Andrae 2010, sections 3.1 and 3.2). It copes with a nonlinear
model and gives asymmetric ranges. The Numerical Recipes forum thread on
delta chi-squared (t-1249) is about projecting multi-parameter fits, which does
not arise when time is the only parameter fitted.

## What v3 does

1. Run the model forward as before (unchanged).
2. Interpolate every model curve to the measured positions. The positions never
   change, so this is one fixed matrix W (two nonzeros per row), not work per time
   step. Points outside the model domain are left out.
3. First pass: plain sum of squares, to find a provisional best fit.
4. Fit an error model to the residuals at that fit by maximum likelihood:

   C = s^2 (I + rho^2 K),  K[i, j] = exp(-(x_i - x_j)^2 / (2 L^2))

   - `s` (`NoiseModel.white_sigma`): independent analytical noise, point to point;
   - `a = rho * s` (`correlated_sigma`): a smooth misfit the model cannot follow
     (initial-condition shape, 1-D geometry, beam convolution);
   - `L` (`correlation_length_um`): how far along the profile that misfit
     persists.

   s has a closed-form best value for each (rho, L), which are searched on a grid.
   A warning is raised if the best (rho, L) is at the edge of that grid.
5. Second pass: the generalised chi-squared r^T C^-1 r for every time step. Points
   closer together than about L count as partly redundant. Its minimum is the best
   fit and the 1-sigma range is where it is within 1 of the minimum, with the ends
   interpolated between time steps. This is a profile-likelihood interval, valid
   for a nonlinear model with one fitted parameter.

The noise model is the same Gaussian process used in kriging. The old code used it
to smooth the data; v3 uses it to describe the residuals.

Cost: the chi-squared is evaluated as c^T A c - 2 b^T c + const with
A = W^T C^-1 W, so its cost depends on the model grid, not on the number of
analyses. On a round robin profile with 29 analyses (105,121 steps x 300 grid
points) it takes 0.04 s, against 0.14 s for v2's grid chi-squared and 1.9 s for
the whole fit. On a dense round robin profile with 298 analyses (525,601 steps x
150 grid points) it takes 0.07 s, against
0.34 s for v2's chi-squared plus 0.30 s for kriging, and 8.1 s for the fit.
Kriging still runs, for plots only.

## Evidence

Coverage: 300 synthetic profiles at the measured positions of a 29-point
round robin profile, made from the model at a known time plus noise, each fitted
with every method. A 1-sigma range should contain the true time 68% of the time
(sampling error about +-3%).

| Method | Model exact (white noise 0.2 Fo) | Plus smooth misfit 0.4 Fo over 10 um |
|---|---|---|
| v2: grid, reduced chi-squared + 1 | 99.7% | 62% |
| Grid, delta chi-squared = 1 | 16% | 4% |
| Measured points, delta chi-squared = 1, independent errors | 69% | 23% |
| Same, errors scaled so reduced chi-squared = 1 | 68% | 46% |
| Pairs bootstrap of measured points | 61% | 41% |
| Moving-block bootstrap | 56% | 52% |
| **v3: measured points, correlated noise** | **71%** | **66%** |

Only v3 is close to 68% in both cases. The bootstraps assume independent points
and undercover when the misfit is smooth (Sohn and Menke 2002 recommend the
bootstrap for nonlinear fits, under that assumption). These figures are from the
exploratory study (2026-10-11) with a coarser noise-model search;
`tests/test_uncertainty.py` repeats the coverage test for the implemented method
(200 profiles per case, seeded).

The real round robin profiles do have smoothly correlated residuals, and the
analyses on the dense round robin profiles scatter by more than the 0.2 Fo that
the round robin kriging nugget assumed. v3 estimates the scatter from the data.

## Not included in the range

- **Temperature, pressure, fO2, orientation, activation energy.** D factors as
  D0(T, P, fO2, orientation) x 10^(3 (0.9 - X_Fo)), so the modelled profile
  depends only on D0 x t, and the fitted time scales as 1/D0. These
  uncertainties can be propagated afterwards by Monte Carlo on D0 without re-running
  the model, as `Round_Robin_Diffusion_Modeling Part 2.py` does. See the issues
  below before reusing that code.
- **Initial condition** (step position and plateau values). Fitting the step
  position as a second parameter and taking the range over both is possible (a few
  seconds per position) but not implemented.
- **Discretization error.** About 1.5% at dx = 1 um on the synthetic test in
  `tests/test_physics.py`. Check by refitting at a finer grid or with
  `scheme="dipra"`.

## Issues found in the round robin diffusivity Monte Carlo (not changed)

In `Round_Robin_Diffusion_Modeling Part 2.py`, which combines the fit range with a
Monte Carlo over T, P, fO2 and activation energy:

1. **Temperature in degrees C where kelvin is expected.** `sample_PT_fO2` draws
   `T_C_rand` and passes it to `D_Fo_For_PT_Uncert_Sampling(T=...)`, which uses T
   in kelvin (as do the defaults, `T=1200`). The median ratio is unaffected; the
   spread is too wide.
2. **33.33% and 66.66% quantiles are labelled +-1 sigma.** They span the middle
   third; 1 sigma is 15.87% to 84.13%. The spread is too narrow.
3. **`Diffusion_Uncert2` (the 1102 C round robin sample) uses the 1200 C samples**
   (`Sample_1to3_Diffusivity_array_scale` in both `match` cases).
4. **Activation energy varied with the pre-factor held fixed.** E = 201 +- 8
   kJ/mol (Dohmen et al. 2007) comes from a fit to experiments at 900-1250 C.
   Varying E at fixed D0 pivots the Arrhenius line at 1/T = 0, so at 1200 C it alone
   moves D by about +-0.28 log units (x1.9). Pivoting near the middle of the
   experimental range would move D much less at 1100-1200 C. This term
   dominates the Monte Carlo spread. Dohmen et al. (2007) publish no E-D0
   correlation; Mutch et al. (2019, 2021) publish a regression with full
   covariance, which gives x0.78 to x1.29 at 1200 C for the calibration term.
   See `diffusivity-literature.md`. [Henry to decide whether to adopt it.]

Items 1 and 2 partly cancel. For samples 1-3 (1200 C, T +-30 C, fO2 +-0.2 log,
E +-8 kJ/mol) the time multiplier reported as +-1 sigma was x0.66 to x1.51; with
kelvin and 1-sigma quantiles it is x0.48 to x2.10 (x0.45 to x2.25 at 1102 C). The
Combined +-1 sigma column of the round robin table also adds the v2 fit range in
quadrature.

## Open items

- [ ] Refit and report round robin and paper results with v3, or state which
      version produced each published number. [Henry: confirm whether the paper's
      reported ranges used the chi-squared range or only the P-T Monte Carlo
      percentiles.]
- [ ] Fix or retire the diffusivity Monte Carlo above, and decide how to treat
      the activation-energy uncertainty (item 4).
- [ ] One round robin profile warns that the noise model is at the edge of its
      grid (a very long correlation length): the misfit is a long-wavelength
      trend. Check that profile's initial condition.
- [ ] Consider putting the D0 Monte Carlo into the package, combined with the v3
      range in log time rather than in quadrature in days.

## References

- Andrae, R. (2010). Error estimation in astronomy: A guide. arXiv:1009.2755.
  Sections 3.1 (brute-force grids), 3.2 (varying chi-squared and the bad practice
  of forcing reduced chi-squared = 1), 3.4, 4.2.
- Andrae, R., Schulze-Hartung, T. and Melchior, P. (2010). Dos and don'ts of
  reduced chi-squared. arXiv:1012.3754.
- Numerical Recipes forum, "multi-parameter fits", thread t-1249
  (<http://www.numerical.recipes/forumarchive/index.php/t-1249.html>):
  delta chi-squared = 1 for one parameter after minimising over the others
  ("Theorem D") is valid; projections of multi-parameter fits need care.
- Sohn, R. A. and Menke, W. (2002). Application of maximum likelihood and
  bootstrap methods to nonlinear curve-fit problems in geochemistry. G-cubed
  3(7), doi:10.1029/2001GC000253.
- Dohmen, R., Becker, H.-W. and Chakraborty, S. (2007). Fe-Mg diffusion in
  olivine I. Phys Chem Minerals 34, 389-407 (E = 201 +- 8 kJ/mol, 900-1250 C).

## Log

### 2026-10-12

v3 added as the default (`forster/uncertainty.py`); v2 kept as "legacy".
Evidence gathered 2026-10-11 to 12 in a Claude Code session: interval
comparison on five round robin profiles, coverage on synthetic profiles,
timings on the full curve arrays, and the 17-profile version comparison in
`docs/plans/2026-10-12-fit-uncertainty/`.
