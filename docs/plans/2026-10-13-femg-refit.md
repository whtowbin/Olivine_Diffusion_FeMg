# Fe-Mg diffusivity refit with a covariance

Status (2026-10-13): data compiled and checked; least-squares reproduction of Mutch et al. (2019) within ~1 standard error (below); bootstrap and ODR fits next.

## Goal

A Fe-Mg diffusivity law with a full parameter covariance, fitted to the published
experiments, so the calibration uncertainty can be sampled with its correlations
(the Mutch et al. 2019 matrices are rounded and not positive definite; Dohmen and
Chakraborty 2007 publish none).

## Approach

1. Compile the calibration data of Mutch et al. (2019): Chakraborty (1997), Petry et
   al. (2004), Dohmen, Becker and Chakraborty (2007, Part I), Dohmen and Chakraborty
   (2007, Part II; check for tabulated data), Holzapfel et al. (2007), Spandler and
   O'Neill (2010). One CSV per source table, transcribed from the PDFs and checked
   row by row against page images; then one combined table.
2. Reproduce Mutch et al. (2019) Supplementary Tables 1-2 by ordinary least squares
   (their method: multiple linear regression of ln D[001] on ln fO2 (bar), X_Fo,
   1/T, P (Pa), P/T; [100] and [010] data scaled to [001] by 6). Matching the
   coefficients and the covariance validates the compilation. Global: all Fe-Mg
   data; TaMED: the TaMED-regime data (selection rule to be found by matching).
3. Our fits, same form and the Dohmen and Chakraborty form:
   - bootstrap resampling by experiment run and by study (rows from one run, or one
     lab, are not independent; OLS treats them as if they were);
   - errors in the predictors (T, fO2, composition) by orthogonal distance
     regression with `odrpack`, following Henry's `odr_bootstrap` method (the
     package itself fits one predictor; this needs five). Expected to change little
     (a 3 C temperature error is ~0.04 in ln D against ~0.5 scatter); measured, not
     assumed.
4. Presets with covariance in `forster.diffusivity`, for the phase 3 Monte Carlo.

Data and scripts: `analyses/femg_refit/` (data/ with one CSV per source table and a
README citing each; fit scripts; results).

## Data in hand

- Dohmen, Becker and Chakraborty (2007) Part I, Table 3: 75 experiments, parsed
  twice independently (identical). Fo86, 1 atm, fO2 1e-12 to 1e-5 Pa, three axes.
- Chakraborty (1997): Table 3, 55 D (12 runs, Fo25-97, [001], 1 atm), T and fO2
  per run in Table 2.

## Open questions

- Which Spandler and O'Neill (2010) quantity Mutch used for Fe-Mg (Fe tracer or
  interdiffusion) and at which composition.
- Whether Dohmen and Chakraborty (2007) Part II tabulates data Mutch could use (the
  Meissner experiments are unpublished).
- How Mutch split TaMED from PED data (fO2 threshold 1e-10 Pa, or temperature).

## Results so far (2026-10-13)

Data: 166 Fe-Mg coefficients from five sources (`analyses/femg_refit/README.md`),
hand-checked against the page images.

Reproduction of Mutch et al. (2019) by ordinary least squares (`fit_ols.py`).
Distance of each coefficient from theirs, in units of their standard error (a, b,
c, q, j, h):

| Data choices | Global, max |z| | TaMED (fO2 > 1e-10 Pa), max |z| |
|---|---|---|
| all axes, [100]/[010] x 6 | 1.76 (a) | 1.19 (b) |
| [001] data only | 1.26 (b) | 0.63 (b) |

Our standard errors are 0.6-1.1x theirs; correlations agree within 0.15-0.44.
Identical data would agree within ~0.05 (their values are printed to 3 figures),
so their compilation differs from ours in details not stated in the paper (the
alternatives tried: Petry composition Fo90 or Fo95, Holzapfel's 11 GPa runs and
unbuffered runs, Spandler's three axes or [001] only, Chakraborty's fO2 per run or
nominal; none closes the gap). An exact match needs their compiled table (DFENS,
Mutch et al. 2021, Data Set S6). Numerical note: the predictors span 13 orders of
magnitude (P ~ 1e10 Pa, 1/T ~ 1e-3), so the columns must be scaled before solving;
unscaled `lstsq` silently drops the 1/T direction (q came out as -0.008).
