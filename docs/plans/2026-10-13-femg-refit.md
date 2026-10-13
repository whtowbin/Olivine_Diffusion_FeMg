# Fe-Mg diffusivity refit with a covariance

Status (2026-10-13): data compiled and checked; Mutch et al. (2019) reproduced within ~1 standard error; four law forms fitted five ways, with figures (`analyses/femg_refit/`). Presets with covariance not yet added (decision for Henry, below).

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

## Refits (2026-10-13)

Four forms x five methods on the 135 TaMED points (`fit_models.py`; forms, methods
and assumed errors in `analyses/femg_refit/README.md`; figures in `figures/`).

Coefficients (OLS, 1-sigma OLS / run bootstrap):

| Quantity | DC07 form, D0 and E free | DC07 form, all but V free | Mutch form | Published |
|---|---|---|---|---|
| E (kJ/mol) | 236 (5 / 6); 223 weighted, 228 ODR | 220 (7 / 10) | 214 (7 / 9) | 201 +- 8 (DC07); 222 +- 7.5 (Mutch TaMED) |
| log10 D0 | -7.89 (0.19 / 0.26) | -8.52 (0.27 / 0.40) | | -9.21 |
| corr(log D0, E) | 0.993 | | | not published |
| fO2 exponent | 1/6 fixed | 0.200 (0.009 / 0.014); 0.218 weighted, 0.246 ODR | 0.26 (0.02 / 0.04); 0.39 ODR | 1/6 (DC07); 0.224 (Mutch) |
| composition, log10 D per X_Fe | 3 fixed | 3.12 (0.16 / 0.39) | 3.08 (0.16 / 0.48) | 3 (DC07); 3.12 (Mutch) |
| V (cm3/mol, 1250 C) | 7 fixed | 7 fixed | 7.9 (0.8 / 1.1) | 7 (DC07); 7.5 (Mutch) |
| [001]/[100], [001]/[010] | 6, 6 fixed | 6.2, 5.1 (+-1.0) | 6.5, 5.3 | 6 |

log10 D[001] at the round robin KS20 conditions (1200 C, fO2 1.4e-3 Pa, Fo85),
minus the current default (time multiplier = 10^-difference):

| Form | OLS | weighted | ODR | run bootstrap 1-sigma |
|---|---|---|---|---|
| DC07, D0 and E free | +0.10 | -0.01 | +0.01 | 0.04 |
| DC07, all but V free | +0.15 | +0.18 | +0.30 | 0.05 |
| Mutch, 6x | +0.37 | +0.53 | +0.85 | 0.14 |
| Mutch, anisotropy fitted | +0.37 | +0.45 | +0.81 | 0.13 |
| Mutch et al. 2019 TaMED (published) | +0.28 | | | |

What the fits show:

1. **The form of the law and the fitting method move D at our conditions more than
   any one fit's 1-sigma does.** Across forms and methods D[001] at 1200 C ranges
   from the default to 0.85 log units above it (times x1.0 to x0.14); single-fit
   1-sigma bars are 0.02-0.17 log.
2. **The cause is the fO2 term, extrapolated.** Only 10 of 135 points lie within 1.5
   log units of the round robin and paper fO2 (10^-3.7 to 10^-2.1 Pa); the 17 above
   are Holzapfel's 6-12 GPa runs, whose fO2 is not measured (computed at Ni-NiO) and
   which reach 1 atm only through the pressure correction. Free fits put the
   exponent at 0.20-0.39, above the 1/6 of the point-defect model, and the higher
   it is the faster D is at our fO2. ODR raises it further: switching on each
   predictor error alone, only the fO2 errors move the prediction (+0.53 to +1.07
   for the Mutch form); tightening Holzapfel's assumed fO2 error from 0.5 to 0.1
   log units barely changes that (+0.85 to +0.84). A control with near-zero
   predictor errors reproduces the weighted fit exactly.
3. **With the point-defect fO2 and composition terms kept, the data want a higher
   activation energy:** E = 223-236 kJ/mol (D0 and E free), against 201. The two
   are correlated at 0.993, so E +- 8 kJ/mol alone (pre-factor fixed) overstates the
   uncertainty of D near the middle of the data by far; the covariance here does
   not. At our conditions this form stays within 0.1 log of the default.
4. **Composition, activation volume and anisotropy agree with the published values**
   within their errors: composition 3.1 log10 D per X_Fe, V 7.9 +- 1.1 cm3/mol (the
   combination V = -R (j T + h) is well determined; h on its own is not, its 1-sigma
   is about 8x its value, because P and P/T are nearly collinear over Holzapfel's
   1350-1550 C), anisotropy 6.2 and 5.1 against 6. A first run without putting the
   1 atm (101325 Pa) and 1 bar (1e5 Pa) rows on one pressure gave the 1 atm data
   spurious leverage on the pressure terms (bootstrap spread of j 1e5 times too
   large, leave-out-Holzapfel predictions off by 1e3 log units); fixed in `tamed()`.
5. **Residual scatter is 0.24 log units in every form**, the stated experimental
   error; the largest residuals are Holzapfel's two 11 GPa runs (which the authors
   excluded as anomalous) and Spandler's two off-axis values (ends of a fit against
   angle).
6. **Leaving one study out** (OLS, change in log D at our conditions): without
   Chakraborty (1997), the only data below Fo85, SH63 (Fo ~0.70) moves by -0.75 to
   -0.79 log in the Mutch forms and -0.37 in the free DC07 form (KS20 by -0.18 to
   -0.20 and -0.10); without Holzapfel
   the free DC07 form moves +0.25 to +0.36; without Spandler and O'Neill the Mutch
   forms move -0.11 to -0.17. Every other single study moves any prediction by
   0.12 log or less, and the DC07 form with D0 and E free by 0.06 or less.

Recommendations (for Henry):

- Report the calibration as a systematic: the default and a refit bracket the
  answer at our fO2 better than any single covariance.
- If one refit becomes a preset with covariance for the phase 3 Monte Carlo, the
  DC07 form with D0 and E free (OLS or run bootstrap covariance) is the least
  extrapolation-sensitive; the free forms' fO2 exponent is the open question.
- The fO2 dependence between 10^-4 and 10^0 Pa is what would most reduce the
  uncertainty: no experiment constrains it at 1 atm.
