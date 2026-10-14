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

## Is our calibration different, or missing something? (2026-10-14)

Residuals (observed - law, log10 D) against the 135 TaMED points, `compare_laws.py`:

| Law | rms | Chakraborty | Dohmen | Holzapfel | Petry | Spandler |
|---|---|---|---|---|---|---|
| Dohmen & Chakraborty 2007 (published) | 0.29 | -0.06 | -0.13 | +0.34 | +0.17 | +0.73 |
| Mutch et al. 2019 TaMED (published) | 0.24 | -0.06 | -0.03 | +0.06 | +0.10 | +0.43 |
| Mutch et al. 2019 Global (published) | 0.24 | -0.05 | -0.03 | +0.09 | +0.15 | +0.59 |
| refit, DC07 form, D0 and E free | 0.25 | -0.07 | -0.01 | +0.09 | +0.09 | +0.55 |
| refit, DC07 form, all but V free | 0.24 | -0.03 | -0.03 | -0.05 | +0.15 | +0.54 |
| refit, Mutch form, anisotropy fitted | 0.23 | -0.03 | -0.03 | -0.02 | +0.12 | +0.37 |

- **Where the data are, our refits and Mutch et al.'s laws agree** (rms 0.23-0.25
  against 0.24). The large parameter differences (E 223-236 against 201) are mostly
  the D0-E trade-off (correlation 0.993). The published DC07 law is the outlier: it
  is slow for the high-pressure (Holzapfel, +0.34) and high-fO2 data (Petry +0.17,
  Spandler +0.73), which it was not fitted to directly (its form comes from the
  point-defect model, its pre-factor from a subset of data).
- **The laws diverge only away from the data**, mostly in fO2: our DC07-form refit
  minus the published DC07 averages +0.01 log at the data's own conditions (range
  -0.35 to +0.49), but +0.15 at the KS20 conditions.
- **Missing terms (F tests on the Mutch form with fitted anisotropy):** an fO2 x 1/T
  interaction is significant (F = 7.3, p = 0.008), consistent with Dohmen et al.
  (2007)'s observation that the fO2 exponent falls at low temperature; added as form
  `mutch_fT`: g = -459 +- 170 (bootstrap +-227); effective E 216 kJ/mol at 1e-7 Pa and
  252 at the KS20 fO2; at KS20 it predicts +0.39 against +0.37 without it, so it does
  not explain the spread there. X_Fo x 1/T (p = 0.76), (ln fO2)^2 (p = 0.10) and
  X_Fo^2 (p = 0.54) are not supported.
- **Study offsets are significant** (F = 5.9, p = 0.0002): the studies disagree with
  each other by more than their internal scatter. Study effects (log, centred):
  Spandler +0.44 to +0.49, Petry -0.04 to +0.13, Chakraborty and Dohmen -0.05 to
  -0.27, Holzapfel -0.50 to +0.07 (confounded with the pressure terms, which only it
  constrains). Between-study sd by method of moments: 0.24-0.36 log (five studies, so
  rough). A covariance used for sampling should include a between-study term,
  since that spread is 5-10x the 0.02-0.05 log statistical errors of the DC07-form
  fits; added below as the `random study` method.
- **The only 1-atm data near our fO2 come from a melt-contact experiment.** Spandler
  and O'Neill (2010) diffused San Carlos olivine against basaltic melt (10^-3.3 Pa,
  1300 C) and sit 0.4-0.7 log above every law. Shea et al. (2023) report Fe-Mg about
  10x faster next to melt. So the constraint closest to the round robin and paper
  conditions may carry a melt effect the solid-source experiments do not.

Bowtie plots for all 20 form x method configurations: `analyses/femg_refit/figures/bowties/`
(index and band widths in the analysis README).

## The unpublished Meissner data (2026-10-14)

Dohmen and Chakraborty (2007) Part II mentions Meissner (2000, Bayreuth
dissertation, TEM-EDX) only in its Fig. 7b legend and reference list; Part I (p. 402)
and Chakraborty (2010, p. 615) say Meissner et al. (1998, Am Mineral 83, 546-552)
obtained the same results as Chakraborty (1997) on similar couples. Fig. 7b is
vector graphics, so its 7 Meissner points were read exactly
(`analyses/femg_refit/data/dohmen_chakraborty2007_fig7b_extracted.csv`): 900-1200 C,
log D -16.48 to -18.14 as plotted. They sit -0.10 +- 0.21 log from the DC07 equation
(-0.22 if normalised like Chakraborty's points, which Fig. 7b puts at Fo86), and 0.2
log below Chakraborty (1997) at the same temperatures.

Adding them to our fits (as plotted, at fO2 1e-7 Pa and either Fo90 or Fo86, [001],
1 atm) moves D at the KS20 conditions by at most 0.007 log and E by at most 2 kJ/mol
in every form. **The missing Meissner data do not explain why our refits differ
from the published DC07 law.** That difference comes from how DC07 was built: its
fO2 and composition terms are fixed by the point-defect model and its pre-factor and
E by a subset at 1e-7 Pa, while the high-fO2 (Petry, Spandler) and high-temperature
(Holzapfel) data, which it fits worst (residuals +0.17, +0.73, +0.34), pull free fits
up at our conditions.

Fig. 7b also shows that DC07 normalised the series inconsistently (Chakraborty to
Fo86 without an fO2 term, Part I high-fO2 points +0.12, Holzapfel not reproducible),
and that the line drawn there is close to log D0 -9.28 and 198 kJ/mol.

## fO2: what each study measures, and why they disagree (2026-10-13)

`analyses/femg_refit/fo2_figures.py`, Figs. 5 and 6 (`figures/fig5_fo2.png`,
`fig6_studies.png`). fO2 exponents (d log D / d log fO2, 1 sigma):

| Source | Exponent |
|---|---|
| Dohmen 2007, 1080 C [001] series | 0.258 +- 0.060 |
| Dohmen 2007, 1050 C [100] series | 0.183 +- 0.029 |
| Dohmen 2007, all TaMED, T and axis removed | 0.190 +- 0.032 |
| Petry 2004, T removed | 0.273 +- 0.038 |
| joint fit, DC07 form: OLS / weighted / ODR | 0.200 / 0.217 / 0.246 (+- 0.01-0.02) |
| joint fit, Mutch form: OLS / ODR | 0.258 +- 0.023 / 0.386 +- 0.030 |
| `mutch_fT` (OLS) at 900 / 1200 / 1400 C | 0.185 / 0.265 / 0.302 (+- 0.02-0.035) |
| DC07 (fixed) / Mutch 2019 TaMED | 1/6 / 0.224 |

- **Only two studies vary fO2, and at different temperatures.** Dohmen's series are
  at 1050-1080 C and give 0.18-0.26; Petry's reach 1400 C and give 0.27. An exponent
  that rises with T, as `mutch_fT` fits (0.19 at 900 C to 0.30 at 1400 C), reconciles
  them, and matches Dohmen et al.'s own remark that the exponent falls at low T. The
  F test for that term is p = 0.008.
- **The other three studies add level, not slope.** Chakraborty (1997) is at one fO2
  (1e-7 Pa). Holzapfel (2007) did not measure fO2 (we assume Ni-NiO, 10^1-10^4 Pa)
  and is alone at 6-12 GPa and 1350-1550 C, so its fO2 lever arm is confounded with
  the pressure terms: its study offset swings from +0.07 (`dc07_fixed`) to -0.30
  (`dc07_free`) to -0.50 (`mutch_aniso`) with the law's form. Spandler and O'Neill
  (2010) is one fO2 (10^-3.3 Pa) in a melt-contact experiment and sits +0.44 to
  +0.49 high in every form.
- **So "the exponent" of a joint fit is a blend**, weighted by where the data sit:
  least-squares fits give 0.20-0.26. The Mutch-form ODR's 0.39 is above every direct
  measurement, so we do not trust it; its D at the round robin conditions (+0.85 log
  against the default) should not be used.
- **The laws agree where the data are and fan out away from them** (Fig. 5a): at
  1e-7 Pa the least-squares forms and the two published laws span 0.12 log; at the
  round robin and paper fO2 (10^-3.75 to 10^-2.1 Pa, at 1200 C, Fo90, 1 atm) they
  span 0.29-0.46, all from the exponent (0.65-1.02 with the Mutch-form ODR). The
  only 1-atm data there are Spandler's melt experiment.
- Study offsets (Fig. 6b, centred, log): Chakraborty -0.24 to -0.05, Dohmen -0.27 to
  -0.07, Petry -0.04 to +0.13, Holzapfel +0.07 to -0.50, Spandler +0.44 to +0.49.

Bowtie update: points moved to a panel's conditions now carry a light bar for the
uncertainty of the move (from the fit's covariance) and fade with the size of the
move; see the analysis README.

## A between-study term: the `random study` method (2026-10-13)

`fit_random` in `fit_models.py`: ln D = law + u_study + e, with u ~ N(0, tau^2) and
e ~ N(0, sigma^2), tau by REML. It is one fitted parameter (tau), not an adjustment
to any study's data; the study offsets come out shrunk toward zero, not fitted
freely. Points of one study share u, so the fit is generalised least squares with a
block covariance, and a study counts for less than its number of points. Its
covariance is for the law averaged over labs; `se_new_lab` adds tau for what one new
lab (or, if the spread is real variability, one new crystal) would show.

| Form | sigma within / tau between (95%), log | KS20 vs default: OLS -> random | 1-sigma: OLS -> random (one new lab) |
|---|---|---|---|
| dc07_fixed | 0.22 / 0.27 (0.12-0.70) | +0.10 -> +0.20 | 0.025 -> 0.124 (0.30) |
| dc07_free | 0.22 / 0.25 (0.11-0.66) | +0.15 -> +0.27 | 0.029 -> 0.122 (0.28) |
| mutch_x6 | 0.22 / 0.26 (0.08-0.76) | +0.37 -> +0.29 | 0.082 -> 0.141 (0.30) |
| mutch_aniso | 0.22 / 0.26 (0.08-0.75) | +0.37 -> +0.29 | 0.085 -> 0.141 (0.29) |
| mutch_fT | 0.22 / 0.29 (0.04-0.92) | +0.39 -> +0.30 | 0.083 -> 0.152 (0.32) |

- **The forms converge.** At KS20 they span +0.10 to +0.39 under OLS and +0.20 to
  +0.30 with the random offset, and the 1-sigma bars (0.12-0.15) are now as wide as
  that spread. OLS let Chakraborty and Dohmen, 99 of the 135 points and both low,
  set the level; with a random offset each study counts closer to once.
- **The fO2 exponents converge on Mutch et al.'s 0.224:** `dc07_free` 0.200 -> 0.221,
  `mutch_x6` 0.258 -> 0.225.
- **E falls** (the D0-E trade-off): `dc07_fixed` 236 -> 216 kJ/mol, `dc07_free`
  220 -> 209, `mutch_x6` (-R q) 213 -> 209. V at 1250 C in the Mutch forms 7.9 -> 6.5
  cm3/mol.
- **The fO2 x 1/T term weakens:** g = -459 +- 170 (OLS, 2.7 sigma) -> -324 +- 190
  (1.7 sigma). Petry (high T, high fO2) sits above Dohmen, so a T-dependent
  exponent and a Petry offset explain the same points; with offsets allowed, the
  data no longer separate the two.
- **Shrunken study offsets** (log): Spandler +0.36 to +0.40, Petry -0.02 to +0.09,
  Chakraborty and Dohmen -0.10 to -0.23, Holzapfel +0.07 to -0.23. Spandler stays
  high, so most of its melt-contact offset is absorbed, not all.
- **The bands no longer widen toward KS20** (0.9-1.0x): the spread between studies
  sets the level everywhere and outweighs the fO2 extrapolation.
- Caveats: tau rests on five studies (95% range roughly 0.1-0.7 log); the model
  assumes offsets unrelated to the conditions, which fails for Holzapfel (alone at
  high P) and Spandler (alone at 1 atm, high fO2); and whether to add tau for a
  natural crystal is a judgment about what the spread is.
