# Fe-Mg diffusivity refit

Experimental Fe-Mg diffusion coefficients in olivine, compiled to refit the
diffusivity law with a full parameter covariance. Plan and results:
`docs/plans/2026-10-13-femg-refit.md`.

## The preset: verified data, 6x anisotropy (2026-10-15)

`femg_refit2026` is fitted to the verified compilation (every value stated in the
sources' tables or text; `docs/references/data-to-source.md`) plus Holzapfel et al.'s
Au-capsule runs, whose unbuffered fO2 is taken at Ni-NiO with a sd of 2 log units.

- **Form:** the Dohmen and Chakraborty (2007) form with log D0, E, the fO2 exponent and
  the composition coefficient fitted; [100] and [010] six times slower than [001] and
  V = 7 cm3/mol fixed (`fit_verified.FORM`).
- **Method** (`fit_verified.fit_preset`): random offsets per study (REML), with each
  run's errors in T (at least 10 C), fO2, X_Fo and P carried as extra variance on ln D
  (effective variance: ODR to first order, inside the REML likelihood).
- **Data:** Chakraborty 1997, Dohmen et al. 2007 (fO2 > 1e-10 Pa), Petry et al. 2004 and
  Holzapfel et al. 2007 (Ni-NiO capsules, and Au capsules at Ni-NiO ± 2 log); 132 points.
- **Scripts:**
  - `fit_verified.py`: `verified()`, `preset_data()`, `oeser_dp()`, the preset's fit
    (`fit_preset`), and its variants (no input errors, no Au runs, the Au runs' fO2 sd
    0.5-3 log). Writes `results/verified/`.
  - `preset_figures.py`: `figures/presets/`, the preset's bowties and residuals and the
    candidate methods (`results/verified/candidates.csv`).
  - `odr_check.py`: ODR against the preset's method, for T-error floors of 0, 10 and
    20 C (`results/verified/odr_check.csv`).
  - `export_preset.py`: builds the preset and checks the shipped one.
  - `unconfirmed_figures.py`, `anisotropy_figures.py` and `oeser_fo2_figures.py`:
    `figures/unconfirmed/`, `figures/anisotropy/` and `figures/oeser_fo2/`, the
    unconfirmed data and scenarios (Oeser et al. 2026; melt contact) against the
    preset (`docs/references/unconfirmed-data.md`).
  - `transcribe_oeser2026.py`: Oeser et al. (2026) Tables 1-3.
  - `transcribe_mutch_ds06.py`: Mutch et al. (2021) Data Set S6, behind the
    `mutch2019_*` presets.
- **The rest of this folder** (`fit_models.py` and `results/*.csv`) is the full
  compilation as first analysed; it is kept for comparison.

## Data (`data/`), transcribed from the published tables

| File | Source | Rows (Fe-Mg) | Conditions |
|---|---|---|---|
| `chakraborty1997_table3.csv` | Chakraborty (1997), JGR 102(B6), 12317-12331, Table 3 (p. 12,322) joined to Table 2 (p. 12,320) | 55 | 980-1300 C, Fo25-97, [001], 1 atm, fO2 ~1e-7 Pa |
| `dohmen2007_part1_table3.csv` | Dohmen, Becker and Chakraborty (2007), Phys Chem Minerals 34, 389-407, Table 3 (p. 393) | 75 | 700-1200 C, D for Fo86, [100]/[010]/[001], 1 atm, fO2 1e-12 to 1e-5 Pa |
| `petry2004.csv` | Petry, Chakraborty and Palme (2004), GCA 68, 4179-4188, Table 3 (p. 4182) | 16 (+12 Mn, 17 Ni) | 902-1403 C, Fo90-Fo100 couples, [001], 1 atm, fO2 1e-9.9 to 1e-3.9 Pa |
| `holzapfel2007.csv` | Holzapfel, Chakraborty, Rubie and Frost (2007), PEPI 162, 186-198, Table 1 (accepted manuscript p. 34) | 17 (+16 Ni, 16 Mn) | 1350-1550 C, 6-12 GPa, Fo97 (couple average), [001]; fO2 not given (Ni-NiO buffer or Au capsule) |
| `spandler_oneill2010.csv` | Spandler and O'Neill (2010), Contrib Mineral Petrol 159, 791-818, Table 2 (p. 799) | 3 (+Mn, Ni) | 1300 C, 1 bar, fO2 1e-3.3 Pa, Fo90.6, three axes |

Every row was transcribed from page images and checked a second time against them
(by the transcribing agent), and each file was hand-checked again against a fresh
render of its pages (Dohmen: 8 rows; Chakraborty, Petry, Holzapfel: every Fe-Mg row).
Derived columns (log10 fO2 in Pa, X_Fo as a fraction, log10 D) are computed from
the printed values, which are kept. Points of judgement are in each row's notes:
Petry's composition (Fo90, the San Carlos side; the paper gives none), Holzapfel's
fO2 (computed at Ni-NiO for each run's P and T in `fit_ols.py`), Spandler's [010]
and [001] values (ends of a fit against angle, not separate measurements), and
flagged rows (Holzapfel's 11 GPa runs; Chakraborty's errors of 50% or more).

Dohmen and Chakraborty (2007) Part II tabulates no diffusion coefficients; the
Meissner (2000) data it uses appear only in its Fig. 7b.
`dohmen_chakraborty2007_fig7b_extracted.csv` holds all 109 points of that figure,
read exactly from the PDF's vector graphics (axis map from 8 x and 5 y tick labels,
residuals < 6e-4; temperatures to about +-1 C), Meissner's 7 first (900-1200 C). The
caption says every point is normalised to 1e5 Pa, fO2 1e-7 Pa and Fo90, but checked
against our tables the series were treated differently: Petry matches that
(+-0.005 log), Chakraborty is normalised to Fo86 with no fO2 term, Part I's
high-fO2 points carry an extra +0.12, Part I's low-fO2 points are unnormalised, and
Holzapfel's could not be reproduced. Which treatment Meissner's points received is
unknown. These are normalised values; Meissner's own fO2, composition and
orientation are not recoverable.

## Scripts

- `fit_ols.py`: the least-squares reproduction of Mutch et al. (2019),
  Supplementary Tables 1-2. `uv run python analyses/femg_refit/fit_ols.py`.
- `fit_models.py`: the refits. `uv run --with odrpack python analyses/femg_refit/fit_models.py`
  writes `results/`: `coefficients.csv` (every form, method and parameter with
  its 1-sigma), `cov_<form>_<method>.csv` (parameter covariances) and
  `predictions.csv` (log10 D[001] at the round robin and paper conditions, and its
  difference from the current default law).
- `figures.py`: the four comparison figures below, from `results/` and the data.
  `uv run --with odrpack python analyses/femg_refit/figures.py`.
- `bowties.py`: an Arrhenius plot with confidence bands for every form and method
  (`figures/bowties/`, listed in `index.csv`).
  `uv run --with odrpack python analyses/femg_refit/bowties.py`.
- `fo2_figures.py`: Figs. 5-6 below, on fO2 and the disagreement between studies.
  `uv run --with odrpack python analyses/femg_refit/fo2_figures.py`.
- `export_preset.py`: writes the `femg_refit2026` preset (`dc07_free`, random study
  offset, with its covariance and between-study spread) for
  `src/forster/diffusivity.py`, and checks that the preset reproduces the refit's
  log10 D and its 1-sigma on every axis at 22 conditions (largest difference 7e-15)
  and equals the preset shipped there (otherwise it prints the values to paste and
  exits with status 1).
  `uv run --with odrpack python analyses/femg_refit/export_preset.py`.
- `compare_laws.py`: residuals of the published laws and our refits against the
  data, by study, and F tests of candidate missing terms.
  `uv run --with odrpack python analyses/femg_refit/compare_laws.py`.

Data used: the TaMED set, fO2 above 1e-10 Pa (135 of 166 points; Holzapfel's fO2
computed at Ni-NiO for each run's P and T; 1 atm and 1 bar rows both at 1e5 Pa).

### Law forms (ln D on each measurement's own axis)

| Form | Free parameters | Fixed |
|---|---|---|
| `mutch_x6` | a, b (fO2), c (X_Fo), q (1/T), j (P), h (P/T) | [100], [010] = [001]/6 |
| `mutch_aniso` | the six, plus the [100] and [010] offsets | |
| `dc07_fixed` | log D0, E | fO2 exponent 1/6, composition 3 (0.9 - X_Fo), V 7 cm3/mol, x6 anisotropy |
| `dc07_free` | log D0, E, fO2 exponent, composition coefficient, both offsets | V 7 cm3/mol |
| `mutch_fT` | `mutch_aniso` plus g ln fO2 / T (an fO2 exponent b + g/T that changes with T) | |

### Methods

- `ols`: unweighted least squares (Mutch et al.'s method), covariance s^2 (X'X)^-1.
- `wls`: weighted by the y errors below.
- `odr`: orthogonal distance regression with errors in log fO2, X_Fo, T and P as
  well (`odrpack`, the engine of Henry's `odr_bootstrap`, which fits one
  predictor; this needs five). A control with the predictor errors set near zero
  reproduces `wls` exactly.
- `bootstrap (runs)`: 2000 resamples of whole experimental runs (rows of one run,
  such as Chakraborty's several compositions per couple, move together), OLS each.
- `random study`: least squares with a random offset per study, u ~ N(0, tau^2),
  tau fitted by REML (profiled over tau^2/sigma^2 on a grid of 400 values; 95% range
  from the profile likelihood). Points of one study share their offset, so they are
  correlated and a study counts for less than its number of points (Chakraborty and
  Dohmen hold 99 of the 135). Covariance: the law averaged over labs.
  `predictions.csv` column `se_new_lab` adds tau for one new lab;
  `coefficients.csv` rows `tau_log10` (and `_lo95`, `_hi95`).
- `without <study>`: leave one study out, OLS; skipped where the remaining data
  cannot identify a form (no pressure terms without Holzapfel's 6-12 GPa runs).

Measurement errors (1 sigma) for `wls` and `odr`; stated in the paper or assumed:

| Study | log D | T | log fO2 | X_Fo | P |
|---|---|---|---|---|---|
| Chakraborty 1997 | printed % per row, at least 10% (stated) | 3 K (assumed) | 0.1 (stated) | 0.01 (assumed) | 0 |
| Dohmen et al. 2007 | 0.2 log (stated) | 2 K (stated) | 0.1 (stated) | 0.005 (assumed) | 0 |
| Petry et al. 2004 | 0.3 log (factor 2, stated for Ni) | 3 K (assumed) | 0.1 (assumed) | 0.01 (assumed) | 0 |
| Holzapfel et al. 2007 | 0.3 log (stated) | 25 K (assumed) | 0.5 (assumed; fO2 not measured) | 0.01 (assumed) | 5% (assumed) |
| Spandler & O'Neill 2010 | 0.1 log (0.04 stated; [010]/[001] from a fit against angle) | 3 K (assumed) | 0.1 (assumed) | 0.002 (stated) | 0 |

## Figures (`figures/`)

1. `fig1_predictions.png`: log10 D[001] at the round robin (KS20, 1200 C; SH63,
   1102 C) and paper (1250 C) conditions from every form and method, minus the
   current default (`dohmen_chakraborty2007`), with 1-sigma bars (the bootstrap
   rows use the OLS estimate with the bootstrap spread; the random study rows add a
   light bar for one new lab); the published Mutch et al. (2019) regressions for
   reference. The top axis is the fitted time as a
   multiple of the default's (time scales as 1/D).
2. `fig2_coefficients.png`: activation energy, fO2 exponent, composition term (log10
   D per unit X_Fe), activation volume at 1250 C (Mutch forms: V = -R (j T + h)) and
   anisotropy factors, per form (colour) and method (marker), with the Dohmen and
   Chakraborty (2007) and Mutch et al. (2019) TaMED values as lines.
3. `fig3_data.png`: the data moved to common conditions (fO2 1e-7 Pa, Fo90, 1 atm,
   [001]; and 1200 C) by the `dc07_free` OLS fit, so only that form's line is
   strictly comparable with the points; the other lines are each form's own
   prediction. Dotted lines: the fO2 of the round robin and paper conditions.
4. `fig4_residuals.png`: observed minus fitted log10 D for `dc07_free` and
   `mutch_aniso` (OLS), by study, against T and fO2.

5. `fig5_fo2.png`: (a) every Fe-Mg point moved to 1200 C, Fo90, 1 atm, [001] by the
   `dc07_free` OLS fit and plotted against fO2 (points below the TaMED cut of 1e-10
   Pa faded; light bars are the uncertainty of that move), with each form's line
   and the published DC07 and Mutch TaMED laws; (b) the fO2 exponent
   (d log D / d log fO2) from each study's own data (Dohmen's isothermal series,
   all of Dohmen with T and axis removed, Petry with T removed) against the joint
   fits and `mutch_fT`'s exponent at 900, 1200 and 1400 C.
6. `fig6_studies.png`: (a) where each study's TaMED points sit in T and fO2, with its
   method, composition and pressure in the legend and the round robin and paper
   conditions as stars; (b) each study's offset (centred, 1 sigma) when an offset per
   study is added to `dc07_fixed`, `dc07_free` and `mutch_aniso` (OLS).

Each figure passes `figcheck` (no overlaps or clipping) and its title's numbers
are computed from the data.

## Bowtie plots (`figures/bowties/`)

One figure per law form and method, in the style of `odr_bootstrap`'s
`plot_regression`: the best fit with 68% and 95% bands (quantiles of the predicted
log D over the bootstrap resamples, or over 4000 draws from the fit's covariance),
the data with their log D error bars, and the published Dohmen & Chakraborty (2007)
and Mutch et al. (2019, TaMED) laws. Panel (a) is at the centre of the data (fO2
1e-7 Pa, Fo90, 1 atm, [001]); panel (b) at the round robin KS20 conditions (fO2
1.4e-3 Pa, Fo85, 42 MPa, [001]). The points are moved to each panel's conditions by
that fit itself, so their scatter about the line is the fit's own residual. That
move has its own uncertainty, sqrt(dG' C dG) for the change dG in the design row and
the fit's covariance C; it is drawn as a light bar outside the measurement bar
(their quadrature sum), and points whose move is less certain (1 sigma 0.03-0.1
and over 0.1 log) are drawn fainter. In panel (b) the move adds 0.07-0.16
log (10th-90th percentile of the points) in the Mutch forms and 0.04-0.09 in
`dc07_free`, against Chakraborty's 0.04 measurement bars; none in `dc07_fixed`,
whose fO2, composition and pressure terms are fixed. It is shared with
the band (the same covariance), not independent scatter, so the bars show why the
points sit less certainly in (b), not extra error to add to the band. Under
each figure a paragraph says what is particular to that form and method, with its
fitted numbers.

| Figure | Form | Method | D at 1200 C, KS20, vs default (log) | 68% band at the data's centre | 68% band at KS20 fO2 |
|---|---|---|---|---|---|
| [mutch_x6__ols.png](figures/bowties/mutch_x6__ols.png) | mutch_x6 | ols | +0.37 | +-0.03 | +-0.08 |
| [mutch_x6__wls.png](figures/bowties/mutch_x6__wls.png) | mutch_x6 | wls | +0.53 | +-0.02 | +-0.16 |
| [mutch_x6__odr.png](figures/bowties/mutch_x6__odr.png) | mutch_x6 | odr | +0.85 | +-0.03 | +-0.12 |
| [mutch_x6__bootstrap.png](figures/bowties/mutch_x6__bootstrap.png) | mutch_x6 | bootstrap (runs) | +0.37 | +-0.06 | +-0.15 |
| [mutch_x6__random.png](figures/bowties/mutch_x6__random.png) | mutch_x6 | random study | +0.29 | +-0.14 | +-0.14 |
| [mutch_aniso__ols.png](figures/bowties/mutch_aniso__ols.png) | mutch_aniso | ols | +0.37 | +-0.04 | +-0.08 |
| [mutch_aniso__wls.png](figures/bowties/mutch_aniso__wls.png) | mutch_aniso | wls | +0.45 | +-0.02 | +-0.16 |
| [mutch_aniso__odr.png](figures/bowties/mutch_aniso__odr.png) | mutch_aniso | odr | +0.81 | +-0.03 | +-0.12 |
| [mutch_aniso__bootstrap.png](figures/bowties/mutch_aniso__bootstrap.png) | mutch_aniso | bootstrap (runs) | +0.37 | +-0.06 | +-0.14 |
| [mutch_aniso__random.png](figures/bowties/mutch_aniso__random.png) | mutch_aniso | random study | +0.29 | +-0.13 | +-0.14 |
| [mutch_fT__ols.png](figures/bowties/mutch_fT__ols.png) | mutch_fT | ols | +0.39 | +-0.03 | +-0.08 |
| [mutch_fT__wls.png](figures/bowties/mutch_fT__wls.png) | mutch_fT | wls | +0.46 | +-0.02 | +-0.16 |
| [mutch_fT__odr.png](figures/bowties/mutch_fT__odr.png) | mutch_fT | odr | +0.83 | +-0.03 | +-0.12 |
| [mutch_fT__bootstrap.png](figures/bowties/mutch_fT__bootstrap.png) | mutch_fT | bootstrap (runs) | +0.39 | +-0.06 | +-0.13 |
| [mutch_fT__random.png](figures/bowties/mutch_fT__random.png) | mutch_fT | random study | +0.30 | +-0.15 | +-0.15 |
| [dc07_fixed__ols.png](figures/bowties/dc07_fixed__ols.png) | dc07_fixed | ols | +0.10 | +-0.02 | +-0.02 |
| [dc07_fixed__wls.png](figures/bowties/dc07_fixed__wls.png) | dc07_fixed | wls | -0.01 | +-0.02 | +-0.02 |
| [dc07_fixed__odr.png](figures/bowties/dc07_fixed__odr.png) | dc07_fixed | odr | +0.01 | +-0.02 | +-0.02 |
| [dc07_fixed__bootstrap.png](figures/bowties/dc07_fixed__bootstrap.png) | dc07_fixed | bootstrap (runs) | +0.10 | +-0.04 | +-0.04 |
| [dc07_fixed__random.png](figures/bowties/dc07_fixed__random.png) | dc07_fixed | random study | +0.20 | +-0.12 | +-0.12 |
| [dc07_free__ols.png](figures/bowties/dc07_free__ols.png) | dc07_free | ols | +0.15 | +-0.04 | +-0.03 |
| [dc07_free__wls.png](figures/bowties/dc07_free__wls.png) | dc07_free | wls | +0.18 | +-0.02 | +-0.08 |
| [dc07_free__odr.png](figures/bowties/dc07_free__odr.png) | dc07_free | odr | +0.30 | +-0.02 | +-0.07 |
| [dc07_free__bootstrap.png](figures/bowties/dc07_free__bootstrap.png) | dc07_free | bootstrap (runs) | +0.15 | +-0.06 | +-0.04 |
| [dc07_free__random.png](figures/bowties/dc07_free__random.png) | dc07_free | random study | +0.27 | +-0.14 | +-0.12 |

Reading them: `dc07_fixed`, which holds the fO2 exponent at the point-defect value,
keeps the band the same width at the KS20 fO2 as at the data's centre, because the
model, not the data, sets how D scales with fO2 (its band is conditional on that
model). Forms that fit the exponent widen when extrapolated there: the Mutch forms
2.2-7.3x; `dc07_free` 3.0-3.4x when weighted or by ODR, but not by OLS or the run
bootstrap (0.7-0.8x), where its log D0, defined at the reference fO2 and Fo90, is
anticorrelated with the exponent. With a random study offset every form's band is
about the same width at both (0.9-1.0x, +-0.12-0.15 log): the spread between studies
sets the level everywhere and outweighs the extrapolation in fO2. Those figures also
draw a dashed 68% line for one new lab (the band plus tau).
