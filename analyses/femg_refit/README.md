# Fe-Mg diffusivity refit

Experimental Fe-Mg diffusion coefficients in olivine, compiled to refit the
diffusivity law with a full parameter covariance. Plan and results:
`docs/plans/2026-10-13-femg-refit.md`.

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

## Scripts

- `fit_ols.py`: the least-squares reproduction of Mutch et al. (2019),
  Supplementary Tables 1-2. `uv run python analyses/femg_refit/fit_ols.py`.
- `fit_models.py`: the refits. `uv run --with odrpack python analyses/femg_refit/fit_models.py`
  writes `results/`: `coefficients.csv` (every form, method and parameter with
  its 1-sigma), `cov_<form>_<method>.csv` (parameter covariances) and
  `predictions.csv` (log10 D[001] at the round robin and paper conditions, and its
  difference from the current default law).
- `figures.py`: the four figures below, from `results/` and the data.
  `uv run --with odrpack python analyses/femg_refit/figures.py`.

Data used: the TaMED set, fO2 above 1e-10 Pa (135 of 166 points; Holzapfel's fO2
computed at Ni-NiO for each run's P and T; 1 atm and 1 bar rows both at 1e5 Pa).

### Law forms (ln D on each measurement's own axis)

| Form | Free parameters | Fixed |
|---|---|---|
| `mutch_x6` | a, b (fO2), c (X_Fo), q (1/T), j (P), h (P/T) | [100], [010] = [001]/6 |
| `mutch_aniso` | the six, plus the [100] and [010] offsets | |
| `dc07_fixed` | log D0, E | fO2 exponent 1/6, composition 3 (0.9 - X_Fo), V 7 cm3/mol, x6 anisotropy |
| `dc07_free` | log D0, E, fO2 exponent, composition coefficient, both offsets | V 7 cm3/mol |

### Methods

- `ols`: unweighted least squares (Mutch et al.'s method), covariance s^2 (X'X)^-1.
- `wls`: weighted by the y errors below.
- `odr`: orthogonal distance regression with errors in log fO2, X_Fo, T and P as
  well (`odrpack`, the engine of Henry's `odr_bootstrap`, which fits one
  predictor; this needs five). A control with the predictor errors set near zero
  reproduces `wls` exactly.
- `bootstrap (runs)`: 2000 resamples of whole experimental runs (rows of one run,
  such as Chakraborty's several compositions per couple, move together), OLS each.
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
   rows use the OLS estimate with the bootstrap spread); the published Mutch et
   al. (2019) regressions for reference. The top axis is the fitted time as a
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

Each figure passes `figcheck` (no overlaps or clipping) and its title's numbers
are computed from the data.
