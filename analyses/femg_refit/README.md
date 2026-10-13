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
