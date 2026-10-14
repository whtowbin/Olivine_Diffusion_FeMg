# Data read from figures, to confirm against the papers' text and tables

Every value in this repo that was read from a published figure rather than from a table or
the text. Each entry gives the source figure, how the values were read, the file, where they
are used, and what to check once the paper's own numbers are in hand. When an entry is
confirmed, record the date and what was checked under its status; do not delete it.

Started 2026-10-14.

## Read from vector graphics

Marker centres are read from the PDF's drawing commands and mapped through a linear fit to
the labelled ticks, so the reading error is the axis-map residual (given per entry), not a
pixel estimate. What a figure cannot give is anything it does not plot: conditions, which
run a point belongs to, and how the plotted values were normalised.

### 1. Dohmen and Chakraborty (2007) Part II, Fig. 7b: 109 points

-   **File:** `analyses/femg_refit/data/dohmen_chakraborty2007_fig7b_extracted.csv`.
-   **Series:** Chakraborty 1997 (48), Dohmen et al. 2007 Part I (14 at fO2 > 1e-10 Pa, 9 below), Petry et al. 2004 (17), Holzapfel et al. 2007 (14), Meissner 2000 (7, unpublished).
-   **Method:** pdfplumber, PDF page 17; axis map from 8 x and 5 y labelled ticks, residual < 4e-4 units.
-   **Used in:** `analyses/femg_refit/compare_laws.py`; the Meissner sensitivity test in `docs/plans/2026-10-13-femg-refit.md` (adding the 7 points moves D at KS20 by at most 0.007 log).
-   **To confirm:**
    -   Meissner's 7 points against Meissner (2000), Bayreuth dissertation (TEM-EDX). It is unpublished, so this may not be possible; Meissner et al. (1998, Am Mineral 83, 546-552) may hold some of them.
    -   The other series against the tables already transcribed. The figure plots them normalised (to fO2 1e-7 Pa and Fo90 or Fo86 by the caption), and the refit plan records that the normalisation is not consistent between series.
-   **Status:** not confirmed.

### 2. Jurewicz and Watson (1988) Fe-Mg, from Shea et al. (2023) Figs 5 and 6

-   **File:** `analyses/melt_contact/data/shea2023_figs_extracted.csv` (`source` = `jurewicz_watson1988`), written by `analyses/melt_contact/extract_shea2023_figs.py` from the accepted manuscript (`1-s2.0-S0012821X23003837-am.pdf`).
-   **What was read:**
    -   Fig. 5 Fe-Mg panel (PDF page 23): 18 grey squares, T and log D. The axis is 1000/T with T in °C, as Shea et al. label it. The points sit in two columns that read as 1219 and 1298 °C.
    -   Fig. 6 Fe-Mg panel (PDF page 24): 19 grey squares, log D measured against log D predicted by DC07. The residual against DC07 is +0.71 ± 0.19 log.
-   **Method:** pymupdf. Fig. 5 axis-map residuals are 0.0004 (x, 3 ticks) and 0.00006 log (y, 3 ticks). Fig. 6 has 0.002 log on y (3 ticks), but its x axis has only 2 labelled ticks, so that map is unchecked.
-   **Used in:** `analyses/melt_contact/figures/fig3_offsets.png`; `docs/references/melt-contact-diffusion.md`. Not used in any fit.
-   **To confirm against Jurewicz and Watson (1988), Contrib Mineral Petrol 99, 186-201:**
    -   Temperatures: nominal values against the 1219 and 1298 °C read here.
    -   Each log D, and which run it belongs to.
    -   fO2, olivine composition and profile orientation. These are needed to put the points on the bowtie and fO2 figures or into a fit.
    -   **Why the two figures differ.** Fig. 5 has 18 points and Fig. 6 has 19. Fig. 5 has two high values (−13.80, −14.31) absent from Fig. 6, and many measured values differ by 0.03-0.1 log between the figures. Different subsets or normalisations are possible.
    -   The conditions Shea et al. used for the DC07 prediction in Fig. 6.
-   **Status:** not confirmed; the paper is not in hand.

### 3. Shea et al. (2023) Fe-Mg points, Fig. 6

-   **File:** the same CSV (`source` = `shea2023`): 20 circles. The legend swatches are excluded; with them the mean was wrongly +0.83.
-   **What was read:** log D measured against log D predicted by DC07. The residual is +0.96 ± 0.17 log, which agrees with the paper's "about 10x".
-   **Used in:** `fig3_offsets.png`, as Shea's full-set offset beside the 8 profiles with a known axis (+1.13).
-   **To confirm:**
    -   Match each point to its Table 1 row. Table 1 has 23 Fe-Mg values and Fig. 6 shows 20, so 3 are missing or hidden behind others.
    -   The orientation and Fo that Shea et al. used for each prediction (their supplementary table).
-   **Status:** not confirmed.

## Read by eye or from figure annotations

### 4. Shea et al. (2023) Figs 2-4: axes, angles, Fo and modelled times

-   **File:** `analyses/melt_contact/data/shea2023_table1.csv`, columns `axis`, `angle_to_c_deg`, `axis_basis`, `X_Fo`, `X_Fo_basis` and `t_model_*_h`.
-   **What was read:**
    -   **From the figures' text** (exact as printed): the modelled times, the profile angles (58°, 86°, 89°) and the orientations ("parallel to b/c/a").
    -   **By eye from the plotted axes** (about ±0.5 mol%): the Fo range of each profile; X_Fo is its midpoint.
    -   **Inferred, not printed:** which Table 1 row each figure profile is. Fig. 2b is taken as ol18 row 4, and Fig. 3 as ol20 rows 7 and 8, from time and D ratios.
    -   **Assumed:** a or b for the profiles at 86° and 89° (taken as [100]).
-   **Used in:** the melt-term fits (`melt_sensitivity.py`, Shea's 8 profiles with a known axis), `fig1_bowties.png` and `fig2_fo2.png`.
-   **To confirm:** against Shea et al.'s supplementary table, which gives the traverse-to-axis angles and the profile data. In particular: the Fo of each profile, the a/b choice, and the row assignments.
-   **Status:** not confirmed; the supplement is not in hand.

### 5. Spandler and O'Neill (2010): Fe-Mg along [010] and [001]

-   **File:** `analyses/femg_refit/data/spandler_oneill2010.csv`.
-   **What was read:** the [010] and [001] values are the two ends (0° and 90°) of the fit line printed in their Fig. 6, log D = 0.0043 × angle − 15.21. They are not separate measurements.
-   **Used in:** every refit, as 2 of Spandler's 3 Fe-Mg rows.
-   **To confirm:** whether their supplement tabulates the individual profiles by angle.
-   **Status:** not confirmed.
