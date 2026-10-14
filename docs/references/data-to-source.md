# Data to source directly

What the Fe-Mg analysis uses or would use that is not yet confirmed from a source paper's own tables or text, or that lacks a detail the fit needs. For each item: what is missing, what would supply it, and what it changes. Started 2026-10-14, when the `femg_refit2026` preset was refitted to the verified data only.

## The rule

A row is **verified** when every value the fit uses is stated in the source's tables or text: T, P, fO2 (measured, or buffered at a named buffer), composition, the diffusion direction, and D. A statement that covers the whole paper counts (decided 2026-10-14). An example is "measured along [001]" for Chakraborty (1997), even though no table gives the orientation run by run.

A row read from a figure, inferred, assumed, or estimated is **unconfirmed**. Such rows stay out of the preset. They can enter variants and scenarios, labelled as such. Values read from figures are also logged, with how they were read, in `figure-extracted-data.md`.

## In the preset (verified)

`analyses/femg_refit/fit_verified.py`, `verified()`: 153 points from 5 studies.

| Study | Points | Caveat (and what would remove it) |
|---|---|---|
| Chakraborty (1997), Table 3 | 55 | [001] is stated for the paper ("perpendicular to [001] for most runs"), not per run. **To source:** each run's orientation (Chakraborty's thesis, or the author). |
| Dohmen et al. (2007) Part I, Table 3, fO2 > 1e-10 Pa | 44 | D as tabulated, already normalised by the authors to Fo86 with a composition exponent of 3. These points cannot constrain the composition term. |
| Petry et al. (2004), Fe-Mg | 16 | The axis comes from the text (other parameters "determined in experiments where diffusion occurred along [001]"), not from a table. **To source:** each run's axis (Petry 1999, dissertation). |
| Holzapfel et al. (2007), Ni-NiO capsules | 3 | Buffered at Ni-NiO, so the fO2 is known. |
| Oeser et al. (2026), Tables 2-3, depth profiles | 35 | D is from the authors' multicomponent model, for X_Fe = 0.085, as their figure captions state. |

## Not in the preset (to source)

The table is in order of how much each item could change the result.

| # | Data | What is missing or unconfirmed | Source that would settle it | What it changes now | Priority |
|---|---|---|---|---|---|
| 1 | **Shea et al. (2023)**, all 26 profiles | Orientation and Fo of each profile. We infer an axis for 8 rows from Figs 2-4, and the Fo is read by eye. | Shea et al. (2023) supplementary table: traverse-to-axis angles and the profile data. | All 26 profiles could be fitted, not just 8. Our 8 run about 0.17 log above Shea's full set, so the melt term estimated with Shea is biased high. | high |
| 2 | **Jurewicz and Watson (1988)** | Everything: so far read from Shea et al.'s Figs 5-6, which do not agree with each other (18 against 19 points). | Jurewicz and Watson (1988), Contrib Mineral Petrol 99, 186-201, tables. | A third melt-contact study (+0.71 against DC07 as read). | high |
| 3 | **Holzapfel et al. (2007)**, Au-capsule runs | fO2 was not buffered; the authors estimate "close to or slightly above Ni-NiO". | No measured value exists. Either a better estimate (from the authors, or a model of fO2 in Au capsules at 6-12 GPa), or keep the wide prior. | Fitted at Ni-NiO with a prior sd of 0, 1, 2 or 3 log: KS20 +0.23, +0.23, +0.22 and +0.21 against DC07, against +0.20 for the preset (`results/verified/variants.csv`). | medium |
| 4 | **Oeser et al. (2026)**, linescan runs 7, 14 and 15 | Sourced, but convolution corrected with an assumed 7.5 um beam. They sit about 0.7 log above the depth-profile runs. | The raw linescan profiles (the paper's online data repository), to redo the deconvolution. | Added to the verified data: E rises to 240 kJ/mol and m falls to 0.149, and KS20 moves to +0.13 (from +0.20; `results/verified/scenarios.csv`). | medium |
| 5 | **Spandler and O'Neill (2010)** | Melt contact, so excluded by decision. Also, two of its three Fe-Mg values ([010], [001]) are ends of a fitted line in log D against angle, not measurements. | Their electronic supplement, if it gives the individual profiles by angle. | As a melt term against the verified law: +0.46 ± 0.29 (+0.73 ± 0.24 with Shea's 8 profiles); `unconfirmed-data.md`. | medium |
| 6 | **Shea, Nizam and Helz (2026)**, Kilauea Iki lava lake | A population-model multiplier (5-15x DC07, a lower bound), not D values. fO2 and the temperature-time paths are in the supplement. | Supplement doi:10.1130/GEOL.S.34036920. | A natural bound on the melt term, near SH63 temperatures. | medium |
| 7 | **Meissner (2000)**, 7 points read from DC07 Fig 7b | Unpublished (dissertation); read from a figure. | Meissner (2000), Bayreuth dissertation; Meissner et al. (1998), Am Mineral 83, 546-552. | At most 0.007 log at KS20 (`docs/plans/2026-10-13-femg-refit.md`). | low |
| 8 | **Mutch et al. (2019)** covariances | Only the rounded matrices are printed; TaMED's is not positive definite. | The full-precision `D_vcov` files (DFENS Data Set S6), or the authors. | The Mutch presets carry no covariance. Our own preset does not depend on these. | low |
| 9 | **Chakraborty (1997)** and **Petry (2004)**, per-run orientation | See the verified table above. | As above. | If a run was not along [001], its D shifts by up to log 6 (0.78). | low (the authors' paper-wide statements) |

Not yet read: Rawlings and Mutch (2026), "Olivine diffusion chronometry: recent developments and future directions", which is in Zotero. It may point to further datasets.

## How unconfirmed data enter

- **Preset:** verified rows only (`fit_verified.verified()`). `export_preset.py` builds the preset from them and exits 1 unless the shipped preset equals it.
- **Variants:** unconfirmed rows added one set at a time, each with its assumption stated (`results/verified/variants.csv`).
- **Scenarios:** melt contact, kept outside the law's uncertainty (`melt-contact-diffusion.md`).
- **When an item is sourced:** move it to the verified table, add it to `verified()`, re-export the preset, and record the change in the CHANGELOG.
