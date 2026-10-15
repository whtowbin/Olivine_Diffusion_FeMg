# Data to source directly

What the Fe-Mg analysis uses or would use that is not yet confirmed from a source paper's own tables or text, or that lacks a detail the fit needs. For each item: what is missing, what would supply it, and what it changes. Started 2026-10-14, when the `femg_refit2026` preset was refitted to the verified data only. Updated since: the preset fixes the anisotropy at 6x (as DC07, DIPRA and DFENS), Oeser et al. (2026) moved to the unconfirmed scenarios, and Holzapfel et al.'s Au-capsule runs entered the preset with their unconfirmed fO2 carried as an uncertainty (item 3).

## The rule

A row is **verified** when every value the fit uses is stated in the source's tables or text: T, P, fO2 (measured, or buffered at a named buffer), composition, the diffusion direction, and D. A statement that covers the whole paper counts (decided 2026-10-14). An example is "measured along [001]" for Chakraborty (1997), even though no table gives the orientation run by run.

A row read from a figure, inferred, assumed, or estimated is **unconfirmed**. Such rows stay out of the preset, with one kind of exception: a row whose only unconfirmed value can be carried in the fit with an honest uncertainty (decided 2026-10-14 for Holzapfel's Au-capsule fO2, at Ni-NiO with a sd of 2 log). They can enter variants and scenarios, labelled as such. Values read from figures are also logged, with how they were read, in `figure-extracted-data.md`.

## In the preset (verified)

`analyses/femg_refit/fit_verified.py`, `preset_data()`: the 118 verified points of `verified()` plus Holzapfel's 14 Au-capsule runs, 132 points from 4 studies, in the Dohmen and Chakraborty (2007) form with D[100] = D[010] = D[001]/6. Fitted by `fit_preset`: REML with a random offset per study, each run's errors in T (at least 10 C), fO2, X_Fo and P carried as extra variance.

| Study | Points | Caveat (and what would remove it) |
|---|---|---|
| Chakraborty (1997), Table 3 | 55 | [001] is stated for the paper ("perpendicular to [001] for most runs"), not per run. **To source:** each run's orientation (Chakraborty's thesis, or the author). |
| Dohmen et al. (2007) Part I, Table 3, fO2 > 1e-10 Pa | 44 | D as tabulated, already normalised by the authors to Fo86 with a composition exponent of 3. These points cannot constrain the composition term. |
| Petry et al. (2004), Fe-Mg | 16 | The axis comes from the text (other parameters "determined in experiments where diffusion occurred along [001]"), not from a table. **To source:** each run's axis (Petry 1999, dissertation). |
| Holzapfel et al. (2007), Ni-NiO capsules | 3 | Buffered at Ni-NiO, so the fO2 is known. |
| Holzapfel et al. (2007), Au capsules | 14 | **fO2 unconfirmed** (item 3): taken at Ni-NiO with a sd of 2 log units, carried as extra variance (ln10 m 2)^2 on their ln D. Everything else is stated. |

## Not in the preset (to source)

The table is in order of how much each item could change the result.

| # | Data | What is missing or unconfirmed | Source that would settle it | What it changes now | Priority |
|---|---|---|---|---|---|
| 1 | **Shea et al. (2023)**, all 26 profiles | Orientation and Fo of each profile. We infer an axis for 8 rows from Figs 2-4, and the Fo is read by eye. | Shea et al. (2023) supplementary table: traverse-to-axis angles and the profile data. | All 26 profiles could be fitted, not just 8. Our 8 run about 0.17 log above Shea's full set, so the melt term estimated with Shea is biased high. | high |
| 2 | **Jurewicz and Watson (1988)** | Everything: so far read from Shea et al.'s Figs 5-6, which do not agree with each other (18 against 19 points). | Jurewicz and Watson (1988), Contrib Mineral Petrol 99, 186-201, tables. | A third melt-contact study (+0.71 against DC07 as read). | high |
| 3 | **Holzapfel et al. (2007)**, Au-capsule runs | fO2 was not buffered; the authors estimate "close to or slightly above Ni-NiO". | No measured value exists. Either a better estimate (from the authors, or a model of fO2 in Au capsules at 6-12 GPa), or keep the wide prior. | In the preset at Ni-NiO with a sd of 2 log. With a sd of 0.5, 1, 2 or 3 log: KS20 +0.17, +0.16, +0.155 and +0.15 against DC07; without them +0.13 (`results/verified/variants.csv`). | medium |
| 4 | **Oeser et al. (2026)**, all runs | Their tables are verified. But their [010] is 1.7-3.2x slower than [001] in every run, against 6x in Dohmen et al. at the same temperatures, which the preset keeps (decided 2026-10-14). Their level also rises above the other studies with temperature, on every axis. Runs 7, 14 and 15 are linescans, convolution corrected with an assumed 7.5 um beam, and sit highest. | Another solid-source study with all three axes at 1150-1300 C, to settle [010]; for runs 7, 14 and 15, the raw linescans (the paper's data repository). | Added to the preset's data: KS20 +0.26 (all depth profiles), +0.21 ([001] and [100] only), +0.22 (with the linescans), against +0.155. Their own fO2 exponent ([001], 0.20 ± 0.05) equals the preset's, so they move the level, not the slope (`unconfirmed-data.md`). | medium |
| 5 | **Spandler and O'Neill (2010)** | Melt contact, so excluded by decision. Also, two of its three Fe-Mg values ([010], [001]) are ends of a fitted line in log D against angle, not measurements. | Their electronic supplement, if it gives the individual profiles by angle. | As a melt term against the preset: +0.57 ± 0.18 (+0.82 ± 0.18 with Shea's 8 profiles); `unconfirmed-data.md`. | medium |
| 6 | **Shea, Nizam and Helz (2026)**, Kilauea Iki lava lake | A population-model multiplier (5-15x DC07, a lower bound), not D values. fO2 and the temperature-time paths are in the supplement. | Supplement doi:10.1130/GEOL.S.34036920. | A natural bound on the melt term, near SH63 temperatures. | medium |
| 7 | **Meissner (2000)**, 7 points read from DC07 Fig 7b | Unpublished (dissertation); read from a figure. | Meissner (2000), Bayreuth dissertation; Meissner et al. (1998), Am Mineral 83, 546-552. | At most 0.007 log at KS20 (`docs/plans/2026-10-13-femg-refit.md`). | low |
| 8 | **Mutch et al. (2019)**, the compiled experiments behind their regressions | Their coefficients and covariances are now sourced (item settled 2026-10-15, below). The rows they regressed are not published, and our OLS on the same six studies reproduces them only to 0.03-1.8 of their standard errors (our fO2 exponent 0.258 against their 0.224 for TaMED). Meissner's points do not close the gap. | The compiled table, from the authors; or a reading of the calibration figure in their 2019 supplement (figure-extracted). | Only a refit of the Mutch form needs it. The Mutch presets use the published fit as it stands. | low |
| 9 | **Chakraborty (1997)** and **Petry (2004)**, per-run orientation | See the verified table above. | As above. | If a run was not along [001], its D shifts by up to log 6 (0.78). | low (the authors' paper-wide statements) |

## Settled

- **Mutch et al. (2019) coefficients and covariances (2026-10-15).** Full precision in Mutch et al. (2021), DFENS, Data Set S6 (`analyses/femg_refit/data/mutch2021_dfens_ds06.xlsx`; transcribed by `transcribe_mutch_ds06.py`). Both Fe-Mg matrices are positive definite; the printed TaMED matrix was not only because of rounding. The `mutch2019_tamed`, `mutch2019_global`, `mutch2019_ni` and `mutch2019_mn` presets carry them.

Not yet read: Rawlings and Mutch (2026), "Olivine diffusion chronometry: recent developments and future directions", which is in Zotero. It may point to further datasets.

## How unconfirmed data enter

- **Preset:** verified rows, plus the Au-capsule runs with their fO2 uncertainty (`fit_verified.preset_data()`, `fit_preset`). `export_preset.py` builds the preset from them and exits 1 unless the shipped preset equals it.
- **Variants:** unconfirmed rows added one set at a time, each with its assumption stated (`results/verified/variants.csv`).
- **Scenarios:** melt contact, and Oeser et al.'s anisotropy and high-temperature level, kept outside the law's uncertainty (`unconfirmed-data.md`).
- **When an item is sourced:** move it to the verified table, add it to `verified()` (or narrow its uncertainty in `fit_preset`), re-export the preset, and record the change in the CHANGELOG.
