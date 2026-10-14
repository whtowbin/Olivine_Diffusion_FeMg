# What the unconfirmed data suggest, and where the analysis goes next

Written 2026-10-14. On that date the `femg_refit2026` preset was refitted to the verified data only. "Verified" means every value is stated in the source's tables or text (`data-to-source.md`). This note covers everything else:
- what each unconfirmed set would do if it were confirmed;
- the melt-contact evidence, which the preset leaves out;
- what each missing source would let us do.

Numbers: `analyses/femg_refit/results/verified/{variants,scenarios}.csv`. Figures: `analyses/femg_refit/unconfirmed_figures.py`. Log units throughout. "KS20" is D[001] at the round-robin conditions (1200 C, fO2 1.4e-3 Pa, Fo85, 42 MPa), given relative to the default law (DC07).

## The verified law

| | Points | Studies | KS20 vs DC07 | 1 sigma, law / one new lab | tau | fO2 exponent | E (kJ/mol) |
|---|---|---|---|---|---|---|---|
| verified, `femg_refit2026` from 2026-10-14 | 153 | 5 | +0.20 | 0.114 / 0.261 | 0.235 | 0.221 | 213 |
| previous preset (all data, Spandler as a study) | 135 | 5 | +0.27 | 0.122 / 0.282 | 0.254 | 0.221 | 209 |

**What changed:**
- **Oeser et al.'s depth-profile runs came in.** That lowered [010]'s offset from [001]: [010] is now 3.5x slower than [001], not 5.3x.
- **Spandler and O'Neill (melt contact) and Holzapfel's 14 Au-capsule runs went out.**
- **The fO2 exponent did not move.** The law sits 0.07 log lower at KS20, and 0.09 lower at SH63 (+0.14). For a fit with this preset, times come out about 1.2x longer than before.

![The verified law and the unconfirmed data](../../analyses/femg_refit/figures/unconfirmed/u1_law_and_data.png)

**Fig. U1.** The verified law at the KS20 conditions, drawn as log D minus DC07. Bands are 68% and 95% of the law; dashed lines add tau for one new laboratory.
- **Points:** solid points are verified and fitted. Faded points are unconfirmed and not fitted; each is moved to the KS20 conditions by the verified law.
- **Holzapfel's Au runs** carry a bar that includes a 2-log fO2 prior.
- **Pink boxes:** melt-contact evidence that cannot be placed point by point (mean ± sd against DC07 at each study's own conditions, spanning its temperatures). For the lava lake, the arrow marks a lower bound.

![Where each unconfirmed addition would move the law](../../analyses/femg_refit/figures/unconfirmed/u2_variants.png)

**Fig. U2.** The KS20 and SH63 levels of the verified law and of each addition. Thick bars are the law's 1 sigma; thin bars are one new laboratory. The melt-term rows predict for solid contact; their melt offset is in brackets.

![The verified law against fO2](../../analyses/femg_refit/figures/unconfirmed/u3_fo2.png)

**Fig. U3.** The verified law against fO2 at 1200 C. Points are moved in T, composition, P and axis (not fO2) by the law. Holzapfel's Au runs are drawn at their assumed fO2 (Ni-NiO), with ±2 log bars. The shaded span is the round robin and paper fO2.

## What each unconfirmed set suggests

### Unconfirmed solid-source data barely move the law: 0.07 log at most

- **Holzapfel et al. (2007), Au capsules (14 runs).** Their fO2 is assumed to be Ni-NiO.
  - **The level barely moves.** At KS20 the law goes to +0.23 with an fO2 prior sd of 0, and +0.21 with sd 3.
  - **tau falls** from 0.235 to between 0.18 (sd 0) and 0.22 (sd 3). The 3 buffered runs sit low against the law (Holzapfel's study offset is -0.33), and the Au runs pull that offset toward the others.
  - **Reading:** either the 3 buffered runs are low, or the Au runs were somewhat more oxidised than Ni-NiO, as the authors suggest ("close to or slightly above"). The two cannot be told apart without a measured fO2.
  - In Fig. U3 the Au runs lie on the law's extrapolation at 10^1 to 10^4 Pa within their bars. Even at face value, the high-pressure data agree with the 1-atm fO2 exponent.
- **Oeser et al. (2026), linescan runs 7, 14 and 15.**
  - **They pull the law down:** KS20 +0.13, through E 240 kJ/mol and an fO2 exponent of 0.149. The rest of the data give 213 and 0.221.
  - **They are the only points that disagree with Oeser's own depth profiles.** They sit about 0.7 log above them at the same temperatures.
  - **Two readings.** If the convolution correction is right, they are the start of the high-temperature regime change Oeser et al. propose: a higher E, and a weaker fO2 dependence above about 1250 C. If it is not, they are an artefact.

### Melt contact is a separate offset of about +0.5 to +0.8 log, not a move of the law

Spandler and O'Neill (2010) are excluded from the preset, and Shea et al. (2023) cannot yet be fitted beyond 8 profiles. Every melt-contact source sits well above the verified law (Fig. U1, pink).

| Source | Against DC07, own conditions | Against the verified law | How it was obtained |
|---|---|---|---|
| Spandler and O'Neill (2010), 1300 C | +0.73 | +0.54; fitted melt term +0.46 ± 0.29 | Table 2 (two of three values are ends of a fitted line) |
| Jurewicz and Watson (1988), 1219-1298 C | +0.71 ± 0.19 | about +0.55 | read from Shea et al.'s Fig 6 |
| Shea et al. (2023), all 20, 1200-1290 C | +0.96 ± 0.17 | about +0.81 | read from Fig 6 |
| Shea et al. (2023), 8 profiles with a known axis | +1.13 | melt term with Spandler: +0.73 ± 0.24 | Table 1 D; axes inferred from Figs 2-4 |
| Kilauea Iki lava lake, about 1090-1150 C | +0.70 to +1.18 (5-15x), lower bound | at least +0.6 to +1.1 (assuming fO2 near QFM) | Shea, Nizam and Helz (2026), text |

**Notes on possible melt effects, from Shea et al. (2023) and what followed:**

- **It is a level, not a slope.**
  - The offset is about the same on every axis, and the anisotropy stays DC07-like.
  - It is about the same for Fe-Mg, Ni and Mn (x10, x7.5 and x9 in Shea et al.; +1.12, +1.01 and +1.00 against the laws they used).
  - Separating it from the solid data changes the fitted fO2 exponent by 0.01 or less.
  - So it belongs as one log offset common to axes and elements, applied only to crystals in melt contact (`melt-contact-diffusion.md`).
- **The proposed cause is extra M-site vacancies made at the melt interface** by Al or H (their Eq. 4: D = X_V D_V). Two other causes were tested and rejected: aSiO2 is not higher in the basalt than in the enstatite-buffered experiments, and the TEM and EBSD found no extended defects.
  - The cause is untested. It predicts that the offset shrinks where the fO2-dependent vacancies dominate (higher fO2 and T), which no experiment has yet checked.
- **It is not a laboratory artefact.** The Kilauea Iki olivine needs 5-15x DC07 over 29 years of natural re-equilibration, with no experimental interface. That is the same size as the experiments, at temperatures below the ~1150 C regime change Oeser et al. report.
- **Size for a fitted time,** for a crystal in melt contact against the verified law:
  - +0.46 (Spandler) → time x0.35;
  - +0.55 (Jurewicz and Watson) → x0.28;
  - +0.73 to +0.81 (Shea) → x0.19 to x0.15.

  The spread between sources (about 0.35 log) is larger than any one of their errors.
- **How forster should carry it, for now:** a named, one-sided scenario reported next to the solid-contact time. It should not be in the preset's law, and not in tau. A symmetric tau would widen the slow side, for which there is no evidence.

## Where the analysis goes when each source is in hand

1. **Shea et al. (2023) supplementary table** (orientation and Fo of all 26 profiles).
   - Fit all 26 instead of 8, and re-estimate the melt term with Spandler. The 8 run about 0.17 above the full set, so expect the term to fall toward +0.55-0.6.
   - Ni and Mn are in the same table, so this enables the joint Fe-Mg/Ni/Mn fit with a shared melt term.
2. **Jurewicz and Watson (1988).**
   - A third melt study, fitted with its own conditions (fO2, Fo, orientation).
   - A melt term from three studies gets a meaningful spread. Spread among melt studies, rather than one offset, is then the honest uncertainty for the scenario.
3. **Shea, Nizam and Helz (2026) supplement** (fO2, temperature-time paths, crystal data).
   - Run forster's forward model on the lava-lake temperature paths with the verified law plus a melt term.
   - That turns the "5-15x, lower bound" into a likelihood on the melt term from natural olivine near SH63 temperatures.
4. **Oeser et al. (2026) raw linescans.**
   - Redo the deconvolution (D = D_measured - sigma^2/2t with the beam width checked, not assumed).
   - If runs 7, 14 and 15 survive, they are evidence for a regime change above about 1250 C. Then the paper conditions (1250 C) need a temperature-dependent anisotropy or E, as `docs/plans/2026-10-14-oeser-additive.md` sets out.
5. **Holzapfel et al. (2007) Au-capsule fO2.**
   - A better estimate would let the 14 runs in; Fig. U2 shows they matter little at KS20.
   - Until then the wide-prior variant is the sensitivity test.
6. **Per-run orientations** (Chakraborty 1997; Petry 2004), **Meissner (2000)**, and the **full-precision Mutch et al. covariances**: confirmation of what is already in, or of reference presets. Each moves KS20 by less than 0.01, unless a run turns out to be off [001].

## Other data that would change the result most

- **Solid-source diffusion at 1e-4 to 1e-2 Pa and 1150-1250 C, by depth profiling.**
  - Our runs sit there, and the fO2 exponent there is an extrapolation.
  - Oeser's 1150 C steps steepen from 0.09-0.24 below 1e-5 Pa to 0.42-0.52 above it, but rest on one run.
- **A melt-contact experiment at a second fO2.**
  - It separates a multiplicative melt effect (a constant offset) from an additive one: extra vacancies that matter less where Fe3+ vacancies dominate.
- **The Ni and Mn compilations:** Petry, Holzapfel and Spandler have all three elements, so they test whether a study's offset, or the melt offset, is shared across elements.
- **Rawlings and Mutch (2026)**, "Olivine diffusion chronometry: recent developments and future directions" (in Zotero), is not yet read for further datasets.
