# Fe-Mg diffusivity in olivine: newer work against the equation forster uses

Where studies since 2007 support, and where they diverge from, the
Dohmen and Chakraborty (2007) expression that forster uses, and what each
divergence would do to fitted times. This is a synthesis to inform later
decisions; the equation in the code is unchanged. Started 2026-10-12.

## Summary

forster computes D_FeMg with Dohmen and Chakraborty (2007, as corrected in their
2007 erratum): [001] from the TaMED equation, [100] and [010] six times slower,
combined by the profile's angles to each axis.

- **Up to about 1150 C the equation holds up.** A later regression of a larger
  experimental compilation (Mutch et al. 2019, 2021) recovers its form and
  activation energy, and newer experiments keep its 6x anisotropy at these
  temperatures (Oeser et al. 2026).
- **Above about 1150 C, newer solid-source experiments find faster diffusion
  along [010]** and higher activation energies, which they read as the start of a
  change in diffusion mechanism (Sio et al. 2018; Oeser et al. 2026; consistent
  with Tachibana et al. 2013 at 1400-1600 C). The round robin samples 1-3 (1200 C)
  and the paper profiles (1250 C) are in this range.
- **With melt present, Fe-Mg diffusion was about 10x faster** than the
  solid-source calibrations at 1200-1255 C (Shea et al. 2023, agreeing with two
  older melt-olivine datasets). Not yet reproduced with controlled geometry; the
  authors call for more experiments.
- **Every divergence found above 1150 C points to faster diffusion than the
  equation gives, so shorter times.** None of them points the other way at these
  temperatures.

Size of each effect on a fitted time (time multiplier; time scales as 1/D):

| Effect | Applies to | Time multiplier | Source |
|---|---|---|---|
| Regression calibration instead of DC07, TaMED / Global | all profiles | x0.52 / x0.78 at 1200 C; x0.66 / x0.91 at 1102 C | Mutch et al. 2019 (this note's calculation) |
| [010] at least 0.5 log faster, [100] and [001] unchanged | profiles at >= 1150 C | median x0.88 (round robin), x0.90 (paper); down to x0.35 for traverses near [010] | Oeser et al. 2026 (this note's calculation) |
| Sio et al. multipliers ([100] x1.1, [010] x4.5, [001] x1.9) | 1200-1400 C | median x0.48 (round robin), x0.53 (paper); range x0.24-x0.86 | Sio et al. 2018 (this note's calculation) |
| Melt in contact with the olivine | melt-olivine interfaces | x0.05-x0.2 (Fe-Mg), about x0.1 typical | Shea et al. 2023 |
| Parameter uncertainty of the calibration (1 sigma, with covariance) | all | x0.78-x1.29 at 1200 C | Mutch et al. 2019 covariance |

These are separate scenarios, not to be multiplied together without thought;
the [010] and melt effects were measured in different kinds of experiment.

## What forster uses

`engine.D_Fo` (and `dipra.D_Fo_and_derivative`): Dohmen and Chakraborty (2007),
equation for fO2 > 1e-10 Pa,

log D[001] = -9.21 - (201000 + (P - 1e5) 7e-6) / (2.303 R T) + 1/6 log(fO2 / 1e-7) + 3 (X_Fe - 0.1)

with D[100] = D[010] = D[001] / 6. The erratum (Phys Chem Minerals 34:597-598)
corrected the composition term from 3 X_Fe to 3 (X_Fe - 0.1); the code's
`10**(3 * (0.9 - XFo))` is the corrected form. The erratum states that the
equations reproduce all 113 experimental points within half an order of
magnitude. Round robin and paper conditions (fO2 about 1e-3 Pa) are well inside
the high-fO2 (TaMED) regime, so the low-fO2 equation does not apply.

## Study by study

### Dohmen et al. (2007), Dohmen and Chakraborty (2007): the baseline

Thin-film experiments at 700-1200 C combined with Chakraborty (1997), Petry et
al. (2004) and Holzapfel et al. (2007), interpreted with a point defect model.
E = 201 +- 8 kJ/mol from a bulk fit of [001] data at 900-1250 C, fO2 = 1e-7 Pa.
No uncertainty on the pre-factor and no E-pre-factor correlation is published.
Each diffusion coefficient in their Table 3 carries about +-0.2 log units (up to
+-0.3) from profile fitting. Refitting the 69 Table 3 points at fO2 >= 1e-10 Pa
against the erratum equation (D at Fo86, [100] and [010] shifted by log 6) gives
a residual standard deviation of 0.24 log units, mean -0.13 (data slightly below
the equation): the scatter is about the stated measurement error, with little
sign of extra crystal-to-crystal variation.

### Mutch et al. (2019, 2021, DFENS): supports the form, shifts the level, adds covariance

A least-squares regression of a compiled database (Chakraborty 1997; Petry et al.
2004; Dohmen et al. 2007; Dohmen and Chakraborty 2007; Holzapfel et al. 2007;
Spandler and O'Neill 2010):

ln D[001] = a + b ln fO2(bar) + c X_Fo + q / T + j P + h P / T

published with full covariance matrices (Mutch et al. 2019, Supplementary Tables
1-2; also DFENS Data Set S6). For the TaMED mechanism E = 222 +- 7.5 kJ/mol;
the regression recovers the experiments within 0.5 log units and agrees with
earlier expressions to that level.

- **Supports:** the composition dependence (c = -7.18 in ln units, i.e. -3.1 per
  unit X_Fo in log10, matching the 3 of DC07), an activation volume close to
  DC07's, and 6x anisotropy (assumed, not refitted).
- **Diverges slightly:** the fO2 exponent. ln D = b ln fO2 is the same as
  log D = b log fO2, so the regression's exponent is 0.22 against DC07's
  1/6 = 0.17.
- **Diverges:** the central value. At the round robin conditions (X_Fo 0.85,
  fO2 from the table) TaMED gives log D 0.28 above DC07 at 1200 C and 0.18 above
  at 1102 C; the Global regression 0.11 and 0.04 above. These gaps exceed the
  regressions' own parameter uncertainty (about +-0.1 log), so the choice of
  calibration is a systematic, not noise.
- **Uncertainty treatment:** DFENS samples the parameters jointly from the
  covariance and reports that this narrows timescale uncertainty 2-3x compared
  with treating them as independent. At 1200 C the calibration term is x0.78 to
  x1.29 (1 sigma); with T +-30 C, P +-10 MPa and fO2 +-0.2 log as well, x0.64 to
  x1.59. Varying E alone with the pre-factor fixed (the round robin Monte Carlo)
  gives x0.48 to x2.11 once its unit and quantile errors are fixed.
- The published matrices are rounded to 3 significant figures and are
  numerically singular (smallest eigenvalue about 0); use the deposited
  full-precision values if this is implemented.

### Couperthwaite et al. (2021): size of the usual uncertainty

Piton de la Fournaise olivine, DC07 diffusivities. Uncertainty from temperature,
activation energy and pre-factor, treated without covariance: 0.34-0.38 log
units (1 sigma), i.e. about x0.42 to x2.4 on time. The same size as the
independent-parameter treatment above.

### Sio et al. (2018): faster [010] at 1200-1400 C

Fo83-Fo89 and Fo89-Fo100 diffusion couples at 1200-1400 C, aimed at Fe isotope
fractionation. Profiles were fitted with the DC07 equation times a free factor
j: on average j = 1.1 along [100], 4.5 along [010] and 1.9 along [001]. Only 4 of
25 profiles needed j beyond the half order of magnitude DC07 quote. Partly
supports DC07 (within its stated uncertainty for most profiles), but the [010]
excess is systematic. The authors note that sectioning not exactly
perpendicular to the interface may explain some of it.

### Tachibana et al. (2013): higher activation energy at 1400-1600 C

Fo100-Fo90 couples at 1400-1600 C, 1 atm. E = 280-320 kJ/mol, 80-120 kJ/mol above
the lower-temperature value, and anisotropy D[001] > D[010] > D[100]. Outside
the temperatures modelled here, but it is the high-temperature end of the trend
Oeser et al. (2026) extend downward.

### Oeser et al. (2026): anisotropy changes near 1150 C

Isotopically doped powder source (25Mg, 57Fe) on oriented San Carlos olivine at
1100-1300 C and fO2 1e-7 to 1e-2 Pa; tracer and interdiffusion coefficients from
a multicomponent model. Against DC07 at fO2 1e-5 Pa:

- **Supports, below 1150 C:** D[100] = D[010], D[001] faster, as DC07 assume;
  D_FeMg increases steadily with fO2 at <= 1150 C.
- **Diverges, at and above 1150 C:** D[001] > D[010] > D[100]. D[010] is more
  than 0.5 log units above DC07; D[001] is comparable to DC07; D[100] is slightly
  above DC07 at >= 1200 C. Activation energies are 230-370 kJ/mol, above DC07's
  200-220. At >= 1250 C the fO2 dependence becomes more complex. The authors
  interpret this as a transition from the TaMED regime toward diffusion affected
  by intrinsic defects, starting near 1200 C, and they decline to fit a combined
  regression.

Effect on forster fits if D[010] is raised by 0.5 log units (the lower bound)
with D[100] and D[001] unchanged, using each run's angles: round robin runs at
1200 C (12 runs) median x0.88, range x0.37-x0.94; paper runs at 1250 C (18 runs)
median x0.90, range x0.35-x1.00. Traverses close to [010] (for example
KS20-527_5, beta = 166 deg; several paper overgrowth profiles) move most. The
SH63 runs (1102 C) are below the transition.

**Data now transcribed (2026-10-14).** `analyses/femg_refit/data/oeser2026.csv` holds 54
profiles from Tables 1-3 (the PDF's text, not figures), written by
`transcribe_oeser2026.py`. The fields are:

- T and fO2 for each run, from Table 1;
- per profile: the tracer D for Fe and Mg, D_Fe-Mg from the multicomponent model, and
  the conventional-fit D_Fe-Mg where given;
- the beta values and the measurement technique;
- an italic flag for convolution-corrected values.

The composition is Fo91.5. A first look (`oeser_first_look.py`, not a fit; numbers in
`analyses/femg_refit/results/oeser2026_*.csv`):

- **Level against femg_refit2026: +0.20 log below 1200 C, +0.75 at 1200-1303 C.** Below
  1200 C, [001] and [100] are within -0.30 to +0.26 of the refit; [010] is +0.09 to +0.46.
- **The fO2 dependence is not one power law.** Same axis, same temperature, depth-profile
  runs only (1150 C):
  - from 10^-7.1 to 10^-5 Pa the slope is 0.09-0.24;
  - from 10^-5 to 10^-3.7 Pa it is 0.42-0.52, well above 1/6 and the refit's 0.22.

  Flat at low fO2 and steep at high fO2 is the shape a sum of an fO2-independent and an
  fO2-dependent mechanism would give.
- **Caveat on the hottest runs.** Runs 7, 14 and 15 (1255-1300 C, 72-96 h) were measured
  by laser linescan after the crystals broke out of sintered powder, and their D values
  are convolution-corrected with an assumed 7.5 um beam (D = D_measured - sigma^2/2t).
  They sit highest: run 15 at 1256 C and 10^-7 Pa is +1.2 above the refit, against +0.1
  to +0.5 for depth-profile run 5 at 1250 C. They make the 1250-1300 C fO2 steps negative
  or flat (-0.22 to +0.08). The "more complex" fO2 dependence above 1250 C may therefore
  be partly a measurement effect. Flag these rows in any fit.

### Shea et al. (2023): melt makes diffusion about 10x faster

Natural Kilauea and San Carlos olivine seeds in Kilauea basalt, superliquidus
anneal at 1290 C, then dwell at 1200-1255 C for 6-84 h at QFM-1. Modelling the
resulting profiles with published diffusivities gives times 5-20x longer than
the true dwell time for Fe-Mg (Ni 2-21x, Mn 4-24x, Ca 5-65x); fitted
diffusivities are about 10x (Fe-Mg), 7.5x (Ni), 9x (Mn) and 10x (Ca) faster.
Error bars on each diffusivity span 0.6 log units.

- **Supports:** anisotropy, with Fe-Mg about 6x faster along [001] than along
  [100] or [010], as DC07.
- **Diverges:** absolute rate, by an order of magnitude. Agrees with the earlier
  melt-olivine datasets of Jurewicz and Watson (1988) and Spandler and O'Neill
  (2010) (as cited by Shea et al.). Growth during diffusion, diffusion during
  dissolution, and interface defects were tested and judged unable to explain
  it; two-step (growth then diffusion) models lowered diffusivities by only
  1.4-1.9x on average. The proposed cause is extra vacancies created at the
  melt interface by Al or H from the basalt.
- The authors call for more melt-bearing experiments with controlled geometry
  before published diffusivities are revised.
- Against the refit's study offsets, the between-study term and the fO2 exponent:
  `melt-contact-diffusion.md` (2026-10-14).

### Shea, Nizam and Helz (2026): the same speed-up in a natural lava lake

Geology, doi:10.1130/G55140.1 (in `~/Downloads/g55140-2.pdf`).

**What they did.** Olivine from the 1959 Kilauea Iki lava lake, from drill cores taken in
1960-1988. A 3-D cooling model of the lake supplies each sample's temperature-time path,
which drives 3-D diffusion models of olivine populations: 300-850 modelled crystals per
sample, against 24-65 cores measured by electron probe. They tested DC07 times 1, 3, 5,
7.5, 10 and 15, scoring each against the measured core Fo distributions by the
Wasserstein distance.

**What they found.**
- **The cores need 5-15x DC07** (+0.7 to +1.18 log). The average misfit falls to about
  1 mol% at 10x or more.
- **A lower bound, not a bracket**: for two samples (1976, 1988) the misfit is still
  falling at 15x.
- **Temperature cannot be the explanation**: a tenfold change would need a history
  120-170 C hotter. The model's temperatures are set deliberately high, so the
  multipliers are minima.
- **Neither sectioning nor growth can explain it**: growth would require faster
  diffusion still.

**Why it matters here.**
- **It is natural, melt-bearing olivine with no laboratory interface**, which answers
  whether the melt-contact rates apply to magmas: they do, at least here.
- **It sits below the ~1150 C transition Oeser report** (lake at about 1090-1150 C), so
  their [010] effect cannot explain it, and it is near the SH63 conditions (1102 C).
- **It is a population constraint, not a measured D.** Its fO2 is in the supplement, so
  it enters as a bound on a melt term, not as data in the refit.

## Where the studies disagree with each other

- **Anisotropy at 1200-1255 C.** Oeser et al. (powder source, San Carlos,
  1e-7 to 1e-2 Pa) find [010] faster than [100]; Shea et al. (basalt melt,
  QFM-1) find [100] and [010] alike and [001] 6x faster, as DC07. The
  experiments differ in diffusant source, fO2 and crystals, and neither paper
  resolves the difference.
- **Activation energy.** 201 +- 8 (DC07, 900-1250 C), 222 +- 7.5 (Mutch TaMED,
  compiled data), 230-370 (Oeser, 1100-1300 C), 280-320 kJ/mol (Tachibana,
  1400-1600 C). Consistent with one mechanism below about 1150 C and a change
  above, rather than one Arrhenius line throughout.
- **Absolute rate with melt.** All solid-source calibrations (DC07, Mutch, Oeser)
  against the melt-bearing experiments (Shea; Spandler and O'Neill; Jurewicz and
  Watson), about 1 log unit apart.

## What this means for forster results (equation unchanged)

1. **Below about 1150 C** (the SH63 runs), DC07 is supported; the main
   uncertainty is the calibration's, best propagated with covariance. The choice
   between DC07 and the Mutch regressions moves times by x0.66 to x0.91.
2. **Above about 1150 C** (round robin samples 1-3, the paper), times from DC07
   are more likely too long than too short: every newer result points to faster
   diffusion. For most profiles the [010] effect is 10% or less; for traverses
   near [010] it can be 2-4x.
3. **For profiles at a melt contact** (the paper's melt veins and contact
   overgrowths), the Shea et al. result would shorten times about 10x if it holds
   for those conditions. This is the largest possible change and the least
   settled.

Low-cost steps short of redeveloping the equations:

- Report the calibration (DC07 or a Mutch regression) with every time, and give
  the alternative as a systematic. Since 2026-10-12 forster records it
  (`diffusivity` in summary.csv) and can fit with either: presets
  `dohmen_chakraborty2007` (default), `mutch2019_tamed`, `mutch2019_global`, or
  a user equation in a diffusivities table.
- Report each profile's share of D coming from [010] (cos^2(beta) / 6 over the
  total) so profiles sensitive to the anisotropy question can be flagged.
- For melt-contact profiles, state the Shea et al. (2023) caveat and the x0.1
  scenario alongside the main result.
- Propagate the calibration uncertainty with the Mutch et al. covariance (see
  `fit-uncertainty.md`, round robin Monte Carlo item 4).

Not covered here: the effect of water on Fe-Mg diffusion (Hier-Majumder et al.
2005, cited by Shea et al. 2023 and Mutch et al. 2021), which forster does not
model; pressure dependence beyond the activation volume in DC07; Ni, Mn and Ca.

## Method notes

Numbers marked "this note's calculation" were computed on 2026-10-12 from the
published coefficients and the forster parameter tables:

- Calibration offsets: DC07 (erratum form) and the Mutch et al. 2019 TaMED and
  Global regressions evaluated at X_Fo 0.85 and the round robin T, P and fO2
  (fO2 converted from Pa to bar for the regression).
- Uncertainty: 400,000 draws from the TaMED covariance (rounded matrix as
  published), with T, P and fO2 drawn independently.
- Anisotropy: time multiplier = [cos^2 g + (cos^2 a + cos^2 b) / 6] /
  [k_c cos^2 g + (k_a cos^2 a + k_b cos^2 b) / 6] with each run's angles from
  `read_runs` on the round robin and paper tables.
- DC07 Table 3 residuals: 75 rows transcribed from the PDF text, 69 at
  fO2 >= 1e-10 Pa.

## References

- Couperthwaite, F. K., Morgan, D. J., Pankhurst, M. J., Lee, P. D. and Day,
  J. M. D. (2021). Reducing epistemic and model uncertainty in ionic
  inter-diffusion chronology. American Mineralogist 106, 481-494.
  doi:10.2138/am-2021-7296CCBY
- Dohmen, R., Becker, H.-W. and Chakraborty, S. (2007). Fe-Mg diffusion in
  olivine I. Phys Chem Minerals 34, 389-407. doi:10.1007/s00269-007-0157-7
- Dohmen, R. and Chakraborty, S. (2007). Fe-Mg diffusion in olivine II. Phys Chem
  Minerals 34, 409-430. doi:10.1007/s00269-007-0158-6. Erratum: Phys Chem
  Minerals 34, 597-598. doi:10.1007/s00269-007-0185-3
- Mutch, E. J. F., Maclennan, J., Shorttle, O., Edmonds, M. and Rudge, J. F.
  (2019). Rapid transcrustal magma movement under Iceland. Nature Geoscience 12,
  569-574. doi:10.1038/s41561-019-0376-9 (Supplementary Tables 1-2)
- Mutch, E. J. F., Maclennan, J., Shorttle, O., Rudge, J. F. and Neave, D. A.
  (2021). DFENS: Diffusion chronometry using Finite Elements and Nested Sampling.
  Geochemistry, Geophysics, Geosystems 22(4). doi:10.1029/2020GC009303
- Oeser, M., Dohmen, R. and Weyer, S. (2026). Fe-Mg in olivine: simultaneous
  experimental determination of the diffusion-driven isotope fractionation along
  with tracer- and inter-diffusion coefficients. Geochimica et Cosmochimica Acta
  413, 48-66. doi:10.1016/j.gca.2025.09.036
- Shea, T., Ruth, D., Jollands, M., Ohtaki, K., Ishii, H. and Bradley, J. (2023).
  The presence of silicate melt may enhance rates of cation diffusion in olivine.
  Earth and Planetary Science Letters 621, 118370.
  doi:10.1016/j.epsl.2023.118370
- Shea, T., Nizam, N. and Helz, R. T. (2026). A 29-year natural experiment constrains
  Fe-Mg diffusion rates in olivine. Geology, doi:10.1130/G55140.1
- Sio, C. K., Roskosz, M., Dauphas, N., Bennett, N. R., Mock, T. and Shahar, A.
  (2018). The isotope effect for Mg-Fe interdiffusion in olivine and its
  dependence on crystal orientation, composition and temperature. Geochimica et
  Cosmochimica Acta 239, 463-480. doi:10.1016/j.gca.2018.06.024
- Tachibana, S., Tamada, S., Kawasaki, H., Ozawa, K. and Nagahara, H. (2013).
  Interdiffusion of Mg-Fe in olivine at 1400-1600 C and 1 atm total pressure.
  Phys Chem Minerals 40, 511-519. doi:10.1007/s00269-013-0588-2

Also seen but not read in full (paywalled): Wu et al. (2022), Diffuser,
Computers & Geosciences 163; Morgado and Morgan (2024), DiffSim, Frontiers in
Earth Science, doi:10.3389/feart.2024.1431516; the 2026 review "Olivine
diffusion chronometry: recent developments and future directions"
(doi:10.1080/29932092.2026.2678716), whose summary also notes that olivine-melt
experiments give Fe-Mg diffusion up to an order of magnitude faster.
