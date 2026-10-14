# Fe-Mg diffusivity: where the uncertainties come from, and the diffusion regimes

Sources checked 2026-10-13 for two questions: where the 201 ± 8 kJ/mol on the
default law comes from and what it covers, and whether the Mutch et al. (2019)
covariances can be used as published. Also, how Chakraborty (2010)'s diffusion
regimes frame the newer experiments as options. Complements
`diffusivity-literature.md`, which covers the newer experiments themselves.

## Summary

- **The ±8 kJ/mol is from Dohmen, Becker and Chakraborty (2007, Part I), not
  Chakraborty (2010).** It is the standard error of E from one bulk fit: San Carlos
  olivine, along [001], at fO2 = 10⁻⁷ Pa, 900-1250 °C. No uncertainty on the
  pre-factor and no E-pre-factor correlation is given anywhere.
- **Chakraborty (2010) restates the law with one uncertainty: half an order of
  magnitude.** The law and its PED companion reproduce 113 experiments to within
  that. It also gives the regime framework (intrinsic, TaMED, PED) used below.
- **The oxygen fugacity exponent is a larger uncertainty than E at our
  conditions.** Part I gives 1/4-1/7 in its abstract and 1/4-1/6 in its
  conclusions; its own fits lie at 1/5-1/7. The law uses 1/6 and is anchored at
  10⁻⁷ Pa, and our runs sit 3.3-4.9 log units above that (round robin 3.3-4.1,
  paper 4.9). Relative to 1/6, an exponent of 1/4 raises log D by 0.27-0.41 (time
  ×0.39-×0.53); 1/7 lowers it by 0.08-0.12 (time ×1.2-×1.3); Mutch's fitted 0.224
  raises it by 0.19-0.28.
- **The Mutch et al. (2019) covariances are in their SI, not in the DFENS repo.**
  The repo's script loads them from a `D_vcov/` folder that was never uploaded.
  As printed (3 significant figures) the TaMED matrix is not positive definite,
  but it propagates to a sensible 1σ: 0.11 log units in D at 1200 °C.

## The 201 ± 8 kJ/mol

Dohmen, Becker and Chakraborty (2007), Phys Chem Minerals 34, 389-407 (Part I):

- "A bulk fit of all data shown in Fig. 6a, which were measured with San Carlos
  olivine between 900 and 1,250 °C at an fO2 of 10⁻⁷ Pa gives an activation energy
  of 201 ± 8 kJ/mol, slightly lower than 226 ± 18 kJ/mol obtained by fitting the
  data of Chakraborty (1997) alone."
- Per axis at 10⁻⁷ Pa: 196 ± 12 ([100]), 191 ± 9 ([010]), 201 ± 8 kJ/mol ([001]).
  From these the authors conclude that the anisotropy does not depend on
  temperature between 800 and 1,250 °C (conclusion 2).
- At lower fO2 the activation energy is higher: 220 ± 9 kJ/mol fitting [001] at
  10⁻¹⁰ and 10⁻¹² Pa together; 223 ± 9 at 10⁻¹² Pa (conclusion 4). This is the
  start of the PED regime.
- fO2 exponent, stated three ways in the same paper: "between 1/4 and 1/7"
  (abstract); "consistent with an fO2 exponent, m, of 1/4-1/6. Within the
  uncertainty of measurement, it is not possible to constrain this exponent any
  better" (conclusion 3). The fits themselves give m between 1/5 and 1/7 at about
  1,050 °C (10⁻⁵ to 10⁻¹¹ Pa), ~1/6-1/8 between 10⁻⁷ and 10⁻¹⁰ Pa at lower
  temperatures, and ~1/10 at 900 °C under reducing conditions. The 1/4 end comes
  from earlier studies the data cannot rule out. Part II (Dohmen and Chakraborty
  2007) fixes 1/6 from the point defect model.
- Profile-fitting errors on each D: ±0.2 log units, up to ±0.3 for flat or short
  profiles.

What the ±8 does not cover: the pre-factor, its correlation with E, the fO2
exponent, and the choice of one Arrhenius line across studies. Shifting E by
±8 kJ/mol with the pre-factor fixed moves D at 1200 °C by ×0.52-×1.92, but D near
the middle of the data (about 1075 °C) is pinned much more tightly than that. A
sample over E must move the pre-factor with it.

## Chakraborty (2010): the law, the PED law, and the regimes

Chakraborty, S. (2010), Diffusion coefficients in olivine, wadsleyite and
ringwoodite. Rev Mineral Geochem 72, 603-639, p. 617:

- TaMED, fO2 > 10⁻¹⁰ Pa, along [001]:
  D = 10^-9.21 (fO2/10⁻⁷)^(1/6) 10^(3(X_Fe - 0.1)) exp(-(201000 + (P - 10⁵) 7×10⁻⁶)/RT),
  with D[100] ≈ D[010] ≈ D[001]/6. Identical to the `dohmen_chakraborty2007` preset.
- PED, lower fO2, independent of fO2:
  log D = -8.91 - (220000 + (P - 10⁵) 7×10⁻⁶)/(2.303 RT) + 3(X_Fe - 0.1), same
  anisotropy. Within olivine's stability field these conditions are reached only
  below about 900 °C. "For less accurate work or where fO2 is unknown, the second
  expression describes diffusion coefficient of olivine at any T-fO2 condition,
  with a larger error."
- Both "reproduce 113 experimental data points from different studies ... to
  within half an order of magnitude."
- Water (Hier-Majumder et al. 2005): log D0 = -14.8 ± 2.7, fH2O exponent
  0.9 ± 0.3, Q = 220 ± 60 kJ/mol, V* = 16 ± 6 cm³/mol, valid along NNO. At equal
  fO2, wet diffusion is about 2x faster (Costa and Chakraborty 2008). The
  wet law applies above about 11 ppm water.

The regimes (pp. 605-608):

1. **Intrinsic**, at the highest temperatures in very pure crystals. D depends on
   P and T only. Its transition to TaMED is "marked by a large change in Arrhenius
   slope (change in activation energies of several 100 kJ/mol), but is unlikely to
   ever be realized in nature in olivines".
2. **TaMED** (transition metal extrinsic), at intermediate temperatures. D depends
   on P, T and fO2 (vacancies from Fe³⁺) but not on trace elements. "Likely to
   dominate in most natural olivines."
3. **PED** (pure extrinsic), at low temperatures, typically below about 800 °C.
   D depends on P, T and heterovalent trace elements (Al, H), not on fO2. The
   TaMED-PED transition is smeared, with an activation energy change of tens of
   kJ/mol; where it falls depends on trace-element content.

Chakraborty (1997)'s data show "no break in the Arrhenius slope between 980 and
1300 °C".

## Where the newer experiments sit in this framework

| Study | Finding | Regime reading | Conflicts with |
|---|---|---|---|
| Oeser et al. (2026) | above ~1150 °C, [010] > [100], E 230-370 kJ/mol | onset of intrinsic-affected diffusion (authors' reading) | DC07 Part I conclusion 2 (anisotropy constant 800-1250 °C); Chakraborty 2010's "unlikely in nature" |
| Tachibana et al. (2013) | E 280-320 kJ/mol at 1400-1600 °C | intrinsic, as Chakraborty 2010 expects at high T | nothing; outside our range |
| Sio et al. (2018) | [010] ×4.5 at 1200-1400 °C | consistent with Oeser | as Oeser |
| Shea et al. (2023) | ~10x faster with melt present, anisotropy as DC07 | extra extrinsic vacancies from melt-supplied Al or H (authors' reading): a trace-element effect, which the TaMED fO2 term cannot carry | the solid-source calibrations' absolute level |
| Hier-Majumder et al. (2005) | water speeds diffusion, larger V* | ×2 at equal fO2 | none; a separate law |

The tension is at 1150-1300 °C. Two experimental generations disagree on whether
the anisotropy and activation energy change there. Neither the round robin
(1102-1200 °C) nor the paper (1250 °C) runs can avoid it.

## Mutch et al. (2019): covariances

- DFENS repository (github.com/EuanMutch/DFENS, commit 0072d41, January 2021):
  `DFENS_Ol_1D.py` reads `./D_vcov/FeMg_TAMED_001_vcov.csv`, `FeMg_all_001_vcov.csv`
  and the Ni, Mn and Ca equivalents. **That folder is not in the repository.**
  The script confirms the parameter order and units: `(Intercept), lnfO2, XFo,
  T_i, P_Pa, T_i:P_Pa` with `T_i = 1/T`, fO2 in bar and P in Pa. So j multiplies P
  and h multiplies P/T, as `diffusivity.py` assumes. It gives the TaMED central
  values to 4 figures: a = -6.755, b = 0.2244, c = -7.181, q = -2.674e4,
  j = -5.213e-10, h = -1.028e-7. The presets use the SI's 3 figures (q = -26700,
  E 222.0 against 222.3 kJ/mol).
- The SI (Supplementary Table 2, p. 49) prints full 6×6 covariance matrices for
  FeMg (Global), FeMg (TaMED), Ni and Mn. "Global" is regressed on all the Fe-Mg
  data, TaMED and PED alike (SI Table 1 caption); "TaMED" on the TaMED data only.
- As printed, the TaMED correlation matrix has a smallest eigenvalue of
  -3.6×10⁻⁴: not positive definite, so it cannot be sampled directly. Global's
  is +4.9×10⁻⁴. Options: clip the negative eigenvalue (nearest positive-definite
  matrix), or obtain the full-precision `D_vcov` files (DFENS Data Set S6, or ask
  the author).
- Propagated to log10 D at the round robin conditions (X_Fo 0.85), 1σ:

  | Regression | 1200 °C | 1102 °C |
  |---|---|---|
  | TaMED | 0.109 (time ×0.78-×1.28) | 0.091 |
  | Global | 0.085 (×0.82-×1.22) | 0.075 |
  | TaMED, parameters taken as independent | 0.526 | |

  σ(E): 7.5 kJ/mol (TaMED), 7.6 (Global). Corr(a, q) = -0.41 (TaMED) and -0.59
  (Global): the intercept and activation energy trade off, which is why the
  joint treatment is five times narrower.

## Can we refit them? (checked 2026-10-13)

Data in hand: Dohmen, Becker and Chakraborty (2007) Part I, Table 3, 75
experiments (all parsed from the PDF; they agree row for row with the other
session's transcription), and Spandler and O'Neill (2010). Table 3 gives D at
Fo86, using a composition exponent of 3; all runs are at 1 atm, fO2 10⁻¹² to
10⁻⁵ Pa.

Refits of Table 3 alone (ordinary least squares on log D):

| Fit | Result | Published |
|---|---|---|
| [100], fO2 10⁻⁷ Pa, n = 11 | E = 196 ± 12 kJ/mol | 196 ± 12 |
| [010], fO2 10⁻⁷ Pa, n = 11 | E = 191 ± 11 kJ/mol | 191 ± 9 |
| [001], fO2 10⁻⁷ Pa, n = 11 (900-1102 °C) | E = 215 ± 34 kJ/mol | 201 ± 8, but fitted with Chakraborty (1997) and Petry et al. (2004) data up to 1250 °C |
| TaMED form, all axes, fO2 ≥ 10⁻¹⁰ Pa, n = 69 | log D0[001] -9.74 ± 0.39, m 0.186 ± 0.020, E 190 ± 10 kJ/mol, [100] /5.1, [010] /3.9; residual sd 0.23 log | (DC07 law: m 1/6, E 201, /6; its residuals on these points: mean -0.13, sd 0.24) |

The [100] and [010] fits reproduce the paper, so the data and method match. The
TaMED-form refit propagates to 1σ = 0.108 log units on D[001] at 1200 °C and
1.38×10⁻³ Pa, close to Mutch's 0.109. But it is fitted on Fo86 at 1 atm only, so
the composition and pressure terms cannot be estimated from it.

What a full refit needs:

- **Mutch et al. (2019), all six calibrant sets:** Chakraborty (1997), JGR 102(B6),
  12317-12331; Petry, Chakraborty and Palme (2004), GCA 68, 4179-4188; Holzapfel,
  Chakraborty, Rubie and Frost (2007), PEPI 162, 186-198; plus Part I, Part II and
  Spandler and O'Neill (2010), which are in hand. All published, so their
  Supplementary Tables 1-2 can be used as a target: matching a-h and the
  covariance would validate the regression and give a full-precision,
  positive-definite matrix.
- **Dohmen and Chakraborty (2007) cannot be rebuilt exactly.** Their calibration
  includes unpublished experiments by Meißner (Mutch et al. 2019, Methods), and the
  law comes from a point defect model, not a free regression. A regression of
  the published data in the DC07 form is the closest substitute.

Our run conditions lie above the experiments' fO2: Part I reaches 10⁻⁵ Pa, while
the round robin and paper runs are at 10⁻³·⁷ to 10⁻²·¹ Pa. Spandler and O'Neill
(10⁻⁸·³ bar = 10⁻³·³ Pa, 1300 °C) is the only calibrant in that range. Any
refit's fO2 term is an extrapolation at our conditions.

## Options for forster

Status (2026-10-13): option 2 is done in a better form: `femg_refit2026`, a refit of
all the TaMED data (not only DC07's Table 3) with a covariance and a between-study
term, propagated analytically to the fitted time (`FitResult.time_range_days`); see
`docs/plans/2026-10-13-law-uncertainty.md`. For Fo fits option 1 no longer needs a
Monte Carlo: a law's covariance maps onto log time directly. The Mutch et al. (2019)
covariances are still not transcribed. Where the analytic mapping holds and where it
does not (changing rims, growth stages, element fits): `law-uncertainty-limits.md`.

Ranked by what they buy for the work they cost:

1. **Sample the Mutch regressions with their covariance** (repaired as above):
   a calibration uncertainty with correlation, about ±0.1 log at 1200 °C. Needs the
   matrices in `diffusivity.py` and a sampler in the phase 3 Monte Carlo.
2. **A covariance for the default law, from its own data.** DC07 publish none. A
   regression of their Table 3 (`diffusivity-literature.md` already transcribed 75
   rows) for log D0 and E at fixed fO2 exponent and anisotropy would give one,
   and with it a defensible "201 ± 8" sample. Without it, the honest choices are
   the authors' ±0.5 log envelope, or E ± 8 kJ/mol pivoted about the data's
   central temperature (~1075 °C) so D there stays fixed.
3. **A `dohmen_chakraborty2007_ped` preset** (the PED law above), plus a warning
   when the TaMED law is used below 10⁻¹⁰ Pa. Cheap, and the PED law is the one
   the authors recommend when fO2 is unknown.
4. **The fO2 exponent as a reported systematic** (1/6 against 1/4 or 1/7, or
   Mutch's fitted 0.224). At round robin and paper fO2 it is the largest single
   calibration term: +0.27 to +0.41 log toward 1/4, -0.08 to -0.12 toward 1/7.
5. **High-temperature anisotropy (Oeser, Sio) and melt contact (Shea) as named
   scenarios, not presets.** Oeser et al. decline to fit a law, and Shea et al. ask
   for more experiments before diffusivities are revised. A per-axis multiplier
   option ([010] ×3, say) or a reported sensitivity per profile matches what the
   evidence supports. A preset would imply a calibration that does not exist.
6. Water (Hier-Majumder) needs fH2O, which forster does not take. Out of scope
   for now.

## Sources read

- Dohmen, Becker and Chakraborty (2007), Part I, PCM 34, 389-407 (Zotero
  ZPZGJJNL): pp. "Results" and "Conclusions", text extracted and searched.
- Dohmen and Chakraborty (2007), Part II, PCM 34, 409-430 (Zotero PBC3BNZK):
  searched for activation energy statements; Q = 201 kJ/mol at 10⁻⁷ Pa, no
  uncertainty on it.
- Chakraborty (2010), RiMG 72, 603-639 (Zotero 7GHEMBB5): pp. 605-608 and 617
  read, p. 617 as a page image.
- Mutch et al. (2019), Nature Geoscience 12, 569-574, Supplementary Information
  (Zotero DU5PLXNX): Tables 1-2, pp. 48-49.
- DFENS repository, github.com/EuanMutch/DFENS, commit 0072d41: `DFENS_Ol_1D.py`,
  `pmc.py` read as text; no code run.
