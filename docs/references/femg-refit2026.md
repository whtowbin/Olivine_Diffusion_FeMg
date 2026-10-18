# The femg_refit2026 diffusion equation

`femg_refit2026` is forster's refit (2026) of Fe-Mg interdiffusion in olivine to published experiments. Select it with `diffusivity="femg_refit2026"`, or the short name `refit` (the `--diffusivity` option on the command line). The default remains the Dohmen and Chakraborty (2007) equation as tabulated in DIPRA. This page gives the equation as shipped, its fitted values and their uncertainties, and how it was fitted. The values are those in `src/forster/diffusivity.py`; `tests/test_refit_doc.py` fails if this page and the code disagree.

## The equation

D along the crystal's [001] axis:

$$
\log_{10} D_{[001]} = \log_{10} D_0
- \frac{E + V\,(P - 10^{5}\ \mathrm{Pa})}{\ln(10)\,R\,T}
+ m \log_{10}\!\left(\frac{f_{\mathrm{O_2}}}{10^{-7}\ \mathrm{Pa}}\right)
+ n\,(0.9 - X_{\mathrm{Fo}})
$$

With the fitted values:

$$
\log_{10} D_{[001]} = -8.954
- \frac{208.8\ \mathrm{kJ\,mol^{-1}} + (P - 10^{5}\ \mathrm{Pa})\,(7.53\times10^{-6}\ \mathrm{m^{3}\,mol^{-1}})}{\ln(10)\,R\,T}
+ 0.216\,\log_{10}\!\left(\frac{f_{\mathrm{O_2}}}{10^{-7}\ \mathrm{Pa}}\right)
+ 2.82\,(0.9 - X_{\mathrm{Fo}})
$$

The other two axes, and a profile at angles θ<sub>i</sub> to the three axes (Costa and Chakraborty 2004):

$$
D_{[100]} = D_{[010]} = \tfrac{1}{6}\,D_{[001]},
\qquad
D_{\text{profile}} = \sum_{i} D_i \cos^{2}\theta_i
$$

D is in m²/s, T in K, P and f<sub>O2</sub> in Pa, R = 8.314 J mol⁻¹ K⁻¹, and X<sub>Fo</sub> is the forsterite mole fraction. The form is that of Dohmen and Chakraborty (2007); five numbers are refitted, and the factor of 6 between the axes is kept from them (as in DIPRA and DFENS).

## Fitted values

| Parameter | Symbol | Value | 1 sigma |
|---|---|---|---|
| log10 of the pre-exponential factor, [001] (m²/s) | log10 D0 | -8.954 | 0.366 |
| Activation energy (kJ/mol) | E | 208.8 | 9.3 |
| Activation volume (cm³/mol) | V | 7.53 | 0.99 |
| fO2 exponent | m | 0.216 | 0.034 |
| Composition coefficient (log10 D per unit X_Fe) | n | 2.82 | 0.13 |
| Spread between laboratories (log10 D) | tau | 0.072 | 95% range 0.00-0.42 |

**The parameters are strongly correlated, so their 1-sigma values must not be combined as if independent.** Correlation coefficients:

| | log10 D0 | E | m | n | V |
|---|---|---|---|---|---|
| log10 D0 | 1 | 0.99 | -0.46 | 0.08 | -0.21 |
| E | 0.99 | 1 | -0.46 | 0.10 | -0.22 |
| m | -0.46 | -0.46 | 1 | -0.04 | 0.85 |
| n | 0.08 | 0.10 | -0.04 | 1 | -0.06 |
| V | -0.21 | -0.22 | 0.85 | -0.06 | 1 |

Treated as independent, the five 1-sigma values would give an uncertainty on D several times too large. The full covariance is `DIFFUSIVITY_PRESETS["femg_refit2026"].uncertainty`; `log10_sigma(conditions, orientation, x_fo)` propagates it (exact here, because log10 D is linear in all five parameters). That uncertainty is for D in an average laboratory. A crystal that may sit off that average, as one more experiment would, adds tau in quadrature; forster reports both ranges (`law_` and `all_` columns; README, "Uncertainty").

## Worked values

To check an implementation. D along [001], Fo85:

| Conditions | log10 D (m²/s) | 1 sigma |
|---|---|---|
| 1200 C, 42 MPa, fO2 1.38e-3 Pa (RR1200: the round robin 1200 C conditions) | -15.330 | 0.132 |
| 1100 C, 1 atm, fO2 1e-4 Pa | -16.105 | 0.106 |
| 900 C, 0.1 GPa, fO2 1e-9 Pa | -18.574 | 0.088 |

At the RR1200 conditions this is 0.18 log10 units (a factor of 1.5) faster than Dohmen and Chakraborty (2007), so Fe-Mg diffusion times come out about 1.5 times shorter.

## What it was fitted to

132 values of D from 4 studies, each value stated in the source's tables or text (`data-to-source.md` lists what was left out and why):

- Chakraborty (1997): 55 values from 12 diffusion couples, along [001], 980-1300 C, Fo25-97, 1 atm.
- Dohmen, Becker and Chakraborty (2007): 44 values, all three axes, 800-1200 C, fO2 above 1e-10 Pa.
- Petry, Chakraborty and Palme (2004): 16 values, [001], 902-1403 C, fO2 1e-9.9 to 1e-3.9 Pa.
- Holzapfel, Chakraborty, Rubie and Frost (2007): 17 values, [001], 1350-1550 C, 6-12 GPa. The fO2 of their 14 Au-capsule runs was not buffered; it is taken as Ni-NiO with an uncertainty of 2 log units.

Not included, and described as scenarios in `unconfirmed-data.md`: diffusion next to melt (Spandler and O'Neill 2010; Shea et al. 2023), which is about 0.55-0.85 log units faster, and Oeser et al. (2026), whose [010] is only about 2x slower than [001].

## How it was fitted

Least squares (restricted maximum likelihood) on log D, in `analyses/femg_refit/fit_verified.py` (`fit_preset`), exported by `export_preset.py`. Each value's variance is the sum of:

- its published error in D: per value for Chakraborty (4-98% of D) and Holzapfel et al. (±0.3 log), otherwise the study's stated error (Dohmen et al. ±0.2 log, Petry et al. a factor of 2);
- the effect of its errors in temperature (at least 10 C), fO2, composition and pressure on D;
- an extra scatter common to all values, estimated by the fit (0.055 log);
- an offset per experiment (0.14 log), shared by the values from one experiment, so that each of Chakraborty's diffusion couples, read at 3-9 compositions, counts once for the temperature dependence;
- an offset per laboratory (tau, 0.072 log).

Checks behind the choice of fitting method (`unconfirmed-data.md`, section "The preset"): a bootstrap that resamples whole experiments gives a spread no wider than this uncertainty for any parameter; fitting the form of Mutch et al. (2019) instead gives the same D (+0.180 against +0.177 at RR1200); and orthogonal distance regression without laboratory offsets is not used, because its answer depends on the temperature error assumed.
