# Is the whitening in FORSTER's default fit valid for these data? (2026-10-15)

Written for Henry and for a later reader of the paper. The method is described in
[`docs/references/likelihood-method.md`](../../references/likelihood-method.md),
sections 2, 4 and 5; this note tests what that note left open (its section 10).

## Short answer

- **Whitening itself works on the real profiles.** After whitening, serial
  correlation is at chance level: Ljung-Box flags 1 of 36 profiles and the runs test
  2 of 36, against about 1.8 expected by chance for each test. The squared-exponential
  shape of the misfit was never beaten by the exponential or Matérn 3/2 shape by more
  than 2 AIC (36 of 36). On synthetic data, a misfit of the wrong shape (exponential or
  Matérn 3/2) lowers coverage by at most 5 points compared with the right shape at the
  same length. **No Matérn option is needed.**
- **Heavy tails and outliers do not break the range.** About a third of the real
  profiles have heavier tails than a normal distribution allows (11 of 36 fail
  Shapiro-Wilk or the largest-|z| test). On synthetic data, Student-t noise and 1–2 outliers of 5 sd per
  profile kept 1σ coverage at 69–74%. Outliers make the range wider, but it still covers.
  **A robust (Student-t) likelihood would buy precision, not calibration.** It is not
  urgent.
- **The real weakness is misfit concentrated near the step.** In 30 of 34 real
  profiles the whitened residuals scatter more within 2 diffusion lengths of the step
  than elsewhere (median variance ratio 1.75). In 12 of them the difference is
  significant (about 2 would be expected by chance). The model assumes the misfit is
  equally large everywhere. When it is not, synthetic 1σ coverage falls to 45–71%
  (69–72% with white noise), and 2σ coverage to 72–95% (target 95.45%). It falls
  furthest on the dense SH63-like profile (45–51%). The four real SH63olv066 profiles are more concentrated than the strongest
  synthetic case, so **their 1σ ranges probably contain the true time less than half
  the time.**
- **A candidate fix works where it matters.** A misfit term whose amplitude may grow
  near the step lifts dense-profile coverage from 46–53% to 61–66% (2σ: 70–83% to
  88–91%), and changes the well-specified cases by at most 3 points. It is clearly preferred
  (ΔAIC < −6) on 7 of 36 real profiles and moves SH63olv066 by −10 to −15%. It is
  worth adding, first as a warning and then, after more validation, as part of the
  model (section 5).

## 1. Terms

- **Residual**: measured Fo minus model Fo at a measured point, at the best-fit time.
- **Noise model**: what the method assumes about the residuals. FORSTER's default is
  *white noise* (independent from point to point, standard deviation s) plus a
  *smooth correlated misfit* (amplitude a, length L). Nearby misfit values are alike
  and values far apart are unrelated. The **kernel** is the formula for how fast that
  likeness decays with distance d:
  - **squared-exponential** (FORSTER's): exp(−d²/2L²). Very smooth misfit.
  - **exponential** (Matérn 1/2): exp(−d/L). Rough, jagged misfit.
  - **Matérn 3/2**: (1 + √3 d/L) exp(−√3 d/L). In between.
- **Whitening**: transforming the residuals with the fitted noise model (z = L⁻¹r, with
  L the Cholesky factor of the covariance) so that, if the model is right, the z are
  independent standard-normal numbers. The χ² that sets the best fit and range is Σz².
  The checks below ask whether the z really look like that.
- **AIC** (Akaike information criterion): −2 × maximised log-likelihood + 2 × number
  of noise parameters. Lower is better. A difference under 2 means the data cannot
  tell the models apart.
- **Coverage**: the share of synthetic profiles of known age whose 1σ range contains
  the true time. A correct 1σ range covers 68.27%, a 2σ range 95.45%. With 1000
  trials, a correct 1σ range lands between 65.4% and 71.1%, and a correct 2σ range
  between 94.1% and 96.7%, 95% of the time (the **sampling band**).
- **Independent-error baseline**: the rule before 2026-10-12. Each point's error is
  independent, σ is fitted to the residuals, and the range is where χ² rises by 1
  (1σ) or 4 (2σ).
- **Diffusion length** l = √(D t) at the best fit. **Near the step** means within 2 l of
  a step of the initial condition.
- **Bootstrap p-value**: the standard tests (Ljung-Box and the others) assume the z are
  exactly independent N(0, 1). Here the noise model was fitted to the same residuals,
  which makes z look slightly too white. Each p-value below therefore comes from 200
  synthetic data sets drawn from the profile's own fitted model and refitted with the
  whole default pipeline. p is the share of those that look at least as extreme as the
  real data.

## 2. What was tested

**A. Real residuals** (`real_profiles.py`, `zone_noise.py real`). The 35 profiles of
the grid-choice study (17 round robin, 18 Towbin and Plank 2026, loaded as
`analyses/ic_sampling/references.py` does) and the packaged example
(`forster.write_template`; the same crystal as WHT01_Ol_63, on the chosen grid). Each
was fitted with the default method (`forster.fit_run`). Then:
1. At the white best fit, where FORSTER fits its noise model, four noise models were
   fitted by maximum likelihood: white only, white + squared-exponential, white +
   exponential, white + Matérn 3/2. Each used the same (ρ, L) grid as FORSTER, refined
   with the same optimiser and bounds for every kernel. The check that my
   squared-exponential grid fit reproduces FORSTER's `NoiseModel` passes for all 36.
   The comparison was repeated at the correlated best fit, with the same conclusions.
2. The residuals at the default best fit were whitened with FORSTER's fitted model and
   tested: Ljung-Box on lags 1–5, a runs test on the signs, Shapiro-Wilk, the variance
   of z and the largest |z|.
3. The scatter near the step(s) and away from them was compared, as raw-residual RMS
   and as variance of z. A rim zone (within 2 l of the rim but not near a step) had 3
   or more points in only 3 profiles, so it could not be tested separately.

**B. Synthetic coverage when the noise model is wrong** (`synthetic.py`). Profiles of
known age were made with the forward model (implicit scheme, grid from
`forster.grid.choose_grid` at the true time, which gives 400 time steps up to it). The
model was run on three real geometries, using only their x positions:

| Geometry | Points | Spacing | True time | Diffusion length | White noise |
|---|---|---|---|---|---|
| KS20-527_2-transect-1 (round robin) | 29 | 10 µm | 40.3 d | 26 µm | 0.2 mol% Fo |
| GCB1A R31_MP_Ol5_prof_2 (paper) | 17 | 2.9 µm | 12.3 d | 8.5 µm | 0.1 mol% Fo |
| SH63olv066-1 (round robin, two steps) | 298 | 0.5 µm | 10.8 d | 7.1 µm | 0.6 mol% Fo |

The white-noise levels follow the 2026-10-13 validation for KS and SH63. For the paper
geometry I used the median white noise fitted to the paper profiles (about 0.1 mol%).
The cases, each with 1000 trials:
- **controls**: white only, and white + squared-exponential misfit (0.4 mol% Fo) at
  L = 3, 10, 30 µm. The noise model is right in these;
- **wrong kernel shape**: exponential and Matérn 3/2 misfit, 0.4 mol% Fo, L = 3, 10,
  30 µm;
- **misfit concentrated near the step**: squared-exponential misfit (L = 3 or 10 µm)
  multiplied by a Gaussian bump of width l around each step, with peak amplitude
  0.4 mol% Fo, or 1.2 mol% ("strong", added after part A showed near-step scatter
  that strong in the real data);
- **heavy tails**: Student-t noise with 3 degrees of freedom, scaled to the same sd;
  and white noise plus 1 or 2 points shifted by 5 sd;
- **heteroscedastic white noise**: sd proportional to |Fo − core Fo| (plus 10% of its
  maximum, so the core is not noise-free), or sd rising linearly 4× from core to rim.
  Both are scaled to the same RMS sd.

Each synthetic profile was fitted with the unchanged default
(`forster.uncertainty.fit_correlated_points` on the model at the points for every time
step). That is what `fit_profile` does, minus early stopping, which does not change
the fit. The same data were also fitted with the independent-error baseline and with
two cheap candidate changes: the kernel chosen by AIC, and one refit without points
whose standardised leave-one-out residual exceeds 4. **Follow-up** (`zone_noise.py`,
500 trials, the first 500 draws of the same cases): two candidate near-step noise
models (section 5).

## 3. Results A: real residuals

![AIC comparison](fig2_aic.png)

**Figure 2.** AIC of each noise model above the best of the four, for every real
profile, from the residuals at the white best fit. Grey: within 2 of the best. RR:
round robin. Profiles are sorted by how much the white-only model loses. In the top 13
a correlated term is needed (white only worse by more than 2). In the rest, white noise alone is best, and the three
correlated models sit near +2 to +4 (the AIC cost of their two extra parameters).

**Kernel shape (A2).**
- The current noise model (white, or white + squared-exponential) is within 2 AIC of
  the best of the four for **36 of 36** profiles. The squared-exponential is within 2
  of the best correlated kernel for 36 of 36, and is the best one for 35.
- A correlated term is clearly needed (white only worse by > 2) for 13 profiles. In 12
  of them the squared-exponential is the best correlated kernel. The 13th,
  WHT01_Ol_192, prefers Matérn 3/2 by 0.01.
- Only the SH63olv066 profiles carry enough points to separate the shapes. They
  favour the squared-exponential over the exponential by 1.8–10.2 and over Matérn 3/2
  by 1.1–4.6.
- **Caveat on power:** for 15–30 points this comparison is weak. In part B, with a
  truly exponential misfit, AIC picked a kernel other than the squared-exponential in
  only 0–32% of KS and paper trials and 65–69% of SH63 trials. With a truly
  squared-exponential misfit it picked another kernel in up to 37% of SH63 trials. So
  "never beaten" mostly means "the data cannot tell". Part B shows this does not
  matter much for coverage.

![Whitened residuals](fig3_whitened.png)

**Figure 3.** (left) Bootstrap p-values of the whiteness tests for the 36 real
profiles. The numbers on top count p < 0.05; about 1.8 of 36 would be expected by
chance. (right) Lag-1 autocorrelation of the residuals before and after whitening.
Only fit statistics are shown, no measured data.

**Whiteness (A3).**
- **Serial correlation is removed.** 12 profiles have raw lag-1 autocorrelation above
  0.3. After whitening, 35 of 36 lie between −0.22 and +0.14. The exception is
  WHT06_ol_overgrowth1 (14 points, −0.62, Ljung-Box p = 0.03). Ljung-Box p < 0.05:
  1 of 36; runs test: 2 of 36 (SH63olv066-2 and -3). That is chance level.
- **The distribution is not always normal.** Shapiro-Wilk p < 0.05: 8 of 36. Largest
  |z| too large: 11 of 36. Excess kurtosis too large: 10 of 36. Twelve profiles fail
  at least one of the four core tests (Ljung-Box, runs, Shapiro-Wilk, largest |z|), so
  24 of 36 pass all four. With a Bonferroni correction for four tests, 28 of 36 pass.
  The failures cluster:
  - SH63olv066-1 to -4 (|z| up to 5.3);
  - KS20-527-8 transects 1, 2, 4 and 5 (|z| 3.4–3.6 in 29 points);
  - WHT01_Ol_190, WHT01_bigol1 and Xenocryst_ol23;
  - WHT06_ol_overgrowth1 (Ljung-Box only).
- **Where the extreme points are.** The most extreme standardised leave-one-out
  residual sits *at the step* in all four SH63olv066 profiles (0.00–0.07 l from it) and
  within 0.3 l in KS20-527-8 transects 1, 2, 4 and 5. These look like misfit at the
  step rather than random bad analyses.
- **The variance of z is 0.79–1.00 by construction**, because s is fitted to these
  residuals. It is not a useful test and is left out of "passes".

**Near-step scatter (A4).**
- Whitened variance near the step against far from it: larger in **30 of 34**
  profiles (34 had 3 or more points in each zone). Median ratio 1.75; median raw-RMS
  ratio 1.59 (round robin 1.70, paper 1.57).
- Bootstrap p < 0.05 in **12 of 34**: KS20-527_2-transects 1 and 2, all four
  SH63olv066, WHT01_Ol_180, _190 and _191, both Xenocryst_ol6 profiles, and
  WHT06_ol48.
- The SH63olv066 profiles have raw-RMS ratios of 1.9–2.7, more than the strongest
  synthetic near-step case (1.5–1.6, section 4).

Per-profile values. ΔAIC columns are relative to the squared-exponential (positive:
squared-exponential better). p-values are bootstrap; "<0.01" means none of the 200
bootstrap sets was as extreme. The last column is the near-step candidate of section 5
(negative: the near-step term is preferred). Full columns are in `real_profiles.csv`,
`real_kernels.csv` and `zone_real.csv`.

| profile | set | n | L (um) | AIC white - sq-exp | AIC exp - sq-exp | AIC M3/2 - sq-exp | lag-1 raw | lag-1 whitened | Ljung-Box p | runs p | Shapiro p | max abs z (p) | near/far var z (p) | near-step term dAIC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| KS20-527_2-transect-1 | RR | 29 | 16.3 | 0.2 | 0.1 | 0.1 | 0.10 | -0.16 | 0.72 | 0.97 | 0.25 | 3.0 (0.08) | 2.48 (0.03) | -2.9 |
| KS20-527_2-transect-2 | RR | 29 | 16.3 | -0.4 | 0.1 | 0.0 | 0.03 | -0.21 | 0.63 | 1.00 | 0.20 | 2.9 (0.05) | 2.33 (0.03) | -2.1 |
| KS20-527_2-transect-3 | RR | 29 | 145.5 | -0.4 | 0.5 | 0.3 | -0.13 | -0.13 | 0.55 | 0.16 | 0.22 | 2.7 (0.13) | 0.87 (0.54) | 2.0 |
| KS20-527_2-transect-4 | RR | 29 | 20.8 | 4.5 | 0.3 | 0.2 | 0.49 | -0.08 | 0.92 | 0.38 | 0.56 | 2.2 (0.67) | 1.19 (0.29) | -3.9 |
| KS20-527_5-transect-1 | RR | 20 | 22.7 | 9.0 | 0.9 | 0.5 | 0.68 | -0.18 | 0.88 | 0.28 | 0.47 | 2.2 (0.39) | 2.18 (0.10) | -0.4 |
| KS20-527_5-transect-2 | RR | 20 | 18.5 | 2.9 | 0.5 | 0.4 | 0.49 | -0.08 | 0.91 | 0.43 | 0.32 | 2.2 (0.50) | 2.60 (0.07) | 0.2 |
| KS20-527_5-transect-3 | RR | 20 | 15.1 | -2.4 | 0.2 | 0.2 | 0.24 | -0.03 | 0.54 | 0.20 | 0.06 | 2.4 (0.28) | 1.93 (0.16) | 1.5 |
| KS20-527-8_transect-1 | RR | 29 | 20.9 | -3.4 | 0.3 | 0.2 | 0.02 | -0.14 | 0.68 | 0.35 | <0.01 | 3.6 (<0.01) | 1.21 (0.33) | 2.0 |
| KS20-527-8_transect-2 | RR | 29 | 20.9 | -3.2 | 0.3 | 0.2 | -0.01 | -0.15 | 0.64 | 0.33 | 0.12 | 3.4 (0.01) | 1.19 (0.36) | 2.0 |
| KS20-527-8_transect-3 | RR | 29 | 20.9 | -1.9 | 0.6 | 0.4 | 0.19 | -0.13 | 0.55 | 0.17 | 0.25 | 2.9 (0.08) | 1.63 (0.19) | 1.2 |
| KS20-527-8_transect-4 | RR | 29 | 20.9 | -3.7 | 0.2 | 0.1 | -0.04 | -0.10 | 0.72 | 0.34 | 0.03 | 3.5 (<0.01) | 1.02 (0.48) | 2.0 |
| KS20-527-8_transect-5 | RR | 29 | 26.8 | -1.6 | 0.7 | 0.4 | 0.12 | -0.18 | 0.55 | 0.35 | 0.01 | 3.6 (<0.01) | 1.34 (0.31) | 2.0 |
| SH63olv066-1 | RR | 298 | 7.7 | 174.5 | 10.2 | 4.6 | 0.61 | 0.08 | 0.08 | 0.31 | <0.01 | 5.2 (<0.01) | 1.70 (<0.01) | -30.6 |
| SH63olv066-2 | RR | 298 | 7.7 | 129.3 | 6.0 | 2.8 | 0.52 | 0.13 | 0.09 | 0.05 | <0.01 | 5.3 (<0.01) | 1.88 (<0.01) | -25.3 |
| SH63olv066-3 | RR | 298 | 7.7 | 99.8 | 2.2 | 1.1 | 0.47 | 0.14 | 0.09 | 0.03 | 0.01 | 4.6 (<0.01) | 1.76 (<0.01) | -19.0 |
| SH63olv066-4 | RR | 298 | 7.7 | 74.1 | 1.8 | 1.1 | 0.42 | 0.14 | 0.07 | 0.08 | 0.02 | 4.6 (<0.01) | 1.74 (<0.01) | -18.2 |
| SH63olv084 | RR | 273 | 21.0 | 9.4 | 0.1 | 0.3 | 0.11 | 0.04 | 0.26 | 0.42 | 0.35 | 3.0 (0.41) | 0.96 (0.56) | 0.7 |
| GCB1A R31_MP_Ol5_prof_1 | TP26 | 17 | 20.3 | -3.9 | 0.0 | 0.0 | -0.16 | -0.16 | 0.21 | 0.77 | 0.84 | 2.1 (0.59) |  | 2.0 |
| GCB1A R31_MP_Ol5_prof_2 | TP26 | 17 | 7.5 | -2.8 | 0.2 | 0.1 | 0.17 | 0.02 | 0.45 | 0.70 | 0.63 | 2.0 (0.57) | 1.95 (0.23) | 0.4 |
| WHT06_ol_overgrowth2_prof_1 | TP26 | 27 | 0.0 | -4.0 | 0.0 | 0.0 | 0.01 | 0.01 | 0.73 | 0.39 | 0.83 | 2.2 (0.61) |  | 2.0 |
| WHT06_ol_overgrowth1_prof_1 | TP26 | 14 | 9.5 | -1.2 | 1.0 | 0.6 | 0.13 | -0.62 | 0.03 | 0.07 | 0.60 | 2.0 (0.53) | 0.99 (0.41) | 2.0 |
| WHT01_bigol1_overgrowth_prof_1 | TP26 | 31 | 10.4 | 17.4 | 1.4 | 0.3 | 0.43 | -0.07 | 0.46 | 0.80 | 0.08 | 3.1 (0.03) | 0.58 (0.85) | 2.0 |
| WHT01_bigol2_overgrowth_prof_1 | TP26 | 19 | 0.0 | -4.0 | 0.0 | 0.0 | -0.09 | -0.09 | 0.14 | 0.76 | 0.95 | 2.5 (0.16) | 3.05 (0.09) | 1.9 |
| WHT01_Ol_63_prof_1 | TP26 | 27 | 4.0 | -1.7 | 0.3 | 0.2 | 0.25 | -0.10 | 0.89 | 0.60 | 0.80 | 2.1 (0.81) | 1.59 (0.18) | -0.8 |
| WHT01_Ol_180_prof_1 | TP26 | 23 | 0.0 | -4.0 | 0.0 | 0.0 | -0.12 | -0.12 | 0.48 | 0.25 | 0.32 | 2.7 (0.10) | 2.63 (0.05) | -0.2 |
| WHT01_Ol_181_prof_2 | TP26 | 24 | 70.0 | -3.6 | 0.2 | 0.1 | -0.20 | -0.21 | 0.36 | 0.49 | 0.91 | 2.5 (0.27) | 1.98 (0.13) | 2.0 |
| WHT01_Ol_190_prof_1 | TP26 | 17 | 14.7 | -4.0 | 0.0 | 0.0 | -0.01 | -0.04 | 0.78 | 1.00 | 0.02 | 2.9 (<0.01) | 4.91 (0.02) | -3.2 |
| WHT01_Ol_191_prof_1 | TP26 | 20 | 0.0 | -4.0 | 0.0 | 0.0 | -0.22 | -0.22 | 0.77 | 0.49 | 0.08 | 2.5 (0.12) | 5.90 (<0.01) | 1.9 |
| WHT01_Ol_192_prof_1 | TP26 | 19 | 34.6 | 3.9 | 0.0 | -0.0 | 0.46 | -0.07 | 0.94 | 0.93 | 0.79 | 2.1 (0.55) | 1.50 (0.21) | 2.0 |
| WHT01_Ol_193_prof_1 | TP26 | 20 | 4.8 | -3.4 | 0.5 | 0.2 | 0.18 | 0.01 | 0.54 | 0.94 | 0.57 | 1.8 (0.95) | 1.40 (0.23) | 2.0 |
| Xenocryst_ol6_prof_1 | TP26 | 15 | 2.5 | 5.0 | 3.1 | 1.3 | 0.59 | 0.01 | 0.50 | 0.52 | 0.49 | 2.1 (0.45) | 6.60 (<0.01) | -9.4 |
| Xenocryst_ol6_prof_2 | TP26 | 23 | 2.2 | 9.6 | 2.2 | 0.4 | 0.64 | -0.17 | 0.85 | 0.05 | 0.81 | 2.6 (0.14) | 2.43 (0.04) | -1.0 |
| WHT06_ol43_prof 1 | TP26 | 28 | 4.0 | 10.5 | 1.9 | 0.9 | 0.59 | -0.14 | 0.70 | 0.15 | 0.76 | 1.9 (0.89) | 1.81 (0.08) | -7.3 |
| Xenocryst_ol23_prof_1 | TP26 | 17 | 21.8 | -2.9 | 0.1 | 0.1 | -0.14 | -0.12 | 0.90 | 0.19 | 0.08 | 2.9 (0.02) | 1.17 (0.23) | 1.0 |
| WHT06_ol48_prof 1 | TP26 | 26 | 2.6 | -1.8 | 0.6 | 0.2 | 0.26 | 0.07 | 0.18 | 0.68 | 0.30 | 2.8 (0.08) | 3.44 (<0.01) | -6.6 |
| packaged example (WHT01_Ol_63_prof_1) | ex | 27 | 4.0 | -1.7 | 0.3 | 0.2 | 0.25 | -0.10 | 0.88 | 0.59 | 0.77 | 2.1 (0.72) | 1.59 (0.22) | -0.8 |

## 4. Results B: synthetic coverage

![Coverage](fig1_coverage.png)

**Figure 1.** 1σ (top) and 2σ (bottom) coverage of the true time, per case and
geometry, 1000 trials each. Diamonds: the default. Open circles: the
independent-error baseline. Triangles: the near-step candidate of section 5 (500
trials, shown only where run). Vertical line: target. Grey band: 95% sampling range
for 1000 trials.

Coverage (%) of the 1σ and 2σ ranges for the default and the independent-error
baseline, median error of the best fit (bias, % of the true time) and median 1σ width
(% of the true time). 1000 trials per case. Sampling band: 65.4–71.1 (1σ), 94.1–96.7
(2σ).

| geometry | case | 1s default | 1s indep. | 2s default | 2s indep. | bias % | width % | indep. width % |
|---|---|---|---|---|---|---|---|---|
| KS 29 pts | white (control) | 70 | 66 | 96.7 | 94.2 | -0.2 | 20 | 19 |
| KS 29 pts | sq-exp L=3 (control) | 70 | 66 | 95.4 | 93.0 | -0.6 | 45 | 40 |
| KS 29 pts | sq-exp L=10 (control) | 67 | 50 | 92.2 | 79.8 | -2.1 | 60 | 38 |
| KS 29 pts | sq-exp L=30 (control) | 58 | 32 | 88.5 | 58.3 | -1.9 | 68 | 35 |
| KS 29 pts | exponential L=3 | 70 | 66 | 95.3 | 92.2 | -0.5 | 45 | 40 |
| KS 29 pts | exponential L=10 | 66 | 53 | 91.6 | 82.1 | -0.9 | 53 | 39 |
| KS 29 pts | exponential L=30 | 58 | 36 | 86.8 | 66.8 | -0.8 | 59 | 36 |
| KS 29 pts | Matern 3/2 L=3 | 70 | 66 | 95.4 | 92.7 | -0.5 | 45 | 40 |
| KS 29 pts | Matern 3/2 L=10 | 66 | 51 | 90.8 | 80.9 | -1.5 | 56 | 39 |
| KS 29 pts | Matern 3/2 L=30 | 58 | 33 | 86.7 | 62.1 | -2.2 | 63 | 35 |
| KS 29 pts | near-step sq-exp L=3 | 66 | 60 | 92.5 | 89.5 | -0.2 | 25 | 23 |
| KS 29 pts | near-step sq-exp L=10 | 60 | 48 | 88.9 | 78.9 | -0.4 | 27 | 22 |
| KS 29 pts | strong near-step L=3 | 60 | 51 | 87.8 | 81.0 | -1.5 | 47 | 41 |
| KS 29 pts | strong near-step L=10 | 50 | 32 | 77.4 | 63.1 | -1.8 | 53 | 36 |
| KS 29 pts | Student-t3 white | 74 | 70 | 96.5 | 94.6 | +0.2 | 18 | 16 |
| KS 29 pts | 1-2 outliers (5 sd) | 72 | 67 | 94.8 | 92.7 | -0.2 | 31 | 28 |
| KS 29 pts | sd grows with \|Fo − core\| | 76 | 64 | 96.2 | 92.1 | +0.0 | 22 | 17 |
| KS 29 pts | sd rises 4x to the rim | 60 | 54 | 91.1 | 86.2 | +0.0 | 20 | 18 |
| paper 17 pts | white (control) | 72 | 65 | 97.1 | 94.7 | +0.0 | 14 | 12 |
| paper 17 pts | sq-exp L=3 (control) | 68 | 43 | 95.0 | 75.6 | +1.0 | 87 | 48 |
| paper 17 pts | sq-exp L=10 (control) | 52 | 30 | 81.1 | 52.9 | -0.1 | 78 | 38 |
| paper 17 pts | sq-exp L=30 (control) | 62 | 30 | 88.9 | 56.2 | +0.8 | 46 | 35 |
| paper 17 pts | exponential L=3 | 63 | 48 | 92.8 | 80.7 | +2.2 | 72 | 50 |
| paper 17 pts | exponential L=10 | 56 | 34 | 85.6 | 62.6 | -0.1 | 79 | 44 |
| paper 17 pts | exponential L=30 | 57 | 34 | 84.1 | 61.9 | -0.6 | 68 | 39 |
| paper 17 pts | Matern 3/2 L=3 | 65 | 46 | 93.2 | 78.7 | +1.8 | 78 | 49 |
| paper 17 pts | Matern 3/2 L=10 | 56 | 31 | 83.4 | 57.0 | +0.8 | 82 | 41 |
| paper 17 pts | Matern 3/2 L=30 | 59 | 32 | 86.8 | 57.1 | -0.5 | 59 | 36 |
| paper 17 pts | near-step sq-exp L=3 | 71 | 47 | 94.5 | 81.3 | +0.2 | 51 | 30 |
| paper 17 pts | near-step sq-exp L=10 | 56 | 33 | 81.6 | 56.6 | +0.0 | 50 | 26 |
| paper 17 pts | strong near-step L=3 | 62 | 44 | 90.3 | 73.2 | -0.6 | 136 | 77 |
| paper 17 pts | strong near-step L=10 | 50 | 26 | 79.4 | 48.0 | -2.2 | 117 | 47 |
| paper 17 pts | Student-t3 white | 70 | 65 | 96.3 | 92.1 | +0.2 | 12 | 10 |
| paper 17 pts | 1-2 outliers (5 sd) | 69 | 63 | 96.3 | 93.8 | +0.5 | 25 | 22 |
| paper 17 pts | sd grows with \|Fo − core\| | 57 | 49 | 89.3 | 82.0 | -0.5 | 14 | 12 |
| paper 17 pts | sd rises 4x to the rim | 61 | 52 | 92.7 | 86.9 | +0.0 | 14 | 12 |
| SH63 298 pts | white (control) | 69 | 66 | 96.2 | 94.8 | +0.0 | 7 | 7 |
| SH63 298 pts | sq-exp L=3 (control) | 65 | 36 | 91.4 | 62.5 | -0.2 | 19 | 8 |
| SH63 298 pts | sq-exp L=10 (control) | 64 | 30 | 89.8 | 51.8 | +0.2 | 17 | 8 |
| SH63 298 pts | sq-exp L=30 (control) | 64 | 35 | 93.3 | 64.7 | +0.0 | 10 | 8 |
| SH63 298 pts | exponential L=3 | 62 | 39 | 89.7 | 68.2 | -0.2 | 14 | 8 |
| SH63 298 pts | exponential L=10 | 59 | 31 | 88.9 | 58.0 | +0.2 | 17 | 8 |
| SH63 298 pts | exponential L=30 | 61 | 35 | 89.7 | 61.6 | +0.2 | 14 | 8 |
| SH63 298 pts | Matern 3/2 L=3 | 64 | 36 | 91.4 | 64.8 | -0.2 | 17 | 8 |
| SH63 298 pts | Matern 3/2 L=10 | 62 | 31 | 90.2 | 53.2 | +0.0 | 17 | 8 |
| SH63 298 pts | Matern 3/2 L=30 | 61 | 35 | 90.6 | 61.7 | +0.2 | 11 | 8 |
| SH63 298 pts | near-step sq-exp L=3 | 51 | 43 | 81.9 | 73.8 | +0.0 | 8 | 7 |
| SH63 298 pts | near-step sq-exp L=10 | 50 | 39 | 80.7 | 67.5 | +0.0 | 8 | 7 |
| SH63 298 pts | strong near-step L=3 | 45 | 21 | 72.4 | 41.6 | -0.5 | 17 | 8 |
| SH63 298 pts | strong near-step L=10 | 45 | 17 | 71.5 | 32.7 | +0.5 | 17 | 8 |
| SH63 298 pts | Student-t3 white | 71 | 69 | 96.5 | 95.8 | +0.0 | 7 | 7 |
| SH63 298 pts | 1-2 outliers (5 sd) | 69 | 64 | 95.9 | 95.0 | -0.2 | 8 | 7 |
| SH63 298 pts | sd grows with \|Fo − core\| | 33 | 30 | 60.6 | 55.2 | -0.2 | 7 | 7 |
| SH63 298 pts | sd rises 4x to the rim | 53 | 49 | 85.6 | 82.9 | +0.0 | 7 | 7 |

What the table shows:
- **Controls reproduce the 2026-10-13 validation.**
  - White noise: 69–72% (1σ), 96.2–97.1% (2σ). The 17-point paper geometry is
    slightly conservative: 72.3% and 97.1%, just above the band.
  - Squared-exponential misfit: 64–70% for L = 3 and 10 µm on KS and SH63, and 58–64%
    at L = 30 µm (the known trade-off when L approaches the diffusion length).
- **New: the paper geometry is hit hardest by misfit as long as the diffusion
  length.** With L = 10 µm against l = 8.5 µm, coverage is 52% (1σ) and 81% (2σ),
  below the 53–62% the method note quotes for KS and SH63. The independent-error
  baseline gives 30%.
- **The wrong kernel shape costs little.** Against the squared-exponential at the
  same L:
  - exponential: −2 to 0 points on KS, −5 to +4 on paper, −3 to −4 on SH63;
  - Matérn 3/2: −1 to 0 on KS, −3 to +4 on paper, −1 to −3 on SH63.
  - The lowest wrong-shape value is 56% (paper, L = 10 µm), and there the
    squared-exponential itself gives 52%.
- **Near-step misfit is the largest loss among the misfit cases.**
  - At 0.4 mol% Fo peak: 66 and 60% on KS, 71 and 56% on paper, 51 and 50% on SH63.
  - At 1.2 mol%: 60 and 50% on KS, 62 and 50% on paper, 45 and 45% on SH63.
  - 2σ coverage falls to 72–95%. Widths barely grow on SH63 (8%, as for white noise),
    so the range does not see the problem.
- **Heavy tails do not hurt coverage.**
  - Student-t: 70–74%. Outliers: 69–72%.
  - Outliers make the range wider: KS 20 to 31%, paper 14 to 25%, SH63 7.2 to 7.7% of
    the true time.
- **Heteroscedastic noise can hurt badly or help, depending on where it sits.**
  - sd rising 4× to the rim: 53–61%.
  - sd proportional to |Fo − core|: 76% on KS (over-covers), 57% on paper and 33% on
    SH63. On SH63 the noisiest points (the rim and the plateau) are about 11× noisier
    than the core, an extreme case.
- **Bias is small in every case**: median within ±2.3% of the true time.
- **The independent-error baseline is worse in every correlated case**: 17–60% at 1σ
  where the point spacing resolves the misfit. It reaches 66% only when the misfit is
  shorter than the spacing (L = 3 µm on the 10 µm KS spacing, where it acts as white
  noise). It is close to the default only for white or heavy-tailed noise.
- **Whiteness tests do not detect near-step misfit.** In the near-step cases,
  Ljung-Box flags 2–18% of trials. Shapiro-Wilk flags 6–17%, except 46–70% for the
  strong cases on KS.

The two cheap candidates help nowhere (table in the appendix):
- **Kernel chosen by AIC:** coverage changes by −8 to +3 points against the default,
  and is lower in most cases.
- **Leave-one-out outlier clip (|z| > 4):** coverage changes by −2.3 to +0.1 points,
  except −3.8 and −4.7 for the Fo-proportional cases on KS and SH63. It narrows the range only on the dense
  profile (outliers: 7.7 to 7.3%). On sparse profiles it found outliers in only 0–11%
  of trials, because one 5-sd outlier among 17–29 points inflates the fitted s enough
  to hide itself.

## 5. A candidate fix for near-step misfit

Two noise models were tried, each with one extra parameter fitted by maximum
likelihood together with s, ρ and L at the white best fit (`zone_noise.py`). Here
e(x) is a Gaussian bump of width l (at the white best fit) around each step, with
peak 1:
- **near-step white**: C = s²(diag(1 + g·e) + ρ²K). The white noise may grow near the
  step. **It does not help:** 1σ coverage stays at 51–54% (SH63 near-step) and 43–45%
  (SH63 strong). Not recommended.
- **near-step correlated**: C = s²(I + ρ² S K S), with S = diag(√(1 + h·e)) and h on
  0, 1, 3, 10, 30, 100. The *correlated misfit's* amplitude may grow near the step
  (h = 0 is FORSTER's model). This matches the shape of the synthetic near-step cases,
  so for them it is a best case. The heteroscedastic cases have different shapes.

Coverage (%) for the default and the near-step correlated candidate, 500 trials. The
"near/far raw rms" column is the median ratio of residual RMS near the step to that
far from it at the default best fit, the same statistic as A4 on the real profiles.
"dAIC" is the AIC change from adding the near-step term (negative favours it); the
two dAIC columns give the share of trials below −2 and −6. The last column is the
near-step white candidate.

| geometry | case | near/far raw rms | 1s default | 1s candidate | 2s default | 2s candidate | width default % | width candidate % | dAIC<-2 % | dAIC<-6 % | 1s white-near-step cand. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| KS 29 pts | white (control) | 0.93 | 73 | 74 | 97 | 97 | 20 | 21 | 1 | 0 | 75 |
| KS 29 pts | sq-exp L=10 (control) | 0.87 | 68 | 68 | 93 | 94 | 60 | 62 | 1 | 0 | 69 |
| KS 29 pts | exponential L=10 | 0.91 | 67 | 67 | 92 | 91 | 53 | 54 | 1 | 0 | 68 |
| KS 29 pts | near-step sq-exp L=3 | 1.55 | 66 | 79 | 94 | 97 | 25 | 31 | 22 | 8 | 76 |
| KS 29 pts | near-step sq-exp L=10 | 1.45 | 60 | 74 | 90 | 95 | 27 | 32 | 31 | 13 | 63 |
| KS 29 pts | strong near-step L=3 | 3.59 | 58 | 90 | 88 | 98 | 46 | 75 | 88 | 76 | 70 |
| KS 29 pts | strong near-step L=10 | 3.14 | 48 | 75 | 77 | 92 | 51 | 59 | 88 | 76 | 51 |
| KS 29 pts | sd grows with \|Fo − core\| | 5.40 | 77 | 97 | 97 | 100 | 23 | 33 | 98 | 96 | 78 |
| KS 29 pts | sd rises 4x to the rim | 1.59 | 61 | 71 | 93 | 95 | 20 | 25 | 23 | 7 | 65 |
| paper 17 pts | white (control) | 1.03 | 74 | 74 | 97 | 97 | 15 | 15 | 1 | 0 | 72 |
| paper 17 pts | sq-exp L=10 (control) | 0.90 | 52 | 49 | 81 | 81 | 78 | 76 | 4 | 0 | 51 |
| paper 17 pts | exponential L=10 | 0.99 | 56 | 53 | 86 | 86 | 81 | 79 | 2 | 0 | 54 |
| paper 17 pts | near-step sq-exp L=3 | 2.66 | 72 | 75 | 94 | 97 | 52 | 47 | 49 | 12 | 65 |
| paper 17 pts | near-step sq-exp L=10 | 2.05 | 55 | 59 | 82 | 83 | 51 | 42 | 23 | 2 | 53 |
| paper 17 pts | strong near-step L=3 | 5.35 | 63 | 66 | 91 | 91 | 142 | 127 | 75 | 37 | 57 |
| paper 17 pts | strong near-step L=10 | 3.28 | 48 | 53 | 79 | 78 | 121 | 91 | 41 | 14 | 48 |
| paper 17 pts | sd grows with \|Fo − core\| | 4.73 | 57 | 64 | 90 | 93 | 14 | 15 | 17 | 5 | 53 |
| paper 17 pts | sd rises 4x to the rim | 2.33 | 62 | 65 | 93 | 94 | 14 | 15 | 8 | 1 | 57 |
| SH63 298 pts | white (control) | 0.99 | 69 | 70 | 97 | 97 | 7 | 7 | 1 | 0 | 70 |
| SH63 298 pts | sq-exp L=10 (control) | 0.94 | 65 | 64 | 91 | 91 | 17 | 17 | 1 | 0 | 65 |
| SH63 298 pts | exponential L=10 | 0.95 | 60 | 61 | 88 | 88 | 16 | 17 | 2 | 0 | 60 |
| SH63 298 pts | near-step sq-exp L=3 | 1.08 | 53 | 63 | 83 | 91 | 8 | 10 | 37 | 15 | 54 |
| SH63 298 pts | near-step sq-exp L=10 | 1.07 | 51 | 61 | 83 | 89 | 8 | 9 | 35 | 18 | 51 |
| SH63 298 pts | strong near-step L=3 | 1.61 | 46 | 66 | 74 | 90 | 17 | 24 | 97 | 94 | 43 |
| SH63 298 pts | strong near-step L=10 | 1.46 | 46 | 62 | 70 | 88 | 16 | 21 | 86 | 77 | 45 |
| SH63 298 pts | sd grows with \|Fo − core\| | 7.23 | 34 | 91 | 63 | 100 | 7 | 23 | 100 | 100 | 36 |
| SH63 298 pts | sd rises 4x to the rim | 1.62 | 53 | 70 | 88 | 96 | 7 | 11 | 89 | 72 | 63 |

- **Well-specified cases barely change:** within ±1 point, except paper SE L = 10
  (52 to 49) and exponential L = 10 (56 to 53).
- **Near-step and rim cases recover:**
  - KS: 48–66% to 74–90%;
  - SH63: 46–53% to 61–66% (2σ: 70–83% to 88–91%);
  - rim-rising sd: 53–62% to 65–71%.
  - Ranges widen where coverage was lost, e.g. SH63 strong 16–17% to 21–24%, KS
    strong L = 3 µm 46% to 75%.
- **The 17-point paper geometry gains only 3–7 points.** Too few points sit near the
  step to estimate the extra term.
- **As a diagnostic** (ΔAIC of the near-step term at the white best fit):
  - false alarms at ΔAIC < −2 in 0.6–4% of control trials, at ΔAIC < −6 in at most
    0.2%;
  - detection at ΔAIC < −6: 76–94% of the strong cases on KS and SH63, but 14–37% on
    paper and 2–18% for the 0.4 mol% cases.
  - A simpler test, Brown-Forsythe on near/far z, is weaker: it fired on 3–34% of SH63
    near-step trials.
- **On the real profiles:** preferred by more than 2 AIC on 11 of 36 and by more than
  6 on 7 (the four SH63olv066, Xenocryst_ol6_prof_1, WHT06_ol43, WHT06_ol48).
  - h reached the top of its grid (100) in 11 profiles, so a continuous fit may move
    these further.
  - The fitted time changes by −10 to −15% for SH63olv066, +54% for
    Xenocryst_ol6_prof_1, +18% for WHT06_ol43 and −12% for WHT06_ol48.
  - 3 of the 11 best fits fall outside the default 1σ range.

| profile | dAIC of near-step term | h | default time (d) | default 1σ low | default 1σ high | candidate time (d) | candidate 1σ low | candidate 1σ high | candidate time outside default 1σ |
|---|---|---|---|---|---|---|---|---|---|
| KS20-527_2-transect-1 | -2.884 | 30 | 40.326 | 33.673 | 47.259 | 41.306 | 31.718 | 50.786 | False |
| KS20-527_2-transect-2 | -2.07 | 10 | 36.646 | 30.124 | 43.52 | 38.271 | 31.62 | 45.133 | False |
| KS20-527_2-transect-4 | -3.888 | 30 | 21.257 | 16.998 | 26.829 | 26.778 | 20.544 | 33.637 | False |
| SH63olv066-1 | -30.562 | 100 | 10.817 | 9.046 | 12.861 | 9.192 | 7.355 | 11.02 | False |
| SH63olv066-2 | -25.29 | 100 | 10.758 | 9.514 | 12.123 | 9.317 | 7.274 | 11.393 | True |
| SH63olv066-3 | -19.049 | 100 | 10.781 | 9.51 | 12.187 | 9.693 | 8.184 | 11.295 | False |
| SH63olv066-4 | -18.173 | 100 | 9.86 | 8.651 | 11.207 | 8.75 | 7.342 | 10.245 | False |
| AZ18 WHT01_Ol_190_prof_1 | -3.17 | 30 | 11.992 | 9.81 | 14.546 | 13.296 | 9.086 | 18.097 | False |
| Xenocryst_ol6_prof_1 | -9.397 | 100 | 0.148 | 0.103 | 0.206 | 0.228 | 0.184 | 0.275 | True |
| AZ18 WHT06_ol43_prof 1 | -7.288 | 30 | 1.923 | 1.528 | 2.392 | 2.268 | 1.862 | 2.703 | False |
| AZ18 WHT06_ol48_prof 1 | -6.635 | 100 | 3.039 | 2.77 | 3.332 | 2.662 | 2.383 | 2.958 | True |

This study took D at the mean measured Fo. FORSTER's `near_step_check` (and so
`near_step_daic` in `summary.csv`) takes D at the middle of the initial condition's Fo
range, so its values differ slightly: Xenocryst_ol6_prof_1 gives −10.4 and +58%, not
−9.4 and +54%. The figures in `docs/references/noise-models-guide.md` use FORSTER's values.

## 6. Verdict

**Valid:**
- Whitening with the squared-exponential model is valid for the serial correlation in
  these profiles. After it, serial correlation is at chance level (Ljung-Box flags 1
  of 36, the runs test 2 of 36).
- The kernel's shape is not a practical concern: wrong shapes cost 0–5 points of
  coverage.
- Heavy tails and isolated outliers leave coverage at 69–74%.

**Degrades:**
1. *Misfit concentrated near the step* (seen in the real data: 30 of 34 profiles,
   12 significant):
   - 1σ coverage 45–71% and 2σ 72–95% in synthetic tests;
   - worst for dense profiles, 45–51%;
   - the real SH63olv066 profiles exceed the strongest synthetic case.
2. *Misfit about as long as the diffusion length* (already known):
   - 52–64% at 1σ;
   - 52% on a paper-like 17-point profile, lower than the method note reports.
3. *Noise that grows toward the rim or with |Fo − core|*: 33–61% where the noisy part
   coincides with the time information. The real data cannot separate this from (1).

**Recommendations**, each backed by the numbers above:
1. **Do not add a Matérn kernel option.** Real profiles never preferred one. Wrong
   shapes cost at most 5 points, and choosing the kernel by AIC lowered coverage by up
   to 8 points.
2. **Do not prioritise a robust likelihood.** Coverage with outliers and Student-t
   noise is 69–74%. The cost is width (+50 to +75% on sparse profiles), which a
   Student-t likelihood might recover. That was not tested. Simple leave-one-out
   clipping did not help sparse profiles.
3. **Add a near-step warning now.** Fit the near-step correlated term at the white
   best fit, and warn when it improves AIC by more than 6.
   - False alarms ≤0.2% of control trials.
   - Detects 76–94% of strong near-step misfit on 29- and 298-point profiles.
   - Fires on 7 real profiles.
   - Extra cost: about 6× the Cholesky factorisations of the noise-model fit, about
     0.16 s on a 298-point profile.
4. **Consider making the near-step term part of the model**, after:
   - fitting h continuously (it reached its grid maximum in 11 real profiles);
   - testing other non-stationary shapes;
   - a differential check that h = 0 reproduces today's fits.

   On synthetic data it restored dense-profile coverage from 46–53% to 61–66%. It
   changes SH63olv066 by −10 to −15%.
5. **Until then:** for dense profiles with near-step scatter (SH63olv066-like), report
   the 2σ range (88–91% with the candidate, 70–83% without), or flag the 1σ range as
   too narrow. Update the method note's section 4 with the paper-geometry number (52%
   at L ≈ l).

## 7. Reproducing

From the repository root. At most 4 worker processes, one heavy job at a time.
`WHITENING_CACHE` is where the model-at-points arrays are cached (default: the system
temp directory).

```bash
uv run python docs/plans/2026-10-15-whitening-validity/real_profiles.py      # 56 s
uv run python docs/plans/2026-10-15-whitening-validity/synthetic.py run 1000  # 13 min, 4 workers
uv run python docs/plans/2026-10-15-whitening-validity/zone_noise.py run 500  # 9.5 min, 4 workers
uv run python docs/plans/2026-10-15-whitening-validity/zone_noise.py real     # 10 s
uv run python docs/plans/2026-10-15-whitening-validity/figures.py
uv run python docs/plans/2026-10-15-whitening-validity/report.py              # the tables above
```

Files: `common.py` (kernels, noise-model fits, whiteness tests), `real_profiles.py` →
`real_profiles.csv`, `real_kernels.csv`; `synthetic.py` → `trials.csv`,
`coverage.csv`; `zone_noise.py` → `zone_trials.csv`, `zone_coverage.csv`,
`zone_real.csv`; `figures.py` → `fig1_coverage.png`, `fig2_aic.png`,
`fig3_whitened.png`; `report.py` prints every table here. The round robin data are
private: no measured profile is plotted or copied here, only fit statistics.

Limitations:
- One true time per geometry.
- Noise models are fitted on FORSTER's grid; only part A refines them continuously.
- The near-step candidate's shape matches the synthetic near-step cases.
- The bootstrap uses 200 sets, so its smallest p-value is 0.005.
- Initial-condition errors are not studied here (see `forster.scan`).

## Appendix: the two cheap candidates and the diagnostics, per case

1000 trials. "kernel by AIC": kernel chosen by AIC at the white best fit. "outlier
clip": one refit without points whose |standardised LOO residual| > 4. "AIC not
sq-exp": share of trials where AIC chose the exponential or Matérn 3/2. "clipped":
share with at least one point dropped. "LB/SW p<.05": share where Ljung-Box (lags 1–5)
or Shapiro-Wilk of the whitened residuals gives nominal p < 0.05. "warn": FORSTER's
profile-wide-misfit warning.

| geometry | case | 1s default | 1s kernel by AIC | 1s outlier clip | 2s default | 2s kernel by AIC | 2s outlier clip | AIC not sq-exp % | clipped % | LB p<.05 % | SW p<.05 % | warn % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| KS 29 pts | white (control) | 70 | 67 | 70 | 97 | 95 | 97 | 1 | 0 | 5 | 5 | 0 |
| KS 29 pts | sq-exp L=3 (control) | 70 | 68 | 70 | 95 | 94 | 95 | 1 | 0 | 7 | 4 | 0 |
| KS 29 pts | sq-exp L=10 (control) | 67 | 66 | 67 | 92 | 89 | 92 | 15 | 0 | 3 | 4 | 4 |
| KS 29 pts | sq-exp L=30 (control) | 58 | 57 | 58 | 88 | 88 | 88 | 23 | 0 | 2 | 5 | 1 |
| KS 29 pts | exponential L=3 | 70 | 67 | 70 | 95 | 94 | 95 | 0 | 0 | 6 | 4 | 0 |
| KS 29 pts | exponential L=10 | 66 | 60 | 66 | 92 | 87 | 92 | 7 | 0 | 3 | 4 | 1 |
| KS 29 pts | exponential L=30 | 58 | 58 | 58 | 87 | 84 | 87 | 25 | 0 | 1 | 4 | 3 |
| KS 29 pts | Matern 3/2 L=3 | 70 | 68 | 70 | 95 | 94 | 95 | 1 | 0 | 7 | 4 | 0 |
| KS 29 pts | Matern 3/2 L=10 | 66 | 62 | 66 | 91 | 88 | 91 | 10 | 0 | 2 | 4 | 2 |
| KS 29 pts | Matern 3/2 L=30 | 58 | 59 | 58 | 87 | 87 | 87 | 31 | 0 | 2 | 4 | 2 |
| KS 29 pts | near-step sq-exp L=3 | 66 | 62 | 66 | 92 | 92 | 93 | 1 | 1 | 7 | 15 | 0 |
| KS 29 pts | near-step sq-exp L=10 | 60 | 55 | 60 | 89 | 85 | 89 | 3 | 0 | 3 | 9 | 1 |
| KS 29 pts | strong near-step L=3 | 60 | 56 | 58 | 88 | 85 | 86 | 1 | 12 | 18 | 70 | 1 |
| KS 29 pts | strong near-step L=10 | 50 | 47 | 50 | 77 | 74 | 77 | 7 | 3 | 6 | 46 | 4 |
| KS 29 pts | Student-t3 white | 74 | 71 | 72 | 96 | 96 | 96 | 0 | 10 | 4 | 39 | 0 |
| KS 29 pts | 1-2 outliers (5 sd) | 72 | 69 | 70 | 95 | 95 | 94 | 1 | 11 | 4 | 80 | 0 |
| KS 29 pts | sd grows with \|Fo − core\| | 76 | 70 | 72 | 96 | 95 | 95 | 4 | 18 | 31 | 85 | 3 |
| KS 29 pts | sd rises 4x to the rim | 60 | 57 | 60 | 91 | 89 | 91 | 1 | 0 | 11 | 14 | 0 |
| paper 17 pts | white (control) | 72 | 69 | 72 | 97 | 96 | 97 | 0 | 0 | 6 | 4 | 1 |
| paper 17 pts | sq-exp L=3 (control) | 68 | 65 | 68 | 95 | 92 | 95 | 7 | 0 | 2 | 5 | 22 |
| paper 17 pts | sq-exp L=10 (control) | 52 | 51 | 52 | 81 | 81 | 81 | 22 | 0 | 3 | 6 | 8 |
| paper 17 pts | sq-exp L=30 (control) | 62 | 61 | 62 | 89 | 87 | 89 | 23 | 0 | 8 | 6 | 33 |
| paper 17 pts | exponential L=3 | 63 | 55 | 63 | 93 | 89 | 93 | 4 | 0 | 3 | 6 | 6 |
| paper 17 pts | exponential L=10 | 56 | 53 | 56 | 86 | 82 | 86 | 21 | 0 | 2 | 5 | 14 |
| paper 17 pts | exponential L=30 | 57 | 56 | 57 | 84 | 84 | 84 | 32 | 0 | 3 | 5 | 20 |
| paper 17 pts | Matern 3/2 L=3 | 65 | 57 | 65 | 93 | 90 | 93 | 7 | 0 | 2 | 5 | 8 |
| paper 17 pts | Matern 3/2 L=10 | 56 | 55 | 56 | 83 | 84 | 83 | 30 | 0 | 1 | 7 | 10 |
| paper 17 pts | Matern 3/2 L=30 | 59 | 60 | 59 | 87 | 86 | 87 | 26 | 0 | 4 | 6 | 25 |
| paper 17 pts | near-step sq-exp L=3 | 71 | 65 | 71 | 94 | 91 | 94 | 8 | 0 | 4 | 12 | 11 |
| paper 17 pts | near-step sq-exp L=10 | 56 | 54 | 56 | 82 | 79 | 82 | 16 | 0 | 2 | 6 | 4 |
| paper 17 pts | strong near-step L=3 | 62 | 60 | 62 | 90 | 88 | 90 | 8 | 0 | 3 | 17 | 27 |
| paper 17 pts | strong near-step L=10 | 50 | 51 | 50 | 79 | 80 | 79 | 18 | 0 | 2 | 6 | 5 |
| paper 17 pts | Student-t3 white | 70 | 67 | 70 | 96 | 96 | 96 | 0 | 0 | 6 | 24 | 1 |
| paper 17 pts | 1-2 outliers (5 sd) | 69 | 67 | 69 | 96 | 96 | 96 | 0 | 0 | 4 | 59 | 1 |
| paper 17 pts | sd grows with \|Fo − core\| | 57 | 52 | 57 | 89 | 88 | 89 | 0 | 0 | 13 | 20 | 2 |
| paper 17 pts | sd rises 4x to the rim | 61 | 56 | 61 | 93 | 90 | 93 | 0 | 0 | 9 | 10 | 1 |
| SH63 298 pts | white (control) | 69 | 66 | 69 | 96 | 95 | 96 | 2 | 2 | 4 | 6 | 0 |
| SH63 298 pts | sq-exp L=3 (control) | 65 | 65 | 65 | 91 | 92 | 91 | 37 | 2 | 4 | 7 | 0 |
| SH63 298 pts | sq-exp L=10 (control) | 64 | 64 | 64 | 90 | 90 | 90 | 37 | 2 | 5 | 6 | 0 |
| SH63 298 pts | sq-exp L=30 (control) | 64 | 65 | 64 | 93 | 94 | 93 | 36 | 2 | 5 | 6 | 4 |
| SH63 298 pts | exponential L=3 | 62 | 63 | 62 | 90 | 92 | 90 | 65 | 2 | 2 | 6 | 0 |
| SH63 298 pts | exponential L=10 | 59 | 62 | 59 | 89 | 90 | 89 | 69 | 3 | 5 | 6 | 0 |
| SH63 298 pts | exponential L=30 | 61 | 63 | 61 | 90 | 90 | 90 | 66 | 2 | 5 | 7 | 1 |
| SH63 298 pts | Matern 3/2 L=3 | 64 | 65 | 64 | 91 | 92 | 91 | 56 | 2 | 4 | 6 | 0 |
| SH63 298 pts | Matern 3/2 L=10 | 62 | 61 | 62 | 90 | 90 | 90 | 52 | 3 | 4 | 7 | 0 |
| SH63 298 pts | Matern 3/2 L=30 | 61 | 63 | 61 | 91 | 92 | 91 | 52 | 2 | 5 | 6 | 2 |
| SH63 298 pts | near-step sq-exp L=3 | 51 | 48 | 51 | 82 | 80 | 82 | 10 | 2 | 3 | 7 | 0 |
| SH63 298 pts | near-step sq-exp L=10 | 50 | 47 | 50 | 81 | 78 | 81 | 14 | 2 | 4 | 6 | 0 |
| SH63 298 pts | strong near-step L=3 | 45 | 44 | 45 | 72 | 72 | 72 | 29 | 2 | 5 | 10 | 0 |
| SH63 298 pts | strong near-step L=10 | 45 | 44 | 45 | 72 | 72 | 71 | 28 | 2 | 6 | 7 | 0 |
| SH63 298 pts | Student-t3 white | 71 | 69 | 70 | 96 | 96 | 96 | 1 | 96 | 4 | 100 | 0 |
| SH63 298 pts | 1-2 outliers (5 sd) | 69 | 65 | 68 | 96 | 95 | 96 | 1 | 87 | 4 | 84 | 0 |
| SH63 298 pts | sd grows with \|Fo − core\| | 33 | 32 | 28 | 61 | 59 | 52 | 2 | 100 | 78 | 100 | 0 |
| SH63 298 pts | sd rises 4x to the rim | 53 | 50 | 52 | 86 | 84 | 84 | 2 | 27 | 14 | 76 | 0 |

## Follow-ups to keep evaluating (added 2026-10-15)

The near-step **warning** is now in the package (`forster.uncertainty.near_step_check`,
run by `fit_profile` on correlated fits without growth stages; summary column
`near_step_daic`; it warns below -6). It fires on the three paper profiles this study
flagged (Xenocryst_ol6_prof_1 -10.4, WHT06_ol43 -7.7, WHT06_ol48 -6.6) and not on
WHT01_Ol_63, _181 or _192. The near-step **term** is not part of the fit. Still to do:

1. **Fit h continuously.** h reaches the top of its grid (100) on all three warned
   paper profiles and on 11 of the 36 real profiles here, so the size of the
   near-step misfit, and the shift in time it implies, is not yet measured.
2. **Decide whether the term becomes part of the model.** Before switching it on:
   - a differential check that h = 0 reproduces today's fits exactly;
   - coverage on more geometries, and on the heteroscedastic cases it did not fix;
   - other non-stationary shapes (a rim zone, an asymmetric zone, two steps);
   - a review of what it does to real profiles (SH63olv066 -10 to -15%,
     Xenocryst_ol6_prof_1 +54%; 3 of 11 move outside today's 1-sigma range).
3. **Power on sparse, paper-like profiles.** The term recovered only +3 to +7 points of
   coverage there; measure the warning's detection rate at 15-20 points.
4. **Does fitting the initial condition remove it?** For SH63olv066 the round robin
   refit's wider Fo priors lowered the correlated misfit from 0.46-0.77 to 0.25 mol%.
   Rerun the check after `scan_method` fits to see whether the warning clears; if so,
   the warning should point users there first.
5. **Wider reach.** Run the check in the samplers' posterior (`forster.sampling`)
   and on growth-stage fits (both skipped today); show it in the GUI (shade the
   near-step zone, run the check on release or idle, not during drags).
6. **Regression harness.** Rerun this study (`real_profiles.py`, `synthetic.py`,
   `zone_noise.py`) after any change to the noise model, and compare with these CSVs.
7. **Round robin note** (private submodule, `outputs/refit_2026/NOTE.md`): the
   SH63olv066 ranges are probably too narrow; add the caveat or the 2-sigma ranges
   before it goes to the organisers.
8. **Done 2026-10-16 (item 4):** fitting the starting profile cleared the warning on
   SH63olv066-1 (ΔAIC −30.6 → −4.3), WHT06_ol43 (−7.7 → +1.2) and Xenocryst_ol6 (−10.4 →
   +2.0), and only just on WHT06_ol48 (−6.6 → −5.4, its near-step bump remains). Times
   moved by −4%, −40%, −52% and +6%; Xenocryst_ol6's first step ended at the edge of its
   prior, so its −52% is not reliable. The warning now says to fit the starting profile first, and the
   starting-profile fit reports `scan_near_step_daic`
   ([noise-model-findings.md](../../references/noise-model-findings.md)).
9. **Plateau length strengthens item 2:** with near-step misfit, a longer plateau makes
   the default 1σ range too narrow (73% → 46% coverage from 3 to 20 diffusion lengths),
   and only the near-step model holds (71-80%).
10. **Third noise refit** in `fit_initial_condition` (refit at the second optimum): the
    second refit moved optima by up to 0.5 µm; whether a third changes anything is
    untested. Also check that the grid method (`scan.fit_candidates`, one noise fit at the
    white best candidate, no second pass) behaves like the sampler.
11. **The model's core end:** FORSTER ends the model at the last measured point and holds
    the core Fo there; data that stop at or before the plateau (which starts about 2
    diffusion lengths past the step) moved the time by −15% to +22% at its start and far
    more on the slope. A warning covers it now (`core_end_diffusion_lengths` < 3); consider
    extending the model to the domain's end instead (a behaviour change, needs a
    differential replay).
