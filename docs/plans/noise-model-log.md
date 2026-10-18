# Running log: noise model, uncertainty and fitting the starting profile

One entry per question asked, newest last: what was tested, what was found, what was
decided, and where the details are. The thorough summary is
[noise-model-findings.md](../references/noise-model-findings.md); advice for users is in
[practical-advice.md](../guide/practical-advice.md). Add an entry whenever the noise
model, the ranges or the starting-profile fit change, or a new test is run.

## 2026-10-12 to 10-13: replacing the legacy uncertainty
- **Question:** the legacy rule (reduced χ² within 1 of its minimum, on a kriged profile)
  covered 99.7% with an exact model and 62% with smooth misfit. What should replace it?
- **Done:** a correlated-noise likelihood (white noise s plus a smooth misfit of size a and
  length L, fitted to the residuals; generalised χ² by whitening), validated by coverage
  on synthetic profiles.
- **Decided:** the correlated method is the default (`uncertainty = correlated`).
- **Details:** [likelihood-method.md](../references/likelihood-method.md),
  [fit-uncertainty.md](../references/fit-uncertainty.md),
  `docs/plans/2026-10-13-likelihood-validation/`.

## 2026-10-14: fitting the starting profile
- **Question:** with fits now taking milliseconds, how should uncertain starting-profile
  parameters be fitted?
- **Done:** compared grids, an optimiser with Laplace importance sampling, nested sampling
  and MCMC against dense reference grids.
- **Found:** the 5-point default grid put the time up to 19% off; the optimiser plus
  importance sampling is within 0.5% when the data pin the parameters, with nested
  sampling as the fallback.
- **Decided:** `scan_method` auto (optimiser, importance sampling, nested fallback); the
  grid stays as an option.
- **Details:** `analyses/ic_sampling/`, `forster.sampling`.

## 2026-10-15: is whitening valid for these profiles?
- **Done:** on 36 real profiles: kernel shape (AIC), serial correlation and normality of
  the whitened residuals, misfit near the step against far from it; on synthetic profiles:
  coverage under seven kinds of misspecified noise.
- **Found:** the squared-exponential misfit is adequate on 36/36 profiles and whitening
  removes the serial correlation. The one real failure: misfit concentrated near the
  step (30 of 34 profiles scatter more there, 12 significantly), which lowers 1σ coverage
  to 45-60%, worst on dense profiles.
- **Decided:** add a near-step check and warning; keep evaluating a near-step noise model
  (follow-ups 1-7 in the study).
- **Details:** [whitening-validity study](2026-10-15-whitening-validity/README.md).

## 2026-10-15: near-step warning built
- **Done:** `forster.uncertainty.near_step_check`: a misfit term whose size may grow
  within about one diffusion length of the steps; warn when it improves AIC by more than
  6 (`near_step_daic` in `summary.csv`).
- **Found:** it fires on exactly the three paper profiles the study flagged
  (Xenocryst_ol6_prof_1, WHT06_ol43, WHT06_ol48) and on the dense round robin profiles
  of one sample, and not on well-behaved ones.
- **Details:** commit a3a8adc; [noise-models-guide.md](../references/noise-models-guide.md).

## 2026-10-16: figures of each kind of misfit
- **Done:** a synthetic gallery (500 draws per kind of misfit) and real paper profiles
  where each noise model works best.
- **Found:** 2σ held up on every warned real profile (the near-step model's best fit lay
  inside it on every one); misfit about as long as the diffusion length and noise growing
  toward the rim go undetected.
- **Details:** [noise-models-guide.md](../references/noise-models-guide.md),
  `docs/references/figures/noise_models/`.

## 2026-10-16: does whitening dampen the choice of starting profile? (Henry's question)
- **Question:** misfit near the step is a signal that the starting profile (step
  positions, edge and plateau Fo) is not tuned to the data, the dense round robin profiles being the example.
  Does whitening dampen that signal during the parameter search? Could the AIC be used in
  the search, or to flag a starting profile that fits badly?
- **Done:** one-parameter scans of every starting-profile parameter on a dense round
  robin profile (private) and the three paper near-step profiles under four scorings (white noise only; noise model
  fixed at the nominal starting profile; refitted at the optimum, as FORSTER does; refitted
  for every candidate), joint fits, a synthetic recovery test (200 draws) and costs.
- **Found:** a noise model fitted at a poor starting profile does flatten the scans
  (1.0-1.7× wider than after the refit on the paper profiles; on synthetic profiles its
  ranges were 2-3× too wide, covering 96-99.5% instead of 68%), but FORSTER's second pass
  (refit at the first optimum) recovers it and
  lands on the same starting profile as a per-candidate refit, which costs 7× more and
  absorbs misfit in the core and plateau Fo. The near-step ΔAIC is a measure of how
  concentrated the misfit is, not how large: in the search it would reward a wrong core
  Fo. As a flag after the fit it works: it fired at the nominal starting profile on all
  near-step profiles and cleared at the fitted one (WHT06_ol48 only just, −5.4).
  Fitting the starting profile moved the time by −40% (WHT06_ol43), −52% (Xenocryst_ol6,
  not reliable: its first step ended at the edge of its prior, asking for a rim zone a
  sharp step cannot represent) and +6% (WHT06_ol48); the round robin profile moved
  little.
- **Decided:** no per-candidate noise refit and no ΔAIC in the objective. The near-step
  warning now tells users to fit the starting profile first; the starting-profile fit
  reports `scan_near_step_daic` and warns if misfit near the step remains.
- **Details:** [findings, section 2](../references/noise-model-findings.md).

## 2026-10-16: does plateau length bias the time or its range? (Henry's question)
- **Done:** synthetic profiles with 3-20 diffusion lengths of plateau under eight kinds of
  misfit (500 draws each), and real profiles with plateau points removed.
- **Found:** the best-fit time is not biased (under 1%). With misfit near the step, a
  longer plateau pulls the fitted misfit down and the default 1σ range becomes too
  narrow (coverage 73% → 46%); whitening helps but does not stop the trend; the near-step
  model holds (71-80%). A profile measured only about 2 l past the step moves the time
  by −37% to +49% (corrected in the next entry: those cuts stopped on the slope), because
  the model ends at the last measured point with the core Fo
  held there. A core Fo 0.1-0.2 mol% off biases the time 5-21%, fixed by fitting it.
- **Decided:** warn when the profile ends within 3 diffusion lengths of the last step
  (`core_end_diffusion_lengths`); the evidence strengthens the case for making the
  near-step model part of the fit (study follow-up 2).
- **Details:** [findings, section 3](../references/noise-model-findings.md).

## 2026-10-16: plateau test redone on real long plateaus (Henry's correction)
- **Problem (Henry):** the real-profile cuts were placed by a rule on the starting profile
  (2 diffusion lengths past the starting step), not on the data: the "−37%" cut on
  WHT06_ol43 stopped on the diffused slope, where the held core Fo is pegged above the
  data, and the test never used the long plateau (the run's domain ends at 80 µm, the data
  run to 132 µm).
- **Done:** the six paper profiles with the longest measured plateaus, domains extended to
  the end of the data (five of six published runs stop short of flat data); where the
  plateau starts read from the data (the full fit within 2 noise sd of its core level,
  1.7-2.3 diffusion lengths past the step); cuts at 8, 4, 2, 1, 0 diffusion lengths of
  plateau and one on the slope; fitted as FORSTER runs a short profile and with the
  full-length model.
- **Found:** with at least 1 diffusion length of plateau (about 3 past the step), every
  time within 4% of the full profile's; at the plateau's start −15% to +22%; on the slope
  −63% to off the scale. The full-length model barely moves at those cuts, so the held
  core Fo causes it. A longer plateau narrows the range and strengthens the near-step
  warning on profiles with near-step misfit (WHT06_ol43 to 132 µm: ΔAIC −18.3, against
  −8.6 at 80 µm).
- **Decided:** keep the 3-diffusion-length threshold (it marks about 1 diffusion length
  into the plateau); the warning and the docs now say the data probably stop at or before
  the plateau. Figures 3-5 of the findings redone.
- **Details:** [findings, section 3](../references/noise-model-findings.md).

## 2026-10-16: the plateau points dropped from the paper runs
- **Done:** every paper run whose data extend past its domain (7 of 18), refitted with the
  domain extended to the data; the dropped points classified as flat plateau or a feature
  (five tests on the residuals, then by eye).
- **Found:** five flat (time moves -1.1% to +1.2%), two features (WHT01_bigol1's dropped
  points sit 0.5 mol% above the core: an older inner core; WHT01_Ol_193 marginal).
  WHT06_ol43 extended: same time, but 1σ narrows by a third and the near-step warning
  strengthens (-7.7 -> -18.3), the overconfidence of section 3 of the findings.
- **Decided:** nothing changed in the paper tables; recommendations await Henry.
- **Details:** `docs/plans/2026-10-16-dropped-plateau/`.

## 2026-10-16: priors from the crystal population (explored)
- **Found:** see `docs/plans/2026-10-16-population-priors/README.md`. Also a fault in the
  starting-profile fit: its first noise model is fitted at the prior's centre and refitted
  once, so an informative prior centred away from the data inflates the correlated noise
  4-18x and leaves the time unconstrained (confirmed by holding the noise model; nested
  sampling rules out a missed mode).
- **Decided:** fix the noise start (iterate the noise refit and the search until stable)
  before offering informative priors.

## 2026-10-16 to 10-17: the joint fit of one crystal, and its sampler
- **Built:** `forster.fit_crystal` (Python): profiles of one crystal share the time, the
  conditions and the segment Fo by label; each keeps its steps, orientation and noise
  model; per-profile Fo drift offsets (one profile the reference) and optional edge
  offsets (Henry's answers in `docs/plans/2026-10-16-joint-fits.md`). The Laplace
  proposal is refined by a second Hessian pass, now also in single-profile fits (template
  range moved < 0.1%).
- **Calibration** (`docs/plans/2026-10-16-joint-calibration/`): on the paper's two
  two-profile crystals against nested references, the ESS did not predict the error and
  seeds flipped the method; nested sampling at the default nlive was no more accurate.
  Henry set the tolerance at about 5%, next to the ~0.1 log uncertainty of the
  diffusivity laws.
- **Decided:** joint fits use a sample-count step (1024, else 4096, nested only below ESS
  20): all 80 decisions within 5%, no seed flips, 12 s per fit. The starting-profile fit's
  model time is capped at 10x the fitted time (it had used the table's 200 days).
- **Not covered:** crystals where the Laplace approximation is plainly wrong (two modes,
  a value at a prior's edge); the round robin has no multi-transect crystals.

## Open
- Make the near-step model part of the fit? (whitening study, follow-ups 1-6).
- Henry to check the fitted starting profiles of WHT06_ol43 (step 1 moved to 6.7 µm, with
  no data between the rim and 7.5 µm) and Xenocryst_ol6 (a rim zone?) before the −40% is
  relied on (findings, Figure 1).
- A third noise refit in the starting-profile fit (untested); check that the grid method,
  which refits once, behaves like the sampler.
- Should the model extend past the last measured point to the domain's end, instead of
  holding the core Fo at the last point? (A behaviour change; the warning covers it now.)
- A mechanisms guide with full derivations and citations, separate from the summary.
- Measured points beyond the model domain's end are dropped without a warning; five of
  the six long-plateau paper runs stop their domain short of flat data (WHT06_ol43: data to
  132 µm, domain to 80 µm). Consider a warning, and whether the paper's tables should be
  extended.
- Warn from where the data flatten instead of from the diffusion length (needs a robust
  plateau finder for noisy data).
- Henry, 2026-10-16: the two published runs the short-plateau warning fires on
  (WHT06_ol_overgrowth2, GCB1A R31_MP_Ol5_prof_1) reach the centre of small crystals, with
  their core Fo set from other crystals of the sample. FORSTER models half profiles with
  a held core; Henry wants whole crystals (two rims set, core free), which needs naming
  that works for both geometries; the warning now says so. Also to do: priors for core, rim and plateau Fo from
  the crystal population (histograms, joint fits), and bring back the plateau points
  dropped from the paper runs (dropped to save model time or because of earlier stages).
  All three are in TODO_local.md.
- Full traverses with both rims (an idea in TODO_local.md) need their own rule for the
  core-end warning, which assumes one rim and a held core Fo at the last point.
