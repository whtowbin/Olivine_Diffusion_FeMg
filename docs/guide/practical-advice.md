# Practical advice for collecting profiles and fitting them

A running record of advice for FORSTER users, each item with the evidence behind it.
It is the seed of the user guide for the release: add to it whenever a test or study
shows something a user should do differently. Keep each item short and practical; the
evidence column points to where the numbers are.

Terms used below:
- **Diffusion length** l = √(D t): roughly how far a step has spread. On a fitted profile
  it is about half the width of the diffused step. FORSTER reports the fitted time, so
  l at your conditions follows from it.
- **Starting profile** (initial condition): the profile before diffusion, set by the
  step positions and the Fo of the edge, of each plateau and of the core.
- **1σ and 2σ ranges**: the ranges of time that should contain the true time 68% and
  95% of the time. Coverage is how often they actually do on synthetic profiles of
  known age.
- **Near-step warning**: FORSTER's warning that the misfit between model and data is
  concentrated near a step (`near_step_daic` below −6 in `summary.csv`).

## Collecting the data

| Advice | Why | Evidence |
|---|---|---|
| Measure until the profile has been flat for at least one more diffusion length past where it flattens; more is better. | FORSTER ends the model at the last measured point and holds the core Fo there. On real profiles the plateau starts about 2 diffusion lengths past the step. Data that stop where it starts moved the time by −15% to +22%; data that stop on the slope by −63% or more; with 1 diffusion length of plateau or more, every profile tested stayed within 4%. FORSTER warns when the data end within 3 diffusion lengths of the step (`core_end_diffusion_lengths`). | [Findings, section 3](../references/noise-model-findings.md) |
| If the profile ends at the centre of a small crystal, treat the short-plateau warning with care. | There the data cannot go further, and the problem is FORSTER's fixed core Fo at the end of the model: at a crystal's centre the core is free to change (both rims set the boundaries, and by symmetry no Fo crosses the centre), so the centre's Fo falls once diffusion arrives. Modelling whole crystals, rim to rim with a free core, is planned; until then, expect the time to be biased if diffusion has reached the centre. Setting the core Fo from other crystals of the same sample is reasonable. | [Findings, section 3](../references/noise-model-findings.md) |
| Do not worry about plateaus being longer on some profiles than others. | Once the data reach the plateau, its length does not move the time (within 4% on six real profiles with 1 to 21 diffusion lengths of plateau; under 1% on synthetic ones). | [Findings, section 3](../references/noise-model-findings.md) |
| Never trim plateau points to tighten a range or remove a warning. | Trimming does not move the time. It only hides misfit near the step: on WHT06_ol43 the near-step ΔAIC goes from −18.3 with the full plateau to −3.0 with 2 diffusion lengths, while the range stays too narrow. | [Findings, section 3](../references/noise-model-findings.md) |
| Set the model domain to include the whole measured plateau. | Points beyond the domain's end are left out without a warning. Five of six long-plateau paper runs stopped short of flat data; a longer plateau makes the near-step warning more reliable. | [Findings, section 3](../references/noise-model-findings.md) |
| Put at least about 13 analyses within two diffusion lengths of each step. | The near-step check depends on the points there: with 7-12 it catches only 10-60% of near-step misfit, with 13 or more 87-97% (`near_step_points` in summary.csv). The default range is too narrow either way when such misfit is present, so the check is what tells you to quote 2σ. | [Near-step follow-ups](../plans/2026-10-15-whitening-validity/followups/README.md) |
| Aim for 25 or more points across the diffused zone and its plateaus. | On 15-17-point profiles the near-step warning catches only 14-37% of strong near-step misfit (94% on 30 points), and a correlated misfit can mimic the profile itself. | [Whitening study](../plans/2026-10-15-whitening-validity/README.md) |
| Record your analytical precision. | Compare it with the fitted white noise (`white_sigma_fo`): much larger means unmodelled scatter or outliers. | [Noise guide](../references/noise-models-guide.md), section 2 |
| Note analyses that may be bad (cracks, inclusions, poor totals, beam overlap with the rim or matrix). | Remove a point only for a known fault, then refit. Outliers inflate the fitted noise, so they are hard to spot from the diagnostics. Rim analyses that are noisier lower coverage (63%). | [Noise guide](../references/noise-models-guide.md), sections 3.6-3.7 |

## Setting up and running fits

| Advice | Why | Evidence |
|---|---|---|
| Give the starting profile uncertainties (`steps_sd_um`, `edge_fo_sd`, `core_fo_sd`, plateau Fo) and let FORSTER fit it (`scan_method` auto), especially when the near-step warning fires. | On the four profiles with near-step misfit tested, fitting the starting profile cleared the warning (one only just) and moved the time by −40% (WHT06_ol43), +6%, −4% and −52% (the last not reliable, see the next row). The starting profile matters more than the noise model. | [Findings, starting profile](../references/noise-model-findings.md) |
| Look at the fitted starting profile before quoting the time. | A fitted value at the edge of its prior (FORSTER warns) means the data want something outside what you allowed. On Xenocryst_ol6 the first step ran to its prior edge, asking for a rim zone a sharp step cannot represent, and the time halved: rethink the starting profile rather than widen the prior. Steps placed where there are no data (between the rim and the first point) are constrained only by the shape of the nearby points. | [Findings, starting profile](../references/noise-model-findings.md) |
| If the core Fo is uncertain, fit it, unless the plateau is visibly sloped. | A core Fo 0.1-0.2 mol% off biases the time by 5-21% at any plateau length, and nothing warns; fitting it removes the bias. With a sloped plateau a fitted core Fo matches the average level, not the level next to the step (+9% bias). | [Findings, plateau length](../references/noise-model-findings.md) |
| Quote the 2σ range (`low2_days`-`high2_days`) when the near-step warning fires and fitting the starting profile does not clear it (`scan_near_step_daic` below −6). | The default 1σ range then covers 45-61%, and as little as 40% on long-plateau profiles; 2σ covered the near-step model's best fit on all 7 warned real profiles. | [Noise guide](../references/noise-models-guide.md), section 3.5 |
| Quote the 2σ range when the fitted misfit length (`correlation_length_um`) is about the diffusion length. | Misfit of that length trades off with time; the 1σ range covers about 57% and the diagnostics look clean. | [Noise guide](../references/noise-models-guide.md), section 3.3 |
| When the misfit length is several diffusion lengths, or FORSTER warns that the misfit spans the profile, check the starting profile before anything else. | A plateau or core level off the model's, not noise. | [Noise guide](../references/noise-models-guide.md), section 3.4 |
| Do not use white-noise-only ranges (the legacy method) for new work. | With real correlated misfit they cover 18-40% (target 68%). | [Findings, starting profile](../references/noise-model-findings.md) |
| Let FORSTER choose the grid; do not raise `FORSTER_MAX_CURVES_GB`. | The automatic grid resolves the fit to the stated accuracy; the memory limit stops a grid mistake from freezing the computer. | `docs/plans/2026-10-13-grid-choice/` |
