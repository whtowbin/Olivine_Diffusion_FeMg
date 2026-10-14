# The diffusivity-law range of a fitted time: what it assumes, and where it fails

`forster` can widen a fitted time's range by the calibration uncertainty of the
diffusivity law (`FitResult.time_range_days("law" | "law_and_studies")`, the
`law_*` and `all_*` columns of summary.csv). This note says how that range is
computed, the one physical assumption it rests on (the profile depends on D t),
the fits for which that assumption holds, the fits for which it does not and what
`forster` does then, and the smaller approximations around it. Every number below
was measured; the scripts are in `docs/plans/2026-10-14-review-fixes/`
(`law_limits.py`, which writes `dt_scaling.csv` and `linear_vs_mc.csv`). Plans:
`docs/plans/2026-10-13-law-uncertainty.md` (the feature),
`docs/plans/2026-10-14-review-fixes.md` (the fixes that followed this note).

## How the range is computed (no Monte Carlo)

The method is linear error propagation, not a Monte Carlo, and it needs no model
runs beyond the fit itself:

1. The law carries a covariance C over its parameters (`LawUncertainty`; at
   present only `femg_refit2026` has one: log D0, E, the fO2 exponent, the
   composition coefficient and the two anisotropy offsets, from the refit in
   `analyses/femg_refit`).
2. At the run's temperature, pressure, fO2, orientation and a composition X, the
   gradient g of log10 D with respect to those parameters is known in closed form,
   so the law's 1-sigma in log10 D is sigma = sqrt(g' C g)
   (`ArrheniusDiffusivity.log10_sigma`). At the round robin KS20 conditions along
   [001] it is 0.114 log (0.122 for the preset before 2026-10-14).
3. **The D t argument.** The diffusion equation with D(X) = D_level f(X) and fixed
   boundary values is unchanged by trading D_level for time: the profile after time t
   with D_level is the profile after time 2t with D_level / 2. So an error delta in
   log10 D_level is an error -delta in the fitted log10 t, exactly.
4. Each side of the fit's own range is combined with sigma in quadrature in log10
   time (`forster.uncertainty.widen_log10`), so an asymmetric range stays
   asymmetric. "law_and_studies" also adds the between-study spread tau (0.235 log for
   `femg_refit2026`) in quadrature: hypot(0.114, 0.235) = 0.261 log at KS20.

### Checked: linear propagation against a Monte Carlo over the covariance

log10 D is linear in every parameter except the two anisotropy offsets, which
enter through log10 of the cos^2-weighted sum of the three axes' D. For a profile
off the crystal axes that weighting is the only nonlinearity. 4000 draws from the
preset's covariance, log10 D at Fo 0.85 (`linear_vs_mc.csv`):

| conditions | orientation | linear sigma | Monte Carlo sd | MC 16-84% half-width |
|---|---|---|---|---|
| 1200 C, 0.042 GPa, KS20 fO2 | [001] | 0.1141 | 0.1138 | 0.1137 |
| 1200 C, 0.042 GPa, KS20 fO2 | [100] | 0.1213 | 0.1209 | 0.1215 |
| 1200 C, 0.042 GPa, KS20 fO2 | oblique (50, 70, 46.5) | 0.1134 | 0.1131 | 0.1139 |
| 1200 C, 0.042 GPa, KS20 fO2 | equal angles (54.7) | 0.1134 | 0.1130 | 0.1141 |
| 900 C, 0.1 GPa, 1e-9 Pa | [001] | 0.1276 | 0.1283 | 0.1280 |
| 900 C, 0.1 GPa, 1e-9 Pa | oblique (50, 70, 46.5) | 0.1271 | 0.1280 | 0.1286 |
| 1300 C, 1 atm, 1e-2 Pa | [100] | 0.1252 | 0.1246 | 0.1246 |
| 1300 C, 1 atm, 1e-2 Pa | equal angles (54.7) | 0.1178 | 0.1174 | 0.1171 |

The two agree within the Monte Carlo's own sampling error (about 1.1% of sigma at
4000 draws; every row is within 1.6%, and the [001] rows, where the propagation is
exactly linear, differ by -0.3% to +0.5%, so the differences are the draws, not the
method; table regenerated for the preset of 2026-10-14). A Monte Carlo over the law adds
nothing for an Fo fit for which the D t argument holds.

## Where the D t argument holds (measured)

A synthetic profile was made with `femg_refit2026` at KS20 conditions (6 days,
noise 5e-4 Fo) and fitted twice: with the law, and with D raised by delta in log10.
Where the argument holds, the fitted log10 t moves by -delta
(`dt_scaling.csv`):

| case | shift of log10 t, delta = 0.122 | delta = 0.301 | fraction of -delta |
|---|---|---|---|
| fixed rim (control) | -0.1221 | -0.3010 | 1.00 |
| fixed `rim_fo`, not the edge Fo | -0.1216 | -0.3010 | 1.00 |
| rim changing (`RimPath` 0.86 to 0.82) over days 0-0.5 | -0.1158 | -0.2825 | 0.94-0.95 |
| rim changing (`RimPath` 0.86 to 0.82) over days 0-3 | -0.0803 | -0.1714 | 0.57-0.66 |
| growth stage (Fo 0.82 zone) at day 3 | -0.0723 | -0.1487 | 0.49-0.59 |

The control and the fixed `rim_fo` miss by at most 0.0004 log, within the time-step
resolution of the fit (dt 0.01 d on 6 d). So the argument is exact, to the grid's resolution, for every fit whose
boundary values are fixed in time: any initial condition built from steps and
plateaus, any fixed rim value, normal or reverse zoning, any orientation and any
conditions. The grid (dx) and the scheme do not enter: the profile on a fixed grid
depends on D t just as the continuous one does.

## Where it fails, and what forster does

**1. A rim value that changes in time** (`fit_profile(..., rim_fo=RimPath(...))` or
any function of time). The rim history is fixed in seconds, so a larger D does not
replay the same history faster; the fitted time moves by only part of -delta (0.57
to 0.66 of it when the rim changes over the first half of the run). The range from
the shortcut would be too wide by up to 1.75x in the law term. `forster` reports
**no law range** for these fits (`law_sigma_log10` is None, the summary columns are
empty, `time_range_days("law")` raises). A rim that settles early in the run comes
close to the D t result (0.94-0.95 of it when the change ends at 1/12 of the time), but
`forster` does not try to judge "early enough".

**2. Growth stages** (`fit_profile(..., stages=...)`). The same reason: stage times
are fixed in seconds. The measured fraction (0.49-0.59) is between the D t result
and the result if only the time after the stage scaled (0.465 for delta = 0.122
here), and it depends on delta, so there is no simple correction. **No law range.**

**3. Element fits** (Mn, Ni, Ca; `fit_element`, `fit_joint`). The time depends on
two laws, the element's and Fe-Mg's (the element diffusivity depends on the Fo
profile through the cross term), each with its own uncertainty and possibly
correlated with each other. A law range would need both covariances and the
gradient of the fitted time with respect to each law's level, which the D t
argument does not give. **Not computed.** The element presets carry no covariance.

**4. The initial-condition scan** (`scan_initial_conditions`, the `scan_*` columns).
Each candidate is a fixed-boundary Fo fit, so the D t argument does hold for each;
the scan's range could be widened with `widen_log10` the same way. **Not wired up
yet**: the `scan_low_days`/`scan_high_days` columns are without the law term. Until
it is, combine by hand: `widen_log10(scan.time_days, low, high, sigma)`.

**5. A best fit at time zero** (a profile already at its initial condition). There
is no log distance to combine in quadrature; `widen_log10` returns the low end 0
and scales the high end by 10^sigma (the law's error moves the bound as it moves
any time). This is the D t argument applied to the bound, not a combined 68%
range, and is cruder than the usual case.

## Approximations inside the cases where it holds

- **The composition term.** D(X) = D_ref 10^(n (X_ref - X)): an error in n changes
  D's *shape* along the profile, not only its level, so the D t argument is not
  exact for it. `forster` evaluates g at X a third of the way from the edge Fo to the
  core Fo. On fits over Fo 0.80-0.90 (both zoning directions) and 0.85-0.88, the
  time shift for dn = 0.3 matched -dn (X_ref - X) at that X to 0.0011 log; the
  midpoint missed by up to 0.005 (`test_the_composition_term_moves_time_as_evaluated_a_third_of_the_way_in`).
  For `femg_refit2026` it barely matters: sigma changes by 0.0005 log over X =
  0.80-0.90, because the composition coefficient's own variance is small (sd 0.140).
  A law with a much less certain composition term, or a profile spanning a much
  wider Fo range, would need this checked again.
- **The plateau Fo** of a two-step initial condition is not used to place X.
- **Independence.** The fit's range comes from the profile's noise, the law's from
  the calibration experiments; they are independent, so quadrature is right. The
  between-study term is treated as independent of the law's covariance (it is the
  REML random-effect variance of the same refit; C is the covariance at the
  REML estimate of tau).
- **Gaussian in log10 D.** Both the law's covariance and tau are 1-sigma Gaussian
  quantities in log10; the fit's range is a 68% likelihood interval. The widened
  range is a 68% range under those assumptions. tau is poorly determined from five
  studies (95% range 0.11-0.64 log), and the "law_and_studies" range moves with it:
  hypot(0.114, tau) is 0.157 to 0.647 log over that interval at KS20.

## What the covariance does not contain

- **Model-form error.** C is the uncertainty of the parameters *within* the DC07
  form with V fixed at 7 cm3/mol. Other forms fitted to the same data predict D at
  KS20 differently: with a random study offset the five forms span +0.20 to +0.30
  log against the default law, a spread comparable to sigma; without it (OLS, WLS,
  ODR, bootstrap) they span -0.01 to +0.85 (`analyses/femg_refit/README.md`, bowtie table).
  Choosing a form is a systematic the law range does not include.
- **Extrapolation beyond the calibration.** g' C g grows away from the data's centre
  as the covariance says it should (the bowtie widening in
  `analyses/femg_refit/README.md`), but only within the assumed form; outside the
  calibrated T, P, fO2 and Fo ranges, and in other diffusion regimes (the PED
  mechanism at low fO2, melt contact, high-temperature anisotropy;
  `diffusivity-uncertainty-sources.md`), the law itself may not hold.
- **Inputs to the law.** Errors in T, P, fO2 and orientation are not in C. They
  move D through the same g, but `forster` does not propagate them (an error of
  10 C at 1200 C is 0.05 log in D for E = 209 kJ/mol, before any change in a
  buffered fO2).

## What a law range for cases 1-3 would need

A refit per draw: sample the law's parameters from C (and an offset from tau), fit
the profile with each sampled law, and take the 16th and 84th percentiles of the
fitted times, combined with the fit's own range. The percentile's sampling error is
about 0.11 sigma at 200 draws (sqrt(0.16 x 0.84 / N) / phi(1)), so 200-400 full
fits per profile, against none for the D t shortcut. For growth stages one could
instead fit the time's derivative with respect to log10 D_level from two extra fits
(the fraction measured above) and propagate sigma through it, if the response is
near linear over +-sigma (here the fraction moved from 0.59 to 0.49 between delta =
0.122 and 0.301, so only roughly).
