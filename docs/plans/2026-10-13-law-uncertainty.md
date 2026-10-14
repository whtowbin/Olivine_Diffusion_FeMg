# Diffusivity-law uncertainty in fitted times

**Goal:** a fit reports its time range three ways: from the profile alone, with the
diffusivity law's calibration uncertainty, and with the spread between laboratories
as well (Henry's decision, `2026-10-13-femg-refit.md`: the first for comparing
crystals that share a law, the last for absolute times).

**Approach.** For an Fo fit the profile depends on D t only, when D's level is what
is uncertain (`test_time_scales_inversely_with_the_diffusivity`), so an error
delta in log10 D shifts log10 t by -delta exactly, with no model runs. The law
carries a covariance over its parameters; at the run's conditions, orientation and
composition its gradient g gives sigma_law = sqrt(g' C g) in log10 D. The fit's own
range is combined with it in log10 time, separately for each side. The composition
term changes D's shape as well as its level. Decision (measured during task 3):
evaluate it a third of the way from the edge Fo to the core Fo. On fits over Fo
0.80-0.90 (normal and reverse zoning) and 0.85-0.88 the time shift for dn = 0.3 was
linear and centred there to 0.0011 log; the midpoint (first plan) missed by up to
0.005. It hardly matters: the preset's sigma varies by 0.0005 log over X = 0.80-0.90.

**Out of scope (follow-ups):** growth stages (stage times are fixed in seconds, so
time no longer scales as 1/D; no law range is given), element fits (time depends on
two laws), the initial-condition scan (combine its range with the same function),
the Mutch et al. (2019) published covariances (not transcribed; the TaMED matrix as
printed is not positive definite), and a Monte Carlo over the law (not needed for
Fo fits, see above).

## Tasks

1. **`forster.diffusivity`**
   - `LAW_PARAMETERS = ("log10_D0_001", "activation_energy_J_mol", "fO2_exponent",
     "fo_coefficient", "activation_volume_m3_mol", "pressure_coefficient_per_Pa",
     "log10_100_vs_001", "log10_010_vs_001")`. The two anisotropy terms are offsets
     from [001], so log10_D0_001 moves every axis.
   - `@dataclass(frozen=True) class LawUncertainty: parameters: tuple[str, ...];
     covariance: tuple[tuple[float, ...], ...]; between_study_log10: float = 0.0;
     source: str = ""`, checked in `__post_init__` (names in LAW_PARAMETERS, square,
     symmetric, positive semi-definite, tau >= 0). Tuples keep the dataclass
     hashable and comparable.
   - `ArrheniusDiffusivity.uncertainty: LawUncertainty | None = None`;
     `ArrheniusDiffusivity.log10_sigma(conditions, orientation, x_fo) -> float | None`.
     Not in the diffusivities table (its columns are unchanged).
   - Preset `femg_refit2026`: the DC07 form with log D0, E, fO2 exponent,
     composition coefficient and both anisotropy factors fitted, V fixed at 7
     cm3/mol, random study offset (`analyses/femg_refit`, method "random study"),
     with its covariance and tau. Values written by
     `analyses/femg_refit/export_preset.py`, which also checks the preset's sigma
     against `fit_models.predict` at several conditions.
2. **`forster.uncertainty.widen_log10(time, low, high, sigma_log10) -> (low, high)`**:
   each side's distance in log10 time combined in quadrature with sigma; a low end
   at zero stays at zero.
3. **`forster.model`**: `FitResult.law_sigma_log10: float | None = None`,
   `FitResult.between_study_log10: float | None = None`; `fit_profile` fills them
   (Fo fits with a law covariance and no stages, a third of the way from edge to
   core Fo);
   `FitResult.time_range_days(include="fit" | "law" | "law_and_studies")`, which
   raises ValueError when the law has no covariance.
4. **`forster.io.summary_row`, `forster.template.SUMMARY_COLUMNS`**: columns
   `law_sigma_log10`, `between_study_log10`, `law_low_days`, `law_high_days`,
   `all_low_days`, `all_high_days` (NaN when not available).
5. **Docs**: README (preset list, the two ranges and when to use each), CHANGELOG,
   `docs/references/diffusivity-uncertainty-sources.md` (options 1-2 status).

## Tests (`tests/test_law_uncertainty.py`)

- `log10_sigma` against a hand-computed g' C g for a two-parameter covariance
  (log10_D0_001, E), along [001] and at an angle (anisotropy weights).
- The preset's sigma at the KS20 conditions, Fo85, [001] equals the refit's
  `se` in `analyses/femg_refit/results/predictions.csv` (0.122) and tau 0.254.
- `widen_log10`: sigma 0 returns the fit range; known numbers; low end at zero.
- A real fit with `femg_refit2026`: `time_range_days("law")` contains the fit range
  and matches widen_log10 of it; "law_and_studies" is wider; default preset gives
  None and `time_range_days("law")` raises.
- The composition approximation, on real fits over Fo 0.80-0.90 both ways round:
  shifting the fo_coefficient by +-0.3 moves the fitted time as -dn (fo_ref - X)
  predicts with X a third of the way in, within 0.002 log.
- Invalid `LawUncertainty` (unknown name, asymmetric, negative eigenvalue,
  negative tau) raises.
- Default results unchanged: the existing suite (round robin differential test).

Verify: `uv run pytest -q`, with and without `FORSTER_NO_NUMBA=1`; `ruff check`,
`ruff format --check`, `ty check`, `complexipy src --max-complexity-allowed 25`.

## Status

- [x] 1-5 (2026-10-13). Verified: full suite with and without Numba, ruff, ty, complexipy.

## Follow-up (2026-10-14)

The review found the law range also attached to fits with a rim value that changes
in time, where the D t argument fails as it does for growth stages (measured: the
fitted time moved by 0.57-0.66 of -delta); it is now withheld there too, and a
time-zero fit no longer makes `summary_row` raise. Limits of the method, measured:
`docs/references/law-uncertainty-limits.md`.
