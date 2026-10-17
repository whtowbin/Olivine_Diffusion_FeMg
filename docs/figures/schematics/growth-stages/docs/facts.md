# Facts and sources

Sources: FORSTER_FACTS.md (research note), docs/plans/2026-10-13-growth-stages.md,
src/forster/boundary.py, tests/test_growth.py (repo, read only). Curves:
figure/data/curves.json, made by figure/data/compute_curves.py with forster.fit_profile.

## Model facts used
- Stage 0 = the starting profile (crystal [40, 199] um of Fo 0.90; rim held at the edge
  value 0.86). Source: tests/test_growth.py (EARLY), boundary.py docstring.
- Stage 1 at 6 d: rim moves outward from 40 um to 0 um; new zone [0, 40) of Fo 0.82;
  rim held at 0.82 from then on (rim_value defaults to zone_value). Source: test GROWTH.
- New zone diffuses together with everything grown before. Source: plan, Goal.
- Distances measured from the final rim (rim-relative). Source: plan, Model.
- Only times after the last stage can fit (the fit considers rows at or after the last
  stage). Source: plan, Fitting.
- Reported: time since first stage (time_days) and since last stage
  (time_since_last_stage_days). Stage start times are fixed inputs. Source: plan, choices.
- Conditions of the illustrative case: 1200 C, 42 MPa, fO2 1.38e-3 Pa, along [001],
  Dohmen & Chakraborty 2007. Source: tests/test_growth.py. (Caption only.)

## Computed (this run, figure/data/curves.json)
- Synthetic measured profile: 66 points, independent flux-form solver at 10 d, noise
  0.001 Fo, seed 5 (as in the test).
- Staged fit: 10.07 d since stage 0 (1 sigma 9.86-10.29), 4.07 d since stage 1;
  chi-squared minimum 66.0.
- Single-stage fit (both zones present from time zero, edge 0.82, core 0.90, step at
  40 um): 6.74 d (1 sigma 6.11-7.49); chi-squared minimum 64.9. Plan reported 6.8 d
  (6.2-7.6), chi2 65.2 vs 66.0: same within grid differences.
- The staged misfit is undefined (no outer zone yet) before 6 d.

## Unknown (kept generic)
- Crystal habit and section shape: a generic elongate eight-sided section is drawn; no
  face indices, no real crystal.
- Real crystal numbers: all values are from the illustrative test case.
- How the stage times are known in practice: not in the sources beyond "fixed inputs";
  the plan says the history must come from petrography, other elements or zoning.

## On-screen strings (quoted for the final gate)
Headings: "Stage 0: starting crystal", "Stage 1: new rim zone", "Measured profile",
"Misfit against time" (plan: stage 0 = starting profile; stage 1 = new zone at the rim).
Time stamps: "t = 0 to 6 d", "at 6 d", "at 10 d" (test: stage 1 at 6 d, observed 10 d).
Crystal: "final rim", "profile", "new zone", "old rim" (boundary.py: rim moves outward from
40 to 0 um). Colour key: "Fo", "0.82", "0.86", "0.90" (zone, early rim, core Fo).
Profile axes: "distance from rim (µm)", ticks "0", "50", "100", "150" (rim-relative
coordinates, plan). Curves: "t = 0" (starting profile), "6 d" and "diffused 6 d" (model at
6 d), "not yet grown" (inactive nodes before stage 1), "sharp step" (new zone filled at
zone_value, undiffused at 6 d), "measured" (synthetic points), "staged model" (staged
fit), "outer step 4 d" and "inner gradient 10 d" (time since stage 1 = 4.07 d; since
stage 0 = 10.07 d).
Time panel: "stage 0: starting crystal", "stage 1: new rim zone", "time since stage 0
(days)", ticks "2" "4" "6" "8" "10" "12" "14" "16", "misfit", "high", "low", "outer zone
not grown: no staged fit" (plan, Fitting), "best fit" (staged minimum, 10.07 d), "6.7 d" (single-stage minimum, 6.74 d), "near-step warning"
(on that fit: FORSTER's near-step check on that fit improves AIC by
22.0, below its warning threshold of -6, while the same check on the staged fit gives
+1.3: `near_step` in curves.json, from compute_curves.py), "both zones from t = 0" (single-stage model, SINGLE in
compute_curves.py), "since first stage: 10.1 d", "since last stage: 4.1 d" (time_days and
time_since_last_stage_days of the staged fit).
