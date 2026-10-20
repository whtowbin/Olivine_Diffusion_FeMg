# Facts and sources

Sources: `../growth-stages/docs/facts.md` (the model facts, unchanged), FORSTER_FACTS.md,
src/forster/boundary.py (Stage: `start_s`, `rim_um`, `rim_value`, `zone_value`; the rim
value may differ from the new zone's value), src/forster/model.py (`fit_profile` with
`stages`, `near_step_at`), src/forster/uncertainty.py (`near_step_check`, warning below
AIC -6). Zone geometry: `../joint-fit/figure/data/profiles.json` (profile 1). Curves:
figure/data/curves.json, made by figure/data/compute_curves.py with forster.fit_profile.

## Zone geometry, read from joint-fit profile 1 (not typed in)
- Steps 15.158 and 36.040 µm from the rim (p1 `steps`), profile 76 µm long (p1 `x_end`),
  points every 2 µm (meta `spacing`); levels edge 0, rim zone 0.9, mantle 2.0, core 3.0
  (meta `edge`, `segments`).
- Fo = 0.80 + 0.10 × level / 3: edge 0.800, rim zone 0.830, mantle 0.8667, core 0.900.
- Not reused: the joint-fit curve, its relative diffusivity and time, its edge offset
  (3 µm between the true rim and the first spot; here the points start at the rim).

## Model history (choices for the illustration)
- 1200 °C, 42 MPa, fO2 1.38e-3 Pa, along [001], Dohmen & Chakraborty (2007), as
  growth-stages. D at Fo 0.80-0.90: 4.4e-16 to 2.2e-16 m²/s.
- Stage 0, t = 0 to 0.5 d: core crystal from 36.04 µm inward (to 120 µm, held at Fo 0.90
  there), its edge held at the mantle Fo 0.867.
- Stage 1 at 0.5 d: rim moves to 15.16 µm; new zone (mantle) at Fo 0.867; rim held at
  0.867.
- Stage 2 at 1.25 d: rim moves to 0 µm; new zone (rim zone) at Fo 0.830; edge held at
  Fo 0.80 (`rim_value` 0.80, `zone_value` 0.830).
- Observed at 1.4 d. Why these times: the outer steps then have sqrt(D t) of about 2 µm
  (sqrt(4 D t) of about 4 µm, the width joint-fit profile 1 was drawn with), which keeps a
  short rim-zone plateau visible; the inner step has 5.5 µm, 2.7 times wider, which keeps
  a short mantle plateau. Longer outer ages (0.3 d tried) erase the rim-zone plateau.
- Diffusion lengths sqrt(D t) at 1.4 d (D at the mean Fo of each step's two sides, t the
  time since the step formed): inner step (mantle / core, 1.4 d) 5.47 µm; outer step (rim
  zone / mantle, 0.15 d) 2.02 µm; edge (edge / rim zone, 0.15 d) 2.27 µm.

## Computed (this run, figure/data/curves.json)
- Snapshots: FORSTER's model (implicit scheme, dx 0.49 µm, dt 114 s), rows just before
  and just after each stage (the "after" row has one time step of diffusion, 0.2 µm).
- Synthetic measured profile: an independent explicit flux-form solver (not FORSTER;
  dx 0.1 µm, each rim a node held at its value), 39 points at 0, 2, ... 76 µm, noise
  0.001 Fo, seed 5. It agrees with FORSTER's snapshot at 1.4 d to 9e-5 Fo.
- Staged fit: 1.392 d since stage 0 (1 sigma 1.383-1.402; true 1.4), 0.142 d since the
  last stage (true 0.15); chi-squared minimum 39.0 for 39 points; rms residual 0.00087.
- Comparison fit, every zone present from time zero (segments 0.80 | 0.830 | 0.867 |
  0.900 with steps at 0, 15.16, 36.04 µm; the zero-width first segment carries the edge
  value): 0.207 d (1 sigma 0.155-0.291); chi-squared minimum 38.1, as low as the staged
  fit's.
- FORSTER's near-step check: comparison fit AIC -3.2 (warns below -6: not flagged);
  staged fit (same check on its residuals, steps of the comparison, length at the time
  since the last stage, as growth-stages does) +1.9. Over 10 noise draws (seeds 0-9) the
  comparison is flagged in 2, the staged fit in 0; the comparison's rms residual is
  0.0039-0.0045 Fo, 3.9-4.5 times the noise, in every draw. The staged fit's time since
  the last stage ranges 0.138-0.173 d over the draws (true 0.15).
- Where the comparison misses: residuals up to 0.014 Fo over 38-56 µm (the inner
  gradient, which it makes too sharp), rms 0.0067 over 30-60 µm; 0.0015-0.0018 elsewhere.
  The misfit is broad, wider than the check's near-step zone (2 sqrt(D t), about 5 µm, at
  the comparison's 0.21 d), so the check's correlated-noise model absorbs it.
- No FORSTER warnings on either fit.

## Unknown (kept generic)
- Crystal habit and section shape: generic elongate eight-sided section, no faces.
- Real numbers: all values are illustrative.

## On-screen strings (quoted for the final gate)
Headings: "Stage 0: core crystal", "Stage 1: mantle grows", "Stage 2: rim zone grows",
"Measured profile", "Misfit against time". Time stamps (stage times and observation):
"t = 0 to 0.5 d", "0.5 to 1.25 d", "at 1.25 d", "at 1.4 d".
Crystal: "final rim", "profile", "new zone", "old rim", "old rims" (stage rims at 36 and
15 µm). Colour key: "Fo", "0.80", "0.85", "0.90" (edge and core Fo, and the middle).
Profile axes: "distance from rim (µm)", "Fo", ticks "0", "20", "40", "60", "80".
Curves: "t = 0" (starting profile), "0.5 d" (model at the end of stage 0, and dashed in b
just after the mantle grew), "1.25 d" (model at the end of stage 1), "not yet grown"
(inactive nodes before a zone grows), "new zone" (zone filled at its Fo), "sharp step"
(rim zone / mantle step just after stage 2), "edge held at 0.80" (stage 2 rim_value),
"diffused 1.25 d" (inner gradient at stage 2), "measured" (synthetic points), "staged
model" (staged fit), "outer steps 0.15 d" and "inner step 1.4 d" (time since each step
formed: 1.4 - 1.25 d and 1.4 d).
Time panel: "stage 0: core crystal", "stage 1: mantle grows", "stage 2: rim zone grows",
"rim zone not grown: no staged fit" (only times after the last stage are fitted), "time
since stage 0 (days)", ticks "0", "0.25", "0.5", "0.75", "1", "1.25", "1.5", "1.75", "2",
"misfit", "high", "low", "best fit" (staged minimum), "staged model", "0.21 d no
near-step warning residuals 4.3 × noise" (comparison minimum 0.207 d; near-step AIC -3.2,
above the -6 threshold; rms residual 0.0043 against noise 0.001), "all zones from t = 0"
(the comparison), "since first stage: 1.39 d", "since last stage: 0.14 d" (time_days and
time_since_last_stage_days of the staged fit), markers "a", "b", "c", "d" (stage starts
0, 0.5, 1.25 d and the best fit).
