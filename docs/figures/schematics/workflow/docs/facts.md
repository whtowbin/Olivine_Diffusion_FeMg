# Facts ledger (sources: FORSTER_FACTS.md research note; repo README.md, .claude/skills/forster-fitting/SKILL.md,
# docs/references/likelihood-method.md, src/forster/plotting.py plot_time_distributions)

## Geometry record (the only physical object: an olivine section with a traverse)
- Object: a polished olivine section with a straight traverse of microprobe spots from the crystal edge
  (rim) inward. Element order along the traverse: rim (distance 0, edge 1) → rim segment → step → core
  segment. Source: note ("Distances are measured from the profile's own edge"; segments a, b, ... from edge 1 inward).
- Crystal outline: UNKNOWN habit; drawn as a generic convex polygon, no faces named (note: "do not claim a
  specific habit or face indices").
- Crystal axes: [100], [010], [001]; the profile's direction relative to them sets D along it
  (D_profile = D[100]cos²α + D[010]cos²β + D[001]cos²γ). Which axis lies in the section plane: UNKNOWN,
  so the axes are drawn as a generic triad beside the crystal, not tied to its edges.
- Zoning direction (rim lower Fo than core, "normal" zoning): one of the two cases in the README examples
  ("0.82, 0.90": rim 0.82, core 0.90). Illustrative choice.

## On-screen strings and their sources
| String | Source |
|---|---|
| "Inputs" | note: inputs list |
| "olivine" | note |
| "microprobe spots" | note: "electron microprobe spot analyses" |
| "measured profile" | note: "a measured profile (Fo vs distance from the crystal edge)" |
| "crystal axes" | note: orientation from EBSD, angles to a, b, c |
| "[100]", "[010]", "[001]" | note: a [100], b [010], c [001] |
| "T", "P", "fO2" (shown with subscript) | note: temperature, pressure, oxygen fugacity |
| "Fo" | note: Fo = Mg/(Mg+Fe) |
| "distance from rim" | note: "Fo vs distance from the crystal edge"; edge 1 is the crystal rim |
| "starting profile" | note |
| "segment a", "segment b", "a", "b", "step" | note: segments labelled a, b, ... from edge 1 inward; step positions |
| "rim held" | note: "edge 1 is the crystal rim, held at the edge Fo" |
| "Forward model" | note: "runs the diffusion model forward in time" |
| "t₁", "t₂", "t₃", "time" | illustrative model times (figure/data/make_curves.py) |
| "√(D t)" (diffusion length) | README "the diffusion length l = sqrt(D t)" |
| "Misfit against time" , "misfit", "χ²" | note: "a misfit (chi-squared) curve versus time" |
| "best-fit time", "best fit" | note: "Best-fit time = minimum misfit" |
| "1σ range" | note: "1σ and 2σ ranges from the misfit curve"; README: where χ² rises by about 1 |
| "residuals", "white", "correlated", "noise model" | note: "white (point-to-point) part and a smooth correlated misfit part" |
| "Uncertain starting profile", "optional" | note: fit step 4 ("can carry a prior ... FORSTER then fits them too") |
| "step position", "segment Fo", "prior" | note: "step positions and segment Fo can carry a prior (sd or range)" |
| "fitted with the time" | forster-fitting skill: "optimises the uncertain steps and segment Fo with time marginalised" |
| "Monte Carlo" | note: step 5 |
| "law parameters" | README: "the equation's parameters from its covariance (mc_law, on by default)" |
| "D" , "log D", "log t" | note: D depends on T, P, fO2, Fo, direction |
| "faster D, shorter t" | note: "A profile fixes D × t, so the time scales as 1/D" |
| "t = (D t)_fit / D" | README: "Times rescale exactly for a fixed rim"; "An Fo profile depends on D t, so an error in log D is the same error in log time" |
| "Time distribution" | README / plotting.py: fit alone, with the initial condition, with conditions and law |
| "fit alone", "+ starting profile", "+ T, P, fO2, law" | plotting.py legend: "Fit alone", "With the initial condition", "With conditions and law" |
| "median", "68% range" | note: "median, 68% and 90% ranges" |
| "time (log scale)" | plotting.py: "Time (days, log scale)" |

## Computed (illustrative) elements: figure/data/make_curves.py → curves.json
- Profiles: constant-D erfc solution of one step on a half line with the rim held (method of images).
  Real FORSTER uses Fo-dependent, anisotropic D and finite differences; shapes here are illustrative.
- Synthetic data: model + white noise + smooth correlated misfit (squared-exponential), 13 spots, seed 7.
- Misfit: generalised χ² with that covariance; 1σ where it rises by 1 (FORSTER uses a small-sample
  threshold slightly above 1).
- Distributions: fit alone ∝ exp(−χ²/2); + starting profile: marginalised over a 9×9 grid of step
  position and core Fo with Gaussian priors (illustrative widths); + conditions: convolved in log t with an
  illustrative Gaussian in log D (sd 0.12). None of these widths come from a real crystal.

## Unknown (kept generic)
- Crystal habit and face indices; which axes lie in the section plane.
- Typical sizes of each uncertainty contribution (no numbers on the figure).
- Real spot spacing, profile length, Fo values, times.

## Strings added in the build (sources as above)
| String | Source |
|---|---|
| "orientation" | note: "the orientation of the profile in the crystal" |
| "Inputs", "Forward model in time", "Misfit against time", "Uncertain starting profile", "Monte Carlo", "Time distribution" | panel headings, named from the note's pipeline steps |
| "(optional; fitted with the time)" | note step 4; forster-fitting skill ("with time marginalised") |
| "segment Fo", "step position", "starting profile, measured Fo" | note: starting profile, segment Fo, step positions, measured profile |
| "noise model", "white", "correlated" | note: noise model with white and smooth correlated parts |
| "best-fit time", "1σ range", "misfit χ2", "time", "t1", "t2", "t3" | note: misfit (chi-squared) curve versus time, best fit, 1σ |
| "√(D t)" | README: diffusion length l = sqrt(D t) |
| "log D", "log t", "faster D, shorter t", "t = (D t)fit / D" | note: "A profile fixes D × t, so the time scales as 1/D"; README rescaling for a fixed rim |
| "fit alone", "+ starting profile", "+ T, P, fO2, law", "median", "68% range", "probability", "time (log scale)" | plotting.py plot_time_distributions; note step 5 |
| "law parameters" | README: mc_law draws the equation's parameters from its covariance |
