# Spec: the FORSTER workflow

**Structure:** a pipeline (Z-order, two rows of three panels). Reason: the content is one causal
sequence from inputs to a time distribution, so reading order = processing order (P8). The unifying idea
(shape fixes D·t) is drawn where it acts: the diffusion-length bracket √(D t) on the evolving profiles (b)
and the rescaling t = (D t)_fit / D in the Monte Carlo (e).

## Concepts considered (step 2)
| | Concept | 5-s takeaway | True to note | Signal state (P1) | Mapping (P2/P3) | Explain/impress |
|---|---|---|---|---|---|---|
| A | Six-panel Z pipeline: inputs → profiles in time → misfit vs time → uncertain start → Monte Carlo → nested time distributions | good: the eye follows one profile to one distribution | high: each panel is a step FORSTER runs | yes: the profile itself is shown at each step | colour link of t₁..t₃ between b and c; D→t mapping in e | explains |
| B | Two converging streams: profile shape → (D·t), conditions → D, meet at t = D·t / D | very fast | **low**: implies the fit measures D·t then divides, but the fit runs the model with D (must-not list) | partial | strong | explains |
| C | One hero panel of nested time distributions with a thin pipeline strip above | fast for the result, weak for the method | medium | weak | weak | impresses |
Kept: **A** (B is the takeaway drawn as a false mechanism; C hides the method the paper is about).

**Takeaway:** a profile's shape fixes D × t; FORSTER turns it into a time with an honest range.
**Size:** 183 mm × 112 mm
**Hero:** none drawn in 3D: the hero is the profile evolving in time (panel b), a computed flat plot; the
only physical object (an olivine section) is a small flat drawing in panel a. Route flat (no physical
object deserves depth; P25).

## Panels
| Panel | Job | Position |
|---|---|---|
| a | Inputs: crystal section with spot traverse and axes; T, P, fO₂; measured profile with the step-shaped starting profile and segments | top left, widest |
| b | Forward model: starting step relaxes, curves at t₁ < t₂ < t₃ over the data; √(D t) bracket | top middle (hero) |
| c | Misfit against time: χ² curve with dots at t₁..t₃, minimum = best-fit time, 1σ range; noise-model strip (residuals: white + correlated) | top right |
| d | Uncertain starting profile (optional): prior on step position and segment Fo, alternative starts, fitted with the time | bottom left |
| e | Monte Carlo: T, P, fO₂, law parameters → spread in log D → spread in log t (inverse mapping) | bottom middle |
| f | Time distribution: fit alone ⊂ + starting profile ⊂ + T, P, fO₂, law; median and 68% range | bottom right |

## Components
| Panel | Component | Label | Look | Source |
|---|---|---|---|---|
| a | olivine section | "olivine" | pale olive-green convex polygon, slightly darker rim band (zoning) | draw (L.polygon); generic, no habit claimed |
| a | traverse with spots | "microprobe spots" | row of small dark dots from edge inward | draw |
| a | axes triad | "[100]" "[010]" "[001]", "crystal axes" | three short grey arrows | draw (L.arrow) |
| a | conditions | "T", "P", "fO₂" | three small rounded chips, green | draw (L.box) |
| a | measured profile | "Fo", "distance from rim", "measured profile" | dots | compute (curves.json) |
| a | starting profile | "starting profile", "segment a", "segment b", "step" | orange step line | compute |
| b | model curves | "t₁" "t₂" "t₃", "time" arrow | blue ramp light → dark; best (t₂) thick | compute |
| b | diffusion length | "√(D t)" | small bracket across the front at t₂ | compute (position) |
| c | misfit curve | "misfit χ²", "time" | black line; coloured dots at t₁..t₃ | compute |
| c | best fit & 1σ | "best-fit time", "1σ range" | blue vertical tick, pale blue band | compute |
| c | noise strip | "residuals", "white", "correlated" | grey dots + smooth grey-blue line | compute (GP mean) |
| d | priors | "step position", "segment Fo", "prior" | orange double arrows, faint orange alternatives | compute |
| e | input spreads | "T", "P", "fO₂", "law parameters" | small green Gaussians | draw (shape only) |
| e | mapping | "log D", "log t", "faster D, shorter t", relation | two axes, draws as ticks joined by lines | compute (draws) |
| f | distributions | "fit alone", "+ starting profile", "+ T, P, fO₂, law", "median", "68% range", "time (log scale)" | blue line, orange line, green fill | compute |

## Relations
| From | To | Kind | Drawn as |
|---|---|---|---|
| a | b | flow | grey arrow between panels |
| b | c | flow (each curve → one misfit value) | grey arrow + colour link of t₁..t₃ |
| c | d | flow | grey elbow arrow through the row gutter |
| d | e | flow | grey arrow |
| e | f | flow | grey arrow |

## Conventions
- Colour per role: data = near-black dots; starting profile = orange (as in the FORSTER app); model/time
  from the fit = blue; conditions and law = green. f uses the same three colours for the three nested
  distributions (fit = blue, starting profile = orange, conditions = green).
- Distance from the rim runs left → right in every profile plot (rim at left); Fo up.
- No numeric ticks anywhere (illustrative).
- Look-up (step 4): web search found traverse figures over BSE images (red traverse lines on zoned
  olivine) but no habit drawing; the generic outline follows the note's instruction. No images saved.

## Negative constraints
- No numbers, no specific habit, no face indices, no growth stages, no second profile, no four-law switch,
  no code identifiers.
