# Brief: the FORSTER workflow (schematic)

- Mode: design. Format: journal, double column, 183 mm wide (about 112 mm high). Methods paper, early figure.
- Reader: petrologists and volcanologists who have not used FORSTER; they know zoned olivine and
  microprobe traverses, not the code.
- Style: own choice. Clean flat Nature-style line art: small schematic plots with no numeric ticks
  (every curve illustrative), a generic olivine section, one colour per role carried through all panels.
  Chosen because the content is a computation on profiles (plots are the native register), not hardware.

**Takeaway (one sentence):** a profile's shape fixes D × t, and FORSTER turns that into a time with an
honest range: forward model in time, misfit against time, best fit with 1σ, then the starting profile and
the conditions widen it into the final time distribution.

## Key ideas (user's criteria first, kept verbatim)
1. "Pipeline: inputs (measured Fo profile from microprobe spots, T, P, fO2, profile orientation in the
   crystal, step-shaped starting profile with segments) → forward diffusion model in time → misfit vs time
   with the noise model → best-fit time with 1σ range → (optional) uncertain starting profile fitted too,
   and Monte Carlo over T, P, fO2 and the diffusivity law → final time distribution."
2. "a profile's shape fixes D × t, and FORSTER turns that into a time with an honest range."
3. The model is run once forward; every time step is compared with the data (misfit is a curve in time).
4. Each added source of uncertainty widens the time distribution (fit alone ⊂ + starting profile ⊂ +
   conditions and law).
5. Faster D means a shorter time for the same shape (t scales as 1/D).

## Must not imply
- That FORSTER fits a curve shape freely and divides by D afterwards for the fit itself (the fit runs the
  diffusion model with D from the conditions; only the Monte Carlo rescales by D, for a fixed rim).
- Specific numbers, a specific crystal habit, face indices, or a real sample.
- That the starting-profile fit is mandatory (it is optional, used when steps or segment Fo are uncertain).
- That the Monte Carlo switches between the four named laws (it draws the chosen law's parameters from
  its covariance; the law comparison is a separate option).
- Growth stages and joint fits of several profiles (out of scope; other figures).

## Scope
One overview figure of the single-profile workflow. Growth stages, joint fits, Ni/Mn/Ca and the GUI are
left to other figures or the text. Numbers, thresholds and options go in the caption.
