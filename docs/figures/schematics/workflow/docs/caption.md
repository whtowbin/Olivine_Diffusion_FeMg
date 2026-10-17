# Caption

**Figure N | How FORSTER turns a zoned olivine profile into a time with an honest range.** The shape of a
diffusion profile fixes the product of diffusivity and time, D·t; FORSTER runs the diffusion model forward
in time with D from the stated conditions, finds the time whose model matches the measured profile, and then
widens that time by the uncertainty of the starting profile, the conditions and the diffusivity law.

**a,** Inputs: a forsterite (Fo) profile measured by electron microprobe spots along a traverse from the
crystal rim inward; temperature *T*, pressure *P* and oxygen fugacity *f*O₂; the profile's orientation
relative to the olivine axes [100], [010] and [001] (from EBSD; diffusion along [001] is about six times
faster than along [100] or [010], and D along the profile combines the three by the angles between them);
and a step-shaped starting profile whose constant-Fo segments (a, b, ... from the rim inward) meet at
steps. **b,** The model is run once forward in time. The rim is held at the rim Fo; the step relaxes, and its
width grows as the diffusion length √(D t). Curves at three times t₁ < t₂ < t₃ are shown against the
measured points. **c,** Every time step is compared with the data: the misfit (χ²) traced against time has its
minimum at the best-fit time, and the 1σ range is where it rises by about 1 (slightly more for profiles with
few points). The misfit uses a noise model of white, point-to-point analytical scatter plus a smooth,
spatially correlated misfit, both fitted to the residuals (lower plot). **d,** When step positions or segment
Fo are uncertain they carry priors (a standard deviation or a range) and are fitted together with the time
(optimisation, then importance sampling, nested sampling if needed); faint lines are alternative starting
profiles allowed by the priors. **e,** A Monte Carlo draws *T*, *P*, *f*O₂ and the diffusivity law's
parameters (from their covariance) and so draws D. For a fixed rim the time rescales exactly as
t = (D t)_fit / D, so a faster D gives a shorter time and no new model runs are needed. **f,** The resulting
time distributions on a log time axis: the fit alone, with the starting-profile uncertainty, and with the
conditions and law as well; FORSTER reports the median and the 68% and 90% ranges.

All curves are illustrative, not data from a real crystal: constant-D error-function solutions of one step
with the rim held, synthetic points with white and correlated noise, and illustrative widths for each source
of uncertainty (figure/data/make_curves.py). FORSTER itself uses an Fo-, *T*-, *P*-, *f*O₂- and
direction-dependent D solved by finite differences. The olivine outline is generic; no crystal habit is implied.
