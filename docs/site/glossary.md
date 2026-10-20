# Glossary

Terms used across this site, in alphabetical order.

1σ range
:   The range of times that should contain the true time 68% of the time, given the
    data and the inputs (`low_days`-`high_days`). See [How the fit
    works](how-it-works.md#4-the-ranges).

2σ range
:   The range of times that should contain the true time 95% of the time (95.45%;
    `low2_days`-`high2_days`). Quote it when a warning says the 1σ range is probably
    too narrow.

AIC (Akaike information criterion)
:   A score for comparing models that penalizes each extra parameter. Lower is better.
    FORSTER uses the change in AIC (ΔAIC, `near_step_daic`) to test whether the misfit
    is concentrated near the steps. A difference under 2 means the data cannot tell the
    models apart. Below -6, FORSTER warns.

Anisotropy
:   Diffusion that runs at different rates in different crystallographic directions.
    Fe-Mg diffusion in olivine is about six times faster along [001] than along [100]
    or [010].

Chi-squared (χ²)
:   A measure of the misfit between model and data. It is the sum of the squared
    residuals, each divided by the expected size of its error. FORSTER computes it from
    the whitened residuals.

Core
:   The interior of the crystal. In the model it is the innermost segment of a half
    profile, and its Fo is held fixed at the far end of the model. The word is also used
    for the flat plateau beyond the last smoothed step.

Correlated misfit
:   The part of the residuals that varies smoothly along the profile, so that
    neighboring points share it. Its size is `correlated_sigma_fo` and its length
    `correlation_length_um`.

Coverage
:   How often a range actually contains the true time, measured on synthetic profiles
    of known age. A good 1σ range covers about 68%.

Diffusion length
:   Roughly how far a step has spread by diffusion, l = √(D t). On a fitted profile it
    is about half the width of the smoothed step. Several checks measure distances in
    diffusion lengths.

Diffusion law (diffusivity equation)
:   The equation that gives the diffusivity D from temperature, pressure, oxygen
    fugacity, Fo and crystallographic direction. FORSTER offers four for Fe-Mg
    ([Setting up a run](setup.md#the-diffusion-law)).

Diffusivity (D)
:   The diffusion coefficient: how fast Fe and Mg exchange, in m²/s. A profile fixes the
    product D × t, so a larger D means a shorter time.

Domain (model domain)
:   The part of the profile that is modeled, from `domain_start_um` (edge 1) to
    `domain_end_um`. Points outside it are not fitted.

Edge 1
:   The end of the profile where the model holds Fo fixed (`domain_start_um`), usually
    the crystal rim. Its Fo is that of the first segment, or `edge1_fo`.

ESS (effective sample size)
:   The number of equally weighted draws that a weighted sample is worth. A low ESS
    means a few draws carry most of the weight and the sample is unreliable. FORSTER
    then switches to nested sampling.

Fo (forsterite content)
:   The molar ratio Mg/(Mg + Fe) in olivine. FORSTER's tables use a fraction (0.895);
    the app shows mol% (89.5).

fO2 (oxygen fugacity)
:   The effective partial pressure of oxygen, in Pa, or in log units relative to a
    buffer such as QFM (quartz-fayalite-magnetite) or NNO (nickel-nickel oxide). Fe-Mg
    diffusion is faster at higher fO2.

Growth stage
:   A zone that grew after diffusion had already begun. At the start of the stage the
    rim moves outward and a new zone appears ([Advanced](advanced.md#growth-stages)).

Half profile
:   A traverse from the rim into a core large enough that diffusion never reaches the
    far end, which is held at the core's Fo. The alternative is a whole crystal,
    mirrored at its center (`centre_um`).

Importance sampling
:   A way to sample a distribution. Draws come from an approximation, and each is
    weighted by how much more or less probable it is under the exact distribution. The
    weights correct the sample to the exact distribution.

Laplace approximation
:   A bell-shaped (multivariate Gaussian) approximation of a posterior, centered on the
    best fit and shaped by the curvature of the misfit there.

Law range
:   The range of time including the diffusion law's own calibration uncertainty
    (`law_low_days`-`law_high_days`).

Marginalized
:   Averaged over the possible values of other parameters, each weighted by its
    probability. The posterior of time marginalized over the starting profile includes
    every starting profile the data and the priors allow.

Monte Carlo
:   Drawing the uncertain inputs (T, P, fO2, the law's parameters) at random many times
    and looking at the spread of the results.

Near-step check
:   FORSTER's test of whether the misfit is larger near the steps than elsewhere
    (`near_step_daic`). When it is, the 1σ range is too narrow.

Nested sampling
:   A sampling method that explores the whole prior. It is slower than importance
    sampling but reliable when the posterior has several peaks or is far from Gaussian.

Noise model
:   FORSTER's description of the scatter in the residuals: white noise plus a smooth
    correlated misfit, both fitted to the data. It sets how much each point counts and
    how wide the range is.

Plateau
:   A flat part of the profile: a segment that diffusion has not reached, between steps
    or beyond the last one.

Posterior
:   What is known about a parameter after fitting: the prior combined with what the
    data say.

Prior
:   What is known about a parameter before fitting this profile: a nominal value and a
    Gaussian 1σ (`_sd` columns) or a uniform ± half-width (`_range` columns).

Residual
:   Measured Fo minus model Fo at a measured point, at the best-fit time.

Run
:   One fit of one profile with one set of inputs: a row of `parameters.csv`.

Segment
:   A stretch of constant Fo in the starting profile, labeled a, b, c, ... from edge
    1 inward (or by `segment_names`).

Starting profile (initial condition)
:   The Fo profile before diffusion began: segments of constant Fo separated by steps.

Step
:   The boundary between two segments of the starting profile, where Fo jumps
    (`steps_um`).

White noise
:   Scatter that is independent from point to point, like analytical precision
    (`white_sigma_fo`).

Whitening
:   Transforming the residuals with the fitted noise model so that, if the model is
    right, they become independent and of equal size. Points on the same stretch of
    smooth misfit then count as one piece of information rather than several.
