# Glossary

Terms used across this site, in alphabetical order.

1σ range
:   The range of times that should contain the true time 68% of the time, given the
    data and the inputs (`low_days`-`high_days`). See [How the fit
    works](how-it-works.md#4-the-ranges).

2σ range
:   The same at 95% (95.45%; `low2_days`-`high2_days`). Quote it when a warning says
    the 1σ range is probably too narrow.

AIC (Akaike information criterion)
:   A score for comparing models that charges for each extra parameter; lower is
    better. FORSTER uses the change in AIC (ΔAIC, `near_step_daic`) to ask whether the
    misfit is concentrated near the steps. A difference under 2 means the data cannot
    tell the models apart; below -6 FORSTER warns.

Anisotropy
:   Diffusion at different rates in different directions. Fe-Mg diffusion in olivine
    is about six times faster along [001] than along [100] or [010].

Chi-squared (χ²)
:   The misfit between model and data: the sum of the squared residuals, each divided
    by the expected size of its error. FORSTER computes it from the whitened residuals.

Core
:   The innermost segment of a half profile, held at its Fo at the far end of the
    model. Also used for the flat plateau beyond the last smoothed step.

Correlated misfit
:   The part of the residuals that varies smoothly along the profile (neighbouring
    points share it): amplitude `correlated_sigma_fo`, length `correlation_length_um`.

Coverage
:   How often a range actually contains the true time, measured on synthetic profiles
    of known age. A good 1σ range covers about 68%.

Diffusion length
:   l = √(D t): roughly how far a step has spread; on a fitted profile, about half the
    width of the smoothed step. Several checks measure distances in diffusion lengths.

Diffusion law (diffusivity equation)
:   The equation giving D from temperature, pressure, oxygen fugacity, Fo and
    direction. FORSTER offers four for Fe-Mg ([Setting up a
    run](setup.md#the-diffusion-law)).

Diffusivity (D)
:   How fast Fe and Mg exchange, in m²/s. A profile fixes the product D × t, so a
    larger D means a shorter time.

Domain (model domain)
:   The part of the profile that is modelled, from `domain_start_um` (edge 1) to
    `domain_end_um`. Points outside it are not fitted.

Edge 1
:   The end of the profile where the model holds Fo (`domain_start_um`), usually the
    crystal rim. Held at the first segment's Fo, or at `edge1_fo`.

ESS (effective sample size)
:   For a weighted sample: the number of equally weighted draws it is worth. Low ESS
    means a few draws carry most of the weight and the sample is unreliable; FORSTER
    then switches to nested sampling.

Fo (forsterite content)
:   Mg/(Mg + Fe) in olivine, in moles. FORSTER's tables use a fraction (0.895); the app
    shows mol% (89.5).

fO2 (oxygen fugacity)
:   The effective partial pressure of oxygen, in Pa, or relative to a buffer such as
    QFM (quartz-fayalite-magnetite) or NNO (nickel-nickel oxide) in log units. Fe-Mg
    diffusion is faster at higher fO2.

Growth stage
:   A zone that grew partway through: at its start the rim moves outward and a new
    zone appears ([Advanced](advanced.md#growth-stages)).

Half profile
:   A profile from the rim into a core large enough that diffusion never reaches the
    far end, which is held at the core's Fo. The alternative is a whole crystal,
    mirrored at its centre (`centre_um`).

Importance sampling
:   Drawing samples from an approximation and weighting each by how much more or less
    probable it really is, which corrects the sample to the exact distribution.

Laplace approximation
:   A bell-shaped (multivariate Gaussian) approximation of a posterior, centred on the
    best fit and shaped by the curvature of the misfit there.

Law range
:   The range of time including the diffusion law's own calibration uncertainty
    (`law_low_days`-`law_high_days`).

Marginalised
:   Averaged over the possible values of other parameters, each weighted by its
    probability. The posterior of time marginalised over the starting profile includes
    every starting profile the data and the priors allow.

Monte Carlo
:   Drawing uncertain inputs (T, P, fO2, the law's parameters) at random many times
    and looking at the spread of the results.

Near-step check
:   FORSTER's test of whether the misfit is larger near the steps than elsewhere
    (`near_step_daic`); when it is, the 1σ range is too narrow.

Nested sampling
:   A sampling method that explores the whole prior; slower than importance sampling
    but robust when the posterior has several peaks or is very non-Gaussian.

Noise model
:   What FORSTER assumes about the residuals: white noise plus a smooth correlated
    misfit, both fitted to the data. It sets how much each point counts and how wide
    the range is.

Plateau
:   A flat part of the profile: a segment diffusion has not reached, between steps or
    beyond the last one.

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
:   A stretch of constant Fo in the starting profile, labelled a, b, c, ... from edge
    1 inward (or by `segment_names`).

Starting profile (initial condition)
:   The Fo profile before diffusion: segments of constant Fo separated by steps.

Step
:   The boundary between two segments of the starting profile, where Fo jumps
    (`steps_um`).

White noise
:   Scatter that is independent from point to point, like analytical precision
    (`white_sigma_fo`).

Whitening
:   Transforming the residuals with the fitted noise model so that, if it is right,
    they become independent and of equal size. Points on the same stretch of smooth
    misfit then count as one piece of information rather than several.
