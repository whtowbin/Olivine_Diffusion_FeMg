# Caption

**Fig. X | A profile with several zones can record several growth events; modelled as
growth stages, each zone diffuses only from the time it grew.**
In FORSTER a crystal can grow in instantaneous stages. Stage 0 is the starting crystal. At
the start of each later stage the rim moves outward and a new zone with its own Fo
appears; from then on it diffuses together with everything grown before it, so inner
steps are older and smoother than outer ones. A stage can also change the Fo held at the
rim. Only model times after the last stage can fit the measured profile, and FORSTER
reports the time since the first stage and the time since the last stage. Stage start
times are fixed inputs. The zones reuse the geometry of profile 1 of Fig. Y (the joint
fit): its step positions (15.2 and 36.0 µm from the rim), its length (76 µm) and point
spacing (2 µm), and its four levels mapped linearly to Fo (edge 0.80, rim zone 0.83,
mantle 0.867, core 0.90). The curves are not that figure's curves: here the zones are
grown and diffused in stages.

**a**, Stage 0: a core crystal of Fo 0.90 (from 36 µm inward), its edge held at the
mantle Fo, 0.867 (the melt the mantle will grow from), for 0.5 days. Dashed: the starting
profile; blue: after 0.5 days. The outer 36 µm has not grown yet. **b**, Stage 1 at
0.5 days: the mantle grows (15-36 µm) at Fo 0.867 and the rim is held at that value.
Dashed: just after it grew; blue: at 1.25 days, the inner gradient having spread into the
mantle. **c**, Stage 2 at 1.25 days: the rim zone (0-15 µm, Fo 0.83) grows, and the edge
is then held at Fo 0.80. Its two steps are sharp, on an inner gradient that has already
diffused for 1.25 days. **d**, The measured profile at 1.4 days (grey circles) and the
staged model's best fit (blue). The outer steps have diffused for 0.15 days (diffusion
length √(Dt) about 2 µm), the inner step for 1.4 days (5.5 µm). **e**, Misfit (χ², log
scale) against time since stage 0. The staged model has no fit before stage 2, because
the rim zone does not exist yet; its best fit is at 1.39 days (1σ 1.38-1.40), which
FORSTER reports as 1.39 days since the first stage and 0.14 days since the last (true
1.4 and 0.15). Fitting the same profile with every zone present from time zero (grey
dashed) gives 0.21 days (1σ 0.15-0.29), which is neither of the two times. Its misfit is
as low (χ² 38.1 against 39.0 for 39 points), because each fit's noise model absorbs its
own systematic misfit. Here FORSTER's near-step check does not flag it either (AIC
improves by 3.2; it warns at 6 or more; flagged in 2 of 10 noise draws): its misfit lies
across the whole inner gradient (up to 0.014 Fo over 38-56 µm), wider than the zone next
to the steps that the check tests. The residuals do show it: 4.3 times the analytical
noise (0.0043 against 0.0010 Fo), against 0.9 times for the staged fit. That the cause
is a growth stage still has to come from petrography, other elements or zoning that only
growth explains. Circled letters mark the start of each stage (**a**-**c**) and the best
fit (**d**).

Illustrative case, computed with FORSTER's own model: 1200 °C, 42 MPa, fO2
1.38 × 10^-3 Pa, profile along [001], Dohmen & Chakraborty (2007) diffusivity. The
"measured" profile is synthetic (39 points from an independent finite-difference solver
at 1.4 days, noise 0.001 Fo). The crystal sections are coloured by the computed profile as
if every face had the same profile (real diffusion is faster along c, so real zones
differ in width by direction); the crystal outline is generic. Everything here is
illustrative; no value is from a real crystal.
