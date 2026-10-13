# Ni, Mn and Ca diffusion: model and fits

Status (2026-10-13): stage 1 (Python API) built and verified, uncommitted: `forster/elements.py`, `FitResult.element`, element plots, 14 tests in `tests/test_elements.py` (full suite passes with `pytest -n 8`; ruff and ty clean). Stage 2 (tables, CLI) waits for Henry's review.

## Goal

Fit diffusion times to Ni, Mn and Ca profiles measured alongside Fo, one element at a time and jointly with Fo, from Python first and then from the tables and CLI.

## Decisions (Henry, 2026-10-13)

- Fit mode: per-element fits first (each element's own time and 1σ range, a consistency check against Fo), then a joint fit (one time; the sum of each species' chi-squared, each with its own noise model).
- Initial condition: each element gives its own edge, core and plateau concentrations and uses Fo's step positions unless it gives its own.
- Scope: stage 1 is the Python API with physics tests; stage 2 adds table columns and CLI output after review.

## Approach

Ni and Mn diffusivities depend on X_Fo, so they are modelled with the Fo profile evolving alongside them on the same grid and time step:

    dC/dt = d/dx(D_el(X_Fo) dC/dx),   dX_Fo/dt = d/dx(D_FeMg(X_Fo) dX_Fo/dx)

Each time step advances Fo with the existing stepper and each element with the same stepper given the current Fo vector (`engine.diffusion_step(c, fo, ...)` for the central scheme, `dipra.diffusion_step(c, fo, ...)` for DIPRA's). The Fo rows are then bit-identical to `fit_profile`'s. Ca's DIPRA diffusivity does not depend on Fo; when no modelled element depends on Fo, Fo is not advanced. The equation is linear in each element's concentration, so any unit (wt% oxide, ppm) works without conversion.

Out of scope: partitioning boundary conditions, multicomponent coupling beyond D(X_Fo), the kriging (legacy) uncertainty method for elements, and choosing dx and dt (session github-drafts-68's `forster/grid.py`; until it lands, the element code checks only that D_max dt / dx² ≤ 0.5).

## Stage 1 tasks

1. `src/forster/elements.py` (new):
   - `ElementProfile(distance_um, values, element, name)`: element "Ni", "Mn" or "Ca"; at least 3 finite points.
   - `ElementInitial(edge, core, step=None, steps_um=None)`: concentrations in the profile's units; steps default to the Fo initial condition's. One step without a plateau, two with.
   - `DEFAULT_DIFFUSIVITY = {"Ni": "girona_costa2013_ni", "Mn": "girona_costa2013_mn", "Ca": "girona_costa2013_ca"}`; `element_diffusivity(element, diffusivity=None)` resolves and checks the element.
   - `coupled_model(...) -> (start, advance)`: the packed state [Fo, element 1, ...] advanced together, resumable in blocks; checks stability for every species.
   - `fit_element(profile, conditions, orientation, fo_initial, initial, grid, *, diffusivity=None, fo_diffusivity=..., scheme="central", stop_early=True) -> FitResult`: the correlated method on the element's points; early stopping and extension as for Fo.
   - `fit_joint(fo_profile, element_profiles, conditions, orientation, fo_initial, element_initials, grid, *, diffusivities=None, fo_diffusivity=..., scheme="central", stop_early=True) -> JointFit`: chi-squared summed over species, each with its noise model fitted at its own first-pass best fit; the range is where the sum rises by `interval_threshold(total points)`. `fo_profile` may be None (elements only).
2. `model.py`: `FitResult.element` ("Fe-Mg" by default). `plotting.py`: element results plotted in their own units.
3. Tests, `tests/test_elements.py`:
   - Mn identity: Mn uses the same law as Fe-Mg, so a Mn start that is a + b·(Fo start) stays a + b·Fo at every step (tests the coupling and the cross term through the Fo gradient).
   - Ni with flat Fo has constant D: matches the error-function solution.
   - Ca: D does not depend on Fo, Fo is not advanced, and the profile matches the error-function solution with the per-axis D combination.
   - Ni with Fo zoning against an independent flux-form solver for both species (D on cell faces): agreement within that solver's grid error.
   - The Fo rows of a coupled run equal `fit_profile`'s curves exactly.
   - Early stopping gives the same fit as a full run (element and joint).
   - `fit_element` recovers the time of a profile made by the reference solver.
   - `fit_joint` with Fo alone equals `fit_profile`'s correlated fit; with Fo and Ni made at one time, the joint range is no wider than the narrower single range.
   - Stability: a dt unstable for the element (not for Fo) is an error naming the species.
   - Input errors: unknown element, a diffusivity for the wrong element, two steps without a plateau, duplicate elements in a joint fit.
4. Real-data check (reported, not asserted): Fo, Ni, Mn and Ca fits on the AZ18 example and a KS20 round robin profile, with times compared across elements.

Verify: `uv run pytest -n 8`, ruff, ty, complexipy.

## Open points

- Small-sample threshold for the joint fit: `interval_threshold(n_total, 1)` treats the summed chi-squared like one species with n_total points, though each species' noise scale is estimated separately. To check by coverage tests before the joint range is relied on.
- Element initial conditions for real profiles are not in any table yet; stage 1's real-data check estimates edge and core values from the data ends.

## Stage 1 results (2026-10-13)

Physics checks (all in `tests/test_elements.py`):

- Mn with the Fe-Mg law stays a + b Fo to 1.5e-15 over 992 steps.
- Ni in zoned olivine against an independent flux-form solver for both species:
  1.2e-4 (central) and 1.35e-4 (DIPRA) on a 0.1 step, within that solver's own grid
  error (3-4e-4). The control with D frozen at the mean Fo is 1.7e-3 off.
- Ni in unzoned olivine and Ca at any Fo match the error-function solution (< 1e-5).
- The Fo rows of a coupled run equal the Fo model's exactly; early stopping gives the
  same element and joint fits as full runs; a joint fit of Fo alone equals
  `fit_profile`.
- Ni diffuses 1.05-1.48x faster than Fe-Mg at round robin conditions and 1.24-1.76x
  at the paper's, so Ni can limit the time step. Ca is 0.25-0.41x.

Real profiles, element edge and core taken as the means of the first 3 and last 5
points in the domain, steps at Fo's (not fitted; times are illustrative):

| Profile | Fo | Ni | Mn | Ca | Joint (all four) |
|---|---|---|---|---|---|
| AZ18 WHT01_Ol_63 (1250 C) | 5.10 (4.20-6.19) | 0.00: NiO contrast 0.012 wt%, within noise | 19.4 (9.3-40.2) | 4.2 (1.8-11.4) | 5.39 (4.49-6.46) |
| KS20-527_2-transect-1 (1200 C) | 42.2 (34.4-50.4) | 6.7 (4.2-11.5) | 16.4 (7.3-39.5) | 46.9 (17.9-125) | 38.1 (31.0-45.6) |

Days, 1-sigma ranges. Findings:

- Ca agrees with Fo on both profiles; Mn does not on either; Ni on KS20 is 6x
  shorter than Fo. On KS20 the plot shows the guessed Ni rim value (0.248) above the
  measured rim (0.228-0.240) and the 50-90 um points below the model, so Ni's
  initial condition, not only its diffusivity, is in question.
- The joint fit hides disagreement: Fo + Ni on KS20 gives 39.4 d, near Fo's 42.2,
  although Ni alone gives 6.7. A joint fit needs a consistency check before its
  range means anything (for example each species' chi-squared rise at the joint best
  fit against its own minimum, or a test of a common time).
- A profile without resolvable zoning (AZ18 Ni) fits at time zero. It should be
  reported as unconstrained, not as 0 days.

## Before stage 2

- A consistency statistic for joint fits.
- Element initial conditions fitted or scanned (the `forster.scan` machinery applies),
  not guessed from the data ends.
- `ElementProfile.units` for axis labels and tables.
- Check the DIPRA Ni law against its source, Petry et al. (2004), and the Ni and Mn
  pressure terms against Holzapfel et al. (2007), now in Henry's Zotero.
