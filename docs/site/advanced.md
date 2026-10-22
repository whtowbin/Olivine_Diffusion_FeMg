# Advanced

These features go beyond fitting one profile with one starting profile. Some are available from Python only for now, and each section says which.

## Growth stages

A zoned crystal can record more than one growth event. A core may diffuse for some
time before a new rim zone grows around it. If such a profile is fitted as if every
zone had been there from the start, the time matches neither event. **Growth stages**
model this history. At the start of each stage the rim moves outward and a new zone
with its own Fo appears. From then on the new zone diffuses together with the older
crystal, so inner zones have diffused for longer than outer ones.

![A crystal growing in two stages](images/schematic-growth-stages.png)

*A crystal growing in two stages, computed with FORSTER's own model on a synthetic
profile. **a** Stage 0: the starting crystal diffuses for 6 days with its rim held.
**b** Stage 1 at 6 days: the rim moves out and a new zone of Fo 0.82 appears, with a
sharp step on top of the gradient that has already diffused. **c** The measured
profile at 10 days and the staged fit. **d** The misfit against time. The staged model
fits at 10.1 days since the first stage (4.1 days since the last). A fit of the same
profile with both zones present from the start gives 6.7 days, which matches neither
stage, with a misfit almost as low. FORSTER's near-step check flags the single-stage
fit, but the growth history itself has to come from other evidence.*

**From the tables**, give one entry per stage. For several stages, separate the entries
with commas and put the cell in quotes (for example `"5, 12"`):

| Column | Meaning |
|---|---|
| `stage_days` | When each stage starts, in days after the start of the run. |
| `stage_rim_um` | The rim position from that stage on. The crystal grows outward, so it is at most the previous rim's (`domain_start_um` before the first stage). |
| `stage_zone_fo` | Fo of the zone the stage adds between the old and the new rim. |
| `stage_rim_fo` | Optional: Fo held at the rim from that stage on (default: the new zone's Fo). |

The domain, step and Fo columns describe the crystal **before** the first stage. Only
times after the last stage are fitted. The reported `time_days` counts from the start
of the run, and `time_since_last_stage_days` counts from the last stage. Growth stages
need `dx_um` and `dt_s` given, the correlated uncertainty method, and no
starting-profile fit (no `_sd` or `_range` columns). In the app, the stages are a
table in the Advanced tab.

**From Python:**

```python
from forster import Stage, fit_profile

overgrowth = Stage(start_s=6 * 86400, rim_um=0.0, zone_value=0.82)
result = fit_profile(profile, conditions, orientation, initial, grid, stages=[overgrowth])
print(result.time_days, result.time_since_last_stage_days)
```

With the implicit scheme, stages start at their exact times and the rim value can
change with time. `fit_profile(..., rim_fo=forster.boundary.RimPath(times_days, values))`
holds the rim at a value interpolated linearly between the given times.

!!! warning "The history must come from elsewhere"
    A staged history and a single-stage history can fit a profile almost equally well
    and give different times. The stage start times are inputs and are not fitted.
    Choose them from petrography, from other elements, or from zoning that only growth
    can explain.

## Joint fits of one crystal

*Python only for now.*

Profiles measured in different directions across one crystal record the same history.
`forster.fit_crystal` fits them together. The profiles share **the time**, **the
conditions** (which must be identical) and **the Fo of each segment**. Segments are
matched by label, so give them the same names on each profile with `segment_names`.
Each profile keeps its **own step positions**, since zones grow at different rates on
different faces. It also keeps its **own orientation** (and so its own D) and its
**own noise model**.

![Two profiles of one crystal fitted jointly](images/schematic-joint-fit.png)

*What is shared and what is per profile. **a** A zoned crystal with two traverses, one
near the fast c axis and one near the slow a axis. **b** The profiles share one time,
the same T, P and fO2, the Fo of each segment (dashed levels) and the edge Fo. Each
profile has its own steps, its own orientation and D, an Fo offset for microprobe
drift between sessions, and optionally an edge position offset. **c** The joint misfit
is the sum of the two profiles', and its range is narrower than either
profile's alone. Illustrative, synthetic crystal, drawn while the joint fit was being built (hence
"planned" in panel c).*

```python
import forster
from forster import fit_crystal

profiles = forster.read_profiles("crystal/profiles.csv")
run_a, run_b = forster.read_runs("crystal/parameters.csv")   # with starting-profile priors
crystal = fit_crystal([run_a, run_b], profiles)
print(crystal.best_days, crystal.range_days, crystal.range2_days)
print(crystal.segment_fo, crystal.fo_offset, crystal.agree)
```

What it does:

- Each profile is first fitted alone (`crystal.individual`). If two profiles' 2σ
  ranges do not overlap, FORSTER warns and `crystal.agree` is False. Before trusting
  the joint time, check the orientations, whether a profile is off the crystal's
  center, and look for a 3D effect near a corner.
- Each profile gets an Fo offset (`<run>:fo_offset`) that absorbs microprobe drift
  between sessions, with a Gaussian prior of 0.1 mol% (`fo_offset_sd`). The profile
  with the most points is the reference and has no offset.
- An edge position offset (`<run>:edge1_offset_um`, where the true edge lies in the
  profile's measured distances) can be switched on with `edge_offset_sd_um=2.0` when
  the steps are well constrained.
- A segment can belong to one profile only. For example,
  `own_segments={"run_b": ["c"]}` lets that profile fit its own core.
- The sampler is the same as for a single starting-profile fit. Its settings were
  calibrated on two real crystals to about 5% on the 16th, 50th and 84th percentiles
  of time.

The result's ranges (`low_days`-`high_days`, `low2_days`-`high2_days`) are the
central 68% and 95% of the joint posterior of time, with every free parameter
included. Joint fits do not yet support growth stages or a held `edge1_fo`, and they
cannot yet be run from the tables, the command line or the app (the `crystal` column
is carried through for now). The design is in
[2026-10-16-joint-fits.md](https://github.com/whtowbin/Olivine_Diffusion_FeMg/blob/master/docs/plans/2026-10-16-joint-fits.md)
on GitHub.

## Ni, Mn and Ca

*Python only for now.*

Ni, Mn and Ca profiles in the same olivine give a check on the Fo time. They diffuse
at different rates, so a single history should explain all of them.

- `forster.fit_element` fits a Ni, Mn or Ca profile for its own time.
- `forster.fit_joint` fits one time to Fo and the elements together and reports
  whether they agree. `consistency_p` is the probability of a disagreement at least
  this large if they do share one time.

The Ni and Mn diffusivities depend on Fo, so Ni and Mn diffuse with the Fo profile
alongside them. Ca does not need it. The element's edge, core and plateau
concentrations are fitted unless you give them. Concentrations can be in any unit.

```python
from forster import ElementProfile, fit_element, fit_joint

ni = ElementProfile(distance_um, nio_wt_pct, "Ni", name="my profile")
result = fit_element(ni, conditions, orientation, fo_initial, None, grid)
joint = fit_joint(fo_profile, [ni], conditions, orientation, fo_initial, None, grid)
print(result.time_days, joint.time_days, joint.consistency_p, joint.species_days)
```

The element fits can use DIPRA's Ni, Mn and Ca equations (Girona and Costa 2013,
Table 1), with DIPRA's Ni equation as the default for Ni. The Ni equation of Petry et
al. (2004) is also available on its own (`petry2004_ni`). DIPRA's Ni equation pairs
Petry's pre-factor with the activation energy of a different fit, which makes Ni 0.14
log units slower at 1200 °C. There are also the Ni and Mn regressions of Mutch
et al. (2019) (`mutch2019_ni`, `mutch2019_mn`). Choose one with
`fit_element(..., diffusivity="petry2004_ni")`. These element fits use the explicit
schemes.

## Project folders

The desktop app works on a **project folder**. This is an ordinary folder of FORSTER
tables:

| File | Holds |
|---|---|
| `profiles.csv` | Every profile, with `Marked_bad` = `bad` on masked points (the table reader drops them, so `forster fit` leaves them out too). |
| `parameters.csv` | One row per run. |
| `diffusivities.csv` | The project's own diffusion equations, if any. |
| `project.json` | What only the app needs: the format version, the selected run, each run's view settings, and which cached result belongs to which run. |
| `.forster/cache/` | Cached fits, starting-profile fits and Monte Carlos (NumPy and JSON files, never pickles), each under a key computed from everything that determines it: the run's row, the profile's numbers, the equations and the FORSTER version. A run whose settings changed has no result rather than a stale one. |
| `.forster/autosave/` | The autosave, written every minute without touching the saved tables. |

The command line reads the same tables, so `forster fit profiles.csv parameters.csv`
reruns a saved project. A project can be shared as a zip (the app's Export) or kept
under version control (leave out `.forster/`). From Python, `forster.project.Project`
opens, edits and saves a project with the same checks as the app
([Python API](api.md#project-folders-forsterprojectproject)).
