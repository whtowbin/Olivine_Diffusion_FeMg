# Naming the starting profile for half profiles and whole crystals

Status (2026-10-16): **draft for Henry's decision.** Nothing is built. Started at Henry's
request so that whole crystals, joint fits, population priors, step proposals, the
edge-position fit and batch import all use one set of names, settled before any of them
is built. Requirements were collected from both sessions working on these features
(`TODO_local.md`: "Whole crystals", "Priors from the crystal population", "Automated
proposals", "Profiles along more than one axis", "Fit the profile edge position", "Batch
data import").

## The problem

FORSTER models **half profiles**: one rim, where Fo is held (`edge_fo`, or `rim_fo` /
`RimPath` when it differs), and a core, where Fo is held fixed at the far end of the
model (`core_fo`). Henry wants FORSTER also to model **whole crystals**: two rims, each
held, and a core free to change. The solvers already hold a value at each end of the
model (`src/forster/implicit.py`: `c[0]` and `c[-1]`), so the numerics need little change.
The names do:

- `InitialCondition(domain_um, edge_fo, core_fo, steps_um, step_fo)` names zones by their
  place in a half profile. `core_fo` is both the last zone's Fo and the value held at the
  model's end.
- One or two steps only, and one intermediate plateau (`step_fo`).
- The parameter table follows: `domain_start_um` ("Fo is held at edge_fo here"),
  `domain_end_um` ("held at core_fo here"), `edge_fo`, `core_fo`, `step_fo`, and the
  uncertainties `edge_fo_sd`, `core_fo_sd`, `step_fo_sd`, `steps_sd_um`.
- Growth stages (`stage_rim_um`, `stage_zone_fo`, `stage_rim_fo`) assume one growing rim,
  at `domain_start_um`.
- The round robin tables call the domain ends `edge_x1` and `edge_x2`.

## What the names must carry

From the whole-crystal and population sections:
1. Either geometry: a half profile (one rim held, the core held at the far end) or a whole
   crystal (both rims held, the core free).
2. Any number of steps, each zone with its own Fo.
3. A held value per end: a rim value (or a rim that changes in time) at each rim; for a
   half profile, the held core value.
4. Uncertainties on every step position and zone Fo, for the starting-profile fit.
5. Growth stages at either rim, or at both.
6. Population priors: a sample's crystals each with their own zone Fo, drawn from a shared
   distribution.

From the joint-fit, edge, proposal and import sections (the other session):
7. An edge offset per held end: one for a half profile, two for a whole crystal.
8. Step positions per profile, with the zone Fo shared across the profiles of one crystal.
   **Within one crystal only the compositions are shared** (Henry): step positions and zone
   widths stay per profile, because olivine grows faster on some faces than others and a
   section may miss the centre.
9. An optional per-profile core Fo that differs from the crystal's shared one.
10. An intersection point: a distance along each of two profiles where they cross.
11. A proposed step that the user accepts or rejects, with its prior sd or range.
12. Template columns that carry units (oxide wt%, element wt%, ppm, molar); stage x/y
    passed through unconverted; beam-scan profiles in their own layout. (These are profile
    table columns, not starting-profile names, but they share the template and its
    column guide.)

And always:
13. Every existing table, summary.csv and round robin table reads unchanged and refits
    identically (a differential replay on the template example, the paper runs and the
    round robin runs).

## Proposal

**Zones and ends named separately.** The starting profile is a list of zones between steps,
read from `domain_start_um` to `domain_end_um`; the ends are named for what is held there.

| Concept | Proposed name (table / Python) | Today's name, kept as an alias |
|---|---|---|
| Geometry | `geometry`: `half` (default) or `whole` | (always half) |
| Model domain | `domain_start_um`, `domain_end_um` (unchanged) | round robin `edge_x1`, `edge_x2` |
| Step positions | `steps_um`: any number, increasing | `steps_um` (1 or 2) |
| Zone Fo, one per zone, start to end | `zone_fo` (list, len(steps) + 1) | one step: `edge_fo`, `core_fo`; two steps: `edge_fo`, `step_fo`, `core_fo` |
| Held value at the start (a rim) | `start_rim_fo` (default: the first zone's Fo); a `RimPath` in Python | `rim_fo` |
| Held value at the end | half: the last zone's Fo (the core); whole: `end_rim_fo` (default: the last zone's Fo) | `core_fo` |
| Uncertainties | `steps_sd_um` / `steps_range_um` (one per step, unchanged), `zone_fo_sd` / `zone_fo_range` (one per zone) | `edge_fo_sd`, `core_fo_sd`, `step_fo_sd` (and `_range`) |
| Edge offsets | `start_offset_um`, `end_offset_um` (whole only), each with `_sd` / `_range` to fit it | (none) |
| Growth stages | `stage_rim_um`, `stage_zone_fo`, `stage_rim_fo` as today for the start rim; `stage_end_rim_um`, `stage_end_zone_fo`, `stage_end_rim_fo` for the end rim of a whole crystal | today's `stage_*` (start rim) |

**Grouping columns for joint fits and priors** (none affects a run fitted on its own):

| Concept | Proposed column | Meaning |
|---|---|---|
| Crystal | `crystal` | Profiles with the same value are one crystal: their `zone_fo` is shared, their `steps_um` and offsets are not (requirement 8). |
| Own core | `own_zones` | Zone indices whose Fo this profile fits on its own instead of sharing, e.g. the core (requirement 9). |
| Sample | `sample` | Crystals with the same value form a population, the source of priors and of the hierarchical joint fit (requirement 6). |
| Intersection | `cross_profile`, `cross_at_um`, `cross_other_at_um` | This profile crosses `cross_profile` at `cross_at_um` along itself and `cross_other_at_um` along the other (requirement 10). |

**Step proposals** (requirement 11) need no table column: proposals live in the GUI project
(`project.json`) until the user accepts one, which then writes its position into
`steps_um` and its prior into `steps_sd_um` or `steps_range_um`, so an accepted proposal is
an ordinary run row that `forster fit` reruns.

**Reading old tables.** A row that gives `edge_fo` / `core_fo` / `step_fo` and no
`zone_fo` is read as today, as a half profile. Giving both forms is an error unless they
agree. summary.csv keeps writing the old names for half profiles until the release, so
existing summaries refit; whole-crystal rows write the new ones.

## How the pieces use it

- **Whole crystal:** `geometry = whole`, steps e.g. `"12, 40"` with `zone_fo` of the outer
  zone, the core and the other outer zone; both ends held (`start_rim_fo`, `end_rim_fo`),
  the core free. The short-plateau warning does not apply; instead check that both rims'
  fronts are resolved and report when they meet in the core.
- **Half profile reaching a small crystal's centre:** by symmetry the same as a whole
  crystal mirrored at the centre. Either model it as `whole` (simple, twice the nodes), or
  add a half-profile end held as "no flux" (cheaper). Open question below.
- **Starting-profile fit:** the uncertain parameters become `steps_um[i]`, `zone_fo[j]` and
  the offsets, so the fit no longer special-cases edge, core and plateau.
- **Population priors:** a histogram of `zone_fo[j]` across the crystals of a `sample`
  becomes the prior for that zone; zones are matched by role (outer zone, plateau, core),
  so a role per zone may be needed (open question).

## Questions for Henry

1. **Words.** "rim" or "edge" for the held outer ends? "zone" or "plateau" for the regions
   between steps? (The proposal uses rim and zone; `edge_fo` stays as an alias.)
2. **Profiles reaching the centre:** model them as whole crystals, or add a cheaper
   half-profile end with no flux across the centre?
3. **Matching zones across crystals** for population priors and joint fits: by position in
   the list (needs the same number of zones everywhere), or by a role label per zone
   (outer, plateau, core)?
4. **Whole-crystal coordinates:** one origin at `domain_start_um` with the second rim at
   `domain_end_um` (proposed), or each rim its own origin?
5. **Aliases:** keep `edge_fo` / `core_fo` / `step_fo` indefinitely, or only until the
   release?

## Steps once decided

1. `InitialCondition` gains `zone_fo`, `geometry` and the held ends, with the old fields as
   aliases; table reader and writer, template and column guide follow.
2. Differential replay: every half-profile run (template example, paper runs, round robin
   runs) refits identically, with a baseline-against-itself control.
3. Whole crystals in the solvers' boundary setup; check a symmetric whole crystal against
   the half profile and an analytical slab solution.
4. The starting-profile fit over `steps_um[i]` / `zone_fo[j]` / offsets.
5. Then the joint fits, population priors, step proposals and batch import, each on these
   names.
