# Naming the starting profile for half profiles and whole crystals

Status (2026-10-16): **design decided by Henry** (section "Decisions"); three small points
still to confirm (last section). Nothing is built. Started at Henry's
request so that whole crystals, joint fits, population priors, step proposals, the
edge-position fit and batch import all use one set of names, settled before any of them
is built. Requirements were collected from both sessions working on these features
(`TODO_local.md`: "Whole crystals", "Priors from the crystal population", "Automated
proposals", "Profiles along more than one axis", "Fit the profile edge position", "Batch
data import"). Both sessions have reviewed it.

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
    column guide.) In detail:
    - each concentration column declares its unit, by a header suffix (`NiO_wt%`,
      `Ni_ppm`) or a units row; a missing or ambiguous unit fails the import loudly
      instead of being guessed;
    - Fo is computed from FeO and MgO when there is no Fo column;
    - Ni, Mn and Ca columns go through the minor-elements stage-2 reader, so there is one
      profile reader, not two.
14. Profiles marked as one crystal "by file name or by selecting a box" (Henry): the
    `crystal` column filled either from the file or profile name, by a rule the batch
    reader proposes and the user confirms (never assumed), or from a GUI check box.

And always:
13. Every existing table, summary.csv and round robin table reads unchanged and refits
    identically (a differential replay on the template example, the paper runs and the
    round robin runs).

## Decisions (Henry, 2026-10-16)

1. **Words.** The held outer ends are **edge 1** and **edge 2**. The regions between steps
   are **segments** in the tables and the code; the docs and the GUI may also call them
   zones and plateaus, to help users pick concentrations for different growth periods.
2. **The core end defaults to no flux.** A half profile's far end becomes a symmetry
   boundary (no Fo crosses it), so the core is free to change; today it is held fixed.
   As an option the user models the whole profile by giving the **distance of the
   crystal's centre**, across which the profile is mirrored. The centre may lie beyond the
   last measured point when the crystal is large and its core plateau flat.
3. **Segments are labelled a, b, c, ...**, renamable by the user, and matched across
   profiles and crystals by label (for joint fits and population priors). In the GUI a
   table at the side defines the segments, one row each: name, start and end position,
   and concentration, with the labels shown above the plot where priors are selected and
   dragged. **An edge concentration may differ from the concentration of the segment next
   to the edge** (today's `rim_fo` against `edge_fo`).
4. **Distances.** For a single profile the edges are the first and last modelled points,
   usually 0 on one side and a larger number on the other; the user may instead put zero
   at the crystal's centre. Profiles on the same crystal do not share an origin (they do
   not lie on one line); across them only segment concentrations are matched.
5. **Names from the papers stay.** (Read as: the papers' tables, column names and sample
   names keep working unchanged, so the old column names remain accepted; see the last
   section.)

## Design

| Concept | Table column / Python | Old name, still accepted |
|---|---|---|
| Model domain: edge 1 and the far end | `domain_start_um` (edge 1), `domain_end_um` (edge 2 of a whole crystal, else the core end) | round robin `edge_x1`, `edge_x2` |
| Far end | `core_end`: `no_flux` (default for new runs) or `fixed` (held at the last segment's Fo, today's behaviour) | (always fixed) |
| Whole profile by mirroring | `centre_um`: distance of the crystal's centre; the model runs from edge 1 to the centre (beyond the data if needed) and mirrors there | (none) |
| Origin | `origin`: `edge` (default: distances from edge 1) or `centre` (zero at `centre_um`) | (always edge) |
| Segment boundaries | `steps_um`: any number, increasing | `steps_um` (1 or 2) |
| Segment labels | `segment_names` (list; default `a, b, c, ...` from edge 1 inward) | (none) |
| Segment Fo | `segment_fo` (list, one per segment) | one step: `edge_fo`, `core_fo`; two: `edge_fo`, `step_fo`, `core_fo` |
| Held edge values | `edge1_fo` (default: segment a's Fo), `edge2_fo` (whole crystals; default: the last segment's) | `rim_fo` |
| Uncertainties | `steps_sd_um` / `steps_range_um` (per step), `segment_fo_sd` / `segment_fo_range` (per segment), `edge1_fo_sd` ... | `edge_fo_sd`, `core_fo_sd`, `step_fo_sd` |
| Edge offsets | `edge1_offset_um`, `edge2_offset_um`, each with `_sd` / `_range` | (none) |
| Growth stages | `stage_*` as today at edge 1; `stage_edge2_*` at edge 2 of a whole crystal | today's `stage_*` |
| Crystal | `crystal`: profiles with the same value share segment Fo by label, not step positions or offsets (requirement 8); filled from a confirmed name rule or a GUI check box, an explicit value winning (requirement 14) | (none) |
| Own segments | `own_segments`: labels this profile fits on its own, e.g. its core (requirement 9) | (none) |
| Sample | `sample`: crystals forming a population, for priors and the hierarchical fit | (none) |
| Intersection | `cross_profile`, `cross_at_um`, `cross_other_at_um` (requirement 10) | (none) |

**Segments in the table and the GUI.** In the parameter table a run's segments are list
columns (`segment_names`, `segment_fo`, with `steps_um` as the boundaries between them),
so one row is still one run. The GUI shows the same thing as Henry's side table, one row
per segment with its start, end and Fo; the starts and ends are the edges and the steps,
so editing one segment's end moves the next one's start.

**Step proposals** need no column: they live in the GUI project (`project.json`) until
accepted, which writes `steps_um` and `steps_sd_um`, so an accepted proposal is an
ordinary run row that `forster fit` reruns.

**The short-plateau warning** changes with the no-flux default: a far end with no flux
where the data still slope (they stop before the plateau) is as wrong as a held core
there. It stays for half profiles without `centre_um`, saying to measure further or give
the centre; with `centre_um`, or for a whole crystal, it is replaced by a check that the
fronts are resolved and a note when they meet in the core.

## Still to confirm

1. **Old tables and the new default.** With `core_end` defaulting to `no_flux`, a table
   written before this change (no `core_end` column) would refit differently wherever
   diffusion reaches the far end: exactly the short profiles the warning flags; long
   plateaus are unaffected. Proposal: rows without a `core_end` column read as `fixed`, so
   published tables and summaries refit identically, and new templates write `no_flux`.
   A differential replay measures how much each paper and round robin run would move under
   `no_flux`.
2. **Decision 5.** Confirm the reading: the old column names (`edge_fo`, `core_fo`,
   `step_fo`, `rim_fo`, and the round robin `edge_x1` / `edge_x2`) stay accepted
   indefinitely, and sample and profile names are never renamed.
3. **Crystal grouping:** an explicit `crystal` value (typed or from the check box) wins over
   the file-name rule, which only proposes values for empty cells. Agreed?

## Steps once confirmed

1. `InitialCondition` gains segments (`steps_um`, `segment_names`, `segment_fo`), the held
   edges, `core_end`, `centre_um` and `origin`, with the old fields accepted; table
   reader and writer, template and column guide follow.
2. Differential replay: every run read from an existing table (template example, paper
   runs, round robin runs) refits identically, with a baseline-against-itself control;
   then the same runs under `no_flux`, to report what the new default changes.
3. No-flux far end and mirroring at `centre_um` in the solvers; check against an
   analytical slab solution and a whole profile modelled explicitly.
4. The starting-profile fit over step positions, segment Fo, edge values and offsets.
5. Then joint fits, population priors, step proposals and batch import on these names.
