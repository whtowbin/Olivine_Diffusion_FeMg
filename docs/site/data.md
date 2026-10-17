# Preparing data

This page covers getting measured profiles into FORSTER and, before that, collecting
profiles that FORSTER can fit well.

## The profiles table

FORSTER reads measured points from a table with one row per analysis point
(`profiles.csv`, or an Excel sheet):

| Column | Unit | Meaning |
|---|---|---|
| `profile` | | Profile name. The `profile` column of `parameters.csv` refers to it. Many profiles can share one table. |
| `distance_um` | µm | Distance of the analysis from the crystal edge. |
| `fo` | fraction | Forsterite content, Mg/(Mg + Fe) in moles, as a fraction: 0.895, not 89.5. |
| `Marked_bad` | | Optional. `bad` leaves the point out of the fits (the desktop app's masked points). |

```text
profile,distance_um,fo
ol12 prof 1,0.0,0.8512
ol12 prof 1,2.5,0.8547
ol12 prof 1,5.0,0.8601
...
```

Points are sorted by distance when read, so their order does not matter. A profile
needs at least 3 points (a name with fewer is skipped with a warning). Extra columns
are ignored, so you can keep oxide data, totals and notes in the same table.

**Distance from which edge?** Distances are measured from **edge 1**, the end of the
profile where the model holds Fo (usually the crystal rim). If your traverse runs from
the core outward, flip it (the app's Data tab and the import step both do this).
Points do not have to start at zero: the model domain (`domain_start_um`,
`domain_end_um`) says which part of the profile is modelled.

## Importing other spreadsheet layouts

A spreadsheet straight from the microprobe has its own column names and units.
`forster import` converts it in two steps, so nothing is converted without you
checking it first.

**Step 1: propose a mapping.**

```bash
forster import probe_session.xlsx
```

FORSTER guesses what each column holds from its header and values: the profile name,
the distance (µm or mm), Fo (fraction or percent), each oxide or element (wt% or ppm),
the analytical total, stage coordinates, or exclusion columns (`Marked_bad`,
`Ignore`, ...). It writes the guess to `probe_session.import.json` and stops. Open
that file, check every column and correct anything it got wrong.

**Step 2: apply the confirmed mapping.**

```bash
forster import probe_session.xlsx --template probe_session.import.json
```

writes `probe_session_profiles.csv` in the layout above. On the way:

- **Fo from oxides.** Without an Fo column, Fo is computed in moles from MgO and FeO
  in wt%: Fo = (MgO/40.3044) / (MgO/40.3044 + FeO/71.8444), with all iron as Fe²⁺
  (FeO is total iron). Mg and Fe as elements (wt% or ppm) work the same way.
- **Flagged totals.** Points whose analytical total lies outside 98.5-101.5 wt% are
  flagged in a `total_ok` column and listed, **not** dropped: a low total is a reason
  to look at a point, not proof that it is bad. Change the window with
  `--total-window 98 102`.
- **Excluded rows.** Rows marked in an exclusion column (`bad`, `yes`, `x`, `1`, ...)
  are left out.

The confirmed `.import.json` is an **import template**: later files with the same
columns import with it directly. A file with columns the template does not know, or
without ones it needs, stops with a message rather than being imported wrongly.
(An import template is a column mapping; it is unrelated to the `forster template`
command, which writes the parameter tables.)

| Option | Does |
|---|---|
| `--sheet NAME` | Excel sheet to read (default: the first). |
| `--save-template FILE` | Where to write the proposed mapping (default `FILE.import.json`). |
| `--out FILE` | Profiles table to write (default `FILE_profiles.csv`). |
| `--total-window LOW HIGH` | Window for flagging totals (default 98.5 101.5). |
| `--force` | Overwrite an existing proposed mapping. |

The desktop app's **Import profiles...** dialog uses the same mapping and reads and
writes the same import templates ([The desktop app](app.md#importing-profiles)). From
Python: `forster.imports.guess`, `ImportTemplate` and `apply` ([Python
API](api.md#importing-spreadsheets-forsterimports)).

Parameter and profile tables in two older layouts (the tables of Towbin and Plank
2026, and the lab layout with `Profile_Name`, `Distance µm`, `Fo#` in percent) are
read too.

## Collecting profiles

This advice comes from tests on synthetic profiles of known age and on the published
profiles of Towbin and Plank (2026). Two terms first:

- The **diffusion length** l = √(D t) is roughly how far a step has spread: on a
  fitted profile it is about half the width of the smoothed step. D is the
  diffusivity (how fast Fe and Mg exchange, in m²/s) and t the time. FORSTER reports
  the fitted time, so l at your conditions follows from it; `core_end_diffusion_lengths`
  in `summary.csv` gives distances in units of l.
- The **plateau** (or core) is the flat part of the profile beyond the smoothed
  step, which diffusion has not yet reached.

| Advice | Why |
|---|---|
| **Measure until the profile has been flat for at least one more diffusion length past where it flattens.** More is better. | FORSTER ends the model at the last measured point and holds the core Fo there. On real profiles the plateau starts about 2 diffusion lengths past the step. Data that stopped where it starts moved the time by -15% to +22%; data that stopped on the slope by -63% or more. With 1 diffusion length of plateau or more, every profile tested stayed within 4%. FORSTER warns when the data end within 3 diffusion lengths of the step. |
| If the profile ends at the centre of a small crystal, treat that warning with care. | There the data cannot go further. Diffusion from both rims meets at the centre, so the centre's Fo falls once diffusion arrives, which a fixed core Fo does not allow. Model the whole crystal instead (`centre_um`, [Setting up a run](setup.md#whole-crystals-centre_um)). |
| **Do not worry about plateaus being longer on some profiles than others.** | Once the data reach the plateau, its length does not move the time (within 4% on six real profiles with 1 to 21 diffusion lengths of plateau). |
| **Never trim plateau points** to tighten a range or remove a warning. | Trimming does not move the time; it only hides misfit near the step. On one paper profile the near-step check went from a clear warning with the full plateau to silence with 2 diffusion lengths, while the range stayed too narrow. |
| **Set the model domain to include the whole measured plateau.** | Points beyond the domain's end are not fitted (`points_outside_domain` counts them). A longer plateau makes the near-step check more reliable. |
| **Put at least about 13 analyses within two diffusion lengths of each step.** | The near-step check (does the misfit pile up near the step?) depends on the points there: with 7-12 it catches only 10-60% of such misfit, with 13 or more 87-97% (`near_step_points` in `summary.csv`). |
| **Aim for 25 or more points across the smoothed zone and its plateaus.** | On 15-17-point profiles the near-step check catches only 14-37% of strong near-step misfit (94% on 30 points), and a smooth misfit can mimic the profile itself. |
| **Record your analytical precision.** | Compare it with the fitted scatter (`white_sigma_fo`): much larger means unmodelled scatter or outliers. |
| **Note analyses that may be bad** (cracks, inclusions, poor totals, beam overlap with the rim or the matrix). | Remove a point only for a known fault, then refit. Outliers inflate the fitted scatter, so they are hard to spot from the diagnostics. Rim analyses that are noisier make the 1σ range too narrow (it then covers about 63% instead of 68%). |
| **Measure the orientation** (EBSD, or optically). | Fe-Mg diffusion is about six times faster along [001] than along [100] or [010], so the same profile gives times up to six times apart depending on its direction ([Setting up a run](setup.md#orientation)). |

![Data that stop on the slope](images/short-plateau.png)

*Why the data must reach the plateau. A published profile (WHT06_ol43, Towbin and
Plank 2026) fitted with every point (blue), cut one diffusion length into the plateau
(green: the time moves 4%), and cut on the slope (pink). The cut model ends at its last
point and holds the core Fo there, where the measured profile is still about 1 mol%
lower; that steepens the modelled step and the time falls 63%.*

The evidence behind each item is in
[practical-advice.md](https://github.com/whtowbin/Olivine_Diffusion_FeMg/blob/master/docs/guide/practical-advice.md)
and
[noise-model-findings.md](https://github.com/whtowbin/Olivine_Diffusion_FeMg/blob/master/docs/references/noise-model-findings.md)
on GitHub.
