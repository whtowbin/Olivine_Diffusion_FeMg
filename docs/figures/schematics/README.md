# Schematic figures for the FORSTER methods paper

Drafts made 2026-10-16 with the `schematic-figure` skill (github.com/lgwrightlab/schematic-figure).
All curves are computed and illustrative; each caption is in `<figure>/docs/caption.md`.

| Folder | Figure |
|---|---|
| `workflow/` | From inputs to a time distribution: starting profile, forward model, misfit, starting-profile uncertainty, Monte Carlo |
| `olivine-concept/` | A zoned olivine, a step relaxing faster along c than along a, and how the profile's direction sets D |
| `growth-stages/` | A crystal growing in two stages, computed with FORSTER's own model, and the misfit against time |
| `growth-stages-v2/` | Three zones from two growth stages, with the zone geometry of joint-fit profile 1; here the near-step check does not flag the wrong history, but the residuals do |
| `joint-fit/` | Two profiles on one crystal: what is shared and what is per profile (the joint fit is being built) |

Rerender: `tools/render.sh <folder> [dpi]`. How to change them, regenerate their data and
what to watch for: `.claude/skills/forster-schematics/SKILL.md`.
