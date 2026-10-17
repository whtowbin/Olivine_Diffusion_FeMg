---
name: forster-schematics
description: Use when changing, re-rendering or adding schematic (concept, setup or workflow) figures for the FORSTER methods paper in docs/figures/schematics/ (the zoned-olivine concept, workflow, growth-stages and joint-fit figures), or when using the global schematic-figure skill in this repo. Not for matplotlib data plots (figure-craft covers those).
---

# FORSTER schematics: how to change them and make more

The figures were drawn with the global `schematic-figure` skill
(`~/.claude/skills/schematic-figure`, code-drawn SVG/PDF rendered in headless Chromium).
Read its SKILL.md before a new figure; for a change to an existing one, this file is enough.

## Layout (docs/figures/schematics/)
- `FORSTER_FACTS.md`: the shared research note all four figures were drawn from.
- `<figure>/docs/`: `brief.md` (the takeaway), `spec.md` (concepts and the layout),
  `facts.md` (the source of every on-screen string; the final gate fails on a string
  missing here), `caption.md`, `self-critique.md` (errors fixed, and why each gate WARN
  was kept).
- `<figure>/figure/figure.js` draws it; `figure/data/*.py` computes the curves into JSON;
  `figure/out/` holds the PNG (600 dpi), PDF (vector, live text) and SVG.
- `tools/render.sh <figure> [dpi]`: PDF and SVG, a PNG that is not tiled, the tiling
  check, then the gate. A rerender takes seconds.

## Regenerate the data (all byte-identical to the committed JSON, checked 2026-10-16)
- workflow, olivine-concept, joint-fit: from the figure folder,
  `uv run --no-project --with numpy python figure/data/<script>.py`
  (joint-fit: `make_crystal.py`, then `make_profiles.py`, which also runs the 12-seed sweep).
- growth-stages uses FORSTER itself: from the repo root,
  `uv run --frozen python docs/figures/schematics/growth-stages/figure/data/compute_curves.py docs/figures/schematics/growth-stages/figure/data/curves.json`.
  If the model changes, this figure's numbers change with it; rerun and recheck the caption.

## Lessons from making them
1. **Look at every PNG.** Under Brave, the skill's `render.mjs` writes PNGs tiled (the
   top-left repeated) while printing success; the PDF and SVG are fine. `render.sh`
   avoids it (`png_settled.mjs` waits for the page to settle) and runs `tiled_check.py`.
   A green gate checks the layout, not that the figure is right.
2. **Facts before drawing.** Start from `FORSTER_FACTS.md` and the repo docs it cites;
   write `facts.md` before drawing a crystal. What the sources do not give stays generic:
   crystal habit, face indices, the section plane, real values (no ticks on schematic plots).
3. **Compute every curve**, seeded, and prefer FORSTER's own model (growth-stages calls
   `fit_profile`). Hand-drawn shapes are where physics errors hide.
4. **Measure a figure's claim before drawing it.** The joint-fit takeaway "fitting
   together pins the time better" held only modestly in a toy model (joint range a median
   14% narrower than the better single profile, wider in 2 of 12 seeds), so its caption
   says less. Growth-stages first claimed "the fit alone cannot tell the two histories
   apart" because both whitened χ² minima are equally low; each fit's noise model absorbs
   its own misfit. The residuals do tell: the single-stage fit leaves 4x the analytical
   noise at the zone boundary and FORSTER's near-step check flags it (AIC -22; the staged
   fit gives +1.3). `compute_curves.py` now computes this (`near_step` in curves.json) and
   panel d labels it. Equal χ² across models with different noise fits is not evidence
   they fit equally well.
5. **Physics checks that caught real errors:** a right-handed axis frame (a × b = c); a
   traverse on a section must lie in its plane; a rim highlight read as an extra zone;
   the same zoning sense in every panel; D along c about 6 × D along a; the D of a
   profile from cos² of its angles; width grows as √(D t).
6. **Gate WARNs expected here:** "text is small" (5.5-6 pt at 183 mm is inside the
   journal 5-7 pt; the gate's 11.3 pt calibration conflicts with the skill's own rule),
   word density on panels with data, and short labels that state a result. Record why
   in `self-critique.md`; fix any FAIL.
7. **Reusable code:** `inset()` (offset a convex outline inward for zones) and `rayHit()`
   (start a traverse exactly on the rim) in `olivine-concept/figure/figure.js`; a
   computed Fo map placed as an image (`joint-fit/figure/data/make_crystal.py`); axes
   drawn by hand from JSON (the library's `L.plot` is a schematic inset only). The
   library has no mineral-section part.
8. **Keep the set consistent.** All four use 183 mm width, 5.5-6 pt Arial labels and no
   numbers on schematic axes. Colour roles: in the workflow figure orange is the starting
   profile, blue the model and fit, green the conditions and law; in the joint-fit figure
   black is shared, blue and vermillion are per profile. **One Fo colour scale for all:**
   light amber `#F5D27F` = low Fo, `#B9B24F`, olive `#56782B` = high Fo, so Fe-rich olivine
   is the bright one, as in backscattered-electron images (unified 2026-10-16;
   growth-stages `FO_STOPS`, joint-fit `RAMP` in make_crystal.py, olivine-concept `COL`).
9. **Cost:** a new figure at the skill's `depth=standard` took 15-30 minutes and about
   200,000-270,000 tokens in a subagent; `full` depth was not needed (the skill's own
   benchmark found its extra drafts and critic did not raise the score). A change to an
   existing figure is an edit to `figure.js` and a rerender.
10. Never draw round robin data here: the repo is public (see `forster-safe-changes`).
