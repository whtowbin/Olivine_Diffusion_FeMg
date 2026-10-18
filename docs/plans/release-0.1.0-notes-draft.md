# Draft: release notes for FORSTER 0.1.0, and the note for the paper repository

Status (2026-10-17): **draft for Henry to review.** Nothing here is published. Per
Henry, the round robin is not mentioned until that work is published. Once approved,
part A becomes the CHANGELOG's 0.1.0 section (the detailed development entries move
to a development log) and part B goes into the Xenolith_Melt_Vein_Modeling README.

---

## A. Release notes (CHANGELOG, "0.1.0 (date)")

FORSTER fits Fe-Mg diffusion times to forsterite profiles in zoned olivine, from a
command line, a desktop app or Python. This first release grew out of the scripts
behind Towbin and Plank (2026, J. Petrology 67(9), egag077). Documentation:
https://forster.readthedocs.io.

### If you used the earlier scripts

Results from FORSTER differ from those of the paper-era scripts for four reasons. None
changes the conclusions of Towbin and Plank (2026); the first is a correction, the
others are changes of method.

1. **A coding error in the diffusion model is fixed (2026-10-11).** The model solves
   dC/dt = D d²C/dx² + (dD/dx)(dC/dx). The second term, which matters only where the
   diffusivity changes across a step, was twice too large: two central differences were
   divided by 2Δx² instead of 4Δx². Fitted times were biased low where D varies
   strongly across the step. For the profiles of Towbin and Plank (2026), run with the
   published script, the same diffusion law and the same random draws:
   - melt-vein medians move by 2-5% (WHT01 17.9 to 17.6 days, GCB1 12.8 to 13.5 days,
     WHT06 5.6 to 5.4 days);
   - the exterior contact-overgrowth median moves from 5.9 to 7.5 days (+28%), because
     their Fo contrast is the largest (7-10 units);
   - the xenocryst median moves by +9% (0.79 to 0.86 days).

   Every shift lies well inside the published 5th-95th percentile ranges, which are
   dominated by the uncertainty of the pressure-temperature path (e.g. WHT01 melt
   veins 5.5-44.9 days before, 5.4-45.3 after; contact overgrowths 2.3-15.2 before,
   3.2-17.9 after). An ascent rate derived from the exterior times would come down by
   up to about 20%, also within its range. Per-profile numbers:
   `docs/plans/2026-10-11-engine-fix/`.
2. **The range on a fitted time is computed differently.** The paper-era scripts
   compared the model with a kriged profile on the model grid and drew the range where
   grid chi-squared rose by sqrt(2(N-1)) (until March 2026; the ranges of individual
   fits in Towbin and Plank 2026). FORSTER compares the model with the measured
   points, allows for smooth misfit between model and data (a correlated noise model),
   and draws 1σ and 2σ ranges from it; tests on profiles of known age give the stated
   coverage. Individual-fit ranges therefore change; the paper's reported ranges, from
   the Monte Carlo over the pressure-temperature path, are dominated by that path and
   are not affected in substance. The old rule (from March 2026: reduced chi-squared
   within 1 of its minimum) is kept as `uncertainty="legacy"` to reproduce earlier
   numbers. Details: `docs/references/fit-uncertainty.md`.
3. **Time is reported as row × time step** (the paper-era code used (row + 1) × time
   step), and the default solver is an implicit scheme with its grid chosen
   automatically; the explicit schemes remain available.
4. **New choices**: four named Fe-Mg diffusion laws (`dipra`, the default, as before;
   `refit`; `dfens_tamed`; `dfens_global`), fitting the starting profile with its
   uncertainty, a Monte Carlo over temperature, pressure, oxygen fugacity and the law,
   growth stages, and the parameter-table columns renamed for starting profiles of any
   number of segments (`segment_fo`, `edge1_fo`; the old `edge_fo`, `core_fo`,
   `step_fo` and `rim_fo` are still read, with a warning).

### Highlights
- Command line (`forster template`, `forster fit`, `forster import`) and a desktop app
  (`forster gui`) on the same project tables.
- Checks that warn when a range is probably too narrow (misfit concentrated near a
  step, a profile that ends before its plateau) or a prior conflicts with the data.
- Joint fits of several profiles of one crystal, and Ni, Mn and Ca fits (Python).

### Known limits
- Joint fits, Ni/Mn/Ca fits and some options (a changing rim value) are Python-only.
- The desktop installers are built in CI but unsigned.
- Citations for each diffusion law and method: see the documentation (being confirmed).

---

## B. Note for the Xenolith_Melt_Vein_Modeling README

> **Note (October 2026).** The code in this repository reproduces Towbin and Plank
> (2026) as published. Since then a coding error was found in one term of the diffusion
> model (the term that matters where the diffusivity changes across a step was twice too
> large). Rerunning the published script with the fix, the same inputs and the same
> random draws moves the melt-vein median timescales by 2-5% (WHT01 17.9 to 17.6 days,
> GCB1 12.8 to 13.5, WHT06 5.6 to 5.4), the exterior contact-overgrowth median from 5.9
> to 7.5 days, and the xenocrysts by +9%. All shifts lie well within the published
> 5th-95th percentile ranges, which are dominated by the pressure-temperature path, and
> the conclusions are unchanged. The maintained code, with this fix, is FORSTER:
> https://github.com/whtowbin/Olivine_Diffusion_FeMg (per-profile numbers in
> `docs/plans/2026-10-11-engine-fix/`).

---

## Questions for Henry
1. Is part A's framing right (one correction, three changes of method; conclusions
   unchanged)?
2. Should the ascent-rate sentence stay, given that the paper's ascent rates (0.05-0.15
   m/s) were not recomputed here?
3. Part B: add it to the Xenolith_Melt_Vein_Modeling README as is? (That repository is
   separate; this session has not touched it.)
