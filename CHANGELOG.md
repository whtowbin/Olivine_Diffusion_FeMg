# Changelog

## 0.1.0 (unreleased)

FORSTER fits Fe-Mg diffusion times to forsterite profiles in zoned olivine, from a
command line, a desktop app or Python. This first release grew out of the scripts
behind Towbin and Plank (2026, J. Petrology 67(9), egag077). Documentation:
https://forster.readthedocs.io. The detailed record of every change made during
development, with the measurements behind each, is in
[docs/development-log.md](docs/development-log.md).

### If you used the earlier scripts

<!-- HENRY: question 1 from the release-notes draft: is this framing right (one correction, three changes of method; conclusions unchanged)? -->
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
   3.2-17.9 after).
   <!-- HENRY: question 2 from the draft: should the next sentence (ascent rate) stay, given that the paper's ascent rates (0.05-0.15 m/s) were not recomputed here? -->
   An ascent rate derived from the exterior times would come down by
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
- Import a whole folder of per-profile spreadsheets at once, from the command line,
  Python or the app, with one column mapping.
- Desktop app installers, each checked by a self-test that starts the built app before
  it is published.
- Checks that warn when a range is probably too narrow (misfit concentrated near a
  step, a profile that ends before its plateau) or a prior conflicts with the data.
- Joint fits of several profiles of one crystal, and Ni, Mn and Ca fits (Python).
- Automatic grids estimate the time in the fit's own scheme, which is faster; fitted
  times move by at most 0.44%.

### Other changes to know about

- User-facing text uses American spelling. Warning text changed, so code that filters
  warnings by their text needs the new spellings. Column names are unchanged.
- The app refuses runs set to the legacy (kriging) uncertainty method; use the command
  line or Python for those.

### Known limits

- Joint fits, Ni/Mn/Ca fits and some options (a changing rim value) are Python-only.
- The desktop installers are built in CI but unsigned.
<!-- HENRY: citations are still being confirmed (docs/references/citations-to-confirm.md); the citation line below stays as is until they are settled. -->
- Citations for each diffusion law and method: see the documentation (being confirmed).

### Before release

- [ ] Paper addendum, to return to later: the cross-term fix changes the timescales in Towbin and Plank (2026, J. Petrology 67(9), egag077). Melt-vein medians move by 2-5% (WHT01 17.9 to 17.6 days, GCB1 12.8 to 13.5, WHT06 5.6 to 5.4); the exterior contact-overgrowth median moves from 5.9 to 7.5 days (+28%); xenocrysts +9%. Check the ascent rates derived from these times. Numbers per profile: `docs/plans/2026-10-11-engine-fix/`.

- [ ] Uncertainty method: state in the release notes which version (pre-March 2026, legacy, correlated) produced each published or submitted range, and refit the round robin with "correlated". The round robin diffusivity Monte Carlo (`Round_Robin_Diffusion_Modeling Part 2.py`) has separate problems: temperature in degrees C passed where kelvin is expected, 33.33/66.66% quantiles labelled +-1 sigma, the 1200 C samples used for the 1102 C samples, and the activation energy varied with the pre-factor held fixed; corrected for the first two problems, its "+-1 sigma" time multipliers become wider. See `docs/references/fit-uncertainty.md`.

- [ ] Come back to the cross-term fix: state the size of the bias for older results in the release notes, refit the round robin profiles (revisit the starting profiles of the round robin profiles whose fits get worse with the fixed engine), correct any submitted round robin results, and add the note on the paper results to the Xenolith_Melt_Vein_Modeling README.

- [x] Move round robin results out of the public tree (Henry, 2026-10-16: fine in the repo for now, private for the release; done 2026-10-17). Round robin results (fits, times, ranges, noise numbers, per-profile rows, figures and plots of round robin profiles) are out of the current public files: the originals of every public file that held them are in the private submodule under `outputs/public_tree_moved/<original path>`, and the scripts that read the round robin data write those rows only to the submodule. Synthetic profiles built on round robin geometries keep their data under generic names ("29-point profile, 10 um spacing"); femg_refit's law predictions at the round robin experimental conditions are kept, labelled RR1200 and RR1102. Old git commits keep the earlier versions; history is not rewritten (Henry, 2026-10-17). `tests/test_public_tree.py` checks that no tracked file outside its allow-list of code paths names a round robin sample. Follow-up the same day: 36 figures in `analyses/xenolith_archive/` named after paper profiles were really one round robin fit (saved under paper names by a stale variable in the old script) and are removed; the likelihood-validation `trials.csv` now gives times relative to the true time (its true time was a round robin fit); the grid-choice scripts send round robin rows to the submodule; tests use round-number truth times.

- [x] User guide: the documentation site (`docs/site/`, MkDocs on Read the Docs) is the release user guide (collecting profiles, setting up a run, reading the results and warnings). The separate mechanisms guide (the likelihood, whitening, the near-step check, the starting-profile fit, with derivations and citations; `docs/references/likelihood-method.md` holds most of it now) follows after 0.1.0.
