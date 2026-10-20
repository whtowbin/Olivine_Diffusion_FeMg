# Changelog and citing

## Changelog

FORSTER is in development (version 0.1.0.dev0). Every change that affects results is
recorded in
[CHANGELOG.md](https://github.com/whtowbin/Olivine_Diffusion_FeMg/blob/master/CHANGELOG.md)
on GitHub, with the results that moved and by how much.

If you used earlier versions of this code, these are the changes to know about:

- **The default uncertainty method** now compares the model with the measured points
  themselves, under a fitted noise model (white noise plus a smooth correlated
  misfit). Earlier versions compared it with a kriged interpolation of the data
  ([How the fit works](how-it-works.md)). Best fits can move relative to earlier fits,
  most where the misfit has a long wavelength, and ranges are wider where the
  residuals are correlated. `uncertainty = legacy` reproduces the old rule.
- **The default numerical scheme** is now implicit (TR-BDF2) on an automatically
  chosen grid, and the time is accurate to about 0.3%. The earlier explicit scheme
  (`central`) is still available and is the default with `uncertainty = legacy`, so
  old results reproduce exactly.
- **Starting-profile names** have changed. Segments (`segment_fo`, `segment_names`)
  and `edge1_fo` replace `edge_fo`, `core_fo`, `step_fo` and `rim_fo`. Until the
  release, FORSTER still reads the old names and gives a warning.
- **New in this version**: fitting the starting profile, the Monte Carlo over T, P,
  fO2 and the diffusion law, four diffusion laws with their calibration uncertainties,
  growth stages, whole-crystal models (`centre_um`), the near-step and short-plateau
  checks, `forster import`, the desktop app, and joint fits of several profiles from
  one crystal (in Python).

## Citing

If you use FORSTER, please cite:

> Towbin, W. H., and Plank, T. (2026). *Journal of Petrology* 67(9), egag077.

and the source of the diffusion law you used:

| Choice | Cite |
|---|---|
| `dipra` | Dohmen, R. and Chakraborty, S. (2007), *Physics and Chemistry of Minerals* 34, 409-430, and their erratum (34, 597-598), as tabulated in DIPRA (Girona, T. and Costa, F. (2013), *Geochemistry, Geophysics, Geosystems* 14, 422-431). |
| `refit` | FORSTER's 2026 refit (this software, with the FORSTER version from `forster_version`); the experiments it was fitted to are listed in [femg-refit2026.md](https://github.com/whtowbin/Olivine_Diffusion_FeMg/blob/master/docs/references/femg-refit2026.md). |
| `dfens_tamed`, `dfens_global` | Mutch, E. J. F. et al. (2019), *Nature Geoscience* 12, 569-574; the covariance from Mutch et al. (2021), DFENS, *Geochemistry, Geophysics, Geosystems* 22, e2020GC009303. |

If you fitted the starting profile, note that its nested-sampling fallback uses dynesty (Speagle
2020, *Monthly Notices of the Royal Astronomical Society* 493, 3132-3158).

!!! note "Citations being confirmed"
    Some of these references are still being checked against the papers before the
    first release, for example whether to cite the Dohmen and Chakraborty erratum
    separately, and the proposed dynesty and nested-sampling references. The open
    items are listed in
    [citations-to-confirm.md](https://github.com/whtowbin/Olivine_Diffusion_FeMg/blob/master/docs/references/citations-to-confirm.md)
    on GitHub. This page will be updated when they are settled.

## License

FORSTER is released under the MIT license.
