# Changelog and citing

## Changelog

FORSTER is in development (version 0.1.0.dev0). Every change that affects results is
recorded, with which results moved and by how much, in
[CHANGELOG.md](https://github.com/whtowbin/Olivine_Diffusion_FeMg/blob/master/CHANGELOG.md)
on GitHub.

Changes users of earlier versions of this code should know about:

- **The default uncertainty method** compares the model with the measured points under
  a fitted noise model (white noise plus a smooth correlated misfit) instead of with a
  kriged interpolation of the data ([How the fit works](how-it-works.md)). Best fits
  can move relative to earlier fits, most where the misfit is long-wavelength, and
  ranges are wider where residuals are correlated. `uncertainty = legacy` reproduces
  the old rule.
- **The default numerical scheme** is implicit (TR-BDF2) on an automatically chosen
  grid, accurate to about 0.3% in the time; the earlier explicit scheme (`central`) is
  kept and is still the default with `uncertainty = legacy`, so old results reproduce
  exactly.
- **Starting-profile names**: segments (`segment_fo`, `segment_names`) and `edge1_fo`
  replace `edge_fo`, `core_fo`, `step_fo` and `rim_fo`; the old names are still read,
  with a warning, until the release.
- **New in this version**: fitting the starting profile, the Monte Carlo over T, P,
  fO2 and the law, four diffusion laws with their calibration uncertainties, growth
  stages, whole-crystal models (`centre_um`), the near-step and short-plateau checks,
  `forster import`, the desktop app, and joint fits of one crystal (Python).

## Citing

If you use FORSTER, please cite:

> Towbin, W. H., and Plank, T. (2026). *Journal of Petrology* 67(9), egag077.

and the diffusion law you used:

| Choice | Cite |
|---|---|
| `dipra` | Dohmen, R. and Chakraborty, S. (2007), *Physics and Chemistry of Minerals* 34, 409-430, and their erratum (34, 597-598), as tabulated in DIPRA (Girona, T. and Costa, F. (2013), *Geochemistry, Geophysics, Geosystems* 14, 422-431). |
| `refit` | FORSTER's 2026 refit (this software, with the FORSTER version from `forster_version`); the experiments it was fitted to are listed in [femg-refit2026.md](https://github.com/whtowbin/Olivine_Diffusion_FeMg/blob/master/docs/references/femg-refit2026.md). |
| `dfens_tamed`, `dfens_global` | Mutch, E. J. F. et al. (2019), *Nature Geoscience* 12, 569-574; the covariance from Mutch et al. (2021), DFENS, *Geochemistry, Geophysics, Geosystems* 22, e2020GC009303. |

If you fitted the starting profile, the nested-sampling fallback uses dynesty (Speagle
2020, *Monthly Notices of the Royal Astronomical Society* 493, 3132-3158).

!!! note "Citations being confirmed"
    Some of these references are still being checked against the papers before the
    first release (for example whether to cite the Dohmen and Chakraborty erratum
    separately, and the proposed dynesty and nested-sampling references). The open
    items are listed in
    [citations-to-confirm.md](https://github.com/whtowbin/Olivine_Diffusion_FeMg/blob/master/docs/references/citations-to-confirm.md)
    on GitHub; this page will be updated when they are settled.

## Licence

FORSTER is released under the MIT licence.
