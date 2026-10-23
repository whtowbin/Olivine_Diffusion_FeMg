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
| `dipra` (the default) | Dohmen and Chakraborty (2007) and their erratum (2007), as tabulated in DIPRA (Girona and Costa 2013). |
| `refit` | FORSTER's 2026 refit (this software, with the version in `forster_version`); the experiments it was fitted to are listed in [femg-refit2026.md](https://github.com/whtowbin/Olivine_Diffusion_FeMg/blob/master/docs/references/femg-refit2026.md). |
| `dfens_tamed`, `dfens_global` | Mutch et al. (2019) for the equations; Mutch et al. (2021) for their covariance. |
| Ni, Mn and Ca (Python) | DIPRA Table 1 (Girona and Costa 2013): Ni from Petry et al. (2004), Ca from Coogan et al. (2005), Mn as given there. `petry2004_ni`: Petry et al. (2004). |

The 201 ± 8 kJ/mol activation energy used by the Monte Carlo over the diffusion law
is from Dohmen, Becker and Chakraborty (2007), part I. For Fe-Mg diffusion above
about 1150 °C, where newer experiments find faster diffusion, see Oeser et al.
(2026).

If you fitted the starting profile, its nested-sampling fallback uses dynesty: cite
Speagle (2020) and the dynesty software record (Koposov et al.), and Skilling (2004,
2006) for nested sampling. The default implicit scheme is TR-BDF2 (Bank et al. 1985;
Hosea and Shampine 1996), whose first stage is the Crank-Nicolson method (Crank and
Nicolson 1947). For an overview of diffusion chronometry, see Costa et al. (2008),
Chakraborty (2008) and Costa et al. (2020).

### References

- Bank, R. E., Coughran, W. M., Fichtner, W., Grosse, E. H., Rose, D. J. and Smith, R. K. (1985). Transient simulation of silicon devices and circuits. *IEEE Transactions on Computer-Aided Design* 4(4), 436-451. [doi:10.1109/TCAD.1985.1270142](https://doi.org/10.1109/TCAD.1985.1270142)
- Chakraborty, S. (2008). Diffusion in solid silicates: a tool to track timescales of processes comes of age. *Annual Review of Earth and Planetary Sciences* 36, 153-190. [doi:10.1146/annurev.earth.36.031207.124125](https://doi.org/10.1146/annurev.earth.36.031207.124125)
- Coogan, L. A., Hain, A., Stahl, S. and Chakraborty, S. (2005). Experimental determination of the diffusion coefficient for calcium in olivine between 900 °C and 1500 °C. *Geochimica et Cosmochimica Acta* 69(14), 3683-3694. [doi:10.1016/j.gca.2005.03.002](https://doi.org/10.1016/j.gca.2005.03.002)
- Costa, F., Dohmen, R. and Chakraborty, S. (2008). Time scales of magmatic processes from modeling the zoning patterns of crystals. *Reviews in Mineralogy and Geochemistry* 69, 545-594. [doi:10.2138/rmg.2008.69.14](https://doi.org/10.2138/rmg.2008.69.14)
- Costa, F., Shea, T. and Ubide, T. (2020). Diffusion chronometry and the timescales of magmatic processes. *Nature Reviews Earth & Environment* 1(4), 201-214. [doi:10.1038/s43017-020-0038-x](https://doi.org/10.1038/s43017-020-0038-x)
- Crank, J. and Nicolson, P. (1947). A practical method for numerical evaluation of solutions of partial differential equations of the heat-conduction type. *Mathematical Proceedings of the Cambridge Philosophical Society* 43(1), 50-67. [doi:10.1017/S0305004100023197](https://doi.org/10.1017/S0305004100023197)
- Dohmen, R., Becker, H.-W. and Chakraborty, S. (2007). Fe-Mg diffusion in olivine I: experimental determination between 700 and 1,200 °C as a function of composition, crystal orientation and oxygen fugacity. *Physics and Chemistry of Minerals* 34(6), 389-407. [doi:10.1007/s00269-007-0157-7](https://doi.org/10.1007/s00269-007-0157-7)
- Dohmen, R. and Chakraborty, S. (2007). Fe-Mg diffusion in olivine II: point defect chemistry, change of diffusion mechanisms and a model for calculation of diffusion coefficients in natural olivine. *Physics and Chemistry of Minerals* 34(6), 409-430. [doi:10.1007/s00269-007-0158-6](https://doi.org/10.1007/s00269-007-0158-6)
- Dohmen, R. and Chakraborty, S. (2007). Erratum to: Fe-Mg diffusion in olivine II. *Physics and Chemistry of Minerals* 34(8), 597-598. [doi:10.1007/s00269-007-0185-3](https://doi.org/10.1007/s00269-007-0185-3)
- Girona, T. and Costa, F. (2013). DIPRA: A user-friendly program to model multi-element diffusion in olivine with applications to timescales of magmatic processes. *Geochemistry, Geophysics, Geosystems* 14(2), 422-431. [doi:10.1029/2012GC004427](https://doi.org/10.1029/2012GC004427)
- Hosea, M. E. and Shampine, L. F. (1996). Analysis and implementation of TR-BDF2. *Applied Numerical Mathematics* 20(1-2), 21-37. [doi:10.1016/0168-9274(95)00115-8](https://doi.org/10.1016/0168-9274(95)00115-8)
- Koposov, S. et al. dynesty v3.1.0 (software). Zenodo. [doi:10.5281/zenodo.21417233](https://doi.org/10.5281/zenodo.21417233)
- Mutch, E. J. F., Maclennan, J., Shorttle, O., Edmonds, M. and Rudge, J. F. (2019). Rapid transcrustal magma movement under Iceland. *Nature Geoscience* 12(7), 569-574. [doi:10.1038/s41561-019-0376-9](https://doi.org/10.1038/s41561-019-0376-9)
- Mutch, E. J. F., Maclennan, J., Shorttle, O., Rudge, J. F. and Neave, D. A. (2021). DFENS: Diffusion chronometry using finite elements and nested sampling. *Geochemistry, Geophysics, Geosystems* 22(4), e2020GC009303. [doi:10.1029/2020GC009303](https://doi.org/10.1029/2020GC009303)
- Oeser, M., Dohmen, R. and Weyer, S. (2026). Fe-Mg in olivine: simultaneous experimental determination of the diffusion-driven isotope fractionation along with tracer- and inter-diffusion coefficients. *Geochimica et Cosmochimica Acta* 413, 48-66. [doi:10.1016/j.gca.2025.09.036](https://doi.org/10.1016/j.gca.2025.09.036)
- Petry, C., Chakraborty, S. and Palme, H. (2004). Experimental determination of Ni diffusion coefficients in olivine and their dependence on temperature, composition, oxygen fugacity, and crystallographic orientation. *Geochimica et Cosmochimica Acta* 68(20), 4179-4188. [doi:10.1016/j.gca.2004.02.024](https://doi.org/10.1016/j.gca.2004.02.024)
- Skilling, J. (2004). Nested sampling. *AIP Conference Proceedings* 735, 395-405. [doi:10.1063/1.1835238](https://doi.org/10.1063/1.1835238)
- Skilling, J. (2006). Nested sampling for general Bayesian computation. *Bayesian Analysis* 1(4), 833-859. [doi:10.1214/06-BA127](https://doi.org/10.1214/06-BA127)
- Speagle, J. S. (2020). dynesty: a dynamic nested sampling package for estimating Bayesian posteriors and evidences. *Monthly Notices of the Royal Astronomical Society* 493(3), 3132-3158. [doi:10.1093/mnras/staa278](https://doi.org/10.1093/mnras/staa278)

The numerical schemes also draw on Crank (1975), *The Mathematics of Diffusion*, 2nd
ed., chapter 8; *Numerical Recipes* (Press et al. 2007), §15.6 and §20.2; Gerya
(2010), *Introduction to Numerical Geodynamic Modelling*; Langtangen and Linge
(2017), *Finite Difference Computing with PDEs*, chapter 3
([doi:10.1007/978-3-319-55456-3](https://doi.org/10.1007/978-3-319-55456-3)); the
University of Warwick CO906 lecture notes (2009); DIPRA's auxiliary material; and
Costa et al. (2008), Appendix I. Using a central difference for the term in the
gradient of D follows T. Girona (personal communication, 2020).

## License

FORSTER is released under the MIT license.
