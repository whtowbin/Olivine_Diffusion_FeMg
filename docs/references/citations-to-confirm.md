# Citations to confirm before the 0.1.0 release

Every source FORSTER cites in its docs, README or code, where the right reference is
not yet certain. Each item says what the code does, where, my current best guess,
and what I need from you. Answer inline (write under **Answer**) or tell me in chat;
I will then update the README "Citing" section, the docstrings and
`docs/plans/2026-10-11-restructure.md` ("Science and citations").

Collected 2026-10-15 from the restructure plan's citation table and the open items
in `docs/references/numerical-methods.md`, plus the libraries added this week.

## A. Diffusion equations (how fast each element diffuses)

A diffusion equation here gives the diffusivity D (m²/s) from temperature, pressure,
oxygen fugacity, olivine composition and direction in the crystal. The default
Fe-Mg equation is the one the DIPRA program uses; the fitted time scales as 1/D, so
the source matters for every result.

1. **Default Fe-Mg equation** (`dohmen_chakraborty2007`, called "Classic DIPRA").
   - Code: D[001] = 10^-9.21 (fO2 / 10^-7 Pa)^(1/6) 10^(3 (0.9 - X_Fo)) exp(-(201 kJ/mol + (P - 10^5 Pa) 7 cm³/mol) / RT); [100] and [010] six times slower. `src/forster/diffusivity.py` (`DOHMEN_CHAKRABORTY_2007`), `engine.D_Fo`.
   - Best guess: Dohmen, R. and Chakraborty, S. (2007), Fe-Mg diffusion in olivine II: point defect chemistry, change of diffusion mechanisms and a model for calculation of diffusion coefficients in natural olivine, *Phys. Chem. Minerals* 34, 409-430, with their erratum (2007, *PCM* 34, 597-598). The numbers match DIPRA's Table 1 (Girona and Costa 2013) exactly.
   - Need: confirm, and whether to cite the erratum separately.
   - **Answer:**

2. **The ± 8 kJ/mol on the activation energy** (201 ± 8 kJ/mol, used by the paper-era Monte Carlo and quoted in docs).
   - Code comment says "Dohmen et al. 2007". Two candidates: Dohmen, Becker and Chakraborty (2007), Fe-Mg diffusion in olivine I: experimental determination between 700 and 1,200 °C as a function of composition, crystal orientation and oxygen fugacity, *PCM* 34, 389-407 (the experiments, where the uncertainty would come from); or Dohmen and Chakraborty (2007) part II (item 1).
   - Need: which paper the ± 8 comes from (part I, I expect).
   - **Answer:**

3. **Ni equation** (`girona_costa2013_ni`, the default for Ni fits).
   - Code: 3.84×10^-9 (fO2 / 10^-6 Pa)^(1/4.25) 10^(1.5 (0.9 - X_Fo)) exp(-(220 kJ/mol + PV)/RT), anisotropy as Fe-Mg. Identical to DIPRA Table 1, which pairs Petry's pre-factor with the activation energy of a different fit (`petry2004_ni` is Petry's own consistent pair; CHANGELOG 2026-10-13).
   - Best guess: Petry, C., Chakraborty, S. and Palme, H. (2004), Experimental determination of Ni diffusion coefficients in olivine and their dependence on temperature, composition, oxygen fugacity, and crystallographic orientation, *Geochim. Cosmochim. Acta* 68, 4179-4188.
   - Need: confirm; and whether the README should say outright that DIPRA's Ni pairing is inconsistent (it does now, in one sentence).
   - **Answer:**

4. **Ca equation** (`girona_costa2013_ca`).
   - Code: separate D along [100], [010], [001] (pre-factors 16.59, 34.67, 95.49 ×10^-12 m²/s; E = 193, 201, 207 kJ/mol), (fO2 / 10^-7 Pa)^0.3. Identical to DIPRA Table 1.
   - Best guess: Coogan, L. A., Hain, A., Stahl, S. and Chakraborty, S. (2005), Experimental determination of the diffusion coefficient for calcium in olivine between 900 °C and 1500 °C, *Geochim. Cosmochim. Acta* 69, 3683-3694.
   - Need: confirm.
   - **Answer:**

5. **Mn equation** (`girona_costa2013_mn`): the Fe-Mg form with E = 201 kJ/mol, as in DIPRA Table 1.
   - DIPRA cites Dohmen and Chakraborty (2007a, b), Chakraborty (2010), Dohmen et al. (2007) and Holzapfel et al. (2007) for the composition dependence, but no single primary Mn source.
   - Need: which paper to cite for Mn (or "cite DIPRA Table 1 only").
   - **Answer:**

## B. Fitting method

6. **The legacy ± 1σ rule**: the times where reduced chi-squared is within 1 of its minimum (`uncertainty = legacy`, the round robin and paper-era rule; `engine.Best_fit_Chi2`).
   - It is not DIPRA's rule (DIPRA counts model points outside the data error bars; Girona and Costa 2013, Auxiliary Material 2). The code comments cite Andrae, Schulze-Hartung and Melchior (2010), Dos and don'ts of reduced chi-squared, arXiv:1012.3754, for the pitfalls, not for the rule.
   - Need: where the rule came from (a paper, a textbook such as Bevington and Robinson, or "our own choice"). The new default method (correlated noise, `docs/references/likelihood-method.md`) does not depend on this answer; the release notes do, because they say which method produced each published range.
   - **Answer:**

7. **Review paper for diffusion chronometry**, cited once in the README as the method overview.
   - Best guess: Costa, F., Dohmen, R. and Chakraborty, S. (2008), Time scales of magmatic processes from modeling the zoning patterns of crystals, *Rev. Mineral. Geochem.* 69, 545-594.
   - Need: confirm, or name the review you prefer (and the page range if you have it to hand).
   - **Answer:**

8. **PyKrige** (kriging of profiles; legacy method only since 2026-10-12).
   - Need: your preferred citation. PyKrige asks for: Murphy, B., Yurchak, R. and Müller, S. (2022), GeoStat-Framework/PyKrige, Zenodo, doi:10.5281/zenodo.3738604 (version DOI). I have not checked the version-specific DOI.
   - **Answer:**

## C. Numerical scheme sources (`docs/references/numerical-methods.md`)

9. **Warwick CO906 lecture notes (2009), chapter 3, "The Diffusion Equation".** The PDF names no author.
   - Best guess: Colm Connaughton.
   - Need: confirm, or say to cite it as "University of Warwick, CO906 lecture notes".
   - **Answer:**

10. **Langtangen and Linge (2017)**, *Finite Difference Computing with PDEs*, Springer, open access.
    - Need: confirm the chapter we lean on (ch. 3, diffusion) and the DOI 10.1007/978-3-319-55456-3.
    - **Answer:**

11. **Which of these were actually used** when writing the original scheme (all in your Zotero library): Crank (1975) *The Mathematics of Diffusion* (and which chapter); Press et al. *Numerical Recipes* ch. 19/20; Demouchy and Thoraval (2014); Gerya *Numerical Geodynamic Modelling* (2010 or 2019 edition); Ford Versypt and Braatz (2014).
    - Need: tick the ones to cite; the rest stay out.
    - **Answer:**

## D. New this week: proposed citations (not yet in the docs)

These are tools the new features rely on. I believe the references below are right,
but have not checked them against the papers.

12. **Nested sampling and dynesty** (`forster.sampling`, the fallback when importance sampling is not enough).
    - Proposed: Skilling, J. (2006), Nested sampling for general Bayesian computation, *Bayesian Analysis* 1, 833-859; Speagle, J. S. (2020), dynesty: a dynamic nested sampling package for estimating Bayesian posteriors and evidences, *MNRAS* 493, 3132-3158.
    - Need: OK to cite both in the README and the `forster.sampling` docstring.
    - **Answer:**

13. **TR-BDF2** (the default time stepper, `forster.implicit`): already cited as Bank et al. (1985) and Hosea and Shampine (1996).
    - Proposed full references: Bank, R. E., Coughran, W. M., Fichtner, W., Grosse, E. H., Rose, D. J. and Smith, R. K. (1985), Transient simulation of silicon devices and circuits, *IEEE Trans. Electron Devices* 32, 1992-2007; Hosea, M. E. and Shampine, L. F. (1996), Analysis and implementation of TR-BDF2, *Appl. Numer. Math.* 20, 21-37.
    - Need: OK to add them in full.
    - **Answer:**

14. **Mutch et al. (2019, 2021)** (the DFENS diffusion equations and their covariance) and **Oeser et al. (2026)** are already cited with full references by the refit work (`docs/references/diffusivity-literature.md`); nothing needed unless you want them in the README's Citing section too.
    - **Answer:**
