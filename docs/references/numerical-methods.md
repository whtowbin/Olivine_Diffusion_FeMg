# Sources for the numerical diffusion scheme

Notes on where the finite-difference method in `src/forster/engine.py`
came from. Started 2026-10-11 from Henry's recollection; entries marked
[confirm] still need checking.

## What the engine does

Explicit finite differences, forward Euler in time and central differences in
space (FTCS), with a composition-dependent diffusivity:

$$C^{n+1} = C^n + \frac{\Delta t}{\Delta x^2}\Big[D\,(C_{i+1} - 2C_i + C_{i-1}) + \tfrac14 (D_{i+1} - D_{i-1})(C_{i+1} - C_{i-1})\Big]$$

The matrix form of one step is $C^{n+1} = (I + \Delta t\,A)\,C^n$, where $A$ is
tridiagonal. The code never builds $A$: `np.convolve` with the 3-point kernels
`[1, -2, 1]` and `[1, 0, -1]` (`diffusion_kernel`, `diffusion_step`) computes
the same product. There is no implicit or Crank-Nicolson solver in the engine
and none in the git history (no `diags`, `spsolve` or `solve_banded`); those
schemes appear in the sources below but were not adopted.

Stability: FTCS is stable only for $D\,\Delta t/\Delta x^2 \le \tfrac12$.
`engine.py` states this in a comment, but nothing enforces it.

## Sources used early on (Henry's recollection)

1. **Warwick CO906 lecture notes (2009), Chapter 3: "The Diffusion Equation".**
   University of Warwick, Complexity Science DTC, module CO906.
   <https://warwick.ac.uk/fac/cross_fac/complexity/study/msc_and_phd/co906/co906online/lecturenotes_2009/chap3.pdf>
   Author is not printed in the PDF [confirm; probably Colm Connaughton, who
   taught the Warwick complexity computing modules].
   Relevant parts:
   - §3.1: the FTCS scheme written as $dv/dt = Av$ with tridiagonal $A$ (eq. 3.4)
     and the forward Euler step $v_{j+1} = v_j + hAv_j$ (eq. 3.5). This is the
     matrix-multiplication view of the engine's step.
   - §3.2.2 and §3.2.3: Dirichlet and Neumann boundary conditions in FTCS.
   - §3.3: consistency, stability and von Neumann analysis of FTCS (source of
     the $D\Delta t/\Delta x^2 \le \tfrac12$ limit).
   - §3.4 and §3.5: Crank-Nicolson and its stability.
   - Constant $D$ only; the cross term for variable $D$ is not covered.

2. **Langtangen, H. P. and Linge, S., "Finite difference methods for diffusion processes"**
   (lecture notes, *Lectures on numerical methods for PDEs*, 2015/2016).
   <https://hplgit.github.io/num-methods-for-PDEs/doc/pub/diffu/html/._diffu001.html#diffu:pde1:CN>
   Relevant parts: forward Euler, backward Euler, "Sparse matrix implementation"
   (building the tridiagonal system with `scipy.sparse`), Crank-Nicolson, the
   θ rule, amplification-factor analysis of each scheme, and "Diffusion in
   heterogeneous media" (variable coefficients).
   Citable published version: Langtangen, H. P. and Linge, S. (2017),
   *Finite Difference Computing with PDEs: A Modern Software Approach*,
   Texts in Computational Science and Engineering 16, Springer (open access),
   ch. 3, doi:10.1007/978-3-319-55456-3 [confirm DOI and chapter].

## Other likely sources (in Zotero; Henry to confirm which were used)

| Source | Zotero added | Why it may be relevant |
|---|---|---|
| Girona, T. and Costa, F. (2013), DIPRA, *G³* 14, 422-431, doi:10.1029/2012GC004427 | 2017-02-22 | Same equation for Fe-Mg in olivine; eq. 5 is the explicit update with a forward-difference cross term. Already cited in the restructure plan and used as the check in "Engine fix". |
| Crank, J. (1975), *The Mathematics of Diffusion*, 2nd ed., Oxford | 2017-02-22 | Analytic erf solution used in `tests/test_physics.py`; ch. 8 covers numerical methods [confirm chapter]. |
| Costa, F., Dohmen, R. and Chakraborty, S. (2008), "Time scales of magmatic processes from modeling the zoning patterns of crystals", *Rev. Mineral. Geochem.* 69, 545-594 [confirm pages] | 2020-08-14 | Standard review of explicit FD for diffusion chronometry, including the concentration-dependent-D term. |
| Press, W. H. et al., *Numerical Recipes*, ch. 19 (20 in 3rd ed.), "Partial Differential Equations" | 2020-08-09 | FTCS, von Neumann stability, Crank-Nicolson, tridiagonal solver. |
| Demouchy, S. and Thoraval, C. (2014), "Numerical models of ionic diffusion in one and three dimensions: application to dehydration of mantle olivine", *Phys. Chem. Minerals*, doi:10.1007/s00269-014-0685-x | 2016-11-20 | FD diffusion models in olivine. |
| Gerya, T., *Numerical Geodynamic Modelling*, Cambridge (2010 or 2019 ed.) | 2016-10-30 | Explicit and implicit FD for the heat/diffusion equation. |
| Ford Versypt, A. N. and Braatz, R. D. (2014), "Analysis of finite difference discretization schemes for diffusion in spheres with variable diffusivity", *Comput. Chem. Eng.*, doi:10.1016/j.compchemeng.2014.05.022 | 2023-04-21 | Discretizing the variable-$D$ cross term. |

## Open items

- Confirm the Warwick author and the Langtangen and Linge DOI and chapter.
- Henry: which of the Zotero items above were actually used for this code.
- The restructure plan's citation table
  (`docs/plans/2026-10-11-restructure.md`, row "Numerical scheme") says "own
  implementation"; once the list above is settled, cite the FTCS sources there.
- Consider enforcing the stability limit in `fit_profile` (raise if
  $\max D\,\Delta t/\Delta x^2 > \tfrac12$) instead of leaving it as a comment.

## Log

Newest first. Record what was checked and how, including searches that found
nothing, so they are not repeated.

### 2026-10-11

- **Engine method.** Read `engine.py`: explicit FTCS, `np.convolve` with
  `[1, -2, 1]` (diffusion) and `[1, 0, -1]` (central derivative, cross term ÷4
  since commit 999a40f). No matrix is built.
- **Git history.** `git log --all -S` for `Crank`, `diags`, `spsolve`,
  `solve_banded`, `matmul` and `@ vector`: no matrix or implicit solver ever
  existed in this repo. The only `Crank` hit is the Crank (1975) erf citation
  added in de3bfd3; `np.convolve` goes back to the first commit (7c6bf5f,
  2026-02-14).
- **No source URLs in the code.** Grepping the repo's `.py` and `.md` files
  for `warwick`, `hplgit`, `langtangen`, `crank` and `nicolson` found only the
  Crank (1975) mention in `tests/test_physics.py`. The engine itself cites
  only DIPRA (in a comment on the cross-term kernel).
- **Stability.** `engine.py:136` states $D\,\Delta t/\Delta x^2 < 0.5$ as a
  comment; nothing in `src/` or `tests/` checks it.
- **Warwick CO906 ch. 3.** Downloaded and read: 11 pages, sections 3.1-3.6
  (FTCS, boundary conditions, consistency and von Neumann stability,
  Crank-Nicolson, similarity solutions). The PDF names no author, and its
  metadata only says it was made with cairo. Constant D throughout.
- **Langtangen notes.** Read the table of contents: FE, BE, sparse matrix
  implementation, Crank-Nicolson, θ rule, amplification-factor analysis,
  "Diffusion in heterogeneous media". Page dated 2015.
- **Zotero.** Searched a copy of `~/Zotero/zotero.sqlite` (titles and URLs)
  for finite diff, Crank, Nicolson, diffusion equation, numerical, explicit,
  implicit, stencil, convolution, Fick, Mathematics of Diffusion, warwick,
  hplgit, Langtangen and DIPRA, then separately for Costa et al. (2008). The
  hits are in the table above. Neither lecture-note source is in Zotero.
- **Not searched.** A search of the other repos in `~/Projects` for older
  matrix-based diffusion code was blocked by a permission check. Henry can run
  `grep -rliE "crank|nicolson|diags|solve_banded|hplgit|warwick"` there
  to check.
