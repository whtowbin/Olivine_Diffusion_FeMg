# Fe-Mg refit: Oeser et al. (2026) added, and a two-mechanism law

Status (2026-10-14): fitted and recorded; nothing adopted. The compilation (`fit_ols.load`),
`results/*.csv` and the `femg_refit2026` preset are unchanged. These fits are a side
branch, `analyses/femg_refit/fit_extended.py`, which writes `results/extended/` and
`figures/extended_fo2.png`.

## Why

Two questions Henry approved on 2026-10-14:

1. **Add Oeser et al. (2026) as a sixth study.** It is a solid powder source at 1100-1303 C
   and 1e-7 to 1e-2.3 Pa, the only solid-source data near our runs' fO2 apart from Petry
   (data: `data/oeser2026.csv`, see `diffusivity-literature.md`).
2. **Test a sum of two mechanisms** against the single power law in fO2:
   D = D_TaMED + D_ext, where D_ext has no fO2 dependence, like the PED law of
   Chakraborty (2010).

   Two observations prompted the test:
   - Oeser's local fO2 slopes at 1150 C steepen with fO2: 0.09-0.24 from 1e-7 to 1e-5 Pa,
     0.42-0.52 above 1e-5 Pa;
   - the earlier refit found an fO2 x 1/T term (p = 0.008).

## Method

- **Forms:**
  - the power law is the `dc07_free` form: log D0, E, fO2 exponent m, composition n and
    both anisotropy offsets, with V fixed at 7 cm3/mol;
  - the sum adds D_ext = 10^(a_x - E_x/(ln10 R T)), which shares the composition,
    anisotropy and pressure terms.
- **Data:**
  - power-law fits use the TaMED set (fO2 > 1e-10 Pa) plus Oeser;
  - the sum is meant to cover both regimes, so it gets every fO2, including Dohmen et
    al.'s 31 points below 1e-10 Pa.
- **Fitting:** nonlinear least squares with a random offset per study, with REML over a
  grid of tau^2/sigma^2.
- **Control:** on the power law and the TaMED data, the machinery reproduces
  `fit_models.fit_random`: log D0 -8.881, E 209.06 kJ/mol, m 0.2206 and n 3.150, all
  identical; tau 0.255 against 0.254, from the coarser grid. The script asserts this
  before fitting anything.
- **Oeser runs 7, 14 and 15** (1255-1300 C) were measured by laser linescan after their
  crystals broke, and are convolution corrected. They are fitted in one variant ("all")
  and left out of the others ("dp").

## Results

Log units throughout. "vs DC07" is D[001] at the RR1200 conditions (the round robin 1200 C conditions: 1200 C, fO2 1.4e-3 Pa,
Fo85, 42 MPa).

| Fit | n | tau (95%) | sd within | m | E (kJ/mol) | RR1200 vs DC07 | RR1200 1 sigma, law / one new lab |
|---|---|---|---|---|---|---|---|
| power, TaMED (= preset) | 135 | 0.255 (0.11-0.60) | 0.221 | 0.221 | 209.1 | +0.27 | 0.122 / 0.283 |
| power + Oeser, all | 189 | 0.245 (0.14-0.57) | 0.283 | 0.161 | 239.4 | +0.24 | 0.110 / 0.268 |
| power + Oeser, dp | 170 | 0.218 (0.13-0.51) | 0.218 | 0.226 | 212.9 | +0.29 | 0.097 / 0.239 |
| power + Oeser dp + melt term | 170 | 0.188 (0.09-0.44) | 0.217 | 0.222 | 212.8 | +0.23 | 0.091 / 0.209 |
| sum, all fO2 | 166 | 0.310 (0.12-0.84) | 0.232 | 0.255 | 220.4 | +0.26 | 0.146 / 0.343 |
| sum + Oeser, dp | 201 | 0.258 (0.13-0.60) | 0.224 | 0.251 | 225.6 | +0.28 | 0.113 / 0.282 |
| sum + Oeser dp + melt term | 201 | 0.258 (0.11-0.70) | 0.223 | 0.251 | 225.0 | +0.23 | 0.121 / 0.285 |

- **The melt term (Spandler)** is +0.42 ± 0.24 with Oeser in (power law) and +0.38 ± 0.31
  (sum). Without Oeser it was +0.57 ± 0.18 (`melt-contact-diffusion.md`).
- **The sum's extrinsic term** is a_x = -11.33 ± 0.83 and E_x = 172 ± 18 kJ/mol.
- **Shrunken study offsets** for Oeser (dp) are +0.15 (power law), +0.21 (with the melt
  term) and +0.17 (sum); with all its runs, +0.31.

What the numbers say:

- **Oeser's depth-profile runs agree with the refit at our conditions.** Adding them moves
  the RR1200 level by +0.01, the exponent by +0.005 and E by +4 kJ/mol. They narrow the law's
  1 sigma at RR1200 from 0.122 to 0.097, and the new-lab range from 0.283 to 0.239.
- **Oeser enters as a sixth study at +0.15.** With the melt term fitted, the solid-source
  spread grows from 0.107 to 0.188. Oeser now sits on the high side, and it takes up part
  of what the melt term had absorbed (+0.57 to +0.42).
- **The linescan runs distort the fit.** With them in, E rises to 239 kJ/mol, m falls to
  0.161 and the within-study sd rises to 0.283. They act through temperature, since all
  three are at 1255-1300 C. They should stay out until the convolution correction is
  checked.
- **The sum fits both regimes with one law, but changes nothing at our conditions.** Its
  extrinsic term matters only below about 1e-10 Pa. There it fits Dohmen et al.'s PED
  points as well as it fits its own TaMED points (rms 0.28 each; the power law's TaMED rms is 0.26). Between 1e-7
  and 1e-2 Pa the two laws differ by at most 0.09 log, and at RR1200 by 0.01.
- **The sum does not reproduce Oeser's steepening at 1150 C.** Its local exponent there
  is 0.24 at 1e-6 Pa and 0.25 at 1e-4.35 Pa, against Oeser's local slopes of 0.09-0.24
  and 0.42-0.52.
  - **Why it can't:** in a sum, the slope can only rise toward m as fO2 rises. The
    steepening would need D_ext to dominate up to about 1e-5 Pa at 1150 C, and Dohmen et
    al.'s series at 1050-1080 C (exponent 0.18-0.26 between 1e-10 and 1e-5 Pa) rule that out.
  - **What the steepening rests on:** one run, run 16 (3 profiles at 1e-3.7 Pa). The joint
    data do not support it.
  - The sum does not give an exponent that changes with temperature either: its local
    exponent is 0.25 at both 1150 and 1250 C.

## What this leaves open

- **Whether to add Oeser to the default compilation** and re-export `femg_refit2026` (dp
  runs only). Effect at RR1200: level +0.01, law 1 sigma 0.122 to 0.097. Because it changes
  the preset, `export_preset.py`'s exact-match check requires updating the preset in the
  same change.
- **The melt-term preset decision is now entangled with Oeser.** With both in, the
  solid-source spread is 0.188, not 0.107, and the melt term is +0.42 ± 0.24.
- **The sum** is a better-founded single law across all fO2: it is the only form that
  fits the PED points. It is not needed at the round robin and paper conditions. Keep it
  as a candidate form for low-fO2 work.
- **Steepening near 1e-4 Pa:** only a solid-source experiment at 1e-4 to 1e-2 Pa near
  1150-1200 C, by depth profiling, can test it.

Run: `uv run python analyses/femg_refit/fit_extended.py` (about 2 s).
