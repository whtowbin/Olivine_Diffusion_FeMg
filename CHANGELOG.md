# Changelog

## Unreleased

### Added

- DIPRA's finite-difference scheme (Girona and Costa 2013, eq. 5) as an alternative to the engine's: `fit_profile(..., scheme="dipra")`, `RunDefaults(scheme="dipra")`, or a `scheme` column in a parameter table. Both schemes solve the same equation and differ only in how the cross term is differenced (DIPRA: forward differences and the analytic dD/dX_Fo; engine: central differences of C and D), so fitting a profile with both is a check on the discretization. Default stays "central".

### Fixed

- Cross term of the diffusion equation was doubled (`diffusion_step`, found 2026-10-11). The engine solves dC/dt = D d²C/dx² + (dD/dx)(dC/dx). The cross term was built from two central differences, each spanning two grid spacings, so their product needs dividing by 4Δx²; the code divided by 2Δx². The fix changes `/ 2` to `/ 4`. It affects every Fe-Mg, Mn and Ni model where D varies along the profile, and only those (D depends on Fo). Checked against an independent flux-form solver and against DIPRA's published scheme (Girona and Costa 2013, eq. 5), which converge to the same answer as the fixed engine while the old engine does not (`tests/test_physics.py`; details in `docs/plans/2026-10-11-restructure.md`, "Engine fix").
  - Effect on fitted times: idealised single steps came out 1.3% too short for Fo 0.92 to 0.895 and 8-11% too short for Fo 0.82 to 0.885 and 0.67 to 0.745. The 17 round robin fits move by -28% to +43% (most become 13-43% longer). For Towbin and Plank (2026) the melt-vein medians move by 2-5% (1-3 weeks unchanged) and the exterior contact-overgrowth median by +28% (5.9 to 7.5 days).

### Before release

- [ ] Paper addendum, to return to later: the cross-term fix changes the timescales in Towbin and Plank (2026, J. Petrology 67(9), egag077). Melt-vein medians move by 2-5% (WHT01 17.9 to 17.6 days, GCB1 12.8 to 13.5, WHT06 5.6 to 5.4); the exterior contact-overgrowth median moves from 5.9 to 7.5 days (+28%); xenocrysts +9%. Check the ascent rates derived from these times. Numbers per profile: `docs/plans/2026-10-11-engine-fix/`.

- [ ] Come back to the cross-term fix: state the size of the bias for older results in the release notes, refit the round robin profiles (revisit the SH63olv066 initial conditions, whose fits get worse with the fixed engine), correct any submitted round robin results, and add the note on the paper results to the Xenolith_Melt_Vein_Modeling README.
