"""Write the `femg_refit2026` preset from the refit, and check it against the refit.

The DC07 form with log D0, E, the fO2 exponent, the composition coefficient and
both anisotropy factors fitted (V fixed at 7 cm3/mol), with a random offset per
study (`fit_models.fit_random`). Prints the `ArrheniusDiffusivity` for
`src/forster/diffusivity.py`, then checks that the preset reproduces the refit's
log10 D and its 1-sigma (`fit_models.predict`) on every axis at the round robin
and paper conditions and over a spread of others. Run:
uv run --with odrpack python analyses/femg_refit/export_preset.py
"""

import math
import warnings

import fit_models as F
import numpy as np

from forster import Conditions, Orientation
from forster.diffusivity import ArrheniusDiffusivity, LawUncertainty

warnings.simplefilter("ignore")
LN10 = math.log(10)
PARAMETERS = (
    "log10_D0_001",
    "activation_energy_J_mol",
    "fO2_exponent",
    "fo_coefficient",
    "log10_100_vs_001",
    "log10_010_vs_001",
)
AXES = {
    "[100]": Orientation(0, 90, 90),
    "[010]": Orientation(90, 0, 90),
    "[001]": Orientation(90, 90, 0),
}


def preset(fit: dict) -> ArrheniusDiffusivity:
    log_d0, e, m, n, k100, k010 = fit["beta"]
    # dc07_free's beta is (log10 D0, E, m, n, ln offsets of [100] and [010]); the
    # offsets become log10.
    jac = np.diag([1.0, 1.0, 1.0, 1.0, 1 / LN10, 1 / LN10])
    cov = jac @ fit["cov"] @ jac.T
    cov = (cov + cov.T) / 2  # exactly symmetric
    lo, hi = (t / LN10 for t in fit["tau_range"])
    return ArrheniusDiffusivity(
        name="femg_refit2026",
        element="Fe-Mg",
        log10_D0_m2_s=(float(log_d0 + k100 / LN10), float(log_d0 + k010 / LN10), float(log_d0)),
        activation_energy_J_mol=(float(e),) * 3,
        fO2_exponent=float(m),
        fO2_ref_Pa=1e-7,
        fo_coefficient=float(n),
        fo_ref=0.9,
        activation_volume_m3_mol=F.V_DC07,
        P_ref_Pa=1e5,
        source=(
            "Refit of the TaMED Fe-Mg data (Chakraborty 1997; Dohmen et al. 2007; Petry et "
            "al. 2004; Holzapfel et al. 2007; Spandler and O'Neill 2010; fO2 > 1e-10 Pa, "
            "135 points) in the Dohmen and Chakraborty (2007) form, V fixed at 7 cm3/mol, "
            "least squares with a random offset per study (REML). analyses/femg_refit, "
            "method 'random study'."
        ),
        uncertainty=LawUncertainty(
            parameters=PARAMETERS,
            covariance=tuple(map(tuple, cov)),
            between_study_log10=float(fit["tau"] / LN10),
            source=(
                f"Covariance of the law averaged over laboratories; between-study sd "
                f"{fit['tau'] / LN10:.3f} log10 (95% {lo:.2f}-{hi:.2f}, five studies)."
            ),
        ),
    )


def check(law: ArrheniusDiffusivity, fit: dict) -> float:
    """Largest difference in log10 D and in its sigma between preset and refit."""
    form = F.FORMS["dc07_free"]
    worst = 0.0
    rng = np.random.default_rng(1)
    cases = [(1200.0, 4.2e7, math.log10(0.00138), 0.85), (1250.0, 5e8, math.log10(8.3e-3), 0.90)]
    cases += [
        (
            rng.uniform(800, 1400),
            rng.uniform(1e5, 1e9),
            rng.uniform(-10, -1),
            rng.uniform(0.7, 0.95),
        )
        for _ in range(20)
    ]
    for t_c, p, log_f, x in cases:
        c = Conditions(t_c, p, 10**log_f)
        for i, (axis, o) in enumerate(AXES.items()):
            v = F.Vars(
                *(
                    np.array([val])
                    for val in (
                        log_f,
                        x,
                        t_c + 273.15,
                        p,
                        float(axis == "[100]"),
                        float(axis == "[010]"),
                    )
                )
            )
            mean, se = F.predict(form, fit["beta"], fit["cov"], v)
            d = abs(math.log10(law.axis_diffusivities(c, x)[i]) - mean)
            s = abs(law.log10_sigma(c, o, x) - se)
            worst = max(worst, d, s)
    return worst


def main() -> None:
    fit = F.fit_random(F.FORMS["dc07_free"], F.tamed())
    law = preset(fit)
    print(f"worst difference, preset vs refit (log10 D and its sigma): {check(law, fit):.2e}")
    u = law.uncertainty
    assert u is not None
    print("\n# --- for src/forster/diffusivity.py ---")
    print(f"log10_D0_m2_s={law.log10_D0_m2_s!r},")
    print(f"activation_energy_J_mol={law.activation_energy_J_mol!r},")
    print(f"fO2_exponent={law.fO2_exponent!r},")
    print(f"fo_coefficient={law.fo_coefficient!r},")
    print(f"parameters={u.parameters!r},")
    print("covariance=(")
    for row in u.covariance:
        print(f"    {row!r},")
    print("),")
    print(f"between_study_log10={u.between_study_log10!r},")
    print(f"source={u.source!r},")


if __name__ == "__main__":
    main()
