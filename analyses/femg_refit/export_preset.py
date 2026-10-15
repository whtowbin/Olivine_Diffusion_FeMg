"""Write the `femg_refit2026` preset from the refit, and check it against the refit.

Fitted to the verified compilation (`fit_verified.verified()`: every value stated in
the sources' tables or text; see docs/references/data-to-source.md). The DC07 form
with log D0, E, the fO2 exponent and the composition coefficient fitted, D[100] =
D[010] = D[001]/6 and V = 7 cm3/mol fixed (`fit_verified.FORM`), with a random offset
per study (`fit_models.fit_random`). Checks that the law reproduces the refit's log10 D
and its 1-sigma (`fit_models.predict`) on every axis at the round robin and paper
conditions and over a spread of others, and that it equals the preset shipped in
`src/forster/diffusivity.py`; if not (the data or the fit changed), prints the
values to paste there and exits with status 1. Run:
uv run --with odrpack python analyses/femg_refit/export_preset.py
"""

import math
import warnings

import fit_models as F
import fit_verified
import numpy as np

from forster import DIFFUSIVITY_PRESETS, Conditions, Orientation
from forster.diffusivity import _REFIT_2026, ArrheniusDiffusivity, LawUncertainty

warnings.simplefilter("ignore")
LN10 = math.log(10)
PARAMETERS = (
    "log10_D0_001",
    "activation_energy_J_mol",
    "fO2_exponent",
    "fo_coefficient",
)  # the anisotropy is fixed at 6x, so it carries no covariance
AXES = {
    "[100]": Orientation(0, 90, 90),
    "[010]": Orientation(90, 0, 90),
    "[001]": Orientation(90, 90, 0),
}


def preset(fit: dict, n_studies: int) -> ArrheniusDiffusivity:
    log_d0, e, m, n = fit["beta"]
    cov = (fit["cov"] + fit["cov"].T) / 2  # exactly symmetric
    off_axis = log_d0 - math.log10(6)
    lo, hi = (t / LN10 for t in fit["tau_range"])
    return ArrheniusDiffusivity(
        name="femg_refit2026",
        element="Fe-Mg",
        log10_D0_m2_s=(float(off_axis), float(off_axis), float(log_d0)),
        activation_energy_J_mol=(float(e),) * 3,
        fO2_exponent=float(m),
        fO2_ref_Pa=1e-7,
        fo_coefficient=float(n),
        fo_ref=0.9,
        activation_volume_m3_mol=F.V_DC07,
        P_ref_Pa=1e5,
        source=_REFIT_2026,
        uncertainty=LawUncertainty(
            parameters=PARAMETERS,
            covariance=tuple(map(tuple, cov)),
            between_study_log10=float(fit["tau"] / LN10),
            source=(
                f"Covariance of the law averaged over laboratories; between-study sd "
                f"{fit['tau'] / LN10:.3f} log10 (95% {lo:.2f}-{hi:.2f}, {n_studies} studies)."
            ),
        ),
    )


def check(law: ArrheniusDiffusivity, fit: dict) -> float:
    """Largest difference in log10 D and in its sigma between preset and refit."""
    form = fit_verified.FORM
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
    data = fit_verified.verified()
    fit = F.fit_random(fit_verified.FORM, data)
    law = preset(fit, data["source"].nunique())
    print(f"worst difference, preset vs refit (log10 D and its sigma): {check(law, fit):.2e}")
    if law == DIFFUSIVITY_PRESETS["femg_refit2026"]:
        print("the preset in src/forster/diffusivity.py is this law, exactly")
        return
    u = law.uncertainty
    assert u is not None
    print("\n# --- the preset differs: paste into src/forster/diffusivity.py ---")
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
    raise SystemExit(1)


if __name__ == "__main__":
    main()
