"""docs/references/femg-refit2026.md must show the femg_refit2026 values the code ships."""

import math
from pathlib import Path

import numpy as np

from forster import DIFFUSIVITY_PRESETS, Conditions, Orientation

DOC = Path(__file__).parents[1] / "docs" / "references" / "femg-refit2026.md"
LAW = DIFFUSIVITY_PRESETS["femg_refit2026"]
ALONG_001 = Orientation(90, 90, 0)


def test_the_doc_shows_the_shipped_equation_and_its_uncertainty():
    text = DOC.read_text()
    u = LAW.uncertainty
    assert u is not None
    se = dict(zip(u.parameters, np.sqrt(np.diag(u.covariance)), strict=True))
    expected = [
        f"{LAW.log10_D0_m2_s[2]:.3f}",
        f"{LAW.activation_energy_J_mol[2] / 1e3:.1f}",
        f"{LAW.activation_volume_m3_mol * 1e6:.2f}",
        f"{LAW.fO2_exponent:.3f}",
        f"{LAW.fo_coefficient:.2f}",
        f"{se['log10_D0_001']:.3f}",
        f"{se['activation_energy_J_mol'] / 1e3:.1f}",
        f"{se['activation_volume_m3_mol'] * 1e6:.2f}",
        f"{se['fO2_exponent']:.3f}",
        f"{se['fo_coefficient']:.2f}",
        f"{u.between_study_log10:.3f}",
        f"{LAW.activation_volume_m3_mol:.2e}".replace("e-06", r"\times10^{-6}"),
    ]
    missing = [value for value in expected if value not in text]
    assert not missing, f"{DOC.name} does not show {missing}: update it from the preset"
    assert math.isclose(
        LAW.log10_D0_m2_s[0], LAW.log10_D0_m2_s[2] - math.log10(6), abs_tol=1e-12
    ), "the doc says [100] and [010] are 6x slower than [001]"


def test_the_doc_worked_values_are_the_codes():
    text = DOC.read_text()
    for t_c, p_pa, fo2_pa in ((1200.0, 4.2e7, 1.38e-3), (1100.0, 1e5, 1e-4), (900.0, 1e8, 1e-9)):
        c = Conditions(t_c, p_pa, fo2_pa)
        log_d = math.log10(LAW.axis_diffusivities(c, 0.85)[2])
        row = f"| {log_d:.3f} | {LAW.log10_sigma(c, ALONG_001, 0.85):.3f} |"
        assert row in text, f"worked value at {t_c} C should read {row}"
