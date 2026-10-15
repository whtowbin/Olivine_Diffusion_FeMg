"""Diffusivity equations: the default (DIPRA's Fe-Mg law), presets, and user-defined ones.

Every equation here has the form, for each crystal axis i ([100], [010], [001]),

    log10 D_i = log10_D0_i + fO2_exponent * log10(fO2 / fO2_ref_Pa)
                + fo_coefficient * (fo_ref - X_Fo)
                - (E_i + activation_volume * (P - P_ref_Pa)) / (ln(10) R T)
                + pressure_coefficient * (P - P_ref_Pa)

and the diffusivity along a profile is sum_i D_i cos^2(angle to axis i)
(Costa and Chakraborty 2004; DIPRA eq. 4). T in K, P and fO2 in Pa, D in m^2/s.
Because the composition term is shared by all axes, dD/dX_Fo = -fo_coefficient
ln(10) D, which the DIPRA finite-difference scheme needs.

The default, "dohmen_chakraborty2007", is the Fe-Mg equation of Dohmen and
Chakraborty (2007, with their erratum) as tabulated in DIPRA (Girona and Costa
2013, Table 1), and it is computed by the same code as before (``engine.D_Fo``),
so default results are unchanged. ``docs/references/diffusivity-literature.md``
compares it with newer work.

An equation may carry a `LawUncertainty`: a covariance over its parameters (and the
spread between laboratories), which `ArrheniusDiffusivity.log10_sigma` turns into
the 1-sigma of log10 D at given conditions. For an Fo fit the profile depends on
D t, so that is also the 1-sigma of log10 of the fitted time
(`forster.model.FitResult.time_range_days`).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, fields, replace
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import pandas as pd

from forster import dipra

if TYPE_CHECKING:
    from forster.model import Conditions, Orientation

R_J_MOL_K = 8.3145  # as in engine.D_Fo
ELEMENTS = ("Fe-Mg", "Ni", "Mn", "Ca")
LOG10_6 = math.log10(6.0)  # [001] is 6x faster than [100] and [010] for Fe-Mg, Ni, Mn

DiffusivityFunctions = tuple[Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]


@dataclass(frozen=True)
class LogLinearDiffusivity:
    """D(X_Fo) = at_ref * 10**(fo_coefficient * (fo_ref - X_Fo)), evaluated by ``function``.

    Calling it calls ``function`` (so results are unchanged); the parameters let the
    compiled time steppers (`forster.compiled`) evaluate D themselves.
    """

    function: Callable[[np.ndarray], np.ndarray]
    at_ref: float
    fo_coefficient: float
    fo_ref: float

    def __call__(self, x_fo: np.ndarray) -> np.ndarray:
        return self.function(x_fo)


class Diffusivity(Protocol):
    """What `fit_profile` needs from a diffusivity; `ArrheniusDiffusivity` is one.

    A Python user can pass any object with these attributes, for an equation that
    does not fit the Arrhenius form. ``functions`` returns D(X_Fo) along the profile
    (m^2/s) and dD/dX_Fo, both taking and returning arrays.
    """

    @property
    def name(self) -> str: ...

    @property
    def element(self) -> str: ...

    def functions(
        self, conditions: Conditions, orientation: Orientation
    ) -> DiffusivityFunctions: ...


# Parameters a `LawUncertainty` may cover. The anisotropy terms are log10 offsets of
# [100] and [010] from [001], so a change in log10_D0_001 moves every axis.
LAW_PARAMETERS = (
    "log10_D0_001",
    "activation_energy_J_mol",
    "fO2_exponent",
    "fo_coefficient",
    "activation_volume_m3_mol",
    "pressure_coefficient_per_Pa",
    "log10_100_vs_001",
    "log10_010_vs_001",
)


@dataclass(frozen=True)
class LawUncertainty:
    """Calibration uncertainty of an `ArrheniusDiffusivity`.

    ``covariance`` is over ``parameters`` (names from `LAW_PARAMETERS`, in the
    equation's own units); parameters not named are taken as exact.
    ``between_study_log10`` is the standard deviation of a laboratory's offset in
    log10 D (a random study effect): what one more experiment, or a crystal that
    behaves like one, adds beyond the law's own uncertainty. Tuples keep the
    dataclass hashable.
    """

    parameters: tuple[str, ...]
    covariance: tuple[tuple[float, ...], ...]
    between_study_log10: float = 0.0
    source: str = ""

    def __post_init__(self) -> None:
        unknown = [p for p in self.parameters if p not in LAW_PARAMETERS]
        if unknown or len(set(self.parameters)) != len(self.parameters):
            raise ValueError(
                f"law uncertainty: parameters must be distinct names from LAW_PARAMETERS, "
                f"got {', '.join(self.parameters)}"
            )
        cov = np.array(self.covariance, dtype=float)
        n = len(self.parameters)
        if cov.shape != (n, n) or not np.all(np.isfinite(cov)):
            raise ValueError(f"law uncertainty: covariance must be a finite {n} x {n} matrix")
        if not np.allclose(cov, cov.T, rtol=1e-9, atol=0.0):
            raise ValueError("law uncertainty: covariance must be symmetric")
        sd = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
        if np.linalg.eigvalsh(cov / np.outer(sd, sd)).min() < -1e-9:
            raise ValueError("law uncertainty: covariance must be positive semi-definite")
        if not (math.isfinite(self.between_study_log10) and self.between_study_log10 >= 0):
            raise ValueError("law uncertainty: between_study_log10 must be zero or more")
        object.__setattr__(self, "parameters", tuple(self.parameters))
        object.__setattr__(self, "covariance", tuple(tuple(map(float, row)) for row in cov))


def _cos2(orientation: Orientation) -> np.ndarray:
    """Squared direction cosines of the profile to [100], [010], [001]."""
    angles = [orientation.alpha_deg, orientation.beta_deg, orientation.gamma_deg]
    return np.cos(np.deg2rad(angles)) ** 2


@dataclass(frozen=True)
class ArrheniusDiffusivity:
    """One diffusivity equation in the form of the module docstring.

    Per-axis values are tuples in the order ([100], [010], [001]). The dataclass is
    frozen; derive variants with `dataclasses.replace` or `shift_activation_energy`.
    """

    name: str
    element: str
    log10_D0_m2_s: tuple[float, ...]  # 3 values, checked in __post_init__
    activation_energy_J_mol: tuple[float, ...]
    fO2_exponent: float = 0.0
    fO2_ref_Pa: float = 1.0
    fo_coefficient: float = 0.0
    fo_ref: float = 0.0
    activation_volume_m3_mol: float = 0.0
    P_ref_Pa: float = 0.0
    pressure_coefficient_per_Pa: float = 0.0
    source: str = ""
    uncertainty: LawUncertainty | None = None  # not in the diffusivities table

    def __post_init__(self) -> None:
        if not self.name or not str(self.name).strip():
            raise ValueError("a diffusivity needs a name")
        if self.element not in ELEMENTS:
            raise ValueError(
                f"diffusivity {self.name!r}: element {self.element!r}; use one of "
                f"{', '.join(ELEMENTS)}"
            )
        for label in ("log10_D0_m2_s", "activation_energy_J_mol"):
            values = tuple(float(v) for v in getattr(self, label))
            if len(values) != 3:
                raise ValueError(
                    f"diffusivity {self.name!r}: {label} needs 3 values (100, 010, 001)"
                )
            object.__setattr__(self, label, values)
        numbers = [*self.log10_D0_m2_s, *self.activation_energy_J_mol] + [
            getattr(self, f.name) for f in fields(self) if f.type == "float"
        ]
        if not all(math.isfinite(v) for v in numbers):
            raise ValueError(
                f"diffusivity {self.name!r}: every coefficient must be a finite number"
            )
        if self.fO2_ref_Pa <= 0:
            raise ValueError(f"diffusivity {self.name!r}: fO2_ref_Pa must be positive")

    def shift_activation_energy(self, delta_J_mol: float) -> ArrheniusDiffusivity:
        """The same equation with every axis's activation energy shifted by ``delta_J_mol``."""
        return replace(
            self,
            activation_energy_J_mol=tuple(e + delta_J_mol for e in self.activation_energy_J_mol),
        )

    def axis_diffusivities(self, conditions: Conditions, x_fo: float | np.ndarray) -> np.ndarray:
        """D along [100], [010], [001] (m^2/s) at ``x_fo``; last axis of the result is the axis."""
        T, P = conditions.T_K, conditions.P_Pa
        x = np.asarray(x_fo, dtype=float)[..., None]
        log10_D = (
            np.array(self.log10_D0_m2_s)
            + self.fO2_exponent * math.log10(conditions.fO2_Pa / self.fO2_ref_Pa)
            + self.fo_coefficient * (self.fo_ref - x)
            - (
                np.array(self.activation_energy_J_mol)
                + self.activation_volume_m3_mol * (P - self.P_ref_Pa)
            )
            / (math.log(10) * R_J_MOL_K * T)
            + self.pressure_coefficient_per_Pa * (P - self.P_ref_Pa)
        )
        return 10.0**log10_D

    def log10_sigma(
        self, conditions: Conditions, orientation: Orientation, x_fo: float
    ) -> float | None:
        """1-sigma of log10 D along the profile at ``x_fo`` from the law's covariance
        (without the between-study spread); None if the law has none."""
        if self.uncertainty is None:
            return None
        cos2 = _cos2(orientation)
        weights = self.axis_diffusivities(conditions, x_fo) * cos2
        weights = weights / weights.sum()
        T, dP = conditions.T_K, conditions.P_Pa - self.P_ref_Pa
        per_rt = 1 / (math.log(10) * R_J_MOL_K * T)
        gradient = {
            "log10_D0_001": 1.0,
            "activation_energy_J_mol": -per_rt,
            "fO2_exponent": math.log10(conditions.fO2_Pa / self.fO2_ref_Pa),
            "fo_coefficient": self.fo_ref - x_fo,
            "activation_volume_m3_mol": -dP * per_rt,
            "pressure_coefficient_per_Pa": dP,
            "log10_100_vs_001": float(weights[0]),
            "log10_010_vs_001": float(weights[1]),
        }
        g = np.array([gradient[p] for p in self.uncertainty.parameters])
        variance = float(g @ np.array(self.uncertainty.covariance) @ g)
        return math.sqrt(max(variance, 0.0))

    def functions(self, conditions: Conditions, orientation: Orientation) -> DiffusivityFunctions:
        """D(X_Fo) along the profile and dD/dX_Fo, for the model's time steppers."""
        cos2 = _cos2(orientation)
        # Everything but the composition term is constant during a run.
        at_ref = float(self.axis_diffusivities(conditions, self.fo_ref) @ cos2)
        slope = -self.fo_coefficient * math.log(10)

        def diffusivity(x_fo: np.ndarray) -> np.ndarray:
            return at_ref * 10.0 ** (self.fo_coefficient * (self.fo_ref - np.asarray(x_fo)))

        def derivative(x_fo: np.ndarray) -> np.ndarray:
            return slope * diffusivity(x_fo)

        return LogLinearDiffusivity(
            diffusivity, at_ref, self.fo_coefficient, self.fo_ref
        ), derivative


@dataclass(frozen=True)
class _EngineFeMg(ArrheniusDiffusivity):
    """Dohmen and Chakraborty (2007) Fe-Mg, computed by ``engine.D_Fo`` as before
    (through ``dipra.D_Fo_and_derivative``).

    Results are bit-identical to earlier versions (the round robin differential
    test depends on that). Any change other than a uniform shift in activation
    energy falls back to the general formula, which agrees to about 1e-15.
    """

    def functions(self, conditions: Conditions, orientation: Orientation) -> DiffusivityFunctions:
        E = self.activation_energy_J_mol
        unchanged = replace(
            self,
            name=DOHMEN_CHAKRABORTY_2007.name,
            source=DOHMEN_CHAKRABORTY_2007.source,
            activation_energy_J_mol=DOHMEN_CHAKRABORTY_2007.activation_energy_J_mol,
        )
        if E[0] == E[1] == E[2] and unchanged == DOHMEN_CHAKRABORTY_2007:
            D, dD = dipra.D_Fo_and_derivative(
                conditions.T_K,
                conditions.P_Pa,
                conditions.fO2_Pa,
                orientation.alpha_deg,
                orientation.beta_deg,
                orientation.gamma_deg,
                E[0],
            )
            at_ref = float(np.asarray(D(np.array([self.fo_ref])))[0])
            return LogLinearDiffusivity(D, at_ref, self.fo_coefficient, self.fo_ref), dD
        return ArrheniusDiffusivity.functions(self, conditions, orientation)


_DIPRA_TABLE_1 = "Girona and Costa (2013), DIPRA, G-cubed 14, 422-431, Table 1"

DOHMEN_CHAKRABORTY_2007 = _EngineFeMg(
    name="dohmen_chakraborty2007",
    element="Fe-Mg",
    log10_D0_m2_s=(-9.21 - LOG10_6, -9.21 - LOG10_6, -9.21),
    activation_energy_J_mol=(201_000.0, 201_000.0, 201_000.0),
    fO2_exponent=1 / 6,
    fO2_ref_Pa=1e-7,
    fo_coefficient=3.0,
    fo_ref=0.9,
    activation_volume_m3_mol=7e-6,
    P_ref_Pa=1e5,
    source=(
        "Dohmen and Chakraborty (2007), Phys Chem Minerals 34, 409-430, and erratum "
        f"34, 597-598; fO2 > 1e-10 Pa. As in {_DIPRA_TABLE_1}."
    ),
)


_REFIT_2026 = (
    "Refit of the verified Fe-Mg data, every value stated in the sources' tables or text "
    "(docs/references/data-to-source.md): Chakraborty 1997; Dohmen et al. 2007, fO2 > "
    "1e-10 Pa; Petry et al. 2004; Holzapfel et al. 2007, Ni-NiO capsules, and Au capsules "
    "with their unbuffered fO2 taken at Ni-NiO +- 2 log units (132 points, 4 studies). "
    "Dohmen and Chakraborty (2007) form with log D0, E, the fO2 exponent and the "
    "composition coefficient fitted; [100] and [010] 6x slower than [001] and V = 7 "
    "cm3/mol fixed; least squares with a random offset per study (REML) and each run's "
    "errors in T (at least 10 C), fO2, X_Fo and P carried as effective variance. Not "
    "included: melt contact (Spandler and O'Neill 2010) and Oeser et al. 2026, whose [010] "
    "is about 2x slower than [001], not 6x (both unconfirmed scenarios). analyses/femg_refit "
    "(fit_verified.py, export_preset.py)."
)


# Mutch et al. (2019) parameters (a, b, c, q, j, h) as forster's: log10 D0, the fO2
# exponent, the composition coefficient, E, the pressure coefficient, V. Each is a
# coefficient times a constant, so the covariance maps by that diagonal.
_MUTCH_PARAMETERS = (
    "log10_D0_001",
    "fO2_exponent",
    "fo_coefficient",
    "activation_energy_J_mol",
    "pressure_coefficient_per_Pa",
    "activation_volume_m3_mol",
)
_MUTCH_SCALE = (
    1 / math.log(10),
    1.0,
    -1 / math.log(10),
    -R_J_MOL_K,
    1 / math.log(10),
    -R_J_MOL_K,
)


def _mutch2019(
    name: str,
    element: str,
    data: str,
    coefficients: tuple[float, float, float, float, float, float],
    covariance: tuple[tuple[float, ...], ...],
    anisotropy: float = 6.0,
) -> ArrheniusDiffusivity:
    """Mutch et al. (2019) regression ln D[001] = a + b ln fO2(bar) + c X_Fo + q/T + j P +
    h P/T, at full precision with its covariance (Mutch et al. 2021, Data Set S6)."""
    a, b, c, q, j, h = coefficients
    ln10 = math.log(10)
    off_axis = a / ln10 - math.log10(anisotropy)
    s = np.array(_MUTCH_SCALE)
    cov = np.array(covariance) * np.outer(s, s)
    return ArrheniusDiffusivity(
        name=name,
        element=element,
        log10_D0_m2_s=(off_axis, off_axis, a / ln10),
        activation_energy_J_mol=(-q * R_J_MOL_K,) * 3,
        fO2_exponent=b,
        fO2_ref_Pa=1e5,  # fO2 in bar in the regression
        fo_coefficient=-c / ln10,
        fo_ref=0.0,
        activation_volume_m3_mol=-h * R_J_MOL_K,
        P_ref_Pa=0.0,
        pressure_coefficient_per_Pa=j / ln10,
        source=(
            f"Mutch et al. (2019), Nature Geoscience 12, 569-574 ({data}); coefficients and "
            "covariance at full precision from Mutch et al. (2021), G-cubed 22, "
            "e2020GC009303, Data Set S6 (Version 1, fO2 in bar). [001], with [100] and "
            f"[010] {anisotropy:g}x slower as in DFENS, which carries no anisotropy "
            "uncertainty."
        ),
        uncertainty=LawUncertainty(
            parameters=_MUTCH_PARAMETERS,
            covariance=tuple(map(tuple, (cov + cov.T) / 2)),
            source="Mutch et al. (2021), Data Set S6: the regression's OLS covariance.",
        ),
    )


DIFFUSIVITY_PRESETS: dict[str, ArrheniusDiffusivity] = {
    d.name: d
    for d in (
        DOHMEN_CHAKRABORTY_2007,
        _mutch2019(
            "mutch2019_tamed",
            "Fe-Mg",
            "regressed on the TaMED-regime Fe-Mg data, fO2 > 1e-10 Pa",
            (-6.755, 0.2244, -7.181, -26740, -5.213e-10, -1.028e-07),
            (
                (
                    0.719988227888066,
                    0.0136049442637915,
                    -0.137206914066245,
                    -316.542691119953,
                    -5.10703480738867e-11,
                    3.56565852966258e-08,
                ),
                (
                    0.0136049442637915,
                    0.000824692307620194,
                    0.00022482943325554,
                    11.8157816556034,
                    -1.60520420649397e-12,
                    2.06662706021798e-10,
                ),
                (
                    -0.137206914066245,
                    0.00022482943325554,
                    0.133506645898472,
                    44.4762047396756,
                    1.76263206439128e-12,
                    -5.04771041057526e-09,
                ),
                (
                    -316.542691119953,
                    11.8157816556034,
                    44.4762047396756,
                    819871.010459364,
                    8.11565428517953e-09,
                    -3.61035571275858e-05,
                ),
                (
                    -5.10703480738867e-11,
                    -1.60520420649397e-12,
                    1.76263206439128e-12,
                    8.11565428517953e-09,
                    2.07869617518516e-19,
                    -3.45519816705532e-16,
                ),
                (
                    3.56565852966258e-08,
                    2.06662706021798e-10,
                    -5.04771041057526e-09,
                    -3.61035571275858e-05,
                    -3.45519816705532e-16,
                    5.82629996012315e-13,
                ),
            ),
        ),
        _mutch2019(
            "mutch2019_global",
            "Fe-Mg",
            "regressed on all the Fe-Mg data, TaMED and PED",
            (-7.861, 0.1868, -7.208, -26580, -4.148e-10, -1.54e-07),
            (
                (
                    0.497039964543917,
                    0.00363381377240425,
                    -0.132319267271482,
                    -378.453948506284,
                    -2.76790679156379e-11,
                    2.69463943024172e-08,
                ),
                (
                    0.00363381377240425,
                    0.000431162727069159,
                    0.00107609125043613,
                    10.2162808414763,
                    -6.40646254133509e-13,
                    -1.98771926260489e-10,
                ),
                (
                    -0.132319267271482,
                    0.00107609125043613,
                    0.149124060624799,
                    51.0034229198428,
                    -1.46324714807073e-13,
                    -4.70997758933153e-09,
                ),
                (
                    -378.453948506284,
                    10.2162808414763,
                    51.0034229198428,
                    839580.703219708,
                    1.33015759093729e-08,
                    -3.94295445360981e-05,
                ),
                (
                    -2.76790679156379e-11,
                    -6.40646254133509e-13,
                    -1.46324714807073e-13,
                    1.33015759093729e-08,
                    2.33230090554607e-19,
                    -3.91323758633186e-16,
                ),
                (
                    2.69463943024172e-08,
                    -1.98771926260489e-10,
                    -4.70997758933153e-09,
                    -3.94295445360981e-05,
                    -3.91323758633186e-16,
                    6.61290384810696e-13,
                ),
            ),
        ),
        ArrheniusDiffusivity(
            name="femg_refit2026",
            element="Fe-Mg",
            log10_D0_m2_s=(-9.646974534486233, -9.646974534486233, -8.86882328410259),
            activation_energy_J_mol=(211256.92006817146, 211256.92006817146, 211256.92006817146),
            fO2_exponent=0.20754312391534094,
            fO2_ref_Pa=1e-7,
            fo_coefficient=3.16282093738726,
            fo_ref=0.9,
            activation_volume_m3_mol=7e-6,
            P_ref_Pa=1e5,
            source=_REFIT_2026,
            uncertainty=LawUncertainty(
                parameters=(
                    "log10_D0_001",
                    "activation_energy_J_mol",
                    "fO2_exponent",
                    "fo_coefficient",
                ),
                covariance=(
                    (
                        0.0721474075255834,
                        1727.9408078733795,
                        -0.0022955311787679032,
                        0.007035854001120223,
                    ),
                    (
                        1727.9408078733795,
                        44379548.797255434,
                        -49.576982192864584,
                        199.34916717874046,
                    ),
                    (
                        -0.0022955311787679032,
                        -49.576982192864584,
                        0.0002576018620512191,
                        -0.00011689453932131425,
                    ),
                    (
                        0.007035854001120223,
                        199.34916717874046,
                        -0.00011689453932131425,
                        0.01893332058630904,
                    ),
                ),
                between_study_log10=0.11824888990356036,
                source=(
                    "Covariance of the law averaged over laboratories; between-study sd "
                    "0.118 log10 (95% 0.03-0.46, 4 studies)."
                ),
            ),
        ),
        _mutch2019(
            "mutch2019_ni",
            "Ni",
            "Ni diffusion",
            (-11.09, 0.2769, -2.185, -25080, -1.246e-09, 9.967e-07),
            (
                (
                    3.32788082296725,
                    0.0108660968622702,
                    -1.77375521776915,
                    -2186.61589865981,
                    -1.40203650629739e-10,
                    1.90307239581102e-07,
                ),
                (
                    0.0108660968622702,
                    0.00216784683169149,
                    -0.0152818744193823,
                    85.006525908944,
                    -1.98169090924804e-12,
                    -1.97548792434352e-09,
                ),
                (
                    -1.77375521776915,
                    -0.0152818744193823,
                    1.87806040681595,
                    -339.86176883119,
                    2.67755733941985e-11,
                    -2.60844977093618e-08,
                ),
                (
                    -2186.61589865981,
                    85.006525908944,
                    -339.86176883119,
                    6793292.28743474,
                    9.49540857947479e-08,
                    -0.000320505347787139,
                ),
                (
                    -1.40203650629739e-10,
                    -1.98169090924804e-12,
                    2.67755733941985e-11,
                    9.49540857947479e-08,
                    2.22508365973983e-19,
                    -3.68519013721069e-16,
                ),
                (
                    1.90307239581102e-07,
                    -1.97548792434352e-09,
                    -2.60844977093618e-08,
                    -0.000320505347787139,
                    -3.68519013721069e-16,
                    6.24915157004132e-13,
                ),
            ),
            anisotropy=10.7,
        ),
        _mutch2019(
            "mutch2019_mn",
            "Mn",
            "Mn diffusion",
            (-7.548, 0.1963, -7.153, -26720, -9.504e-10, 7.195e-07),
            (
                (
                    3.23909781874242,
                    0.00394323922982214,
                    -0.678808849146587,
                    -3680.51039678914,
                    -1.94956354613448e-10,
                    2.68546039329982e-07,
                ),
                (
                    0.00394323922982214,
                    0.00347810091754163,
                    0.002784587490056,
                    119.28822156565,
                    -4.0328467747652e-12,
                    -2.03960098719852e-09,
                ),
                (
                    -0.678808849146587,
                    0.002784587490056,
                    0.323011631927977,
                    726.306288511926,
                    2.81952079772358e-11,
                    -5.3690017575469e-08,
                ),
                (
                    -3680.51039678914,
                    119.28822156565,
                    726.306288511926,
                    8787975.0418876,
                    9.60676117507748e-08,
                    -0.00039878531957198,
                ),
                (
                    -1.94956354613448e-10,
                    -4.0328467747652e-12,
                    2.81952079772358e-11,
                    9.60676117507748e-08,
                    2.83349563026499e-19,
                    -4.64960532776102e-16,
                ),
                (
                    2.68546039329982e-07,
                    -2.03960098719852e-09,
                    -5.3690017575469e-08,
                    -0.00039878531957198,
                    -4.64960532776102e-16,
                    7.86972428160841e-13,
                ),
            ),
        ),
        ArrheniusDiffusivity(
            name="girona_costa2013_mn",
            element="Mn",
            log10_D0_m2_s=(-9.21 - LOG10_6, -9.21 - LOG10_6, -9.21),
            activation_energy_J_mol=(201_000.0,) * 3,
            fO2_exponent=1 / 6,
            fO2_ref_Pa=1e-7,
            fo_coefficient=3.0,
            fo_ref=0.9,
            activation_volume_m3_mol=7e-6,
            P_ref_Pa=1e5,
            source=_DIPRA_TABLE_1,
        ),
        ArrheniusDiffusivity(
            name="girona_costa2013_ni",
            element="Ni",
            log10_D0_m2_s=(math.log10(3.84e-9) - LOG10_6,) * 2 + (math.log10(3.84e-9),),
            activation_energy_J_mol=(220_000.0,) * 3,
            fO2_exponent=1 / 4.25,
            fO2_ref_Pa=1e-6,
            fo_coefficient=1.5,
            fo_ref=0.9,
            activation_volume_m3_mol=7e-6,
            P_ref_Pa=1e5,
            source=_DIPRA_TABLE_1,
        ),
        ArrheniusDiffusivity(
            name="petry2004_ni",
            element="Ni",
            log10_D0_m2_s=(math.log10(3.84e-9) - LOG10_6,) * 2 + (math.log10(3.84e-9),),
            activation_energy_J_mol=(216_000.0,) * 3,
            fO2_exponent=1 / 4.25,
            fO2_ref_Pa=1e-6,
            fo_coefficient=1.5,
            fo_ref=0.9,
            activation_volume_m3_mol=7e-6,
            P_ref_Pa=1e5,
            source=(
                "Petry, Chakraborty and Palme (2004), GCA 68, 4179-4188: pre-factor and "
                "E = 216 kJ/mol of their constant-fO2 fit (p. 4184), a consistent pair. "
                f"girona_costa2013_ni ({_DIPRA_TABLE_1}) pairs this pre-factor with "
                "E = 220 kJ/mol from their all-data fit, 0.14 log units slower at 1200 C. "
                "Anisotropy and activation volume as in DIPRA; Petry measured at 1 atm "
                "and fO2 up to 1e-4 Pa."
            ),
        ),
        ArrheniusDiffusivity(
            name="girona_costa2013_ca",
            element="Ca",
            log10_D0_m2_s=(math.log10(16.59e-12), math.log10(34.67e-12), math.log10(95.49e-12)),
            activation_energy_J_mol=(193_000.0, 201_000.0, 207_000.0),
            fO2_exponent=0.3,
            fO2_ref_Pa=1e-7,
            source=f"{_DIPRA_TABLE_1} (no Fo or pressure dependence)",
        ),
    )
}
DEFAULT_DIFFUSIVITY = DOHMEN_CHAKRABORTY_2007.name

# The Fe-Mg equations offered by name in tables, the CLI and the GUI: a short alias and a
# display label for each preset. The preset names keep working everywhere.
FE_MG_CHOICES: dict[str, tuple[str, str]] = {
    "dipra": (DEFAULT_DIFFUSIVITY, "Classic DIPRA (Dohmen & Chakraborty 2007)"),
    "refit": ("femg_refit2026", "FORSTER refit (2026)"),
    "dfens_tamed": ("mutch2019_tamed", "DFENS TaMED (Mutch et al. 2019)"),
    "dfens_global": ("mutch2019_global", "DFENS Global (Mutch et al. 2019)"),
}
ALIASES = {alias: name for alias, (name, _) in FE_MG_CHOICES.items()}
LABELS = {name: label for name, label in FE_MG_CHOICES.values()}


def label(diffusivity: Diffusivity) -> str:
    """The display label of an offered preset, otherwise the equation's name."""
    return LABELS.get(diffusivity.name, diffusivity.name)


def resolve(
    diffusivity: Diffusivity | str,
    custom: Mapping[str, Diffusivity] | None = None,
) -> Diffusivity:
    """A diffusivity object, looking up a name (or an alias in `ALIASES`) among the presets
    and ``custom``."""
    if not isinstance(diffusivity, str):
        return diffusivity
    name = ALIASES.get(diffusivity, diffusivity)
    available = {**DIFFUSIVITY_PRESETS, **(custom or {})}
    if name not in available:
        raise ValueError(
            f"unknown diffusivity {diffusivity!r}; choices are {', '.join(ALIASES)}, presets "
            f"are {', '.join(DIFFUSIVITY_PRESETS)}"
            + (f"; from the table: {', '.join(custom)}" if custom else "")
        )
    return available[name]


# Columns of a diffusivities table (``forster template`` writes one with the presets).
TABLE_COLUMNS = (
    "name",
    "element",
    "log10_D0_100",
    "log10_D0_010",
    "log10_D0_001",
    "E_100_J_mol",
    "E_010_J_mol",
    "E_001_J_mol",
    "fO2_exponent",
    "fO2_ref_Pa",
    "fo_coefficient",
    "fo_ref",
    "activation_volume_m3_mol",
    "P_ref_Pa",
    "pressure_coefficient_per_Pa",
    "source",
)
_SCALARS = TABLE_COLUMNS[8:15]


def to_row(diffusivity: ArrheniusDiffusivity) -> dict[str, Any]:
    """One table row; `from_row` reads it back as an equal equation."""
    d = diffusivity
    return {
        "name": d.name,
        "element": d.element,
        **dict(zip(TABLE_COLUMNS[2:5], d.log10_D0_m2_s, strict=True)),
        **dict(zip(TABLE_COLUMNS[5:8], d.activation_energy_J_mol, strict=True)),
        **{c: getattr(d, c) for c in _SCALARS},
        "source": d.source,
    }


def from_row(row: Mapping[str, Any]) -> ArrheniusDiffusivity:
    """An equation from one table row (every column but ``source`` is required)."""
    name = str(row.get("name", "")).strip()
    missing = [c for c in TABLE_COLUMNS[:-1] if c not in row or pd.isna(row[c])]
    if missing:
        raise ValueError(f"diffusivity {name or '(no name)'!r}: missing {', '.join(missing)}")
    source = row.get("source")
    return ArrheniusDiffusivity(
        name=name,
        element=str(row["element"]).strip(),
        log10_D0_m2_s=tuple(float(row[c]) for c in TABLE_COLUMNS[2:5]),
        activation_energy_J_mol=tuple(float(row[c]) for c in TABLE_COLUMNS[5:8]),
        fO2_exponent=float(row["fO2_exponent"]),
        fO2_ref_Pa=float(row["fO2_ref_Pa"]),
        fo_coefficient=float(row["fo_coefficient"]),
        fo_ref=float(row["fo_ref"]),
        activation_volume_m3_mol=float(row["activation_volume_m3_mol"]),
        P_ref_Pa=float(row["P_ref_Pa"]),
        pressure_coefficient_per_Pa=float(row["pressure_coefficient_per_Pa"]),
        source="" if source is None or pd.isna(source) else str(source),
    )


def diffusivity_table(diffusivities: Iterable[ArrheniusDiffusivity]) -> pd.DataFrame:
    """A diffusivities table with one row per equation."""
    return pd.DataFrame([to_row(d) for d in diffusivities]).reindex(columns=list(TABLE_COLUMNS))


def from_table(
    df: pd.DataFrame, origin: str = "diffusivities table"
) -> dict[str, ArrheniusDiffusivity]:
    """Equations of a diffusivities table, by name.

    A row may repeat a preset only with the preset's own values (as in the table
    ``forster template`` writes); it then stands for the preset itself.
    """
    missing = [c for c in TABLE_COLUMNS[:-1] if c not in df.columns]
    if missing:
        raise ValueError(f"{origin}: missing column(s) {', '.join(missing)}")
    out: dict[str, ArrheniusDiffusivity] = {}
    for _, row in df[df["name"].notna()].iterrows():
        equation = from_row(row)
        if equation.name in out:
            raise ValueError(f"{origin}: diffusivity {equation.name!r} is defined twice")
        if equation.name in ALIASES:
            raise ValueError(
                f"{origin}: {equation.name!r} names the preset {ALIASES[equation.name]}; give "
                "your equation another name"
            )
        preset = DIFFUSIVITY_PRESETS.get(equation.name)
        if preset is not None:
            if {**to_row(equation), "source": ""} != {**to_row(preset), "source": ""}:
                raise ValueError(
                    f"{origin}: {equation.name!r} is a preset name; give your equation another name"
                )
            out[equation.name] = preset
        else:
            out[equation.name] = equation
    return out
