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

    def functions(self, conditions: Conditions, orientation: Orientation) -> DiffusivityFunctions:
        """D(X_Fo) along the profile and dD/dX_Fo, for the model's time steppers."""
        cos2 = (
            np.cos(np.deg2rad([orientation.alpha_deg, orientation.beta_deg, orientation.gamma_deg]))
            ** 2
        )
        # Everything but the composition term is constant during a run.
        at_ref = float(self.axis_diffusivities(conditions, self.fo_ref) @ cos2)
        slope = -self.fo_coefficient * math.log(10)

        def diffusivity(x_fo: np.ndarray) -> np.ndarray:
            return at_ref * 10.0 ** (self.fo_coefficient * (self.fo_ref - np.asarray(x_fo)))

        def derivative(x_fo: np.ndarray) -> np.ndarray:
            return slope * diffusivity(x_fo)

        return diffusivity, derivative


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
            return dipra.D_Fo_and_derivative(
                conditions.T_K,
                conditions.P_Pa,
                conditions.fO2_Pa,
                orientation.alpha_deg,
                orientation.beta_deg,
                orientation.gamma_deg,
                E[0],
            )
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


def _mutch2019(
    name: str, data: str, a: float, b: float, c: float, q: float, j: float, h: float
) -> ArrheniusDiffusivity:
    """Mutch et al. (2019) regression ln D[001] = a + b ln fO2(bar) + c X_Fo + q/T + j P + h P/T."""
    ln10 = math.log(10)
    return ArrheniusDiffusivity(
        name=name,
        element="Fe-Mg",
        log10_D0_m2_s=(a / ln10 - LOG10_6, a / ln10 - LOG10_6, a / ln10),
        activation_energy_J_mol=(-q * R_J_MOL_K,) * 3,
        fO2_exponent=b,
        fO2_ref_Pa=1e5,  # fO2 in bar in the regression
        fo_coefficient=-c / ln10,
        fo_ref=0.0,
        activation_volume_m3_mol=-h * R_J_MOL_K,
        P_ref_Pa=0.0,
        pressure_coefficient_per_Pa=j / ln10,
        source=(
            "Mutch et al. (2019), Nature Geoscience 12, 569-574, Supplementary Table 1 "
            f"({name.removeprefix('mutch2019_')}: regressed on {data}), as printed to 3 "
            "figures; [001], with [100] and [010] 6x slower as in DFENS (Mutch et al. 2021). "
            "Covariance: Supplementary Table 2."
        ),
    )


DIFFUSIVITY_PRESETS: dict[str, ArrheniusDiffusivity] = {
    d.name: d
    for d in (
        DOHMEN_CHAKRABORTY_2007,
        _mutch2019(
            "mutch2019_tamed",
            "the TaMED-regime Fe-Mg data only",
            -6.76,
            0.224,
            -7.18,
            -26700,
            -5.21e-10,
            -1.03e-7,
        ),
        _mutch2019(
            "mutch2019_global",
            "all the Fe-Mg data, TaMED and PED",
            -7.86,
            0.187,
            -7.21,
            -26600,
            -4.15e-10,
            -1.54e-7,
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


def resolve(
    diffusivity: Diffusivity | str,
    custom: Mapping[str, Diffusivity] | None = None,
) -> Diffusivity:
    """A diffusivity object, looking up a name among the presets and ``custom``."""
    if not isinstance(diffusivity, str):
        return diffusivity
    available = {**DIFFUSIVITY_PRESETS, **(custom or {})}
    if diffusivity not in available:
        raise ValueError(
            f"unknown diffusivity {diffusivity!r}; presets are {', '.join(DIFFUSIVITY_PRESETS)}"
            + (f"; from the table: {', '.join(custom)}" if custom else "")
        )
    return available[diffusivity]


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
        **{c: float(row[c]) for c in _SCALARS},
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
