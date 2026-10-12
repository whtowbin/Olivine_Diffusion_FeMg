"""FORSTER: Fitting Olivine Residence & Storage Timescales from Elemental Relaxation.

Fe-Mg diffusion modeling of zoned olivine profiles.
"""

from forster import dipra, uncertainty
from forster.io import RunDefaults, fit_run, read_profiles, read_runs
from forster.model import (
    SCHEMES,
    UNCERTAINTY_METHODS,
    Conditions,
    FitResult,
    Grid,
    InitialCondition,
    KrigingSettings,
    Orientation,
    Profile,
    fit_profile,
)
from forster.uncertainty import NoiseModel

__all__ = [
    "SCHEMES",
    "UNCERTAINTY_METHODS",
    "Conditions",
    "FitResult",
    "Grid",
    "InitialCondition",
    "KrigingSettings",
    "NoiseModel",
    "Orientation",
    "Profile",
    "RunDefaults",
    "dipra",
    "fit_profile",
    "fit_run",
    "read_profiles",
    "read_runs",
    "uncertainty",
]
