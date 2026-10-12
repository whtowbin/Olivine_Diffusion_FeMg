"""FORSTER: Fitting Olivine Residence & Storage Timescales from Elemental Relaxation.

Fe-Mg diffusion modeling of zoned olivine profiles.
"""

from forster import dipra
from forster.io import RunDefaults, fit_run, read_profiles, read_runs
from forster.model import (
    SCHEMES,
    Conditions,
    FitResult,
    Grid,
    InitialCondition,
    KrigingSettings,
    Orientation,
    Profile,
    fit_profile,
)

__all__ = [
    "SCHEMES",
    "Conditions",
    "FitResult",
    "Grid",
    "InitialCondition",
    "KrigingSettings",
    "Orientation",
    "Profile",
    "RunDefaults",
    "dipra",
    "fit_profile",
    "fit_run",
    "read_profiles",
    "read_runs",
]
