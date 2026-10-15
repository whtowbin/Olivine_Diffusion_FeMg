"""FORSTER: Fitting Olivine Residence & Storage Timescales from Elemental Relaxation.

Fe-Mg diffusion modeling of zoned olivine profiles.
"""

from forster import boundary, diffusivity, dipra, elements, scan, uncertainty
from forster.boundary import BoundaryHistory, Stage
from forster.diffusivity import (
    DIFFUSIVITY_PRESETS,
    ArrheniusDiffusivity,
    Diffusivity,
    LawUncertainty,
)
from forster.elements import ElementInitial, ElementProfile, JointFit, fit_element, fit_joint
from forster.grid import GridRequest, check_grid, choose_grid, stable_dt
from forster.io import (
    RunDefaults,
    fit_run,
    fit_tables,
    read_diffusivities,
    read_profiles,
    read_runs,
    scan_run,
    scan_summary,
    scan_table,
    summary_row,
)
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
from forster.plotting import plot_fit
from forster.sampling import ICPosterior, fit_initial_condition
from forster.scan import Uncertainty, candidate_grid
from forster.template import write_template
from forster.uncertainty import NoiseModel

__all__ = [
    "DIFFUSIVITY_PRESETS",
    "SCHEMES",
    "UNCERTAINTY_METHODS",
    "ArrheniusDiffusivity",
    "BoundaryHistory",
    "Conditions",
    "Diffusivity",
    "ElementInitial",
    "ElementProfile",
    "FitResult",
    "Grid",
    "GridRequest",
    "ICPosterior",
    "InitialCondition",
    "JointFit",
    "KrigingSettings",
    "LawUncertainty",
    "NoiseModel",
    "Orientation",
    "Profile",
    "RunDefaults",
    "Stage",
    "Uncertainty",
    "boundary",
    "candidate_grid",
    "check_grid",
    "choose_grid",
    "diffusivity",
    "dipra",
    "elements",
    "fit_element",
    "fit_initial_condition",
    "fit_joint",
    "fit_profile",
    "fit_run",
    "fit_tables",
    "plot_fit",
    "read_diffusivities",
    "read_profiles",
    "read_runs",
    "scan",
    "scan_run",
    "scan_summary",
    "scan_table",
    "stable_dt",
    "summary_row",
    "uncertainty",
    "write_template",
]
