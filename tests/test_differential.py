"""The package gives exactly what the round robin script's model_diffusion gave.

Both sides run the same engine on the same inputs: the reference reads the
parameter row inside model_diffusion (verbatim copy in tests/reference), the new
side goes through read_runs, read_profiles and fit_run. The fits are
deterministic, so any difference at all is a bug. Full model curves are compared
by SHA-256 to keep memory down (one run holds about 250 MB of curves).
"""

import hashlib

import numpy as np
import pandas as pd
import pytest
from conftest import HAS_ROUND_ROBIN, ROUND_ROBIN_DATA, ROUND_ROBIN_MISSING, ROUND_ROBIN_PARAMS
from reference import round_robin_model_diffusion as ref

from forster import KrigingSettings, RunDefaults, fit_run, read_profiles, read_runs

# The settings the round robin batch loop passed to model_diffusion.
BATCH_KWARGS = {
    "Total_time_days": 2 * 365,
    "kriging_variogram_model": "gaussian",
    "kriging_variogram_parameters": {
        "sill": (10 / 100) ** 2,
        "range": 25,
        "nugget": (0.2 / 100) ** 2,
    },
    "uniform_uncert": 0.001,
}
BATCH_DEFAULTS = RunDefaults(
    max_time_days=BATCH_KWARGS["Total_time_days"],
    kriging=KrigingSettings(
        variogram_model=BATCH_KWARGS["kriging_variogram_model"],
        variogram_parameters=BATCH_KWARGS["kriging_variogram_parameters"],
        uniform_uncertainty=BATCH_KWARGS["uniform_uncert"],
    ),
    uncertainty="legacy",  # the reference is the old kriged-grid chi-squared rule
)

if not HAS_ROUND_ROBIN:
    pytest.skip(ROUND_ROBIN_MISSING, allow_module_level=True)

PARAMS = pd.read_excel(ROUND_ROBIN_PARAMS, engine="openpyxl")
DATA = pd.read_excel(ROUND_ROBIN_DATA, engine="openpyxl")
SAMPLES = list(PARAMS.Sample.dropna().unique())


def fingerprint(time, indices, sum_chi2, model_x, curves, interp_fo, interp_sigma):
    return {
        "time": np.asarray(time),
        "indices": tuple(int(i) for i in indices),
        "sum_chi2": np.asarray(sum_chi2),
        "model_x": np.asarray(model_x),
        "interp_fo": np.asarray(interp_fo),
        "interp_sigma": np.asarray(interp_sigma),
        "curves_shape": curves.shape,
        # .data hashes the buffer in place (tobytes() made a full copy of the largest array)
        "curves_sha256": hashlib.sha256(np.ascontiguousarray(curves).data).hexdigest(),
    }


def reference_fit(sample):
    ref.ol_param_db = PARAMS
    out = ref.model_diffusion(sample, data_db=DATA, parameter_db=PARAMS, **BATCH_KWARGS)
    return fingerprint(
        out["time"],
        out["idx_min"],
        out["sum_chi2"],
        out["model_x"],
        out["Fo_diffusion_results"],
        out["Y_interp"],
        out["Y_interp_std"],
    )


def package_fit(sample):
    runs = {run.name: run for run in read_runs(ROUND_ROBIN_PARAMS, defaults=BATCH_DEFAULTS)}
    result = fit_run(runs[sample], read_profiles(ROUND_ROBIN_DATA))
    time = (result.time_s, result.time_minus_s, result.time_plus_s)
    return fingerprint(
        time,
        result.indices,
        result.sum_chi2,
        result.model_x_um,
        result.curves,
        result.interp_fo,
        result.interp_fo_sigma,
    )


def assert_same(a, b):
    assert a.keys() == b.keys()
    for key in a:
        if isinstance(a[key], np.ndarray):
            np.testing.assert_array_equal(a[key], b[key], err_msg=key)
        else:
            assert a[key] == b[key], key


def test_package_reads_the_same_runs():
    names = [run.name for run in read_runs(ROUND_ROBIN_PARAMS, defaults=BATCH_DEFAULTS)]
    assert names == SAMPLES
    assert len(names) == 17


@pytest.mark.parametrize("sample", SAMPLES)
def test_package_matches_reference_exactly(sample):
    assert_same(package_fit(sample), reference_fit(sample))


@pytest.mark.parametrize("sample", SAMPLES[:3])
def test_reference_is_deterministic(sample):
    # Control: if the reference were not repeatable, exact equality above would mean nothing.
    assert_same(reference_fit(sample), reference_fit(sample))
