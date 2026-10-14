"""The compiled (Numba) steppers (`forster.compiled`) against the NumPy ones they replace.

Agreement is to rounding, not bit for bit, so fits are compared by their fitted time
(a whole time step) and the profiles to 1e-12 Fo.
"""

from dataclasses import replace

import numpy as np
import pytest
from conftest import ROUND_ROBIN_DATA, ROUND_ROBIN_PARAMS

from forster import (
    DIFFUSIVITY_PRESETS,
    Conditions,
    Orientation,
    RunDefaults,
    compiled,
    fit_run,
    read_profiles,
    read_runs,
)
from forster.boundary import RimPath
from forster.grid import GridRequest
from forster.implicit import cell_average, node_grid, stepper
from forster.model import _stepper

pytestmark = pytest.mark.skipif(not compiled.AVAILABLE, reason="Numba is not installed")

CONDITIONS = Conditions(T_C=1200, P_Pa=4.2e7, fO2_Pa=1.4e-3)
ALONG_001 = Orientation(90, 90, 0)


@pytest.fixture(params=["dohmen_chakraborty2007", "mutch2019_tamed"])
def law(request):
    return DIFFUSIVITY_PRESETS[request.param].functions(CONDITIONS, ALONG_001)


@pytest.mark.parametrize("scheme", ["central", "dipra"])
def test_explicit_kernels_match_numpy(law, scheme):
    D, dD = law
    x = np.linspace(0.0, 150.0, 151)
    dx = 1e-6
    dt = 0.4 * dx**2 / float(D(np.array([0.82]))[0])
    start = np.where(x < 20.3, 0.82, 0.88)
    fast = _stepper(scheme, D, dD, (0.82, 0.88), dt, dx)
    slow = _stepper(scheme, D, dD, (0.82, 0.88), dt, dx, fast=False)
    assert fast is not slow
    a, b = fast(start, 2000), slow(start, 2000)
    assert np.abs(a - b).max() < 1e-12
    blocks = np.concatenate([fast(start, 700), fast(fast(start, 700)[-1], 1300)[1:]])
    assert np.array_equal(blocks, a)


def test_implicit_kernel_matches_numpy_with_a_changing_rim(law):
    D, _ = law
    x = node_grid([0.0, 40.3, 150.0], 2.0)  # uneven spacing next to 40.3
    rim = RimPath((0.0, 2.0, 5.0), (0.82, 0.86, 0.84))
    start = cell_average(x, [20.3], [0.82, 0.88])
    fast = compiled.implicit_stepper(D, rim, 0.88, 3600.0, x)
    assert fast is not None
    slow = stepper(D, rim, 0.88, 3600.0, x)
    a, b = fast(start, 200), slow(start, 200)
    assert np.abs(a - b).max() < 1e-12
    rest = fast(a[80], 120, first_row=80)
    assert np.array_equal(np.concatenate([fast(start, 80), rest[1:]]), a)


def test_a_custom_diffusivity_falls_back_to_numpy():
    def D(x):
        return 1e-17 * (1.0 + np.asarray(x))  # not of the log-linear form

    assert compiled.explicit_stepper("central", D, (0.8, 0.9), 1.0, 1e-6) is None
    assert compiled.implicit_stepper(D, 0.8, 0.9, 1.0, np.linspace(0, 10, 11)) is None


@pytest.mark.parametrize("scheme", ["implicit", "central"])
def test_fits_are_the_same_with_and_without_numba(monkeypatch, scheme):
    runs = {r.name: r for r in read_runs(ROUND_ROBIN_PARAMS, defaults=RunDefaults(730))}
    profiles = read_profiles(ROUND_ROBIN_DATA)
    for name in ("KS20-527_2-transect-1", "KS20-527-8_transect-2"):
        run = replace(runs[name], scheme=scheme)
        if scheme == "implicit":
            run = replace(run, grid=GridRequest())
        fast = fit_run(run, profiles)
        monkeypatch.setattr(compiled, "_kernels", None)
        slow = fit_run(run, profiles)
        monkeypatch.undo()
        assert fast.indices == slow.indices
        assert fast.time_s == slow.time_s
        assert np.abs(fast.curves[fast.indices[0]] - slow.curves[slow.indices[0]]).max() < 1e-10
