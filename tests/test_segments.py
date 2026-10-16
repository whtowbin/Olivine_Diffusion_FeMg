"""Starting profiles as named segments, the old half-profile names, and the crystal
centre (docs/plans/2026-10-16-geometry-naming.md).

The renaming itself is checked by a differential replay outside the suite: every run of
the template example, the paper and the round robin refits identically under the new
names (scratchpad naming/replay.py, 2026-10-16). These tests pin the new behaviour.
"""

import math
import warnings

import numpy as np
import pandas as pd
import pytest

from forster import (
    InitialCondition,
    Profile,
    Uncertainty,
    fit_profile,
    fit_run,
    read_profiles,
    read_runs,
    write_template,
)
from forster.boundary import Stage
from forster.implicit import stepper
from forster.io import RenamedColumnWarning, run_row
from forster.scan import candidate_grid, parameter_names


@pytest.fixture(scope="module")
def example(tmp_path_factory):
    out = tmp_path_factory.mktemp("template")
    write_template(out)
    row = pd.read_csv(out / "parameters.csv").iloc[0].to_dict()
    return row, read_profiles(out / "profiles.csv")


def _read(tmp_path, row, **kwargs):
    path = tmp_path / "parameters.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    return read_runs(path, **kwargs)[0]


def test_old_column_names_read_as_segments_with_a_warning(tmp_path, example):
    row, _ = example
    old = {k: v for k, v in row.items() if k != "segment_fo"}
    old |= {"edge_fo": 0.9146, "core_fo": 0.8985, "rim_fo": 0.912, "edge_fo_sd": 0.002}
    new = dict(row, edge1_fo=0.912, segment_fo_sd="0.002, ")
    with pytest.warns(RenamedColumnWarning, match="edge_fo -> segment_fo"):
        from_old = _read(tmp_path, old)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RenamedColumnWarning)
        from_new = _read(tmp_path, new)
    assert from_old == from_new
    assert from_new.uncertainties == {"a_fo": Uncertainty(0.002)}
    with pytest.raises(ValueError, match="not both"):
        _read(tmp_path, dict(new, edge_fo=0.9146))


def test_named_segments_round_trip_and_fit(tmp_path, example):
    row, profiles = example
    rich = dict(
        row,
        steps_um="10, 24, 40",
        segment_fo="0.914, 0.91, 0.905, 0.8985",
        segment_names="rim, m1, m2, core",
        segment_fo_sd="0.002, , , ",
        segment_fo_range=", , 0.003, ",
    )
    run = _read(tmp_path, rich)
    assert run.initial.segment_names == ("rim", "m1", "m2", "core")
    assert run.initial.step_fo is None  # only a three-segment profile has a middle one
    assert run.uncertainties == {
        "rim_fo": Uncertainty(0.002),
        "m2_fo": Uncertainty(0.003, "uniform"),
    }
    back = _read(tmp_path, run_row(run))
    assert back == run
    assert fit_run(run, profiles).time_days > 0


def test_parameter_order_keeps_the_old_edge_core_plateau_order():
    """Steps, first segment, last, then the middle ones: for two and three segments the
    order of the old edge_fo, core_fo, step_fo, so earlier fits are unchanged."""
    three = InitialCondition((0, 100), 0.82, 0.88, steps_um=(20, 25), step_fo=0.85)
    assert parameter_names(three) == ("step1_um", "step2_um", "a_fo", "c_fo", "b_fo")
    four = InitialCondition((0, 100), steps_um=(10, 20, 30), segment_fo=(0.8, 0.82, 0.84, 0.86))
    assert parameter_names(four)[3:] == ("a_fo", "d_fo", "b_fo", "c_fo")
    grid = candidate_grid(three, {"core_fo": Uncertainty(0.01), "step_fo": Uncertainty(0.01)})
    assert grid.names == ("c_fo", "b_fo")
    with pytest.raises(ValueError, match="no parameter"):
        candidate_grid(four, {"step_fo": Uncertainty(0.01)})  # no middle with four segments


def test_bad_segments_name_the_problem():
    with pytest.raises(ValueError, match="one more Fo than steps"):
        InitialCondition((0, 100), steps_um=(10, 20), segment_fo=(0.8, 0.9))
    with pytest.raises(ValueError, match="distinct"):
        InitialCondition((0, 100), steps_um=(10,), segment_fo=(0.8, 0.9), segment_names=("a", "a"))
    with pytest.raises(ValueError, match="distinct"):
        InitialCondition(
            (0, 100), steps_um=(10,), segment_fo=(0.8, 0.9), segment_names=("a", "step2")
        )
    with pytest.raises(ValueError, match="centre_um"):
        InitialCondition((0, 100), 0.8, 0.9, steps_um=(10,), centre_um=50)


def test_free_far_end_matches_the_slab_solution():
    """No flux at the centre: a slab held at the rim, constant D, against the Fourier
    series c = c_rim + (c0 - c_rim) sum 4/(k pi) sin(k pi x / 2L) exp(-(k pi / 2L)^2 D t),
    k odd. The centre is free, so its Fo falls."""
    D, L = 1e-16, 100e-6
    t = 0.2 * L**2 / D
    x_um = np.linspace(0.0, L * 1e6, 401)
    advance = stepper(lambda c: np.full_like(c, D), 0.80, None, t / 800, x_um)
    c = advance(np.full(x_um.size, 0.90), 800)[-1]
    x = x_um * 1e-6
    k = 2 * np.arange(200) + 1
    series = (4 / (k * math.pi))[:, None] * np.sin(k[:, None] * math.pi * x / (2 * L))
    exact = 0.80 + 0.10 * (series * np.exp(-((k * math.pi / (2 * L)) ** 2) * D * t)[:, None]).sum(0)
    assert np.max(np.abs(c - exact)) < 2e-5
    assert c[-1] < 0.895  # the centre has changed: it is not held


def test_a_distant_centre_fits_like_a_half_profile(example):
    """With the centre far beyond a long plateau, diffusion never reaches it, so a
    mirrored crystal and a half profile with the core held give the same time."""
    row, profiles = example
    profile = profiles[row["profile"]]
    run = read_runs_row(row)
    half = fit_run(run, profiles)
    initial = InitialCondition(
        run.initial.domain_um,
        steps_um=run.initial.steps_um,
        segment_fo=run.initial.segment_fo,
        centre_um=400.0,
    )
    grid = half.inputs["grid"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mirrored = fit_profile(
            profile, run.conditions, run.orientation, initial, grid, None, run.diffusivity
        )
    assert mirrored.time_days == pytest.approx(half.time_days, rel=0.01)
    assert mirrored.model_x_um[-1] == 400.0
    assert mirrored.core_end_lengths is None  # no short-plateau check when mirrored
    assert not any("past the last step" in str(w.message) for w in caught)


def test_a_centre_needs_the_implicit_scheme_and_no_stages(example):
    row, profiles = example
    run = read_runs_row(row)
    profile = profiles[row["profile"]]
    initial = InitialCondition(
        run.initial.domain_um, 0.9146, 0.8985, steps_um=run.initial.steps_um, centre_um=100.0
    )
    grid = fit_run(run, profiles).inputs["grid"]
    args = (profile, run.conditions, run.orientation, initial, grid, None, run.diffusivity)
    with pytest.raises(ValueError, match=r"needs scheme=.implicit"):
        fit_profile(*args, "central")
    with pytest.raises(ValueError, match="growth stages"):
        fit_profile(*args, "implicit", stages=(Stage(86400.0, rim_um=1.0, zone_value=0.915),))


def test_points_past_the_domain_end_are_not_fitted_when_mirrored(example):
    """The model runs on to the centre, but only points up to domain_um[1] are fitted."""
    row, profiles = example
    run = read_runs_row(row)
    profile = profiles[row["profile"]]
    lo, hi = run.initial.domain_um
    initial = InitialCondition(
        (lo, hi - 20), 0.9146, 0.8985, steps_um=run.initial.steps_um, centre_um=hi + 50
    )
    grid = fit_run(run, profiles).inputs["grid"]
    fit = fit_profile(
        profile, run.conditions, run.orientation, initial, grid, None, run.diffusivity
    )
    assert fit.fit_x_um.max() <= hi - 20
    assert Profile(fit.fit_x_um, fit.fit_fo).distance_um.size < profile.distance_um.size


def read_runs_row(row):
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "p.csv"
        pd.DataFrame([row]).to_csv(path, index=False)
        return read_runs(path)[0]
