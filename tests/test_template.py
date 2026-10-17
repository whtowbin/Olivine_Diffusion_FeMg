"""The template parameter layout, the packaged example and the summary table."""

import warnings
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from conftest import PAPER_DATA, PAPER_PARAMS, REPO_ROOT

from forster import (
    GridRequest,
    InitialCondition,
    Orientation,
    fit_run,
    read_profiles,
    read_runs,
    summary_row,
)
from forster.io import _steps, _steps_text
from forster.template import (
    PARAMETER_COLUMNS,
    PROFILE_COLUMNS,
    SUMMARY_COLUMNS,
    column_table,
    write_template,
)

EXAMPLE = "AZ18 WHT01_Ol_63_prof_1"


@pytest.fixture(scope="module")
def example(tmp_path_factory):
    out = tmp_path_factory.mktemp("template")
    write_template(out)
    return out


@pytest.fixture
def example_row(example):
    return pd.read_csv(example / "parameters.csv").iloc[0].to_dict()


def _read_row(tmp_path, row):
    path = tmp_path / "parameters.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    return read_runs(path)[0]


def test_example_matches_the_paper_row_and_profile_it_was_taken_from(example):
    (run,) = read_runs(example / "parameters.csv")
    paper_run = {r.name: r for r in read_runs(PAPER_PARAMS)}[EXAMPLE]
    assert (run.name, run.profile_name) == (EXAMPLE, EXAMPLE)
    assert run.conditions == paper_run.conditions  # fO2 from the QFM buffer
    assert run.orientation == paper_run.orientation  # from the Euler angles
    assert run.initial == paper_run.initial
    assert run.grid == GridRequest(max_time_days=40)  # chosen by forster.grid

    profiles = read_profiles(example / "profiles.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # the paper table has short profiles
        paper_profiles = read_profiles(PAPER_DATA, sheet="Olivine")
    assert list(profiles) == [EXAMPLE]
    np.testing.assert_array_equal(
        profiles[EXAMPLE].distance_um, paper_profiles[EXAMPLE].distance_um
    )
    np.testing.assert_array_equal(profiles[EXAMPLE].fo, paper_profiles[EXAMPLE].fo)

    result = fit_run(run, profiles)
    paper_result = fit_run(replace(paper_run, grid=run.grid, scheme=run.scheme), paper_profiles)
    assert result.indices == paper_result.indices
    assert result.time_s == paper_result.time_s


def test_angle_and_pa_forms_give_the_same_run_as_euler_and_buffer(tmp_path, example_row):
    euler = _read_row(tmp_path, example_row)
    angles = dict(example_row, phi1_deg=None, Phi_deg=None, phi2_deg=None, profile_angle_deg=None)
    angles.update(
        alpha_deg=euler.orientation.alpha_deg,
        beta_deg=euler.orientation.beta_deg,
        gamma_deg=euler.orientation.gamma_deg,
        fO2_buffer=None,
        fO2_delta_log=None,
        fO2_Pa=euler.conditions.fO2_Pa,
    )
    direct = _read_row(tmp_path, angles)
    assert direct.orientation == euler.orientation
    assert direct.conditions == euler.conditions
    assert euler.orientation == Orientation.from_euler(77.0, 61.2, 334.2, -80.0)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"fO2_Pa": 1e-3}, r"give fO2 as \(fO2_Pa\) or \(fO2_buffer, fO2_delta_log\), not both"),
        ({"fO2_delta_log": None}, "fO2 is missing fO2_delta_log"),
        ({"fO2_buffer": None, "fO2_delta_log": None}, r"give fO2 as \(fO2_Pa\) or"),
        ({"alpha_deg": 30}, "orientation as .* not both"),
        ({"profile_angle_deg": None}, "orientation is missing profile_angle_deg"),
        ({"fO2_buffer": "IW"}, "unknown fO2 buffer 'IW'"),
        ({"steps_um": "10; 33"}, "step positions '10; 33'"),
        ({"steps_um": "10, 33"}, "give one more Fo than steps"),
        ({"uncertainty": "bayes"}, "uncertainty 'bayes'"),
    ],
)
def test_bad_template_rows_name_the_problem(tmp_path, example_row, change, message):
    with pytest.raises(ValueError, match=message):
        _read_row(tmp_path, dict(example_row, **change))


def test_two_steps_with_a_plateau(tmp_path, example_row):
    run = _read_row(
        tmp_path, dict(example_row, steps_um="10, 33.5", segment_fo="0.9146, 0.93, 0.8985")
    )
    assert run.initial.steps_um == (10, 33.5)
    assert run.initial.step_fo == 0.93


@pytest.mark.parametrize("steps", [(10.0, 33.0), (24.0,), (0.5, 23.25)])
def test_step_text_reads_back_exactly(steps):
    assert _steps(_steps_text(steps)) == steps


def test_initial_condition_needs_step_fo_with_two_steps():
    with pytest.raises(ValueError, match="two with step_fo"):
        InitialCondition((0, 50), 0.9, 0.85, steps_um=(10, 20))
    with pytest.raises(ValueError, match="3 step positions"):
        InitialCondition((0, 50), 0.9, 0.85, steps_um=(10, 20, 30))


def test_summary_reads_back_as_the_same_runs(tmp_path, example):
    (run,) = read_runs(example / "parameters.csv")
    result = fit_run(run, read_profiles(example / "profiles.csv"))
    row = summary_row(result)
    assert row["time_days"] == result.time_days
    assert row["low_days"] <= row["time_days"] <= row["high_days"]
    documented = {c.name for c in (*PARAMETER_COLUMNS, *SUMMARY_COLUMNS)}
    assert set(row) - documented == set()
    rerun = _read_row(tmp_path, dict(row, profile=run.profile_name))
    # The example leaves the grid to forster.grid; summary.csv records the one chosen.
    assert rerun == replace(run, metadata={}, grid=result.inputs["grid"])


def test_summary_with_a_stage_and_edge1_fo_reads_back_and_refits(tmp_path, example, example_row):
    staged = dict(example_row, domain_start_um=8.0, dx_um=1.5, dt_s=2000.0, max_time_days=20)
    staged.update(edge1_fo=0.912, stage_days=1.5, stage_rim_um=2.5, stage_zone_fo=0.911)
    run = _read_row(tmp_path, staged)
    assert run.stages and run.rim_fo == 0.912
    profiles = read_profiles(example / "profiles.csv")
    result = fit_run(run, profiles)
    row = summary_row(result)
    assert set(row) - {c.name for c in (*PARAMETER_COLUMNS, *SUMMARY_COLUMNS)} == set()
    assert (row["edge1_fo"], row["stage_days"], row["stage_rim_um"]) == (0.912, "1.5", "2.5")
    assert row["time_since_last_stage_days"] == result.time_since_last_stage_days
    rerun = _read_row(tmp_path, dict(row, profile=run.profile_name))
    assert rerun == replace(run, metadata={})
    again = fit_run(rerun, profiles)
    assert (again.time_s, again.indices) == (result.time_s, result.indices)


def test_the_column_guide_documents_every_template_column():
    """The column tables live in the documentation site (docs/site/columns.md, generated
    by docs/gen_columns.py; tests/test_docs.py checks it is current), no longer in the
    README, which links to them."""
    page = (REPO_ROOT / "docs" / "site" / "columns.md").read_text(encoding="utf-8")
    assert column_table(PROFILE_COLUMNS) in page
    assert column_table(PARAMETER_COLUMNS) in page
    assert column_table(SUMMARY_COLUMNS) in page


def test_write_template_refuses_to_overwrite(tmp_path):
    write_template(tmp_path)
    (tmp_path / "parameters.csv").write_text("mine")
    with pytest.raises(FileExistsError, match=r"parameters\.csv"):
        write_template(tmp_path)
    assert (tmp_path / "parameters.csv").read_text() == "mine"
    write_template(tmp_path, overwrite=True)
    assert (tmp_path / "parameters.csv").read_text() != "mine"


def test_diffusivity_columns_match_the_table_reader():
    from forster.diffusivity import TABLE_COLUMNS
    from forster.template import DIFFUSIVITY_COLUMNS

    assert tuple(c.name for c in DIFFUSIVITY_COLUMNS) == TABLE_COLUMNS


def test_the_diffusivity_column_names_every_fe_mg_preset():
    from forster import DIFFUSIVITY_PRESETS
    from forster.template import PARAMETER_COLUMNS

    (column,) = [c for c in PARAMETER_COLUMNS if c.name == "diffusivity"]
    for name, preset in DIFFUSIVITY_PRESETS.items():
        if preset.element == "Fe-Mg":
            assert name in column.description
