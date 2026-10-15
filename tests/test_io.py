import math

import numpy as np
import pandas as pd
import pytest
from conftest import (
    PAPER_DATA,
    PAPER_PARAMS,
    ROUND_ROBIN_DATA,
    ROUND_ROBIN_PARAMS,
    require_round_robin,
    requires_round_robin,
)

from forster import RunDefaults, engine, fit_run, orientation, read_profiles, read_runs
from forster.io import Run


@pytest.fixture(scope="module")
def round_robin_runs():
    require_round_robin()
    return {run.name: run for run in read_runs(ROUND_ROBIN_PARAMS)}


@pytest.fixture(scope="module")
def paper_runs():
    return {run.name: run for run in read_runs(PAPER_PARAMS)}


def test_round_robin_single_step_row(round_robin_runs):
    run = round_robin_runs["KS20-527_2-transect-1"]
    assert run.profile_name == "KS20-527_2-transect"
    assert (run.orientation.alpha_deg, run.orientation.beta_deg, run.orientation.gamma_deg) == (
        88,
        123,
        33,
    )
    assert (run.conditions.T_C, run.conditions.P_Pa) == (1200, 4.2e7)
    # The cell holds 0.0013803842646..., shown as 0.00138 in Excel.
    assert run.conditions.fO2_Pa == pytest.approx(0.00138, rel=1e-3)
    assert run.initial.domain_um == (0, 300)
    assert run.initial.steps_um == (41,)
    assert run.initial.step_fo is None
    assert (run.initial.edge_fo, run.initial.core_fo) == (0.82, 0.885)
    assert (run.grid.dx_um, run.grid.dt_s) == (1, 600)
    assert run.metadata["Descriptive_Name"] == "Step 41µm"


def test_round_robin_two_step_row(round_robin_runs):
    run = round_robin_runs["KS20-527-8_transect-1"]
    assert run.initial.steps_um == (0, 100)
    assert run.initial.step_fo == pytest.approx(0.885)


def test_paper_layout_converts_euler_angles_and_buffer(paper_runs):
    raw = pd.read_excel(PAPER_PARAMS, engine="openpyxl").set_index("File_Name")
    assert len(paper_runs) == 18
    for name, run in paper_runs.items():
        row = raw.loc[name]
        expected_angles = orientation.vector_direction(row.prof_angle, row.phi1, row.Phi, row.phi2)
        assert (
            run.orientation.alpha_deg,
            run.orientation.beta_deg,
            run.orientation.gamma_deg,
        ) == expected_angles
        expected_fO2 = engine.fo2buffer(row["T"] + 273.15, row.P, row.FO2_dQFM, "FMQ")
        assert run.conditions.fO2_Pa == expected_fO2
        # row.dt would be pandas' datetime accessor, so index by name.
        assert (run.grid.dx_um, run.grid.dt_s) == (row["dx"], row["dt"])


@requires_round_robin
def test_max_time_from_table_or_defaults():
    raw = pd.read_excel(ROUND_ROBIN_PARAMS, engine="openpyxl").set_index("Sample")
    runs = read_runs(ROUND_ROBIN_PARAMS, defaults=RunDefaults(max_time_days=123))
    from_table = [r for r in runs if pd.notna(raw.loc[r.name, "max_model_time_days"])]
    from_default = [r for r in runs if pd.isna(raw.loc[r.name, "max_model_time_days"])]
    assert from_table and from_default  # the table has both kinds of row
    for run in from_table:
        assert run.grid.max_time_days == raw.loc[run.name, "max_model_time_days"]
    assert {run.grid.max_time_days for run in from_default} == {123}


@requires_round_robin
def test_read_profiles_drops_flagged_rows_and_sorts():
    raw = pd.read_excel(ROUND_ROBIN_DATA, engine="openpyxl")
    profiles = read_profiles(ROUND_ROBIN_DATA)
    assert set(profiles) == set(raw.Profile_Name.dropna().unique())
    for name, profile in profiles.items():
        usable = raw[(raw.Profile_Name == name) & (raw.Marked_bad != "bad") & (raw.Ignore != "yes")]
        assert len(profile.distance_um) == len(usable)
        assert np.all(np.diff(profile.distance_um) >= 0)
        assert profile.fo.max() <= 1


@requires_round_robin
def test_csv_tables_read_like_excel(tmp_path):
    csv = tmp_path / "params.csv"
    pd.read_excel(ROUND_ROBIN_PARAMS, engine="openpyxl").to_csv(csv, index=False)
    from_csv = read_runs(csv)
    from_xlsx = read_runs(ROUND_ROBIN_PARAMS)
    assert [r.name for r in from_csv] == [r.name for r in from_xlsx]
    assert from_csv[0].initial == from_xlsx[0].initial


def test_unrecognised_table_names_both_layouts(tmp_path):
    bad = tmp_path / "bad.csv"
    pd.DataFrame(
        {
            "edge_x1": [0],
            "edge_x2": [1],
            "inflection_x": [0],
            "inflection_c": [math.nan],
            "edge_c": [80],
            "center_c": [90],
        }
    ).to_csv(bad, index=False)
    with pytest.raises(ValueError, match=r"Round robin layout needs.*paper layout needs"):
        read_runs(bad)


@requires_round_robin
def test_duplicate_run_names_are_rejected(tmp_path):
    table = pd.read_excel(ROUND_ROBIN_PARAMS, engine="openpyxl")
    table = pd.concat([table, table.iloc[[0]]])
    path = tmp_path / "dup.csv"
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="unique"):
        read_runs(path)


def test_missing_profile_names_what_is_available(round_robin_runs):
    run: Run = round_robin_runs["KS20-527_2-transect-1"]
    with pytest.raises(KeyError, match="available"):
        fit_run(run, {})


def test_paper_run_fits_end_to_end():
    """The paper layout through to a fit: Euler angles, the QFM buffer and multi-sheet data."""
    runs = {run.name: run for run in read_runs(PAPER_PARAMS)}
    with pytest.warns(UserWarning, match="fewer than 3 points"):
        profiles = read_profiles(PAPER_DATA, sheet="Olivine")
    assert all(run.profile_name in profiles for run in runs.values())
    result = fit_run(runs["AZ18 WHT06_ol_overgrowth1_prof_1"], profiles)
    assert result.time_s > 0
    assert result.time_minus_s <= 0 <= result.time_plus_s


@requires_round_robin
def test_scheme_column_is_read_per_run(tmp_path):
    table = pd.read_excel(ROUND_ROBIN_PARAMS, engine="openpyxl")
    table["scheme"] = None
    table.loc[0, "scheme"] = "DIPRA"
    path = tmp_path / "params.csv"
    table.to_csv(path, index=False)
    schemes = {run.name: run.scheme for run in read_runs(path)}
    assert schemes[table.loc[0, "Sample"]] == "dipra"
    assert set(schemes.values()) == {"implicit", "dipra"}  # implicit: the default
    assert {r.scheme for r in read_runs(path, defaults=RunDefaults(scheme="dipra"))} == {"dipra"}


@requires_round_robin
def test_bad_scheme_in_table_is_an_error(tmp_path):
    table = pd.read_excel(ROUND_ROBIN_PARAMS, engine="openpyxl")
    table["scheme"] = "crank_nicolson"
    path = tmp_path / "params.csv"
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="scheme 'crank_nicolson'"):
        read_runs(path)


def test_uncertainty_columns_are_read_and_written_back(tmp_path):
    from forster import write_template
    from forster.io import uncertainty_columns
    from forster.scan import Uncertainty

    write_template(tmp_path / "t")
    table = pd.read_csv(tmp_path / "t" / "parameters.csv")
    table = table.assign(steps_sd_um=2.0, edge_fo_range=0.003, scan_points=7)
    path = tmp_path / "params.csv"
    table.to_csv(path, index=False)
    (run,) = read_runs(path)
    assert run.uncertainties == {
        "step1_um": Uncertainty(2.0, "normal"),
        "edge_fo": Uncertainty(0.003, "uniform"),
    }
    assert run.scan_points == 7
    again = tmp_path / "again.csv"
    table.drop(columns=["steps_sd_um", "edge_fo_range", "scan_points"]).assign(
        **uncertainty_columns(run)
    ).to_csv(again, index=False)
    assert read_runs(again)[0] == run


@pytest.mark.parametrize(
    ("columns", "message"),
    [
        ({"steps_sd_um": 2.0, "steps_range_um": 3.0}, "not both"),
        ({"steps_sd_um": "1, 2"}, "2 values for 1 steps"),
        ({"step_fo_sd": 0.01}, "needs a plateau"),
    ],
)
def test_bad_uncertainty_columns_are_errors(tmp_path, columns, message):
    from forster import write_template

    write_template(tmp_path / "t")
    table = pd.read_csv(tmp_path / "t" / "parameters.csv").assign(**columns)
    path = tmp_path / "params.csv"
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match=message):
        read_runs(path)


@pytest.fixture
def template_tables(tmp_path):
    from forster import write_template

    write_template(tmp_path / "t")
    return (
        pd.read_csv(tmp_path / "t" / "profiles.csv"),
        pd.read_csv(tmp_path / "t" / "parameters.csv"),
    )


def test_flags_are_case_insensitive(tmp_path, template_tables):
    profiles, _ = template_tables
    flags = ["Bad", " BAD ", "bad"] + [""] * (len(profiles) - 3)
    path = tmp_path / "flagged.csv"
    profiles.assign(
        Marked_bad=flags, Ignore=["", "", "", "Yes"] + [""] * (len(profiles) - 4)
    ).to_csv(path, index=False)
    (profile,) = read_profiles(path).values()
    assert len(profile.distance_um) == len(profiles) - 4


def test_rows_without_distance_or_fo_are_dropped_with_a_warning(tmp_path, template_tables):
    profiles, _ = template_tables
    profiles.loc[5, "fo"] = np.nan
    path = tmp_path / "blank.csv"
    profiles.to_csv(path, index=False)
    with pytest.warns(UserWarning, match="dropped rows with no distance or Fo"):
        (profile,) = read_profiles(path).values()
    assert len(profile.fo) == len(profiles) - 1
    assert np.isfinite(profile.fo).all()


def test_unknown_template_column_warns(tmp_path, template_tables):
    _, parameters = template_tables
    path = tmp_path / "params.csv"
    parameters.assign(steps_sd=2.0).to_csv(path, index=False)  # meant steps_sd_um
    with pytest.warns(UserWarning, match=r"not used by FORSTER.*steps_sd"):
        read_runs(path)


def test_bad_inflection_c_is_a_value_error():
    from forster.io import _plateau_fraction

    assert _plateau_fraction("88.5") == pytest.approx(0.885)
    with pytest.raises(ValueError, match="inflection_c"):
        _plateau_fraction("88,,5")


def test_xls_is_refused(tmp_path):
    from forster.io import read_table

    path = tmp_path / "old.xls"
    path.write_bytes(b"")
    with pytest.raises(ValueError, match=r"\.csv or \.xlsx"):
        read_table(path)
