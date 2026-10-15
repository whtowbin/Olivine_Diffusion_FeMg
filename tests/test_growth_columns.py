"""Growth stages and a fixed rim value from the template layout (``rim_fo`` and the
``stage_*`` columns), through `read_runs`, `fit_run`, the scan guard and `forster fit`."""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forster import Grid, Profile, fit_profile, fit_run, read_profiles, read_runs, write_template
from forster.boundary import RimPath, Stage
from forster.io import _boundary_columns, _list_text, _number_list, scan_run, scan_size

DAY = 86400.0
FORSTER = Path(sys.executable).parent / "forster"
# The packaged example with a grid given in full (staged runs need one) and a short run.
GRID = {"dx_um": 1.5, "dt_s": 2000.0, "max_time_days": 20}
RIM = {"rim_fo": 0.912}
# A crystal from 12 um that grows to 6 um at day 1 (zone Fo 0.913), then to 2.5 um at
# day 2.5 (zone 0.911) with its rim held at 0.912.
TWO_STAGES = {
    "domain_start_um": 12.0,
    "stage_days": "1, 2.5",
    "stage_rim_um": "6, 2.5",
    "stage_zone_fo": "0.913, 0.911",
    "stage_rim_fo": ", 0.912",
}
EXPECTED_STAGES = (
    Stage(1 * DAY, 6.0, rim_value=None, zone_value=0.913),
    Stage(2.5 * DAY, 2.5, rim_value=0.912, zone_value=0.911),
)


@pytest.fixture(scope="module")
def example(tmp_path_factory):
    out = tmp_path_factory.mktemp("template")
    write_template(out)
    row = pd.read_csv(out / "parameters.csv").iloc[0].to_dict()
    return dict(row, **GRID), read_profiles(out / "profiles.csv")


def _read_row(tmp_path, row):
    path = tmp_path / "parameters.csv"
    pd.DataFrame([row]).to_csv(path, index=False)
    return read_runs(path)[0]


def _direct_fit(run, profiles, **boundary):
    """`fit_profile` called directly, with the grid and boundary written out by hand."""
    measured = profiles[run.profile_name]
    return fit_profile(
        Profile(measured.distance_um, measured.fo, name=run.name),
        run.conditions,
        run.orientation,
        run.initial,
        Grid(**GRID),
        run.kriging,
        run.diffusivity,
        run.scheme,
        run.uncertainty,
        **boundary,
    )


def _assert_same_fit(a, b):
    assert (a.time_s, a.time_minus_s, a.time_plus_s) == (b.time_s, b.time_minus_s, b.time_plus_s)
    assert a.indices == b.indices
    np.testing.assert_array_equal(a.best_fit, b.best_fit)


def test_rim_fo_reads_into_the_run_and_fits_like_fit_profile(tmp_path, example):
    row, profiles = example
    run = _read_row(tmp_path, dict(row, **RIM))
    assert run.rim_fo == 0.912 and run.stages == ()
    assert run.initial.edge_fo == row["edge_fo"]  # the start profile keeps edge_fo
    fitted = fit_run(run, profiles)
    _assert_same_fit(fitted, _direct_fit(run, profiles, rim_fo=0.912))
    plain = fit_run(_read_row(tmp_path, row), profiles)
    assert fitted.time_s != plain.time_s  # the rim value reaches the model


def test_two_stages_read_into_the_run_and_fit_like_fit_profile(tmp_path, example):
    row, profiles = example
    run = _read_row(tmp_path, dict(row, **TWO_STAGES))
    assert run.stages == EXPECTED_STAGES
    assert run.rim_fo is None and run.initial.domain_um == (12.0, 72.0)
    fitted = fit_run(run, profiles)
    _assert_same_fit(fitted, _direct_fit(run, profiles, stages=list(EXPECTED_STAGES)))
    assert fitted.last_stage_s == 2.5 * DAY
    assert fitted.time_since_last_stage_days == pytest.approx(fitted.time_days - 2.5)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"stage_rim_um": "6"}, "stage_rim_um has 1 entries for 2 stages"),
        ({"stage_rim_fo": "0.9, 0.9, 0.9"}, "stage_rim_fo has 3 entries for 2 stages"),
        ({"stage_days": None}, "growth stages need stage_days"),
        ({"stage_days": None, "stage_rim_um": None}, "need stage_days and stage_rim_um"),
        ({"stage_days": "1; 2.5"}, "stage_days '1; 2.5': give numbers separated by commas"),
        ({"stage_zone_fo": "0.913, high"}, "stage_zone_fo '0.913, high': give numbers"),
        ({"stage_days": ", 2.5"}, "stage_days needs a value for every stage"),
        ({"stage_days": "2.5, 1"}, "stage_days must increase"),
        ({"stage_days": "0, 1"}, "stage_days must increase and be after 0"),
        ({"stage_rim_um": "20, 2.5"}, "the rim cannot move inward"),
        ({"stage_zone_fo": ", 0.911"}, "give zone_value"),
        ({"stage_zone_fo": "0.913, 1.2"}, "Fo values must be fractions"),
        ({"rim_fo": 1.5}, "rim_fo 1.5 must be a fraction"),
    ],
)
def test_bad_stage_columns_name_the_run_and_the_problem(tmp_path, example, change, message):
    row, _ = example
    with pytest.raises(ValueError, match=f"run 'AZ18 WHT01_Ol_63_prof_1': .*{message}"):
        _read_row(tmp_path, {**row, **TWO_STAGES, **change})


def test_stages_need_a_grid_given_in_full(tmp_path, example):
    row, profiles = example
    run = _read_row(tmp_path, dict(row, **TWO_STAGES, dt_s=None))
    with pytest.raises(ValueError, match="growth stages need dx_um and dt_s"):
        fit_run(run, profiles)


@pytest.mark.parametrize(("boundary", "what"), [(RIM, "rim_fo"), (TWO_STAGES, "growth stages")])
def test_the_scan_refuses_stages_and_rim_fo(tmp_path, example, boundary, what):
    row, profiles = example
    run = _read_row(tmp_path, dict(row, **boundary, edge_fo_sd=0.002))
    message = f"the initial-condition scan does not model {what}"
    with pytest.raises(ValueError, match=message):
        scan_size(run)
    with pytest.raises(ValueError, match=message):
        scan_run(run, profiles)


@pytest.mark.parametrize("values", [(0.91, None), (None, 0.5), (1.0, 2.5, 0.125), (6.0,)])
def test_list_text_reads_back_exactly(values):
    row = pd.Series({"run": "r", "c": _list_text(values)})
    assert _number_list(row, "c") == values


def test_a_changing_rim_value_has_no_table_form():
    path = RimPath((0.0, 3.0), (0.86, 0.82))
    with pytest.raises(ValueError, match="no parameter-table form"):
        _boundary_columns({"rim_fo": path, "stages": ()})
    with pytest.raises(ValueError, match="no parameter-table form"):
        _boundary_columns({"stages": (Stage(DAY, 0.0, rim_value=path, zone_value=0.82),)})


def test_forster_fit_with_a_stage_writes_a_summary_that_refits(tmp_path, example):
    row, _ = example
    staged = dict(row, **RIM, domain_start_um=8.0, stage_days=1.5, stage_rim_um=2.5)
    staged["stage_zone_fo"] = 0.911
    scanned = dict(staged, run="scanned", edge_fo_sd=0.002)
    pd.DataFrame([staged, scanned]).to_csv(tmp_path / "parameters.csv", index=False)
    write_template(tmp_path / "in")

    def forster(params, out):
        return subprocess.run(
            [str(FORSTER), "fit", tmp_path / "in" / "profiles.csv", params, "--out", out],
            capture_output=True,
            text=True,
            timeout=300,
        )

    done = forster(tmp_path / "parameters.csv", tmp_path / "results")
    assert done.returncode == 1  # the scanned run fails; the staged one is kept
    assert "the initial-condition scan does not model growth stages or rim_fo" in " ".join(
        done.stderr.split()
    )
    summary = pd.read_csv(tmp_path / "results" / "summary.csv", float_precision="round_trip")  # ty: ignore[no-matching-overload]
    (fitted,) = summary.itertuples()
    assert fitted.run == row["run"]
    assert (fitted.stage_days, fitted.stage_rim_um, fitted.stage_zone_fo) == (1.5, 2.5, 0.911)
    assert fitted.rim_fo == 0.912
    assert fitted.time_since_last_stage_days == pytest.approx(fitted.time_days - 1.5)

    again = forster(tmp_path / "results" / "summary.csv", tmp_path / "again")
    assert again.returncode == 0, again.stderr
    refit = pd.read_csv(tmp_path / "again" / "summary.csv", float_precision="round_trip")  # ty: ignore[no-matching-overload]
    assert refit.time_days[0] == fitted.time_days
