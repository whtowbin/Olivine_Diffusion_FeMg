"""The app's second milestone without Qt: profile import with a column mapping, masking
and moving profiles, copying settings, the Monte Carlo and advanced settings (grid,
scheme, growth stages) to and from a run, and the views' inputs."""

import numpy as np
import pandas as pd
import pytest

from forster.boundary import Stage
from forster.grid import GridRequest
from forster.gui.importing import (
    Column,
    add_imported,
    fo_unit,
    guess_mapping,
    profiles_from_table,
    read_columns,
    rim_offset,
)
from forster.gui.state import (
    RunEditor,
    StageRow,
    copy_settings,
    law_tasks,
    new_run,
    result_rows,
    stage_rows,
    stages_of,
    summary_table,
    time_view,
)
from forster.model import Grid
from forster.project import Project
from forster.rescale import ConditionUncertainty

EXAMPLE = "AZ18 WHT01_Ol_63_prof_1"


@pytest.fixture
def project(tmp_path):
    return Project.new(tmp_path / "p")


@pytest.fixture
def editor(project):
    return RunEditor(project, EXAMPLE)


def lab_table(path, *, rim_far=False):
    """Two profiles in a lab layout (mol%, distances from 10 um; the rim at the far end
    when ``rim_far``), one flagged point and one blank row."""
    x = np.arange(10, 60, 5.0)
    fo = 90 - 2 * np.exp(-(x - 10) / 10)
    if rim_far:
        fo = fo[::-1]
    df = pd.DataFrame(
        {
            "Sample": ["A"] * 10 + ["B"] * 10 + ["C"] * 2,
            "Distance (um)": [*x, *x, 1.0, 2.0],
            "Fo#": [*fo, *(fo - 1), 88.0, 88.0],
            "Marked_bad": [""] * 3 + ["bad"] + [""] * 18,
            "Comment": "x",
        }
    )
    df.loc[5, "Fo#"] = np.nan
    df.to_csv(path, index=False)
    return path


def test_the_mapping_is_guessed_and_fo_in_percent_detected(tmp_path):
    df = read_columns(lab_table(tmp_path / "lab.csv"))
    mapping = guess_mapping(df)
    assert mapping["Sample"] == Column("profile")
    assert mapping["Distance (um)"] == Column("distance", "um")
    assert mapping["Fo#"] == Column("fo", "mol%")
    assert mapping["Comment"] == Column() and mapping["Marked_bad"] == Column()
    assert fo_unit(pd.Series([0.9, 0.91])) == "fraction"


def test_import_with_a_mapping_flips_and_offsets(tmp_path, project):
    df = read_columns(lab_table(tmp_path / "lab.csv", rim_far=True))
    mapping = guess_mapping(df)
    offset = rim_offset(pd.to_numeric(df["Distance (um)"]).to_numpy()[:20], flip=True)
    assert offset == 55.0
    imported = profiles_from_table(df, mapping, flip=True, offset_um=offset)
    assert set(imported.profiles) == {"A", "B"}  # C has too few points
    a = imported.profiles["A"]
    assert a.distance_um[0] == 0.0 and np.all(np.diff(a.distance_um) > 0)
    assert a.fo[0] == pytest.approx(0.88)  # the rim, now at 0
    assert len(a.fo) == 8  # 10 points: one flagged (masked), one with no Fo
    assert imported.masked["A"][0].tolist() == [30.0]  # x = 25 flipped about 55
    assert any("no distance or Fo" in n for n in imported.notes)
    assert any("fewer than 3" in n for n in imported.notes)
    names = add_imported(project, imported)
    assert names == ["A", "B"] and project.points("A").masked.sum() == 1
    project.save()
    again = Project.open(project.folder)
    assert again.points("A").masked.sum() == 1
    with pytest.raises(ValueError, match="already in the project"):
        add_imported(again, imported)


def test_import_of_one_profile_without_a_name_column(tmp_path):
    df = pd.DataFrame({"x": [0.001, 0.002, 0.003, 0.004], "Fo": [0.9, 0.89, 0.88, 0.88]})
    mapping = {"x": Column("distance", "mm"), "Fo": Column("fo", "fraction")}
    imported = profiles_from_table(df, mapping, name="single")
    np.testing.assert_allclose(imported.profiles["single"].distance_um, [1, 2, 3, 4])
    with pytest.raises(ValueError, match="choose a column for Fo"):
        profiles_from_table(df, {"x": Column("distance", "um")})
    with pytest.raises(ValueError, match="both"):
        profiles_from_table(df, {"x": Column("distance", "um"), "Fo": Column("distance", "um")})
    with pytest.raises(ValueError, match="check its unit"):
        profiles_from_table(
            df.assign(Fo=[90, 89, 88, 88]),
            {"x": Column("distance", "um"), "Fo": Column("fo", "fraction")},
        )


def test_a_run_on_an_imported_profile_fits(tmp_path, project):
    imported = profiles_from_table(
        read_columns(lab_table(tmp_path / "lab.csv")),
        guess_mapping(read_columns(tmp_path / "lab.csv")),
    )
    add_imported(project, imported)
    run = project.add_run(new_run(project, "A run", "A", like=EXAMPLE))
    assert run.initial.domain_um == (10.0, 55.0)
    assert project.fit("A run").time_days > 0


def test_masking_changes_the_fit_and_survives_a_reopen(project):
    first = project.fit(EXAMPLE)
    flags = project.points(EXAMPLE).masked.to_numpy().copy()
    flags[[10, 11]] = True
    project.set_masked(EXAMPLE, flags)
    assert project.result(EXAMPLE) is None
    second = project.fit(EXAMPLE)
    assert len(second.data_fo) == len(first.data_fo) - 2
    project.save()
    assert Project.open(project.folder).result(EXAMPLE) is not None  # same key, cached


def test_copy_settings_by_group(project):
    project.add_run(new_run(project, "other", EXAMPLE))
    before = project.run("other")
    source = project.run(EXAMPLE)
    copy_settings(project, EXAMPLE, ["other"], ["conditions", "equation"])
    after = project.run("other")
    assert after.conditions == source.conditions and after.fO2_form == source.fO2_form
    assert after.diffusivity == source.diffusivity
    assert after.orientation == before.orientation  # not chosen
    copy_settings(project, EXAMPLE, ["other"], ["orientation", "grid"])
    after = project.run("other")
    assert after.orientation == source.orientation and after.euler_form == source.euler_form
    assert after.grid == source.grid and after.scheme == source.scheme
    with pytest.raises(ValueError, match="at least one"):
        copy_settings(project, EXAMPLE, ["other"], [])
    with pytest.raises(ValueError, match="no settings group"):
        copy_settings(project, EXAMPLE, ["other"], ["colour"])


def test_copying_uncertainties_a_target_cannot_take_changes_nothing(project, editor):
    assert editor.add_segment(1)  # three segments: a step2_um prior is possible
    assert editor.set_step_spread(1, "2")
    project.add_run(new_run(project, "two segments", EXAMPLE))
    before = project.run("two segments")
    with pytest.raises(ValueError):
        copy_settings(project, EXAMPLE, ["two segments"], ["uncertainties", "conditions"])
    assert project.run("two segments") == before


def test_monte_carlo_settings_to_and_from_a_run(editor, project):
    fields = editor.mc_text()
    assert fields["on"] == "no" and fields["fO2_buffer"] == "QFM"
    assert editor.mc_buffers() == ["QFM", "NNO"]  # a buffer-relative run is always held
    on = {**fields, "on": "yes", "T_sd_C": "10", "P_sd_MPa": "50", "fO2_sd_log": "0.2"}
    assert editor.set_mc({**on, "samples": "200", "seed": "4", "law": "no"}, "T_sd_C")
    run = project.run(EXAMPLE)
    assert run.mc == ConditionUncertainty(10.0, 5e7, 0.2, "QFM", law=False)
    assert (run.mc_samples, run.mc_seed) == (200, 4)
    assert editor.mc_text()["P_sd_MPa"] == "50" and editor.mc_text()["samples"] == "200"
    assert not editor.set_mc({**on, "samples": "2.5"}, "samples")
    assert "whole number" in editor.errors["samples"]
    assert not editor.set_mc({**on, "T_sd_C": "-1"}, "T_sd_C")
    assert "zero or more" in editor.errors["T_sd_C"]
    assert project.run(EXAMPLE).mc_samples == 200  # refused edits change nothing
    assert editor.set_mc({**on, "on": "no"}, "on")
    assert project.run(EXAMPLE).mc is None
    # A run with fO2 in Pa may leave fO2 unheld.
    fields = editor.conditions_text()
    assert editor.set_conditions({**fields, "fO2_mode": "Pa", "fO2_Pa": "1e-6"}, "fO2_Pa")
    assert editor.mc_buffers() == ["", "QFM", "NNO"]
    assert editor.set_mc({**on, "fO2_buffer": ""}, "fO2_buffer")
    assert project.run(EXAMPLE).mc.fO2_buffer is None
    project.save()
    assert Project.open(project.folder).run(EXAMPLE).mc == project.run(EXAMPLE).mc


def test_the_monte_carlo_feeds_the_results_and_the_time_view(editor, project):
    assert time_view(project, EXAMPLE, "fit") == "Fit the run first."
    project.fit(EXAMPLE)
    view = time_view(project, EXAMPLE, "fit")
    assert not isinstance(view, str)
    result, posterior, mc = view
    assert result.time_s == project.fit(EXAMPLE).time_s
    assert posterior is None and mc is None
    assert "Fit starting profile" in time_view(project, EXAMPLE, "start")
    on = {**editor.mc_text(), "on": "yes", "T_sd_C": "10", "samples": "200"}
    assert editor.set_mc(on, "T_sd_C")
    assert "Monte Carlo" in time_view(project, EXAMPLE, "all")
    task = project.mc_task(EXAMPLE)
    project.store_mc(task, task())
    view = time_view(project, EXAMPLE, "all")
    assert not isinstance(view, str) and view[2] is not None
    assert len(view[2].days) == 200
    row = result_rows(project)[0]
    assert row["mc_low"] < row["mc_median"] < row["mc_high"]
    assert summary_table(project).mc_draws.tolist() == [200]


def test_grid_and_scheme_to_and_from_a_run(editor, project):
    fields = editor.grid_text()
    assert fields["scheme"] == "implicit"
    assert editor.set_grid({**fields, "dx_um": "0.5"}, "dx_um")
    assert fields["max_time_days"] == "40"
    assert project.run(EXAMPLE).grid == GridRequest(0.5, None, 40.0)
    assert editor.set_grid({**fields, "dx_um": "0.5", "dt_s": "2000", "max_time_days": ""}, "dt_s")
    assert project.run(EXAMPLE).grid == Grid(0.5, 2000.0, 200.0)  # the table's default
    assert not editor.set_grid({**fields, "dx_um": "-1"}, "dx_um")
    assert "positive" in editor.errors["dx_um"]
    assert editor.set_grid({"dx_um": "", "dt_s": "", "max_time_days": ""}, "dx_um")
    assert project.run(EXAMPLE).grid == GridRequest()
    assert editor.set_scheme("central") and project.run(EXAMPLE).scheme == "central"
    assert not editor.set_scheme("euler")
    assert editor.grid_lines() == ["Fit the run to see the grid it used."]
    project.fit(EXAMPLE)
    lines = editor.grid_lines()
    assert lines[0].startswith("Used: dx") and "Stability ratio" in lines[2]


def test_growth_stages_to_and_from_a_run(editor, project):
    assert stage_rows(project.run(EXAMPLE)) == []
    assert editor.add_stage()
    run = project.run(EXAMPLE)
    assert run.stages == (Stage(86400.0, 2.5, rim_value=0.9146),)
    assert not editor.set_stage_cell(0, "rim_um", "1")  # grows the crystal: needs a zone Fo
    assert "zone_value" in editor.errors["stage:0:rim_um"]
    assert editor.set_stage_cell(0, "zone_fo", "90.5")
    assert editor.set_stage_cell(0, "rim_um", "1")
    assert editor.set_stage_cell(0, "days", "3")
    assert stage_rows(project.run(EXAMPLE)) == [StageRow(3.0, 1.0, 0.905, 0.9146)]
    assert editor.add_stage()
    assert not editor.set_stage_cell(1, "days", "2")  # stages must come in time order
    assert "increase" in editor.errors["stage:1:days"]
    assert not editor.set_stage_cell(0, "zone_fo", "120")
    assert "at most 100" in editor.errors["stage:0:zone_fo"]
    rows = stage_rows(project.run(EXAMPLE))
    assert stages_of(rows) == project.run(EXAMPLE).stages
    project.save()
    assert Project.open(project.folder).run(EXAMPLE).stages == project.run(EXAMPLE).stages
    assert editor.remove_stage(1) and editor.remove_stage(0)
    assert project.run(EXAMPLE).stages == ()


def test_equations_offered_with_their_sources(editor):
    assert "Dohmen" in editor.diffusivity_source("dipra")
    assert editor.diffusivity_source("nonsense") == ""


def test_the_law_comparison_tasks(project):
    tasks = law_tasks(project, EXAMPLE)
    assert tasks[0].key == project.result_key(EXAMPLE)
    names = [t.run.diffusivity.name for t in tasks]
    assert len(names) == 4 and len(set(names)) == 4
    assert project.run(EXAMPLE).diffusivity.name == names[0]
