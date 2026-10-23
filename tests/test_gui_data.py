"""The app's second milestone without Qt: profile import with a column mapping, masking
and moving profiles, copying settings, the Monte Carlo and advanced settings (grid,
scheme, growth stages) to and from a run, and the views' inputs."""

import numpy as np
import pandas as pd
import pytest

from forster.boundary import Stage
from forster.grid import GridRequest
from forster.gui.importing import (
    RIM_ADVICE,
    add_imported,
    import_files,
    import_table,
    position_um,
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
from forster.imports import ColumnMap, ImportTemplate, guess
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


def test_the_mapping_is_the_core_guess_and_fo_in_percent_detected(tmp_path):
    df = read_columns(lab_table(tmp_path / "lab.csv"))
    roles = {c.column: (c.role, c.unit) for c in guess(df).columns}
    assert roles["Sample"] == ("profile", None)
    assert roles["Distance (um)"] == ("position", "um")
    assert roles["Fo#"] == ("fo", "percent")
    assert roles["Marked_bad"] == ("exclude", None) and roles["Comment"] == ("ignore", None)
    np.testing.assert_array_equal(position_um(df, guess(df))[:2], [10.0, 15.0])


def test_import_flips_offsets_and_masks_the_excluded_points(tmp_path, project):
    df = read_columns(lab_table(tmp_path / "lab.csv", rim_far=True))
    template = guess(df)
    offset = rim_offset(position_um(df, template)[:20], flip=True)
    assert offset == 55.0
    imported = import_table(df, template, flip=True, offset_um=offset)
    assert set(imported.profiles) == {"A", "B"}  # C has too few points
    a = imported.profiles["A"]
    assert a.distance_um[0] == 0.0 and np.all(np.diff(a.distance_um) > 0)
    assert a.fo[0] == pytest.approx(0.88)  # the rim, now at 0
    assert len(a.fo) == 8  # 10 points: one excluded (masked), one with no Fo
    assert imported.masked["A"][0].tolist() == [30.0]  # x = 25 flipped about 55
    assert any("left out 1 row(s)" in n for n in imported.notes)
    assert any("fewer than 3" in n for n in imported.notes)
    assert any("1 point(s) marked in an exclusion column" in n for n in imported.notes)
    names = add_imported(project, imported)
    assert names == ["A", "B"] and project.points("A").masked.sum() == 1
    project.save()
    again = Project.open(project.folder)
    assert again.points("A").masked.sum() == 1
    with pytest.raises(ValueError, match="already in the project"):
        add_imported(again, imported)


def test_import_of_one_profile_without_a_name_column():
    df = pd.DataFrame({"x": [0.001, 0.002, 0.003, 0.004], "Fo": [0.9, 0.89, 0.88, 0.88]})
    template = ImportTemplate(
        (ColumnMap("x", "position", unit="mm"), ColumnMap("Fo", "fo", unit="fraction"))
    )
    imported = import_table(df, template, name="single.1")
    np.testing.assert_allclose(imported.profiles["single.1"].distance_um, [1, 2, 3, 4])
    with pytest.raises(ValueError, match="no Fo"):
        import_table(df, ImportTemplate((ColumnMap("x", "position", unit="um"),)))
    with pytest.raises(ValueError, match="more than one position"):
        import_table(
            df,
            ImportTemplate(
                (ColumnMap("x", "position", unit="um"), ColumnMap("Fo", "position", unit="um"))
            ),
        )
    with pytest.raises(ValueError, match="Fo must be a fraction"):
        import_table(df.assign(Fo=[90, 89, 88, 88]), template)
    with pytest.raises(ValueError, match="no profile with enough points"):
        import_table(df.head(2), template)


def test_fo_from_oxides_and_flagged_totals_kept_with_moved_distances():
    """MgO and FeO in wt% with a Total column: Fo in moles (all iron as FeO); the point
    with a low total stays in the profile and is listed with its moved distance."""
    mgo, feo = np.array([48.0, 47.0, 46.0, 45.0]), np.array([9.0, 10.0, 11.0, 12.0])
    df = pd.DataFrame(
        {
            "Distance (mm)": [0.010, 0.012, 0.014, 0.016],
            "MgO": mgo,
            "FeO": feo,
            "Total (wt%)": [100.0, 96.0, 100.2, 99.9],
            "Marked_bad": ["", "", "", "bad"],
        }
    )
    template = guess(df)
    assert template.problems() == []
    imported = import_table(df, template, name="ox", offset_um=10.0)
    p = imported.profiles["ox"]
    mg, fe = mgo / 40.3044, feo / 71.8444
    np.testing.assert_allclose(p.fo, (mg / (mg + fe))[:3], rtol=1e-12)
    np.testing.assert_allclose(p.distance_um, [0.0, 2.0, 4.0])
    assert imported.flagged["distance_um"].tolist() == [2.0]
    assert imported.flagged["total_wt%"].tolist() == [96.0]
    assert imported.masked["ox"][0].tolist() == [6.0]


def test_a_run_on_an_imported_profile_fits(tmp_path, project):
    df = read_columns(lab_table(tmp_path / "lab.csv"))
    imported = import_table(df, guess(df))
    add_imported(project, imported)
    run = project.add_run(new_run(project, "A run", "A", like=EXAMPLE))
    assert run.initial.domain_um == (10.0, 55.0)
    # The example's longest time (40 days) is for its own profile: chosen afresh here.
    assert project.run(EXAMPLE).grid.max_time_days == 40.0
    assert run.grid == GridRequest() and run.scheme == project.run(EXAMPLE).scheme
    assert new_run(project, "same", EXAMPLE, like=EXAMPLE).grid.max_time_days == 40.0
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


def test_the_editor_masks_points_and_moves_its_profile(editor, project):
    points = editor.points()
    i = editor.nearest_point(float(points.distance_um[4]), float(points.fo[4] * 100), 1.0, 1.0)
    assert i == 4
    assert editor.set_point_masked(4, True)
    assert editor.points().masked.tolist() == [k == 4 for k in range(len(points))]
    assert editor.set_point_masked(4, False) and not project.masked
    for k in range(len(points) - 3):
        assert editor.set_point_masked(k, True)
    assert not editor.set_point_masked(len(points) - 1, True)
    assert "at least 3" in editor.errors["mask"]
    assert editor.rim_offset_text(False) == "0"
    assert not editor.move_profile(False, "")
    assert "give an offset" in editor.errors["move"]
    assert editor.move_profile(False, "-2.5")
    assert project.run(EXAMPLE).initial.domain_um == (5.0, 74.5)
    assert editor.rows[0].start_um == 5.0  # the side table follows
    assert editor.move_profile(True, editor.rim_offset_text(True))
    assert abs(editor.points().distance_um.iloc[0]) < 1e-9


def test_the_views_figures_are_recoloured_for_the_dark_theme():
    """`restyle_figure` turns the command line's figure colours (near-black data, white
    marker faces, the model blue, a light grid) into the dark palette's, backgrounds
    last; the light palette leaves them as drawn."""
    from matplotlib.colors import to_hex
    from matplotlib.figure import Figure

    from forster.gui.style import DARK, LIGHT, restyle_figure
    from forster.plotting import DATA_COLOR, MODEL_COLOR

    def drawn():
        figure = Figure()
        top, _ = figure.subfigures(2, 1)
        ax = top.subplots()
        ax.plot([0, 1], [0, 1], "o", color=DATA_COLOR, markerfacecolor="white", label="data")
        ax.plot([0, 1], [1, 0], color=MODEL_COLOR, label="model")
        ax.grid(color="0.9")
        ax.legend()
        ax.set_title("title")
        return figure, ax

    figure, ax = drawn()
    restyle_figure(figure, DARK)
    data, model = ax.get_lines()
    assert to_hex(data.get_color()) == DARK.foreground
    assert to_hex(data.get_markerfacecolor()) == DARK.background
    assert to_hex(model.get_color()) == to_hex(DARK.model)
    assert to_hex(ax.xaxis.get_gridlines()[0].get_color()) == DARK.grid
    assert to_hex(ax.get_facecolor()) == DARK.background
    assert to_hex(figure.get_facecolor()) == DARK.background
    assert to_hex(figure.subfigs[0].get_facecolor()) == DARK.background
    assert to_hex(ax.title.get_color()) == DARK.foreground
    assert {to_hex(t.get_color()) for t in ax.get_legend().get_texts()} == {DARK.foreground}
    figure, ax = drawn()
    restyle_figure(figure, LIGHT)
    assert to_hex(ax.get_lines()[0].get_color()) == to_hex(DATA_COLOR)


def staggered_table(path, starts, *, bad_first=False):
    """One profile per start distance, each 10 points 5 um apart from its start."""
    rows = []
    for name, start in starts.items():
        x = start + np.arange(10) * 5.0
        for i, xi in enumerate(x):
            bad = "bad" if bad_first and i == 0 else ""
            rows.append(
                {"Sample": name, "Distance (um)": xi, "Fo#": 90 - i * 0.1, "Marked_bad": bad}
            )
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_each_profile_can_start_at_its_own_first_point(tmp_path):
    """offset_um=None shifts each profile by its own first point (one offset for the
    whole table put only the profile with the smallest distance at 0), counting masked
    points so none lands below 0, and says that rim distances fit better."""
    df = read_columns(staggered_table(tmp_path / "s.csv", {"A": 10.0, "B": 3.0}))
    imported = import_table(df, guess(df), offset_um=None)
    assert imported.profiles["A"].distance_um[0] == 0.0
    assert imported.profiles["B"].distance_um[0] == 0.0
    assert RIM_ADVICE in imported.notes
    shared = import_table(df, guess(df), offset_um=3.0)
    assert shared.profiles["A"].distance_um[0] == 7.0 and RIM_ADVICE not in shared.notes

    df = read_columns(staggered_table(tmp_path / "m.csv", {"A": 10.0}, bad_first=True))
    masked = import_table(df, guess(df), offset_um=None)
    assert masked.masked["A"][0].tolist() == [0.0]  # the masked first point
    assert masked.profiles["A"].distance_um[0] == 5.0


def test_a_batch_starts_each_file_at_its_own_first_point(tmp_path):
    folder = tmp_path / "batch"
    folder.mkdir()
    paths = [
        staggered_table(folder / "one.csv", {"A": 12.0}),
        staggered_table(folder / "two.csv", {"B": 4.0}),
    ]
    template = guess(read_columns(paths[0]))
    imported, problems = import_files(paths, template, offset_um=None)
    assert problems == {}
    assert [imported.profiles[k].distance_um[0] for k in ("A", "B")] == [0.0, 0.0]
    assert imported.notes.count(RIM_ADVICE) == 1
