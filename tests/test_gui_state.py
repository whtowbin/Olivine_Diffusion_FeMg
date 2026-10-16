"""`forster.gui.state`: the side panel's logic without Qt: the segment table, its
validation through the table reader, and the mapping to and from a run."""

import pytest

from forster import Uncertainty
from forster.gui.state import (
    RunEditor,
    SegmentRow,
    initial_of,
    new_run,
    result_rows,
    rows_of,
    summary_table,
)
from forster.project import Project

EXAMPLE = "AZ18 WHT01_Ol_63_prof_1"


@pytest.fixture
def project(tmp_path):
    return Project.new(tmp_path / "p")


@pytest.fixture
def editor(project):
    return RunEditor(project, EXAMPLE)


def test_the_example_as_a_segment_table_and_back(project):
    run = project.run(EXAMPLE)
    rows = rows_of(run)
    assert rows == [
        SegmentRow("a", 2.5, 24.0, 0.9146),
        SegmentRow("b", 24.0, 72.0, 0.8985),
    ]
    assert initial_of(rows, None) == run.initial


def test_editing_an_end_moves_the_next_start(editor, project):
    assert editor.set_cell(0, "end", "30")
    assert project.run(EXAMPLE).initial.steps_um == (30.0,)
    assert editor.rows[1].start_um == 30.0
    assert editor.set_cell(1, "end", "70")  # the last end is the domain's end
    assert project.run(EXAMPLE).initial.domain_um == (2.5, 70.0)
    assert editor.set_cell(0, "start", "0")
    assert project.run(EXAMPLE).initial.domain_um == (0.0, 70.0)
    assert project.dirty


def test_a_refused_value_keeps_the_run_and_names_the_field(editor, project):
    before = project.run(EXAMPLE)
    assert not editor.set_cell(0, "end", "abc")
    assert "not a number" in editor.errors["segment:0:end"]
    assert not editor.set_cell(0, "end", "80")  # past the domain's end (72)
    assert "inside the model domain" in editor.errors["segment:0:end"]
    assert not editor.set_cell(0, "fo", "120")
    assert "at most 100" in editor.errors["segment:0:fo"]
    assert not editor.set_cell(1, "name", "step2")
    assert "segment names" in editor.errors["segment:1:name"]
    assert project.run(EXAMPLE) == before
    assert editor.rows == rows_of(before)
    assert editor.set_cell(0, "fo", "91.2")
    assert "segment:0:fo" not in editor.errors  # an accepted value clears its error


def test_fo_is_typed_in_mol_percent_and_stored_exactly(editor, project):
    assert editor.set_cell(0, "fo", "91.46")
    assert project.run(EXAMPLE).initial.segment_fo[0] == 0.9146
    assert editor.set_cell(1, "spread", "0.2")
    assert project.run(EXAMPLE).uncertainties == {"b_fo": Uncertainty(0.002)}
    assert editor.set_cell(1, "kind", "range")
    assert project.run(EXAMPLE).uncertainties == {"b_fo": Uncertainty(0.002, "uniform")}
    assert not editor.set_cell(1, "kind", "wide")


def test_renaming_a_segment_keeps_its_uncertainty(editor, project):
    assert editor.set_cell(0, "spread", "0.3")
    assert editor.set_cell(0, "name", "rim")
    run = project.run(EXAMPLE)
    assert run.initial.segment_names == ("rim", "b")
    assert run.uncertainties == {"rim_fo": Uncertainty(0.003)}


def test_adding_and_removing_segments(editor, project):
    assert editor.set_step_spread(0, "2")
    assert editor.add_segment(1)  # b split at its middle
    run = project.run(EXAMPLE)
    assert run.initial.steps_um == (24.0, 48.0)
    assert run.initial.segment_names == ("a", "b", "c")
    assert run.initial.segment_fo == (0.9146, 0.8985, 0.8985)
    assert run.uncertainties == {"step1_um": Uncertainty(2.0)}
    assert editor.remove_segment(0)  # b takes a's place; the step at a's end goes
    run = project.run(EXAMPLE)
    assert run.initial.steps_um == (48.0,) and run.initial.domain_um == (2.5, 72.0)
    assert run.initial.segment_names == ("a", "b")  # default names stay a, b, c ...
    assert run.uncertainties == {}
    assert not editor.remove_segment(0)
    assert "at least two segments" in editor.errors["segments"]


def test_step_uncertainties_share_one_kind(editor, project):
    assert editor.set_step_spread(0, "2.5")
    assert project.run(EXAMPLE).uncertainties == {"step1_um": Uncertainty(2.5)}
    assert editor.set_steps_kind("range")
    assert project.run(EXAMPLE).uncertainties == {"step1_um": Uncertainty(2.5, "uniform")}
    assert editor.set_step_spread(0, "")
    assert project.run(EXAMPLE).uncertainties == {}


def test_drags_preview_then_commit(editor, project):
    before = project.run(EXAMPLE)
    preview = editor.dragged_rows("step", 0, editor.snap("step", 26.04))
    assert preview[0].end_um == 26.0 and preview[1].start_um == 26.0
    assert project.run(EXAMPLE) == before  # a preview changes nothing
    assert editor.drag("fo", 1, editor.snap("fo", 89.5049))
    assert project.run(EXAMPLE).initial.segment_fo[1] == 0.895
    assert editor.drag("end", 0, 60.0)
    assert project.run(EXAMPLE).initial.domain_um[1] == 60.0
    assert not editor.drag("step", 0, 100.0)
    assert editor.errors["drag"]


def test_edge_value_and_crystal_centre(editor, project):
    assert editor.set_edge1("91.2")
    assert project.run(EXAMPLE).rim_fo == 0.912 and editor.edge1_text == "91.2"
    assert editor.set_edge1("")
    assert project.run(EXAMPLE).rim_fo is None
    assert editor.set_centre("100")
    assert project.run(EXAMPLE).initial.centre_um == 100.0
    assert not editor.set_centre("50")  # inside the domain
    assert "centre_um" in editor.errors["centre"]
    assert editor.set_centre("")
    assert project.run(EXAMPLE).initial.centre_um is None


def test_conditions_orientation_and_equation(editor, project):
    fields = editor.conditions_text()
    assert fields["fO2_mode"] == "buffer" and fields["P_MPa"] == "500"
    assert editor.set_conditions({**fields, "T_C": "1260"}, "T_C")
    run = project.run(EXAMPLE)
    assert run.conditions.T_C == 1260.0 and run.fO2_form == ("QFM", 0.3)
    assert editor.set_conditions({**fields, "fO2_mode": "Pa", "fO2_Pa": "1e-6"}, "fO2_Pa")
    assert project.run(EXAMPLE).fO2_form is None
    assert not editor.set_conditions({**fields, "fO2_buffer": "XYZ"}, "fO2_buffer")
    assert "QFM" in editor.errors["fO2_buffer"]
    orientation = editor.orientation_text()
    assert orientation["mode"] == "euler" and orientation["phi1_deg"] == "77"
    assert editor.set_orientation(
        {**orientation, "mode": "angles", "alpha_deg": "90", "beta_deg": "90", "gamma_deg": "0"},
        "alpha_deg",
    )
    assert project.run(EXAMPLE).euler_form is None
    assert [k for k, _ in editor.diffusivity_choices()] == [
        "dipra",
        "refit",
        "dfens_tamed",
        "dfens_global",
    ]
    assert editor.set_diffusivity("refit")
    assert editor.diffusivity_key == "refit"
    assert project.run(EXAMPLE).diffusivity.name == "femg_refit2026"


def test_edits_survive_save_and_reopen(editor, project):
    editor.set_cell(0, "name", "rim")
    editor.set_cell(1, "spread", "0.2")
    editor.add_segment(1)
    project.save()
    again = Project.open(project.folder)
    assert RunEditor(again, EXAMPLE).rows == editor.rows


def test_new_runs_and_the_results_table(project):
    added = project.add_run(new_run(project, "second", EXAMPLE, like=EXAMPLE))
    x = project.profiles[EXAMPLE].distance_um
    lo, hi = added.initial.domain_um
    assert (lo, hi) == (x.min(), x.max())
    assert added.initial.steps_um == (round((lo + hi) / 2, 1),)
    assert added.conditions == project.run(EXAMPLE).conditions
    plain = new_run(project, "third", EXAMPLE)
    project.add_run(plain)
    rows = result_rows(project)
    assert [r["status"] for r in rows] == ["not fitted"] * 3
    project.fit(EXAMPLE)
    rows = result_rows(project)
    assert rows[0]["status"] == "fitted" and rows[0]["days"] > 0
    table = summary_table(project)
    assert table.run.tolist() == [EXAMPLE]
    assert {"profile", "time_days", "warnings"} <= set(table.columns)
