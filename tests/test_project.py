"""`forster.project`: open, edit, save and reopen a project folder; the result cache."""

import subprocess
import sys
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from forster import Conditions, InitialCondition, Uncertainty, read_runs
from forster.boundary import Stage
from forster.diffusivity import resolve
from forster.io import run_row
from forster.project import WORK, Project, profile_table
from forster.rescale import ConditionUncertainty

EXAMPLE = "AZ18 WHT01_Ol_63_prof_1"


@pytest.fixture
def project(tmp_path):
    return Project.new(tmp_path / "p")


def test_new_project_saves_and_reopens_identically(project):
    project.save()
    again = Project.open(project.folder)
    assert again.runs == project.runs
    assert again.profiles.keys() == project.profiles.keys()
    for name, p in project.profiles.items():
        np.testing.assert_array_equal(again.profiles[name].fo, p.fo)
        np.testing.assert_array_equal(again.profiles[name].distance_um, p.distance_um)
    # The example gives fO2 relative to QFM and the orientation as Euler angles: kept.
    run = again.run(EXAMPLE)
    assert run.fO2_form == ("QFM", 0.3)
    assert run.euler_form == (77.0, 61.2, 334.2, -80.0)
    table = pd.read_csv(project.folder / "parameters.csv")
    assert "fO2_buffer" in table and "phi1_deg" in table and "fO2_Pa" not in table


def test_run_row_round_trips_a_run_with_every_option(project, tmp_path):
    run = project.run(EXAMPLE)
    rich = replace(
        run,
        name="rich",
        diffusivity=resolve("refit"),
        uncertainties={"step1_um": Uncertainty(2.0), "edge_fo": Uncertainty(0.004, "uniform")},
        scan_method="laplace",
        scan_seed=7,
        mc=ConditionUncertainty(T_sd_C=10, fO2_sd_log=0.2, fO2_buffer="QFM", between_study=True),
        mc_samples=900,
        mc_seed=3,
        stages=(Stage(86400.0, rim_um=1.0, zone_value=0.915),),
        rim_fo=0.913,
        fO2_form=None,
        euler_form=None,
    )
    path = tmp_path / "one.csv"
    pd.DataFrame([run_row(rich)]).to_csv(path, index=False)
    (back,) = read_runs(path)
    assert back == rich


def test_editing_runs(project):
    first = project.run_names[0]
    copy = project.duplicate_run(first)
    assert project.run_names == [first, copy.name] and copy.name == f"{first} copy"
    project.rename_run(copy.name, "warmer")
    warmer = project.update_run(
        "warmer", conditions=replace(project.run("warmer").conditions, T_C=1300.0)
    )
    assert warmer.conditions.T_C == 1300.0 and project.run(first).conditions.T_C == 1250.0
    project.copy_settings("warmer", [first], ["conditions"])
    assert project.run(first).conditions == warmer.conditions
    with pytest.raises(ValueError, match="already a run"):
        project.rename_run(first, "warmer")
    with pytest.raises(ValueError, match="already a run"):
        project.duplicate_run(first, "warmer")
    # A bad edit is refused and changes nothing.
    before = project.run("warmer")
    with pytest.raises(ValueError):
        project.update_run(
            "warmer",
            initial=InitialCondition((0, 50), 0.9, 0.88, steps_um=(10, 20)),  # 2 steps, no plateau
        )
    assert project.run("warmer") == before
    project.remove_run("warmer")
    assert project.run_names == [first]
    with pytest.raises(KeyError, match="no run"):
        project.run("warmer")
    assert project.dirty


def test_importing_profiles(project, tmp_path):
    percent = tmp_path / "lab.csv"
    pd.DataFrame(
        {
            "sample": ["a"] * 5 + ["b"] * 5,
            "x": list(range(5)) * 2,
            "Fo": [90.1, 90.0, 89.5, 89.0, 88.9] * 2,
        }
    ).to_csv(percent, index=False)
    added = project.import_profiles(
        percent, name_column="sample", distance_column="x", fo_column="Fo"
    )
    assert added == ["a", "b"]
    assert project.profiles["a"].fo.max() == pytest.approx(0.901)  # percent detected
    with pytest.raises(ValueError, match="already in the project"):
        project.import_profiles(percent, name_column="sample", distance_column="x", fo_column="Fo")
    with pytest.raises(ValueError, match="used by runs"):
        project.remove_profile(EXAMPLE)
    project.remove_profile("b")
    assert "b" not in project.profiles
    project.save()
    assert set(Project.open(project.folder).profiles) == {EXAMPLE, "a"}
    assert set(pd.read_csv(project.folder / "profiles.csv").profile) == {EXAMPLE, "a"}


def test_fit_cache_reloads_and_goes_stale(project):
    result = project.fit(EXAMPLE)
    cached = project.result(EXAMPLE)
    assert cached is not None and cached.time_days == result.time_days
    np.testing.assert_array_equal(cached.curves, result.curves)
    assert cached.time_range_days() == result.time_range_days()
    assert project.summary(EXAMPLE)["time_days"] == result.time_days
    # No pickles anywhere in the cache: a shared folder must not run code when opened.
    files = list((project.folder / WORK / "cache").rglob("*"))
    assert {f.suffix for f in files if f.is_file()} == {".npz", ".json"}
    project.save()
    again = Project.open(project.folder)
    reopened = again.result(EXAMPLE)  # survives reopening
    assert reopened is not None and reopened.time_days == result.time_days
    again.update_run(EXAMPLE, conditions=replace(again.run(EXAMPLE).conditions, T_C=1240.0))
    assert again.result(EXAMPLE) is None and again.is_stale(EXAMPLE)
    assert not project.is_stale(EXAMPLE)


def test_autosave_restore_and_zip_export(project, tmp_path):
    project.save()
    project.duplicate_run(EXAMPLE, "draft")
    project.autosave()
    assert project.has_autosave
    assert Project.open(project.folder).run_names == [EXAMPLE]  # saved tables untouched
    restored = project.restore_autosave()
    assert restored.run_names == [EXAMPLE, "draft"] and restored.dirty
    restored.save()
    assert not restored.has_autosave
    project.fit(EXAMPLE)
    archive = restored.export_zip(tmp_path / "shared")
    names = zipfile.ZipFile(archive).namelist()
    assert any(n.endswith("parameters.csv") for n in names)
    assert not any(WORK in n for n in names)


def test_a_saved_project_runs_with_the_command_line(project, tmp_path):
    project.update_run(EXAMPLE, conditions=Conditions(1240.0, 5e8, 1e-6), fO2_form=None)
    project.save()
    forster = Path(sys.executable).parent / "forster"
    done = subprocess.run(
        [str(forster), "fit", "profiles.csv", "parameters.csv", "--out", str(tmp_path / "out")],
        cwd=project.folder,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert done.returncode == 0, done.stderr
    summary = pd.read_csv(tmp_path / "out" / "summary.csv")
    assert summary.T_C.tolist() == [1240.0]


def test_profile_table_of_no_profiles_has_the_columns():
    assert list(profile_table({}).columns) == ["profile", "distance_um", "fo"]


def test_fo2_and_orientation_follow_the_form_they_were_entered_in(project):
    run = project.run(EXAMPLE)  # fO2 at QFM + 0.3, orientation as Euler angles
    hotter = project.update_run(EXAMPLE, conditions=replace(run.conditions, T_C=1300.0))
    at_buffer = Conditions.from_buffer(1300.0, run.conditions.P_Pa, 0.3, "QFM").fO2_Pa
    assert hotter.fO2_form == ("QFM", 0.3) and hotter.conditions.fO2_Pa == at_buffer
    direct = project.update_run(EXAMPLE, conditions=replace(hotter.conditions, fO2_Pa=1e-5))
    assert direct.fO2_form is None and direct.conditions.fO2_Pa == 1e-5
    turned = project.update_run(EXAMPLE, euler_form=(10.0, 20.0, 30.0, 0.0))
    from forster import Orientation

    assert turned.orientation == Orientation.from_euler(10.0, 20.0, 30.0, 0.0)
    plain = project.update_run(EXAMPLE, orientation=Orientation(90.0, 90.0, 0.0))
    assert plain.euler_form is None
    project.save()
    assert Project.open(project.folder).run(EXAMPLE) == plain


def test_links_in_a_project_folder_are_not_followed(project, tmp_path):
    """A shared project folder may hold symbolic links: export must not package what
    they point to, and saving or caching must not write through them."""
    secret = tmp_path / "secret.txt"
    secret.write_text("private key")
    (project.folder / "notes.txt").symlink_to(secret)
    archive = project.export_zip(tmp_path / "shared")
    with zipfile.ZipFile(archive) as z:
        assert not any("notes" in n for n in z.namelist())
        assert all(b"private key" not in z.read(n) for n in z.namelist())
    outside = tmp_path / "elsewhere.json"
    outside.write_text("{}")
    project.save()
    (project.folder / "project.json").unlink()
    (project.folder / "project.json").symlink_to(outside)
    with pytest.raises(ValueError, match="symbolic link"):
        project.save()
    assert outside.read_text() == "{}"
    (project.folder / "project.json").unlink()
    elsewhere = tmp_path / "cache_elsewhere"
    elsewhere.mkdir()
    (project.folder / WORK).mkdir(exist_ok=True)
    (project.folder / WORK / "cache").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic link"):
        project.fit(EXAMPLE)
    assert list(elsewhere.iterdir()) == []
