"""`forster.project`: open, edit, save and reopen a project folder; the result cache."""

import json
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
from forster.model import Grid
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


def test_opening_refuses_project_files_that_are_links(project, tmp_path):
    """A received folder whose tables link elsewhere is refused on open, so a linked
    file's contents never reach memory (and a later export)."""
    project.save()
    elsewhere = tmp_path / "other.csv"
    elsewhere.write_text((project.folder / "profiles.csv").read_text())
    (project.folder / "profiles.csv").unlink()
    (project.folder / "profiles.csv").symlink_to(elsewhere)
    with pytest.raises(ValueError, match="symbolic link"):
        Project.open(project.folder)


def test_a_fit_in_a_worker_is_filed_under_the_settings_it_used(project):
    """Edit the run while its fit runs: the result is stored for the old settings, the
    run shows none, and going back to the old settings finds it."""
    task = project.fit_task(EXAMPLE)
    original = project.run(EXAMPLE).conditions
    project.update_run(EXAMPLE, conditions=replace(original, T_C=1260.0))
    assert project.store(task, task()) is False
    assert project.result(EXAMPLE) is None
    project.update_run(EXAMPLE, conditions=original)
    assert project.result(EXAMPLE) is not None


def test_cache_keeps_one_entry_per_run_and_ignores_scan_settings(project):
    project.fit(EXAMPLE)
    cache = project.folder / WORK / "cache"
    assert len(list(cache.iterdir())) == 1
    project.update_run(EXAMPLE, uncertainties={"step1_um": Uncertainty(2.0)}, scan_seed=4)
    assert project.result(EXAMPLE) is not None and not project.is_stale(EXAMPLE)
    project.update_run(EXAMPLE, conditions=replace(project.run(EXAMPLE).conditions, T_C=1255.0))
    project.fit(EXAMPLE)
    assert len(list(cache.iterdir())) == 1  # the old entry went


def test_steps_outside_the_domain_are_refused(project):
    run = project.run(EXAMPLE)
    for steps in ((100.0,), (-5.0,)):
        with pytest.raises(ValueError, match="inside the model domain"):
            project.update_run(EXAMPLE, initial=replace(run.initial, steps_um=steps))
    assert project.run(EXAMPLE) == run


def test_selected_run_and_views_are_saved(project):
    project.selected_run = EXAMPLE
    project.view(EXAMPLE)["xlim"] = [0, 50]
    with pytest.raises(KeyError):
        project.selected_run = "nope"
    project.save()
    again = Project.open(project.folder)
    assert again.selected_run == EXAMPLE and again.view(EXAMPLE) == {"xlim": [0, 50]}


def test_a_crafted_project_file_cannot_delete_outside_the_cache(project, tmp_path):
    """project.json comes from whoever sent the folder: a cache key that is a path must
    not be used to build paths or to delete anything."""
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep.txt").write_text("keep")
    project.save()
    state = json.loads((project.folder / "project.json").read_text())
    state["results"] = {EXAMPLE: "../../../victim"}
    (project.folder / "project.json").write_text(json.dumps(state))
    opened = Project.open(project.folder)
    assert opened.state["results"] == {}  # the malformed key is dropped on open
    opened.state["results"] = {EXAMPLE: "../../../victim"}  # even if set in memory
    opened.fit(EXAMPLE)
    assert (victim / "keep.txt").read_text() == "keep"
    with pytest.raises(ValueError, match="not a cache key"):
        opened._cache("../x")


def test_a_fits_warnings_are_cached_with_it(project):
    """`FitTask.run_collecting` returns the warnings as data; `store` keeps them with the
    result, so a reopened project shows them without refitting."""
    run = project.run(EXAMPLE)
    project.update_run(EXAMPLE, initial=replace(run.initial, domain_um=(2.5, 40.0)))
    assert project.fit_warnings(EXAMPLE) is None
    task = project.fit_task(EXAMPLE)
    outcome = task.run_collecting()
    assert any("past the last step" in m for m in outcome.warnings)
    assert project.store(task, outcome.result, outcome.warnings)
    project.save()
    again = Project.open(project.folder)
    assert again.fit_warnings(EXAMPLE) == outcome.warnings
    reopened = again.result(EXAMPLE)
    assert reopened is not None and reopened.time_s == outcome.result.time_s
    # Project.fit, outside a worker, issues them as fit_run does.
    again.update_run(EXAMPLE, conditions=replace(run.conditions, T_C=1251.0))
    with pytest.warns(UserWarning, match="past the last step"):
        again.fit(EXAMPLE)


def test_the_starting_profile_fit_is_cached_under_its_settings(project):
    from forster.sampling import Cancelled

    priors = {"step1_um": Uncertainty(2.0), "a_fo": Uncertainty(0.002)}
    project.update_run(EXAMPLE, uncertainties=priors, scan_method="laplace")
    project.fit(EXAMPLE)
    assert project.scan(EXAMPLE) is None
    task = project.scan_task(EXAMPLE)
    summary = project.summary(EXAMPLE)
    assert summary is not None and task.time_days == summary["time_days"]
    with pytest.raises(Cancelled):
        task(cancel=lambda: True)
    stages = []
    record = task(stage=stages.append)
    assert stages[:2] == ["best fit", "importance sampling"]
    assert project.store_scan(task, record)
    project.save()
    again = Project.open(project.folder)
    cached = again.scan(EXAMPLE)
    assert cached is not None and record.density is not None
    assert cached.summary == record.summary
    assert cached.best_initial == record.best_initial
    np.testing.assert_array_equal(cached.density, record.density)
    # Another prior is another scan; going back finds the cached one. A Monte Carlo
    # setting does not change the scan.
    again.update_run(EXAMPLE, uncertainties={"step1_um": Uncertainty(3.0)})
    assert again.scan(EXAMPLE) is None
    again.update_run(EXAMPLE, uncertainties=priors)
    assert again.scan(EXAMPLE) is not None
    mc = ConditionUncertainty(T_sd_C=10, fO2_sd_log=0.2, fO2_buffer="QFM", between_study=True)
    again.update_run(EXAMPLE, mc=mc, mc_samples=900, mc_seed=3)
    assert again.scan(EXAMPLE) is not None
    again.rename_run(EXAMPLE, "renamed")
    assert again.state["scans"]["renamed"] == task.key
    again.remove_run("renamed")
    assert "renamed" not in again.state["scans"]


def test_masked_points_stay_in_the_project_and_out_of_the_fits(project, tmp_path):
    points = project.points(EXAMPLE)
    assert not points.masked.any() and len(points) == len(project.profiles[EXAMPLE].fo)
    project.save()
    assert "Marked_bad" not in (project.folder / "profiles.csv").read_text()  # unchanged
    key = project.result_key(EXAMPLE)
    flags = np.zeros(len(points), bool)
    flags[[3, 7]] = True
    project.set_masked(EXAMPLE, flags)
    assert len(project.profiles[EXAMPLE].fo) == len(points) - 2
    assert project.result_key(EXAMPLE) != key  # masking refits
    assert project.points(EXAMPLE).masked.tolist() == flags.tolist()
    project.save()
    # The command line's reader drops the masked rows; the project keeps them.
    from forster.io import read_profiles

    assert len(read_profiles(project.folder / "profiles.csv")[EXAMPLE].fo) == len(points) - 2
    again = Project.open(project.folder)
    assert again.points(EXAMPLE).masked.tolist() == flags.tolist()
    np.testing.assert_array_equal(again.points(EXAMPLE).fo, points.fo)
    again.set_masked(EXAMPLE, np.zeros(len(points), bool))
    assert again.result_key(EXAMPLE) == key and not again.masked
    with pytest.raises(ValueError, match="at least 3"):
        again.set_masked(EXAMPLE, np.arange(len(points)) > 1)
    with pytest.raises(ValueError, match="flags"):
        again.set_masked(EXAMPLE, [True])


def test_moving_a_profile_moves_its_runs_and_a_flip_needs_none(project):
    x = project.profiles[EXAMPLE].distance_um.copy()
    flags = np.zeros(len(x), bool)
    flags[0] = True
    project.set_masked(EXAMPLE, flags)
    project.update_run(EXAMPLE, stages=(Stage(86400.0, 1.0, zone_value=0.9),))
    before = project.run(EXAMPLE)
    project.move_profile(EXAMPLE, offset_um=2.5)
    np.testing.assert_allclose(project.points(EXAMPLE).distance_um, x - 2.5)
    run = project.run(EXAMPLE)
    assert run.initial.domain_um == (before.initial.domain_um[0] - 2.5, 69.5)
    assert run.initial.steps_um == (21.5,) and run.stages[0].rim_um == -1.5
    with pytest.raises(ValueError, match="flip it before adding runs"):
        project.move_profile(EXAMPLE, flip=True, offset_um=70.0)
    project.remove_run(EXAMPLE)
    project.move_profile(EXAMPLE, flip=True, offset_um=x.max() - 2.5)
    moved = project.points(EXAMPLE)
    assert moved.distance_um.iloc[0] == 0.0 and moved.distance_um.is_monotonic_increasing
    assert moved.masked.iloc[-1]  # the masked first point is now the last


def test_the_monte_carlo_is_cached_and_matches_the_command_line(project):
    from forster.io import mc_run

    with pytest.raises(ValueError, match="Monte Carlo"):
        project.mc_task(EXAMPLE)
    mc = ConditionUncertainty(T_sd_C=10.0, fO2_sd_log=0.2)
    project.update_run(EXAMPLE, mc=mc, mc_samples=400, mc_seed=2)
    # Blank mc_fO2_buffer on a buffer-relative run reads back as its own buffer.
    assert project.run(EXAMPLE).mc == replace(mc, fO2_buffer="QFM")
    with pytest.raises(ValueError, match="fit the run first"):
        project.mc_task(EXAMPLE)
    result = project.fit(EXAMPLE)
    task = project.mc_task(EXAMPLE)
    assert task.posterior is None
    record = task()
    expected = mc_run(project.run(EXAMPLE), result)
    np.testing.assert_array_equal(record.days, expected.days)
    assert record.summary["mc_draws"] == 400
    assert project.mc(EXAMPLE) is None
    assert project.store_mc(task, record)
    project.save()
    again = Project.open(project.folder)
    cached = again.mc(EXAMPLE)
    assert cached is not None and cached.summary == record.summary
    np.testing.assert_array_equal(cached.distribution.days, expected.days)
    again.update_run(EXAMPLE, mc_seed=5)
    assert again.mc(EXAMPLE) is None
    again.rename_run(EXAMPLE, "renamed")
    again.remove_run("renamed")
    assert "renamed" not in again.state["mcs"]


def test_a_refitted_monte_carlo_reports_progress_and_cancels(project):
    from forster.sampling import Cancelled

    project.update_run(
        EXAMPLE,
        stages=(Stage(86400.0, 1.0, zone_value=0.9),),
        grid=Grid(0.5, 2000.0, 20.0),
        mc=ConditionUncertainty(T_sd_C=5.0),
        mc_samples=3,
    )
    project.fit(EXAMPLE)
    task = project.mc_task(EXAMPLE)
    with pytest.raises(Cancelled):
        task(cancel=lambda: True)
    seen = []
    record = task(progress=lambda done, n: seen.append((done, n)))
    assert record.method == "refit" and seen == [(1, 3), (2, 3), (3, 3)]


def test_a_variant_fit_is_keyed_by_its_own_settings(project):
    other = resolve("refit")
    task = project.variant_task(EXAMPLE, diffusivity=other)
    assert task.run.diffusivity == other and project.run(EXAMPLE).diffusivity != other
    assert task.key != project.result_key(EXAMPLE)
    assert project.variant_task(EXAMPLE).key == project.result_key(EXAMPLE)
    assert project.cached(task) is None
    outcome = task.run_collecting()
    assert not project.store(task, outcome.result)  # not the run's own settings
    cached = project.cached(task)
    assert cached is not None and cached.time_s == outcome.result.time_s
    assert project.result(EXAMPLE) is None
