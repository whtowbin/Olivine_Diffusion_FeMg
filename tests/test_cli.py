"""`forster template` and `forster fit`, run as the installed command in a subprocess."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest
from conftest import PAPER_DATA, PAPER_PARAMS

from forster import fit_run, read_profiles, read_runs

FORSTER = Path(sys.executable).parent / "forster"
EXAMPLE = "AZ18 WHT01_Ol_63_prof_1"
STEM = "AZ18_WHT01_Ol_63_prof_1"


def read_csv(path):
    """Exactly the floats written (pandas' default parser can be 1 ulp off)."""
    return pd.read_csv(path, float_precision="round_trip")  # ty: ignore[no-matching-overload]


def forster(*args, cwd):
    return subprocess.run(
        [str(FORSTER), *map(str, args)], cwd=cwd, capture_output=True, text=True, timeout=300
    )


@pytest.fixture
def template(tmp_path):
    done = forster("template", "in", cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    return tmp_path / "in"


def test_template_then_fit_writes_every_output(tmp_path, template):
    done = forster("fit", template / "profiles.csv", template / "parameters.csv", cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    out = tmp_path / "results"
    for suffix in ("fit.png", "curves.csv", "kriged.csv", "points.csv"):
        assert (out / f"{STEM}_{suffix}").stat().st_size > 0
    summary = read_csv(out / "summary.csv")
    assert summary.run.tolist() == [EXAMPLE]
    expected = fit_run(
        read_runs(template / "parameters.csv")[0], read_profiles(template / "profiles.csv")
    )
    assert summary.time_days[0] == expected.time_days
    assert summary.uncertainty[0] == "correlated"
    curves = read_csv(out / f"{STEM}_curves.csv")
    assert curves.best_fit_fo.tolist() == expected.best_fit.tolist()

    # The summary is itself a parameter table: rerunning it gives the same fit.
    again = forster(
        "fit", template / "profiles.csv", out / "summary.csv", "--out", "again", cwd=tmp_path
    )
    assert again.returncode == 0, again.stderr
    assert read_csv(tmp_path / "again" / "summary.csv").time_days[0] == expected.time_days


def test_template_refuses_to_overwrite(tmp_path, template):
    (template / "parameters.csv").write_text("mine")
    done = forster("template", "in", cwd=tmp_path)
    assert done.returncode == 2
    assert "--force" in done.stderr
    assert (template / "parameters.csv").read_text() == "mine"


def test_a_failing_run_is_skipped_and_reported(tmp_path, template):
    params = read_csv(template / "parameters.csv")
    broken = params.assign(run="no profile", profile="not in the table")
    pd.concat([broken, params]).to_csv(template / "parameters.csv", index=False)
    done = forster("fit", template / "profiles.csv", template / "parameters.csv", cwd=tmp_path)
    assert done.returncode == 1
    assert "not in the table" in done.stderr
    assert read_csv(tmp_path / "results" / "summary.csv").run.tolist() == [EXAMPLE]


def test_run_option_selects_runs_and_rejects_unknown_names(tmp_path, template):
    params = read_csv(template / "parameters.csv")
    pd.concat([params, params.assign(run="copy")]).to_csv(template / "parameters.csv", index=False)
    args = ("fit", template / "profiles.csv", template / "parameters.csv")
    done = forster(*args, "--run", "copy", cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    assert read_csv(tmp_path / "results" / "summary.csv").run.tolist() == ["copy"]
    unknown = forster(*args, "--run", "nope", cwd=tmp_path)
    assert unknown.returncode == 2
    assert "nope" in unknown.stderr


def test_runs_that_would_share_file_names_are_rejected(tmp_path, template):
    params = read_csv(template / "parameters.csv")
    pd.concat([params.assign(run="a b"), params.assign(run="a/b")]).to_csv(
        template / "parameters.csv", index=False
    )
    done = forster("fit", template / "profiles.csv", template / "parameters.csv", cwd=tmp_path)
    assert done.returncode == 2
    assert "same files" in done.stderr


def test_excel_tables_in_the_paper_layout(tmp_path):
    run = "AZ18 WHT06_ol_overgrowth1_prof_1"
    done = forster(
        "fit",
        PAPER_DATA,
        PAPER_PARAMS,
        "--profiles-sheet",
        "Olivine",
        "--run",
        run,
        cwd=tmp_path,
    )
    assert done.returncode == 0, done.stderr
    assert "warning: " in done.stderr and "fewer than 3 points" in done.stderr
    summary = read_csv(tmp_path / "results" / "summary.csv")
    assert summary.run.tolist() == [run]
    assert summary.time_days[0] > 0
