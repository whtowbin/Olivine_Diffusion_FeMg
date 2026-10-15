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
    for suffix in ("fit.png", "curves.csv", "points.csv"):
        assert (out / f"{STEM}_{suffix}").stat().st_size > 0
    assert not (out / f"{STEM}_kriged.csv").exists()  # kriging: legacy method only
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
    assert 'failed: "' not in done.stderr  # a KeyError's message, without repr quotes
    assert read_csv(tmp_path / "results" / "summary.csv").run.tolist() == [EXAMPLE]


def test_a_run_that_raises_does_not_lose_the_summary(tmp_path, template):
    """Not only input errors: an OSError writing one run's figure (here a folder where
    its PNG goes) is reported with its type, and the other run still reaches summary.csv."""
    params = read_csv(template / "parameters.csv")
    pd.concat([params.assign(run="blocked"), params]).to_csv(
        template / "parameters.csv", index=False
    )
    (tmp_path / "results" / "blocked_fit.png").mkdir(parents=True)
    done = forster("fit", template / "profiles.csv", template / "parameters.csv", cwd=tmp_path)
    assert done.returncode == 1
    assert "Error" in done.stderr  # IsADirectoryError (or PermissionError on Windows)
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


def test_custom_diffusivity_from_a_table_and_refit_from_the_outputs(tmp_path, template):
    """A run naming a row of diffusivities.csv; the fit writes the equations it used, so
    summary.csv refits with the output folder's diffusivities.csv."""
    table = read_csv(template / "diffusivities.csv")
    mine = table[table.name == "dohmen_chakraborty2007"].assign(name="half_D")
    for axis in ("100", "010", "001"):
        mine[f"log10_D0_{axis}"] -= 0.30103  # D / 2
    pd.concat([table, mine]).to_csv(template / "diffusivities.csv", index=False)
    params = read_csv(template / "parameters.csv")
    pd.concat([params, params.assign(run="slow", diffusivity="half_D")]).to_csv(
        template / "parameters.csv", index=False
    )
    args = ("fit", template / "profiles.csv", template / "parameters.csv")
    missing = forster(*args, cwd=tmp_path)
    assert missing.returncode == 2
    assert "unknown diffusivity 'half_D'" in missing.stderr

    done = forster(*args, "--diffusivities", template / "diffusivities.csv", cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    summary = read_csv(tmp_path / "results" / "summary.csv").set_index("run")
    assert summary.diffusivity.to_dict() == {EXAMPLE: "dohmen_chakraborty2007", "slow": "half_D"}
    assert summary.time_days["slow"] > 1.5 * summary.time_days[EXAMPLE]
    used = read_csv(tmp_path / "results" / "diffusivities.csv")
    assert sorted(used.name) == ["dohmen_chakraborty2007", "half_D"]

    again = forster(
        "fit",
        template / "profiles.csv",
        tmp_path / "results" / "summary.csv",
        "--diffusivities",
        tmp_path / "results" / "diffusivities.csv",
        "--out",
        "again",
        cwd=tmp_path,
    )
    assert again.returncode == 0, again.stderr
    refit = read_csv(tmp_path / "again" / "summary.csv").set_index("run")
    assert refit.time_days.to_dict() == summary.time_days.to_dict()


def test_fit_scans_runs_with_uncertainties_and_the_summary_refits(tmp_path, template):
    params = read_csv(template / "parameters.csv")
    params.assign(steps_sd_um=1.0, edge_fo_range=0.002, scan_points=3).to_csv(
        template / "parameters.csv", index=False
    )
    done = forster("fit", template / "profiles.csv", template / "parameters.csv", cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    assert "scanning 9 initial conditions" in done.stdout
    out = tmp_path / "results"
    scan = read_csv(out / f"{STEM}_scan.csv")
    assert len(scan) == 9 and abs(scan.probability.sum() - 1) < 1e-9
    summary = read_csv(out / "summary.csv")
    row = summary.iloc[0]
    assert row.scan_low_days <= row.scan_best_days <= row.scan_high_days
    assert row.scan_candidates == 9 and row.steps_sd_um == 1.0
    again = forster(
        "fit", template / "profiles.csv", out / "summary.csv", "--out", "again", cwd=tmp_path
    )
    assert again.returncode == 0, again.stderr
    refit = read_csv(tmp_path / "again" / "summary.csv").iloc[0]
    assert (refit.scan_best_days, refit.scan_low_days, refit.scan_high_days) == (
        row.scan_best_days,
        row.scan_low_days,
        row.scan_high_days,
    )
    too_big = forster(
        "fit",
        template / "profiles.csv",
        template / "parameters.csv",
        "--max-candidates",
        "4",
        "--out",
        "big",
        cwd=tmp_path,
    )
    assert too_big.returncode == 1
    assert "more than 4" in too_big.stderr


def test_blank_dx_and_dt_are_chosen_and_recorded(tmp_path, template):
    params = read_csv(template / "parameters.csv")
    params.assign(dx_um=float("nan"), dt_s=float("nan")).to_csv(
        template / "parameters.csv", index=False
    )
    done = forster("fit", template / "profiles.csv", template / "parameters.csv", cwd=tmp_path)
    assert done.returncode == 0, done.stderr
    row = read_csv(tmp_path / "results" / "summary.csv").iloc[0]
    assert row.dx_um > 0 and row.dt_s > 0
    assert row.grid_error_pct <= 1.05 and pd.isna(row.grid_stability)  # implicit: no limit
    again = forster(
        "fit",
        template / "profiles.csv",
        tmp_path / "results" / "summary.csv",
        "--out",
        "again",
        cwd=tmp_path,
    )
    assert again.returncode == 0, again.stderr
    assert read_csv(tmp_path / "again" / "summary.csv").time_days[0] == row.time_days


def test_compare_laws_fits_each_run_with_the_four_choices(tmp_path, template):
    from forster.diffusivity import FE_MG_CHOICES

    done = forster(
        "fit",
        template / "profiles.csv",
        template / "parameters.csv",
        "--compare-laws",
        cwd=tmp_path,
    )
    assert done.returncode == 0, done.stderr
    summary = read_csv(tmp_path / "results" / "summary.csv")
    assert summary.run.tolist() == [f"{EXAMPLE}_{alias}" for alias in FE_MG_CHOICES]
    assert summary.diffusivity.tolist() == [name for name, _ in FE_MG_CHOICES.values()]
    assert summary.diffusivity_label.tolist() == [lab for _, lab in FE_MG_CHOICES.values()]
    assert summary.time_days.nunique() == 4  # the laws differ, so do the times
    for alias in FE_MG_CHOICES:
        assert (tmp_path / "results" / f"{STEM}_{alias}_fit.png").stat().st_size > 0
    # Every law but Classic DIPRA carries a covariance, so it has a law_ range.
    has_law_range = summary.law_low_days.notna().tolist()
    assert has_law_range == [False, True, True, True]


def test_diffusivity_option_overrides_the_column(tmp_path, template):
    done = forster(
        "fit",
        template / "profiles.csv",
        template / "parameters.csv",
        "--diffusivity",
        "refit",
        cwd=tmp_path,
    )
    assert done.returncode == 0, done.stderr
    summary = read_csv(tmp_path / "results" / "summary.csv")
    assert summary.diffusivity.tolist() == ["femg_refit2026"]
    both = forster(
        "fit",
        template / "profiles.csv",
        template / "parameters.csv",
        "--diffusivity",
        "refit",
        "--compare-laws",
        cwd=tmp_path,
    )
    assert both.returncode == 2
    assert "not allowed with" in " ".join(both.stderr.split())
    unknown = forster(
        "fit",
        template / "profiles.csv",
        template / "parameters.csv",
        "--diffusivity",
        "dc07",
        cwd=tmp_path,
    )
    assert unknown.returncode == 2
    assert "unknown diffusivity 'dc07'" in unknown.stderr
