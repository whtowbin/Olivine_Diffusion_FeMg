"""`forster.imports`: column mapping, units, Fo from oxides, flagged totals, import
templates, and the `forster import` command."""

import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from forster.imports import (
    ColumnMap,
    ImportTemplate,
    TemplateMismatch,
    apply,
    check_columns,
    guess,
    import_files,
    list_profile_files,
    read_with_template,
    template_table,
)
from forster.io import read_profiles

MGO, FEO = 40.3044, 71.8444


def _epma():
    """Two profiles of raw EPMA analyses: distance in mm, oxides in wt%, Ni in ppm, stage
    coordinates, and one point with a low total."""
    rows = []
    for name, step in (("ol1 line A", 0.010), ("ol2 line B", 0.020)):
        for i in range(6):
            mgo = 46.0 + 0.6 * i
            feo = 14.0 - 0.6 * i
            rows.append(
                {
                    "Sample": name,
                    "Distance (mm)": round(i * step, 4),
                    "SiO2": 40.5,
                    "FeO": feo,
                    "MgO": mgo,
                    "NiO": 0.30,
                    "Ni (ppm)": 2400.0,
                    "Total": 40.5 + feo + mgo + 0.30,
                    "X": 1000.0 + i,
                    "Y": 2000.0,
                    "Comment": "spot",
                }
            )
    rows[3]["Total"] = 97.2  # a bad point: flagged, kept
    return pd.DataFrame(rows)


def test_guess_proposes_roles_and_units_from_headers_and_values():
    template = guess(_epma())
    roles = {c.column: (c.role, c.quantity, c.unit) for c in template.columns}
    assert roles["Sample"] == ("profile", None, None)
    assert roles["Distance (mm)"] == ("position", None, "mm")
    assert roles["FeO"] == ("oxide", "FeO", "wt%")
    assert roles["Ni (ppm)"] == ("element", "Ni", "ppm")
    assert roles["Total"][0] == "total"
    assert (roles["X"][0], roles["Y"][0]) == ("stage_x", "stage_y")
    assert roles["Comment"][0] == "ignore"
    assert template.problems() == []


def test_fo_from_oxides_in_moles_with_all_iron_as_feo_and_flagged_totals():
    df = _epma()
    result = apply(df, guess(df))
    first = result.profiles["ol1 line A"]
    mg, fe = (46.0 + 0.6 * np.arange(6)) / MGO, (14.0 - 0.6 * np.arange(6)) / FEO
    np.testing.assert_allclose(first.fo, mg / (mg + fe), rtol=1e-12)
    np.testing.assert_allclose(first.distance_um, 10.0 * np.arange(6))  # mm -> um
    assert len(first.fo) == 6  # the low-total point is kept
    assert len(result.flagged) == 1 and result.flagged["total_wt%"].iloc[0] == 97.2
    assert any("outside 98.5-101.5 wt%" in n for n in result.notes)
    assert {"Ni_wt%", "NiO_wt%", "stage_x", "stage_y"} <= set(result.points.columns)
    assert result.points["Ni_wt%"].iloc[0] == pytest.approx(0.24)  # 2400 ppm
    wider = apply(df, ImportTemplate(guess(df).columns, total_window=(97.0, 102.0)))
    assert len(wider.flagged) == 0


def test_an_ambiguous_unit_stops_the_import():
    df = _epma().rename(columns={"Ni (ppm)": "Ni"})
    template = guess(df)
    assert any("give the unit" in p for p in template.problems())
    with pytest.raises(ValueError, match="not complete"):
        apply(df, template)


def test_templates_save_reload_and_refuse_unknown_or_missing_columns(tmp_path):
    df = _epma()
    path = guess(df).save(tmp_path / "epma.import.json")
    template = ImportTemplate.load(path)
    assert template.columns == guess(df).columns and template.name == "epma.import.json"
    check_columns(df.columns, template)  # same columns: fine
    with pytest.raises(TemplateMismatch, match="does not know: \\['Beam current'\\]"):
        check_columns([*df.columns, "Beam current"], template)
    with pytest.raises(TemplateMismatch, match="lacks: \\['MgO'\\]"):
        apply(df.drop(columns=["MgO"]), template)


def test_the_lab_sheet_imports_like_read_profiles():
    """The real lab spreadsheet (44 columns): the guess picks Profile_Name, Distance µm and
    Fo# in percent, and the import gives exactly read_profiles' profiles."""
    from conftest import PAPER_DATA

    from forster.io import read_table

    df = read_table(PAPER_DATA, "Olivine")
    template = guess(df)
    picked = {c.role: c.column for c in template.columns if c.role in ("profile", "position", "fo")}
    assert picked == {"profile": "Profile_Name", "position": "Distance µm", "fo": "Fo#"}
    assert template.problems() == []
    imported = apply(df, template).profiles
    with pytest.warns(UserWarning):
        expected = read_profiles(PAPER_DATA, sheet="Olivine")
    assert imported.keys() == expected.keys()
    for name, p in expected.items():
        q = imported[name]
        a, b = np.lexsort((p.fo, p.distance_um)), np.lexsort((q.fo, q.distance_um))
        np.testing.assert_array_equal(q.distance_um[b], p.distance_um[a])
        np.testing.assert_array_equal(q.fo[b], p.fo[a])


def test_forster_import_proposes_then_applies_a_confirmed_template(tmp_path):
    source = tmp_path / "epma.csv"
    _epma().to_csv(source, index=False)
    forster = [sys.executable, "-m", "forster.cli", "import", str(source)]
    proposed = subprocess.run(forster, capture_output=True, text=True, timeout=120)
    assert proposed.returncode == 3, proposed.stderr
    draft = tmp_path / "epma.import.json"
    assert draft.exists() and "not applied" in proposed.stdout
    out = tmp_path / "profiles.csv"
    done = subprocess.run(
        [*forster, "--template", str(draft), "--out", str(out)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    assert "imported with epma.import.json" in done.stdout
    profiles = read_profiles(out)
    assert set(profiles) == {"ol1 line A", "ol2 line B"}
    result = read_with_template(source, ImportTemplate.load(draft))
    np.testing.assert_array_equal(profiles["ol2 line B"].fo, result.profiles["ol2 line B"].fo)


# ------------------------------------------------------------------------- a folder


def _one_profile(path, shift=0.0, name=None):
    """A small per-profile file: distance in um, Fo in percent (and a name column when
    ``name`` is given)."""
    df = pd.DataFrame({"Distance (um)": np.arange(6) * 10.0, "Fo": 88.0 + shift + np.arange(6)})
    if name:
        df.insert(0, "Profile", name)
    df.to_csv(path, index=False)
    return path


def test_a_folder_without_a_name_column_names_profiles_after_the_files(tmp_path):
    for i, stem in enumerate(("c_grain", "a_grain", "b_grain")):
        _one_profile(tmp_path / f"{stem}.csv", shift=i)
    batch = import_files(tmp_path)
    assert list(batch.profiles) == ["a_grain", "b_grain", "c_grain"]  # sorted by file name
    assert not batch.problems and len(batch.points) == 18
    np.testing.assert_allclose(batch.profiles["c_grain"].fo[0], 0.88)
    np.testing.assert_allclose(batch.profiles["a_grain"].fo[0], 0.89)
    assert batch.sources["b_grain"].name == "b_grain.csv"
    assert batch.template.by_role("position")[0].unit == "um"


def test_a_folder_with_a_name_column_uses_it(tmp_path):
    _one_profile(tmp_path / "one.csv", name="olA")
    _one_profile(tmp_path / "two.csv", shift=2, name="olB")
    batch = import_files(tmp_path)
    assert list(batch.profiles) == ["olA", "olB"]


def test_a_file_that_does_not_match_is_reported_by_name(tmp_path):
    _one_profile(tmp_path / "a.csv")
    _one_profile(tmp_path / "c.csv", shift=1)
    pd.DataFrame({"Distance (um)": [0, 1, 2], "Fo": [88, 89, 90], "Extra": [1, 2, 3]}).to_csv(
        tmp_path / "b.csv", index=False
    )
    batch = import_files(tmp_path)
    assert list(batch.profiles) == ["a", "c"]  # the others still come in
    assert [p.name for p in batch.problems] == ["b.csv"]
    assert "Extra" in next(iter(batch.problems.values()))
    with pytest.raises(ValueError, match=r"b\.csv.*Extra"):
        batch.raise_for_problems()


def test_the_same_profile_name_in_two_files_is_refused_naming_both(tmp_path):
    _one_profile(tmp_path / "one.csv", name="ol")
    _one_profile(tmp_path / "two.csv", name="ol")
    with pytest.raises(ValueError, match=r"'ol'.*one\.csv.*two\.csv"):
        import_files(tmp_path)


def test_lock_and_hidden_files_and_other_types_are_ignored(tmp_path):
    _one_profile(tmp_path / "real.csv")
    (tmp_path / "~$real.xlsx").write_bytes(b"lock")
    (tmp_path / ".hidden.csv").write_text("not,a,table\n")
    (tmp_path / "notes.txt").write_text("x")
    (tmp_path / "sub").mkdir()
    assert [p.name for p in list_profile_files(tmp_path)] == ["real.csv"]
    assert list(import_files(tmp_path).profiles) == ["real"]


def test_a_list_of_files_and_an_xlsx_work_and_a_given_template_is_used(tmp_path):
    csv = _one_profile(tmp_path / "x.csv")
    xlsx = tmp_path / "y.xlsx"
    pd.read_csv(csv).to_excel(xlsx, index=False)
    template = guess(pd.read_csv(csv), name="mine.import.json")
    batch = import_files([csv, xlsx], template)
    assert list(batch.profiles) == ["x", "y"] and batch.template.name == "mine.import.json"


def test_an_empty_folder_is_an_error(tmp_path):
    with pytest.raises(ValueError, match=r"no \.csv"):
        import_files(tmp_path)


def test_forster_import_on_a_folder(tmp_path):
    folder = tmp_path / "grains"
    folder.mkdir()
    for i, stem in enumerate(("g1", "g2")):
        _one_profile(folder / f"{stem}.csv", shift=i)
    forster = [sys.executable, "-m", "forster.cli", "import", str(folder)]
    proposed = subprocess.run(forster, capture_output=True, text=True, timeout=120)
    assert proposed.returncode == 3, proposed.stderr
    draft = tmp_path / "grains.import.json"
    assert draft.exists() and "first of 2 files" in proposed.stdout
    done = subprocess.run([*forster, "--template", str(draft)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert set(read_profiles(tmp_path / "grains_profiles.csv")) == {"g1", "g2"}
    # a file with another layout: nothing is written, the file is named
    (tmp_path / "grains_profiles.csv").unlink()
    pd.DataFrame({"Distance (um)": [0, 1, 2], "Fo": [88, 89, 90], "Z": [1, 2, 3]}).to_csv(
        folder / "g3.csv", index=False
    )
    bad = subprocess.run([*forster, "--template", str(draft)], capture_output=True, text=True)
    assert bad.returncode == 1 and "g3.csv" in bad.stderr
    assert not (tmp_path / "grains_profiles.csv").exists()


def test_numeric_exclusion_flags_work_when_the_column_has_blanks(tmp_path):
    """A column with blanks reads as float, so the flag 1 arrives as 1.0."""
    df = pd.DataFrame(
        {
            "Profile": "p1",
            "Distance (um)": np.arange(8) * 5.0,
            "Fo": 88.0 + np.arange(8),
            "Exclude": [np.nan, 1, np.nan, np.nan, 1, np.nan, np.nan, np.nan],
        }
    )
    path = tmp_path / "flags.csv"
    df.to_csv(path, index=False)
    result = read_with_template(path, guess(pd.read_csv(path)))
    assert len(result.excluded) == 2 and len(result.points) == 6


@pytest.mark.parametrize(
    ("header", "unit"),
    [
        ("Distance [mm]", "mm"),
        ("Distance/mm", "mm"),
        ("Distance (mm)", "mm"),
        ("Distance_mm", "mm"),
        ("Distance mm", "mm"),
        ("Distance (µm)", "um"),
        ("Distance/um", "um"),
        ("Distance, comment", "um"),
        ("Distance summary", "um"),
    ],
)
def test_the_position_unit_is_read_from_the_header(header, unit):
    from forster.imports import _position_unit

    assert _position_unit(header) == unit


def test_forsters_own_oxide_headers_import_again(tmp_path):
    """`template_table` writes MgO_wt%, FeO_wt% and total_wt%; the mapping must find them."""
    result = apply(_epma(), guess(_epma()))
    out = template_table(result.points.drop(columns=["total_ok"]))
    path = tmp_path / "again.csv"
    out.to_csv(path, index=False)
    again = guess(pd.read_csv(path))
    quantities = {(c.role, c.quantity) for c in again.columns if c.role == "oxide"}
    assert {("oxide", "MgO"), ("oxide", "FeO"), ("oxide", "SiO2")} <= quantities
    assert ("MgO_wt%" in [c.column for c in again.columns]) and again.problems() == []


def test_a_template_round_trips_non_ascii_names(tmp_path):
    template = ImportTemplate(
        (
            ColumnMap("Distance µm", "position", unit="um"),
            ColumnMap("Fo μ", "fo", unit="percent"),
        ),
        name="lab µ",
    )
    path = template.save(tmp_path / "t.import.json")
    assert "µ" in path.read_bytes().decode("utf-8")
    assert ImportTemplate.load(path) == template


def test_a_hand_edited_template_with_a_bad_key_is_a_clear_error(tmp_path):
    good = guess(_epma()).to_dict()
    good["columns"][0]["colum"] = good["columns"][0].pop("column")
    path = tmp_path / "bad.import.json"
    path.write_text(json.dumps(good), encoding="utf-8")
    with pytest.raises(ValueError, match="colum"):
        ImportTemplate.load(path)
    path.write_text(json.dumps({"format": "forster-import", "version": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="columns"):
        ImportTemplate.load(path)
    source = tmp_path / "epma.csv"
    _epma().to_csv(source, index=False)
    done = subprocess.run(
        [sys.executable, "-m", "forster.cli", "import", str(source), "--template", str(path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 2 and "columns" in done.stderr
    assert "Traceback" not in done.stderr


def test_blank_unnamed_columns_do_not_refuse_a_file(tmp_path):
    folder = tmp_path / "grains"
    folder.mkdir()
    _one_profile(folder / "g1.csv")
    template = guess(pd.read_csv(folder / "g1.csv"))
    stray = pd.read_csv(folder / "g1.csv").assign(**{"Unnamed: 7": np.nan})
    stray.to_csv(folder / "g2.csv", index=False)
    stray["Unnamed: 8"] = 1.0  # a column with something in it is still refused
    stray.to_csv(folder / "g3.csv", index=False)
    batch = import_files(folder, template)
    assert set(batch.profiles) == {"g1", "g2"} and list(batch.problems) == [folder / "g3.csv"]


def test_unreadable_excel_files_are_listed_as_problems(tmp_path):
    folder = tmp_path / "grains"
    folder.mkdir()
    _one_profile(folder / "g1.csv")
    (folder / "g2.xls").write_text("Distance (um)\tFo\n0\t88\n")  # tab text, not Excel
    (folder / "g3.xlsx").write_bytes(b"PK not a workbook")
    batch = import_files(folder, guess(pd.read_csv(folder / "g1.csv")))
    assert set(batch.profiles) == {"g1"}
    assert {p.name for p in batch.problems} == {"g2.xls", "g3.xlsx"}
    assert all("Excel file" in why for why in batch.problems.values())
    draft = tmp_path / "grains.import.json"
    guess(pd.read_csv(folder / "g1.csv")).save(draft)
    done = subprocess.run(
        [sys.executable, "-m", "forster.cli", "import", str(folder), "--template", str(draft)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 1 and "g2.xls" in done.stderr and "Excel file" in done.stderr


def test_cli_prints_an_unreadable_excel_file_as_a_refusal(tmp_path):
    path = tmp_path / "bad.xls"
    path.write_text("not an excel file")
    done = subprocess.run(
        [sys.executable, "-m", "forster.cli", "import", str(path)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 2 and done.stderr.startswith("forster: ")
    assert "Traceback" not in done.stderr


def test_forster_import_dot_resolves_the_folder(tmp_path):
    folder = tmp_path / "grains"
    folder.mkdir()
    _one_profile(folder / "g1.csv")
    done = subprocess.run(
        [sys.executable, "-m", "forster.cli", "import", "."],
        cwd=folder,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 3, done.stderr  # proposed, awaiting a check
    assert (tmp_path / "grains.import.json").exists()
