"""`forster.imports`: column mapping, units, Fo from oxides, flagged totals, import
templates, and the `forster import` command."""

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from forster.imports import (
    ImportTemplate,
    TemplateMismatch,
    apply,
    check_columns,
    guess,
    read_with_template,
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
