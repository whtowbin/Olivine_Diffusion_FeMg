"""CITATION.cff, `forster.cite` and `forster cite` agree with each other and the package."""

import subprocess
import sys
import tomllib
from importlib.metadata import version
from pathlib import Path

import pytest
import yaml

from forster import citation, cite

ROOT = Path(__file__).resolve().parent.parent
FORSTER = Path(sys.executable).parent / "forster"


@pytest.fixture(scope="module")
def cff():
    return yaml.safe_load((ROOT / "CITATION.cff").read_text(encoding="utf-8"))


def test_cff_has_the_required_keys_and_cites_the_software(cff):
    for key in ("cff-version", "message", "title", "authors"):
        assert key in cff
    assert cff["cff-version"] == "1.2.0"
    assert cff["type"] == "software"
    # GitHub's "Cite this repository" must cite the software, not the paper.
    assert "preferred-citation" not in cff
    assert {r["type"] for r in cff["references"]} == {"article", "thesis"}
    assert cff["references"][0]["doi"] == citation.PAPER_DOI


def test_cff_and_the_package_agree(cff):
    assert cff["title"] == citation.TITLE
    assert cff["url"] == citation.URL
    assert cff["repository-code"] == citation.REPOSITORY
    names = [f"{a['family-names']}, {a['given-names']}" for a in cff["authors"]]
    assert tuple(names) == citation.AUTHORS
    # the Zenodo DOI goes under identifiers, as the TODO in CITATION.cff says
    dois = [i["value"] for i in cff.get("identifiers", []) if i["type"] == "doi"]
    assert "doi" not in cff
    assert (dois[0] if dois else None) == citation.DOI
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert cff["version"] == pyproject["version"]
    assert cff["license"] == pyproject["license"]
    assert cff["keywords"] == pyproject["keywords"]


def test_cite_names_the_software_version_and_paper():
    text = cite()
    assert "FORSTER" in text
    assert version("forster") in text
    assert "10.1093/petrology/egag077" in text
    assert "diffusion equation" not in text


def test_cite_gives_the_source_of_the_diffusion_equation():
    text = cite("dipra")
    assert "Dohmen and Chakraborty" in text
    assert "DIPRA" in text
    # an alias and the preset name it stands for give the same text
    assert cite("refit") == cite("femg_refit2026")
    assert "refit" in cite("refit").lower()
    with pytest.raises(ValueError, match="unknown diffusivity"):
        cite("nonsense")


def test_forster_cite_prints_the_same_text():
    done = subprocess.run(
        [str(FORSTER), "cite", "--diffusivity", "dipra"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout == cite("dipra")
    plain = subprocess.run([str(FORSTER), "cite"], capture_output=True, text=True, timeout=120)
    assert plain.stdout == cite()
    bad = subprocess.run(
        [str(FORSTER), "cite", "--diffusivity", "nonsense"],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert bad.returncode == 2
