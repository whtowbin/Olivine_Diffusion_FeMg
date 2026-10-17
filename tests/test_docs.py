"""The documentation site (docs/site, built with MkDocs): pages generated from the code
are up to date, and the site holds no round robin material."""

import importlib.util
import re

from conftest import REPO_ROOT

SITE = REPO_ROOT / "docs" / "site"


def _gen_columns():
    spec = importlib.util.spec_from_file_location(
        "gen_columns", REPO_ROOT / "docs" / "gen_columns.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_column_reference_page_matches_the_template_columns():
    gen = _gen_columns()
    committed = (SITE / "columns.md").read_text(encoding="utf-8")
    assert committed == gen.columns_page(), (
        "docs/site/columns.md is out of date; run: uv run python docs/gen_columns.py"
    )


def test_column_reference_page_has_every_template_column():
    from forster.template import PARAMETER_COLUMNS, PROFILE_COLUMNS, SUMMARY_COLUMNS

    page = (SITE / "columns.md").read_text(encoding="utf-8")
    for column in (*PROFILE_COLUMNS, *PARAMETER_COLUMNS, *SUMMARY_COLUMNS):
        assert f"| `{column.name}` |" in page


def test_site_pages_name_no_round_robin_sample():
    # The round robin profiles belong to other laboratories and stay private until the
    # round robin is published (forster-safe-changes skill): no sample names, numbers or
    # figures of theirs on the public site.
    private = re.compile(r"SH63|KS20|KS-20|olv066|527[_-]")
    hits = [
        f"{path.relative_to(SITE)}: {match.group(0)}"
        for path in SITE.rglob("*.md")
        for match in private.finditer(path.read_text(encoding="utf-8"))
    ]
    assert not hits, hits


def test_every_image_on_the_site_is_used_and_every_used_image_exists():
    pages = "\n".join(p.read_text(encoding="utf-8") for p in SITE.rglob("*.md"))
    used = set(re.findall(r"images/([\w.-]+\.png)", pages))
    present = {p.name for p in (SITE / "images").glob("*.png")}
    assert used <= present, sorted(used - present)
    assert present <= used, sorted(present - used)
