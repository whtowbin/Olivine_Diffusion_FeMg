"""The public tree holds no round robin results (Henry, 2026-10-17).

The round robin profiles (other laboratories' samples) and everything derived from them
(fits, times, ranges, noise numbers, per-profile rows, figures) live in the private
submodule "Diffusion Round Robin/". Old commits keep earlier versions; the current files
must not. The check is deliberately simple: no tracked file outside the submodule names a
round robin sample, in its path or its contents, unless it is code on the allow-list
below, which may use the names only as identifiers to read the private data when it is
checked out.
"""

import re
import subprocess

import pytest
from conftest import REPO_ROOT

SAMPLE_NAMES = re.compile(rb"SH63|KS20")

# Path -> why it may name a round robin sample. Only code that reads the private data when
# the submodule is present (and writes nothing derived from it to the public tree), and
# tests of this rule. Keep the list short; prefer removing the name.
ALLOWED = {
    # This file and the docs-site test hold the pattern they search for.
    "tests/test_public_tree.py": "holds the pattern it searches for",
    "tests/test_docs.py": "holds the pattern the docs-site check searches for",
    # Instructions to agents: name the samples so they recognise private data.
    ".claude/skills/forster-safe-changes/SKILL.md": "names the private samples as examples",
    # Tests that read the private tables and skip without the submodule.
    "tests/test_compiled.py": "reads private runs by name; skips without the submodule",
    "tests/test_differential.py": "reads private runs by name; skips without the submodule",
    "tests/test_grid.py": "reads private runs by name; skips without the submodule",
    "tests/test_implicit.py": "reads private runs by name; skips without the submodule",
    "tests/test_io.py": "reads private runs by name; skips without the submodule",
    "tests/test_scan.py": "reads private runs by name; skips without the submodule",
    "tests/test_uncertainty.py": "reads private runs by name; skips without the submodule",
    # Research scripts that read the private data and write round robin rows only there.
    "analyses/round_robin/Round_Robin_Diffusion_Modeling Part 2.py": "round robin script",
    "analyses/round_robin/refit_2026.py": "round robin refit; writes to the submodule",
    "analyses/ic_sampling/references.py": "reads private runs; their rows go to the submodule",
    "analyses/ic_sampling/sweep_methods.py": "reads private runs; prints to the terminal only",
    "analyses/femg_refit/fit_models.py": "reads the 1102 C run's P and fO2 from the private table",
    "docs/plans/2026-10-15-whitening-validity/synthetic.py": (
        "reads two private profiles' point positions to build synthetic geometries"
    ),
    "docs/plans/2026-10-13-likelihood-validation/validate.py": (
        "reads private profiles to build synthetic geometries"
    ),
    "docs/plans/2026-10-13-likelihood-validation/compare_real.py": (
        "reads private runs; their rows go to the submodule"
    ),
    "docs/references/figures/noise_findings/findings.py": "reads private runs when present",
    "docs/references/figures/noise_findings/make_figures.py": (
        "private figures only with --private, written to the submodule"
    ),
    "docs/references/figures/noise_models/make_figures.py": (
        "private figures only with --private, written to the submodule"
    ),
    # Grid-choice benchmarks: run lists of the private profiles (timings and grid errors).
    "docs/plans/2026-10-13-grid-choice/bench_numba.py": "grid benchmark; reads private runs",
    "docs/plans/2026-10-13-grid-choice/calib_grid.py": "grid benchmark; reads private runs",
    "docs/plans/2026-10-13-grid-choice/calib_grid2.py": "grid benchmark; reads private runs",
    "docs/plans/2026-10-13-grid-choice/compiled.py": "grid benchmark; reads private runs",
    "docs/plans/2026-10-13-grid-choice/explicit_control.py": "grid benchmark; reads private runs",
    "docs/plans/2026-10-13-grid-choice/gridbench.py": "grid benchmark; reads private runs",
    "docs/plans/2026-10-13-grid-choice/implicit_study.py": "grid benchmark; reads private runs",
    "docs/plans/2026-10-13-grid-choice/trbdf2_check.py": "grid benchmark; reads private runs",
}


def _tracked_files() -> list[str]:
    try:
        out = subprocess.run(
            ["git", "ls-files", "-s", "-z"],
            cwd=REPO_ROOT,
            capture_output=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("not a git checkout")
    paths = []
    for entry in out.split(b"\0"):
        if not entry:
            continue
        meta, path = entry.split(b"\t", 1)
        if meta.startswith(b"160000"):  # a submodule (the private round robin data)
            continue
        paths.append(path.decode())
    return paths


def test_no_tracked_file_names_a_round_robin_sample():
    hits = []
    for path in _tracked_files():
        if path in ALLOWED:
            continue
        file = REPO_ROOT / path
        if SAMPLE_NAMES.search(path.encode()):
            hits.append(f"{path} (file name)")
        elif file.is_file() and SAMPLE_NAMES.search(file.read_bytes()):
            hits.append(path)
    assert not hits, (
        "round robin sample names in public files (move the results to the private "
        f"submodule, or justify an allow-list entry in {__name__}): {hits}"
    )


def test_every_allowed_path_is_tracked_and_still_needs_its_entry():
    tracked = set(_tracked_files())
    stale = [
        path
        for path in ALLOWED
        if path not in tracked or not SAMPLE_NAMES.search((REPO_ROOT / path).read_bytes())
    ]
    assert not stale, f"allow-list entries no longer needed: {stale}"
