---
name: forster-safe-changes
description: Use before committing any change to FORSTER's src/forster that should leave results unchanged (a refactor, a rename, a new output column), or any run that could be memory-heavy, or any work touching the round robin data in "Diffusion Round Robin/".
---

# Changing FORSTER safely

## Prove a behaviour-preserving change with a differential replay
- `analyses/replay/replay.py OUT.json` fits every existing run (the template example
  with both starting-profile fit methods, the Monte Carlo and a summary round trip,
  every paper run, every round robin run) and writes the outputs, curves as hashes.
- Run it on the old code, on the new code, and on the old code again (the control must
  show 0 differences); compare with `analyses/replay/compare.py OLD.json NEW.json`.
  Hand-check a few compared values before trusting a 0.
- Write the JSON outside the repository (it holds round robin results).
- An intended change (a fix) must show up only where expected: say which outputs moved
  and by how much in the commit message and CHANGELOG.

## Checks before calling work done
`uv run ruff check src tests`, `uv run ruff format --check src tests`, `uv run ty check`,
`uv run complexipy src/forster --max-complexity-allowed 25`, `uv run pytest -n 4`.
Do not add `-q`: pyproject's addopts already has it, and `-qq` drops the
"N passed" summary line, so the result cannot be quoted.
Run them once over the finished change. A new summary or parameter column needs its
`forster.template` entry, and the documentation's column page must be regenerated
(`uv run python docs/gen_columns.py`; `tests/test_docs.py` checks it). The README no
longer holds the column tables; the user guide is the MkDocs site in `docs/site/`
(`uv run --group docs mkdocs build --strict`).

## Memory: one heavy job at a time
The machine has 48 GB and a nearly full disk; it has frozen in swap.
- `pytest -n 4` at most (the suite holds a lock in the system temp folder, so a second
  run waits). Subagents: at most 2 processes, pools of 2 workers, no installer builds.
- `FORSTER_MAX_CURVES_GB` (default 4) stops a fit that would allocate more; do not raise
  it to get past a grid problem.

## Round robin privacy
The round robin profiles (SH63olv066, KS20-527, ...) come from other labs.
- Raw profiles and plots of them stay in the private submodule `Diffusion Round Robin/`
  (commit and push there, then commit the pointer in the main repo).
- Results derived from them (times, ranges, noise numbers, per-profile rows, figures) never
  go in public files either (Henry, 2026-10-17; old commits may keep them).
  `tests/test_public_tree.py` fails on any tracked file outside the submodule that names a
  round robin sample, unless it is on its allow-list of code that only reads the private
  data. Scripts write round robin rows to `Diffusion Round Robin/outputs/public_tree_moved/`
  and keep only paper rows next to themselves; a table whose numbers would reveal a round
  robin time is made relative (e.g. times divided by the true time).
- Never commit round robin data as test fixtures; tests skip when the submodule is absent.

## Commits
Other Claude sessions share this repository: commit with explicit paths
(`git commit -- <paths>`), never the bare index; leave files you did not change alone.
Pushing is outward-facing: ask first.
