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
Run them once over the finished change. A new summary or parameter column needs its
`forster.template` entry, and the README column tables must be regenerated (replace the
old `column_table(...)` text; `tests/test_template.py` checks it).

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
- Derived numbers may sit in public docs until the release; every public file holding
  them is listed in CHANGELOG "Before release" and must move first. Add new ones there.
- Never commit round robin data as test fixtures; tests skip when the submodule is absent.

## Commits
Other Claude sessions share this repository: commit with explicit paths
(`git commit -- <paths>`), never the bare index; leave files you did not change alone.
Pushing is outward-facing: ask first.
