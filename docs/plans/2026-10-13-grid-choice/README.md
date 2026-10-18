# Grid choice: calibration scripts and data (2026-10-13)

Behind `docs/references/grid-choice.md` and `src/forster/grid.py`. Run each
script from the repository root with `uv run python docs/plans/2026-10-13-grid-choice/<script>`.

The scripts fit round robin profiles as well as paper profiles when the private
round robin submodule is checked out. The round robin data belong to other labs,
so the CSVs here hold only the paper rows; the full tables, with the round robin
rows, are in the submodule under
`Diffusion Round Robin/outputs/public_tree_moved/docs/plans/2026-10-13-grid-choice/`.
`old_tables.py` and `trbdf2_check.py` read their reference times from there when
it is present. Rerunning a script with the submodule present writes the round
robin rows back into the CSV here; remove them before committing.

- `calib_grid.py` → `calib_grid.csv`: fitted time against l / dx = k
  (1 to 30) on six real profiles, dt at D_max dt / dx² = 0.4.
- `calib_grid2.py` → `calib_grid2.csv`: k = 12 to 60, with the first step
  left where the grid snaps it (`aligned = false`) or placed on a node
  (`aligned = true`); the k = 60 runs are the reference.
- `gridbench.py`: cost per explicit step against grid size, and a
  backward-Euler (implicit) prototype.
- `compiled.py` (`uv run --with numba`): the explicit step as a Numba loop,
  checked against `engine.py` to 1e-16.
