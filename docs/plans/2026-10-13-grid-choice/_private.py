"""Write a results table: every row to the round robin submodule, the paper rows here.

The round robin profiles are other laboratories' data, and results derived from them stay
private (Henry, 2026-10-17). Rows are told apart by the run name.
"""

import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
PRIVATE = REPO / "Diffusion Round Robin" / "outputs" / "public_tree_moved" / HERE.relative_to(REPO)
ROUND_ROBIN_RUN = re.compile(r"SH63|KS20")


def write_rows(rows, name):
    table = pd.DataFrame(rows)
    PRIVATE.mkdir(parents=True, exist_ok=True)
    table.to_csv(PRIVATE / name, index=False)
    public = table[~table["run"].astype(str).str.contains(ROUND_ROBIN_RUN)]
    public.to_csv(HERE / name, index=False)
