import fcntl
import os
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Round robin tables: samples from other labs, kept in a private repository checked out as the
# git submodule "Diffusion Round Robin" (README, Development). Tests that need them skip
# when it is not checked out (a public clone).
ROUND_ROBIN_PARAMS = (
    REPO_ROOT
    / "Diffusion Round Robin"
    / "Round_Robin_Olivine_Profile_Diffusion_Modeling_Parameters_DB.xlsx"
)
ROUND_ROBIN_DATA = REPO_ROOT / "Diffusion Round Robin" / "Diffusion Round Robin Data Reorg.xlsx"

# Tables from Towbin and Plank (2026), in the paper-era layout.
PAPER_ARCHIVE = REPO_ROOT / "analyses" / "xenolith_archive"
PAPER_PARAMS = PAPER_ARCHIVE / "Olivine_Profile_Diffusion_Modeling_Parameters_DB.xlsx"
PAPER_DATA = PAPER_ARCHIVE / "Xenolith_MeltVein_EMPA_Master.xlsx"

HAS_ROUND_ROBIN = ROUND_ROBIN_PARAMS.exists() and ROUND_ROBIN_DATA.exists()
ROUND_ROBIN_MISSING = (
    "round robin data not checked out (private submodule 'Diffusion Round Robin'; "
    "git submodule update --init, with access)"
)
requires_round_robin = pytest.mark.skipif(not HAS_ROUND_ROBIN, reason=ROUND_ROBIN_MISSING)


def require_round_robin() -> None:
    """Skip the calling fixture (and the tests using it) without the round robin data;
    marks on fixtures have no effect, so fixtures call this."""
    if not HAS_ROUND_ROBIN:
        pytest.skip(ROUND_ROBIN_MISSING)


# One FORSTER test run at a time on this machine: a full run with 8 workers peaks near
# 18 GB, and runs started at once by several sessions or worktrees pushed the machine
# into swap and froze it (2026-10-15). The second run waits here for the first.
_LOCK = Path(tempfile.gettempdir()) / "forster-pytest.lock"
_lock_file = None


def pytest_configure(config):
    global _lock_file
    if hasattr(config, "workerinput") or os.environ.get("FORSTER_TEST_NO_LOCK") == "1":
        return  # xdist workers run under their controller's lock
    _lock_file = _LOCK.open("w")
    try:
        fcntl.flock(_lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"waiting for another FORSTER test run to finish ({_LOCK})", file=sys.stderr)
        fcntl.flock(_lock_file, fcntl.LOCK_EX)
