from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Round robin tables (public in this repo; samples from other labs, so not used as examples).
ROUND_ROBIN_PARAMS = (
    REPO_ROOT
    / "Diffusion Round Robin"
    / "Round_Robin_Olivine_Profile_Diffusion_Modeling_Parameters_DB.xlsx"
)
ROUND_ROBIN_DATA = REPO_ROOT / "Diffusion Round Robin" / "Diffusion Round Robin Data Reorg.xlsx"

# Tables from Towbin and Plank (2026), in the paper-era layout.
PAPER_PARAMS = (
    REPO_ROOT / "old Xenolith Code" / "Olivine_Profile_Diffusion_Modeling_Parameters_DB.xlsx"
)
PAPER_DATA = REPO_ROOT / "old Xenolith Code" / "Xenolith_MeltVein_EMPA_Master.xlsx"
