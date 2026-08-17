"""Create an isolated UE map for the chroma8-to-L0 color-collapse trajectory."""

from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[4]
BASE_PATH = SCRIPT_PATH.with_name("setup_scifihelmet_affine_preview.py")

spec = importlib.util.spec_from_file_location("_affine_preview_base", BASE_PATH)
if spec is None or spec.loader is None:
    raise RuntimeError("could not load shared affine UE setup implementation")
base = importlib.util.module_from_spec(spec)
base.AFFINE_PREVIEW_AUTORUN = False
spec.loader.exec_module(base)

DEPLOYMENT_ROOT = (
    REPO_ROOT
    / "outputs/scifihelmet_c4_affine_v1/ue_preview/"
    "a874ad-chroma8-l0-camera31-trajectory-r1"
)
EXPECTED_PREVIEW_MANIFEST_SHA256 = (
    "45b0866ef525082e7f2bba1bcaef82d46b26f8767f5ee7b8a6bd4693eae01d63"
)
EXPECTED_CANDIDATES = (
    "p0_chroma8_parent",
    "l0_s001k",
    "l0_s005k",
    "l0_s010k",
    "l0_s080k",
)

ASSET_ROOT = "/Game/CGCompression/AffineTrajectory"
SOURCE_MAP = "/Game/CGCompression/Maps/MaterialLab"
PREVIEW_MAP = f"{ASSET_ROOT}/Maps/MaterialLab_Affine_Chroma8_Trajectory"
MASTER_MATERIAL = (
    f"{ASSET_ROOT}/Materials/M_SciFiHelmet_Affine_Trajectory_Master"
)

CANDIDATE_LABELS = {
    "p0_chroma8_parent": "Helmet_Affine_P0_CHROMA8_PARENT",
    "l0_s001k": "Helmet_Affine_L0_S001K",
    "l0_s005k": "Helmet_Affine_L0_S005K",
    "l0_s010k": "Helmet_Affine_L0_S010K",
    "l0_s080k": "Helmet_Affine_L0_S080K_CAMERA31",
}

base.DEPLOYMENT_ROOT = DEPLOYMENT_ROOT
base.PREVIEW_MANIFEST_PATH = DEPLOYMENT_ROOT / "trajectory_manifest.json"
base.REPORT_PATH = DEPLOYMENT_ROOT / "ue_setup_report.json"
base.EVIDENCE_ROOT = DEPLOYMENT_ROOT / "ue_evidence"
base.EXPECTED_PREVIEW_MANIFEST_SHA256 = EXPECTED_PREVIEW_MANIFEST_SHA256
base.EXPECTED_CANDIDATES = EXPECTED_CANDIDATES
base.ASSET_ROOT = ASSET_ROOT
base.SOURCE_MAP = SOURCE_MAP
base.PREVIEW_MAP = PREVIEW_MAP
base.MASTER_MATERIAL = MASTER_MATERIAL
base.CANDIDATE_LABELS = CANDIDATE_LABELS

base.setup()
