from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_scifihelmet_c4_affine_40k import (
    endpoint_artifact_directory,
    freeze_training_decision,
    validate_completed_report,
)


def test_endpoint_artifact_directory_is_qualified_by_actual_optimizer_step() -> None:
    class Candidate:
        candidate_id = "L0"
        optimizer_updates = 80_000

    assert endpoint_artifact_directory(Path("isolated"), Candidate()) == (
        Path("isolated/candidates/L0/step-080000")
    )


def test_formal_decision_freezes_selected_ratios_lineage_and_dense_evidence() -> None:
    preflight = {
        "p0": {
            "raw": {"artifact_hash": "raw-hash"},
            "safe": {"artifact_hash": "safe-hash"},
            "increment": {"material_mae": 0.14, "masked_linear_hdr_mae": 0.058},
        },
        "calibration": {
            "tv": {"lambdas": {"0.05": 1.9851209640502931}},
            "cube": {"lambdas": {"0.05": 0.10291787385940553}},
        },
    }
    config = {
        "authorization": {
            "authorized_40k": True,
            "retain_p0_raw_gap_report": True,
            "candidate_order": ["L0", "L1", "L2"],
        },
        "selection": {"tv_ratio": 0.05, "cube_ratio": 0.05},
        "training": {
            "total_steps": 40_000,
            "checkpoint_steps": [1_000, 5_000, 10_000, 20_000, 30_000, 35_000, 40_000],
            "trend_interval": 1_000,
        },
    }

    decision = freeze_training_decision(config, preflight)

    assert decision.parent_p0_hash == "safe-hash"
    assert decision.raw_p0_hash == "raw-hash"
    assert decision.tv_ratio == decision.cube_ratio == 0.05
    assert decision.tv_lambda == 1.9851209640502931
    assert decision.cube_lambda == 0.10291787385940553
    assert decision.candidate_order == ("L0", "L1", "L2")
    assert decision.observation_plan.checkpoint_steps[-1] == 40_000
    assert len(decision.observation_plan.trend_steps) == 40
    assert decision.raw_gap_report == preflight["p0"]


def test_completed_report_is_a_read_only_idempotent_resume_endpoint() -> None:
    decision = freeze_training_decision(
        {
            "authorization": {
                "authorized_40k": True,
                "retain_p0_raw_gap_report": True,
                "candidate_order": ["L0", "L1", "L2"],
            },
            "selection": {"tv_ratio": 0.05, "cube_ratio": 0.05},
            "training": {
                "total_steps": 40_000,
                "checkpoint_steps": [1_000, 5_000, 10_000, 20_000, 30_000, 35_000, 40_000],
                "trend_interval": 1_000,
            },
        },
        {
            "p0": {
                "raw": {"artifact_hash": "raw-hash"},
                "safe": {"artifact_hash": "safe-hash"},
                "increment": {},
            },
            "calibration": {
                "tv": {"lambdas": {"0.05": 1.0}},
                "cube": {"lambdas": {"0.05": 2.0}},
            },
        },
    )
    report = {
        "experiment": "scifihelmet_c4_affine_v1_40k",
        "common_endpoint": 40_000,
        "source_preflight_report_sha256": "report-hash",
        "candidates": {
            name: {
                "manifest": {
                    "candidate_id": name,
                    "parent_p0_hash": "safe-hash",
                    "optimizer_updates": 40_000,
                }
            }
            for name in ("L0", "L1", "L2")
        },
    }

    assert validate_completed_report(
        report, decision, expected_preflight_hash="report-hash"
    ) is report
