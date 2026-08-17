from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_scifihelmet_c4_affine_l0_80k import (
    freeze_continuation_decision,
    validate_completed_continuation_report,
    validate_source_40k_report,
)


def _config() -> dict[str, object]:
    return {
        "authorization": {
            "authorized_l0_80k": True,
            "candidate_id": "L0",
            "retain_p0_raw_gap_report": True,
        },
        "source": {
            "common_40k_report_sha256": "report-hash",
            "checkpoint_hash": "checkpoint-hash",
            "parent_p0_hash": "parent-hash",
        },
        "training": {
            "source_step": 40_000,
            "total_steps": 80_000,
            "checkpoint_steps": [45_000, 50_000, 55_000, 60_000, 65_000, 70_000, 75_000, 80_000],
            "trend_interval": 1_000,
        },
    }


def test_l0_80k_decision_freezes_source_lineage_and_dense_evidence() -> None:
    decision = freeze_continuation_decision(_config())

    assert decision.candidate_id == "L0"
    assert decision.source_step == 40_000
    assert decision.observation_plan.total_steps == 80_000
    assert decision.observation_plan.checkpoint_steps == (
        45_000, 50_000, 55_000, 60_000, 65_000, 70_000, 75_000, 80_000
    )
    assert decision.observation_plan.trend_steps[-1] == 80_000
    assert decision.source_report_hash == "report-hash"
    assert decision.source_checkpoint_hash == "checkpoint-hash"
    assert decision.parent_p0_hash == "parent-hash"


def test_l0_80k_decision_rejects_non_l0_or_wrong_endpoint() -> None:
    for section, key, value in (
        ("authorization", "candidate_id", "L1"),
        ("training", "source_step", 39_999),
        ("training", "total_steps", 120_000),
    ):
        config = _config()
        config[section][key] = value
        with pytest.raises(ValueError):
            freeze_continuation_decision(config)


def test_source_report_requires_exact_common_l0_40k_lineage_and_raw_gap() -> None:
    decision = freeze_continuation_decision(_config())
    report = {
        "experiment": "scifihelmet_c4_affine_v1_40k",
        "common_endpoint": 40_000,
        "p0_raw_gap_report": {"raw": {}, "safe": {}, "increment": {}},
        "candidates": {
            "L0": {
                "manifest": {
                    "candidate_id": "L0",
                    "objective_id": "material+helmet",
                    "optimizer_updates": 40_000,
                    "parent_p0_hash": "parent-hash",
                },
                "endpoint": {"checkpoint_hash": "checkpoint-hash"},
            }
        },
    }

    assert validate_source_40k_report(report, decision) is report

    report["candidates"]["L0"]["manifest"]["objective_id"] = "material+helmet+tv"
    with pytest.raises(ValueError, match="lineage"):
        validate_source_40k_report(report, decision)


def test_completed_l0_80k_report_is_bound_to_source_checkpoint() -> None:
    decision = freeze_continuation_decision(_config())
    report = {
        "experiment": "scifihelmet_c4_affine_v1_l0_80k_continuation",
        "endpoint": 80_000,
        "source_common_40k_report_sha256": "report-hash",
        "p0_raw_gap_report": {"raw": {}, "safe": {}, "increment": {}},
        "candidate": {
            "manifest": {
                "candidate_id": "L0",
                "objective_id": "material+helmet",
                "optimizer_updates": 80_000,
                "parent_p0_hash": "parent-hash",
                "continuation": {
                    "source_checkpoint_hash": "checkpoint-hash",
                    "source_step": 40_000,
                },
            }
        },
    }

    assert validate_completed_continuation_report(report, decision) is report

    report["candidate"]["manifest"]["continuation"]["source_step"] = 39_999
    with pytest.raises(ValueError, match="lineage"):
        validate_completed_continuation_report(report, decision)
