from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_scifihelmet_c4_affine_chroma8_l0_40k import (
    freeze_chroma8_l0_decision,
    freeze_lightrel_decision,
    freeze_render_coverage,
    select_render_pair,
    validate_completed_chroma8_l0_report,
)


def _config() -> dict[str, object]:
    return {
        "authorization": {
            "authorized_chroma8_l0_80k": True,
            "candidate_id": "L0",
            "retain_p0_raw_gap_report": True,
        },
        "source": {
            "pca_audit_report_sha256": "audit-hash",
            "parent_manifest_sha256": "manifest-hash",
            "parent_p0_hash": "safe-hash",
            "render_pool_config_sha256": "render-pool-hash",
        },
        "training": {
            "total_steps": 80_000,
            "checkpoint_steps": [1_000, *range(5_000, 80_001, 5_000)],
            "trend_interval": 1_000,
        },
    }


def _parent_manifest() -> dict[str, object]:
    return {
        "pipeline_id": "scifihelmet_c4_affine_pca_enhanced_v1",
        "pca": {
            "rank": 4,
            "chroma_tail_strength": 7.0,
            "opponent_chroma_weight": 2.0,
            "semantic_group_balance": True,
            "material_cluster_count": 0,
            "residual_reweight_iterations": 0,
        },
        "safe_artifact": {
            "artifact_hash": "safe-hash",
            "certificate": {"valid": True},
        },
    }


def _camera31_pool() -> dict[str, object]:
    return {
        "training_pool": {
            "camera_limit": 48,
            "light_limit": 6,
            "camera_selection_strategy": "explicit_audited_pool_v1",
            "randomize_lights_online": True,
        },
        "render": {"resolution": [256, 256]},
        "camera_finetune": {"audit": {"selected_camera_count": 31}},
        "train_cameras": [
            {"name": f"camera-{index}", "yaw_degrees": float(index), "elevation_degrees": 0.0}
            for index in range(31)
        ],
        "train_lights": [
            {"name": f"light-{index}", "position": [0.0, 1.0, float(index + 1)]}
            for index in range(6)
        ],
    }


def test_chroma8_l0_decision_freezes_parent_and_l0_only_80k() -> None:
    decision = freeze_chroma8_l0_decision(_config(), _parent_manifest())

    assert decision.candidate_id == "L0"
    assert decision.parent_p0_hash == "safe-hash"
    assert decision.parent_manifest_hash == "manifest-hash"
    assert decision.source_audit_hash == "audit-hash"
    assert decision.render_pool_hash == "render-pool-hash"
    assert decision.observation_plan.total_steps == 80_000
    assert decision.observation_plan.checkpoint_steps == (
        1_000,
        *range(5_000, 80_001, 5_000),
    )
    assert len(decision.observation_plan.trend_steps) == 80


def test_chroma8_l0_decision_rejects_other_parent_family_or_candidate() -> None:
    config = _config()
    config["authorization"]["candidate_id"] = "L1"
    with pytest.raises(ValueError):
        freeze_chroma8_l0_decision(config, _parent_manifest())

    manifest = _parent_manifest()
    manifest["pca"]["chroma_tail_strength"] = 3.0
    with pytest.raises(ValueError):
        freeze_chroma8_l0_decision(_config(), manifest)


def test_completed_chroma8_l0_report_is_bound_to_parent_and_80k() -> None:
    decision = freeze_chroma8_l0_decision(_config(), _parent_manifest())
    report = {
        "experiment": "scifihelmet_c4_affine_chroma8_l0_camera31_light6_80k",
        "endpoint": 80_000,
        "source_pca_audit_report_sha256": "audit-hash",
        "source_parent_manifest_sha256": "manifest-hash",
        "source_render_pool_config_sha256": "render-pool-hash",
        "p0_raw_gap_report": {"raw": {}, "safe": {}},
        "candidate": {
            "manifest": {
                "candidate_id": "L0",
                "objective_id": "material+helmet",
                "parent_p0_hash": "safe-hash",
                "optimizer_updates": 80_000,
            }
        },
    }

    assert validate_completed_chroma8_l0_report(report, decision) is report

    report["candidate"]["manifest"]["parent_p0_hash"] = "old-parent"
    with pytest.raises(ValueError, match="lineage"):
        validate_completed_chroma8_l0_report(report, decision)


def test_chroma8_l0_freezes_camera31_light6_coverage() -> None:
    coverage = freeze_render_coverage(_camera31_pool())

    assert coverage.resolution == (256, 256)
    assert coverage.camera_count == 31
    assert coverage.light_count == 6
    assert coverage.randomize_lights_online is True
    assert coverage.camera_selection_strategy == "explicit_audited_pool_v1"


def test_render_pair_uses_two_independent_resumable_batch_draws() -> None:
    first = select_render_pair([62, 17], camera_count=31, light_count=6)
    resumed = select_render_pair([62, 17], camera_count=31, light_count=6)

    assert first == (0, 5)
    assert resumed == first

    # Changing only the second draw must not change the selected camera.
    assert select_render_pair([62, 18], camera_count=31, light_count=6) == (0, 0)


def test_lightrel_decision_stops_at_1k_and_5k_before_any_long_run() -> None:
    config = _config()
    config["authorization"] = {
        "authorized_lightrel_diagnostic": True,
        "candidate_id": "L0",
        "retain_p0_raw_gap_report": True,
        "long_training_authorized": False,
    }
    config["training"] = {
        "total_steps": 5_000,
        "checkpoint_steps": [1_000, 5_000],
        "trend_interval": 1_000,
    }

    decision = freeze_lightrel_decision(config, _parent_manifest())

    assert decision.candidate_id == "L0"
    assert decision.parent_p0_hash == "safe-hash"
    assert decision.observation_plan.total_steps == 5_000
    assert decision.observation_plan.checkpoint_steps == (1_000, 5_000)
