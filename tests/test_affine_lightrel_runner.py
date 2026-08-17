from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_scifihelmet_c4_affine_chroma8_l0_lightrel import (
    certified_parent_certificate,
    evaluate_lightrel_training_gate,
    freeze_lightrel_40k_continuation,
)


def test_1k_gate_requires_color_retention_and_bounded_error_before_5k_resume() -> None:
    parent = {
        "chroma_contrast_retention": 0.16,
        "yellow_mean_r_minus_b": 0.08,
        "seven_channel_mae": 0.07,
        "masked_linear_hdr_mae": 0.02,
    }
    step_1k = {
        "chroma_contrast_retention": 0.15,
        "yellow_mean_r_minus_b": 0.07,
        "seven_channel_mae": 0.08,
        "masked_linear_hdr_mae": 0.03,
        "certificate_valid": True,
    }
    thresholds = {
        "generic_chroma_retention_fraction_of_parent_min": 0.90,
        "yellow_r_minus_b_fraction_of_parent_min": 0.80,
        "seven_channel_mae_multiplier_max": 2.0,
        "masked_linear_hdr_mae_multiplier_max": 2.0,
        "certificate_valid_required": True,
    }

    passed = evaluate_lightrel_training_gate(parent, step_1k, thresholds)
    assert passed["passed"] is True

    step_1k["yellow_mean_r_minus_b"] = 0.01
    failed = evaluate_lightrel_training_gate(parent, step_1k, thresholds)
    assert failed["passed"] is False
    assert failed["gates"]["yellow_r_minus_b"] is False


def test_parent_certificate_is_read_from_the_hashed_manifest() -> None:
    certificate = {"valid": True, "certificate_margin": 0.001}

    assert certified_parent_certificate(
        {"safe_artifact": {"certificate": certificate}}
    ) == certificate


def test_lightrel_40k_continuation_freezes_dense_observation_schedule() -> None:
    decision = freeze_lightrel_40k_continuation(
        {
            "authorization": {
                "authorized_lightrel_l0_40k": True,
                "candidate_id": "L0",
                "retain_p0_raw_gap_report": True,
            },
            "source": {
                "parent_p0_hash": "parent",
                "source_checkpoint_hash": "checkpoint",
                "source_step": 1000,
            },
            "training": {
                "total_steps": 40000,
                "checkpoint_steps": list(range(5000, 40001, 5000)),
                "trend_interval": 1000,
            },
        }
    )

    assert decision.source_step == 1000
    assert decision.source_checkpoint_hash == "checkpoint"
    assert decision.observation_plan.checkpoint_steps == tuple(
        range(5000, 40001, 5000)
    )
    assert decision.observation_plan.trend_steps == tuple(range(2000, 40001, 1000))
