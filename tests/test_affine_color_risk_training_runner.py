from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
for path in (SCRIPTS, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from run_scifihelmet_c4_affine_color_risk_1k import (
    evaluate_color_risk_gate,
    freeze_color_risk_training_spec,
    freeze_tail_hue_gate_policy,
    validate_g0_mean_scale_replay,
)


def _config() -> dict[str, object]:
    return {
        "training": {
            "total_ratio": 0.10,
            "tail_mass": 0.25,
            "preflight_steps": 10,
            "total_steps": 1000,
            "checkpoint_steps": [250, 500, 750, 1000],
            "metric_interval": 250,
            "progress_interval": 100,
            "trajectory_gradient_batches": 5,
            "material_batch_size": 4096,
            "color_batch_size": 4096,
            "color_seed_offset": 23,
            "charbonnier_epsilon": 0.001,
            "hue_group_count": 8,
            "hue_min_group_size": 4096,
        }
    }


def test_risk_training_spec_freezes_bounded_four_candidate_matrix() -> None:
    spec = freeze_color_risk_training_spec(_config())

    assert spec.candidate_ids == (
        "G0-mean",
        "G1-yc-cvar25",
        "G2-hue8-macro",
        "G3-cvar25-hue8",
    )
    assert spec.total_ratio == 0.10
    assert spec.tail_mass == 0.25
    assert spec.total_steps == 1000

    invalid = _config()
    invalid["training"]["tail_mass"] = 0.50
    with pytest.raises(ValueError, match="tail mass"):
        freeze_color_risk_training_spec(invalid)


def test_g0_mean_scale_uses_raw_replay_and_rounded_frozen_report_value() -> None:
    raw = 8.492847018062633
    validate_g0_mean_scale_replay(raw, raw)
    with pytest.raises(RuntimeError, match="replay mismatch"):
        validate_g0_mean_scale_replay(raw + 1.0e-12, raw)


def _endpoint(*, original: float, new: float, yellow_selection: bool = False):
    return {
        "color": {
            "uniform_base_color_l1": 1.0,
            "chroma_contrast_retention": 1.0,
            "uniform_opponent_error": original,
            "macro_bin_opponent_error": original,
            "worst_bin_opponent_error": original,
            "fixed_pair_opponent_error": original,
            "macro_bin_cvar25_opponent_error": new,
            "worst_bin_cvar25_opponent_error": new,
            "hue_macro_opponent_error": new,
            "worst_hue_group_opponent_error": new,
        },
        "material": {
            "seven_channel_mae": 1.0,
            "normal_cosine": 1.0,
            "roughness_l1": 1.0,
            "metallic_l1": 1.0,
        },
        "render": {"masked_linear_hdr_mae": 1.0, "display_ssim": 0.9},
        "certificate": {"valid": True},
        "yellow_diagnostic": {"selection_metric": yellow_selection},
    }


def test_risk_gate_requires_new_metric_gain_without_yellow_selection() -> None:
    policy = freeze_tail_hue_gate_policy()
    parent = _endpoint(original=1.0, new=1.0)
    c0 = _endpoint(original=1.0, new=1.0)
    candidate = _endpoint(original=0.94, new=0.94)

    result = evaluate_color_risk_gate(parent, c0, c0, candidate, policy)
    assert result["passed"] is True
    assert result["new_metric_improvement_count"] == 4
    assert policy["yellow_diagnostic"]["selection_metric"] is False

    regressed = _endpoint(original=0.94, new=1.03)
    assert evaluate_color_risk_gate(parent, c0, c0, regressed, policy)["passed"] is False
    with pytest.raises(ValueError, match="yellow"):
        evaluate_color_risk_gate(
            parent,
            c0,
            c0,
            _endpoint(original=0.94, new=0.94, yellow_selection=True),
            policy,
        )
