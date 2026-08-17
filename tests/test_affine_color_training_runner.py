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

from run_scifihelmet_c4_affine_color_guard_1k import (
    cli_summary,
    evaluate_balanced_color_gate,
    freeze_candidate_matrix,
    freeze_color_guard_training_spec,
)


def _training_config() -> dict[str, object]:
    return {
        "training": {
            "ratios": [0.10, 0.25],
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
        }
    }


def test_training_spec_freezes_shared_c0_and_two_ratio_matrix() -> None:
    spec = freeze_color_guard_training_spec(_training_config())

    assert spec.ratios == (0.10, 0.25)
    assert spec.candidate_keys == (
        "C0",
        "C1-r010",
        "C2-r010",
        "C1-r025",
        "C2-r025",
    )
    assert spec.checkpoint_steps == (250, 500, 750, 1000)
    assert spec.trajectory_gradient_batches == 5

    invalid = _training_config()
    invalid["training"]["ratios"] = [0.25, 0.10]
    with pytest.raises(ValueError, match="ratios"):
        freeze_color_guard_training_spec(invalid)


def test_candidate_matrix_uses_audited_lambdas_and_one_shared_c0() -> None:
    audit = {
        "status": "complete_waiting_for_ratio_confirmation",
        "training_started": False,
        "selected_ratio": None,
        "calibration": {
            "selected_ratio": None,
            "candidates": {
                "0.100000": {
                    "C1": {"opponent_lambda": 0.8, "pair_lambda": 0.0},
                    "C2": {"opponent_lambda": 0.4, "pair_lambda": 0.2},
                },
                "0.250000": {
                    "C1": {"opponent_lambda": 2.0, "pair_lambda": 0.0},
                    "C2": {"opponent_lambda": 1.0, "pair_lambda": 0.5},
                },
            },
        },
    }

    matrix = freeze_candidate_matrix(audit, (0.10, 0.25))

    assert tuple(matrix) == ("C0", "C1-r010", "C2-r010", "C1-r025", "C2-r025")
    assert matrix["C0"]["ratio"] is None
    assert matrix["C1-r010"]["opponent_lambda"] == 0.8
    assert matrix["C2-r025"]["pair_lambda"] == 0.5

    audit["training_started"] = True
    with pytest.raises(ValueError, match="zero-update"):
        freeze_candidate_matrix(audit, (0.10, 0.25))


def _endpoint(
    *,
    chroma: float = 0.20,
    color: float = 1.0,
    base: float = 1.0,
    seven: float = 1.0,
    hdr: float = 1.0,
    ssim: float = 0.95,
    normal: float = 1.0,
    roughness: float = 1.0,
    metallic: float = 1.0,
    certificate: bool = True,
) -> dict[str, object]:
    return {
        "color": {
            "chroma_contrast_retention": chroma,
            "uniform_opponent_error": color,
            "macro_bin_opponent_error": color,
            "worst_bin_opponent_error": color,
            "fixed_pair_opponent_error": color,
            "uniform_base_color_l1": base,
        },
        "material": {
            "seven_channel_mae": seven,
            "normal_cosine": normal,
            "roughness_l1": roughness,
            "metallic_l1": metallic,
        },
        "render": {
            "masked_linear_hdr_mae": hdr,
            "display_ssim": ssim,
        },
        "certificate": {"valid": certificate},
        "yellow_diagnostic": {"selection_metric": False},
    }


def test_balanced_gate_requires_color_gain_without_global_regression() -> None:
    policy = {
        "parent_chroma_retention_floor": 0.90,
        "global_error_multiplier": 1.05,
        "color_improvement_vs_c0": 0.05,
        "color_regression_vs_c0": 0.02,
        "display_ssim_absolute_drop": 0.005,
        "primary_color_metrics": [
            "uniform_opponent_error",
            "macro_bin_opponent_error",
            "worst_bin_opponent_error",
            "fixed_pair_opponent_error",
        ],
    }
    parent = _endpoint(chroma=0.20)
    c0 = _endpoint(chroma=0.15)
    candidate = _endpoint(chroma=0.19, color=0.94)

    result = evaluate_balanced_color_gate(parent, c0, candidate, policy)

    assert result["passed"] is True
    assert result["color_improvement_count"] == 4
    assert "yellow_diagnostic" not in result["selection_values"]

    failed = evaluate_balanced_color_gate(
        parent,
        c0,
        _endpoint(chroma=0.19, color=0.94, roughness=1.06),
        policy,
    )
    assert failed["passed"] is False
    assert failed["gates"]["roughness_l1"] is False


def test_cli_summary_does_not_echo_the_full_training_report() -> None:
    report = {
        "status": "completed_1k_stop_before_5k_or_ue",
        "passing_candidates": [],
        "preferred_candidate": None,
        "wall_seconds": 544.0,
        "candidates": {"C0": {"curve": [0] * 1000}},
        "gates": {"C1-r010": {"passed": False}},
    }

    summary = cli_summary(report)

    assert summary == {
        "status": "completed_1k_stop_before_5k_or_ue",
        "passing_candidates": [],
        "preferred_candidate": None,
        "wall_seconds": 544.0,
    }
    assert "candidates" not in summary
