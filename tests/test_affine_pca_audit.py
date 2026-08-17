from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cg_frontier.compression.affine_pca_audit import (
    cluster_balanced_candidate_specs,
    enhanced_candidate_specs,
    material_region_metrics,
    residual_reweighted_candidate_specs,
)


def test_material_region_metrics_reports_generic_chroma_contrast_retention() -> None:
    target = torch.tensor(
        [
            [0.10, 0.10, 0.10, 0.0, 0.0, 0.4, 0.0],
            [0.12, 0.11, 0.10, 0.0, 0.0, 0.5, 0.0],
            [0.80, 0.45, 0.10, 0.0, 0.0, 0.6, 0.0],
            [0.10, 0.55, 0.85, 0.0, 0.0, 0.7, 0.0],
        ],
        dtype=torch.float64,
    )
    prediction = target.clone()
    prediction[2:, :3] = prediction[2:, :3].mean(dim=-1, keepdim=True)

    metrics = material_region_metrics(prediction, target, tail_quantile=0.5)

    assert metrics["tail_definition"] == "source_linear_rgb_chroma_quantile"
    assert metrics["tail_texels"] == 2
    assert metrics["source_chroma_contrast"] > 1.0
    assert metrics["predicted_chroma_contrast"] == pytest.approx(0.0, abs=1.0e-12)
    assert metrics["chroma_contrast_retention"] == pytest.approx(0.0, abs=1.0e-12)
    assert metrics["seven_mae"] > 0.0


def test_enhanced_candidate_matrix_is_small_frozen_and_generic() -> None:
    specs = enhanced_candidate_specs()

    assert list(specs) == ["balanced", "opponent2", "chroma4", "chroma8"]
    assert [spec.chroma_tail_strength for spec in specs.values()] == [0.0, 0.0, 3.0, 7.0]
    assert [spec.opponent_chroma_weight for spec in specs.values()] == [1.0, 2.0, 2.0, 2.0]
    assert all(spec.semantic_group_balance for spec in specs.values())


def test_cluster_balanced_candidate_matrix_only_varies_balance_power() -> None:
    specs = cluster_balanced_candidate_specs()

    assert list(specs) == ["chroma8_k4_b05", "chroma8_k4_b10"]
    assert [spec.material_cluster_count for spec in specs.values()] == [4, 4]
    assert [spec.material_cluster_balance_power for spec in specs.values()] == [0.5, 1.0]
    assert {spec.material_cluster_seed for spec in specs.values()} == {20260807}
    assert {spec.chroma_tail_strength for spec in specs.values()} == {7.0}
    assert {spec.opponent_chroma_weight for spec in specs.values()} == {2.0}
    assert all(spec.semantic_group_balance for spec in specs.values())


def test_residual_reweighted_candidate_matrix_only_varies_tail_strength() -> None:
    specs = residual_reweighted_candidate_specs()

    assert list(specs) == ["chroma8_resid3", "chroma8_resid7"]
    assert [spec.residual_tail_strength for spec in specs.values()] == [3.0, 7.0]
    assert {spec.residual_reweight_iterations for spec in specs.values()} == {2}
    assert {spec.chroma_tail_strength for spec in specs.values()} == {7.0}
    assert {spec.opponent_chroma_weight for spec in specs.values()} == {2.0}
    assert all(spec.semantic_group_balance for spec in specs.values())
    assert {spec.material_cluster_count for spec in specs.values()} == {0}
