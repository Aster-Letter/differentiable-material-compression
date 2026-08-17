from __future__ import annotations

import pytest
import torch

from cg_frontier.compression.basecolor_dynamic_constraint import (
    DynamicBaseColorConstraintConfig,
    compose_dynamic_basecolor_constraint,
    update_dynamic_multipliers,
)


def _config() -> DynamicBaseColorConstraintConfig:
    return DynamicBaseColorConstraintConfig(
        rgb_rmse_ceiling=0.012,
        chroma_retention_floor=0.2,
        penalty_rho=0.25,
        dual_learning_rate=0.05,
        multiplier_max=100.0,
    )


def test_dynamic_constraint_is_inactive_for_exact_rgb() -> None:
    target = torch.tensor(
        [[0.1, 0.2, 0.3], [0.8, 0.2, 0.1], [0.3, 0.7, 0.2]],
        dtype=torch.float64,
    )
    residual = torch.tensor(0.7, dtype=torch.float64)
    result = compose_dynamic_basecolor_constraint(
        target,
        target,
        residual,
        torch.tensor((3.0, 4.0), dtype=torch.float64),
        _config(),
    )

    assert result.rgb_violation < 0.0
    assert result.chroma_violation < 0.0
    assert result.rgb_penalty == 0.0
    assert result.chroma_penalty == 0.0
    assert result.total == residual


def test_dynamic_constraint_penalizes_grey_chroma_collapse() -> None:
    target = torch.tensor(
        [[0.8, 0.2, 0.1], [0.1, 0.7, 0.2], [0.2, 0.1, 0.9]],
        dtype=torch.float64,
    )
    prediction = target.mean(dim=-1, keepdim=True).expand_as(target).clone()
    result = compose_dynamic_basecolor_constraint(
        prediction,
        target,
        torch.tensor(0.1, dtype=torch.float64),
        torch.tensor((1.0, 1.0), dtype=torch.float64),
        _config(),
    )

    assert result.rgb_violation > 0.0
    assert result.chroma_retention == pytest.approx(0.0, abs=1.0e-12)
    assert result.chroma_violation == pytest.approx(1.0)
    assert result.total > result.residual


def test_dynamic_multiplier_update_projects_and_relaxes() -> None:
    multipliers = torch.tensor((0.0, 1.0), dtype=torch.float64)
    update_dynamic_multipliers(
        multipliers,
        torch.tensor(2.0, dtype=torch.float64),
        torch.tensor(-30.0, dtype=torch.float64),
        _config(),
    )

    assert multipliers.tolist() == pytest.approx([0.1, 0.0])


def test_dynamic_constraint_rejects_invalid_floor() -> None:
    with pytest.raises(ValueError, match="must not exceed one"):
        DynamicBaseColorConstraintConfig(
            rgb_rmse_ceiling=0.012,
            chroma_retention_floor=1.1,
            penalty_rho=0.25,
            dual_learning_rate=0.05,
            multiplier_max=100.0,
        )
