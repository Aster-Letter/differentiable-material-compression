"""Generic dynamic constraints for the C4 BaseColor-priority lineage."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch

from cg_frontier.compression.affine_color import orthogonal_color_coordinates


@dataclass(frozen=True)
class DynamicBaseColorConstraintConfig:
    rgb_rmse_ceiling: float
    chroma_retention_floor: float
    penalty_rho: float
    dual_learning_rate: float
    multiplier_max: float
    epsilon: float = 1.0e-12

    def __post_init__(self) -> None:
        values = (
            self.rgb_rmse_ceiling,
            self.chroma_retention_floor,
            self.penalty_rho,
            self.dual_learning_rate,
            self.multiplier_max,
            self.epsilon,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("dynamic BaseColor constraint values must be finite and positive")
        if self.chroma_retention_floor > 1.0:
            raise ValueError("chroma retention floor must not exceed one")


@dataclass(frozen=True)
class DynamicBaseColorConstraintTerms:
    total: torch.Tensor
    residual: torch.Tensor
    rgb_rmse: torch.Tensor
    chroma_retention: torch.Tensor
    rgb_violation: torch.Tensor
    chroma_violation: torch.Tensor
    rgb_penalty: torch.Tensor
    chroma_penalty: torch.Tensor


def compose_dynamic_basecolor_constraint(
    prediction_rgb: torch.Tensor,
    target_rgb: torch.Tensor,
    residual: torch.Tensor,
    multipliers: torch.Tensor,
    config: DynamicBaseColorConstraintConfig,
) -> DynamicBaseColorConstraintTerms:
    """Compose a residual objective with generic RGB/chroma inequality constraints."""

    if (
        prediction_rgb.shape != target_rgb.shape
        or prediction_rgb.ndim != 2
        or prediction_rgb.shape[1] != 3
        or not prediction_rgb.is_floating_point()
        or not target_rgb.is_floating_point()
    ):
        raise ValueError("dynamic BaseColor constraints require matching Nx3 floating RGB")
    if residual.ndim != 0 or multipliers.shape != (2,):
        raise ValueError("residual must be scalar and multipliers must contain two values")
    if not bool(
        torch.isfinite(prediction_rgb).all()
        and torch.isfinite(target_rgb).all()
        and torch.isfinite(residual)
        and torch.isfinite(multipliers).all()
    ):
        raise ValueError("dynamic BaseColor constraint inputs must be finite")

    difference = prediction_rgb - target_rgb
    rgb_rmse = torch.sqrt(difference.square().mean() + config.epsilon)
    prediction_opponent = orthogonal_color_coordinates(prediction_rgb)
    target_opponent = orthogonal_color_coordinates(target_rgb)
    prediction_chroma = torch.linalg.vector_norm(prediction_opponent[:, 1:], dim=-1)
    target_chroma = torch.linalg.vector_norm(target_opponent[:, 1:], dim=-1)
    chroma_retention = prediction_chroma.std() / target_chroma.std().clamp_min(
        config.epsilon
    )

    rgb_violation = rgb_rmse / config.rgb_rmse_ceiling - 1.0
    chroma_violation = (
        config.chroma_retention_floor - chroma_retention
    ) / config.chroma_retention_floor
    positive_rgb = torch.relu(rgb_violation)
    positive_chroma = torch.relu(chroma_violation)
    rgb_penalty = (
        multipliers[0] * positive_rgb
        + 0.5 * config.penalty_rho * positive_rgb.square()
    )
    chroma_penalty = (
        multipliers[1] * positive_chroma
        + 0.5 * config.penalty_rho * positive_chroma.square()
    )
    total = residual + rgb_penalty + chroma_penalty
    return DynamicBaseColorConstraintTerms(
        total=total,
        residual=residual,
        rgb_rmse=rgb_rmse,
        chroma_retention=chroma_retention,
        rgb_violation=rgb_violation,
        chroma_violation=chroma_violation,
        rgb_penalty=rgb_penalty,
        chroma_penalty=chroma_penalty,
    )


@torch.no_grad()
def update_dynamic_multipliers(
    multipliers: torch.Tensor,
    rgb_violation: torch.Tensor,
    chroma_violation: torch.Tensor,
    config: DynamicBaseColorConstraintConfig,
) -> None:
    """Perform one projected dual-ascent update in place."""

    if multipliers.shape != (2,) or rgb_violation.ndim != 0 or chroma_violation.ndim != 0:
        raise ValueError("invalid dynamic multiplier update shapes")
    delta = torch.stack((rgb_violation, chroma_violation)).to(multipliers)
    if not bool(torch.isfinite(delta).all() and torch.isfinite(multipliers).all()):
        raise ValueError("dynamic multiplier update must remain finite")
    multipliers.add_(config.dual_learning_rate * delta)
    multipliers.clamp_(0.0, config.multiplier_max)
