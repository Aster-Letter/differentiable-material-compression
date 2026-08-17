"""Regularizers and gradient-scale calibration for affine latent training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import torch


@dataclass(frozen=True)
class ChartAwareTV:
    """Normalized TV value and the number of accepted atlas edges."""

    loss: torch.Tensor
    edge_count: int


@dataclass(frozen=True)
class GradientRatioCalibration:
    batch_count: int
    target_ratios: tuple[float, ...]
    base_gradient_norms: tuple[float, ...]
    extension_gradient_norms: tuple[float, ...]
    median_base_over_extension: float
    lambdas: dict[float, float]


def hard_quantize_unorm8(latent: torch.Tensor) -> torch.Tensor:
    """Half-up UNORM8 quantization and dequantization."""

    clamped = torch.clamp(latent, 0.0, 1.0)
    return torch.floor(clamped * 255.0 + 0.5) / 255.0


def calibrate_gradient_ratios(
    latent: torch.Tensor,
    batches: Sequence[object],
    base_objective: Callable[[torch.Tensor, object], torch.Tensor],
    extension_objective: Callable[[torch.Tensor, object], torch.Tensor],
    *,
    target_ratios: tuple[float, ...] = (0.02, 0.05, 0.10),
    epsilon: float = 1.0e-12,
) -> GradientRatioCalibration:
    """Calibrate fixed extension weights from eight latent-gradient batches."""

    if len(batches) != 8:
        raise ValueError("gradient calibration requires exactly 8 batches")
    base_norms: list[float] = []
    extension_norms: list[float] = []
    scales: list[torch.Tensor] = []
    for batch in batches:
        base_loss = base_objective(latent, batch)
        base_gradient = torch.autograd.grad(base_loss, latent)[0]
        extension_loss = extension_objective(latent, batch)
        extension_gradient = torch.autograd.grad(extension_loss, latent)[0]
        base_norm = torch.linalg.vector_norm(base_gradient)
        extension_norm = torch.linalg.vector_norm(extension_gradient)
        base_norms.append(float(base_norm.detach().cpu()))
        extension_norms.append(float(extension_norm.detach().cpu()))
        scales.append(base_norm / (extension_norm + epsilon))
    median_scale = torch.median(torch.stack(scales))
    median_value = float(median_scale.detach().cpu())
    return GradientRatioCalibration(
        batch_count=8,
        target_ratios=tuple(float(value) for value in target_ratios),
        base_gradient_norms=tuple(base_norms),
        extension_gradient_norms=tuple(extension_norms),
        median_base_over_extension=median_value,
        lambdas={float(ratio): float(ratio) * median_value for ratio in target_ratios},
    )


def fake_quantize_unorm8(latent: torch.Tensor) -> torch.Tensor:
    """Half-up UNORM8 forward with an identity STE inside the legal range."""

    clamped = torch.clamp(latent, 0.0, 1.0)
    hard = hard_quantize_unorm8(clamped)
    return clamped + (hard - clamped).detach()


def chart_aware_quantized_tv(
    latent: torch.Tensor,
    valid_mask: torch.Tensor,
    chart_ids: torch.Tensor,
    *,
    epsilon: float = 1.0 / 255.0,
) -> ChartAwareTV:
    """Charbonnier TV over fake-quantized valid edges within one UV chart."""

    quantized = fake_quantize_unorm8(latent)
    horizontal_mask = (
        valid_mask[:, :-1]
        & valid_mask[:, 1:]
        & (chart_ids[:, :-1] == chart_ids[:, 1:])
    )
    vertical_mask = (
        valid_mask[:-1, :]
        & valid_mask[1:, :]
        & (chart_ids[:-1, :] == chart_ids[1:, :])
    )
    horizontal = quantized[:, 1:] - quantized[:, :-1]
    vertical = quantized[1:, :] - quantized[:-1, :]
    edge_count = int(horizontal_mask.sum() + vertical_mask.sum())
    if edge_count == 0:
        return ChartAwareTV(loss=quantized.sum() * 0.0, edge_count=0)

    horizontal_penalty = (
        torch.sqrt(horizontal.square() + epsilon**2) - epsilon
    ).mean(dim=-1)
    vertical_penalty = (
        torch.sqrt(vertical.square() + epsilon**2) - epsilon
    ).mean(dim=-1)
    loss = (
        horizontal_penalty[horizontal_mask].sum()
        + vertical_penalty[vertical_mask].sum()
    ) / edge_count
    return ChartAwareTV(loss=loss, edge_count=edge_count)
