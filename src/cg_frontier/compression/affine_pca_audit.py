"""Train-only metrics for C4 affine PCA representation audits."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from cg_frontier.compression.affine_pca import EnhancedPCASpec


def enhanced_candidate_specs() -> dict[str, EnhancedPCASpec]:
    """Return the frozen small generic metric ablation matrix."""

    return {
        "balanced": EnhancedPCASpec(semantic_group_balance=True),
        "opponent2": EnhancedPCASpec(
            opponent_chroma_weight=2.0,
            semantic_group_balance=True,
        ),
        "chroma4": EnhancedPCASpec(
            chroma_tail_strength=3.0,
            opponent_chroma_weight=2.0,
            semantic_group_balance=True,
        ),
        "chroma8": EnhancedPCASpec(
            chroma_tail_strength=7.0,
            opponent_chroma_weight=2.0,
            semantic_group_balance=True,
        ),
    }


def cluster_balanced_candidate_specs() -> dict[str, EnhancedPCASpec]:
    """Return the bounded cluster-balance power ablation for enhanced P0."""

    common = {
        "chroma_tail_strength": 7.0,
        "opponent_chroma_weight": 2.0,
        "semantic_group_balance": True,
        "material_cluster_count": 4,
        "material_cluster_seed": 20260807,
    }
    return {
        "chroma8_k4_b05": EnhancedPCASpec(
            **common, material_cluster_balance_power=0.5
        ),
        "chroma8_k4_b10": EnhancedPCASpec(
            **common, material_cluster_balance_power=1.0
        ),
    }


def residual_reweighted_candidate_specs() -> dict[str, EnhancedPCASpec]:
    """Return the bounded residual-tail ablation for enhanced global P0."""

    common = {
        "chroma_tail_strength": 7.0,
        "opponent_chroma_weight": 2.0,
        "semantic_group_balance": True,
        "residual_reweight_iterations": 2,
    }
    return {
        "chroma8_resid3": EnhancedPCASpec(
            **common, residual_tail_strength=3.0
        ),
        "chroma8_resid7": EnhancedPCASpec(
            **common, residual_tail_strength=7.0
        ),
    }


def material_region_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    tail_quantile: float = 0.95,
) -> dict[str, float | int | str]:
    """Measure generic source-defined chroma separation and material error."""

    if prediction.shape != target.shape or target.ndim != 2 or target.shape[1] != 7:
        raise ValueError("prediction and target must have matching (N, 7) shape")
    if not 0.0 < tail_quantile < 1.0:
        raise ValueError("tail quantile must be in (0, 1)")
    source_chroma = target[:, :3].amax(dim=-1) - target[:, :3].amin(dim=-1)
    predicted_chroma = (
        prediction[:, :3].amax(dim=-1) - prediction[:, :3].amin(dim=-1)
    )
    threshold = torch.quantile(source_chroma, tail_quantile)
    tail = source_chroma > threshold
    if not bool(tail.any()) or not bool((~tail).any()):
        raise ValueError("chroma quantile did not produce two non-empty regions")
    epsilon = torch.finfo(target.dtype).eps
    source_contrast = source_chroma[tail].mean() / torch.clamp(
        source_chroma[~tail].mean(), min=epsilon
    )
    predicted_contrast = predicted_chroma[tail].mean() / torch.clamp(
        predicted_chroma[~tail].mean(), min=epsilon
    )
    return {
        "tail_definition": "source_linear_rgb_chroma_quantile",
        "tail_quantile": tail_quantile,
        "tail_threshold": float(threshold.detach().cpu()),
        "tail_texels": int(torch.count_nonzero(tail)),
        "seven_mae": float(F.l1_loss(prediction, target).detach().cpu()),
        "base_color_mae": float(
            F.l1_loss(prediction[:, :3], target[:, :3]).detach().cpu()
        ),
        "tail_base_color_mae": float(
            F.l1_loss(prediction[tail, :3], target[tail, :3]).detach().cpu()
        ),
        "nontail_base_color_mae": float(
            F.l1_loss(prediction[~tail, :3], target[~tail, :3]).detach().cpu()
        ),
        "normal_xy_mae": float(
            F.l1_loss(prediction[:, 3:5], target[:, 3:5]).detach().cpu()
        ),
        "roughness_mae": float(
            F.l1_loss(prediction[:, 5], target[:, 5]).detach().cpu()
        ),
        "metallic_mae": float(
            F.l1_loss(prediction[:, 6], target[:, 6]).detach().cpu()
        ),
        "source_chroma_contrast": float(source_contrast.detach().cpu()),
        "predicted_chroma_contrast": float(predicted_contrast.detach().cpu()),
        "chroma_contrast_retention": float(
            (predicted_contrast / source_contrast).detach().cpu()
        ),
    }
