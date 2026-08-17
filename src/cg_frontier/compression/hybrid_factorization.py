"""Causal shared/split-head/partitioned Hybrid auxiliary representations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from cg_frontier.compression.hybrid import (
    AuxMaterialDecoder,
    HybridInitialization,
    _inverse_auxiliary_targets,
    deterministic_pca_initialization,
)
from cg_frontier.compression.material import Core4Targets
from cg_frontier.compression.render_loss import hard_quantize_unorm8


@dataclass(frozen=True)
class FactorizationSpec:
    name: str
    aux_channels: int
    normal_input: tuple[int, ...]
    scalar_input: tuple[int, ...]
    normal_hidden: int
    scalar_hidden: int
    architecture: str


SPECS: dict[str, FactorizationSpec] = {
    "d6_h": FactorizationSpec("d6_h", 3, (0, 1, 2), (0, 1, 2), 5, 5, "shared_latent_split_heads"),
    "d6_p": FactorizationSpec("d6_p", 3, (0, 1), (2,), 6, 8, "partitioned_normal2_scalar1"),
    "d7_p": FactorizationSpec("d7_p", 4, (0, 1), (2, 3), 6, 6, "partitioned_normal2_scalar2"),
}


def candidate_aux_channels(candidate: str) -> int:
    if candidate == "d6_s":
        return 3
    if candidate == "o7_direct":
        return 4
    return SPECS[candidate].aux_channels


class FactorizedAuxDecoder(nn.Module):
    """Two ReLU heads with explicit normal/scalar latent ownership."""

    def __init__(self, spec: FactorizationSpec) -> None:
        super().__init__()
        self.spec = spec
        self.normal_head = nn.Sequential(
            nn.Linear(len(spec.normal_input), spec.normal_hidden),
            nn.ReLU(),
            nn.Linear(spec.normal_hidden, 2),
        )
        self.scalar_head = nn.Sequential(
            nn.Linear(len(spec.scalar_input), spec.scalar_hidden),
            nn.ReLU(),
            nn.Linear(spec.scalar_hidden, 2),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.shape[-1:] != (self.spec.aux_channels,):
            raise ValueError("factorized auxiliary latent width does not match decoder")
        normal_ids = torch.as_tensor(self.spec.normal_input, device=latent.device)
        scalar_ids = torch.as_tensor(self.spec.scalar_input, device=latent.device)
        normal = self.normal_head(torch.index_select(latent, -1, normal_ids))
        scalar = self.scalar_head(torch.index_select(latent, -1, scalar_ids))
        return torch.cat((normal, scalar), dim=-1)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def weight_bytes_float32(self) -> int:
        return self.parameter_count * 4

    @property
    def macs_per_pixel(self) -> int:
        spec = self.spec
        return (
            len(spec.normal_input) * spec.normal_hidden
            + spec.normal_hidden * 2
            + len(spec.scalar_input) * spec.scalar_hidden
            + spec.scalar_hidden * 2
        )


@dataclass(frozen=True)
class CausalHybridInitialization:
    direct_base_linear: torch.Tensor
    auxiliary_latent: torch.Tensor
    decoder: nn.Module | None
    metadata: dict[str, Any]


def _pca_latent(
    raw: torch.Tensor,
    optimizer_ids: torch.Tensor,
    rank: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    optimizer_raw = raw[optimizer_ids].to(torch.float64)
    mean = optimizer_raw.mean(dim=0)
    std = optimizer_raw.std(dim=0, unbiased=False).clamp_min(1.0e-8)
    standardized = (raw.to(torch.float64) - mean) / std
    selected = standardized[optimizer_ids]
    covariance = selected.T @ selected / float(selected.shape[0])
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    components = eigenvectors[:, order[:rank]]
    for column in range(rank):
        pivot = torch.argmax(torch.abs(components[:, column]))
        if components[pivot, column] < 0:
            components[:, column].mul_(-1.0)
    scores = standardized @ components
    optimizer_scores = scores[optimizer_ids]
    score_min = optimizer_scores.amin(dim=0)
    score_max = optimizer_scores.amax(dim=0)
    span = (score_max - score_min).clamp_min(1.0e-8)
    latent = hard_quantize_unorm8(((scores - score_min) / span).clamp(0.0, 1.0).to(torch.float32))
    return latent, {
        "rank": rank,
        "raw_mean": mean.cpu().tolist(),
        "raw_std": std.cpu().tolist(),
        "eigenvalues": eigenvalues.cpu().tolist(),
        "components": components.cpu().tolist(),
        "score_min": score_min.cpu().tolist(),
        "score_max": score_max.cpu().tolist(),
    }


def _affine_solution(
    latent: torch.Tensor,
    raw: torch.Tensor,
    optimizer_ids: torch.Tensor,
    ridge: float,
) -> torch.Tensor:
    selected = latent[optimizer_ids].to(torch.float64)
    design = torch.cat(
        (selected, torch.ones((selected.shape[0], 1), dtype=torch.float64, device=selected.device)),
        dim=-1,
    )
    xtx = design.T @ design
    xty = design.T @ raw[optimizer_ids].to(torch.float64)
    return torch.linalg.solve(
        xtx + torch.eye(design.shape[1], dtype=torch.float64, device=design.device) * ridge,
        xty,
    ).to(torch.float32)


@torch.no_grad()
def _embed_affine(head: nn.Sequential, solution: torch.Tensor) -> None:
    hidden, output = head[0], head[2]
    assert isinstance(hidden, nn.Linear) and isinstance(output, nn.Linear)
    rank = solution.shape[0] - 1
    hidden.weight.zero_()
    hidden.bias.zero_()
    output.weight.zero_()
    output.bias.copy_(solution[rank])
    for channel in range(rank):
        hidden.weight[channel, channel] = 1.0
    output.weight[:, :rank].copy_(solution[:rank].T)
    for unit in range(rank, hidden.out_features):
        for channel in range(hidden.in_features):
            hidden.weight[unit, channel] = 1.0e-3 * (1.0 if (unit + channel) % 2 == 0 else -1.0)
        hidden.bias[unit] = 0.1


def _optimizer_ids(mask: np.ndarray, targets: Core4Targets) -> torch.Tensor:
    array = np.asarray(mask, dtype=bool)
    if array.shape != (targets.height, targets.width) or not np.any(array):
        raise ValueError("optimizer mask must be non-empty and match the atlas")
    return torch.from_numpy(np.flatnonzero(array.reshape(-1))).to(targets.base_color_linear.device)


@torch.no_grad()
def deterministic_causal_initialization(
    targets: Core4Targets,
    optimizer_mask: np.ndarray,
    candidate: str,
    *,
    epsilon: float = 1.0e-4,
    ridge: float = 1.0e-10,
) -> CausalHybridInitialization:
    """Initialize one causal candidate without selection/validation information."""

    direct = hard_quantize_unorm8(
        targets.base_color_linear.reshape(targets.height, targets.width, 3)
    ).detach()
    if candidate == "d6_s":
        original: HybridInitialization = deterministic_pca_initialization(
            targets, optimizer_mask, 3, epsilon=epsilon, ridge=ridge
        )
        return CausalHybridInitialization(
            original.direct_base_linear,
            original.auxiliary_latent,
            original.decoder,
            {"candidate": candidate, "architecture": "shared_3_to_8_to_4", "joint": original.metadata},
        )
    raw = _inverse_auxiliary_targets(targets, epsilon)
    if candidate == "o7_direct":
        normal_xy = targets.normal_xyz[:, :2].reshape(targets.height, targets.width, 2)
        normal_encoded = ((normal_xy + 1.0) * 0.5).clamp(0.0, 1.0)
        scalar = torch.cat((targets.roughness, targets.metallic), dim=-1).reshape(
            targets.height, targets.width, 2
        )
        latent = hard_quantize_unorm8(torch.cat((normal_encoded, scalar), dim=-1))
        return CausalHybridInitialization(
            direct,
            latent,
            None,
            {"candidate": candidate, "architecture": "direct_normalXY_roughness_metallic"},
        )
    spec = SPECS[candidate]
    ids = _optimizer_ids(optimizer_mask, targets)
    if candidate == "d6_h":
        joint = deterministic_pca_initialization(targets, optimizer_mask, 3, epsilon=epsilon, ridge=ridge)
        latent_flat = joint.auxiliary_latent.reshape(-1, 3)
        normal_meta = scalar_meta = {"source": "shared_joint_rank3"}
    else:
        normal_latent, normal_meta = _pca_latent(raw[:, :2], ids, 2)
        scalar_rank = 1 if candidate == "d6_p" else 2
        scalar_latent, scalar_meta = _pca_latent(raw[:, 2:4], ids, scalar_rank)
        latent_flat = torch.cat((normal_latent, scalar_latent), dim=-1)
    decoder = FactorizedAuxDecoder(spec).to(device=raw.device, dtype=torch.float32)
    normal_input = latent_flat[:, list(spec.normal_input)]
    scalar_input = latent_flat[:, list(spec.scalar_input)]
    _embed_affine(decoder.normal_head, _affine_solution(normal_input, raw[:, :2], ids, ridge))
    _embed_affine(decoder.scalar_head, _affine_solution(scalar_input, raw[:, 2:4], ids, ridge))
    return CausalHybridInitialization(
        direct,
        latent_flat.reshape(targets.height, targets.width, spec.aux_channels),
        decoder,
        {
            "candidate": candidate,
            "architecture": spec.architecture,
            "optimizer_texels": int(ids.numel()),
            "inverse_postprocess_epsilon": epsilon,
            "normal_initializer": normal_meta,
            "scalar_initializer": scalar_meta,
        },
    )


def decoder_for_candidate(candidate: str) -> nn.Module:
    if candidate == "d6_s":
        return AuxMaterialDecoder(3)
    return FactorizedAuxDecoder(SPECS[candidate])


def direct_semantic_material(auxiliary: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if auxiliary.shape[-1:] != (4,):
        raise ValueError("O7-Direct requires four auxiliary channels")
    normal_xy = auxiliary[..., :2] * 2.0 - 1.0
    radius = torch.linalg.vector_norm(normal_xy, dim=-1, keepdim=True)
    normal_xy = normal_xy / torch.clamp(radius / (1.0 - 1.0e-6), min=1.0)
    normal_z = torch.sqrt(torch.clamp(1.0 - torch.sum(normal_xy * normal_xy, dim=-1, keepdim=True), min=1.0e-8))
    return F.normalize(torch.cat((normal_xy, normal_z), dim=-1), dim=-1), auxiliary[..., 2:3], auxiliary[..., 3:4]


def flattened_gradients(
    loss: torch.Tensor,
    variables: Iterable[torch.Tensor],
    *,
    retain_graph: bool,
) -> torch.Tensor:
    items = tuple(variables)
    gradients = torch.autograd.grad(loss, items, retain_graph=retain_graph, allow_unused=True)
    return torch.cat(
        [torch.zeros_like(value).reshape(-1) if grad is None else grad.reshape(-1) for value, grad in zip(items, gradients, strict=True)]
    )


def gradient_conflict_report(
    decoder: nn.Module,
    latent_samples: torch.Tensor,
    target: Core4Targets,
) -> dict[str, Any]:
    """Measure fixed-probe task gradients without updating parameters."""

    latent = latent_samples.detach().clone().requires_grad_(True)
    raw = decoder(latent)
    normal_xy = torch.tanh(raw[..., :2])
    radius = torch.linalg.vector_norm(normal_xy, dim=-1, keepdim=True)
    normal_xy = normal_xy / torch.clamp(radius / (1.0 - 1.0e-6), min=1.0)
    normal_z = torch.sqrt(torch.clamp(1.0 - torch.sum(normal_xy * normal_xy, dim=-1, keepdim=True), min=1.0e-8))
    normal = F.normalize(torch.cat((normal_xy, normal_z), dim=-1), dim=-1)
    losses = {
        "normal": torch.mean(1.0 - torch.sum(normal * target.normal_xyz, dim=-1)),
        "roughness": F.l1_loss(torch.sigmoid(raw[..., 2:3]), target.roughness),
        "metallic": F.l1_loss(torch.sigmoid(raw[..., 3:4]), target.metallic),
    }
    variables = (latent, *tuple(decoder.parameters()))
    vectors: dict[str, torch.Tensor] = {}
    names = tuple(losses)
    for index, name in enumerate(names):
        vectors[name] = flattened_gradients(
            losses[name], variables, retain_graph=index < len(names) - 1
        ).detach()
    result: dict[str, Any] = {
        "losses": {name: float(value.detach()) for name, value in losses.items()},
        "gradient_l2": {name: float(torch.linalg.vector_norm(value)) for name, value in vectors.items()},
        "cosines": {},
    }
    for left_index, left in enumerate(names):
        for right in names[left_index + 1 :]:
            denominator = torch.linalg.vector_norm(vectors[left]) * torch.linalg.vector_norm(vectors[right])
            cosine = torch.sum(vectors[left] * vectors[right]) / denominator if denominator > 0 else torch.tensor(0.0)
            result["cosines"][f"{left}__{right}"] = float(cosine)
    return result
