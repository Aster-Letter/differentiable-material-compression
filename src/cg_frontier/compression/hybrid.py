"""Interpolation-safe direct-BaseColor plus auxiliary-latent material representation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

from cg_frontier.compression.material import (
    Core4Targets,
    DecodedMaterial,
    reconstruct_normal,
)
from cg_frontier.compression.render_loss import (
    bilinear_sample_top_down_wrap,
    decoded_to_material,
    fake_quantize_unorm8,
    hard_quantize_unorm8,
    unorm8_encode_half_up,
)
from cg_frontier.render.gbuffer import Camera, GBufferResult
from cg_frontier.render.pbr import PointLight, shade_ggx


@dataclass(frozen=True)
class HybridInitialization:
    """Deterministic optimizer-only auxiliary initialization and its evidence."""

    direct_base_linear: torch.Tensor
    auxiliary_latent: torch.Tensor
    decoder: "AuxMaterialDecoder"
    metadata: dict[str, Any]


class AuxMaterialDecoder(nn.Module):
    """A 2/3→8→4 decoder for normal XY, roughness, and metallic raw values."""

    def __init__(self, input_channels: int) -> None:
        super().__init__()
        if input_channels not in (2, 3):
            raise ValueError("hybrid auxiliary input must contain two or three channels")
        self.input_channels = int(input_channels)
        self.network = nn.Sequential(
            nn.Linear(self.input_channels, 8), nn.ReLU(), nn.Linear(8, 4)
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.shape[-1:] != (self.input_channels,):
            raise ValueError("auxiliary latent width does not match decoder")
        return self.network(latent)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def weight_bytes_float32(self) -> int:
        return self.parameter_count * 4

    @property
    def macs_per_pixel(self) -> int:
        return self.input_channels * 8 + 8 * 4


def decode_auxiliary(
    decoder: AuxMaterialDecoder,
    auxiliary: torch.Tensor,
    direct_base_linear: torch.Tensor,
) -> DecodedMaterial:
    """Decode only auxiliary semantics; BaseColor is a direct linear value."""

    raw = decoder(auxiliary)
    normal_xy = torch.tanh(raw[..., 0:2])
    normal_xyz = reconstruct_normal(normal_xy)
    roughness = torch.sigmoid(raw[..., 2:3])
    metallic = torch.sigmoid(raw[..., 3:4])
    return DecodedMaterial(
        direct_base_linear,
        normal_xy,
        normal_xyz,
        roughness,
        metallic,
    )


def pack_hybrid_textures(
    direct_base_linear: torch.Tensor,
    auxiliary_latent: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack RGB+aux0 and aux1[/aux2] without allowing BaseColor gradients."""

    if direct_base_linear.ndim != 3 or direct_base_linear.shape[-1:] != (3,):
        raise ValueError("direct BaseColor texture must be HWC RGB")
    if auxiliary_latent.ndim != 3 or auxiliary_latent.shape[-1] not in (2, 3, 4):
        raise ValueError("auxiliary latent texture must be HWC with two, three, or four channels")
    if direct_base_linear.shape[:2] != auxiliary_latent.shape[:2]:
        raise ValueError("hybrid texture extents must match")
    texture_a = torch.cat(
        (direct_base_linear.detach(), auxiliary_latent[..., 0:1]), dim=-1
    )
    return texture_a, auxiliary_latent[..., 1:]


def sample_and_decode_hybrid(
    geometry: GBufferResult,
    direct_base_linear: torch.Tensor,
    auxiliary_latent: torch.Tensor,
    decoder: AuxMaterialDecoder,
    *,
    quantization: str,
):
    """Sample two physical textures, bypassing all BaseColor decoding."""

    if quantization == "float":
        deployed_auxiliary = auxiliary_latent
    elif quantization == "hard":
        deployed_auxiliary = hard_quantize_unorm8(auxiliary_latent)
    elif quantization == "fake":
        deployed_auxiliary = fake_quantize_unorm8(auxiliary_latent)
    else:
        raise ValueError(f"unsupported quantization mode: {quantization}")
    texture_a, texture_b = pack_hybrid_textures(
        hard_quantize_unorm8(direct_base_linear.detach()), deployed_auxiliary
    )
    uv = geometry.torch_buffers["uv"]
    sampled_a = bilinear_sample_top_down_wrap(texture_a, uv)
    sampled_b = bilinear_sample_top_down_wrap(texture_b, uv)
    sampled_auxiliary = torch.cat((sampled_a[..., 3:4], sampled_b), dim=-1)
    decoded = decode_auxiliary(decoder, sampled_auxiliary, sampled_a[..., 0:3])
    return decoded_to_material(geometry, decoded)


def render_hybrid_material(
    geometry: GBufferResult,
    camera: Camera,
    light: PointLight,
    direct_base_linear: torch.Tensor,
    auxiliary_latent: torch.Tensor,
    decoder: AuxMaterialDecoder,
    *,
    quantization: str,
    minimum_roughness: float,
):
    material = sample_and_decode_hybrid(
        geometry,
        direct_base_linear,
        auxiliary_latent,
        decoder,
        quantization=quantization,
    )
    return (
        shade_ggx(
            geometry,
            camera,
            light,
            material_override=material,
            minimum_roughness=minimum_roughness,
        ),
        material,
    )


def _inverse_auxiliary_targets(targets: Core4Targets, epsilon: float) -> torch.Tensor:
    normal_xy = targets.normal_xyz[:, 0:2].to(torch.float64)
    radius = torch.linalg.vector_norm(normal_xy, dim=-1, keepdim=True)
    normal_xy = normal_xy / torch.clamp(radius / (1.0 - epsilon), min=1.0)
    normal_raw = torch.atanh(normal_xy.clamp(-1.0 + epsilon, 1.0 - epsilon))

    def logit(values: torch.Tensor) -> torch.Tensor:
        bounded = values.to(torch.float64).clamp(epsilon, 1.0 - epsilon)
        return torch.log(bounded) - torch.log1p(-bounded)

    return torch.cat((normal_raw, logit(targets.roughness), logit(targets.metallic)), dim=-1)


@torch.no_grad()
def deterministic_pca_initialization(
    targets: Core4Targets,
    optimizer_mask: np.ndarray,
    input_channels: int,
    *,
    epsilon: float = 1.0e-4,
    ridge: float = 1.0e-10,
) -> HybridInitialization:
    """Build a deterministic PCA latent and exact-function affine ReLU initializer."""

    if input_channels not in (2, 3):
        raise ValueError("PCA initialization supports rank two or three")
    mask = np.asarray(optimizer_mask, dtype=bool)
    if mask.shape != (targets.height, targets.width) or not np.any(mask):
        raise ValueError("optimizer mask must be non-empty and match the atlas")
    device = targets.base_color_linear.device
    optimizer_ids = torch.from_numpy(np.flatnonzero(mask.reshape(-1))).to(device=device)
    raw = _inverse_auxiliary_targets(targets, epsilon)
    optimizer_raw = raw[optimizer_ids]
    mean = optimizer_raw.mean(dim=0)
    std = optimizer_raw.std(dim=0, unbiased=False).clamp_min(1.0e-8)
    standardized = (raw - mean) / std
    optimizer_standardized = standardized[optimizer_ids]
    covariance = optimizer_standardized.T @ optimizer_standardized
    covariance /= float(optimizer_standardized.shape[0])
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    components = eigenvectors[:, order[:input_channels]]
    for column in range(input_channels):
        vector = components[:, column]
        pivot = torch.argmax(torch.abs(vector))
        if vector[pivot] < 0:
            components[:, column].mul_(-1.0)
    scores = standardized @ components
    optimizer_scores = scores[optimizer_ids]
    score_min = optimizer_scores.amin(dim=0)
    score_max = optimizer_scores.amax(dim=0)
    score_span = (score_max - score_min).clamp_min(1.0e-8)
    latent = ((scores - score_min) / score_span).clamp(0.0, 1.0).to(torch.float32)
    latent = hard_quantize_unorm8(latent)

    design = torch.cat(
        (
            latent[optimizer_ids].to(torch.float64),
            torch.ones((optimizer_ids.numel(), 1), dtype=torch.float64, device=device),
        ),
        dim=-1,
    )
    xtx = design.T @ design
    xty = design.T @ optimizer_raw
    solution = torch.linalg.solve(
        xtx + torch.eye(input_channels + 1, dtype=torch.float64, device=device) * ridge,
        xty,
    ).to(torch.float32)

    decoder = AuxMaterialDecoder(input_channels).to(device=device, dtype=torch.float32)
    hidden = decoder.network[0]
    output = decoder.network[2]
    assert isinstance(hidden, nn.Linear) and isinstance(output, nn.Linear)
    hidden.weight.zero_()
    hidden.bias.zero_()
    output.weight.zero_()
    output.bias.copy_(solution[input_channels])
    for channel in range(input_channels):
        hidden.weight[channel, channel] = 1.0
    output.weight[:, :input_channels].copy_(solution[:input_channels].T)
    for unit in range(input_channels, 8):
        for channel in range(input_channels):
            hidden.weight[unit, channel] = 1.0e-3 * (
                1.0 if (unit + channel) % 2 == 0 else -1.0
            )
        hidden.bias[unit] = 0.1

    direct = hard_quantize_unorm8(
        targets.base_color_linear.reshape(targets.height, targets.width, 3)
    ).detach()
    return HybridInitialization(
        direct_base_linear=direct,
        auxiliary_latent=latent.reshape(targets.height, targets.width, input_channels),
        decoder=decoder,
        metadata={
            "method": "optimizer_only_standardized_raw_pca_then_float64_least_squares",
            "input_channels": input_channels,
            "inverse_postprocess_epsilon": epsilon,
            "optimizer_texels": int(optimizer_ids.numel()),
            "raw_mean": mean.cpu().tolist(),
            "raw_std": std.cpu().tolist(),
            "eigenvalues": eigenvalues.cpu().tolist(),
            "components": components.cpu().tolist(),
            "score_min": score_min.cpu().tolist(),
            "score_max": score_max.cpu().tolist(),
        },
    )


def export_hybrid_textures(
    direct_base_linear: torch.Tensor,
    auxiliary_latent: torch.Tensor,
    texture_a_path: Path,
    texture_b_path: Path,
) -> dict[str, Any]:
    """Export exact logical RGBA8 plus R8/RG8/RGB8 PNG payloads."""

    texture_a, texture_b = pack_hybrid_textures(
        hard_quantize_unorm8(direct_base_linear), hard_quantize_unorm8(auxiliary_latent)
    )
    a_u8 = unorm8_encode_half_up(texture_a).cpu().numpy()
    b_u8 = unorm8_encode_half_up(texture_b).cpu().numpy()
    texture_a_path.parent.mkdir(parents=True, exist_ok=True)
    texture_b_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a_u8, mode="RGBA").save(texture_a_path, compress_level=9)
    if b_u8.shape[-1] == 1:
        Image.fromarray(b_u8[..., 0], mode="L").save(texture_b_path, compress_level=9)
        logical_format = "R8_UNORM"
    elif b_u8.shape[-1] == 2:
        Image.fromarray(b_u8, mode="LA").save(texture_b_path, compress_level=9)
        logical_format = "RG8_UNORM"
    elif b_u8.shape[-1] == 3:
        Image.fromarray(b_u8, mode="RGB").save(texture_b_path, compress_level=9)
        logical_format = "RGB8_UNORM_LOGICAL"
    else:
        raise ValueError("Hybrid Texture B must contain one to three logical channels")
    height, width = a_u8.shape[:2]
    return {
        "texture_a": {
            "format": "RGBA8_UNORM",
            "shape": list(a_u8.shape),
            "raw_bytes": int(a_u8.size),
        },
        "texture_b": {
            "format": logical_format,
            "shape": list(b_u8.shape),
            "raw_bytes": int(b_u8.size),
        },
        "logical_raw_bytes": int(height * width * (3 + auxiliary_latent.shape[-1])),
        "texture_samples": 2,
    }
