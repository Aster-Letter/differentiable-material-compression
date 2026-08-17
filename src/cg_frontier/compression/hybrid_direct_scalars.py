"""Hybrid material with direct-filtered BaseColor/roughness/metallic and neural normal."""

from __future__ import annotations

import torch
from torch import nn

from cg_frontier.compression.hybrid import pack_hybrid_textures
from cg_frontier.compression.material import Core4Targets, DecodedMaterial, reconstruct_normal
from cg_frontier.compression.render_loss import (
    bilinear_sample_top_down_wrap,
    decoded_to_material,
    fake_quantize_unorm8,
    hard_quantize_unorm8,
)
from cg_frontier.render.gbuffer import Camera, GBufferResult
from cg_frontier.render.pbr import PointLight, shade_ggx


class NormalAuxDecoder(nn.Module):
    """The bounded normal-only 2→6→2 ReLU decoder used by D7-D."""

    def __init__(self) -> None:
        super().__init__()
        self.normal_head = nn.Sequential(nn.Linear(2, 6), nn.ReLU(), nn.Linear(6, 2))

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.shape[-1:] != (2,):
            raise ValueError("normal latent must contain exactly two channels")
        return self.normal_head(latent)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def weight_bytes_float32(self) -> int:
        return self.parameter_count * 4

    @property
    def macs_per_pixel(self) -> int:
        return 2 * 6 + 6 * 2


def build_direct_scalar_auxiliary(
    normal_latent: torch.Tensor,
    targets: Core4Targets,
) -> torch.Tensor:
    """Pack independent normal latent plus immutable direct linear scalar texels."""

    if normal_latent.shape != (targets.height, targets.width, 2):
        raise ValueError("normal latent extent must match the Core-4 atlas")
    scalars = torch.cat((targets.roughness, targets.metallic), dim=-1).reshape(
        targets.height, targets.width, 2
    )
    return torch.cat(
        (hard_quantize_unorm8(normal_latent), hard_quantize_unorm8(scalars).detach()), dim=-1
    )


def decode_direct_scalars(
    decoder: NormalAuxDecoder,
    auxiliary: torch.Tensor,
    direct_base_linear: torch.Tensor,
) -> DecodedMaterial:
    """Decode normal only; direct scalars bypass every neural/postprocess scalar path."""

    if auxiliary.shape[-1:] != (4,):
        raise ValueError("D7-D auxiliary must contain normal2 + roughness + metallic")
    normal_xy = torch.tanh(decoder(auxiliary[..., :2]))
    normal_xyz = reconstruct_normal(normal_xy)
    return DecodedMaterial(
        direct_base_linear,
        normal_xy,
        normal_xyz,
        auxiliary[..., 2:3],
        auxiliary[..., 3:4],
    )


def sample_and_decode_direct_scalars(
    geometry: GBufferResult,
    direct_base_linear: torch.Tensor,
    auxiliary: torch.Tensor,
    decoder: NormalAuxDecoder,
    *,
    quantization: str,
):
    """Use two identical samplers while keeping all direct semantics immutable."""

    normal = auxiliary[..., :2]
    if quantization == "float":
        deployed_normal = normal
    elif quantization == "hard":
        deployed_normal = hard_quantize_unorm8(normal)
    elif quantization == "fake":
        deployed_normal = fake_quantize_unorm8(normal)
    else:
        raise ValueError(f"unsupported quantization mode: {quantization}")
    deployed_auxiliary = torch.cat(
        (deployed_normal, hard_quantize_unorm8(auxiliary[..., 2:]).detach()), dim=-1
    )
    texture_a, texture_b = pack_hybrid_textures(
        hard_quantize_unorm8(direct_base_linear).detach(), deployed_auxiliary
    )
    uv = geometry.torch_buffers["uv"]
    sampled_a = bilinear_sample_top_down_wrap(texture_a, uv)
    sampled_b = bilinear_sample_top_down_wrap(texture_b, uv)
    sampled_auxiliary = torch.cat((sampled_a[..., 3:4], sampled_b), dim=-1)
    decoded = decode_direct_scalars(decoder, sampled_auxiliary, sampled_a[..., :3])
    return decoded_to_material(geometry, decoded)


def render_direct_scalar_material(
    geometry: GBufferResult,
    camera: Camera,
    light: PointLight,
    direct_base_linear: torch.Tensor,
    auxiliary: torch.Tensor,
    decoder: NormalAuxDecoder,
    *,
    quantization: str,
    minimum_roughness: float,
):
    material = sample_and_decode_direct_scalars(
        geometry,
        direct_base_linear,
        auxiliary,
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
