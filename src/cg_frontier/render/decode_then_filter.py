"""Independent decode-then-filter renderer for learned Core-4 materials."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch

from cg_frontier.compression.decode_then_filter import (
    DecodeThenFilterDecoder,
    decode_then_filter_sample,
)
from cg_frontier.compression.material import DecodedMaterial
from cg_frontier.compression.render_loss import decoded_to_material, display_transform
from cg_frontier.render.gbuffer import Camera, GBufferResult, MaterialBuffers
from cg_frontier.render.pbr import PointLight, shade_ggx


@dataclass(frozen=True)
class DTFLatentMaterialSource:
    """Stored C4/C5 latent and shared four-corner decoder."""

    latent_texture: torch.Tensor
    decoder: DecodeThenFilterDecoder
    quantization: str = "hard"


@dataclass(frozen=True)
class DTFReferenceMaterialSource:
    """Direct Core-4 reference material for renderer parity."""

    material: MaterialBuffers


@dataclass(frozen=True)
class DTFRenderBundle:
    """Differentiable products emitted by the standalone DTF renderer."""

    renderer_identifier: str
    coverage: torch.Tensor
    uv: torch.Tensor
    geometry_buffers: Mapping[str, torch.Tensor]
    material: MaterialBuffers
    linear_hdr: torch.Tensor
    display_rgb: torch.Tensor
    metadata: Mapping[str, Any]


def _reshape_decoded(decoded: DecodedMaterial, shape: tuple[int, ...]) -> DecodedMaterial:
    def field(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(*shape, value.shape[-1])

    return DecodedMaterial(
        base_color_linear=field(decoded.base_color_linear),
        normal_xy=field(decoded.normal_xy),
        normal_xyz=field(decoded.normal_xyz),
        roughness=field(decoded.roughness),
        metallic=field(decoded.metallic),
    )


class DecodeThenFilterRenderer:
    """Decode four quantized texels before filtering material semantics."""

    renderer_identifier = "decode_then_filter_renderer_v1"

    def __init__(self, *, display_exposure: float, minimum_roughness: float) -> None:
        if display_exposure <= 0.0:
            raise ValueError("display exposure must be positive")
        if not (0.0 < minimum_roughness <= 1.0):
            raise ValueError("minimum roughness must be within (0, 1]")
        self.display_exposure = float(display_exposure)
        self.minimum_roughness = float(minimum_roughness)

    def render(
        self,
        geometry: GBufferResult,
        camera: Camera,
        light: PointLight,
        source: DTFLatentMaterialSource | DTFReferenceMaterialSource,
        *,
        input_hashes: Mapping[str, str],
    ) -> DTFRenderBundle:
        buffers = geometry.torch_buffers
        missing = sorted({"mask", "uv"}.difference(buffers))
        if missing:
            raise ValueError(f"DTF geometry GBuffer is missing: {', '.join(missing)}")
        uv = buffers["uv"]
        if uv.ndim != 3 or uv.shape[-1] != 2:
            raise ValueError("DTF renderer requires an HxWx2 UV buffer")
        if isinstance(source, DTFLatentMaterialSource):
            sample = decode_then_filter_sample(
                source.latent_texture,
                uv.reshape(-1, 2),
                source.decoder,
                quantization=source.quantization,
            )
            decoded = _reshape_decoded(sample.material, tuple(uv.shape[:-1]))
            material = decoded_to_material(geometry, decoded)
            deployment_order = [
                "unorm8_quantize_corner_texels",
                "four_point_corner_fetches_lod0_wrap",
                "shared_decoder_per_corner",
                "core4_postprocess_per_corner",
                "bilinear_filter_material_semantics",
                "normalize_tangent_normal_once",
                "shared_tbn_ggx_pbr",
                "display_transform",
            ]
            source_metadata: dict[str, Any] = {
                "latent_channels": source.decoder.latent_channels,
                "decoder_width": source.decoder.width,
                "activation": source.decoder.activation,
                "quantization": source.quantization,
                "lod": 0,
                "mipmaps": False,
                "address_mode": "wrap",
            }
        elif isinstance(source, DTFReferenceMaterialSource):
            material = source.material
            deployment_order = [
                "reference_core4_material",
                "shared_tbn_ggx_pbr",
                "display_transform",
            ]
            source_metadata = {"quantization": "not_applicable"}
        else:
            raise TypeError(f"unsupported DTF material source: {type(source)!r}")
        linear_hdr = shade_ggx(
            geometry,
            camera,
            light,
            material_override=material,
            minimum_roughness=self.minimum_roughness,
        )
        display_rgb = display_transform(linear_hdr, self.display_exposure)
        return DTFRenderBundle(
            renderer_identifier=self.renderer_identifier,
            coverage=buffers["mask"],
            uv=uv,
            geometry_buffers=dict(buffers),
            material=material,
            linear_hdr=linear_hdr,
            display_rgb=display_rgb,
            metadata={
                "schema_version": 1,
                "renderer_version": self.renderer_identifier,
                "deployment_order": deployment_order,
                **source_metadata,
                "input_hashes": dict(input_hashes),
                "minimum_roughness": self.minimum_roughness,
                "display_exposure": self.display_exposure,
            },
        )
