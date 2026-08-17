"""Minimal differentiable Cook-Torrance GGX reference shading."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as torch_functional
import yaml
from PIL import Image

from cg_frontier.assets.preprocess import sha256_file
from cg_frontier.render.gbuffer import (
    Camera,
    GBufferResult,
    MaterialBuffers,
    material_from_gbuffer,
    srgb_to_linear_torch,
)


@dataclass(frozen=True)
class PointLight:
    position: tuple[float, float, float]
    color: tuple[float, float, float]
    radiant_intensity: float
    ambient_intensity: float


@dataclass(frozen=True)
class ReferenceResult:
    images: Mapping[str, torch.Tensor]
    metadata: Mapping[str, Any]


def linear_to_srgb_torch(linear: torch.Tensor) -> torch.Tensor:
    """Encode non-negative linear RGB for display only."""

    linear = torch.clamp(linear, min=0.0)
    return torch.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * torch.pow(linear, 1.0 / 2.4) - 0.055,
    )


def _ggx_distribution(n_dot_h: torch.Tensor, alpha_squared: torch.Tensor) -> torch.Tensor:
    """Evaluate GGX NDF with a finite denominator at grazing/low-roughness limits."""

    denominator = n_dot_h.square() * (alpha_squared - 1.0) + 1.0
    return alpha_squared / (math.pi * denominator.square().clamp_min(1e-8))


def _smith_g1(n_dot_x: torch.Tensor, alpha_squared: torch.Tensor) -> torch.Tensor:
    """Evaluate one Smith masking term with guarded square root and division."""

    root = torch.sqrt(
        (alpha_squared + (1.0 - alpha_squared) * n_dot_x.square()).clamp_min(1e-8)
    )
    return (2.0 * n_dot_x) / (n_dot_x + root).clamp_min(1e-8)


def shade_ggx(
    gbuffer: GBufferResult,
    camera: Camera,
    light: PointLight,
    *,
    normal_buffer: str = "normal_world",
    base_color_override: torch.Tensor | None = None,
    material_override: MaterialBuffers | None = None,
    minimum_roughness: float = 0.045,
) -> torch.Tensor:
    """Shade shared geometry with either its legacy material or one complete override."""

    buffers = gbuffer.torch_buffers
    required = {"mask", "position_world"}
    missing = sorted(required.difference(buffers))
    if missing:
        raise ValueError(f"GBuffer is missing PBR inputs: {', '.join(missing)}")
    if not (0.0 < minimum_roughness <= 1.0):
        raise ValueError("minimum roughness must be within (0, 1]")

    position = buffers["position_world"]
    if material_override is not None and base_color_override is not None:
        raise ValueError("base_color_override cannot be combined with a complete material_override")
    material = material_from_gbuffer(gbuffer) if material_override is None else material_override
    normal_source = material.normal_world
    if normal_buffer == "normal_world_y_flipped":
        if material.normal_world_y_flipped is None:
            raise ValueError("complete material override has no normal_world_y_flipped diagnostic")
        normal_source = material.normal_world_y_flipped
    elif normal_buffer != "normal_world":
        raise ValueError(f"unsupported material normal buffer: {normal_buffer}")
    normal = torch_functional.normalize(normal_source, dim=-1, eps=1e-8)
    base_color = material.base_color_linear if base_color_override is None else base_color_override
    # The floor bounds the alpha^4 GGX denominator; it is a shading-only clamp
    # and does not change the stored or decoded roughness target.
    roughness = material.roughness[..., None].clamp(min=minimum_roughness, max=1.0)
    metallic = material.metallic[..., None].clamp(0.0, 1.0)
    device = position.device
    camera_position = torch.tensor(camera.eye, dtype=position.dtype, device=device)
    light_position = torch.tensor(light.position, dtype=position.dtype, device=device)
    light_color = torch.tensor(light.color, dtype=position.dtype, device=device)
    if light.radiant_intensity <= 0.0 or light.ambient_intensity < 0.0:
        raise ValueError("point-light intensities must be non-negative")

    to_light = light_position - position
    distance_squared = torch.sum(to_light.square(), dim=-1, keepdim=True).clamp_min(1e-6)
    light_direction = torch_functional.normalize(to_light, dim=-1, eps=1e-8)
    view_direction = torch_functional.normalize(
        camera_position - position, dim=-1, eps=1e-8
    )
    half_vector = torch_functional.normalize(
        light_direction + view_direction, dim=-1, eps=1e-8
    )

    n_dot_l = torch.sum(normal * light_direction, dim=-1, keepdim=True).clamp(0.0, 1.0)
    n_dot_v = torch.sum(normal * view_direction, dim=-1, keepdim=True).clamp(0.0, 1.0)
    n_dot_h = torch.sum(normal * half_vector, dim=-1, keepdim=True).clamp(0.0, 1.0)
    v_dot_h = torch.sum(view_direction * half_vector, dim=-1, keepdim=True).clamp(0.0, 1.0)

    alpha = roughness.square()
    alpha_squared = alpha.square()
    distribution = _ggx_distribution(n_dot_h, alpha_squared)
    geometry = _smith_g1(n_dot_l, alpha_squared) * _smith_g1(
        n_dot_v, alpha_squared
    )
    dielectric_f0 = torch.full_like(base_color, 0.04)
    f0 = dielectric_f0 * (1.0 - metallic) + base_color * metallic
    fresnel = f0 + (1.0 - f0) * torch.pow(1.0 - v_dot_h, 5.0)
    # Clamp the projected-area denominator for back/grazing configurations.
    # n_dot_l still multiplies the final direct term, preserving zero backlight.
    specular = (
        distribution
        * geometry
        * fresnel
        / (4.0 * n_dot_l * n_dot_v).clamp_min(1e-6)
    )
    diffuse_weight = (1.0 - fresnel) * (1.0 - metallic)
    diffuse = diffuse_weight * base_color / math.pi
    incoming_radiance = (
        light_color * float(light.radiant_intensity) / distance_squared
    )
    direct = (diffuse + specular) * incoming_radiance * n_dot_l
    ambient = base_color * (1.0 - metallic) * float(light.ambient_intensity)
    color = direct + ambient
    return torch.where(
        buffers["mask"][..., None], color, torch.zeros_like(color)
    )


def render_reference_variants(
    gbuffer: GBufferResult,
    camera: Camera,
    light: PointLight,
    *,
    material: MaterialBuffers | None = None,
    minimum_roughness: float = 0.045,
) -> ReferenceResult:
    """Render the correct path plus fixed normal-Y and double-sRGB error controls."""

    source = material_from_gbuffer(gbuffer) if material is None else material
    correct = shade_ggx(
        gbuffer,
        camera,
        light,
        material_override=source,
        minimum_roughness=minimum_roughness,
    )
    normal_y_flipped = shade_ggx(
        gbuffer,
        camera,
        light,
        normal_buffer="normal_world_y_flipped",
        material_override=source,
        minimum_roughness=minimum_roughness,
    )
    double_decoded_base = srgb_to_linear_torch(source.base_color_linear)
    double_decoded_material = MaterialBuffers(
        base_color_linear=double_decoded_base,
        normal_world=source.normal_world,
        roughness=source.roughness,
        metallic=source.metallic,
        normal_ts_raw=source.normal_ts_raw,
        normal_ts_unit=source.normal_ts_unit,
        normal_world_y_flipped=source.normal_world_y_flipped,
        normal_y_flip_angle_degrees=source.normal_y_flip_angle_degrees,
    )
    base_color_double_decoded = shade_ggx(
        gbuffer,
        camera,
        light,
        material_override=double_decoded_material,
        minimum_roughness=minimum_roughness,
    )
    mask = gbuffer.torch_buffers["mask"]

    def error_statistics(candidate: torch.Tensor) -> dict[str, float]:
        difference = candidate[mask] - correct[mask]
        absolute = torch.abs(difference)
        return {
            "mae_linear": float(absolute.mean().detach().cpu()),
            "max_abs_linear": float(absolute.max().detach().cpu()),
            "rmse_linear": float(torch.sqrt(torch.mean(difference.square())).detach().cpu()),
        }

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "brdf": {
            "model": "Cook-Torrance GGX metallic-roughness",
            "normal_distribution": "Trowbridge-Reitz GGX",
            "geometry": "Smith GGX",
            "fresnel": "Schlick",
            "dielectric_f0": 0.04,
            "minimum_roughness_for_brdf": minimum_roughness,
        },
        "light": {
            "type": "point",
            "position": list(light.position),
            "color": list(light.color),
            "radiant_intensity": light.radiant_intensity,
            "inverse_square_falloff": True,
            "ambient_intensity": light.ambient_intensity,
        },
        "camera": {
            "eye": list(camera.eye),
            "target": list(camera.target),
            "up": list(camera.up),
            "vertical_fov_degrees": camera.vertical_fov_degrees,
            "near": camera.near,
            "far": camera.far,
        },
        "variants": {
            "reference": "single BaseColor sRGB decode; glTF +Y tangent normal",
            "normal_y_flipped": "diagnostic error: negate tangent-space normal Y",
            "base_color_double_decoded": "diagnostic error: apply sRGB decode twice",
        },
        "variant_error": {
            "normal_y_flipped": error_statistics(normal_y_flipped),
            "base_color_double_decoded": error_statistics(base_color_double_decoded),
        },
    }
    return ReferenceResult(
        images={
            "reference_linear": correct,
            "normal_y_flipped_linear": normal_y_flipped,
            "base_color_double_decoded_linear": base_color_double_decoded,
        },
        metadata=metadata,
    )


def _display_encode(linear: torch.Tensor, exposure: float) -> np.ndarray:
    if exposure <= 0.0:
        raise ValueError("display exposure must be positive")
    mapped = linear * exposure
    mapped = mapped / (1.0 + mapped)
    encoded = linear_to_srgb_torch(mapped).clamp(0.0, 1.0)
    return np.rint(encoded.detach().cpu().numpy() * 255.0).astype(np.uint8)


def export_reference(
    result: ReferenceResult,
    output_dir: Path | str,
    *,
    display_exposure: float,
) -> dict[str, Any]:
    """Save linear HDR arrays, display PNGs, error maps, and deterministic metadata."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}
    display_images: dict[str, np.ndarray] = {}
    for name, tensor in result.images.items():
        array = tensor.detach().cpu().numpy().astype(np.float32)
        npy_path = output_dir / f"{name}.npy"
        np.save(npy_path, array, allow_pickle=False)
        files[npy_path.name] = {
            "sha256": sha256_file(npy_path),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }
        display_name = name.removesuffix("_linear")
        display = _display_encode(tensor, display_exposure)
        png_path = output_dir / f"{display_name}.png"
        Image.fromarray(display).save(png_path, format="PNG")
        files[png_path.name] = {"sha256": sha256_file(png_path)}
        display_images[display_name] = display

    reference = result.images["reference_linear"]
    error_sources = {
        "normal_y_abs_difference": result.images["normal_y_flipped_linear"],
        "base_color_double_decode_abs_difference": result.images[
            "base_color_double_decoded_linear"
        ],
    }
    for name, candidate in error_sources.items():
        difference = torch.abs(candidate - reference)
        visualization = torch.clamp(difference * 4.0, 0.0, 1.0)
        encoded = np.rint(visualization.detach().cpu().numpy() * 255.0).astype(np.uint8)
        path = output_dir / f"{name}.png"
        Image.fromarray(encoded).save(path, format="PNG")
        files[path.name] = {"sha256": sha256_file(path), "display_gain": 4.0}

    panel = np.concatenate(
        [
            display_images["reference"],
            display_images["normal_y_flipped"],
            display_images["base_color_double_decoded"],
        ],
        axis=1,
    )
    panel_path = output_dir / "comparison_reference__normal_y_flip__double_srgb.png"
    Image.fromarray(panel).save(panel_path, format="PNG")
    files[panel_path.name] = {
        "sha256": sha256_file(panel_path),
        "order": ["reference", "normal_y_flipped", "base_color_double_decoded"],
    }

    metadata = dict(result.metadata)
    metadata["display"] = {
        "exposure": display_exposure,
        "tone_mapping": "Reinhard x/(1+x)",
        "encoding": "linear_to_sRGB",
    }
    metadata["files"] = files
    metadata_path = output_dir / "reference.yaml"
    text = yaml.safe_dump(
        metadata, sort_keys=False, allow_unicode=True, default_flow_style=False
    )
    with metadata_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    return metadata
