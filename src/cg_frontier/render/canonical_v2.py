"""Canonical deployment-order renderer for filter-aware material compression."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Callable, Mapping

import torch

from cg_frontier.compression.deployment_parity import (
    DeploymentParityDecoder,
    deployment_parity_sample,
)
from cg_frontier.compression.material import DecodedMaterial
from cg_frontier.compression.render_loss import decoded_to_material, display_transform
from cg_frontier.render.gbuffer import Camera, GBufferResult, MaterialBuffers
from cg_frontier.render.pbr import PointLight, shade_ggx


@dataclass(frozen=True)
class LatentMaterialSource:
    """A stored RGBA latent and its per-pixel deployment decoder."""

    latent_rgba: torch.Tensor
    decoder: DeploymentParityDecoder
    quantization: str = "hard"


@dataclass(frozen=True)
class ReferenceMaterialSource:
    """An already sampled Core-4 reference material on the shared geometry."""

    material: MaterialBuffers


@dataclass(frozen=True)
class RenderBundle:
    """All differentiable products shared by training and evaluation."""

    renderer_identifier: str
    coverage: torch.Tensor
    uv: torch.Tensor
    geometry_buffers: Mapping[str, torch.Tensor]
    material: MaterialBuffers
    linear_hdr: torch.Tensor
    display_rgb: torch.Tensor
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class RendererParityReport:
    """Frozen renderer-v2 takeover metrics and their aggregate decision."""

    passed: bool
    metrics: Mapping[str, float | bool]
    thresholds: Mapping[str, float]


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


class CanonicalRendererV2:
    """Render the exact quantize/filter/decode/postprocess deployment contract."""

    renderer_identifier = "canonical_renderer_v2"

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
        source: LatentMaterialSource | ReferenceMaterialSource,
        *,
        input_hashes: Mapping[str, str],
    ) -> RenderBundle:
        buffers = geometry.torch_buffers
        missing = sorted({"mask", "uv"}.difference(buffers))
        if missing:
            raise ValueError(f"geometry GBuffer is missing: {', '.join(missing)}")
        uv = buffers["uv"]
        if uv.ndim != 3 or uv.shape[-1] != 2:
            raise ValueError("canonical renderer requires an HxWx2 UV buffer")
        if isinstance(source, LatentMaterialSource):
            sample = deployment_parity_sample(
                source.latent_rgba,
                uv.reshape(-1, 2),
                source.decoder,
                quantization=source.quantization,
            )
            material = decoded_to_material(
                geometry,
                _reshape_decoded(sample.runtime, tuple(uv.shape[:-1])),
            )
            deployment_order = [
                "rgba8_quantize_texels",
                "single_bilinear_latent_sample",
                "relu_decoder_4_to_8_to_7",
                "core4_postprocess_once",
                "shared_ggx_pbr",
                "display_transform",
            ]
            quantization = source.quantization
        elif isinstance(source, ReferenceMaterialSource):
            material = source.material
            deployment_order = [
                "reference_core4_bilinear",
                "shared_ggx_pbr",
                "display_transform",
            ]
            quantization = "not_applicable"
        else:
            raise TypeError(f"unsupported canonical material source: {type(source)!r}")
        linear_hdr = shade_ggx(
            geometry,
            camera,
            light,
            material_override=material,
            minimum_roughness=self.minimum_roughness,
        )
        display_rgb = display_transform(linear_hdr, self.display_exposure)
        return RenderBundle(
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
                "input_hashes": dict(input_hashes),
                "quantization": quantization,
                "minimum_roughness": self.minimum_roughness,
                "display_exposure": self.display_exposure,
            },
        )


_PARITY_THRESHOLDS = {
    "uv_max_abs": 2.0e-6,
    "normal_p99_degrees": 0.01,
    "core4_p99_abs": 1.0e-5,
    "linear_hdr_mae": 1.0e-4,
    "linear_hdr_p99_abs": 1.0e-3,
    "display_ssim_minimum": 0.9999,
}


def _p99(values: torch.Tensor) -> float:
    return float(torch.quantile(values.reshape(-1), 0.99).detach().cpu())


def _display_ssim(reference: torch.Tensor, candidate: torch.Tensor) -> float:
    x = reference.reshape(-1)
    y = candidate.reshape(-1)
    c1, c2 = 0.01**2, 0.03**2
    mean_x, mean_y = x.mean(), y.mean()
    variance_x = ((x - mean_x) ** 2).mean()
    variance_y = ((y - mean_y) ** 2).mean()
    covariance = ((x - mean_x) * (y - mean_y)).mean()
    value = ((2 * mean_x * mean_y + c1) * (2 * covariance + c2)) / (
        (mean_x.square() + mean_y.square() + c1)
        * (variance_x + variance_y + c2)
    )
    return float(value.detach().cpu())


def compare_render_bundles(
    reference: RenderBundle,
    candidate: RenderBundle,
) -> RendererParityReport:
    """Evaluate the immutable forward-parity portion of the v2 takeover gate."""

    coverage_exact = torch.equal(reference.coverage, candidate.coverage)
    if reference.uv.shape != candidate.uv.shape:
        raise ValueError("renderer parity UV shapes differ")
    mask = reference.coverage
    if mask.shape != reference.uv.shape[:-1] or not torch.any(mask):
        raise ValueError("renderer parity requires non-empty matching coverage")
    uv_max = float((reference.uv - candidate.uv).abs().max().detach().cpu())
    reference_normal = torch.nn.functional.normalize(
        reference.material.normal_world[mask], dim=-1, eps=1.0e-8
    )
    candidate_normal = torch.nn.functional.normalize(
        candidate.material.normal_world[mask], dim=-1, eps=1.0e-8
    )
    normal_dot = torch.sum(reference_normal * candidate_normal, dim=-1).clamp(-1.0, 1.0)
    normal_cross = torch.linalg.cross(reference_normal, candidate_normal, dim=-1)
    normal_angle = torch.rad2deg(
        torch.atan2(torch.linalg.vector_norm(normal_cross, dim=-1), normal_dot)
    )
    material_errors = torch.cat(
        [
            (reference.material.base_color_linear[mask] - candidate.material.base_color_linear[mask]).abs().reshape(-1),
            normal_angle.reshape(-1) / 180.0,
            (reference.material.roughness[mask] - candidate.material.roughness[mask]).abs().reshape(-1),
            (reference.material.metallic[mask] - candidate.material.metallic[mask]).abs().reshape(-1),
        ]
    )
    hdr_error = (reference.linear_hdr[mask] - candidate.linear_hdr[mask]).abs()
    metrics: dict[str, float | bool] = {
        "coverage_exact": coverage_exact,
        "uv_max_abs": uv_max,
        "normal_p99_degrees": _p99(normal_angle),
        "core4_p99_abs": _p99(material_errors),
        "linear_hdr_mae": float(hdr_error.mean().detach().cpu()),
        "linear_hdr_p99_abs": _p99(hdr_error),
        "display_ssim": _display_ssim(
            reference.display_rgb[mask], candidate.display_rgb[mask]
        ),
    }
    passed = bool(
        coverage_exact
        and metrics["uv_max_abs"] <= _PARITY_THRESHOLDS["uv_max_abs"]
        and metrics["normal_p99_degrees"] <= _PARITY_THRESHOLDS["normal_p99_degrees"]
        and metrics["core4_p99_abs"] <= _PARITY_THRESHOLDS["core4_p99_abs"]
        and metrics["linear_hdr_mae"] <= _PARITY_THRESHOLDS["linear_hdr_mae"]
        and metrics["linear_hdr_p99_abs"] <= _PARITY_THRESHOLDS["linear_hdr_p99_abs"]
        and metrics["display_ssim"] >= _PARITY_THRESHOLDS["display_ssim_minimum"]
    )
    return RendererParityReport(passed, metrics, dict(_PARITY_THRESHOLDS))


def parity_report_json(report: RendererParityReport) -> bytes:
    """Serialize parity evidence with stable ordering and line endings."""

    payload = {
        "schema_version": 1,
        "renderer": CanonicalRendererV2.renderer_identifier,
        "passed": report.passed,
        "metrics": dict(report.metrics),
        "thresholds": dict(report.thresholds),
    }
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def compare_gradient_probes(
    reference: Callable[[torch.Tensor], torch.Tensor],
    candidate: Callable[[torch.Tensor], torch.Tensor],
    probes: torch.Tensor,
) -> dict[str, float | int | bool]:
    """Compare deterministic per-probe input gradients for renderer takeover."""

    if probes.ndim != 2 or probes.shape[0] != 32:
        raise ValueError("gradient parity requires exactly 32 row probes")
    cosine_values: list[float] = []
    relative_values: list[float] = []
    all_finite = True
    for row in probes:
        left_input = row.detach().clone().requires_grad_(True)
        right_input = row.detach().clone().requires_grad_(True)
        left_output = reference(left_input[None, :])
        right_output = candidate(right_input[None, :])
        if left_output.shape != right_output.shape or left_output.numel() == 0:
            raise ValueError("gradient parity callables must return matching non-empty tensors")
        coefficients = torch.linspace(
            0.5,
            1.5,
            left_output.numel(),
            dtype=left_output.dtype,
            device=left_output.device,
        ).reshape(left_output.shape)
        left_gradient = torch.autograd.grad((left_output * coefficients).sum(), left_input)[0]
        right_gradient = torch.autograd.grad((right_output * coefficients).sum(), right_input)[0]
        finite = bool(torch.isfinite(left_gradient).all() and torch.isfinite(right_gradient).all())
        all_finite = all_finite and finite
        if not finite:
            cosine_values.append(float("-inf"))
            relative_values.append(float("inf"))
            continue
        left_norm = torch.linalg.vector_norm(left_gradient)
        right_norm = torch.linalg.vector_norm(right_gradient)
        denominator = (left_norm * right_norm).clamp_min(1.0e-12)
        cosine = torch.dot(left_gradient, right_gradient) / denominator
        relative = torch.linalg.vector_norm(right_gradient - left_gradient) / left_norm.clamp_min(1.0e-12)
        cosine_values.append(float(cosine.detach().cpu()))
        relative_values.append(float(relative.detach().cpu()))
    minimum_cosine = min(cosine_values)
    maximum_relative_l2 = max(relative_values)
    return {
        "probe_count": int(probes.shape[0]),
        "all_finite": all_finite,
        "minimum_cosine": minimum_cosine,
        "maximum_relative_l2": maximum_relative_l2,
        "passed": bool(all_finite and minimum_cosine >= 0.999 and maximum_relative_l2 <= 0.01),
    }
