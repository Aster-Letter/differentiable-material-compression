"""Filter-aware 4→8→7 decoder primitives and deterministic artifact metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch import nn

from cg_frontier.compression.artifact_analysis import tail_statistics
from cg_frontier.compression.material import DecodedMaterial, MaterialDecoder, reconstruct_normal


FILTER_AWARE_KINDS = ("f_relu", "f_softplus", "f_sigmoid")
FILTER_AWARE_COST = {
    "parameters": 103,
    "weight_bytes_float32": 412,
    "macs_per_pixel": 88,
}
ACTIVATION_COST = {
    "f_relu": {"hidden_units": 8, "operation_class": "max", "special_functions_per_pixel": 0},
    "f_softplus": {
        "hidden_units": 8,
        "operation_class": "softplus_beta_4_exp_log",
        "special_functions_per_pixel": 16,
    },
    "f_sigmoid": {
        "hidden_units": 8,
        "operation_class": "sigmoid_exp_reciprocal",
        "special_functions_per_pixel": 8,
    },
}


class FilterAwareDecoder(nn.Module):
    """Fixed 4→8→7 decoder whose hidden activation is an explicit candidate."""

    def __init__(self, kind: str) -> None:
        super().__init__()
        if kind not in FILTER_AWARE_KINDS:
            raise ValueError(f"unsupported filter-aware decoder kind: {kind}")
        self.hidden = nn.Linear(4, 8)
        self.output = nn.Linear(8, 7)
        self.kind = kind
        self.module_identifier = f"filter_aware_decoder_v1.{kind}"

    def activate(self, values: torch.Tensor) -> torch.Tensor:
        if self.kind == "f_relu":
            return F.relu(values)
        if self.kind == "f_softplus":
            return F.softplus(values, beta=4.0)
        return torch.sigmoid(values)

    def forward(self, latent_rgba: torch.Tensor) -> torch.Tensor:
        return self.output(self.activate(self.hidden(latent_rgba)))

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def macs_per_pixel(self) -> int:
        return 88

    @property
    def activation_cost(self) -> dict[str, Any]:
        return dict(ACTIVATION_COST[self.kind])


def initialize_filter_aware_from_tiny(
    source: MaterialDecoder, kind: str
) -> FilterAwareDecoder:
    """Copy the frozen pre-QAT affine layers into an independent candidate."""

    if source.kind != "tiny_mlp":
        raise ValueError("filter-aware initialization requires the frozen tiny MLP")
    hidden, output = source.network[0], source.network[2]
    if not isinstance(hidden, nn.Linear) or not isinstance(output, nn.Linear):
        raise TypeError("unexpected frozen tiny-MLP layout")
    first = next(source.parameters())
    candidate = FilterAwareDecoder(kind).to(device=first.device, dtype=first.dtype)
    with torch.no_grad():
        candidate.hidden.weight.copy_(hidden.weight)
        candidate.hidden.bias.copy_(hidden.bias)
        candidate.output.weight.copy_(output.weight)
        candidate.output.bias.copy_(output.bias)
    return candidate


def calculate_filter_aware_cost(decoder: FilterAwareDecoder) -> dict[str, Any]:
    """Calculate neural and activation costs from the instantiated module."""

    result = {
        "module_identifier": decoder.module_identifier,
        "shape": "4->8->7",
        "parameters": decoder.parameter_count,
        "weight_bytes_float32": sum(
            parameter.numel() * parameter.element_size() for parameter in decoder.parameters()
        ),
        "macs_per_pixel": decoder.macs_per_pixel,
        "activation": decoder.activation_cost,
    }
    for key, expected in FILTER_AWARE_COST.items():
        if result[key] != expected:
            raise ValueError(f"filter-aware cost mismatch for {key}: {result[key]} != {expected}")
    return result


def postprocess_raw_torch(raw: torch.Tensor) -> DecodedMaterial:
    """Apply the frozen seven-channel postprocess exactly once."""

    base_color = torch.sigmoid(raw[..., 0:3])
    normal_xy = torch.tanh(raw[..., 3:5])
    normal_xyz = reconstruct_normal(normal_xy)
    roughness = torch.sigmoid(raw[..., 5:6])
    metallic = torch.sigmoid(raw[..., 6:7])
    return DecodedMaterial(base_color, normal_xy, normal_xyz, roughness, metallic)


def bilinear_corners_top_down_wrap_torch(
    texture: torch.Tensor, uv: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather HWC texture corners and weights under the frozen sampler contract."""

    if texture.ndim != 3 or uv.ndim != 2 or uv.shape[-1] != 2:
        raise ValueError("texture must be HWC and uv must be Nx2")
    height, width = texture.shape[:2]
    x = uv[:, 0] * float(width) - 0.5
    y = uv[:, 1] * float(height) - 0.5
    x0_floor, y0_floor = torch.floor(x), torch.floor(y)
    wx, wy = x - x0_floor, y - y0_floor
    x0 = x0_floor.to(torch.int64).remainder(width)
    y0 = y0_floor.to(torch.int64).remainder(height)
    x1, y1 = (x0 + 1).remainder(width), (y0 + 1).remainder(height)
    corners = torch.stack(
        (texture[y0, x0], texture[y0, x1], texture[y1, x0], texture[y1, x1]), dim=1
    )
    weights = torch.stack(((1 - wx) * (1 - wy), wx * (1 - wy), (1 - wx) * wy, wx * wy), dim=1)
    return corners, weights


def _weighted(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.sum(values * weights[..., None], dim=1)


@dataclass(frozen=True)
class CommutativityResult:
    loss: torch.Tensor
    runtime: DecodedMaterial
    filtered: DecodedMaterial


def postprocess_commutativity_loss(
    decoder: FilterAwareDecoder, latent_texture: torch.Tensor, uv: torch.Tensor
) -> CommutativityResult:
    """Compare postprocess(decode(bilinear latent)) with filtered corner materials."""

    corners, weights = bilinear_corners_top_down_wrap_torch(latent_texture, uv)
    sampled = _weighted(corners, weights)
    runtime = postprocess_raw_torch(decoder(sampled))
    corner_material = postprocess_raw_torch(decoder(corners.reshape(-1, 4)))
    count = corners.shape[0]

    def reshape(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(count, 4, value.shape[-1])

    filtered = DecodedMaterial(
        _weighted(reshape(corner_material.base_color_linear), weights),
        _weighted(reshape(corner_material.normal_xy), weights),
        _weighted(reshape(corner_material.normal_xyz), weights),
        _weighted(reshape(corner_material.roughness), weights),
        _weighted(reshape(corner_material.metallic), weights),
    )
    loss = (
        F.l1_loss(runtime.base_color_linear, filtered.base_color_linear)
        + F.l1_loss(runtime.normal_xyz, filtered.normal_xyz)
        + F.l1_loss(runtime.roughness, filtered.roughness)
        + F.l1_loss(runtime.metallic, filtered.metallic)
    )
    return CommutativityResult(loss=loss, runtime=runtime, filtered=filtered)


def component_rectangularity(
    mask: np.ndarray, *, rectangularity_threshold: float = 0.65
) -> dict[str, Any]:
    """Report 4-connected areas and area/bounding-box rectangularity."""

    selection = np.asarray(mask, dtype=bool)
    if selection.ndim != 2 or not (0.0 < rectangularity_threshold <= 1.0):
        raise ValueError("component mask/threshold is invalid")
    labels, count = ndimage.label(
        selection, structure=np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    )
    if count == 0:
        return {
            "component_count": 0,
            "total_area": 0,
            "max_area": 0,
            "max_area_rectangularity": 0.0,
            "rectangularity_p99": 0.0,
            "rectangular_component_max_area": 0,
            "rectangularity_threshold": rectangularity_threshold,
        }
    areas = np.bincount(labels.reshape(-1), minlength=count + 1)[1:]
    rectangles = []
    for label_id, bounds in enumerate(ndimage.find_objects(labels), start=1):
        if bounds is None:
            rectangles.append(0.0)
            continue
        box_area = int(np.prod([item.stop - item.start for item in bounds]))
        rectangles.append(float(areas[label_id - 1] / box_area) if box_area else 0.0)
    rectangularity = np.asarray(rectangles, dtype=np.float64)
    largest = int(np.argmax(areas))
    rectangular_areas = areas[rectangularity >= rectangularity_threshold]
    return {
        "component_count": int(count),
        "total_area": int(areas.sum()),
        "max_area": int(areas.max()),
        "max_area_rectangularity": float(rectangularity[largest]),
        "rectangularity_p99": float(np.percentile(rectangularity, 99.0)),
        "rectangular_component_max_area": int(rectangular_areas.max()) if rectangular_areas.size else 0,
        "rectangularity_threshold": rectangularity_threshold,
    }


def dilate_mask(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Deterministically dilate a 2D mask with a square footprint."""

    if radius < 0:
        raise ValueError("dilation radius must be non-negative")
    return ndimage.binary_dilation(
        np.asarray(mask, dtype=bool), structure=np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    )


def halo_statistics(
    reference_luminance: np.ndarray,
    candidate_luminance: np.ndarray,
    boundary_band: np.ndarray,
    *,
    threshold: float = 0.02,
) -> dict[str, Any]:
    """Summarize signed boundary-band luminance error and two-sided halos."""

    reference = np.asarray(reference_luminance, dtype=np.float32).reshape(-1)
    candidate = np.asarray(candidate_luminance, dtype=np.float32).reshape(-1)
    band = np.asarray(boundary_band, dtype=bool).reshape(-1)
    if reference.shape != candidate.shape or band.shape != reference.shape or not np.any(band):
        raise ValueError("halo arrays must match and boundary band must be non-empty")
    signed = candidate[band] - reference[band]
    positive = np.maximum(signed, 0.0)
    negative = np.maximum(-signed, 0.0)
    return {
        "count": int(signed.size),
        "signed_mean": float(signed.mean(dtype=np.float64)),
        "signed_p01": float(np.percentile(signed, 1.0)),
        "signed_p99": float(np.percentile(signed, 99.0)),
        "absolute_p99": float(np.percentile(np.abs(signed), 99.0)),
        "positive_fraction": float(np.mean(signed > threshold)),
        "positive_p99": float(np.percentile(positive, 99.0)),
        "negative_fraction": float(np.mean(signed < -threshold)),
        "negative_p99": float(np.percentile(negative, 99.0)),
        "threshold": threshold,
    }


def commutativity_statistics(
    runtime: Mapping[str, np.ndarray], filtered: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    """Summarize postprocessed runtime-versus-corner-filter divergence."""

    base = np.max(
        np.abs(np.asarray(runtime["base_color_linear"]) - np.asarray(filtered["base_color_linear"])), axis=-1
    )
    normal_a = np.asarray(runtime["normal_xyz"], dtype=np.float32)
    normal_b = np.asarray(filtered["normal_xyz"], dtype=np.float32)
    normal_a /= np.maximum(np.linalg.norm(normal_a, axis=-1, keepdims=True), 1.0e-8)
    normal_b /= np.maximum(np.linalg.norm(normal_b, axis=-1, keepdims=True), 1.0e-8)
    normal = np.rad2deg(np.arccos(np.sum(normal_a * normal_b, axis=-1).clip(-1.0, 1.0))) / 180.0
    roughness = np.abs(np.asarray(runtime["roughness_linear"]) - np.asarray(filtered["roughness_linear"]))
    metallic = np.abs(np.asarray(runtime["metallic_linear"]) - np.asarray(filtered["metallic_linear"]))
    composite = np.maximum.reduce((base, normal, roughness, metallic))
    return {
        "base_color_max_channel": tail_statistics(base),
        "normal_fraction_of_180": tail_statistics(normal),
        "roughness_l1": tail_statistics(roughness),
        "metallic_l1": tail_statistics(metallic),
        "composite_max": tail_statistics(composite),
    }


def replacement_oracle_statistics(
    reference: Mapping[str, np.ndarray], runtime: Mapping[str, np.ndarray]
) -> dict[str, Any]:
    """Replace one runtime semantic at a time and report material-error reduction."""

    channels = ("base_color_linear", "normal_xyz", "roughness_linear", "metallic_linear")

    def score(material: Mapping[str, np.ndarray]) -> tuple[float, float]:
        base = np.max(
            np.abs(np.asarray(material["base_color_linear"]) - np.asarray(reference["base_color_linear"])), axis=-1
        )
        a = np.asarray(material["normal_xyz"], dtype=np.float32)
        b = np.asarray(reference["normal_xyz"], dtype=np.float32)
        a /= np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1.0e-8)
        b /= np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1.0e-8)
        normal = np.rad2deg(np.arccos(np.sum(a * b, axis=-1).clip(-1.0, 1.0))) / 180.0
        roughness = np.abs(np.asarray(material["roughness_linear"]) - np.asarray(reference["roughness_linear"]))
        metallic = np.abs(np.asarray(material["metallic_linear"]) - np.asarray(reference["metallic_linear"]))
        composite = np.maximum.reduce((base, normal, roughness, metallic))
        return float(composite.mean(dtype=np.float64)), float(np.percentile(composite, 99.0))

    baseline_mean, baseline_p99 = score(runtime)
    result: dict[str, Any] = {
        "runtime": {"composite_mean": baseline_mean, "composite_p99": baseline_p99},
        "replacements": {},
    }
    for channel in channels:
        hybrid = {name: np.asarray(runtime[name]) for name in channels}
        hybrid[channel] = np.asarray(reference[channel])
        mean, p99 = score(hybrid)
        result["replacements"][channel] = {
            "composite_mean": mean,
            "composite_p99": p99,
            "mean_reduction": (baseline_mean - mean) / baseline_mean if baseline_mean else 0.0,
            "p99_reduction": (baseline_p99 - p99) / baseline_p99 if baseline_p99 else 0.0,
        }
    return result
