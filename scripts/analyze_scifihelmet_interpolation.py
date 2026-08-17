"""Reproduce SciFiHelmet subpixel interpolation defects without GPU or holdout data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cg_frontier.compression.artifact_analysis import (  # noqa: E402
    connected_patch_statistics,
    deterministic_json,
    sha256_file,
)
from cg_frontier.compression.interpolation_analysis import (  # noqa: E402
    LUMINANCE_WEIGHTS,
    activation_crossings,
    bilinear_footprint_top_down_wrap,
    dark_fraction_counts,
    error_tail,
    fraction_report,
    normal_angular_error,
    normalize_vectors,
    phase0_manifest,
    sample_float_top_down_wrap,
    subpixel_boundary_mask,
)
from cg_frontier.compression.material import load_core4_targets  # noqa: E402
from cg_frontier.compression.render_loss import (  # noqa: E402
    bilinear_sample_top_down_wrap as torch_bilinear_sample,
)
from cg_frontier.compression.ue_export import (  # noqa: E402
    bilinear_sample_top_down_wrap as ue_bilinear_sample,
    decoder_postprocess,
    decoder_raw_forward,
)


def _repo_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"config field {label} must be a non-empty path")
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"config field {label} escapes the repository")
    if "formal_holdout" in path.as_posix().lower():
        raise ValueError(f"sealed formal holdout path is forbidden: {label}")
    return path


def _load_reference(core4_dir: Path) -> dict[str, np.ndarray]:
    targets = load_core4_targets(core4_dir)
    return {
        "base_color_linear": targets.base_color_linear.numpy().reshape(targets.height, targets.width, 3),
        "normal_xyz": targets.normal_xyz.numpy().reshape(targets.height, targets.width, 3),
        "roughness_linear": targets.roughness.numpy().reshape(targets.height, targets.width),
        "metallic_linear": targets.metallic.numpy().reshape(targets.height, targets.width),
    }


def _load_decoder(path: Path) -> dict[str, np.ndarray]:
    tiny = {
        "network.0.weight": (8, 4),
        "network.0.bias": (8,),
        "network.2.weight": (7, 8),
        "network.2.bias": (7,),
    }
    split = {
        "base_affine.weight": (3, 4),
        "base_affine.bias": (3,),
        "aux_hidden.weight": (8, 4),
        "aux_hidden.bias": (8,),
        "aux_output.weight": (4, 8),
        "aux_output.bias": (4,),
    }
    hybrid2 = {
        "network.0.weight": (8, 2),
        "network.0.bias": (8,),
        "network.2.weight": (4, 8),
        "network.2.bias": (4,),
    }
    hybrid3 = {
        "network.0.weight": (8, 3),
        "network.0.bias": (8,),
        "network.2.weight": (4, 8),
        "network.2.bias": (4,),
    }
    d6_h = {
        "normal_head.0.weight": (5, 3), "normal_head.0.bias": (5,),
        "normal_head.2.weight": (2, 5), "normal_head.2.bias": (2,),
        "scalar_head.0.weight": (5, 3), "scalar_head.0.bias": (5,),
        "scalar_head.2.weight": (2, 5), "scalar_head.2.bias": (2,),
    }
    d6_p = {
        "normal_head.0.weight": (6, 2), "normal_head.0.bias": (6,),
        "normal_head.2.weight": (2, 6), "normal_head.2.bias": (2,),
        "scalar_head.0.weight": (8, 1), "scalar_head.0.bias": (8,),
        "scalar_head.2.weight": (2, 8), "scalar_head.2.bias": (2,),
    }
    d7_p = {
        "normal_head.0.weight": (6, 2), "normal_head.0.bias": (6,),
        "normal_head.2.weight": (2, 6), "normal_head.2.bias": (2,),
        "scalar_head.0.weight": (6, 2), "scalar_head.0.bias": (6,),
        "scalar_head.2.weight": (2, 6), "scalar_head.2.bias": (2,),
    }
    d7_direct_scalars = {
        "normal_head.0.weight": (6, 2), "normal_head.0.bias": (6,),
        "normal_head.2.weight": (2, 6), "normal_head.2.bias": (2,),
        "direct_scalars.marker": (1,),
    }
    o7_direct = {"direct_semantic.marker": (1,)}
    with np.load(path, allow_pickle=False) as stored:
        arrays = {name: np.asarray(stored[name], dtype=np.float32).copy() for name in stored.files}
    expected = None
    for schema in (tiny, split, hybrid2, hybrid3, d6_h, d6_p, d7_p, d7_direct_scalars, o7_direct):
        if set(arrays) == set(schema) and all(arrays[name].shape == shape for name, shape in schema.items()):
            expected = schema
            break
    if expected is None:
        raise ValueError(f"decoder arrays differ from supported interpolation schemas: {sorted(arrays)}")
    for name, shape in expected.items():
        if arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
            raise ValueError(f"decoder array {name} is invalid")
    return arrays


def _decoder_raw(latent: np.ndarray, arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Evaluate either supported decoder without invoking optional BLAS."""

    if "network.0.weight" in arrays:
        return decoder_raw_forward(latent, arrays)
    if _is_direct_scalars(arrays):
        values = np.asarray(latent[..., :2], dtype=np.float32)
        hidden = np.maximum(
            np.sum(values[..., None, :] * arrays["normal_head.0.weight"], axis=-1, dtype=np.float32)
            + arrays["normal_head.0.bias"],
            np.float32(0.0),
        )
        return np.sum(
            hidden[..., None, :] * arrays["normal_head.2.weight"], axis=-1, dtype=np.float32
        ) + arrays["normal_head.2.bias"]
    if "normal_head.0.weight" in arrays:
        values = np.asarray(latent, dtype=np.float32)
        normal_width = arrays["normal_head.0.weight"].shape[1]
        scalar_width = arrays["scalar_head.0.weight"].shape[1]
        normal_input = values if normal_width == values.shape[-1] else values[..., :normal_width]
        scalar_input = values if scalar_width == values.shape[-1] else values[..., -scalar_width:]

        def head(prefix: str, inputs: np.ndarray) -> np.ndarray:
            hidden = np.maximum(
                np.sum(inputs[..., None, :] * arrays[f"{prefix}.0.weight"], axis=-1, dtype=np.float32)
                + arrays[f"{prefix}.0.bias"],
                np.float32(0.0),
            )
            return np.sum(
                hidden[..., None, :] * arrays[f"{prefix}.2.weight"], axis=-1, dtype=np.float32
            ) + arrays[f"{prefix}.2.bias"]

        return np.concatenate((head("normal_head", normal_input), head("scalar_head", scalar_input)), axis=-1)
    if "direct_semantic.marker" in arrays:
        raise ValueError("O7-Direct has no raw neural decoder")
    values = np.asarray(latent, dtype=np.float32)
    base = np.sum(
        values[..., None, :] * arrays["base_affine.weight"], axis=-1, dtype=np.float32
    ) + arrays["base_affine.bias"]
    hidden = np.maximum(
        np.sum(values[..., None, :] * arrays["aux_hidden.weight"], axis=-1, dtype=np.float32)
        + arrays["aux_hidden.bias"],
        np.float32(0.0),
    )
    auxiliary = np.sum(
        hidden[..., None, :] * arrays["aux_output.weight"], axis=-1, dtype=np.float32
    ) + arrays["aux_output.bias"]
    return np.concatenate((base, auxiliary), axis=-1).astype(np.float32, copy=False)


def _is_hybrid(arrays: Mapping[str, np.ndarray]) -> bool:
    return (
        ("network.2.weight" in arrays and arrays["network.2.weight"].shape[0] == 4)
        or "normal_head.0.weight" in arrays
        or "direct_semantic.marker" in arrays
    )


def _is_direct_semantic(arrays: Mapping[str, np.ndarray]) -> bool:
    return "direct_semantic.marker" in arrays


def _is_direct_scalars(arrays: Mapping[str, np.ndarray]) -> bool:
    return "direct_scalars.marker" in arrays


def _direct_semantic_postprocess(latent: np.ndarray, direct_base: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(latent, dtype=np.float32)
    normal_xy = values[..., :2] * np.float32(2.0) - np.float32(1.0)
    radius = np.sqrt(np.sum(normal_xy * normal_xy, axis=-1, keepdims=True, dtype=np.float32))
    normal_xy = normal_xy / np.maximum(radius / np.float32(1.0 - 1.0e-6), np.float32(1.0))
    normal_z = np.sqrt(np.maximum(
        np.float32(1.0) - np.sum(normal_xy * normal_xy, axis=-1, keepdims=True, dtype=np.float32),
        np.float32(1.0e-8),
    )).astype(np.float32)
    return {
        "base_color_linear": np.asarray(direct_base, dtype=np.float32),
        "normal_xyz": normalize_vectors(np.concatenate((normal_xy, normal_z), axis=-1)),
        "roughness_linear": values[..., 2],
        "metallic_linear": values[..., 3],
    }


def _direct_scalars_postprocess(
    normal_raw: np.ndarray,
    latent: np.ndarray,
    direct_base: np.ndarray,
) -> dict[str, np.ndarray]:
    values = np.asarray(normal_raw, dtype=np.float32)
    normal_xy = np.tanh(values[..., :2]).astype(np.float32)
    radius = np.sqrt(np.sum(normal_xy * normal_xy, axis=-1, keepdims=True, dtype=np.float32))
    normal_xy = normal_xy / np.maximum(radius / np.float32(1.0 - 1.0e-6), np.float32(1.0))
    normal_z = np.sqrt(np.maximum(
        np.float32(1.0) - np.sum(normal_xy * normal_xy, axis=-1, keepdims=True, dtype=np.float32),
        np.float32(1.0e-8),
    )).astype(np.float32)
    auxiliary = np.asarray(latent, dtype=np.float32)
    return {
        "base_color_linear": np.asarray(direct_base, dtype=np.float32),
        "normal_xyz": normalize_vectors(np.concatenate((normal_xy, normal_z), axis=-1)),
        "roughness_linear": auxiliary[..., 2],
        "metallic_linear": auxiliary[..., 3],
    }


def _hybrid_postprocess(raw: np.ndarray, direct_base: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(raw, dtype=np.float32)
    normal_xy = np.tanh(values[..., 0:2]).astype(np.float32)
    radius = np.sqrt(np.sum(normal_xy * normal_xy, axis=-1, keepdims=True, dtype=np.float32))
    normal_xy = normal_xy / np.maximum(radius / np.float32(1.0 - 1.0e-6), np.float32(1.0))
    normal_z = np.sqrt(
        np.maximum(
            np.float32(1.0) - np.sum(normal_xy * normal_xy, axis=-1, keepdims=True, dtype=np.float32),
            np.float32(1.0e-8),
        )
    ).astype(np.float32)
    normal = normalize_vectors(np.concatenate((normal_xy, normal_z), axis=-1))
    roughness = np.asarray(1.0 / (1.0 + np.exp(-values[..., 2:3])), dtype=np.float32)
    metallic = np.asarray(1.0 / (1.0 + np.exp(-values[..., 3:4])), dtype=np.float32)
    return {
        "base_color_linear": np.asarray(direct_base, dtype=np.float32),
        "normal_xyz": normal,
        "roughness_linear": roughness[..., 0],
        "metallic_linear": metallic[..., 0],
    }


def _decode_texels(
    latent: np.ndarray,
    arrays: dict[str, np.ndarray],
    chunk: int,
    direct_base: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    flat = latent.reshape(-1, latent.shape[-1])
    if _is_hybrid(arrays):
        if direct_base is None:
            raise ValueError("hybrid decoding requires direct BaseColor")
        base_flat = direct_base.reshape(-1, 3)
        hybrid_parts = {name: [] for name in ("base_color_linear", "normal_xyz", "roughness_linear", "metallic_linear")}
        for start in range(0, flat.shape[0], chunk):
            if _is_direct_semantic(arrays):
                post = _direct_semantic_postprocess(
                    flat[start : start + chunk], base_flat[start : start + chunk]
                )
            elif _is_direct_scalars(arrays):
                post = _direct_scalars_postprocess(
                    _decoder_raw(flat[start : start + chunk], arrays),
                    flat[start : start + chunk],
                    base_flat[start : start + chunk],
                )
            else:
                post = _hybrid_postprocess(
                    _decoder_raw(flat[start : start + chunk], arrays),
                    base_flat[start : start + chunk],
                )
            for name in hybrid_parts:
                hybrid_parts[name].append(post[name])
        height, width = latent.shape[:2]
        return {
            name: np.concatenate(values, axis=0).reshape(height, width, -1)
            if name in ("base_color_linear", "normal_xyz")
            else np.concatenate(values, axis=0).reshape(height, width)
            for name, values in hybrid_parts.items()
        }
    parts: dict[str, list[np.ndarray]] = {
        "base_color_linear": [],
        "normal_xyz": [],
        "roughness_linear": [],
        "metallic_linear": [],
    }
    for start in range(0, flat.shape[0], chunk):
        post = decoder_postprocess(_decoder_raw(flat[start : start + chunk], arrays))
        parts["base_color_linear"].append(post["base_color_linear"])
        parts["normal_xyz"].append(post["normal_tangent_gltf_positive_y"])
        parts["roughness_linear"].append(post["roughness_linear"][..., 0])
        parts["metallic_linear"].append(post["metallic_linear"][..., 0])
    height, width = latent.shape[:2]
    return {
        name: np.concatenate(values, axis=0).reshape(height, width, -1)
        if name in ("base_color_linear", "normal_xyz")
        else np.concatenate(values, axis=0).reshape(height, width)
        for name, values in parts.items()
    }


def _add_counts(target: dict[str, dict[str, int]], source: Mapping[str, Mapping[str, int]]) -> None:
    for threshold, values in source.items():
        row = target.setdefault(threshold, {name: 0 for name in values})
        for name, value in values.items():
            row[name] += int(value)


class _ScopeAccumulator:
    def __init__(self, hidden_units: int) -> None:
        self.samples = 0
        self.dark_counts: dict[str, dict[str, int]] = {}
        self.values: dict[str, list[np.ndarray]] = {
            "base_runtime": [],
            "base_filter": [],
            "normal_runtime": [],
            "roughness_runtime": [],
            "metallic_runtime": [],
            "normal_filter_divergence": [],
            "roughness_filter_divergence": [],
            "metallic_filter_divergence": [],
        }
        self.boundary_count = 0
        self.boundary_abs_sum = 0.0
        self.boundary_above_0_1 = 0
        self.crossing_count = np.zeros(hidden_units, dtype=np.int64)
        self.novel_crossing_count = np.zeros(hidden_units, dtype=np.int64)
        self.any_crossing_count = 0
        self.novel_count = 0
        self.novel_any_crossing_count = 0
        self.correlation_count = 0
        self.correlation_sum = np.zeros(4, dtype=np.float64)
        self.correlation_cross = np.zeros((4, 4), dtype=np.float64)

    def ingest(
        self,
        selection: np.ndarray,
        reference: Mapping[str, np.ndarray],
        runtime: Mapping[str, np.ndarray],
        filtered: Mapping[str, np.ndarray],
        boundary: np.ndarray,
        crossings: np.ndarray,
        thresholds: Sequence[float],
        dark_ratio: float,
    ) -> np.ndarray:
        selected = np.asarray(selection, dtype=bool).reshape(-1)
        if not np.any(selected):
            return np.zeros(selected.shape, dtype=bool)
        ref_base = np.asarray(reference["base_color_linear"], dtype=np.float32)
        run_base = np.asarray(runtime["base_color_linear"], dtype=np.float32)
        filter_base = np.asarray(filtered["base_color_linear"], dtype=np.float32)
        # Keep this verifier independent of optional BLAS DLLs. The project
        # Windows runtime can terminate natively (0xc06d007f) when NumPy's
        # matrix-multiply entry point is used, even though elementwise ufuncs
        # are healthy and deterministic.
        ref_luma = np.sum(ref_base * LUMINANCE_WEIGHTS, axis=-1, dtype=np.float32)
        run_luma = np.sum(run_base * LUMINANCE_WEIGHTS, axis=-1, dtype=np.float32)
        filter_luma = np.sum(filter_base * LUMINANCE_WEIGHTS, axis=-1, dtype=np.float32)
        _add_counts(
            self.dark_counts,
            dark_fraction_counts(
                ref_luma[selected], run_luma[selected], filter_luma[selected], thresholds, ratio=dark_ratio
            ),
        )
        base_runtime = np.max(np.abs(run_base - ref_base), axis=-1)
        base_filter = np.max(np.abs(filter_base - ref_base), axis=-1)
        normal_runtime = normal_angular_error(reference["normal_xyz"], runtime["normal_xyz"])
        roughness_runtime = np.abs(
            np.asarray(runtime["roughness_linear"], dtype=np.float32)
            - np.asarray(reference["roughness_linear"], dtype=np.float32)
        )
        metallic_runtime = np.abs(
            np.asarray(runtime["metallic_linear"], dtype=np.float32)
            - np.asarray(reference["metallic_linear"], dtype=np.float32)
        )
        normal_filter_divergence = normal_angular_error(runtime["normal_xyz"], filtered["normal_xyz"])
        roughness_filter_divergence = np.abs(
            np.asarray(runtime["roughness_linear"], dtype=np.float32)
            - np.asarray(filtered["roughness_linear"], dtype=np.float32)
        )
        metallic_filter_divergence = np.abs(
            np.asarray(runtime["metallic_linear"], dtype=np.float32)
            - np.asarray(filtered["metallic_linear"], dtype=np.float32)
        )
        metrics = (
            base_runtime,
            base_filter,
            normal_runtime,
            roughness_runtime,
            metallic_runtime,
            normal_filter_divergence,
            roughness_filter_divergence,
            metallic_filter_divergence,
        )
        for name, values in zip(self.values, metrics, strict=True):
            self.values[name].append(np.asarray(values[selected], dtype=np.float32))
        errors = np.stack(metrics[0:1] + metrics[2:5], axis=-1)[selected].astype(np.float64)
        self.correlation_count += int(errors.shape[0])
        self.correlation_sum += errors.sum(axis=0, dtype=np.float64)
        for left in range(4):
            for right in range(left, 4):
                value = float(np.sum(errors[:, left] * errors[:, right], dtype=np.float64))
                self.correlation_cross[left, right] += value
                if left != right:
                    self.correlation_cross[right, left] += value
        selected_boundary = selected & np.asarray(boundary, dtype=bool).reshape(-1)
        if np.any(selected_boundary):
            edge_values = metallic_runtime[selected_boundary]
            self.boundary_count += int(edge_values.size)
            self.boundary_abs_sum += float(edge_values.sum(dtype=np.float64))
            self.boundary_above_0_1 += int(np.sum(edge_values > 0.1))
        runtime_dark = run_luma < ref_luma * np.float32(dark_ratio)
        filter_dark = filter_luma < ref_luma * np.float32(dark_ratio)
        novel = selected & (ref_luma > np.float32(0.05)) & runtime_dark & ~filter_dark
        selected_crossings = np.asarray(crossings, dtype=bool)[selected]
        selected_novel = novel[selected]
        any_crossing = np.any(selected_crossings, axis=-1)
        self.samples += int(selected.sum())
        self.crossing_count += selected_crossings.sum(axis=0, dtype=np.int64)
        self.novel_crossing_count += (selected_crossings & selected_novel[:, None]).sum(axis=0, dtype=np.int64)
        self.any_crossing_count += int(any_crossing.sum())
        self.novel_count += int(selected_novel.sum())
        self.novel_any_crossing_count += int(np.sum(any_crossing & selected_novel))
        return novel

    def _correlations(self) -> dict[str, float]:
        names = ("base_color", "normal", "roughness", "metallic")
        if self.correlation_count == 0:
            return {f"{a}__{b}": 0.0 for i, a in enumerate(names) for b in names[i + 1 :]}
        count = float(self.correlation_count)
        centered = self.correlation_cross - np.outer(self.correlation_sum, self.correlation_sum) / count
        result: dict[str, float] = {}
        for left in range(4):
            for right in range(left + 1, 4):
                denominator = float(np.sqrt(max(centered[left, left], 0.0) * max(centered[right, right], 0.0)))
                result[f"{names[left]}__{names[right]}"] = (
                    float(centered[left, right] / denominator) if denominator else 0.0
                )
        return result

    def report(self) -> dict[str, Any]:
        novel_count = self.novel_count
        units: dict[str, Any] = {}
        for unit in range(self.crossing_count.size):
            crossing_count = int(self.crossing_count[unit])
            joint = int(self.novel_crossing_count[unit])
            units[str(unit)] = {
                "crossing_count": crossing_count,
                "novel_dark_crossing_count": joint,
                "novel_dark_coverage": float(joint / novel_count) if novel_count else 0.0,
                "conditional_novel_dark_rate": float(joint / crossing_count) if crossing_count else 0.0,
            }
        tails = {name: error_tail(values) for name, values in self.values.items()}
        return {
            "sample_count": self.samples,
            "dark_fractions": fraction_report(self.dark_counts),
            "base_color_max_channel": {
                "runtime": tails["base_runtime"],
                "decode_then_filter": tails["base_filter"],
            },
            "normal_degrees": tails["normal_runtime"],
            "roughness_absolute_error": tails["roughness_runtime"],
            "metallic_absolute_error": tails["metallic_runtime"],
            "filter_divergence": {
                "normal_degrees": tails["normal_filter_divergence"],
                "roughness_l1": tails["roughness_filter_divergence"],
                "metallic_l1": tails["metallic_filter_divergence"],
            },
            "metallic_boundary": {
                "count": self.boundary_count,
                "mae": float(self.boundary_abs_sum / self.boundary_count) if self.boundary_count else 0.0,
                "fraction_above_0_1": float(self.boundary_above_0_1 / self.boundary_count)
                if self.boundary_count
                else 0.0,
            },
            "cross_channel_error_correlation": self._correlations(),
            "activation_crossing": {
                "any_crossing_count": self.any_crossing_count,
                "any_crossing_fraction": float(self.any_crossing_count / self.samples) if self.samples else 0.0,
                "novel_dark_count": novel_count,
                "novel_dark_with_any_crossing_count": self.novel_any_crossing_count,
                "novel_dark_with_any_crossing_fraction": (
                    float(self.novel_any_crossing_count / novel_count) if novel_count else 0.0
                ),
                "units": units,
            },
        }


def _sample_material(texture: Mapping[str, np.ndarray], uv: np.ndarray) -> dict[str, np.ndarray]:
    normal = normalize_vectors(sample_float_top_down_wrap(texture["normal_xyz"], uv))
    return {
        "base_color_linear": sample_float_top_down_wrap(texture["base_color_linear"], uv),
        "normal_xyz": normal,
        "roughness_linear": sample_float_top_down_wrap(texture["roughness_linear"], uv)[..., 0],
        "metallic_linear": sample_float_top_down_wrap(texture["metallic_linear"], uv)[..., 0],
    }


def _runtime_material(
    latent: np.ndarray,
    arrays: dict[str, np.ndarray],
    uv: np.ndarray,
    direct_base: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    sampled = sample_float_top_down_wrap(latent, uv)
    if _is_hybrid(arrays):
        if direct_base is None:
            raise ValueError("hybrid runtime requires direct BaseColor")
        sampled_base = sample_float_top_down_wrap(direct_base, uv)
        if _is_direct_semantic(arrays):
            return _direct_semantic_postprocess(sampled, sampled_base)
        if _is_direct_scalars(arrays):
            return _direct_scalars_postprocess(_decoder_raw(sampled, arrays), sampled, sampled_base)
        return _hybrid_postprocess(_decoder_raw(sampled, arrays), sampled_base)
    post = decoder_postprocess(_decoder_raw(sampled, arrays))
    return {
        "base_color_linear": post["base_color_linear"],
        "normal_xyz": post["normal_tangent_gltf_positive_y"],
        "roughness_linear": post["roughness_linear"][..., 0],
        "metallic_linear": post["metallic_linear"][..., 0],
    }


def _scope_masks(
    scopes: Mapping[str, Mapping[str, Any]],
    base_xy: np.ndarray,
    boundary: np.ndarray,
) -> dict[str, np.ndarray]:
    x, y = base_xy[..., 0], base_xy[..., 1]
    result: dict[str, np.ndarray] = {}
    for name, definition in scopes.items():
        kind = definition["kind"]
        if kind == "all":
            mask = np.ones(x.shape, dtype=bool)
        elif kind == "metallic_boundary":
            mask = np.asarray(boundary, dtype=bool)
        elif kind == "bbox_xyxy":
            x0, y0, x1, y1 = (int(value) for value in definition["bbox_xyxy"])
            mask = (x >= x0) & (x < x1) & (y >= y0) & (y < y1)
        else:
            raise ValueError(f"unsupported scope kind: {kind}")
        result[name] = np.asarray(mask, dtype=bool).reshape(-1)
    return result


def _sampler_contract(latent_u8: np.ndarray) -> dict[str, Any]:
    height, width = latent_u8.shape[:2]
    probes = np.asarray(
        [
            [0.5 / width, 0.5 / height],
            [(width - 0.5) / width, (height - 0.5) / height],
            [(width + 0.5) / width, 0.5 / height],
            [0.0, 0.0],
            [0.73125, 0.26875],
        ],
        dtype=np.float32,
    )
    latent = latent_u8.astype(np.float32) / np.float32(255.0)
    numpy_sample = sample_float_top_down_wrap(latent, probes)
    ue_sample = ue_bilinear_sample(latent_u8, probes)
    torch_sample = torch_bilinear_sample(
        torch.from_numpy(latent), torch.from_numpy(probes)
    ).numpy()
    center_expected = latent_u8[0, 0].astype(np.float32) / np.float32(255.0)
    return {
        "contract": "top-down glTF UV; v=0 row 0; wrap; half-texel centers; UNORM8 dequantize before bilinear",
        "probe_count": int(probes.shape[0]),
        "numpy_vs_ue_max_abs": float(np.max(np.abs(numpy_sample - ue_sample))),
        "numpy_vs_torch_max_abs": float(np.max(np.abs(numpy_sample - torch_sample))),
        "first_center_max_abs": float(np.max(np.abs(numpy_sample[0] - center_expected))),
        "wrap_center_max_abs": float(np.max(np.abs(numpy_sample[2] - center_expected))),
        "passed": bool(
            np.max(np.abs(numpy_sample - ue_sample)) <= 1.0e-7
            and np.max(np.abs(numpy_sample - torch_sample)) <= 1.0e-7
            and np.array_equal(numpy_sample[0], center_expected)
            and np.array_equal(numpy_sample[2], center_expected)
        ),
    }


def _analyze_probe_family(
    probe_chunks: Sequence[tuple[str, np.ndarray]],
    reference_texture: Mapping[str, np.ndarray],
    decoded_texels: Mapping[str, np.ndarray],
    latent: np.ndarray,
    arrays: dict[str, np.ndarray],
    config: Mapping[str, Any],
    *,
    connected: bool,
    direct_base: np.ndarray | None = None,
) -> dict[str, Any]:
    scopes = config["scopes"]
    thresholds = [float(value) for value in config["analysis"]["luminance_thresholds"]]
    dark_ratio = float(config["analysis"]["dark_ratio"])
    boundary_threshold = float(config["analysis"]["metallic_boundary_threshold"])
    hidden_units = (
        0 if _is_direct_semantic(arrays)
        else int(arrays["normal_head.0.bias"].size) if _is_direct_scalars(arrays)
        else int(arrays["normal_head.0.bias"].size + arrays["scalar_head.0.bias"].size)
        if "normal_head.0.weight" in arrays else 8
    )
    accumulators = {name: _ScopeAccumulator(hidden_units) for name in scopes}
    connected_records: dict[str, list[dict[str, Any]]] = {name: [] for name in scopes}
    for chunk_name, uv in probe_chunks:
        flat_uv = np.asarray(uv, dtype=np.float32).reshape(-1, 2)
        reference = _sample_material(reference_texture, flat_uv)
        runtime = _runtime_material(latent, arrays, flat_uv, direct_base)
        filtered = _sample_material(decoded_texels, flat_uv)
        boundary = subpixel_boundary_mask(
            reference_texture["metallic_linear"], flat_uv, threshold=boundary_threshold
        )
        if _is_direct_semantic(arrays):
            crossings = np.zeros((flat_uv.shape[0], 0), dtype=bool)
        elif _is_direct_scalars(arrays):
            crossings = activation_crossings(
                latent[..., :2], flat_uv,
                arrays["normal_head.0.weight"], arrays["normal_head.0.bias"],
            )
        elif "normal_head.0.weight" in arrays:
            normal_width = arrays["normal_head.0.weight"].shape[1]
            scalar_width = arrays["scalar_head.0.weight"].shape[1]
            normal_latent = latent if normal_width == latent.shape[-1] else latent[..., :normal_width]
            scalar_latent = latent if scalar_width == latent.shape[-1] else latent[..., -scalar_width:]
            crossings = np.concatenate(
                (
                    activation_crossings(normal_latent, flat_uv, arrays["normal_head.0.weight"], arrays["normal_head.0.bias"]),
                    activation_crossings(scalar_latent, flat_uv, arrays["scalar_head.0.weight"], arrays["scalar_head.0.bias"]),
                ),
                axis=-1,
            )
        else:
            hidden_prefix = "network.0" if "network.0.weight" in arrays else "aux_hidden"
            crossings = activation_crossings(
                latent,
                flat_uv,
                arrays[f"{hidden_prefix}.weight"],
                arrays[f"{hidden_prefix}.bias"],
            )
        _, _, base_xy = bilinear_footprint_top_down_wrap(latent.shape[:2], flat_uv)
        masks = _scope_masks(scopes, base_xy, boundary)
        chunk_novel: dict[str, np.ndarray] = {}
        for name, accumulator in accumulators.items():
            chunk_novel[name] = accumulator.ingest(
                masks[name], reference, runtime, filtered, boundary, crossings, thresholds, dark_ratio
            )
        if connected:
            for name in scopes:
                connected_records[name].append(
                    {"phase": chunk_name, "values": chunk_novel[name].astype(bool, copy=False)}
                )
    result: dict[str, Any] = {"scopes": {name: value.report() for name, value in accumulators.items()}}
    if connected:
        height = int(config["inputs"]["height"])
        width = int(config["inputs"]["width"])
        grouped_records: dict[str, list[dict[str, Any]]] = {}
        for name, chunks in connected_records.items():
            phase_order: list[str] = []
            phase_values: dict[str, list[np.ndarray]] = {}
            for item in chunks:
                phase = str(item["phase"])
                if phase not in phase_values:
                    phase_order.append(phase)
                    phase_values[phase] = []
                phase_values[phase].append(np.asarray(item["values"], dtype=bool))
            grouped_records[name] = []
            for phase in phase_order:
                mask = np.concatenate(phase_values[phase]).reshape(height, width)
                grouped_records[name].append(
                    {"phase": phase, **connected_patch_statistics(mask)}
                )
        result["connected_novel_dark_luminance_gt_0_05"] = {
            name: {
                "per_phase": records,
                "max_patch_pixels": max((int(item["max_patch_pixels"]) for item in records), default=0),
                "max_total_pixels": max((int(item["total_pixels"]) for item in records), default=0),
            }
            for name, records in grouped_records.items()
        }
    return result


def _fixed_phase_chunks(
    height: int,
    width: int,
    phases: Sequence[Sequence[float]],
    row_chunk_size: int,
) -> list[tuple[str, np.ndarray]]:
    chunks: list[tuple[str, np.ndarray]] = []
    x = np.arange(width, dtype=np.float32)
    for phase_x, phase_y in phases:
        phase_name = f"x{float(phase_x):.3f}_y{float(phase_y):.3f}"
        for row_start in range(0, height, row_chunk_size):
            y = np.arange(row_start, min(row_start + row_chunk_size, height), dtype=np.float32)
            grid_y, grid_x = np.meshgrid(
                (y + np.float32(phase_y)) / np.float32(height),
                (x + np.float32(phase_x)) / np.float32(width),
                indexing="ij",
            )
            chunks.append((phase_name, np.stack((grid_x, grid_y), axis=-1)))
    return chunks


def _random_chunks(count: int, seed: int, chunk_size: int) -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    uv = rng.random((count, 2), dtype=np.float32)
    return [
        (f"seed_{seed}_{start:08d}", uv[start : start + chunk_size])
        for start in range(0, count, chunk_size)
    ]


def run(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping) or config.get("schema_version") != 1:
        raise ValueError("unsupported interpolation-analysis config schema")
    inputs = config["inputs"]
    core4_dir = _repo_path(inputs["core4_dir"], "inputs.core4_dir")
    hybrid_input = "texture_a_png" in inputs
    latent_path = _repo_path(
        inputs["texture_a_png"] if hybrid_input else inputs["latent_hard_png"],
        "inputs.texture_a_png" if hybrid_input else "inputs.latent_hard_png",
    )
    extra_path = _repo_path(inputs["texture_b_png"], "inputs.texture_b_png") if hybrid_input else None
    decoder_path = _repo_path(inputs["decoder_npz"], "inputs.decoder_npz")
    output_dir = _repo_path(config["output_dir"], "output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    expected_hashes = config["frozen_sha256"]
    actual_hashes = (
        {
            "texture_a_png": sha256_file(latent_path),
            "texture_b_png": sha256_file(extra_path),
            "decoder_npz": sha256_file(decoder_path),
        }
        if hybrid_input and extra_path is not None
        else {"latent_hard_png": sha256_file(latent_path), "decoder_npz": sha256_file(decoder_path)}
    )
    if actual_hashes != dict(expected_hashes):
        raise ValueError(f"frozen input hash mismatch: {actual_hashes}")
    reference = _load_reference(core4_dir)
    height, width = reference["metallic_linear"].shape
    config["inputs"]["height"] = height
    config["inputs"]["width"] = width
    if [height, width] != list(config["analysis"]["expected_atlas_hw"]):
        raise ValueError("Core-4 atlas shape differs from the frozen analysis contract")
    latent_u8 = np.asarray(Image.open(latent_path).convert("RGBA"), dtype=np.uint8)
    if latent_u8.shape[:2] != (height, width):
        raise ValueError("latent and Core-4 atlas dimensions differ")
    direct_base = latent_u8[..., 0:3].astype(np.float32) / np.float32(255.0) if hybrid_input else None
    if hybrid_input:
        assert extra_path is not None
        extra_image = Image.open(extra_path)
        expected_extra = int(config["representation"]["aux_channels"]) - 1
        mode = {1: "L", 2: "LA", 3: "RGB"}.get(expected_extra)
        if mode is None:
            raise ValueError("Hybrid Texture B must contain one to three logical channels")
        extra_u8 = np.asarray(extra_image.convert(mode), dtype=np.uint8)
        if extra_u8.ndim == 2:
            extra_u8 = extra_u8[..., None]
        latent = np.concatenate((latent_u8[..., 3:4], extra_u8), axis=-1).astype(np.float32) / np.float32(255.0)
    else:
        extra_u8 = None
        latent = latent_u8.astype(np.float32) / np.float32(255.0)
    arrays = _load_decoder(decoder_path)
    decoded_texels = _decode_texels(
        latent, arrays, int(config["analysis"]["decode_chunk_size"]), direct_base
    )
    phases = config["analysis"]["fixed_phases_xy"]
    fixed = _analyze_probe_family(
        _fixed_phase_chunks(
            height,
            width,
            phases,
            int(config["analysis"]["fixed_row_chunk_size"]),
        ),
        reference,
        decoded_texels,
        latent,
        arrays,
        config,
        connected=True,
        direct_base=direct_base,
    )
    random_count = int(config["analysis"]["random_probe_count"])
    random_seed = int(config["analysis"]["random_seed"])
    random = _analyze_probe_family(
        _random_chunks(random_count, random_seed, int(config["analysis"]["random_chunk_size"])),
        reference,
        decoded_texels,
        latent,
        arrays,
        config,
        connected=False,
        direct_base=direct_base,
    )
    core4_hashes = {
        name: sha256_file(core4_dir / filename)
        for name, filename in {
            "base_color": "base_color.png",
            "normal": "normal.png",
            "roughness": "roughness.png",
            "metallic": "metallic.png",
        }.items()
    }
    sampler_a = _sampler_contract(latent_u8)
    sampler_b = _sampler_contract(extra_u8) if extra_u8 is not None else None
    sampler = {
        "texture_a": sampler_a,
        "texture_b": sampler_b,
        "identical_uv_filter_contract": bool(sampler_a["passed"] and (sampler_b is None or sampler_b["passed"])),
        "passed": bool(sampler_a["passed"] and (sampler_b is None or sampler_b["passed"])),
    } if hybrid_input else sampler_a
    if not sampler["passed"]:
        raise RuntimeError(f"UV/sampler contract mismatch: {sampler}")
    is_split = "base_affine.weight" in arrays
    is_hybrid = _is_hybrid(arrays)
    if _is_direct_semantic(arrays):
        architecture = "direct_linear_RGB + direct_normalXY_roughness_metallic"
        macs = 0
    elif _is_direct_scalars(arrays):
        architecture = "direct_linear_RGB + normal_2->6->2 + direct_linear_roughness_metallic"
        macs = int(sum(value.size for name, value in arrays.items() if name.endswith("weight")))
    elif "normal_head.0.weight" in arrays:
        architecture = str(config.get("representation", {}).get("architecture", "factorized_auxiliary"))
        macs = int(sum(value.size for name, value in arrays.items() if name.endswith("weight")))
    else:
        architecture = f"direct_linear_RGB + auxiliary_{latent.shape[-1]}->8->4" if is_hybrid else None
        if is_hybrid:
            macs = int(sum(value.size for name, value in arrays.items() if name.endswith("weight")))
    decoder_report: dict[str, Any] = {
        "architecture": (
            architecture
            if is_hybrid
            else "base_affine_4->3 + auxiliary_4->8->4"
            if is_split
            else "4->8->7"
        ),
        "parameters": 0 if _is_direct_semantic(arrays) else int(sum(
            value.size for name, value in arrays.items() if not name.endswith(".marker")
        )),
        "weight_bytes_float32": 0 if _is_direct_semantic(arrays) else int(sum(
            value.nbytes for name, value in arrays.items() if not name.endswith(".marker")
        )),
        "macs_per_pixel": macs if is_hybrid else 76 if is_split else 88,
        "activation_attribution_hidden_head": "auxiliary_only" if is_split or is_hybrid else "shared_all_outputs",
    }
    if is_hybrid:
        decoder_report["base_color_path"] = "direct_linear_UNORM8_bilinear_no_decoder_no_sigmoid"
        if _is_direct_scalars(arrays):
            decoder_report["roughness_metallic_path"] = "direct_linear_UNORM8_bilinear_no_decoder_no_sigmoid"
    elif is_split:
        decoder_report["base_color_affine_weight"] = arrays["base_affine.weight"].tolist()
    else:
        decoder_report["base_color_output_weights_by_hidden_unit"] = arrays["network.2.weight"][:3].T.tolist()
    report: dict[str, Any] = {
        "schema_version": 1,
        "scope": "scifihelmet_hybrid_interpolation" if hybrid_input else "scifihelmet_interpolation_repair_phase0_baseline",
        "formal_holdout_accessed": False,
        "inputs": {
            "core4_dir": inputs["core4_dir"],
            "core4_sha256": core4_hashes,
            **(
                {
                    "texture_a_png": {"path": inputs["texture_a_png"], "sha256": actual_hashes["texture_a_png"]},
                    "texture_b_png": {"path": inputs["texture_b_png"], "sha256": actual_hashes["texture_b_png"]},
                }
                if hybrid_input
                else {"latent_hard_png": {"path": inputs["latent_hard_png"], "sha256": actual_hashes["latent_hard_png"]}}
            ),
            "decoder_npz": {"path": inputs["decoder_npz"], "sha256": actual_hashes["decoder_npz"]},
            "config": {"path": config_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(config_path)},
        },
        "decoder": decoder_report,
        "runtime_order": (
            "RGBA8(RGB direct+aux0) and R8/RG8/logical-RGB8(aux extra) texels -> two identical bilinear samples -> direct linear BaseColor + per-pixel auxiliary path -> one auxiliary postprocess"
            if hybrid_input
            else "RGBA8 texel -> bilinear latent -> per-pixel decoder -> one channel postprocess"
        ),
        "representation": config.get("representation") if hybrid_input else None,
        "reference_filter": "Core-4 linear material channels -> same top-down wrap bilinear; normal renormalized",
        "sampler_uv_contract": sampler,
        "analysis_contract": {
            "fixed_phases_xy": phases,
            "random_probe_count": random_count,
            "random_seed": random_seed,
            "luminance_thresholds": config["analysis"]["luminance_thresholds"],
            "dark_definition": f"prediction_luminance < reference_luminance * {float(config['analysis']['dark_ratio'])}",
            "novel_definition": "runtime_dark AND NOT decode_then_filter_dark",
            "scope_definitions": config["scopes"],
            "gate_probe_family": "seeded_random",
            "connected_patch_probe_family": "fixed_phase_grid",
        },
        "probe_families": {
            "fixed_phase_grid": fixed,
            "seeded_random": random,
        },
    }
    report_path = output_dir / "interpolation_analysis.json"
    report_path.write_text(deterministic_json(report), encoding="utf-8", newline="\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/eval/scifihelmet_interpolation_repair.yaml",
    )
    parser.add_argument("--verify-twice", action="store_true")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = _repo_path(config["output_dir"], "output_dir")
    first = run(config_path)
    report_path = output_dir / "interpolation_analysis.json"
    first_hash = sha256_file(report_path)
    run_hashes = [first_hash]
    if args.verify_twice:
        second = run(config_path)
        second_hash = sha256_file(report_path)
        run_hashes.append(second_hash)
        if deterministic_json(first) != deterministic_json(second) or first_hash != second_hash:
            raise RuntimeError(f"two deterministic runs disagree: {run_hashes}")
    if not args.verify_twice:
        raise RuntimeError("Phase 0 completion requires --verify-twice")
    manifest = phase0_manifest(
        sha256_file(report_path),
        run_hashes,
        first["inputs"],
        sampler_uv_contract_passed=bool(first["sampler_uv_contract"]["passed"]),
    )
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(deterministic_json(manifest), encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                "status": "complete",
                "report_sha256": sha256_file(report_path),
                "two_runs_identical": manifest["determinism"]["identical"],
                "formal_holdout_accessed": False,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
