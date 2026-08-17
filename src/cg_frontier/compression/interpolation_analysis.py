"""Deterministic subpixel sampling and ReLU-boundary diagnostics."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from cg_frontier.compression.artifact_analysis import tail_statistics


LUMINANCE_WEIGHTS = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)


def bilinear_footprint_top_down_wrap(
    shape: tuple[int, int], uv: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return wrapped YX corner indices, weights, and base pixel coordinates.

    UV follows the project contract: top-down arrays, glTF ``v=0`` at row zero,
    wrap addressing, and texel centers at ``(i + 0.5) / extent``.
    """

    height, width = (int(value) for value in shape)
    coordinates = np.asarray(uv, dtype=np.float32)
    if min(height, width) <= 0 or coordinates.ndim < 2 or coordinates.shape[-1] != 2:
        raise ValueError("shape must be positive and uv must end in two coordinates")
    x = coordinates[..., 0] * np.float32(width) - np.float32(0.5)
    y = coordinates[..., 1] * np.float32(height) - np.float32(0.5)
    x0_floor, y0_floor = np.floor(x), np.floor(y)
    wx = np.asarray(x - x0_floor, dtype=np.float32)
    wy = np.asarray(y - y0_floor, dtype=np.float32)
    x0, y0 = x0_floor.astype(np.int64) % width, y0_floor.astype(np.int64) % height
    x1, y1 = (x0 + 1) % width, (y0 + 1) % height
    corners_yx = np.stack(
        (
            np.stack((y0, x0), axis=-1),
            np.stack((y0, x1), axis=-1),
            np.stack((y1, x0), axis=-1),
            np.stack((y1, x1), axis=-1),
        ),
        axis=-2,
    )
    weights = np.stack(
        (
            (np.float32(1.0) - wx) * (np.float32(1.0) - wy),
            wx * (np.float32(1.0) - wy),
            (np.float32(1.0) - wx) * wy,
            wx * wy,
        ),
        axis=-1,
    ).astype(np.float32, copy=False)
    base_xy = np.stack(
        (
            np.floor(coordinates[..., 0] * np.float32(width)).astype(np.int64) % width,
            np.floor(coordinates[..., 1] * np.float32(height)).astype(np.int64) % height,
        ),
        axis=-1,
    )
    return corners_yx, weights, base_xy


def sample_corners_top_down_wrap(
    texture: np.ndarray, uv: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gather the four bilinear corners without applying interpolation."""

    image = np.asarray(texture)
    if image.ndim == 2:
        image = image[..., None]
    if image.ndim != 3:
        raise ValueError("texture must be HW or HWC")
    corners_yx, weights, base_xy = bilinear_footprint_top_down_wrap(image.shape[:2], uv)
    corners = image[corners_yx[..., 0], corners_yx[..., 1]]
    return np.asarray(corners), weights, base_xy


def sample_float_top_down_wrap(texture: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Sample float HW/HWC data using the explicit frozen bilinear footprint."""

    corners, weights, _ = sample_corners_top_down_wrap(texture, uv)
    return np.sum(
        np.asarray(corners, dtype=np.float32) * weights[..., None],
        axis=-2,
        dtype=np.float32,
    )


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    """Normalize trailing XYZ values with a deterministic positive epsilon."""

    values = np.asarray(vectors, dtype=np.float32)
    if values.shape[-1:] != (3,):
        raise ValueError("normal values must end in XYZ")
    length = np.sqrt(np.sum(values * values, axis=-1, keepdims=True, dtype=np.float32))
    return np.asarray(values / np.maximum(length, np.float32(1.0e-8)), dtype=np.float32)


def normal_angular_error(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Return normalized tangent-space angular error in degrees."""

    truth = normalize_vectors(reference)
    prediction = normalize_vectors(candidate)
    dot = np.sum(truth * prediction, axis=-1, dtype=np.float32).clip(-1.0, 1.0)
    return np.asarray(np.rad2deg(np.arccos(dot)), dtype=np.float32)


def activation_crossings(
    latent_texture: np.ndarray,
    uv: np.ndarray,
    hidden_weight: np.ndarray,
    hidden_bias: np.ndarray,
) -> np.ndarray:
    """Mark hidden units whose contributing bilinear corners cross ReLU zero."""

    corners, weights, _ = sample_corners_top_down_wrap(latent_texture, uv)
    weight = np.asarray(hidden_weight, dtype=np.float32)
    bias = np.asarray(hidden_bias, dtype=np.float32)
    if weight.ndim != 2 or corners.shape[-1] != weight.shape[1]:
        raise ValueError("activation attribution requires latent corners matching hidden input width")
    if bias.shape != (weight.shape[0],):
        raise ValueError("hidden bias shape does not match hidden weights")
    preactivation = np.sum(
        np.asarray(corners, dtype=np.float32)[..., :, None, :] * weight[None, None, :, :]
        if corners.ndim == 3
        else np.asarray(corners, dtype=np.float32)[..., :, None, :] * weight,
        axis=-1,
        dtype=np.float32,
    ) + bias
    support = weights[..., :, None] > np.float32(0.0)
    has_positive = np.any((preactivation > np.float32(0.0)) & support, axis=-2)
    has_nonpositive = np.any((preactivation <= np.float32(0.0)) & support, axis=-2)
    return has_positive & has_nonpositive


def subpixel_boundary_mask(
    scalar_texture: np.ndarray, uv: np.ndarray, *, threshold: float
) -> np.ndarray:
    """Select samples whose contributing scalar corners span a true boundary."""

    if threshold <= 0.0:
        raise ValueError("boundary threshold must be positive")
    corners, weights, _ = sample_corners_top_down_wrap(scalar_texture, uv)
    values = np.asarray(corners[..., 0], dtype=np.float32)
    support = weights > np.float32(0.0)
    low = np.min(np.where(support, values, np.float32(np.inf)), axis=-1)
    high = np.max(np.where(support, values, np.float32(-np.inf)), axis=-1)
    return (high - low) > np.float32(threshold)


def dark_fraction_counts(
    reference_luminance: np.ndarray,
    runtime_luminance: np.ndarray,
    filtered_luminance: np.ndarray,
    thresholds: Sequence[float],
    *,
    ratio: float = 0.5,
) -> dict[str, dict[str, int]]:
    """Count runtime, decode-then-filter, and interpolation-novel dark samples."""

    truth = np.asarray(reference_luminance, dtype=np.float32).reshape(-1)
    runtime = np.asarray(runtime_luminance, dtype=np.float32).reshape(-1)
    filtered = np.asarray(filtered_luminance, dtype=np.float32).reshape(-1)
    if truth.shape != runtime.shape or truth.shape != filtered.shape or not (0.0 < ratio < 1.0):
        raise ValueError("dark-fraction arrays must match and ratio must lie in (0,1)")
    runtime_dark = runtime < truth * np.float32(ratio)
    filtered_dark = filtered < truth * np.float32(ratio)
    novel = runtime_dark & ~filtered_dark
    result: dict[str, dict[str, int]] = {}
    for threshold in thresholds:
        eligible = truth > np.float32(threshold)
        result[f"{float(threshold):.2f}"] = {
            "eligible": int(eligible.sum()),
            "runtime_dark": int(np.sum(eligible & runtime_dark)),
            "filter_dark": int(np.sum(eligible & filtered_dark)),
            "novel_dark": int(np.sum(eligible & novel)),
        }
    return result


def fraction_report(counts: Mapping[str, Mapping[str, int]]) -> dict[str, dict[str, float | int]]:
    """Convert additive dark counts to stable fractions."""

    result: dict[str, dict[str, float | int]] = {}
    for threshold, values in counts.items():
        denominator = int(values["eligible"])
        result[threshold] = {"eligible": denominator}
        for name in ("runtime_dark", "filter_dark", "novel_dark"):
            count = int(values[name])
            result[threshold][f"{name}_count"] = count
            result[threshold][f"{name}_fraction"] = float(count / denominator) if denominator else 0.0
    return result


def hidden_unit_attribution(
    crossings: np.ndarray, novel_dark: np.ndarray
) -> dict[str, Any]:
    """Summarize crossing prevalence and per-unit novel-dark attribution."""

    crossing = np.asarray(crossings, dtype=bool)
    novel = np.asarray(novel_dark, dtype=bool).reshape(-1)
    if crossing.ndim != 2 or crossing.shape[0] != novel.size:
        raise ValueError("crossings must be NxH and novel_dark must be N")
    any_crossing = np.any(crossing, axis=-1)
    novel_count = int(novel.sum())
    units: dict[str, dict[str, float | int]] = {}
    for unit in range(crossing.shape[1]):
        crossed = crossing[:, unit]
        joint = int(np.sum(crossed & novel))
        crossed_count = int(crossed.sum())
        units[str(unit)] = {
            "crossing_count": crossed_count,
            "novel_dark_crossing_count": joint,
            "novel_dark_coverage": float(joint / novel_count) if novel_count else 0.0,
            "conditional_novel_dark_rate": float(joint / crossed_count) if crossed_count else 0.0,
        }
    joint_any = int(np.sum(any_crossing & novel))
    return {
        "sample_count": int(novel.size),
        "novel_dark_count": novel_count,
        "any_crossing_count": int(any_crossing.sum()),
        "novel_dark_with_any_crossing_count": joint_any,
        "novel_dark_with_any_crossing_fraction": float(joint_any / novel_count) if novel_count else 0.0,
        "units": units,
    }


def error_tail(values: Sequence[np.ndarray]) -> dict[str, float | int]:
    """Concatenate deterministic chunks and return the frozen tail schema."""

    if not values:
        return {"count": 0, "mean": 0.0, "p50_0": 0.0, "p95_0": 0.0, "p99_0": 0.0, "p99_9": 0.0, "max": 0.0}
    return tail_statistics(np.concatenate([np.asarray(value, dtype=np.float32).reshape(-1) for value in values]))


def phase0_manifest(
    report_sha256: str,
    run_hashes: Sequence[str],
    frozen_inputs: Mapping[str, Any],
    *,
    sampler_uv_contract_passed: bool,
) -> dict[str, Any]:
    """Build the signed Phase-0 manifest, rejecting weak determinism evidence."""

    hashes = [str(value).lower() for value in run_hashes]
    if len(report_sha256) != 64 or any(len(value) != 64 for value in hashes):
        raise ValueError("manifest SHA-256 values must contain 64 hexadecimal characters")
    if len(hashes) != 2 or len(set(hashes)) != 1 or hashes[0] != report_sha256.lower():
        raise ValueError("Phase 0 requires two byte-identical report runs")
    if not sampler_uv_contract_passed:
        raise ValueError("Phase 0 manifest requires a passing sampler/UV contract")
    return {
        "schema_version": 1,
        "formal_holdout_accessed": False,
        "report": {"path": "interpolation_analysis.json", "sha256": report_sha256.lower()},
        "determinism": {
            "two_runs_requested": True,
            "run_report_sha256": hashes,
            "identical": True,
        },
        "frozen_input_sha256": dict(frozen_inputs),
        "sampler_uv_contract_passed": True,
    }
