"""Deterministic tail, spatial, ROI, and oracle-hybrid material diagnostics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


TAIL_PERCENTILES = (50.0, 95.0, 99.0, 99.9)
MATERIAL_CHANNELS = (
    "base_color_linear",
    "normal_xyz",
    "roughness_linear",
    "metallic_linear",
)


def sha256_file(path: Path | str) -> str:
    """Return a streaming lowercase SHA-256 digest."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tail_statistics(values: np.ndarray, mask: np.ndarray | None = None) -> dict[str, float | int]:
    """Summarize finite scalar samples with frozen tail percentiles."""

    array = np.asarray(values, dtype=np.float64)
    if mask is not None:
        selection = np.asarray(mask, dtype=bool)
        if selection.shape != array.shape:
            raise ValueError("tail mask must match scalar values")
        array = array[selection]
    else:
        array = array.reshape(-1)
    if array.size == 0 or not np.isfinite(array).all():
        raise ValueError("tail statistics require non-empty finite values")
    result: dict[str, float | int] = {
        "count": int(array.size),
        "mean": float(array.mean(dtype=np.float64)),
    }
    for percentile in TAIL_PERCENTILES:
        key = f"p{str(percentile).replace('.', '_')}"
        result[key] = float(np.percentile(array, percentile))
    result["max"] = float(array.max())
    return result


def metallic_boundary_mask(reference: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """Mark both sides of 4-neighbour metallic jumps above ``threshold``."""

    values = np.asarray(reference, dtype=np.float32)
    if values.ndim != 2 or threshold <= 0.0:
        raise ValueError("metallic reference must be 2D and threshold positive")
    result = np.zeros(values.shape, dtype=bool)
    horizontal = np.abs(values[:, 1:] - values[:, :-1]) > threshold
    vertical = np.abs(values[1:, :] - values[:-1, :]) > threshold
    result[:, 1:] |= horizontal
    result[:, :-1] |= horizontal
    result[1:, :] |= vertical
    result[:-1, :] |= vertical
    return result


def low_gradient_mask(values: np.ndarray, threshold: float = 0.01) -> np.ndarray:
    """Select pixels whose maximum 4-neighbour channel gradient is small."""

    image = np.asarray(values, dtype=np.float32)
    if image.ndim not in (2, 3) or threshold <= 0.0:
        raise ValueError("gradient input must be HW/HWC and threshold positive")
    if image.ndim == 2:
        image = image[..., None]
    gradient = np.zeros(image.shape[:2], dtype=np.float32)
    horizontal = np.max(np.abs(image[:, 1:] - image[:, :-1]), axis=-1)
    vertical = np.max(np.abs(image[1:] - image[:-1]), axis=-1)
    gradient[:, 1:] = np.maximum(gradient[:, 1:], horizontal)
    gradient[:, :-1] = np.maximum(gradient[:, :-1], horizontal)
    gradient[1:] = np.maximum(gradient[1:], vertical)
    gradient[:-1] = np.maximum(gradient[:-1], vertical)
    return gradient < threshold


def connected_patch_statistics(mask: np.ndarray) -> dict[str, float | int]:
    """Summarize 4-connected true patches without requiring image metadata."""

    from scipy import ndimage

    selection = np.asarray(mask, dtype=bool)
    if selection.ndim != 2:
        raise ValueError("connected patch mask must be 2D")
    labels, count = ndimage.label(selection, structure=np.asarray([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    if count == 0:
        return {"patch_count": 0, "total_pixels": 0, "max_patch_pixels": 0, "mean_patch_pixels": 0.0}
    areas = np.bincount(labels.reshape(-1), minlength=count + 1)[1:]
    return {
        "patch_count": int(count),
        "total_pixels": int(areas.sum()),
        "max_patch_pixels": int(areas.max()),
        "mean_patch_pixels": float(areas.mean(dtype=np.float64)),
    }


def bilinear_sample_float_top_down_wrap(texture: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Sample float HWC data using the frozen top-down wrap convention."""

    image = np.asarray(texture, dtype=np.float32)
    coordinates = np.asarray(uv, dtype=np.float32)
    if image.ndim != 3 or coordinates.shape[-1:] != (2,):
        raise ValueError("bilinear inputs must be HWC texture and UV pairs")
    height, width, channels = image.shape
    x = coordinates[..., 0] * np.float32(width) - np.float32(0.5)
    y = coordinates[..., 1] * np.float32(height) - np.float32(0.5)
    x0_floor, y0_floor = np.floor(x), np.floor(y)
    wx, wy = (x - x0_floor)[..., None], (y - y0_floor)[..., None]
    x0, y0 = x0_floor.astype(np.int64) % width, y0_floor.astype(np.int64) % height
    x1, y1 = (x0 + 1) % width, (y0 + 1) % height
    top = image[y0, x0].reshape(*x0.shape, channels) * (1.0 - wx) + image[y0, x1].reshape(*x0.shape, channels) * wx
    bottom = image[y1, x0].reshape(*x0.shape, channels) * (1.0 - wx) + image[y1, x1].reshape(*x0.shape, channels) * wx
    return np.asarray(top * (1.0 - wy) + bottom * wy, dtype=np.float32)


def deterministic_tile_partitions(
    height: int,
    width: int,
    *,
    tile_size: int = 64,
    seed: int = 20260803,
    selection_fraction: float = 0.1,
    validation_fraction: float = 0.1,
) -> dict[str, np.ndarray]:
    """Assign whole tiles to optimizer, selection, or repair-validation.

    Assignment hashes tile coordinates, so it is stable across traversal order.
    Selection and validation tiles are never eligible for hard-example pools.
    """

    if min(height, width, tile_size) <= 0:
        raise ValueError("image and tile dimensions must be positive")
    if selection_fraction < 0.0 or validation_fraction < 0.0:
        raise ValueError("partition fractions must be non-negative")
    if selection_fraction + validation_fraction >= 1.0:
        raise ValueError("selection plus validation must leave optimizer tiles")
    masks = {
        "optimizer": np.zeros((height, width), dtype=bool),
        "selection": np.zeros((height, width), dtype=bool),
        "repair_validation": np.zeros((height, width), dtype=bool),
    }
    tiles: list[tuple[bytes, int, int, int, int]] = []
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            token = f"{seed}:{y // tile_size}:{x // tile_size}".encode("ascii")
            tiles.append(
                (hashlib.sha256(token).digest(), y, x, min(y + tile_size, height), min(x + tile_size, width))
            )
    tiles.sort(key=lambda item: item[0])
    selection_count = max(1, int(round(len(tiles) * selection_fraction)))
    validation_count = max(1, int(round(len(tiles) * validation_fraction)))
    if selection_count + validation_count >= len(tiles):
        raise ValueError("image has too few tiles for three non-empty partitions")
    for index, (_, y, x, y1, x1) in enumerate(tiles):
        name = (
            "selection"
            if index < selection_count
            else "repair_validation"
            if index < selection_count + validation_count
            else "optimizer"
        )
        masks[name][y:y1, x:x1] = True
    if np.any(sum(mask.astype(np.uint8) for mask in masks.values()) != 1):
        raise AssertionError("tile partitions must be exhaustive and disjoint")
    return masks


def worst_tiles(
    error: np.ndarray,
    *,
    tile_size: int = 64,
    limit: int = 16,
    include_mask: np.ndarray | None = None,
    exclude_mask: np.ndarray | None = None,
) -> list[dict[str, float | int | list[int]]]:
    """Return a stable descending ranking of mean-error tiles."""

    values = np.asarray(error, dtype=np.float64)
    if values.ndim != 2 or tile_size <= 0 or limit <= 0:
        raise ValueError("worst-tile input must be 2D with positive sizes")
    include = np.ones(values.shape, dtype=bool) if include_mask is None else np.asarray(include_mask, dtype=bool)
    exclude = np.zeros(values.shape, dtype=bool) if exclude_mask is None else np.asarray(exclude_mask, dtype=bool)
    if include.shape != values.shape or exclude.shape != values.shape:
        raise ValueError("tile masks must match the error image")
    records: list[dict[str, float | int | list[int]]] = []
    height, width = values.shape
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            y1, x1 = min(y + tile_size, height), min(x + tile_size, width)
            valid = include[y:y1, x:x1] & ~exclude[y:y1, x:x1]
            if not np.any(valid):
                continue
            samples = values[y:y1, x:x1][valid]
            records.append(
                {
                    "bbox_xyxy": [x, y, x1, y1],
                    "valid_pixels": int(samples.size),
                    "mean_error": float(samples.mean(dtype=np.float64)),
                    "max_error": float(samples.max()),
                }
            )
    records.sort(key=lambda item: (-float(item["mean_error"]), int(item["bbox_xyxy"][1]), int(item["bbox_xyxy"][0])))
    return records[:limit]


def normal_angular_error(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Return per-pixel angular error in degrees after safe normalization."""

    truth = np.asarray(reference, dtype=np.float64)
    prediction = np.asarray(candidate, dtype=np.float64)
    if truth.shape != prediction.shape or truth.shape[-1:] != (3,):
        raise ValueError("normal arrays must have matching trailing XYZ")
    truth /= np.maximum(np.linalg.norm(truth, axis=-1, keepdims=True), 1.0e-12)
    prediction /= np.maximum(np.linalg.norm(prediction, axis=-1, keepdims=True), 1.0e-12)
    dot = np.sum(truth * prediction, axis=-1).clip(-1.0, 1.0)
    return np.rad2deg(np.arccos(dot))


def material_error_maps(
    reference: Mapping[str, np.ndarray], candidate: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Build scalar per-pixel errors for the four compressed semantics."""

    missing = [name for name in MATERIAL_CHANNELS if name not in reference or name not in candidate]
    if missing:
        raise ValueError(f"material mapping is missing channels: {', '.join(missing)}")
    base = np.max(
        np.abs(np.asarray(candidate["base_color_linear"]) - np.asarray(reference["base_color_linear"])),
        axis=-1,
    )
    normal = normal_angular_error(reference["normal_xyz"], candidate["normal_xyz"])
    roughness = np.abs(np.asarray(candidate["roughness_linear"]) - np.asarray(reference["roughness_linear"]))
    metallic = np.abs(np.asarray(candidate["metallic_linear"]) - np.asarray(reference["metallic_linear"]))
    return {
        "base_color_max_channel": base.astype(np.float32),
        "normal_degrees": normal.astype(np.float32),
        "roughness": roughness.astype(np.float32),
        "metallic": metallic.astype(np.float32),
    }


def roi_mask(shape: tuple[int, int], bbox_xyxy: Sequence[int]) -> np.ndarray:
    """Create a clipped rectangular ROI mask using exclusive max coordinates."""

    if len(bbox_xyxy) != 4:
        raise ValueError("ROI must contain x0,y0,x1,y1")
    height, width = shape
    x0, y0, x1, y1 = (int(value) for value in bbox_xyxy)
    x0, x1 = max(0, x0), min(width, x1)
    y0, y1 = max(0, y0), min(height, y1)
    if x0 >= x1 or y0 >= y1:
        raise ValueError("ROI does not intersect the image")
    result = np.zeros(shape, dtype=bool)
    result[y0:y1, x0:x1] = True
    return result


def roi_material_metrics(
    reference: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    mask: np.ndarray,
    *,
    metallic_edges: np.ndarray | None = None,
) -> dict[str, Any]:
    """Measure one defect ROI, including dark prediction and composite tails."""

    errors = material_error_maps(reference, candidate)
    selection = np.asarray(mask, dtype=bool)
    if selection.shape != errors["metallic"].shape or not np.any(selection):
        raise ValueError("ROI mask must select pixels from the material image")
    base_truth = np.asarray(reference["base_color_linear"], dtype=np.float64)
    base_candidate = np.asarray(candidate["base_color_linear"], dtype=np.float64)
    luminance_weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float64)
    truth_luma = base_truth @ luminance_weights
    candidate_luma = base_candidate @ luminance_weights
    visible_truth = truth_luma > 1.0e-6
    dark = selection & visible_truth & (candidate_luma < truth_luma * 0.5)
    dark_denominator = selection & visible_truth
    composite = (
        errors["base_color_max_channel"]
        + errors["normal_degrees"] / 180.0
        + errors["roughness"]
        + errors["metallic"]
    ) / 4.0
    result: dict[str, Any] = {
        "pixels": int(selection.sum()),
        "base_color_max_channel": tail_statistics(errors["base_color_max_channel"], selection),
        "normal_degrees": tail_statistics(errors["normal_degrees"], selection),
        "roughness": tail_statistics(errors["roughness"], selection),
        "metallic": tail_statistics(errors["metallic"], selection),
        "composite": tail_statistics(composite, selection),
        "predicted_luminance_below_reference_half_fraction": (
            float(dark.sum() / dark_denominator.sum()) if np.any(dark_denominator) else 0.0
        ),
    }
    if metallic_edges is not None:
        edge_selection = selection & np.asarray(metallic_edges, dtype=bool)
        result["metallic_boundary"] = (
            {
                **tail_statistics(errors["metallic"], edge_selection),
                "fraction_above_0_1": float(np.mean(errors["metallic"][edge_selection] > 0.1)),
            }
            if np.any(edge_selection)
            else {"count": 0, "mean": 0.0, "fraction_above_0_1": 0.0}
        )
    return result


def oracle_hybrid(
    reference: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    oracle_channels: Sequence[str],
) -> dict[str, np.ndarray]:
    """Replace selected candidate semantics with reference values for attribution."""

    requested = set(oracle_channels)
    unknown = requested.difference(MATERIAL_CHANNELS)
    if unknown:
        raise ValueError(f"unknown oracle channels: {sorted(unknown)}")
    return {
        name: np.asarray(reference[name] if name in requested else candidate[name]).copy()
        for name in MATERIAL_CHANNELS
    }


def cross_channel_error_correlation(error_maps: Mapping[str, np.ndarray]) -> dict[str, float]:
    """Return deterministic Pearson correlations for scalar error maps."""

    names = sorted(error_maps)
    result: dict[str, float] = {}
    for index, left in enumerate(names):
        x = np.asarray(error_maps[left], dtype=np.float64).reshape(-1)
        for right in names[index + 1 :]:
            y = np.asarray(error_maps[right], dtype=np.float64).reshape(-1)
            if x.size != y.size:
                raise ValueError("error maps must have matching sizes")
            # Avoid np.corrcoef/BLAS so the analysis remains usable on the
            # project's Windows runtime when an optional BLAS DLL is absent.
            x_centered = x - x.mean(dtype=np.float64)
            y_centered = y - y.mean(dtype=np.float64)
            denominator = float(
                np.sqrt(
                    np.sum(x_centered * x_centered, dtype=np.float64)
                    * np.sum(y_centered * y_centered, dtype=np.float64)
                )
            )
            value = (
                0.0
                if denominator == 0.0
                else float(np.sum(x_centered * y_centered, dtype=np.float64) / denominator)
            )
            result[f"{left}__{right}"] = value
    return result


def deterministic_json(value: Mapping[str, Any]) -> str:
    """Serialize a manifest with stable key and whitespace ordering."""

    return json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
