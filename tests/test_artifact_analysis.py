from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cg_frontier.compression.artifact_analysis import (
    bilinear_sample_float_top_down_wrap,
    connected_patch_statistics,
    deterministic_json,
    deterministic_tile_partitions,
    metallic_boundary_mask,
    oracle_hybrid,
    roi_mask,
    tail_statistics,
    worst_tiles,
)


def test_tail_statistics_include_frozen_extreme_percentiles() -> None:
    result = tail_statistics(np.arange(1000, dtype=np.float32))
    assert result["count"] == 1000
    assert result["p95_0"] == np.percentile(np.arange(1000), 95.0)
    assert result["p99_9"] == np.percentile(np.arange(1000), 99.9)
    assert result["max"] == 999.0


def test_metallic_boundary_marks_both_sides_only() -> None:
    values = np.zeros((4, 5), dtype=np.float32)
    values[:, 3:] = 1.0
    boundary = metallic_boundary_mask(values, 0.1)
    assert np.all(boundary[:, 2:4])
    assert not np.any(boundary[:, :2])
    assert not np.any(boundary[:, 4:])


def test_worst_tiles_are_stable_and_honor_exclusion() -> None:
    error = np.zeros((8, 8), dtype=np.float32)
    error[:4, :4] = 2.0
    error[4:, 4:] = 1.0
    exclude = np.zeros_like(error, dtype=bool)
    exclude[:4, :4] = True
    result = worst_tiles(error, tile_size=4, limit=2, exclude_mask=exclude)
    assert result[0]["bbox_xyxy"] == [4, 4, 8, 8]
    assert result[0]["mean_error"] == 1.0


def test_roi_clips_and_excludes_outside_pixels() -> None:
    mask = roi_mask((4, 5), (-2, 1, 3, 8))
    assert mask.sum() == 9
    assert np.all(mask[1:, :3])
    assert not np.any(mask[:, 3:])


def test_oracle_hybrid_replaces_only_requested_channel() -> None:
    reference = {name: np.full((2, 2), index, dtype=np.float32) for index, name in enumerate(("base_color_linear", "normal_xyz", "roughness_linear", "metallic_linear"))}
    candidate = {name: value + 10 for name, value in reference.items()}
    hybrid = oracle_hybrid(reference, candidate, ["metallic_linear"])
    assert np.array_equal(hybrid["metallic_linear"], reference["metallic_linear"])
    assert np.array_equal(hybrid["base_color_linear"], candidate["base_color_linear"])


def test_tile_split_is_deterministic_disjoint_and_exhaustive() -> None:
    first = deterministic_tile_partitions(256, 256, tile_size=64)
    second = deterministic_tile_partitions(256, 256, tile_size=64)
    assert all(np.array_equal(first[name], second[name]) for name in first)
    total = sum(mask.astype(np.uint8) for mask in first.values())
    assert np.all(total == 1)


def test_report_serialization_is_deterministic() -> None:
    left = deterministic_json({"b": 2, "a": {"z": 1}})
    right = deterministic_json(json.loads(left))
    assert left == right


def test_connected_patch_statistics_use_four_connectivity() -> None:
    mask = np.eye(3, dtype=bool)
    result = connected_patch_statistics(mask)
    assert result["patch_count"] == 3
    assert result["max_patch_pixels"] == 1


def test_float_bilinear_wrap_has_half_texel_centers() -> None:
    texture = np.asarray([[[0.0], [1.0]], [[2.0], [3.0]]], dtype=np.float32)
    uv = np.asarray([[0.25, 0.25], [0.75, 0.75], [1.25, 0.25]], dtype=np.float32)
    sampled = bilinear_sample_float_top_down_wrap(texture, uv)[:, 0]
    assert np.array_equal(sampled, np.asarray([0.0, 3.0, 0.0], dtype=np.float32))
