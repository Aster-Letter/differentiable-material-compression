from __future__ import annotations

from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cg_frontier.render.canonical_cube import (
    build_canonical_cube,
    cube_tangent_normal_to_world,
    make_cube_rng,
    masked_cube_l1,
    sample_cube_atlas,
)


def test_cube_faces_cover_full_uv_with_explicit_orientation() -> None:
    cube = build_canonical_cube(resolution=5, dtype=torch.float64)

    assert cube.face_names == ("+X", "-X", "+Y", "-Y", "+Z", "-Z")
    assert cube.uv.shape == (6, 5, 5, 2)
    assert torch.all(cube.uv.amin(dim=(1, 2)) == 0.0)
    assert torch.all(cube.uv.amax(dim=(1, 2)) == 1.0)
    expected_positions = (
        0.5 * cube.normals[:, None, None, :]
        + (cube.uv[..., 0:1] - 0.5) * cube.tangents[:, None, None, :]
        + (cube.uv[..., 1:2] - 0.5) * cube.bitangents[:, None, None, :]
    )
    assert torch.equal(cube.positions, expected_positions)
    assert torch.all(
        torch.sum(
            (cube.positions[:, 0, -1] - cube.positions[:, 0, 0])
            * cube.tangents,
            dim=-1,
        )
        > 0.0
    )
    assert torch.all(
        torch.sum(
            (cube.positions[:, -1, 0] - cube.positions[:, 0, 0])
            * cube.bitangents,
            dim=-1,
        )
        > 0.0
    )


def test_cube_tbn_is_right_handed_and_preserves_training_space_positive_y() -> None:
    cube = build_canonical_cube(resolution=3, dtype=torch.float64)
    assert torch.equal(torch.linalg.cross(cube.tangents, cube.bitangents), cube.normals)
    assert torch.equal(cube.camera_positions, 2.0 * cube.normals)
    assert torch.equal(cube.camera_targets, torch.zeros_like(cube.normals))
    assert torch.equal(cube.camera_up, cube.bitangents)

    tangent_x = cube_tangent_normal_to_world(
        cube, torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    )
    tangent_y = cube_tangent_normal_to_world(
        cube, torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    )
    tangent_z = cube_tangent_normal_to_world(
        cube, torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64)
    )

    assert torch.equal(tangent_x, cube.tangents)
    assert torch.equal(tangent_y, cube.bitangents)
    assert torch.equal(tangent_z, cube.normals)


def test_cube_material_and_valid_mask_share_footprint_and_mask_gradient() -> None:
    cube = build_canonical_cube(resolution=3, dtype=torch.float64)
    atlas = (torch.arange(9, dtype=torch.float64) + 1.0).reshape(3, 3, 1)
    atlas = atlas.clone().requires_grad_(True)
    valid = torch.ones(3, 3, dtype=torch.bool)
    valid[1, 1] = False

    samples = sample_cube_atlas(atlas, valid, cube)
    result = masked_cube_l1(samples.material, torch.zeros_like(samples.material), samples.valid)
    result.loss.backward()

    assert samples.material.shape == (6, 3, 3, 1)
    assert torch.all(samples.material[:, 1, 1, 0] == atlas[1, 1, 0])
    assert torch.all(samples.valid[:, 1, 1] == 0.0)
    assert result.valid_pixel_count == 48
    assert atlas.grad is not None
    assert atlas.grad[1, 1, 0] == 0.0
    assert torch.count_nonzero(atlas.grad) == 8


def test_cube_loss_normalizes_identical_error_by_total_valid_pixels() -> None:
    small_prediction = torch.full((6, 2, 2, 3), 0.25, dtype=torch.float64)
    small_reference = torch.zeros_like(small_prediction)
    small_valid = torch.ones(6, 2, 2, dtype=torch.bool)

    large_prediction = torch.full((6, 5, 5, 3), 1000.0, dtype=torch.float64)
    large_reference = torch.zeros_like(large_prediction)
    large_valid = torch.zeros(6, 5, 5, dtype=torch.bool)
    large_valid[:, ::2, ::2] = True
    large_prediction[large_valid] = 0.25

    small = masked_cube_l1(small_prediction, small_reference, small_valid)
    large = masked_cube_l1(large_prediction, large_reference, large_valid)

    assert small.valid_pixel_count == 24
    assert large.valid_pixel_count == 54
    assert small.loss.item() == 0.25
    assert large.loss.item() == 0.25


def test_cube_rng_advance_does_not_change_core_sampling_sequence() -> None:
    core = torch.Generator().manual_seed(127)
    expected_core = torch.Generator().manual_seed(127)
    cube_rng = make_cube_rng(seed=991)

    torch.rand(100, generator=cube_rng)
    actual = torch.rand(16, generator=core)
    expected = torch.rand(16, generator=expected_core)

    assert cube_rng is not core
    assert torch.equal(actual, expected)
