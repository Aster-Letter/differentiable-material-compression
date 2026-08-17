from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cg_frontier.render.camera_relative_lighting import (
    CameraRelativeLightFamily,
    build_camera_relative_light_grid,
    camera_local_to_world,
    parse_camera_relative_light_families,
)
from cg_frontier.render.gbuffer import Camera


def _camera(
    *,
    eye: tuple[float, float, float],
    target: tuple[float, float, float],
) -> Camera:
    return Camera(
        eye=eye,
        target=target,
        up=(0.0, 1.0, 0.0),
        vertical_fov_degrees=35.0,
        near=0.1,
        far=100.0,
    )


def test_camera_local_light_coordinates_follow_shifted_and_rotated_camera() -> None:
    front = _camera(eye=(2.0, 3.0, 8.0), target=(2.0, 3.0, 4.0))
    side = _camera(eye=(6.0, 3.0, 4.0), target=(2.0, 3.0, 4.0))

    assert camera_local_to_world(front, (1.0, 2.0, 3.0)) == pytest.approx(
        (3.0, 5.0, 7.0)
    )
    assert camera_local_to_world(side, (1.0, 2.0, 3.0)) == pytest.approx(
        (5.0, 5.0, 3.0)
    )


def test_light_family_grid_preserves_photometry_while_following_each_camera() -> None:
    cameras = (
        _camera(eye=(0.0, 0.0, 4.0), target=(0.0, 0.0, 0.0)),
        _camera(eye=(4.0, 2.0, 1.0), target=(1.0, 2.0, 1.0)),
    )
    family = CameraRelativeLightFamily(
        name="key",
        local_position=(1.0, 2.0, 3.0),
        color=(1.0, 0.9, 0.8),
        radiant_intensity=90.0,
        ambient_intensity=0.04,
        role="camera_side",
    )

    grid = build_camera_relative_light_grid(cameras, (family,))

    assert len(grid) == 1
    assert grid[0][0].position == pytest.approx((1.0, 2.0, 3.0))
    assert grid[0][1].position == pytest.approx((4.0, 4.0, 0.0))
    assert all(light.color == family.color for light in grid[0])
    assert all(light.radiant_intensity == 90.0 for light in grid[0])
    assert all(light.ambient_intensity == 0.04 for light in grid[0])


def test_six_light_policy_allows_only_one_deliberate_opposite_side_rim() -> None:
    specs = [
        {
            "name": f"camera-side-{index}",
            "local_position": [float(index), 2.0, 3.0],
            "color": [1.0, 0.9, 0.8],
            "radiant_intensity": 80.0,
            "ambient_intensity": 0.04,
            "role": "camera_side",
        }
        for index in range(5)
    ]
    specs.append(
        {
            "name": "rim",
            "local_position": [0.0, 3.0, -3.5],
            "color": [0.9, 0.95, 1.0],
            "radiant_intensity": 72.0,
            "ambient_intensity": 0.04,
            "role": "rim",
        }
    )

    families = parse_camera_relative_light_families(specs)

    assert [family.role for family in families].count("camera_side") == 5
    assert [family.role for family in families].count("rim") == 1

    specs[0]["local_position"] = [0.0, 2.0, -3.0]
    with pytest.raises(ValueError, match="camera-side"):
        parse_camera_relative_light_families(specs)
