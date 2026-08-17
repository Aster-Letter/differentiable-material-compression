from __future__ import annotations

import numpy as np

from cg_frontier.render.generic_c4_rig import build_generic_c4_rig


def test_generic_rig_has_frozen_camera_light_split_and_hash() -> None:
    first = build_generic_c4_rig()
    second = build_generic_c4_rig()

    assert len(first.cameras) == 31
    assert sum(camera.split == "train" for camera in first.cameras) == 24
    assert sum(camera.split == "audit" for camera in first.cameras) == 7
    assert len(first.lights) == 6
    assert sum(light.frame == "world" for light in first.lights) == 3
    assert sum(light.frame == "camera" for light in first.lights) == 3
    assert set(first.presentation_views) == {"front", "rear", "upper_side", "top"}
    assert first.rig_hash == second.rig_hash
    assert first == second
    assert all(
        np.isclose(np.linalg.norm(camera.direction), 1.0, atol=1.0e-12)
        for camera in first.cameras
    )
