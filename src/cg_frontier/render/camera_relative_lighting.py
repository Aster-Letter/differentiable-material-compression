"""Camera-relative point-light placement for render-pool experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

from cg_frontier.render.gbuffer import Camera
from cg_frontier.render.pbr import PointLight


Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class CameraRelativeLightFamily:
    name: str
    local_position: Vector3
    color: Vector3
    radiant_intensity: float
    ambient_intensity: float
    role: str


def _vector3(value: object, label: str) -> Vector3:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"{label} must contain three values")
    return tuple(float(component) for component in value)  # type: ignore[return-value]


def parse_camera_relative_light_families(
    specs: Sequence[Mapping[str, object]],
) -> tuple[CameraRelativeLightFamily, ...]:
    """Parse and fail closed on the frozen six-family hemisphere policy."""

    families = tuple(
        CameraRelativeLightFamily(
            name=str(spec["name"]),
            local_position=_vector3(spec["local_position"], "local_position"),
            color=_vector3(spec["color"], "color"),
            radiant_intensity=float(spec["radiant_intensity"]),
            ambient_intensity=float(spec["ambient_intensity"]),
            role=str(spec["role"]),
        )
        for spec in specs
    )
    if len(families) != 6 or len({family.name for family in families}) != 6:
        raise ValueError("camera-relative light pool requires six unique families")
    camera_side = [family for family in families if family.role == "camera_side"]
    rims = [family for family in families if family.role == "rim"]
    if len(camera_side) < 4 or len(rims) > 1 or len(camera_side) + len(rims) != 6:
        raise ValueError("camera-relative pool requires camera-side lights and at most one rim")
    if any(family.local_position[2] <= 0.0 for family in camera_side):
        raise ValueError("camera-side lights require positive local camera-side depth")
    if any(family.local_position[2] >= 0.0 for family in rims):
        raise ValueError("rim lights require negative local camera-side depth")
    return families


def _subtract(left: Vector3, right: Vector3) -> Vector3:
    return tuple(a - b for a, b in zip(left, right, strict=True))  # type: ignore[return-value]


def _cross(left: Vector3, right: Vector3) -> Vector3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _normalize(value: Vector3) -> Vector3:
    length = math.sqrt(sum(component * component for component in value))
    if length <= 1.0e-12:
        raise ValueError("camera-relative frame requires non-degenerate vectors")
    return tuple(component / length for component in value)  # type: ignore[return-value]


def camera_local_to_world(camera: Camera, local_position: Vector3) -> Vector3:
    """Map right/up/camera-side coordinates around the camera's own target."""

    camera_side = _normalize(_subtract(camera.eye, camera.target))
    right = _normalize(_cross(camera.up, camera_side))
    corrected_up = _cross(camera_side, right)
    return tuple(
        camera.target[index]
        + local_position[0] * right[index]
        + local_position[1] * corrected_up[index]
        + local_position[2] * camera_side[index]
        for index in range(3)
    )  # type: ignore[return-value]


def build_camera_relative_light_grid(
    cameras: Sequence[Camera],
    families: Sequence[CameraRelativeLightFamily],
) -> tuple[tuple[PointLight, ...], ...]:
    """Build a family-major light grid aligned with the camera pool."""

    return tuple(
        tuple(
            PointLight(
                position=camera_local_to_world(camera, family.local_position),
                color=family.color,
                radiant_intensity=family.radiant_intensity,
                ambient_intensity=family.ambient_intensity,
            )
            for camera in cameras
        )
        for family in families
    )
