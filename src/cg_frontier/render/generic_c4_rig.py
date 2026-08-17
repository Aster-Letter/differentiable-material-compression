"""Deterministic normalized camera/light specification for generic C4 assets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Literal, Mapping

import numpy as np

from cg_frontier.render.gbuffer import Camera
from cg_frontier.render.pbr import PointLight


def _unit(values: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(sum(value * value for value in values))
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("rig direction must be finite and non-zero")
    return tuple(value / length for value in values)


@dataclass(frozen=True)
class RigCamera:
    camera_id: str
    direction: tuple[float, float, float]
    split: Literal["train", "audit"]


@dataclass(frozen=True)
class RigLight:
    light_id: str
    frame: Literal["world", "camera"]
    direction: tuple[float, float, float]
    color: tuple[float, float, float]
    radiant_intensity_per_radius_squared: float
    ambient_intensity: float


@dataclass(frozen=True)
class GenericC4Rig:
    cameras: tuple[RigCamera, ...]
    lights: tuple[RigLight, ...]
    presentation_view_indices: tuple[tuple[str, int], ...]
    resolution: tuple[int, int]
    rig_hash: str

    @property
    def presentation_views(self) -> Mapping[str, int]:
        return dict(self.presentation_view_indices)


def build_generic_c4_rig() -> GenericC4Rig:
    """Build the frozen 31-camera, 6-light normalized-bounds rig."""

    audit_indices = frozenset((2, 6, 10, 14, 18, 22, 26))
    presentation = {"front": 0, "rear": 4, "upper_side": 8, "top": 12}
    overrides = {
        0: (0.0, 0.0, 1.0),
        4: (0.0, 0.0, -1.0),
        8: (1.0, 0.65, 1.0),
        12: (0.0, 1.0, 0.05),
    }
    cameras: list[RigCamera] = []
    golden_angle = math.pi * (3.0 - math.sqrt(5.0))
    for index in range(31):
        y = 1.0 - 2.0 * (index + 0.5) / 31.0
        radial = math.sqrt(max(0.0, 1.0 - y * y))
        azimuth = index * golden_angle
        direction = overrides.get(
            index,
            (radial * math.sin(azimuth), y, radial * math.cos(azimuth)),
        )
        cameras.append(
            RigCamera(
                camera_id=f"camera_{index:02d}",
                direction=_unit(direction),
                split="audit" if index in audit_indices else "train",
            )
        )
    light_rows = (
        ("world_key", "world", (1.7, 2.1, 2.8), (1.0, 0.98, 0.95), 24.0, 0.055),
        ("world_fill", "world", (-2.2, 1.0, 1.3), (0.78, 0.88, 1.0), 13.0, 0.04),
        ("world_grazing", "world", (0.4, 0.3, -2.8), (1.0, 0.82, 0.7), 18.0, 0.035),
        ("camera_key", "camera", (1.2, 1.4, 2.4), (1.0, 0.98, 0.95), 22.0, 0.05),
        ("camera_fill", "camera", (-1.7, 0.6, 1.8), (0.78, 0.88, 1.0), 12.0, 0.04),
        ("camera_grazing", "camera", (1.8, 0.15, 0.35), (1.0, 0.82, 0.7), 17.0, 0.035),
    )
    lights = tuple(
        RigLight(
            light_id=light_id,
            frame=frame,
            direction=_unit(direction),
            color=color,
            radiant_intensity_per_radius_squared=intensity,
            ambient_intensity=ambient,
        )
        for light_id, frame, direction, color, intensity, ambient in light_rows
    )
    identity = {
        "schema_version": 1,
        "normalized_bounds": "bounding_sphere",
        "resolution": [256, 256],
        "cameras": [asdict(camera) for camera in cameras],
        "lights": [asdict(light) for light in lights],
        "presentation_views": presentation,
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("ascii")
    return GenericC4Rig(
        cameras=tuple(cameras),
        lights=lights,
        presentation_view_indices=tuple(presentation.items()),
        resolution=(256, 256),
        rig_hash=hashlib.sha256(payload).hexdigest(),
    )


def instantiate_camera(
    mesh: object,
    camera: RigCamera,
    *,
    vertical_fov_degrees: float = 38.0,
    distance_padding: float = 1.18,
) -> tuple[Camera, tuple[float, float, float], float]:
    """Map one normalized rig direction onto an asset bounding sphere."""

    lower = np.asarray(getattr(mesh, "bounds_min"), dtype=np.float64)
    upper = np.asarray(getattr(mesh, "bounds_max"), dtype=np.float64)
    positions = np.asarray(getattr(mesh, "positions"), dtype=np.float64)
    if lower.shape != (3,) or upper.shape != (3,) or positions.ndim != 2:
        raise ValueError("mesh does not expose finite 3D bounds")
    center = (lower + upper) * 0.5
    radius = float(np.max(np.linalg.norm(positions - center, axis=1)))
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("mesh bounding sphere must be finite and non-empty")
    if not 1.0 < vertical_fov_degrees < 179.0 or distance_padding <= 1.0:
        raise ValueError("invalid generic rig camera settings")
    direction = np.asarray(camera.direction, dtype=np.float64)
    distance = radius / math.sin(math.radians(vertical_fov_degrees) * 0.5) * distance_padding
    eye = center + direction * distance
    world_up = np.array((0.0, 1.0, 0.0), dtype=np.float64)
    if abs(float(np.dot(direction, world_up))) > 0.95:
        world_up = np.array((0.0, 0.0, 1.0), dtype=np.float64)
    value = Camera(
        eye=tuple(float(component) for component in eye),
        target=tuple(float(component) for component in center),
        up=tuple(float(component) for component in world_up),
        vertical_fov_degrees=float(vertical_fov_degrees),
        near=max(radius * 0.01, 1.0e-4),
        far=distance + radius * 3.0,
    )
    return value, tuple(float(component) for component in center), radius


def instantiate_lights(
    camera: Camera,
    center: tuple[float, float, float],
    radius: float,
    rig: GenericC4Rig,
) -> tuple[PointLight, ...]:
    """Instantiate the three world and three camera-relative rig lights."""

    center_vector = np.asarray(center, dtype=np.float64)
    outward = np.asarray(camera.eye, dtype=np.float64) - center_vector
    outward /= np.linalg.norm(outward)
    up_hint = np.asarray(camera.up, dtype=np.float64)
    forward = -outward
    right = np.cross(forward, up_hint)
    right /= np.linalg.norm(right)
    camera_up = np.cross(right, forward)
    lights: list[PointLight] = []
    for spec in rig.lights:
        direction = np.asarray(spec.direction, dtype=np.float64)
        if spec.frame == "camera":
            direction = (
                right * direction[0]
                + camera_up * direction[1]
                + outward * direction[2]
            )
            direction /= np.linalg.norm(direction)
        position = center_vector + direction * radius * 3.0
        lights.append(
            PointLight(
                position=tuple(float(component) for component in position),
                color=spec.color,
                radiant_intensity=spec.radiant_intensity_per_radius_squared
                * radius
                * radius,
                ambient_intensity=spec.ambient_intensity,
            )
        )
    return tuple(lights)
