"""Fixed-camera nvdiffrast GBuffer for the SciFiHelmet Stage-B reference."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as torch_functional
import yaml
from PIL import Image

from cg_frontier.assets.gltf_mesh import GltfMesh, reconstruct_tangents
from cg_frontier.assets.preprocess import AssetValidationError, sha256_file


@dataclass(frozen=True)
class Camera:
    eye: tuple[float, float, float]
    target: tuple[float, float, float]
    up: tuple[float, float, float]
    vertical_fov_degrees: float
    near: float
    far: float


@dataclass(frozen=True)
class Core4Textures:
    base_color_linear: torch.Tensor
    normal: torch.Tensor
    roughness: torch.Tensor
    metallic: torch.Tensor
    source_hashes: Mapping[str, str]


@dataclass(frozen=True)
class GBufferResult:
    buffers: Mapping[str, np.ndarray]
    torch_buffers: Mapping[str, torch.Tensor]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class MaterialBuffers:
    """Complete per-pixel material consumed by the shared GGX shader."""

    base_color_linear: torch.Tensor
    normal_world: torch.Tensor
    roughness: torch.Tensor
    metallic: torch.Tensor
    normal_ts_raw: torch.Tensor | None = None
    normal_ts_unit: torch.Tensor | None = None
    normal_world_y_flipped: torch.Tensor | None = None
    normal_y_flip_angle_degrees: torch.Tensor | None = None


def _vector3(values: tuple[float, float, float], device: torch.device) -> torch.Tensor:
    tensor = torch.tensor(values, dtype=torch.float32, device=device)
    if tensor.shape != (3,) or not torch.isfinite(tensor).all():
        raise ValueError("camera vectors must contain three finite values")
    return tensor


def look_at_matrix(camera: Camera, device: torch.device | str = "cpu") -> torch.Tensor:
    """Return a right-handed OpenGL view matrix whose camera looks along -Z."""

    device = torch.device(device)
    eye = _vector3(camera.eye, device)
    target = _vector3(camera.target, device)
    up_hint = _vector3(camera.up, device)
    forward = torch_functional.normalize(target - eye, dim=0)
    side = torch_functional.normalize(torch.linalg.cross(forward, up_hint), dim=0)
    up = torch.linalg.cross(side, forward)
    view = torch.eye(4, dtype=torch.float32, device=device)
    view[0, :3] = side
    view[1, :3] = up
    view[2, :3] = -forward
    view[0, 3] = -torch.dot(side, eye)
    view[1, 3] = -torch.dot(up, eye)
    view[2, 3] = torch.dot(forward, eye)
    if not torch.isfinite(view).all():
        raise ValueError("camera eye, target, and up must form a valid frame")
    return view


def perspective_matrix(
    vertical_fov_degrees: float,
    aspect: float,
    near: float,
    far: float,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Return an OpenGL clip-space perspective projection matrix."""

    if not (0.0 < vertical_fov_degrees < 180.0):
        raise ValueError("vertical field of view must be between 0 and 180 degrees")
    if aspect <= 0.0 or near <= 0.0 or far <= near:
        raise ValueError("camera aspect/near/far values are invalid")
    focal = 1.0 / math.tan(math.radians(vertical_fov_degrees) * 0.5)
    projection = torch.zeros((4, 4), dtype=torch.float32, device=device)
    projection[0, 0] = focal / aspect
    projection[1, 1] = focal
    projection[2, 2] = (far + near) / (near - far)
    projection[2, 3] = (2.0 * far * near) / (near - far)
    projection[3, 2] = -1.0
    return projection


def orthonormal_tangent_frame(
    normal: torch.Tensor, tangent_with_handedness: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize N, Gram-Schmidt T against N, then apply glTF handedness to B."""

    normal_unit = torch_functional.normalize(normal, dim=-1, eps=1e-8)
    tangent = tangent_with_handedness[..., :3]
    tangent = tangent - normal_unit * torch.sum(tangent * normal_unit, dim=-1, keepdim=True)
    tangent_unit = torch_functional.normalize(tangent, dim=-1, eps=1e-8)
    handedness = torch.where(
        tangent_with_handedness[..., 3:4] < 0.0,
        -torch.ones_like(tangent_with_handedness[..., 3:4]),
        torch.ones_like(tangent_with_handedness[..., 3:4]),
    )
    bitangent_unit = torch.linalg.cross(normal_unit, tangent_unit, dim=-1) * handedness
    return normal_unit, tangent_unit, bitangent_unit


def _read_texture(path: Path, channels: int) -> np.ndarray:
    try:
        with Image.open(path) as image:
            image.load()
            pixels = np.array(image, dtype=np.uint8, copy=True)
    except OSError as error:
        raise AssetValidationError(f"failed to load Core-4 texture: {path}") from error
    if pixels.ndim == 2:
        pixels = pixels[..., None]
    if pixels.ndim != 3 or pixels.shape[-1] < channels:
        raise AssetValidationError(f"Core-4 texture has too few channels: {path.name}")
    return pixels[..., :channels]


def srgb_to_linear_torch(encoded: torch.Tensor) -> torch.Tensor:
    """Decode normalized sRGB values once while preserving autograd."""

    if torch.any((encoded < 0.0) | (encoded > 1.0)):
        raise ValueError("sRGB tensor values must stay within [0, 1]")
    return torch.where(
        encoded <= 0.04045,
        encoded / 12.92,
        torch.pow((encoded + 0.055) / 1.055, 2.4),
    )


def load_core4_textures(
    manifest_path: Path | str, device: torch.device | str
) -> Core4Textures:
    """Load Stage-A outputs while preserving glTF's top-left texture origin.

    nvdiffrast treats tensor row zero as the logical bottom row. Keeping the
    top-down PNG array unchanged intentionally maps glTF v=0 to PNG row zero.
    Raster outputs are flipped separately when converted back to top-down files.
    """

    manifest_path = Path(manifest_path)
    try:
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        raise AssetValidationError(f"failed to read Core-4 manifest: {manifest_path}") from error
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != 1:
        raise AssetValidationError("unsupported Core-4 manifest schema")
    output_root = manifest.get("output_root")
    outputs = manifest.get("outputs")
    if not isinstance(output_root, str) or not isinstance(outputs, Mapping):
        raise AssetValidationError("Core-4 manifest is missing output_root or outputs")
    root = (manifest_path.parent / output_root).resolve()
    loaded: dict[str, torch.Tensor] = {}
    hashes: dict[str, str] = {}
    channel_counts = {
        "base_color": 3,
        "normal": 3,
        "roughness": 1,
        "metallic": 1,
    }
    for semantic, channels in channel_counts.items():
        metadata = outputs.get(semantic)
        if not isinstance(metadata, Mapping) or not isinstance(metadata.get("uri"), str):
            raise AssetValidationError(f"Core-4 manifest is missing {semantic} output")
        path = (root / metadata["uri"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise AssetValidationError(f"Core-4 output path is invalid: {semantic}")
        digest = sha256_file(path)
        if digest != metadata.get("sha256"):
            raise AssetValidationError(f"Core-4 output hash mismatch: {semantic}")
        pixels = _read_texture(path, channels)
        tensor = torch.from_numpy(pixels).to(
            device=device, dtype=torch.float32
        ) / 255.0
        loaded[semantic] = srgb_to_linear_torch(tensor) if semantic == "base_color" else tensor
        hashes[semantic] = digest
    return Core4Textures(
        base_color_linear=loaded["base_color"],
        normal=loaded["normal"],
        roughness=loaded["roughness"],
        metallic=loaded["metallic"],
        source_hashes=hashes,
    )


def _transform_points(points: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    homogeneous = torch.cat(
        [points, torch.ones((points.shape[0], 1), dtype=points.dtype, device=points.device)],
        dim=1,
    )
    return homogeneous @ matrix.T


def _vector_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean(dtype=np.float64)),
    }


def _handedness_counts(tangents: np.ndarray) -> dict[str, int]:
    return {
        "negative": int(np.count_nonzero(tangents[:, 3] < 0.0)),
        "positive": int(np.count_nonzero(tangents[:, 3] > 0.0)),
    }


def select_render_tangents(
    mesh: GltfMesh, *, source: str = "reconstructed_uv"
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select an explicit, validated tangent lineage for GBuffer rendering."""

    if source == "reconstructed_uv":
        tangents = reconstruct_tangents(mesh)
        return tangents, {
            "render_source": source,
            "uv_reconstruction_attempted": True,
        }
    if source != "source_gltf":
        raise ValueError(f"unsupported tangent source: {source}")
    tangents = mesh.tangents
    if (
        not isinstance(tangents, np.ndarray)
        or tangents.shape != (mesh.positions.shape[0], 4)
        or not np.isfinite(tangents).all()
        or np.any(np.linalg.norm(tangents[:, :3], axis=1) < 1e-8)
        or np.any(np.abs(np.abs(tangents[:, 3]) - 1.0) > 1e-6)
    ):
        raise ValueError("source glTF tangents are missing or invalid")
    return tangents, {
        "render_source": source,
        "uv_reconstruction_attempted": False,
    }


def render_geometry_gbuffer(
    mesh: GltfMesh,
    camera: Camera,
    resolution: tuple[int, int],
    *,
    device: torch.device | str = "cuda",
    cull_backfaces: bool = True,
    tangent_source: str = "reconstructed_uv",
) -> GBufferResult:
    """Rasterize only geometry/TBN/UV into top-down buffers, without material sampling."""

    import nvdiffrast.torch as dr

    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("nvdiffrast Stage-B GBuffer requires a CUDA device")
    height, width = resolution
    if height <= 0 or width <= 0:
        raise ValueError("GBuffer resolution must be positive")

    render_tangents, tangent_selection = select_render_tangents(
        mesh, source=tangent_source
    )
    all_triangles = mesh.triangles.astype(np.int64, copy=False)
    face_positions = mesh.positions[all_triangles]
    face_cross = np.cross(
        face_positions[:, 1] - face_positions[:, 0],
        face_positions[:, 2] - face_positions[:, 0],
    )
    face_lengths = np.linalg.norm(face_cross, axis=1)
    if np.any(face_lengths < 1e-12):
        raise ValueError("mesh contains degenerate triangles")
    face_normals = face_cross / face_lengths[:, None]
    face_centers = face_positions.mean(axis=1)
    camera_eye = np.asarray(camera.eye, dtype=np.float32)
    # Cull in the asset's unchanged right-handed world space before rasterization.
    front_facing = np.sum(face_normals * (camera_eye - face_centers), axis=1) > 0.0
    selected = front_facing if cull_backfaces else np.ones_like(front_facing)
    selected_indices = np.flatnonzero(selected).astype(np.int32)
    triangles = np.ascontiguousarray(mesh.triangles[selected], dtype=np.int32)
    selected_face_normals = np.ascontiguousarray(face_normals[selected], dtype=np.float32)
    if triangles.size == 0:
        raise ValueError("back-face culling removed every triangle")

    positions = torch.from_numpy(mesh.positions).to(device=device)
    normals = torch.from_numpy(mesh.normals).to(device=device)
    tangents = torch.from_numpy(render_tangents).to(device=device)
    texcoords = torch.from_numpy(mesh.texcoords).to(device=device)
    triangle_tensor = torch.from_numpy(triangles).to(device=device)
    face_normal_tensor = torch.from_numpy(selected_face_normals).to(device=device)
    selected_index_tensor = torch.from_numpy(selected_indices).to(device=device)

    view = look_at_matrix(camera, device)
    projection = perspective_matrix(
        camera.vertical_fov_degrees, width / height, camera.near, camera.far, device
    )
    view_positions = _transform_points(positions, view)
    clip_positions = _transform_points(positions, projection @ view)[None, ...]
    context = dr.RasterizeCudaContext(device=device)
    raster, _ = dr.rasterize(
        context, clip_positions, triangle_tensor, resolution=[height, width]
    )
    mask = raster[..., 3:4] > 0.0

    position_pixel, _ = dr.interpolate(positions[None, ...], raster, triangle_tensor)
    view_position_pixel, _ = dr.interpolate(
        view_positions[None, :, :3].contiguous(), raster, triangle_tensor
    )
    uv_pixel, _ = dr.interpolate(texcoords[None, ...], raster, triangle_tensor)
    normal_vertex_pixel, _ = dr.interpolate(normals[None, ...], raster, triangle_tensor)
    tangent_pixel, _ = dr.interpolate(tangents[None, ...], raster, triangle_tensor)
    normal_vertex_unit, tangent_unit, bitangent_unit = orthonormal_tangent_frame(
        normal_vertex_pixel, tangent_pixel
    )

    local_triangle_id = raster[..., 3].to(torch.int64) - 1
    safe_triangle_id = local_triangle_id.clamp(0, triangles.shape[0] - 1)
    geometric_normal = face_normal_tensor[safe_triangle_id]
    original_triangle_id = selected_index_tensor[safe_triangle_id]

    depth = -view_position_pixel[..., 2:3]

    vector_buffers = {
        "position_world": position_pixel,
        "uv": uv_pixel,
        "geometric_normal_world": geometric_normal,
        "vertex_normal_world": normal_vertex_unit,
        "tangent_world": tangent_unit,
        "bitangent_world": bitangent_unit,
    }
    scalar_buffers = {
        "depth_camera": depth,
    }
    arrays: dict[str, np.ndarray] = {}
    torch_arrays: dict[str, torch.Tensor] = {}
    # nvdiffrast raster rows are bottom-up.  Flip exactly once here so every
    # exported and downstream in-memory GBuffer uses conventional top-down rows.
    for name, value in vector_buffers.items():
        masked = torch.where(mask, value, torch.zeros_like(value))
        top_down = torch.flip(masked[0], dims=(0,))
        torch_arrays[name] = top_down
        arrays[name] = top_down.detach().cpu().numpy().astype(np.float32)
    for name, value in scalar_buffers.items():
        masked = torch.where(mask, value, torch.zeros_like(value))
        top_down = torch.flip(masked[0, ..., 0], dims=(0,))
        torch_arrays[name] = top_down
        arrays[name] = top_down.detach().cpu().numpy().astype(np.float32)
    mask_top_down = torch.flip(mask[0, ..., 0], dims=(0,))
    triangle_id_top_down = torch.flip(
        torch.where(mask[..., 0], original_triangle_id, -1)[0], dims=(0,)
    )
    torch_arrays["mask"] = mask_top_down
    torch_arrays["triangle_id"] = triangle_id_top_down
    arrays["mask"] = mask_top_down.detach().cpu().numpy()
    arrays["triangle_id"] = triangle_id_top_down.detach().cpu().numpy().astype(np.int32)

    valid = arrays["mask"]
    source_dot = np.sum(mesh.normals * mesh.tangents[:, :3], axis=1)
    render_dot = np.sum(mesh.normals * render_tangents[:, :3], axis=1)
    n_dot_t = np.sum(
        arrays["vertex_normal_world"][valid] * arrays["tangent_world"][valid], axis=1
    )
    n_dot_b = np.sum(
        arrays["vertex_normal_world"][valid] * arrays["bitangent_world"][valid], axis=1
    )
    t_dot_b = np.sum(
        arrays["tangent_world"][valid] * arrays["bitangent_world"][valid], axis=1
    )
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "coordinate_system": "glTF right-handed, +Y up, asset front +Z",
        "image_memory": {
            "texture_input": "top-down PNG rows kept unchanged so glTF v=0 samples row 0",
            "raster_internal": "nvdiffrast bottom-up",
            "saved_arrays": "top-down after one vertical flip",
        },
        "camera": {
            "eye": list(camera.eye),
            "target": list(camera.target),
            "up": list(camera.up),
            "vertical_fov_degrees": camera.vertical_fov_degrees,
            "near": camera.near,
            "far": camera.far,
        },
        "resolution": [height, width],
        "mesh": {
            "vertex_count": int(mesh.positions.shape[0]),
            "triangle_count": int(mesh.triangles.shape[0]),
            "rendered_triangle_count": int(triangles.shape[0]),
            "backface_culling": cull_backfaces,
            "bounds_min": mesh.bounds_min.tolist(),
            "bounds_max": mesh.bounds_max.tolist(),
        },
        "tangent_basis": {
            **tangent_selection,
            "formula": "B = cross(N, T) * tangent.w",
            "source_abs_normal_dot_tangent": _vector_stats(np.abs(source_dot)),
            "source_handedness": _handedness_counts(mesh.tangents),
            "render_abs_normal_dot_tangent": _vector_stats(np.abs(render_dot)),
            "render_handedness": _handedness_counts(render_tangents),
            **(
                {
                    "reconstructed_abs_normal_dot_tangent": _vector_stats(
                        np.abs(render_dot)
                    ),
                    "reconstructed_handedness": _handedness_counts(render_tangents),
                }
                if tangent_source == "reconstructed_uv"
                else {}
            ),
            "pixel_max_abs_dot": {
                "N_T": float(np.max(np.abs(n_dot_t))),
                "N_B": float(np.max(np.abs(n_dot_b))),
                "T_B": float(np.max(np.abs(t_dot_b))),
            },
        },
        "coverage": {
            "pixels": int(np.count_nonzero(valid)),
            "ratio": float(np.mean(valid)),
        },
        "valid_pixel_statistics": {
            "depth_camera": _vector_stats(arrays["depth_camera"][valid]),
        },
    }
    return GBufferResult(buffers=arrays, torch_buffers=torch_arrays, metadata=metadata)


def tangent_normal_to_world(
    geometry: GBufferResult, normal_ts: torch.Tensor
) -> torch.Tensor:
    """Transform a glTF/OpenGL +Y tangent normal through the shared pixel TBN."""

    buffers = geometry.torch_buffers
    required = {"vertex_normal_world", "tangent_world", "bitangent_world"}
    missing = sorted(required.difference(buffers))
    if missing:
        raise ValueError(f"geometry GBuffer is missing TBN inputs: {', '.join(missing)}")
    # The bitangent already contains glTF tangent.w, so +Y is preserved here.
    normal_unit = torch_functional.normalize(normal_ts, dim=-1, eps=1e-8)
    return torch_functional.normalize(
        buffers["tangent_world"] * normal_unit[..., 0:1]
        + buffers["bitangent_world"] * normal_unit[..., 1:2]
        + buffers["vertex_normal_world"] * normal_unit[..., 2:3],
        dim=-1,
        eps=1e-8,
    )


def sample_core4_material(
    geometry: GBufferResult, textures: Core4Textures
) -> MaterialBuffers:
    """Sample original Core-4 at geometry UVs without changing the frozen row convention."""

    import nvdiffrast.torch as dr

    buffers = geometry.torch_buffers
    if "uv" not in buffers or "mask" not in buffers:
        raise ValueError("geometry GBuffer is missing UV or mask")
    uv = buffers["uv"][None, ...]

    def sample(texture: torch.Tensor) -> torch.Tensor:
        # Reference sampling is wrap + bilinear, matching the faithful UE
        # baseline.  Textures remain top-down per load_core4_textures().
        return dr.texture(
            texture[None, ...], uv, filter_mode="linear", boundary_mode="wrap"
        )[0]

    base_color_linear = sample(textures.base_color_linear)
    normal_ts_raw = sample(textures.normal) * 2.0 - 1.0
    normal_ts_unit = torch_functional.normalize(normal_ts_raw, dim=-1, eps=1e-8)
    normal_world = tangent_normal_to_world(geometry, normal_ts_unit)
    # This flipped variant is diagnostic only.  The training truth remains +Y;
    # the single UE convention conversion belongs at the engine boundary.
    normal_ts_y_flipped = torch.cat(
        (normal_ts_raw[..., 0:1], -normal_ts_raw[..., 1:2], normal_ts_raw[..., 2:3]),
        dim=-1,
    )
    normal_world_y_flipped = tangent_normal_to_world(geometry, normal_ts_y_flipped)
    variant_dot = torch.sum(normal_world * normal_world_y_flipped, dim=-1)
    angle = torch.rad2deg(torch.acos(variant_dot.clamp(-1.0, 1.0)))
    roughness = sample(textures.roughness)[..., 0]
    metallic = sample(textures.metallic)[..., 0]
    mask = buffers["mask"]

    def masked_vector(value: torch.Tensor) -> torch.Tensor:
        return torch.where(mask[..., None], value, torch.zeros_like(value))

    def masked_scalar(value: torch.Tensor) -> torch.Tensor:
        return torch.where(mask, value, torch.zeros_like(value))

    return MaterialBuffers(
        base_color_linear=masked_vector(base_color_linear),
        normal_world=masked_vector(normal_world),
        roughness=masked_scalar(roughness),
        metallic=masked_scalar(metallic),
        normal_ts_raw=masked_vector(normal_ts_raw),
        normal_ts_unit=masked_vector(normal_ts_unit),
        normal_world_y_flipped=masked_vector(normal_world_y_flipped),
        normal_y_flip_angle_degrees=masked_scalar(angle),
    )


def material_from_gbuffer(gbuffer: GBufferResult) -> MaterialBuffers:
    """Read a complete material from the legacy combined GBuffer representation."""

    buffers = gbuffer.torch_buffers
    required = {"base_color_linear", "normal_world", "roughness", "metallic"}
    missing = sorted(required.difference(buffers))
    if missing:
        raise ValueError(f"GBuffer is missing material inputs: {', '.join(missing)}")
    return MaterialBuffers(
        base_color_linear=buffers["base_color_linear"],
        normal_world=buffers["normal_world"],
        roughness=buffers["roughness"],
        metallic=buffers["metallic"],
        normal_ts_raw=buffers.get("normal_ts_raw"),
        normal_ts_unit=buffers.get("normal_ts_unit"),
        normal_world_y_flipped=buffers.get("normal_world_y_flipped"),
        normal_y_flip_angle_degrees=buffers.get("normal_y_flip_angle_degrees"),
    )


def compose_material_gbuffer(
    geometry: GBufferResult,
    material: MaterialBuffers,
    *,
    source_hashes: Mapping[str, str] | None = None,
) -> GBufferResult:
    """Create the backward-compatible combined view without resampling geometry."""

    torch_buffers = dict(geometry.torch_buffers)
    optional = {
        "normal_ts_raw": material.normal_ts_raw,
        "normal_ts_unit": material.normal_ts_unit,
        "normal_world_y_flipped": material.normal_world_y_flipped,
        "normal_y_flip_angle_degrees": material.normal_y_flip_angle_degrees,
    }
    torch_buffers.update(
        {
            "base_color_linear": material.base_color_linear,
            "normal_world": material.normal_world,
            "roughness": material.roughness,
            "metallic": material.metallic,
            **{name: value for name, value in optional.items() if value is not None},
        }
    )
    arrays = dict(geometry.buffers)
    for name, value in torch_buffers.items():
        if name in arrays:
            continue
        dtype = np.int32 if name == "triangle_id" else None
        array = value.detach().cpu().numpy()
        arrays[name] = array.astype(dtype or np.float32, copy=False)

    metadata = dict(geometry.metadata)
    metadata["textures"] = dict(source_hashes or {})
    valid = arrays["mask"]
    stats = dict(metadata.get("valid_pixel_statistics", {}))
    stats.update(
        {
            "base_color_linear": _vector_stats(arrays["base_color_linear"][valid]),
            "roughness": _vector_stats(arrays["roughness"][valid]),
            "metallic": _vector_stats(arrays["metallic"][valid]),
        }
    )
    if "normal_ts_raw" in arrays:
        stats["normal_ts_raw_length"] = _vector_stats(
            np.linalg.norm(arrays["normal_ts_raw"][valid], axis=1)
        )
    if "normal_y_flip_angle_degrees" in arrays:
        stats["normal_y_flip_angle_degrees"] = _vector_stats(
            arrays["normal_y_flip_angle_degrees"][valid]
        )
    metadata["valid_pixel_statistics"] = stats
    return GBufferResult(buffers=arrays, torch_buffers=torch_buffers, metadata=metadata)


def render_gbuffer(
    mesh: GltfMesh,
    textures: Core4Textures,
    camera: Camera,
    resolution: tuple[int, int],
    *,
    device: torch.device | str = "cuda",
    cull_backfaces: bool = True,
) -> GBufferResult:
    """Compatibility wrapper: geometry rasterization followed by Core-4 material sampling."""

    geometry = render_geometry_gbuffer(
        mesh,
        camera,
        resolution,
        device=device,
        cull_backfaces=cull_backfaces,
    )
    material = sample_core4_material(geometry, textures)
    return compose_material_gbuffer(
        geometry, material, source_hashes=textures.source_hashes
    )


def _save_png(path: Path, values: np.ndarray) -> None:
    encoded = np.rint(np.clip(values, 0.0, 1.0) * 255.0).astype(np.uint8)
    Image.fromarray(encoded).save(path, format="PNG")


def _linear_to_srgb_numpy(linear: np.ndarray) -> np.ndarray:
    linear = np.clip(linear, 0.0, 1.0)
    return np.where(
        linear <= 0.0031308,
        linear * 12.92,
        1.055 * np.power(linear, 1.0 / 2.4) - 0.055,
    )


def export_gbuffer(
    result: GBufferResult, output_dir: Path | str, mesh: GltfMesh
) -> dict[str, Any]:
    """Save exact arrays plus human-readable diagnostic PNGs and metadata."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}
    for name, array in result.buffers.items():
        path = output_dir / f"{name}.npy"
        np.save(path, array, allow_pickle=False)
        files[f"{name}.npy"] = {
            "sha256": sha256_file(path),
            "shape": list(array.shape),
            "dtype": str(array.dtype),
        }

    mask = result.buffers["mask"]
    depth = result.buffers["depth_camera"]
    valid_depth = depth[mask]
    depth_visual = np.zeros_like(depth)
    depth_span = float(valid_depth.max() - valid_depth.min())
    if depth_span > 0.0:
        depth_visual[mask] = (valid_depth.max() - valid_depth) / depth_span
    bounds_min = mesh.bounds_min
    bounds_span = np.maximum(mesh.bounds_max - bounds_min, 1e-8)
    position_visual = (result.buffers["position_world"] - bounds_min) / bounds_span
    vector_visuals = {
        "base_color": _linear_to_srgb_numpy(result.buffers["base_color_linear"]),
        "uv": np.concatenate(
            [result.buffers["uv"], np.zeros((*result.buffers["uv"].shape[:2], 1), dtype=np.float32)],
            axis=2,
        ),
        "position_world": position_visual,
        "geometric_normal_world": result.buffers["geometric_normal_world"] * 0.5 + 0.5,
        "vertex_normal_world": result.buffers["vertex_normal_world"] * 0.5 + 0.5,
        "tangent_world": result.buffers["tangent_world"] * 0.5 + 0.5,
        "bitangent_world": result.buffers["bitangent_world"] * 0.5 + 0.5,
        "normal_ts_raw": result.buffers["normal_ts_raw"] * 0.5 + 0.5,
        "normal_world": result.buffers["normal_world"] * 0.5 + 0.5,
        "normal_world_y_flipped": result.buffers["normal_world_y_flipped"] * 0.5 + 0.5,
    }
    scalar_visuals = {
        "mask": mask.astype(np.float32),
        "depth_camera": depth_visual,
        "roughness": result.buffers["roughness"],
        "metallic": result.buffers["metallic"],
        "normal_y_flip_angle": result.buffers["normal_y_flip_angle_degrees"] / 180.0,
    }
    for name, values in {**vector_visuals, **scalar_visuals}.items():
        if values.ndim == 3:
            values = np.where(mask[..., None], values, 0.0)
        else:
            values = np.where(mask, values, 0.0)
        path = output_dir / f"{name}.png"
        _save_png(path, values)
        files[path.name] = {"sha256": sha256_file(path)}

    metadata = dict(result.metadata)
    metadata["files"] = files
    metadata_path = output_dir / "gbuffer.yaml"
    text = yaml.safe_dump(
        metadata, sort_keys=False, allow_unicode=True, default_flow_style=False
    )
    with metadata_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
    return metadata
