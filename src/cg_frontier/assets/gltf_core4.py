"""Explicit single-material glTF ingestion for generic Core-4 audits."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence
from urllib.parse import unquote, urlparse

import numpy as np
from PIL import Image
import torch

from cg_frontier.assets.gltf_mesh import GltfMesh, GltfMeshError, load_gltf_mesh
from cg_frontier.assets.preprocess import sha256_file, srgb_to_linear
from cg_frontier.compression.material import Core4Targets
from cg_frontier.render.gbuffer import Core4Textures


class GltfCore4Error(ValueError):
    """Raised when a source asset is outside the generic Core-4 contract."""


@dataclass(frozen=True)
class GltfCore4Asset:
    """One transformed mesh plus its seven-channel material truth."""

    name: str
    mesh: GltfMesh
    targets: Core4Targets
    textures: Core4Textures
    manifest: Mapping[str, Any]


def _require_index(items: Sequence[Any], value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value < len(items):
        raise GltfCore4Error(f"{label} must be a valid integer index")
    return value


def _external_image_path(root: Path, uri: Any, label: str) -> Path:
    if not isinstance(uri, str) or not uri:
        raise GltfCore4Error(f"{label} image must use a non-empty external URI")
    parsed = urlparse(uri)
    relative = PurePosixPath(unquote(parsed.path))
    if (
        parsed.scheme
        or parsed.netloc
        or parsed.query
        or parsed.fragment
        or relative.is_absolute()
        or ".." in relative.parts
    ):
        raise GltfCore4Error(f"{label} image URI must stay inside the glTF directory")
    path = (root / Path(*relative.parts)).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise GltfCore4Error(f"{label} image does not exist: {uri}")
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise GltfCore4Error(f"{label} image must be PNG or JPEG")
    return path


def _texture_path(
    document: Mapping[str, Any], root: Path, texture_info: Any, label: str
) -> tuple[Path, int, int]:
    if not isinstance(texture_info, Mapping):
        raise GltfCore4Error(f"material is missing {label} texture")
    if int(texture_info.get("texCoord", 0)) != 0:
        raise GltfCore4Error(f"{label} must use TEXCOORD_0")
    if "extensions" in texture_info:
        raise GltfCore4Error(f"{label} texture extensions are outside the audit contract")
    textures = document.get("textures", [])
    images = document.get("images", [])
    if not isinstance(textures, list) or not isinstance(images, list):
        raise GltfCore4Error("glTF textures and images must be arrays")
    texture_index = _require_index(textures, texture_info.get("index"), f"{label} texture")
    texture = textures[texture_index]
    if not isinstance(texture, Mapping):
        raise GltfCore4Error(f"{label} texture must be an object")
    image_index = _require_index(images, texture.get("source"), f"{label} image")
    image = images[image_index]
    if not isinstance(image, Mapping):
        raise GltfCore4Error(f"{label} image must be an object")
    return _external_image_path(root, image.get("uri"), label), texture_index, image_index


def _read_rgb(path: Path, expected_size: tuple[int, int], label: str) -> np.ndarray:
    try:
        with Image.open(path) as image:
            image.load()
            if image.size != expected_size:
                raise GltfCore4Error(
                    f"{label} size {image.size} does not match {expected_size}"
                )
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except OSError as error:
        raise GltfCore4Error(f"failed to read {label} image: {path.name}") from error


def load_gltf_core4_asset(
    gltf_path: Path | str,
    *,
    name: str | None = None,
    expected_size: tuple[int, int] = (2048, 2048),
    device: torch.device | str = "cpu",
) -> GltfCore4Asset:
    """Load a transformed one-primitive glTF and decode its Core-4 material once."""

    path = Path(gltf_path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GltfCore4Error(f"failed to read glTF: {path}") from error
    if not isinstance(document, Mapping):
        raise GltfCore4Error("glTF root must be an object")
    meshes = document.get("meshes", [])
    if not isinstance(meshes, list) or len(meshes) != 1:
        raise GltfCore4Error("Core-4 audit requires exactly one mesh")
    primitives = meshes[0].get("primitives", []) if isinstance(meshes[0], Mapping) else []
    if not isinstance(primitives, list) or len(primitives) != 1:
        raise GltfCore4Error("Core-4 audit requires exactly one primitive")
    primitive = primitives[0]
    if not isinstance(primitive, Mapping):
        raise GltfCore4Error("glTF primitive must be an object")
    materials = document.get("materials", [])
    if not isinstance(materials, list):
        raise GltfCore4Error("glTF materials must be an array")
    material_index = _require_index(materials, primitive.get("material"), "material")
    material = materials[material_index]
    if not isinstance(material, Mapping):
        raise GltfCore4Error("glTF material must be an object")
    pbr = material.get("pbrMetallicRoughness")
    if not isinstance(pbr, Mapping):
        raise GltfCore4Error("material must use metallic-roughness PBR")

    root = path.parent
    base_path, base_texture, base_image = _texture_path(
        document, root, pbr.get("baseColorTexture"), "base_color"
    )
    mr_path, mr_texture, mr_image = _texture_path(
        document, root, pbr.get("metallicRoughnessTexture"), "metallic_roughness"
    )
    normal_info = material.get("normalTexture")
    normal_path, normal_texture, normal_image = _texture_path(
        document, root, normal_info, "normal"
    )

    base_u8 = _read_rgb(base_path, expected_size, "base_color")
    mr_u8 = _read_rgb(mr_path, expected_size, "metallic_roughness")
    normal_u8 = _read_rgb(normal_path, expected_size, "normal")
    base_factor = np.asarray(pbr.get("baseColorFactor", (1.0, 1.0, 1.0, 1.0)), dtype=np.float64)
    if base_factor.shape != (4,) or not np.isfinite(base_factor).all():
        raise GltfCore4Error("baseColorFactor must contain four finite values")
    roughness_factor = float(pbr.get("roughnessFactor", 1.0))
    metallic_factor = float(pbr.get("metallicFactor", 1.0))
    normal_scale = float(normal_info.get("scale", 1.0))
    if not np.isfinite((roughness_factor, metallic_factor, normal_scale)).all():
        raise GltfCore4Error("material factors must be finite")

    base = srgb_to_linear(base_u8.astype(np.float64) / 255.0) * base_factor[:3]
    base = np.clip(base, 0.0, 1.0).astype(np.float32)
    roughness = np.clip(mr_u8[..., 1:2].astype(np.float32) / 255.0 * roughness_factor, 0.0, 1.0)
    metallic = np.clip(mr_u8[..., 2:3].astype(np.float32) / 255.0 * metallic_factor, 0.0, 1.0)
    normal = normal_u8.astype(np.float32) / 255.0 * 2.0 - 1.0
    normal[..., :2] *= normal_scale
    lengths = np.linalg.norm(normal, axis=-1, keepdims=True)
    if np.any(lengths < 1e-8):
        raise GltfCore4Error("normal texture contains a degenerate vector")
    normal /= lengths

    try:
        mesh = load_gltf_mesh(
            path, apply_node_transform=True, reconstruct_missing_tangents=True
        )
    except GltfMeshError as error:
        raise GltfCore4Error(str(error)) from error
    height, width = expected_size[1], expected_size[0]
    target_device = torch.device(device)
    base_tensor = torch.from_numpy(base).to(target_device).contiguous()
    normal_tensor = torch.from_numpy(normal).to(target_device).contiguous()
    roughness_tensor = torch.from_numpy(roughness).to(target_device).contiguous()
    metallic_tensor = torch.from_numpy(metallic).to(target_device).contiguous()
    hashes = {
        "gltf": sha256_file(path),
        "base_color": sha256_file(base_path),
        "metallic_roughness": sha256_file(mr_path),
        "normal": sha256_file(normal_path),
    }
    targets = Core4Targets(
        base_color_linear=base_tensor.reshape(-1, 3),
        normal_xyz=normal_tensor.reshape(-1, 3),
        roughness=roughness_tensor.reshape(-1, 1),
        metallic=metallic_tensor.reshape(-1, 1),
        height=height,
        width=width,
    )
    textures = Core4Textures(
        base_color_linear=base_tensor,
        normal=(normal_tensor + 1.0) * 0.5,
        roughness=roughness_tensor,
        metallic=metallic_tensor,
        source_hashes=hashes,
    )
    manifest = {
        "schema_version": 1,
        "asset": name or path.stem,
        "gltf": str(path.resolve()),
        "source_hashes": hashes,
        "texture_bindings": {
            "base_color": {"texture": base_texture, "image": base_image, "path": str(base_path.resolve()), "color_space": "sRGB decoded exactly once"},
            "metallic_roughness": {"texture": mr_texture, "image": mr_image, "path": str(mr_path.resolve()), "color_space": "linear", "roughness_channel": "G", "metallic_channel": "B"},
            "normal": {"texture": normal_texture, "image": normal_image, "path": str(normal_path.resolve()), "color_space": "linear", "y_flip": False},
        },
        "material_factors": {
            "base_color": base_factor.tolist(),
            "roughness": roughness_factor,
            "metallic": metallic_factor,
            "normal_scale": normal_scale,
        },
        "geometry": {
            "node_transform": "applied_to_positions_normals_and_tangents",
            "missing_tangent_policy": "deterministic_position_uv_reconstruction",
            "vertices": int(mesh.positions.shape[0]),
            "triangles": int(mesh.triangles.shape[0]),
            "bounds_min": mesh.bounds_min.tolist(),
            "bounds_max": mesh.bounds_max.tolist(),
        },
        "core4": {
            "resolution": [width, height],
            "channels": ["base_color_linear_rgb", "normal_tangent_xy", "roughness", "metallic"],
            "ambient_occlusion": "excluded",
            "emissive": "excluded",
        },
    }
    return GltfCore4Asset(name or path.stem, mesh, targets, textures, manifest)
