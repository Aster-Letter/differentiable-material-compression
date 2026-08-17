"""Deterministic Core-4 texture preprocessing for the SciFiHelmet glTF asset.

This module deliberately stops at glTF metadata and PNG pixels. It does not load
mesh buffers, construct a GBuffer, or perform any rendering.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence
from urllib.parse import unquote, urlparse

import numpy as np
import yaml
from PIL import Image


class AssetValidationError(ValueError):
    """Raised when the source asset violates the frozen Stage-A contract."""


_SEMANTIC_PATHS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("base_color", ("pbrMetallicRoughness", "baseColorTexture")),
    (
        "metallic_roughness",
        ("pbrMetallicRoughness", "metallicRoughnessTexture"),
    ),
    ("normal", ("normalTexture",)),
    ("ambient_occlusion", ("occlusionTexture",)),
)

_ALLOWED_MODES: Mapping[str, tuple[str, ...]] = {
    "base_color": ("RGB", "RGBA"),
    "metallic_roughness": ("RGB", "RGBA"),
    "normal": ("RGB", "RGBA"),
    "ambient_occlusion": ("L", "LA", "RGB", "RGBA"),
}

_REQUIRED_ATTRIBUTES = ("POSITION", "NORMAL", "TANGENT", "TEXCOORD_0")


def sha256_file(path: Path | str) -> str:
    """Return the lowercase SHA-256 digest of a file without loading it at once."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def srgb_to_linear(values: np.ndarray | Sequence[float]) -> np.ndarray:
    """Decode normalized sRGB values to linear exactly once.

    Inputs and outputs are floating point arrays. This function is intentionally
    separate from PNG preprocessing: BaseColor stays sRGB-encoded on disk.
    """

    encoded = np.asarray(values, dtype=np.float64)
    if np.any((encoded < 0.0) | (encoded > 1.0)):
        raise AssetValidationError("sRGB values must be within [0, 1]")
    return np.where(
        encoded <= 0.04045,
        encoded / 12.92,
        ((encoded + 0.055) / 1.055) ** 2.4,
    )


def decode_normal_unorm8(rgb: np.ndarray) -> np.ndarray:
    """Decode RGB UNORM8 to [-1, 1] without normalization or Y flipping."""

    array = np.asarray(rgb)
    if array.dtype != np.uint8:
        raise AssetValidationError("normal pixels must be uint8")
    if array.ndim < 1 or array.shape[-1] < 3:
        raise AssetValidationError("normal pixels must have at least three channels")
    return array[..., :3].astype(np.float64) * (2.0 / 255.0) - 1.0


def _require_index(items: Sequence[Any], index: Any, label: str) -> int:
    if not isinstance(index, int) or isinstance(index, bool):
        raise AssetValidationError(f"{label} must be an integer index")
    if index < 0 or index >= len(items):
        raise AssetValidationError(f"{label} index {index} is out of range")
    return index


def _nested_mapping(root: Mapping[str, Any], keys: Sequence[str], label: str) -> Mapping[str, Any]:
    value: Any = root
    for key in keys:
        if not isinstance(value, Mapping) or key not in value:
            raise AssetValidationError(f"missing glTF texture semantic: {label}")
        value = value[key]
    if not isinstance(value, Mapping):
        raise AssetValidationError(f"glTF texture semantic {label} must be an object")
    return value


def _safe_image_path(gltf_dir: Path, uri: Any, semantic: str) -> tuple[str, Path]:
    if not isinstance(uri, str) or not uri:
        raise AssetValidationError(f"{semantic} image must use a non-empty external URI")
    parsed = urlparse(uri)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        raise AssetValidationError(f"{semantic} image URI must be a local relative path")
    uri_path = PurePosixPath(unquote(parsed.path))
    if uri_path.is_absolute() or ".." in uri_path.parts:
        raise AssetValidationError(f"{semantic} image URI must be relative")
    relative = Path(*uri_path.parts)
    candidate = (gltf_dir / relative).resolve()
    root = gltf_dir.resolve()
    if not candidate.is_relative_to(root):
        raise AssetValidationError(f"{semantic} image URI escapes the glTF directory")
    if candidate.suffix.lower() != ".png":
        raise AssetValidationError(f"{semantic} image must be a PNG: {uri}")
    if not candidate.is_file():
        raise AssetValidationError(f"{semantic} image does not exist: {uri}")
    return relative.as_posix(), candidate


def _texture_bindings(
    document: Mapping[str, Any], gltf_dir: Path, material_index: int
) -> dict[str, dict[str, Any]]:
    """Resolve the four source semantics without decoding their color spaces."""

    materials = document.get("materials", [])
    textures = document.get("textures", [])
    images = document.get("images", [])
    if not all(isinstance(values, list) for values in (materials, textures, images)):
        raise AssetValidationError("glTF materials, textures, and images must be arrays")
    material_index = _require_index(materials, material_index, "material")
    material = materials[material_index]
    if not isinstance(material, Mapping):
        raise AssetValidationError("glTF material must be an object")

    bindings: dict[str, dict[str, Any]] = {}
    for semantic, path in _SEMANTIC_PATHS:
        texture_info = _nested_mapping(material, path, semantic)
        texture_index = _require_index(textures, texture_info.get("index"), f"{semantic} texture")
        texture = textures[texture_index]
        if not isinstance(texture, Mapping):
            raise AssetValidationError(f"{semantic} texture must be an object")
        image_index = _require_index(images, texture.get("source"), f"{semantic} image")
        image = images[image_index]
        if not isinstance(image, Mapping):
            raise AssetValidationError(f"{semantic} image must be an object")
        uri, path_on_disk = _safe_image_path(gltf_dir, image.get("uri"), semantic)
        texcoord = texture_info.get("texCoord", 0)
        if not isinstance(texcoord, int) or isinstance(texcoord, bool) or texcoord < 0:
            raise AssetValidationError(f"{semantic} texCoord must be a non-negative integer")
        bindings[semantic] = {
            "texture_index": texture_index,
            "image_index": image_index,
            "texcoord": texcoord,
            "uri": uri,
            "path": path_on_disk,
        }

    if len({binding["path"] for binding in bindings.values()}) != 4:
        raise AssetValidationError("Core-4 source must reference four distinct PNG images")
    return bindings


def _accessor_metadata(document: Mapping[str, Any], index: Any, label: str) -> dict[str, Any]:
    accessors = document.get("accessors", [])
    if not isinstance(accessors, list):
        raise AssetValidationError("glTF accessors must be an array")
    accessor_index = _require_index(accessors, index, f"{label} accessor")
    accessor = accessors[accessor_index]
    if not isinstance(accessor, Mapping):
        raise AssetValidationError(f"{label} accessor must be an object")
    for required in ("componentType", "count", "type"):
        if required not in accessor:
            raise AssetValidationError(f"{label} accessor is missing {required}")
    metadata: dict[str, Any] = {
        "index": accessor_index,
        "component_type": accessor["componentType"],
        "count": accessor["count"],
        "type": accessor["type"],
        "normalized": bool(accessor.get("normalized", False)),
    }
    for source, target in (
        ("bufferView", "buffer_view"),
        ("byteOffset", "byte_offset"),
        ("min", "min"),
        ("max", "max"),
    ):
        if source in accessor:
            metadata[target] = accessor[source]
    return metadata


def _geometry_metadata(document: Mapping[str, Any]) -> dict[str, Any]:
    """Record the single-primitive geometry contract without reading buffers."""

    meshes = document.get("meshes", [])
    if not isinstance(meshes, list) or len(meshes) != 1:
        raise AssetValidationError("Stage A expects exactly one glTF mesh")
    mesh = meshes[0]
    primitives = mesh.get("primitives", []) if isinstance(mesh, Mapping) else []
    if not isinstance(primitives, list) or len(primitives) != 1:
        raise AssetValidationError("Stage A expects exactly one glTF primitive")
    primitive = primitives[0]
    if not isinstance(primitive, Mapping):
        raise AssetValidationError("glTF primitive must be an object")
    attributes = primitive.get("attributes")
    if not isinstance(attributes, Mapping):
        raise AssetValidationError("glTF primitive attributes must be an object")
    missing = [name for name in _REQUIRED_ATTRIBUTES if name not in attributes]
    if missing:
        raise AssetValidationError(f"glTF primitive is missing attributes: {', '.join(missing)}")
    if "indices" not in primitive or "material" not in primitive:
        raise AssetValidationError("glTF primitive must reference indices and a material")

    accessor_data = {
        semantic: _accessor_metadata(document, attributes[semantic], semantic)
        for semantic in _REQUIRED_ATTRIBUTES
    }
    indices = _accessor_metadata(document, primitive["indices"], "indices")
    mode = primitive.get("mode", 4)
    if mode != 4:
        raise AssetValidationError("Stage A expects TRIANGLES primitive mode (4)")
    count = indices["count"]
    if not isinstance(count, int) or count % 3 != 0:
        raise AssetValidationError("triangle index count must be divisible by three")
    return {
        "mesh_index": 0,
        "primitive_index": 0,
        "material_index": primitive["material"],
        "mode": mode,
        "triangle_count": count // 3,
        "indices": indices,
        "attributes": accessor_data,
    }


def _channel_ranges(array: np.ndarray, bands: Sequence[str]) -> dict[str, list[int]]:
    if array.ndim == 2:
        array = array[..., np.newaxis]
    return {
        band: [int(array[..., channel].min()), int(array[..., channel].max())]
        for channel, band in enumerate(bands)
    }


def _validate_png(
    path: Path, semantic: str, expected_size: tuple[int, int]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Validate lossless source pixels and return their storage-domain metadata."""

    try:
        with Image.open(path) as image:
            image.load()
            mode = image.mode
            size = image.size
            bands = image.getbands()
            pixels = np.asarray(image)
    except (OSError, ValueError) as error:
        raise AssetValidationError(f"failed to read {semantic} PNG: {path.name}") from error
    if size != expected_size:
        raise AssetValidationError(
            f"{semantic} PNG size is {size}, expected {expected_size}"
        )
    if mode not in _ALLOWED_MODES[semantic]:
        allowed = ", ".join(_ALLOWED_MODES[semantic])
        raise AssetValidationError(f"{semantic} PNG mode is {mode}, expected one of {allowed}")
    if pixels.dtype != np.uint8:
        raise AssetValidationError(f"{semantic} PNG must contain 8-bit unsigned pixels")
    return pixels, {
        "uri": path.name,
        "sha256": sha256_file(path),
        "size": [size[0], size[1]],
        "mode": mode,
        "channel_ranges_u8": _channel_ranges(pixels, bands),
    }


def _scalar_statistics(array: np.ndarray) -> dict[str, float | int]:
    return {
        "min_u8": int(array.min()),
        "max_u8": int(array.max()),
        "mean_u8": float(array.mean(dtype=np.float64)),
        "min_linear": float(array.min()) / 255.0,
        "max_linear": float(array.max()) / 255.0,
        "mean_linear": float(array.mean(dtype=np.float64)) / 255.0,
    }


def _normal_statistics(pixels: np.ndarray) -> dict[str, Any]:
    """Describe source normal texels without normalizing or changing +Y."""

    decoded = decode_normal_unorm8(pixels)
    lengths = np.linalg.norm(decoded, axis=-1)
    components: dict[str, dict[str, float]] = {}
    for channel, name in enumerate(("x", "y", "z")):
        values = decoded[..., channel]
        components[name] = {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean(dtype=np.float64)),
        }
    return {
        "decode": "2 * (u8 / 255) - 1",
        "normalized_after_decode": False,
        "y_flipped": False,
        "components": components,
        "vector_length": {
            "min": float(lengths.min()),
            "max": float(lengths.max()),
            "mean": float(lengths.mean(dtype=np.float64)),
        },
    }


def _write_yaml(path: Path, manifest: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(
        dict(manifest),
        sort_keys=False,
        allow_unicode=True,
        default_flow_style=False,
    )
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(text)


def _relative_uri(path: Path, base_directory: Path) -> str:
    try:
        relative = os.path.relpath(path.resolve(), base_directory.resolve())
    except ValueError as error:
        raise AssetValidationError(
            "manifest and referenced assets must be on the same filesystem volume"
        ) from error
    return Path(relative).as_posix()


def preprocess_scifihelmet(
    gltf_path: Path | str,
    output_dir: Path | str,
    manifest_path: Path | str,
    *,
    expected_size: tuple[int, int] = (2048, 2048),
) -> dict[str, Any]:
    """Validate and deterministically preprocess SciFiHelmet Core-4 textures."""

    gltf_path = Path(gltf_path)
    output_dir = Path(output_dir)
    manifest_path = Path(manifest_path)
    if not gltf_path.is_file():
        raise AssetValidationError(f"source glTF does not exist: {gltf_path}")
    try:
        document = json.loads(gltf_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise AssetValidationError(f"failed to parse glTF JSON: {gltf_path.name}") from error
    if not isinstance(document, Mapping):
        raise AssetValidationError("glTF root must be an object")
    asset = document.get("asset")
    if not isinstance(asset, Mapping) or asset.get("version") != "2.0":
        raise AssetValidationError("Stage A requires a glTF 2.0 asset")

    geometry = _geometry_metadata(document)
    bindings = _texture_bindings(document, gltf_path.parent, geometry["material_index"])

    source_pixels: dict[str, np.ndarray] = {}
    source_images: dict[str, dict[str, Any]] = {}
    for semantic, _ in _SEMANTIC_PATHS:
        pixels, image_info = _validate_png(
            bindings[semantic]["path"], semantic, expected_size
        )
        image_info["uri"] = bindings[semantic]["uri"]
        image_info["texture_index"] = bindings[semantic]["texture_index"]
        image_info["image_index"] = bindings[semantic]["image_index"]
        image_info["texcoord"] = bindings[semantic]["texcoord"]
        source_pixels[semantic] = pixels
        source_images[semantic] = image_info

    metallic_roughness = source_pixels["metallic_roughness"]
    if metallic_roughness.ndim != 3 or metallic_roughness.shape[-1] < 3:
        raise AssetValidationError("metallic-roughness PNG must provide RGB channels")
    # glTF fixes roughness in G and metallic in B; neither scalar is sRGB-decoded.
    roughness = np.ascontiguousarray(metallic_roughness[..., 1])
    metallic = np.ascontiguousarray(metallic_roughness[..., 2])

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {
        "base_color": output_dir / "base_color.png",
        "normal": output_dir / "normal.png",
        "roughness": output_dir / "roughness.png",
        "metallic": output_dir / "metallic.png",
    }
    # Preserve encoded BaseColor bytes and the source +Y normal map exactly.
    # Color decoding and the UE-only Y convention bridge occur downstream.
    shutil.copyfile(bindings["base_color"]["path"], output_paths["base_color"])
    shutil.copyfile(bindings["normal"]["path"], output_paths["normal"])
    Image.fromarray(roughness).save(output_paths["roughness"], format="PNG")
    Image.fromarray(metallic).save(output_paths["metallic"], format="PNG")

    texture_references = {
        semantic: {
            "texture_index": bindings[semantic]["texture_index"],
            "image_index": bindings[semantic]["image_index"],
            "texcoord": bindings[semantic]["texcoord"],
            "uri": bindings[semantic]["uri"],
        }
        for semantic, _ in _SEMANTIC_PATHS
    }
    outputs = {
        "base_color": {
            "uri": output_paths["base_color"].name,
            "sha256": sha256_file(output_paths["base_color"]),
            "size": list(expected_size),
            "mode": source_images["base_color"]["mode"],
            "storage_encoding": "srgb_unorm8",
            "training_semantic": "decode_srgb_to_linear_rgb_once",
            "alpha_semantic": "ignored" if source_images["base_color"]["mode"] == "RGBA" else "absent",
        },
        "normal": {
            "uri": output_paths["normal"].name,
            "sha256": sha256_file(output_paths["normal"]),
            "size": list(expected_size),
            "mode": source_images["normal"]["mode"],
            "storage_encoding": "linear_unorm8",
            "training_semantic": "tangent_space_xyz",
            "decode": "2 * (u8 / 255) - 1",
            "normalize_after_decode": False,
            "flip_y": False,
        },
        "roughness": {
            "uri": output_paths["roughness"].name,
            "sha256": sha256_file(output_paths["roughness"]),
            "size": list(expected_size),
            "mode": "L",
            "storage_encoding": "linear_unorm8",
            "source_channel": "G",
        },
        "metallic": {
            "uri": output_paths["metallic"].name,
            "sha256": sha256_file(output_paths["metallic"]),
            "size": list(expected_size),
            "mode": "L",
            "storage_encoding": "linear_unorm8",
            "source_channel": "B",
        },
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "asset": "SciFiHelmet",
        "source": {
            "gltf": {
                "uri": gltf_path.name,
                "sha256": sha256_file(gltf_path),
                "version": "2.0",
            },
            "texture_references": texture_references,
            "images": source_images,
            "geometry": geometry,
        },
        "core4": {
            "base_color": "linear RGB after exactly one sRGB decode",
            "normal": "linear tangent-space XYZ; no normalization and no Y flip in Stage A",
            "roughness": "linear scalar copied exactly from metallicRoughness.G",
            "metallic": "linear scalar copied exactly from metallicRoughness.B",
            "ambient_occlusion": "validated source only; excluded from Core-4 outputs",
        },
        "output_root": _relative_uri(output_dir, manifest_path.parent),
        "outputs": outputs,
        "statistics": {
            "normal_decoded": _normal_statistics(source_pixels["normal"]),
            "roughness": _scalar_statistics(roughness),
            "metallic": _scalar_statistics(metallic),
        },
    }
    _write_yaml(manifest_path, manifest)
    return manifest
