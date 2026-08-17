"""Deterministic shared-material glTF primitive merging for C4 assets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

import numpy as np
from PIL import Image

from cg_frontier.assets.gltf_mesh import (
    GltfMesh,
    GltfMeshError,
    _load_buffers,
    _mesh_node_transform,
    _read_accessor,
)
from cg_frontier.assets.preprocess import sha256_file


def _one_primitive(mesh: object, index: int) -> Mapping[str, Any]:
    if not isinstance(mesh, Mapping):
        raise GltfMeshError(f"mesh {index} must be an object")
    primitives = mesh.get("primitives")
    if not isinstance(primitives, list) or len(primitives) != 1:
        raise GltfMeshError("merge input requires one primitive per mesh")
    primitive = primitives[0]
    if not isinstance(primitive, Mapping) or primitive.get("mode", 4) != 4:
        raise GltfMeshError("merge input requires TRIANGLES primitives")
    if primitive.get("material") != 0 or primitive.get("targets"):
        raise GltfMeshError("all merge primitives must use material 0 without morphs")
    return primitive


def merge_shared_material_meshes(path: Path | str) -> tuple[GltfMesh, dict[str, Any]]:
    """Merge all active, one-primitive meshes after applying world transforms."""

    source = Path(path).resolve()
    document = json.loads(source.read_text(encoding="utf-8"))
    meshes = document.get("meshes")
    materials = document.get("materials")
    if not isinstance(meshes, list) or len(meshes) < 2:
        raise GltfMeshError("merge input requires at least two meshes")
    if not isinstance(materials, list) or len(materials) != 1:
        raise GltfMeshError("merge input requires exactly one shared material")
    buffers = _load_buffers(document, source.parent)
    positions_all: list[np.ndarray] = []
    normals_all: list[np.ndarray] = []
    tangents_all: list[np.ndarray] = []
    texcoords_all: list[np.ndarray] = []
    triangles_all: list[np.ndarray] = []
    transforms: list[list[float]] = []
    vertex_offset = 0
    for mesh_index, mesh_value in enumerate(meshes):
        primitive = _one_primitive(mesh_value, mesh_index)
        attributes = primitive.get("attributes")
        if not isinstance(attributes, Mapping):
            raise GltfMeshError("merge primitive attributes must be an object")
        required = ("POSITION", "NORMAL", "TANGENT", "TEXCOORD_0")
        missing = [name for name in required if name not in attributes]
        if missing or "indices" not in primitive:
            raise GltfMeshError(f"merge primitive is missing: {', '.join(missing)}")
        positions = _read_accessor(document, buffers, attributes["POSITION"], "POSITION")
        normals = _read_accessor(document, buffers, attributes["NORMAL"], "NORMAL")
        tangents = _read_accessor(document, buffers, attributes["TANGENT"], "TANGENT")
        texcoords = _read_accessor(document, buffers, attributes["TEXCOORD_0"], "TEXCOORD_0")
        indices = _read_accessor(document, buffers, primitive["indices"], "indices")[:, 0]
        if (
            positions.dtype != np.float32
            or positions.shape[1:] != (3,)
            or normals.shape != positions.shape
            or tangents.shape != (positions.shape[0], 4)
            or texcoords.shape != (positions.shape[0], 2)
            or not np.issubdtype(indices.dtype, np.integer)
            or indices.size % 3
        ):
            raise GltfMeshError("merge primitive attribute layout is unsupported")
        world = _mesh_node_transform(document, mesh_index, require_identity=False)
        transforms.append(world.reshape(-1).tolist())
        linear = world[:3, :3]
        position_h = np.concatenate(
            (positions.astype(np.float64), np.ones((positions.shape[0], 1))), axis=1
        )
        positions = (position_h @ world.T)[:, :3]
        normal_matrix = np.linalg.inv(linear).T
        normals = normals.astype(np.float64) @ normal_matrix.T
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        tangent_xyz = tangents[:, :3].astype(np.float64) @ linear.T
        tangent_xyz -= normals * np.sum(normals * tangent_xyz, axis=1, keepdims=True)
        tangent_xyz /= np.linalg.norm(tangent_xyz, axis=1, keepdims=True)
        tangent_w = tangents[:, 3:4].astype(np.float64) * np.sign(np.linalg.det(linear))
        positions_all.append(np.asarray(positions, dtype=np.float32))
        normals_all.append(np.asarray(normals, dtype=np.float32))
        tangents_all.append(
            np.asarray(np.concatenate((tangent_xyz, tangent_w), axis=1), dtype=np.float32)
        )
        texcoords_all.append(np.asarray(texcoords, dtype=np.float32))
        triangles_all.append(
            np.asarray(indices, dtype=np.int64).reshape(-1, 3) + vertex_offset
        )
        vertex_offset += positions.shape[0]
    merged = GltfMesh(
        positions=np.ascontiguousarray(np.concatenate(positions_all), dtype=np.float32),
        normals=np.ascontiguousarray(np.concatenate(normals_all), dtype=np.float32),
        tangents=np.ascontiguousarray(np.concatenate(tangents_all), dtype=np.float32),
        texcoords=np.ascontiguousarray(np.concatenate(texcoords_all), dtype=np.float32),
        triangles=np.ascontiguousarray(np.concatenate(triangles_all), dtype=np.int32),
    )
    return merged, {
        "source_meshes": len(meshes),
        "source_primitives": len(meshes),
        "world_transforms_row_major": transforms,
    }


def _append_blob(payload: bytearray, value: np.ndarray) -> tuple[int, int]:
    while len(payload) % 4:
        payload.append(0)
    offset = len(payload)
    raw = np.ascontiguousarray(value).tobytes(order="C")
    payload.extend(raw)
    return offset, len(raw)


def _emissive_fraction(path: Path, threshold: float = 0.05) -> float:
    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    return float(np.mean(np.max(rgb, axis=-1) > threshold))


def derive_single_primitive_c4_asset(
    source_gltf: Path | str,
    output_directory: Path | str,
    *,
    upstream_commit: str,
) -> dict[str, Any]:
    """Write a one-primitive derivative with emissive explicitly excluded."""

    source = Path(source_gltf).resolve()
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_document = json.loads(source.read_text(encoding="utf-8"))
    mesh, merge = merge_shared_material_meshes(source)
    material = source_document["materials"][0]
    pbr = material["pbrMetallicRoughness"]
    images = source_document["images"]
    textures = source_document["textures"]

    def image_for(texture_info: Mapping[str, Any]) -> Path:
        texture = textures[int(texture_info["index"])]
        return source.parent / str(images[int(texture["source"])]["uri"])

    texture_paths = {
        "base_color": image_for(pbr["baseColorTexture"]),
        "metallic_roughness": image_for(pbr["metallicRoughnessTexture"]),
        "normal": image_for(material["normalTexture"]),
        "emissive": image_for(material["emissiveTexture"]),
    }
    derived_names = {
        "base_color": "Lantern_baseColor.png",
        "metallic_roughness": "Lantern_roughnessMetallic.png",
        "normal": "Lantern_normal.png",
    }
    for role, name in derived_names.items():
        shutil.copyfile(texture_paths[role], output / name)

    payload = bytearray()
    arrays = (
        ("POSITION", mesh.positions.astype("<f4"), 5126, "VEC3"),
        ("NORMAL", mesh.normals.astype("<f4"), 5126, "VEC3"),
        ("TANGENT", mesh.tangents.astype("<f4"), 5126, "VEC4"),
        ("TEXCOORD_0", mesh.texcoords.astype("<f4"), 5126, "VEC2"),
        ("indices", mesh.triangles.reshape(-1).astype("<u4"), 5125, "SCALAR"),
    )
    buffer_views: list[dict[str, Any]] = []
    accessors: list[dict[str, Any]] = []
    accessor_by_name: dict[str, int] = {}
    for name, array, component_type, value_type in arrays:
        offset, length = _append_blob(payload, array)
        buffer_views.append({"buffer": 0, "byteOffset": offset, "byteLength": length})
        count = int(array.shape[0])
        accessor: dict[str, Any] = {
            "bufferView": len(buffer_views) - 1,
            "componentType": component_type,
            "count": count,
            "type": value_type,
        }
        if name == "POSITION":
            accessor["min"] = mesh.positions.min(axis=0).astype(float).tolist()
            accessor["max"] = mesh.positions.max(axis=0).astype(float).tolist()
        accessors.append(accessor)
        accessor_by_name[name] = len(accessors) - 1
    binary = output / "Lantern.bin"
    binary.write_bytes(bytes(payload))
    derived_material = {
        "name": str(material.get("name", "LanternPost_Mat")) + "_Core4_NoEmissive",
        "pbrMetallicRoughness": {
            "baseColorTexture": {"index": 0},
            "metallicRoughnessTexture": {"index": 1},
            "baseColorFactor": pbr.get("baseColorFactor", [1.0, 1.0, 1.0, 1.0]),
            "roughnessFactor": pbr.get("roughnessFactor", 1.0),
            "metallicFactor": pbr.get("metallicFactor", 1.0),
        },
        "normalTexture": {"index": 2, "scale": material.get("normalTexture", {}).get("scale", 1.0)},
        "alphaMode": "OPAQUE",
    }
    document = {
        "asset": {
            "version": "2.0",
            "generator": "cg_frontier deterministic shared-material merge v1",
        },
        "buffers": [{"uri": binary.name, "byteLength": len(payload)}],
        "bufferViews": buffer_views,
        "accessors": accessors,
        "images": [{"uri": name} for name in derived_names.values()],
        "textures": [{"source": index} for index in range(3)],
        "materials": [derived_material],
        "meshes": [{"name": "Lantern_Core4_Merged", "primitives": [{
            "attributes": {name: accessor_by_name[name] for name in ("POSITION", "NORMAL", "TANGENT", "TEXCOORD_0")},
            "indices": accessor_by_name["indices"],
            "material": 0,
            "mode": 4,
        }]}],
        "nodes": [{"mesh": 0, "name": "Lantern_Core4_Merged"}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    gltf = output / "Lantern.gltf"
    gltf.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    upstream_files = [source, source.parent / "Lantern.bin", *texture_paths.values()]
    for documentation in (source.parent.parent / "metadata.json", source.parent.parent / "LICENSE.md"):
        if documentation.is_file():
            upstream_files.append(documentation)
    derived_files = [gltf, binary, *(output / name for name in derived_names.values())]
    manifest = {
        "schema_version": 1,
        "asset": "Lantern",
        "upstream": {
            "repository": "https://github.com/KhronosGroup/glTF-Sample-Assets",
            "commit": upstream_commit,
            "license_spdx": "CC0-1.0",
            "files": {
                path.relative_to(source.parent.parent).as_posix(): sha256_file(path)
                for path in upstream_files
            },
        },
        "derivation": {
            **merge,
            "output_meshes": 1,
            "output_primitives": 1,
            "vertices": int(mesh.positions.shape[0]),
            "triangles": int(mesh.triangles.shape[0]),
            "node_transforms": "baked_into_positions_normals_tangents",
            "texture_resampling": False,
            "emissive": {
                "policy": "excluded_from_reference_pca_and_both_training_arms",
                "max_rgb_gt_0_05_fraction": _emissive_fraction(texture_paths["emissive"]),
                "conclusion_scope": "Core4 BaseColor Normal Roughness Metallic only",
            },
        },
        "derived_files": {path.name: sha256_file(path) for path in derived_files},
    }
    manifest_path = output.parent / "derived_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest
