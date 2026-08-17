"""Minimal, explicit glTF 2.0 mesh loading for the SciFiHelmet reference asset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence
from urllib.parse import unquote, urlparse

import numpy as np


class GltfMeshError(ValueError):
    """Raised when a glTF mesh falls outside the supported reference contract."""


_COMPONENT_DTYPES: Mapping[int, np.dtype[Any]] = {
    5120: np.dtype("<i1"),
    5121: np.dtype("<u1"),
    5122: np.dtype("<i2"),
    5123: np.dtype("<u2"),
    5125: np.dtype("<u4"),
    5126: np.dtype("<f4"),
}

_TYPE_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
}


@dataclass(frozen=True)
class GltfMesh:
    """Single-indexed glTF primitive in its unmodified native coordinate system."""

    positions: np.ndarray
    normals: np.ndarray
    tangents: np.ndarray
    texcoords: np.ndarray
    triangles: np.ndarray
    coordinate_system: str = "right_handed_y_up_front_plus_z"

    @property
    def bounds_min(self) -> np.ndarray:
        return self.positions.min(axis=0)

    @property
    def bounds_max(self) -> np.ndarray:
        return self.positions.max(axis=0)


def reconstruct_tangents(mesh: GltfMesh) -> np.ndarray:
    """Rebuild an orthonormal tangent basis from position/UV derivatives.

    SciFiHelmet is fully de-indexed (each vertex belongs to one triangle), so
    this calculation has no cross-triangle accumulation or seam ambiguity.
    Source tangents remain available as ``mesh.tangents`` for diagnostics.
    """

    triangles = mesh.triangles.astype(np.int64, copy=False)
    p0, p1, p2 = (mesh.positions[triangles[:, corner]] for corner in range(3))
    uv0, uv1, uv2 = (mesh.texcoords[triangles[:, corner]] for corner in range(3))
    edge1 = p1 - p0
    edge2 = p2 - p0
    delta1 = uv1 - uv0
    delta2 = uv2 - uv0
    determinant = delta1[:, 0] * delta2[:, 1] - delta1[:, 1] * delta2[:, 0]
    if np.any(np.abs(determinant) < 1e-12):
        count = int(np.count_nonzero(np.abs(determinant) < 1e-12))
        raise GltfMeshError(f"cannot reconstruct tangents: {count} degenerate UV triangles")
    reciprocal = 1.0 / determinant
    tangent_per_triangle = (
        edge1 * delta2[:, 1:2] - edge2 * delta1[:, 1:2]
    ) * reciprocal[:, None]
    bitangent_per_triangle = (
        edge2 * delta1[:, 0:1] - edge1 * delta2[:, 0:1]
    ) * reciprocal[:, None]

    tangent_sum = np.zeros_like(mesh.positions, dtype=np.float64)
    bitangent_sum = np.zeros_like(mesh.positions, dtype=np.float64)
    for corner in range(3):
        np.add.at(tangent_sum, triangles[:, corner], tangent_per_triangle)
        np.add.at(bitangent_sum, triangles[:, corner], bitangent_per_triangle)
    normal = mesh.normals.astype(np.float64)
    tangent = tangent_sum - normal * np.sum(normal * tangent_sum, axis=1, keepdims=True)
    lengths = np.linalg.norm(tangent, axis=1, keepdims=True)
    if np.any(lengths < 1e-8):
        count = int(np.count_nonzero(lengths[:, 0] < 1e-8))
        raise GltfMeshError(
            f"cannot reconstruct tangents: {count} projected tangents are degenerate"
        )
    tangent /= lengths
    # glTF stores the bitangent sign in tangent.w.  Derive the same handedness
    # after Gram-Schmidt projection so TBN reconstruction stays deterministic.
    handedness = np.where(
        np.sum(np.cross(normal, tangent) * bitangent_sum, axis=1) < 0.0,
        -1.0,
        1.0,
    )
    return np.ascontiguousarray(
        np.concatenate([tangent, handedness[:, None]], axis=1), dtype=np.float32
    )


def _require_index(items: Sequence[Any], index: Any, label: str) -> int:
    if not isinstance(index, int) or isinstance(index, bool):
        raise GltfMeshError(f"{label} must be an integer index")
    if index < 0 or index >= len(items):
        raise GltfMeshError(f"{label} index {index} is out of range")
    return index


def _safe_external_path(root: Path, uri: Any, label: str) -> Path:
    if not isinstance(uri, str) or not uri:
        raise GltfMeshError(f"{label} must use a non-empty external URI")
    parsed = urlparse(uri)
    if parsed.scheme or parsed.netloc or parsed.query or parsed.fragment:
        raise GltfMeshError(f"{label} URI must be a local relative path")
    uri_path = PurePosixPath(unquote(parsed.path))
    if uri_path.is_absolute() or ".." in uri_path.parts:
        raise GltfMeshError(f"{label} URI must stay inside the glTF directory")
    path = (root / Path(*uri_path.parts)).resolve()
    if not path.is_relative_to(root.resolve()):
        raise GltfMeshError(f"{label} URI escapes the glTF directory")
    if not path.is_file():
        raise GltfMeshError(f"{label} file does not exist: {uri}")
    return path


def _identity_node_transform(node: Mapping[str, Any]) -> bool:
    if "matrix" in node:
        matrix = np.asarray(node["matrix"], dtype=np.float64)
        if matrix.shape != (16,) or not np.allclose(
            matrix.reshape(4, 4, order="F"), np.eye(4), rtol=0.0, atol=1e-8
        ):
            return False
    defaults = {
        "translation": np.array([0.0, 0.0, 0.0]),
        "rotation": np.array([0.0, 0.0, 0.0, 1.0]),
        "scale": np.array([1.0, 1.0, 1.0]),
    }
    for key, expected in defaults.items():
        if key in node:
            value = np.asarray(node[key], dtype=np.float64)
            if value.shape != expected.shape or not np.allclose(
                value, expected, rtol=0.0, atol=1e-8
            ):
                return False
    return True


def _node_local_transform(node: Mapping[str, Any]) -> np.ndarray:
    """Return one glTF node's column-vector local transform."""

    if "matrix" in node:
        if any(key in node for key in ("translation", "rotation", "scale")):
            raise GltfMeshError("glTF node may not combine matrix and TRS transforms")
        matrix = np.asarray(node["matrix"], dtype=np.float64)
        if matrix.shape != (16,) or not np.isfinite(matrix).all():
            raise GltfMeshError("glTF node matrix must contain 16 finite values")
        return matrix.reshape(4, 4, order="F")

    translation = np.asarray(node.get("translation", (0.0, 0.0, 0.0)), dtype=np.float64)
    rotation = np.asarray(node.get("rotation", (0.0, 0.0, 0.0, 1.0)), dtype=np.float64)
    scale = np.asarray(node.get("scale", (1.0, 1.0, 1.0)), dtype=np.float64)
    if (
        translation.shape != (3,)
        or rotation.shape != (4,)
        or scale.shape != (3,)
        or not np.isfinite(translation).all()
        or not np.isfinite(rotation).all()
        or not np.isfinite(scale).all()
    ):
        raise GltfMeshError("glTF node TRS values have invalid shape or non-finite data")
    rotation_length = np.linalg.norm(rotation)
    if rotation_length < 1e-12 or np.any(np.abs(scale) < 1e-12):
        raise GltfMeshError("glTF node rotation and scale must be non-degenerate")
    x, y, z, w = rotation / rotation_length
    rotation_matrix = np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation_matrix @ np.diag(scale)
    matrix[:3, 3] = translation
    return matrix


def _mesh_node_transform(
    document: Mapping[str, Any], mesh_index: int, *, require_identity: bool
) -> np.ndarray:
    """Require one active-scene mesh node with an identity ancestor chain.

    Stage B deliberately consumes vertex data in native glTF coordinates.  Any
    node transform would otherwise be an implicit coordinate conversion that
    the GBuffer path does not apply to normals and tangents.
    """

    nodes = document.get("nodes", [])
    scenes = document.get("scenes", [])
    if not isinstance(nodes, list) or not isinstance(scenes, list) or not scenes:
        raise GltfMeshError("glTF must define nodes and at least one scene")
    mesh_nodes = [
        index
        for index, node in enumerate(nodes)
        if isinstance(node, Mapping) and node.get("mesh") == mesh_index
    ]
    if len(mesh_nodes) != 1:
        raise GltfMeshError("Stage B expects exactly one node referencing the mesh")
    mesh_node = mesh_nodes[0]
    if nodes[mesh_node].get("skin") is not None:
        raise GltfMeshError("skinned meshes are outside the Stage-B contract")

    parents: dict[int, int] = {}
    for parent_index, node in enumerate(nodes):
        if not isinstance(node, Mapping):
            raise GltfMeshError("glTF node must be an object")
        children = node.get("children", [])
        if not isinstance(children, list):
            raise GltfMeshError("glTF node children must be an array")
        for child in children:
            child_index = _require_index(nodes, child, "child node")
            if child_index in parents:
                raise GltfMeshError("a node may not have multiple parents")
            parents[child_index] = parent_index

    chain: list[int] = []
    current = mesh_node
    while True:
        if current in chain:
            raise GltfMeshError("node hierarchy contains a cycle")
        chain.append(current)
        if require_identity and not _identity_node_transform(nodes[current]):
            raise GltfMeshError(
                "mesh node and its ancestors must use identity transforms in Stage B"
            )
        if current not in parents:
            break
        current = parents[current]

    scene_index = document.get("scene", 0)
    scene_index = _require_index(scenes, scene_index, "scene")
    scene = scenes[scene_index]
    roots = scene.get("nodes", []) if isinstance(scene, Mapping) else []
    if not isinstance(roots, list):
        raise GltfMeshError("active scene nodes must be an array")
    reachable: set[int] = set()
    stack = [_require_index(nodes, root, "scene node") for root in roots]
    while stack:
        index = stack.pop()
        if index in reachable:
            continue
        reachable.add(index)
        stack.extend(
            _require_index(nodes, child, "child node")
            for child in nodes[index].get("children", [])
        )
    if mesh_node not in reachable:
        raise GltfMeshError("mesh node is not reachable from the active scene")
    world = np.eye(4, dtype=np.float64)
    for index in reversed(chain):
        world = world @ _node_local_transform(nodes[index])
    return world


def _load_buffers(document: Mapping[str, Any], gltf_dir: Path) -> list[bytes]:
    buffers = document.get("buffers", [])
    if not isinstance(buffers, list) or not buffers:
        raise GltfMeshError("glTF must define at least one external buffer")
    loaded: list[bytes] = []
    for index, buffer in enumerate(buffers):
        if not isinstance(buffer, Mapping):
            raise GltfMeshError("glTF buffer must be an object")
        path = _safe_external_path(gltf_dir, buffer.get("uri"), f"buffer {index}")
        data = path.read_bytes()
        declared_length = buffer.get("byteLength")
        if not isinstance(declared_length, int) or declared_length < 0:
            raise GltfMeshError(f"buffer {index} byteLength must be non-negative")
        if len(data) < declared_length:
            raise GltfMeshError(
                f"buffer {index} is shorter than declared byteLength {declared_length}"
            )
        loaded.append(data)
    return loaded


def _normalized_values(values: np.ndarray, component_type: int) -> np.ndarray:
    if component_type == 5120:
        return np.maximum(values.astype(np.float32) / 127.0, -1.0)
    if component_type == 5121:
        return values.astype(np.float32) / 255.0
    if component_type == 5122:
        return np.maximum(values.astype(np.float32) / 32767.0, -1.0)
    if component_type == 5123:
        return values.astype(np.float32) / 65535.0
    raise GltfMeshError(
        f"component type {component_type} cannot use normalized decoding"
    )


def _read_accessor(
    document: Mapping[str, Any], buffers: Sequence[bytes], accessor_index: Any, label: str
) -> np.ndarray:
    """Materialize one dense accessor while honoring both offsets and stride."""

    accessors = document.get("accessors", [])
    buffer_views = document.get("bufferViews", [])
    if not isinstance(accessors, list) or not isinstance(buffer_views, list):
        raise GltfMeshError("glTF accessors and bufferViews must be arrays")
    index = _require_index(accessors, accessor_index, f"{label} accessor")
    accessor = accessors[index]
    if not isinstance(accessor, Mapping):
        raise GltfMeshError(f"{label} accessor must be an object")
    if "sparse" in accessor:
        raise GltfMeshError(f"sparse {label} accessors are outside the Stage-B contract")
    if "bufferView" not in accessor:
        raise GltfMeshError(f"{label} accessor must reference a bufferView")

    component_type = accessor.get("componentType")
    if component_type not in _COMPONENT_DTYPES:
        raise GltfMeshError(f"unsupported {label} componentType: {component_type}")
    accessor_type = accessor.get("type")
    if accessor_type not in _TYPE_COMPONENTS:
        raise GltfMeshError(f"unsupported {label} accessor type: {accessor_type}")
    count = accessor.get("count")
    if not isinstance(count, int) or count <= 0:
        raise GltfMeshError(f"{label} accessor count must be positive")

    view_index = _require_index(
        buffer_views, accessor["bufferView"], f"{label} bufferView"
    )
    view = buffer_views[view_index]
    if not isinstance(view, Mapping):
        raise GltfMeshError(f"{label} bufferView must be an object")
    buffer_index = _require_index(buffers, view.get("buffer"), f"{label} buffer")
    dtype = _COMPONENT_DTYPES[component_type]
    component_count = _TYPE_COMPONENTS[accessor_type]
    packed_size = dtype.itemsize * component_count
    stride = view.get("byteStride", packed_size)
    if not isinstance(stride, int) or stride < packed_size:
        raise GltfMeshError(f"invalid {label} byteStride: {stride}")
    # Both offsets are relative to the start of the underlying buffer.
    offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    if not isinstance(offset, int) or offset < 0:
        raise GltfMeshError(f"invalid {label} byte offset")
    required_end = offset + (count - 1) * stride + packed_size
    view_end = view.get("byteOffset", 0) + view.get("byteLength", 0)
    if required_end > view_end or required_end > len(buffers[buffer_index]):
        raise GltfMeshError(f"{label} accessor exceeds its bufferView")

    # Copy out of the immutable byte buffer so callers receive an owned,
    # contiguous array independent of interleaved bufferView storage.
    values = np.ndarray(
        shape=(count, component_count),
        dtype=dtype,
        buffer=buffers[buffer_index],
        offset=offset,
        strides=(stride, dtype.itemsize),
    ).copy()
    if accessor.get("normalized", False):
        values = _normalized_values(values, component_type)
    return values


def _unit_length_error(vectors: np.ndarray) -> float:
    return float(np.max(np.abs(np.linalg.norm(vectors, axis=1) - 1.0)))


def load_gltf_mesh(
    gltf_path: Path | str,
    *,
    apply_node_transform: bool = False,
    reconstruct_missing_tangents: bool = False,
) -> GltfMesh:
    """Load one identity-transformed indexed TRIANGLES primitive.

    Positions, normals, tangents, and UVs retain their glTF values: this loader
    performs no axis swap, handedness conversion, or UV vertical flip.
    """

    gltf_path = Path(gltf_path)
    if not gltf_path.is_file():
        raise GltfMeshError(f"glTF does not exist: {gltf_path}")
    try:
        document = json.loads(gltf_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise GltfMeshError(f"failed to parse glTF JSON: {gltf_path.name}") from error
    if not isinstance(document, Mapping):
        raise GltfMeshError("glTF root must be an object")
    asset = document.get("asset")
    if not isinstance(asset, Mapping) or asset.get("version") != "2.0":
        raise GltfMeshError("Stage B requires glTF 2.0")

    meshes = document.get("meshes", [])
    if not isinstance(meshes, list) or len(meshes) != 1:
        raise GltfMeshError("Stage B expects exactly one mesh")
    mesh = meshes[0]
    primitives = mesh.get("primitives", []) if isinstance(mesh, Mapping) else []
    if not isinstance(primitives, list) or len(primitives) != 1:
        raise GltfMeshError("Stage B expects exactly one primitive")
    primitive = primitives[0]
    if not isinstance(primitive, Mapping):
        raise GltfMeshError("glTF primitive must be an object")
    if primitive.get("mode", 4) != 4:
        raise GltfMeshError("Stage B expects TRIANGLES primitive mode (4)")
    if primitive.get("targets"):
        raise GltfMeshError("morph targets are outside the Stage-B contract")
    attributes = primitive.get("attributes")
    if not isinstance(attributes, Mapping):
        raise GltfMeshError("glTF primitive attributes must be an object")
    required = ("POSITION", "NORMAL", "TEXCOORD_0")
    if not reconstruct_missing_tangents:
        required += ("TANGENT",)
    missing = [semantic for semantic in required if semantic not in attributes]
    if missing:
        raise GltfMeshError(f"missing primitive attributes: {', '.join(missing)}")
    if "indices" not in primitive:
        raise GltfMeshError("Stage B requires an indexed primitive")

    node_transform = _mesh_node_transform(
        document, 0, require_identity=not apply_node_transform
    )
    buffers = _load_buffers(document, gltf_path.parent)
    positions = _read_accessor(document, buffers, attributes["POSITION"], "POSITION")
    normals = _read_accessor(document, buffers, attributes["NORMAL"], "NORMAL")
    tangents = (
        _read_accessor(document, buffers, attributes["TANGENT"], "TANGENT")
        if "TANGENT" in attributes
        else None
    )
    texcoords = _read_accessor(document, buffers, attributes["TEXCOORD_0"], "TEXCOORD_0")
    indices = _read_accessor(document, buffers, primitive["indices"], "indices")

    if positions.dtype != np.float32 or positions.shape[1] != 3:
        raise GltfMeshError("POSITION must be float32 VEC3")
    if normals.dtype != np.float32 or normals.shape[1] != 3:
        raise GltfMeshError("NORMAL must be float32 VEC3")
    if tangents is not None and (tangents.dtype != np.float32 or tangents.shape[1] != 4):
        raise GltfMeshError("TANGENT must be float32 VEC4")
    if texcoords.shape[1] != 2:
        raise GltfMeshError("TEXCOORD_0 must be VEC2")
    if indices.shape[1] != 1 or not np.issubdtype(indices.dtype, np.integer):
        raise GltfMeshError("indices must use an integer SCALAR accessor")
    vertex_count = positions.shape[0]
    vertex_arrays = (normals, texcoords) if tangents is None else (normals, tangents, texcoords)
    if any(array.shape[0] != vertex_count for array in vertex_arrays):
        raise GltfMeshError("all vertex attributes must have matching counts")
    flat_indices = indices[:, 0].astype(np.int64)
    if flat_indices.size % 3 != 0:
        raise GltfMeshError("triangle index count must be divisible by three")
    if flat_indices.min() < 0 or flat_indices.max() >= vertex_count:
        raise GltfMeshError("triangle indices reference vertices out of range")
    if flat_indices.max() > np.iinfo(np.int32).max:
        raise GltfMeshError("nvdiffrast requires triangle indices that fit int32")
    if _unit_length_error(normals) > 2e-3:
        raise GltfMeshError("source NORMAL values are not unit length within tolerance")
    if tangents is not None and _unit_length_error(tangents[:, :3]) > 2e-3:
        raise GltfMeshError("source TANGENT.xyz values are not unit length within tolerance")
    if tangents is not None and not np.all(np.isin(tangents[:, 3], (-1.0, 1.0))):
        raise GltfMeshError("source TANGENT.w must be exactly -1 or +1")
    if apply_node_transform:
        linear = node_transform[:3, :3]
        position_h = np.concatenate(
            (positions.astype(np.float64), np.ones((vertex_count, 1), dtype=np.float64)), axis=1
        )
        positions = (position_h @ node_transform.T)[:, :3]
        normal_matrix = np.linalg.inv(linear).T
        normals = normals.astype(np.float64) @ normal_matrix.T
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        if tangents is not None:
            tangent_xyz = tangents[:, :3].astype(np.float64) @ linear.T
            tangent_xyz -= normals * np.sum(normals * tangent_xyz, axis=1, keepdims=True)
            tangent_xyz /= np.linalg.norm(tangent_xyz, axis=1, keepdims=True)
            tangent_w = tangents[:, 3:4].astype(np.float64) * np.sign(np.linalg.det(linear))
            tangents = np.concatenate((tangent_xyz, tangent_w), axis=1)

    mesh = GltfMesh(
        positions=np.ascontiguousarray(positions, dtype=np.float32),
        normals=np.ascontiguousarray(normals, dtype=np.float32),
        tangents=np.ascontiguousarray(
            tangents if tangents is not None else np.zeros((vertex_count, 4)),
            dtype=np.float32,
        ),
        texcoords=np.ascontiguousarray(texcoords, dtype=np.float32),
        triangles=np.ascontiguousarray(flat_indices.reshape(-1, 3), dtype=np.int32),
    )
    if tangents is None:
        mesh = GltfMesh(
            positions=mesh.positions,
            normals=mesh.normals,
            tangents=reconstruct_tangents(mesh),
            texcoords=mesh.texcoords,
            triangles=mesh.triangles,
            coordinate_system=mesh.coordinate_system,
        )
    return mesh
