"""CPU contract tests for the minimal direct glTF mesh loader."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cg_frontier.assets.gltf_mesh import (  # noqa: E402
    GltfMeshError,
    load_gltf_mesh,
    reconstruct_tangents,
)


REAL_GLTF = (
    REPOSITORY_ROOT
    / "assets"
    / "source"
    / "glTF-Sample-Assets"
    / "Models"
    / "SciFiHelmet"
    / "glTF"
    / "SciFiHelmet.gltf"
)


def _append(blob: bytearray, values: np.ndarray) -> tuple[int, int]:
    while len(blob) % 4:
        blob.append(0)
    offset = len(blob)
    data = np.ascontiguousarray(values).tobytes()
    blob.extend(data)
    return offset, len(data)


def _synthetic_gltf(root: Path) -> Path:
    root.mkdir(parents=True)
    positions_interleaved = np.array(
        [
            [-1.0, -1.0, 0.0, 11.0],
            [1.0, -1.0, 0.0, 12.0],
            [1.0, 1.0, 0.0, 13.0],
            [-1.0, 1.0, 0.0, 14.0],
        ],
        dtype="<f4",
    )
    normals = np.tile(np.array([[0.0, 0.0, 1.0]], dtype="<f4"), (4, 1))
    tangents = np.tile(np.array([[1.0, 0.0, 0.0, 1.0]], dtype="<f4"), (4, 1))
    texcoords_u16 = np.array(
        [[0, 0], [65535, 0], [65535, 65535], [0, 65535]], dtype="<u2"
    )
    indices = np.array([0, 1, 2, 0, 2, 3], dtype="<u2")

    blob = bytearray()
    arrays = [positions_interleaved, normals, tangents, texcoords_u16, indices]
    ranges = [_append(blob, values) for values in arrays]
    (root / "mesh.bin").write_bytes(blob)
    buffer_views = [
        {"buffer": 0, "byteOffset": ranges[0][0], "byteLength": ranges[0][1], "byteStride": 16},
        {"buffer": 0, "byteOffset": ranges[1][0], "byteLength": ranges[1][1]},
        {"buffer": 0, "byteOffset": ranges[2][0], "byteLength": ranges[2][1]},
        {"buffer": 0, "byteOffset": ranges[3][0], "byteLength": ranges[3][1]},
        {"buffer": 0, "byteOffset": ranges[4][0], "byteLength": ranges[4][1]},
    ]
    document = {
        "asset": {"version": "2.0"},
        "buffers": [{"uri": "mesh.bin", "byteLength": len(blob)}],
        "bufferViews": buffer_views,
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": 4,
                "type": "VEC3",
                "min": [-1.0, -1.0, 0.0],
                "max": [1.0, 1.0, 0.0],
            },
            {"bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5126, "count": 4, "type": "VEC4"},
            {
                "bufferView": 3,
                "componentType": 5123,
                "normalized": True,
                "count": 4,
                "type": "VEC2",
            },
            {"bufferView": 4, "componentType": 5123, "count": 6, "type": "SCALAR"},
        ],
        "materials": [{}],
        "meshes": [
            {
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": 0,
                            "NORMAL": 1,
                            "TANGENT": 2,
                            "TEXCOORD_0": 3,
                        },
                        "indices": 4,
                        "material": 0,
                        "mode": 4,
                    }
                ]
            }
        ],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }
    path = root / "mesh.gltf"
    path.write_text(json.dumps(document), encoding="utf-8", newline="\n")
    return path


def test_loads_stride_normalized_uv_and_indices_without_axis_conversion(tmp_path: Path) -> None:
    path = _synthetic_gltf(tmp_path / "asset")
    mesh = load_gltf_mesh(path)

    np.testing.assert_array_equal(
        mesh.positions,
        np.array(
            [[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]],
            dtype=np.float32,
        ),
    )
    np.testing.assert_array_equal(
        mesh.texcoords,
        np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32),
    )
    np.testing.assert_array_equal(
        mesh.triangles, np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    )
    assert mesh.coordinate_system == "right_handed_y_up_front_plus_z"
    np.testing.assert_array_equal(mesh.bounds_min, [-1.0, -1.0, 0.0])
    np.testing.assert_array_equal(mesh.bounds_max, [1.0, 1.0, 0.0])
    np.testing.assert_allclose(reconstruct_tangents(mesh), mesh.tangents, rtol=0.0, atol=1e-7)


def test_rejects_non_identity_mesh_node_transform(tmp_path: Path) -> None:
    path = _synthetic_gltf(tmp_path / "asset")
    document = json.loads(path.read_text(encoding="utf-8"))
    document["nodes"][0]["translation"] = [1.0, 0.0, 0.0]
    path.write_text(json.dumps(document), encoding="utf-8", newline="\n")
    with pytest.raises(GltfMeshError, match="identity transforms"):
        load_gltf_mesh(path)


def test_applies_node_transform_and_reconstructs_missing_tangents(tmp_path: Path) -> None:
    path = _synthetic_gltf(tmp_path / "asset")
    document = json.loads(path.read_text(encoding="utf-8"))
    del document["meshes"][0]["primitives"][0]["attributes"]["TANGENT"]
    document["nodes"][0]["rotation"] = [0.0, 1.0, 0.0, 0.0]
    document["nodes"][0]["translation"] = [3.0, 0.0, 0.0]
    path.write_text(json.dumps(document), encoding="utf-8", newline="\n")

    mesh = load_gltf_mesh(
        path, apply_node_transform=True, reconstruct_missing_tangents=True
    )

    np.testing.assert_allclose(
        mesh.positions,
        np.array([[4, -1, 0], [2, -1, 0], [2, 1, 0], [4, 1, 0]], dtype=np.float32),
        rtol=0.0,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        np.linalg.norm(mesh.tangents[:, :3], axis=1), 1.0, rtol=0.0, atol=1e-6
    )
    np.testing.assert_allclose(
        np.sum(mesh.normals * mesh.tangents[:, :3], axis=1), 0.0, rtol=0.0, atol=1e-6
    )


@pytest.mark.skipif(not REAL_GLTF.is_file(), reason="ignored SciFiHelmet source asset is absent")
def test_real_scifihelmet_mesh_contract() -> None:
    mesh = load_gltf_mesh(REAL_GLTF)
    assert mesh.positions.shape == (70074, 3)
    assert mesh.normals.shape == (70074, 3)
    assert mesh.tangents.shape == (70074, 4)
    assert mesh.texcoords.shape == (70074, 2)
    assert mesh.triangles.shape == (23358, 3)
    np.testing.assert_allclose(
        mesh.bounds_min, [-1.1511525, -1.4587183, -1.2511287], rtol=0.0, atol=1e-7
    )
    np.testing.assert_allclose(
        mesh.bounds_max, [1.1511525, 1.4587184, 1.2511277], rtol=0.0, atol=1e-7
    )
    assert np.all(mesh.tangents[:, 3] == 1.0)
    source_dot = np.sum(mesh.normals * mesh.tangents[:, :3], axis=1)
    assert float(np.max(np.abs(source_dot))) > 0.99
    rebuilt = reconstruct_tangents(mesh)
    rebuilt_dot = np.sum(mesh.normals * rebuilt[:, :3], axis=1)
    assert float(np.max(np.abs(rebuilt_dot))) < 1e-5
    handedness, counts = np.unique(rebuilt[:, 3], return_counts=True)
    assert dict(zip(handedness.tolist(), counts.tolist(), strict=True)) == {
        -1.0: 54062,
        1.0: 16012,
    }
