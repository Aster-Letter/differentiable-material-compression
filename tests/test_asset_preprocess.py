"""Contract tests for deterministic Stage-A Core-4 texture preprocessing."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cg_frontier.assets.preprocess import (  # noqa: E402
    decode_normal_unorm8,
    preprocess_scifihelmet,
    sha256_file,
    srgb_to_linear,
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


def _write_png(path: Path, pixels: np.ndarray, mode: str) -> None:
    Image.fromarray(pixels, mode=mode).save(path, format="PNG")


def _synthetic_asset(root: Path) -> tuple[Path, dict[str, np.ndarray]]:
    root.mkdir(parents=True)
    base_color = np.array(
        [
            [[0, 16, 32, 255], [64, 96, 128, 255]],
            [[160, 192, 224, 255], [255, 128, 0, 255]],
        ],
        dtype=np.uint8,
    )
    metallic_roughness = np.array(
        [
            [[9, 10, 20, 255], [8, 30, 40, 255]],
            [[7, 50, 60, 255], [6, 70, 80, 255]],
        ],
        dtype=np.uint8,
    )
    normal = np.array(
        [
            [[128, 64, 255], [0, 255, 128]],
            [[255, 0, 128], [64, 128, 192]],
        ],
        dtype=np.uint8,
    )
    ambient_occlusion = np.array([[0, 85], [170, 255]], dtype=np.uint8)
    _write_png(root / "albedo.png", base_color, "RGBA")
    _write_png(root / "packed.png", metallic_roughness, "RGBA")
    _write_png(root / "tangent.png", normal, "RGB")
    _write_png(root / "ao.png", ambient_occlusion, "L")

    document = {
        "asset": {"version": "2.0", "generator": "synthetic-test"},
        "images": [
            {"uri": "ao.png"},
            {"uri": "packed.png"},
            {"uri": "albedo.png"},
            {"uri": "tangent.png"},
        ],
        "textures": [
            {"source": 1},
            {"source": 3},
            {"source": 0},
            {"source": 2},
        ],
        "materials": [
            {
                "name": "Synthetic",
                "pbrMetallicRoughness": {
                    "baseColorTexture": {"index": 3, "texCoord": 0},
                    "metallicRoughnessTexture": {"index": 0},
                },
                "normalTexture": {"index": 1},
                "occlusionTexture": {"index": 2},
            }
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5125,
                "count": 6,
                "type": "SCALAR",
                "min": [0],
                "max": [3],
            },
            {
                "bufferView": 1,
                "componentType": 5126,
                "count": 4,
                "type": "VEC3",
                "min": [-1.0, -1.0, -1.0],
                "max": [1.0, 1.0, 1.0],
            },
            {
                "bufferView": 2,
                "componentType": 5126,
                "count": 4,
                "type": "VEC3",
            },
            {
                "bufferView": 3,
                "componentType": 5126,
                "count": 4,
                "type": "VEC4",
                "min": [-1.0, -1.0, -1.0, 1.0],
                "max": [1.0, 1.0, 1.0, 1.0],
            },
            {
                "bufferView": 4,
                "componentType": 5126,
                "count": 4,
                "type": "VEC2",
                "min": [0.0, 0.0],
                "max": [1.0, 1.0],
            },
        ],
        "meshes": [
            {
                "name": "Synthetic",
                "primitives": [
                    {
                        "attributes": {
                            "POSITION": 1,
                            "NORMAL": 2,
                            "TANGENT": 3,
                            "TEXCOORD_0": 4,
                        },
                        "indices": 0,
                        "material": 0,
                        "mode": 4,
                    }
                ],
            }
        ],
    }
    gltf_path = root / "fixture.gltf"
    gltf_path.write_text(json.dumps(document), encoding="utf-8", newline="\n")
    return gltf_path, {
        "base_color": base_color,
        "metallic_roughness": metallic_roughness,
        "normal": normal,
        "ambient_occlusion": ambient_occlusion,
    }


def test_srgb_decode_anchor_points() -> None:
    encoded = np.array([0.0, 0.04045, 0.5, 1.0])
    decoded = srgb_to_linear(encoded)
    expected_midpoint = ((0.5 + 0.055) / 1.055) ** 2.4
    np.testing.assert_allclose(
        decoded,
        np.array([0.0, 0.04045 / 12.92, expected_midpoint, 1.0]),
        rtol=0.0,
        atol=1e-12,
    )


def test_normal_decode_does_not_normalize_or_flip_y() -> None:
    pixels = np.array([[[128, 64, 255]]], dtype=np.uint8)
    decoded = decode_normal_unorm8(pixels)
    expected = pixels.astype(np.float64) * (2.0 / 255.0) - 1.0
    np.testing.assert_array_equal(decoded, expected)
    assert decoded[0, 0, 1] < 0.0
    assert not np.isclose(np.linalg.norm(decoded[0, 0]), 1.0)


def test_channel_mapping_gltf_semantics_hashes_and_statistics(tmp_path: Path) -> None:
    gltf_path, source = _synthetic_asset(tmp_path / "source")
    output_dir = tmp_path / "processed"
    manifest_path = tmp_path / "manifest.yaml"
    manifest = preprocess_scifihelmet(
        gltf_path, output_dir, manifest_path, expected_size=(2, 2)
    )

    np.testing.assert_array_equal(np.asarray(Image.open(output_dir / "base_color.png")), source["base_color"])
    np.testing.assert_array_equal(np.asarray(Image.open(output_dir / "normal.png")), source["normal"])
    np.testing.assert_array_equal(
        np.asarray(Image.open(output_dir / "roughness.png")),
        source["metallic_roughness"][..., 1],
    )
    np.testing.assert_array_equal(
        np.asarray(Image.open(output_dir / "metallic.png")),
        source["metallic_roughness"][..., 2],
    )

    references = manifest["source"]["texture_references"]
    assert manifest["source"]["gltf"]["uri"] == "fixture.gltf"
    assert manifest["output_root"] == "processed"
    assert references["base_color"] == {
        "texture_index": 3,
        "image_index": 2,
        "texcoord": 0,
        "uri": "albedo.png",
    }
    assert references["metallic_roughness"]["uri"] == "packed.png"
    geometry = manifest["source"]["geometry"]
    assert geometry["triangle_count"] == 2
    assert geometry["attributes"]["POSITION"]["min"] == [-1.0, -1.0, -1.0]
    assert geometry["attributes"]["NORMAL"]["normalized"] is False

    images = manifest["source"]["images"]
    assert images["base_color"]["mode"] == "RGBA"
    assert images["normal"]["mode"] == "RGB"
    assert images["ambient_occlusion"]["mode"] == "L"
    assert images["metallic_roughness"]["channel_ranges_u8"]["G"] == [10, 70]
    assert images["base_color"]["sha256"] == sha256_file(gltf_path.parent / "albedo.png")
    for output in manifest["outputs"].values():
        assert output["sha256"] == sha256_file(output_dir / output["uri"])

    decoded = decode_normal_unorm8(source["normal"])
    normal_statistics = manifest["statistics"]["normal_decoded"]
    assert normal_statistics["normalized_after_decode"] is False
    assert normal_statistics["y_flipped"] is False
    assert normal_statistics["components"]["y"]["mean"] == pytest.approx(
        float(decoded[..., 1].mean())
    )
    assert normal_statistics["vector_length"]["mean"] == pytest.approx(
        float(np.linalg.norm(decoded, axis=-1).mean())
    )
    assert yaml.safe_load(manifest_path.read_text(encoding="utf-8")) == manifest


def test_repeated_runs_are_byte_deterministic(tmp_path: Path) -> None:
    gltf_path, _ = _synthetic_asset(tmp_path / "source")
    manifests: list[bytes] = []
    output_hashes: list[dict[str, str]] = []
    for name in ("first", "second"):
        output_dir = tmp_path / name / "core4"
        manifest_path = tmp_path / name / "manifest.yaml"
        manifest = preprocess_scifihelmet(
            gltf_path, output_dir, manifest_path, expected_size=(2, 2)
        )
        manifests.append(manifest_path.read_bytes())
        output_hashes.append(
            {
                semantic: sha256_file(output_dir / metadata["uri"])
                for semantic, metadata in manifest["outputs"].items()
            }
        )
    assert manifests[0] == manifests[1]
    assert output_hashes[0] == output_hashes[1]


@pytest.mark.skipif(not REAL_GLTF.is_file(), reason="ignored SciFiHelmet source asset is absent")
def test_real_scifihelmet_stage_a_integration(tmp_path: Path) -> None:
    output_dir = tmp_path / "core4"
    manifest = preprocess_scifihelmet(
        REAL_GLTF, output_dir, tmp_path / "scifihelmet_core4.yaml"
    )
    assert set(manifest["source"]["images"]) == {
        "base_color",
        "metallic_roughness",
        "normal",
        "ambient_occlusion",
    }
    assert all(image["size"] == [2048, 2048] for image in manifest["source"]["images"].values())
    assert manifest["outputs"]["base_color"]["storage_encoding"] == "srgb_unorm8"
    assert manifest["outputs"]["normal"]["flip_y"] is False
    with Image.open(output_dir / "roughness.png") as roughness_image:
        assert roughness_image.mode == "L"
        assert roughness_image.size == (2048, 2048)
    with Image.open(output_dir / "metallic.png") as metallic_image:
        assert metallic_image.mode == "L"
        assert metallic_image.size == (2048, 2048)
