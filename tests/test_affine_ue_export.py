from __future__ import annotations

import io
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest


pytestmark = pytest.mark.asset


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cg_frontier.compression.affine_ue_export import (  # noqa: E402
    DEFAULT_OUTPUT_RELATIVE_PATH,
    PREVIEW_MANIFEST_NAME,
    affine_material_parameters,
    compare_rgba8_png_bytes,
    export_affine_ue_preview_package,
    generate_shared_affine_hlsl,
    load_affine_ue_candidates,
    sha256_file,
)


def test_rgba8_readback_compares_pixels_not_png_encoding_bytes() -> None:
    pixels = np.arange(4 * 5 * 4, dtype=np.uint8).reshape(4, 5, 4)
    source = io.BytesIO()
    readback = io.BytesIO()
    Image.fromarray(pixels, mode="RGBA").save(source, format="PNG", compress_level=0)
    Image.fromarray(pixels, mode="RGBA").save(
        readback, format="PNG", compress_level=9
    )

    comparison = compare_rgba8_png_bytes(source.getvalue(), readback.getvalue())

    assert source.getvalue() != readback.getvalue()
    assert comparison == {
        "shape_match": True,
        "max_abs": 0,
        "changed_values": 0,
        "exact_pixel_match": True,
    }


def test_loads_progress_candidates_plus_two_certified_enhanced_pcas() -> None:
    candidates = load_affine_ue_candidates(ROOT)

    assert list(candidates) == [
        "p0_safe",
        "p0_safe_repair",
        "p0_enhanced_chroma4",
        "p0_enhanced_chroma8",
        "l0_s040k",
        "l0_s080k",
        "l1_tv_r005_s040k",
        "l2_cube_r005_s040k",
    ]
    assert [candidate.optimizer_updates for candidate in candidates.values()] == [
        0,
        0,
        0,
        0,
        40_000,
        80_000,
        40_000,
        40_000,
    ]
    assert [candidate.objective_id for candidate in candidates.values()] == [
        "pca-safe-certified",
        "pca-safe-certified-constrained",
        "pca-safe-enhanced-chroma4",
        "pca-safe-enhanced-chroma8",
        "material+helmet",
        "material+helmet",
        "material+helmet+tv",
        "material+helmet+cube",
    ]
    assert {candidate.parent_p0_hash for candidate in candidates.values()} == {
        "22a57aac4f2b86108dddf32ee76c7dbd1487d58a4f37b741eb143f34278dd98e",
        "1f2e66359a4d855e06010be2a2a9714ac54bde162161c4b07fd0cb79963fe709",
        "03a8f489f1a594f62e6eaaa1f1390455958941f0b196fd69692297a168f2e320",
        "d9e630cfd93be58b7e083744fea71f0fc77247f55db7d8fd1693c83d016cb748",
    }
    assert {candidate.decoder_weights.shape for candidate in candidates.values()} == {
        (7, 4)
    }
    assert {candidate.decoder_bias.shape for candidate in candidates.values()} == {(7,)}
    assert {candidate.static_cost["parameters"] for candidate in candidates.values()} == {
        35
    }
    assert {
        candidate.static_cost["filtered_samples_per_pixel"]
        for candidate in candidates.values()
    } == {1}
    assert all(candidate.certificate["valid"] for candidate in candidates.values())
    assert all(candidate.latent_path.is_file() for candidate in candidates.values())
    assert all(candidate.decoder_path.is_file() for candidate in candidates.values())


def test_shared_material_parameters_round_trip_the_direct_affine_decoder() -> None:
    candidates = load_affine_ue_candidates(ROOT)
    source = generate_shared_affine_hlsl()

    assert "Texture2DSample" not in source
    assert source.count("dot(LatentRGBA, AffineW") == 7
    assert "sqrt(max(1.0 - dot(NormalXY, NormalXY), 1.0e-8))" in source
    assert "NormalXY.y * NormalYSign" in source
    for forbidden in ("sigmoid", "tanh", "clamp", "normalize"):
        assert forbidden not in source.lower()

    for candidate in candidates.values():
        parameters = affine_material_parameters(candidate)
        reconstructed_weights = [
            parameters["vector_parameters"][f"AffineW{row}"]
            for row in range(7)
        ]
        reconstructed_bias = [
            parameters["scalar_parameters"][f"AffineB{row}"]
            for row in range(7)
        ]
        assert parameters["scalar_parameters"]["NormalYSign"] == -1.0
        assert reconstructed_weights == candidate.decoder_weights.tolist()
        assert reconstructed_bias == candidate.decoder_bias.tolist()


def test_exports_one_shared_master_and_eight_deterministic_material_instances() -> None:
    output = ROOT / DEFAULT_OUTPUT_RELATIVE_PATH
    manifest = export_affine_ue_preview_package(ROOT, output)

    assert manifest["status"] == "exported_ue_setup_pending"
    assert manifest["formal_holdout_accessed"] is False
    assert manifest["ue_assets"]["asset_root"] == "/Game/CGCompression/AffinePreview"
    assert manifest["ue_assets"]["source_map"] == "/Game/CGCompression/Maps/MaterialLab"
    assert manifest["ue_assets"]["preview_map"].startswith(
        "/Game/CGCompression/AffinePreview/"
    )
    assert manifest["ue_assets"]["master_material"].endswith(
        "M_SciFiHelmet_Affine_Master"
    )
    assert list(manifest["candidates"]) == [
        "p0_safe",
        "p0_safe_repair",
        "p0_enhanced_chroma4",
        "p0_enhanced_chroma8",
        "l0_s040k",
        "l0_s080k",
        "l1_tv_r005_s040k",
        "l2_cube_r005_s040k",
    ]
    assert "p0_raw" not in manifest["candidates"]
    assert len({item["ue_assets"]["material_instance"] for item in manifest["candidates"].values()}) == 8
    assert {item["ue_assets"]["master_material"] for item in manifest["candidates"].values()} == {
        manifest["ue_assets"]["master_material"]
    }

    for item in manifest["candidates"].values():
        package_root = output / item["package_directory"]
        for filename, expected_hash in item["generated_files"].items():
            assert sha256_file(package_root / filename) == expected_hash

    preview_path = output / PREVIEW_MANIFEST_NAME
    first_hash = sha256_file(preview_path)
    assert export_affine_ue_preview_package(ROOT, output) == manifest
    assert sha256_file(preview_path) == first_hash


def test_ue_setup_uses_one_shared_master_and_preserves_materiallab() -> None:
    setup_path = (
        ROOT
        / "ue_demo/CGCompressionDemo/Content/Python/"
        "setup_scifihelmet_affine_preview.py"
    )
    source = setup_path.read_text(encoding="utf-8")

    assert 'ASSET_ROOT = "/Game/CGCompression/AffinePreview"' in source
    assert 'SOURCE_MAP = "/Game/CGCompression/Maps/MaterialLab"' in source
    assert 'PREVIEW_MAP = f"{ASSET_ROOT}/Maps/MaterialLab_Affine_Progress"' in source
    assert "EXPECTED_PREVIEW_MANIFEST_SHA256" in source
    assert "07251d55276a885eb928bf13d97d2fc629c4beb5345631c9c2d8d4ceed7d0f5c" in source
    assert "set(preview.get(\"candidates\", {})) != set(EXPECTED_CANDIDATES)" in source
    assert "MaterialExpressionTextureSampleParameter2D" in source
    assert 'expressions = [(latent, "RGBA", "LatentRGBA")]' in source
    assert "MaterialExpressionTextureCoordinate" not in source
    assert '"uv_source": "default_texcoord0"' in source
    assert "MaterialInstanceConstantFactoryNew" in source
    instance_start = source.index("def _create_instance(")
    instance_end = source.index("def _component(", instance_start)
    instance_source = source[instance_start:instance_end]
    assert "does_asset_exist(asset_path)" in instance_source
    assert instance_source.index("does_asset_exist(asset_path)") < instance_source.index(
        "load_asset(asset_path)"
    )
    assert "set_material_instance_parent(instance, master)" in source
    assert "update_material_instance(instance)" in source
    assert source.index("set_material_instance_parent(instance, master)") < source.index(
        "set_material_instance_texture_parameter_value"
    )
    assert source.index("instance = unreal.EditorAssetLibrary.load_asset(asset_path)", source.index("set_material_instance_parent(instance, master)")) < source.index(
        "set_material_instance_texture_parameter_value"
    )
    assert "set_material_instance_texture_parameter_value" in source
    assert "set_material_instance_vector_parameter_value" in source
    assert "set_material_instance_scalar_parameter_value" in source
    assert "is_material_instance_parameter_overridden" in source
    assert "if not library.set_material_instance_texture_parameter_value" not in source
    assert "TextureMipGenSettings.TMGS_SIMPLE_AVERAGE" in source
    assert "TextureFilter.TF_DEFAULT" in source
    assert 'texture.set_editor_property("never_stream", False)' in source
    assert 'readback["file_sha256_exact_source_match"]' in source
    assert 'readback["exact_source_match"]' not in source
    assert "save_dirty_packages" not in source
    assert "EditorAssetLibrary.save_asset(PREVIEW_MAP" in source
    assert "EditorAssetLibrary.save_asset(SOURCE_MAP" not in source
    assert 'execute_console_command(None, "MAP CHECK")' not in source
    for label in (
        "Helmet_Affine_P0_SAFE",
        "Helmet_Affine_P0_SAFE_REPAIR",
        "Helmet_Affine_P0_ENH_CHROMA4",
        "Helmet_Affine_P0_ENH_CHROMA8",
        "Helmet_Affine_L0_S040K",
        "Helmet_Affine_L0_S080K",
        "Helmet_Affine_L1_TV_R005_S040K",
        "Helmet_Affine_L2_CUBE_R005_S040K",
    ):
        assert label in source
