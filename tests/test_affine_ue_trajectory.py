from __future__ import annotations

from pathlib import Path
import sys

import pytest


pytestmark = pytest.mark.asset


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cg_frontier.compression.affine_trajectory_export import (  # noqa: E402
    DEFAULT_TRAJECTORY_OUTPUT_RELATIVE_PATH,
    TRAJECTORY_MANIFEST_NAME,
    export_chroma8_l0_trajectory_package,
    load_chroma8_l0_trajectory,
)
from cg_frontier.compression.ue_export import sha256_file  # noqa: E402


def test_loads_hash_bound_chroma8_parent_and_color_collapse_trajectory() -> None:
    candidates = load_chroma8_l0_trajectory(ROOT)

    assert list(candidates) == [
        "p0_chroma8_parent",
        "l0_s001k",
        "l0_s005k",
        "l0_s010k",
        "l0_s080k",
    ]
    assert [candidate.optimizer_updates for candidate in candidates.values()] == [
        0,
        1_000,
        5_000,
        10_000,
        80_000,
    ]
    assert {candidate.parent_p0_hash for candidate in candidates.values()} == {
        "d9e630cfd93be58b7e083744fea71f0fc77247f55db7d8fd1693c83d016cb748"
    }
    assert {candidate.objective_id for candidate in candidates.values()} == {
        "pca-safe-enhanced-chroma8",
        "material+helmet",
    }
    assert all(candidate.certificate["valid"] for candidate in candidates.values())
    assert {candidate.decoder_weights.shape for candidate in candidates.values()} == {
        (7, 4)
    }
    assert {candidate.decoder_bias.shape for candidate in candidates.values()} == {
        (7,)
    }
    assert {candidate.static_cost["parameters"] for candidate in candidates.values()} == {
        35
    }


def test_exports_isolated_deterministic_ue_trajectory_without_overwriting_progress() -> None:
    output = ROOT / DEFAULT_TRAJECTORY_OUTPUT_RELATIVE_PATH
    manifest = export_chroma8_l0_trajectory_package(ROOT, output)

    assert manifest["status"] == "exported_ue_setup_pending"
    assert manifest["formal_holdout_accessed"] is False
    assert manifest["ue_assets"] == {
        "asset_root": "/Game/CGCompression/AffineTrajectory",
        "source_map": "/Game/CGCompression/Maps/MaterialLab",
        "preview_map": (
            "/Game/CGCompression/AffineTrajectory/Maps/"
            "MaterialLab_Affine_Chroma8_Trajectory"
        ),
        "master_material": (
            "/Game/CGCompression/AffineTrajectory/Materials/"
            "M_SciFiHelmet_Affine_Trajectory_Master"
        ),
    }
    assert list(manifest["candidates"]) == [
        "p0_chroma8_parent",
        "l0_s001k",
        "l0_s005k",
        "l0_s010k",
        "l0_s080k",
    ]
    assert len(
        {
            item["ue_assets"]["material_instance"]
            for item in manifest["candidates"].values()
        }
    ) == 5
    assert {
        item["ue_assets"]["master_material"]
        for item in manifest["candidates"].values()
    } == {manifest["ue_assets"]["master_material"]}

    for item in manifest["candidates"].values():
        package_root = output / item["package_directory"]
        for filename, expected_hash in item["generated_files"].items():
            assert sha256_file(package_root / filename) == expected_hash

    endpoint = (
        ROOT
        / "outputs/scifihelmet_c4_affine_v1/train80k/"
        "a874ad-chroma8-l0-camera31-light6-r1/candidates/L0/step-080000"
    )
    final = manifest["candidates"]["l0_s080k"]
    final_root = output / final["package_directory"]
    assert (final_root / "decoder.bin").read_bytes() == (
        endpoint / "decoder.bin"
    ).read_bytes()
    from PIL import Image
    import numpy as np

    with Image.open(final_root / "latent_rgba8.png") as actual_image:
        actual = np.asarray(actual_image.convert("RGBA"))
    with Image.open(endpoint / "latent_rgba8.png") as expected_image:
        expected = np.asarray(expected_image.convert("RGBA"))
    assert np.array_equal(actual, expected)

    manifest_path = output / TRAJECTORY_MANIFEST_NAME
    first_hash = sha256_file(manifest_path)
    assert export_chroma8_l0_trajectory_package(ROOT, output) == manifest
    assert sha256_file(manifest_path) == first_hash


def test_ue_setup_adds_five_isolated_trajectory_nodes_and_preserves_source_map() -> None:
    base_path = (
        ROOT
        / "ue_demo/CGCompressionDemo/Content/Python/"
        "setup_scifihelmet_affine_preview.py"
    )
    setup_path = (
        ROOT
        / "ue_demo/CGCompressionDemo/Content/Python/"
        "setup_scifihelmet_affine_chroma8_trajectory.py"
    )
    base_source = base_path.read_text(encoding="utf-8")
    source = setup_path.read_text(encoding="utf-8")

    assert 'if globals().get("AFFINE_PREVIEW_AUTORUN", True):' in base_source
    assert "default_candidate = EXPECTED_CANDIDATES[0]" in base_source
    assert 'default_package = packages[default_candidate]' in base_source
    assert 'default_package = packages["p0_safe"]' not in base_source
    master_start = base_source.index("def _create_master_material(")
    master_end = base_source.index("def _create_instance(", master_start)
    master_source = base_source[master_start:master_end]
    assert "does_asset_exist(MASTER_MATERIAL)" in master_source
    assert master_source.index("does_asset_exist(MASTER_MATERIAL)") < master_source.index(
        "load_asset(MASTER_MATERIAL)"
    )
    assert 'ASSET_ROOT = "/Game/CGCompression/AffineTrajectory"' in source
    assert 'SOURCE_MAP = "/Game/CGCompression/Maps/MaterialLab"' in source
    assert (
        'PREVIEW_MAP = f"{ASSET_ROOT}/Maps/MaterialLab_Affine_Chroma8_Trajectory"'
        in source
    )
    assert "45b0866ef525082e7f2bba1bcaef82d46b26f8767f5ee7b8a6bd4693eae01d63" in source
    for label in (
        "Helmet_Affine_P0_CHROMA8_PARENT",
        "Helmet_Affine_L0_S001K",
        "Helmet_Affine_L0_S005K",
        "Helmet_Affine_L0_S010K",
        "Helmet_Affine_L0_S080K_CAMERA31",
    ):
        assert label in source
    assert "base.AFFINE_PREVIEW_AUTORUN = False" in source
    assert "base.setup()" in source
    assert "save_dirty_packages" not in source
    assert "EditorAssetLibrary.save_asset(SOURCE_MAP" not in source
