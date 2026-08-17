from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


pytestmark = pytest.mark.asset


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cg_frontier.compression.ue_decode_then_filter_export import (  # noqa: E402
    DEFAULT_OUTPUT_RELATIVE_PATH,
    PREVIEW_MANIFEST_NAME,
    export_dtf_preview_package,
    generate_candidate_hlsl,
    load_frozen_candidate,
    parse_candidate_hlsl_constants,
    sha256_file,
)


def test_c4_steps_are_distinct_and_preserve_decode_then_material_filter() -> None:
    candidate_80k = load_frozen_candidate(ROOT, "c4_dtf_16_s080k")
    candidate = load_frozen_candidate(ROOT, "c4_dtf_16_s160k")
    assert candidate_80k.manifest["status"] == "trained_local_80k"
    assert candidate_80k.manifest["training_result"]["completed_steps"] == 80_000
    assert candidate.manifest["status"] == "trained_local_160k_continuation"
    assert candidate.manifest["training_result"]["completed_steps"] == 160_000
    assert candidate.manifest["checkpoint_selection"]["best_render"]["step"] == 155_000
    assert (
        candidate.manifest["checkpoint_selection"]["best_artifact_safe"]["step"]
        == 160_000
    )
    assert candidate.latent_rgba_u8.shape == (2048, 2048, 4)
    assert candidate.latent_r_u8 is None
    assert candidate.metadata["latent_channels"] == 4
    assert candidate.metadata["texture_resources"] == 1
    assert candidate.metadata["point_texel_loads_per_pixel"] == 4
    assert candidate.metadata["theoretical_raw_bytes_unorm8"] == 16 * 1024 * 1024
    assert {name: array.shape for name, array in candidate.arrays.items()} == {
        "hidden_in.weight": (16, 4),
        "hidden_in.bias": (16,),
        "hidden_mid.weight": (16, 16),
        "hidden_mid.bias": (16,),
        "output.weight": (7, 16),
        "output.bias": (7,),
    }
    assert not np.array_equal(
        candidate_80k.latent_rgba_u8, candidate.latent_rgba_u8
    )

    source = generate_candidate_hlsl(candidate)
    reloaded = parse_candidate_hlsl_constants(source, latent_channels=4)
    for name, expected in candidate.arrays.items():
        np.testing.assert_array_equal(reloaded[name], expected)

    assert source.count("Texture2DSampleLevel(LatentRGBA") == 4
    assert "Texture2DSampleLevel(LatentR," not in source
    assert source.count("CGD_DecodeCorner(") == 5  # declaration plus four calls
    assert "CGD_FilteredNormalGLTF = normalize" in source
    assert "CGD_FilteredNormalGLTF.y * NormalYSign" in source


def test_c5_steps_are_distinct_frozen_candidates_with_the_same_runtime() -> None:
    candidate_80k = load_frozen_candidate(ROOT, "c5_dtf_16_s080k")
    candidate = load_frozen_candidate(ROOT, "c5_dtf_16_s120k")
    assert candidate_80k.manifest["status"] == "trained_local_80k"
    assert candidate_80k.manifest["training_result"]["completed_steps"] == 80_000
    assert candidate.manifest["status"] == "trained_local_120k_continuation"
    assert candidate.manifest["training_result"]["completed_steps"] == 120_000
    assert candidate.manifest["checkpoint_selection"]["best_render"]["step"] == 120_000
    assert (
        candidate.manifest["checkpoint_selection"]["best_artifact_safe"]["step"]
        == 120_000
    )
    assert candidate.latent_rgba_u8.shape == (2048, 2048, 4)
    assert candidate.latent_r_u8 is not None
    assert candidate.latent_r_u8.shape == (2048, 2048)
    assert candidate.metadata["latent_channels"] == 5
    assert candidate.metadata["texture_resources"] == 2
    assert candidate.metadata["point_texel_loads_per_pixel"] == 8
    assert candidate.metadata["theoretical_raw_bytes_unorm8"] == 20 * 1024 * 1024
    assert not np.array_equal(
        candidate_80k.latent_rgba_u8, candidate.latent_rgba_u8
    )

    source = generate_candidate_hlsl(candidate)
    reloaded = parse_candidate_hlsl_constants(source, latent_channels=5)
    for name, expected in candidate.arrays.items():
        np.testing.assert_array_equal(reloaded[name], expected)

    assert source.count("Texture2DSampleLevel(LatentRGBA") == 4
    assert source.count("Texture2DSampleLevel(LatentR,") == 4
    assert source.count("CGD_DecodeCorner(") == 5
    assert source.index("CGD.CGD_DecodeCorner") < source.index("CGD_FilteredBase")


def test_export_package_freezes_step_qualified_ue_materials() -> None:
    output = ROOT / DEFAULT_OUTPUT_RELATIVE_PATH
    manifest = export_dtf_preview_package(ROOT, output)
    assert manifest["status"] == "exported_ue_setup_pending"
    assert manifest["formal_holdout_accessed"] is False
    assert set(manifest["candidates"]) == {
        "c4_dtf_16_s080k",
        "c4_dtf_16_s160k",
        "c5_dtf_16_s080k",
        "c5_dtf_16_s120k",
    }

    c4_80k = manifest["candidates"]["c4_dtf_16_s080k"]
    c4_160k = manifest["candidates"]["c4_dtf_16_s160k"]
    c5_80k = manifest["candidates"]["c5_dtf_16_s080k"]
    c5_120k = manifest["candidates"]["c5_dtf_16_s120k"]
    assert c4_80k["ue_assets"]["material"].endswith(
        "M_SciFiHelmet_C4_DTF16_S080K"
    )
    assert c4_160k["ue_assets"]["material"].endswith(
        "M_SciFiHelmet_C4_DTF16_S160K"
    )
    assert c5_80k["ue_assets"]["material"].endswith(
        "M_SciFiHelmet_C5_DTF16_S080K"
    )
    assert c5_120k["ue_assets"]["material"].endswith(
        "M_SciFiHelmet_C5_DTF16_S120K"
    )
    assert c4_80k["runtime"]["point_texel_loads_per_pixel"] == 4
    assert c4_160k["runtime"]["point_texel_loads_per_pixel"] == 4
    assert c5_80k["runtime"]["point_texel_loads_per_pixel"] == 8
    assert c5_120k["runtime"]["point_texel_loads_per_pixel"] == 8
    assert c4_80k["selection_step"] == 80_000
    assert c4_160k["selection_step"] == 160_000
    assert c5_80k["selection_step"] == 80_000
    assert c5_120k["selection_step"] == 120_000
    assert c5_120k["source"]["training_manifest_sha256"] == (
        "0ba3797d9528c9a8bad77ac16e508ca480b7cbb29e4cb1c94121e796774c1bcb"
    )
    assert c4_160k["runtime"]["filter_order"] == c5_120k["runtime"]["filter_order"] == (
        "four_point_fetches_per_resource -> per_corner_decode_postprocess -> "
        "material_bilinear_filter -> one_normalize"
    )

    for candidate in (c4_80k, c4_160k, c5_80k, c5_120k):
        candidate_root = output / candidate["package_directory"]
        for filename, expected_hash in candidate["generated_files"].items():
            path = candidate_root / filename
            assert path.is_file()
            assert sha256_file(path) == expected_hash
    assert (output / PREVIEW_MANIFEST_NAME).is_file()


def test_ue_setup_isolated_from_existing_materiallab_state() -> None:
    setup_path = (
        ROOT
        / "ue_demo/CGCompressionDemo/Content/Python/setup_scifihelmet_dtf_preview.py"
    )
    source = setup_path.read_text(encoding="utf-8")
    assert 'ASSET_ROOT = "/Game/CGCompression/DTFPreview"' in source
    assert "MaterialExpressionTextureObject" in source
    assert "TextureFilter.TF_NEAREST" in source
    assert 'task.set_editor_property("replace_existing", True)' in source
    assert "save_dirty_packages" not in source
    assert "EditorAssetLibrary.save_asset(PREVIEW_MAP" in source
    assert "SOURCE_MAP" in source and "PREVIEW_MAP" in source
    assert "EXPECTED_PREVIEW_MANIFEST_SHA256" in source
    assert "duplicate_actors" not in source
    assert "spawn_actor_from_class" in source
    preview_hash = sha256_file(ROOT / DEFAULT_OUTPUT_RELATIVE_PATH / PREVIEW_MANIFEST_NAME)
    assert preview_hash in source
    assert '"c5_dtf_16_s080k"' in source
    assert '"c5_dtf_16_s120k"' in source
    assert '"c4_dtf_16_s080k"' in source
    assert '"c4_dtf_16_s160k"' in source
    assert 'label="Helmet_C4_DTF16_S080K"' in source
    assert 'label="Helmet_C4_DTF16_S160K"' in source
    assert 'label="Helmet_C5_DTF16_S080K"' in source
    assert 'label="Helmet_C5_DTF16_S120K"' in source
    assert '"C5_DTF16_S080K"' in source
    assert '"C5_DTF16_S120K"' in source
    assert '"C4_DTF16_S080K"' in source
    assert '"C4_DTF16_S160K"' in source
    assert "0ba3797d9528c9a8bad77ac16e508ca480b7cbb29e4cb1c94121e796774c1bcb" in source
