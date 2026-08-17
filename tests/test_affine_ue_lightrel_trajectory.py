from __future__ import annotations

from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cg_frontier.compression.affine_lightrel_trajectory_export import (  # noqa: E402
    DEFAULT_LIGHTREL_OUTPUT_RELATIVE_PATH,
    UE_ASSET_ROOT,
    load_lightrel_l0_trajectory,
)


@pytest.mark.asset
def test_loads_hash_bound_lightrel_parent_and_key_checkpoints() -> None:
    candidates = load_lightrel_l0_trajectory(ROOT)

    assert list(candidates) == [
        "p0_chroma8_parent",
        "l0_lightrel_s001k",
        "l0_lightrel_s005k",
        "l0_lightrel_s010k",
        "l0_lightrel_s020k",
        "l0_lightrel_s030k",
        "l0_lightrel_s040k",
    ]
    assert [candidate.optimizer_updates for candidate in candidates.values()] == [
        0,
        1_000,
        5_000,
        10_000,
        20_000,
        30_000,
        40_000,
    ]
    assert {candidate.parent_p0_hash for candidate in candidates.values()} == {
        "d9e630cfd93be58b7e083744fea71f0fc77247f55db7d8fd1693c83d016cb748"
    }
    assert all(candidate.certificate["valid"] for candidate in candidates.values())
    assert UE_ASSET_ROOT == "/Game/CGCompression/AffineLightrel40k"
    assert "lightrel-40k" in DEFAULT_LIGHTREL_OUTPUT_RELATIVE_PATH.as_posix()


def test_ue_setup_uses_an_isolated_lightrel_map_and_seven_nodes() -> None:
    setup_path = (
        ROOT
        / "ue_demo/CGCompressionDemo/Content/Python/"
        "setup_scifihelmet_affine_lightrel_40k_trajectory.py"
    )
    source = setup_path.read_text(encoding="utf-8")

    assert 'ASSET_ROOT = "/Game/CGCompression/AffineLightrel40k"' in source
    assert 'SOURCE_MAP = "/Game/CGCompression/Maps/MaterialLab"' in source
    assert "base.AFFINE_PREVIEW_AUTORUN = False" in source
    assert "base.setup()" in source
    for label in (
        "Helmet_Affine_P0_CHROMA8_PARENT",
        "Helmet_Affine_L0_LIGHTREL_S001K",
        "Helmet_Affine_L0_LIGHTREL_S005K",
        "Helmet_Affine_L0_LIGHTREL_S010K",
        "Helmet_Affine_L0_LIGHTREL_S020K",
        "Helmet_Affine_L0_LIGHTREL_S030K",
        "Helmet_Affine_L0_LIGHTREL_S040K",
    ):
        assert label in source
    assert "save_dirty_packages" not in source
    assert "EditorAssetLibrary.save_asset(SOURCE_MAP" not in source
