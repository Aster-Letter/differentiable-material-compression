from __future__ import annotations

import pytest
from pathlib import Path
import yaml

from cg_frontier.assets.complex_c4_validation import (
    audit_complex_gltf_document,
    select_complex_validators,
)
from cg_frontier.assets.gltf_mesh import GltfMeshError
import scripts.audit_complex_c4_assets as asset_screen
from scripts.audit_complex_c4_assets import _selection_render_metrics, run


def _document(*, emissive: bool = False, extensions: bool = False):
    material = {
        "pbrMetallicRoughness": {
            "baseColorTexture": {"index": 0},
            "metallicRoughnessTexture": {"index": 1},
        },
        "normalTexture": {"index": 2},
    }
    if emissive:
        material["emissiveTexture"] = {"index": 3}
    value = {
        "meshes": [{"primitives": [{"material": 0}]}],
        "materials": [material],
    }
    if extensions:
        value["extensionsUsed"] = ["KHR_materials_clearcoat"]
    return value


def test_complex_asset_contract_fails_closed_on_license_extensions_and_emissive_review() -> None:
    clean = audit_complex_gltf_document(
        _document(), license_spdx="CC0-1.0", emissive_identity_preserved=None
    )
    assert clean.eligible is True
    assert clean.emissive_review == "not_required"

    pending = audit_complex_gltf_document(
        _document(emissive=True),
        license_spdx="CC0-1.0",
        emissive_identity_preserved=None,
    )
    assert pending.eligible is False
    assert pending.emissive_review == "pending"

    with pytest.raises(ValueError):
        audit_complex_gltf_document(
            _document(), license_spdx="CC-BY-4.0", emissive_identity_preserved=True
        )
    with pytest.raises(ValueError):
        audit_complex_gltf_document(
            _document(extensions=True),
            license_spdx="CC0-1.0",
            emissive_identity_preserved=True,
        )


def test_complex_validator_selection_applies_capacity_gates_and_complementarity() -> None:
    reports = [
        {
            "asset_id": "BoomBox",
            "contract_eligible": True,
            "core4_identity_preserved": True,
            "q4_basecolor_error": 1.0,
            "q6_basecolor_error": 0.7,
            "q4_seven_error": 1.0,
            "q6_seven_error": 0.9,
            "q4_hdr_mae": 1.0,
            "q6_hdr_mae": 0.85,
            "q4_ssim": 0.80,
            "q6_ssim": 0.805,
            "basecolor_q4_excess": 4.0,
            "nonbase_standardized_residual": 1.0,
        },
        {
            "asset_id": "Corset",
            "contract_eligible": True,
            "core4_identity_preserved": True,
            "q4_basecolor_error": 1.0,
            "q6_basecolor_error": 0.9,
            "q4_seven_error": 1.0,
            "q6_seven_error": 0.7,
            "q4_hdr_mae": 1.0,
            "q6_hdr_mae": 0.95,
            "q4_ssim": 0.80,
            "q6_ssim": 0.82,
            "basecolor_q4_excess": 1.5,
            "nonbase_standardized_residual": 5.0,
        },
        {
            "asset_id": "WaterBottle",
            "contract_eligible": True,
            "core4_identity_preserved": True,
            "q4_basecolor_error": 1.0,
            "q6_basecolor_error": 0.9,
            "q4_seven_error": 1.0,
            "q6_seven_error": 0.9,
            "q4_hdr_mae": 1.0,
            "q6_hdr_mae": 0.95,
            "q4_ssim": 0.80,
            "q6_ssim": 0.805,
            "basecolor_q4_excess": 8.0,
            "nonbase_standardized_residual": 8.0,
        },
    ]

    selection = select_complex_validators(reports)

    assert selection.basecolor_dominant == "BoomBox"
    assert selection.cross_channel_dominant == "Corset"
    assert selection.eligible_assets == ("BoomBox", "Corset")


@pytest.mark.asset
def test_checked_in_complex_asset_config_passes_contract_only_audit() -> None:
    result = run(
        Path("configs/eval/complex_c4_asset_screen_v1.yaml"), contract_only=True
    )
    rows = {row["asset_id"]: row for row in result["assets"]}

    assert result["status"] == "complete_contract_only"
    assert rows["Corset"]["contract_eligible"] is True
    assert rows["BoomBox"]["contract_eligible"] is True
    assert rows["BoomBox"]["emissive_review"] == "accepted"
    assert rows["WaterBottle"]["contract_eligible"] is True
    assert rows["WaterBottle"]["emissive_review"] == "accepted"


def test_asset_screen_selection_metrics_use_masked_hdr_schema() -> None:
    render_metrics = {
        "raw_q4": {
            "mean": {"masked_linear_hdr_mae": 0.25, "display_ssim": 0.8}
        },
        "oracle_q6": {
            "mean": {"masked_linear_hdr_mae": 0.1, "display_ssim": 0.9}
        },
    }

    assert _selection_render_metrics(render_metrics) == {
        "q4_hdr_mae": 0.25,
        "q6_hdr_mae": 0.1,
        "q4_ssim": 0.8,
        "q6_ssim": 0.9,
    }


@pytest.mark.asset
def test_asset_screen_marks_unsupported_mesh_runtime_ineligible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = yaml.safe_load(
        Path("configs/eval/complex_c4_asset_screen_v1.yaml").read_text(encoding="utf-8")
    )

    def reject_mesh(*_args, **_kwargs):
        raise GltfMeshError("cannot reconstruct tangents: 80 degenerate UV triangles")

    writes = []
    monkeypatch.setattr(asset_screen, "_process_asset", reject_mesh)
    monkeypatch.setattr(
        asset_screen,
        "_write_json",
        lambda path, value: writes.append((path, value)) or "report-hash",
    )
    report = asset_screen._process_asset_or_rejection(
        config["assets"][2], config, Path("unused-output")
    )

    assert report["asset_id"] == "WaterBottle"
    assert report["source_contract_eligible"] is True
    assert report["contract_eligible"] is False
    assert report["runtime_eligible"] is False
    assert report["screen_rejection"]["type"] == "GltfMeshError"
    assert "80 degenerate UV triangles" in report["screen_rejection"]["message"]
    assert writes == [(Path("unused-output/WaterBottle/report.json"), report)]
