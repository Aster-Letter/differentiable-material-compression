from __future__ import annotations

import hashlib
import json

import numpy as np
from PIL import Image

from cg_frontier.compression.exact_basecolor_experiment import stable_json_bytes
from cg_frontier.compression.ue_exact_basecolor_export import (
    CAMERAS,
    CANDIDATES,
    build_exact_affine_hlsl,
    export_ue_preview_bundle,
    verify_ue_texture_readbacks,
)


def _sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime_export(path, latent, weight, bias, candidate) -> None:
    path.mkdir(parents=True)
    texture = path / "latent_rgba_unorm8.png"
    decoder = path / "decoder_affine.npz"
    Image.fromarray(latent).save(texture)
    np.savez(decoder, weight=weight, bias=bias)
    manifest = {
        "schema_version": 1,
        "candidate": candidate,
        "codec": {},
        "runtime_inputs": [texture.name, decoder.name],
        "source_basecolor_required": False,
        "files": {texture.name: _sha256(texture), decoder.name: _sha256(decoder)},
    }
    (path / "export_manifest.json").write_bytes(stable_json_bytes(manifest))


def test_hlsl_is_one_affine_with_direct_basecolor_and_expected_postprocess() -> None:
    weight = np.arange(28, dtype=np.float32).reshape(7, 4) / 17.0
    bias = np.arange(7, dtype=np.float32) / 13.0
    hlsl = build_exact_affine_hlsl(weight, bias)
    assert hlsl.count("dot(z, float4(") == 7
    assert "return float3(raw0, raw1, raw2);" in hlsl
    assert "normalXY = tanh" in hlsl
    assert "normalXY /= max(normalRadius" in hlsl
    assert "Roughness = saturate(1.0f / (1.0f + exp(-raw5)))" in hlsl
    assert "Metallic = saturate(1.0f / (1.0f + exp(-raw6)))" in hlsl


def test_export_bundle_is_self_contained_and_selects_separated(tmp_path) -> None:
    experiment = tmp_path / "experiment"
    training = experiment / "training"
    report = experiment / "report"
    report.mkdir(parents=True)
    latent = np.array(
        [
            [[10, 20, 30, 0], [40, 50, 60, 64]],
            [[70, 80, 90, 128], [100, 110, 120, 255]],
        ],
        dtype=np.uint8,
    )
    weight = np.zeros((7, 4), dtype=np.float32)
    weight[:3, :3] = np.eye(3, dtype=np.float32)
    bias = np.zeros(7, dtype=np.float32)
    metrics = {}
    for candidate in CANDIDATES:
        candidate_root = training / candidate
        _runtime_export(candidate_root / "export", latent, weight, bias, candidate)
        render_root = candidate_root / "render_10000"
        render_root.mkdir(parents=True)
        for camera in CAMERAS:
            Image.new("RGB", (8, 8), (camera, 20, 30)).save(render_root / f"camera_{camera:02d}.png")
        metrics[candidate] = {"mean_hdr_mae": 0.1}
    (report / "final_summary.json").write_text(
        json.dumps({"10k_main_metrics": metrics, "scope": {"formal_holdout_accessed": False}}),
        encoding="utf-8",
    )
    Image.new("RGB", (8, 8), "red").save(report / "basecolor_u0_s_m.png")
    Image.new("RGB", (8, 8), "blue").save(report / "training_trajectories.png")

    output = tmp_path / "bundle"
    manifest = export_ue_preview_bundle(experiment, output)
    assert manifest["selection"] == "S-separated"
    assert manifest["runtime"]["samples_per_pixel"] == 1
    assert manifest["candidates"]["S-separated"]["recommended"] is True
    assert manifest["candidates"]["M-mixed"]["runtime_verification"]["finite"] is True
    assert (output / "M_SciFiHelmet_ExactBC_S.custom.hlsl").is_file()
    assert Image.open(output / "SciFiHelmet_ExactBaseColor_10k_RenderMontage.png").size == (1134, 802)
    assert Image.open(output / "SciFiHelmet_S_Separated_10k_4View.png").size == (512, 560)
    reloaded = json.loads((output / "ue_preview_manifest.json").read_text(encoding="utf-8"))
    assert reloaded["preview_map"].endswith("/ExactBaseColorV1")
    evidence = output / "evidence"
    evidence.mkdir()
    for record in manifest["candidates"].values():
        source = output / record["files"]["texture"]["name"]
        Image.open(source).save(evidence / f"ue_readback_{record['slug']}.png")
    readbacks = verify_ue_texture_readbacks(output)
    assert readbacks["passed"] is True
    assert all(value["max_abs"] == 0 for value in readbacks["candidates"].values())
