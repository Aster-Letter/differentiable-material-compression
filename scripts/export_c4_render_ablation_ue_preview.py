"""Export the verified three-asset C4 20k endpoints for Unreal preview."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

from PIL import Image
import torch
from torch.nn import functional as F
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cg_frontier.assets.gltf_core4 import load_gltf_core4_asset  # noqa: E402
from cg_frontier.assets.preprocess import sha256_file  # noqa: E402
from cg_frontier.compression.affine_pca import export_p0_bundle, rasterize_uv_charts  # noqa: E402
from cg_frontier.compression.basecolor_priority import postprocess_affine_output  # noqa: E402
from cg_frontier.compression.render_ablation import load_ablation_checkpoint  # noqa: E402


CONFIG_PATH = ROOT / "configs/train/c4_render_ablation_20k_v1.yaml"
EXTRACTED_ROOT = ROOT / "outputs/analysis/c4-render-ablation-20k-v1-job-37489/extracted"
RESULT_ROOT = EXTRACTED_ROOT / "outputs/remote/c4-render-ablation-20k-v1"
OUTPUT_ROOT = ROOT / "outputs/deployment/c4_render_ablation_20k_v1/ue_preview_job_37489"
UE_ASSET_ROOT = "/Game/CGCompression/C4RenderAblation20k"
ENDPOINTS = ("raw_q4", "material_only", "material_render")
RUNS = {"Corset": "37474", "Lantern": "37477", "BoomBox": "37478"}


def _seven(targets: object) -> torch.Tensor:
    return torch.cat(
        (
            targets.base_color_linear.reshape(targets.height, targets.width, 3),
            targets.normal_xyz.reshape(targets.height, targets.width, 3)[..., :2],
            targets.roughness.reshape(targets.height, targets.width, 1),
            targets.metallic.reshape(targets.height, targets.width, 1),
        ),
        dim=-1,
    )


def _write_json(path: Path, value: object) -> str:
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _hlsl() -> str:
    lines = [
        *(f"float Y{row} = dot(LatentRGBA, AffineW{row}) + AffineB{row};" for row in range(7)),
        "float2 NormalXY = float2(Y3, Y4);",
        "NormalXY /= max(1.0, length(NormalXY));",
        "float NormalZ = sqrt(max(1.0 - dot(NormalXY, NormalXY), 0.0));",
        "NormalTS = float3(NormalXY.x, NormalXY.y * NormalYSign, NormalZ);",
        "Roughness = saturate(Y5);",
        "Metallic = saturate(Y6);",
        "return saturate(float3(Y0, Y1, Y2));",
        "",
    ]
    return "\n".join(lines)


def _parameters(weight: torch.Tensor, bias: torch.Tensor) -> dict[str, object]:
    return {
        "texture_parameter": "LatentRGBA",
        "vector_parameters": {
            f"AffineW{row}": [float(value) for value in weight[row].tolist()]
            for row in range(7)
        },
        "scalar_parameters": {
            **{f"AffineB{row}": float(bias[row]) for row in range(7)},
            "NormalYSign": -1.0,
        },
        "postprocess": {
            "base_color": "saturate",
            "normal_xy": "unit_disk_projection",
            "normal_z": "positive_reconstruction",
            "roughness": "saturate",
            "metallic": "saturate",
        },
    }


def _source_files(gltf: Path) -> dict[str, str]:
    return {
        str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path)
        for path in sorted(gltf.parent.iterdir())
        if path.is_file()
    }


def _raw_state(
    gltf: Path,
    expected_parent_hash: str,
    frozen_metrics: dict[str, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
    asset = load_gltf_core4_asset(gltf, device="cpu")
    target = _seven(asset.targets)
    valid_mask, chart_ids = rasterize_uv_charts(
        asset.mesh.texcoords,
        asset.mesh.triangles,
        height=asset.targets.height,
        width=asset.targets.width,
    )
    raw = export_p0_bundle(target, valid_mask, chart_ids, margin=1.0e-3).calibration.raw
    prediction = postprocess_affine_output(
        F.linear(raw.latent_unorm8.float() / 255.0, raw.weight, raw.bias),
        compander_parameters=None,
        straight_through=False,
    ).seven[valid_mask]
    difference = (prediction - target[valid_mask]).abs()
    reconstructed_metrics = {
        "base_color_linear_mae": float(difference[:, :3].mean()),
        "seven_channel_mae": float(difference.mean()),
    }
    metric_abs_delta = {
        name: abs(value - float(frozen_metrics[name]))
        for name, value in reconstructed_metrics.items()
    }
    metric_tolerance = 5.0e-5
    if any(value > metric_tolerance for value in metric_abs_delta.values()):
        raise ValueError(
            f"raw q4 reconstruction differs materially from the frozen endpoint: "
            f"{gltf}; reconstructed={reconstructed_metrics}; "
            f"frozen={frozen_metrics}; delta={metric_abs_delta}"
        )
    evidence = {
        "expected_parent_hash": expected_parent_hash,
        "reconstructed_parent_hash": raw.artifact_hash,
        "bitwise_parent_hash_match": raw.artifact_hash == expected_parent_hash,
        "representation_note": (
            "CPU/LAPACK PCA basis is not bitwise canonical across the SCOW and local hosts; "
            "the reconstructed deployment is accepted only after frozen material metrics match."
        ),
        "reconstructed_metrics": reconstructed_metrics,
        "frozen_metrics": {
            name: float(frozen_metrics[name]) for name in reconstructed_metrics
        },
        "metric_abs_delta": metric_abs_delta,
        "metric_tolerance": metric_tolerance,
    }
    return raw.latent_unorm8, raw.weight, raw.bias, evidence


def export() -> dict[str, object]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    specs = {item["id"]: item for item in config["assets"]}
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    hlsl_path = OUTPUT_ROOT / "shared_affine_decoder.hlsl"
    hlsl_path.write_text(_hlsl(), encoding="utf-8", newline="\n")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "status": "exported_ue_setup_pending",
        "experiment": "c4_render_ablation_20k_v1",
        "archive_job_id": "37489",
        "formal_holdout_accessed": False,
        "deployment": {
            "texture_samples": 1,
            "latent_format": "RGBA8_UNORM",
            "decoder": "single_4_to_7_affine",
            "quantization_before_bilinear_sampling": True,
            "postprocess_matches_training": True,
        },
        "ue": {
            "asset_root": UE_ASSET_ROOT,
            "preview_map": f"{UE_ASSET_ROOT}/Maps/C4_Render_Ablation_20k",
            "master_material": f"{UE_ASSET_ROOT}/Materials/M_C4_Render_Ablation_Master",
        },
        "shared_files": {"shared_affine_decoder.hlsl": sha256_file(hlsl_path)},
        "assets": {},
    }
    for asset_id, job_id in RUNS.items():
        spec = specs[asset_id]
        gltf = ROOT / spec["gltf"]
        if sha256_file(gltf) != spec["gltf_sha256"]:
            raise ValueError(f"asset glTF hash mismatch: {asset_id}")
        pair_root = RESULT_ROOT / job_id / asset_id
        preparation = json.loads((pair_root / "preparation.json").read_text(encoding="utf-8"))
        identity = preparation["identity"]
        material_only_report = json.loads(
            (pair_root / "material_only/training_report.json").read_text(encoding="utf-8")
        )
        latent, weight, bias, raw_evidence = _raw_state(
            gltf,
            identity["parent_hash"],
            material_only_report["raw_parent"]["material"],
        )
        endpoints: dict[str, object] = {}
        for endpoint in ENDPOINTS:
            source: dict[str, object]
            if endpoint == "raw_q4":
                endpoint_latent, endpoint_weight, endpoint_bias = latent, weight, bias
                source = {"kind": "deterministic_raw_parent_reconstruction", **raw_evidence}
            else:
                checkpoint = pair_root / endpoint / "checkpoints/step_20000/checkpoint.pt"
                report = json.loads((pair_root / endpoint / "training_report.json").read_text(encoding="utf-8"))
                expected_hash = report["checkpoints"]["20000"]["sha256"]
                if sha256_file(checkpoint) != expected_hash:
                    raise ValueError(f"20k checkpoint hash mismatch: {asset_id}/{endpoint}")
                payload = load_ablation_checkpoint(
                    checkpoint,
                    expected_asset=asset_id,
                    expected_arm=endpoint,
                    expected_identity=identity,
                )
                endpoint_latent = torch.floor(payload["latent"].clamp(0.0, 1.0) * 255.0 + 0.5).to(torch.uint8)
                endpoint_weight = payload["weight"].float()
                endpoint_bias = payload["bias"].float()
                source = {
                    "kind": "formal_20k_checkpoint",
                    "job_id": job_id,
                    "arm": endpoint,
                    "checkpoint_sha256": expected_hash,
                    "identity_hash": preparation["identity_hash"],
                }
            package = OUTPUT_ROOT / "packages" / asset_id / endpoint
            package.mkdir(parents=True, exist_ok=True)
            latent_path = package / "latent_rgba8.png"
            Image.fromarray(endpoint_latent.cpu().numpy(), mode="RGBA").save(latent_path)
            parameters_path = package / "material_parameters.json"
            _write_json(parameters_path, _parameters(endpoint_weight, endpoint_bias))
            token = endpoint.upper()
            endpoints[endpoint] = {
                "package_directory": str(package.relative_to(OUTPUT_ROOT)).replace("\\", "/"),
                "source": source,
                "generated_files": {
                    "latent_rgba8.png": sha256_file(latent_path),
                    "material_parameters.json": sha256_file(parameters_path),
                },
                "ue_assets": {
                    "latent_texture": f"{UE_ASSET_ROOT}/Textures/T_{asset_id}_{token}_Latent",
                    "material_instance": f"{UE_ASSET_ROOT}/Materials/MI_{asset_id}_{token}",
                },
            }
        manifest["assets"][asset_id] = {
            "job_id": job_id,
            "gltf": str(gltf.relative_to(ROOT)).replace("\\", "/"),
            "gltf_sha256": spec["gltf_sha256"],
            "gltf_directory_files": _source_files(gltf),
            "identity": identity,
            "identity_hash": preparation["identity_hash"],
            "emissive_policy": spec["emissive_policy"],
            "ue_mesh_folder": f"{UE_ASSET_ROOT}/Meshes/{asset_id}",
            "endpoints": endpoints,
        }
    manifest_path = OUTPUT_ROOT / "preview_manifest.json"
    manifest_hash = _write_json(manifest_path, manifest)
    (OUTPUT_ROOT / "preview_manifest.json.sha256").write_text(manifest_hash + "\n", encoding="ascii")
    print(json.dumps({"status": "exported", "manifest": str(manifest_path), "sha256": manifest_hash}, sort_keys=True))
    return manifest


if __name__ == "__main__":
    export()
