"""Export the verified Lantern material-render 160k endpoint for Unreal preview."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = (
    ROOT
    / "outputs/remote-archives/scow-c4-render-ablation-lantern-render-160k-v1-job-37824-archive-38202"
    / "extracted/outputs/remote/c4-render-ablation-lantern-render-160k-v1/37824"
)
LANTERN = RUN_ROOT / "Lantern"
OBSERVATION = LANTERN / "material_render/observations/step_160000"
REPORT = LANTERN / "material_render/training_report.json"
FORMAL = RUN_ROOT / "formal_verified.json"
RESULT_MANIFEST = RUN_ROOT / "result_manifest.json"
OLD_DEPLOYMENT = ROOT / "outputs/deployment/c4_render_ablation_20k_v1/ue_preview_job_37489"
OUTPUT = ROOT / "outputs/deployment/c4_render_ablation_lantern_160k_v1/ue_preview_job_37824"
UE_ROOT = "/Game/CGCompression/C4LanternRender160k"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> str:
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def parameters(affine: dict[str, object]) -> dict[str, object]:
    weight = affine["weight"]
    bias = affine["bias"]
    return {
        "texture_parameter": "LatentRGBA",
        "vector_parameters": {
            f"AffineW{row}": [float(value) for value in weight[row]] for row in range(7)
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


def export() -> dict[str, object]:
    formal = json.loads(FORMAL.read_text(encoding="utf-8"))
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    result_manifest = json.loads(RESULT_MANIFEST.read_text(encoding="utf-8"))
    old_manifest_path = OLD_DEPLOYMENT / "preview_manifest.json"
    old_manifest = json.loads(old_manifest_path.read_text(encoding="utf-8"))
    if (
        formal.get("status") != "formal_run_verified"
        or formal.get("formal_holdout_accessed") is not False
        or formal.get("endpoint_step") != 160000
        or report.get("steps") != 160000
        or report.get("arm") != "material_render"
        or report.get("asset") != "Lantern"
        or result_manifest.get("status") != "result_manifest_verified"
        or old_manifest.get("formal_holdout_accessed") is not False
    ):
        raise ValueError("Lantern 160k UE export contract mismatch")

    indexed = {entry["path"]: entry for entry in result_manifest["files"]}
    relative_files = {
        "material_render/observations/step_160000/latent_rgba8.png": OBSERVATION / "latent_rgba8.png",
        "material_render/observations/step_160000/affine.json": OBSERVATION / "affine.json",
    }
    for relative, path in relative_files.items():
        if relative not in indexed or sha256(path) != indexed[relative]["sha256"]:
            raise ValueError(f"verified result file mismatch: {relative}")

    image = Image.open(OBSERVATION / "latent_rgba8.png")
    if image.mode != "RGBA":
        raise ValueError(f"expected RGBA8 latent, got {image.mode}")
    package = OUTPUT / "packages/Lantern/material_render_160k"
    package.mkdir(parents=True, exist_ok=True)
    latent_path = package / "latent_rgba8.png"
    image.save(latent_path)
    affine = json.loads((OBSERVATION / "affine.json").read_text(encoding="utf-8"))
    parameter_path = package / "material_parameters.json"
    write_json(parameter_path, parameters(affine))

    manifest = {
        "schema_version": 1,
        "status": "exported_ue_setup_pending",
        "experiment": "c4_render_ablation_lantern_render_160k_v1",
        "formal_job_id": "37824",
        "archive_job_id": "38202",
        "source_job_id": "37581",
        "source_step": 40000,
        "endpoint_step": 160000,
        "arm": "material_render",
        "formal_holdout_accessed": False,
        "emissive_excluded_fraction_approx": 0.0327,
        "lineage": {
            "checkpoint_sha256": formal["checkpoint_hashes"]["160000"],
            "source_identity": formal["source_identity"],
            "old_20k_preview_manifest_sha256": sha256(old_manifest_path),
        },
        "deployment": {
            "latent_format": "RGBA8_UNORM",
            "texture_samples": 1,
            "decoder": "single_4_to_7_affine",
            "quantization_before_bilinear_sampling": True,
        },
        "package_directory": str(package.relative_to(OUTPUT)).replace("\\", "/"),
        "generated_files": {
            "latent_rgba8.png": sha256(latent_path),
            "material_parameters.json": sha256(parameter_path),
        },
        "ue": {
            "asset_root": UE_ROOT,
            "preview_map": f"{UE_ROOT}/Maps/C4_Lantern_Source_Raw_20k_160k",
            "latent_texture": f"{UE_ROOT}/Textures/T_Lantern_MATERIAL_RENDER_160K_Latent",
            "material_instance": f"{UE_ROOT}/Materials/MI_Lantern_MATERIAL_RENDER_160K",
            "existing_master_material": "/Game/CGCompression/C4RenderAblation20k/Materials/M_C4_Render_Ablation_Master",
            "existing_source_map": "/Game/CGCompression/C4RenderAblation20k/Maps/C4_Render_Ablation_20k",
        },
    }
    manifest_path = OUTPUT / "preview_manifest.json"
    digest = write_json(manifest_path, manifest)
    (OUTPUT / "preview_manifest.json.sha256").write_text(digest + "\n", encoding="ascii")
    print(json.dumps({"status": "exported", "manifest": str(manifest_path), "sha256": digest}, sort_keys=True))
    return manifest


if __name__ == "__main__":
    export()
