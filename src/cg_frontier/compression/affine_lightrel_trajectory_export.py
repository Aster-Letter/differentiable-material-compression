"""Hash-bound UE preview inputs for the camera-relative L0 40k trajectory."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from cg_frontier.compression.affine_material import AFFINE_STATIC_COST
from cg_frontier.compression.affine_trajectory_export import (
    AffineTrajectoryCandidate,
    _load_checkpoint,
    _load_parent,
    _write_json,
    _write_rgba8_png,
)
from cg_frontier.compression.affine_ue_export import (
    affine_material_parameters,
    generate_shared_affine_hlsl,
)
from cg_frontier.compression.ue_export import sha256_file


SOURCE_1K_ROOT = Path(
    "outputs/scifihelmet_c4_affine_v1/train5k/"
    "a874ad-chroma8-l0-lightrel-r1"
)
CONTINUATION_ROOT = Path(
    "outputs/scifihelmet_c4_affine_v1/train40k/"
    "a874ad-chroma8-l0-lightrel-40k-r1"
)
DEFAULT_LIGHTREL_OUTPUT_RELATIVE_PATH = Path(
    "outputs/scifihelmet_c4_affine_v1/ue_preview/"
    "a874ad-lightrel-40k-trajectory-r1"
)
LIGHTREL_MANIFEST_NAME = "trajectory_manifest.json"
UE_ASSET_ROOT = "/Game/CGCompression/AffineLightrel40k"

_CHECKPOINT_SPECS = (
    (
        "l0_lightrel_s001k",
        1_000,
        "83053ef6ea5006c7f68168f904aad10ee941c891181a98591636c9d0e969a1fd",
        SOURCE_1K_ROOT,
    ),
    (
        "l0_lightrel_s005k",
        5_000,
        "e1674c3de7192b8ca6a028c747c5ebfd865dbd6f2435afc01d60bbea08b0a01e",
        CONTINUATION_ROOT,
    ),
    (
        "l0_lightrel_s010k",
        10_000,
        "5e11e50e80493268fdbe7aab2195c5dbd728e26e51fef62c0b2c383a4dcf3159",
        CONTINUATION_ROOT,
    ),
    (
        "l0_lightrel_s020k",
        20_000,
        "176fc534dd102b1b1bdec16dacc969d6359b43347fa532b6abe5238d316ce0fb",
        CONTINUATION_ROOT,
    ),
    (
        "l0_lightrel_s030k",
        30_000,
        "657f8027601a81b45f5e38fd4c8f4fce60faf9ceb523b9c9980d4508c3fa8462",
        CONTINUATION_ROOT,
    ),
    (
        "l0_lightrel_s040k",
        40_000,
        "1e73ce648be7d495ab21da66280b1346ef7f31157b300c4e0b7a59bff431cefa",
        CONTINUATION_ROOT,
    ),
)

_UE_TOKENS = {
    "p0_chroma8_parent": "P0_CHROMA8_PARENT",
    "l0_lightrel_s001k": "L0_LIGHTREL_S001K",
    "l0_lightrel_s005k": "L0_LIGHTREL_S005K",
    "l0_lightrel_s010k": "L0_LIGHTREL_S010K",
    "l0_lightrel_s020k": "L0_LIGHTREL_S020K",
    "l0_lightrel_s030k": "L0_LIGHTREL_S030K",
    "l0_lightrel_s040k": "L0_LIGHTREL_S040K",
}


def load_lightrel_l0_trajectory(
    repo_root: Path,
    *,
    fold_device: str = "cpu",
) -> dict[str, AffineTrajectoryCandidate]:
    """Load the certified parent and six immutable camera-relative checkpoints."""

    root = repo_root.resolve()
    result = {"p0_chroma8_parent": _load_parent(root)}
    for name, step, expected_file_hash, training_root in _CHECKPOINT_SPECS:
        result[name] = _load_checkpoint(
            root,
            name,
            step,
            expected_file_hash,
            fold_device=fold_device,
            training_root=training_root,
        )
    return result


def export_lightrel_l0_trajectory_package(
    repo_root: Path,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Export the seven isolated UE packages without touching existing previews."""

    root = repo_root.resolve()
    output = (
        root / DEFAULT_LIGHTREL_OUTPUT_RELATIVE_PATH
        if output_root is None
        else output_root.resolve()
    )
    output.mkdir(parents=True, exist_ok=True)
    candidates = load_lightrel_l0_trajectory(root, fold_device="cuda")
    hlsl_path = output / "shared_affine_decoder.hlsl"
    hlsl_path.write_text(generate_shared_affine_hlsl(), encoding="utf-8", newline="\n")
    master = f"{UE_ASSET_ROOT}/Materials/M_SciFiHelmet_Affine_Lightrel_Master"
    records: dict[str, Any] = {}
    for name, candidate in candidates.items():
        token = _UE_TOKENS[name]
        package_directory = Path("candidates") / name
        package_root = output / package_directory
        package_root.mkdir(parents=True, exist_ok=True)
        latent_path = package_root / "latent_rgba8.png"
        decoder_path = package_root / "decoder.bin"
        parameters_path = package_root / "material_parameters.json"
        _write_rgba8_png(latent_path, candidate.latent_unorm8)
        decoder_path.write_bytes(candidate.decoder_payload)
        _write_json(parameters_path, affine_material_parameters(candidate))
        record = {
            "artifact_hash": candidate.artifact_hash,
            "certificate": candidate.certificate,
            "generated_files": {
                "decoder.bin": sha256_file(decoder_path),
                "latent_rgba8.png": sha256_file(latent_path),
                "material_parameters.json": sha256_file(parameters_path),
            },
            "objective_id": candidate.objective_id,
            "optimizer_updates": candidate.optimizer_updates,
            "package_directory": package_directory.as_posix(),
            "parent_p0_hash": candidate.parent_p0_hash,
            "runtime": dict(candidate.static_cost),
            "source": {
                "relative_path": candidate.source_relative_path,
                "sha256": candidate.source_sha256,
            },
            "ue_assets": {
                "latent_texture": f"{UE_ASSET_ROOT}/Textures/T_SciFiHelmet_Affine_{token}",
                "master_material": master,
                "material_instance": f"{UE_ASSET_ROOT}/Materials/MI_SciFiHelmet_Affine_{token}",
            },
        }
        _write_json(package_root / "candidate_manifest.json", record)
        records[name] = record

    preview_map = f"{UE_ASSET_ROOT}/Maps/MaterialLab_Affine_Lightrel_40K_Trajectory"
    manifest = {
        "schema_version": 1,
        "pipeline_id": "scifihelmet_c4_affine_lightrel_l0_40k_trajectory_v1",
        "status": "exported_ue_setup_pending",
        "formal_holdout_accessed": False,
        "runtime_contract": {
            **dict(AFFINE_STATIC_COST),
            "decoder": "direct 4x7 affine plus positive normal-Z reconstruction",
            "filtering": "one derivative-LOD filtered RGBA8 sample",
            "mip_generation": "simple linear average without sharpening",
            "texture_streaming": True,
        },
        "ue_assets": {
            "asset_root": UE_ASSET_ROOT,
            "source_map": "/Game/CGCompression/Maps/MaterialLab",
            "preview_map": preview_map,
            "master_material": master,
        },
        "shared_files": {
            "shared_affine_decoder.hlsl": sha256_file(hlsl_path),
        },
        "candidates": records,
    }
    _write_json(output / LIGHTREL_MANIFEST_NAME, manifest)
    return manifest
