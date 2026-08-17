"""Validated Unreal Engine export inputs for the C4 affine progress preview."""

from __future__ import annotations

from dataclasses import dataclass
import io
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
from PIL import Image

from cg_frontier.compression.affine_material import (
    AFFINE_OUTPUT_SEMANTICS,
    AFFINE_STATIC_COST,
)
from cg_frontier.compression.ue_export import sha256_file


PARENT_P0_HASH = (
    "22a57aac4f2b86108dddf32ee76c7dbd1487d58a4f37b741eb143f34278dd98e"
)
DEFAULT_OUTPUT_RELATIVE_PATH = Path(
    "outputs/scifihelmet_c4_affine_v1/ue_preview/a874ad-progress-r3-enhanced"
)
PREVIEW_MANIFEST_NAME = "preview_manifest.json"
UE_ASSET_ROOT = "/Game/CGCompression/AffinePreview"

_UE_TOKENS = {
    "p0_safe": "P0_SAFE",
    "p0_safe_repair": "P0_SAFE_REPAIR",
    "p0_enhanced_chroma4": "P0_ENH_CHROMA4",
    "p0_enhanced_chroma8": "P0_ENH_CHROMA8",
    "l0_s040k": "L0_S040K",
    "l0_s080k": "L0_S080K",
    "l1_tv_r005_s040k": "L1_TV_R005_S040K",
    "l2_cube_r005_s040k": "L2_CUBE_R005_S040K",
}

_CANDIDATE_SPECS: tuple[tuple[str, dict[str, Any]], ...] = (
    (
        "p0_safe",
        {
            "relative_root": Path(
                "outputs/scifihelmet_c4_affine_v1/preflight/a874ad-r2/p0-safe"
            ),
            "manifest_sha256": (
                "2d3f0a8001c398356a497e015f0df5985b8ca0c32e750ccdf8bcde5bb0ede796"
            ),
            "optimizer_updates": 0,
            "objective_id": "pca-safe-certified",
            "candidate_id": "P0-safe",
        },
    ),
    (
        "p0_safe_repair",
        {
            "relative_root": Path(
                "outputs/scifihelmet_c4_affine_v1/pca_repair/a874ad-r2/"
                "p0-safe-constrained-v1"
            ),
            "manifest_sha256": (
                "ed9f59b5be3986469ec0f262fcaaed1f866d555ec861812e068d286705eb4b19"
            ),
            "optimizer_updates": 0,
            "objective_id": "pca-safe-certified-constrained",
            "candidate_id": "P0-safe-constrained-v1",
            "parent_p0_hash": (
                "1f2e66359a4d855e06010be2a2a9714ac54bde162161c4b07fd0cb79963fe709"
            ),
        },
    ),
    (
        "p0_enhanced_chroma4",
        {
            "relative_root": Path(
                "outputs/scifihelmet_c4_affine_v1/pca_audit/"
                "a874ad-enhanced-r1/candidates/chroma4"
            ),
            "manifest_sha256": (
                "3a9ea21240e9c7bc5c857f1bc5975f3a16b340fc868b574410f3128fece4c73b"
            ),
            "optimizer_updates": 0,
            "objective_id": "pca-safe-enhanced-chroma4",
            "candidate_id": "P0-safe-enhanced-chroma4",
            "parent_p0_hash": (
                "03a8f489f1a594f62e6eaaa1f1390455958941f0b196fd69692297a168f2e320"
            ),
        },
    ),
    (
        "p0_enhanced_chroma8",
        {
            "relative_root": Path(
                "outputs/scifihelmet_c4_affine_v1/pca_audit/"
                "a874ad-enhanced-r1/candidates/chroma8"
            ),
            "manifest_sha256": (
                "e50f6d1bc1645c6c40384cdc9b03d580a81954ca2d210e7f9f4a87c805416fcd"
            ),
            "optimizer_updates": 0,
            "objective_id": "pca-safe-enhanced-chroma8",
            "candidate_id": "P0-safe-enhanced-chroma8",
            "parent_p0_hash": (
                "d9e630cfd93be58b7e083744fea71f0fc77247f55db7d8fd1693c83d016cb748"
            ),
        },
    ),
    (
        "l0_s040k",
        {
            "relative_root": Path(
                "outputs/scifihelmet_c4_affine_v1/train40k/a874ad-r005/"
                "candidates/L0/step-040000"
            ),
            "manifest_sha256": (
                "8275cea8300481b7f239be510f65c17827d7de8b568546d1a3749b7ebc8d1030"
            ),
            "optimizer_updates": 40_000,
            "objective_id": "material+helmet",
            "candidate_id": "L0",
        },
    ),
    (
        "l0_s080k",
        {
            "relative_root": Path(
                "outputs/scifihelmet_c4_affine_v1/continuations/"
                "a874ad-l0-80k-r1/candidates/L0/step-080000"
            ),
            "manifest_sha256": (
                "dc9121844e3f6864fefc9f91d24d6a4075b34ad103183520b2bc25a06ce0aaca"
            ),
            "optimizer_updates": 80_000,
            "objective_id": "material+helmet",
            "candidate_id": "L0",
        },
    ),
    (
        "l1_tv_r005_s040k",
        {
            "relative_root": Path(
                "outputs/scifihelmet_c4_affine_v1/train40k/a874ad-r005/"
                "candidates/L1/step-040000"
            ),
            "manifest_sha256": (
                "3bf605696c732aa5ee1c7413d6745f1ae0a214fd51d08195e7736df1c9e6d5f6"
            ),
            "optimizer_updates": 40_000,
            "objective_id": "material+helmet+tv",
            "candidate_id": "L1",
        },
    ),
    (
        "l2_cube_r005_s040k",
        {
            "relative_root": Path(
                "outputs/scifihelmet_c4_affine_v1/train40k/a874ad-r005/"
                "candidates/L2/step-040000"
            ),
            "manifest_sha256": (
                "21fad9ae51c38c494b6c015fad3eca9f834a54dd3d558b7419a2cfadec2f8b3f"
            ),
            "optimizer_updates": 40_000,
            "objective_id": "material+helmet+cube",
            "candidate_id": "L2",
        },
    ),
)


@dataclass(frozen=True)
class AffineUECandidate:
    """One immutable, hash-validated affine material endpoint."""

    name: str
    root: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    latent_path: Path
    decoder_path: Path
    decoder_weights: np.ndarray
    decoder_bias: np.ndarray
    certificate: dict[str, Any]
    static_cost: dict[str, int]
    parent_p0_hash: str
    optimizer_updates: int
    objective_id: str
    artifact_hash: str


def compare_rgba8_png_bytes(source: bytes, readback: bytes) -> dict[str, Any]:
    """Compare decoded RGBA8 pixels independently of PNG file encoding."""

    with Image.open(io.BytesIO(source)) as source_image:
        source_pixels = np.asarray(source_image.convert("RGBA"), dtype=np.int16)
    with Image.open(io.BytesIO(readback)) as readback_image:
        readback_pixels = np.asarray(readback_image.convert("RGBA"), dtype=np.int16)
    if source_pixels.shape != readback_pixels.shape:
        return {
            "shape_match": False,
            "max_abs": None,
            "changed_values": None,
            "exact_pixel_match": False,
        }
    difference = np.abs(source_pixels - readback_pixels)
    max_abs = int(difference.max()) if difference.size else 0
    changed_values = int(np.count_nonzero(difference))
    return {
        "shape_match": True,
        "max_abs": max_abs,
        "changed_values": changed_values,
        "exact_pixel_match": changed_values == 0,
    }


def _load_one(repo_root: Path, name: str, spec: dict[str, Any]) -> AffineUECandidate:
    root = repo_root.resolve() / spec["relative_root"]
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file() or sha256_file(manifest_path) != spec["manifest_sha256"]:
        raise ValueError(f"frozen affine manifest hash mismatch: {name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    p0_names = {
        "p0_safe",
        "p0_safe_repair",
        "p0_enhanced_chroma4",
        "p0_enhanced_chroma8",
    }
    if name in p0_names:
        decoder_manifest = manifest["decoder_manifest"]
        certificate = manifest["safe_artifact"]["certificate"]
        static_cost = decoder_manifest["cost"]
        parent_p0_hash = manifest["safe_artifact"]["artifact_hash"]
        artifact_hash = parent_p0_hash
        if name == "p0_safe":
            identity_valid = (
                manifest.get("pipeline_id") == "scifihelmet_c4_affine_v1"
                and manifest["safe_artifact"].get("artifact_id") == "p0-safe"
            )
        elif name == "p0_safe_repair":
            identity_valid = (
                manifest.get("pipeline_id")
                == "scifihelmet_c4_affine_pca_repair_v1"
                and manifest["safe_artifact"].get("artifact_id")
                == "p0-safe-constrained-v1"
            )
        else:
            identity_valid = (
                manifest.get("pipeline_id")
                == "scifihelmet_c4_affine_pca_enhanced_v1"
                and manifest["safe_artifact"].get("artifact_id")
                == "p0-safe-enhanced-v1"
            )
    else:
        decoder_manifest = manifest["decoder_manifest"]
        certificate = manifest["certificate"]
        static_cost = manifest["static_cost"]
        parent_p0_hash = manifest["parent_p0_hash"]
        artifact_hash = manifest["artifact_hash"]
        identity_valid = (
            manifest.get("candidate_id") == spec["candidate_id"]
            and manifest.get("learned_linear") is True
        )

    if (
        not identity_valid
        or parent_p0_hash != spec.get("parent_p0_hash", PARENT_P0_HASH)
        or int(spec["optimizer_updates"]) != int(manifest.get("optimizer_updates", 0))
        or certificate.get("valid") is not True
        or decoder_manifest.get("semantics") != list(AFFINE_OUTPUT_SEMANTICS)
        or static_cost != AFFINE_STATIC_COST
    ):
        raise ValueError(f"affine endpoint contract mismatch: {name}")
    if name not in p0_names and manifest.get(
        "objective_id"
    ) != spec["objective_id"]:
        raise ValueError(f"affine endpoint objective mismatch: {name}")

    latent_path = root / "latent_rgba8.png"
    decoder_path = root / "decoder.bin"
    expected_hashes = manifest["hashes"]
    if sha256_file(latent_path) != expected_hashes["latent_png_sha256"]:
        raise ValueError(f"affine latent hash mismatch: {name}")
    if sha256_file(decoder_path) != expected_hashes["decoder_sha256"]:
        raise ValueError(f"affine decoder hash mismatch: {name}")
    payload = decoder_path.read_bytes()
    if len(payload) != 140 or sha256_file(decoder_path) != decoder_manifest["payload_sha256"]:
        raise ValueError(f"affine decoder payload mismatch: {name}")
    packed = np.frombuffer(payload, dtype="<f4").copy()
    if packed.shape != (35,) or not np.isfinite(packed).all():
        raise ValueError(f"invalid affine decoder values: {name}")
    with Image.open(latent_path) as image:
        if image.mode != "RGBA" or image.size != (2048, 2048):
            raise ValueError(f"affine latent must be 2048x2048 RGBA8: {name}")
        image.verify()

    return AffineUECandidate(
        name=name,
        root=root,
        manifest=manifest,
        manifest_sha256=spec["manifest_sha256"],
        latent_path=latent_path,
        decoder_path=decoder_path,
        decoder_weights=packed[:28].reshape(7, 4),
        decoder_bias=packed[28:],
        certificate=dict(certificate),
        static_cost=dict(static_cost),
        parent_p0_hash=parent_p0_hash,
        optimizer_updates=int(spec["optimizer_updates"]),
        objective_id=str(spec["objective_id"]),
        artifact_hash=artifact_hash,
    )


def load_affine_ue_candidates(repo_root: Path) -> dict[str, AffineUECandidate]:
    """Load the exact user-authorized progress and enhanced PCA candidates."""

    return {
        name: _load_one(repo_root, name, spec)
        for name, spec in _CANDIDATE_SPECS
    }


def affine_material_parameters(candidate: AffineUECandidate) -> dict[str, Any]:
    """Map one decoder payload to the shared UE master parameter names."""

    vector_parameters = {
        f"AffineW{row}": candidate.decoder_weights[row].tolist()
        for row in range(7)
    }
    scalar_parameters = {
        f"AffineB{row}": float(candidate.decoder_bias[row])
        for row in range(7)
    }
    scalar_parameters["NormalYSign"] = -1.0
    return {
        "texture_parameter": "LatentRGBA",
        "vector_parameters": vector_parameters,
        "scalar_parameters": scalar_parameters,
    }


def generate_shared_affine_hlsl() -> str:
    """Emit the shared UE Custom-expression body used by every preview MI."""

    lines: list[str] = []
    lines.extend(
        f"float Y{row} = dot(LatentRGBA, AffineW{row}) + AffineB{row};"
        for row in range(7)
    )
    lines.extend(
        [
            "float2 NormalXY = float2(Y3, Y4);",
            "float NormalZ = sqrt(max(1.0 - dot(NormalXY, NormalXY), 1.0e-8));",
            "NormalTS = float3(NormalXY.x, NormalXY.y * NormalYSign, NormalZ);",
            "Roughness = Y5;",
            "Metallic = Y6;",
            "return float3(Y0, Y1, Y2);",
            "",
        ]
    )
    return "\n".join(lines)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def export_affine_ue_preview_package(
    repo_root: Path,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Export a deterministic, hash-bound affine UE preview package."""

    repo_root = repo_root.resolve()
    output = (
        repo_root / DEFAULT_OUTPUT_RELATIVE_PATH
        if output_root is None
        else output_root.resolve()
    )
    output.mkdir(parents=True, exist_ok=True)
    candidates = load_affine_ue_candidates(repo_root)

    hlsl_path = output / "shared_affine_decoder.hlsl"
    hlsl_path.write_text(generate_shared_affine_hlsl(), encoding="utf-8", newline="\n")
    master_material = f"{UE_ASSET_ROOT}/Materials/M_SciFiHelmet_Affine_Master"
    result_candidates: dict[str, Any] = {}
    for name, candidate in candidates.items():
        token = _UE_TOKENS[name]
        package_directory = Path("candidates") / name
        package_root = output / package_directory
        package_root.mkdir(parents=True, exist_ok=True)
        latent_output = package_root / "latent_rgba8.png"
        decoder_output = package_root / "decoder.bin"
        parameters_output = package_root / "material_parameters.json"
        shutil.copyfile(candidate.latent_path, latent_output)
        shutil.copyfile(candidate.decoder_path, decoder_output)
        parameters = affine_material_parameters(candidate)
        _write_json(parameters_output, parameters)
        generated_files = {
            "decoder.bin": sha256_file(decoder_output),
            "latent_rgba8.png": sha256_file(latent_output),
            "material_parameters.json": sha256_file(parameters_output),
        }
        item = {
            "artifact_hash": candidate.artifact_hash,
            "certificate": candidate.certificate,
            "generated_files": generated_files,
            "objective_id": candidate.objective_id,
            "optimizer_updates": candidate.optimizer_updates,
            "package_directory": package_directory.as_posix(),
            "parent_p0_hash": candidate.parent_p0_hash,
            "runtime": dict(candidate.static_cost),
            "source": {
                "manifest_relative_path": candidate.root.relative_to(repo_root).as_posix()
                + "/manifest.json",
                "manifest_sha256": candidate.manifest_sha256,
            },
            "ue_assets": {
                "latent_texture": f"{UE_ASSET_ROOT}/Textures/T_SciFiHelmet_Affine_{token}",
                "master_material": master_material,
                "material_instance": f"{UE_ASSET_ROOT}/Materials/MI_SciFiHelmet_Affine_{token}",
            },
        }
        _write_json(package_root / "candidate_manifest.json", item)
        result_candidates[name] = item

    manifest = {
        "schema_version": 1,
        "pipeline_id": "scifihelmet_c4_affine_v1",
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
            "preview_map": f"{UE_ASSET_ROOT}/Maps/MaterialLab_Affine_Progress",
            "master_material": master_material,
        },
        "shared_files": {
            "shared_affine_decoder.hlsl": sha256_file(hlsl_path),
        },
        "candidates": result_candidates,
    }
    _write_json(output / PREVIEW_MANIFEST_NAME, manifest)
    return manifest
