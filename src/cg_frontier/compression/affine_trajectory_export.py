"""Hash-bound UE preview inputs for the chroma8 L0 training trajectory."""

from __future__ import annotations

from dataclasses import dataclass
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch

from cg_frontier.compression.affine_material import (
    AFFINE_STATIC_COST,
    SafeAffineMaterialDecoder,
    export_affine_decoder,
)
from cg_frontier.compression.affine_training import _checkpoint_hash
from cg_frontier.compression.affine_ue_export import (
    affine_material_parameters,
    generate_shared_affine_hlsl,
)
from cg_frontier.compression.ue_export import sha256_file


PARENT_P0_HASH = (
    "d9e630cfd93be58b7e083744fea71f0fc77247f55db7d8fd1693c83d016cb748"
)
PARENT_ROOT = Path(
    "outputs/scifihelmet_c4_affine_v1/pca_audit/"
    "a874ad-residual-r3/candidates/chroma8"
)
PARENT_MANIFEST_SHA256 = (
    "95fa5e05b55fbb2f613e5c8080f4598339700465561aa01c73d21becfa096e3f"
)
TRAINING_ROOT = Path(
    "outputs/scifihelmet_c4_affine_v1/train80k/"
    "a874ad-chroma8-l0-camera31-light6-r1"
)
DEFAULT_TRAJECTORY_OUTPUT_RELATIVE_PATH = Path(
    "outputs/scifihelmet_c4_affine_v1/ue_preview/"
    "a874ad-chroma8-l0-camera31-trajectory-r1"
)
TRAJECTORY_MANIFEST_NAME = "trajectory_manifest.json"
UE_ASSET_ROOT = "/Game/CGCompression/AffineTrajectory"

_UE_TOKENS = {
    "p0_chroma8_parent": "P0_CHROMA8_PARENT",
    "l0_s001k": "L0_S001K",
    "l0_s005k": "L0_S005K",
    "l0_s010k": "L0_S010K",
    "l0_s080k": "L0_S080K",
}

_CHECKPOINT_SPECS = (
    ("l0_s001k", 1_000, "ec0cedcd97c812c5dcb5e0d2109287a46da10b16383252ebf8b80d9036dd2204"),
    ("l0_s005k", 5_000, "b21fd100029dd7d7365bb7a49dce8110cf6c2dc8c30dd4948e20273fe0a8b624"),
    ("l0_s010k", 10_000, "2a860a7f30cda1f8d96a2e60a9fe274fb429f51b751ef1f404d7074d55b3029c"),
    ("l0_s080k", 80_000, "c91f921f670e4b61e10fca64e976b6c0a5f5d2217ba8ed908218010e26a10050"),
)


@dataclass(frozen=True)
class AffineTrajectoryCandidate:
    """One certified parent or immutable training checkpoint for preview."""

    name: str
    optimizer_updates: int
    objective_id: str
    parent_p0_hash: str
    artifact_hash: str
    latent_unorm8: np.ndarray
    decoder_payload: bytes
    decoder_weights: np.ndarray
    decoder_bias: np.ndarray
    certificate: dict[str, Any]
    static_cost: dict[str, int]
    source_relative_path: str
    source_sha256: str


def _unpack_decoder(payload: bytes) -> tuple[np.ndarray, np.ndarray]:
    packed = np.frombuffer(payload, dtype="<f4").copy()
    if packed.shape != (35,) or not np.isfinite(packed).all():
        raise ValueError("invalid affine decoder payload")
    return packed[:28].reshape(7, 4), packed[28:]


def _load_parent(repo_root: Path) -> AffineTrajectoryCandidate:
    root = repo_root / PARENT_ROOT
    manifest_path = root / "manifest.json"
    if sha256_file(manifest_path) != PARENT_MANIFEST_SHA256:
        raise ValueError("chroma8 parent manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    safe = manifest["safe_artifact"]
    if (
        manifest.get("pipeline_id") != "scifihelmet_c4_affine_pca_enhanced_v1"
        or safe.get("artifact_hash") != PARENT_P0_HASH
        or safe.get("certificate", {}).get("valid") is not True
    ):
        raise ValueError("chroma8 parent contract mismatch")
    latent_path = root / "latent_rgba8.png"
    decoder_path = root / "decoder.bin"
    if sha256_file(latent_path) != manifest["hashes"]["latent_png_sha256"]:
        raise ValueError("chroma8 parent latent hash mismatch")
    if sha256_file(decoder_path) != manifest["hashes"]["decoder_sha256"]:
        raise ValueError("chroma8 parent decoder hash mismatch")
    with Image.open(latent_path) as image:
        latent = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    payload = decoder_path.read_bytes()
    weight, bias = _unpack_decoder(payload)
    return AffineTrajectoryCandidate(
        name="p0_chroma8_parent",
        optimizer_updates=0,
        objective_id="pca-safe-enhanced-chroma8",
        parent_p0_hash=PARENT_P0_HASH,
        artifact_hash=PARENT_P0_HASH,
        latent_unorm8=latent,
        decoder_payload=payload,
        decoder_weights=weight,
        decoder_bias=bias,
        certificate=dict(safe["certificate"]),
        static_cost=dict(AFFINE_STATIC_COST),
        source_relative_path=(PARENT_ROOT / "manifest.json").as_posix(),
        source_sha256=PARENT_MANIFEST_SHA256,
    )


def _load_checkpoint(
    repo_root: Path,
    name: str,
    step: int,
    expected_file_hash: str,
    *,
    fold_device: str,
    training_root: Path = TRAINING_ROOT,
) -> AffineTrajectoryCandidate:
    relative_path = (
        training_root
        / "checkpoints/L0/endpoints"
        / f"step-{step:06d}/checkpoint.pt"
    )
    path = repo_root / relative_path
    if sha256_file(path) != expected_file_hash:
        raise ValueError(f"trajectory checkpoint file hash mismatch: {name}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("checkpoint_hash") != _checkpoint_hash(checkpoint):
        raise ValueError(f"trajectory checkpoint content hash mismatch: {name}")
    if (
        checkpoint.get("candidate_id") != "L0"
        or checkpoint.get("objective_id") != "material+helmet"
        or checkpoint.get("parent_p0_hash") != PARENT_P0_HASH
        or checkpoint.get("optimizer_updates") != step
    ):
        raise ValueError(f"trajectory checkpoint contract mismatch: {name}")
    latent = checkpoint.get("latent")
    parameters = checkpoint.get("safe_affine_raw_parameters")
    if not isinstance(latent, torch.Tensor) or not isinstance(parameters, dict):
        raise ValueError(f"trajectory checkpoint state missing: {name}")
    if fold_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to reproduce training-time affine folds")
    decoder = SafeAffineMaterialDecoder(margin=1.0e-3).to(fold_device)
    decoder.load_state_dict(
        {key: value.to(fold_device) for key, value in parameters.items()}
    )
    artifact = export_affine_decoder(decoder)
    weight, bias = _unpack_decoder(artifact.payload)
    latent_unorm8 = (
        torch.floor(latent.detach().clamp(0.0, 1.0) * 255.0 + 0.5)
        .to(torch.uint8)
        .numpy()
    )
    return AffineTrajectoryCandidate(
        name=name,
        optimizer_updates=step,
        objective_id="material+helmet",
        parent_p0_hash=PARENT_P0_HASH,
        artifact_hash=str(checkpoint["checkpoint_hash"]),
        latent_unorm8=latent_unorm8,
        decoder_payload=artifact.payload,
        decoder_weights=weight,
        decoder_bias=bias,
        certificate=dict(artifact.manifest["certificate"]),
        static_cost=dict(AFFINE_STATIC_COST),
        source_relative_path=relative_path.as_posix(),
        source_sha256=expected_file_hash,
    )


def load_chroma8_l0_trajectory(
    repo_root: Path,
    *,
    fold_device: str = "cpu",
) -> dict[str, AffineTrajectoryCandidate]:
    """Load the frozen parent and four diagnostic L0 checkpoints."""

    root = repo_root.resolve()
    result = {"p0_chroma8_parent": _load_parent(root)}
    result.update(
        {
            name: _load_checkpoint(
                root,
                name,
                step,
                expected_file_hash,
                fold_device=fold_device,
            )
            for name, step, expected_file_hash in _CHECKPOINT_SPECS
        }
    )
    return result


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_rgba8_png(path: Path, pixels: np.ndarray) -> None:
    if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[-1] != 4:
        raise ValueError("trajectory latent must be HxWx4 uint8")
    buffer = io.BytesIO()
    Image.fromarray(pixels, mode="RGBA").save(buffer, format="PNG")
    path.write_bytes(buffer.getvalue())


def export_chroma8_l0_trajectory_package(
    repo_root: Path,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Export five isolated, deterministic packages for UE trajectory review."""

    root = repo_root.resolve()
    output = (
        root / DEFAULT_TRAJECTORY_OUTPUT_RELATIVE_PATH
        if output_root is None
        else output_root.resolve()
    )
    output.mkdir(parents=True, exist_ok=True)
    # Training and the immutable 80k endpoint were folded on CUDA. Reusing that
    # device is required for byte-exact historical decoder reconstruction.
    candidates = load_chroma8_l0_trajectory(root, fold_device="cuda")
    hlsl_path = output / "shared_affine_decoder.hlsl"
    hlsl_path.write_text(generate_shared_affine_hlsl(), encoding="utf-8", newline="\n")
    master = f"{UE_ASSET_ROOT}/Materials/M_SciFiHelmet_Affine_Trajectory_Master"
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
                "latent_texture": (
                    f"{UE_ASSET_ROOT}/Textures/T_SciFiHelmet_Affine_{token}"
                ),
                "master_material": master,
                "material_instance": (
                    f"{UE_ASSET_ROOT}/Materials/MI_SciFiHelmet_Affine_{token}"
                ),
            },
        }
        _write_json(package_root / "candidate_manifest.json", record)
        records[name] = record

    manifest = {
        "schema_version": 1,
        "pipeline_id": "scifihelmet_c4_affine_chroma8_l0_trajectory_v1",
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
            "preview_map": (
                f"{UE_ASSET_ROOT}/Maps/MaterialLab_Affine_Chroma8_Trajectory"
            ),
            "master_material": master,
        },
        "shared_files": {
            "shared_affine_decoder.hlsl": sha256_file(hlsl_path),
        },
        "candidates": records,
    }
    _write_json(output / TRAJECTORY_MANIFEST_NAME, manifest)
    return manifest
