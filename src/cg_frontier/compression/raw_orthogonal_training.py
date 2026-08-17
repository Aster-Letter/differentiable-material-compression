"""Checkpoint primitives for raw-PCA orthogonal BaseColor experiments."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import torch
from torch import nn


RAW_ORTHOGONAL_CHECKPOINT_SCHEMA = 2
RAW_ORTHOGONAL_LINEAGE_FIELDS = (
    "parent_artifact_hash",
    "config_sha256",
    "input_sha256",
    "basecolor_profile_hash",
    "visibility_hash",
)


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_lineage(lineage: Mapping[str, str]) -> dict[str, str]:
    missing = [field for field in RAW_ORTHOGONAL_LINEAGE_FIELDS if field not in lineage]
    if missing:
        raise ValueError(f"raw orthogonal lineage is missing: {','.join(missing)}")
    return {field: str(lineage[field]) for field in RAW_ORTHOGONAL_LINEAGE_FIELDS}


def save_raw_orthogonal_checkpoint(
    path: Path | str,
    *,
    step: int,
    candidate_id: str,
    objective_id: str,
    ratio: float,
    latent: nn.Parameter,
    weight: nn.Parameter,
    bias: nn.Parameter,
    latent_optimizer: torch.optim.Optimizer,
    affine_optimizer: torch.optim.Optimizer,
    core_rng: torch.Generator,
    color_rng: torch.Generator,
    lineage: Mapping[str, str],
) -> str:
    """Write one immutable raw schema-v2 checkpoint and return its SHA-256."""

    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {target}")
    if step < 0 or objective_id not in {"O1", "O2"} or not 0.0 <= ratio <= 1.0:
        raise ValueError("invalid raw orthogonal checkpoint identity")
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": RAW_ORTHOGONAL_CHECKPOINT_SCHEMA,
            "candidate_id": str(candidate_id),
            "objective_id": str(objective_id),
            "ratio": float(ratio),
            "step": int(step),
            **_validate_lineage(lineage),
            "latent": latent.detach().cpu(),
            "weight": weight.detach().cpu(),
            "bias": bias.detach().cpu(),
            "latent_optimizer": latent_optimizer.state_dict(),
            "affine_optimizer": affine_optimizer.state_dict(),
            "rng_state": core_rng.get_state().cpu(),
            "color_rng_state": color_rng.get_state().cpu(),
        },
        target,
    )
    return sha256_file(target)


def load_raw_orthogonal_checkpoint(
    path: Path | str,
    *,
    expected_candidate_id: str,
    expected_objective_id: str,
    expected_ratio: float,
    expected_lineage: Mapping[str, str],
) -> dict[str, object]:
    """Load schema v2 and fail closed on every experiment identity hash."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("schema_version") != 2:
        raise ValueError("raw orthogonal checkpoint must use schema v2")
    if payload.get("candidate_id") != expected_candidate_id:
        raise ValueError("raw orthogonal candidate mismatch")
    if payload.get("objective_id") != expected_objective_id:
        raise ValueError("raw orthogonal objective mismatch")
    if float(payload.get("ratio", -1.0)) != float(expected_ratio):
        raise ValueError("raw orthogonal ratio mismatch")
    lineage = _validate_lineage(expected_lineage)
    for field, expected in lineage.items():
        if payload.get(field) != expected:
            raise ValueError(f"raw orthogonal {field} mismatch")
    for field in (
        "latent",
        "weight",
        "bias",
        "latent_optimizer",
        "affine_optimizer",
        "rng_state",
        "color_rng_state",
    ):
        if field not in payload:
            raise ValueError(f"raw orthogonal checkpoint is missing {field}")
    return payload
