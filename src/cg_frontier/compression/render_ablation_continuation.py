"""Checkpoint contract for the Lantern 20k-to-40k C4 continuation."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Mapping

import torch

from cg_frontier.compression.render_ablation import (
    ARMS,
    checkpoint_identity_hash,
    tensor_sha256,
)


CHECKPOINT_SCHEMA = 1
CHECKPOINT_TYPE = "c4_render_ablation_lantern_40k_v1"
CHECKPOINT_STEPS = (30000, 40000)
OBSERVATION_STEPS = (25000, 30000, 35000, 40000)


def save_continuation_checkpoint(
    path: Path | str,
    *,
    arm: str,
    step: int,
    latent: torch.nn.Parameter,
    weight: torch.nn.Parameter,
    bias: torch.nn.Parameter,
    latent_optimizer: torch.optim.Optimizer,
    affine_optimizer: torch.optim.Optimizer,
    rng: torch.Generator,
    source_identity: Mapping[str, str],
    initial_rng_hash: str,
    continuation_config_hash: str,
    source_checkpoint_sha256: str,
) -> str:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite continuation checkpoint: {target}")
    if arm not in ARMS or step not in CHECKPOINT_STEPS:
        raise ValueError("invalid Lantern continuation checkpoint identity")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA,
        "checkpoint_type": CHECKPOINT_TYPE,
        "asset": "Lantern",
        "arm": arm,
        "step": int(step),
        "source_identity": dict(source_identity),
        "source_identity_hash": checkpoint_identity_hash(source_identity),
        "continuation_config_hash": continuation_config_hash,
        "source_checkpoint_sha256": source_checkpoint_sha256,
        "initial_rng_hash": initial_rng_hash,
        "current_rng_hash": tensor_sha256(rng.get_state()),
        "latent": latent.detach().cpu(),
        "weight": weight.detach().cpu(),
        "bias": bias.detach().cpu(),
        "latent_optimizer": latent_optimizer.state_dict(),
        "affine_optimizer": affine_optimizer.state_dict(),
        "rng_state": rng.get_state().cpu(),
    }
    torch.save(payload, target)
    return hashlib.sha256(target.read_bytes()).hexdigest()


def load_continuation_checkpoint(
    path: Path | str,
    *,
    expected_arm: str,
    expected_source_identity: Mapping[str, str],
    expected_continuation_config_hash: str,
    expected_source_checkpoint_sha256: str,
) -> dict[str, object]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != CHECKPOINT_SCHEMA
        or payload.get("checkpoint_type") != CHECKPOINT_TYPE
        or payload.get("asset") != "Lantern"
        or payload.get("arm") != expected_arm
        or payload.get("source_identity") != dict(expected_source_identity)
        or payload.get("source_identity_hash")
        != checkpoint_identity_hash(expected_source_identity)
        or payload.get("continuation_config_hash") != expected_continuation_config_hash
        or payload.get("source_checkpoint_sha256") != expected_source_checkpoint_sha256
        or int(payload.get("step", -1)) not in CHECKPOINT_STEPS
    ):
        raise ValueError("continuation checkpoint lineage does not match")
    for name in (
        "latent",
        "weight",
        "bias",
        "latent_optimizer",
        "affine_optimizer",
        "rng_state",
        "initial_rng_hash",
        "current_rng_hash",
    ):
        if name not in payload:
            raise ValueError("continuation checkpoint is incomplete")
    if (
        not math.isfinite(float(payload["latent"].float().mean()))
        or not math.isfinite(float(payload["weight"].float().mean()))
        or not math.isfinite(float(payload["bias"].float().mean()))
        or payload["current_rng_hash"] != tensor_sha256(payload["rng_state"])
    ):
        raise ValueError("continuation checkpoint contains invalid state")
    return payload
