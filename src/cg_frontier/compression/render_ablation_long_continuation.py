"""Checkpoint contract for the Lantern material-render 40k-to-160k continuation."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Mapping

import torch

from cg_frontier.compression.render_ablation import checkpoint_identity_hash, tensor_sha256


CHECKPOINT_SCHEMA = 1
CHECKPOINT_TYPE = "c4_render_ablation_lantern_render_160k_v1"
CHECKPOINT_STEPS = (80000, 120000, 160000)
OBSERVATION_STEPS = (60000, 80000, 100000, 120000, 140000, 160000)


def save_long_continuation_checkpoint(
    path: Path | str,
    *,
    step: int,
    latent: torch.nn.Parameter,
    weight: torch.nn.Parameter,
    bias: torch.nn.Parameter,
    latent_optimizer: torch.optim.Optimizer,
    affine_optimizer: torch.optim.Optimizer,
    rng: torch.Generator,
    source_identity: Mapping[str, str],
    continuation_initial_rng_hash: str,
    original_initial_rng_hash: str,
    continuation_config_hash: str,
    source_40k_checkpoint_sha256: str,
) -> str:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite continuation checkpoint: {target}")
    if step not in CHECKPOINT_STEPS:
        raise ValueError("invalid long-continuation checkpoint step")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA,
        "checkpoint_type": CHECKPOINT_TYPE,
        "asset": "Lantern",
        "arm": "material_render",
        "step": int(step),
        "source_identity": dict(source_identity),
        "source_identity_hash": checkpoint_identity_hash(source_identity),
        "continuation_config_hash": continuation_config_hash,
        "source_40k_checkpoint_sha256": source_40k_checkpoint_sha256,
        "continuation_initial_rng_hash": continuation_initial_rng_hash,
        "original_initial_rng_hash": original_initial_rng_hash,
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


def load_long_continuation_checkpoint(
    path: Path | str,
    *,
    expected_source_identity: Mapping[str, str],
    expected_continuation_config_hash: str,
    expected_source_40k_checkpoint_sha256: str,
) -> dict[str, object]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != CHECKPOINT_SCHEMA
        or payload.get("checkpoint_type") != CHECKPOINT_TYPE
        or payload.get("asset") != "Lantern"
        or payload.get("arm") != "material_render"
        or payload.get("source_identity") != dict(expected_source_identity)
        or payload.get("source_identity_hash") != checkpoint_identity_hash(expected_source_identity)
        or payload.get("continuation_config_hash") != expected_continuation_config_hash
        or payload.get("source_40k_checkpoint_sha256") != expected_source_40k_checkpoint_sha256
        or int(payload.get("step", -1)) not in CHECKPOINT_STEPS
    ):
        raise ValueError("long-continuation checkpoint lineage does not match")
    for name in (
        "latent",
        "weight",
        "bias",
        "latent_optimizer",
        "affine_optimizer",
        "rng_state",
        "continuation_initial_rng_hash",
        "original_initial_rng_hash",
        "current_rng_hash",
    ):
        if name not in payload:
            raise ValueError("long-continuation checkpoint is incomplete")
    if (
        not math.isfinite(float(payload["latent"].float().mean()))
        or not math.isfinite(float(payload["weight"].float().mean()))
        or not math.isfinite(float(payload["bias"].float().mean()))
        or payload["current_rng_hash"] != tensor_sha256(payload["rng_state"])
    ):
        raise ValueError("long-continuation checkpoint contains invalid state")
    return payload
