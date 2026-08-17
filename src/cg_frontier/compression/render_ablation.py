"""Frozen objectives, sampling evidence, and checkpoints for the C4 20k ablation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import torch


CHECKPOINT_SCHEMA = 1
CHECKPOINT_TYPE = "c4_render_ablation_20k_v1"
ARMS = ("material_only", "material_render")
OBSERVATION_STEPS = (1000, 5000, 10000, 15000, 20000)
FULL_CHECKPOINT_STEPS = (10000, 20000)
LOSS_KEYS = (
    "base_color_l1",
    "render_linear",
    "render_log",
    "normal_cosine",
    "roughness_l1",
    "metallic_l1",
)


@dataclass(frozen=True)
class LossWeights:
    base_color_l1: float = 1.0
    render_linear: float = 1.0
    render_log: float = 0.25
    normal_cosine: float = 0.25
    roughness_l1: float = 0.5
    metallic_l1: float = 0.5

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "LossWeights":
        if set(value) != set(LOSS_KEYS):
            raise ValueError("C4 render-ablation loss keys do not match the contract")
        result = cls(**{name: float(value[name]) for name in LOSS_KEYS})
        if result != cls():
            raise ValueError("C4 render-ablation loss weights differ from the frozen contract")
        return result


def compose_ablation_loss(
    terms: Mapping[str, torch.Tensor],
    *,
    arm: str,
    weights: LossWeights,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compose the paired material-only or material+render objective."""

    if arm not in ARMS or set(terms) != set(LOSS_KEYS):
        raise ValueError("invalid C4 render-ablation arm or term set")
    material = (
        weights.base_color_l1 * terms["base_color_l1"]
        + weights.normal_cosine * terms["normal_cosine"]
        + weights.roughness_l1 * terms["roughness_l1"]
        + weights.metallic_l1 * terms["metallic_l1"]
    )
    render = (
        weights.render_linear * terms["render_linear"]
        + weights.render_log * terms["render_log"]
    )
    included_render = render if arm == "material_render" else torch.zeros_like(render)
    total = material + included_render
    return total, {
        "material": material,
        "render": included_render,
        "diagnostic_render": render,
        "total": total,
    }


def tensor_sha256(value: torch.Tensor) -> str:
    cpu = value.detach().cpu().contiguous()
    header = json.dumps(
        {"dtype": str(cpu.dtype), "shape": list(cpu.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(header + b"\0" + cpu.numpy().tobytes(order="C")).hexdigest()


def sampling_contract_hash(
    *, seed: int, valid_texels: int, training_camera_indices: tuple[int, ...], lights: int
) -> str:
    payload = json.dumps(
        {
            "draw_order": ["material_positions", "training_camera", "light"],
            "positions_per_step": "training.material_batch_size",
            "rng": "torch.Generator(device=cuda)",
            "seed": int(seed),
            "valid_texels": int(valid_texels),
            "training_camera_indices": list(training_camera_indices),
            "lights": int(lights),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def checkpoint_identity_hash(identity: Mapping[str, str]) -> str:
    required = {
        "asset_hash",
        "config_hash",
        "parent_hash",
        "rig_hash",
        "sampling_contract_hash",
    }
    if set(identity) != required or any(not str(value) for value in identity.values()):
        raise ValueError("checkpoint identity fields do not match the contract")
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def sampling_trajectory_hash(
    *,
    sampling_contract: str,
    initial_rng: str,
    final_rng: str,
    steps: int,
) -> str:
    """Commit to the deterministic draw trajectory without logging sampled texels."""

    if not sampling_contract or not initial_rng or not final_rng or steps <= 0:
        raise ValueError("sampling trajectory commitment inputs are invalid")
    payload = json.dumps(
        {
            "sampling_contract_hash": sampling_contract,
            "initial_rng_hash": initial_rng,
            "final_rng_hash": final_rng,
            "steps": int(steps),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def save_ablation_checkpoint(
    path: Path | str,
    *,
    asset: str,
    arm: str,
    step: int,
    latent: torch.nn.Parameter,
    weight: torch.nn.Parameter,
    bias: torch.nn.Parameter,
    latent_optimizer: torch.optim.Optimizer,
    affine_optimizer: torch.optim.Optimizer,
    rng: torch.Generator,
    identity: Mapping[str, str],
    initial_rng_hash: str,
) -> str:
    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {target}")
    if arm not in ARMS or step not in FULL_CHECKPOINT_STEPS or not asset:
        raise ValueError("invalid C4 render-ablation checkpoint identity")
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": CHECKPOINT_SCHEMA,
        "checkpoint_type": CHECKPOINT_TYPE,
        "asset": asset,
        "arm": arm,
        "step": int(step),
        "identity": dict(identity),
        "identity_hash": checkpoint_identity_hash(identity),
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


def load_ablation_checkpoint(
    path: Path | str,
    *,
    expected_asset: str,
    expected_arm: str,
    expected_identity: Mapping[str, str],
) -> dict[str, object]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != CHECKPOINT_SCHEMA
        or payload.get("checkpoint_type") != CHECKPOINT_TYPE
        or payload.get("asset") != expected_asset
        or payload.get("arm") != expected_arm
        or payload.get("identity") != dict(expected_identity)
        or payload.get("identity_hash") != checkpoint_identity_hash(expected_identity)
    ):
        raise ValueError("checkpoint lineage does not match the requested ablation")
    required = (
        "latent",
        "weight",
        "bias",
        "latent_optimizer",
        "affine_optimizer",
        "rng_state",
        "initial_rng_hash",
        "current_rng_hash",
    )
    if any(name not in payload for name in required):
        raise ValueError("checkpoint is incomplete")
    step = int(payload.get("step", -1))
    if step not in FULL_CHECKPOINT_STEPS:
        raise ValueError("checkpoint step is outside the 10k/20k contract")
    if not math.isfinite(float(payload["latent"].float().mean())):
        raise ValueError("checkpoint contains non-finite parameters")
    return payload


def paired_sampling_evidence(left: Mapping[str, object], right: Mapping[str, object]) -> dict[str, object]:
    fields = (
        "sampling_contract_hash",
        "initial_rng_hash",
        "final_rng_hash",
        "sampling_trajectory_hash",
        "steps",
    )
    matches = {name: left.get(name) == right.get(name) for name in fields}
    return {"fields": matches, "identical": all(matches.values())}
