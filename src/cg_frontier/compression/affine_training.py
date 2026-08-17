"""Paired L0/L1/L2 state, checkpointing, and short-run timing."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib
import io
import os
from pathlib import Path
import time
from typing import Callable, Protocol, Sequence

import torch
from torch import nn

from cg_frontier.compression.affine_material import SafeAffineMaterialDecoder, certify_affine
from cg_frontier.compression.affine_color import (
    ColorQuantilePartition,
    draw_color_batch,
)
from cg_frontier.compression.affine_pca import P0Calibration


@dataclass
class AffineCandidateState:
    candidate_id: str
    objective_id: str
    parent_p0_hash: str
    config_hash: str
    input_hash: str
    latent: nn.Parameter
    decoder: SafeAffineMaterialDecoder
    latent_optimizer: torch.optim.Optimizer
    affine_optimizer: torch.optim.Optimizer
    core_rng: torch.Generator
    cube_rng: torch.Generator | None
    color_rng: torch.Generator | None = None
    color_partition_hash: str | None = None
    color_group_hash: str | None = None
    optimizer_updates: int = 0
    phase: str = "warmup"
    best_metadata: dict[str, object] = field(default_factory=dict)
    continuation_from_checkpoint_hash: str | None = None
    continuation_from_step: int | None = None


@dataclass(frozen=True)
class AffineTrainingBatch:
    core_indices: torch.Tensor
    cube_samples: torch.Tensor | None
    color_indices: torch.Tensor | None = None
    color_bin_ids: torch.Tensor | None = None


@dataclass(frozen=True)
class AffineTrainingStep:
    batch: AffineTrainingBatch
    loss: float
    terms: dict[str, float]


@dataclass(frozen=True)
class CheckpointWrite:
    path: Path
    checkpoint_hash: str
    endpoint: bool


@dataclass(frozen=True)
class CandidateTrainingRun:
    manifest: dict[str, object]
    curve: tuple[dict[str, object], ...]
    parameter_trends: tuple[dict[str, object], ...]
    checkpoints: tuple[CheckpointWrite, ...]


@dataclass(frozen=True)
class CandidateTiming:
    warmup_steps: int
    measured_steps: int
    step_times_ms: tuple[float, ...]
    median_step_ms: float
    p95_step_ms: float
    samples_per_second: float
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    mean_loss_terms: dict[str, float]


def select_render_pair(
    independent_draws: Sequence[int], *, camera_count: int, light_count: int
) -> tuple[int, int]:
    """Select camera/light from two independently sampled, resumable batch values."""

    if len(independent_draws) < 2 or camera_count <= 0 or light_count <= 0:
        raise ValueError("render-pair selection requires two draws and non-empty pools")
    return int(independent_draws[0]) % camera_count, int(independent_draws[1]) % light_count


@dataclass(frozen=True)
class TrainingObservationPlan:
    total_steps: int
    checkpoint_steps: tuple[int, ...]
    trend_steps: tuple[int, ...]


def plan_training_observations(
    *,
    total_steps: int,
    checkpoint_steps: tuple[int, ...],
    trend_interval: int,
) -> TrainingObservationPlan:
    """Validate and freeze step-qualified checkpoints plus dense trend samples."""

    if total_steps <= 0 or trend_interval <= 0:
        raise ValueError("training observation steps must be positive")
    frozen_checkpoints = tuple(int(step) for step in checkpoint_steps)
    if (
        not frozen_checkpoints
        or frozen_checkpoints != tuple(sorted(set(frozen_checkpoints)))
        or frozen_checkpoints[-1] != total_steps
        or frozen_checkpoints[0] <= 0
    ):
        raise ValueError("checkpoint steps must be unique, increasing, and end at total_steps")
    trend_steps = tuple(range(trend_interval, total_steps + 1, trend_interval))
    if not trend_steps or trend_steps[-1] != total_steps:
        raise ValueError("trend interval must land exactly on total_steps")
    return TrainingObservationPlan(
        total_steps=total_steps,
        checkpoint_steps=frozen_checkpoints,
        trend_steps=trend_steps,
    )


def candidate_parameter_trend(
    state: AffineCandidateState,
    parent_p0: P0Calibration,
) -> dict[str, object]:
    """Summarize one candidate against its immutable P0-safe parent."""

    if state.parent_p0_hash != parent_p0.safe.artifact_hash:
        raise ValueError("candidate parent_p0_hash mismatch")
    latent = state.latent.detach()
    parent_latent = parent_p0.safe.latent_unorm8.to(
        device=latent.device, dtype=latent.dtype
    ) / 255.0
    if latent.shape != parent_latent.shape:
        raise ValueError("candidate latent shape does not match P0-safe")
    weight, bias = state.decoder.fold_affine()
    parent_weight = parent_p0.safe.weight.to(device=weight.device, dtype=weight.dtype)
    parent_bias = parent_p0.safe.bias.to(device=bias.device, dtype=bias.dtype)
    channel_dims = tuple(range(latent.ndim - 1))
    certificate = certify_affine(weight, bias, margin=state.decoder.margin)
    return {
        "candidate_id": state.candidate_id,
        "objective_id": state.objective_id,
        "parent_p0_hash": state.parent_p0_hash,
        "step": state.optimizer_updates,
        "phase": state.phase,
        "latent_delta_rmse": float(
            torch.sqrt(torch.mean((latent - parent_latent).square())).cpu()
        ),
        "latent_channel_mean": [
            float(value) for value in latent.mean(dim=channel_dims).cpu()
        ],
        "latent_channel_std": [
            float(value) for value in latent.std(dim=channel_dims, unbiased=False).cpu()
        ],
        "latent_saturation_fraction": float(
            torch.mean(((latent <= 0.0) | (latent >= 1.0)).to(torch.float32)).cpu()
        ),
        "affine_weight_l2": float(torch.linalg.vector_norm(weight).detach().cpu()),
        "affine_bias_l2": float(torch.linalg.vector_norm(bias).detach().cpu()),
        "affine_weight_delta_l2": float(
            torch.linalg.vector_norm(weight - parent_weight).detach().cpu()
        ),
        "affine_bias_delta_l2": float(
            torch.linalg.vector_norm(bias - parent_bias).detach().cpu()
        ),
        "latent_learning_rate": float(state.latent_optimizer.param_groups[0]["lr"]),
        "affine_learning_rate": float(state.affine_optimizer.param_groups[0]["lr"]),
        "certificate": certificate,
    }


class CheckpointStorage(Protocol):
    def write(self, path: Path, payload: bytes, *, immutable: bool) -> None: ...


@dataclass
class InMemoryCheckpointStorage:
    """Deterministic storage boundary used by synthetic checkpoint tests."""

    files: dict[Path, bytes] = field(default_factory=dict)

    def write(self, path: Path, payload: bytes, *, immutable: bool) -> None:
        if immutable and path in self.files:
            raise FileExistsError(path)
        self.files[path] = payload


class FilesystemCheckpointStorage:
    """Atomic filesystem checkpoint storage with immutable endpoint directories."""

    def write(self, path: Path, payload: bytes, *, immutable: bool) -> None:
        directory = path.parent
        if immutable:
            directory.parent.mkdir(parents=True, exist_ok=True)
            directory.mkdir(exist_ok=False)
        else:
            directory.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_bytes(payload)
        os.replace(temporary, path)


def draw_training_batch(
    state: AffineCandidateState,
    *,
    texel_count: int,
    batch_size: int,
    cube_sample_count: int,
    color_partition: ColorQuantilePartition | None = None,
    color_batch_size: int = 0,
) -> AffineTrainingBatch:
    """Draw paired core samples and optional L2-only cube samples."""

    core_indices = torch.randint(
        0,
        texel_count,
        (batch_size,),
        generator=state.core_rng,
        device=state.latent.device,
    )
    cube_samples = None
    if state.cube_rng is not None:
        cube_samples = torch.rand(
            cube_sample_count,
            2,
            generator=state.cube_rng,
            device=state.latent.device,
            dtype=state.latent.dtype,
        )
    color_indices = None
    color_bin_ids = None
    if color_partition is not None:
        if state.color_rng is None or state.color_partition_hash is None:
            raise ValueError("color partition requires a color-enabled candidate")
        if state.color_partition_hash != color_partition.partition_hash:
            raise ValueError("candidate color partition hash mismatch")
        color = draw_color_batch(
            color_partition, generator=state.color_rng, batch_size=color_batch_size
        )
        color_indices = color.valid_positions
        color_bin_ids = color.logical_bin_ids
    elif state.color_rng is not None or color_batch_size != 0:
        raise ValueError("color-enabled candidates require their frozen partition")
    return AffineTrainingBatch(
        core_indices=core_indices,
        cube_samples=cube_samples,
        color_indices=color_indices,
        color_bin_ids=color_bin_ids,
    )


def train_candidate_step(
    state: AffineCandidateState,
    objective: Callable[
        [AffineCandidateState, AffineTrainingBatch],
        tuple[torch.Tensor, dict[str, torch.Tensor]],
    ],
    *,
    texel_count: int,
    batch_size: int,
    cube_sample_count: int,
    color_partition: ColorQuantilePartition | None = None,
    color_batch_size: int = 0,
) -> AffineTrainingStep:
    """Perform exactly one optimizer update for one paired candidate."""

    batch = draw_training_batch(
        state,
        texel_count=texel_count,
        batch_size=batch_size,
        cube_sample_count=cube_sample_count,
        color_partition=color_partition,
        color_batch_size=color_batch_size,
    )
    state.latent_optimizer.zero_grad(set_to_none=True)
    state.affine_optimizer.zero_grad(set_to_none=True)
    total, terms = objective(state, batch)
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("non-finite affine training loss")
    total.backward()
    state.latent_optimizer.step()
    state.affine_optimizer.step()
    with torch.no_grad():
        state.latent.clamp_(0.0, 1.0)
    state.optimizer_updates += 1
    if state.optimizer_updates <= 2_000:
        state.phase = "warmup"
    elif state.optimizer_updates <= 35_000:
        state.phase = "joint"
    else:
        state.phase = "polish"
    return AffineTrainingStep(
        batch=batch,
        loss=float(total.detach().cpu()),
        terms={name: float(value.detach().cpu()) for name, value in terms.items()},
    )


def time_candidate_steps(
    state: AffineCandidateState,
    objective: Callable[
        [AffineCandidateState, AffineTrainingBatch],
        tuple[torch.Tensor, dict[str, torch.Tensor]],
    ],
    *,
    texel_count: int,
    batch_size: int,
    cube_sample_count: int,
    warmup_steps: int,
    measured_steps: int,
    color_partition: ColorQuantilePartition | None = None,
    color_batch_size: int = 0,
) -> CandidateTiming:
    """Run fixed warmup/measured updates and report synchronized step timing."""

    if warmup_steps < 0 or measured_steps <= 0:
        raise ValueError("timing requires non-negative warmup and positive measured steps")
    is_cuda = state.latent.device.type == "cuda"
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(state.latent.device)
    for _ in range(warmup_steps):
        train_candidate_step(
            state,
            objective,
            texel_count=texel_count,
            batch_size=batch_size,
            cube_sample_count=cube_sample_count,
            color_partition=color_partition,
            color_batch_size=color_batch_size,
        )
    if is_cuda:
        torch.cuda.synchronize(state.latent.device)

    step_times: list[float] = []
    term_sums: dict[str, float] = {}
    for _ in range(measured_steps):
        if is_cuda:
            torch.cuda.synchronize(state.latent.device)
        started = time.perf_counter()
        result = train_candidate_step(
            state,
            objective,
            texel_count=texel_count,
            batch_size=batch_size,
            cube_sample_count=cube_sample_count,
            color_partition=color_partition,
            color_batch_size=color_batch_size,
        )
        if is_cuda:
            torch.cuda.synchronize(state.latent.device)
        step_times.append((time.perf_counter() - started) * 1000.0)
        for name, value in result.terms.items():
            term_sums[name] = term_sums.get(name, 0.0) + value
    timing_tensor = torch.tensor(step_times, dtype=torch.float64)
    median = float(torch.median(timing_tensor))
    p95 = float(torch.quantile(timing_tensor, 0.95))
    return CandidateTiming(
        warmup_steps=warmup_steps,
        measured_steps=measured_steps,
        step_times_ms=tuple(step_times),
        median_step_ms=median,
        p95_step_ms=p95,
        samples_per_second=batch_size / (median / 1000.0),
        peak_allocated_bytes=(
            int(torch.cuda.max_memory_allocated(state.latent.device)) if is_cuda else 0
        ),
        peak_reserved_bytes=(
            int(torch.cuda.max_memory_reserved(state.latent.device)) if is_cuda else 0
        ),
        mean_loss_terms={
            name: value / measured_steps for name, value in term_sums.items()
        },
    )


def checkpoint_candidate(state: AffineCandidateState) -> dict[str, object]:
    """Capture all mutable state required for exact continuation."""

    if (state.color_rng is None) != (state.color_partition_hash is None):
        raise ValueError("color RNG and partition hash must be configured together")
    if state.color_group_hash is not None and state.color_rng is None:
        raise ValueError("color group hash requires color training state")
    checkpoint = {
        "schema_version": (
            3
            if state.color_group_hash is not None
            else 2 if state.color_rng is not None else 1
        ),
        "candidate_id": state.candidate_id,
        "objective_id": state.objective_id,
        "parent_p0_hash": state.parent_p0_hash,
        "config_hash": state.config_hash,
        "input_hash": state.input_hash,
        "optimizer_updates": state.optimizer_updates,
        "phase": state.phase,
        "latent": state.latent.detach().clone(),
        "safe_affine_raw_parameters": state.decoder.state_dict(),
        "latent_optimizer": state.latent_optimizer.state_dict(),
        "affine_optimizer": state.affine_optimizer.state_dict(),
        "core_rng_state": state.core_rng.get_state(),
        "cube_rng_state": (
            state.cube_rng.get_state() if state.cube_rng is not None else None
        ),
        "best_metadata": state.best_metadata,
    }
    if state.color_rng is not None:
        checkpoint["color_rng_state"] = state.color_rng.get_state()
        checkpoint["color_partition_hash"] = state.color_partition_hash
    if state.color_group_hash is not None:
        checkpoint["color_group_hash"] = state.color_group_hash
    if state.continuation_from_checkpoint_hash is not None:
        checkpoint["continuation_from_checkpoint_hash"] = (
            state.continuation_from_checkpoint_hash
        )
        checkpoint["continuation_from_step"] = state.continuation_from_step
    checkpoint["checkpoint_hash"] = _checkpoint_hash(checkpoint)
    return copy.deepcopy(checkpoint)


def begin_candidate_continuation(
    state: AffineCandidateState,
    *,
    source_checkpoint: dict[str, object],
    continuation_config_hash: str,
) -> None:
    """Rebind an exact checkpoint state to a separately configured continuation."""

    source_hash = source_checkpoint.get("checkpoint_hash")
    if source_hash != _checkpoint_hash(source_checkpoint):
        raise ValueError("source checkpoint_hash mismatch")
    if checkpoint_candidate(state).get("checkpoint_hash") != source_hash:
        raise ValueError("continuation state does not exactly match source checkpoint")
    if not continuation_config_hash:
        raise ValueError("continuation config hash is required")
    state.continuation_from_checkpoint_hash = str(source_hash)
    state.continuation_from_step = int(source_checkpoint["optimizer_updates"])
    state.config_hash = continuation_config_hash


def _update_checkpoint_digest(digest: "hashlib._Hash", value: object) -> None:
    if isinstance(value, torch.Tensor):
        host = value.detach().cpu().contiguous()
        digest.update(b"tensor")
        digest.update(str(host.dtype).encode("ascii"))
        digest.update(str(tuple(host.shape)).encode("ascii"))
        digest.update(host.numpy().tobytes(order="C"))
    elif isinstance(value, dict):
        digest.update(b"dict")
        for key in sorted(value, key=lambda item: repr(item)):
            _update_checkpoint_digest(digest, key)
            _update_checkpoint_digest(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode("ascii"))
        for item in value:
            _update_checkpoint_digest(digest, item)
    else:
        digest.update(type(value).__name__.encode("ascii"))
        digest.update(repr(value).encode("utf-8"))


def _checkpoint_hash(checkpoint: dict[str, object]) -> str:
    digest = hashlib.sha256()
    content = {key: value for key, value in checkpoint.items() if key != "checkpoint_hash"}
    _update_checkpoint_digest(digest, content)
    return digest.hexdigest()


def candidate_manifest(state: AffineCandidateState) -> dict[str, object]:
    """Describe one step-qualified candidate without duplicating mutable state."""

    manifest = {
        "schema_version": 1,
        "candidate_id": state.candidate_id,
        "objective_id": state.objective_id,
        "parent_p0_hash": state.parent_p0_hash,
        "config_hash": state.config_hash,
        "input_hash": state.input_hash,
        "optimizer_updates": state.optimizer_updates,
        "phase": state.phase,
        "learned_linear": state.optimizer_updates > 0,
    }
    if state.continuation_from_checkpoint_hash is not None:
        manifest["continuation"] = {
            "source_checkpoint_hash": state.continuation_from_checkpoint_hash,
            "source_step": state.continuation_from_step,
        }
    if state.color_rng is not None:
        manifest["color_partition_hash"] = state.color_partition_hash
    if state.color_group_hash is not None:
        manifest["color_group_hash"] = state.color_group_hash
    return manifest


def write_candidate_checkpoint(
    output_root: Path | str,
    state: AffineCandidateState,
    *,
    endpoint: bool,
    storage: CheckpointStorage | None = None,
) -> CheckpointWrite:
    """Atomically update rolling state or create one immutable step endpoint."""

    root = Path(output_root)
    if endpoint:
        path = (
            root
            / state.candidate_id
            / "endpoints"
            / f"step-{state.optimizer_updates:06d}"
            / "checkpoint.pt"
        )
    else:
        path = root / state.candidate_id / "rolling" / "checkpoint.pt"
    checkpoint = checkpoint_candidate(state)
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    backend = storage if storage is not None else FilesystemCheckpointStorage()
    backend.write(path, buffer.getvalue(), immutable=endpoint)
    return CheckpointWrite(
        path=path,
        checkpoint_hash=str(checkpoint["checkpoint_hash"]),
        endpoint=endpoint,
    )


def run_candidate_training(
    state: AffineCandidateState,
    objective: Callable[
        [AffineCandidateState, AffineTrainingBatch],
        tuple[torch.Tensor, dict[str, torch.Tensor]],
    ],
    parent_p0: P0Calibration,
    *,
    output_root: Path | str,
    observation_plan: TrainingObservationPlan,
    texel_count: int,
    batch_size: int,
    cube_sample_count: int,
    color_partition: ColorQuantilePartition | None = None,
    color_batch_size: int = 0,
    storage: CheckpointStorage | None = None,
    on_step: Callable[[dict[str, object]], None] | None = None,
    on_trend: Callable[[dict[str, object]], None] | None = None,
) -> CandidateTrainingRun:
    """Train one candidate to a fixed endpoint with auditable observations."""

    if state.optimizer_updates > observation_plan.total_steps:
        raise ValueError("candidate is already beyond the requested endpoint")
    curve: list[dict[str, object]] = []
    trends: list[dict[str, object]] = []
    checkpoints: list[CheckpointWrite] = []
    trend_steps = set(observation_plan.trend_steps)
    checkpoint_steps = set(observation_plan.checkpoint_steps)
    while state.optimizer_updates < observation_plan.total_steps:
        result = train_candidate_step(
            state,
            objective,
            texel_count=texel_count,
            batch_size=batch_size,
            cube_sample_count=cube_sample_count,
            color_partition=color_partition,
            color_batch_size=color_batch_size,
        )
        point = {
            "step": state.optimizer_updates,
            "phase": state.phase,
            "loss": result.loss,
            "terms": result.terms,
        }
        curve.append(point)
        best_loss = float(state.best_metadata.get("best_loss", float("inf")))
        if result.loss < best_loss:
            state.best_metadata = {
                "best_loss": result.loss,
                "best_step": state.optimizer_updates,
            }
        if state.optimizer_updates in trend_steps:
            trend = candidate_parameter_trend(state, parent_p0)
            trends.append(trend)
            if on_trend is not None:
                on_trend(trend)
        if state.optimizer_updates in checkpoint_steps:
            write_candidate_checkpoint(
                output_root, state, endpoint=False, storage=storage
            )
            checkpoints.append(
                write_candidate_checkpoint(
                    output_root, state, endpoint=True, storage=storage
                )
            )
        if on_step is not None:
            on_step(point)
    return CandidateTrainingRun(
        manifest=candidate_manifest(state),
        curve=tuple(curve),
        parameter_trends=tuple(trends),
        checkpoints=tuple(checkpoints),
    )


def resume_candidate(
    checkpoint: dict[str, object],
    p0: P0Calibration,
    *,
    expected_parent_p0_hash: str,
    expected_config_hash: str,
    expected_input_hash: str,
    expected_color_partition_hash: str | None = None,
    expected_color_group_hash: str | None = None,
) -> AffineCandidateState:
    """Fail closed on lineage mismatch and restore an exact candidate state."""

    if checkpoint.get("checkpoint_hash") != _checkpoint_hash(checkpoint):
        raise ValueError("checkpoint_hash mismatch")
    schema_version = int(checkpoint.get("schema_version", 0))
    if schema_version not in (1, 2, 3):
        raise ValueError("unsupported checkpoint schema_version")
    expected = {
        "parent_p0_hash": expected_parent_p0_hash,
        "config_hash": expected_config_hash,
        "input_hash": expected_input_hash,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise ValueError(f"checkpoint {key} mismatch")
    if p0.safe.artifact_hash != expected_parent_p0_hash:
        raise ValueError("checkpoint parent_p0_hash does not match supplied P0")

    latent_value = checkpoint["latent"]
    if not isinstance(latent_value, torch.Tensor):
        raise ValueError("checkpoint latent is missing")
    latent = nn.Parameter(latent_value.clone())
    decoder = copy.deepcopy(p0.safe_decoder).to(
        device=latent.device, dtype=latent.dtype
    )
    decoder.load_state_dict(checkpoint["safe_affine_raw_parameters"])
    latent_optimizer_state = checkpoint["latent_optimizer"]
    affine_optimizer_state = checkpoint["affine_optimizer"]
    latent_lr = float(latent_optimizer_state["param_groups"][0]["lr"])
    affine_lr = float(affine_optimizer_state["param_groups"][0]["lr"])
    latent_optimizer = torch.optim.Adam((latent,), lr=latent_lr)
    affine_optimizer = torch.optim.Adam(decoder.parameters(), lr=affine_lr)
    latent_optimizer.load_state_dict(copy.deepcopy(latent_optimizer_state))
    affine_optimizer.load_state_dict(copy.deepcopy(affine_optimizer_state))
    core_rng = torch.Generator(device=latent.device)
    core_rng_state = checkpoint["core_rng_state"]
    if not isinstance(core_rng_state, torch.Tensor):
        raise ValueError("checkpoint core RNG state is missing")
    core_rng.set_state(core_rng_state.detach().cpu())
    cube_rng_state = checkpoint["cube_rng_state"]
    cube_rng = None
    if cube_rng_state is not None:
        if not isinstance(cube_rng_state, torch.Tensor):
            raise ValueError("checkpoint cube RNG state is invalid")
        cube_rng = torch.Generator(device=latent.device)
        cube_rng.set_state(cube_rng_state.detach().cpu())
    color_rng = None
    color_partition_hash = None
    color_group_hash = None
    if schema_version in (2, 3):
        color_rng_state = checkpoint.get("color_rng_state")
        color_partition_hash_value = checkpoint.get("color_partition_hash")
        if not isinstance(color_rng_state, torch.Tensor):
            raise ValueError("checkpoint color RNG state is missing")
        if not isinstance(color_partition_hash_value, str) or not color_partition_hash_value:
            raise ValueError("checkpoint color partition hash is missing")
        if expected_color_partition_hash != color_partition_hash_value:
            raise ValueError("checkpoint color partition hash mismatch")
        color_rng = torch.Generator(device=latent.device)
        color_rng.set_state(color_rng_state.detach().cpu())
        color_partition_hash = color_partition_hash_value
        if schema_version == 3:
            color_group_hash_value = checkpoint.get("color_group_hash")
            if not isinstance(color_group_hash_value, str) or not color_group_hash_value:
                raise ValueError("checkpoint color group hash is missing")
            if expected_color_group_hash != color_group_hash_value:
                raise ValueError("checkpoint color group hash mismatch")
            color_group_hash = color_group_hash_value
        elif expected_color_group_hash is not None:
            raise ValueError("checkpoint has no color group hash")
    elif expected_color_partition_hash is not None:
        raise ValueError("legacy checkpoint has no color partition")
    elif expected_color_group_hash is not None:
        raise ValueError("legacy checkpoint has no color group hash")
    return AffineCandidateState(
        candidate_id=str(checkpoint["candidate_id"]),
        objective_id=str(checkpoint["objective_id"]),
        parent_p0_hash=str(checkpoint["parent_p0_hash"]),
        config_hash=str(checkpoint["config_hash"]),
        input_hash=str(checkpoint["input_hash"]),
        latent=latent,
        decoder=decoder,
        latent_optimizer=latent_optimizer,
        affine_optimizer=affine_optimizer,
        core_rng=core_rng,
        cube_rng=cube_rng,
        color_rng=color_rng,
        color_partition_hash=color_partition_hash,
        color_group_hash=color_group_hash,
        optimizer_updates=int(checkpoint["optimizer_updates"]),
        phase=str(checkpoint["phase"]),
        best_metadata=copy.deepcopy(checkpoint["best_metadata"]),
        continuation_from_checkpoint_hash=(
            str(checkpoint["continuation_from_checkpoint_hash"])
            if checkpoint.get("continuation_from_checkpoint_hash") is not None
            else None
        ),
        continuation_from_step=(
            int(checkpoint["continuation_from_step"])
            if checkpoint.get("continuation_from_step") is not None
            else None
        ),
    )


def create_paired_candidates(
    p0: P0Calibration,
    *,
    core_seed: int,
    cube_seed: int,
    config_hash: str,
    input_hash: str,
    latent_learning_rate: float,
    affine_learning_rate: float,
) -> dict[str, AffineCandidateState]:
    """Deep-copy three independent trainable children from one frozen P0-safe."""

    specifications = {
        "L0": "material+helmet",
        "L1": "material+helmet+tv",
        "L2": "material+helmet+cube",
    }
    children: dict[str, AffineCandidateState] = {}
    dtype = p0.safe.weight.dtype
    device = p0.safe.weight.device
    initial_latent = p0.safe.latent_unorm8.to(device=device, dtype=dtype) / 255.0
    for candidate_id, objective_id in specifications.items():
        latent = nn.Parameter(initial_latent.clone())
        decoder = copy.deepcopy(p0.safe_decoder)
        latent_optimizer = torch.optim.Adam(
            (latent,), lr=latent_learning_rate
        )
        affine_optimizer = torch.optim.Adam(
            decoder.parameters(), lr=affine_learning_rate
        )
        core_rng = torch.Generator(device=device).manual_seed(core_seed)
        cube_rng = (
            torch.Generator(device=device).manual_seed(cube_seed)
            if candidate_id == "L2"
            else None
        )
        children[candidate_id] = AffineCandidateState(
            candidate_id=candidate_id,
            objective_id=objective_id,
            parent_p0_hash=p0.safe.artifact_hash,
            config_hash=config_hash,
            input_hash=input_hash,
            latent=latent,
            decoder=decoder,
            latent_optimizer=latent_optimizer,
            affine_optimizer=affine_optimizer,
            core_rng=core_rng,
            cube_rng=cube_rng,
        )
    return children


def create_color_candidates(
    p0: P0Calibration,
    *,
    core_seed: int,
    color_seed: int,
    color_partition_hash: str,
    config_hash: str,
    input_hash: str,
    latent_learning_rate: float,
    affine_learning_rate: float,
) -> dict[str, AffineCandidateState]:
    """Deep-copy paired C0/C1/C2 children with independent color RNG streams."""

    if not color_partition_hash:
        raise ValueError("color_partition_hash is required")
    specifications = {
        "C0": "material+helmet",
        "C1": "material+helmet+opponent",
        "C2": "material+helmet+opponent+pair",
    }
    children: dict[str, AffineCandidateState] = {}
    dtype = p0.safe.weight.dtype
    device = p0.safe.weight.device
    initial_latent = p0.safe.latent_unorm8.to(device=device, dtype=dtype) / 255.0
    for candidate_id, objective_id in specifications.items():
        latent = nn.Parameter(initial_latent.clone())
        decoder = copy.deepcopy(p0.safe_decoder)
        children[candidate_id] = AffineCandidateState(
            candidate_id=candidate_id,
            objective_id=objective_id,
            parent_p0_hash=p0.safe.artifact_hash,
            config_hash=config_hash,
            input_hash=input_hash,
            latent=latent,
            decoder=decoder,
            latent_optimizer=torch.optim.Adam((latent,), lr=latent_learning_rate),
            affine_optimizer=torch.optim.Adam(
                decoder.parameters(), lr=affine_learning_rate
            ),
            core_rng=torch.Generator(device=device).manual_seed(core_seed),
            cube_rng=None,
            color_rng=torch.Generator(device=device).manual_seed(color_seed),
            color_partition_hash=color_partition_hash,
        )
    return children


def create_color_risk_candidates(
    p0: P0Calibration,
    *,
    core_seed: int,
    color_seed: int,
    color_partition_hash: str,
    color_group_hash: str,
    config_hash: str,
    input_hash: str,
    latent_learning_rate: float,
    affine_learning_rate: float,
) -> dict[str, AffineCandidateState]:
    """Deep-copy the four independent v3 color-risk candidates."""

    if not color_partition_hash or not color_group_hash:
        raise ValueError("color partition and group hashes are required")
    specifications = {
        "G0-mean": "material+helmet+opponent-mean",
        "G1-yc-cvar25": "material+helmet+yc-cvar25",
        "G2-hue8-macro": "material+helmet+hue8-macro",
        "G3-cvar25-hue8": "material+helmet+yc-cvar25+hue8-macro",
    }
    children: dict[str, AffineCandidateState] = {}
    dtype = p0.safe.weight.dtype
    device = p0.safe.weight.device
    initial_latent = p0.safe.latent_unorm8.to(device=device, dtype=dtype) / 255.0
    for candidate_id, objective_id in specifications.items():
        latent = nn.Parameter(initial_latent.clone())
        decoder = copy.deepcopy(p0.safe_decoder)
        children[candidate_id] = AffineCandidateState(
            candidate_id=candidate_id,
            objective_id=objective_id,
            parent_p0_hash=p0.safe.artifact_hash,
            config_hash=config_hash,
            input_hash=input_hash,
            latent=latent,
            decoder=decoder,
            latent_optimizer=torch.optim.Adam((latent,), lr=latent_learning_rate),
            affine_optimizer=torch.optim.Adam(
                decoder.parameters(), lr=affine_learning_rate
            ),
            core_rng=torch.Generator(device=device).manual_seed(core_seed),
            cube_rng=None,
            color_rng=torch.Generator(device=device).manual_seed(color_seed),
            color_partition_hash=color_partition_hash,
            color_group_hash=color_group_hash,
        )
    return children
