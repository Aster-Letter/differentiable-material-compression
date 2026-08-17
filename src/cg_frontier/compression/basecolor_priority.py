"""BaseColor-priority objectives and deployment-consistent postprocessing."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import statistics
from typing import Callable, Mapping, Sequence

import torch
from torch import nn


SCALAR_CHANNELS = (0, 1, 2, 5, 6)
NORMAL_Z_BACKWARD_MIN_RADICAND = 1.0e-6
BASECOLOR_PRIORITY_CHECKPOINT_SCHEMA = 3
BASECOLOR_PRIORITY_CHECKPOINT_TYPE = "c4_basecolor_priority_v3"
BASECOLOR_PRIORITY_IDENTITY_FIELDS = (
    "parent_hash",
    "input_hash",
    "config_hash",
    "objective_hash",
    "postprocess_hash",
    "rig_hash",
)


def _checkpoint_identity(identity: Mapping[str, str]) -> dict[str, str]:
    missing = [name for name in BASECOLOR_PRIORITY_IDENTITY_FIELDS if name not in identity]
    if missing:
        raise ValueError(f"BaseColor checkpoint identity is missing: {','.join(missing)}")
    return {name: str(identity[name]) for name in BASECOLOR_PRIORITY_IDENTITY_FIELDS}


def save_basecolor_priority_checkpoint(
    path: Path | str,
    *,
    step: int,
    candidate_id: str,
    objective_id: str,
    target_share: float,
    lambda_value: float,
    actual_shares: Mapping[str, float],
    latent: nn.Parameter,
    weight: nn.Parameter,
    bias: nn.Parameter,
    compander_parameters: nn.Parameter,
    latent_optimizer: torch.optim.Optimizer,
    affine_optimizer: torch.optim.Optimizer,
    compander_optimizer: torch.optim.Optimizer,
    core_rng: torch.Generator,
    identity: Mapping[str, str],
) -> str:
    """Write an immutable BaseColor-priority v3 checkpoint."""

    target = Path(path)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {target}")
    if (
        step < 0
        or not candidate_id
        or objective_id not in {
            "r0_control",
            "basecolor_priority",
            "basecolor_priority_compander",
            "basecolor_only_oracle",
        }
        or not 0.0 <= target_share <= 1.0
        or not math.isfinite(lambda_value)
        or lambda_value < 0.0
        or set(actual_shares) != {"latent", "affine"}
        or any(not math.isfinite(float(value)) for value in actual_shares.values())
        or compander_parameters.numel() != 2
    ):
        raise ValueError("invalid BaseColor checkpoint state")
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": BASECOLOR_PRIORITY_CHECKPOINT_SCHEMA,
            "checkpoint_type": BASECOLOR_PRIORITY_CHECKPOINT_TYPE,
            "step": int(step),
            "candidate_id": str(candidate_id),
            "objective_id": str(objective_id),
            "target_share": float(target_share),
            "lambda_value": float(lambda_value),
            "actual_shares": {name: float(value) for name, value in actual_shares.items()},
            **_checkpoint_identity(identity),
            "latent": latent.detach().cpu(),
            "weight": weight.detach().cpu(),
            "bias": bias.detach().cpu(),
            "compander_parameters": compander_parameters.detach().cpu(),
            "latent_optimizer": latent_optimizer.state_dict(),
            "affine_optimizer": affine_optimizer.state_dict(),
            "compander_optimizer": compander_optimizer.state_dict(),
            "core_rng_state": core_rng.get_state().cpu(),
        },
        target,
    )
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    return digest


def load_basecolor_priority_checkpoint(
    path: Path | str,
    *,
    expected_candidate_id: str,
    expected_objective_id: str,
    expected_target_share: float,
    expected_identity: Mapping[str, str],
) -> dict[str, object]:
    """Load v3 state and fail closed on candidate, target share, and lineage."""

    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != BASECOLOR_PRIORITY_CHECKPOINT_SCHEMA
        or payload.get("checkpoint_type") != BASECOLOR_PRIORITY_CHECKPOINT_TYPE
    ):
        raise ValueError("checkpoint is not C4 BaseColor-priority v3")
    if payload.get("candidate_id") != expected_candidate_id:
        raise ValueError("BaseColor checkpoint candidate mismatch")
    if payload.get("objective_id") != expected_objective_id:
        raise ValueError("BaseColor checkpoint objective mismatch")
    if float(payload.get("target_share", -1.0)) != float(expected_target_share):
        raise ValueError("BaseColor checkpoint target share mismatch")
    for name, expected in _checkpoint_identity(expected_identity).items():
        if payload.get(name) != expected:
            raise ValueError(f"BaseColor checkpoint {name} mismatch")
    required = (
        "latent",
        "weight",
        "bias",
        "compander_parameters",
        "latent_optimizer",
        "affine_optimizer",
        "compander_optimizer",
        "core_rng_state",
        "lambda_value",
        "actual_shares",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"BaseColor checkpoint is missing: {','.join(missing)}")
    return payload


@dataclass(frozen=True)
class ChromaCompanderOutput:
    rgb: torch.Tensor
    gain: torch.Tensor


@dataclass(frozen=True)
class PostprocessedAffineOutput:
    seven: torch.Tensor
    normal_xyz: torch.Tensor
    compander_gain: torch.Tensor | None


def postprocess_affine_output(
    raw_seven: torch.Tensor,
    *,
    compander_parameters: torch.Tensor | None,
    straight_through: bool,
) -> PostprocessedAffineOutput:
    """Create the only seven-channel surface consumed by objectives and rendering."""

    final = apply_material_safety(raw_seven, straight_through=straight_through)
    gain = None
    if compander_parameters is not None:
        if compander_parameters.shape != (2,):
            raise ValueError("compander parameters must contain G0 and G1")
        companded = apply_chroma_compander(
            final[..., :3],
            compander_parameters[0],
            compander_parameters[1],
            straight_through_saturate=straight_through,
        )
        final = torch.cat((companded.rgb, final[..., 3:]), dim=-1)
        gain = companded.gain
    xy = final[..., 3:5]
    radical = torch.clamp(
        1.0 - xy.square().sum(dim=-1, keepdim=True),
        min=0.0,
    )
    exact_z = torch.sqrt(radical)
    if straight_through:
        stable_z = torch.sqrt(
            torch.clamp(radical, min=NORMAL_Z_BACKWARD_MIN_RADICAND)
        )
        z = _straight_through(stable_z, exact_z)
    else:
        z = exact_z
    return PostprocessedAffineOutput(
        seven=final,
        normal_xyz=torch.cat((xy, z), dim=-1),
        compander_gain=gain,
    )


@dataclass(frozen=True)
class C4PostprocessConfig:
    scalar_saturate: bool = True
    normal_disk_projection: bool = True
    reconstruct_normal_z: bool = True
    compander: bool = False


@dataclass(frozen=True)
class GradientBudgetGroup:
    median_residual_over_base: float
    lambda_candidate: float
    achieved_share: float
    base_norms: tuple[float, ...]
    residual_norms: tuple[float, ...]
    ratios: tuple[float, ...]
    cosines: tuple[float, ...]


@dataclass(frozen=True)
class GradientBudgetAudit:
    batch_count: int
    target_share: float
    lambda_value: float
    limiting_group: str
    groups: Mapping[str, GradientBudgetGroup]


@dataclass(frozen=True)
class BaseColorObjectiveConfig:
    candidate_id: str
    target_share: float
    lambda_value: float

    def __post_init__(self) -> None:
        if not self.candidate_id or not math.isfinite(self.lambda_value):
            raise ValueError("objective identity and lambda must be finite")
        if self.candidate_id == "N0-control":
            if self.target_share != 0.0:
                raise ValueError("N0-control must use target_share=0")
        elif self.candidate_id.startswith(("BC80", "BC90")):
            expected = 0.8 if self.candidate_id.startswith("BC80") else 0.9
            if self.target_share != expected or self.lambda_value <= 0.0:
                raise ValueError("BaseColor candidate identity/share mismatch")
        elif self.candidate_id == "BC-only":
            if self.target_share != 1.0 or self.lambda_value != 1.0:
                raise ValueError("BaseColor-only oracle must use unit identity weight")
        else:
            raise ValueError("unsupported BaseColor-priority candidate")


def compose_basecolor_priority_objective(
    terms: Mapping[str, torch.Tensor],
    config: BaseColorObjectiveConfig,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compose N0 or calibrated BC objective with the frozen R0 residual weights."""

    expected = {
        "base_color_l1",
        "base_color_charbonnier",
        "render_linear",
        "render_log",
        "normal_cosine",
        "roughness_l1",
        "metallic_l1",
    }
    if set(terms) != expected:
        raise ValueError("objective terms do not match the C4 BaseColor contract")
    residual = (
        terms["render_linear"]
        + 0.25 * terms["render_log"]
        + 0.25 * terms["normal_cosine"]
        + 0.5 * terms["roughness_l1"]
        + 0.5 * terms["metallic_l1"]
    )
    if config.candidate_id == "N0-control":
        base = terms["base_color_l1"]
    elif config.candidate_id == "BC-only":
        base = terms["base_color_charbonnier"]
        zero_residual = torch.zeros_like(residual)
        return base, {
            "base_color": base,
            "residual": zero_residual,
            "total": base,
        }
    else:
        base = config.lambda_value * terms["base_color_charbonnier"]
    total = base + residual
    return total, {"base_color": base, "residual": residual, "total": total}


def calibrate_basecolor_gradient_budget(
    batches: Sequence[object],
    loss_builder: Callable[[object], tuple[torch.Tensor, torch.Tensor]],
    *,
    parameter_groups: Mapping[str, Sequence[nn.Parameter]],
    target_share: float,
    epsilon: float = 1.0e-12,
    share_tolerance: float = 0.01,
) -> GradientBudgetAudit:
    """Calibrate BaseColor weight from eight frozen batches without state updates."""

    if len(batches) != 8:
        raise ValueError("BaseColor gradient audit requires exactly 8 batches")
    if set(parameter_groups) != {"latent", "affine"}:
        raise ValueError("gradient audit requires latent and affine parameter groups")
    if not 0.0 < target_share < 1.0 or epsilon <= 0.0 or share_tolerance < 0.0:
        raise ValueError("invalid gradient-budget calibration settings")
    parameters: list[nn.Parameter] = []
    group_slices: dict[str, slice] = {}
    for name, values in parameter_groups.items():
        start = len(parameters)
        parameters.extend(values)
        if len(parameters) == start:
            raise ValueError(f"parameter group {name} is empty")
        group_slices[name] = slice(start, len(parameters))

    rows: dict[str, dict[str, list[float]]] = {
        name: {key: [] for key in ("base", "residual", "ratio", "cosine")}
        for name in parameter_groups
    }
    for batch_index, batch in enumerate(batches):
        base_loss, residual_loss = loss_builder(batch)
        if base_loss.ndim != 0 or residual_loss.ndim != 0:
            raise ValueError("gradient-audit losses must be scalar")
        base_gradients = torch.autograd.grad(
            base_loss, parameters, retain_graph=True, allow_unused=False
        )
        residual_gradients = torch.autograd.grad(
            residual_loss, parameters, allow_unused=False
        )
        for name, indices in group_slices.items():
            base_values = base_gradients[indices]
            residual_values = residual_gradients[indices]
            base_norm = torch.sqrt(sum(value.square().sum() for value in base_values))
            residual_norm = torch.sqrt(
                sum(value.square().sum() for value in residual_values)
            )
            dot = sum(
                (left * right).sum()
                for left, right in zip(base_values, residual_values, strict=True)
            )
            ratio = residual_norm / (base_norm + epsilon)
            cosine = dot / (base_norm * residual_norm + epsilon)
            values = (base_norm, residual_norm, ratio, cosine)
            if not all(math.isfinite(float(value.detach())) for value in values):
                base_finite = all(
                    bool(torch.isfinite(value).all()) for value in base_values
                )
                residual_finite = all(
                    bool(torch.isfinite(value).all()) for value in residual_values
                )
                raise FloatingPointError(
                    "non-finite gradient audit result: "
                    f"batch={batch_index} group={name} "
                    f"base_gradients_finite={base_finite} "
                    f"residual_gradients_finite={residual_finite} "
                    f"base_norm={float(base_norm.detach())} "
                    f"residual_norm={float(residual_norm.detach())} "
                    f"ratio={float(ratio.detach())} "
                    f"cosine={float(cosine.detach())}"
                )
            if float(base_norm.detach()) <= 0.0 or float(residual_norm.detach()) <= 0.0:
                raise ValueError("gradient audit requires non-zero gradients")
            rows[name]["base"].append(float(base_norm.detach()))
            rows[name]["residual"].append(float(residual_norm.detach()))
            rows[name]["ratio"].append(float(ratio.detach()))
            rows[name]["cosine"].append(float(cosine.detach()))

    multiplier = target_share / (1.0 - target_share)
    candidates = {
        name: multiplier * statistics.median(values["ratio"])
        for name, values in rows.items()
    }
    limiting_group = max(candidates, key=candidates.__getitem__)
    lambda_value = candidates[limiting_group]
    summaries: dict[str, GradientBudgetGroup] = {}
    for name, values in rows.items():
        median_ratio = statistics.median(values["ratio"])
        achieved_share = lambda_value / (lambda_value + median_ratio)
        summaries[name] = GradientBudgetGroup(
            median_residual_over_base=median_ratio,
            lambda_candidate=candidates[name],
            achieved_share=achieved_share,
            base_norms=tuple(values["base"]),
            residual_norms=tuple(values["residual"]),
            ratios=tuple(values["ratio"]),
            cosines=tuple(values["cosine"]),
        )
    limiting_share = summaries[limiting_group].achieved_share
    if abs(limiting_share - target_share) > share_tolerance:
        raise ValueError("limiting parameter group missed target gradient share")
    if any(value.achieved_share + 1.0e-12 < target_share for value in summaries.values()):
        raise ValueError("a parameter group is below target BaseColor gradient share")
    return GradientBudgetAudit(
        batch_count=8,
        target_share=float(target_share),
        lambda_value=float(lambda_value),
        limiting_group=limiting_group,
        groups=summaries,
    )


def decoder_instruction_audit(
    config: C4PostprocessConfig,
    *,
    budget: int = 80,
) -> dict[str, object]:
    """Return a conservative analytic scalar-instruction count, not GPU timing."""

    if budget <= 0:
        raise ValueError("decoder instruction budget must be positive")
    if not (
        config.scalar_saturate
        and config.normal_disk_projection
        and config.reconstruct_normal_z
    ):
        raise ValueError("C4 safety and Normal-Z reconstruction are mandatory")
    breakdown = {
        "affine_mac": 28,
        "scalar_saturate": 5,
        "normal_xy_disk_projection": 7,
        "normal_z_reconstruction": 6,
        "chroma_compander": 24 if config.compander else 0,
    }
    total = sum(breakdown.values())
    if total > budget:
        raise ValueError(
            f"decoder analytic cost {total} exceeds scalar instruction budget {budget}"
        )
    return {
        "breakdown": breakdown,
        "total_scalar_instruction_equivalents": total,
        "budget": int(budget),
        "within_budget": True,
        "measured_gpu_cost": False,
    }


def apply_chroma_compander(
    rgb: torch.Tensor,
    g0: torch.Tensor | float,
    g1: torch.Tensor | float,
    *,
    straight_through_saturate: bool = True,
) -> ChromaCompanderOutput:
    """Apply the bounded square-chroma compander without square roots or powers."""

    if rgb.shape[-1] != 3 or not rgb.is_floating_point():
        raise ValueError("compander input must be a floating RGB tensor")
    if not bool(torch.isfinite(rgb).all()):
        raise ValueError("compander input must be finite")
    g0_tensor = torch.as_tensor(g0, dtype=rgb.dtype, device=rgb.device)
    g1_tensor = torch.as_tensor(g1, dtype=rgb.dtype, device=rgb.device)
    if not bool(torch.isfinite(g0_tensor).all() and torch.isfinite(g1_tensor).all()):
        raise ValueError("compander parameters must be finite")
    y = rgb.mean(dim=-1, keepdim=True)
    difference = rgb - y
    squared_chroma = difference.square().sum(dim=-1, keepdim=True)
    gain = torch.clamp(g0_tensor + g1_tensor * squared_chroma, 0.5, 2.0)
    expanded = y + gain * difference
    saturated = expanded.clamp(0.0, 1.0)
    final = (
        _straight_through(expanded, saturated)
        if straight_through_saturate
        else saturated
    )
    return ChromaCompanderOutput(rgb=final, gain=gain)


def _straight_through(source: torch.Tensor, hard: torch.Tensor) -> torch.Tensor:
    return source + (hard - source).detach()


def apply_material_safety(
    raw_seven: torch.Tensor,
    *,
    straight_through: bool,
) -> torch.Tensor:
    """Apply scalar saturate and branchless unit-disk Normal XY projection."""

    if raw_seven.shape[-1] != 7 or not raw_seven.is_floating_point():
        raise ValueError("raw material must be a floating tensor ending in 7 channels")
    if not bool(torch.isfinite(raw_seven).all()):
        raise ValueError("raw material must be finite")
    final = raw_seven.clone()
    scalar_raw = raw_seven[..., list(SCALAR_CHANNELS)]
    scalar_hard = scalar_raw.clamp(0.0, 1.0)
    final[..., list(SCALAR_CHANNELS)] = (
        _straight_through(scalar_raw, scalar_hard)
        if straight_through
        else scalar_hard
    )
    normal_raw = raw_seven[..., 3:5]
    inverse_length = torch.rsqrt(
        torch.clamp(normal_raw.square().sum(dim=-1, keepdim=True), min=1.0)
    )
    normal_hard = normal_raw * inverse_length
    final[..., 3:5] = (
        _straight_through(normal_raw, normal_hard)
        if straight_through
        else normal_hard
    )
    return final


def basecolor_charbonnier(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    epsilon: float = 1.0e-3,
) -> torch.Tensor:
    """Return zero-based uniform linear-RGB Charbonnier over texels and channels."""

    if prediction.shape != target.shape or prediction.shape[-1] != 3:
        raise ValueError("BaseColor tensors must have equal shape ending in RGB")
    if epsilon <= 0.0:
        raise ValueError("Charbonnier epsilon must be positive")
    difference = prediction - target
    return (torch.sqrt(difference.square() + epsilon * epsilon) - epsilon).mean()
