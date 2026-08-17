"""Monotone BaseColor continuation from the historical raw PCA artifact.

The candidate keeps the deployment contract at one RGBA8 sample followed by
one direct 4-to-7 affine.  Training is stochastic, but accepted audit states
are constrained by deterministic hard-rounded full-texture BaseColor metrics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

from cg_frontier.compression.exact_basecolor_experiment import (
    RenderCase,
    TexelTargets,
    display_transform,
    learning_rates,
    material_losses,
    render_pair_loss,
)
from cg_frontier.compression.material import DecodedMaterial, reconstruct_normal
from cg_frontier.render.pbr import PointLight


SCHEMA_VERSION = 1
CHECKPOINT_TYPE = "c4_monotone_basecolor_v1"
OPPONENT = torch.tensor(
    [
        [1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0), 0.0],
        [1.0 / math.sqrt(6.0), 1.0 / math.sqrt(6.0), -2.0 / math.sqrt(6.0)],
    ],
    dtype=torch.float32,
)


@dataclass(frozen=True)
class ConstraintFractions:
    rgb_mean: float
    rgb_tail: float
    opponent_mean: float
    opponent_macro: float
    opponent_relative_mean: float
    opponent_relative_macro: float

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not (0.0 < float(value) <= 1.0):
                raise ValueError(f"constraint fraction {name} must be in (0, 1]")


@dataclass(frozen=True)
class BaseColorMetrics:
    rgb_mean: float
    rgb_rmse: float
    rgb_tail: float
    opponent_mean: float
    opponent_macro: float
    opponent_relative_mean: float
    opponent_relative_macro: float
    generic_chroma_retention: float
    chromatic_texel_fraction_losing_25pct: float

    def constrained(self) -> dict[str, float]:
        return {
            "rgb_mean": self.rgb_mean,
            "rgb_tail": self.rgb_tail,
            "opponent_mean": self.opponent_mean,
            "opponent_macro": self.opponent_macro,
            "opponent_relative_mean": self.opponent_relative_mean,
            "opponent_relative_macro": self.opponent_relative_macro,
        }

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class ColorPartition:
    group_ids: torch.Tensor
    members: tuple[torch.Tensor, ...]
    group_count: int
    luminance_edges: tuple[float, ...]
    chroma_edges: tuple[float, ...]
    group_counts: tuple[int, ...]

    def to(self, device: torch.device | str) -> "ColorPartition":
        return ColorPartition(
            group_ids=self.group_ids.to(device),
            members=tuple(value.to(device) for value in self.members),
            group_count=self.group_count,
            luminance_edges=self.luminance_edges,
            chroma_edges=self.chroma_edges,
            group_counts=self.group_counts,
        )

    def specification(self) -> dict[str, Any]:
        return {
            "group_count": self.group_count,
            "luminance_edges": list(self.luminance_edges),
            "chroma_edges": list(self.chroma_edges),
            "group_counts": list(self.group_counts),
        }


def _strict_quantile_edges(values: torch.Tensor, bins: int) -> tuple[float, ...]:
    candidates = torch.quantile(
        values.to(torch.float64),
        torch.linspace(0.0, 1.0, bins + 1, dtype=torch.float64)[1:-1],
    )
    edges: list[float] = []
    for value in candidates.tolist():
        if not edges or value > edges[-1] + 1.0e-12:
            edges.append(float(value))
    return tuple(edges)


def build_color_partition(colors_q8: torch.Tensor, *, bins: int = 4) -> ColorPartition:
    if colors_q8.ndim != 2 or colors_q8.shape[1] != 3 or bins < 2:
        raise ValueError("colors_q8 must be Nx3 and bins must be at least two")
    colors = colors_q8.to(torch.float32) / 255.0
    luminance = colors.sum(dim=-1) / math.sqrt(3.0)
    chroma = torch.linalg.vector_norm(colors @ OPPONENT.T, dim=-1)
    luminance_edges = _strict_quantile_edges(luminance, bins)
    chroma_edges = _strict_quantile_edges(chroma, bins)
    y_bin = torch.bucketize(luminance, torch.tensor(luminance_edges, dtype=luminance.dtype))
    c_bin = torch.bucketize(chroma, torch.tensor(chroma_edges, dtype=chroma.dtype))
    raw = y_bin * (len(chroma_edges) + 1) + c_bin
    _, group_ids = torch.unique(raw, sorted=True, return_inverse=True)
    group_count = int(group_ids.max()) + 1
    counts = torch.bincount(group_ids, minlength=group_count)
    if torch.any(counts == 0):
        raise AssertionError("color partition contains an empty compact group")
    return ColorPartition(
        group_ids=group_ids.to(torch.int64),
        members=tuple(
            torch.nonzero(group_ids == group, as_tuple=False).reshape(-1).to(torch.int64)
            for group in range(group_count)
        ),
        group_count=group_count,
        luminance_edges=luminance_edges,
        chroma_edges=chroma_edges,
        group_counts=tuple(int(value) for value in counts.tolist()),
    )


class DirectAffineDecoder(nn.Module):
    """Historical direct 4-to-7 affine semantics used by raw PCA."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor) -> None:
        super().__init__()
        if weight.shape != (7, 4) or bias.shape != (7,):
            raise ValueError("direct affine decoder must be 7x4 plus seven biases")
        self.weight = nn.Parameter(weight.to(torch.float32).clone())
        self.bias = nn.Parameter(bias.to(torch.float32).clone())

    def raw_affine(self, latent_unorm: torch.Tensor) -> torch.Tensor:
        return F.linear(latent_unorm, self.weight, self.bias)

    def forward(self, latent_unorm: torch.Tensor) -> DecodedMaterial:
        raw = self.raw_affine(latent_unorm)
        normal_xy = raw[..., 3:5]
        return DecodedMaterial(
            base_color_linear=raw[..., :3],
            normal_xy=normal_xy,
            normal_xyz=reconstruct_normal(normal_xy),
            roughness=raw[..., 5:6],
            metallic=raw[..., 6:7],
        )

    def combined_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.weight, self.bias


class MonotoneBaseColorCandidate(nn.Module):
    """Trainable RGBA8 code and direct affine initialized byte-for-byte from PCA."""

    name = "C-monotone"
    strict = False

    def __init__(
        self,
        *,
        latent_u8: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor,
        colors_u8: torch.Tensor,
    ) -> None:
        super().__init__()
        if latent_u8.ndim != 3 or latent_u8.shape[-1] != 4:
            raise ValueError("raw PCA latent must be HxWx4")
        if colors_u8.shape != (latent_u8.shape[0] * latent_u8.shape[1], 3):
            raise ValueError("BaseColor target does not match raw PCA texture")
        self.height, self.width = int(latent_u8.shape[0]), int(latent_u8.shape[1])
        self.latent_byte = nn.Parameter(latent_u8.reshape(-1, 4).to(torch.float32))
        self.decoder = DirectAffineDecoder(weight, bias)
        self.register_buffer("colors_u8", colors_u8.to(torch.uint8).clone())

    @property
    def texel_count(self) -> int:
        return int(self.latent_byte.shape[0])

    def code_parameters(self) -> Iterable[nn.Parameter]:
        return (self.latent_byte,)

    def decoder_parameters(self) -> Iterable[nn.Parameter]:
        return self.decoder.parameters()

    def latent_for_ids(self, flat_ids: torch.Tensor, *, ste: bool) -> torch.Tensor:
        values = self.latent_byte[flat_ids].clamp(0.0, 255.0)
        hard = torch.floor(values + 0.5)
        quantized = values + (hard - values).detach() if ste else hard
        return quantized / 255.0

    def sample_uv(self, uv: torch.Tensor, *, height: int, width: int, ste: bool) -> torch.Tensor:
        if (height, width) != (self.height, self.width):
            raise ValueError("runtime dimensions differ from PCA artifact")
        x = uv[:, 0] * float(width) - 0.5
        y = uv[:, 1] * float(height) - 0.5
        x0f, y0f = torch.floor(x), torch.floor(y)
        wx, wy = x - x0f, y - y0f
        x0, y0 = x0f.to(torch.int64).remainder(width), y0f.to(torch.int64).remainder(height)
        x1, y1 = (x0 + 1).remainder(width), (y0 + 1).remainder(height)
        ids = torch.stack((y0 * width + x0, y0 * width + x1, y1 * width + x0, y1 * width + x1), dim=1)
        weights = torch.stack(((1 - wx) * (1 - wy), wx * (1 - wy), (1 - wx) * wy, wx * wy), dim=1)
        corners = self.latent_for_ids(ids.reshape(-1), ste=ste).reshape(ids.shape[0], 4, 4)
        return torch.sum(corners * weights[..., None], dim=1)

    @torch.no_grad()
    def project_codes_(self) -> None:
        self.latent_byte.clamp_(0.0, 255.0)

    @torch.no_grad()
    def hard_texture_bytes(self) -> torch.Tensor:
        return torch.floor(self.latent_byte.clamp(0.0, 255.0) + 0.5).to(torch.uint8).reshape(self.height, self.width, 4)


def load_raw_pca_candidate(
    artifact_root: Path,
    targets: TexelTargets,
    *,
    expected_artifact_hash: str,
    device: torch.device | str,
) -> tuple[MonotoneBaseColorCandidate, dict[str, Any]]:
    manifest_path = artifact_root / "manifest.json"
    texture_path = artifact_root / "latent_rgba8.png"
    decoder_path = artifact_root / "decoder.bin"
    manifest = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    if (
        manifest.get("artifact_id") != "p0-raw"
        or manifest.get("artifact_hash") != expected_artifact_hash
        or manifest.get("deployable") is not False
        or sha(texture_path) != manifest.get("latent_png_sha256")
        or sha(decoder_path) != manifest.get("decoder_sha256")
    ):
        raise ValueError("raw PCA artifact identity or payload hash mismatch")
    latent = torch.from_numpy(np.asarray(Image.open(texture_path).convert("RGBA"), dtype=np.uint8).copy())
    packed = np.frombuffer(decoder_path.read_bytes(), dtype="<f4").copy()
    if packed.shape != (35,) or not np.isfinite(packed).all():
        raise ValueError("raw PCA decoder payload is invalid")
    candidate = MonotoneBaseColorCandidate(
        latent_u8=latent,
        weight=torch.from_numpy(packed[:28].reshape(7, 4)),
        bias=torch.from_numpy(packed[28:]),
        colors_u8=targets.base_q8,
    ).to(device)
    return candidate, {
        "artifact_id": manifest["artifact_id"],
        "artifact_hash": manifest["artifact_hash"],
        "manifest_sha256": sha(manifest_path),
        "latent_png_sha256": sha(texture_path),
        "decoder_sha256": sha(decoder_path),
        "historical_deployable_flag": False,
    }


def sample_balanced_ids(
    partition: ColorPartition,
    *,
    sample_count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    if sample_count < partition.group_count:
        raise ValueError("balanced sample must contain every source color group")
    device = partition.group_ids.device
    per_group = math.ceil(sample_count / partition.group_count)
    pieces: list[torch.Tensor] = []
    for group in range(partition.group_count):
        members = partition.members[group]
        positions = torch.randint(0, members.numel(), (per_group,), generator=generator, device=device)
        pieces.append(members[positions])
    return torch.cat(pieces)[:sample_count]


def balanced_basecolor_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    group_ids: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    difference = prediction - target
    rgb = difference.abs().mean()
    per_texel_rgb = difference.abs().mean(dim=-1)
    tail_count = max(1, math.ceil(per_texel_rgb.numel() * 0.25))
    rgb_tail = torch.topk(per_texel_rgb, tail_count, sorted=False).values.mean()
    opponent_error = torch.mean(torch.abs(difference @ OPPONENT.to(difference).T), dim=-1)
    group_values = []
    for group in torch.unique(group_ids, sorted=True):
        group_values.append(opponent_error[group_ids == group].mean())
    opponent_macro = torch.stack(group_values).mean()
    opponent_mean = opponent_error.mean()
    opponent = OPPONENT.to(difference)
    target_opponent = target @ opponent.T
    prediction_opponent = prediction @ opponent.T
    source_chroma = torch.linalg.vector_norm(target_opponent, dim=-1)
    reliable = source_chroma >= 0.02
    relative = torch.linalg.vector_norm(prediction_opponent - target_opponent, dim=-1) / source_chroma.clamp_min(0.02)
    relative_mean = relative[reliable].mean()
    relative_groups = []
    for group in torch.unique(group_ids[reliable], sorted=True):
        selected = reliable & (group_ids == group)
        relative_groups.append(relative[selected].mean())
    relative_macro = torch.stack(relative_groups).mean()
    total = (
        rgb
        + 2.0 * rgb_tail
        + opponent_mean
        + opponent_macro
        + 0.01 * relative_mean
        + 0.01 * relative_macro
    )
    return total, {
        "rgb": rgb,
        "rgb_tail": rgb_tail,
        "opponent_mean": opponent_mean,
        "opponent_macro": opponent_macro,
        "opponent_relative_mean": relative_mean,
        "opponent_relative_macro": relative_macro,
    }


@torch.no_grad()
def basecolor_tail_ids(
    candidate: MonotoneBaseColorCandidate,
    targets: TexelTargets,
    *,
    tail_fraction: float,
    chunk: int = 262_144,
) -> torch.Tensor:
    """Return the current hard-rounded worst RGB texels deterministically."""

    if not (0.0 < tail_fraction <= 1.0):
        raise ValueError("tail_fraction must be in (0, 1]")
    errors: list[torch.Tensor] = []
    target = targets.base_q8.to(candidate.latent_byte.device).to(torch.float32) / 255.0
    for start in range(0, candidate.texel_count, chunk):
        stop = min(start + chunk, candidate.texel_count)
        ids = torch.arange(start, stop, device=candidate.latent_byte.device)
        prediction = candidate.decoder(candidate.latent_for_ids(ids, ste=False)).base_color_linear
        errors.append(torch.mean(torch.abs(prediction - target[ids]), dim=-1))
    values = torch.cat(errors)
    count = max(1, math.ceil(candidate.texel_count * tail_fraction))
    return torch.topk(values, count, sorted=True).indices


@torch.no_grad()
def refit_basecolor_l2_(
    candidate: MonotoneBaseColorCandidate,
    targets: TexelTargets,
    *,
    chunk: int = 262_144,
    ridge: float = 1.0e-10,
) -> dict[str, float]:
    """Solve the 15-parameter BaseColor affine subproblem on all hard texels."""

    device = candidate.latent_byte.device
    gram = torch.zeros((5, 5), dtype=torch.float64, device=device)
    cross = torch.zeros((5, 3), dtype=torch.float64, device=device)
    target = targets.base_q8.to(device).to(torch.float64) / 255.0
    for start in range(0, candidate.texel_count, chunk):
        stop = min(start + chunk, candidate.texel_count)
        ids = torch.arange(start, stop, device=device)
        latent = candidate.latent_for_ids(ids, ste=False).to(torch.float64)
        design = torch.cat((latent, torch.ones((latent.shape[0], 1), dtype=torch.float64, device=device)), dim=-1)
        gram.add_(design.T @ design)
        cross.add_(design.T @ target[ids])
    gram.diagonal().add_(float(ridge))
    solution = torch.linalg.solve(gram, cross)
    candidate.decoder.weight[:3].copy_(solution[:4].T.to(torch.float32))
    candidate.decoder.bias[:3].copy_(solution[4].to(torch.float32))
    residual = gram @ solution - cross
    return {
        "normal_equation_max_abs": float(residual.abs().max()),
        "weight_l2": float(torch.linalg.vector_norm(solution[:4])),
        "bias_l2": float(torch.linalg.vector_norm(solution[4])),
    }


@torch.no_grad()
def refit_basecolor_irls_(
    candidate: MonotoneBaseColorCandidate,
    targets: TexelTargets,
    *,
    iterations: int = 5,
    epsilon: float = 1.0e-3,
    chunk: int = 262_144,
    ridge: float = 1.0e-8,
) -> dict[str, Any]:
    """Deterministically refit BaseColor rows for a Charbonnier/L1 proxy."""

    if iterations <= 0 or epsilon <= 0.0:
        raise ValueError("IRLS iterations and epsilon must be positive")
    device = candidate.latent_byte.device
    target = targets.base_q8.to(device).to(torch.float64) / 255.0
    solution = torch.cat(
        (
            candidate.decoder.weight[:3].detach().to(torch.float64).T,
            candidate.decoder.bias[:3].detach().to(torch.float64)[None, :],
        ),
        dim=0,
    )
    residual_norms: list[float] = []
    for _ in range(iterations):
        grams = torch.zeros((3, 5, 5), dtype=torch.float64, device=device)
        crosses = torch.zeros((3, 5), dtype=torch.float64, device=device)
        for start in range(0, candidate.texel_count, chunk):
            stop = min(start + chunk, candidate.texel_count)
            ids = torch.arange(start, stop, device=device)
            latent = candidate.latent_for_ids(ids, ste=False).to(torch.float64)
            design = torch.cat((latent, torch.ones((latent.shape[0], 1), dtype=torch.float64, device=device)), dim=-1)
            batch_target = target[ids]
            residual = design @ solution - batch_target
            weights = torch.rsqrt(residual.square() + float(epsilon) ** 2)
            for channel in range(3):
                weighted = design * weights[:, channel : channel + 1]
                grams[channel].add_(design.T @ weighted)
                crosses[channel].add_(design.T @ (weights[:, channel] * batch_target[:, channel]))
        new_solution = torch.empty_like(solution)
        maximum = 0.0
        for channel in range(3):
            grams[channel].diagonal().add_(float(ridge))
            new_solution[:, channel] = torch.linalg.solve(grams[channel], crosses[channel])
            maximum = max(maximum, float((grams[channel] @ new_solution[:, channel] - crosses[channel]).abs().max()))
        solution = new_solution
        residual_norms.append(maximum)
    candidate.decoder.weight[:3].copy_(solution[:4].T.to(torch.float32))
    candidate.decoder.bias[:3].copy_(solution[4].to(torch.float32))
    return {
        "iterations": iterations,
        "epsilon": epsilon,
        "normal_equation_max_abs": residual_norms,
        "weight_l2": float(torch.linalg.vector_norm(solution[:4])),
        "bias_l2": float(torch.linalg.vector_norm(solution[4])),
    }


@torch.no_grad()
def evaluate_basecolor_constraints(
    candidate: MonotoneBaseColorCandidate,
    targets: TexelTargets,
    partition: ColorPartition,
    *,
    chunk: int = 262_144,
    tail_fraction: float = 0.05,
) -> BaseColorMetrics:
    if not (0.0 < tail_fraction <= 1.0):
        raise ValueError("tail_fraction must be in (0, 1]")
    target_device = targets.base_q8.to(candidate.latent_byte.device).to(torch.float32) / 255.0
    group_device = partition.group_ids.to(candidate.latent_byte.device)
    rgb_sum = rgb_sq_sum = opponent_sum = source_chroma_sum = prediction_chroma_sum = 0.0
    relative_sum = 0.0
    relative_count = 0
    chromatic_count = chroma_loss_count = 0
    per_texel_rgb: list[torch.Tensor] = []
    group_sums = torch.zeros(partition.group_count, dtype=torch.float64)
    group_counts = torch.zeros(partition.group_count, dtype=torch.int64)
    relative_group_sums = torch.zeros(partition.group_count, dtype=torch.float64)
    relative_group_counts = torch.zeros(partition.group_count, dtype=torch.int64)
    opponent = OPPONENT.to(candidate.latent_byte.device)
    for start in range(0, candidate.texel_count, chunk):
        stop = min(start + chunk, candidate.texel_count)
        ids = torch.arange(start, stop, device=candidate.latent_byte.device)
        prediction = candidate.decoder(candidate.latent_for_ids(ids, ste=False)).base_color_linear
        target = target_device[ids]
        difference = prediction - target
        absolute = difference.abs()
        rgb_sum += float(absolute.sum())
        rgb_sq_sum += float(difference.square().sum())
        per_texel_rgb.append(absolute.mean(dim=-1).cpu())
        opponent_error = torch.mean(torch.abs(difference @ opponent.T), dim=-1)
        opponent_sum += float(opponent_error.sum())
        groups = group_device[ids]
        group_sums += torch.bincount(groups.cpu(), weights=opponent_error.cpu().to(torch.float64), minlength=partition.group_count)
        group_counts += torch.bincount(groups.cpu(), minlength=partition.group_count)
        source_chroma = torch.linalg.vector_norm(target @ opponent.T, dim=-1)
        prediction_chroma = torch.linalg.vector_norm(prediction @ opponent.T, dim=-1)
        source_chroma_sum += float(source_chroma.sum())
        prediction_chroma_sum += float(prediction_chroma.sum())
        chromatic = source_chroma >= 0.02
        relative = torch.linalg.vector_norm(difference @ opponent.T, dim=-1) / source_chroma.clamp_min(0.02)
        relative_sum += float(relative[chromatic].sum())
        relative_count += int(chromatic.sum())
        relative_group_sums += torch.bincount(
            groups[chromatic].cpu(),
            weights=relative[chromatic].cpu().to(torch.float64),
            minlength=partition.group_count,
        )
        relative_group_counts += torch.bincount(
            groups[chromatic].cpu(), minlength=partition.group_count
        )
        chromatic_count += int(chromatic.sum())
        chroma_loss_count += int((chromatic & (prediction_chroma < 0.75 * source_chroma)).sum())
    per_texel = torch.cat(per_texel_rgb)
    tail_count = max(1, math.ceil(per_texel.numel() * tail_fraction))
    count = candidate.texel_count
    reliable_groups = relative_group_counts > 0
    return BaseColorMetrics(
        rgb_mean=rgb_sum / (count * 3),
        rgb_rmse=math.sqrt(rgb_sq_sum / (count * 3)),
        rgb_tail=float(torch.topk(per_texel, tail_count, sorted=False).values.mean()),
        opponent_mean=opponent_sum / count,
        opponent_macro=float(torch.mean(group_sums / group_counts.clamp_min(1))),
        opponent_relative_mean=relative_sum / max(relative_count, 1),
        opponent_relative_macro=float(
            torch.mean(relative_group_sums[reliable_groups] / relative_group_counts[reliable_groups])
        ),
        generic_chroma_retention=prediction_chroma_sum / max(source_chroma_sum, 1.0e-12),
        chromatic_texel_fraction_losing_25pct=chroma_loss_count / max(chromatic_count, 1),
    )


def constraint_targets(
    initial: BaseColorMetrics,
    fractions: ConstraintFractions,
    *,
    step: int,
    final_step: int,
) -> dict[str, float]:
    if not 0 <= step <= final_step or final_step <= 0:
        raise ValueError("invalid continuation step")
    progress = step / float(final_step)
    continuation = 0.5 * (1.0 - math.cos(math.pi * progress))
    initial_values = initial.constrained()
    fraction_values = asdict(fractions)
    return {
        name: initial_values[name]
        * ((1.0 - continuation) + continuation * float(fraction_values[name]))
        for name in initial_values
    }


def audit_acceptance(
    current: BaseColorMetrics,
    previous: BaseColorMetrics,
    targets: Mapping[str, float],
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> tuple[bool, dict[str, Any]]:
    checks: dict[str, Any] = {}
    for name, value in current.constrained().items():
        monotone_limit = previous.constrained()[name] * (1.0 + relative_tolerance) + absolute_tolerance
        target_limit = float(targets[name]) * (1.0 + relative_tolerance) + absolute_tolerance
        checks[name] = {
            "value": value,
            "monotone_limit": monotone_limit,
            "target_limit": target_limit,
            "monotone": value <= monotone_limit,
            "target": value <= target_limit,
        }
    return all(item["monotone"] and item["target"] for item in checks.values()), checks


def normalized_basecolor_merit(
    metrics: BaseColorMetrics,
    initial: BaseColorMetrics,
) -> float:
    current_values = metrics.constrained()
    initial_values = initial.constrained()
    ratios = [
        current_values[name] / max(initial_values[name], 1.0e-12)
        for name in initial_values
    ]
    return float(sum(ratios) / len(ratios))


def trust_region_acceptance(
    current: BaseColorMetrics,
    initial: BaseColorMetrics,
    historical_best: Mapping[str, float],
    *,
    target_merit: float,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> tuple[bool, dict[str, Any]]:
    current_values = current.constrained()
    checks: dict[str, Any] = {}
    for name, value in current_values.items():
        guard = float(historical_best[name]) * (1.0 + relative_tolerance) + absolute_tolerance
        checks[name] = {"value": value, "historical_best": float(historical_best[name]), "guard": guard, "passed": value <= guard}
    merit = normalized_basecolor_merit(current, initial)
    checks["merit"] = {"value": merit, "target": float(target_merit), "passed": merit <= target_merit}
    return all(item["passed"] for item in checks.values()), checks


def preset_curve_targets(
    initial: BaseColorMetrics,
    *,
    alpha: float,
    exponents: Mapping[str, float],
    floors: Mapping[str, float],
    starts: Mapping[str, float] | None = None,
    headrooms: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Evaluate a predeclared monotone BaseColor constraint homotopy.

    ``alpha`` is achieved constraint progress, not optimizer-step progress.
    The smoothstep has zero slope at both endpoints, which avoids an abrupt
    first audit while still converging exactly to the declared floors.
    """

    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError("preset-curve alpha must be in [0, 1]")
    initial_values = initial.constrained()
    starts = {name: 0.0 for name in initial_values} if starts is None else starts
    headrooms = {name: 0.0 for name in initial_values} if headrooms is None else headrooms
    if (
        set(exponents) != set(initial_values)
        or set(floors) != set(initial_values)
        or set(starts) != set(initial_values)
        or set(headrooms) != set(initial_values)
    ):
        raise ValueError("preset-curve exponents/floors/starts/headrooms must name every constrained metric exactly once")
    targets: dict[str, float] = {}
    for name, initial_value in initial_values.items():
        exponent = float(exponents[name])
        floor = float(floors[name])
        start = float(starts[name])
        headroom = float(headrooms[name])
        if (
            exponent <= 0.0
            or floor < 0.0
            or floor > initial_value
            or not 0.0 <= start < 1.0
            or headroom < 0.0
        ):
            raise ValueError(f"invalid preset-curve parameters for {name}")
        local_progress = min(1.0, max(0.0, (float(alpha) - start) / (1.0 - start)))
        eased = local_progress * local_progress * (3.0 - 2.0 * local_progress)
        ceiling = initial_value * (1.0 + headroom)
        targets[name] = floor + (ceiling - floor) * math.pow(1.0 - eased, exponent)
    return targets


def preset_curve_acceptance(
    current: BaseColorMetrics,
    targets: Mapping[str, float],
    *,
    relative_tolerance: float,
    absolute_tolerance: float,
) -> tuple[bool, dict[str, Any]]:
    """Require every hard-rounded metric to lie inside one curve cross-section."""

    if relative_tolerance < 0.0 or absolute_tolerance < 0.0:
        raise ValueError("curve tolerances must be non-negative")
    current_values = current.constrained()
    if set(targets) != set(current_values):
        raise ValueError("curve targets must name every constrained metric exactly once")
    checks: dict[str, Any] = {}
    for name, value in current_values.items():
        target = float(targets[name])
        limit = target * (1.0 + relative_tolerance) + absolute_tolerance
        checks[name] = {
            "value": value,
            "target": target,
            "limit": limit,
            "passed": value <= limit,
        }
    return all(item["passed"] for item in checks.values()), checks


def normalized_basecolor_composite(
    metrics: BaseColorMetrics,
    initial: BaseColorMetrics,
    weights: Mapping[str, float],
) -> float:
    """Return a normalized scalar BaseColor objective with value one at start."""

    current_values = metrics.constrained()
    initial_values = initial.constrained()
    if not weights or not set(weights).issubset(initial_values):
        raise ValueError("composite weights must select at least one constrained metric")
    numeric = {name: float(value) for name, value in weights.items()}
    if any(value < 0.0 for value in numeric.values()) or not math.isclose(
        sum(numeric.values()), 1.0, rel_tol=0.0, abs_tol=1.0e-9
    ):
        raise ValueError("composite weights must be non-negative and sum to one")
    return float(
        sum(
            numeric[name] * current_values[name] / max(initial_values[name], 1.0e-12)
            for name in numeric
        )
    )


def composite_curve_target(
    *,
    alpha: float,
    floor: float,
    exponent: float,
) -> float:
    """Evaluate the frozen smoothstep-power target for the scalar composite."""

    if not 0.0 <= float(alpha) <= 1.0 or not 0.0 <= float(floor) <= 1.0 or exponent <= 0.0:
        raise ValueError("invalid composite curve parameters")
    progress = float(alpha)
    eased = progress * progress * (3.0 - 2.0 * progress)
    return float(floor) + (1.0 - float(floor)) * math.pow(1.0 - eased, float(exponent))


def composite_curve_alpha_for_value(
    value: float,
    *,
    floor: float,
    exponent: float,
) -> float:
    """Invert ``composite_curve_target`` for achieved hard-metric progress."""

    if not 0.0 <= float(floor) < 1.0 or exponent <= 0.0 or not math.isfinite(float(value)):
        raise ValueError("invalid inverse composite curve parameters")
    if value >= 1.0:
        return 0.0
    if value <= floor:
        return 1.0
    remaining = math.pow((float(value) - floor) / (1.0 - floor), 1.0 / float(exponent))
    target_eased = 1.0 - remaining
    lower, upper = 0.0, 1.0
    for _ in range(64):
        midpoint = 0.5 * (lower + upper)
        eased = midpoint * midpoint * (3.0 - 2.0 * midpoint)
        if eased < target_eased:
            lower = midpoint
        else:
            upper = midpoint
    return 0.5 * (lower + upper)


def composite_curve_acceptance(
    current: BaseColorMetrics,
    initial: BaseColorMetrics,
    *,
    weights: Mapping[str, float],
    target: float,
    guard_multipliers: Mapping[str, float],
    relative_tolerance: float,
    absolute_tolerance: float,
) -> tuple[bool, dict[str, Any]]:
    """Accept scalar curve progress only when every frozen component guard holds."""

    current_values = current.constrained()
    initial_values = initial.constrained()
    if set(guard_multipliers) != set(initial_values):
        raise ValueError("guard multipliers must name every constrained metric exactly once")
    if relative_tolerance < 0.0 or absolute_tolerance < 0.0:
        raise ValueError("curve tolerances must be non-negative")
    merit = normalized_basecolor_composite(current, initial, weights)
    merit_limit = float(target) * (1.0 + relative_tolerance) + absolute_tolerance
    checks: dict[str, Any] = {
        "composite": {
            "value": merit,
            "target": float(target),
            "limit": merit_limit,
            "passed": merit <= merit_limit,
        }
    }
    for name, value in current_values.items():
        multiplier = float(guard_multipliers[name])
        if multiplier < 1.0:
            raise ValueError(f"guard multiplier must be at least one: {name}")
        guard = initial_values[name] * multiplier
        limit = guard * (1.0 + relative_tolerance) + absolute_tolerance
        checks[name] = {
            "value": value,
            "initial": initial_values[name],
            "guard_multiplier": multiplier,
            "guard": guard,
            "limit": limit,
            "passed": value <= limit,
        }
    return all(item["passed"] for item in checks.values()), checks


def train_stochastic_step(
    candidate: MonotoneBaseColorCandidate,
    targets: TexelTargets,
    partition: ColorPartition,
    tail_ids: torch.Tensor,
    cases: Sequence[RenderCase],
    lights: Sequence[PointLight],
    *,
    generator: torch.Generator,
    texel_batch_size: int,
    color_batch_size: int,
    screen_batch_size: int,
    minimum_roughness: float,
    display_exposure: float,
    code_optimizer: torch.optim.Optimizer,
    decoder_optimizer: torch.optim.Optimizer,
    step: int,
    stop: int,
    base_scale: float,
    learning_rate_scale: float = 1.0,
) -> dict[str, float | int]:
    device = candidate.latent_byte.device
    texel_ids = torch.randint(0, candidate.texel_count, (texel_batch_size,), generator=generator, device=device)
    balanced_count = (color_batch_size * 3) // 4
    balanced_ids = sample_balanced_ids(partition, sample_count=balanced_count, generator=generator)
    tail_positions = torch.randint(
        0,
        tail_ids.numel(),
        (color_batch_size - balanced_count,),
        generator=generator,
        device=device,
    )
    color_ids = torch.cat((balanced_ids, tail_ids[tail_positions]))
    camera_index = int(torch.randint(0, len(cases), (1,), generator=generator, device=device))
    light_index = int(torch.randint(0, len(lights), (1,), generator=generator, device=device))
    case = cases[camera_index]
    positions = torch.randint(0, case.valid_flat_indices.shape[0], (screen_batch_size,), generator=generator, device=device)
    screen_ids = case.valid_flat_indices[positions]

    decoded = candidate.decoder(candidate.latent_for_ids(texel_ids, ste=True))
    material = material_losses(decoded, targets, texel_ids)
    color_decoded = candidate.decoder(candidate.latent_for_ids(color_ids, ste=True)).base_color_linear
    color_target = targets.base_q8[color_ids].to(torch.float32) / 255.0
    color_loss, color_terms = balanced_basecolor_loss(color_decoded, color_target, partition.group_ids[color_ids])
    hdr, display, _ = render_pair_loss(
        candidate,
        case,
        lights[light_index],
        screen_ids,
        height=targets.height,
        width=targets.width,
        minimum_roughness=minimum_roughness,
        display_exposure=display_exposure,
        ste=True,
    )
    auxiliary = hdr + 0.25 * display + 0.5 * material["normal"] + 0.25 * material["roughness"] + 0.25 * material["metallic"]
    # Uniform RGB keeps the objective tied to the actual texel-frequency
    # measure.  The balanced/tail term adds coverage without replacing it.
    uniform_rgb = material["base_color"]
    total = auxiliary + float(base_scale) * (8.0 * uniform_rgb + color_loss)
    if not torch.isfinite(total):
        raise FloatingPointError(f"non-finite C-monotone objective at step {step}")
    code_lr, decoder_lr = learning_rates(step, stop=stop)
    code_lr *= learning_rate_scale
    decoder_lr *= learning_rate_scale
    for group in code_optimizer.param_groups:
        group["lr"] = code_lr
    for group in decoder_optimizer.param_groups:
        group["lr"] = decoder_lr
    code_optimizer.zero_grad(set_to_none=True)
    decoder_optimizer.zero_grad(set_to_none=True)
    total.backward()
    gradients = [parameter.grad for parameter in candidate.parameters() if parameter.grad is not None]
    if not gradients or any(not bool(torch.isfinite(value).all()) for value in gradients):
        raise FloatingPointError(f"invalid C-monotone gradients at step {step}")
    code_optimizer.step()
    decoder_optimizer.step()
    candidate.project_codes_()
    return {
        "step": step,
        "camera_index": camera_index,
        "light_index": light_index,
        "total": float(total.detach()),
        "auxiliary": float(auxiliary.detach()),
        "hdr": float(hdr.detach()),
        "display": float(display.detach()),
        "base_scale": float(base_scale),
        "color_uniform_rgb": float(uniform_rgb.detach()),
        "code_lr": code_lr,
        "decoder_lr": decoder_lr,
        **{f"color_{name}": float(value.detach()) for name, value in color_terms.items()},
        **{name: float(value.detach()) for name, value in material.items()},
    }


def clone_training_state(
    candidate: MonotoneBaseColorCandidate,
    code_optimizer: torch.optim.Optimizer,
    decoder_optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
) -> dict[str, Any]:
    def cpu_clone(value: Any) -> Any:
        if torch.is_tensor(value):
            return value.detach().cpu().clone()
        if isinstance(value, dict):
            return {key: cpu_clone(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cpu_clone(item) for item in value]
        if isinstance(value, tuple):
            return tuple(cpu_clone(item) for item in value)
        return copy.deepcopy(value)

    return {
        "candidate_state": cpu_clone(candidate.state_dict()),
        "code_optimizer": cpu_clone(code_optimizer.state_dict()),
        "decoder_optimizer": cpu_clone(decoder_optimizer.state_dict()),
        "generator_state": generator.get_state().cpu().clone(),
        "torch_rng_state": torch.get_rng_state().clone(),
        "cuda_rng_state": [value.cpu().clone() for value in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
    }


def restore_training_state(
    snapshot: Mapping[str, Any],
    candidate: MonotoneBaseColorCandidate,
    code_optimizer: torch.optim.Optimizer,
    decoder_optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
) -> None:
    candidate.load_state_dict(snapshot["candidate_state"])
    code_optimizer.load_state_dict(snapshot["code_optimizer"])
    decoder_optimizer.load_state_dict(snapshot["decoder_optimizer"])
    generator.set_state(snapshot["generator_state"])
    torch.set_rng_state(snapshot["torch_rng_state"])
    if torch.cuda.is_available() and snapshot["cuda_rng_state"]:
        torch.cuda.set_rng_state_all(snapshot["cuda_rng_state"])


@torch.no_grad()
def interpolate_candidate_state_(
    candidate: MonotoneBaseColorCandidate,
    start_state: Mapping[str, torch.Tensor],
    end_state: Mapping[str, torch.Tensor],
    fraction: float,
) -> None:
    """Project a block update by interpolating model state, leaving Adam moments intact."""

    if not 0.0 <= float(fraction) <= 1.0:
        raise ValueError("interpolation fraction must be in [0, 1]")
    current = candidate.state_dict()
    if set(current) != set(start_state) or set(current) != set(end_state):
        raise ValueError("candidate interpolation states have different keys")
    projected: dict[str, torch.Tensor] = {}
    for name, destination in current.items():
        start = start_state[name].to(device=destination.device, dtype=destination.dtype)
        end = end_state[name].to(device=destination.device, dtype=destination.dtype)
        if destination.is_floating_point() or destination.is_complex():
            projected[name] = torch.lerp(start, end, float(fraction))
        else:
            if not torch.equal(start, end):
                raise ValueError(f"cannot interpolate changing non-floating state: {name}")
            projected[name] = start
    candidate.load_state_dict(projected)


def checkpoint_payload(
    *,
    candidate: MonotoneBaseColorCandidate,
    step: int,
    code_optimizer: torch.optim.Optimizer,
    decoder_optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    config_hash: str,
    target_hash: str,
    artifact_hash: str,
    partition: ColorPartition,
    initial_metrics: BaseColorMetrics,
    accepted_metrics: BaseColorMetrics,
    controller: Mapping[str, Any],
) -> dict[str, Any]:
    payload = clone_training_state(candidate, code_optimizer, decoder_optimizer, generator)
    payload.update(
        {
            "schema_version": SCHEMA_VERSION,
            "checkpoint_type": CHECKPOINT_TYPE,
            "candidate": candidate.name,
            "step": int(step),
            "config_hash": config_hash,
            "target_hash": target_hash,
            "artifact_hash": artifact_hash,
            "partition": partition.specification(),
            "initial_metrics": initial_metrics.as_dict(),
            "accepted_metrics": accepted_metrics.as_dict(),
            "controller": copy.deepcopy(dict(controller)),
        }
    )
    return payload


def validate_checkpoint(
    payload: Mapping[str, Any],
    *,
    config_hash: str,
    target_hash: str,
    artifact_hash: str,
) -> None:
    expected = {
        "schema_version": SCHEMA_VERSION,
        "checkpoint_type": CHECKPOINT_TYPE,
        "candidate": MonotoneBaseColorCandidate.name,
        "config_hash": config_hash,
        "target_hash": target_hash,
        "artifact_hash": artifact_hash,
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise ValueError(f"C-monotone checkpoint {name} mismatch")


def export_candidate(candidate: MonotoneBaseColorCandidate, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    latent = candidate.hard_texture_bytes().cpu().numpy()
    texture_path = output_dir / "latent_rgba_unorm8.png"
    decoder_path = output_dir / "decoder_affine.npz"
    Image.fromarray(latent, mode="RGBA").save(texture_path)
    weight, bias = candidate.decoder.combined_parameters()
    np.savez(decoder_path, weight=weight.detach().cpu().numpy(), bias=bias.detach().cpu().numpy())
    reread = np.asarray(Image.open(texture_path).convert("RGBA"), dtype=np.uint8)
    if not np.array_equal(reread, latent):
        raise AssertionError("C-monotone PNG roundtrip changed bytes")
    files = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (texture_path, decoder_path)
    }
    return {
        "schema_version": 1,
        "candidate": candidate.name,
        "runtime": "one RGBA8 bilinear sample plus one direct 4-to-7 affine",
        "source_basecolor_required": False,
        "files": files,
    }
