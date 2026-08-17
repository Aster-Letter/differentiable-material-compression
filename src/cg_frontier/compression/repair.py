"""Bounded SciFiHelmet R1/R2 repair primitives and gate calculations."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from cg_frontier.compression.material import MaterialDecoder


R1_COST = {"parameters": 103, "weight_bytes_float32": 412, "macs_per_pixel": 88}
R2_COST = {"parameters": 107, "weight_bytes_float32": 428, "macs_per_pixel": 92}
SPLIT_HEAD_COST = {"parameters": 91, "weight_bytes_float32": 364, "macs_per_pixel": 76}


class SplitHeadDecoder(nn.Module):
    """Affine BaseColor head plus a 4→8→4 auxiliary ReLU head."""

    def __init__(self) -> None:
        super().__init__()
        self.base_affine = nn.Linear(4, 3)
        self.aux_hidden = nn.Linear(4, 8)
        self.aux_output = nn.Linear(8, 4)
        self.kind = "split_affine_base_aux_mlp"

    def forward(self, latent_rgba: torch.Tensor) -> torch.Tensor:
        base = self.base_affine(latent_rgba)
        auxiliary = self.aux_output(torch.relu(self.aux_hidden(latent_rgba)))
        return torch.cat((base, auxiliary), dim=-1)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def macs_per_pixel(self) -> int:
        return 76


def initialize_split_head_from_tiny(base: MaterialDecoder) -> SplitHeadDecoder:
    """Copy hidden/auxiliary outputs from frozen 4→8→7; leave affine RGB zero."""

    if base.kind != "tiny_mlp":
        raise ValueError("split-head initialization requires the frozen tiny MLP")
    hidden, output = base.network[0], base.network[2]
    if not isinstance(hidden, nn.Linear) or not isinstance(output, nn.Linear):
        raise TypeError("unexpected tiny MLP module layout")
    first = next(base.parameters())
    target = SplitHeadDecoder().to(device=first.device, dtype=first.dtype)
    with torch.no_grad():
        target.base_affine.weight.zero_()
        target.base_affine.bias.zero_()
        target.aux_hidden.weight.copy_(hidden.weight)
        target.aux_hidden.bias.copy_(hidden.bias)
        target.aux_output.weight.copy_(output.weight[3:7])
        target.aux_output.bias.copy_(output.bias[3:7])
    return target


class MetallicResidualDecoder(nn.Module):
    """Add a zero-initialized bias-free latent-to-metallic raw residual."""

    def __init__(self, base: MaterialDecoder) -> None:
        super().__init__()
        if base.kind != "tiny_mlp":
            raise ValueError("R2 requires the frozen 4→8→7 tiny MLP")
        self.base = base
        self.metallic_residual = nn.Linear(4, 1, bias=False)
        nn.init.zeros_(self.metallic_residual.weight)
        self.kind = "tiny_mlp_metallic_residual"

    def forward(self, latent_rgba: torch.Tensor) -> torch.Tensor:
        raw = self.base(latent_rgba)
        residual = self.metallic_residual(latent_rgba)
        return torch.cat((raw[..., :6], raw[..., 6:7] + residual), dim=-1)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def macs_per_pixel(self) -> int:
        return 92


def deterministic_case_partitions(
    names: Sequence[str], *, seed: int = 20260803
) -> dict[str, list[str]]:
    """Split original train cases into optimizer/selection/repair-validation."""

    unique = sorted(set(names))
    if len(unique) != len(names) or len(unique) < 10:
        raise ValueError("case names must be unique and contain at least ten cases")
    ranked = sorted(
        unique,
        key=lambda name: hashlib.sha256(f"{seed}:{name}".encode("utf-8")).digest(),
    )
    selection_count = max(1, int(round(len(ranked) * 0.15)))
    validation_count = max(1, int(round(len(ranked) * 0.15)))
    optimizer_count = len(ranked) - selection_count - validation_count
    return {
        "optimizer": ranked[:optimizer_count],
        "selection": ranked[optimizer_count : optimizer_count + selection_count],
        "repair_validation": ranked[optimizer_count + selection_count :],
    }


def hard_example_indices(
    error: np.ndarray,
    eligible: np.ndarray,
    *,
    top_fraction: float,
) -> np.ndarray:
    """Return stable descending hard-example indices within eligible texels."""

    values = np.asarray(error, dtype=np.float64).reshape(-1)
    mask = np.asarray(eligible, dtype=bool).reshape(-1)
    if values.shape != mask.shape or not (0.0 < top_fraction <= 1.0) or not np.any(mask):
        raise ValueError("hard-pool inputs are invalid")
    indices = np.flatnonzero(mask)
    count = max(1, int(np.ceil(indices.size * top_fraction)))
    order = np.lexsort((indices, -values[indices]))
    return indices[order[:count]].astype(np.int64, copy=False)


def stratified_batch_indices(
    texel_count: int,
    optimizer_indices: torch.Tensor,
    base_pool: torch.Tensor,
    metallic_pool: torch.Tensor,
    *,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, dict[str, slice]]:
    """Sample exactly 50% uniform, 25% BaseColor, and 25% metallic texels."""

    if texel_count <= 0 or batch_size < 4 or batch_size % 4 != 0:
        raise ValueError("batch size must be a positive multiple of four")
    if any(pool.numel() == 0 for pool in (optimizer_indices, base_pool, metallic_pool)):
        raise ValueError("stratified pools must be non-empty")
    uniform_count = batch_size // 2
    hard_count = batch_size // 4

    def draw(pool: torch.Tensor, count: int) -> torch.Tensor:
        positions = torch.randint(
            0, pool.numel(), (count,), generator=generator, device=pool.device
        )
        return pool[positions]

    uniform = draw(optimizer_indices, uniform_count)
    base = draw(base_pool, hard_count)
    metallic = draw(metallic_pool, hard_count)
    batch = torch.cat((uniform, base, metallic))
    slices = {
        "uniform": slice(0, uniform_count),
        "base": slice(uniform_count, uniform_count + hard_count),
        "metallic": slice(uniform_count + hard_count, batch_size),
    }
    return batch, slices


def top_fraction_mean(values: torch.Tensor, fraction: float = 0.05) -> torch.Tensor:
    """Average the largest deterministic ceil(fraction*N) scalar values."""

    flat = values.reshape(-1)
    if flat.numel() == 0 or not (0.0 < fraction <= 1.0):
        raise ValueError("top-tail input is invalid")
    count = max(1, int(np.ceil(flat.numel() * fraction)))
    return torch.topk(flat, count, sorted=False).values.mean()


def relative_reduction(baseline: float, candidate: float) -> float:
    """Return fractional error reduction with an exact-zero convention."""

    if baseline < 0.0 or candidate < 0.0:
        raise ValueError("gate errors must be non-negative")
    if baseline == 0.0:
        return 0.0 if candidate == 0.0 else float("-inf")
    return (baseline - candidate) / baseline


def evaluate_acceptance_gates(
    baseline: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, Any]:
    """Apply the frozen repair gates and classify an R2-eligible R1 result."""

    roi_reductions = {
        name: relative_reduction(
            float(baseline["rois"][name]["composite"]["p99_0"]),
            float(candidate["rois"][name]["composite"]["p99_0"]),
        )
        for name in ("D1_metallic_boundary_full", "D2_yellow_tube_uv", "D3_gray_panel_proxy")
    }
    yellow_dark = relative_reduction(
        float(baseline["rois"]["D2_yellow_tube_uv"]["predicted_luminance_below_reference_half_fraction"]),
        float(candidate["rois"]["D2_yellow_tube_uv"]["predicted_luminance_below_reference_half_fraction"]),
    )
    yellow_base_p99 = relative_reduction(
        float(baseline["rois"]["D2_yellow_tube_uv"]["base_color_max_channel"]["p99_0"]),
        float(candidate["rois"]["D2_yellow_tube_uv"]["base_color_max_channel"]["p99_0"]),
    )
    metallic_mean = relative_reduction(
        float(baseline["metallic_boundary"]["mean"]),
        float(candidate["metallic_boundary"]["mean"]),
    )
    metallic_fraction = relative_reduction(
        float(baseline["metallic_boundary"]["fraction_above_0_1"]),
        float(candidate["metallic_boundary"]["fraction_above_0_1"]),
    )
    render_hdr_ok = float(candidate["render_selection"]["hdr_mae"]) <= float(
        baseline["render_selection"]["hdr_mae"]
    ) * 1.02
    render_ssim_ok = float(candidate["render_selection"]["display_ssim"]) >= float(
        baseline["render_selection"]["display_ssim"]
    ) - 0.001
    normal_ok = float(candidate["global"]["normal_degrees"]["p95_0"]) <= float(
        baseline["global"]["normal_degrees"]["p95_0"]
    ) + 0.1
    roughness_ok = float(candidate["global"]["roughness"]["mean"]) <= float(
        baseline["global"]["roughness"]["mean"]
    ) * 1.05
    base_ok = yellow_dark >= 0.50 and yellow_base_p99 >= 0.25
    roi_ok = all(value >= 0.30 for value in roi_reductions.values())
    metallic_ok = metallic_mean >= 0.30 and metallic_fraction >= 0.30
    non_metal_ok = base_ok and roi_ok and render_hdr_ok and render_ssim_ok and normal_ok and roughness_ok
    return {
        "roi_composite_p99_reduction": roi_reductions,
        "yellow_dark_fraction_reduction": yellow_dark,
        "yellow_base_p99_reduction": yellow_base_p99,
        "metallic_boundary_mae_reduction": metallic_mean,
        "metallic_boundary_above_0_1_fraction_reduction": metallic_fraction,
        "checks": {
            "roi_each_at_least_30_percent": roi_ok,
            "yellow_basecolor": base_ok,
            "metallic_boundary": metallic_ok,
            "render_hdr": render_hdr_ok,
            "display_ssim": render_ssim_ok,
            "normal_p95": normal_ok,
            "roughness_mae": roughness_ok,
        },
        "offline_pass": non_metal_ok and metallic_ok,
        "r2_eligible_metallic_only_failure": non_metal_ok and not metallic_ok,
    }
