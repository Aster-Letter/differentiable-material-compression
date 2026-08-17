"""Frozen sampling and optimization protocol for deployment-parity candidates."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import torch


FIXED_BATCH_MIX = {
    "visible_screen_subpixel": 0.40,
    "uniform_atlas_subpixel": 0.20,
    "D2_D3_dark_hard_cell": 0.15,
    "material_boundary_halo_cell": 0.15,
    "texel_center_reference_anchor": 0.10,
}

FIXED_LOSS = {
    "subpixel_reference_material": 0.5,
    "postprocess_filter_commutativity": 0.5,
    "luminance_underprediction_top_tail": 0.5,
    "metallic_boundary_halo": 0.25,
    "texel_center_reference_anchor": 0.5,
    "dark_envelope": 0.5,
    "activation_region": 0.5,
    "activation_region_margin": 2.0 / 255.0,
}


@dataclass(frozen=True)
class DeploymentParityBatch:
    uv: torch.Tensor
    slices: Mapping[str, slice]
    screen_positions: torch.Tensor


def _draw_rows(pool: torch.Tensor, count: int, generator: torch.Generator) -> torch.Tensor:
    if pool.shape[0] == 0:
        raise ValueError("deployment-parity sampling pool is empty")
    positions = torch.randint(
        0,
        pool.shape[0],
        (count,),
        generator=generator,
        device=pool.device,
    )
    return pool[positions]


def _cell_uv(
    ids: torch.Tensor,
    *,
    height: int,
    width: int,
    generator: torch.Generator,
    centered: bool,
) -> torch.Tensor:
    if centered:
        offsets = torch.full((ids.numel(), 2), 0.5, device=ids.device)
    else:
        offsets = torch.rand((ids.numel(), 2), generator=generator, device=ids.device)
    x = ids.remainder(width).to(torch.float32) + offsets[:, 0]
    y = torch.div(ids, width, rounding_mode="floor").to(torch.float32) + offsets[:, 1]
    return torch.stack((x / float(width), y / float(height)), dim=-1)


def deployment_parity_batch(
    *,
    screen_uv: torch.Tensor,
    uniform_pool: torch.Tensor,
    dark_pool: torch.Tensor,
    boundary_pool: torch.Tensor,
    batch_size: int,
    height: int,
    width: int,
    generator: torch.Generator,
) -> DeploymentParityBatch:
    """Draw the immutable 40/20/15/15/10 deployment-parity mixture."""

    if batch_size <= 0 or batch_size % 20:
        raise ValueError("deployment-parity batch size must be a positive multiple of 20")
    if screen_uv.ndim != 2 or screen_uv.shape[-1] != 2:
        raise ValueError("visible screen-space pool must have shape Nx2")
    if height <= 0 or width <= 0:
        raise ValueError("atlas dimensions must be positive")
    unit = batch_size // 20
    counts = (8 * unit, 4 * unit, 3 * unit, 3 * unit, 2 * unit)
    screen_positions = torch.randint(
        0,
        screen_uv.shape[0],
        (counts[0],),
        generator=generator,
        device=screen_uv.device,
    )
    screen = screen_uv[screen_positions]
    uniform_ids = _draw_rows(uniform_pool, counts[1] + counts[4], generator)
    dark_ids = _draw_rows(dark_pool, counts[2], generator)
    boundary_ids = _draw_rows(boundary_pool, counts[3], generator)
    chunks = (
        screen,
        _cell_uv(
            uniform_ids[: counts[1]],
            height=height,
            width=width,
            generator=generator,
            centered=False,
        ),
        _cell_uv(
            dark_ids,
            height=height,
            width=width,
            generator=generator,
            centered=False,
        ),
        _cell_uv(
            boundary_ids,
            height=height,
            width=width,
            generator=generator,
            centered=False,
        ),
        _cell_uv(
            uniform_ids[counts[1] :],
            height=height,
            width=width,
            generator=generator,
            centered=True,
        ),
    )
    names = (
        "visible_screen_subpixel",
        "uniform_atlas_subpixel",
        "dark_hard_cell",
        "material_boundary_halo_cell",
        "texel_center_reference_anchor",
    )
    offset = 0
    slices: dict[str, slice] = {}
    for name, count in zip(names, counts, strict=True):
        slices[name] = slice(offset, offset + count)
        offset += count
    return DeploymentParityBatch(torch.cat(chunks, dim=0), slices, screen_positions)


def _cosine(start: float, stop: float, fraction: float) -> float:
    blend = 0.5 * (1.0 - math.cos(math.pi * fraction))
    return start + (stop - start) * blend


def learning_rates_at_step(step: int) -> tuple[float, float]:
    """Return latent and decoder LR under the immutable 120k schedule."""

    if not (0 <= step <= 120_000):
        raise ValueError("training step must be within [0, 120000]")
    if step == 120_000:
        return 5.0e-4, 2.0e-5
    if step <= 5_000:
        fraction = step / 5_000.0
        return 2.0e-2 * fraction, 1.0e-3 * fraction
    if step <= 100_000:
        fraction = (step - 5_000) / 95_000.0
        return _cosine(2.0e-2, 2.0e-3, fraction), _cosine(1.0e-3, 1.0e-4, fraction)
    fraction = (step - 100_000) / 20_000.0
    return _cosine(2.0e-3, 5.0e-4, fraction), _cosine(1.0e-4, 2.0e-5, fraction)


def validate_fixed_protocol(config: Mapping[str, Any]) -> dict[str, Any]:
    """Reject configs that silently alter the approved candidate comparison."""

    if dict(config.get("batch_mix", {})) != FIXED_BATCH_MIX:
        raise ValueError("deployment-parity batch mix differs from the frozen protocol")
    if dict(config.get("loss", {})) != FIXED_LOSS:
        raise ValueError("deployment-parity loss weights differ from the frozen protocol")
    return {
        "seed": 20260804,
        "max_steps": 120_000,
        "max_minutes": 120,
        "preflight_steps": 1_000,
        "probe_interval": 1_000,
        "atlas_evaluation_interval": 5_000,
        "float_dtype": "float32",
        "quantization": "rgba8_ste",
        "batch_mix": dict(FIXED_BATCH_MIX),
        "loss": dict(FIXED_LOSS),
    }
