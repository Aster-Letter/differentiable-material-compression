"""Strict-BaseColor affine codecs for one-sample RGBA8 material decoding.

This module is intentionally independent of the historical affine experiment
stack.  The only representation it exposes is an exact linear-UNORM8
BaseColor fiber plus one scalar residual degree of freedom.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Protocol, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from cg_frontier.compression.material import DecodedMaterial, reconstruct_normal


def half_up_unorm8(values: torch.Tensor, *, ste: bool) -> torch.Tensor:
    """Clamp and quantize normalized floats with floor(x*255+0.5)."""

    hard = torch.floor(torch.clamp(values, 0.0, 1.0) * 255.0 + 0.5) / 255.0
    return values + (hard - values).detach() if ste else hard


def _round_byte_ste(values: torch.Tensor) -> torch.Tensor:
    hard = torch.floor(values + 0.5)
    return values + (hard - values).detach()


def bilinear_sample_wrap(texture: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    """Sample top-down HWC data using the frozen LOD0/Wrap convention."""

    if texture.ndim != 3 or uv.ndim != 2 or uv.shape[-1] != 2:
        raise ValueError("texture must be HWC and uv must be Nx2")
    height, width = texture.shape[:2]
    if height <= 0 or width <= 0:
        raise ValueError("texture dimensions must be positive")
    x = uv[:, 0] * float(width) - 0.5
    y = uv[:, 1] * float(height) - 0.5
    x0_floor, y0_floor = torch.floor(x), torch.floor(y)
    wx, wy = x - x0_floor, y - y0_floor
    x0 = x0_floor.to(torch.int64).remainder(width)
    y0 = y0_floor.to(torch.int64).remainder(height)
    x1, y1 = (x0 + 1).remainder(width), (y0 + 1).remainder(height)
    return (
        texture[y0, x0] * ((1.0 - wx) * (1.0 - wy))[:, None]
        + texture[y0, x1] * (wx * (1.0 - wy))[:, None]
        + texture[y1, x0] * ((1.0 - wx) * wy)[:, None]
        + texture[y1, x1] * (wx * wy)[:, None]
    )


class ExactBaseColorCodec(Protocol):
    """Runtime-facing contract implemented by strict BaseColor codecs."""

    identifier: str

    def valid_bounds(self, colors_u8: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]: ...
    def encode_hard(self, colors_u8: torch.Tensor, residual_byte: torch.Tensor) -> torch.Tensor: ...
    def encode_fake(self, colors_u8: torch.Tensor, residual_byte: torch.Tensor) -> torch.Tensor: ...
    def decode_basecolor(self, latent_unorm: torch.Tensor) -> torch.Tensor: ...
    def decoder_parameters(self) -> tuple[torch.Tensor, torch.Tensor]: ...
    def specification(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class SeparatedCodec:
    """Canonical RGB24 plus one independent residual byte."""

    identifier: str = "separated_rgb24_residual8_v1"

    def _validate(self, colors_u8: torch.Tensor) -> None:
        if colors_u8.shape[-1] != 3:
            raise ValueError("BaseColor byte tensor must end in three channels")

    def valid_bounds(self, colors_u8: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate(colors_u8)
        shape = colors_u8.shape[:-1]
        return (
            torch.zeros(shape, dtype=torch.int64, device=colors_u8.device),
            torch.full(shape, 255, dtype=torch.int64, device=colors_u8.device),
        )

    def encode_hard(self, colors_u8: torch.Tensor, residual_byte: torch.Tensor) -> torch.Tensor:
        self._validate(colors_u8)
        residual = torch.floor(residual_byte.to(torch.float64) + 0.5).clamp(0, 255).to(torch.int64)
        return torch.cat((colors_u8.to(torch.int64), residual[..., None]), dim=-1)

    def encode_fake(self, colors_u8: torch.Tensor, residual_byte: torch.Tensor) -> torch.Tensor:
        residual = _round_byte_ste(residual_byte.clamp(0.0, 255.0))
        return torch.cat((colors_u8.to(residual.dtype), residual[..., None]), dim=-1) / 255.0

    def decode_basecolor(self, latent_unorm: torch.Tensor) -> torch.Tensor:
        return latent_unorm[..., :3]

    def decoder_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        weight = torch.zeros((3, 4), dtype=torch.float32)
        weight[:, :3] = torch.eye(3)
        return weight, torch.zeros(3, dtype=torch.float32)

    def specification(self) -> dict[str, Any]:
        return {"identifier": self.identifier, "kernel": [0, 0, 0, 1], "t0": 0}


def _ceil_div_tensor(numerator: torch.Tensor, denominator: int) -> torch.Tensor:
    return -torch.div(-numerator, denominator, rounding_mode="floor")


@dataclass(frozen=True)
class AffineLatticeCodec:
    """Integer shear whose null direction changes all four stored bytes."""

    kernel_rgb: tuple[int, int, int]
    t0: int
    identifier: str = "affine_mixed_lattice_v1"

    def __post_init__(self) -> None:
        if len(self.kernel_rgb) != 3 or any(value == 0 for value in self.kernel_rgb):
            raise ValueError("mixed lattice kernel must have three non-zero RGB entries")
        if any(abs(value) > 2 for value in self.kernel_rgb):
            raise ValueError("mixed lattice kernel entries must be in {-2,-1,1,2}")
        if not (0 <= self.t0 <= 255):
            raise ValueError("mixed lattice t0 must be a byte")

    @property
    def kernel(self) -> tuple[int, int, int, int]:
        return (*self.kernel_rgb, 1)

    def valid_bounds(self, colors_u8: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if colors_u8.shape[-1] != 3:
            raise ValueError("BaseColor byte tensor must end in three channels")
        colors = colors_u8.to(torch.int64)
        lower = torch.full(colors.shape[:-1], -self.t0, dtype=torch.int64, device=colors.device)
        upper = torch.full(colors.shape[:-1], 255 - self.t0, dtype=torch.int64, device=colors.device)
        for channel, coefficient in enumerate(self.kernel_rgb):
            color = colors[..., channel]
            if coefficient > 0:
                channel_lower = _ceil_div_tensor(-color, coefficient)
                channel_upper = torch.div(255 - color, coefficient, rounding_mode="floor")
            else:
                magnitude = -coefficient
                channel_lower = _ceil_div_tensor(color - 255, magnitude)
                channel_upper = torch.div(color, magnitude, rounding_mode="floor")
            lower = torch.maximum(lower, channel_lower)
            upper = torch.minimum(upper, channel_upper)
        return lower, upper

    def _project(self, colors_u8: torch.Tensor, residual_byte: torch.Tensor) -> torch.Tensor:
        lower, upper = self.valid_bounds(colors_u8)
        return torch.maximum(torch.minimum(residual_byte, upper.to(residual_byte.dtype)), lower.to(residual_byte.dtype))

    def encode_hard(self, colors_u8: torch.Tensor, residual_byte: torch.Tensor) -> torch.Tensor:
        residual = torch.floor(self._project(colors_u8, residual_byte.to(torch.float64)) + 0.5).to(torch.int64)
        kernel = torch.tensor(self.kernel_rgb, dtype=torch.int64, device=colors_u8.device)
        rgb = colors_u8.to(torch.int64) + residual[..., None] * kernel
        alpha = self.t0 + residual
        encoded = torch.cat((rgb, alpha[..., None]), dim=-1)
        if torch.any((encoded < 0) | (encoded > 255)):
            raise AssertionError("projected mixed lattice code escaped the byte cube")
        return encoded

    def encode_fake(self, colors_u8: torch.Tensor, residual_byte: torch.Tensor) -> torch.Tensor:
        residual = _round_byte_ste(self._project(colors_u8, residual_byte))
        kernel = torch.tensor(self.kernel_rgb, dtype=residual.dtype, device=residual.device)
        rgb = colors_u8.to(residual.dtype) + residual[..., None] * kernel
        alpha = float(self.t0) + residual
        return torch.cat((rgb, alpha[..., None]), dim=-1) / 255.0

    def decode_basecolor(self, latent_unorm: torch.Tensor) -> torch.Tensor:
        kernel = torch.tensor(self.kernel_rgb, dtype=latent_unorm.dtype, device=latent_unorm.device)
        return latent_unorm[..., :3] - kernel * (latent_unorm[..., 3:4] - float(self.t0) / 255.0)

    def decoder_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        weight = torch.zeros((3, 4), dtype=torch.float32)
        weight[:, :3] = torch.eye(3)
        weight[:, 3] = -torch.tensor(self.kernel_rgb, dtype=torch.float32)
        bias = torch.tensor(self.kernel_rgb, dtype=torch.float32) * (float(self.t0) / 255.0)
        return weight, bias

    def specification(self) -> dict[str, Any]:
        return {"identifier": self.identifier, "kernel": list(self.kernel), "t0": self.t0}


class ExactAffineMaterialDecoder(nn.Module):
    """One affine 4->7 mapping with optionally frozen exact BaseColor rows."""

    def __init__(self, *, codec: ExactBaseColorCodec, train_basecolor: bool) -> None:
        super().__init__()
        base_weight, base_bias = codec.decoder_parameters()
        self.base_weight = nn.Parameter(base_weight, requires_grad=train_basecolor)
        self.base_bias = nn.Parameter(base_bias, requires_grad=train_basecolor)
        self.auxiliary = nn.Linear(4, 4)
        nn.init.normal_(self.auxiliary.weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.auxiliary.bias)
        self.codec_identifier = codec.identifier
        self.train_basecolor = bool(train_basecolor)

    def raw_affine(self, latent_unorm: torch.Tensor) -> torch.Tensor:
        base = F.linear(latent_unorm, self.base_weight, self.base_bias)
        auxiliary = self.auxiliary(latent_unorm)
        return torch.cat((base, auxiliary), dim=-1)

    def forward(self, latent_unorm: torch.Tensor) -> DecodedMaterial:
        raw = self.raw_affine(latent_unorm)
        normal_xy = torch.tanh(raw[..., 3:5])
        return DecodedMaterial(
            base_color_linear=raw[..., :3],
            normal_xy=normal_xy,
            normal_xyz=reconstruct_normal(normal_xy),
            roughness=torch.sigmoid(raw[..., 5:6]),
            metallic=torch.sigmoid(raw[..., 6:7]),
        )

    def combined_parameters(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.cat((self.base_weight, self.auxiliary.weight), dim=0),
            torch.cat((self.base_bias, self.auxiliary.bias), dim=0),
        )


@dataclass(frozen=True)
class LatticeCapacity:
    kernel_rgb: tuple[int, int, int]
    t0: int
    min_states: int
    max_states: int
    mean_states: float
    p05_states: float
    weighted_mean_log2_states: float

    @property
    def capacity_sort_key(self) -> tuple[Any, ...]:
        return (
            -self.weighted_mean_log2_states,
            -self.min_states,
            sum(abs(value) for value in self.kernel_rgb),
            self.kernel_rgb,
            self.t0,
        )

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _numpy_bounds(colors: np.ndarray, kernel_rgb: Sequence[int], t0: int) -> tuple[np.ndarray, np.ndarray]:
    lower = np.full(colors.shape[0], -t0, dtype=np.int16)
    upper = np.full(colors.shape[0], 255 - t0, dtype=np.int16)
    for channel, coefficient in enumerate(kernel_rgb):
        color = colors[:, channel].astype(np.int32)
        if coefficient > 0:
            channel_lower = np.ceil((-color) / coefficient).astype(np.int16)
            channel_upper = np.floor((255 - color) / coefficient).astype(np.int16)
        else:
            magnitude = -coefficient
            channel_lower = np.ceil((color - 255) / magnitude).astype(np.int16)
            channel_upper = np.floor(color / magnitude).astype(np.int16)
        lower = np.maximum(lower, channel_lower)
        upper = np.minimum(upper, channel_upper)
    return lower, upper


def enumerate_lattice_capacity(
    colors_u8: np.ndarray,
    frequencies: np.ndarray,
    *,
    min_states: int,
    top_k: int,
) -> list[LatticeCapacity]:
    """Enumerate the frozen 64x256 shear family and return ranked capacity candidates."""

    colors = np.asarray(colors_u8, dtype=np.int16)
    counts = np.asarray(frequencies, dtype=np.int64)
    if colors.ndim != 2 or colors.shape[1] != 3:
        raise ValueError("colors_u8 must be Nx3")
    if counts.shape != (colors.shape[0],) or np.any(counts <= 0):
        raise ValueError("frequencies must be positive and match colors")
    if min_states <= 0 or top_k <= 0:
        raise ValueError("min_states and top_k must be positive")
    probability = counts.astype(np.float64) / float(counts.sum())
    candidates: list[LatticeCapacity] = []
    entries = (-2, -1, 1, 2)
    for a in entries:
        for b in entries:
            for c in entries:
                kernel = (a, b, c)
                for t0 in range(256):
                    lower, upper = _numpy_bounds(colors, kernel, t0)
                    states = upper.astype(np.int32) - lower.astype(np.int32) + 1
                    if int(states.min()) < min_states:
                        continue
                    candidates.append(
                        LatticeCapacity(
                            kernel_rgb=kernel,
                            t0=t0,
                            min_states=int(states.min()),
                            max_states=int(states.max()),
                            mean_states=float(np.sum(states * probability)),
                            p05_states=float(np.percentile(states, 5.0)),
                            weighted_mean_log2_states=float(np.sum(np.log2(states) * probability)),
                        )
                    )
    candidates.sort(key=lambda item: item.capacity_sort_key)
    return candidates[:top_k]


def codec_certificate(
    codec: ExactBaseColorCodec,
    colors_u8: torch.Tensor,
    residual_byte: torch.Tensor,
) -> dict[str, Any]:
    """Build the texel-center half of the strict BaseColor certificate."""

    encoded = codec.encode_hard(colors_u8, residual_byte)
    decoded = codec.decode_basecolor(encoded.to(torch.float32) / 255.0)
    target = colors_u8.to(torch.float32) / 255.0
    error = (decoded - target).abs()
    lower, upper = codec.valid_bounds(colors_u8)
    return {
        "schema_version": 1,
        "codec": codec.specification(),
        "texel_count": int(colors_u8.numel() // 3),
        "byte_exact": bool(torch.equal(torch.floor(decoded * 255.0 + 0.5).to(torch.int64), colors_u8.to(torch.int64))),
        "max_abs": float(error.max().detach().cpu()),
        "mean_abs": float(error.mean().detach().cpu()),
        "min_states": int((upper - lower + 1).min().detach().cpu()),
        "max_states": int((upper - lower + 1).max().detach().cpu()),
    }
