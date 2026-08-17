"""Deployment-order latent filtering and activation-region coherence primitives."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Mapping

import torch
import torch.nn.functional as F
from torch import nn

from cg_frontier.compression.filter_aware import (
    bilinear_corners_top_down_wrap_torch,
    postprocess_raw_torch,
)
from cg_frontier.compression.material import DecodedMaterial
from cg_frontier.compression.render_loss import (
    fake_quantize_unorm8,
    hard_quantize_unorm8,
)


class DeploymentParityDecoder(nn.Module):
    """ReLU decoder used by fresh deployment-parity candidates."""

    def __init__(self, *, width: int = 8) -> None:
        super().__init__()
        if width not in (8, 12):
            raise ValueError("deployment-parity decoder width must be 8 or 12")
        self.hidden = nn.Linear(4, width)
        self.output = nn.Linear(width, 7)
        self.width = width
        self.module_identifier = f"deployment_parity_decoder_v1.relu_w{width}"

    def forward(self, latent_rgba: torch.Tensor) -> torch.Tensor:
        return self.output(F.relu(self.hidden(latent_rgba)))


@dataclass(frozen=True)
class DeploymentParitySample:
    corners: torch.Tensor
    weights: torch.Tensor
    sampled_latent: torch.Tensor
    runtime: DecodedMaterial
    decode_then_filter: DecodedMaterial
    postprocess_commutativity_l1: torch.Tensor


@dataclass(frozen=True)
class ActivationRegionResult:
    loss: torch.Tensor
    crossing_fraction: float
    normalized_margin: torch.Tensor


@dataclass(frozen=True)
class FreshInitialization:
    seed: int
    decoder_width: int
    latent_values: torch.Tensor
    decoder_state: Mapping[str, torch.Tensor]
    sha256: str


@dataclass(frozen=True)
class FreshCandidate:
    candidate: str
    initial_sha256: str
    latent: nn.Parameter
    decoder: DeploymentParityDecoder


def _weighted(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.sum(values * weights[..., None], dim=1)


def _quantize(values: torch.Tensor, mode: str) -> torch.Tensor:
    if mode in {"float", "prequantized"}:
        return values
    if mode == "hard":
        return hard_quantize_unorm8(values)
    if mode == "fake":
        return fake_quantize_unorm8(values)
    raise ValueError(f"unsupported quantization mode: {mode}")


def deployment_parity_sample(
    latent_texture: torch.Tensor,
    uv: torch.Tensor,
    decoder: DeploymentParityDecoder,
    *,
    quantization: str,
) -> DeploymentParitySample:
    """Quantize texels, bilinear sample once, decode, then postprocess once."""

    deployed = _quantize(latent_texture, quantization)
    corners, weights = bilinear_corners_top_down_wrap_torch(deployed, uv)
    sampled = _weighted(corners, weights)
    runtime = postprocess_raw_torch(decoder(sampled))
    corner_material = postprocess_raw_torch(decoder(corners.reshape(-1, 4)))
    count = corners.shape[0]

    def filtered(value: torch.Tensor) -> torch.Tensor:
        return _weighted(value.reshape(count, 4, value.shape[-1]), weights)

    decode_then_filter = DecodedMaterial(
        filtered(corner_material.base_color_linear),
        filtered(corner_material.normal_xy),
        filtered(corner_material.normal_xyz),
        filtered(corner_material.roughness),
        filtered(corner_material.metallic),
    )
    gap = (
        F.l1_loss(runtime.base_color_linear, decode_then_filter.base_color_linear)
        + F.l1_loss(runtime.normal_xyz, decode_then_filter.normal_xyz)
        + F.l1_loss(runtime.roughness, decode_then_filter.roughness)
        + F.l1_loss(runtime.metallic, decode_then_filter.metallic)
    )
    return DeploymentParitySample(
        corners=corners,
        weights=weights,
        sampled_latent=sampled,
        runtime=runtime,
        decode_then_filter=decode_then_filter,
        postprocess_commutativity_l1=gap,
    )


def activation_region_coherence(
    decoder: DeploymentParityDecoder,
    corners: torch.Tensor,
    *,
    margin: float = 2.0 / 255.0,
    epsilon: float = 1.0e-8,
) -> ActivationRegionResult:
    """Penalize 2×2 cells whose affine hidden units do not share one ReLU region."""

    if corners.ndim != 3 or corners.shape[1:] != (4, 4):
        raise ValueError("activation-region corners must have shape Nx4x4")
    if margin <= 0.0 or epsilon <= 0.0:
        raise ValueError("activation-region margin and epsilon must be positive")
    preactivation = F.linear(corners, decoder.hidden.weight, decoder.hidden.bias)
    weight_norm = torch.linalg.vector_norm(decoder.hidden.weight, dim=1).clamp_min(epsilon)
    normalized = preactivation / weight_norm
    minimum = normalized.amin(dim=1)
    maximum = normalized.amax(dim=1)
    normalized_margin = torch.maximum(minimum, -maximum)
    loss = F.relu(float(margin) - normalized_margin).mean()
    crossing = torch.logical_and(minimum < 0.0, maximum > 0.0)
    return ActivationRegionResult(
        loss=loss,
        crossing_fraction=float(crossing.to(torch.float32).mean().detach().cpu()),
        normalized_margin=normalized_margin,
    )


def dark_envelope_loss(
    runtime: DecodedMaterial,
    decode_then_filter: DecodedMaterial,
    *,
    tolerance: float = 1.0 / 255.0,
) -> torch.Tensor:
    """Penalize novel BaseColor darkness relative to the filtered corner material."""

    if tolerance < 0.0:
        raise ValueError("dark-envelope tolerance must be non-negative")
    luma_weights = runtime.base_color_linear.new_tensor((0.2126, 0.7152, 0.0722))
    runtime_luma = torch.sum(runtime.base_color_linear * luma_weights, dim=-1)
    filtered_luma = torch.sum(decode_then_filter.base_color_linear * luma_weights, dim=-1)
    luma_undercut = F.relu(filtered_luma - runtime_luma - float(tolerance))
    runtime_max = runtime.base_color_linear.amax(dim=-1)
    filtered_max = decode_then_filter.base_color_linear.amax(dim=-1)
    max_channel_undercut = F.relu(filtered_max - runtime_max - float(tolerance))
    return 0.5 * (luma_undercut.mean() + max_channel_undercut.mean())


def _initialization_hash(
    *,
    seed: int,
    decoder_width: int,
    latent_values: torch.Tensor,
    decoder_state: Mapping[str, torch.Tensor],
) -> str:
    digest = hashlib.sha256()
    digest.update(f"seed={seed};decoder_width={decoder_width}".encode("ascii"))
    tensors = {"latent_values": latent_values, **decoder_state}
    for name in sorted(tensors):
        tensor = tensors[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def make_fresh_initialization(
    *,
    height: int,
    width: int,
    decoder_width: int,
    seed: int,
) -> FreshInitialization:
    """Create one deterministic CPU state that candidates can clone independently."""

    if height <= 0 or width <= 0:
        raise ValueError("fresh latent dimensions must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    latent_values = torch.rand((height, width, 4), generator=generator)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 1)
        decoder = DeploymentParityDecoder(width=decoder_width)
    decoder_state = {
        name: value.detach().cpu().clone() for name, value in decoder.state_dict().items()
    }
    digest = _initialization_hash(
        seed=seed,
        decoder_width=decoder_width,
        latent_values=latent_values,
        decoder_state=decoder_state,
    )
    return FreshInitialization(
        seed=seed,
        decoder_width=decoder_width,
        latent_values=latent_values.detach().clone(),
        decoder_state=decoder_state,
        sha256=digest,
    )


def instantiate_fresh_candidate(
    initialization: FreshInitialization,
    *,
    candidate: str,
    device: torch.device | str = "cpu",
) -> FreshCandidate:
    """Clone an immutable shared state without sharing candidate storage."""

    allowed = {"dp_relu_fresh", "arc_relu_fresh", "arc12_diagnostic"}
    if candidate not in allowed:
        raise ValueError(f"unsupported fresh candidate: {candidate}")
    if candidate == "arc12_diagnostic" and initialization.decoder_width != 12:
        raise ValueError("arc12_diagnostic requires a width-12 initialization")
    if candidate != "arc12_diagnostic" and initialization.decoder_width != 8:
        raise ValueError("exact-budget fresh candidates require a width-8 initialization")
    decoder = DeploymentParityDecoder(width=initialization.decoder_width).to(device)
    decoder.load_state_dict(
        {name: value.detach().clone().to(device) for name, value in initialization.decoder_state.items()}
    )
    decoder.module_identifier = (
        f"deployment_parity_arc_v1.{candidate}.relu_w{initialization.decoder_width}"
    )
    latent = nn.Parameter(initialization.latent_values.detach().clone().to(device))
    return FreshCandidate(
        candidate=candidate,
        initial_sha256=initialization.sha256,
        latent=latent,
        decoder=decoder,
    )


def calculate_deployment_parity_cost(
    decoder: DeploymentParityDecoder,
) -> dict[str, object]:
    """Derive neural, activation, texture-sample, and resident costs."""

    parameters = sum(parameter.numel() for parameter in decoder.parameters())
    weight_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in decoder.parameters()
    )
    macs = 4 * decoder.width + decoder.width * 7
    return {
        "shape": f"4->{decoder.width}->7",
        "parameters": parameters,
        "weight_bytes_float32": weight_bytes,
        "macs_per_pixel": macs,
        "texture_samples": 1,
        "actual_resident_bytes": 2048 * 2048 * 4,
        "activation": {
            "kind": "relu",
            "hidden_units": decoder.width,
            "special_functions_per_pixel": 0,
        },
        "deployable_exact_budget": decoder.width == 8,
    }
