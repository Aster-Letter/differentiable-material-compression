"""Four-corner decode-then-material-filter primitives."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Mapping

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from cg_frontier.compression.filter_aware import (
    bilinear_corners_top_down_wrap_torch,
    postprocess_raw_torch,
)
from cg_frontier.compression.material import DecodedMaterial
from cg_frontier.compression.render_loss import (
    fake_quantize_unorm8,
    hard_quantize_unorm8,
    unorm8_encode_half_up,
)


class DecodeThenFilterDecoder(nn.Module):
    """Shared C4/C5 decoder evaluated independently at four texel corners."""

    def __init__(
        self,
        *,
        latent_channels: int = 4,
        width: int = 16,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        if latent_channels not in (4, 5):
            raise ValueError("DTF latent channels must be C4 or C5")
        if width not in (8, 16, 32):
            raise ValueError("DTF decoder width must be 8, 16, or 32")
        if activation not in ("relu", "silu"):
            raise ValueError("DTF activation must be relu or silu")
        self.hidden_in = nn.Linear(latent_channels, width)
        self.hidden_mid = nn.Linear(width, width)
        self.output = nn.Linear(width, 7)
        self.latent_channels = latent_channels
        self.width = width
        self.activation = activation
        self.module_identifier = (
            f"decode_then_filter_decoder_v1.c{latent_channels}_w{width}_{activation}"
        )

    def _activate(self, values: torch.Tensor) -> torch.Tensor:
        return F.relu(values) if self.activation == "relu" else F.silu(values)

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        hidden = self._activate(self.hidden_in(latent))
        hidden = self._activate(self.hidden_mid(hidden))
        return self.output(hidden)


@dataclass(frozen=True)
class DecodeThenFilterSample:
    """Observable products of one four-corner DTF texture lookup."""

    corners: torch.Tensor
    weights: torch.Tensor
    filtered_latent: torch.Tensor
    corner_material: DecodedMaterial
    material: DecodedMaterial


@dataclass(frozen=True)
class PairedPrecheckInitialization:
    """Activation-independent CPU state shared by the paired prechecks."""

    seed: int
    latent_channels: int
    decoder_width: int
    latent_values: torch.Tensor
    decoder_state: Mapping[str, torch.Tensor]
    sha256: str


@dataclass(frozen=True)
class PairedPrecheckCandidate:
    """One mutable candidate cloned from a paired immutable initialization."""

    activation: str
    initial_sha256: str
    latent: nn.Parameter
    decoder: DecodeThenFilterDecoder


def _quantize(values: torch.Tensor, mode: str) -> torch.Tensor:
    if mode in {"float", "prequantized"}:
        return values
    if mode == "hard":
        return hard_quantize_unorm8(values)
    if mode == "fake":
        return fake_quantize_unorm8(values)
    raise ValueError(f"unsupported DTF quantization mode: {mode}")


def _weighted(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return torch.sum(values * weights[..., None], dim=1)


def decode_then_filter_sample(
    latent_texture: torch.Tensor,
    uv: torch.Tensor,
    decoder: DecodeThenFilterDecoder,
    *,
    quantization: str,
) -> DecodeThenFilterSample:
    """Quantize texels, decode four corners, then filter Core-4 semantics."""

    if latent_texture.ndim != 3:
        raise ValueError("DTF latent texture must be HWC")
    if latent_texture.shape[-1] != decoder.latent_channels:
        raise ValueError("DTF latent texture channels do not match the decoder")
    deployed = _quantize(latent_texture, quantization)
    corners, weights = bilinear_corners_top_down_wrap_torch(deployed, uv)
    filtered_latent = _weighted(corners, weights)
    corner_material = postprocess_raw_torch(decoder(corners))

    filtered_normal = F.normalize(
        _weighted(corner_material.normal_xyz, weights),
        dim=-1,
        eps=1.0e-8,
    )
    material = DecodedMaterial(
        base_color_linear=_weighted(corner_material.base_color_linear, weights),
        normal_xy=filtered_normal[..., :2],
        normal_xyz=filtered_normal,
        roughness=_weighted(corner_material.roughness, weights),
        metallic=_weighted(corner_material.metallic, weights),
    )
    return DecodeThenFilterSample(
        corners=corners,
        weights=weights,
        filtered_latent=filtered_latent,
        corner_material=corner_material,
        material=material,
    )


def calculate_decode_then_filter_cost(
    decoder: DecodeThenFilterDecoder,
    *,
    height: int,
    width: int,
) -> dict[str, object]:
    """Report DTF arithmetic, texture traffic, and honest storage boundaries."""

    if height <= 0 or width <= 0:
        raise ValueError("DTF texture dimensions must be positive")
    parameters = sum(parameter.numel() for parameter in decoder.parameters())
    weight_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in decoder.parameters()
    )
    macs_per_corner = (
        decoder.latent_channels * decoder.width
        + decoder.width * decoder.width
        + decoder.width * 7
    )
    texture_resources = (decoder.latent_channels + 3) // 4
    activation_evaluations = 4 * 2 * decoder.width
    return {
        "module_identifier": decoder.module_identifier,
        "shape": (
            f"{decoder.latent_channels}->{decoder.width}->{decoder.width}->7"
        ),
        "parameters": parameters,
        "weight_bytes_float32": weight_bytes,
        "decoder_macs_per_corner": macs_per_corner,
        "decoder_macs_per_pixel": 4 * macs_per_corner,
        "texture_resources": texture_resources,
        "point_texel_loads_per_pixel": 4 * texture_resources,
        "material_filter_scalar_channels": 8,
        "normal_normalizations_per_pixel": 1,
        "theoretical_raw_bytes_unorm8": (
            height * width * decoder.latent_channels
        ),
        "actual_resident_bytes": None,
        "actual_resident_measurement": "required_in_target_runtime",
        "activation": {
            "kind": decoder.activation,
            "hidden_units_per_layer": decoder.width,
            "hidden_layers": 2,
            "activation_evaluations_per_pixel": activation_evaluations,
            "special_functions_per_pixel": (
                activation_evaluations if decoder.activation == "silu" else 0
            ),
        },
    }


def export_decode_then_filter_latent_unorm8(
    latent_bounded: torch.Tensor,
    *,
    output_dir: Path | str,
    stem: str,
) -> dict[str, object]:
    """Export C4 as RGBA8 or C5 as RGBA8 plus one R8 resource."""

    if not stem or Path(stem).name != stem:
        raise ValueError("DTF latent export stem must be one plain file stem")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    packed = pack_decode_then_filter_latent_unorm8(latent_bounded)
    resources: list[dict[str, object]] = []
    for item in packed:
        path = directory / f"{stem}_{item['suffix']}_unorm8.png"
        Image.fromarray(item["encoded"], mode=str(item["mode"])).save(
            path, format="PNG"
        )
        resources.append(
            {
                "semantic": item["semantic"],
                "storage_channels": item["storage_channels"],
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "file_bytes": path.stat().st_size,
                "raw_bytes": int(item["encoded"].size),
            }
        )
    return {
        "latent_channels": int(latent_bounded.shape[-1]),
        "texture_resources": len(resources),
        "theoretical_raw_bytes_unorm8": sum(
            int(item["encoded"].size) for item in packed
        ),
        "resources": resources,
    }


def pack_decode_then_filter_latent_unorm8(
    latent_bounded: torch.Tensor,
) -> list[dict[str, object]]:
    """Return exact C4/C5 UNORM8 resource planes without filesystem effects."""

    if latent_bounded.ndim != 3 or latent_bounded.shape[-1] not in (4, 5):
        raise ValueError("DTF latent packing requires an HxWx4 or HxWx5 tensor")
    encoded = unorm8_encode_half_up(latent_bounded).detach().cpu().numpy()
    packed: list[dict[str, object]] = [
        {
            "semantic": "latent_channels_0_3",
            "storage_channels": "rgba8",
            "suffix": "rgba",
            "mode": "RGBA",
            "encoded": encoded[..., :4],
        }
    ]
    if encoded.shape[-1] == 5:
        packed.append(
            {
                "semantic": "latent_channel_4",
                "storage_channels": "r8",
                "suffix": "r",
                "mode": "L",
                "encoded": encoded[..., 4],
            }
        )
    return packed


def _paired_initialization_hash(
    *,
    seed: int,
    latent_channels: int,
    decoder_width: int,
    latent_values: torch.Tensor,
    decoder_state: Mapping[str, torch.Tensor],
) -> str:
    digest = hashlib.sha256()
    digest.update(
        (
            f"seed={seed};latent_channels={latent_channels};"
            f"decoder_width={decoder_width}"
        ).encode("ascii")
    )
    tensors = {"latent_values": latent_values, **decoder_state}
    for name in sorted(tensors):
        tensor = tensors[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def make_paired_precheck_initialization(
    *,
    height: int,
    width: int,
    latent_channels: int,
    decoder_width: int,
    seed: int,
) -> PairedPrecheckInitialization:
    """Create one deterministic C4/C5 latent and affine decoder state on CPU."""

    if height <= 0 or width <= 0:
        raise ValueError("paired DTF latent dimensions must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    latent_values = torch.rand(
        (height, width, latent_channels), generator=generator
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed + 1)
        decoder = DecodeThenFilterDecoder(
            latent_channels=latent_channels,
            width=decoder_width,
            activation="relu",
        )
    decoder_state = {
        name: value.detach().cpu().clone()
        for name, value in decoder.state_dict().items()
    }
    digest = _paired_initialization_hash(
        seed=seed,
        latent_channels=latent_channels,
        decoder_width=decoder_width,
        latent_values=latent_values,
        decoder_state=decoder_state,
    )
    return PairedPrecheckInitialization(
        seed=seed,
        latent_channels=latent_channels,
        decoder_width=decoder_width,
        latent_values=latent_values.detach().clone(),
        decoder_state=decoder_state,
        sha256=digest,
    )


def instantiate_paired_precheck_candidate(
    initialization: PairedPrecheckInitialization,
    *,
    activation: str,
    device: torch.device | str = "cpu",
) -> PairedPrecheckCandidate:
    """Clone a ReLU or SiLU precheck without sharing mutable storage."""

    if activation not in {"relu", "silu"}:
        raise ValueError("paired DTF precheck activation must be relu or silu")
    decoder = DecodeThenFilterDecoder(
        latent_channels=initialization.latent_channels,
        width=initialization.decoder_width,
        activation=activation,
    ).to(device)
    decoder.load_state_dict(
        {
            name: value.detach().clone().to(device)
            for name, value in initialization.decoder_state.items()
        }
    )
    decoder.module_identifier = (
        f"scifihelmet_c4_dtf_v1.precheck.{activation}."
        f"c{initialization.latent_channels}_w{initialization.decoder_width}"
    )
    return PairedPrecheckCandidate(
        activation=activation,
        initial_sha256=initialization.sha256,
        latent=nn.Parameter(initialization.latent_values.detach().clone().to(device)),
        decoder=decoder,
    )
