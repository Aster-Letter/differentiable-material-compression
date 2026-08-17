"""Safe single-affine decoding for the C4 material-compression mainline."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import torch
from torch import nn
import torch.nn.functional as F


SCALAR_ROWS = (0, 1, 2, 5, 6)
AFFINE_OUTPUT_SEMANTICS = (
    "base_color_linear_r",
    "base_color_linear_g",
    "base_color_linear_b",
    "normal_tangent_x",
    "normal_tangent_y",
    "roughness_linear",
    "metallic_linear",
)
AFFINE_STATIC_COST = {
    "parameters": 35,
    "weight_bytes_fp32": 140,
    "decoder_macs_per_pixel": 28,
    "texture_resources": 1,
    "filtered_samples_per_pixel": 1,
}


@dataclass(frozen=True)
class AffineDecodedMaterial:
    """Direct seven-channel affine output plus reconstructed positive-Z normal."""

    base_color_linear: torch.Tensor
    normal_xy: torch.Tensor
    normal_xyz: torch.Tensor
    roughness: torch.Tensor
    metallic: torch.Tensor


@dataclass(frozen=True)
class AffineDecoderArtifact:
    """Portable FP32 affine payload and its hash-bound manifest."""

    payload: bytes
    manifest: dict[str, object]


def reconstruct_positive_normal(normal_xy: torch.Tensor) -> torch.Tensor:
    """Reconstruct positive Z with only a numerical sqrt guard."""

    normal_z = torch.sqrt(
        torch.clamp(
            1.0 - torch.sum(normal_xy.square(), dim=-1, keepdim=True),
            min=torch.finfo(normal_xy.dtype).eps,
        )
    )
    return torch.cat((normal_xy, normal_z), dim=-1)


def certify_affine(
    weight: torch.Tensor,
    bias: torch.Tensor,
    *,
    margin: float,
) -> dict[str, object]:
    """Return a fail-closed full-domain safety certificate for deployment ``W,b``."""

    if weight.shape != (7, 4) or bias.shape != (7,):
        raise ValueError("affine weight/bias shape must be (7, 4) and (7,)")
    if not weight.is_floating_point() or not bias.is_floating_point():
        raise ValueError("affine certificate requires floating-point tensors")
    if weight.dtype != bias.dtype or weight.device != bias.device:
        raise ValueError("affine weight/bias dtype and device must match")
    if not 0.0 < margin < 0.5:
        raise ValueError("certificate margin must be in (0, 0.5)")
    if not bool(torch.isfinite(weight).all() and torch.isfinite(bias).all()):
        raise ValueError("affine certificate requires finite weight and bias")

    scalar_weight = weight[list(SCALAR_ROWS)]
    scalar_bias = bias[list(SCALAR_ROWS)]
    exact_lower = scalar_bias + torch.minimum(
        scalar_weight, torch.zeros_like(scalar_weight)
    ).sum(dim=-1)
    exact_upper = scalar_bias + torch.maximum(
        scalar_weight, torch.zeros_like(scalar_weight)
    ).sum(dim=-1)
    scalar_roundoff_guard = (
        8.0
        * torch.finfo(weight.dtype).eps
        * (scalar_bias.abs() + scalar_weight.abs().sum(dim=-1)).clamp_min(1.0)
    )
    lower = exact_lower - scalar_roundoff_guard
    upper = exact_upper + scalar_roundoff_guard

    normal_vectors = 0.5 * weight[3:5].transpose(0, 1)
    normal_center = bias[3:5] + normal_vectors.sum(dim=0)
    normal_radius = torch.linalg.vector_norm(normal_center) + torch.linalg.vector_norm(
        normal_vectors, dim=-1
    ).sum()
    normal_roundoff_guard = (
        16.0
        * torch.finfo(weight.dtype).eps
        * normal_radius.detach().abs().clamp_min(1.0)
    )
    normal_max_radius = normal_radius + normal_roundoff_guard
    certificate_margin = torch.minimum(
        torch.minimum(lower.amin(), (1.0 - upper).amin()),
        1.0 - normal_max_radius,
    )
    valid = bool(
        torch.all(exact_lower >= margin)
        and torch.all(exact_upper <= 1.0 - margin)
        and normal_radius <= 1.0 - margin
        and certificate_margin > 0.0
    )
    if not valid:
        raise ValueError("affine certificate invalid for the full latent domain")

    return {
        "valid": True,
        "finite": True,
        "dtype": str(weight.dtype).removeprefix("torch."),
        "margin": float(margin),
        "scalar_lower_bounds": [float(value) for value in lower.detach().cpu()],
        "scalar_upper_bounds": [float(value) for value in upper.detach().cpu()],
        "normal_max_radius": float(normal_max_radius.detach().cpu()),
        "certificate_margin": float(certificate_margin.detach().cpu()),
        "scalar_roundoff_guard": [
            float(value) for value in scalar_roundoff_guard.detach().cpu()
        ],
        "normal_roundoff_guard": float(normal_roundoff_guard.detach().cpu()),
    }


def decode_affine_material(
    latent_rgba: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> AffineDecodedMaterial:
    """Decode one filtered RGBA sample with direct affine material semantics."""

    seven = F.linear(latent_rgba, weight, bias)
    normal_xy = seven[..., 3:5]
    return AffineDecodedMaterial(
        base_color_linear=seven[..., 0:3],
        normal_xy=normal_xy,
        normal_xyz=reconstruct_positive_normal(normal_xy),
        roughness=seven[..., 5:6],
        metallic=seven[..., 6:7],
    )


def decode_filtered_affine_material(
    latent_samples: torch.Tensor,
    filter_weights: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> AffineDecodedMaterial:
    """Simulate one hardware-filtered latent sample followed by affine decode."""

    filtered_latent = torch.sum(
        latent_samples * filter_weights[..., None], dim=-2
    )
    return decode_affine_material(filtered_latent, weight, bias)


def export_affine_decoder(decoder: "SafeAffineMaterialDecoder") -> AffineDecoderArtifact:
    """Pack a safe decoder into the exact 140-byte deployment payload."""

    weight, bias = decoder.fold_affine()
    packed = torch.cat((weight.reshape(-1), bias)).detach().to(
        device="cpu", dtype=torch.float32
    ).contiguous()
    payload = packed.numpy().tobytes(order="C")
    export_weight = packed[:28].reshape(7, 4)
    export_bias = packed[28:]
    manifest: dict[str, object] = {
        "schema_version": 1,
        "semantics": list(AFFINE_OUTPUT_SEMANTICS),
        "cost": dict(AFFINE_STATIC_COST),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "certificate": certify_affine(
            export_weight, export_bias, margin=decoder.margin
        ),
    }
    return AffineDecoderArtifact(payload=payload, manifest=manifest)


def reload_affine_decoder(
    artifact: AffineDecoderArtifact,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate and reload an exported decoder without changing its values."""

    expected_hash = artifact.manifest.get("payload_sha256")
    actual_hash = hashlib.sha256(artifact.payload).hexdigest()
    if actual_hash != expected_hash:
        raise ValueError("affine payload SHA-256 mismatch")
    if len(artifact.payload) != AFFINE_STATIC_COST["weight_bytes_fp32"]:
        raise ValueError("affine payload length must be 140 bytes")
    if artifact.manifest.get("semantics") != list(AFFINE_OUTPUT_SEMANTICS):
        raise ValueError("affine output semantics mismatch")
    if artifact.manifest.get("cost") != AFFINE_STATIC_COST:
        raise ValueError("affine static cost mismatch")

    packed = torch.frombuffer(bytearray(artifact.payload), dtype=torch.float32).clone()
    weight = packed[:28].reshape(7, 4)
    bias = packed[28:]
    certificate = artifact.manifest.get("certificate")
    if not isinstance(certificate, dict):
        raise ValueError("affine certificate is missing")
    certify_affine(weight, bias, margin=float(certificate.get("margin", 0.0)))
    return weight, bias


class SafeAffineMaterialDecoder(nn.Module):
    """Smoothly parameterized affine whose full RGBA domain is material-safe."""

    def __init__(self, margin: float = 1.0e-3, direction_epsilon: float = 1.0e-8) -> None:
        super().__init__()
        if not 0.0 < margin < 0.5:
            raise ValueError("margin must be in (0, 0.5)")
        if direction_epsilon <= 0.0:
            raise ValueError("direction_epsilon must be positive")
        self.margin = float(margin)
        self.direction_epsilon = float(direction_epsilon)

        # Five scalar rows each use five coefficient shares plus one slack share.
        self.scalar_share_logits = nn.Parameter(torch.zeros(5, 6))
        self.scalar_raw_coefficients = nn.Parameter(torch.zeros(5, 5))

        # Normal XY uses five two-dimensional coefficient vectors and one slack.
        self.normal_share_logits = nn.Parameter(torch.zeros(6))
        self.normal_raw_vectors = nn.Parameter(torch.zeros(5, 2))

    @classmethod
    def from_safe_affine(
        cls,
        weight: torch.Tensor,
        bias: torch.Tensor,
        *,
        margin: float,
        direction_epsilon: float = 1.0e-8,
    ) -> "SafeAffineMaterialDecoder":
        """Initialize raw training parameters by an explicit interior inverse."""

        certify_affine(weight, bias, margin=margin)
        decoder = cls(margin=margin, direction_epsilon=direction_epsilon).to(
            device=weight.device, dtype=weight.dtype
        )
        scalar_weight = weight[list(SCALAR_ROWS)]
        scalar = torch.cat(
            (
                (
                    bias[list(SCALAR_ROWS)]
                    - 0.5
                    + 0.5 * scalar_weight.sum(dim=-1)
                )[:, None],
                0.5 * scalar_weight,
            ),
            dim=-1,
        )
        normal_vectors = 0.5 * weight[3:5].transpose(0, 1)
        normal = torch.cat(
            ((bias[3:5] + normal_vectors.sum(dim=0))[None, :], normal_vectors),
            dim=0,
        )

        def shares_from_magnitudes(magnitudes: torch.Tensor) -> torch.Tensor:
            remaining = 1.0 - magnitudes.sum(dim=-1, keepdim=True)
            coefficient_shares = magnitudes + remaining / (2.0 * 5.0)
            slack_share = remaining / 2.0
            return torch.cat((coefficient_shares, slack_share), dim=-1)

        scalar_budget = 0.5 - margin
        scalar_shares = shares_from_magnitudes(torch.abs(scalar) / scalar_budget)
        scalar_ratio = scalar / (scalar_budget * scalar_shares[..., :5])

        normal_budget = 1.0 - margin
        normal_magnitudes = torch.linalg.vector_norm(normal, dim=-1) / normal_budget
        normal_shares = shares_from_magnitudes(normal_magnitudes[None, :])[0]
        normal_direction = normal / (normal_budget * normal_shares[:5, None])
        inverse_denominator = torch.sqrt(
            1.0 - torch.sum(normal_direction.square(), dim=-1, keepdim=True)
        )
        normal_raw = direction_epsilon * normal_direction / inverse_denominator

        with torch.no_grad():
            decoder.scalar_share_logits.copy_(torch.log(scalar_shares))
            decoder.scalar_raw_coefficients.copy_(torch.atanh(scalar_ratio))
            decoder.normal_share_logits.copy_(torch.log(normal_shares))
            decoder.normal_raw_vectors.copy_(normal_raw)
        return decoder

    def _centered_coefficients(self) -> tuple[torch.Tensor, torch.Tensor]:
        scalar_shares = torch.softmax(self.scalar_share_logits, dim=-1)[..., :5]
        scalar = (
            (0.5 - self.margin)
            * scalar_shares
            * torch.tanh(self.scalar_raw_coefficients)
        )

        normal_shares = torch.softmax(self.normal_share_logits, dim=-1)[:5]
        lengths = torch.sqrt(
            torch.sum(self.normal_raw_vectors.square(), dim=-1, keepdim=True)
            + self.direction_epsilon**2
        )
        normal = (
            (1.0 - self.margin)
            * normal_shares[:, None]
            * self.normal_raw_vectors
            / lengths
        )
        return scalar, normal

    def fold_affine(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Fold centered-domain safe coefficients into deployment ``Wz+b``."""

        scalar, normal = self._centered_coefficients()
        weight = scalar.new_empty((7, 4))
        bias = scalar.new_empty(7)

        weight[list(SCALAR_ROWS)] = 2.0 * scalar[:, 1:]
        bias[list(SCALAR_ROWS)] = 0.5 + scalar[:, 0] - scalar[:, 1:].sum(dim=-1)
        weight[3:5] = 2.0 * normal[1:].transpose(0, 1)
        bias[3:5] = normal[0] - normal[1:].sum(dim=0)
        return weight, bias

    def forward(self, latent_rgba: torch.Tensor) -> AffineDecodedMaterial:
        weight, bias = self.fold_affine()
        seven = F.linear(latent_rgba, weight, bias)
        normal_xy = seven[..., 3:5]
        return AffineDecodedMaterial(
            base_color_linear=seven[..., 0:3],
            normal_xy=normal_xy,
            normal_xyz=reconstruct_positive_normal(normal_xy),
            roughness=seven[..., 5:6],
            metallic=seven[..., 6:7],
        )
