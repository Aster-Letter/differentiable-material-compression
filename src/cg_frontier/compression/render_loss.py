"""Deployment-order RGBA latent sampling, shared-PBR losses, metrics, and export."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from typing import Mapping

import numpy as np
from PIL import Image
import torch

from cg_frontier.compression.material import (
    DecodedMaterial,
    MaterialDecoder,
    decode_material,
)
from cg_frontier.render.gbuffer import (
    GBufferResult,
    MaterialBuffers,
    tangent_normal_to_world,
)
from cg_frontier.render.pbr import (
    PointLight,
    linear_to_srgb_torch,
    shade_ggx,
)
from cg_frontier.render.gbuffer import Camera


@dataclass(frozen=True)
class RenderLossTerms:
    total: torch.Tensor
    charbonnier_hdr: torch.Tensor
    log1p_hdr: torch.Tensor
    material_auxiliary: torch.Tensor


def unorm8_encode_half_up(values: torch.Tensor) -> torch.Tensor:
    """Encode [0,1] floats with the frozen floor(x*255+0.5) rule."""

    return torch.floor(torch.clamp(values, 0.0, 1.0) * 255.0 + 0.5).to(torch.uint8)


def hard_quantize_unorm8(values: torch.Tensor) -> torch.Tensor:
    """Return exact UNORM8 dequantized values while retaining the input float dtype."""

    encoded = unorm8_encode_half_up(values)
    return encoded.to(dtype=values.dtype) / 255.0


def fake_quantize_unorm8(values: torch.Tensor) -> torch.Tensor:
    """Exact hard-quant forward with an identity straight-through estimator."""

    hard = hard_quantize_unorm8(values)
    return values + (hard - values).detach()


def latent_float_to_logits(
    latent: torch.Tensor, *, clamp_epsilon: float = 1.0e-6
) -> torch.Tensor:
    """Clamp an existing bounded latent, then convert it to trainable sigmoid logits."""

    if not (0.0 < clamp_epsilon < 0.5):
        raise ValueError("clamp_epsilon must be within (0, 0.5)")
    bounded = latent.clamp(clamp_epsilon, 1.0 - clamp_epsilon)
    return torch.log(bounded) - torch.log1p(-bounded)


def bilinear_sample_top_down_wrap(
    texture: torch.Tensor, uv: torch.Tensor
) -> torch.Tensor:
    """Sample HWC top-down data with glTF v=0 at row 0 and periodic half-texel bilinear filtering."""

    if texture.ndim != 3 or uv.ndim < 2 or uv.shape[-1] != 2:
        raise ValueError("texture must be HWC and uv must end in two coordinates")
    height, width, channels = texture.shape
    if height <= 0 or width <= 0:
        raise ValueError("texture dimensions must be positive")
    # The -0.5 offset places integer texel centers at (i + 0.5) / extent,
    # matching GPU bilinear sampling before the per-pixel decoder executes.
    x = uv[..., 0] * width - 0.5
    y = uv[..., 1] * height - 0.5
    x0_floor = torch.floor(x)
    y0_floor = torch.floor(y)
    wx = (x - x0_floor)[..., None]
    wy = (y - y0_floor)[..., None]
    x0 = x0_floor.to(torch.int64).remainder(width)
    y0 = y0_floor.to(torch.int64).remainder(height)
    x1 = (x0 + 1).remainder(width)
    y1 = (y0 + 1).remainder(height)
    flat = texture.reshape(height * width, channels)

    def gather(row: torch.Tensor, column: torch.Tensor) -> torch.Tensor:
        return flat[(row * width + column).reshape(-1)].reshape(*row.shape, channels)

    top = gather(y0, x0) * (1.0 - wx) + gather(y0, x1) * wx
    bottom = gather(y1, x0) * (1.0 - wx) + gather(y1, x1) * wx
    return top * (1.0 - wy) + bottom * wy


def sparse_fake_quantized_bilinear_sample_top_down_wrap(
    texture: torch.Tensor, uv: torch.Tensor
) -> torch.Tensor:
    """Fake-quantize only the four texels used by each wrapped bilinear sample.

    The forward result is identical to quantizing the complete texture before
    sampling, but avoids materializing a full quantized 2048-square atlas on
    every differentiable-rendering step.
    """

    if texture.ndim != 3 or uv.ndim < 2 or uv.shape[-1] != 2:
        raise ValueError("texture must be HWC and uv must end in two coordinates")
    height, width, channels = texture.shape
    if height <= 0 or width <= 0:
        raise ValueError("texture dimensions must be positive")
    x = uv[..., 0] * width - 0.5
    y = uv[..., 1] * height - 0.5
    x0_floor = torch.floor(x)
    y0_floor = torch.floor(y)
    wx = (x - x0_floor)[..., None]
    wy = (y - y0_floor)[..., None]
    x0 = x0_floor.to(torch.int64).remainder(width)
    y0 = y0_floor.to(torch.int64).remainder(height)
    x1 = (x0 + 1).remainder(width)
    y1 = (y0 + 1).remainder(height)
    flat = texture.reshape(height * width, channels)

    def gather(row: torch.Tensor, column: torch.Tensor) -> torch.Tensor:
        selected = flat[(row * width + column).reshape(-1)].reshape(
            *row.shape, channels
        )
        return fake_quantize_unorm8(selected)

    top = gather(y0, x0) * (1.0 - wx) + gather(y0, x1) * wx
    bottom = gather(y1, x0) * (1.0 - wx) + gather(y1, x1) * wx
    return top * (1.0 - wy) + bottom * wy


def decoded_to_material(
    geometry: GBufferResult, decoded: DecodedMaterial
) -> MaterialBuffers:
    """Convert decoder outputs to a complete PBR material without any color decoding."""

    mask = geometry.torch_buffers["mask"]
    normal_world = tangent_normal_to_world(geometry, decoded.normal_xyz)

    def vector(value: torch.Tensor) -> torch.Tensor:
        return torch.where(mask[..., None], value, torch.zeros_like(value))

    def scalar(value: torch.Tensor) -> torch.Tensor:
        squeezed = value[..., 0] if value.shape[-1:] == (1,) else value
        return torch.where(mask, squeezed, torch.zeros_like(squeezed))

    return MaterialBuffers(
        base_color_linear=vector(decoded.base_color_linear),
        normal_world=vector(normal_world),
        roughness=scalar(decoded.roughness),
        metallic=scalar(decoded.metallic),
        normal_ts_raw=vector(decoded.normal_xyz),
        normal_ts_unit=vector(decoded.normal_xyz),
    )


def sample_and_decode_latent(
    geometry: GBufferResult,
    latent_bounded: torch.Tensor,
    decoder: MaterialDecoder,
    *,
    quantization: str,
) -> MaterialBuffers:
    """Quantize texels first, sample RGBA second, then decode per pixel."""

    # Quantization belongs to stored texels, never to interpolated samples.
    if quantization == "float":
        deployed = latent_bounded
    elif quantization == "hard":
        deployed = hard_quantize_unorm8(latent_bounded)
    elif quantization == "fake":
        deployed = fake_quantize_unorm8(latent_bounded)
    else:
        raise ValueError(f"unsupported quantization mode: {quantization}")
    # Frozen deployment order: RGBA texel -> bilinear RGBA -> decoder -> one
    # semantic postprocess in decode_material/decoded_to_material.
    sampled = bilinear_sample_top_down_wrap(deployed, geometry.torch_buffers["uv"])
    return decoded_to_material(geometry, decode_material(decoder, sampled))


def render_latent_material(
    geometry: GBufferResult,
    camera: Camera,
    light: PointLight,
    latent_bounded: torch.Tensor,
    decoder: MaterialDecoder,
    *,
    quantization: str,
    minimum_roughness: float,
) -> tuple[torch.Tensor, MaterialBuffers]:
    material = sample_and_decode_latent(
        geometry, latent_bounded, decoder, quantization=quantization
    )
    image = shade_ggx(
        geometry,
        camera,
        light,
        material_override=material,
        minimum_roughness=minimum_roughness,
    )
    return image, material


def display_transform(linear_hdr: torch.Tensor, exposure: float) -> torch.Tensor:
    """Frozen exposure, Reinhard, and linear-to-sRGB display transform."""

    if exposure <= 0.0:
        raise ValueError("display exposure must be positive")
    mapped = torch.clamp(linear_hdr, min=0.0) * exposure
    mapped = mapped / (1.0 + mapped)
    return linear_to_srgb_torch(mapped).clamp(0.0, 1.0)


def masked_render_metrics(
    reference_hdr: torch.Tensor,
    candidate_hdr: torch.Tensor,
    mask: torch.Tensor,
    *,
    linear_psnr_data_range: float,
    display_exposure: float,
) -> dict[str, float | int | str]:
    """Compute only foreground metrics; SSIM is the explicit masked global variant."""

    if reference_hdr.shape != candidate_hdr.shape or reference_hdr.shape[-1] != 3:
        raise ValueError("reference and candidate HDR images must have matching RGB shapes")
    if mask.shape != reference_hdr.shape[:-1] or not torch.any(mask):
        raise ValueError("mask must match the image and contain foreground pixels")
    if linear_psnr_data_range <= 0.0:
        raise ValueError("linear PSNR data range must be positive")
    difference = candidate_hdr[mask] - reference_hdr[mask]
    absolute = difference.abs()
    mse = difference.square().mean()
    linear_psnr = 10.0 * torch.log10(
        torch.tensor(linear_psnr_data_range**2, device=mse.device, dtype=mse.dtype)
        / mse.clamp_min(torch.finfo(mse.dtype).tiny)
    )
    reference_display = display_transform(reference_hdr, display_exposure)[mask]
    candidate_display = display_transform(candidate_hdr, display_exposure)[mask]
    display_mse = (candidate_display - reference_display).square().mean()
    display_psnr = -10.0 * torch.log10(
        display_mse.clamp_min(torch.finfo(display_mse.dtype).tiny)
    )
    x = reference_display.reshape(-1)
    y = candidate_display.reshape(-1)
    c1 = 0.01**2
    c2 = 0.03**2
    mean_x = x.mean()
    mean_y = y.mean()
    variance_x = ((x - mean_x) ** 2).mean()
    variance_y = ((y - mean_y) ** 2).mean()
    covariance = ((x - mean_x) * (y - mean_y)).mean()
    ssim = ((2 * mean_x * mean_y + c1) * (2 * covariance + c2)) / (
        (mean_x.square() + mean_y.square() + c1) * (variance_x + variance_y + c2)
    )
    return {
        "masked_linear_hdr_mae": float(absolute.mean().detach().cpu()),
        "masked_linear_hdr_rmse": float(torch.sqrt(mse).detach().cpu()),
        "linear_psnr_db": float(linear_psnr.detach().cpu()),
        "linear_psnr_data_range": float(linear_psnr_data_range),
        "display_psnr_db": float(display_psnr.detach().cpu()),
        "display_ssim": float(ssim.detach().cpu()),
        "display_ssim_definition": "foreground-only global RGB SSIM, C1=0.01^2, C2=0.03^2",
        "max_absolute_error": float(absolute.max().detach().cpu()),
        "foreground_pixel_count": int(mask.sum().detach().cpu()),
    }


def pbr_render_loss(
    reference_hdr: torch.Tensor,
    candidate_hdr: torch.Tensor,
    mask: torch.Tensor,
    material_auxiliary: torch.Tensor,
    *,
    charbonnier_epsilon: float,
    charbonnier_weight: float,
    log1p_weight: float,
    material_weight: float,
) -> RenderLossTerms:
    difference = candidate_hdr[mask] - reference_hdr[mask]
    charbonnier = torch.sqrt(difference.square() + charbonnier_epsilon**2).mean()
    log1p = torch.abs(
        torch.log1p(candidate_hdr[mask].clamp_min(0.0))
        - torch.log1p(reference_hdr[mask].clamp_min(0.0))
    ).mean()
    total = (
        charbonnier * charbonnier_weight
        + log1p * log1p_weight
        + material_auxiliary * material_weight
    )
    return RenderLossTerms(total, charbonnier, log1p, material_auxiliary)


def export_latent_unorm8_png(latent_bounded: torch.Tensor, path: Path | str) -> dict[str, int | str]:
    """Export an RGBA PNG whose decoded bytes are exactly the hard-quantized latent."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = unorm8_encode_half_up(latent_bounded).detach().cpu().numpy()
    if encoded.ndim != 3 or encoded.shape[-1] != 4:
        raise ValueError("latent PNG export requires an HxWx4 tensor")
    Image.fromarray(encoded, mode="RGBA").save(path, format="PNG")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"sha256": digest, "file_bytes": path.stat().st_size, "raw_bytes": int(encoded.size)}


def load_latent_unorm8_png(
    path: Path | str, *, device: torch.device | str = "cpu"
) -> tuple[np.ndarray, torch.Tensor]:
    """Return exact RGBA bytes and normalized float dequantization from a PNG."""

    with Image.open(path) as image:
        image.load()
        encoded = np.array(image.convert("RGBA"), dtype=np.uint8, copy=True)
    tensor = torch.from_numpy(encoded).to(device=device, dtype=torch.float32) / 255.0
    return encoded, tensor


def decoder_weight_bytes(decoder: MaterialDecoder) -> int:
    return sum(parameter.numel() * parameter.element_size() for parameter in decoder.parameters())


def orbit_camera(
    *,
    yaw_degrees: float,
    elevation_degrees: float,
    radius: float,
    target: tuple[float, float, float],
    up: tuple[float, float, float],
    vertical_fov_degrees: float,
    near: float,
    far: float,
) -> Camera:
    """Construct the frozen +Z-front orbit camera from reviewable yaw/elevation values."""

    yaw = math.radians(yaw_degrees)
    elevation = math.radians(elevation_degrees)
    horizontal = radius * math.cos(elevation)
    eye = (
        target[0] + horizontal * math.sin(yaw),
        target[1] + radius * math.sin(elevation),
        target[2] + horizontal * math.cos(yaw),
    )
    return Camera(
        eye=eye,
        target=target,
        up=up,
        vertical_fov_degrees=vertical_fov_degrees,
        near=near,
        far=far,
    )


def assert_disjoint_case_partitions(
    train_cameras: list[Mapping[str, object]],
    holdout_cameras: list[Mapping[str, object]],
    train_lights: list[Mapping[str, object]],
    holdout_lights: list[Mapping[str, object]],
) -> None:
    """Reject selection leakage by physical parameters, not merely case labels."""

    def camera_key(value: Mapping[str, object]) -> tuple[float, float]:
        return (float(value["elevation_degrees"]), float(value["yaw_degrees"]) % 360.0)

    def light_key(value: Mapping[str, object]) -> tuple[float, ...]:
        position = tuple(float(item) for item in value["position"])  # type: ignore[arg-type]
        color = tuple(float(item) for item in value["color"])  # type: ignore[arg-type]
        return position + color + (
            float(value["radiant_intensity"]),
            float(value["ambient_intensity"]),
        )

    camera_overlap = set(map(camera_key, train_cameras)) & set(map(camera_key, holdout_cameras))
    light_overlap = set(map(light_key, train_lights)) & set(map(light_key, holdout_lights))
    if camera_overlap:
        raise ValueError(f"train/holdout camera overlap: {sorted(camera_overlap)}")
    if light_overlap:
        raise ValueError(f"train/holdout light overlap: {sorted(light_overlap)}")
