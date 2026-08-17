"""Fresh SciFiHelmet experiment for exact BaseColor affine fibers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F

from cg_frontier.compression.exact_basecolor import (
    AffineLatticeCodec,
    ExactAffineMaterialDecoder,
    ExactBaseColorCodec,
    LatticeCapacity,
    SeparatedCodec,
    codec_certificate,
    enumerate_lattice_capacity,
)
from cg_frontier.compression.material import DecodedMaterial, reconstruct_normal
from cg_frontier.render.gbuffer import (
    Camera,
    Core4Textures,
    GBufferResult,
    MaterialBuffers,
    tangent_normal_to_world,
)
from cg_frontier.render.pbr import PointLight, linear_to_srgb_torch, shade_ggx


SCHEMA_VERSION = 1
CHECKPOINT_TYPE = "scifihelmet_exact_basecolor_v1"
CANDIDATE_NAMES = ("U0-unconstrained", "S-separated", "M-mixed")


@dataclass(frozen=True)
class TexelTargets:
    base_float: torch.Tensor
    base_q8: torch.Tensor
    normal_xyz: torch.Tensor
    roughness: torch.Tensor
    metallic: torch.Tensor
    height: int
    width: int

    @property
    def count(self) -> int:
        return self.height * self.width

    def to(self, device: torch.device | str) -> "TexelTargets":
        return TexelTargets(
            self.base_float.to(device),
            self.base_q8.to(device),
            self.normal_xyz.to(device),
            self.roughness.to(device),
            self.metallic.to(device),
            self.height,
            self.width,
        )


@dataclass(frozen=True)
class RankOneInitialization:
    normalized_scalar: torch.Tensor
    direction: torch.Tensor
    regression: torch.Tensor
    sample_indices: torch.Tensor
    scalar_center: float
    scalar_scale: float


@dataclass(frozen=True)
class LatticeOracleRecord:
    capacity: LatticeCapacity
    material_score: float
    auxiliary_weight: torch.Tensor
    auxiliary_bias: torch.Tensor
    audit_residual: torch.Tensor

    def summary(self) -> dict[str, Any]:
        return {**self.capacity.as_dict(), "material_score": self.material_score}


@dataclass(frozen=True)
class RenderCase:
    name: str
    camera: Camera
    geometry: GBufferResult
    reference: MaterialBuffers
    valid_flat_indices: torch.Tensor


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return sha256_bytes(array.tobytes(order="C"))


def stable_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")


def load_texel_targets(textures: Core4Textures) -> TexelTargets:
    base = textures.base_color_linear.detach().cpu().to(torch.float32)
    encoded = torch.floor(base.clamp(0.0, 1.0) * 255.0 + 0.5).to(torch.uint8)
    normal = textures.normal.detach().cpu().to(torch.float32) * 2.0 - 1.0
    normal = F.normalize(normal, dim=-1, eps=1e-8)
    height, width = base.shape[:2]
    return TexelTargets(
        base.reshape(-1, 3),
        encoded.reshape(-1, 3),
        normal.reshape(-1, 3),
        textures.roughness.detach().cpu().reshape(-1, 1),
        textures.metallic.detach().cpu().reshape(-1, 1),
        height,
        width,
    )


def auxiliary_raw_targets(targets: TexelTargets, indices: torch.Tensor | None = None) -> torch.Tensor:
    ids = slice(None) if indices is None else indices
    normal_xy = targets.normal_xyz[ids, :2].clamp(-0.999, 0.999)
    roughness = targets.roughness[ids].clamp(1e-4, 1.0 - 1e-4)
    metallic = targets.metallic[ids].clamp(1e-4, 1.0 - 1e-4)
    return torch.cat((torch.atanh(normal_xy), torch.logit(roughness), torch.logit(metallic)), dim=-1)


def deterministic_sample_indices(count: int, sample_count: int, seed: int) -> torch.Tensor:
    if count <= 0 or sample_count <= 0:
        raise ValueError("sample counts must be positive")
    generator = np.random.default_rng(seed)
    chosen = generator.choice(count, size=min(count, sample_count), replace=False)
    return torch.from_numpy(np.sort(chosen).astype(np.int64))


def conditional_rank_one_initialization(
    targets: TexelTargets,
    *,
    sample_count: int,
    seed: int,
) -> RankOneInitialization:
    """Fit auxiliary raw values as affine(BaseColor)+one scalar residual."""

    indices = deterministic_sample_indices(targets.count, sample_count, seed)
    color = targets.base_q8[indices].to(torch.float64) / 255.0
    design = torch.cat((color, torch.ones((color.shape[0], 1), dtype=torch.float64)), dim=-1)
    raw = auxiliary_raw_targets(targets, indices).to(torch.float64)
    regression = torch.linalg.lstsq(design, raw).solution
    residual = raw - design @ regression
    _, _, vh = torch.linalg.svd(residual, full_matrices=False)
    direction = vh[0]
    sample_scalar = residual @ direction
    lower = float(torch.quantile(sample_scalar, 0.01))
    upper = float(torch.quantile(sample_scalar, 0.99))
    center = 0.5 * (lower + upper)
    scale = max(0.5 * (upper - lower), 1e-8)
    scalar_parts: list[torch.Tensor] = []
    chunk = 262_144
    for start in range(0, targets.count, chunk):
        stop = min(start + chunk, targets.count)
        full_color = targets.base_q8[start:stop].to(torch.float64) / 255.0
        full_design = torch.cat((full_color, torch.ones((stop - start, 1), dtype=torch.float64)), dim=-1)
        full_raw = auxiliary_raw_targets(targets, torch.arange(start, stop)).to(torch.float64)
        scalar = ((full_raw - full_design @ regression) @ direction - center) / scale
        scalar_parts.append(scalar.clamp(-1.0, 1.0).to(torch.float32))
    return RankOneInitialization(
        normalized_scalar=torch.cat(scalar_parts),
        direction=direction.to(torch.float32),
        regression=regression.to(torch.float32),
        sample_indices=indices,
        scalar_center=center,
        scalar_scale=scale,
    )


def _fit_affine_raw(latent: torch.Tensor, raw_targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = latent.to(torch.float64)
    y = raw_targets.to(torch.float64)
    design = torch.cat((x, torch.ones((x.shape[0], 1), dtype=x.dtype)), dim=-1)
    solution = torch.linalg.lstsq(design, y).solution
    return solution[:4].T.to(torch.float32), solution[4].to(torch.float32)


def _postprocessed_auxiliary(raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    normal_xy = torch.tanh(raw[..., :2])
    return reconstruct_normal(normal_xy), torch.sigmoid(raw[..., 2:3]), torch.sigmoid(raw[..., 3:4])


def _material_score(raw: torch.Tensor, targets: TexelTargets, indices: torch.Tensor) -> float:
    normal, roughness, metallic = _postprocessed_auxiliary(raw)
    cosine = 1.0 - torch.sum(normal * targets.normal_xyz[indices], dim=-1)
    score = 0.5 * cosine.mean() + 0.25 * F.l1_loss(roughness, targets.roughness[indices]) + 0.25 * F.l1_loss(metallic, targets.metallic[indices])
    return float(score)


def fit_lattice_oracle(
    capacity: LatticeCapacity,
    targets: TexelTargets,
    initialization: RankOneInitialization,
    *,
    iterations: int = 5,
) -> LatticeOracleRecord:
    codec = AffineLatticeCodec(capacity.kernel_rgb, capacity.t0)
    ids = initialization.sample_indices
    colors = targets.base_q8[ids]
    lower, upper = codec.valid_bounds(colors)
    normalized = initialization.normalized_scalar[ids]
    residual = lower.to(torch.float32) + 0.5 * (normalized + 1.0) * (upper - lower).to(torch.float32)
    residual = torch.floor(residual + 0.5)
    raw_targets = auxiliary_raw_targets(targets, ids)
    kernel = torch.tensor(codec.kernel, dtype=torch.float64) / 255.0
    weight = torch.empty((4, 4))
    bias = torch.empty(4)
    for _ in range(iterations):
        latent = codec.encode_hard(colors, residual).to(torch.float32) / 255.0
        weight, bias = _fit_affine_raw(latent, raw_targets)
        direction = weight.to(torch.float64) @ kernel
        denominator = float(torch.dot(direction, direction))
        if denominator <= 1e-16:
            break
        current = latent.to(torch.float64) @ weight.to(torch.float64).T + bias.to(torch.float64)
        zero = current - residual.to(torch.float64)[:, None] * direction[None, :]
        optimum = torch.sum((raw_targets.to(torch.float64) - zero) * direction[None, :], dim=-1) / denominator
        residual = torch.floor(torch.maximum(torch.minimum(optimum, upper.to(torch.float64)), lower.to(torch.float64)) + 0.5).to(torch.float32)
    latent = codec.encode_hard(colors, residual).to(torch.float32) / 255.0
    raw = latent @ weight.T + bias
    return LatticeOracleRecord(capacity, _material_score(raw, targets, ids), weight, bias, residual)


def search_lattice_oracles(
    targets: TexelTargets,
    initialization: RankOneInitialization,
    *,
    min_states: int = 64,
    capacity_top_k: int = 64,
    material_top_k: int = 8,
) -> list[LatticeOracleRecord]:
    colors, frequencies = np.unique(targets.base_q8.numpy(), axis=0, return_counts=True)
    capacities = enumerate_lattice_capacity(colors, frequencies, min_states=min_states, top_k=capacity_top_k)
    records = [fit_lattice_oracle(item, targets, initialization) for item in capacities]
    records.sort(key=lambda item: (item.material_score, item.capacity.capacity_sort_key))
    return records[:material_top_k]


class ExactExperimentCandidate(nn.Module):
    """One of U0/S/M with byte-domain trainable parameters."""

    def __init__(
        self,
        *,
        name: str,
        codec: ExactBaseColorCodec,
        colors_u8: torch.Tensor,
        initial_residual: torch.Tensor,
        auxiliary_weight: torch.Tensor,
        auxiliary_bias: torch.Tensor,
    ) -> None:
        super().__init__()
        if name not in CANDIDATE_NAMES:
            raise ValueError(f"unsupported exact BaseColor candidate: {name}")
        self.name = name
        self.codec = codec
        self.register_buffer("colors_u8", colors_u8.to(torch.uint8))
        self.decoder = ExactAffineMaterialDecoder(codec=codec, train_basecolor=name == "U0-unconstrained")
        with torch.no_grad():
            self.decoder.auxiliary.weight.copy_(auxiliary_weight)
            self.decoder.auxiliary.bias.copy_(auxiliary_bias)
        if name == "U0-unconstrained":
            initial = codec.encode_hard(colors_u8, initial_residual).to(torch.float32)
            self.latent_byte = nn.Parameter(initial)
            self.register_parameter("residual_byte", None)
        else:
            self.residual_byte = nn.Parameter(initial_residual.to(torch.float32).clone())
            self.register_parameter("latent_byte", None)

    @property
    def strict(self) -> bool:
        return self.name != "U0-unconstrained"

    @property
    def texel_count(self) -> int:
        return int(self.colors_u8.shape[0])

    def code_parameters(self) -> Iterable[nn.Parameter]:
        return (self.latent_byte,) if self.latent_byte is not None else (self.residual_byte,)

    def decoder_parameters(self) -> Iterable[nn.Parameter]:
        return (parameter for parameter in self.decoder.parameters() if parameter.requires_grad)

    def latent_for_ids(self, flat_ids: torch.Tensor, *, ste: bool) -> torch.Tensor:
        if self.latent_byte is not None:
            values = self.latent_byte[flat_ids].clamp(0.0, 255.0)
            hard = torch.floor(values + 0.5)
            values = values + (hard - values).detach() if ste else hard
            return values / 255.0
        colors = self.colors_u8[flat_ids]
        residual = self.residual_byte[flat_ids]
        return self.codec.encode_fake(colors, residual) if ste else self.codec.encode_hard(colors, residual).to(torch.float32) / 255.0

    def sample_uv(self, uv: torch.Tensor, *, height: int, width: int, ste: bool) -> torch.Tensor:
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
        if self.latent_byte is not None:
            self.latent_byte.clamp_(0.0, 255.0)
        else:
            lower, upper = self.codec.valid_bounds(self.colors_u8)
            self.residual_byte.copy_(torch.maximum(torch.minimum(self.residual_byte, upper.to(self.residual_byte.dtype)), lower.to(self.residual_byte.dtype)))

    @torch.no_grad()
    def hard_texture_bytes(self, *, height: int, width: int, chunk: int = 262_144) -> torch.Tensor:
        pieces: list[torch.Tensor] = []
        for start in range(0, self.texel_count, chunk):
            ids = torch.arange(start, min(start + chunk, self.texel_count), device=self.colors_u8.device)
            pieces.append(torch.floor(self.latent_for_ids(ids, ste=False) * 255.0 + 0.5).to(torch.uint8).cpu())
        return torch.cat(pieces).reshape(height, width, 4)


def initialize_candidates(
    targets: TexelTargets,
    initialization: RankOneInitialization,
    mixed_record: LatticeOracleRecord,
    *,
    device: torch.device | str,
) -> dict[str, ExactExperimentCandidate]:
    ids = initialization.sample_indices
    separated = SeparatedCodec()
    separated_residual = (initialization.normalized_scalar + 1.0) * 127.5
    separated_latent = separated.encode_hard(targets.base_q8[ids], separated_residual[ids]).to(torch.float32) / 255.0
    separated_weight, separated_bias = _fit_affine_raw(separated_latent, auxiliary_raw_targets(targets, ids))
    mixed = AffineLatticeCodec(mixed_record.capacity.kernel_rgb, mixed_record.capacity.t0)
    lower, upper = mixed.valid_bounds(targets.base_q8)
    mixed_residual = lower.to(torch.float32) + 0.5 * (initialization.normalized_scalar + 1.0) * (upper - lower).to(torch.float32)
    mixed_residual = torch.floor(mixed_residual + 0.5)
    mixed_latent = mixed.encode_hard(targets.base_q8[ids], mixed_residual[ids]).to(torch.float32) / 255.0
    mixed_weight, mixed_bias = _fit_affine_raw(mixed_latent, auxiliary_raw_targets(targets, ids))
    values = {
        "S-separated": ExactExperimentCandidate(
            name="S-separated", codec=separated, colors_u8=targets.base_q8,
            initial_residual=separated_residual, auxiliary_weight=separated_weight, auxiliary_bias=separated_bias,
        ),
        "U0-unconstrained": ExactExperimentCandidate(
            name="U0-unconstrained", codec=separated, colors_u8=targets.base_q8,
            initial_residual=separated_residual, auxiliary_weight=separated_weight, auxiliary_bias=separated_bias,
        ),
        "M-mixed": ExactExperimentCandidate(
            name="M-mixed", codec=mixed, colors_u8=targets.base_q8,
            initial_residual=mixed_residual, auxiliary_weight=mixed_weight, auxiliary_bias=mixed_bias,
        ),
    }
    return {name: candidate.to(device) for name, candidate in values.items()}


def learning_rates(step: int, *, stop: int = 10_000) -> tuple[float, float]:
    if not 1 <= step <= stop:
        raise ValueError("training step is outside the configured range")
    if step <= 500:
        return 0.02, 0.001
    fraction = (step - 500) / float(max(stop - 500, 1))
    blend = 0.5 * (1.0 - math.cos(math.pi * fraction))
    return 0.02 + (0.002 - 0.02) * blend, 0.001 + (0.0001 - 0.001) * blend


def orbit_camera(
    *, yaw_degrees: float, elevation_degrees: float, radius: float,
    target: Sequence[float], up: Sequence[float], vertical_fov_degrees: float,
    near: float, far: float,
) -> Camera:
    yaw, elevation = math.radians(yaw_degrees), math.radians(elevation_degrees)
    horizontal = radius * math.cos(elevation)
    target3 = tuple(float(value) for value in target)
    return Camera(
        eye=(target3[0] + horizontal * math.sin(yaw), target3[1] + radius * math.sin(elevation), target3[2] + horizontal * math.cos(yaw)),
        target=target3,
        up=tuple(float(value) for value in up),
        vertical_fov_degrees=float(vertical_fov_degrees), near=float(near), far=float(far),
    )


def display_transform(linear_hdr: torch.Tensor, exposure: float) -> torch.Tensor:
    mapped = torch.clamp(linear_hdr * float(exposure), min=0.0)
    mapped = mapped / (1.0 + mapped)
    # Clamp the inactive power branch to its sRGB breakpoint.  torch.where
    # evaluates both branches, and pow(0, 1/2.4) otherwise has an infinite
    # derivative that contaminates the selected linear branch as 0*inf.
    encoded = torch.where(
        mapped <= 0.0031308,
        mapped * 12.92,
        1.055 * torch.pow(mapped.clamp_min(0.0031308), 1.0 / 2.4) - 0.055,
    )
    return encoded.clamp(0.0, 1.0)


def subset_geometry(geometry: GBufferResult, flat_ids: torch.Tensor) -> GBufferResult:
    height, width = geometry.torch_buffers["mask"].shape
    selected: dict[str, torch.Tensor] = {}
    for name, value in geometry.torch_buffers.items():
        if value.shape[:2] == (height, width):
            selected[name] = value.reshape(height * width, *value.shape[2:])[flat_ids]
    selected["mask"] = torch.ones(flat_ids.shape[0], dtype=torch.bool, device=flat_ids.device)
    return GBufferResult(buffers={}, torch_buffers=selected, metadata={"subset": True})


def subset_material(material: MaterialBuffers, flat_ids: torch.Tensor) -> MaterialBuffers:
    def select(value: torch.Tensor | None) -> torch.Tensor | None:
        if value is None:
            return None
        return value.reshape(-1, *value.shape[2:])[flat_ids] if value.ndim >= 3 else value.reshape(-1)[flat_ids]
    return MaterialBuffers(
        base_color_linear=select(material.base_color_linear), normal_world=select(material.normal_world),
        roughness=select(material.roughness), metallic=select(material.metallic),
        normal_ts_raw=select(material.normal_ts_raw), normal_ts_unit=select(material.normal_ts_unit),
    )


def decoded_subset_material(geometry: GBufferResult, decoded: DecodedMaterial) -> MaterialBuffers:
    normal_world = tangent_normal_to_world(geometry, decoded.normal_xyz)
    return MaterialBuffers(
        base_color_linear=decoded.base_color_linear,
        normal_world=normal_world,
        roughness=decoded.roughness[..., 0], metallic=decoded.metallic[..., 0],
        normal_ts_raw=decoded.normal_xyz, normal_ts_unit=decoded.normal_xyz,
    )


def render_pair_loss(
    candidate: ExactExperimentCandidate,
    case: RenderCase,
    light: PointLight,
    selected_ids: torch.Tensor,
    *,
    height: int,
    width: int,
    minimum_roughness: float,
    display_exposure: float,
    ste: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    geometry = subset_geometry(case.geometry, selected_ids)
    reference_material = subset_material(case.reference, selected_ids)
    uv = geometry.torch_buffers["uv"]
    decoded = candidate.decoder(candidate.sample_uv(uv, height=height, width=width, ste=ste))
    predicted_material = decoded_subset_material(geometry, decoded)
    reference_hdr = shade_ggx(geometry, case.camera, light, material_override=reference_material, minimum_roughness=minimum_roughness)
    predicted_hdr = shade_ggx(geometry, case.camera, light, material_override=predicted_material, minimum_roughness=minimum_roughness)
    hdr_l1 = F.l1_loss(predicted_hdr, reference_hdr)
    display_l1 = F.l1_loss(display_transform(predicted_hdr, display_exposure), display_transform(reference_hdr, display_exposure))
    return hdr_l1, display_l1, {"predicted_hdr": predicted_hdr, "reference_hdr": reference_hdr}


def material_losses(decoded: DecodedMaterial, targets: TexelTargets, ids: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "base_color": F.l1_loss(decoded.base_color_linear, targets.base_q8[ids].to(decoded.base_color_linear.dtype) / 255.0),
        "normal": torch.mean(1.0 - torch.sum(decoded.normal_xyz * targets.normal_xyz[ids], dim=-1)),
        "roughness": F.l1_loss(decoded.roughness, targets.roughness[ids]),
        "metallic": F.l1_loss(decoded.metallic, targets.metallic[ids]),
    }


def objective(hdr: torch.Tensor, display: torch.Tensor, material: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return hdr + 0.25 * display + 0.5 * material["base_color"] + 0.5 * material["normal"] + 0.25 * material["roughness"] + 0.25 * material["metallic"]


def candidate_state_hash(candidate: ExactExperimentCandidate) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(candidate.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def checkpoint_payload(
    *, candidate: ExactExperimentCandidate, step: int, code_optimizer: torch.optim.Optimizer,
    decoder_optimizer: torch.optim.Optimizer, generator: torch.Generator,
    config_hash: str, lattice_manifest_hash: str, target_hash: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION, "checkpoint_type": CHECKPOINT_TYPE,
        "candidate": candidate.name, "codec": candidate.codec.specification(), "step": int(step),
        "candidate_state": candidate.state_dict(), "code_optimizer": code_optimizer.state_dict(),
        "decoder_optimizer": decoder_optimizer.state_dict(), "generator_state": generator.get_state(),
        "python_random_state": random.getstate(), "numpy_random_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(), "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "config_hash": config_hash, "lattice_manifest_hash": lattice_manifest_hash, "target_hash": target_hash,
    }


def validate_checkpoint(payload: Mapping[str, Any], *, candidate_name: str, config_hash: str, lattice_manifest_hash: str, target_hash: str) -> None:
    expected = {
        "schema_version": SCHEMA_VERSION, "checkpoint_type": CHECKPOINT_TYPE,
        "candidate": candidate_name, "config_hash": config_hash,
        "lattice_manifest_hash": lattice_manifest_hash, "target_hash": target_hash,
    }
    for name, value in expected.items():
        if payload.get(name) != value:
            raise ValueError(f"checkpoint {name} mismatch")


def atomic_torch_save(value: Any, path: Path, *, immutable: bool = False) -> None:
    if immutable and path.exists():
        raise FileExistsError(f"refusing to replace immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def export_candidate(
    candidate: ExactExperimentCandidate,
    *,
    output_dir: Path,
    height: int,
    width: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    encoded = candidate.hard_texture_bytes(height=height, width=width).numpy()
    texture_path = output_dir / "latent_rgba_unorm8.png"
    Image.fromarray(encoded, mode="RGBA").save(texture_path, format="PNG")
    weight, bias = candidate.decoder.combined_parameters()
    decoder_path = output_dir / "decoder_affine.npz"
    np.savez(decoder_path, weight=weight.detach().cpu().numpy(), bias=bias.detach().cpu().numpy())
    reread = np.asarray(Image.open(texture_path).convert("RGBA"), dtype=np.uint8)
    if not np.array_equal(reread, encoded):
        raise AssertionError("RGBA PNG readback changed latent bytes")
    files = {}
    for path in (texture_path, decoder_path):
        files[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1, "candidate": candidate.name, "codec": candidate.codec.specification(),
        "runtime_inputs": [texture_path.name, decoder_path.name], "source_basecolor_required": False,
        "files": files,
    }
    (output_dir / "export_manifest.json").write_bytes(stable_json_bytes(manifest))
    return manifest


def verify_runtime_export(export_dir: Path, *, chunk: int = 262_144) -> dict[str, Any]:
    """Reload an export using only its declared runtime files.

    The returned BaseColor hash is over half-up rounded RGB bytes in row-major
    order.  A caller can compare it with the target hash from the audit without
    exposing the source BaseColor texture to this runtime-only path.
    """

    manifest_path = export_dir / "export_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_inputs = ["latent_rgba_unorm8.png", "decoder_affine.npz"]
    if manifest.get("runtime_inputs") != expected_inputs:
        raise ValueError("export declares unexpected runtime inputs")
    if manifest.get("source_basecolor_required") is not False:
        raise ValueError("export must not require source BaseColor")
    for name in expected_inputs:
        actual = sha256_bytes((export_dir / name).read_bytes())
        if actual != manifest["files"].get(name):
            raise ValueError(f"export file hash mismatch: {name}")

    latent = np.asarray(Image.open(export_dir / expected_inputs[0]).convert("RGBA"), dtype=np.uint8)
    with np.load(export_dir / expected_inputs[1], allow_pickle=False) as decoder:
        weight = np.asarray(decoder["weight"], dtype=np.float32)
        bias = np.asarray(decoder["bias"], dtype=np.float32)
    if weight.shape != (7, 4) or bias.shape != (7,):
        raise ValueError("decoder must be one 4-to-7 affine transform")

    flat = latent.reshape(-1, 4)
    decoded_hash = hashlib.sha256()
    finite = True
    decoded_min = math.inf
    decoded_max = -math.inf
    for start in range(0, flat.shape[0], chunk):
        encoded = flat[start : start + chunk].astype(np.float32) / np.float32(255.0)
        decoded = encoded @ weight[:3].T + bias[:3]
        finite = finite and bool(np.isfinite(decoded).all())
        decoded_min = min(decoded_min, float(decoded.min()))
        decoded_max = max(decoded_max, float(decoded.max()))
        decoded_u8 = np.floor(np.clip(decoded, 0.0, 1.0) * np.float32(255.0) + np.float32(0.5)).astype(np.uint8)
        decoded_hash.update(decoded_u8.tobytes(order="C"))
    if not finite:
        raise ValueError("decoder produced non-finite BaseColor")

    return {
        "schema_version": 1,
        "candidate": manifest["candidate"],
        "runtime_inputs": expected_inputs,
        "source_basecolor_required": False,
        "latent_shape": list(latent.shape),
        "latent_rgba_sha256": tensor_sha256(torch.from_numpy(latent.copy())),
        "decoded_basecolor_u8_sha256": decoded_hash.hexdigest(),
        "decoded_basecolor_min": decoded_min,
        "decoded_basecolor_max": decoded_max,
        "finite": finite,
    }


def _linear_to_srgb_numpy(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, 0.0, 1.0)
    return np.where(
        clipped <= 0.0031308,
        12.92 * clipped,
        1.055 * np.power(clipped, 1.0 / 2.4) - 0.055,
    )


def _image_u8(value: np.ndarray) -> Image.Image:
    encoded = np.floor(np.clip(value, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(encoded)


def write_runtime_material_diagnostics(
    export_dir: Path,
    targets: TexelTargets,
    *,
    output_dir: Path,
    chunk: int = 262_144,
    uv_count: int = 1_000_000,
    uv_seed: int = 20260811,
) -> dict[str, Any]:
    """Create material/error atlases from runtime files plus audit targets."""

    verification = verify_runtime_export(export_dir, chunk=chunk)
    latent = np.asarray(Image.open(export_dir / "latent_rgba_unorm8.png").convert("RGBA"), dtype=np.uint8)
    with np.load(export_dir / "decoder_affine.npz", allow_pickle=False) as decoder:
        weight = np.asarray(decoder["weight"], dtype=np.float32)
        bias = np.asarray(decoder["bias"], dtype=np.float32)
    flat = latent.reshape(-1, 4)
    base = np.empty((flat.shape[0], 3), dtype=np.float32)
    normal = np.empty((flat.shape[0], 3), dtype=np.float32)
    roughness = np.empty((flat.shape[0], 1), dtype=np.float32)
    metallic = np.empty((flat.shape[0], 1), dtype=np.float32)
    for start in range(0, flat.shape[0], chunk):
        stop = min(start + chunk, flat.shape[0])
        encoded = flat[start:stop].astype(np.float32) / np.float32(255.0)
        raw = encoded @ weight.T + bias
        xy = np.tanh(raw[:, 3:5])
        z = np.sqrt(np.clip(1.0 - np.sum(xy * xy, axis=-1, keepdims=True), 0.0, 1.0))
        xyz = np.concatenate((xy, z), axis=-1)
        xyz /= np.maximum(np.linalg.norm(xyz, axis=-1, keepdims=True), 1e-8)
        base[start:stop] = raw[:, :3]
        normal[start:stop] = xyz
        roughness[start:stop] = 1.0 / (1.0 + np.exp(-raw[:, 5:6]))
        metallic[start:stop] = 1.0 / (1.0 + np.exp(-raw[:, 6:7]))

    height, width = targets.height, targets.width
    base_target = targets.base_q8.reshape(height, width, 3).numpy().astype(np.float32) / 255.0
    normal_target = targets.normal_xyz.reshape(height, width, 3).numpy()
    rough_target = targets.roughness.reshape(height, width, 1).numpy()
    metal_target = targets.metallic.reshape(height, width, 1).numpy()
    base = base.reshape(height, width, 3)
    normal = normal.reshape(height, width, 3)
    roughness = roughness.reshape(height, width, 1)
    metallic = metallic.reshape(height, width, 1)

    dot = np.clip(np.sum(normal * normal_target, axis=-1), -1.0, 1.0)
    error_maps = {
        "basecolor": np.mean(np.abs(base - base_target), axis=-1),
        "normal": np.degrees(np.arccos(dot)),
        "roughness": np.abs(roughness[..., 0] - rough_target[..., 0]),
        "metallic": np.abs(metallic[..., 0] - metal_target[..., 0]),
    }
    predicted_bytes = np.floor(np.clip(base, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    target_bytes = targets.base_q8.reshape(height, width, 3).numpy()
    exact_texels = np.all(predicted_bytes == target_bytes, axis=-1)

    latent_torch = torch.from_numpy(latent.astype(np.float32) / np.float32(255.0))
    target_torch = targets.base_q8.reshape(height, width, 3).to(torch.float32) / 255.0
    weight_torch = torch.from_numpy(weight[:3].copy())
    bias_torch = torch.from_numpy(bias[:3].copy())
    generator = torch.Generator().manual_seed(uv_seed)
    uv_maximum = 0.0
    uv_absolute_sum = 0.0
    for start in range(0, uv_count, 100_000):
        count = min(100_000, uv_count - start)
        uv = torch.rand((count, 2), generator=generator)
        decoded_uv = F.linear(_bilinear_cpu(latent_torch, uv), weight_torch, bias_torch)
        target_uv = _bilinear_cpu(target_torch, uv)
        difference = (decoded_uv - target_uv).abs()
        uv_maximum = max(uv_maximum, float(difference.max()))
        uv_absolute_sum += float(difference.sum())
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "basecolor": output_dir / "basecolor_decoded.png",
        "normal": output_dir / "normal_decoded.png",
        "roughness": output_dir / "roughness_decoded.png",
        "metallic": output_dir / "metallic_decoded.png",
        "errors": output_dir / "material_error_atlas.png",
    }
    _image_u8(_linear_to_srgb_numpy(base)).save(files["basecolor"])
    _image_u8(normal * 0.5 + 0.5).save(files["normal"])
    _image_u8(roughness[..., 0]).save(files["roughness"])
    _image_u8(metallic[..., 0]).save(files["metallic"])

    scales = {"basecolor": 20.0, "normal": 1.0 / 45.0, "roughness": 4.0, "metallic": 4.0}
    panels = []
    for name in ("basecolor", "normal", "roughness", "metallic"):
        panel = _image_u8(error_maps[name] * scales[name]).resize((512, 512), Image.Resampling.BILINEAR).convert("RGB")
        panels.append(panel)
    atlas = Image.new("RGB", (512 * len(panels), 512), "black")
    for index, panel in enumerate(panels):
        atlas.paste(panel, (index * 512, 0))
    atlas.save(files["errors"])
    return {
        "runtime_verification": verification,
        "files": {name: path.name for name, path in files.items()},
        "error_visualization_scales": scales,
        "metrics": {
            "base_color_q8_mae": float(error_maps["basecolor"].mean()),
            "base_color_byte_exact": bool(exact_texels.all()),
            "base_color_byte_exact_fraction": float(exact_texels.mean()),
            "linear_q8_uv_max_abs": uv_maximum,
            "linear_q8_uv_mae": uv_absolute_sum / (uv_count * 3),
            "linear_q8_uv_probe_count": uv_count,
            "normal_angle_degrees": float(error_maps["normal"].mean()),
            "roughness_mae": float(error_maps["roughness"].mean()),
            "metallic_mae": float(error_maps["metallic"].mean()),
        },
    }


@torch.no_grad()
def candidate_forward_parity(
    candidate: ExactExperimentCandidate,
    *,
    sample_count: int = 100_000,
    seed: int = 20260811,
) -> dict[str, Any]:
    generator = torch.Generator(device=candidate.colors_u8.device).manual_seed(seed)
    ids = torch.randint(
        0,
        candidate.texel_count,
        (min(sample_count, candidate.texel_count),),
        generator=generator,
        device=candidate.colors_u8.device,
    )
    fake_latent = candidate.latent_for_ids(ids, ste=True)
    hard_latent = candidate.latent_for_ids(ids, ste=False)
    fake_raw = candidate.decoder.raw_affine(fake_latent)
    hard_raw = candidate.decoder.raw_affine(hard_latent)
    latent_error = float((fake_latent - hard_latent).abs().max())
    decoder_error = float((fake_raw - hard_raw).abs().max())
    return {
        "sample_count": int(ids.numel()),
        "latent_max_abs": latent_error,
        "decoder_max_abs": decoder_error,
        "passed": latent_error == 0.0 and decoder_error == 0.0,
    }


@torch.no_grad()
def evaluate_texels(
    candidate: ExactExperimentCandidate,
    targets: TexelTargets,
    *,
    chunk: int = 262_144,
) -> dict[str, float]:
    device_targets = targets.to(candidate.colors_u8.device)
    base_abs = base_sq = normal_sum = normal_angle_sum = rough_abs = metal_abs = 0.0
    byte_exact_texels = 0
    chroma_source_sum = chroma_prediction_sum = 0.0
    chromatic_count = chroma_loss_count = 0
    opponent = candidate.colors_u8.new_tensor(
        [[1.0 / math.sqrt(2.0), -1.0 / math.sqrt(2.0), 0.0],
         [1.0 / math.sqrt(6.0), 1.0 / math.sqrt(6.0), -2.0 / math.sqrt(6.0)]],
        dtype=torch.float32,
    )
    for start in range(0, candidate.texel_count, chunk):
        stop = min(start + chunk, candidate.texel_count)
        ids = torch.arange(start, stop, device=candidate.colors_u8.device)
        decoded = candidate.decoder(candidate.latent_for_ids(ids, ste=False))
        base_target = device_targets.base_q8[ids].to(torch.float32) / 255.0
        difference = decoded.base_color_linear - base_target
        base_abs += float(difference.abs().sum())
        base_sq += float(difference.square().sum())
        normal_dot = torch.sum(decoded.normal_xyz * device_targets.normal_xyz[ids], dim=-1).clamp(-1.0, 1.0)
        normal_sum += float((1.0 - normal_dot).sum())
        normal_angle_sum += float(torch.rad2deg(torch.acos(normal_dot)).sum())
        rough_abs += float((decoded.roughness - device_targets.roughness[ids]).abs().sum())
        metal_abs += float((decoded.metallic - device_targets.metallic[ids]).abs().sum())
        source_chroma = torch.linalg.vector_norm(base_target @ opponent.T, dim=-1)
        prediction_chroma = torch.linalg.vector_norm(decoded.base_color_linear @ opponent.T, dim=-1)
        chroma_source_sum += float(source_chroma.sum())
        chroma_prediction_sum += float(prediction_chroma.sum())
        chromatic = source_chroma >= 0.02
        chromatic_count += int(chromatic.sum())
        chroma_loss_count += int((chromatic & (prediction_chroma < 0.75 * source_chroma)).sum())
        decoded_bytes = torch.floor(decoded.base_color_linear.clamp(0.0, 1.0) * 255.0 + 0.5).to(torch.uint8)
        byte_exact_texels += int(torch.all(decoded_bytes == device_targets.base_q8[ids], dim=-1).sum())
    count = candidate.texel_count
    return {
        "base_color_q8_mae": base_abs / (count * 3),
        "base_color_q8_rmse": math.sqrt(base_sq / (count * 3)),
        "base_color_byte_exact": byte_exact_texels == count,
        "base_color_byte_exact_fraction": byte_exact_texels / count,
        "normal_cosine_error": normal_sum / count,
        "normal_angle_degrees": normal_angle_sum / count,
        "roughness_mae": rough_abs / count,
        "metallic_mae": metal_abs / count,
        "generic_chroma_retention": chroma_prediction_sum / max(chroma_source_sum, 1e-12),
        "chromatic_texel_fraction_losing_25pct": chroma_loss_count / max(chromatic_count, 1),
    }


def _ssim_global(reference: torch.Tensor, prediction: torch.Tensor) -> float:
    x, y = reference.reshape(-1), prediction.reshape(-1)
    c1, c2 = 0.01**2, 0.03**2
    mx, my = x.mean(), y.mean()
    vx, vy = ((x - mx) ** 2).mean(), ((y - my) ** 2).mean()
    covariance = ((x - mx) * (y - my)).mean()
    value = ((2 * mx * my + c1) * (2 * covariance + c2)) / ((mx.square() + my.square() + c1) * (vx + vy + c2))
    return float(value)


@torch.no_grad()
def evaluate_render_pool(
    candidate: ExactExperimentCandidate,
    cases: Sequence[RenderCase],
    lights: Sequence[PointLight],
    *,
    height: int,
    width: int,
    minimum_roughness: float,
    display_exposure: float,
    case_indices: Sequence[int] | None = None,
    save_images: Path | None = None,
) -> dict[str, Any]:
    selected_cases = list(range(len(cases))) if case_indices is None else list(case_indices)
    pair_metrics: list[dict[str, Any]] = []
    for case_index in selected_cases:
        case = cases[case_index]
        ids = case.valid_flat_indices
        for light_index, light in enumerate(lights):
            hdr_l1, _, products = render_pair_loss(
                candidate, case, light, ids,
                height=height, width=width, minimum_roughness=minimum_roughness,
                display_exposure=display_exposure, ste=False,
            )
            reference_display = display_transform(products["reference_hdr"], display_exposure)
            prediction_display = display_transform(products["predicted_hdr"], display_exposure)
            pair_metrics.append({
                "camera_index": case_index, "camera": case.name, "light_index": light_index,
                "hdr_mae": float(hdr_l1), "display_ssim": _ssim_global(reference_display, prediction_display),
            })
            if save_images is not None and light_index == 0:
                save_images.mkdir(parents=True, exist_ok=True)
                mask = case.geometry.torch_buffers["mask"]
                canvas = torch.zeros((*mask.shape, 3), dtype=prediction_display.dtype, device=prediction_display.device)
                canvas.reshape(-1, 3)[ids] = prediction_display
                array = torch.floor(canvas.clamp(0, 1) * 255.0 + 0.5).to(torch.uint8).cpu().numpy()
                Image.fromarray(array, mode="RGB").save(save_images / f"camera_{case_index:02d}.png")
    hdr_values = np.asarray([item["hdr_mae"] for item in pair_metrics], dtype=np.float64)
    ssim_values = np.asarray([item["display_ssim"] for item in pair_metrics], dtype=np.float64)
    return {
        "pair_count": len(pair_metrics), "mean_hdr_mae": float(hdr_values.mean()),
        "worst_hdr_mae": float(hdr_values.max()), "mean_display_ssim": float(ssim_values.mean()),
        "worst_display_ssim": float(ssim_values.min()), "pairs": pair_metrics,
    }


def train_one_step(
    candidate: ExactExperimentCandidate,
    targets: TexelTargets,
    cases: Sequence[RenderCase],
    lights: Sequence[PointLight],
    *,
    generator: torch.Generator,
    texel_batch_size: int,
    screen_batch_size: int,
    minimum_roughness: float,
    display_exposure: float,
    code_optimizer: torch.optim.Optimizer,
    decoder_optimizer: torch.optim.Optimizer,
    step: int,
    stop: int,
) -> dict[str, float | int]:
    device = candidate.colors_u8.device
    texel_ids = torch.randint(0, candidate.texel_count, (texel_batch_size,), generator=generator, device=device)
    camera_index = int(torch.randint(0, len(cases), (1,), generator=generator, device=device))
    light_index = int(torch.randint(0, len(lights), (1,), generator=generator, device=device))
    case = cases[camera_index]
    positions = torch.randint(0, case.valid_flat_indices.shape[0], (screen_batch_size,), generator=generator, device=device)
    screen_ids = case.valid_flat_indices[positions]
    decoded_texels = candidate.decoder(candidate.latent_for_ids(texel_ids, ste=True))
    material = material_losses(decoded_texels, targets, texel_ids)
    hdr, display, _ = render_pair_loss(
        candidate, case, lights[light_index], screen_ids,
        height=targets.height, width=targets.width, minimum_roughness=minimum_roughness,
        display_exposure=display_exposure, ste=True,
    )
    total = objective(hdr, display, material)
    if not torch.isfinite(total):
        raise FloatingPointError(f"non-finite objective at step {step}")
    code_lr, decoder_lr = learning_rates(step, stop=stop)
    for group in code_optimizer.param_groups:
        group["lr"] = code_lr
    for group in decoder_optimizer.param_groups:
        group["lr"] = decoder_lr
    code_optimizer.zero_grad(set_to_none=True)
    decoder_optimizer.zero_grad(set_to_none=True)
    total.backward()
    non_finite_gradients = [
        name
        for name, parameter in candidate.named_parameters()
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all()
    ]
    if non_finite_gradients:
        raise FloatingPointError(
            f"non-finite gradients at step {step}: {','.join(non_finite_gradients)}"
        )
    code_optimizer.step()
    decoder_optimizer.step()
    candidate.project_codes_()
    return {
        "step": step, "camera_index": camera_index, "light_index": light_index,
        "total": float(total.detach()), "hdr": float(hdr.detach()), "display": float(display.detach()),
        **{name: float(value.detach()) for name, value in material.items()},
        "code_lr": code_lr, "decoder_lr": decoder_lr,
    }


@torch.no_grad()
def strict_candidate_certificate(
    candidate: ExactExperimentCandidate,
    targets: TexelTargets,
    *,
    uv_count: int,
    seed: int,
    batch_size: int = 100_000,
) -> dict[str, Any]:
    if not candidate.strict:
        raise ValueError("strict certificate only applies to S/M")
    residual = candidate.residual_byte.detach().cpu()
    texel = codec_certificate(candidate.codec, targets.base_q8, residual)
    texture = candidate.hard_texture_bytes(height=targets.height, width=targets.width).to(torch.float32) / 255.0
    target_texture = targets.base_q8.reshape(targets.height, targets.width, 3).to(torch.float32) / 255.0
    generator = torch.Generator().manual_seed(seed)
    maximum = 0.0
    for start in range(0, uv_count, batch_size):
        count = min(batch_size, uv_count - start)
        uv = torch.rand((count, 2), generator=generator)
        sampled_code = _bilinear_cpu(texture, uv)
        decoded = candidate.codec.decode_basecolor(sampled_code)
        direct = _bilinear_cpu(target_texture, uv)
        maximum = max(maximum, float((decoded - direct).abs().max()))
    texel.update({"uv_probe_count": uv_count, "uv_max_abs": maximum, "uv_passed": maximum <= 2e-6})
    texel["passed"] = bool(texel["byte_exact"] and texel["max_abs"] <= 1e-7 and texel["uv_passed"])
    return texel


def _bilinear_cpu(texture: torch.Tensor, uv: torch.Tensor) -> torch.Tensor:
    height, width = texture.shape[:2]
    x, y = uv[:, 0] * width - 0.5, uv[:, 1] * height - 0.5
    x0f, y0f = torch.floor(x), torch.floor(y)
    wx, wy = x - x0f, y - y0f
    x0, y0 = x0f.to(torch.int64).remainder(width), y0f.to(torch.int64).remainder(height)
    x1, y1 = (x0 + 1).remainder(width), (y0 + 1).remainder(height)
    return (
        texture[y0, x0] * ((1 - wx) * (1 - wy))[:, None]
        + texture[y0, x1] * (wx * (1 - wy))[:, None]
        + texture[y1, x0] * ((1 - wx) * wy)[:, None]
        + texture[y1, x1] * (wx * wy)[:, None]
    )
