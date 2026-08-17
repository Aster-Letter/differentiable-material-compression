"""Minimal RGBA-latent material reconstruction for SciFiHelmet Core-4."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any, Iterable

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
import yaml

from cg_frontier.assets.preprocess import srgb_to_linear


@contextmanager
def keep_system_awake() -> Iterable[None]:
    """Prevent automatic Windows sleep while a bounded training command is alive."""

    if os.name != "nt":
        yield
        return
    es_continuous = 0x80000000
    es_system_required = 0x00000001
    result = ctypes.windll.kernel32.SetThreadExecutionState(es_continuous | es_system_required)
    if result == 0:
        raise OSError("SetThreadExecutionState failed")
    try:
        yield
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(es_continuous)


@dataclass(frozen=True)
class Core4Targets:
    """Flattened material truth; normal is a normalized tangent-space XYZ vector."""

    base_color_linear: torch.Tensor
    normal_xyz: torch.Tensor
    roughness: torch.Tensor
    metallic: torch.Tensor
    height: int
    width: int

    @property
    def texel_count(self) -> int:
        return self.height * self.width

    def select(self, indices: torch.Tensor) -> "Core4Targets":
        return Core4Targets(
            base_color_linear=self.base_color_linear[indices],
            normal_xyz=self.normal_xyz[indices],
            roughness=self.roughness[indices],
            metallic=self.metallic[indices],
            height=1,
            width=indices.numel(),
        )


@dataclass(frozen=True)
class DecodedMaterial:
    """Postprocessed seven-channel decoder output in training-space semantics."""

    base_color_linear: torch.Tensor
    normal_xy: torch.Tensor
    normal_xyz: torch.Tensor
    roughness: torch.Tensor
    metallic: torch.Tensor


class _LinearNonlinearResidual(nn.Module):
    """Seven-channel raw-domain linear shortcut plus a bias-free MLP residual."""

    def __init__(self, residual_width: int = 8) -> None:
        super().__init__()
        self.direct = nn.Linear(4, 7)
        self.residual_hidden = nn.Linear(4, residual_width)
        self.residual_output = nn.Linear(residual_width, 7, bias=False)

    def forward(self, latent_rgba: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.residual_hidden(latent_rgba))
        # Fuse b0 into the nonlinear F.linear call. With W0=0 this preserves the
        # source tiny-MLP operation order exactly while still representing
        # W0 z + b0 + W2 ReLU(W1 z + b1) in the seven-channel raw domain.
        nonlinear = F.linear(hidden, self.residual_output.weight, self.direct.bias)
        direct = F.linear(latent_rgba, self.direct.weight, bias=None)
        return nonlinear + direct


class _WidenedTinyMLP12(nn.Module):
    """A 4→12→7 single-hidden-layer MLP with exact 8+4 block evaluation."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(4, 12)
        self.output = nn.Linear(12, 7)

    def forward(self, latent_rgba: torch.Tensor) -> torch.Tensor:
        # Evaluating the old 8 units with their original GEMM shapes avoids the
        # rounding drift caused by changing a fused 4→8 GEMM into 4→12.
        core_hidden = F.relu(
            F.linear(latent_rgba, self.hidden.weight[:8], self.hidden.bias[:8])
        )
        extra_hidden = F.relu(
            F.linear(latent_rgba, self.hidden.weight[8:], self.hidden.bias[8:])
        )
        core_output = F.linear(
            core_hidden, self.output.weight[:, :8], self.output.bias
        )
        extra_output = F.linear(
            extra_hidden, self.output.weight[:, 8:], bias=None
        )
        return core_output + extra_output


class MaterialDecoder(nn.Module):
    """A per-texel 4-channel latent decoder with seven raw semantic outputs.

    Raw channels are ordered BaseColor RGB, tangent normal XY, roughness, and
    metallic.  ``decode_material`` applies the channel postprocessing once.
    """

    def __init__(self, kind: str) -> None:
        super().__init__()
        if kind == "linear":
            self.network = nn.Linear(4, 7)
        elif kind == "tiny_mlp":
            self.network = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 7))
        elif kind == "tiny_mlp_12":
            self.network = _WidenedTinyMLP12()
        elif kind == "linear_nonlinear_residual":
            self.network = _LinearNonlinearResidual(residual_width=8)
        else:
            raise ValueError(f"unsupported decoder kind: {kind}")
        self.kind = kind

    def forward(self, latent_rgba: torch.Tensor) -> torch.Tensor:
        return self.network(latent_rgba)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def macs_per_pixel(self) -> int:
        return {
            "linear": 28,
            "tiny_mlp": 88,
            "tiny_mlp_12": 132,
            "linear_nonlinear_residual": 116,
        }[self.kind]


def initialize_decoder_from_tiny(
    source: MaterialDecoder, target_kind: str
) -> MaterialDecoder:
    """Create an exact-function-preserving ablation decoder from a 4→8→7 source."""

    if source.kind != "tiny_mlp":
        raise ValueError("function-preserving initialization requires a tiny_mlp source")
    source_hidden = source.network[0]
    source_output = source.network[2]
    assert isinstance(source_hidden, nn.Linear) and isinstance(source_output, nn.Linear)
    first_parameter = next(source.parameters())
    target = MaterialDecoder(target_kind).to(
        device=first_parameter.device, dtype=first_parameter.dtype
    )
    with torch.no_grad():
        if target_kind == "tiny_mlp":
            target.load_state_dict(source.state_dict())
        elif target_kind == "tiny_mlp_12":
            target_network = target.network
            assert isinstance(target_network, _WidenedTinyMLP12)
            target_hidden = target_network.hidden
            target_output = target_network.output
            target_hidden.weight.zero_()
            target_hidden.bias.zero_()
            target_output.weight.zero_()
            target_hidden.weight[:8].copy_(source_hidden.weight)
            target_hidden.bias[:8].copy_(source_hidden.bias)
            target_output.weight[:, :8].copy_(source_output.weight)
            target_output.bias.copy_(source_output.bias)
        elif target_kind == "linear_nonlinear_residual":
            target_network = target.network
            assert isinstance(target_network, _LinearNonlinearResidual)
            target_network.direct.weight.zero_()
            target_network.direct.bias.copy_(source_output.bias)
            target_network.residual_hidden.weight.copy_(source_hidden.weight)
            target_network.residual_hidden.bias.copy_(source_hidden.bias)
            target_network.residual_output.weight.copy_(source_output.weight)
        else:
            raise ValueError(f"unsupported tiny-MLP ablation target: {target_kind}")
    return target


def reconstruct_normal(normal_xy: torch.Tensor, epsilon: float = 1.0e-8) -> torch.Tensor:
    """Project XY into the positive unit disk, reconstruct +Z, and normalize."""

    radius = torch.linalg.vector_norm(normal_xy, dim=-1, keepdim=True)
    normal_xy = normal_xy / torch.clamp(radius / (1.0 - 1.0e-6), min=1.0)
    z = torch.sqrt(torch.clamp(1.0 - torch.sum(normal_xy.square(), dim=-1, keepdim=True), min=epsilon))
    return F.normalize(torch.cat((normal_xy, z), dim=-1), dim=-1, eps=epsilon)


def decode_material(decoder: MaterialDecoder, latent_rgba: torch.Tensor) -> DecodedMaterial:
    """Decode RGBA into linear RGB, +Z tangent normal, and linear PBR scalars."""

    raw = decoder(latent_rgba)
    base_color = torch.sigmoid(raw[..., 0:3])
    normal_xy = torch.tanh(raw[..., 3:5])
    normal_xyz = reconstruct_normal(normal_xy)
    roughness = torch.sigmoid(raw[..., 5:6])
    metallic = torch.sigmoid(raw[..., 6:7])
    return DecodedMaterial(base_color, normal_xy, normal_xyz, roughness, metallic)


def load_core4_targets(core4_dir: Path | str, device: torch.device | str = "cpu") -> Core4Targets:
    """Load Core-4 truth with one BaseColor sRGB decode and unchanged normal +Y."""

    root = Path(core4_dir)
    base_u8 = np.asarray(Image.open(root / "base_color.png").convert("RGB"), dtype=np.uint8)
    normal_u8 = np.asarray(Image.open(root / "normal.png").convert("RGB"), dtype=np.uint8)
    rough_u8 = np.asarray(Image.open(root / "roughness.png").convert("L"), dtype=np.uint8)
    metal_u8 = np.asarray(Image.open(root / "metallic.png").convert("L"), dtype=np.uint8)
    height, width = base_u8.shape[:2]
    if normal_u8.shape[:2] != (height, width) or rough_u8.shape != (height, width) or metal_u8.shape != (height, width):
        raise ValueError("Core-4 texture dimensions do not match")

    base = srgb_to_linear(base_u8.astype(np.float32) / 255.0).astype(np.float32)
    normal = normal_u8.astype(np.float32) / 255.0 * 2.0 - 1.0
    # Normalize source texels for the cosine target; do not flip tangent-space Y.
    normal /= np.maximum(np.linalg.norm(normal, axis=-1, keepdims=True), 1.0e-8)
    rough = rough_u8.astype(np.float32)[..., None] / 255.0
    metal = metal_u8.astype(np.float32)[..., None] / 255.0
    target_device = torch.device(device)
    return Core4Targets(
        base_color_linear=torch.from_numpy(base.reshape(-1, 3)).to(target_device),
        normal_xyz=torch.from_numpy(normal.reshape(-1, 3)).to(target_device),
        roughness=torch.from_numpy(rough.reshape(-1, 1)).to(target_device),
        metallic=torch.from_numpy(metal.reshape(-1, 1)).to(target_device),
        height=height,
        width=width,
    )


def material_loss(
    prediction: DecodedMaterial,
    target: Core4Targets,
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the global material objective without tail or spatial reweighting."""

    terms = {
        "base_color_l1": F.l1_loss(prediction.base_color_linear, target.base_color_linear),
        "normal_cosine": torch.mean(1.0 - torch.sum(prediction.normal_xyz * target.normal_xyz, dim=-1)),
        "roughness_l1": F.l1_loss(prediction.roughness, target.roughness),
        "metallic_l1": F.l1_loss(prediction.metallic, target.metallic),
    }
    total = sum(terms[name] * float(weights[name]) for name in terms)
    return total, terms


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _iter_chunks(count: int, chunk_size: int) -> Iterable[tuple[int, int]]:
    for start in range(0, count, chunk_size):
        yield start, min(start + chunk_size, count)


@torch.no_grad()
def evaluate_full(
    latent: nn.Embedding,
    decoder: MaterialDecoder,
    targets: Core4Targets,
    chunk_size: int,
) -> tuple[dict[str, float], np.ndarray]:
    """Evaluate every texel and preserve the fixed seven-channel export order."""

    decoder.eval()
    predictions: list[np.ndarray] = []
    sums = {"base_abs": 0.0, "base_sq": 0.0, "angle": 0.0, "rough_abs": 0.0, "metal_abs": 0.0}
    angles: list[np.ndarray] = []
    for start, end in _iter_chunks(targets.texel_count, chunk_size):
        ids = torch.arange(start, end, device=latent.weight.device)
        decoded = decode_material(decoder, torch.sigmoid(latent(ids)))
        truth = targets.select(ids)
        base_diff = decoded.base_color_linear - truth.base_color_linear
        dot = torch.clamp(torch.sum(decoded.normal_xyz * truth.normal_xyz, dim=-1), -1.0, 1.0)
        angle = torch.rad2deg(torch.acos(dot))
        sums["base_abs"] += torch.sum(torch.abs(base_diff)).item()
        sums["base_sq"] += torch.sum(base_diff.square()).item()
        sums["angle"] += torch.sum(angle).item()
        sums["rough_abs"] += torch.sum(torch.abs(decoded.roughness - truth.roughness)).item()
        sums["metal_abs"] += torch.sum(torch.abs(decoded.metallic - truth.metallic)).item()
        angles.append(angle.cpu().numpy())
        # Export order is the deployment contract consumed by later analyses:
        # linear BaseColor RGB, tangent XY, linear roughness, linear metallic.
        predictions.append(
            torch.cat(
                (decoded.base_color_linear, decoded.normal_xy, decoded.roughness, decoded.metallic), dim=-1
            ).cpu().numpy().astype(np.float32)
        )

    texels = targets.texel_count
    base_values = texels * 3
    mse = sums["base_sq"] / base_values
    all_angles = np.concatenate(angles)
    metrics = {
        "base_color_linear_mae": sums["base_abs"] / base_values,
        "base_color_linear_psnr_db": float("inf") if mse == 0.0 else -10.0 * math.log10(mse),
        "normal_mean_degrees": sums["angle"] / texels,
        "normal_p95_degrees": float(np.percentile(all_angles, 95.0)),
        "roughness_mae": sums["rough_abs"] / texels,
        "metallic_mae": sums["metal_abs"] / texels,
    }
    return metrics, np.concatenate(predictions, axis=0).reshape(targets.height, targets.width, 7)


def _atomic_torch_save(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False, allow_unicode=True), encoding="utf-8")


def train_material_model(
    *,
    kind: str,
    targets: Core4Targets,
    output_dir: Path,
    seed: int,
    steps: int,
    batch_size: int,
    latent_learning_rate: float,
    decoder_learning_rate: float,
    weights: dict[str, float],
    log_interval: int,
    checkpoint_interval: int,
    evaluation_chunk_size: int,
    max_minutes: float | None = None,
    resume: bool = False,
) -> dict[str, Any]:
    """Train one bounded material model and export a reloadable result."""

    output_dir.mkdir(parents=True, exist_ok=True)
    _set_seed(seed)
    device = targets.base_color_linear.device
    latent = nn.Embedding(targets.texel_count, 4, sparse=True, device=device)
    decoder = MaterialDecoder(kind).to(device)
    nn.init.normal_(latent.weight, mean=0.0, std=0.05)
    latent_optimizer = torch.optim.SparseAdam(latent.parameters(), lr=latent_learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=decoder_learning_rate)
    start_step = 0
    checkpoint_path = output_dir / "checkpoint.pt"
    if resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        latent.load_state_dict(checkpoint["latent"])
        decoder.load_state_dict(checkpoint["decoder"])
        latent_optimizer.load_state_dict(checkpoint["latent_optimizer"])
        decoder_optimizer.load_state_dict(checkpoint["decoder_optimizer"])
        start_step = int(checkpoint["step"])

    log_path = output_dir / "train.jsonl"
    started = time.monotonic()
    last_terms: dict[str, float] = {}
    for step in range(start_step + 1, steps + 1):
        decoder.train()
        ids = torch.randint(0, targets.texel_count, (batch_size,), device=device)
        prediction = decode_material(decoder, torch.sigmoid(latent(ids)))
        total, terms = material_loss(prediction, targets.select(ids), weights)
        latent_optimizer.zero_grad(set_to_none=True)
        decoder_optimizer.zero_grad(set_to_none=True)
        total.backward()
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite loss at step {step}")
        latent_optimizer.step()
        decoder_optimizer.step()
        last_terms = {name: float(value.detach().cpu()) for name, value in terms.items()}

        if step == 1 or step % log_interval == 0:
            elapsed = time.monotonic() - started
            record = {
                "step": step,
                "elapsed_seconds": round(elapsed, 3),
                "total": float(total.detach().cpu()),
                **last_terms,
            }
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            print(record, flush=True)

        timed_out = max_minutes is not None and (time.monotonic() - started) >= max_minutes * 60.0
        if step % checkpoint_interval == 0 or step == steps or timed_out:
            _atomic_torch_save(
                {
                    "schema_version": 1,
                    "kind": kind,
                    "step": step,
                    "seed": seed,
                    "latent": latent.state_dict(),
                    "decoder": decoder.state_dict(),
                    "latent_optimizer": latent_optimizer.state_dict(),
                    "decoder_optimizer": decoder_optimizer.state_dict(),
                },
                checkpoint_path,
            )
        if timed_out:
            break

    final_step = step
    metrics, reconstruction = evaluate_full(latent, decoder, targets, evaluation_chunk_size)
    latent_path = output_dir / "latent_float.npy"
    reconstruction_path = output_dir / "reconstruction_float.npy"
    decoder_path = output_dir / "decoder_weights.npz"
    np.save(latent_path, torch.sigmoid(latent.weight).detach().cpu().numpy().reshape(targets.height, targets.width, 4))
    np.save(reconstruction_path, reconstruction)
    arrays = {name: value.detach().cpu().numpy() for name, value in decoder.state_dict().items()}
    np.savez(decoder_path, **arrays)
    result = {
        "schema_version": 1,
        "model": {
            "kind": kind,
            "latent_channels": 4,
            "output_channels": 7,
            "decoder_parameters": decoder.parameter_count,
            "final_step": final_step,
            "seed": seed,
        },
        "semantics": [
            "base_color_linear_r",
            "base_color_linear_g",
            "base_color_linear_b",
            "normal_tangent_x",
            "normal_tangent_y",
            "roughness_linear",
            "metallic_linear",
        ],
        "metrics": metrics,
        "files": {
            "latent_float.npy": _sha256(latent_path),
            "decoder_weights.npz": _sha256(decoder_path),
            "reconstruction_float.npy": _sha256(reconstruction_path),
        },
    }
    _write_yaml(output_dir / "export.yaml", result)
    _write_yaml(output_dir / "metrics.yaml", metrics)
    return result


def reload_export(output_dir: Path | str, device: torch.device | str = "cpu") -> tuple[np.ndarray, MaterialDecoder]:
    root = Path(output_dir)
    manifest = yaml.safe_load((root / "export.yaml").read_text(encoding="utf-8"))
    latent = np.load(root / "latent_float.npy")
    decoder = MaterialDecoder(manifest["model"]["kind"]).to(device)
    arrays = np.load(root / "decoder_weights.npz")
    state = {name: torch.from_numpy(arrays[name]).to(device) for name in arrays.files}
    decoder.load_state_dict(state)
    decoder.eval()
    return latent, decoder
