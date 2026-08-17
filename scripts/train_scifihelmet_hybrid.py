"""Train bounded C1/C2 SciFiHelmet Hybrid auxiliary representations."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cg_frontier.assets.gltf_mesh import load_gltf_mesh  # noqa: E402
from cg_frontier.compression.artifact_analysis import (  # noqa: E402
    deterministic_json,
    deterministic_tile_partitions,
    material_error_maps,
    metallic_boundary_mask,
    sha256_file,
)
from cg_frontier.compression.hybrid import (  # noqa: E402
    AuxMaterialDecoder,
    decode_auxiliary,
    export_hybrid_textures,
    render_hybrid_material,
)
from cg_frontier.compression.material import Core4Targets, load_core4_targets, material_loss  # noqa: E402
from cg_frontier.compression.render_loss import (  # noqa: E402
    bilinear_sample_top_down_wrap,
    fake_quantize_unorm8,
    hard_quantize_unorm8,
    latent_float_to_logits,
    load_latent_unorm8_png,
    masked_render_metrics,
    pbr_render_loss,
)
from cg_frontier.compression.repair import hard_example_indices  # noqa: E402
from cg_frontier.render.gbuffer import load_core4_textures  # noqa: E402
from train_scifihelmet_repair import (  # noqa: E402
    _atomic_save,
    _case_specs,
    _decoder_from_npz,
    _decode_mapping as _baseline_decode_mapping,
    _material_report,
    _prepare_cases,
    _reference_mapping,
    _render_summary,
    _repo_path,
)


def _deep_update(target: dict[str, Any], source: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in source.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
    return target


def _load_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Hybrid training config must be a mapping")
    if "extends" not in raw:
        return dict(raw)
    base = yaml.safe_load(_repo_path(raw["extends"], "extends").read_text(encoding="utf-8"))
    if not isinstance(base, dict):
        raise ValueError("Hybrid base config must be a mapping")
    return _deep_update(base, {key: value for key, value in raw.items() if key != "extends"})


def _load_initialization(
    texture_a_path: Path,
    texture_b_path: Path,
    decoder_path: Path,
    channels: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, AuxMaterialDecoder]:
    a = np.asarray(Image.open(texture_a_path).convert("RGBA"), dtype=np.uint8)
    b_image = Image.open(texture_b_path)
    b = np.asarray(b_image.convert("L" if channels == 2 else "LA"), dtype=np.uint8)
    if b.ndim == 2:
        b = b[..., None]
    direct = torch.from_numpy(a[..., :3].copy()).to(device=device, dtype=torch.float32) / 255.0
    auxiliary = torch.from_numpy(np.concatenate((a[..., 3:4], b), axis=-1).copy()).to(
        device=device, dtype=torch.float32
    ) / 255.0
    decoder = AuxMaterialDecoder(channels).to(device)
    with np.load(decoder_path, allow_pickle=False) as stored:
        decoder.load_state_dict(
            {name: torch.from_numpy(np.asarray(stored[name])).to(device) for name in stored.files}
        )
    return direct, auxiliary, decoder


def _reference_textures(targets: Core4Targets) -> dict[str, torch.Tensor]:
    return {
        "base": targets.base_color_linear.reshape(targets.height, targets.width, 3),
        "normal": targets.normal_xyz.reshape(targets.height, targets.width, 3),
        "roughness": targets.roughness.reshape(targets.height, targets.width, 1),
        "metallic": targets.metallic.reshape(targets.height, targets.width, 1),
    }


def _sample_reference(textures: Mapping[str, torch.Tensor], uv: torch.Tensor) -> Core4Targets:
    base = bilinear_sample_top_down_wrap(textures["base"], uv)
    normal = F.normalize(bilinear_sample_top_down_wrap(textures["normal"], uv), dim=-1, eps=1.0e-8)
    roughness = bilinear_sample_top_down_wrap(textures["roughness"], uv)
    metallic = bilinear_sample_top_down_wrap(textures["metallic"], uv)
    return Core4Targets(base, normal, roughness, metallic, height=1, width=base.shape[0])


@torch.no_grad()
def _decode_mapping(
    direct: torch.Tensor,
    auxiliary: torch.Tensor,
    decoder: AuxMaterialDecoder,
    chunk_size: int,
) -> dict[str, np.ndarray]:
    height, width = auxiliary.shape[:2]
    flat_aux, flat_base = auxiliary.reshape(-1, auxiliary.shape[-1]), direct.reshape(-1, 3)
    pieces: dict[str, list[np.ndarray]] = {name: [] for name in ("base_color_linear", "normal_xyz", "roughness_linear", "metallic_linear")}
    for start in range(0, flat_aux.shape[0], chunk_size):
        decoded = decode_auxiliary(decoder, flat_aux[start : start + chunk_size], flat_base[start : start + chunk_size])
        pieces["base_color_linear"].append(decoded.base_color_linear.cpu().numpy())
        pieces["normal_xyz"].append(decoded.normal_xyz.cpu().numpy())
        pieces["roughness_linear"].append(decoded.roughness[..., 0].cpu().numpy())
        pieces["metallic_linear"].append(decoded.metallic[..., 0].cpu().numpy())
    return {
        "base_color_linear": np.concatenate(pieces["base_color_linear"]).reshape(height, width, 3),
        "normal_xyz": np.concatenate(pieces["normal_xyz"]).reshape(height, width, 3),
        "roughness_linear": np.concatenate(pieces["roughness_linear"]).reshape(height, width),
        "metallic_linear": np.concatenate(pieces["metallic_linear"]).reshape(height, width),
    }


@torch.no_grad()
def _hybrid_render_summary(cases, direct, auxiliary, decoder, config) -> dict[str, float | int]:
    maes: list[float] = []
    ssims: list[float] = []
    for _, geometry, camera, light, reference in cases:
        candidate, _ = render_hybrid_material(
            geometry,
            camera,
            light,
            direct,
            auxiliary,
            decoder,
            quantization="hard",
            minimum_roughness=float(config["render"]["minimum_roughness"]),
        )
        metrics = masked_render_metrics(
            reference,
            candidate,
            geometry.torch_buffers["mask"],
            linear_psnr_data_range=float(config["render"]["linear_psnr_data_range"]),
            display_exposure=float(config["render"]["display_exposure"]),
        )
        maes.append(float(metrics["masked_linear_hdr_mae"]))
        ssims.append(float(metrics["display_ssim"]))
    return {"case_count": len(cases), "hdr_mae": float(np.mean(maes)), "display_ssim": float(np.mean(ssims))}


def _draw(pool: torch.Tensor, count: int, generator: torch.Generator) -> torch.Tensor:
    positions = torch.randint(0, pool.numel(), (count,), generator=generator, device=pool.device)
    return pool[positions]


def _subpixel_batch(uniform, hard, metallic, *, batch_size, height, width, generator):
    if batch_size % 4:
        raise ValueError("Hybrid batch size must be divisible by four")
    half, quarter = batch_size // 2, batch_size // 4
    ids = torch.cat((_draw(uniform, half, generator), _draw(hard, quarter, generator), _draw(metallic, quarter, generator)))
    offsets = torch.rand((batch_size, 2), generator=generator, device=ids.device)
    x = ids.remainder(width).to(torch.float32) + offsets[:, 0]
    y = torch.div(ids, width, rounding_mode="floor").to(torch.float32) + offsets[:, 1]
    return ids, torch.stack((x / width, y / height), dim=-1), slice(half + quarter, batch_size)


def run(config_path: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    candidate = str(config["candidate"])
    channels = int(config["aux_channels"])
    if config.get("schema_version") != 1 or (candidate, channels) not in (("c1", 2), ("c2", 3)):
        raise ValueError("unsupported Hybrid candidate")
    inputs = config["inputs"]
    paths = {name: _repo_path(value, f"inputs.{name}") for name, value in inputs.items()}
    actual_hashes = {name: sha256_file(paths[name]) for name in config["frozen_sha256"]}
    if actual_hashes != dict(config["frozen_sha256"]):
        raise ValueError("Hybrid frozen input hash mismatch")
    phase0 = json.loads(paths["phase0_summary"].read_text(encoding="utf-8"))
    if phase0.get("gpu_training_allowed") is not True or phase0.get("formal_holdout_accessed") is not False:
        raise RuntimeError("Hybrid Phase 0 does not allow GPU training")
    if candidate == "c2":
        near_path = ROOT / "outputs/compression/scifihelmet/hybrid_interpolation_v1/c1/near_gate.json"
        if not near_path.is_file() or json.loads(near_path.read_text(encoding="utf-8")).get("c2_allowed") is not True:
            raise RuntimeError("C2 is forbidden because C1 did not satisfy the frozen near-gate")
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("Hybrid training requires the existing CUDA environment")
    training = config["training"]
    seed = int(training["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    direct, initial_auxiliary, decoder = _load_initialization(
        paths["initialization_texture_a"], paths["initialization_texture_b"], paths["initialization_decoder"], channels, device
    )
    expected_cost = (60, 240, 48) if candidate == "c1" else (68, 272, 56)
    actual_cost = (decoder.parameter_count, decoder.weight_bytes_float32, decoder.macs_per_pixel)
    if actual_cost != expected_cost:
        raise RuntimeError(f"Hybrid decoder cost mismatch: {actual_cost}")
    logits = torch.nn.Parameter(latent_float_to_logits(initial_auxiliary, clamp_epsilon=float(training["latent_clamp_epsilon"])))
    logits.requires_grad_(False)
    latent_optimizer = torch.optim.Adam([logits], lr=float(training["latent_learning_rate"]))
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=float(training["decoder_learning_rate"]))
    targets = load_core4_targets(paths["core4_dir"], device)
    references = _reference_textures(targets)
    reference_cpu = _reference_mapping(targets)
    partitions = deterministic_tile_partitions(targets.height, targets.width, tile_size=64, seed=int(training["split_seed"]))
    initial_mapping = _decode_mapping(direct, initial_auxiliary, decoder, int(training["decode_chunk_size"]))
    initial_errors = material_error_maps(reference_cpu, initial_mapping)
    optimizer_mask = partitions["optimizer"]
    composite = initial_errors["normal_degrees"] / 180.0 + initial_errors["roughness"]
    uniform_np = np.flatnonzero(optimizer_mask.reshape(-1)).astype(np.int64)
    hard_np = hard_example_indices(composite, optimizer_mask, top_fraction=float(training["hard_pool_top_fraction"]))
    edges = metallic_boundary_mask(reference_cpu["metallic_linear"], 0.1)
    metallic_np = np.flatnonzero((edges & optimizer_mask).reshape(-1)).astype(np.int64)
    if min(uniform_np.size, hard_np.size, metallic_np.size) == 0:
        raise RuntimeError("Hybrid optimizer-only pool is empty")
    uniform = torch.from_numpy(uniform_np).to(device)
    hard = torch.from_numpy(hard_np).to(device)
    metallic = torch.from_numpy(metallic_np).to(device)

    mesh = load_gltf_mesh(paths["gltf"])
    textures = load_core4_textures(paths["core4_manifest"], device)
    case_specs, case_partitions = _case_specs(config)
    optimizer_cases = _prepare_cases(config, case_partitions["optimizer"], case_specs, mesh, textures, device)
    selection_cases = _prepare_cases(config, case_partitions["selection"][: int(training["selection_render_case_limit"])], case_specs, mesh, textures, device)
    _, baseline_latent = load_latent_unorm8_png(paths["baseline_latent"], device=device)
    baseline_decoder = _decoder_from_npz(paths["baseline_decoder"], device)
    baseline_render = _render_summary(selection_cases, baseline_latent, baseline_decoder, config)
    baseline_mapping = _baseline_decode_mapping(
        baseline_latent, baseline_decoder, chunk_size=int(training["decode_chunk_size"])
    )
    baseline_material = _material_report(reference_cpu, baseline_mapping, config)

    output_dir = _repo_path(config["output_dir"], "output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path, checkpoint_path = output_dir / "train.jsonl", output_dir / "checkpoint.pt"
    started = time.monotonic()
    max_seconds = float(training["max_minutes"]) * 60.0
    completed_step = 0
    warmup = int(training["warmup_steps"])
    weights = {name: float(value) for name, value in config["loss"]["auxiliary_channels"].items()}
    for step in range(1, int(training["max_steps"]) + 1):
        if time.monotonic() - started >= max_seconds:
            break
        if step == warmup + 1:
            logits.requires_grad_(True)
        latent_optimizer.zero_grad(set_to_none=True)
        decoder_optimizer.zero_grad(set_to_none=True)
        auxiliary = fake_quantize_unorm8(torch.sigmoid(logits))
        _, uv, metallic_slice = _subpixel_batch(uniform, hard, metallic, batch_size=int(training["batch_size"]), height=targets.height, width=targets.width, generator=generator)
        target = _sample_reference(references, uv)
        texture_a = torch.cat((direct.detach(), auxiliary[..., 0:1]), dim=-1)
        sampled_a = bilinear_sample_top_down_wrap(texture_a, uv)
        sampled_b = bilinear_sample_top_down_wrap(auxiliary[..., 1:], uv)
        prediction = decode_auxiliary(decoder, torch.cat((sampled_a[..., 3:4], sampled_b), dim=-1), sampled_a[..., :3])
        subpixel_aux, sub_terms = material_loss(prediction, target, weights)
        metallic_boundary = F.l1_loss(prediction.metallic[metallic_slice], target.metallic[metallic_slice])
        texel_ids = _draw(uniform, int(training["batch_size"]) // 2, generator)
        texel_prediction = decode_auxiliary(decoder, auxiliary.reshape(-1, channels)[texel_ids], direct.reshape(-1, 3)[texel_ids])
        texel_aux, texel_terms = material_loss(texel_prediction, targets.select(texel_ids), weights)
        case_name, geometry, camera, light, reference_hdr = optimizer_cases[(step - 1) % len(optimizer_cases)]
        candidate_hdr, _ = render_hybrid_material(
            geometry, camera, light, direct, torch.sigmoid(logits), decoder,
            quantization="fake", minimum_roughness=float(config["render"]["minimum_roughness"]),
        )
        render_terms = pbr_render_loss(
            reference_hdr, candidate_hdr, geometry.torch_buffers["mask"], texel_aux,
            charbonnier_epsilon=1.0e-3, charbonnier_weight=1.0, log1p_weight=0.25, material_weight=0.25,
        )
        total = (
            render_terms.total
            + float(config["loss"]["subpixel_auxiliary_weight"]) * subpixel_aux
            + float(config["loss"]["subpixel_metallic_boundary_l1_weight"]) * metallic_boundary
            + float(config["loss"]["texel_center_auxiliary_guard_weight"]) * texel_aux
        )
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite Hybrid loss at step {step}")
        total.backward()
        decoder_optimizer.step()
        if logits.requires_grad:
            latent_optimizer.step()
        completed_step = step
        elapsed = time.monotonic() - started
        due = step % int(training["evaluation_interval"]) == 0
        timed_out = elapsed >= max_seconds
        if due or step == 1 or step % int(training["log_interval"]) == 0 or timed_out:
            record = {
                "step": step,
                "phase": "decoder_only" if step <= warmup else "joint",
                "case": case_name,
                "elapsed_seconds": round(elapsed, 3),
                "total": float(total.detach().cpu()),
                "render_charbonnier": float(render_terms.charbonnier_hdr.detach().cpu()),
                "render_log1p": float(render_terms.log1p_hdr.detach().cpu()),
                "subpixel_auxiliary": float(subpixel_aux.detach().cpu()),
                "metallic_boundary_l1": float(metallic_boundary.detach().cpu()),
                "texel_auxiliary": float(texel_aux.detach().cpu()),
                **{f"subpixel_{name}": float(value.detach().cpu()) for name, value in sub_terms.items()},
                **{f"texel_{name}": float(value.detach().cpu()) for name, value in texel_terms.items()},
            }
            if due or timed_out:
                hard_aux = hard_quantize_unorm8(torch.sigmoid(logits).detach())
                mapping = _decode_mapping(direct, hard_aux, decoder, int(training["decode_chunk_size"]))
                errors_now = material_error_maps(reference_cpu, mapping)
                record["selection_material"] = {
                    "base_p99": float(np.percentile(errors_now["base_color_max_channel"][partitions["selection"]], 99.0)),
                    "normal_p95": float(np.percentile(errors_now["normal_degrees"][partitions["selection"]], 95.0)),
                    "roughness_mae": float(np.mean(errors_now["roughness"][partitions["selection"]])),
                    "metallic_mae": float(np.mean(errors_now["metallic"][partitions["selection"]])),
                }
                record["selection_render"] = _hybrid_render_summary(selection_cases, direct, hard_aux, decoder, config)
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            print(record, flush=True)
        if due or timed_out or step == int(training["max_steps"]):
            _atomic_save({"schema_version": 1, "candidate": candidate, "step": step, "auxiliary_logits": logits.detach(), "decoder": decoder.state_dict()}, checkpoint_path)
        if timed_out:
            break
    if completed_step <= warmup:
        raise RuntimeError("Hybrid candidate did not complete warmup plus one joint step")
    final_aux = hard_quantize_unorm8(torch.sigmoid(logits).detach())
    texture_a_path = output_dir / "texture_a_base_rgb_aux0_rgba8.png"
    texture_b_path = output_dir / ("texture_b_aux1_r8.png" if channels == 2 else "texture_b_aux12_rg8.png")
    packing = export_hybrid_textures(direct, final_aux, texture_a_path, texture_b_path)
    decoder_path = output_dir / "decoder_weights.npz"
    np.savez(decoder_path, **{name: value.detach().cpu().numpy() for name, value in decoder.state_dict().items()})
    final_mapping = _decode_mapping(direct, final_aux, decoder, int(training["decode_chunk_size"]))
    candidate_material = _material_report(reference_cpu, final_mapping, config)
    candidate_render = _hybrid_render_summary(selection_cases, direct, final_aux, decoder, config)
    manifest = {
        "schema_version": 1,
        "candidate": candidate,
        "status": "trained_pending_full_interpolation_gates",
        "formal_holdout_accessed": False,
        "initialization": {"family": f"independent_optimizer_only_rank_{channels}_pca", "rejected_checkpoint_used": False, "c1_checkpoint_used_for_c2": False, "sha256": actual_hashes},
        "cost": {"parameters": decoder.parameter_count, "weight_bytes_float32": decoder.weight_bytes_float32, "macs_per_pixel": decoder.macs_per_pixel, "logical_raw_bytes": packing["logical_raw_bytes"], "texture_samples": 2},
        "training": {"completed_step": completed_step, "warmup_steps": warmup, "max_steps": int(training["max_steps"]), "elapsed_seconds": round(time.monotonic() - started, 3), "max_minutes": float(training["max_minutes"]), "latent_learning_rate": float(training["latent_learning_rate"]), "decoder_learning_rate": float(training["decoder_learning_rate"]), "evaluation_interval": int(training["evaluation_interval"]), "batch_mix": {"uniform_subpixel": 0.5, "normal_roughness_hard": 0.25, "metallic_boundary": 0.25}, "pool_sizes": {"uniform": int(uniform_np.size), "normal_roughness_hard": int(hard_np.size), "metallic_boundary": int(metallic_np.size)}, "case_partitions": case_partitions, "tile_partition_pixels": {name: int(mask.sum()) for name, mask in partitions.items()}, "selection_or_validation_used_for_optimization": False},
        "loss": config["loss"],
        "baseline": {"texel_center_material": baseline_material, "repair_selection_render": baseline_render},
        "candidate_metrics": {"texel_center_material": candidate_material, "repair_selection_render": candidate_render},
        "files": {"texture_a": {"path": texture_a_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(texture_a_path)}, "texture_b": {"path": texture_b_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(texture_b_path)}, "decoder": {"path": decoder_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(decoder_path)}, "checkpoint": sha256_file(checkpoint_path), "train_log": sha256_file(log_path)},
        "deployment_exported": False,
    }
    (output_dir / "training_manifest.json").write_text(deterministic_json(manifest), encoding="utf-8", newline="\n")
    print(json.dumps({"candidate": candidate, "completed_step": completed_step, "elapsed_seconds": manifest["training"]["elapsed_seconds"], "formal_holdout_accessed": False}, ensure_ascii=False))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train/scifihelmet_hybrid_c1.yaml")
    args = parser.parse_args()
    run(args.config.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
