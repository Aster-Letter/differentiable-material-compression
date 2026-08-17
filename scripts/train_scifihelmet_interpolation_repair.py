"""Train bounded S1 interpolation-aware repair from the frozen hard baseline."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np
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
from cg_frontier.compression.material import (  # noqa: E402
    Core4Targets,
    decode_material,
    load_core4_targets,
    material_loss,
)
from cg_frontier.compression.render_loss import (  # noqa: E402
    bilinear_sample_top_down_wrap,
    export_latent_unorm8_png,
    fake_quantize_unorm8,
    latent_float_to_logits,
    load_latent_unorm8_png,
    pbr_render_loss,
    render_latent_material,
)
from cg_frontier.compression.repair import (  # noqa: E402
    SplitHeadDecoder,
    hard_example_indices,
    initialize_split_head_from_tiny,
    top_fraction_mean,
)
from cg_frontier.render.gbuffer import load_core4_textures  # noqa: E402
from train_scifihelmet_repair import (  # noqa: E402
    _atomic_save,
    _case_specs,
    _decode_mapping,
    _decoder_from_npz,
    _material_report,
    _prepare_cases,
    _reference_mapping,
    _render_summary,
    _repo_path,
    _selection_snapshot,
)


def _draw(pool: torch.Tensor, count: int, generator: torch.Generator) -> torch.Tensor:
    if pool.numel() == 0:
        raise ValueError("subpixel sampling pool is empty")
    positions = torch.randint(0, pool.numel(), (count,), generator=generator, device=pool.device)
    return pool[positions]


def _deep_update(target: dict[str, Any], source: Mapping[str, Any]) -> dict[str, Any]:
    for key, value in source.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)
    return target


def _load_config(config_path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("interpolation-repair config must be a mapping")
    if "extends" not in raw:
        return dict(raw)
    base_path = _repo_path(raw["extends"], "extends")
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
    if not isinstance(base, dict):
        raise ValueError("base interpolation-repair config must be a mapping")
    overrides = {key: value for key, value in raw.items() if key != "extends"}
    return _deep_update(base, overrides)


@torch.no_grad()
def _fit_base_affine_on_optimizer(
    decoder: SplitHeadDecoder,
    baseline_decoder: torch.nn.Module,
    hard_latent: torch.Tensor,
    optimizer_indices: torch.Tensor,
    *,
    chunk_size: int,
) -> None:
    """Fit raw RGB by optimizer-only deterministic sufficient statistics."""

    xtx = torch.zeros((5, 5), dtype=torch.float64, device=hard_latent.device)
    xty = torch.zeros((5, 3), dtype=torch.float64, device=hard_latent.device)
    flat = hard_latent.reshape(-1, 4)
    for start in range(0, optimizer_indices.numel(), chunk_size):
        latent = flat[optimizer_indices[start : start + chunk_size]]
        target = baseline_decoder(latent)[..., :3].to(torch.float64)
        design = torch.cat(
            (latent.to(torch.float64), torch.ones((latent.shape[0], 1), dtype=torch.float64, device=latent.device)),
            dim=-1,
        )
        xtx += design.T @ design
        xty += design.T @ target
    solution = torch.linalg.solve(
        xtx + torch.eye(5, dtype=torch.float64, device=xtx.device) * 1.0e-10,
        xty,
    ).to(torch.float32)
    decoder.base_affine.weight.copy_(solution[:4].T)
    decoder.base_affine.bias.copy_(solution[4])


def _subpixel_batch(
    uniform_pool: torch.Tensor,
    base_pool: torch.Tensor,
    metallic_pool: torch.Tensor,
    *,
    batch_size: int,
    height: int,
    width: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, slice]]:
    """Draw the frozen 50/25/25 mixture and uniform within-texel offsets."""

    if batch_size < 4 or batch_size % 4:
        raise ValueError("batch size must be a positive multiple of four")
    uniform_count, focused_count = batch_size // 2, batch_size // 4
    ids = torch.cat(
        (
            _draw(uniform_pool, uniform_count, generator),
            _draw(base_pool, focused_count, generator),
            _draw(metallic_pool, focused_count, generator),
        )
    )
    offsets = torch.rand((batch_size, 2), generator=generator, device=ids.device)
    x = ids.remainder(width).to(torch.float32) + offsets[:, 0]
    y = torch.div(ids, width, rounding_mode="floor").to(torch.float32) + offsets[:, 1]
    uv = torch.stack((x / float(width), y / float(height)), dim=-1)
    slices = {
        "uniform": slice(0, uniform_count),
        "base_hard_bright": slice(uniform_count, uniform_count + focused_count),
        "metallic_boundary": slice(uniform_count + focused_count, batch_size),
    }
    return ids, uv, slices


def _reference_textures(targets: Core4Targets) -> dict[str, torch.Tensor]:
    return {
        "base_color": targets.base_color_linear.reshape(targets.height, targets.width, 3),
        "normal": targets.normal_xyz.reshape(targets.height, targets.width, 3),
        "roughness": targets.roughness.reshape(targets.height, targets.width, 1),
        "metallic": targets.metallic.reshape(targets.height, targets.width, 1),
    }


def _sample_reference(textures: Mapping[str, torch.Tensor], uv: torch.Tensor) -> Core4Targets:
    base = bilinear_sample_top_down_wrap(textures["base_color"], uv)
    normal = F.normalize(bilinear_sample_top_down_wrap(textures["normal"], uv), dim=-1, eps=1.0e-8)
    roughness = bilinear_sample_top_down_wrap(textures["roughness"], uv)
    metallic = bilinear_sample_top_down_wrap(textures["metallic"], uv)
    return Core4Targets(base, normal, roughness, metallic, height=1, width=base.shape[0])


def _pool_indices(
    targets: Core4Targets,
    baseline_errors: Mapping[str, np.ndarray],
    optimizer_mask: np.ndarray,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = targets.height, targets.width
    y, x = np.indices((height, width))
    focus = np.zeros((height, width), dtype=bool)
    for bbox in config["scopes"].values():
        x0, y0, x1, y1 = (int(value) for value in bbox)
        focus[y0:y1, x0:x1] = True
    truth_base = targets.base_color_linear.detach().cpu().numpy().reshape(height, width, 3)
    truth_luma = np.sum(
        truth_base * np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32),
        axis=-1,
        dtype=np.float32,
    )
    bright_focus = optimizer_mask & focus & (truth_luma > 0.05)
    base_pool = hard_example_indices(
        baseline_errors["base_color_max_channel"],
        bright_focus,
        top_fraction=float(config["training"]["hard_bright_pool_top_fraction"]),
    )
    edges = metallic_boundary_mask(
        targets.metallic.detach().cpu().numpy().reshape(height, width), 0.1
    )
    metallic_pool = np.flatnonzero(edges.reshape(-1) & optimizer_mask.reshape(-1)).astype(np.int64)
    uniform_pool = np.flatnonzero(optimizer_mask.reshape(-1)).astype(np.int64)
    if min(uniform_pool.size, base_pool.size, metallic_pool.size) == 0:
        raise RuntimeError("optimizer-only subpixel pool construction failed")
    return uniform_pool, base_pool, metallic_pool, edges


def run(config_path: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    candidate_name = str(config.get("candidate"))
    if config.get("schema_version") != 1 or candidate_name not in ("s1", "s2"):
        raise ValueError("unsupported interpolation-repair config")
    inputs, training = config["inputs"], config["training"]
    latent_path = _repo_path(inputs["latent_hard_png"], "inputs.latent_hard_png")
    decoder_path = _repo_path(inputs["decoder_npz"], "inputs.decoder_npz")
    actual_hashes = {
        "latent_hard_png": sha256_file(latent_path),
        "decoder_npz": sha256_file(decoder_path),
    }
    if actual_hashes != dict(config["frozen_sha256"]):
        raise ValueError(f"{candidate_name.upper()} initialization does not match the frozen baseline")
    output_dir = _repo_path(config["output_dir"], "output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    max_seconds = float(training["max_minutes"]) * 60.0
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError(f"{candidate_name.upper()} requires the existing CUDA environment")
    seed = int(training["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    _, hard_latent = load_latent_unorm8_png(latent_path, device=device)
    base_decoder = _decoder_from_npz(decoder_path, device)
    decoder: torch.nn.Module = (
        base_decoder if candidate_name == "s1" else initialize_split_head_from_tiny(base_decoder)
    )
    expected_cost = (103, 412, 88) if candidate_name == "s1" else (91, 364, 76)
    actual_cost = (
        sum(parameter.numel() for parameter in decoder.parameters()),
        sum(parameter.numel() * parameter.element_size() for parameter in decoder.parameters()),
        int(decoder.macs_per_pixel),  # type: ignore[attr-defined]
    )
    if actual_cost != expected_cost:
        raise RuntimeError(f"{candidate_name.upper()} decoder cost mismatch: {actual_cost}")
    logits = torch.nn.Parameter(
        latent_float_to_logits(hard_latent, clamp_epsilon=float(training["latent_clamp_epsilon"]))
    )
    latent_optimizer = torch.optim.Adam([logits], lr=float(training["latent_learning_rate"]))
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=float(training["decoder_learning_rate"]))
    if candidate_name == "s1":
        for parameter in decoder.parameters():
            parameter.requires_grad_(False)
    else:
        logits.requires_grad_(False)
        for parameter in decoder.parameters():
            parameter.requires_grad_(False)
        assert isinstance(decoder, SplitHeadDecoder)
        for parameter in decoder.base_affine.parameters():
            parameter.requires_grad_(True)

    targets = load_core4_targets(_repo_path(inputs["core4_dir"], "inputs.core4_dir"), device)
    reference_textures = _reference_textures(targets)
    reference_cpu = _reference_mapping(targets)
    baseline_mapping = _decode_mapping(
        hard_latent, base_decoder, chunk_size=int(training["decode_chunk_size"])
    )
    baseline_errors = material_error_maps(reference_cpu, baseline_mapping)
    partitions = deterministic_tile_partitions(
        targets.height,
        targets.width,
        tile_size=64,
        seed=int(training["split_seed"]),
    )
    uniform_np, base_np, metallic_np, edges = _pool_indices(
        targets, baseline_errors, partitions["optimizer"], config
    )
    uniform_pool = torch.from_numpy(uniform_np).to(device)
    base_pool = torch.from_numpy(base_np).to(device)
    metallic_pool = torch.from_numpy(metallic_np).to(device)
    if candidate_name == "s2":
        assert isinstance(decoder, SplitHeadDecoder)
        _fit_base_affine_on_optimizer(
            decoder,
            base_decoder,
            hard_latent,
            uniform_pool,
            chunk_size=int(training["decode_chunk_size"]),
        )

    mesh = load_gltf_mesh(_repo_path(inputs["gltf"], "inputs.gltf"))
    textures = load_core4_textures(_repo_path(inputs["core4_manifest"], "inputs.core4_manifest"), device)
    case_specs, case_partitions = _case_specs(config)
    optimizer_cases = _prepare_cases(
        config, case_partitions["optimizer"], case_specs, mesh, textures, device
    )
    selection_cases = _prepare_cases(
        config,
        case_partitions["selection"][: int(training["selection_render_case_limit"])],
        case_specs,
        mesh,
        textures,
        device,
    )
    baseline_render = _render_summary(selection_cases, hard_latent, base_decoder, config)
    baseline_material = _material_report(reference_cpu, baseline_mapping, config)

    warmup_steps = int(training["warmup_steps"])
    max_steps = int(training["max_steps"])
    batch_size = int(training["batch_size"])
    evaluation_interval = int(training["evaluation_interval"])
    log_interval = int(training["log_interval"])
    log_path = output_dir / "train.jsonl"
    checkpoint_path = output_dir / "checkpoint.pt"
    completed_step = 0
    last_bounded = hard_latent
    for step in range(1, max_steps + 1):
        if time.monotonic() - started >= max_seconds:
            break
        if step == warmup_steps + 1:
            logits.requires_grad_(True)
            for parameter in decoder.parameters():
                parameter.requires_grad_(True)
        latent_optimizer.zero_grad(set_to_none=True)
        decoder_optimizer.zero_grad(set_to_none=True)
        bounded = torch.sigmoid(logits)
        deployed = fake_quantize_unorm8(bounded)
        _, uv, slices = _subpixel_batch(
            uniform_pool,
            base_pool,
            metallic_pool,
            batch_size=batch_size,
            height=targets.height,
            width=targets.width,
            generator=generator,
        )
        subpixel_target = _sample_reference(reference_textures, uv)
        sampled_latent = bilinear_sample_top_down_wrap(deployed, uv)
        raw_prediction = decoder(sampled_latent)
        subpixel_prediction = decode_material(decoder, sampled_latent)  # type: ignore[arg-type]
        subpixel_material, subpixel_terms = material_loss(
            subpixel_prediction,
            subpixel_target,
            {name: float(value) for name, value in config["loss"]["material_channels"].items()},
        )
        luma_weights = torch.tensor([0.2126, 0.7152, 0.0722], device=device)
        target_luma = torch.sum(subpixel_target.base_color_linear * luma_weights, dim=-1)
        prediction_luma = torch.sum(subpixel_prediction.base_color_linear * luma_weights, dim=-1)
        eligible = target_luma > 0.05
        if not torch.any(eligible):
            raise RuntimeError("S1 bright underprediction batch is empty")
        underprediction = torch.relu(target_luma[eligible] - prediction_luma[eligible])
        underprediction_tail = top_fraction_mean(
            underprediction, float(training["underprediction_top_fraction"])
        )
        metallic_slice = slices["metallic_boundary"]
        metallic_boundary_l1 = F.l1_loss(
            subpixel_prediction.metallic[metallic_slice],
            subpixel_target.metallic[metallic_slice],
        )
        uniform_ids = _draw(uniform_pool, batch_size // 2, generator)
        texel_prediction = decode_material(decoder, deployed.reshape(-1, 4)[uniform_ids])
        texel_global, global_terms = material_loss(
            texel_prediction,
            targets.select(uniform_ids),
            {"base_color_l1": 1.0, "normal_cosine": 1.0, "roughness_l1": 0.5, "metallic_l1": 0.5},
        )
        case_name, geometry, camera, light, reference_hdr = optimizer_cases[(step - 1) % len(optimizer_cases)]
        candidate_hdr, _ = render_latent_material(
            geometry,
            camera,
            light,
            bounded,
            decoder,
            quantization="fake",
            minimum_roughness=float(config["render"]["minimum_roughness"]),
        )
        render_terms = pbr_render_loss(
            reference_hdr,
            candidate_hdr,
            geometry.torch_buffers["mask"],
            texel_global,
            charbonnier_epsilon=1.0e-3,
            charbonnier_weight=1.0,
            log1p_weight=0.25,
            material_weight=0.25,
        )
        full_total = (
            render_terms.total
            + float(config["loss"]["subpixel_material_weight"]) * subpixel_material
            + float(config["loss"]["luminance_underprediction_top_tail_weight"]) * underprediction_tail
            + float(config["loss"]["subpixel_metallic_boundary_l1_weight"]) * metallic_boundary_l1
        )
        if candidate_name == "s2" and step <= warmup_steps:
            with torch.no_grad():
                baseline_raw_rgb = base_decoder(sampled_latent)[..., :3]
            total = F.mse_loss(raw_prediction[..., :3], baseline_raw_rgb)
        else:
            total = full_total
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite S1 loss at step {step}")
        total.backward()
        if logits.requires_grad:
            latent_optimizer.step()
        if candidate_name == "s2" or step > warmup_steps:
            decoder_optimizer.step()
        completed_step = step
        last_bounded = bounded.detach()
        elapsed = time.monotonic() - started
        evaluation_due = step % evaluation_interval == 0
        timed_out = elapsed >= max_seconds
        record: dict[str, Any] = {
            "step": step,
            "phase": ("latent_only" if candidate_name == "s1" else "base_affine_head_warmup") if step <= warmup_steps else "joint",
            "case": case_name,
            "elapsed_seconds": round(elapsed, 3),
            "total": float(total.detach().cpu()),
            "render_charbonnier": float(render_terms.charbonnier_hdr.detach().cpu()),
            "render_log1p": float(render_terms.log1p_hdr.detach().cpu()),
            "texel_global_material": float(texel_global.detach().cpu()),
            "subpixel_material": float(subpixel_material.detach().cpu()),
            "underprediction_top_tail": float(underprediction_tail.detach().cpu()),
            "subpixel_metallic_boundary_l1": float(metallic_boundary_l1.detach().cpu()),
            **{f"subpixel_{name}": float(value.detach().cpu()) for name, value in subpixel_terms.items()},
            **{f"texel_{name}": float(value.detach().cpu()) for name, value in global_terms.items()},
        }
        if evaluation_due or step == 1 or step % log_interval == 0 or timed_out:
            if evaluation_due or timed_out:
                record["selection_material"] = _selection_snapshot(
                    partitions["selection"],
                    targets,
                    deployed.detach(),
                    decoder,
                    int(training["decode_chunk_size"]),
                )
                record["selection_render"] = _render_summary(
                    selection_cases, deployed.detach(), decoder, config
                )
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            print(record, flush=True)
        if evaluation_due or timed_out or step == max_steps:
            _atomic_save(
                {
                    "schema_version": 1,
                    "candidate": candidate_name,
                    "step": step,
                    "latent_logits": logits.detach(),
                    "decoder": decoder.state_dict(),
                },
                checkpoint_path,
            )
        if timed_out:
            break
    if completed_step == 0:
        raise RuntimeError(f"{candidate_name.upper()} hard time limit elapsed before the first optimization step")

    final_latent = fake_quantize_unorm8(torch.sigmoid(logits).detach())
    latent_metadata = export_latent_unorm8_png(
        final_latent, output_dir / f"latent_{candidate_name}_rgba_unorm8.png"
    )
    decoder_arrays = {name: value.detach().cpu().numpy() for name, value in decoder.state_dict().items()}
    np.savez(output_dir / "decoder_weights.npz", **decoder_arrays)
    final_mapping = _decode_mapping(
        final_latent, decoder, chunk_size=int(training["decode_chunk_size"])
    )
    candidate_material = _material_report(reference_cpu, final_mapping, config)
    candidate_render = _render_summary(selection_cases, final_latent, decoder, config)
    elapsed = time.monotonic() - started
    manifest = {
        "schema_version": 1,
        "candidate": candidate_name,
        "status": "trained_pending_full_interpolation_gates",
        "formal_holdout_accessed": False,
        "initialization": {
            "latent_sha256": actual_hashes["latent_hard_png"],
            "decoder_sha256": actual_hashes["decoder_npz"],
            "checkpoint_family": "frozen_pre_qat_hard_only",
            "rejected_checkpoint_used": False,
        },
        "cost": {
            "parameters": int(sum(parameter.numel() for parameter in decoder.parameters())),
            "weight_bytes_float32": int(sum(parameter.numel() * parameter.element_size() for parameter in decoder.parameters())),
            "macs_per_pixel": int(decoder.macs_per_pixel),  # type: ignore[attr-defined]
        },
        "training": {
            "completed_step": completed_step,
            "max_steps": max_steps,
            "warmup_steps": warmup_steps,
            "elapsed_seconds": round(elapsed, 3),
            "max_minutes": float(training["max_minutes"]),
            "latent_learning_rate": float(training["latent_learning_rate"]),
            "decoder_learning_rate": float(training["decoder_learning_rate"]),
            "evaluation_interval": evaluation_interval,
            "batch_mix": {"uniform_subpixel": 0.50, "D2_D3_base_hard_bright": 0.25, "metallic_boundary_subpixel": 0.25},
            "pool_sizes": {"uniform": int(uniform_np.size), "base_hard_bright": int(base_np.size), "metallic_boundary": int(metallic_np.size)},
            "case_partitions": case_partitions,
            "tile_partition_pixels": {name: int(mask.sum()) for name, mask in partitions.items()},
            "selection_or_validation_used_for_optimization": False,
        },
        "loss": config["loss"],
        "baseline": {"texel_center_material": baseline_material, "repair_selection_render": baseline_render},
        "candidate_metrics": {"texel_center_material": candidate_material, "repair_selection_render": candidate_render},
        "files": {
            f"latent_{candidate_name}_rgba_unorm8.png": latent_metadata,
            "decoder_weights.npz": sha256_file(output_dir / "decoder_weights.npz"),
            "checkpoint.pt": sha256_file(checkpoint_path),
            "train.jsonl": sha256_file(log_path),
        },
        "deployment_exported": False,
    }
    (output_dir / "training_manifest.json").write_text(
        deterministic_json(manifest), encoding="utf-8", newline="\n"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/train/scifihelmet_interpolation_repair.yaml",
    )
    args = parser.parse_args()
    result = run(args.config.resolve())
    print(
        json.dumps(
            {
                "candidate": result["candidate"],
                "status": result["status"],
                "completed_step": result["training"]["completed_step"],
                "elapsed_seconds": result["training"]["elapsed_seconds"],
                "formal_holdout_accessed": False,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
