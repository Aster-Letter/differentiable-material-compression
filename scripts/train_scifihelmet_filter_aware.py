"""Train the fixed F-ReLU/F-Softplus/F-Sigmoid filter-aware matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cg_frontier.assets.gltf_mesh import load_gltf_mesh  # noqa: E402
from cg_frontier.compression.artifact_analysis import (  # noqa: E402
    deterministic_json,
    deterministic_tile_partitions,
    metallic_boundary_mask,
    sha256_file,
)
from cg_frontier.compression.filter_aware import (  # noqa: E402
    FILTER_AWARE_KINDS,
    FilterAwareDecoder,
    calculate_filter_aware_cost,
    dilate_mask,
    initialize_filter_aware_from_tiny,
    postprocess_commutativity_loss,
)
from cg_frontier.compression.material import (  # noqa: E402
    Core4Targets,
    DecodedMaterial,
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
from cg_frontier.compression.repair import hard_example_indices, top_fraction_mean  # noqa: E402
from cg_frontier.render.gbuffer import load_core4_textures  # noqa: E402
from train_scifihelmet_interpolation_repair import (  # noqa: E402
    _load_config,
    _reference_textures,
    _sample_reference,
)
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
        raise ValueError("filter-aware sampling pool is empty")
    positions = torch.randint(0, pool.numel(), (count,), generator=generator, device=pool.device)
    return pool[positions]


def filter_aware_batch(
    uniform_pool: torch.Tensor,
    dark_pool: torch.Tensor,
    boundary_pool: torch.Tensor,
    *,
    batch_size: int,
    height: int,
    width: int,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, slice]]:
    """Draw exact 40/20/20 subpixel plus 20% texel-center anchors."""

    if batch_size < 5 or batch_size % 5:
        raise ValueError("filter-aware batch size must be a positive multiple of five")
    unit = batch_size // 5
    counts = (2 * unit, unit, unit, unit)
    ids = torch.cat(
        (
            _draw(uniform_pool, counts[0], generator),
            _draw(dark_pool, counts[1], generator),
            _draw(boundary_pool, counts[2], generator),
            _draw(uniform_pool, counts[3], generator),
        )
    )
    offsets = torch.rand((sum(counts[:3]), 2), generator=generator, device=ids.device)
    subpixel_ids = ids[: sum(counts[:3])]
    subpixel_x = subpixel_ids.remainder(width).to(torch.float32) + offsets[:, 0]
    subpixel_y = torch.div(subpixel_ids, width, rounding_mode="floor").to(torch.float32) + offsets[:, 1]
    anchor_ids = ids[sum(counts[:3]) :]
    anchor_x = anchor_ids.remainder(width).to(torch.float32) + 0.5
    anchor_y = torch.div(anchor_ids, width, rounding_mode="floor").to(torch.float32) + 0.5
    uv = torch.cat(
        (
            torch.stack((subpixel_x / float(width), subpixel_y / float(height)), dim=-1),
            torch.stack((anchor_x / float(width), anchor_y / float(height)), dim=-1),
        )
    )
    slices = {
        "uniform_subpixel": slice(0, counts[0]),
        "dark_hard_subpixel": slice(counts[0], counts[0] + counts[1]),
        "boundary_halo_subpixel": slice(counts[0] + counts[1], sum(counts[:3])),
        "texel_center_anchor": slice(sum(counts[:3]), batch_size),
        "all_subpixel": slice(0, sum(counts[:3])),
    }
    return ids, uv, slices


def _material_distance(left: DecodedMaterial, right: DecodedMaterial) -> torch.Tensor:
    return (
        F.l1_loss(left.base_color_linear, right.base_color_linear)
        + torch.mean(1.0 - torch.sum(left.normal_xyz * right.normal_xyz, dim=-1))
        + 0.5 * F.l1_loss(left.roughness, right.roughness)
        + 0.5 * F.l1_loss(left.metallic, right.metallic)
    )


def _focus_mask(height: int, width: int, scopes: Mapping[str, Any]) -> np.ndarray:
    result = np.zeros((height, width), dtype=bool)
    for bbox in scopes.values():
        x0, y0, x1, y1 = (int(value) for value in bbox)
        result[y0:y1, x0:x1] = True
    return result


@torch.no_grad()
def _dark_hard_pool(
    targets: Core4Targets,
    reference_textures: Mapping[str, torch.Tensor],
    hard_latent: torch.Tensor,
    baseline_decoder: torch.nn.Module,
    optimizer_mask: np.ndarray,
    config: Mapping[str, Any],
) -> np.ndarray:
    height, width = targets.height, targets.width
    eligible_ids = np.flatnonzero(
        optimizer_mask.reshape(-1) & _focus_mask(height, width, config["scopes"]).reshape(-1)
    ).astype(np.int64)
    device = hard_latent.device
    errors = np.zeros(eligible_ids.size, dtype=np.float32)
    phase_x, phase_y = (float(value) for value in config["training"]["dark_pool_phase_xy"])
    luma_weights = torch.tensor([0.2126, 0.7152, 0.0722], device=device)
    chunk = int(config["training"]["decode_chunk_size"])
    flat = hard_latent.reshape(-1, 4)
    valid = np.zeros(eligible_ids.size, dtype=bool)
    for start in range(0, eligible_ids.size, chunk):
        stop = min(start + chunk, eligible_ids.size)
        ids = torch.from_numpy(eligible_ids[start:stop]).to(device)
        x = ids.remainder(width).to(torch.float32) + phase_x
        y = torch.div(ids, width, rounding_mode="floor").to(torch.float32) + phase_y
        uv = torch.stack((x / float(width), y / float(height)), dim=-1)
        reference = _sample_reference(reference_textures, uv)
        sampled = bilinear_sample_top_down_wrap(hard_latent, uv)
        prediction = decode_material(baseline_decoder, sampled)  # type: ignore[arg-type]
        ref_luma = torch.sum(reference.base_color_linear * luma_weights, dim=-1)
        pred_luma = torch.sum(prediction.base_color_linear * luma_weights, dim=-1)
        errors[start:stop] = torch.relu(ref_luma - pred_luma).cpu().numpy()
        valid[start:stop] = (ref_luma > 0.05).cpu().numpy()
    local = hard_example_indices(
        errors,
        valid,
        top_fraction=float(config["training"]["dark_hard_pool_top_fraction"]),
    )
    return eligible_ids[local]


def _boundary_pool(
    targets: Core4Targets, optimizer_mask: np.ndarray, config: Mapping[str, Any]
) -> np.ndarray:
    height, width = targets.height, targets.width
    base = targets.base_color_linear.detach().cpu().numpy().reshape(height, width, 3)
    luma = np.sum(base * np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32), axis=-1)
    gradient = np.zeros((height, width), dtype=np.float32)
    horizontal = np.abs(luma[:, 1:] - luma[:, :-1])
    vertical = np.abs(luma[1:, :] - luma[:-1, :])
    gradient[:, 1:] = np.maximum(gradient[:, 1:], horizontal)
    gradient[:, :-1] = np.maximum(gradient[:, :-1], horizontal)
    gradient[1:] = np.maximum(gradient[1:], vertical)
    gradient[:-1] = np.maximum(gradient[:-1], vertical)
    metallic = targets.metallic.detach().cpu().numpy().reshape(height, width)
    boundary = dilate_mask((gradient > 0.02) | metallic_boundary_mask(metallic, 0.1), radius=1)
    pool = np.flatnonzero(boundary.reshape(-1) & optimizer_mask.reshape(-1)).astype(np.int64)
    if pool.size == 0:
        raise RuntimeError("material-boundary/halo pool is empty")
    return pool


def _checkpoint_payload(
    *,
    candidate: str,
    step: int,
    logits: torch.Tensor,
    decoder: FilterAwareDecoder,
    latent_optimizer: torch.optim.Optimizer,
    decoder_optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "candidate": candidate,
        "step": step,
        "latent_logits": logits.detach(),
        "decoder": decoder.state_dict(),
        "optimizers": {
            "latent": latent_optimizer.state_dict(),
            "decoder": decoder_optimizer.state_dict(),
        },
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all(),
            "sampling_generator": generator.get_state(),
        },
    }


def run(config_path: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    candidate_name = str(config.get("candidate"))
    if config.get("schema_version") != 1 or candidate_name not in FILTER_AWARE_KINDS:
        raise ValueError("unsupported filter-aware config")
    if "formal_holdout" in config_path.as_posix().lower():
        raise ValueError("formal holdout paths are forbidden")
    inputs, training = config["inputs"], config["training"]
    latent_path = _repo_path(inputs["latent_hard_png"], "inputs.latent_hard_png")
    decoder_path = _repo_path(inputs["decoder_npz"], "inputs.decoder_npz")
    actual_hashes = {
        "latent_hard_png": sha256_file(latent_path),
        "decoder_npz": sha256_file(decoder_path),
    }
    if actual_hashes != dict(config["frozen_sha256"]):
        raise ValueError("filter-aware initialization does not match the frozen pre-QAT baseline")
    output_dir = _repo_path(config["output_dir"], "output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("filter-aware training requires the existing CUDA environment")
    seed = int(training["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    started = time.monotonic()
    max_seconds = float(training["max_minutes"]) * 60.0
    _, hard_latent = load_latent_unorm8_png(latent_path, device=device)
    baseline_decoder = _decoder_from_npz(decoder_path, device)
    decoder = initialize_filter_aware_from_tiny(baseline_decoder, candidate_name)
    cost = calculate_filter_aware_cost(decoder)
    probes = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [0.5, 0.5, 0.5, 0.5]],
        device=device,
    )
    step0_exact = bool(torch.equal(decoder(probes), baseline_decoder(probes)))
    if candidate_name == "f_relu" and not step0_exact:
        raise RuntimeError("F-ReLU step 0 is not function-identical to the frozen baseline")
    logits = torch.nn.Parameter(
        latent_float_to_logits(hard_latent, clamp_epsilon=float(training["latent_clamp_epsilon"]))
    )
    warmup_steps = int(training["warmup_steps"])
    if candidate_name != "f_relu":
        logits.requires_grad_(False)
    latent_optimizer = torch.optim.Adam([logits], lr=float(training["joint_latent_learning_rate"]))
    decoder_optimizer = torch.optim.Adam(
        decoder.parameters(),
        lr=(
            float(training["warmup_decoder_learning_rate"])
            if warmup_steps
            else float(training["joint_decoder_learning_rate"])
        ),
    )
    targets = load_core4_targets(_repo_path(inputs["core4_dir"], "inputs.core4_dir"), device)
    reference_textures = _reference_textures(targets)
    reference_cpu = _reference_mapping(targets)
    baseline_mapping = _decode_mapping(
        hard_latent, baseline_decoder, chunk_size=int(training["decode_chunk_size"])
    )
    partitions = deterministic_tile_partitions(
        targets.height, targets.width, tile_size=64, seed=int(training["split_seed"])
    )
    uniform_np = np.flatnonzero(partitions["optimizer"].reshape(-1)).astype(np.int64)
    dark_np = _dark_hard_pool(
        targets,
        reference_textures,
        hard_latent,
        baseline_decoder,
        partitions["optimizer"],
        config,
    )
    boundary_np = _boundary_pool(targets, partitions["optimizer"], config)
    uniform_pool = torch.from_numpy(uniform_np).to(device)
    dark_pool = torch.from_numpy(dark_np).to(device)
    boundary_pool = torch.from_numpy(boundary_np).to(device)
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
    baseline_render = _render_summary(selection_cases, hard_latent, baseline_decoder, config)
    baseline_material = _material_report(reference_cpu, baseline_mapping, config)
    max_steps = int(training["max_steps"])
    batch_size = int(training["batch_size"])
    evaluation_interval = int(training["evaluation_interval"])
    log_interval = int(training["log_interval"])
    log_path = output_dir / "train.jsonl"
    checkpoint_path = output_dir / "checkpoint.pt"
    if log_path.exists():
        raise FileExistsError(f"refusing to inherit an existing candidate log: {log_path}")
    completed_step = 0
    joint_evaluations = 0
    for step in range(1, max_steps + 1):
        if time.monotonic() - started >= max_seconds:
            break
        if warmup_steps and step == warmup_steps + 1:
            logits.requires_grad_(True)
            decoder_optimizer.param_groups[0]["lr"] = float(training["joint_decoder_learning_rate"])
        latent_optimizer.zero_grad(set_to_none=True)
        decoder_optimizer.zero_grad(set_to_none=True)
        bounded = torch.sigmoid(logits)
        deployed = fake_quantize_unorm8(bounded)
        ids, uv, slices = filter_aware_batch(
            uniform_pool,
            dark_pool,
            boundary_pool,
            batch_size=batch_size,
            height=targets.height,
            width=targets.width,
            generator=generator,
        )
        subpixel_slice = slices["all_subpixel"]
        subpixel_uv = uv[subpixel_slice]
        subpixel_target = _sample_reference(reference_textures, subpixel_uv)
        commute = postprocess_commutativity_loss(decoder, deployed, subpixel_uv)
        subpixel_prediction = commute.runtime
        subpixel_material, subpixel_terms = material_loss(
            subpixel_prediction,
            subpixel_target,
            {name: float(value) for name, value in config["loss"]["material_channels"].items()},
        )
        luma_weights = torch.tensor([0.2126, 0.7152, 0.0722], device=device)
        dark_slice = slices["dark_hard_subpixel"]
        target_luma = torch.sum(subpixel_target.base_color_linear * luma_weights, dim=-1)
        prediction_luma = torch.sum(subpixel_prediction.base_color_linear * luma_weights, dim=-1)
        dark_local_start = dark_slice.start
        dark_local_stop = dark_slice.stop
        eligible_under = target_luma[dark_local_start:dark_local_stop] > 0.05
        if not torch.any(eligible_under):
            raise RuntimeError("dark-hard batch contains no bright reference samples")
        underprediction = torch.relu(
            target_luma[dark_local_start:dark_local_stop][eligible_under]
            - prediction_luma[dark_local_start:dark_local_stop][eligible_under]
        )
        under_tail = top_fraction_mean(
            underprediction, float(training["underprediction_top_fraction"])
        )
        boundary_slice = slices["boundary_halo_subpixel"]
        boundary_local_start = boundary_slice.start
        boundary_local_stop = boundary_slice.stop
        boundary_metallic = F.l1_loss(
            subpixel_prediction.metallic[boundary_local_start:boundary_local_stop],
            subpixel_target.metallic[boundary_local_start:boundary_local_stop],
        )
        boundary_luma = F.l1_loss(
            prediction_luma[boundary_local_start:boundary_local_stop],
            target_luma[boundary_local_start:boundary_local_stop],
        )
        metallic_boundary_halo = 0.5 * (boundary_metallic + boundary_luma)
        anchor_slice = slices["texel_center_anchor"]
        anchor_ids = ids[anchor_slice]
        candidate_anchor = decode_material(decoder, deployed.reshape(-1, 4)[anchor_ids])  # type: ignore[arg-type]
        with torch.no_grad():
            baseline_anchor = decode_material(baseline_decoder, hard_latent.reshape(-1, 4)[anchor_ids])
        reference_anchor = targets.select(anchor_ids)
        anchor_reference, _ = material_loss(
            candidate_anchor,
            reference_anchor,
            {name: float(value) for name, value in config["loss"]["material_channels"].items()},
        )
        anchor_baseline = _material_distance(candidate_anchor, baseline_anchor)
        anchor_loss = 0.5 * (anchor_reference + anchor_baseline)
        case_name, geometry, camera, light, reference_hdr = optimizer_cases[(step - 1) % len(optimizer_cases)]
        candidate_hdr, _ = render_latent_material(
            geometry,
            camera,
            light,
            bounded,
            decoder,  # type: ignore[arg-type]
            quantization="fake",
            minimum_roughness=float(config["render"]["minimum_roughness"]),
        )
        render_terms = pbr_render_loss(
            reference_hdr,
            candidate_hdr,
            geometry.torch_buffers["mask"],
            anchor_reference,
            charbonnier_epsilon=1.0e-3,
            charbonnier_weight=1.0,
            log1p_weight=0.25,
            material_weight=0.25,
        )
        full_total = (
            render_terms.total
            + float(config["loss"]["subpixel_reference_material_weight"]) * subpixel_material
            + float(config["loss"]["postprocess_filter_commutativity_weight"]) * commute.loss
            + float(config["loss"]["luminance_underprediction_top_tail_weight"]) * under_tail
            + float(config["loss"]["metallic_boundary_halo_weight"]) * metallic_boundary_halo
            + float(config["loss"]["texel_center_baseline_reference_anchor_weight"]) * anchor_loss
        )
        if warmup_steps and step <= warmup_steps:
            total = 0.5 * anchor_baseline + 0.5 * subpixel_material
            phase = "latent_frozen_decoder_distillation"
        else:
            total = full_total
            phase = "joint"
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite filter-aware loss at step {step}")
        total.backward()
        if logits.requires_grad:
            latent_optimizer.step()
        decoder_optimizer.step()
        completed_step = step
        elapsed = time.monotonic() - started
        evaluation_due = step % evaluation_interval == 0
        timed_out = elapsed >= max_seconds
        record: dict[str, Any] = {
            "step": step,
            "phase": phase,
            "case": case_name,
            "elapsed_seconds": round(elapsed, 3),
            "total": float(total.detach().cpu()),
            "render_total": float(render_terms.total.detach().cpu()),
            "subpixel_reference_material": float(subpixel_material.detach().cpu()),
            "postprocess_filter_commutativity": float(commute.loss.detach().cpu()),
            "luminance_underprediction_top_tail": float(under_tail.detach().cpu()),
            "metallic_boundary_halo": float(metallic_boundary_halo.detach().cpu()),
            "texel_center_baseline_reference_anchor": float(anchor_loss.detach().cpu()),
            **{f"subpixel_{name}": float(value.detach().cpu()) for name, value in subpixel_terms.items()},
        }
        if evaluation_due or step == 1 or step % log_interval == 0 or timed_out:
            if evaluation_due or timed_out:
                record["selection_material"] = _selection_snapshot(
                    partitions["selection"],
                    targets,
                    deployed.detach(),
                    decoder,  # type: ignore[arg-type]
                    int(training["decode_chunk_size"]),
                )
                record["selection_render"] = _render_summary(
                    selection_cases, deployed.detach(), decoder, config  # type: ignore[arg-type]
                )
                if phase == "joint":
                    joint_evaluations += 1
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            print(record, flush=True)
        if evaluation_due or timed_out or step == max_steps:
            _atomic_save(
                _checkpoint_payload(
                    candidate=candidate_name,
                    step=step,
                    logits=logits,
                    decoder=decoder,
                    latent_optimizer=latent_optimizer,
                    decoder_optimizer=decoder_optimizer,
                    generator=generator,
                ),
                checkpoint_path,
            )
        if timed_out:
            break
    warmup_complete = completed_step >= warmup_steps
    valid = warmup_complete and joint_evaluations >= 1 and checkpoint_path.is_file()
    if completed_step == 0:
        raise RuntimeError("filter-aware time limit elapsed before the first step")
    final_latent = fake_quantize_unorm8(torch.sigmoid(logits).detach())
    latent_filename = f"latent_{candidate_name}_rgba_unorm8.png"
    latent_metadata = export_latent_unorm8_png(final_latent, output_dir / latent_filename)
    decoder_arrays = {
        name: value.detach().cpu().numpy() for name, value in decoder.state_dict().items()
    }
    np.savez(output_dir / "decoder_weights.npz", **decoder_arrays)
    final_mapping = _decode_mapping(
        final_latent, decoder, chunk_size=int(training["decode_chunk_size"])  # type: ignore[arg-type]
    )
    candidate_material = _material_report(reference_cpu, final_mapping, config)
    candidate_render = _render_summary(
        selection_cases, final_latent, decoder, config  # type: ignore[arg-type]
    )
    elapsed = time.monotonic() - started
    manifest = {
        "schema_version": 1,
        "candidate": candidate_name,
        "module_identifier": decoder.module_identifier,
        "status": "trained_pending_double_analysis" if valid else "invalid_training_run",
        "valid": valid,
        "formal_holdout_accessed": False,
        "initialization": {
            "latent_sha256": actual_hashes["latent_hard_png"],
            "decoder_sha256": actual_hashes["decoder_npz"],
            "checkpoint_family": "frozen_pre_qat_hard_only",
            "rejected_checkpoint_used": False,
            "step0_function_exact": step0_exact,
            "activation_swap_exact_initialization_claimed": candidate_name == "f_relu",
        },
        "cost": cost,
        "training": {
            "completed_step": completed_step,
            "max_steps": max_steps,
            "warmup_steps": warmup_steps,
            "warmup_complete": warmup_complete,
            "joint_evaluations": joint_evaluations,
            "elapsed_seconds": round(elapsed, 3),
            "max_minutes": float(training["max_minutes"]),
            "warmup_decoder_learning_rate": float(training["warmup_decoder_learning_rate"]),
            "joint_latent_learning_rate": float(training["joint_latent_learning_rate"]),
            "joint_decoder_learning_rate": float(training["joint_decoder_learning_rate"]),
            "evaluation_interval": evaluation_interval,
            "batch_mix": config["batch_mix"],
            "pool_sizes": {
                "uniform": int(uniform_np.size),
                "D2_D3_dark_hard": int(dark_np.size),
                "material_boundary_halo": int(boundary_np.size),
            },
            "case_partitions": case_partitions,
            "selection_or_validation_used_for_optimization": False,
            "checkpoint_contains_optimizer_rng_sampling_generator": True,
        },
        "loss": config["loss"],
        "baseline": {
            "texel_center_material": baseline_material,
            "repair_selection_render": baseline_render,
        },
        "candidate_metrics": {
            "texel_center_material": candidate_material,
            "repair_selection_render": candidate_render,
        },
        "files": {
            latent_filename: latent_metadata,
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
        "--config", type=Path, default=ROOT / "configs/train/scifihelmet_filter_aware.yaml"
    )
    args = parser.parse_args()
    result = run(args.config.resolve())
    print(
        json.dumps(
            {
                "candidate": result["candidate"],
                "status": result["status"],
                "valid": result["valid"],
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
