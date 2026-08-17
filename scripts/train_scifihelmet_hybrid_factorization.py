"""Train one bounded causal Hybrid candidate without checkpoint inheritance."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from cg_frontier.assets.gltf_mesh import load_gltf_mesh  # noqa: E402
from cg_frontier.compression.artifact_analysis import (  # noqa: E402
    deterministic_json,
    deterministic_tile_partitions,
    material_error_maps,
    metallic_boundary_mask,
    sha256_file,
)
from cg_frontier.compression.hybrid import decode_auxiliary, export_hybrid_textures, render_hybrid_material  # noqa: E402
from cg_frontier.compression.hybrid_factorization import candidate_aux_channels, decoder_for_candidate  # noqa: E402
from cg_frontier.compression.material import load_core4_targets, material_loss  # noqa: E402
from cg_frontier.compression.render_loss import (  # noqa: E402
    bilinear_sample_top_down_wrap,
    fake_quantize_unorm8,
    hard_quantize_unorm8,
    latent_float_to_logits,
    pbr_render_loss,
)
from cg_frontier.compression.repair import hard_example_indices  # noqa: E402
from cg_frontier.render.gbuffer import load_core4_textures  # noqa: E402
import train_scifihelmet_hybrid as base  # noqa: E402


def _wall_clock_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _checkpoint_payload(
    *,
    candidate: str,
    step: int,
    elapsed_seconds: float,
    logits: torch.Tensor,
    decoder: torch.nn.Module,
    latent_optimizer: torch.optim.Optimizer,
    decoder_optimizer: torch.optim.Optimizer,
    batch_generator: torch.Generator,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "candidate": candidate,
        "step": int(step),
        "training_loop_elapsed_seconds": round(float(elapsed_seconds), 3),
        "checkpoint_inherited": False,
        "auxiliary_logits": logits.detach(),
        "decoder": decoder.state_dict(),
        "latent_optimizer": latent_optimizer.state_dict(),
        "decoder_optimizer": decoder_optimizer.state_dict(),
        "torch_cpu_rng_state": torch.get_rng_state(),
        "torch_cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "batch_generator_state": batch_generator.get_state(),
    }


def _load_initialization(
    texture_a_path: Path,
    texture_b_path: Path,
    decoder_path: Path,
    candidate: str,
    device: torch.device,
):
    channels = candidate_aux_channels(candidate)
    a = np.asarray(Image.open(texture_a_path).convert("RGBA"), dtype=np.uint8)
    mode = {3: "LA", 4: "RGB"}[channels]
    b = np.asarray(Image.open(texture_b_path).convert(mode), dtype=np.uint8)
    if b.ndim == 2:
        b = b[..., None]
    direct = torch.from_numpy(a[..., :3].copy()).to(device=device, dtype=torch.float32) / 255.0
    auxiliary = torch.from_numpy(np.concatenate((a[..., 3:4], b), axis=-1).copy()).to(
        device=device, dtype=torch.float32
    ) / 255.0
    decoder = decoder_for_candidate(candidate).to(device)
    with np.load(decoder_path, allow_pickle=False) as stored:
        decoder.load_state_dict({name: torch.from_numpy(np.asarray(stored[name])).to(device) for name in stored.files})
    return direct, auxiliary, decoder


def _write_analysis_config(
    config: dict[str, Any],
    candidate: str,
    output_dir: Path,
    texture_a: Path,
    texture_b: Path,
    decoder: Path,
) -> Path:
    phase0_config = yaml.safe_load((ROOT / "configs/eval/scifihelmet_hybrid_factorization_phase0.yaml").read_text(encoding="utf-8"))
    channels = candidate_aux_channels(candidate)
    payload = {
        "schema_version": 1,
        "inputs": {
            "core4_dir": config["inputs"]["core4_dir"],
            "texture_a_png": texture_a.relative_to(ROOT).as_posix(),
            "texture_b_png": texture_b.relative_to(ROOT).as_posix(),
            "decoder_npz": decoder.relative_to(ROOT).as_posix(),
        },
        "frozen_sha256": {
            "texture_a_png": sha256_file(texture_a),
            "texture_b_png": sha256_file(texture_b),
            "decoder_npz": sha256_file(decoder),
        },
        "representation": {
            "candidate": candidate,
            "architecture": candidate,
            "texture_a": "RGBA8_linear_BaseColor_RGB_plus_aux0",
            "texture_b": "RG8_aux12" if channels == 3 else "logical_RGB8_aux123",
            "aux_channels": channels,
            "logical_channels": 3 + channels,
            "theoretical_raw_bytes_no_mips": 2048 * 2048 * (3 + channels),
            "physical_ceiling_bytes": 2048 * 2048 * (4 + (2 if channels == 3 else 4)),
            "texture_samples": 2,
        },
        "analysis": phase0_config["analysis"],
        "scopes": phase0_config["scopes"],
        "output_dir": (output_dir / "analysis").relative_to(ROOT).as_posix(),
    }
    path = output_dir / "analysis_config.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8", newline="\n")
    return path


def run(config_path: Path, candidate: str) -> dict[str, Any]:
    if candidate not in ("d6_s", "d6_h", "d6_p", "d7_p"):
        raise ValueError("unsupported causal Hybrid candidate")
    config = base._load_config(config_path)
    inputs = config["inputs"]
    paths = {
        name: base._repo_path(inputs[name], f"inputs.{name}")
        for name in ("gltf", "core4_manifest", "core4_dir", "baseline_latent", "baseline_decoder", "phase0_summary", "initialization_report")
    }
    baseline_hashes = {
        "baseline_latent": sha256_file(paths["baseline_latent"]),
        "baseline_decoder": sha256_file(paths["baseline_decoder"]),
    }
    if baseline_hashes != dict(config["frozen_baseline_sha256"]):
        raise ValueError("frozen baseline hash mismatch")
    phase0 = json.loads(paths["phase0_summary"].read_text(encoding="utf-8"))
    initialization_report = json.loads(paths["initialization_report"].read_text(encoding="utf-8"))
    if phase0.get("gpu_training_allowed") is not True or phase0.get("formal_holdout_accessed") is not False:
        raise RuntimeError("causal Hybrid Phase 0 does not allow GPU training")
    if candidate == "d7_p":
        trigger_path = base._repo_path(f"{config['output_root']}/d7_trigger.json", "d7_trigger")
        if not trigger_path.is_file() or json.loads(trigger_path.read_text(encoding="utf-8")).get("d7_allowed") is not True:
            raise RuntimeError("D7-P is forbidden because the frozen trigger is false or absent")
    file_info = initialization_report["files"][candidate]
    init_paths = {name: base._repo_path(file_info[name]["path"], f"initialization.{name}") for name in ("texture_a", "texture_b", "decoder")}
    actual_init_hashes = {name: sha256_file(path) for name, path in init_paths.items()}
    expected_init_hashes = {name: file_info[name]["sha256"] for name in init_paths}
    if actual_init_hashes != expected_init_hashes:
        raise ValueError("causal Hybrid initialization hash mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("causal Hybrid training requires the existing CUDA environment")
    device = torch.device("cuda")
    training = config["training"]
    seed = int(training["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device=device).manual_seed(seed)
    direct, initial_auxiliary, decoder = _load_initialization(
        init_paths["texture_a"], init_paths["texture_b"], init_paths["decoder"], candidate, device
    )
    expected_cost = initialization_report["candidates"][candidate]["cost"]
    actual_cost = {
        "parameters": decoder.parameter_count,
        "weight_bytes_float32": decoder.weight_bytes_float32,
        "macs_per_pixel": decoder.macs_per_pixel,
    }
    if actual_cost != expected_cost:
        raise RuntimeError(f"causal Hybrid decoder cost mismatch: {actual_cost} != {expected_cost}")
    logits = torch.nn.Parameter(latent_float_to_logits(initial_auxiliary, clamp_epsilon=float(training["latent_clamp_epsilon"])))
    logits.requires_grad_(False)
    latent_optimizer = torch.optim.Adam([logits], lr=float(training["latent_learning_rate"]))
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=float(training["decoder_learning_rate"]))
    targets = load_core4_targets(paths["core4_dir"], device)
    references = base._reference_textures(targets)
    reference_cpu = base._reference_mapping(targets)
    partitions = deterministic_tile_partitions(targets.height, targets.width, tile_size=64, seed=int(training["split_seed"]))
    initial_mapping = base._decode_mapping(direct, initial_auxiliary, decoder, int(training["decode_chunk_size"]))
    initial_errors = material_error_maps(reference_cpu, initial_mapping)
    optimizer_mask = partitions["optimizer"]
    composite = initial_errors["normal_degrees"] / 180.0 + initial_errors["roughness"]
    uniform_np = np.flatnonzero(optimizer_mask.reshape(-1)).astype(np.int64)
    hard_np = hard_example_indices(composite, optimizer_mask, top_fraction=float(training["hard_pool_top_fraction"]))
    edges = metallic_boundary_mask(reference_cpu["metallic_linear"], 0.1)
    metallic_np = np.flatnonzero((edges & optimizer_mask).reshape(-1)).astype(np.int64)
    if min(uniform_np.size, hard_np.size, metallic_np.size) == 0:
        raise RuntimeError("causal Hybrid optimizer-only pool is empty")
    uniform, hard, metallic = (torch.from_numpy(value).to(device) for value in (uniform_np, hard_np, metallic_np))

    mesh = load_gltf_mesh(paths["gltf"])
    textures = load_core4_textures(paths["core4_manifest"], device)
    case_specs, case_partitions = base._case_specs(config)
    optimizer_cases = base._prepare_cases(config, case_partitions["optimizer"], case_specs, mesh, textures, device)
    selection_cases = base._prepare_cases(config, case_partitions["selection"][: int(training["selection_render_case_limit"])], case_specs, mesh, textures, device)
    _, baseline_latent = base.load_latent_unorm8_png(paths["baseline_latent"], device=device)
    baseline_decoder = base._decoder_from_npz(paths["baseline_decoder"], device)
    baseline_render = base._render_summary(selection_cases, baseline_latent, baseline_decoder, config)
    baseline_mapping = base._baseline_decode_mapping(baseline_latent, baseline_decoder, chunk_size=int(training["decode_chunk_size"]))
    baseline_material = base._material_report(reference_cpu, baseline_mapping, config)

    output_dir = base._repo_path(f"{config['output_root']}/{candidate}", "output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path, checkpoint_path = output_dir / "train.jsonl", output_dir / "checkpoint.pt"
    training_started_at = _wall_clock_iso()
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
        _, uv, metallic_slice = base._subpixel_batch(
            uniform, hard, metallic, batch_size=int(training["batch_size"]),
            height=targets.height, width=targets.width, generator=generator,
        )
        target = base._sample_reference(references, uv)
        texture_a = torch.cat((direct.detach(), auxiliary[..., 0:1]), dim=-1)
        sampled_a = bilinear_sample_top_down_wrap(texture_a, uv)
        sampled_b = bilinear_sample_top_down_wrap(auxiliary[..., 1:], uv)
        prediction = decode_auxiliary(decoder, torch.cat((sampled_a[..., 3:4], sampled_b), dim=-1), sampled_a[..., :3])
        subpixel_aux, sub_terms = material_loss(prediction, target, weights)
        metallic_boundary = F.l1_loss(prediction.metallic[metallic_slice], target.metallic[metallic_slice])
        texel_ids = base._draw(uniform, int(training["batch_size"]) // 2, generator)
        texel_prediction = decode_auxiliary(decoder, auxiliary.reshape(-1, auxiliary.shape[-1])[texel_ids], direct.reshape(-1, 3)[texel_ids])
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
            raise FloatingPointError(f"non-finite causal Hybrid loss at step {step}")
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
                "step": step, "phase": "decoder_only" if step <= warmup else "joint", "case": case_name,
                "elapsed_seconds": round(elapsed, 3), "total": float(total.detach().cpu()),
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
                mapping = base._decode_mapping(direct, hard_aux, decoder, int(training["decode_chunk_size"]))
                errors_now = material_error_maps(reference_cpu, mapping)
                record["selection_material"] = {
                    "base_p99": float(np.quantile(errors_now["base_color_max_channel"][partitions["selection"]], 0.99)),
                    "normal_p95": float(np.quantile(errors_now["normal_degrees"][partitions["selection"]], 0.95)),
                    "roughness_mae": float(np.mean(errors_now["roughness"][partitions["selection"]])),
                    "metallic_mae": float(np.mean(errors_now["metallic"][partitions["selection"]])),
                }
                record["selection_render"] = base._hybrid_render_summary(selection_cases, direct, hard_aux, decoder, config)
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            print(record, flush=True)
        if due or timed_out or step == int(training["max_steps"]):
            base._atomic_save(
                _checkpoint_payload(
                    candidate=candidate,
                    step=step,
                    elapsed_seconds=elapsed,
                    logits=logits,
                    decoder=decoder,
                    latent_optimizer=latent_optimizer,
                    decoder_optimizer=decoder_optimizer,
                    batch_generator=generator,
                ),
                checkpoint_path,
            )
        if timed_out:
            break
    if completed_step <= warmup:
        raise RuntimeError("causal Hybrid candidate did not complete warmup plus one joint step")
    training_loop_elapsed = time.monotonic() - started
    termination_reason = "max_steps" if completed_step == int(training["max_steps"]) else "time_limit"
    final_aux = hard_quantize_unorm8(torch.sigmoid(logits).detach())
    texture_a_path = output_dir / "texture_a_base_rgb_aux0_rgba8.png"
    texture_b_path = output_dir / ("texture_b_aux12_rg8.png" if final_aux.shape[-1] == 3 else "texture_b_aux123_rgb8.png")
    packing = export_hybrid_textures(direct, final_aux, texture_a_path, texture_b_path)
    decoder_path = output_dir / "decoder_weights.npz"
    np.savez(decoder_path, **{name: value.detach().cpu().numpy() for name, value in decoder.state_dict().items()})
    final_mapping = base._decode_mapping(direct, final_aux, decoder, int(training["decode_chunk_size"]))
    candidate_material = base._material_report(reference_cpu, final_mapping, config)
    candidate_render = base._hybrid_render_summary(selection_cases, direct, final_aux, decoder, config)
    analysis_config_path = _write_analysis_config(config, candidate, output_dir, texture_a_path, texture_b_path, decoder_path)
    manifest = {
        "schema_version": 1, "candidate": candidate, "status": "trained_pending_full_interpolation_gates",
        "formal_holdout_accessed": False,
        "initialization": {"family": initialization_report["candidates"][candidate]["initializer"], "checkpoint_inherited": False, "sha256": actual_init_hashes},
        "cost": {**actual_cost, "logical_raw_bytes": packing["logical_raw_bytes"], "physical_ceiling_bytes": 2048 * 2048 * (6 if final_aux.shape[-1] == 3 else 8), "texture_samples": 2},
        "training": {
            "completed_step": completed_step, "warmup_steps": warmup, "max_steps": int(training["max_steps"]),
            "elapsed_seconds": round(training_loop_elapsed, 3),
            "training_loop_elapsed_seconds": round(training_loop_elapsed, 3),
            "total_wall_elapsed_seconds": round(time.monotonic() - started, 3),
            "started_at": training_started_at,
            "finished_at": _wall_clock_iso(),
            "termination_reason": termination_reason,
            "max_minutes": float(training["max_minutes"]),
            "latent_learning_rate": float(training["latent_learning_rate"]), "decoder_learning_rate": float(training["decoder_learning_rate"]),
            "evaluation_interval": int(training["evaluation_interval"]),
            "batch_mix": {"uniform_subpixel": 0.5, "normal_roughness_hard": 0.25, "metallic_boundary": 0.25},
            "selection_or_validation_used_for_optimization": False,
        },
        "loss": config["loss"],
        "baseline": {"texel_center_material": baseline_material, "repair_selection_render": baseline_render},
        "candidate_metrics": {"texel_center_material": candidate_material, "repair_selection_render": candidate_render},
        "files": {
            "texture_a": {"path": texture_a_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(texture_a_path)},
            "texture_b": {"path": texture_b_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(texture_b_path)},
            "decoder": {"path": decoder_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(decoder_path)},
            "checkpoint": sha256_file(checkpoint_path), "checkpoint_schema_version": 2,
            "train_log": sha256_file(log_path),
            "analysis_config": {"path": analysis_config_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(analysis_config_path)},
        },
        "deployment_exported": False,
    }
    manifest_path = output_dir / "training_manifest.json"
    manifest_path.write_text(deterministic_json(manifest), encoding="utf-8", newline="\n")
    print(json.dumps({"candidate": candidate, "completed_step": completed_step, "elapsed_seconds": manifest["training"]["elapsed_seconds"], "formal_holdout_accessed": False}, ensure_ascii=False))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train/scifihelmet_hybrid_factorization.yaml")
    parser.add_argument("--candidate", choices=("d6_s", "d6_h", "d6_p", "d7_p"), required=True)
    args = parser.parse_args()
    run(args.config.resolve(), args.candidate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
