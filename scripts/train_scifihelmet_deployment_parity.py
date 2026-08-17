"""Train fresh DP-ReLU and ARC-ReLU candidates under the canonical v2 contract."""

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
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cg_frontier.assets.gltf_mesh import load_gltf_mesh  # noqa: E402
from cg_frontier.compression.artifact_analysis import (  # noqa: E402
    deterministic_json,
    deterministic_tile_partitions,
    sha256_file,
)
from cg_frontier.compression.deployment_parity import (  # noqa: E402
    DeploymentParityDecoder,
    activation_region_coherence,
    calculate_deployment_parity_cost,
    dark_envelope_loss,
    deployment_parity_sample,
    instantiate_fresh_candidate,
    make_fresh_initialization,
)
from cg_frontier.compression.deployment_parity_training import (  # noqa: E402
    deployment_parity_batch,
    learning_rates_at_step,
    validate_fixed_protocol,
)
from cg_frontier.compression.filter_aware import component_rectangularity  # noqa: E402
from cg_frontier.compression.material import (  # noqa: E402
    Core4Targets,
    DecodedMaterial,
    load_core4_targets,
    material_loss,
)
from cg_frontier.compression.render_loss import (  # noqa: E402
    export_latent_unorm8_png,
    fake_quantize_unorm8,
    pbr_render_loss,
)
from cg_frontier.compression.repair import top_fraction_mean  # noqa: E402
from cg_frontier.render.canonical_v2 import CanonicalRendererV2, LatentMaterialSource  # noqa: E402
from cg_frontier.render.gbuffer import GBufferResult, load_core4_textures  # noqa: E402
from train_scifihelmet_filter_aware import _boundary_pool, _focus_mask  # noqa: E402
from train_scifihelmet_interpolation_repair import (  # noqa: E402
    _load_config,
    _reference_textures,
    _sample_reference,
)
from train_scifihelmet_repair import (  # noqa: E402
    _atomic_save,
    _case_specs,
    _prepare_cases,
    _repo_path,
)


FORMAL_CANDIDATES = ("dp_relu_fresh", "arc_relu_fresh")


def _decoded_slice(value: DecodedMaterial, selected: slice) -> DecodedMaterial:
    return DecodedMaterial(
        value.base_color_linear[selected],
        value.normal_xy[selected],
        value.normal_xyz[selected],
        value.roughness[selected],
        value.metallic[selected],
    )


def _target_slice(value: Core4Targets, selected: slice) -> Core4Targets:
    return Core4Targets(
        value.base_color_linear[selected],
        value.normal_xyz[selected],
        value.roughness[selected],
        value.metallic[selected],
        height=1,
        width=value.base_color_linear[selected].shape[0],
    )


def _sampled_geometry(geometry: GBufferResult, positions: torch.Tensor) -> GBufferResult:
    mask = geometry.torch_buffers["mask"]
    valid_flat = torch.nonzero(mask.reshape(-1), as_tuple=False)[:, 0]
    absolute = valid_flat[positions]
    sampled: dict[str, torch.Tensor] = {}
    for name, value in geometry.torch_buffers.items():
        if value.shape[:2] != mask.shape:
            continue
        tail = value.shape[2:]
        sampled[name] = value.reshape(mask.numel(), *tail)[absolute].reshape(
            1, positions.numel(), *tail
        )
    sampled["mask"] = torch.ones(
        (1, positions.numel()), dtype=torch.bool, device=mask.device
    )
    return GBufferResult(buffers={}, torch_buffers=sampled, metadata={"sampled": True})


def _reference_dark_pool(targets: Any, optimizer_mask: np.ndarray, config: Mapping[str, Any]) -> np.ndarray:
    base = targets.base_color_linear.detach().cpu().numpy().reshape(
        targets.height, targets.width, 3
    )
    luma = np.sum(base * np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32), axis=-1)
    focus = _focus_mask(targets.height, targets.width, config["scopes"])
    pool = np.flatnonzero(
        optimizer_mask.reshape(-1) & focus.reshape(-1) & (luma.reshape(-1) > 0.05)
    ).astype(np.int64)
    if pool.size == 0:
        raise RuntimeError("reference-only D2/D3 dark-hard cell pool is empty")
    return pool


def _checkpoint_payload(
    *,
    candidate: str,
    step: int,
    latent: torch.Tensor,
    decoder: DeploymentParityDecoder,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    initialization_sha256: str,
    renderer_sha256: str,
    input_hashes: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "candidate": candidate,
        "step": step,
        "latent": latent.detach(),
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all(),
            "sampling_generator": generator.get_state(),
        },
        "initialization_sha256": initialization_sha256,
        "renderer_sha256": renderer_sha256,
        "input_hashes": dict(input_hashes),
    }


@torch.no_grad()
def _full_atlas_evaluation(
    latent: torch.Tensor,
    decoder: DeploymentParityDecoder,
    targets: Any,
    reference_textures: Mapping[str, torch.Tensor],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    height, width = targets.height, targets.width
    count = height * width
    chunk_size = int(config["training"]["decode_chunk_size"])
    reference_luma = np.empty(count, dtype=np.float32)
    runtime_luma = np.empty(count, dtype=np.float32)
    filtered_luma = np.empty(count, dtype=np.float32)
    commutativity = np.empty(count, dtype=np.float32)
    crossing_sum = 0.0
    crossing_count = 0
    luma_weights = latent.new_tensor((0.2126, 0.7152, 0.0722))
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        ids = torch.arange(start, stop, device=latent.device)
        x = ids.remainder(width).to(torch.float32) + 0.25
        y = torch.div(ids, width, rounding_mode="floor").to(torch.float32) + 0.25
        uv = torch.stack((x / float(width), y / float(height)), dim=-1)
        reference = _sample_reference(reference_textures, uv)
        sample = deployment_parity_sample(
            latent, uv, decoder, quantization="prequantized"
        )
        ref = torch.sum(reference.base_color_linear * luma_weights, dim=-1)
        runtime = torch.sum(sample.runtime.base_color_linear * luma_weights, dim=-1)
        filtered = torch.sum(
            sample.decode_then_filter.base_color_linear * luma_weights, dim=-1
        )
        reference_luma[start:stop] = ref.cpu().numpy()
        runtime_luma[start:stop] = runtime.cpu().numpy()
        filtered_luma[start:stop] = filtered.cpu().numpy()
        commutativity[start:stop] = torch.abs(runtime - filtered).cpu().numpy()
        region = activation_region_coherence(
            decoder,
            sample.corners,
            margin=float(config["loss"]["activation_region_margin"]),
        )
        crossing_sum += region.crossing_fraction * (stop - start)
        crossing_count += stop - start
    eligible = reference_luma > 0.05
    runtime_dark = runtime_luma < reference_luma * 0.5
    filtered_dark = filtered_luma < reference_luma * 0.5
    novel = eligible & runtime_dark & ~filtered_dark
    novel_map = novel.reshape(height, width)
    scopes: dict[str, Any] = {}
    for name, bbox in config["scopes"].items():
        x0, y0, x1, y1 = (int(value) for value in bbox)
        roi_eligible = eligible.reshape(height, width)[y0:y1, x0:x1]
        roi_novel = novel_map[y0:y1, x0:x1]
        components = component_rectangularity(roi_novel)
        scopes[name] = {
            "novel_dark_fraction": float(roi_novel[roi_eligible].mean())
            if np.any(roi_eligible)
            else 0.0,
            "components": components,
        }
    under = np.maximum(reference_luma - runtime_luma, 0.0)
    return {
        "phase_xy": [0.25, 0.25],
        "texel_count": count,
        "novel_dark_fraction": float(novel[eligible].mean()) if np.any(eligible) else 0.0,
        "luminance_underprediction_p99": float(np.quantile(under[eligible], 0.99)),
        "commutativity_luminance_p99": float(np.quantile(commutativity, 0.99)),
        "activation_crossing_fraction": crossing_sum / max(crossing_count, 1),
        "scopes": scopes,
    }


def _set_learning_rates(optimizer: torch.optim.Optimizer, step: int) -> tuple[float, float]:
    latent_lr, decoder_lr = learning_rates_at_step(step)
    optimizer.param_groups[0]["lr"] = latent_lr
    optimizer.param_groups[1]["lr"] = decoder_lr
    return latent_lr, decoder_lr


def _preflight_path(config: Mapping[str, Any], candidate: str) -> Path:
    first_output = _repo_path(
        config["candidates"]["dp_relu_fresh"]["output_dir"],
        "candidates.dp_relu_fresh.output_dir",
    )
    batch_size = int(config["training"]["batch_size"])
    implementation = str(config["training"]["optimizer_implementation"])
    return first_output.parent / f"preflight_{candidate}_b{batch_size}_{implementation}.json"


def run(config_path: Path, *, candidate_name: str, preflight_only: bool) -> dict[str, Any]:
    config = _load_config(config_path)
    validate_fixed_protocol(config)
    if config.get("schema_version") != 1 or config.get("experiment") != "deployment_parity_arc_v1":
        raise ValueError("unsupported deployment-parity training config")
    if candidate_name not in config["candidates"]:
        raise ValueError(f"unsupported deployment-parity candidate: {candidate_name}")
    if "formal_holdout" in config_path.as_posix().lower():
        raise ValueError("sealed formal holdout is forbidden")
    if not torch.cuda.is_available():
        raise RuntimeError("deployment-parity training requires CUDA")
    if candidate_name == "arc12_diagnostic" and not preflight_only:
        eligibility = _preflight_path(config, "arc12_diagnostic").with_name(
            "arc12_diagnostic_eligibility.json"
        )
        if not eligibility.is_file() or not json.loads(eligibility.read_text(encoding="utf-8")).get(
            "eligible", False
        ):
            raise RuntimeError("ARC-12 diagnostic is conditional and has no signed eligibility decision")
    training = config["training"]
    if int(training["max_steps"]) != 120_000 or float(training["max_minutes"]) != 120.0:
        raise ValueError("training budget differs from the approved protocol")
    device = torch.device("cuda")
    seed = int(training["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generator = torch.Generator(device=device).manual_seed(seed)

    inputs = config["inputs"]
    input_paths = {
        name: _repo_path(value, f"inputs.{name}") for name, value in inputs.items()
    }
    input_hashes = {
        "gltf": sha256_file(input_paths["gltf"]),
        "core4_manifest": sha256_file(input_paths["core4_manifest"]),
    }
    renderer_path = ROOT / "src/cg_frontier/render/canonical_v2.py"
    renderer_sha256 = sha256_file(renderer_path)
    targets = load_core4_targets(input_paths["core4_dir"], device)
    reference_textures = _reference_textures(targets)
    initialization = make_fresh_initialization(
        height=targets.height,
        width=targets.width,
        decoder_width=int(config["candidates"][candidate_name]["decoder_width"]),
        seed=seed,
    )
    fresh = instantiate_fresh_candidate(
        initialization, candidate=candidate_name, device=device
    )
    latent, decoder = fresh.latent, fresh.decoder
    cost = calculate_deployment_parity_cost(decoder)
    if (
        training.get("optimizer") != "Adam"
        or training.get("optimizer_implementation") != "fused_cuda_direct_unorm"
    ):
        raise ValueError("deployment-parity training requires fused CUDA Adam on direct UNORM latent")
    optimizer = torch.optim.Adam(
        [
            {"params": [latent], "lr": 0.0},
            {"params": list(decoder.parameters()), "lr": 0.0},
        ],
        fused=True,
    )
    partitions = deterministic_tile_partitions(
        targets.height,
        targets.width,
        tile_size=64,
        seed=int(training["split_seed"]),
    )
    uniform_np = np.flatnonzero(partitions["optimizer"].reshape(-1)).astype(np.int64)
    dark_np = _reference_dark_pool(targets, partitions["optimizer"], config)
    boundary_np = _boundary_pool(targets, partitions["optimizer"], config)
    uniform_pool = torch.from_numpy(uniform_np).to(device)
    dark_pool = torch.from_numpy(dark_np).to(device)
    boundary_pool = torch.from_numpy(boundary_np).to(device)
    mesh = load_gltf_mesh(input_paths["gltf"])
    textures = load_core4_textures(input_paths["core4_manifest"], device)
    specs, case_partitions = _case_specs(config)
    optimizer_cases = _prepare_cases(
        config, case_partitions["optimizer"], specs, mesh, textures, device
    )
    renderer = CanonicalRendererV2(
        display_exposure=float(config["render"]["display_exposure"]),
        minimum_roughness=float(config["render"]["minimum_roughness"]),
    )
    output_dir = _repo_path(
        config["candidates"][candidate_name]["output_dir"],
        f"candidates.{candidate_name}.output_dir",
    )
    if preflight_only:
        output_dir = (
            output_dir.parent
            / f"preflight_b{int(training['batch_size'])}"
            / str(training["optimizer_implementation"])
            / candidate_name
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.jsonl"
    checkpoint_path = output_dir / "checkpoint.pt"
    if log_path.exists():
        raise FileExistsError(f"refusing to inherit an existing candidate run: {log_path}")
    if not preflight_only:
        preflight = _preflight_path(config, candidate_name)
        if not preflight.is_file():
            raise RuntimeError("a successful 1000-step performance preflight is required")
        preflight_value = json.loads(preflight.read_text(encoding="utf-8"))
        if (
            not preflight_value.get("passed")
            or preflight_value.get("initialization_sha256") != initialization.sha256
            or int(preflight_value.get("batch_size", -1)) != int(training["batch_size"])
            or preflight_value.get("optimizer_implementation")
            != training["optimizer_implementation"]
        ):
            raise RuntimeError("performance preflight is missing, failed, or uses another initialization")
        if candidate_name == "arc_relu_fresh":
            dp_manifest = _repo_path(
                config["candidates"]["dp_relu_fresh"]["output_dir"],
                "candidates.dp_relu_fresh.output_dir",
            ) / "training_manifest.json"
            if not dp_manifest.is_file():
                raise RuntimeError("formal candidates must run serially: DP-ReLU is not complete")

    max_steps = int(training["preflight_steps"] if preflight_only else training["max_steps"])
    hard_stop_seconds = float(training["max_minutes"]) * 60.0
    batch_size = int(training["batch_size"])
    luma_weights = torch.tensor((0.2126, 0.7152, 0.0722), device=device)
    started = time.monotonic()
    completed_step = 0
    last_full_evaluation: dict[str, Any] | None = None
    for step in range(1, max_steps + 1):
        latent_lr, decoder_lr = _set_learning_rates(optimizer, step)
        optimizer.zero_grad(set_to_none=True)
        deployed = fake_quantize_unorm8(latent)
        case_name, geometry, camera, light, reference_hdr = optimizer_cases[
            (step - 1) % len(optimizer_cases)
        ]
        screen_uv = geometry.torch_buffers["uv"][geometry.torch_buffers["mask"]]
        batch = deployment_parity_batch(
            screen_uv=screen_uv,
            uniform_pool=uniform_pool,
            dark_pool=dark_pool,
            boundary_pool=boundary_pool,
            batch_size=batch_size,
            height=targets.height,
            width=targets.width,
            generator=generator,
        )
        target = _sample_reference(reference_textures, batch.uv)
        sample = deployment_parity_sample(
            deployed, batch.uv, decoder, quantization="prequantized"
        )
        anchor_slice = batch.slices["texel_center_reference_anchor"]
        subpixel_slice = slice(0, anchor_slice.start)
        subpixel_material, subpixel_terms = material_loss(
            _decoded_slice(sample.runtime, subpixel_slice),
            _target_slice(target, subpixel_slice),
            {name: float(value) for name, value in config["material_channels"].items()},
        )
        anchor_material, _ = material_loss(
            _decoded_slice(sample.runtime, anchor_slice),
            _target_slice(target, anchor_slice),
            {name: float(value) for name, value in config["material_channels"].items()},
        )
        target_luma = torch.sum(target.base_color_linear * luma_weights, dim=-1)
        runtime_luma = torch.sum(sample.runtime.base_color_linear * luma_weights, dim=-1)
        dark_slice = batch.slices["dark_hard_cell"]
        dark_target = target_luma[dark_slice]
        dark_runtime = runtime_luma[dark_slice]
        eligible = dark_target > 0.05
        if not torch.any(eligible):
            raise RuntimeError("dark-hard batch contains no bright reference samples")
        dark_tail = top_fraction_mean(
            torch.relu(dark_target[eligible] - dark_runtime[eligible]),
            float(training["underprediction_top_fraction"]),
        )
        boundary_slice = batch.slices["material_boundary_halo_cell"]
        boundary_halo = 0.5 * (
            F.l1_loss(sample.runtime.metallic[boundary_slice], target.metallic[boundary_slice])
            + F.l1_loss(runtime_luma[boundary_slice], target_luma[boundary_slice])
        )
        envelope = dark_envelope_loss(sample.runtime, sample.decode_then_filter)
        region = activation_region_coherence(
            decoder,
            sample.corners,
            margin=float(config["loss"]["activation_region_margin"]),
        )
        sampled_geometry = _sampled_geometry(geometry, batch.screen_positions)
        candidate_bundle = renderer.render(
            sampled_geometry,
            camera,
            light,
            LatentMaterialSource(deployed, decoder, quantization="prequantized"),
            input_hashes=input_hashes,
        )
        reference_screen = reference_hdr[geometry.torch_buffers["mask"]][
            batch.screen_positions
        ].reshape(1, -1, 3)
        render_terms = pbr_render_loss(
            reference_screen,
            candidate_bundle.linear_hdr,
            candidate_bundle.coverage,
            subpixel_material,
            charbonnier_epsilon=1.0e-3,
            charbonnier_weight=1.0,
            log1p_weight=0.25,
            material_weight=0.25,
        )
        loss = config["loss"]
        total = (
            render_terms.total
            + float(loss["subpixel_reference_material"]) * subpixel_material
            + float(loss["postprocess_filter_commutativity"])
            * sample.postprocess_commutativity_l1
            + float(loss["luminance_underprediction_top_tail"]) * dark_tail
            + float(loss["metallic_boundary_halo"]) * boundary_halo
            + float(loss["texel_center_reference_anchor"]) * anchor_material
            + float(loss["dark_envelope"]) * envelope
            + float(config["candidates"][candidate_name]["activation_region_weight"])
            * region.loss
        )
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite deployment-parity loss at step {step}")
        total.backward()
        optimizer.step()
        with torch.no_grad():
            latent.clamp_(0.0, 1.0)
        completed_step = step
        elapsed = time.monotonic() - started
        record = {
            "step": step,
            "case": case_name,
            "elapsed_seconds": round(elapsed, 3),
            "latent_lr": latent_lr,
            "decoder_lr": decoder_lr,
            "total": float(total.detach().cpu()),
            "render_total": float(render_terms.total.detach().cpu()),
            "subpixel_reference_material": float(subpixel_material.detach().cpu()),
            "postprocess_filter_commutativity": float(
                sample.postprocess_commutativity_l1.detach().cpu()
            ),
            "luminance_underprediction_top_tail": float(dark_tail.detach().cpu()),
            "metallic_boundary_halo": float(boundary_halo.detach().cpu()),
            "texel_center_reference_anchor": float(anchor_material.detach().cpu()),
            "dark_envelope": float(envelope.detach().cpu()),
            "activation_region": float(region.loss.detach().cpu()),
            "activation_crossing_fraction": region.crossing_fraction,
            **{
                f"subpixel_{name}": float(value.detach().cpu())
                for name, value in subpixel_terms.items()
            },
        }
        fixed_probe_due = step % int(training["fixed_probe_interval"]) == 0
        full_due = not preflight_only and step % int(training["full_atlas_evaluation_interval"]) == 0
        if full_due:
            last_full_evaluation = _full_atlas_evaluation(
                fake_quantize_unorm8(latent).detach(),
                decoder,
                targets,
                reference_textures,
                config,
            )
            record["full_atlas_evaluation"] = last_full_evaluation
        if step == 1 or step % int(training["log_interval"]) == 0 or fixed_probe_due or full_due:
            with log_path.open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
            print(
                json.dumps(
                    {
                        "candidate": candidate_name,
                        "step": step,
                        "elapsed_seconds": round(elapsed, 3),
                        "total": record["total"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        if fixed_probe_due or full_due or step == max_steps:
            _atomic_save(
                _checkpoint_payload(
                    candidate=candidate_name,
                    step=step,
                    latent=latent,
                    decoder=decoder,
                    optimizer=optimizer,
                    generator=generator,
                    initialization_sha256=initialization.sha256,
                    renderer_sha256=renderer_sha256,
                    input_hashes=input_hashes,
                ),
                checkpoint_path,
            )

    elapsed = time.monotonic() - started
    if preflight_only:
        projected = elapsed / max(completed_step, 1) * int(training["max_steps"])
        result = {
            "schema_version": 1,
            "candidate": candidate_name,
            "passed": bool(completed_step == int(training["preflight_steps"])),
            "completed_steps": completed_step,
            "elapsed_seconds": round(elapsed, 3),
            "projected_full_seconds": round(projected, 3),
            "budget_seconds": hard_stop_seconds,
            "time_budget_enforced": bool(training["enforce_time_limit"]),
            "batch_size": batch_size,
            "optimizer_implementation": training["optimizer_implementation"],
            "initialization_sha256": initialization.sha256,
            "renderer_sha256": renderer_sha256,
            "formal_holdout_accessed": False,
        }
        path = _preflight_path(config, candidate_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(deterministic_json(result), encoding="utf-8", newline="\n")
        return result

    valid = completed_step == int(training["max_steps"]) and checkpoint_path.is_file()
    final_latent = fake_quantize_unorm8(latent.detach())
    latent_name = f"latent_{candidate_name}_rgba_unorm8.png"
    latent_metadata = export_latent_unorm8_png(final_latent, output_dir / latent_name)
    decoder_path = output_dir / "decoder_weights.npz"
    np.savez(
        decoder_path,
        **{
            name: value.detach().cpu().numpy()
            for name, value in decoder.state_dict().items()
        },
    )
    manifest = {
        "schema_version": 1,
        "candidate": candidate_name,
        "status": "trained_pending_double_analysis" if valid else "invalid_training_run",
        "valid": valid,
        "formal_holdout_accessed": False,
        "initialization": {
            "kind": "shared_fresh_random_state",
            "latent_parameterization": "direct_unorm_projected_after_adam",
            "seed": seed,
            "sha256": initialization.sha256,
            "legacy_or_rejected_checkpoint_used": False,
        },
        "renderer": {
            "identifier": renderer.renderer_identifier,
            "source_sha256": renderer_sha256,
            "deployment_order": "RGBA8_STE->one_bilinear->decoder->postprocess->shared_PBR",
        },
        "cost": cost,
        "training": {
            "completed_steps": completed_step,
            "required_steps": int(training["max_steps"]),
            "elapsed_seconds": round(elapsed, 3),
            "max_minutes_advisory": float(training["max_minutes"]),
            "time_limit_enforced": bool(training["enforce_time_limit"]),
            "batch_size": batch_size,
            "batch_mix": config["batch_mix"],
            "schedule": training["schedule"],
            "optimizer": "Adam",
            "optimizer_implementation": training["optimizer_implementation"],
            "dtype": "float32",
            "amp": False,
            "checkpoint_contains_optimizer_rng_sampling_generator": True,
            "case_partitions": case_partitions,
            "selection_or_formal_holdout_used_for_optimization": False,
        },
        "loss": config["loss"],
        "last_full_atlas_evaluation": last_full_evaluation,
        "input_hashes": input_hashes,
        "files": {
            latent_name: latent_metadata,
            "decoder_weights.npz": sha256_file(decoder_path),
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
        default=ROOT / "configs/train/scifihelmet_deployment_parity_arc_v1.yaml",
    )
    parser.add_argument("--candidate", choices=(*FORMAL_CANDIDATES, "arc12_diagnostic"), required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    result = run(
        args.config.resolve(), candidate_name=args.candidate, preflight_only=args.preflight_only
    )
    print(
        json.dumps(
            {
                "candidate": args.candidate,
                "preflight_only": args.preflight_only,
                "valid_or_passed": result.get("valid", result.get("passed", False)),
                "formal_holdout_accessed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
