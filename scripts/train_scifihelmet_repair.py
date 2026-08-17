"""Run bounded R1/R2 SciFiHelmet repair candidates from the frozen hard baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch
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
    roi_mask,
    roi_material_metrics,
    sha256_file,
    tail_statistics,
)
from cg_frontier.compression.material import (  # noqa: E402
    Core4Targets,
    MaterialDecoder,
    decode_material,
    load_core4_targets,
    material_loss,
)
from cg_frontier.compression.render_loss import (  # noqa: E402
    display_transform,
    export_latent_unorm8_png,
    fake_quantize_unorm8,
    latent_float_to_logits,
    load_latent_unorm8_png,
    masked_render_metrics,
    orbit_camera,
    pbr_render_loss,
    render_latent_material,
)
from cg_frontier.compression.repair import (  # noqa: E402
    MetallicResidualDecoder,
    R1_COST,
    R2_COST,
    deterministic_case_partitions,
    evaluate_acceptance_gates,
    hard_example_indices,
    stratified_batch_indices,
    top_fraction_mean,
)
from cg_frontier.render.gbuffer import (  # noqa: E402
    load_core4_textures,
    render_geometry_gbuffer,
    sample_core4_material,
)
from cg_frontier.render.pbr import PointLight, shade_ggx  # noqa: E402


EXPECTED_LATENT_SHA256 = "5a1781afc1a877be452a87a3d958e48cab921b45237faebf2be3668a60ae5fdc"
EXPECTED_DECODER_SHA256 = "d676ade8294600eb0064a835eabfe86d4d35e39ee787d512574fbef8d7346baa"


def _repo_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"config field {label} must be a path")
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"config field {label} escapes the repository")
    if "formal_holdout" in path.as_posix().lower():
        raise ValueError(f"sealed formal holdout path is forbidden: {label}")
    return path


def _decoder_from_npz(path: Path, device: torch.device) -> MaterialDecoder:
    decoder = MaterialDecoder("tiny_mlp").to(device)
    with np.load(path, allow_pickle=False) as stored:
        decoder.load_state_dict(
            {name: torch.from_numpy(np.asarray(stored[name])).to(device) for name in stored.files}
        )
    return decoder


def _atomic_save(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _camera(spec: Mapping[str, Any], render: Mapping[str, Any]):
    return orbit_camera(
        yaw_degrees=float(spec["yaw_degrees"]),
        elevation_degrees=float(spec["elevation_degrees"]),
        radius=float(render["camera_radius"]),
        target=tuple(float(value) for value in render["target"]),
        up=tuple(float(value) for value in render["up"]),
        vertical_fov_degrees=float(render["vertical_fov_degrees"]),
        near=float(render["near"]),
        far=float(render["far"]),
    )


def _light(spec: Mapping[str, Any]) -> PointLight:
    return PointLight(
        position=tuple(float(value) for value in spec["position"]),
        color=tuple(float(value) for value in spec["color"]),
        radiant_intensity=float(spec["radiant_intensity"]),
        ambient_intensity=float(spec["ambient_intensity"]),
    )


def _case_specs(config: Mapping[str, Any]) -> tuple[dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]], dict[str, list[str]]]:
    cases = {
        f"{camera['name']}__{light['name']}": (camera, light)
        for camera in config["train_cameras"]
        for light in config["train_lights"]
    }
    partitions = deterministic_case_partitions(
        list(cases), seed=int(config["training"]["split_seed"])
    )
    return cases, partitions


def _prepare_cases(
    config: Mapping[str, Any],
    names: list[str],
    specs: Mapping[str, tuple[Mapping[str, Any], Mapping[str, Any]]],
    mesh: Any,
    textures: Any,
    device: torch.device,
) -> list[tuple[str, Any, Any, Any, torch.Tensor]]:
    render = config["render"]
    resolution = tuple(int(value) for value in render["resolution"])
    minimum_roughness = float(render["minimum_roughness"])
    geometry_by_camera: dict[str, tuple[Any, Any]] = {}
    result: list[tuple[str, Any, Any, Any, torch.Tensor]] = []
    print(f"prepare cases: {len(names)} cases", flush=True)
    for case_index, name in enumerate(names, start=1):
        camera_spec, light_spec = specs[name]
        camera_name = str(camera_spec["name"])
        if camera_name not in geometry_by_camera:
            print(f"  rasterize camera {camera_name}", flush=True)
            camera = _camera(camera_spec, render)
            geometry = render_geometry_gbuffer(
                mesh,
                camera,
                resolution,
                device=device,
                cull_backfaces=True,
            )
            geometry_by_camera[camera_name] = (camera, geometry)
        camera, geometry = geometry_by_camera[camera_name]
        light = _light(light_spec)
        with torch.no_grad():
            reference = shade_ggx(
                geometry,
                camera,
                light,
                material_override=sample_core4_material(geometry, textures),
                minimum_roughness=minimum_roughness,
            ).detach()
        result.append((name, geometry, camera, light, reference))
        if case_index == 1 or case_index % 5 == 0 or case_index == len(names):
            print(f"  prepared {case_index}/{len(names)}: {name}", flush=True)
    return result


@torch.no_grad()
def _decode_mapping(
    latent: torch.Tensor,
    decoder: torch.nn.Module,
    *,
    chunk_size: int,
) -> dict[str, np.ndarray]:
    height, width = latent.shape[:2]
    flat = latent.reshape(-1, 4)
    pieces: dict[str, list[np.ndarray]] = {name: [] for name in ("base_color_linear", "normal_xyz", "roughness_linear", "metallic_linear")}
    for start in range(0, flat.shape[0], chunk_size):
        decoded = decode_material(decoder, flat[start : start + chunk_size])
        pieces["base_color_linear"].append(decoded.base_color_linear.cpu().numpy())
        pieces["normal_xyz"].append(decoded.normal_xyz.cpu().numpy())
        pieces["roughness_linear"].append(decoded.roughness[:, 0].cpu().numpy())
        pieces["metallic_linear"].append(decoded.metallic[:, 0].cpu().numpy())
    return {
        "base_color_linear": np.concatenate(pieces["base_color_linear"]).reshape(height, width, 3),
        "normal_xyz": np.concatenate(pieces["normal_xyz"]).reshape(height, width, 3),
        "roughness_linear": np.concatenate(pieces["roughness_linear"]).reshape(height, width),
        "metallic_linear": np.concatenate(pieces["metallic_linear"]).reshape(height, width),
    }


def _reference_mapping(targets: Core4Targets) -> dict[str, np.ndarray]:
    return {
        "base_color_linear": targets.base_color_linear.detach().cpu().numpy().reshape(targets.height, targets.width, 3),
        "normal_xyz": targets.normal_xyz.detach().cpu().numpy().reshape(targets.height, targets.width, 3),
        "roughness_linear": targets.roughness.detach().cpu().numpy().reshape(targets.height, targets.width),
        "metallic_linear": targets.metallic.detach().cpu().numpy().reshape(targets.height, targets.width),
    }


@torch.no_grad()
def _render_summary(
    cases: list[tuple[str, Any, Any, Any, torch.Tensor]],
    latent: torch.Tensor,
    decoder: torch.nn.Module,
    config: Mapping[str, Any],
) -> dict[str, float | int]:
    maes: list[float] = []
    ssims: list[float] = []
    for _, geometry, camera, light, reference in cases:
        candidate, _ = render_latent_material(
            geometry,
            camera,
            light,
            latent,
            decoder,  # type: ignore[arg-type]
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
    return {
        "case_count": len(cases),
        "hdr_mae": float(np.mean(maes)),
        "display_ssim": float(np.mean(ssims)),
    }


def _material_report(
    reference: Mapping[str, np.ndarray],
    candidate: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    errors = material_error_maps(reference, candidate)
    edges = metallic_boundary_mask(reference["metallic_linear"], 0.1)
    rois = {
        name: roi_material_metrics(
            reference,
            candidate,
            roi_mask(reference["metallic_linear"].shape, bbox),
            metallic_edges=edges,
        )
        for name, bbox in config["rois_xyxy"].items()
    }
    return {
        "global": {name: tail_statistics(values) for name, values in errors.items()},
        "metallic_boundary": {
            **tail_statistics(errors["metallic"], edges),
            "fraction_above_0_1": float(np.mean(errors["metallic"][edges] > 0.1)),
        },
        "rois": rois,
    }


def _selection_snapshot(
    selection_mask: np.ndarray,
    targets: Core4Targets,
    latent: torch.Tensor,
    decoder: torch.nn.Module,
    chunk_size: int,
) -> dict[str, float]:
    reference = _reference_mapping(targets)
    candidate = _decode_mapping(latent, decoder, chunk_size=chunk_size)
    errors = material_error_maps(reference, candidate)
    return {
        "base_p99": float(np.percentile(errors["base_color_max_channel"][selection_mask], 99.0)),
        "normal_p95": float(np.percentile(errors["normal_degrees"][selection_mask], 95.0)),
        "roughness_mae": float(np.mean(errors["roughness"][selection_mask])),
        "metallic_boundary_proxy_mae": float(np.mean(errors["metallic"][selection_mask])),
    }


def run(config_path: Path, candidate_name: str) -> dict[str, Any]:
    print(f"repair start: candidate={candidate_name} config={config_path}", flush=True)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping) or config.get("schema_version") != 1:
        raise ValueError("unsupported repair config schema")
    if candidate_name not in ("r1", "r2"):
        raise ValueError("candidate must be r1 or r2")
    inputs = config["inputs"]
    latent_path = _repo_path(inputs["latent_hard_png"], "inputs.latent_hard_png")
    decoder_path = _repo_path(inputs["decoder_npz"], "inputs.decoder_npz")
    print("validate frozen input hashes", flush=True)
    if sha256_file(latent_path) != EXPECTED_LATENT_SHA256 or sha256_file(decoder_path) != EXPECTED_DECODER_SHA256:
        raise ValueError("frozen repair initialization hash mismatch")
    output_root = _repo_path(config["output_dir"], "output_dir")
    output_dir = output_root / candidate_name
    output_dir.mkdir(parents=True, exist_ok=True)
    if candidate_name == "r2":
        r1_gate_path = output_root / "r1" / "gates.json"
        if not r1_gate_path.is_file():
            raise RuntimeError("R2 requires completed R1 gates")
        r1_gates = json.loads(r1_gate_path.read_text(encoding="utf-8"))
        if not r1_gates.get("r2_eligible_metallic_only_failure", False):
            raise RuntimeError("R2 is forbidden because R1 was not a metallic-only failure")

    print("initialize CUDA and frozen latent/decoder", flush=True)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("repair training requires the existing CUDA environment")
    seed = int(config["training"]["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    _, hard_latent = load_latent_unorm8_png(latent_path, device=device)
    print("  latent loaded", flush=True)
    base_decoder = _decoder_from_npz(decoder_path, device)
    print("  decoder loaded", flush=True)
    if candidate_name == "r1":
        decoder: torch.nn.Module = base_decoder
        initial_latent = hard_latent
    else:
        r1_dir = output_root / "r1"
        _, initial_latent = load_latent_unorm8_png(r1_dir / "latent_repair_rgba_unorm8.png", device=device)
        base_decoder = _decoder_from_npz(r1_dir / "decoder_weights.npz", device)
        decoder = MetallicResidualDecoder(base_decoder).to(device)

    print("load Core-4 targets", flush=True)
    targets = load_core4_targets(_repo_path(inputs["core4_dir"], "inputs.core4_dir"), device)
    print("  Core-4 targets loaded", flush=True)
    print("initialize trainable latent and optimizers", flush=True)
    logits = torch.nn.Parameter(
        latent_float_to_logits(initial_latent, clamp_epsilon=float(config["training"]["latent_clamp_epsilon"]))
    )
    latent_optimizer = torch.optim.Adam([logits], lr=float(config["training"]["latent_learning_rate"]))
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=float(config["training"]["decoder_learning_rate"]))
    generator = torch.Generator(device=device).manual_seed(seed)

    print("prepare material reference and frozen hard pools", flush=True)
    reference_cpu = _reference_mapping(targets)
    baseline_mapping = _decode_mapping(hard_latent, base_decoder, chunk_size=int(config["training"]["decode_chunk_size"]))
    baseline_errors = material_error_maps(reference_cpu, baseline_mapping)
    partitions = deterministic_tile_partitions(
        targets.height,
        targets.width,
        tile_size=64,
        seed=int(config["training"]["split_seed"]),
    )
    optimizer_flat = partitions["optimizer"].reshape(-1)
    base_pool_np = hard_example_indices(
        baseline_errors["base_color_max_channel"], optimizer_flat, top_fraction=0.10
    )
    edges = metallic_boundary_mask(reference_cpu["metallic_linear"], 0.1).reshape(-1)
    metal_high = hard_example_indices(baseline_errors["metallic"], optimizer_flat, top_fraction=0.10)
    metal_pool_np = np.union1d(np.flatnonzero(edges & optimizer_flat), metal_high).astype(np.int64)
    optimizer_indices = torch.from_numpy(np.flatnonzero(optimizer_flat).astype(np.int64)).to(device)
    base_pool = torch.from_numpy(base_pool_np).to(device)
    metallic_pool = torch.from_numpy(metal_pool_np).to(device)
    edge_flat = torch.from_numpy(edges).to(device)

    print("prepare render cases from original train partition", flush=True)
    mesh = load_gltf_mesh(_repo_path(inputs["gltf"], "inputs.gltf"))
    textures = load_core4_textures(_repo_path(inputs["core4_manifest"], "inputs.core4_manifest"), device)
    case_specs, case_partitions = _case_specs(config)
    optimizer_cases = _prepare_cases(config, case_partitions["optimizer"], case_specs, mesh, textures, device)
    selection_limit = int(config["training"]["selection_render_case_limit"])
    selection_cases = _prepare_cases(
        config, case_partitions["selection"][:selection_limit], case_specs, mesh, textures, device
    )
    baseline_render = _render_summary(selection_cases, hard_latent, base_decoder, config)

    training = config["training"]
    warmup_steps = int(training["warmup_steps"])
    max_steps = int(training["max_steps"])
    max_seconds = float(training["max_minutes"]) * 60.0
    batch_size = int(training["batch_size"])
    chunk_size = int(training["decode_chunk_size"])
    log_path = output_dir / "train.jsonl"
    checkpoint_path = output_dir / "checkpoint.pt"
    started = time.monotonic()
    completed_step = 0
    if candidate_name == "r1":
        for parameter in decoder.parameters():
            parameter.requires_grad_(False)
    else:
        logits.requires_grad_(False)
        for parameter in base_decoder.parameters():
            parameter.requires_grad_(False)
    for step in range(1, max_steps + 1):
        if step == warmup_steps + 1:
            logits.requires_grad_(True)
            for parameter in decoder.parameters():
                parameter.requires_grad_(True)
        latent_optimizer.zero_grad(set_to_none=True)
        decoder_optimizer.zero_grad(set_to_none=True)
        bounded = torch.sigmoid(logits)
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
        batch, slices = stratified_batch_indices(
            targets.texel_count,
            optimizer_indices,
            base_pool,
            metallic_pool,
            batch_size=batch_size,
            generator=generator,
        )
        deployed = fake_quantize_unorm8(bounded).reshape(-1, 4)
        decoded = decode_material(decoder, deployed[batch])
        uniform_slice = slices["uniform"]
        global_material, global_terms = material_loss(
            type(decoded)(
                decoded.base_color_linear[uniform_slice],
                decoded.normal_xy[uniform_slice],
                decoded.normal_xyz[uniform_slice],
                decoded.roughness[uniform_slice],
                decoded.metallic[uniform_slice],
            ),
            targets.select(batch[uniform_slice]),
            {"base_color_l1": 1.0, "normal_cosine": 1.0, "roughness_l1": 0.5, "metallic_l1": 0.5},
        )
        base_values = torch.max(
            torch.abs(decoded.base_color_linear[slices["base"]] - targets.select(batch[slices["base"]]).base_color_linear),
            dim=-1,
        ).values
        metallic_values = torch.abs(
            decoded.metallic[slices["metallic"]] - targets.select(batch[slices["metallic"]]).metallic
        )[:, 0]
        metallic_batch_ids = batch[slices["metallic"]]
        edge_selection = edge_flat[metallic_batch_ids]
        edge_l1 = metallic_values[edge_selection].mean() if torch.any(edge_selection) else metallic_values.mean()
        base_tail = top_fraction_mean(base_values, 0.05)
        metallic_tail = top_fraction_mean(metallic_values, 0.05)
        render_terms = pbr_render_loss(
            reference_hdr,
            candidate_hdr,
            geometry.torch_buffers["mask"],
            global_material,
            charbonnier_epsilon=1.0e-3,
            charbonnier_weight=1.0,
            log1p_weight=0.25,
            material_weight=0.25,
        )
        total = render_terms.total + 0.25 * base_tail + 0.25 * metallic_tail + 0.25 * edge_l1
        total.backward()
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite repair loss at step {step}")
        if logits.requires_grad:
            latent_optimizer.step()
        decoder_optimizer.step()
        completed_step = step
        elapsed = time.monotonic() - started
        record = {
            "step": step,
            "phase": "warmup" if step <= warmup_steps else "joint",
            "case": case_name,
            "elapsed_seconds": round(elapsed, 3),
            "total": float(total.detach().cpu()),
            "render_charbonnier": float(render_terms.charbonnier_hdr.detach().cpu()),
            "render_log1p": float(render_terms.log1p_hdr.detach().cpu()),
            "global_material": float(global_material.detach().cpu()),
            "base_top5": float(base_tail.detach().cpu()),
            "metallic_top5": float(metallic_tail.detach().cpu()),
            "metallic_edge_l1": float(edge_l1.detach().cpu()),
            **{f"global_{name}": float(value.detach().cpu()) for name, value in global_terms.items()},
        }
        evaluation_due = step % int(training["evaluation_interval"]) == 0
        timed_out = elapsed >= max_seconds
        if step == 1 or step % int(training["log_interval"]) == 0 or evaluation_due or timed_out:
            if evaluation_due or timed_out:
                record["selection_material"] = _selection_snapshot(
                    partitions["selection"], targets, fake_quantize_unorm8(bounded.detach()), decoder, chunk_size
                )
                record["selection_render"] = _render_summary(
                    selection_cases, fake_quantize_unorm8(bounded.detach()), decoder, config
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

    final_latent = fake_quantize_unorm8(torch.sigmoid(logits).detach())
    latent_metadata = export_latent_unorm8_png(final_latent, output_dir / "latent_repair_rgba_unorm8.png")
    decoder_arrays = {name: value.detach().cpu().numpy() for name, value in decoder.state_dict().items()}
    np.savez(output_dir / "decoder_weights.npz", **decoder_arrays)
    final_mapping = _decode_mapping(final_latent, decoder, chunk_size=chunk_size)
    candidate_report = _material_report(reference_cpu, final_mapping, config)
    candidate_report["render_selection"] = _render_summary(selection_cases, final_latent, decoder, config)
    baseline_analysis = json.loads(
        _repo_path(inputs["baseline_analysis"], "inputs.baseline_analysis").read_text(encoding="utf-8")
    )["candidates"]["hard"]
    baseline_report = {
        "global": baseline_analysis["global"],
        "metallic_boundary": baseline_analysis["metallic_boundary"],
        "rois": baseline_analysis["rois"],
        "render_selection": baseline_render,
    }
    gates = evaluate_acceptance_gates(baseline_report, candidate_report)
    gates["ue_visual_pass"] = False
    gates["final_pass"] = False
    (output_dir / "gates.json").write_text(deterministic_json(gates), encoding="utf-8", newline="\n")
    cost = R1_COST if candidate_name == "r1" else R2_COST
    manifest = {
        "schema_version": 1,
        "candidate": candidate_name,
        "status": "offline_pass_ue_pending" if gates["offline_pass"] else "offline_rejected",
        "formal_holdout_accessed": False,
        "initialization": {
            "latent_sha256": EXPECTED_LATENT_SHA256,
            "decoder_sha256": EXPECTED_DECODER_SHA256,
            "checkpoint_family": "frozen_pre_qat_hard_only",
        },
        "cost": cost,
        "training": {
            "completed_step": completed_step,
            "max_steps": max_steps,
            "warmup_steps": warmup_steps,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "max_minutes": float(training["max_minutes"]),
            "batch_mix": {"uniform": 0.50, "base_hard": 0.25, "metallic_edge_or_hard": 0.25},
            "case_partitions": case_partitions,
            "tile_partition_pixels": {name: int(mask.sum()) for name, mask in partitions.items()},
        },
        "loss": {
            "base_top5_weight": 0.25,
            "metallic_top5_weight": 0.25,
            "metallic_boundary_l1_weight": 0.25,
        },
        "files": {
            "latent_repair_rgba_unorm8.png": latent_metadata,
            "decoder_weights.npz": sha256_file(output_dir / "decoder_weights.npz"),
            "checkpoint.pt": sha256_file(checkpoint_path),
            "gates.json": sha256_file(output_dir / "gates.json"),
        },
        "baseline": baseline_report,
        "candidate_metrics": candidate_report,
        "gates": gates,
        "deployment_exported": False,
    }
    (output_dir / "repair_manifest.json").write_text(
        deterministic_json(manifest), encoding="utf-8", newline="\n"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train/scifihelmet_repair.yaml")
    parser.add_argument("--candidate", choices=("r1", "r2"), default="r1")
    args = parser.parse_args()
    result = run(args.config.resolve(), args.candidate)
    print(json.dumps({"candidate": args.candidate, "status": result["status"], "gates": result["gates"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
