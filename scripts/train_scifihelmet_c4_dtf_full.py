"""Train a routed fresh SciFiHelmet C4/C5 DTF candidate for 80k steps."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
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
    sha256_file,
)
from cg_frontier.compression.decode_then_filter import (  # noqa: E402
    DecodeThenFilterDecoder,
    decode_then_filter_sample,
    export_decode_then_filter_latent_unorm8,
    instantiate_paired_precheck_candidate,
    make_paired_precheck_initialization,
)
from cg_frontier.compression.decode_then_filter_training import (  # noqa: E402
    build_decode_then_filter_camera31_fresh_manifest,
    build_decode_then_filter_camera_finetune_manifest,
    build_decode_then_filter_continuation_manifest,
    build_decode_then_filter_manifest,
    camera_finetune_learning_rates_at_step,
    continuation_learning_rates_at_step,
    decode_then_filter_batch,
    full_training_learning_rates_at_step,
    full_training_objective,
    full_training_phase_at_step,
    material_subset_metrics,
    paired_precheck_objective,
    restore_decode_then_filter_resume_state,
    resolve_dtf_camera_spec,
    select_explicit_dtf_camera_pool,
    select_camera_triangle_coverage,
    validate_decode_then_filter_resume_payload,
    validate_decode_then_filter_protocol,
)
from cg_frontier.compression.filter_aware import (  # noqa: E402
    bilinear_corners_top_down_wrap_torch,
    component_rectangularity,
    postprocess_raw_torch,
)
from cg_frontier.compression.material import (  # noqa: E402
    Core4Targets,
    DecodedMaterial,
    keep_system_awake,
    load_core4_targets,
)
from cg_frontier.compression.render_loss import (  # noqa: E402
    decoded_to_material,
    fake_quantize_unorm8,
    masked_render_metrics,
    orbit_camera,
)
from cg_frontier.render.decode_then_filter import (  # noqa: E402
    DTFLatentMaterialSource,
    DTFReferenceMaterialSource,
    DecodeThenFilterRenderer,
)
from cg_frontier.render.gbuffer import (  # noqa: E402
    GBufferResult,
    load_core4_textures,
    render_geometry_gbuffer,
    sample_core4_material,
)
from cg_frontier.render.pbr import PointLight  # noqa: E402
from train_scifihelmet_c4_dtf_precheck import (  # noqa: E402
    _atomic_save,
    _evaluate_precheck,
    _generic_high_gradient_pool,
    _reference_textures,
    _repo_path,
    _sample_reference,
)


SUPPORTED_CANDIDATES = ("c4_dtf_16_selected", "c5_dtf_16")


def _camera(spec: Mapping[str, Any], render: Mapping[str, Any]):
    resolved = resolve_dtf_camera_spec(spec, render)
    return orbit_camera(
        yaw_degrees=resolved["yaw_degrees"],
        elevation_degrees=resolved["elevation_degrees"],
        radius=resolved["radius"],
        target=resolved["target"],
        up=resolved["up"],
        vertical_fov_degrees=resolved["vertical_fov_degrees"],
        near=resolved["near"],
        far=resolved["far"],
    )


def _light(spec: Mapping[str, Any]) -> PointLight:
    return PointLight(
        position=tuple(float(value) for value in spec["position"]),
        color=tuple(float(value) for value in spec["color"]),
        radiant_intensity=float(spec["radiant_intensity"]),
        ambient_intensity=float(spec["ambient_intensity"]),
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


def _decoded_target(target: Core4Targets) -> DecodedMaterial:
    return DecodedMaterial(
        target.base_color_linear.reshape(1, -1, 3),
        target.normal_xyz[:, :2].reshape(1, -1, 2),
        target.normal_xyz.reshape(1, -1, 3),
        target.roughness.reshape(1, -1, 1),
        target.metallic.reshape(1, -1, 1),
    )


def _checkpoint_payload(
    *,
    candidate: str,
    step: int,
    latent: torch.Tensor,
    decoder: DecodeThenFilterDecoder,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    initialization_sha256: str,
    activation_selection_sha256: str,
    input_hashes: Mapping[str, str],
    camera_selection: Mapping[str, Any],
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
        "activation_selection_sha256": activation_selection_sha256,
        "input_hashes": dict(input_hashes),
        "camera_selection": dict(camera_selection),
    }


def _prepare_camera_pool(
    config: Mapping[str, Any],
    mesh: Any,
    device: torch.device,
) -> tuple[list[tuple[str, Any, GBufferResult]], dict[str, Any]]:
    render = config["render"]
    specs = list(config["train_cameras"])
    camera_limit = int(config["training_pool"]["camera_limit"])
    if not (1 <= len(specs) <= camera_limit <= 48):
        raise ValueError("DTF camera candidates exceed the frozen pool limit")
    resolution = tuple(int(value) for value in render["resolution"])
    candidates: list[tuple[str, Any, GBufferResult]] = []
    triangle_ids: list[torch.Tensor] = []
    selection: dict[str, Any] | None = None
    strategy = str(
        config["training_pool"].get(
            "camera_selection_strategy", "triangle_increment_v1"
        )
    )
    if strategy not in {"triangle_increment_v1", "explicit_audited_pool_v1"}:
        raise ValueError(f"unsupported DTF camera selection strategy: {strategy}")
    for spec in specs:
        camera = _camera(spec, render)
        geometry = render_geometry_gbuffer(
            mesh,
            camera,
            resolution,
            device=device,
            cull_backfaces=True,
        )
        ids = torch.unique(geometry.torch_buffers["triangle_id"])
        ids = ids[ids >= 0]
        triangle_ids.append(ids)
        candidates.append((str(spec["name"]), camera, geometry))
        if strategy == "triangle_increment_v1":
            selection = select_camera_triangle_coverage(
                triangle_ids,
                triangle_count=int(mesh.triangles.shape[0]),
                increment_stop=float(
                    config["training_pool"][
                        "camera_triangle_coverage_increment_stop"
                    ]
                ),
                camera_limit=camera_limit,
            )
            if len(selection["selected_indices"]) < len(triangle_ids):
                triangle_ids.pop()
                candidates.pop()
                break
    if strategy == "explicit_audited_pool_v1":
        selection = select_explicit_dtf_camera_pool(
            triangle_ids,
            camera_names=[name for name, _, _ in candidates],
            triangle_count=int(mesh.triangles.shape[0]),
            camera_limit=camera_limit,
        )
    if not candidates or selection is None:
        raise RuntimeError("DTF camera coverage selection produced no camera")
    selection = {
        **selection,
        "strategy": strategy,
        "selected_names": [name for name, _, _ in candidates],
        "candidate_count_evaluated": len(selection["records"]),
        "increment_stop": float(
            config["training_pool"]["camera_triangle_coverage_increment_stop"]
        ),
    }
    return candidates, selection


@torch.no_grad()
def _selection_evaluation(
    *,
    latent: torch.Tensor,
    decoder: DecodeThenFilterDecoder,
    reference_textures: Mapping[str, torch.Tensor],
    core4_textures: Any,
    cameras: list[tuple[str, Any, GBufferResult]],
    lights: list[tuple[str, PointLight]],
    renderer: DecodeThenFilterRenderer,
    config: Mapping[str, Any],
    input_hashes: Mapping[str, str],
) -> dict[str, Any]:
    material = _evaluate_precheck(
        latent,
        decoder,
        reference_textures,
        sample_count=int(config["training"]["selection_evaluation_samples"]),
        seed=int(config["training"]["seed"]) + 777,
    )
    render_metrics: list[dict[str, float | int | str]] = []
    dark_fractions: list[float] = []
    halo_fractions: list[float] = []
    rectangle_max_area = 0
    luma = latent.new_tensor((0.2126, 0.7152, 0.0722))
    camera_limit = int(config["training"]["selection_render_camera_limit"])
    deployed = fake_quantize_unorm8(latent).detach()
    for _, camera, geometry in cameras[:camera_limit]:
        reference_material = sample_core4_material(geometry, core4_textures)
        for _, light in lights:
            reference = renderer.render(
                geometry,
                camera,
                light,
                DTFReferenceMaterialSource(reference_material),
                input_hashes=input_hashes,
            )
            candidate = renderer.render(
                geometry,
                camera,
                light,
                DTFLatentMaterialSource(deployed, decoder, quantization="prequantized"),
                input_hashes=input_hashes,
            )
            metrics = masked_render_metrics(
                reference.linear_hdr,
                candidate.linear_hdr,
                geometry.torch_buffers["mask"],
                linear_psnr_data_range=float(config["render"]["linear_psnr_data_range"]),
                display_exposure=float(config["render"]["display_exposure"]),
            )
            render_metrics.append(metrics)
            reference_luma = torch.sum(reference.display_rgb * luma, dim=-1)
            candidate_luma = torch.sum(candidate.display_rgb * luma, dim=-1)
            eligible = geometry.torch_buffers["mask"] & (reference_luma > 0.05)
            dark = eligible & (candidate_luma < reference_luma * 0.5)
            halo = eligible & (candidate_luma > reference_luma + 0.05)
            dark_fractions.append(
                float(dark[eligible].to(torch.float32).mean().cpu())
                if torch.any(eligible)
                else 0.0
            )
            halo_fractions.append(
                float(halo[eligible].to(torch.float32).mean().cpu())
                if torch.any(eligible)
                else 0.0
            )
            components = component_rectangularity(dark.cpu().numpy())
            rectangle_max_area = max(
                rectangle_max_area,
                int(components["rectangular_component_max_area"]),
            )
    mean_hdr = float(
        np.mean([float(value["masked_linear_hdr_mae"]) for value in render_metrics])
    )
    mean_ssim = float(np.mean([float(value["display_ssim"]) for value in render_metrics]))
    material_error = (
        float(material["base_color_linear_mae"])
        + math.radians(float(material["normal_mean_degrees"]))
        + float(material["roughness_mae"])
        + float(material["metallic_mae"])
    )
    return {
        "render": {
            "case_count": len(render_metrics),
            "multi_light_hdr_mae": mean_hdr,
            "display_ssim": mean_ssim,
            "material_error": material_error,
        },
        "artifact": {
            "obvious_rectangular_black_blocks": rectangle_max_area >= 16,
            "rectangular_component_max_area": rectangle_max_area,
            "novel_dark_fraction": float(np.mean(dark_fractions)),
            "halo_fraction": float(np.mean(halo_fractions)),
            "metallic_boundary_proxy_mae": float(material["metallic_mae"]),
        },
        "material": material,
    }


@torch.no_grad()
def _yellow_material_evaluation(
    *,
    latent: torch.Tensor,
    decoder: DecodeThenFilterDecoder,
    targets: Core4Targets,
) -> dict[str, float | int]:
    """Track yellow-pipe texels without changing sampling or the loss."""

    linear = targets.base_color_linear
    srgb = torch.where(
        linear <= 0.0031308,
        12.92 * linear,
        1.055 * torch.pow(linear.clamp_min(0.0), 1.0 / 2.4) - 0.055,
    )
    yellow = (
        (srgb[:, 0] > 0.12)
        & (srgb[:, 1] > 0.06)
        & (srgb[:, 0] > srgb[:, 1] * 1.06)
        & (srgb[:, 1] > srgb[:, 2] * 1.22)
    )
    indices = torch.nonzero(yellow, as_tuple=False).reshape(-1)
    if indices.numel() == 0:
        raise RuntimeError("yellow evaluation subset is empty")
    deployed = fake_quantize_unorm8(latent).detach().reshape(-1, latent.shape[-1])
    prediction = postprocess_raw_torch(decoder(deployed[indices]))
    reference = targets.select(indices)
    return material_subset_metrics(
        reference,
        prediction,
        torch.ones(indices.numel(), dtype=torch.bool, device=indices.device),
    )


def _best_render_score(evaluation: Mapping[str, Any]) -> tuple[float, float, float]:
    render = evaluation["render"]
    return (
        float(render["multi_light_hdr_mae"]),
        -float(render["display_ssim"]),
        float(render["material_error"]),
    )


def _artifact_score(evaluation: Mapping[str, Any]) -> tuple[float, float, float] | None:
    artifact = evaluation["artifact"]
    if bool(artifact["obvious_rectangular_black_blocks"]):
        return None
    return (
        float(artifact["novel_dark_fraction"]),
        float(artifact["halo_fraction"]),
        float(artifact["metallic_boundary_proxy_mae"]),
    )


def recover_logged_checkpoint_selection(
    source_selection: Mapping[str, Any],
    logged_records: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Recover dual-track best metadata after an interrupted resume run."""

    best_render = dict(source_selection.get("best_render") or {}) or None
    best_artifact = dict(source_selection.get("best_artifact_safe") or {}) or None
    for record in logged_records:
        evaluation = record.get("selection_evaluation")
        if not isinstance(evaluation, Mapping):
            continue
        step = int(record["step"])
        render_score = list(_best_render_score(evaluation))
        if best_render is None or tuple(render_score) < tuple(best_render["score"]):
            best_render = {"step": step, "score": render_score}
        artifact_score = _artifact_score(evaluation)
        if artifact_score is not None and (
            best_artifact is None
            or artifact_score < tuple(best_artifact["score"])
        ):
            best_artifact = {"step": step, "score": list(artifact_score)}
    return {
        "best_render": best_render,
        "best_artifact_safe": best_artifact,
    }


def summarize_resume_training_result(
    *,
    planned_start_step: int,
    segment_start_step: int,
    completed_step: int,
    required_step: int,
    segment_elapsed_seconds: float,
) -> dict[str, Any]:
    """Report full lineage progress separately from the final process segment."""

    if not (
        0 <= planned_start_step <= segment_start_step <= completed_step <= required_step
    ):
        raise ValueError("invalid DTF resume training-result step ordering")
    return {
        "completed_steps": completed_step,
        "required_steps": required_step,
        "source_step": planned_start_step,
        "resume_segment_start_step": segment_start_step,
        "steps_executed": completed_step - planned_start_step,
        "segment_steps_executed": completed_step - segment_start_step,
        "elapsed_seconds": round(segment_elapsed_seconds, 3),
        "elapsed_scope": "final_process_segment",
    }


def run(
    config_path: Path,
    *,
    candidate_name: str,
    preflight_only: bool,
    continuation_stop_step: int | None = None,
    camera_finetune: bool = False,
    camera31_fresh: bool = False,
) -> dict[str, Any]:
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    if not isinstance(config, Mapping):
        raise ValueError("DTF config must be a mapping")
    validate_decode_then_filter_protocol(config)
    expected_experiment = (
        "scifihelmet_c4_dtf_camera31_v1"
        if camera_finetune or camera31_fresh
        else "scifihelmet_c4_dtf_v1"
    )
    if config.get("schema_version") != 1 or config.get("experiment") != expected_experiment:
        raise ValueError("unsupported DTF full-training config")
    if not torch.cuda.is_available():
        raise RuntimeError("DTF full training requires CUDA")
    if candidate_name not in SUPPORTED_CANDIDATES:
        raise ValueError(f"unsupported DTF full candidate: {candidate_name}")
    approved_continuations = {
        ("c5_dtf_16", 120_000): "c5_dtf_16_resume_120k",
        ("c4_dtf_16_selected", 160_000): "c4_dtf_16_resume_160k",
    }
    if sum(bool(value) for value in (camera_finetune, camera31_fresh, continuation_stop_step is not None)) > 1:
        raise ValueError("DTF continuation, camera fine-tune, and camera31 fresh are mutually exclusive")
    is_continuation = continuation_stop_step is not None
    is_resume_training = is_continuation or camera_finetune
    tracks_yellow_material = is_resume_training or camera31_fresh
    continuation_output_name: str | None = None
    if is_continuation:
        try:
            continuation_output_name = approved_continuations[
                (candidate_name, continuation_stop_step)
            ]
        except KeyError as exc:
            raise ValueError("unsupported DTF continuation candidate/endpoint") from exc
    finetune_profile = dict(config.get("camera_finetune", {}))
    if camera_finetune:
        if candidate_name != "c4_dtf_16_selected":
            raise ValueError("camera fine-tune is approved only for C4-DTF-16")
        if (
            finetune_profile.get("source_step") != 160_000
            or finetune_profile.get("stop_step") != 180_000
            or finetune_profile.get("exact_continuation") is not False
        ):
            raise ValueError("invalid C4 camera fine-tune profile")
    if camera31_fresh and candidate_name != "c4_dtf_16_selected":
        raise ValueError("camera31 fresh training is approved only for C4-DTF-16")
    candidate_config = config["candidates"][candidate_name]
    if (
        int(candidate_config["max_steps"]) != 80_000
        or candidate_config["activation"] != "selected_precheck_winner"
    ):
        raise ValueError("DTF full candidate differs from the frozen 80k protocol")
    if len(config["train_lights"]) > int(config["training_pool"]["light_limit"]):
        raise ValueError("DTF light candidates exceed the frozen pool limit")
    selection_path = _repo_path(
        "outputs/compression/scifihelmet/c4_dtf_v1/activation_selection.json",
        "activation_selection",
    )
    if not selection_path.is_file():
        raise FileNotFoundError("paired activation-selection evidence is required")
    activation_selection = json.loads(selection_path.read_text(encoding="utf-8"))
    activation = str(activation_selection.get("selected_activation"))
    if activation != "relu":
        raise ValueError("current signed paired selection does not choose ReLU")
    activation_selection_sha256 = sha256_file(selection_path)
    capacity_route_sha256: str | None = None
    if candidate_name == "c5_dtf_16":
        route_path = _repo_path(
            "outputs/compression/scifihelmet/c4_dtf_v1/capacity_route.json",
            "capacity_route",
        )
        if not route_path.is_file():
            raise FileNotFoundError("C5 requires signed capacity-route evidence")
        route = json.loads(route_path.read_text(encoding="utf-8"))
        if (
            route.get("significant_capacity_benefit") is not False
            or route.get("next_candidate") != candidate_name
            or route.get("c5_is_channel_upper_bound") is not True
        ):
            raise RuntimeError("capacity route does not authorize fresh C5 training")
        capacity_route_sha256 = sha256_file(route_path)

    training = config["training"]
    seed = int(training["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)

    inputs = {
        name: _repo_path(value, f"inputs.{name}")
        for name, value in config["inputs"].items()
    }
    input_hashes = {
        name: sha256_file(path) for name, path in inputs.items() if path.is_file()
    }
    input_hashes.update(
        {
            "dtf_decoder_source": sha256_file(
                ROOT / "src/cg_frontier/compression/decode_then_filter.py"
            ),
            "dtf_renderer_source": sha256_file(
                ROOT / "src/cg_frontier/render/decode_then_filter.py"
            ),
            "dtf_training_source": sha256_file(
                ROOT / "src/cg_frontier/compression/decode_then_filter_training.py"
            ),
        }
    )
    targets = load_core4_targets(inputs["core4_dir"], device)
    reference_textures = _reference_textures(targets)
    core4_textures = load_core4_textures(inputs["core4_manifest"], device)
    initialization = make_paired_precheck_initialization(
        height=targets.height,
        width=targets.width,
        latent_channels=int(candidate_config["latent_channels"]),
        decoder_width=int(candidate_config["decoder_width"]),
        seed=seed,
    )
    fresh = instantiate_paired_precheck_candidate(
        initialization, activation=activation, device=device
    )
    latent, decoder = fresh.latent, fresh.decoder

    mesh = load_gltf_mesh(inputs["gltf"])
    cameras, camera_selection = _prepare_camera_pool(config, mesh, device)
    lights = [(str(spec["name"]), _light(spec)) for spec in config["train_lights"]]
    if not lights:
        raise RuntimeError("DTF training requires at least one light")
    renderer = DecodeThenFilterRenderer(
        display_exposure=float(config["render"]["display_exposure"]),
        minimum_roughness=float(config["render"]["minimum_roughness"]),
    )

    output_root = _repo_path(config["output_root"], "output_root")
    fresh_camera31_output_name = "c4_dtf_16_camera31_fresh_80k"
    if camera31_fresh:
        output_dir = output_root / fresh_camera31_output_name
        if preflight_only:
            output_dir = (
                output_root / "camera31_fresh_preflight" / fresh_camera31_output_name
            )
    elif camera_finetune:
        finetune_output_name = str(finetune_profile["output_name"])
        output_dir = output_root / finetune_output_name
        if preflight_only:
            output_dir = (
                output_root / "camera_finetune_preflight" / finetune_output_name
            )
    elif is_continuation:
        assert continuation_output_name is not None
        output_dir = output_root / continuation_output_name
        if preflight_only:
            output_dir = output_root / "resume_preflight" / continuation_output_name
    else:
        output_dir = output_root / candidate_name
    if preflight_only and not is_resume_training and not camera31_fresh:
        output_dir = output_root / "full_preflight" / candidate_name
    log_path = output_dir / "train.jsonl"
    if log_path.exists() and not is_resume_training:
        raise FileExistsError(f"refusing to inherit an existing DTF full run: {log_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_dir = (
        output_root / str(finetune_profile["source_output_name"])
        if camera_finetune
        else output_root / candidate_name
    )
    source_manifest_path = source_dir / "training_manifest.json"
    source_checkpoint_path = source_dir / "rolling-resume.pt"
    if is_resume_training:
        if not source_manifest_path.is_file() or not source_checkpoint_path.is_file():
            raise FileNotFoundError("DTF parent manifest and rolling checkpoint are required")
        if camera_finetune:
            planned_manifest = build_decode_then_filter_camera_finetune_manifest(
                config,
                source_training_manifest_sha256=sha256_file(source_manifest_path),
                source_checkpoint_sha256=sha256_file(source_checkpoint_path),
                config_sha256=hashlib.sha256(config_bytes).hexdigest(),
                input_hashes=input_hashes,
                output_dir=output_dir.relative_to(ROOT).as_posix(),
            )
        else:
            planned_manifest = build_decode_then_filter_continuation_manifest(
                config,
                candidate_name=candidate_name,
                stop_step=int(continuation_stop_step),
                source_training_manifest_sha256=sha256_file(source_manifest_path),
                source_checkpoint_sha256=sha256_file(source_checkpoint_path),
                config_sha256=hashlib.sha256(config_bytes).hexdigest(),
                input_hashes=input_hashes,
                output_dir=output_dir.relative_to(ROOT).as_posix(),
            )
    elif camera31_fresh:
        planned_manifest = build_decode_then_filter_camera31_fresh_manifest(
            config,
            config_sha256=hashlib.sha256(config_bytes).hexdigest(),
            input_hashes=input_hashes,
            git_commit=str(config["version_control"]["git_commit"]),
            selected_activation=activation,
            activation_selection_sha256=activation_selection_sha256,
            output_dir=output_dir.relative_to(ROOT).as_posix(),
        )
    else:
        planned_manifest = build_decode_then_filter_manifest(
            config,
            candidate=candidate_name,
            config_sha256=hashlib.sha256(config_bytes).hexdigest(),
            input_hashes=input_hashes,
            git_commit=str(config["version_control"]["git_commit"]),
            selected_activation=activation,
            activation_selection_sha256=activation_selection_sha256,
        )
    run_manifest_path = output_dir / "run_manifest.json"
    if not run_manifest_path.exists():
        run_manifest_path.write_text(
            deterministic_json(planned_manifest), encoding="utf-8", newline="\n"
        )

    if not preflight_only and not is_resume_training:
        preflight_manifest = (
            output_root
            / ("camera31_fresh_preflight" if camera31_fresh else "full_preflight")
            / (fresh_camera31_output_name if camera31_fresh else candidate_name)
            / "training_manifest.json"
        )
        if not preflight_manifest.is_file():
            raise RuntimeError("successful fresh full-training preflight is required")
        preflight = json.loads(preflight_manifest.read_text(encoding="utf-8"))
        if (
            not preflight.get("valid")
            or preflight.get("initialization", {}).get("sha256") != initialization.sha256
            or preflight.get("activation_selection_sha256")
            != activation_selection_sha256
            or preflight.get("camera_selection", {}).get("selected_names")
            != camera_selection["selected_names"]
            or preflight.get("capacity_route_sha256") != capacity_route_sha256
            or (
                camera31_fresh
                and preflight.get("camera_distribution")
                != planned_manifest.get("camera_distribution")
            )
        ):
            raise RuntimeError("full-training preflight evidence is invalid or mismatched")
    if is_continuation and not preflight_only:
        assert continuation_output_name is not None
        preflight_manifest = (
            output_root
            / "resume_preflight"
            / continuation_output_name
            / "training_manifest.json"
        )
        if not preflight_manifest.is_file():
            raise RuntimeError("successful DTF continuation preflight is required")
        preflight = json.loads(preflight_manifest.read_text(encoding="utf-8"))
        if not preflight.get("valid") or preflight.get("lineage") != planned_manifest.get(
            "lineage"
        ):
            raise RuntimeError("DTF continuation preflight evidence is invalid or mismatched")
    if camera_finetune and not preflight_only:
        finetune_output_name = str(finetune_profile["output_name"])
        preflight_manifest = (
            output_root
            / "camera_finetune_preflight"
            / finetune_output_name
            / "training_manifest.json"
        )
        if not preflight_manifest.is_file():
            raise RuntimeError("successful DTF camera fine-tune preflight is required")
        preflight = json.loads(preflight_manifest.read_text(encoding="utf-8"))
        if (
            not preflight.get("valid")
            or preflight.get("lineage") != planned_manifest.get("lineage")
            or preflight.get("camera_distribution")
            != planned_manifest.get("camera_distribution")
        ):
            raise RuntimeError("DTF camera fine-tune preflight is invalid or mismatched")

    uniform_pool = torch.arange(targets.texel_count, device=device)
    high_gradient_pool = torch.from_numpy(
        _generic_high_gradient_pool(
            targets, top_fraction=float(training["high_gradient_top_fraction"])
        )
    ).to(device)
    if training.get("optimizer") != "Adam" or training.get("optimizer_implementation") != "fused_cuda_direct_unorm":
        raise ValueError("DTF full training requires fused CUDA Adam")
    optimizer = torch.optim.Adam(
        [
            {"params": [latent], "lr": 0.0},
            {"params": list(decoder.parameters()), "lr": 0.0},
        ],
        fused=True,
    )
    rolling_path = output_dir / "rolling-resume.pt"
    best_render_path = output_dir / "best-render.pt"
    best_artifact_path = output_dir / "best-artifact-safe.pt"
    final_path = output_dir / "final.pt"
    start_step = 0
    if is_resume_training:
        resume_path = rolling_path if rolling_path.is_file() else source_checkpoint_path
        resume_payload = torch.load(resume_path, map_location="cpu", weights_only=False)
        start_step = int(resume_payload.get("step", -1))
        resume_start = 160_000 if camera_finetune else 80_000
        resume_stop = (
            int(finetune_profile["stop_step"])
            if camera_finetune
            else int(continuation_stop_step)
        )
        if not (resume_start <= start_step <= resume_stop):
            raise ValueError("DTF resume checkpoint step is outside its range")
        payload_input_hashes = dict(resume_payload.get("input_hashes", {}))
        for name in config["inputs"]:
            if payload_input_hashes.get(name) != input_hashes.get(name):
                raise ValueError(f"DTF continuation input asset hash changed: {name}")
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        expected_camera_names = (
            list(camera_selection["selected_names"])
            if resume_path == rolling_path
            else list(
                dict(source_manifest.get("camera_selection", {})).get(
                    "selected_names", []
                )
            )
        )
        validate_decode_then_filter_resume_payload(
            resume_payload,
            expected_candidate=candidate_name,
            expected_start_step=start_step,
            expected_latent_shape=tuple(latent.shape),
            expected_decoder_keys=set(decoder.state_dict()),
            expected_input_hashes=payload_input_hashes,
            expected_camera_names=expected_camera_names,
        )
        restore_decode_then_filter_resume_state(
            resume_payload,
            latent=latent,
            decoder=decoder,
            optimizer=optimizer,
            generator=generator,
        )
    max_steps = int(
        start_step + int(training["preflight_steps"])
        if preflight_only and is_resume_training
        else training["preflight_steps"]
        if preflight_only
        else finetune_profile["stop_step"]
        if camera_finetune
        else continuation_stop_step
        if is_continuation
        else candidate_config["max_steps"]
    )
    if is_resume_training:
        resume_stop = (
            int(finetune_profile["stop_step"])
            if camera_finetune
            else int(continuation_stop_step)
        )
        max_steps = min(max_steps, resume_stop)
    batch_size = int(training["batch_size"])
    best_render: dict[str, Any] | None = None
    best_artifact: dict[str, Any] | None = None
    final_evaluation: dict[str, Any] | None = None
    if is_resume_training:
        source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
        source_selection = dict(source_manifest.get("checkpoint_selection", {}))
        best_render = source_selection.get("best_render")
        best_artifact = source_selection.get("best_artifact_safe")
        if rolling_path.is_file() and log_path.is_file():
            logged_records = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            recovered = recover_logged_checkpoint_selection(
                source_selection, logged_records
            )
            best_render = recovered["best_render"]
            best_artifact = recovered["best_artifact_safe"]
        source_best_render = source_dir / "best-render.pt"
        source_best_artifact = source_dir / "best-artifact-safe.pt"
        if best_render and not best_render_path.exists() and source_best_render.is_file():
            shutil.copyfile(source_best_render, best_render_path)
        if (
            best_artifact
            and not best_artifact_path.exists()
            and source_best_artifact.is_file()
        ):
            shutil.copyfile(source_best_artifact, best_artifact_path)
    started = time.monotonic()
    completed_step = start_step
    with keep_system_awake():
        for step in range(start_step + 1, max_steps + 1):
            if camera_finetune:
                phase = str(finetune_profile["phase"])
                latent_lr, decoder_lr = camera_finetune_learning_rates_at_step(step)
            elif is_continuation:
                phase = "low_lr_continuation"
                assert continuation_stop_step is not None
                latent_lr, decoder_lr = continuation_learning_rates_at_step(
                    step, stop_step=continuation_stop_step
                )
            else:
                phase = (
                    "render_first_joint"
                    if preflight_only
                    else full_training_phase_at_step(step)
                )
                latent_lr, decoder_lr = full_training_learning_rates_at_step(step)
            optimizer.param_groups[0]["lr"] = latent_lr
            optimizer.param_groups[1]["lr"] = decoder_lr
            optimizer.zero_grad(set_to_none=True)
            camera_index = random.randrange(len(cameras))
            light_index = random.randrange(len(lights))
            camera_name, camera, geometry = cameras[camera_index]
            light_name, light = lights[light_index]
            screen_uv = geometry.torch_buffers["uv"][geometry.torch_buffers["mask"]]
            batch = decode_then_filter_batch(
                screen_uv=screen_uv,
                uniform_pool=uniform_pool,
                high_gradient_pool=high_gradient_pool,
                batch_size=batch_size,
                height=targets.height,
                width=targets.width,
                generator=generator,
            )
            target = _sample_reference(reference_textures, batch.uv)
            sample = decode_then_filter_sample(
                latent, batch.uv, decoder, quantization="fake"
            )
            raw_corners, _ = bilinear_corners_top_down_wrap_torch(latent, batch.uv)
            quantization_error = F.mse_loss(raw_corners, sample.corners.detach())
            material_total, material_terms = paired_precheck_objective(
                sample.material,
                target,
                anchor_slice=batch.slices["texel_center_quantization_anchor"],
                quantization_error=quantization_error,
                loss_config=config["loss"],
            )
            render_terms: dict[str, torch.Tensor] = {}
            if phase == "material_continuous_pretrain":
                total = material_total
            else:
                screen_slice = batch.slices["screen_space_render"]
                sampled_geometry = _sampled_geometry(geometry, batch.screen_positions)
                screen_target = Core4Targets(
                    target.base_color_linear[screen_slice],
                    target.normal_xyz[screen_slice],
                    target.roughness[screen_slice],
                    target.metallic[screen_slice],
                    height=1,
                    width=screen_slice.stop - screen_slice.start,
                )
                reference_material = decoded_to_material(
                    sampled_geometry, _decoded_target(screen_target)
                )
                reference = renderer.render(
                    sampled_geometry,
                    camera,
                    light,
                    DTFReferenceMaterialSource(reference_material),
                    input_hashes=input_hashes,
                )
                candidate = renderer.render(
                    sampled_geometry,
                    camera,
                    light,
                    DTFLatentMaterialSource(latent, decoder, quantization="fake"),
                    input_hashes=input_hashes,
                )
                total, render_terms = full_training_objective(
                    material_total,
                    reference_hdr=reference.linear_hdr,
                    candidate_hdr=candidate.linear_hdr,
                    reference_display=reference.display_rgb,
                    candidate_display=candidate.display_rgb,
                    phase="low_lr_polish" if is_continuation else phase,
                    loss_config=config["loss"],
                )
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite DTF full loss at step {step}")
            total.backward()
            optimizer.step()
            with torch.no_grad():
                latent.clamp_(0.0, 1.0)
            completed_step = step
            elapsed = time.monotonic() - started
            record: dict[str, Any] = {
                "step": step,
                "phase": phase,
                "camera": camera_name,
                "light": light_name,
                "elapsed_seconds": round(elapsed, 3),
                "latent_lr": latent_lr,
                "decoder_lr": decoder_lr,
                "total": float(total.detach().cpu()),
                **{
                    name: float(value.detach().cpu())
                    for name, value in material_terms.items()
                },
                **{
                    name: float(value.detach().cpu())
                    for name, value in render_terms.items()
                },
            }
            selection_due = (
                not preflight_only
                and step >= int(training["selection_start"])
                and (
                    step % int(training["selection_interval"]) == 0
                    or step == max_steps
                )
            )
            if selection_due:
                final_evaluation = _selection_evaluation(
                    latent=latent.detach(),
                    decoder=decoder,
                    reference_textures=reference_textures,
                    core4_textures=core4_textures,
                    cameras=cameras,
                    lights=lights,
                    renderer=renderer,
                    config=config,
                    input_hashes=input_hashes,
                )
                if tracks_yellow_material:
                    final_evaluation["yellow_material"] = (
                        _yellow_material_evaluation(
                            latent=latent.detach(),
                            decoder=decoder,
                            targets=targets,
                        )
                    )
                record["selection_evaluation"] = final_evaluation
                payload = _checkpoint_payload(
                    candidate=candidate_name,
                    step=step,
                    latent=latent,
                    decoder=decoder,
                    optimizer=optimizer,
                    generator=generator,
                    initialization_sha256=initialization.sha256,
                    activation_selection_sha256=activation_selection_sha256,
                    input_hashes=input_hashes,
                    camera_selection=camera_selection,
                )
                render_score = _best_render_score(final_evaluation)
                if best_render is None or render_score < tuple(best_render["score"]):
                    _atomic_save(payload, best_render_path)
                    best_render = {"step": step, "score": list(render_score)}
                artifact_score = _artifact_score(final_evaluation)
                if artifact_score is not None and (
                    best_artifact is None
                    or artifact_score < tuple(best_artifact["score"])
                ):
                    _atomic_save(payload, best_artifact_path)
                    best_artifact = {"step": step, "score": list(artifact_score)}
            if (
                step == 1
                or step % int(training["log_interval"]) == 0
                or step == max_steps
                or selection_due
            ):
                with log_path.open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(
                        json.dumps(record, sort_keys=True, separators=(",", ":"))
                        + "\n"
                    )
                print(
                    json.dumps(
                        {
                            "candidate": candidate_name,
                            "step": step,
                            "phase": phase,
                            "elapsed_seconds": round(elapsed, 3),
                            "total": record["total"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
            if step % int(training["checkpoint_interval"]) == 0 or step == max_steps:
                _atomic_save(
                    _checkpoint_payload(
                        candidate=candidate_name,
                        step=step,
                        latent=latent,
                        decoder=decoder,
                        optimizer=optimizer,
                        generator=generator,
                        initialization_sha256=initialization.sha256,
                        activation_selection_sha256=activation_selection_sha256,
                        input_hashes=input_hashes,
                        camera_selection=camera_selection,
                    ),
                    rolling_path,
                )

    elapsed = time.monotonic() - started
    final_payload = _checkpoint_payload(
        candidate=candidate_name,
        step=completed_step,
        latent=latent,
        decoder=decoder,
        optimizer=optimizer,
        generator=generator,
        initialization_sha256=initialization.sha256,
        activation_selection_sha256=activation_selection_sha256,
        input_hashes=input_hashes,
        camera_selection=camera_selection,
    )
    _atomic_save(final_payload, final_path)
    if preflight_only:
        final_evaluation = _evaluate_precheck(
            latent.detach(),
            decoder,
            reference_textures,
            sample_count=int(training["preflight_evaluation_samples"]),
            seed=seed + 778,
        )
    latent_stem = f"latent_{candidate_name}"
    latent_metadata = export_decode_then_filter_latent_unorm8(
        fake_quantize_unorm8(latent.detach()),
        output_dir=output_dir,
        stem=latent_stem,
    )
    decoder_path = output_dir / "decoder_weights.npz"
    np.savez(
        decoder_path,
        **{
            name: value.detach().cpu().numpy()
            for name, value in decoder.state_dict().items()
        },
    )
    valid = completed_step == max_steps and rolling_path.is_file() and final_path.is_file()
    files: dict[str, Any] = {
        "latent_resources": latent_metadata,
        "decoder_weights.npz": sha256_file(decoder_path),
        "rolling-resume.pt": sha256_file(rolling_path),
        "final.pt": sha256_file(final_path),
        "train.jsonl": sha256_file(log_path),
        "run_manifest.json": sha256_file(output_dir / "run_manifest.json"),
    }
    if best_render_path.is_file():
        files["best-render.pt"] = sha256_file(best_render_path)
    if best_artifact_path.is_file():
        files["best-artifact-safe.pt"] = sha256_file(best_artifact_path)
    manifest = {
        **planned_manifest,
        "status": (
            "camera31_fresh_preflight_complete"
            if preflight_only and camera31_fresh
            else "camera_distribution_finetune_preflight_complete"
            if preflight_only and camera_finetune
            else "continuation_preflight_complete"
            if preflight_only and is_continuation
            else "preflight_complete"
            if preflight_only
            else "trained_local_80k_camera31_fresh"
            if camera31_fresh
            else "trained_local_180k_camera_distribution_finetune"
            if camera_finetune
            else f"trained_local_{int(continuation_stop_step / 1000)}k_continuation"
            if is_continuation and continuation_stop_step is not None
            else "trained_local_80k"
        ),
        "valid": valid,
        "activation_selection_sha256": activation_selection_sha256,
        "capacity_route_sha256": capacity_route_sha256,
        "initialization": {
            "kind": "fresh_random_state_after_paired_selection",
            "seed": seed,
            "sha256": initialization.sha256,
            "precheck_checkpoint_used": False,
        },
        "camera_selection": camera_selection,
        "light_pool": {
            "count": len(lights),
            "names": [name for name, _ in lights],
            "randomized_online": True,
        },
        "training_result": {
            **(
                summarize_resume_training_result(
                    planned_start_step=int(planned_manifest["training"]["start_step"]),
                    segment_start_step=start_step,
                    completed_step=completed_step,
                    required_step=max_steps,
                    segment_elapsed_seconds=elapsed,
                )
                if is_resume_training
                else {
                    "completed_steps": completed_step,
                    "required_steps": max_steps,
                    "source_step": start_step,
                    "steps_executed": completed_step - start_step,
                    "elapsed_seconds": round(elapsed, 3),
                    "elapsed_scope": "full_process_run",
                }
            ),
            "batch_size": batch_size,
            "optimizer": "Adam",
            "optimizer_implementation": "fused_cuda_direct_unorm",
            "checkpoint_contains_optimizer_rng_sampling_generator": True,
            "phases": (
                [
                    {
                        "name": (
                            str(finetune_profile["phase"])
                            if camera_finetune
                            else "low_lr_continuation"
                        ),
                        "start": start_step,
                        "stop": max_steps,
                    }
                ]
                if is_resume_training
                else config["training"]["phases"]
            ),
        },
        "checkpoint_selection": {
            "start_step": int(training["selection_start"]),
            "tracks": ["best-render", "best-artifact-safe"],
            "best_render": best_render,
            "best_artifact_safe": best_artifact,
        },
        "final_evaluation": final_evaluation,
        "files": files,
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
        default=ROOT / "configs/train/scifihelmet_c4_dtf_v1.yaml",
    )
    parser.add_argument(
        "--candidate", choices=SUPPORTED_CANDIDATES, default="c4_dtf_16_selected"
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument(
        "--continue-c5-to-120k",
        action="store_true",
        help="resume C5-DTF-16 from its exact 80k optimizer/RNG state",
    )
    parser.add_argument(
        "--continue-c4-to-160k",
        action="store_true",
        help="resume C4-DTF-16 from its exact 80k optimizer/RNG state",
    )
    parser.add_argument(
        "--finetune-c4-camera31-to-180k",
        action="store_true",
        help=(
            "fine-tune the C4-160k parent with the explicitly audited 31-camera "
            "distribution; this is not an exact continuation"
        ),
    )
    parser.add_argument(
        "--fresh-c4-camera31-to-80k",
        action="store_true",
        help=(
            "train C4-DTF-16 from fresh initialization with the explicitly "
            "audited 31-camera distribution"
        ),
    )
    args = parser.parse_args()
    selected_modes = sum(
        bool(value)
        for value in (
            args.continue_c5_to_120k,
            args.continue_c4_to_160k,
            args.finetune_c4_camera31_to_180k,
            args.fresh_c4_camera31_to_80k,
        )
    )
    if selected_modes > 1:
        parser.error("choose only one DTF resume or fine-tune mode")
    continuation_stop_step = (
        120_000
        if args.continue_c5_to_120k
        else 160_000
        if args.continue_c4_to_160k
        else None
    )
    result = run(
        args.config.resolve(),
        candidate_name=args.candidate,
        preflight_only=args.preflight_only,
        continuation_stop_step=continuation_stop_step,
        camera_finetune=args.finetune_c4_camera31_to_180k,
        camera31_fresh=args.fresh_c4_camera31_to_80k,
    )
    print(
        json.dumps(
            {
                "candidate": args.candidate,
                "preflight_only": args.preflight_only,
                "continuation_stop_step": continuation_stop_step,
                "camera_finetune": args.finetune_c4_camera31_to_180k,
                "camera31_fresh": args.fresh_c4_camera31_to_80k,
                "valid": result["valid"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
