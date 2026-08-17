"""Run paired material-only and material+render C4 affine training for one asset."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.nn import functional as F
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cg_frontier.assets.gltf_core4 import load_gltf_core4_asset  # noqa: E402
from cg_frontier.assets.preprocess import sha256_file  # noqa: E402
from cg_frontier.compression.affine_color import (  # noqa: E402
    linear_srgb_to_oklab,
    orthogonal_color_coordinates,
)
from cg_frontier.compression.affine_pca import export_p0_bundle, rasterize_uv_charts  # noqa: E402
from cg_frontier.compression.basecolor_priority import postprocess_affine_output  # noqa: E402
from cg_frontier.compression.render_ablation import (  # noqa: E402
    ARMS,
    FULL_CHECKPOINT_STEPS,
    OBSERVATION_STEPS,
    LossWeights,
    checkpoint_identity_hash,
    compose_ablation_loss,
    load_ablation_checkpoint,
    paired_sampling_evidence,
    sampling_contract_hash,
    save_ablation_checkpoint,
    tensor_sha256,
    sampling_trajectory_hash,
)
from cg_frontier.compression.render_loss import (  # noqa: E402
    bilinear_sample_top_down_wrap,
    decoded_to_material,
    fake_quantize_unorm8,
    hard_quantize_unorm8,
    masked_render_metrics,
    sparse_fake_quantized_bilinear_sample_top_down_wrap,
)
from cg_frontier.compression.affine_material import AffineDecodedMaterial  # noqa: E402
from cg_frontier.render.gbuffer import Core4Textures, render_geometry_gbuffer, sample_core4_material  # noqa: E402
from cg_frontier.render.generic_c4_rig import build_generic_c4_rig, instantiate_camera, instantiate_lights  # noqa: E402
from cg_frontier.render.pbr import shade_ggx  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/train/c4_render_ablation_20k_v1.yaml"


def _repo_path(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a repository-relative path")
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"{label} escapes the repository")
    return path


def _write_json(path: Path, value: object) -> str:
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n", encoding="ascii")
    return digest


def _config(path: Path) -> Mapping[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping) or value.get("schema_version") != 1:
        raise ValueError("unsupported C4 render-ablation config")
    if value.get("experiment") != "c4_render_ablation_20k_v1":
        raise ValueError("unexpected experiment lineage")
    if value.get("formal_holdout_access") != "forbidden":
        raise ValueError("formal holdout must remain forbidden")
    training = value["training"]
    if (
        int(training["steps"]) != 20000
        or tuple(training["observation_steps"]) != OBSERVATION_STEPS
        or tuple(training["full_checkpoint_steps"]) != FULL_CHECKPOINT_STEPS
        or float(training["latent_learning_rate"]) != 2.0e-4
        or float(training["affine_learning_rate"]) != 2.0e-5
        or tuple(value["arms"]) != ARMS
    ):
        raise ValueError("training schedule differs from the frozen 20k contract")
    LossWeights.from_mapping(value["loss"])
    postprocess = value["postprocess"]
    if set(postprocess) != {
        "scalar_saturate",
        "normal_disk_projection",
        "normal_z_reconstruction",
        "straight_through_backward",
    } or not all(bool(item) for item in postprocess.values()):
        raise ValueError("deployment safety differs from the frozen contract")
    expected_assets = ("Corset", "Lantern", "BoomBox")
    if tuple(item["id"] for item in value["assets"]) != expected_assets:
        raise ValueError("asset order differs from the frozen three-model contract")
    if tuple(item.get("tangent_source") for item in value["assets"]) != (
        "reconstructed_uv",
        "source_gltf",
        "reconstructed_uv",
    ):
        raise ValueError("asset tangent sources differ from the frozen contract")
    if tuple(int(item.get("degenerate_uv_triangles", -1)) for item in value["assets"]) != (
        0,
        3,
        0,
    ):
        raise ValueError("asset degenerate UV counts differ from the frozen contract")
    return value


def _asset_spec(config: Mapping[str, Any], asset_id: str) -> Mapping[str, Any]:
    matches = [item for item in config["assets"] if item["id"] == asset_id]
    if len(matches) != 1:
        raise ValueError("asset is outside the frozen three-model contract")
    return matches[0]


def _seven(targets: object) -> torch.Tensor:
    return torch.cat(
        (
            targets.base_color_linear.reshape(targets.height, targets.width, 3),
            targets.normal_xyz.reshape(targets.height, targets.width, 3)[..., :2],
            targets.roughness.reshape(targets.height, targets.width, 1),
            targets.metallic.reshape(targets.height, targets.width, 1),
        ),
        dim=-1,
    )


def prepare_asset(config_path: Path, config: Mapping[str, Any], asset_id: str) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the C4 render ablation")
    spec = _asset_spec(config, asset_id)
    gltf = _repo_path(spec["gltf"], f"{asset_id}.gltf")
    metadata = _repo_path(spec["metadata"], f"{asset_id}.metadata")
    if sha256_file(gltf) != spec["gltf_sha256"] or sha256_file(metadata) != spec["metadata_sha256"]:
        raise ValueError("asset source or metadata SHA-256 mismatch")
    asset = load_gltf_core4_asset(gltf, name=asset_id, device="cpu")
    target_atlas = _seven(asset.targets)
    valid_mask, chart_ids = rasterize_uv_charts(
        asset.mesh.texcoords,
        asset.mesh.triangles,
        height=asset.targets.height,
        width=asset.targets.width,
    )
    bundle = export_p0_bundle(target_atlas, valid_mask, chart_ids, margin=1.0e-3)
    rig = build_generic_c4_rig()
    if tuple(config["rig"]["resolution"]) != rig.resolution:
        raise ValueError("config resolution differs from generic_c4_rig_v1")
    textures = Core4Textures(
        base_color_linear=asset.textures.base_color_linear.cuda().contiguous(),
        normal=asset.textures.normal.cuda().contiguous(),
        roughness=asset.textures.roughness.cuda().contiguous(),
        metallic=asset.textures.metallic.cuda().contiguous(),
        source_hashes=asset.textures.source_hashes,
    )
    cameras, geometries, camera_lights, source_materials = [], [], [], []
    for camera_spec in rig.cameras:
        camera, center, radius = instantiate_camera(
            asset.mesh,
            camera_spec,
            vertical_fov_degrees=float(config["rig"]["vertical_fov_degrees"]),
            distance_padding=float(config["rig"]["distance_padding"]),
        )
        geometry = render_geometry_gbuffer(
            asset.mesh,
            camera,
            rig.resolution,
            device="cuda",
            cull_backfaces=bool(config["rig"]["backface_culling"]),
            tangent_source=str(spec["tangent_source"]),
        )
        lights = instantiate_lights(camera, center, radius, rig)
        cameras.append(camera)
        geometries.append(geometry)
        camera_lights.append(lights)
        source_materials.append(sample_core4_material(geometry, textures))
    with torch.no_grad():
        references = [
            [
                shade_ggx(
                    geometries[camera_index],
                    cameras[camera_index],
                    camera_lights[camera_index][light_index],
                    material_override=source_materials[camera_index],
                    minimum_roughness=float(config["rig"]["minimum_roughness"]),
                )
                for camera_index in range(len(cameras))
            ]
            for light_index in range(len(rig.lights))
        ]
    valid_indices = torch.nonzero(valid_mask.reshape(-1), as_tuple=False).flatten().cuda()
    target_valid = target_atlas.reshape(-1, 7).cuda()[valid_indices]
    asset_hash = hashlib.sha256(
        "".join(sorted(asset.manifest["source_hashes"].values())).encode("ascii")
    ).hexdigest()
    training_cameras = tuple(
        index for index, item in enumerate(rig.cameras) if item.split == "train"
    )
    audit_cameras = tuple(
        index for index, item in enumerate(rig.cameras) if item.split == "audit"
    )
    contract_hash = sampling_contract_hash(
        seed=int(config["seed"]),
        valid_texels=valid_indices.numel(),
        training_camera_indices=training_cameras,
        lights=len(rig.lights),
    )
    identity = {
        "asset_hash": asset_hash,
        "config_hash": sha256_file(config_path),
        "parent_hash": bundle.calibration.raw.artifact_hash,
        "rig_hash": rig.rig_hash,
        "sampling_contract_hash": contract_hash,
    }
    return {
        "asset_id": asset_id,
        "spec": spec,
        "mesh": asset.mesh,
        "target_atlas": target_atlas.cuda(),
        "valid_mask": valid_mask.cuda(),
        "valid_indices": valid_indices,
        "target_valid": target_valid,
        "raw": bundle.calibration.raw,
        "rig": rig,
        "cameras": cameras,
        "geometries": geometries,
        "camera_lights": camera_lights,
        "references": references,
        "training_camera_indices": training_cameras,
        "audit_camera_indices": audit_cameras,
        "identity": identity,
        "identity_hash": checkpoint_identity_hash(identity),
    }


def _new_state(prepared: Mapping[str, Any], config: Mapping[str, Any]):
    raw = prepared["raw"]
    latent = nn.Parameter(raw.latent_unorm8.cuda().float() / 255.0)
    weight = nn.Parameter(raw.weight.cuda().float().clone())
    bias = nn.Parameter(raw.bias.cuda().float().clone())
    latent_optimizer = torch.optim.Adam((latent,), lr=float(config["training"]["latent_learning_rate"]))
    affine_optimizer = torch.optim.Adam((weight, bias), lr=float(config["training"]["affine_learning_rate"]))
    rng = torch.Generator(device="cuda").manual_seed(int(config["seed"]))
    return (latent, weight, bias), (latent_optimizer, affine_optimizer), rng


def _draw_batch(prepared: Mapping[str, Any], config: Mapping[str, Any], rng: torch.Generator):
    device = prepared["valid_indices"].device
    positions = torch.randint(
        prepared["valid_indices"].numel(),
        (int(config["training"]["material_batch_size"]),),
        generator=rng,
        device=device,
    )
    camera_slot = int(torch.randint(len(prepared["training_camera_indices"]), (1,), generator=rng, device=device))
    camera_index = int(prepared["training_camera_indices"][camera_slot])
    light_index = int(torch.randint(len(prepared["rig"].lights), (1,), generator=rng, device=device))
    return positions, camera_index, light_index


def _material_override(processed: object) -> AffineDecodedMaterial:
    return AffineDecodedMaterial(
        base_color_linear=processed.seven[..., :3],
        normal_xy=processed.seven[..., 3:5],
        normal_xyz=processed.normal_xyz,
        roughness=processed.seven[..., 5:6],
        metallic=processed.seven[..., 6:7],
    )


def _terms(
    state,
    prepared: Mapping[str, Any],
    config: Mapping[str, Any],
    batch,
    *,
    compute_render: bool,
):
    latent, weight, bias = state
    positions, camera_index, light_index = batch
    indices = prepared["valid_indices"][positions]
    material = postprocess_affine_output(
        F.linear(fake_quantize_unorm8(latent.reshape(-1, 4)[indices]), weight, bias),
        compander_parameters=None,
        straight_through=True,
    )
    target = postprocess_affine_output(
        prepared["target_valid"][positions], compander_parameters=None, straight_through=False
    )
    epsilon = 1.0e-3
    terms = {
        "base_color_l1": F.l1_loss(material.seven[:, :3], target.seven[:, :3]),
        "normal_cosine": torch.mean(1.0 - (material.normal_xyz * target.normal_xyz).sum(dim=-1)),
        "roughness_l1": F.l1_loss(material.seven[:, 5], target.seven[:, 5]),
        "metallic_l1": F.l1_loss(material.seven[:, 6], target.seven[:, 6]),
    }
    if compute_render:
        geometry = prepared["geometries"][camera_index]
        sampled = sparse_fake_quantized_bilinear_sample_top_down_wrap(
            latent, geometry.torch_buffers["uv"]
        )
        render_material = postprocess_affine_output(
            F.linear(sampled, weight, bias), compander_parameters=None, straight_through=True
        )
        candidate = shade_ggx(
            geometry,
            prepared["cameras"][camera_index],
            prepared["camera_lights"][camera_index][light_index],
            material_override=decoded_to_material(geometry, _material_override(render_material)),
            minimum_roughness=float(config["rig"]["minimum_roughness"]),
        )
        mask = geometry.torch_buffers["mask"]
        prediction = candidate[mask]
        reference = prepared["references"][light_index][camera_index][mask]
        difference = prediction - reference
        terms["render_linear"] = torch.sqrt(
            difference.square() + epsilon * epsilon
        ).mean()
        terms["render_log"] = torch.abs(
            torch.log1p(prediction.clamp_min(0.0))
            - torch.log1p(reference.clamp_min(0.0))
        ).mean()
    else:
        zero = terms["base_color_l1"] * 0.0
        terms["render_linear"] = zero
        terms["render_log"] = zero
    return terms, {"camera_index": camera_index, "light_index": light_index}


@torch.no_grad()
def _atlas_metrics(state, prepared: Mapping[str, Any]) -> dict[str, float]:
    latent, weight, bias = state
    indices = prepared["valid_indices"]
    target = prepared["target_valid"]
    deployed = hard_quantize_unorm8(latent).reshape(-1, 4)
    predictions = []
    for start in range(0, indices.numel(), 262144):
        raw = F.linear(deployed[indices[start : start + 262144]], weight, bias)
        predictions.append(postprocess_affine_output(raw, compander_parameters=None, straight_through=False).seven.cpu())
    prediction = torch.cat(predictions)
    target_cpu = target.cpu()
    difference = prediction - target_cpu
    pred_normal = postprocess_affine_output(prediction, compander_parameters=None, straight_through=False).normal_xyz
    target_normal = postprocess_affine_output(target_cpu, compander_parameters=None, straight_through=False).normal_xyz
    angle = torch.rad2deg(torch.acos((pred_normal * target_normal).sum(dim=-1).clamp(-1.0, 1.0)))
    pred_oklab = linear_srgb_to_oklab(prediction[:, :3])
    target_oklab = linear_srgb_to_oklab(target_cpu[:, :3])
    pred_opponent = orthogonal_color_coordinates(prediction[:, :3])
    target_opponent = orthogonal_color_coordinates(target_cpu[:, :3])
    return {
        "seven_channel_mae": float(difference.abs().mean()),
        "base_color_linear_mae": float(difference[:, :3].abs().mean()),
        "normal_mean_degrees": float(angle.mean()),
        "roughness_mae": float(difference[:, 5].abs().mean()),
        "metallic_mae": float(difference[:, 6].abs().mean()),
        "oklab_delta_e_mean": float(torch.linalg.vector_norm(pred_oklab - target_oklab, dim=-1).mean()),
        "opponent_error": float((pred_opponent[:, 1:] - target_opponent[:, 1:]).abs().mean()),
        "chroma_magnitude_retention": float(
            torch.linalg.vector_norm(pred_opponent[:, 1:], dim=-1).std()
            / torch.linalg.vector_norm(target_opponent[:, 1:], dim=-1).std().clamp_min(1.0e-12)
        ),
    }


@torch.no_grad()
def _render_metrics(state, prepared: Mapping[str, Any], config: Mapping[str, Any], pairs) -> dict[str, Any]:
    latent, weight, bias = state
    deployed = hard_quantize_unorm8(latent)
    rows = []
    for camera_index, light_index in pairs:
        geometry = prepared["geometries"][camera_index]
        sampled = bilinear_sample_top_down_wrap(deployed, geometry.torch_buffers["uv"])
        processed = postprocess_affine_output(
            F.linear(sampled, weight, bias), compander_parameters=None, straight_through=False
        )
        candidate = shade_ggx(
            geometry,
            prepared["cameras"][camera_index],
            prepared["camera_lights"][camera_index][light_index],
            material_override=decoded_to_material(geometry, _material_override(processed)),
            minimum_roughness=float(config["rig"]["minimum_roughness"]),
        )
        row = masked_render_metrics(
            prepared["references"][light_index][camera_index],
            candidate,
            geometry.torch_buffers["mask"],
            linear_psnr_data_range=float(config["rig"]["linear_psnr_data_range"]),
            display_exposure=float(config["rig"]["display_exposure"]),
        )
        row.update({"camera_index": camera_index, "light_index": light_index})
        rows.append(row)
    numeric = [key for key, value in rows[0].items() if isinstance(value, (int, float)) and not isinstance(value, bool)]
    return {
        "case_count": len(rows),
        "mean": {key: float(np.mean([row[key] for row in rows])) for key in numeric},
        "worst": {key: float(max(row[key] for row in rows)) for key in numeric},
        "cases": rows,
    }


def _state_from_raw(prepared: Mapping[str, Any]):
    raw = prepared["raw"]
    return (
        raw.latent_unorm8.cuda().float() / 255.0,
        raw.weight.cuda().float(),
        raw.bias.cuda().float(),
    )


def _save_observation(output: Path, step: int, state, prepared, config) -> dict[str, Any]:
    directory = output / "observations" / f"step_{step:05d}"
    directory.mkdir(parents=True, exist_ok=False)
    latent, weight, bias = state
    Image.fromarray(
        torch.floor(latent.detach().clamp(0.0, 1.0) * 255.0 + 0.5).to(torch.uint8).cpu().numpy(),
        mode="RGBA",
    ).save(directory / "latent_rgba8.png")
    affine = {
        "weight": weight.detach().cpu().tolist(),
        "bias": bias.detach().cpu().tolist(),
    }
    _write_json(directory / "affine.json", affine)
    metrics = _atlas_metrics(state, prepared)
    _write_json(directory / "material_metrics.json", metrics)
    _write_presentation_image(directory / "fixed_views.png", state, prepared, config)
    return metrics


def _display(hdr: torch.Tensor, exposure: float) -> Image.Image:
    value = hdr.detach().cpu().numpy().astype(np.float64) * exposure
    value = value / (1.0 + value)
    srgb = np.where(value <= 0.0031308, value * 12.92, 1.055 * np.power(value, 1.0 / 2.4) - 0.055)
    return Image.fromarray(np.clip(np.rint(srgb * 255.0), 0, 255).astype(np.uint8), mode="RGB")


@torch.no_grad()
def _write_presentation_image(path: Path, state, prepared, config) -> None:
    images = []
    for name, camera_index in prepared["rig"].presentation_views.items():
        geometry = prepared["geometries"][camera_index]
        latent, weight, bias = state
        sampled = bilinear_sample_top_down_wrap(hard_quantize_unorm8(latent), geometry.torch_buffers["uv"])
        processed = postprocess_affine_output(F.linear(sampled, weight, bias), compander_parameters=None, straight_through=False)
        hdr = shade_ggx(
            geometry,
            prepared["cameras"][camera_index],
            prepared["camera_lights"][camera_index][0],
            material_override=decoded_to_material(geometry, _material_override(processed)),
            minimum_roughness=float(config["rig"]["minimum_roughness"]),
        )
        image = _display(hdr, float(config["rig"]["display_exposure"]))
        canvas = Image.new("RGB", (image.width, image.height + 24), (20, 20, 23))
        canvas.paste(image, (0, 24))
        ImageDraw.Draw(canvas).text((6, 5), name, fill=(235, 235, 235))
        images.append(canvas)
    contact = Image.new("RGB", (sum(image.width for image in images), max(image.height for image in images)), (20, 20, 23))
    x = 0
    for image in images:
        contact.paste(image, (x, 0))
        x += image.width
    contact.save(path, format="PNG")


def run_arm(
    prepared: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    arm: str,
    output: Path,
    max_steps: int | None = None,
    resume: Path | None = None,
    raw_parent_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if arm not in ARMS:
        raise ValueError("unknown ablation arm")
    formal_steps = int(config["training"]["steps"])
    steps = formal_steps if max_steps is None else min(formal_steps, int(max_steps))
    if steps <= 0:
        raise ValueError("training steps must be positive")
    state, optimizers, rng = _new_state(prepared, config)
    initial_rng_hash = tensor_sha256(rng.get_state())
    start_step = 1
    curve, sample_metrics, observations, checkpoints = [], [], {}, {}
    if resume is None:
        if output.exists():
            raise FileExistsError(f"refusing to overwrite arm output: {output}")
        output.mkdir(parents=True)
    else:
        if not output.is_dir():
            raise ValueError("resume requires the existing arm output")
        payload = load_ablation_checkpoint(
            resume,
            expected_asset=prepared["asset_id"],
            expected_arm=arm,
            expected_identity=prepared["identity"],
        )
        for parameter, name in zip(state, ("latent", "weight", "bias"), strict=True):
            parameter.data.copy_(payload[name].to(parameter))
        for optimizer, name in zip(optimizers, ("latent_optimizer", "affine_optimizer"), strict=True):
            optimizer.load_state_dict(payload[name])
        rng.set_state(payload["rng_state"])
        initial_rng_hash = str(payload["initial_rng_hash"])
        start_step = int(payload["step"]) + 1
        checkpoint_step = int(payload["step"])
        snapshot = resume.parent / "progress_snapshot.json"
        if not snapshot.is_file():
            raise ValueError("resume checkpoint is missing its progress snapshot")
        quarantine = output / f"superseded_after_step_{checkpoint_step:05d}"
        stale: list[Path] = []
        for directory_name in ("observations", "checkpoints"):
            directory = output / directory_name
            if directory.is_dir():
                for child in directory.iterdir():
                    if child.is_dir() and child.name.startswith("step_"):
                        child_step = int(child.name.split("_", 1)[1])
                        if child_step > checkpoint_step:
                            stale.append(child)
        for file_name in ("progress.json", "progress.json.sha256", "training_report.json", "training_report.json.sha256"):
            candidate = output / file_name
            if candidate.exists():
                stale.append(candidate)
        if stale:
            if quarantine.exists():
                raise FileExistsError("resume quarantine already exists")
            quarantine.mkdir(parents=True)
            for source in stale:
                destination = quarantine / source.relative_to(output)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(destination))
        progress = json.loads(snapshot.read_text(encoding="utf-8"))
        curve = list(progress["curve"])
        sample_metrics = list(progress["sample_metrics"])
        observations = dict(progress["observations"])
        checkpoints = dict(progress["checkpoints"])
    if start_step > steps:
        raise ValueError("resume checkpoint is at or beyond the endpoint")
    weights = LossWeights.from_mapping(config["loss"])
    started = time.perf_counter()
    last_checkpoint = None
    for step in range(start_step, steps + 1):
        log_interval = int(config["training"]["log_interval"])
        compute_render = (
            arm == "material_render"
            or step == 1
            or step % log_interval == 0
            or step == steps
        )
        terms, context = _terms(
            state,
            prepared,
            config,
            _draw_batch(prepared, config, rng),
            compute_render=compute_render,
        )
        total, pieces = compose_ablation_loss(terms, arm=arm, weights=weights)
        if not bool(torch.isfinite(total)):
            raise FloatingPointError(f"non-finite training objective at step {step}")
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        total.backward()
        for parameter in state:
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError(f"non-finite or missing gradient at step {step}")
        for optimizer in optimizers:
            optimizer.step()
        with torch.no_grad():
            state[0].clamp_(0.0, 1.0)
        if step == 1 or step % log_interval == 0 or step == steps:
            row = {
                "step": step,
                "loss": float(total.detach()),
                **{name: float(value.detach()) for name, value in terms.items()},
                "weighted_material": float(pieces["material"].detach()),
                "weighted_render": float(pieces["render"].detach()),
                "diagnostic_render": float(pieces["diagnostic_render"].detach()),
                **context,
                "finite": True,
            }
            curve.append(row)
            print(json.dumps({"arm": arm, **row}, sort_keys=True), flush=True)
        if step % int(config["training"]["metric_interval"]) == 0 or step == steps:
            sample_metrics.append({"step": step, **_atlas_metrics(state, prepared)})
        if step in OBSERVATION_STEPS or step == steps:
            observations[str(step)] = _save_observation(output, step, state, prepared, config)
        if step in FULL_CHECKPOINT_STEPS:
            path = output / "checkpoints" / f"step_{step:05d}" / "checkpoint.pt"
            digest = save_ablation_checkpoint(
                path,
                asset=prepared["asset_id"],
                arm=arm,
                step=step,
                latent=state[0],
                weight=state[1],
                bias=state[2],
                latent_optimizer=optimizers[0],
                affine_optimizer=optimizers[1],
                rng=rng,
                identity=prepared["identity"],
                initial_rng_hash=initial_rng_hash,
            )
            checkpoints[str(step)] = {"path": str(path.relative_to(ROOT)), "sha256": digest}
            last_checkpoint = path
        if (
            step == 1
            or step % int(config["training"]["log_interval"]) == 0
            or step in OBSERVATION_STEPS
            or step in FULL_CHECKPOINT_STEPS
            or step == steps
        ):
            progress = {
                "schema_version": 1,
                "asset": prepared["asset_id"],
                "arm": arm,
                "steps": step,
                "curve": curve,
                "sample_metrics": sample_metrics,
                "observations": observations,
                "checkpoints": checkpoints,
                "sampling_contract_hash": prepared["identity"]["sampling_contract_hash"],
                "initial_rng_hash": initial_rng_hash,
                "final_rng_hash": tensor_sha256(rng.get_state()),
            }
            _write_json(output / "progress.json", progress)
            if step in FULL_CHECKPOINT_STEPS:
                _write_json(
                    output / "checkpoints" / f"step_{step:05d}" / "progress_snapshot.json",
                    progress,
                )
    preflight_checkpoint = None
    if steps < FULL_CHECKPOINT_STEPS[0]:
        preflight_checkpoint = output / "preflight_checkpoint.pt"
        torch.save(
            {
                "schema_version": 1,
                "checkpoint_type": "c4_render_ablation_preflight_v1",
                "asset": prepared["asset_id"],
                "arm": arm,
                "step": steps,
                "identity": prepared["identity"],
                "latent": state[0].detach().cpu(),
                "weight": state[1].detach().cpu(),
                "bias": state[2].detach().cpu(),
                "latent_optimizer": optimizers[0].state_dict(),
                "affine_optimizer": optimizers[1].state_dict(),
                "rng_state": rng.get_state().cpu(),
            },
            preflight_checkpoint,
        )
        reloaded = torch.load(preflight_checkpoint, map_location="cpu", weights_only=False)
        if (
            reloaded.get("step") != steps
            or reloaded.get("identity") != prepared["identity"]
            or not torch.equal(reloaded["latent"], state[0].detach().cpu())
        ):
            raise RuntimeError("preflight checkpoint reload verification failed")
    formal_run = steps == formal_steps
    train_pairs = (
        tuple(
            (camera, light)
            for camera in prepared["training_camera_indices"]
            for light in range(len(prepared["rig"].lights))
        )
        if formal_run
        else ((prepared["training_camera_indices"][0], 0),)
    )
    audit_pairs = (
        tuple(
            (camera, light)
            for camera in prepared["audit_camera_indices"]
            for light in range(len(prepared["rig"].lights))
        )
        if formal_run
        else ((prepared["audit_camera_indices"][0], 0),)
    )
    report = {
        "schema_version": 1,
        "status": "complete_20k" if steps == formal_steps else "complete_bounded_preflight",
        "asset": prepared["asset_id"],
        "arm": arm,
        "steps": steps,
        "elapsed_seconds": time.perf_counter() - started,
        "identity": prepared["identity"],
        "identity_hash": prepared["identity_hash"],
        "sampling_contract_hash": prepared["identity"]["sampling_contract_hash"],
        "initial_rng_hash": initial_rng_hash,
        "final_rng_hash": tensor_sha256(rng.get_state()),
        "sampling_trajectory_hash": sampling_trajectory_hash(
            sampling_contract=prepared["identity"]["sampling_contract_hash"],
            initial_rng=initial_rng_hash,
            final_rng=tensor_sha256(rng.get_state()),
            steps=steps,
        ),
        "raw_parent": dict(raw_parent_override) if raw_parent_override is not None else {
            "material": _atlas_metrics(_state_from_raw(prepared), prepared),
            "train_render": _render_metrics(_state_from_raw(prepared), prepared, config, train_pairs),
            "audit_render": _render_metrics(_state_from_raw(prepared), prepared, config, audit_pairs),
        },
        "endpoint": {
            "material": _atlas_metrics(state, prepared),
            "train_render": _render_metrics(state, prepared, config, train_pairs),
            "audit_render": _render_metrics(state, prepared, config, audit_pairs),
        },
        "observation_steps": sorted(int(value) for value in observations),
        "checkpoints": checkpoints,
        "last_checkpoint": str(last_checkpoint.relative_to(ROOT)) if last_checkpoint else None,
        "preflight_checkpoint": (
            {
                "path": str(preflight_checkpoint.relative_to(ROOT)),
                "sha256": sha256_file(preflight_checkpoint),
                "reload_verified": True,
            }
            if preflight_checkpoint is not None
            else None
        ),
        "formal_holdout_accessed": False,
        "audit_used_for_training": False,
        "early_stopping": False,
        "single_seed_statistical_significance_claimed": False,
    }
    _write_json(output / "training_report.json", report)
    return report


def run_pair(config_path: Path, asset_id: str, output: Path, *, max_steps: int | None = None) -> dict[str, Any]:
    config = _config(config_path)
    prepared = prepare_asset(config_path, config, asset_id)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite paired output: {output}")
    output.mkdir(parents=True)
    _write_json(output / "preparation.json", {
        "schema_version": 1,
        "asset": asset_id,
        "identity": prepared["identity"],
        "identity_hash": prepared["identity_hash"],
        "role": prepared["spec"]["role"],
        "tangent_source": prepared["spec"]["tangent_source"],
        "degenerate_uv_triangles": prepared["spec"]["degenerate_uv_triangles"],
        "emissive_policy": prepared["spec"]["emissive_policy"],
        "emissive_max_rgb_gt_0_05_fraction": prepared["spec"].get("emissive_max_rgb_gt_0_05_fraction"),
        "train_cameras": len(prepared["training_camera_indices"]),
        "audit_cameras": len(prepared["audit_camera_indices"]),
        "lights": len(prepared["rig"].lights),
        "audit_used_for_training": False,
    })
    formal_run = max_steps is None or int(max_steps) >= int(config["training"]["steps"])
    train_pairs = (
        tuple(
            (camera, light)
            for camera in prepared["training_camera_indices"]
            for light in range(len(prepared["rig"].lights))
        )
        if formal_run
        else ((prepared["training_camera_indices"][0], 0),)
    )
    audit_pairs = (
        tuple(
            (camera, light)
            for camera in prepared["audit_camera_indices"]
            for light in range(len(prepared["rig"].lights))
        )
        if formal_run
        else ((prepared["audit_camera_indices"][0], 0),)
    )
    raw_state = _state_from_raw(prepared)
    raw_parent = {
        "material": _atlas_metrics(raw_state, prepared),
        "train_render": _render_metrics(raw_state, prepared, config, train_pairs),
        "audit_render": _render_metrics(raw_state, prepared, config, audit_pairs),
    }
    reports = {
        arm: run_arm(
            prepared,
            config,
            arm=arm,
            output=output / arm,
            max_steps=max_steps,
            raw_parent_override=raw_parent,
        )
        for arm in ARMS
    }
    evidence = paired_sampling_evidence(reports[ARMS[0]], reports[ARMS[1]])
    if not evidence["identical"]:
        raise RuntimeError("paired training arms did not consume the same random sequence")
    summary = {
        "schema_version": 1,
        "status": "complete_paired_20k" if max_steps is None else "complete_paired_preflight",
        "asset": asset_id,
        "paired_sampling_evidence": evidence,
        "reports": {
            arm: str((output / arm / "training_report.json").relative_to(ROOT)) for arm in ARMS
        },
        "formal_holdout_accessed": False,
        "audit_used_for_training": False,
    }
    _write_json(output / "paired_summary.json", summary)
    return summary


def finalize_pair(output: Path, asset_id: str) -> dict[str, Any]:
    """Verify two independently completed arms and write paired evidence."""

    reports = {
        arm: json.loads((output / arm / "training_report.json").read_text(encoding="utf-8"))
        for arm in ARMS
    }
    for arm, report in reports.items():
        if (
            report.get("status") != "complete_20k"
            or report.get("asset") != asset_id
            or report.get("arm") != arm
            or report.get("steps") != 20000
        ):
            raise ValueError(f"cannot finalize incomplete paired arm: {arm}")
    evidence = paired_sampling_evidence(reports[ARMS[0]], reports[ARMS[1]])
    if not evidence["identical"]:
        raise RuntimeError("paired training arms did not consume the same random sequence")
    target = output / "paired_summary.json"
    if target.exists():
        raise FileExistsError("refusing to overwrite paired summary")
    summary = {
        "schema_version": 1,
        "status": "complete_paired_20k",
        "asset": asset_id,
        "paired_sampling_evidence": evidence,
        "reports": {
            arm: str((output / arm / "training_report.json").relative_to(ROOT)) for arm in ARMS
        },
        "formal_holdout_accessed": False,
        "audit_used_for_training": False,
    }
    _write_json(target, summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--asset", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--arm", choices=ARMS)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--finalize-pair", action="store_true")
    arguments = parser.parse_args()
    config_path = arguments.config.resolve()
    output = arguments.output_root.resolve()
    for path in (config_path, output):
        if not path.is_relative_to(ROOT):
            raise ValueError("config and output must remain inside the repository")
    if arguments.finalize_pair:
        if arguments.arm or arguments.resume or arguments.max_steps is not None:
            raise ValueError("--finalize-pair cannot be combined with training options")
        result = finalize_pair(output, arguments.asset)
    elif arguments.arm:
        config = _config(config_path)
        prepared = prepare_asset(config_path, config, arguments.asset)
        result = run_arm(
            prepared,
            config,
            arm=arguments.arm,
            output=output,
            max_steps=arguments.max_steps,
            resume=arguments.resume.resolve() if arguments.resume else None,
        )
    else:
        if arguments.resume:
            raise ValueError("paired mode does not accept --resume; resume one arm explicitly")
        result = run_pair(config_path, arguments.asset, output, max_steps=arguments.max_steps)
    print(json.dumps({"status": result["status"], "asset": arguments.asset}, sort_keys=True))


if __name__ == "__main__":
    main()
