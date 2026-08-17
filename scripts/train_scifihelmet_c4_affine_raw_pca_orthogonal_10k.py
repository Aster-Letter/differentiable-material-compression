"""Audit and train raw-PCA O1/O2 orthogonal BaseColor candidates."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for directory in (SRC, SCRIPTS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from cg_frontier.assets.gltf_mesh import load_gltf_mesh  # noqa: E402
from cg_frontier.compression.adaptive_basecolor import (  # noqa: E402
    AdaptiveBaseColorProfile,
    RenderColorVisibility,
    adaptive_group_chroma_loss,
    assign_basecolor_groups,
    draw_adaptive_color_batch,
    orthogonal_error_components,
    visibility_corrected_group_loss,
    weighted_orthogonal_domain_loss,
)
from cg_frontier.compression.affine_material import SCALAR_ROWS, decode_affine_material  # noqa: E402
from cg_frontier.compression.affine_pca import export_p0_bundle, rasterize_uv_charts  # noqa: E402
from cg_frontier.compression.affine_regularizers import fake_quantize_unorm8  # noqa: E402
from cg_frontier.compression.material import load_core4_targets  # noqa: E402
from cg_frontier.compression.raw_orthogonal_training import (  # noqa: E402
    load_raw_orthogonal_checkpoint,
    save_raw_orthogonal_checkpoint,
    sha256_file,
)
from cg_frontier.compression.render_loss import (  # noqa: E402
    bilinear_sample_top_down_wrap,
    decoded_to_material,
    hard_quantize_unorm8,
    sparse_fake_quantized_bilinear_sample_top_down_wrap,
)
from cg_frontier.render.gbuffer import (  # noqa: E402
    load_core4_textures,
    render_geometry_gbuffer,
    sample_core4_material,
)
from cg_frontier.render.pbr import shade_ggx  # noqa: E402
from render_scifihelmet_c4_affine_raw_pca import (  # noqa: E402
    _material_metrics,
    _repo_path,
    _support,
    _verify,
)
from run_scifihelmet_c4_affine_chroma8_l0_40k import _orbit_camera_from_spec  # noqa: E402
from run_scifihelmet_c4_affine_preflight import _light, _targets_to_seven  # noqa: E402
from train_scifihelmet_c4_affine_raw_pca_render_10k import (  # noqa: E402
    _atlas_metrics as _legacy_atlas_metrics,
    _render_metrics,
    _support_penalty,
)


DEFAULT_CONFIG = ROOT / "configs/train/scifihelmet_c4_affine_raw_pca_orthogonal_10k_v1.yaml"
GRADIENT_SUMMARY_STEPS = frozenset((1, 1000, 5000, 10000, 20000, 30000))


def _completion_status(steps: int) -> str:
    if steps <= 0:
        raise ValueError("completion steps must be positive")
    if steps % 1000 == 0:
        return f"complete_{steps // 1000}k"
    return f"complete_{steps}_steps"


def _load(path: Path) -> Mapping[str, Any]:
    payload = path.read_bytes()
    value = yaml.safe_load(payload) if path.suffix.lower() in {".yaml", ".yml"} else json.loads(payload)
    if not isinstance(value, Mapping):
        raise ValueError(f"mapping expected: {path}")
    return value


def _write_json(path: Path, value: Any) -> str:
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n", encoding="ascii")
    return digest


def _state_hash(
    latent: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    latent_optimizer: torch.optim.Optimizer,
    affine_optimizer: torch.optim.Optimizer,
    core_rng: torch.Generator,
    color_rng: torch.Generator,
) -> str:
    stream = io.BytesIO()
    torch.save(
        {
            "latent": latent.detach().cpu(),
            "weight": weight.detach().cpu(),
            "bias": bias.detach().cpu(),
            "latent_optimizer": latent_optimizer.state_dict(),
            "affine_optimizer": affine_optimizer.state_dict(),
            "core_rng": core_rng.get_state().cpu(),
            "color_rng": color_rng.get_state().cpu(),
        },
        stream,
    )
    return hashlib.sha256(stream.getvalue()).hexdigest()


def _prepare(config: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    if config.get("formal_holdout_access") != "forbidden":
        raise ValueError("formal holdout must remain forbidden")
    source = config["source"]
    preflight_path = _repo_path(source["preflight_config"], "source.preflight_config")
    pool_path = _repo_path(source["render_pool_config"], "source.render_pool_config")
    manifest_path = _repo_path(source["standard_p0_manifest"], "source.standard_p0_manifest")
    profile_path = _repo_path(source["adaptive_profile"], "source.adaptive_profile")
    _verify(preflight_path, source["preflight_config_sha256"], "preflight config")
    _verify(pool_path, source["render_pool_config_sha256"], "render pool config")
    _verify(manifest_path, source["standard_p0_manifest_sha256"], "standard P0 manifest")
    _verify(profile_path, source["adaptive_profile_sha256"], "adaptive profile")
    preflight, pool = _load(preflight_path), _load(pool_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mesh = load_gltf_mesh(_repo_path(preflight["inputs"]["gltf"], "inputs.gltf"))
    core4_dir = _repo_path(preflight["inputs"]["core4_dir"], "inputs.core4_dir")
    core4_manifest = _repo_path(preflight["inputs"]["core4_manifest"], "inputs.core4_manifest")
    targets_cpu = load_core4_targets(core4_dir, "cpu")
    target_seven_cpu = _targets_to_seven(targets_cpu)
    valid_mask, chart_ids = rasterize_uv_charts(
        mesh.texcoords,
        mesh.triangles,
        height=targets_cpu.height,
        width=targets_cpu.width,
    )
    bundle = export_p0_bundle(
        target_seven_cpu,
        valid_mask,
        chart_ids,
        margin=float(preflight["p0"]["safety_margin"]),
    )
    if bundle.manifest != manifest:
        raise RuntimeError("reconstructed standard P0 manifest mismatch")
    raw = bundle.calibration.raw
    if raw.artifact_hash != source["raw_parent_artifact_hash"]:
        raise RuntimeError("raw parent artifact hash mismatch")
    profile_payload = torch.load(profile_path, map_location="cpu", weights_only=False)
    profile = profile_payload.get("profile")
    visibility = profile_payload.get("visibility")
    profile_valid_positions = profile_payload.get("valid_positions")
    if not isinstance(profile, AdaptiveBaseColorProfile) or not isinstance(visibility, RenderColorVisibility):
        raise ValueError("adaptive profile artifact has invalid public types")
    if profile.profile_hash != source["basecolor_profile_hash"]:
        raise ValueError("adaptive BaseColor profile hash mismatch")
    if visibility.visibility_hash != source["visibility_hash"]:
        raise ValueError("render visibility hash mismatch")
    valid_positions_cpu = torch.nonzero(valid_mask.reshape(-1), as_tuple=False).flatten()
    if not torch.equal(valid_positions_cpu, profile_valid_positions):
        raise ValueError("adaptive profile valid positions mismatch")

    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    render = pool["render"]
    if tuple(render["resolution"]) != (256, 256) or len(pool["train_cameras"]) != 31 or len(pool["train_lights"]) != 6:
        raise ValueError("runner requires frozen camera31/light6 at 256x256")
    cameras = [_orbit_camera_from_spec(value, render) for value in pool["train_cameras"]]
    geometries = [render_geometry_gbuffer(mesh, camera, (256, 256), device=device) for camera in cameras]
    lights = [_light(value) for value in pool["train_lights"]]
    textures = load_core4_textures(core4_manifest, device)
    source_materials = [sample_core4_material(geometry, textures) for geometry in geometries]
    with torch.no_grad():
        references = [
            [
                shade_ggx(
                    geometry,
                    camera,
                    light,
                    material_override=source_material,
                    minimum_roughness=float(render["minimum_roughness"]),
                )
                for geometry, camera, source_material in zip(
                    geometries, cameras, source_materials, strict=True
                )
            ]
            for light in lights
        ]
    profile_cuda = profile.to(device)
    visibility_cuda = visibility.to(device)
    render_group_ids = [
        assign_basecolor_groups(material.base_color_linear, profile_cuda)
        for material in source_materials
    ]
    valid_indices = valid_positions_cpu.to(device)
    target_valid = target_seven_cpu.reshape(-1, 7).to(device)[valid_indices]
    lineage = {
        "parent_artifact_hash": raw.artifact_hash,
        "config_sha256": sha256_file(config_path),
        "input_sha256": hashlib.sha256(
            (
                sha256_file(_repo_path(preflight["inputs"]["gltf"], "inputs.gltf"))
                + sha256_file(core4_manifest)
            ).encode("ascii")
        ).hexdigest(),
        "basecolor_profile_hash": profile.profile_hash,
        "visibility_hash": visibility.visibility_hash,
    }
    return {
        "mesh": mesh,
        "textures": textures,
        "raw": raw,
        "render": render,
        "cameras": cameras,
        "geometries": geometries,
        "lights": lights,
        "references": references,
        "source_materials": source_materials,
        "render_group_ids": render_group_ids,
        "profile": profile_cuda,
        "visibility": visibility_cuda,
        "valid_indices": valid_indices,
        "target_valid": target_valid,
        "lineage": lineage,
    }


def _new_state(prepared: Mapping[str, Any], config: Mapping[str, Any]):
    raw = prepared["raw"]
    device = torch.device("cuda")
    latent = nn.Parameter(raw.latent_unorm8.to(device=device, dtype=torch.float32) / 255.0)
    weight = nn.Parameter(raw.weight.to(device=device, dtype=torch.float32).clone())
    bias = nn.Parameter(raw.bias.to(device=device, dtype=torch.float32).clone())
    training = config["training"]
    latent_optimizer = torch.optim.Adam((latent,), lr=float(training["latent_learning_rate"]))
    affine_optimizer = torch.optim.Adam((weight, bias), lr=float(training["affine_learning_rate"]))
    core_rng = torch.Generator(device=device).manual_seed(int(config["seed"]))
    color_rng = torch.Generator(device=device).manual_seed(int(config["seed"]) + 23)
    return latent, weight, bias, latent_optimizer, affine_optimizer, core_rng, color_rng


def _draw_batch(prepared, config, core_rng, color_rng):
    device = prepared["valid_indices"].device
    training = config["training"]
    positions = torch.randint(
        0,
        prepared["valid_indices"].numel(),
        (int(training["material_batch_size"]),),
        generator=core_rng,
        device=device,
    )
    camera_index = int(torch.randint(0, len(prepared["cameras"]), (1,), generator=core_rng, device=device))
    light_index = int(torch.randint(0, len(prepared["lights"]), (1,), generator=core_rng, device=device))
    color = draw_adaptive_color_batch(
        prepared["profile"],
        samples_per_group=int(training["color_samples_per_group"]),
        generator=color_rng,
    )
    return positions, camera_index, light_index, color


def _domain_losses(
    latent,
    weight,
    bias,
    prepared,
    config,
    batch,
    *,
    objective_id: str,
    scales: Mapping[str, Any] | None,
    ratio: float,
):
    positions, camera_index, light_index, color_batch = batch
    valid_indices = prepared["valid_indices"]
    target_valid = prepared["target_valid"]
    flat_indices = valid_indices[positions]
    material_seven = F.linear(
        fake_quantize_unorm8(latent.reshape(-1, 4)[flat_indices]), weight, bias
    )
    target_batch = target_valid[positions]
    epsilon = float(config["loss"]["charbonnier_epsilon"])
    base_old = F.l1_loss(material_seven[:, :3], target_batch[:, :3])
    base_y_errors, base_c_uniform_errors = orthogonal_error_components(
        material_seven[:, :3], target_batch[:, :3], epsilon=epsilon
    )
    color_flat = valid_indices[color_batch.valid_positions]
    color_prediction = F.linear(
        fake_quantize_unorm8(latent.reshape(-1, 4)[color_flat]), weight, bias
    )[:, :3]
    color_source = target_valid[color_batch.valid_positions, :3]
    _, color_errors = orthogonal_error_components(color_prediction, color_source, epsilon=epsilon)
    base_c_adaptive = adaptive_group_chroma_loss(
        color_errors,
        color_batch.group_ids,
        colored_group_count=prepared["profile"].k,
    )

    geometry = prepared["geometries"][camera_index]
    sampled = sparse_fake_quantized_bilinear_sample_top_down_wrap(
        latent, geometry.torch_buffers["uv"]
    )
    sampled_seven = F.linear(sampled, weight, bias)
    candidate = shade_ggx(
        geometry,
        prepared["cameras"][camera_index],
        prepared["lights"][light_index],
        material_override=decoded_to_material(
            geometry, decode_affine_material(sampled, weight, bias)
        ),
        minimum_roughness=float(prepared["render"]["minimum_roughness"]),
    )
    mask = geometry.torch_buffers["mask"]
    prediction_rgb = candidate[mask]
    source_rgb = prepared["references"][light_index][camera_index][mask]
    difference = prediction_rgb - source_rgb
    render_old = torch.sqrt(difference.square() + epsilon * epsilon).mean()
    render_y_errors, render_c_errors = orthogonal_error_components(
        prediction_rgb, source_rgb, epsilon=epsilon
    )
    render_ids = prepared["render_group_ids"][camera_index][mask]
    render_c_adaptive = visibility_corrected_group_loss(
        render_c_errors,
        render_ids,
        camera_index=camera_index,
        visibility=prepared["visibility"],
    )
    log_prediction = torch.log1p(prediction_rgb.clamp_min(0.0))
    log_source = torch.log1p(source_rgb.clamp_min(0.0))
    log_old = torch.abs(log_prediction - log_source).mean()
    log_y_errors, log_c_errors = orthogonal_error_components(
        log_prediction, log_source, epsilon=epsilon
    )
    log_c_adaptive = visibility_corrected_group_loss(
        log_c_errors,
        render_ids,
        camera_index=camera_index,
        visibility=prepared["visibility"],
    )
    raw_domains = {
        "base_color": {
            "old": base_old,
            "y_errors": base_y_errors,
            "c_uniform": base_c_uniform_errors.mean(),
            "c_adaptive": base_c_adaptive,
        },
        "render_linear": {
            "old": render_old,
            "y_errors": render_y_errors,
            "c_uniform": render_c_errors.mean(),
            "c_adaptive": render_c_adaptive,
        },
        "render_log": {
            "old": log_old,
            "y_errors": log_y_errors,
            "c_uniform": log_c_errors.mean(),
            "c_adaptive": log_c_adaptive,
        },
    }
    if scales is None:
        return raw_domains, material_seven, target_batch, sampled_seven, mask, camera_index, light_index
    variant = "uniform" if objective_id == "O1" else "adaptive"
    weighted: dict[str, torch.Tensor] = {}
    terms: dict[str, float] = {}
    for domain, values in raw_domains.items():
        total, pieces = weighted_orthogonal_domain_loss(
            values["y_errors"],
            values[f"c_{variant}"],
            ratio=ratio,
            y_scale=float(scales[domain]["y"]),
            chroma_scale=float(scales[domain][f"c_{variant}"]),
        )
        weighted[domain] = total
        for name, value in pieces.items():
            terms[f"{domain}_{name}"] = float(value.detach())
    return weighted, terms, material_seven, target_batch, sampled_seven, mask, camera_index, light_index


def _gradients(loss, latent, weight, bias, *, retain_graph: bool):
    values = torch.autograd.grad(
        loss, (latent, weight, bias), retain_graph=retain_graph, allow_unused=False
    )
    return values


def _group_norms(grads):
    latent, weight, bias = grads
    return {
        "latent": float(torch.linalg.vector_norm(latent).detach()),
        "affine": float(torch.sqrt(weight.square().sum() + bias.square().sum()).detach()),
    }


def _group_cosine(left, right):
    result = {}
    for name, indices in {"latent": (0,), "affine": (1, 2)}.items():
        dot = sum((left[i] * right[i]).sum() for i in indices)
        left_norm = torch.sqrt(sum(left[i].square().sum() for i in indices))
        right_norm = torch.sqrt(sum(right[i].square().sum() for i in indices))
        result[name] = float((dot / (left_norm * right_norm + 1.0e-30)).detach())
    return result


def run_audit(config_path: Path) -> dict[str, Any]:
    config = copy.deepcopy(_load(config_path))
    output = (ROOT / str(config["audit"]["output_root"])).resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite audit root: {output}")
    output.mkdir(parents=True)
    prepared = _prepare(config, config_path)
    state = _new_state(prepared, config)
    latent, weight, bias, latent_optimizer, affine_optimizer, core_rng, color_rng = state
    before = _state_hash(*state)
    core_rng_before = core_rng.get_state().clone()
    color_rng_before = color_rng.get_state().clone()
    batch_rows = []
    ratios: dict[str, dict[str, dict[str, list[float]]]] = {
        domain: {
            key: {"latent": [], "affine": []}
            for key in ("y", "c_uniform", "c_adaptive")
        }
        for domain in ("base_color", "render_linear", "render_log")
    }
    for batch_index in range(int(config["audit"]["batches"])):
        batch = _draw_batch(prepared, config, core_rng, color_rng)
        domains, *_ = _domain_losses(
            latent,
            weight,
            bias,
            prepared,
            config,
            batch,
            objective_id="O1",
            scales=None,
            ratio=0.25,
        )
        domain_rows = {}
        for domain_index, (domain, values) in enumerate(domains.items()):
            losses = {
                "old": values["old"],
                "y": values["y_errors"].mean(),
                "c_uniform": values["c_uniform"],
                "c_adaptive": values["c_adaptive"],
            }
            gradients = {}
            loss_items = list(losses.items())
            for loss_index, (name, loss) in enumerate(loss_items):
                retain = not (
                    domain_index == len(domains) - 1
                    and loss_index == len(loss_items) - 1
                )
                gradients[name] = _gradients(
                    loss, latent, weight, bias, retain_graph=retain
                )
            norms = {name: _group_norms(value) for name, value in gradients.items()}
            for name in ("y", "c_uniform", "c_adaptive"):
                for group in ("latent", "affine"):
                    ratios[domain][name][group].append(
                        norms["old"][group] / (norms[name][group] + 1.0e-30)
                    )
            domain_rows[domain] = {
                "losses": {name: float(loss.detach()) for name, loss in losses.items()},
                "norms": norms,
                "cosines": {
                    f"old_vs_{name}": _group_cosine(gradients["old"], gradients[name])
                    for name in ("y", "c_uniform", "c_adaptive")
                },
            }
            del gradients
        positions, camera_index, light_index, color_batch = batch
        batch_digest = hashlib.sha256(
            positions.detach().cpu().numpy().tobytes()
            + color_batch.valid_positions.detach().cpu().numpy().tobytes()
        ).hexdigest()
        batch_rows.append(
            {
                "batch": batch_index,
                "camera_index": camera_index,
                "light_index": light_index,
                "indices_hash": batch_digest,
                "domains": domain_rows,
            }
        )
    scales = {}
    for domain, values in ratios.items():
        scales[domain] = {}
        for name, groups in values.items():
            scales[domain][name] = min(
                float(np.median(groups["latent"])),
                float(np.median(groups["affine"])),
            )
    core_rng.set_state(core_rng_before)
    color_rng.set_state(color_rng_before)
    after = _state_hash(*state)
    if before != after:
        raise RuntimeError("zero-update gradient audit mutated training state")
    report = {
        "schema_version": 1,
        "status": "complete_zero_update_audit",
        "lineage": prepared["lineage"],
        "profile_k": prepared["profile"].k,
        "scales": scales,
        "ratios": ratios,
        "batches": batch_rows,
        "state_hash_before": before,
        "state_hash_after": after,
        "formal_holdout_accessed": False,
        "optimizer_updates": 0,
    }
    digest = _write_json(output / "gradient_audit.json", report)
    print(json.dumps({"status": report["status"], "sha256": digest}, sort_keys=True))
    return report


@torch.no_grad()
def _color_metrics(latent, weight, bias, prepared, positions=None):
    valid_indices = prepared["valid_indices"]
    target = prepared["target_valid"][:, :3]
    group_ids = prepared["profile"].valid_group_ids
    if positions is not None:
        valid_indices = valid_indices[positions]
        target = target[positions]
        group_ids = group_ids[positions]
    prediction = F.linear(
        hard_quantize_unorm8(latent).reshape(-1, 4)[valid_indices], weight, bias
    )[:, :3]
    y_errors, c_errors = orthogonal_error_components(prediction, target)
    group_means = [float(c_errors[group_ids == group].mean()) for group in range(prepared["profile"].group_count)]
    source_chroma = torch.linalg.vector_norm(
        orthogonal_error_coordinates(target), dim=-1
    )
    prediction_chroma = torch.linalg.vector_norm(
        orthogonal_error_coordinates(prediction), dim=-1
    )
    source_contrast = torch.std(source_chroma, unbiased=False)
    prediction_contrast = torch.std(prediction_chroma, unbiased=False)
    return {
        "uniform_y_error": float(y_errors.mean()),
        "uniform_opponent_error": float(c_errors.mean()),
        "adaptive_macro_opponent_error": 0.5 * group_means[0] + 0.5 * float(np.mean(group_means[1:])),
        "worst_adaptive_group_opponent_error": max(group_means),
        "group_opponent_errors": group_means,
        "chroma_contrast_retention": float(prediction_contrast / (source_contrast + 1.0e-30)),
    }


def orthogonal_error_coordinates(rgb: torch.Tensor) -> torch.Tensor:
    # Chroma coordinates only; local helper avoids materializing Y for metrics.
    red, green, blue = rgb.unbind(dim=-1)
    return torch.stack(
        ((red - green) / math.sqrt(2.0), (red + green - 2.0 * blue) / math.sqrt(6.0)),
        dim=-1,
    )


def run_candidate(
    config_path: Path,
    *,
    candidate_id: str,
    output_override: str | None = None,
    max_steps: int | None = None,
    resume: Path | None = None,
) -> dict[str, Any]:
    config = copy.deepcopy(_load(config_path))
    if candidate_id not in config["candidates"]:
        raise ValueError(f"unknown candidate: {candidate_id}")
    candidate = config["candidates"][candidate_id]
    objective_id = str(candidate["objective_id"])
    ratio = float(candidate["ratio"])
    output = (ROOT / str(output_override or candidate["output_root"])).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("candidate output must stay inside the repository")
    if resume is None:
        if output.exists():
            raise FileExistsError(f"refusing to overwrite candidate root: {output}")
        output.mkdir(parents=True)
    elif not output.is_dir():
        raise ValueError("resume requires the existing candidate output root")
    audit_path = (ROOT / str(config["audit"]["output_root"]) / "gradient_audit.json").resolve()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    prepared = _prepare(config, config_path)
    if audit["lineage"] != prepared["lineage"]:
        raise ValueError("gradient audit lineage mismatch")
    scales = audit["scales"]
    state = _new_state(prepared, config)
    latent, weight, bias, latent_optimizer, affine_optimizer, core_rng, color_rng = state
    parent = (latent.detach().clone(), weight.detach().clone(), bias.detach().clone())
    start_step = 1
    if resume is not None:
        payload = load_raw_orthogonal_checkpoint(
            resume,
            expected_candidate_id=candidate_id,
            expected_objective_id=objective_id,
            expected_ratio=ratio,
            expected_lineage=prepared["lineage"],
        )
        latent.data.copy_(payload["latent"].to(latent))
        weight.data.copy_(payload["weight"].to(weight))
        bias.data.copy_(payload["bias"].to(bias))
        latent_optimizer.load_state_dict(payload["latent_optimizer"])
        affine_optimizer.load_state_dict(payload["affine_optimizer"])
        core_rng.set_state(payload["rng_state"])
        color_rng.set_state(payload["color_rng_state"])
        start_step = int(payload["step"]) + 1
    steps = int(max_steps or config["training"]["steps"])
    if start_step > steps:
        raise ValueError("resume checkpoint is already at or beyond requested steps")
    audit_pairs = [tuple(int(v) for v in pair) for pair in config["audit_pairs"]]
    parent_report = {
        **_legacy_atlas_metrics(*parent, prepared["valid_indices"], prepared["target_valid"]),
        "color": _color_metrics(*parent, prepared),
        "audit_render": _render_metrics(
            *parent,
            prepared["geometries"],
            prepared["cameras"],
            prepared["lights"],
            prepared["references"],
            audit_pairs,
            prepared["render"],
        ),
    }
    metric_count = min(262_144, prepared["valid_indices"].numel())
    metric_positions = torch.linspace(
        0,
        prepared["valid_indices"].numel() - 1,
        metric_count,
        device="cuda",
    ).round().to(torch.int64)
    curve, trajectory, sample_metrics, gradient_summaries = [], [], [], []
    if resume is not None:
        progress_path = output / "progress.json"
        if not progress_path.is_file():
            raise ValueError("resume requires an existing progress.json")
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("candidate_id") != candidate_id:
            raise ValueError("resume progress candidate mismatch")
        curve = list(progress.get("curve", []))
        trajectory = list(progress.get("trajectory", []))
        sample_metrics = list(progress.get("sample_metrics", []))
        gradient_summaries = list(progress.get("gradient_summaries", []))
    started = time.perf_counter()
    last_checkpoint = None
    for step in range(start_step, steps + 1):
        batch = _draw_batch(prepared, config, core_rng, color_rng)
        weighted, orthogonal_terms, material_seven, target_batch, sampled_seven, mask, camera_index, light_index = _domain_losses(
            latent,
            weight,
            bias,
            prepared,
            config,
            batch,
            objective_id=objective_id,
            scales=scales,
            ratio=ratio,
        )
        pred_xy, target_xy = material_seven[:, 3:5], target_batch[:, 3:5]
        pred_z = torch.sqrt(torch.clamp(1.0 - pred_xy.square().sum(-1, keepdim=True), min=1.0e-8))
        target_z = torch.sqrt(torch.clamp(1.0 - target_xy.square().sum(-1, keepdim=True), min=1.0e-8))
        pred_normal = F.normalize(torch.cat((pred_xy, pred_z), dim=-1), dim=-1)
        target_normal = F.normalize(torch.cat((target_xy, target_z), dim=-1), dim=-1)
        material_terms = {
            "normal_cosine": torch.mean(1.0 - (pred_normal * target_normal).sum(-1)),
            "roughness_l1": F.l1_loss(material_seven[:, 5], target_batch[:, 5]),
            "metallic_l1": F.l1_loss(material_seven[:, 6], target_batch[:, 6]),
        }
        support = _support_penalty(material_seven, float(config["training"]["support_margin"])) + _support_penalty(
            sampled_seven[mask], float(config["training"]["support_margin"])
        )
        total = (
            weighted["base_color"] * float(config["loss"]["base_color"])
            + weighted["render_linear"] * float(config["loss"]["helmet_linear"])
            + weighted["render_log"] * float(config["loss"]["helmet_log1p"])
            + sum(material_terms[name] * float(config["loss"][name]) for name in material_terms)
            + support * float(config["training"]["support_penalty_weight"])
        )
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite loss at step {step}")
        latent_optimizer.zero_grad(set_to_none=True)
        affine_optimizer.zero_grad(set_to_none=True)
        total.backward()
        if step in GRADIENT_SUMMARY_STEPS:
            gradient_summaries.append(
                {
                    "step": 0 if step == 1 else step,
                    "latent": float(torch.linalg.vector_norm(latent.grad)),
                    "affine": float(torch.sqrt(weight.grad.square().sum() + bias.grad.square().sum())),
                }
            )
        latent_optimizer.step()
        affine_optimizer.step()
        with torch.no_grad():
            latent.clamp_(0.0, 1.0)
        if step == 1 or step % int(config["training"]["log_interval"]) == 0:
            point = {
                "step": step,
                "loss": float(total.detach()),
                **orthogonal_terms,
                **{name: float(value.detach()) for name, value in material_terms.items()},
                "support": float(support.detach()),
                "camera_index": camera_index,
                "light_index": light_index,
                "finite": True,
            }
            curve.append(point)
            print(json.dumps(point, sort_keys=True), flush=True)
        if step % int(config["training"]["sample_metric_interval"]) == 0 or step == steps:
            sample_metrics.append({"step": step, **_color_metrics(latent, weight, bias, prepared, metric_positions)})
        if step % int(config["training"]["checkpoint_interval"]) == 0 or step == steps:
            checkpoint_path = output / "checkpoints" / f"step_{step:05d}" / "checkpoint.pt"
            digest = save_raw_orthogonal_checkpoint(
                checkpoint_path,
                step=step,
                candidate_id=candidate_id,
                objective_id=objective_id,
                ratio=ratio,
                latent=latent,
                weight=weight,
                bias=bias,
                latent_optimizer=latent_optimizer,
                affine_optimizer=affine_optimizer,
                core_rng=core_rng,
                color_rng=color_rng,
                lineage=prepared["lineage"],
            )
            last_checkpoint = checkpoint_path
            trajectory.append(
                {
                    "step": step,
                    "checkpoint": checkpoint_path.relative_to(output).as_posix(),
                    "checkpoint_sha256": digest,
                    **_legacy_atlas_metrics(
                        latent,
                        weight,
                        bias,
                        prepared["valid_indices"],
                        prepared["target_valid"],
                    ),
                    "color": _color_metrics(latent, weight, bias, prepared),
                    "audit_render": _render_metrics(
                        latent,
                        weight,
                        bias,
                        prepared["geometries"],
                        prepared["cameras"],
                        prepared["lights"],
                        prepared["references"],
                        audit_pairs,
                        prepared["render"],
                    ),
                }
            )
            _write_json(
                output / "progress.json",
                {
                    "candidate_id": candidate_id,
                    "trajectory": trajectory,
                    "curve": curve,
                    "sample_metrics": sample_metrics,
                    "gradient_summaries": gradient_summaries,
                },
            )
            print(json.dumps({"checkpoint": step, "sha256": digest}), flush=True)
    assert last_checkpoint is not None
    load_raw_orthogonal_checkpoint(
        last_checkpoint,
        expected_candidate_id=candidate_id,
        expected_objective_id=objective_id,
        expected_ratio=ratio,
        expected_lineage=prepared["lineage"],
    )
    torch.cuda.synchronize()
    all_pairs = (
        [(camera, light) for light in range(len(prepared["lights"])) for camera in range(len(prepared["cameras"]))]
        if steps >= 10000
        else audit_pairs
    )
    endpoint_render = _render_metrics(
        latent,
        weight,
        bias,
        prepared["geometries"],
        prepared["cameras"],
        prepared["lights"],
        prepared["references"],
        all_pairs,
        prepared["render"],
    )
    parent_render = _render_metrics(
        *parent,
        prepared["geometries"],
        prepared["cameras"],
        prepared["lights"],
        prepared["references"],
        all_pairs,
        prepared["render"],
    )
    report = {
        "schema_version": 2,
        "status": _completion_status(steps),
        "candidate_id": candidate_id,
        "objective_id": objective_id,
        "ratio": ratio,
        "steps": steps,
        "lineage": prepared["lineage"],
        "profile_k": prepared["profile"].k,
        "gradient_audit_sha256": sha256_file(audit_path),
        "runtime_contract": {
            "texture": "2048x2048 RGBA8",
            "filtered_samples_per_pixel": 1,
            "decoder": "unconstrained single 4_to_7 affine",
            "decoder_macs_per_pixel": 28,
        },
        "parent": {**parent_report, "full_render_31x6": parent_render},
        "trajectory": trajectory,
        "sample_metrics": sample_metrics,
        "gradient_summaries": gradient_summaries,
        "endpoint": {**trajectory[-1], "full_render_31x6": endpoint_render},
        "curve": curve,
        "wall_seconds": time.perf_counter() - started,
        "formal_holdout_accessed": False,
        "ue_started": False,
        "yellow_diagnostics": {"selection_metric": False},
    }
    _write_json(output / "training_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--candidate")
    parser.add_argument("--output-root")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    if args.audit_only:
        run_audit(config_path)
        return
    if not args.candidate:
        raise SystemExit("--candidate is required unless --audit-only is used")
    report = run_candidate(
        config_path,
        candidate_id=args.candidate,
        output_override=args.output_root,
        max_steps=args.max_steps,
        resume=args.resume,
    )
    print(json.dumps({"status": report["status"], "candidate": report["candidate_id"], "steps": report["steps"]}))


if __name__ == "__main__":
    main()
