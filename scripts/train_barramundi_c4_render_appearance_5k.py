"""Train bounded fish-appearance affine candidates from a fresh random state."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw
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

from cg_frontier.assets.gltf_core4 import load_gltf_core4_asset  # noqa: E402
from cg_frontier.assets.preprocess import sha256_file  # noqa: E402
from cg_frontier.compression.affine_pca import (  # noqa: E402
    encode_pca_latent,
    fit_uniform_valid_pca,
    rasterize_uv_charts,
)
from cg_frontier.compression.render_loss import (  # noqa: E402
    bilinear_sample_top_down_wrap,
    fake_quantize_unorm8,
    hard_quantize_unorm8,
    masked_render_metrics,
    sparse_fake_quantized_bilinear_sample_top_down_wrap,
)
from cg_frontier.render.gbuffer import (  # noqa: E402
    Core4Textures,
    render_geometry_gbuffer,
    sample_core4_material,
)
from cg_frontier.render.pbr import PointLight, linear_to_srgb_torch, shade_ggx  # noqa: E402
from train_simple_nonmetal_c4_affine_render import (  # noqa: E402
    _camera,
    _full_metrics,
    _material_from_seven,
    _support_penalty,
    _support_statistics,
)


DEFAULT_CONFIG = ROOT / "configs/train/barramundi_c4_render_appearance_5k_v1.yaml"


def _repo_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a repository-relative path")
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"{label} escapes the repository")
    lowered = path.as_posix().lower()
    if "formal_holdout" in lowered or "sealed" in lowered:
        raise ValueError(f"{label} points at forbidden evaluation state")
    return path


def _write_json(path: Path, value: Any) -> str:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n", encoding="ascii")
    return digest


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    header = json.dumps(
        {"dtype": str(tensor.dtype), "shape": list(tensor.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(header + b"\0" + tensor.numpy().tobytes()).hexdigest()


def _initialization_spec(config: Mapping[str, Any]) -> dict[str, Any]:
    raw = config.get("initialization", {"mode": "uniform_raw_pca"})
    if not isinstance(raw, Mapping):
        raise ValueError("initialization must be a mapping")
    spec = dict(raw)
    mode = spec.get("mode")
    if mode == "uniform_raw_pca":
        if spec.get("pca_used", True) is not True:
            raise ValueError("uniform_raw_pca must set pca_used: true")
        if spec.get("train_all_affine_rows", False) is not False:
            raise ValueError("uniform_raw_pca must freeze the metallic affine row")
        return {
            "mode": mode,
            "pca_used": True,
            "train_all_affine_rows": False,
        }
    if mode != "seeded_random_mean_centered":
        raise ValueError(f"unsupported initialization mode: {mode}")
    if spec.get("pca_used") is not False:
        raise ValueError("seeded_random_mean_centered must set pca_used: false")
    if spec.get("latent_distribution") != "uniform_0_1":
        raise ValueError("random initialization requires latent_distribution: uniform_0_1")
    if spec.get("decoder_weight_distribution") != "normal_zero_mean":
        raise ValueError(
            "random initialization requires decoder_weight_distribution: normal_zero_mean"
        )
    if spec.get("decoder_bias_strategy") != "source_valid_mean_centered":
        raise ValueError(
            "random initialization requires decoder_bias_strategy: source_valid_mean_centered"
        )
    if spec.get("train_all_affine_rows") is not True:
        raise ValueError("random initialization must train all affine rows")
    seed = spec.get("seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("initialization.seed must be a non-negative integer")
    weight_std = spec.get("decoder_weight_std")
    if not isinstance(weight_std, (int, float)) or not math.isfinite(float(weight_std)) or float(weight_std) <= 0:
        raise ValueError("initialization.decoder_weight_std must be finite and positive")
    return {
        "mode": mode,
        "seed": seed,
        "pca_used": False,
        "latent_distribution": "uniform_0_1",
        "decoder_weight_distribution": "normal_zero_mean",
        "decoder_weight_std": float(weight_std),
        "decoder_bias_strategy": "source_valid_mean_centered",
        "train_all_affine_rows": True,
    }


def _initialize_parent_cpu(
    target: torch.Tensor,
    valid: torch.Tensor,
    spec: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if target.ndim != 3 or target.shape[-1] != 7:
        raise ValueError("target must have shape [height, width, 7]")
    if valid.shape != target.shape[:2] or valid.dtype != torch.bool or not bool(valid.any()):
        raise ValueError("valid mask must be non-empty bool [height, width]")
    if not bool(torch.isfinite(target[valid]).all()):
        raise ValueError("target valid texels must be finite")

    mode = spec["mode"]
    if mode == "uniform_raw_pca":
        pca = fit_uniform_valid_pca(target, valid)
        encoding = encode_pca_latent(pca)
        scores = (target - pca.mean) @ pca.components.transpose(0, 1)
        latent = torch.full(scores.shape, 0.5, dtype=scores.dtype)
        active = encoding.score_span > 0
        latent[..., active] = (
            scores[..., active] - encoding.score_min[active]
        ) / encoding.score_span[active]
        weight = encoding.weight.float()
        bias = encoding.bias.float()
    elif mode == "seeded_random_mean_centered":
        generator = torch.Generator(device="cpu").manual_seed(int(spec["seed"]))
        latent = torch.rand((*target.shape[:2], 4), dtype=target.dtype, generator=generator)
        weight = torch.randn((7, 4), dtype=target.dtype, generator=generator)
        weight = weight * float(spec["decoder_weight_std"])
        latent_mean = latent[valid].mean(dim=0)
        target_mean = target[valid].mean(dim=0)
        bias = target_mean - weight @ latent_mean
    else:
        raise ValueError(f"unsupported initialization mode: {mode}")

    latent = latent.clamp(0.0, 1.0).contiguous()
    weight = weight.contiguous()
    bias = bias.contiguous()
    if not all(bool(torch.isfinite(value).all()) for value in (latent, weight, bias)):
        raise FloatingPointError("initialization produced non-finite tensors")
    parent = {
        "latent": latent,
        "weight": weight,
        "bias": bias,
        "train_all_affine_rows": bool(spec["train_all_affine_rows"]),
    }
    metadata = {
        **dict(spec),
        "valid_texel_count": int(valid.sum()),
        "latent_valid_mean": [float(value) for value in latent[valid].mean(dim=0)],
        "source_valid_channel_mean": [float(value) for value in target[valid].mean(dim=0)],
        "decoded_valid_channel_mean": [
            float(value) for value in F.linear(latent[valid], weight, bias).mean(dim=0)
        ],
        "tensor_sha256": {
            "latent": _tensor_sha256(latent),
            "weight": _tensor_sha256(weight),
            "bias": _tensor_sha256(bias),
        },
    }
    return parent, metadata


def _affine_parameters(parent: Mapping[str, Any]):
    train_rows = 7 if bool(parent["train_all_affine_rows"]) else 6
    train_weight = nn.Parameter(parent["weight"][:train_rows].clone())
    train_bias = nn.Parameter(parent["bias"][:train_rows].clone())
    frozen_weight = parent["weight"][train_rows:].clone()
    frozen_bias = parent["bias"][train_rows:].clone()
    return train_weight, train_bias, frozen_weight, frozen_bias


def _seven(targets) -> torch.Tensor:
    return torch.cat(
        (
            targets.base_color_linear,
            targets.normal_xyz[:, :2],
            targets.roughness,
            targets.metallic,
        ),
        dim=-1,
    ).reshape(targets.height, targets.width, 7)


def _lights(camera, radius: float, offsets: list[list[float]], render: Mapping[str, Any]) -> list[PointLight]:
    center = np.asarray(camera.target, dtype=np.float64)
    intensity = float(render["light_radiant_intensity_per_radius_squared"]) * radius * radius
    return [
        PointLight(
            position=tuple(float(value) for value in center + radius * np.asarray(offset)),
            color=(1.0, 0.98, 0.95),
            radiant_intensity=intensity,
            ambient_intensity=float(render["ambient_intensity"]),
        )
        for offset in offsets
    ]


def _opponent(rgb: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        ((rgb[..., 0] - rgb[..., 1]) / math.sqrt(2.0), (rgb[..., 0] + rgb[..., 1] - 2.0 * rgb[..., 2]) / math.sqrt(6.0)),
        dim=-1,
    )


def _charbonnier(first: torch.Tensor, second: torch.Tensor, epsilon: float) -> torch.Tensor:
    difference = first - second
    return torch.sqrt(torch.sum(difference.square(), dim=-1) + epsilon * epsilon).mean()


def _gradient_loss(candidate: torch.Tensor, reference: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    horizontal_mask = mask[:, 1:] & mask[:, :-1]
    vertical_mask = mask[1:, :] & mask[:-1, :]
    candidate_x = candidate[:, 1:] - candidate[:, :-1]
    reference_x = reference[:, 1:] - reference[:, :-1]
    candidate_y = candidate[1:, :] - candidate[:-1, :]
    reference_y = reference[1:, :] - reference[:-1, :]
    values = []
    if bool(horizontal_mask.any()):
        values.append(torch.abs(candidate_x[horizontal_mask] - reference_x[horizontal_mask]).mean())
    if bool(vertical_mask.any()):
        values.append(torch.abs(candidate_y[vertical_mask] - reference_y[vertical_mask]).mean())
    if not values:
        raise ValueError("render gradient mask contains no adjacent valid pixels")
    return torch.stack(values).mean()


def _saliency_mask(references: list[torch.Tensor], mask: torch.Tensor, tail_mass: float) -> torch.Tensor:
    stack = torch.stack(references)
    variance = stack.std(dim=0, unbiased=False).mean(dim=-1)
    gray = stack.mean(dim=-1)
    gradient = torch.zeros_like(variance)
    gradient[:, 1:] += torch.abs(gray[:, :, 1:] - gray[:, :, :-1]).mean(dim=0)
    gradient[1:, :] += torch.abs(gray[:, 1:, :] - gray[:, :-1, :]).mean(dim=0)
    valid_variance = variance[mask]
    valid_gradient = gradient[mask]
    variance_scale = torch.quantile(valid_variance, 0.90).clamp_min(1e-8)
    gradient_scale = torch.quantile(valid_gradient, 0.90).clamp_min(1e-8)
    score = variance / variance_scale + gradient / gradient_scale
    threshold = torch.quantile(score[mask], 1.0 - tail_mass)
    tail = mask & (score >= threshold)
    if not bool(tail.any()):
        raise ValueError("source saliency tail is empty")
    return tail


def _prepare_pool(asset, textures, directions, offsets, resolution, render, tail_mass):
    pool = []
    for camera_index, direction in enumerate(directions):
        camera, radius = _camera(asset.mesh, direction, render)
        geometry = render_geometry_gbuffer(
            asset.mesh, camera, resolution, device="cuda", cull_backfaces=True
        )
        source_material = sample_core4_material(geometry, textures)
        references = []
        lights = _lights(camera, radius, offsets, render)
        for light in lights:
            references.append(
                shade_ggx(
                    geometry,
                    camera,
                    light,
                    material_override=source_material,
                    minimum_roughness=float(render["minimum_roughness"]),
                )
            )
        tail = _saliency_mask(
            references, geometry.torch_buffers["mask"], tail_mass
        )
        for light_index, (light, reference) in enumerate(zip(lights, references, strict=True)):
            pool.append(
                {
                    "name": f"camera{camera_index}_light{light_index}",
                    "geometry": geometry,
                    "camera": camera,
                    "light": light,
                    "reference": reference,
                    "tail": tail,
                }
            )
    return pool


def _fold_affine(train_weight, train_bias, metallic_weight, metallic_bias):
    return torch.cat((train_weight, metallic_weight), dim=0), torch.cat((train_bias, metallic_bias), dim=0)


@torch.no_grad()
def _pool_metrics(latent, weight, bias, pool, render, margin):
    deployed = hard_quantize_unorm8(latent)
    rows = []
    tail_errors = []
    opponent_errors = []
    gradient_errors = []
    support_values = []
    for case in pool:
        geometry = case["geometry"]
        mask = geometry.torch_buffers["mask"]
        sampled = bilinear_sample_top_down_wrap(deployed, geometry.torch_buffers["uv"])
        seven = F.linear(sampled, weight, bias)
        candidate = shade_ggx(
            geometry,
            case["camera"],
            case["light"],
            material_override=_material_from_seven(geometry, seven),
            minimum_roughness=float(render["minimum_roughness"]),
        )
        rows.append(
            masked_render_metrics(
                case["reference"],
                candidate,
                mask,
                linear_psnr_data_range=2.0,
                display_exposure=float(render["display_exposure"]),
            )
        )
        tail_errors.append(float(torch.mean(torch.abs(candidate[case["tail"]] - case["reference"][case["tail"]]))))
        opponent_errors.append(float(_charbonnier(_opponent(candidate[mask]), _opponent(case["reference"][mask]), 0.001)))
        gradient_errors.append(float(_gradient_loss(candidate, case["reference"], mask)))
        support_values.append(_support_statistics(seven[mask], margin))
    numeric_keys = [key for key, value in rows[0].items() if isinstance(value, (int, float)) and not isinstance(value, bool)]
    return {
        "case_count": len(rows),
        "mean": {key: float(np.mean([row[key] for row in rows])) for key in numeric_keys},
        "worst": {key: float(max(row[key] for row in rows)) for key in numeric_keys},
        "tail_hdr_mae": float(np.mean(tail_errors)),
        "opponent_error": float(np.mean(opponent_errors)),
        "gradient_error": float(np.mean(gradient_errors)),
        "support": {
            key: float(max(value[key] for value in support_values))
            for key in support_values[0]
        },
    }


def _checkpoint(path, *, candidate_id, step, latent, train_weight, train_bias, metallic_weight, metallic_bias, latent_optimizer, affine_optimizer, generator):
    path.parent.mkdir(parents=True, exist_ok=False)
    payload = {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "step": step,
        "latent": latent.detach().cpu(),
        "train_weight": train_weight.detach().cpu(),
        "train_bias": train_bias.detach().cpu(),
        "metallic_weight_frozen": metallic_weight.detach().cpu(),
        "metallic_bias_frozen": metallic_bias.detach().cpu(),
        "latent_optimizer": latent_optimizer.state_dict(),
        "affine_optimizer": affine_optimizer.state_dict(),
        "rng_state": generator.get_state().cpu(),
    }
    torch.save(payload, path)
    return sha256_file(path)


def _checkpoint_steps(training: Mapping[str, Any]) -> frozenset[int]:
    steps = int(training["steps"])
    configured = training.get("checkpoint_steps")
    if configured is None:
        interval = int(training["checkpoint_interval"])
        if interval <= 0:
            raise ValueError("checkpoint_interval must be positive")
        return frozenset((*range(interval, steps + 1, interval), steps))
    if not isinstance(configured, list) or not configured:
        raise ValueError("checkpoint_steps must be a non-empty list")
    checkpoint_steps = [int(value) for value in configured]
    if checkpoint_steps != sorted(set(checkpoint_steps)):
        raise ValueError("checkpoint_steps must be sorted and unique")
    if checkpoint_steps[0] <= 0 or checkpoint_steps[-1] > steps:
        raise ValueError("checkpoint_steps must be within the training range")
    if checkpoint_steps[-1] != steps:
        raise ValueError("checkpoint_steps must include the final training step")
    return frozenset(checkpoint_steps)


def _run_candidate(candidate, parent, target_flat, valid_indices, train_pool, audit_pool, config, root):
    device = torch.device("cuda")
    training = config["training"]
    loss_config = config["loss"]
    render = config["render"]
    latent = nn.Parameter(parent["latent"].clone())
    train_weight, train_bias, metallic_weight, metallic_bias = _affine_parameters(parent)
    latent_optimizer = torch.optim.Adam([latent], lr=float(training["latent_learning_rate"]))
    affine_optimizer = torch.optim.Adam([train_weight, train_bias], lr=float(training["affine_learning_rate"]))
    generator = torch.Generator(device=device).manual_seed(int(config["seed"]))
    margin = float(training["support_margin"])
    candidate_root = root / str(candidate["id"])
    candidate_root.mkdir(parents=True, exist_ok=False)
    curve = []
    trajectory = []
    checkpoint_steps = _checkpoint_steps(training)
    for step in range(1, int(training["steps"]) + 1):
        positions = torch.randint(
            0,
            valid_indices.numel(),
            (int(training["material_batch_size"]),),
            generator=generator,
            device=device,
        )
        indices = valid_indices[positions]
        weight, bias = _fold_affine(train_weight, train_bias, metallic_weight, metallic_bias)
        seven_batch = F.linear(fake_quantize_unorm8(latent.reshape(-1, 4)[indices]), weight, bias)
        target_batch = target_flat[indices]
        xy = seven_batch[:, 3:5]
        target_xy = target_batch[:, 3:5]
        normal = F.normalize(torch.cat((xy, torch.sqrt(torch.clamp(1.0 - xy.square().sum(-1, keepdim=True), min=1e-8))), dim=-1), dim=-1)
        target_normal = F.normalize(torch.cat((target_xy, torch.sqrt(torch.clamp(1.0 - target_xy.square().sum(-1, keepdim=True), min=1e-8))), dim=-1), dim=-1)
        material_terms = {
            "base_color": F.l1_loss(seven_batch[:, :3], target_batch[:, :3]),
            "normal": torch.mean(1.0 - (normal * target_normal).sum(-1)),
            "roughness": F.l1_loss(seven_batch[:, 5], target_batch[:, 5]),
            "metallic": F.l1_loss(seven_batch[:, 6], target_batch[:, 6]),
        }
        material = sum(material_terms[name] * float(loss_config[f"{name}_l1" if name != "normal" else "normal_cosine"]) for name in material_terms)
        case = train_pool[(step - 1) % len(train_pool)]
        geometry = case["geometry"]
        mask = geometry.torch_buffers["mask"]
        sampled = sparse_fake_quantized_bilinear_sample_top_down_wrap(latent, geometry.torch_buffers["uv"])
        sampled_seven = F.linear(sampled, weight, bias)
        rendered = shade_ggx(
            geometry,
            case["camera"],
            case["light"],
            material_override=_material_from_seven(geometry, sampled_seven),
            minimum_roughness=float(render["minimum_roughness"]),
        )
        reference = case["reference"]
        epsilon = float(loss_config["charbonnier_epsilon"])
        difference = rendered[mask] - reference[mask]
        hdr = torch.sqrt(difference.square() + epsilon * epsilon).mean()
        log_hdr = torch.abs(torch.log1p(rendered[mask].clamp_min(0.0)) - torch.log1p(reference[mask].clamp_min(0.0))).mean()
        opponent = _charbonnier(_opponent(rendered[mask]), _opponent(reference[mask]), epsilon)
        gradient = _gradient_loss(rendered, reference, mask)
        tail = torch.mean(torch.abs(rendered[case["tail"]] - reference[case["tail"]]))
        support = _support_penalty(seven_batch, margin) + _support_penalty(sampled_seven[mask], margin)
        total = (
            material
            + hdr * float(loss_config["render_charbonnier"])
            + log_hdr * float(loss_config["render_log1p"])
            + opponent * float(loss_config["render_opponent"])
            + gradient * float(loss_config["render_gradient"])
            + tail * float(candidate["tail_weight"])
            + support * float(training["support_penalty_weight"])
        )
        if not torch.isfinite(total):
            raise FloatingPointError(f"{candidate['id']} non-finite loss at step {step}")
        latent_optimizer.zero_grad(set_to_none=True)
        affine_optimizer.zero_grad(set_to_none=True)
        total.backward()
        latent_optimizer.step()
        affine_optimizer.step()
        with torch.no_grad():
            latent.clamp_(0.0, 1.0)
        if step == 1 or step % int(training["log_interval"]) == 0:
            record = {
                "step": step,
                "loss": float(total.detach()),
                "material": float(material.detach()),
                "hdr": float(hdr.detach()),
                "opponent": float(opponent.detach()),
                "gradient": float(gradient.detach()),
                "tail": float(tail.detach()),
                "support": float(support.detach()),
                "case": case["name"],
            }
            curve.append(record)
            print(f"[{candidate['id']}] {record}", flush=True)
        if step in checkpoint_steps:
            weight, bias = _fold_affine(train_weight, train_bias, metallic_weight, metallic_bias)
            material_metrics, atlas_support = _full_metrics(latent, weight, bias, valid_indices, target_flat, margin)
            train_metrics = _pool_metrics(latent, weight, bias, train_pool, render, margin)
            audit_metrics = _pool_metrics(latent, weight, bias, audit_pool, render, margin)
            checkpoint_path = candidate_root / "checkpoints" / f"step_{step:04d}" / "checkpoint.pt"
            checkpoint_hash = _checkpoint(
                checkpoint_path,
                candidate_id=str(candidate["id"]),
                step=step,
                latent=latent,
                train_weight=train_weight,
                train_bias=train_bias,
                metallic_weight=metallic_weight,
                metallic_bias=metallic_bias,
                latent_optimizer=latent_optimizer,
                affine_optimizer=affine_optimizer,
                generator=generator,
            )
            trajectory.append(
                {
                    "step": step,
                    "material": material_metrics,
                    "atlas_support": atlas_support,
                    "train_render": train_metrics,
                    "audit_render": audit_metrics,
                    "checkpoint": str(checkpoint_path.relative_to(root)),
                    "checkpoint_sha256": checkpoint_hash,
                }
            )
    report = {
        "schema_version": 1,
        "candidate": candidate,
        "steps": int(training["steps"]),
        "trainable_affine_rows": list(range(int(train_weight.shape[0]))),
        "metallic_affine_row_frozen": not bool(parent["train_all_affine_rows"]),
        "curve": curve,
        "trajectory": trajectory,
        "endpoint": trajectory[-1],
    }
    _write_json(candidate_root / "training_report.json", report)
    return report


def _display(linear, exposure):
    mapped = torch.clamp(linear * exposure, min=0.0)
    mapped = mapped / (1.0 + mapped)
    return np.rint(linear_to_srgb_torch(mapped).clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)


def _render_visual(asset, textures, parent, reports, config, root):
    render = config["render"]
    resolution = tuple(int(value) for value in render["display_resolution"])
    states = []
    for report in reports:
        checkpoint = torch.load(
            root / report["endpoint"]["checkpoint"], map_location="cuda", weights_only=False
        )
        weight, bias = _fold_affine(
            checkpoint["train_weight"].cuda(),
            checkpoint["train_bias"].cuda(),
            checkpoint["metallic_weight_frozen"].cuda(),
            checkpoint["metallic_bias_frozen"].cuda(),
        )
        states.append((str(report["candidate"]["id"]), checkpoint["latent"].cuda(), weight, bias))
    panels = []
    for direction in config["render"]["audit_camera_directions"][:2]:
        camera, radius = _camera(asset.mesh, direction, render)
        geometry = render_geometry_gbuffer(asset.mesh, camera, resolution, device="cuda", cull_backfaces=True)
        light = _lights(camera, radius, render["audit_light_offsets"][:1], render)[0]
        source = shade_ggx(
            geometry,
            camera,
            light,
            material_override=sample_core4_material(geometry, textures),
            minimum_roughness=float(render["minimum_roughness"]),
        )
        images = [_display(source, float(render["display_exposure"]))]
        labels = ["source"]
        parent_state = (
            str(config["initialization"]["mode"]),
            parent["latent"],
            parent["weight"],
            parent["bias"],
        )
        for name, latent, weight, bias in [parent_state] + states:
            sampled = bilinear_sample_top_down_wrap(hard_quantize_unorm8(latent), geometry.torch_buffers["uv"])
            seven = F.linear(sampled, weight, bias)
            hdr = shade_ggx(
                geometry,
                camera,
                light,
                material_override=_material_from_seven(geometry, seven),
                minimum_roughness=float(render["minimum_roughness"]),
            )
            images.append(_display(hdr, float(render["display_exposure"])))
            labels.append(name)
        height, width = images[0].shape[:2]
        panel = Image.new("RGB", (width * len(images), height + 34), (22, 22, 25))
        draw = ImageDraw.Draw(panel)
        for index, (image, label) in enumerate(zip(images, labels, strict=True)):
            panel.paste(Image.fromarray(image), (index * width, 34))
            draw.text((index * width + 10, 10), label, fill=(235, 235, 235))
        panels.append(panel)
    combined = Image.new("RGB", (panels[0].width, sum(panel.height for panel in panels)), (22, 22, 25))
    offset = 0
    for panel in panels:
        combined.paste(panel, (0, offset))
        offset += panel.height
    path = root / "comparison_source_parent_candidates.png"
    combined.save(path, format="PNG")
    return sha256_file(path)


def run(
    config_path: Path,
    *,
    output_root: str | None = None,
    max_steps: int | None = None,
    candidate_id: str | None = None,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping) or config.get("schema_version") != 1:
        raise ValueError("unsupported config schema")
    if config.get("formal_holdout_access") != "forbidden":
        raise ValueError("config must forbid formal holdout")
    config = copy.deepcopy(config)
    initialization = _initialization_spec(config)
    config["initialization"] = initialization
    if output_root is not None:
        config["output_root"] = output_root
    if max_steps is not None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        config["training"]["steps"] = max_steps
        if "checkpoint_steps" in config["training"]:
            config["training"]["checkpoint_steps"] = [
                step
                for step in config["training"]["checkpoint_steps"]
                if int(step) < max_steps
            ] + [max_steps]
        else:
            config["training"]["checkpoint_interval"] = min(
                int(config["training"]["checkpoint_interval"]), max_steps
            )
    if candidate_id is not None:
        candidates = [
            candidate
            for candidate in config["training"]["candidates"]
            if candidate["id"] == candidate_id
        ]
        if len(candidates) != 1:
            raise ValueError(f"unknown or duplicate candidate id: {candidate_id}")
        config["training"]["candidates"] = candidates
    root = _repo_path(config["output_root"], "output_root")
    if root.exists():
        raise FileExistsError(f"refusing to overwrite output root: {root}")
    root.mkdir(parents=True)
    asset_spec = config["asset"]
    asset = load_gltf_core4_asset(
        _repo_path(asset_spec["gltf"], "asset.gltf"),
        name=str(asset_spec["id"]),
        expected_size=tuple(int(value) for value in asset_spec["atlas_resolution"]),
        device="cpu",
    )
    target_cpu = _seven(asset.targets)
    valid_cpu, _ = rasterize_uv_charts(
        asset.mesh.texcoords,
        asset.mesh.triangles,
        height=asset.targets.height,
        width=asset.targets.width,
    )
    parent_cpu, initialization_metadata = _initialize_parent_cpu(
        target_cpu, valid_cpu, initialization
    )
    parent = {
        "latent": parent_cpu["latent"].cuda(),
        "weight": parent_cpu["weight"].cuda(),
        "bias": parent_cpu["bias"].cuda(),
        "train_all_affine_rows": parent_cpu["train_all_affine_rows"],
    }
    target = target_cpu.cuda()
    target_flat = target.reshape(-1, 7)
    valid_indices = torch.nonzero(valid_cpu.reshape(-1), as_tuple=False)[:, 0].cuda()
    textures = Core4Textures(
        base_color_linear=asset.textures.base_color_linear.cuda().contiguous(),
        normal=asset.textures.normal.cuda().contiguous(),
        roughness=asset.textures.roughness.cuda().contiguous(),
        metallic=asset.textures.metallic.cuda().contiguous(),
        source_hashes=asset.textures.source_hashes,
    )
    render = config["render"]
    resolution = tuple(int(value) for value in config["training"]["render_resolution"])
    with torch.no_grad():
        train_pool = _prepare_pool(
            asset,
            textures,
            render["train_camera_directions"],
            render["train_light_offsets"],
            resolution,
            render,
            float(config["training"]["tail_mass"]),
        )
        audit_pool = _prepare_pool(
            asset,
            textures,
            render["audit_camera_directions"],
            render["audit_light_offsets"],
            resolution,
            render,
            float(config["training"]["tail_mass"]),
        )
        parent_material, parent_support = _full_metrics(
            parent["latent"], parent["weight"], parent["bias"], valid_indices, target_flat, float(config["training"]["support_margin"])
        )
        parent_train = _pool_metrics(parent["latent"], parent["weight"], parent["bias"], train_pool, render, float(config["training"]["support_margin"]))
        parent_audit = _pool_metrics(parent["latent"], parent["weight"], parent["bias"], audit_pool, render, float(config["training"]["support_margin"]))
    reports = []
    for candidate in config["training"]["candidates"]:
        print(f"start {candidate['id']} fresh {config['training']['steps']} steps", flush=True)
        reports.append(
            _run_candidate(candidate, parent, target_flat, valid_indices, train_pool, audit_pool, config, root)
        )
    with torch.no_grad():
        visual_hash = _render_visual(asset, textures, parent, reports, config, root)
    summary = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "status": "complete_bounded_run",
        "initialization": initialization_metadata,
        "runtime_contract": {
            "texture": "2048x2048 RGBA8",
            "filtered_samples_per_pixel": 1,
            "decoder": "single 4_to_7 affine",
            "decoder_macs_per_pixel": 28,
            "network": False,
            "full_cube_certificate": False,
            "metallic_affine_row_frozen": not bool(parent["train_all_affine_rows"]),
        },
        "parent": {
            "material": parent_material,
            "atlas_support": parent_support,
            "train_render": parent_train,
            "audit_render": parent_audit,
        },
        "candidates": [
            {
                "id": report["candidate"]["id"],
                "endpoint": report["endpoint"],
            }
            for report in reports
        ],
        "comparison_sha256": visual_hash,
        "formal_holdout_accessed": False,
        "ue_started": False,
    }
    _write_json(root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--candidate")
    args = parser.parse_args()
    summary = run(
        args.config,
        output_root=args.output_root,
        max_steps=args.max_steps,
        candidate_id=args.candidate,
    )
    print(json.dumps({"status": summary["status"], "candidates": [value["id"] for value in summary["candidates"]]}, indent=2))


if __name__ == "__main__":
    main()
