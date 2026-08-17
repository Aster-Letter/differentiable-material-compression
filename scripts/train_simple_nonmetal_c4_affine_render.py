"""Bounded differentiable-render training for simple non-metal C4 assets."""

from __future__ import annotations

import argparse
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
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cg_frontier.assets.gltf_core4 import load_gltf_core4_asset  # noqa: E402
from cg_frontier.assets.preprocess import sha256_file  # noqa: E402
from cg_frontier.compression.affine_material import decode_affine_material  # noqa: E402
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
    Camera,
    Core4Textures,
    MaterialBuffers,
    render_geometry_gbuffer,
    sample_core4_material,
    tangent_normal_to_world,
)
from cg_frontier.render.pbr import PointLight, linear_to_srgb_torch, shade_ggx  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/train/simple_nonmetal_c4_affine_render_1k_v1.yaml"


def _repo_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty repository-relative path")
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"{label} escapes the repository")
    if "formal_holdout" in path.as_posix().lower() or "sealed" in path.as_posix().lower():
        raise ValueError(f"{label} points at forbidden evaluation state")
    return path


def _json_write(path: Path, value: Any) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    path.with_suffix(path.suffix + ".sha256").write_text(
        hashlib.sha256(payload).hexdigest() + "\n", encoding="ascii"
    )


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


def _camera(mesh, direction_value: list[float], render: Mapping[str, Any]) -> tuple[Camera, float]:
    lower = mesh.bounds_min.astype(np.float64)
    upper = mesh.bounds_max.astype(np.float64)
    center = (lower + upper) * 0.5
    radius = float(np.max(np.linalg.norm(mesh.positions.astype(np.float64) - center, axis=1)))
    direction = np.asarray(direction_value, dtype=np.float64)
    if direction.shape != (3,) or not np.isfinite(direction).all() or np.linalg.norm(direction) < 1e-8:
        raise ValueError("camera direction must contain three finite nonzero values")
    direction /= np.linalg.norm(direction)
    fov = float(render["vertical_fov_degrees"])
    distance = radius / math.sin(math.radians(fov) * 0.5) * float(render["distance_padding"])
    eye = center + direction * distance
    return Camera(
        eye=tuple(float(value) for value in eye),
        target=tuple(float(value) for value in center),
        up=(0.0, 1.0, 0.0),
        vertical_fov_degrees=fov,
        near=max(radius * 0.01, 1e-4),
        far=distance + radius * 2.5,
    ), radius


def _lights(camera: Camera, radius: float, render: Mapping[str, Any]) -> list[PointLight]:
    center = np.asarray(camera.target, dtype=np.float64)
    intensity = float(render["light_radiant_intensity_per_radius_squared"]) * radius * radius
    offsets = ((1.7, 2.1, 2.8), (-1.6, 1.45, 2.15))
    return [
        PointLight(
            position=tuple(float(value) for value in center + radius * np.asarray(offset)),
            color=(1.0, 0.98, 0.95),
            radiant_intensity=intensity,
            ambient_intensity=float(render["ambient_intensity"]),
        )
        for offset in offsets
    ]


def _material_from_seven(geometry, seven: torch.Tensor) -> MaterialBuffers:
    xy = seven[..., 3:5]
    z = torch.sqrt(
        torch.clamp(1.0 - torch.sum(xy.square(), dim=-1, keepdim=True), min=torch.finfo(seven.dtype).eps)
    )
    normal_ts = F.normalize(torch.cat((xy, z), dim=-1), dim=-1)
    normal_world = tangent_normal_to_world(geometry, normal_ts)
    mask = geometry.torch_buffers["mask"]
    return MaterialBuffers(
        base_color_linear=torch.where(mask[..., None], seven[..., :3], torch.zeros_like(seven[..., :3])),
        normal_world=torch.where(mask[..., None], normal_world, torch.zeros_like(normal_world)),
        roughness=torch.where(mask, seven[..., 5], torch.zeros_like(seven[..., 5])),
        metallic=torch.where(mask, seven[..., 6], torch.zeros_like(seven[..., 6])),
        normal_ts_raw=normal_ts,
        normal_ts_unit=normal_ts,
    )


def _support_penalty(seven: torch.Tensor, margin: float) -> torch.Tensor:
    scalars = seven[..., (0, 1, 2, 5, 6)]
    low = F.relu(margin - scalars)
    high = F.relu(scalars - (1.0 - margin))
    radius = torch.linalg.vector_norm(seven[..., 3:5], dim=-1)
    normal = F.relu(radius - (1.0 - margin))
    # L1 barriers retain a non-vanishing corrective gradient for the tiny
    # near-zero metallic excursions common in non-metal materials.
    return low.mean() + high.mean() + normal.mean()


def _support_statistics(seven: torch.Tensor, margin: float) -> dict[str, float]:
    scalars = seven[..., (0, 1, 2, 5, 6)]
    radius = torch.linalg.vector_norm(seven[..., 3:5], dim=-1)
    scalar_violation = torch.maximum(F.relu(margin - scalars), F.relu(scalars - (1.0 - margin)))
    normal_violation = F.relu(radius - (1.0 - margin))
    return {
        "scalar_violation_fraction": float(torch.mean((scalar_violation > 0).to(torch.float32))),
        "scalar_max_violation": float(scalar_violation.max()),
        "normal_violation_fraction": float(torch.mean((normal_violation > 0).to(torch.float32))),
        "normal_max_violation": float(normal_violation.max()),
    }


def _material_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    prediction_normal = F.normalize(
        torch.cat(
            (
                prediction[:, 3:5],
                torch.sqrt(torch.clamp(1.0 - prediction[:, 3:5].square().sum(-1, keepdim=True), min=1e-8)),
            ),
            dim=-1,
        ),
        dim=-1,
    )
    target_normal = F.normalize(
        torch.cat(
            (
                target[:, 3:5],
                torch.sqrt(torch.clamp(1.0 - target[:, 3:5].square().sum(-1, keepdim=True), min=1e-8)),
            ),
            dim=-1,
        ),
        dim=-1,
    )
    angles = torch.rad2deg(torch.acos((prediction_normal * target_normal).sum(-1).clamp(-1.0, 1.0)))
    return {
        "seven_channel_mae": float(F.l1_loss(prediction, target)),
        "base_color_linear_mae": float(F.l1_loss(prediction[:, :3], target[:, :3])),
        "normal_mean_degrees": float(angles.mean()),
        "roughness_mae": float(F.l1_loss(prediction[:, 5], target[:, 5])),
        "metallic_mae": float(F.l1_loss(prediction[:, 6], target[:, 6])),
    }


@torch.no_grad()
def _full_metrics(
    latent: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    valid_indices: torch.Tensor,
    target_flat: torch.Tensor,
    margin: float,
) -> tuple[dict[str, float], dict[str, float]]:
    deployed = hard_quantize_unorm8(latent).reshape(-1, 4)
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    for start in range(0, valid_indices.numel(), 262_144):
        indices = valid_indices[start : start + 262_144]
        predictions.append(F.linear(deployed[indices], weight, bias).cpu())
        targets.append(target_flat[indices].cpu())
    prediction = torch.cat(predictions)
    target = torch.cat(targets)
    return _material_metrics(prediction, target), _support_statistics(prediction, margin)


@torch.no_grad()
def _render_metrics(
    latent: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    cases: list[tuple[Any, Camera, PointLight, torch.Tensor]],
    render: Mapping[str, Any],
) -> dict[str, Any]:
    deployed = hard_quantize_unorm8(latent)
    rows: list[dict[str, float]] = []
    for geometry, camera, light, reference in cases:
        sampled = bilinear_sample_top_down_wrap(deployed, geometry.torch_buffers["uv"])
        seven = F.linear(sampled, weight, bias)
        hdr = shade_ggx(
            geometry,
            camera,
            light,
            material_override=_material_from_seven(geometry, seven),
            minimum_roughness=float(render["minimum_roughness"]),
        )
        rows.append(
            masked_render_metrics(
                reference,
                hdr,
                geometry.torch_buffers["mask"],
                linear_psnr_data_range=2.0,
                display_exposure=float(render["display_exposure"]),
            )
        )
    keys = [
        key
        for key, value in rows[0].items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return {
        "case_count": len(rows),
        "mean": {key: float(np.mean([row[key] for row in rows])) for key in keys},
        "worst": {key: float(max(row[key] for row in rows)) for key in keys},
    }


def _display(linear: torch.Tensor, exposure: float) -> np.ndarray:
    mapped = torch.clamp(linear * exposure, min=0.0)
    mapped = mapped / (1.0 + mapped)
    return np.rint(linear_to_srgb_torch(mapped).clamp(0.0, 1.0).cpu().numpy() * 255.0).astype(np.uint8)


def _panel(images: list[np.ndarray], labels: list[str]) -> Image.Image:
    height, width = images[0].shape[:2]
    result = Image.new("RGB", (width * len(images), height + 34), (22, 22, 25))
    draw = ImageDraw.Draw(result)
    for index, (image, label) in enumerate(zip(images, labels, strict=True)):
        result.paste(Image.fromarray(image), (index * width, 34))
        draw.text((index * width + 10, 10), label, fill=(235, 235, 235))
    return result


def _save_snapshot(root: Path, step: int, latent: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> None:
    target = root / "snapshots" / f"step_{step:04d}"
    target.mkdir(parents=True, exist_ok=False)
    latent_u8 = torch.floor(latent.detach().clamp(0.0, 1.0) * 255.0 + 0.5).to(torch.uint8).cpu().numpy()
    Image.fromarray(latent_u8, mode="RGBA").save(target / "latent_rgba8.png", format="PNG")
    np.savez(
        target / "decoder_raw_affine.npz",
        weight=weight.detach().cpu().numpy().astype(np.float32),
        bias=bias.detach().cpu().numpy().astype(np.float32),
    )


def _process_asset(spec: Mapping[str, Any], config: Mapping[str, Any], output_root: Path, steps: int) -> dict[str, Any]:
    asset_id = str(spec["id"])
    root = output_root / asset_id
    root.mkdir(parents=True)
    device = torch.device("cuda")
    print(f"[{asset_id}] load and initialize uniform raw PCA", flush=True)
    atlas_size = tuple(int(value) for value in config["atlas_resolution"])
    asset = load_gltf_core4_asset(
        _repo_path(spec["gltf"], f"assets.{asset_id}.gltf"),
        name=asset_id,
        expected_size=atlas_size,
        device="cpu",
    )
    target_cpu = _seven(asset.targets)
    valid_cpu, _ = rasterize_uv_charts(
        asset.mesh.texcoords,
        asset.mesh.triangles,
        height=asset.targets.height,
        width=asset.targets.width,
    )
    pca = fit_uniform_valid_pca(target_cpu, valid_cpu)
    encoding = encode_pca_latent(pca)
    full_scores = (target_cpu - pca.mean) @ pca.components.transpose(0, 1)
    latent_cpu = torch.full(full_scores.shape, 0.5, dtype=full_scores.dtype)
    active = encoding.score_span > 0
    latent_cpu[..., active] = (
        full_scores[..., active] - encoding.score_min[active]
    ) / encoding.score_span[active]
    latent = nn.Parameter(latent_cpu.clamp(0.0, 1.0).to(device))
    weight = nn.Parameter(encoding.weight.to(device=device, dtype=torch.float32))
    bias = nn.Parameter(encoding.bias.to(device=device, dtype=torch.float32))
    target = target_cpu.to(device)
    target_flat = target.reshape(-1, 7)
    valid_indices = torch.nonzero(valid_cpu.reshape(-1), as_tuple=False)[:, 0].to(device)

    source_textures = Core4Textures(
        base_color_linear=asset.textures.base_color_linear.to(device).contiguous(),
        normal=asset.textures.normal.to(device).contiguous(),
        roughness=asset.textures.roughness.to(device).contiguous(),
        metallic=asset.textures.metallic.to(device).contiguous(),
        source_hashes=asset.textures.source_hashes,
    )
    training = config["training"]
    render = config["render"]
    train_resolution = tuple(int(value) for value in training["render_resolution"])
    cases: list[tuple[Any, Camera, PointLight, torch.Tensor]] = []
    cameras: list[Camera] = []
    print(f"[{asset_id}] prepare 4 cameras x 2 lights", flush=True)
    with torch.no_grad():
        for direction in spec["camera_directions"]:
            camera, radius = _camera(asset.mesh, direction, render)
            cameras.append(camera)
            geometry = render_geometry_gbuffer(
                asset.mesh, camera, train_resolution, device=device, cull_backfaces=True
            )
            source_material = sample_core4_material(geometry, source_textures)
            for light in _lights(camera, radius, render):
                reference = shade_ggx(
                    geometry,
                    camera,
                    light,
                    material_override=source_material,
                    minimum_roughness=float(render["minimum_roughness"]),
                )
                cases.append((geometry, camera, light, reference))

    margin = float(training["support_margin"])
    parent_material, parent_support = _full_metrics(
        latent, weight, bias, valid_indices, target_flat, margin
    )
    parent_render = _render_metrics(latent, weight, bias, cases, render)
    parent_latent = latent.detach().clone()
    parent_weight = weight.detach().clone()
    parent_bias = bias.detach().clone()
    latent_optimizer = torch.optim.Adam([latent], lr=float(training["latent_learning_rate"]))
    affine_optimizer = torch.optim.Adam([weight, bias], lr=float(training["affine_learning_rate"]))
    generator = torch.Generator(device=device).manual_seed(int(config["seed"]))
    loss_config = config["loss"]
    curve: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    snapshot_steps = {value for value in training["checkpoint_steps"] if value <= steps}
    snapshot_steps.add(steps)
    print(f"[{asset_id}] train {steps} steps", flush=True)
    for step in range(1, steps + 1):
        selected_positions = torch.randint(
            0,
            valid_indices.numel(),
            (int(training["material_batch_size"]),),
            generator=generator,
            device=device,
        )
        selected = valid_indices[selected_positions]
        latent_batch = fake_quantize_unorm8(latent.reshape(-1, 4)[selected])
        seven_batch = F.linear(latent_batch, weight, bias)
        target_batch = target_flat[selected]
        xy = seven_batch[:, 3:5]
        predicted_normal = F.normalize(
            torch.cat(
                (xy, torch.sqrt(torch.clamp(1.0 - xy.square().sum(-1, keepdim=True), min=1e-8))),
                dim=-1,
            ),
            dim=-1,
        )
        target_xy = target_batch[:, 3:5]
        target_normal = F.normalize(
            torch.cat(
                (target_xy, torch.sqrt(torch.clamp(1.0 - target_xy.square().sum(-1, keepdim=True), min=1e-8))),
                dim=-1,
            ),
            dim=-1,
        )
        terms = {
            "base_color_l1": F.l1_loss(seven_batch[:, :3], target_batch[:, :3]),
            "normal_cosine": torch.mean(1.0 - (predicted_normal * target_normal).sum(-1)),
            "roughness_l1": F.l1_loss(seven_batch[:, 5], target_batch[:, 5]),
            "metallic_l1": F.l1_loss(seven_batch[:, 6], target_batch[:, 6]),
        }
        material_loss = (
            terms["base_color_l1"] * float(loss_config["base_color_l1"])
            + terms["normal_cosine"] * float(loss_config["normal_cosine"])
            + terms["roughness_l1"] * float(loss_config["roughness_l1"])
            + terms["metallic_l1"] * float(loss_config["metallic_l1"])
        )
        case_index = (step - 1) % len(cases)
        geometry, camera, light, reference = cases[case_index]
        sampled = sparse_fake_quantized_bilinear_sample_top_down_wrap(
            latent, geometry.torch_buffers["uv"]
        )
        sampled_seven = F.linear(sampled, weight, bias)
        candidate = shade_ggx(
            geometry,
            camera,
            light,
            material_override=_material_from_seven(geometry, sampled_seven),
            minimum_roughness=float(render["minimum_roughness"]),
        )
        mask = geometry.torch_buffers["mask"]
        difference = candidate[mask] - reference[mask]
        render_charbonnier = torch.sqrt(
            difference.square() + float(loss_config["charbonnier_epsilon"]) ** 2
        ).mean()
        render_log1p = torch.abs(
            torch.log1p(candidate[mask].clamp_min(0.0))
            - torch.log1p(reference[mask].clamp_min(0.0))
        ).mean()
        support = _support_penalty(seven_batch, margin) + _support_penalty(
            sampled_seven[mask], margin
        )
        total = (
            material_loss
            + render_charbonnier * float(loss_config["render_charbonnier"])
            + render_log1p * float(loss_config["render_log1p"])
            + support * float(training["support_penalty_weight"])
        )
        if not torch.isfinite(total):
            raise FloatingPointError(f"{asset_id} non-finite loss at step {step}")
        latent_optimizer.zero_grad(set_to_none=True)
        affine_optimizer.zero_grad(set_to_none=True)
        total.backward()
        latent_optimizer.step()
        affine_optimizer.step()
        with torch.no_grad():
            latent.clamp_(0.0, 1.0)
        if step == 1 or step % int(training["report_interval"]) == 0:
            record = {
                "step": step,
                "loss": float(total.detach()),
                "material_loss": float(material_loss.detach()),
                "render_charbonnier": float(render_charbonnier.detach()),
                "render_log1p": float(render_log1p.detach()),
                "support_penalty": float(support.detach()),
                "case_index": case_index,
            }
            curve.append(record)
            print(f"[{asset_id}] {record}", flush=True)
        if step in snapshot_steps:
            material_metrics, support_metrics = _full_metrics(
                latent, weight, bias, valid_indices, target_flat, margin
            )
            render_metrics = _render_metrics(latent, weight, bias, cases, render)
            _save_snapshot(root, step, latent, weight, bias)
            checkpoints.append(
                {
                    "step": step,
                    "material": material_metrics,
                    "support": support_metrics,
                    "render": render_metrics,
                }
            )

    endpoint_checkpoint = {
        "schema_version": 1,
        "asset": asset_id,
        "step": steps,
        "latent": latent.detach().cpu(),
        "weight": weight.detach().cpu(),
        "bias": bias.detach().cpu(),
        "latent_optimizer": latent_optimizer.state_dict(),
        "affine_optimizer": affine_optimizer.state_dict(),
        "rng_state": generator.get_state().cpu(),
    }
    torch.save(endpoint_checkpoint, root / "endpoint_checkpoint.pt")

    display_resolution = tuple(int(value) for value in render["display_resolution"])
    display_panels: list[Image.Image] = []
    with torch.no_grad():
        for view_index in (0, 3):
            camera, radius = _camera(asset.mesh, spec["camera_directions"][view_index], render)
            geometry = render_geometry_gbuffer(
                asset.mesh, camera, display_resolution, device=device, cull_backfaces=True
            )
            light = _lights(camera, radius, render)[0]
            source_hdr = shade_ggx(
                geometry,
                camera,
                light,
                material_override=sample_core4_material(geometry, source_textures),
                minimum_roughness=float(render["minimum_roughness"]),
            )
            images = [_display(source_hdr, float(render["display_exposure"]))]
            for candidate_latent, candidate_weight, candidate_bias in (
                (parent_latent, parent_weight, parent_bias),
                (latent, weight, bias),
            ):
                deployed = hard_quantize_unorm8(candidate_latent)
                sampled = bilinear_sample_top_down_wrap(deployed, geometry.torch_buffers["uv"])
                seven = F.linear(sampled, candidate_weight, candidate_bias)
                hdr = shade_ggx(
                    geometry,
                    camera,
                    light,
                    material_override=_material_from_seven(geometry, seven),
                    minimum_roughness=float(render["minimum_roughness"]),
                )
                images.append(_display(hdr, float(render["display_exposure"])))
            display_panels.append(_panel(images, ["source", "uniform_raw_q4", f"render_trained_{steps}"]))
    combined = Image.new(
        "RGB",
        (display_panels[0].width, sum(panel.height for panel in display_panels)),
        (22, 22, 25),
    )
    offset = 0
    for panel in display_panels:
        combined.paste(panel, (0, offset))
        offset += panel.height
    comparison_path = root / "comparison_source_uniform_render_trained.png"
    combined.save(comparison_path, format="PNG")

    report = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "asset": asset.manifest,
        "stage": spec["stage"],
        "steps": steps,
        "runtime_contract": {
            "texture": "2048x2048 RGBA8",
            "filtered_samples_per_pixel": 1,
            "decoder": "single unconstrained 4_to_7 affine",
            "decoder_macs_per_pixel": 28,
            "network": False,
            "full_cube_certificate": False,
            "safety_scope": "observed valid texels and fixed training render samples only",
        },
        "parent": {
            "material": parent_material,
            "support": parent_support,
            "render": parent_render,
        },
        "trajectory": checkpoints,
        "endpoint": checkpoints[-1],
        "curve": curve,
        "comparison_sha256": sha256_file(comparison_path),
        "formal_holdout_accessed": False,
        "ue_started": False,
    }
    _json_write(root / "training_report.json", report)
    del latent, weight, bias, target, source_textures
    torch.cuda.empty_cache()
    return report


def run(config_path: Path, output_override: Path | None, asset_filter: str | None, max_steps: int | None) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping) or config.get("schema_version") != 1:
        raise ValueError("unsupported config schema")
    if config.get("formal_holdout_access") != "forbidden":
        raise ValueError("training config must forbid formal holdout")
    output_root = output_override.resolve() if output_override else _repo_path(config["output_root"], "output_root")
    if not output_root.is_relative_to(ROOT):
        raise ValueError("output root must stay inside the repository")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite output root: {output_root}")
    output_root.mkdir(parents=True)
    configured_steps = int(config["training"]["steps"])
    steps = configured_steps if max_steps is None else min(configured_steps, max_steps)
    specs = [spec for spec in config["assets"] if asset_filter is None or spec["id"] == asset_filter]
    if not specs:
        raise ValueError("asset filter did not select an asset")
    reports = [_process_asset(spec, config, output_root, steps) for spec in specs]
    images = [Image.open(output_root / report["asset"]["asset"] / "comparison_source_uniform_render_trained.png").convert("RGB") for report in reports]
    contact = Image.new("RGB", (max(image.width for image in images), sum(image.height for image in images)), (22, 22, 25))
    offset = 0
    for image in images:
        contact.paste(image, (0, offset))
        offset += image.height
        image.close()
    contact_path = output_root / "teacher_contact_sheet.png"
    contact.save(contact_path, format="PNG")
    summary = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "status": "complete_bounded_differentiable_render_training",
        "steps": steps,
        "assets": [
            {
                "id": report["asset"]["asset"],
                "parent": report["parent"],
                "endpoint": report["endpoint"],
            }
            for report in reports
        ],
        "teacher_contact_sheet_sha256": sha256_file(contact_path),
        "formal_holdout_accessed": False,
        "ue_started": False,
    }
    _json_write(output_root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--asset")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    summary = run(args.config, args.output_root, args.asset, args.max_steps)
    print(json.dumps({"status": summary["status"], "steps": summary["steps"], "assets": [row["id"] for row in summary["assets"]]}, indent=2))


if __name__ == "__main__":
    main()
