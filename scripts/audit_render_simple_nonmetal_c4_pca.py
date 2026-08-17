"""Audit q1-q7 capacity and render source/raw/safe PCA on simple materials."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cg_frontier.assets.gltf_core4 import load_gltf_core4_asset  # noqa: E402
from cg_frontier.assets.preprocess import sha256_file  # noqa: E402
from cg_frontier.compression.affine_pca import (  # noqa: E402
    EnhancedPCASpec,
    export_p0_enhanced_bundle,
    export_p0_bundle,
    fit_global_valid_pca_oracle,
    rasterize_uv_charts,
)
from cg_frontier.render.gbuffer import (  # noqa: E402
    Camera,
    Core4Textures,
    render_geometry_gbuffer,
    sample_core4_material,
)
from cg_frontier.render.pbr import PointLight, linear_to_srgb_torch, shade_ggx  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/eval/simple_nonmetal_c4_pca_v1.yaml"
COLUMNS = ("source", "uniform_raw_q4", "chroma8_raw_q4", "oracle_q6")


def _repo_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty repository-relative path")
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"{label} escapes the repository")
    return path


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


def _metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    normal_prediction = torch.cat(
        (
            prediction[:, 3:5],
            torch.sqrt(
                torch.clamp(
                    1.0 - torch.sum(prediction[:, 3:5].square(), dim=-1, keepdim=True),
                    min=torch.finfo(prediction.dtype).eps,
                )
            ),
        ),
        dim=-1,
    )
    normal_prediction = F.normalize(normal_prediction, dim=-1)
    target_normal = torch.cat(
        (
            target[:, 3:5],
            torch.sqrt(
                torch.clamp(
                    1.0 - torch.sum(target[:, 3:5].square(), dim=-1, keepdim=True),
                    min=torch.finfo(target.dtype).eps,
                )
            ),
        ),
        dim=-1,
    )
    target_normal = F.normalize(target_normal, dim=-1)
    angle = torch.rad2deg(
        torch.acos(torch.sum(normal_prediction * target_normal, dim=-1).clamp(-1.0, 1.0))
    )
    return {
        "seven_channel_mae": float(F.l1_loss(prediction, target)),
        "base_color_linear_mae": float(F.l1_loss(prediction[:, :3], target[:, :3])),
        "normal_xy_mae": float(F.l1_loss(prediction[:, 3:5], target[:, 3:5])),
        "normal_mean_degrees": float(angle.mean()),
        "roughness_mae": float(F.l1_loss(prediction[:, 5], target[:, 5])),
        "metallic_mae": float(F.l1_loss(prediction[:, 6], target[:, 6])),
    }


def _prediction_textures(seven: torch.Tensor, hashes: Mapping[str, str]) -> Core4Textures:
    xy = seven[..., 3:5]
    z = torch.sqrt(
        torch.clamp(
            1.0 - torch.sum(xy.square(), dim=-1, keepdim=True),
            min=torch.finfo(seven.dtype).eps,
        )
    )
    normal = F.normalize(torch.cat((xy, z), dim=-1), dim=-1)
    return Core4Textures(
        base_color_linear=seven[..., :3].contiguous(),
        normal=((normal + 1.0) * 0.5).contiguous(),
        roughness=seven[..., 5:6].contiguous(),
        metallic=seven[..., 6:7].contiguous(),
        source_hashes=hashes,
    )


def _camera(
    mesh, view: str, render: Mapping[str, Any], spec: Mapping[str, Any]
) -> tuple[Camera, float]:
    lower = mesh.bounds_min.astype(np.float64)
    upper = mesh.bounds_max.astype(np.float64)
    center = (lower + upper) * 0.5
    radius = float(np.max(np.linalg.norm(mesh.positions.astype(np.float64) - center, axis=1)))
    if radius <= 0.0:
        raise ValueError("mesh bounds are degenerate")
    direction_value = spec.get(f"{view}_direction")
    if not isinstance(direction_value, list) or len(direction_value) != 3:
        raise ValueError(f"unsupported view: {view}")
    direction = np.asarray(direction_value, dtype=np.float64)
    if not np.isfinite(direction).all() or np.linalg.norm(direction) < 1e-8:
        raise ValueError(f"invalid camera direction for {view}")
    direction /= np.linalg.norm(direction)
    fov = float(render["vertical_fov_degrees"])
    distance = radius / math.sin(math.radians(fov) * 0.5) * float(
        render["distance_padding"]
    )
    eye = center + direction * distance
    return (
        Camera(
            eye=tuple(float(value) for value in eye),
            target=tuple(float(value) for value in center),
            up=(0.0, 1.0, 0.0),
            vertical_fov_degrees=fov,
            near=max(radius * 0.01, 1.0e-4),
            far=distance + radius * 2.5,
        ),
        radius,
    )


def _display(linear: torch.Tensor, exposure: float) -> np.ndarray:
    mapped = torch.clamp(linear * exposure, min=0.0)
    mapped = mapped / (1.0 + mapped)
    encoded = linear_to_srgb_torch(mapped).clamp(0.0, 1.0)
    return np.rint(encoded.detach().cpu().numpy() * 255.0).astype(np.uint8)


def _labeled_panel(images: list[np.ndarray], labels: list[str]) -> Image.Image:
    height, width = images[0].shape[:2]
    label_height = 34
    panel = Image.new("RGB", (width * len(images), height + label_height), (22, 22, 25))
    draw = ImageDraw.Draw(panel)
    for index, (array, label) in enumerate(zip(images, labels, strict=True)):
        panel.paste(Image.fromarray(array), (index * width, label_height))
        draw.text((index * width + 10, 10), label, fill=(235, 235, 235))
    return panel


def _write_json(path: Path, value: Any) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(payload)
    path.with_suffix(path.suffix + ".sha256").write_text(
        hashlib.sha256(payload).hexdigest() + "\n", encoding="ascii"
    )


def _process_asset(
    spec: Mapping[str, Any], config: Mapping[str, Any], output_root: Path
) -> dict[str, Any]:
    asset_id = str(spec["id"])
    print(f"[{asset_id}] load Core-4", flush=True)
    atlas_size = tuple(int(value) for value in config["atlas_resolution"])
    asset = load_gltf_core4_asset(
        _repo_path(spec["gltf"], f"assets.{asset_id}.gltf"),
        name=asset_id,
        expected_size=atlas_size,
        device="cpu",
    )
    target_atlas = _seven(asset.targets)
    valid_mask, chart_ids = rasterize_uv_charts(
        asset.mesh.texcoords,
        asset.mesh.triangles,
        height=asset.targets.height,
        width=asset.targets.width,
    )
    valid_target = target_atlas[valid_mask]
    if valid_target.shape[0] < 1024:
        raise ValueError(f"{asset_id} has too few valid UV texels")

    print(f"[{asset_id}] q1-q7 rank audit", flush=True)
    oracle = fit_global_valid_pca_oracle(target_atlas, valid_mask, rank=7)
    ranks: dict[str, Any] = {}
    oracle_predictions: dict[int, torch.Tensor] = {}
    for rank in range(1, 8):
        prediction = oracle.mean + oracle.valid_scores[:, :rank] @ oracle.components[:rank]
        ranks[f"q{rank}"] = {"deployable": rank <= 4, "metrics": _metrics(prediction, valid_target)}
        if rank == 6:
            oracle_predictions[rank] = prediction

    print(f"[{asset_id}] raw/safe deployable q4", flush=True)
    uniform_bundle = export_p0_bundle(
        target_atlas, valid_mask, chart_ids, margin=float(config["margin"])
    )
    chroma8_bundle = export_p0_enhanced_bundle(
        target_atlas,
        valid_mask,
        chart_ids,
        spec=EnhancedPCASpec(
            chroma_tail_strength=7.0,
            opponent_chroma_weight=2.0,
            semantic_group_balance=True,
        ),
        margin=float(config["margin"]),
    )
    uniform_latent = (
        uniform_bundle.calibration.raw.latent_unorm8[valid_mask].to(torch.float32)
        / 255.0
    )
    chroma8_latent = (
        chroma8_bundle.calibration.raw.latent_unorm8[valid_mask].to(torch.float32)
        / 255.0
    )
    raw_valid = F.linear(
        uniform_latent,
        uniform_bundle.calibration.raw.weight,
        uniform_bundle.calibration.raw.bias,
    )
    safe_valid = F.linear(
        uniform_latent,
        uniform_bundle.calibration.safe.weight,
        uniform_bundle.calibration.safe.bias,
    )
    chroma8_valid = F.linear(
        chroma8_latent,
        chroma8_bundle.calibration.raw.weight,
        chroma8_bundle.calibration.raw.bias,
    )
    reconstructed = {
        "uniform_raw_q4": target_atlas.clone(),
        "chroma8_raw_q4": target_atlas.clone(),
        "oracle_q6": target_atlas.clone(),
    }
    reconstructed["uniform_raw_q4"][valid_mask] = raw_valid
    reconstructed["chroma8_raw_q4"][valid_mask] = chroma8_valid
    reconstructed["oracle_q6"][valid_mask] = oracle_predictions[6]

    asset_root = output_root / asset_id
    for bundle_name, candidate_bundle in (
        ("uniform_pca_bundle", uniform_bundle),
        ("chroma8_pca_bundle", chroma8_bundle),
    ):
        artifact_root = asset_root / bundle_name
        artifact_root.mkdir(parents=True)
        for filename, payload in candidate_bundle.files.items():
            (artifact_root / filename).write_bytes(payload)
        bundle_manifest = dict(candidate_bundle.manifest)
        bundle_manifest["pipeline_id"] = "simple_nonmetal_c4_affine_pca_v1"
        _write_json(artifact_root / "manifest.json", bundle_manifest)

    print(f"[{asset_id}] CUDA render", flush=True)
    render = config["render"]
    resolution = tuple(int(value) for value in render["resolution"])
    render_images: dict[str, list[np.ndarray]] = {name: [] for name in render["views"]}
    render_metrics: dict[str, Any] = {}
    device_textures = {
        "source": Core4Textures(
            base_color_linear=asset.textures.base_color_linear.cuda().contiguous(),
            normal=asset.textures.normal.cuda().contiguous(),
            roughness=asset.textures.roughness.cuda().contiguous(),
            metallic=asset.textures.metallic.cuda().contiguous(),
            source_hashes=asset.textures.source_hashes,
        ),
        **{
            name: _prediction_textures(seven.cuda(), asset.textures.source_hashes)
            for name, seven in reconstructed.items()
        },
    }
    view_panels: list[Image.Image] = []
    for view_name in render["views"]:
        camera, radius = _camera(asset.mesh, str(view_name), render, spec)
        geometry = render_geometry_gbuffer(
            asset.mesh,
            camera,
            resolution,
            device="cuda",
            cull_backfaces=bool(render["backface_culling"]),
        )
        center = np.asarray(camera.target)
        light_position = center + radius * np.array((1.7, 2.1, 2.8))
        light = PointLight(
            position=tuple(float(value) for value in light_position),
            color=(1.0, 0.98, 0.95),
            radiant_intensity=float(render["light_radiant_intensity_per_radius_squared"]) * radius * radius,
            ambient_intensity=float(render["ambient_intensity"]),
        )
        images: list[np.ndarray] = []
        source_hdr = None
        for name in COLUMNS:
            material = sample_core4_material(geometry, device_textures[name])
            hdr = shade_ggx(
                geometry,
                camera,
                light,
                material_override=material,
                minimum_roughness=float(render["minimum_roughness"]),
            )
            if source_hdr is None:
                source_hdr = hdr
            mask = geometry.torch_buffers["mask"]
            images.append(_display(hdr, float(render["display_exposure"])))
            render_metrics[f"{view_name}.{name}"] = {
                "hdr_mae_vs_source": float(torch.mean(torch.abs(hdr[mask] - source_hdr[mask]))),
                "coverage_pixels": int(torch.count_nonzero(mask)),
            }
        panel = _labeled_panel(images, list(COLUMNS))
        panel_path = asset_root / f"{view_name}__source_uniform_chroma8_q6.png"
        panel.save(panel_path, format="PNG")
        view_panels.append(panel)
    combined_width = max(panel.width for panel in view_panels)
    combined_height = sum(panel.height for panel in view_panels)
    combined = Image.new("RGB", (combined_width, combined_height), (22, 22, 25))
    offset = 0
    for panel in view_panels:
        combined.paste(panel, (0, offset))
        offset += panel.height
    combined_path = asset_root / "comparison_source_uniform_chroma8_q6.png"
    combined.save(combined_path, format="PNG")

    report = {
        "schema_version": 1,
        "asset": asset.manifest,
        "stage": spec["stage"],
        "formal_holdout_accessed": False,
        "valid_uv": {
            "texels": int(torch.count_nonzero(valid_mask)),
            "ratio": float(valid_mask.to(torch.float64).mean()),
            "charts": int(torch.unique(chart_ids[valid_mask]).numel()),
        },
        "rank_distortion": ranks,
        "deployable_q4": {
            "uniform_raw_metrics": _metrics(raw_valid, valid_target),
            "chroma8_raw_metrics": _metrics(chroma8_valid, valid_target),
            "uniform_safe_failure_control_metrics": _metrics(safe_valid, valid_target),
            "uniform_raw_artifact_hash": uniform_bundle.calibration.raw.artifact_hash,
            "chroma8_raw_artifact_hash": chroma8_bundle.calibration.raw.artifact_hash,
            "uniform_safe_artifact_hash": uniform_bundle.calibration.safe.artifact_hash,
            "uniform_safe_certificate": uniform_bundle.calibration.safe.certificate,
            "runtime_cost": uniform_bundle.manifest["decoder_manifest"]["cost"],
            "runtime_cost_equal": (
                uniform_bundle.manifest["decoder_manifest"]["cost"]
                == chroma8_bundle.manifest["decoder_manifest"]["cost"]
            ),
        },
        "render": {
            "columns": list(COLUMNS),
            "oracle_q6_deployable": False,
            "metrics": render_metrics,
            "comparison_sha256": sha256_file(combined_path),
        },
    }
    _write_json(asset_root / "report.json", report)
    del device_textures, target_atlas, valid_target, reconstructed
    torch.cuda.empty_cache()
    return report


def run(config_path: Path, output_override: Path | None = None) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping) or config.get("schema_version") != 1:
        raise ValueError("unsupported config schema")
    if config.get("formal_holdout_access") != "forbidden":
        raise ValueError("simple-material audit must explicitly forbid formal holdout")
    output_root = (
        output_override.resolve()
        if output_override is not None
        else _repo_path(config["output_root"], "output_root")
    )
    if not output_root.is_relative_to(ROOT):
        raise ValueError("output root must stay inside the repository")
    lowered = output_root.as_posix().lower()
    if "formal_holdout" in lowered or "sealed" in lowered:
        raise ValueError("output root points at forbidden evaluation state")
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite output root: {output_root}")
    output_root.mkdir(parents=True)
    reports = [
        _process_asset(spec, config, output_root)
        for spec in config["assets"]
    ]
    contact_images = [
        Image.open(
            output_root
            / str(spec["id"])
            / "comparison_source_uniform_chroma8_q6.png"
        ).convert("RGB")
        for spec in config["assets"]
    ]
    contact_width = max(image.width for image in contact_images)
    contact_height = sum(image.height for image in contact_images)
    contact = Image.new("RGB", (contact_width, contact_height), (22, 22, 25))
    offset = 0
    for image in contact_images:
        contact.paste(image, (0, offset))
        offset += image.height
        image.close()
    contact_path = output_root / "teacher_contact_sheet.png"
    contact.save(contact_path, format="PNG")
    summary = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "status": "complete_cpu_audit_cuda_render",
        "formal_holdout_accessed": False,
        "config": str(config_path.resolve()),
        "config_sha256": sha256_file(config_path),
        "teacher_contact_sheet_sha256": sha256_file(contact_path),
        "assets": [
            {
                "id": report["asset"]["asset"],
                "stage": report["stage"],
                "valid_uv_texels": report["valid_uv"]["texels"],
                "uniform_raw_q4": report["deployable_q4"]["uniform_raw_metrics"],
                "chroma8_raw_q4": report["deployable_q4"]["chroma8_raw_metrics"],
                "uniform_safe_failure_control": report["deployable_q4"][
                    "uniform_safe_failure_control_metrics"
                ],
                "q6": report["rank_distortion"]["q6"]["metrics"],
            }
            for report in reports
        ],
    }
    _write_json(output_root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
