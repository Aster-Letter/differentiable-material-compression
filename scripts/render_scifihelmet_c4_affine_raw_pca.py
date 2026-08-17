"""Render a deterministic SciFiHelmet raw-vs-safe PCA preview."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for directory in (SRC, SCRIPTS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from cg_frontier.assets.gltf_mesh import load_gltf_mesh  # noqa: E402
from cg_frontier.compression.affine_material import (  # noqa: E402
    SCALAR_ROWS,
    decode_affine_material,
)
from cg_frontier.compression.affine_pca import (  # noqa: E402
    EnhancedPCASpec,
    export_p0_bundle,
    export_p0_enhanced_bundle,
    rasterize_uv_charts,
)
from cg_frontier.compression.material import load_core4_targets  # noqa: E402
from cg_frontier.compression.render_loss import (  # noqa: E402
    bilinear_sample_top_down_wrap,
    decoded_to_material,
    masked_render_metrics,
    orbit_camera,
)
from cg_frontier.render.gbuffer import (  # noqa: E402
    load_core4_textures,
    render_geometry_gbuffer,
    sample_core4_material,
)
from cg_frontier.render.pbr import linear_to_srgb_torch, shade_ggx  # noqa: E402
from run_scifihelmet_c4_affine_chroma8_l0_40k import _orbit_camera_from_spec  # noqa: E402
from run_scifihelmet_c4_affine_preflight import _light, _targets_to_seven  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/eval/scifihelmet_c4_affine_raw_pca_preview_v1.yaml"
DISPLAY_NAMES = {
    "source": ("Source", "original Core-4"),
    "uniform_raw_pca": ("Uniform raw PCA", "no cube safety"),
    "uniform_safe_pca": ("Uniform safe PCA", "full-cube certified"),
    "chroma8_safe": ("Chroma8 safe", "certified parent"),
}


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify(path: Path, expected: str, label: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA-256 mismatch: {actual} != {expected}")


def _load_yaml(path: Path) -> Mapping[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"mapping expected: {path}")
    return value


def _write_json(path: Path, value: Any) -> str:
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n", encoding="ascii")
    return digest


def _font(size: int, *, bold: bool = False):
    candidates = ("arialbd.ttf", "segoeuib.ttf") if bold else ("arial.ttf", "segoeui.ttf")
    windows_dir = os.environ.get("WINDIR")
    if not windows_dir:
        return ImageFont.load_default()
    for name in candidates:
        path = Path(windows_dir) / "Fonts" / name
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _display(linear: torch.Tensor, exposure: float) -> np.ndarray:
    mapped = torch.clamp(linear * exposure, min=0.0)
    mapped = mapped / (1.0 + mapped)
    return (
        linear_to_srgb_torch(mapped)
        .clamp(0.0, 1.0)
        .mul(255.0)
        .add(0.5)
        .to(torch.uint8)
        .cpu()
        .numpy()
    )


def _camera(view: Mapping[str, Any], render: Mapping[str, Any], cameras: list[object]):
    if "camera_index" in view:
        return cameras[int(view["camera_index"])]
    return orbit_camera(
        yaw_degrees=float(view["yaw_degrees"]),
        elevation_degrees=float(view["elevation_degrees"]),
        radius=float(view.get("radius", render["camera_radius"])),
        target=tuple(float(value) for value in view.get("target", render["target"])),
        up=tuple(float(value) for value in render["up"]),
        vertical_fov_degrees=float(render["vertical_fov_degrees"]),
        near=float(render["near"]),
        far=float(render["far"]),
    )


def _material_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred_xy = prediction[:, 3:5]
    target_xy = target[:, 3:5]
    pred_z = torch.sqrt(torch.clamp(1.0 - pred_xy.square().sum(-1), min=torch.finfo(prediction.dtype).eps))
    target_z = torch.sqrt(torch.clamp(1.0 - target_xy.square().sum(-1), min=torch.finfo(target.dtype).eps))
    pred_normal = F.normalize(torch.cat((pred_xy, pred_z[:, None]), dim=-1), dim=-1)
    target_normal = F.normalize(torch.cat((target_xy, target_z[:, None]), dim=-1), dim=-1)
    degrees = torch.rad2deg(torch.acos(torch.clamp((pred_normal * target_normal).sum(-1), -1.0, 1.0)))
    return {
        "seven_channel_mae": float(F.l1_loss(prediction, target)),
        "base_color_linear_mae": float(F.l1_loss(prediction[:, :3], target[:, :3])),
        "normal_mean_degrees": float(degrees.mean()),
        "roughness_mae": float(F.l1_loss(prediction[:, 5], target[:, 5])),
        "metallic_mae": float(F.l1_loss(prediction[:, 6], target[:, 6])),
    }


def _support(prediction: torch.Tensor) -> dict[str, float]:
    scalar = prediction[:, list(SCALAR_ROWS)]
    scalar_violation = torch.clamp(-scalar, min=0.0) + torch.clamp(scalar - 1.0, min=0.0)
    radius = torch.linalg.vector_norm(prediction[:, 3:5], dim=-1)
    normal_violation = torch.clamp(radius - 1.0, min=0.0)
    return {
        "scalar_violation_fraction": float((scalar_violation > 0.0).any(-1).float().mean()),
        "scalar_max_violation": float(scalar_violation.max()),
        "normal_violation_fraction": float((normal_violation > 0.0).float().mean()),
        "normal_max_violation": float(normal_violation.max()),
    }


def _cube_diagnostic(weight: torch.Tensor, bias: torch.Tensor) -> dict[str, Any]:
    scalar_weight = weight[list(SCALAR_ROWS)]
    scalar_bias = bias[list(SCALAR_ROWS)]
    lower = scalar_bias + torch.minimum(scalar_weight, torch.zeros_like(scalar_weight)).sum(-1)
    upper = scalar_bias + torch.maximum(scalar_weight, torch.zeros_like(scalar_weight)).sum(-1)
    vectors = 0.5 * weight[3:5].transpose(0, 1)
    center = bias[3:5] + vectors.sum(0)
    normal_bound = torch.linalg.vector_norm(center) + torch.linalg.vector_norm(vectors, dim=-1).sum()
    valid = bool(torch.all(lower >= 0.0) and torch.all(upper <= 1.0) and normal_bound < 1.0)
    return {
        "valid": valid,
        "scalar_lower_bounds": [float(value) for value in lower.cpu()],
        "scalar_upper_bounds": [float(value) for value in upper.cpu()],
        "normal_radius_upper_bound": float(normal_bound.cpu()),
    }


def _sheet(views, columns, images):
    tile = 256
    left, top, gap = 188, 108, 4
    width = left + len(columns) * (tile + gap) - gap
    height = top + len(views) * (tile + gap) - gap + 42
    canvas = Image.new("RGB", (width, height), (247, 247, 245))
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 14), "SciFiHelmet — raw PCA preview", fill=(28, 31, 35), font=_font(24, bold=True))
    draw.text((20, 48), "2048² RGBA8 · one filtered sample · single 4→7 affine · no training", fill=(78, 83, 90), font=_font(15))
    draw.text((20, 72), "Raw and safe share the same quantized latent; only the affine safety calibration differs.", fill=(90, 94, 100), font=_font(13))
    for column_index, column in enumerate(columns):
        x = left + column_index * (tile + gap)
        title, subtitle = DISPLAY_NAMES[column]
        draw.text((x + 5, 77), title, fill=(32, 35, 39), font=_font(14, bold=True))
        draw.text((x + 5, 95), subtitle, fill=(96, 100, 106), font=_font(11))
    for row_index, view in enumerate(views):
        y = top + row_index * (tile + gap)
        draw.text((20, y + 104), str(view["label"]), fill=(35, 38, 42), font=_font(17, bold=True))
        for column_index, column in enumerate(columns):
            x = left + column_index * (tile + gap)
            canvas.paste(Image.fromarray(images[(view["id"], column)], mode="RGB"), (x, y))
            draw.rectangle((x, y, x + tile - 1, y + tile - 1), outline=(185, 187, 190), width=1)
    draw.text((20, height - 28), "Top and rear-focus are presentation-only; no formal holdout was accessed.", fill=(100, 84, 38), font=_font(12))
    return canvas


def _atlas_sheet(columns, atlases):
    tile = 384
    top = 82
    gap = 4
    width = len(columns) * (tile + gap) - gap
    height = top + tile + 34
    canvas = Image.new("RGB", (width, height), (247, 247, 245))
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 13), "SciFiHelmet BaseColor atlas — raw versus safe", fill=(28, 31, 35), font=_font(22, bold=True))
    draw.text((18, 44), "Identical valid atlas and area downsampling", fill=(78, 83, 90), font=_font(13))
    for index, column in enumerate(columns):
        x = index * (tile + gap)
        draw.text((x + 8, 65), DISPLAY_NAMES[column][0], fill=(35, 38, 42), font=_font(13, bold=True))
        canvas.paste(Image.fromarray(atlases[column], mode="RGB"), (x, top))
    return canvas


def run(config_path: Path) -> dict[str, Any]:
    config = _load_yaml(config_path)
    if config.get("schema_version") != 1:
        raise ValueError("unsupported config schema")
    scope = config.get("scope")
    if not isinstance(scope, Mapping) or scope.get("formal_holdout_accessed") is not False:
        raise ValueError("formal holdout must remain forbidden")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for deterministic preview rendering")
    output = _repo_path(config["output_root"], "output_root")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output root: {output}")
    source = config["source"]
    paths = {}
    for key in ("preflight_config", "render_pool_config", "standard_p0_manifest", "chroma8_parent_manifest"):
        path = _repo_path(source[key], f"source.{key}")
        _verify(path, str(source[f"{key}_sha256"]), key)
        paths[key] = path

    preflight = _load_yaml(paths["preflight_config"])
    pool = _load_yaml(paths["render_pool_config"])
    standard_manifest = json.loads(paths["standard_p0_manifest"].read_text(encoding="utf-8"))
    chroma_manifest = json.loads(paths["chroma8_parent_manifest"].read_text(encoding="utf-8"))
    mesh = load_gltf_mesh(_repo_path(preflight["inputs"]["gltf"], "inputs.gltf"))
    core4_dir = _repo_path(preflight["inputs"]["core4_dir"], "inputs.core4_dir")
    core4_manifest = _repo_path(preflight["inputs"]["core4_manifest"], "inputs.core4_manifest")
    targets = load_core4_targets(core4_dir, "cpu")
    target_seven = _targets_to_seven(targets)
    valid_mask, chart_ids = rasterize_uv_charts(
        mesh.texcoords, mesh.triangles, height=targets.height, width=targets.width
    )
    margin = float(preflight["p0"]["safety_margin"])
    standard = export_p0_bundle(target_seven, valid_mask, chart_ids, margin=margin)
    if standard.manifest != standard_manifest:
        raise RuntimeError("reconstructed standard P0 manifest mismatch")
    pca = chroma_manifest["pca"]
    chroma = export_p0_enhanced_bundle(
        target_seven,
        valid_mask,
        chart_ids,
        spec=EnhancedPCASpec(
            chroma_tail_strength=float(pca["chroma_tail_strength"]),
            opponent_chroma_weight=float(pca["opponent_chroma_weight"]),
            semantic_group_balance=bool(pca["semantic_group_balance"]),
        ),
        margin=margin,
    )
    if chroma.manifest != chroma_manifest:
        raise RuntimeError("reconstructed chroma8 manifest mismatch")

    device = torch.device("cuda")
    variants = {
        "uniform_raw_pca": standard.calibration.raw,
        "uniform_safe_pca": standard.calibration.safe,
        "chroma8_safe": chroma.calibration.safe,
    }
    deployed = {
        key: (
            artifact.latent_unorm8.to(device=device, dtype=torch.float32) / 255.0,
            artifact.weight.to(device=device, dtype=torch.float32),
            artifact.bias.to(device=device, dtype=torch.float32),
        )
        for key, artifact in variants.items()
    }
    valid_indices = torch.nonzero(valid_mask.reshape(-1), as_tuple=False)[:, 0].to(device)
    target_valid = target_seven.reshape(-1, 7).to(device)[valid_indices]
    variant_report = {}
    source_rgb = torch.zeros(targets.height * targets.width, 3, device=device)
    source_rgb[valid_indices] = target_valid[:, :3]
    atlas_rgb = {"source": source_rgb.reshape(targets.height, targets.width, 3)}
    with torch.no_grad():
        for key, (latent, weight, bias) in deployed.items():
            prediction = F.linear(latent.reshape(-1, 4)[valid_indices], weight, bias)
            variant_report[key] = {
                "artifact_hash": variants[key].artifact_hash,
                "material": _material_metrics(prediction, target_valid),
                "observed_valid_support": _support(prediction),
                "full_cube": _cube_diagnostic(weight, bias),
            }
            full_rgb = torch.zeros(targets.height * targets.width, 3, device=device)
            full_rgb[valid_indices] = prediction[:, :3]
            atlas_rgb[key] = full_rgb.reshape(targets.height, targets.width, 3)

    output.mkdir(parents=True)
    render = pool["render"]
    cameras = [_orbit_camera_from_spec(value, render) for value in pool["train_cameras"]]
    lights = [_light(value) for value in pool["train_lights"]]
    textures = load_core4_textures(core4_manifest, device)
    views = [dict(value) for value in config["visual_contract"]["views"]]
    columns = [str(value) for value in config["visual_contract"]["columns"]]
    resolution = tuple(int(value) for value in config["visual_contract"]["resolution"])
    exposure = float(render["display_exposure"])
    images = {}
    cases = []
    render_root = output / "renders"
    render_root.mkdir()
    with torch.no_grad():
        for view in views:
            camera = _camera(view, render, cameras)
            geometry = render_geometry_gbuffer(mesh, camera, resolution, device=device)
            light = lights[int(view["light_index"])]
            reference = shade_ggx(
                geometry,
                camera,
                light,
                material_override=sample_core4_material(geometry, textures),
                minimum_roughness=float(render["minimum_roughness"]),
            )
            view_root = render_root / str(view["id"])
            view_root.mkdir()
            source_u8 = _display(reference, exposure)
            Image.fromarray(source_u8).save(view_root / "source.png")
            images[(view["id"], "source")] = source_u8
            case = {"view": view, "variants": {}}
            for key, (latent, weight, bias) in deployed.items():
                sampled = bilinear_sample_top_down_wrap(latent, geometry.torch_buffers["uv"])
                material = decoded_to_material(geometry, decode_affine_material(sampled, weight, bias))
                candidate = shade_ggx(
                    geometry,
                    camera,
                    light,
                    material_override=material,
                    minimum_roughness=float(render["minimum_roughness"]),
                )
                candidate_u8 = _display(candidate, exposure)
                Image.fromarray(candidate_u8).save(view_root / f"{key}.png")
                images[(view["id"], key)] = candidate_u8
                case["variants"][key] = masked_render_metrics(
                    reference,
                    candidate,
                    geometry.torch_buffers["mask"],
                    linear_psnr_data_range=float(render["linear_psnr_data_range"]),
                    display_exposure=exposure,
                )
            cases.append(case)

    comparison = output / "raw_safe_four_views.png"
    _sheet(views, columns, images).save(comparison, optimize=True)
    atlas_images = {}
    for key in columns:
        rgb = atlas_rgb[key].clamp(0.0, 1.0).permute(2, 0, 1)[None]
        small = F.interpolate(rgb, size=(384, 384), mode="area")[0].permute(1, 2, 0)
        atlas_images[key] = linear_to_srgb_torch(small).mul(255.0).add(0.5).to(torch.uint8).cpu().numpy()
    atlas_path = output / "basecolor_atlas_raw_safe.png"
    _atlas_sheet(columns, atlas_images).save(atlas_path, optimize=True)

    report = {
        "schema_version": 1,
        "status": "complete_read_only_raw_pca_preview",
        "runtime_contract": {
            "texture": "2048x2048 RGBA8",
            "filtered_samples_per_pixel": 1,
            "decoder": "single 4_to_7 affine",
            "decoder_macs_per_pixel": 28,
            "training": False,
        },
        "variants": variant_report,
        "render_cases": cases,
        "figures": {
            "raw_safe_four_views": {"path": comparison.relative_to(ROOT).as_posix(), "sha256": _sha256(comparison)},
            "basecolor_atlas_raw_safe": {"path": atlas_path.relative_to(ROOT).as_posix(), "sha256": _sha256(atlas_path)},
        },
        "formal_holdout_accessed": False,
        "training_started": False,
        "ue_started": False,
    }
    _write_json(output / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    report = run(args.config.resolve())
    print(json.dumps({"status": report["status"], "figures": list(report["figures"])}, sort_keys=True))


if __name__ == "__main__":
    main()
