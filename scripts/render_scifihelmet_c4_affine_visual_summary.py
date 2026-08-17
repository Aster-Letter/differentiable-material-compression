"""Build report-ready C4 affine progress and color-risk comparison figures."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for entry in (ROOT, SRC):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from cg_frontier.assets.gltf_mesh import load_gltf_mesh
from cg_frontier.compression.affine_color import (
    build_color_hue_partition,
    build_color_quantile_partition,
    opponent_vector_charbonnier_per_texel,
    orthogonal_color_coordinates,
)
from cg_frontier.compression.affine_pca import (
    EnhancedPCASpec,
    export_p0_bundle,
    export_p0_enhanced_bundle,
    rasterize_uv_charts,
)
from cg_frontier.compression.affine_regularizers import hard_quantize_unorm8
from cg_frontier.compression.material import load_core4_targets
from cg_frontier.compression.render_loss import (
    bilinear_sample_top_down_wrap,
    decoded_to_material,
    masked_render_metrics,
    orbit_camera,
)
from cg_frontier.render.gbuffer import (
    load_core4_textures,
    render_geometry_gbuffer,
    sample_core4_material,
)
from cg_frontier.render.pbr import shade_ggx
from diagnose_scifihelmet_c4_affine_color_guard_1k import (
    _atlas_u8,
    _display_u8,
    _save_png,
)
from run_scifihelmet_c4_affine_40k import _load_mapping
from run_scifihelmet_c4_affine_chroma8_l0_40k import _orbit_camera_from_spec
from run_scifihelmet_c4_affine_preflight import (
    _light,
    _move_p0,
    _repo_path,
    _targets_to_seven,
    _write_new,
)


DISPLAY_NAMES = {
    "source": ("Source", "Core-4 reference"),
    "standard_p0": ("Standard P0-safe", "uniform PCA + certificate"),
    "chroma8_parent": ("Chroma8 parent", "best certified global-q4"),
    "l0_80k": ("L0 at 80k", "render/material optimized"),
    "c0_1k": ("C0 at 1k", "no color guard"),
    "g2_1k": ("G2 at 1k", "hue8 macro, failed gate"),
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_sha(path: Path, expected: str, label: str) -> None:
    actual = _sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA-256 mismatch: {actual} != {expected}")


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = ("arialbd.ttf", "segoeuib.ttf") if bold else ("arial.ttf", "segoeui.ttf")
    windows_dir = os.environ.get("WINDIR")
    if not windows_dir:
        return ImageFont.load_default()
    for name in names:
        path = Path(windows_dir) / "Fonts" / name
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _save_figure(image: Image.Image, path: Path) -> dict[str, object]:
    image.save(path, format="PNG", optimize=True)
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "width": image.width,
        "height": image.height,
        "sha256": _sha256_file(path),
    }


def _progress_sheet(
    columns: list[str],
    views: list[dict[str, object]],
    images: Mapping[tuple[str, str], np.ndarray],
) -> Image.Image:
    tile = 256
    left = 212
    top = 112
    gap = 4
    footer = 42
    width = left + len(columns) * (tile + gap) - gap
    height = top + len(views) * (tile + gap) - gap + footer
    canvas = Image.new("RGB", (width, height), color=(247, 247, 245))
    draw = ImageDraw.Draw(canvas)
    draw.text((22, 16), "SciFiHelmet C4 affine — progress across four views", fill=(28, 31, 35), font=_font(24, bold=True))
    draw.text(
        (22, 49),
        "One RGBA8 texture · one filtered sample · single 4→7 affine · 256×256 deterministic renderer",
        fill=(78, 83, 90),
        font=_font(15),
    )
    draw.text(
        (22, 73),
        "Columns are separate lineages from frozen artifacts; within each row camera, light, exposure and tone mapping are identical.",
        fill=(94, 98, 104),
        font=_font(13),
    )
    for column_index, column in enumerate(columns):
        x = left + column_index * (tile + gap)
        title, subtitle = DISPLAY_NAMES[column]
        draw.text((x + 5, 82), title, fill=(32, 35, 39), font=_font(14, bold=True))
        draw.text((x + 5, 99), subtitle, fill=(96, 100, 106), font=_font(11))
    for row_index, view in enumerate(views):
        y = top + row_index * (tile + gap)
        label = str(view["label"])
        detail = str(view["detail"])
        draw.text((22, y + 93), label, fill=(35, 38, 42), font=_font(18, bold=True))
        draw.multiline_text((22, y + 120), detail, fill=(92, 96, 101), font=_font(12), spacing=3)
        for column_index, column in enumerate(columns):
            x = left + column_index * (tile + gap)
            canvas.paste(Image.fromarray(images[(str(view["id"]), column)], mode="RGB"), (x, y))
            draw.rectangle((x, y, x + tile - 1, y + tile - 1), outline=(185, 187, 190), width=1)
    draw.text(
        (22, height - 28),
        "Presentation note: top and rear-focus views are selection_metric=false; no formal holdout was accessed.",
        fill=(100, 84, 38),
        font=_font(12),
    )
    return canvas


def _error_atlas_u8(
    values: torch.Tensor,
    valid_indices: torch.Tensor,
    *,
    height: int,
    width: int,
    scale: float,
    palette: str,
) -> np.ndarray:
    normalized = (values / scale).clamp(0.0, 1.0)
    if palette == "tail":
        rgb = torch.stack((normalized, normalized * 0.28, normalized * 0.04), dim=-1)
    elif palette == "hue":
        rgb = torch.stack(
            (normalized, 0.15 + normalized * 0.35, 1.0 - normalized * 0.85), dim=-1
        )
    else:
        raise ValueError("unknown diagnostic palette")
    flat = torch.zeros(height * width, 3, device=values.device, dtype=values.dtype)
    flat[valid_indices] = rgb
    small = F.interpolate(
        flat.reshape(height, width, 3).permute(2, 0, 1)[None],
        size=(256, 256),
        mode="area",
    )[0].permute(1, 2, 0)
    return small.mul(255.0).add(0.5).to(torch.uint8).cpu().numpy()


def _atlas_sheet(
    columns: list[str],
    rows: list[dict[str, str]],
    images: Mapping[tuple[str, str], np.ndarray],
) -> Image.Image:
    tile = 256
    left = 230
    top = 104
    gap = 4
    footer = 48
    width = left + len(columns) * (tile + gap) - gap
    height = top + len(rows) * (tile + gap) - gap + footer
    canvas = Image.new("RGB", (width, height), color=(247, 247, 245))
    draw = ImageDraw.Draw(canvas)
    draw.text((22, 16), "SciFiHelmet C4 affine — material and tail-error progression", fill=(28, 31, 35), font=_font(24, bold=True))
    draw.text(
        (22, 50),
        "Source-defined partitions; identical valid atlas, scales and downsampling for every column",
        fill=(78, 83, 90),
        font=_font(15),
    )
    for column_index, column in enumerate(columns):
        x = left + column_index * (tile + gap)
        title, _ = DISPLAY_NAMES[column]
        draw.text((x + 5, 78), title, fill=(32, 35, 39), font=_font(13, bold=True))
    for row_index, row in enumerate(rows):
        y = top + row_index * (tile + gap)
        draw.text((22, y + 89), row["label"], fill=(35, 38, 42), font=_font(17, bold=True))
        draw.multiline_text((22, y + 116), row["detail"], fill=(92, 96, 101), font=_font(12), spacing=3)
        for column_index, column in enumerate(columns):
            x = left + column_index * (tile + gap)
            canvas.paste(Image.fromarray(images[(row["id"], column)], mode="RGB"), (x, y))
            draw.rectangle((x, y, x + tile - 1, y + tile - 1), outline=(185, 187, 190), width=1)
    draw.text(
        (22, height - 30),
        "Orange tail: black=0, saturated orange≥0.06465. Hue-group map: blue=low, orange≥0.03162. Yellow remains diagnostic only.",
        fill=(88, 91, 96),
        font=_font(12),
    )
    return canvas


def _stack_existing(
    title: str,
    subtitle: str,
    paths: list[Path],
) -> Image.Image:
    panels = [Image.open(path).convert("RGB") for path in paths]
    width = max(panel.width for panel in panels)
    top = 80
    gap = 8
    height = top + sum(panel.height for panel in panels) + gap * (len(panels) - 1) + 28
    canvas = Image.new("RGB", (width, height), color=(247, 247, 245))
    draw = ImageDraw.Draw(canvas)
    draw.text((18, 13), title, fill=(28, 31, 35), font=_font(22, bold=True))
    draw.text((18, 45), subtitle, fill=(78, 83, 90), font=_font(13))
    y = top
    for panel in panels:
        canvas.paste(panel, ((width - panel.width) // 2, y))
        y += panel.height + gap
    return canvas


def _checkpoint_model(
    path: Path,
    parent_decoder: torch.nn.Module,
    device: torch.device,
) -> tuple[torch.Tensor, torch.nn.Module, str]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    decoder = copy.deepcopy(parent_decoder)
    decoder.load_state_dict(checkpoint["safe_affine_raw_parameters"])
    decoder.to(device).eval()
    return checkpoint["latent"].to(device), decoder, str(checkpoint["checkpoint_hash"])


def _view_camera(
    view: Mapping[str, Any],
    render: Mapping[str, Any],
    pool_cameras: list[object],
) -> object:
    if "camera_index" in view:
        return pool_cameras[int(view["camera_index"])]
    target = tuple(float(value) for value in view.get("target", render["target"]))
    return orbit_camera(
        yaw_degrees=float(view["yaw_degrees"]),
        elevation_degrees=float(view["elevation_degrees"]),
        radius=float(view.get("radius", render["camera_radius"])),
        target=target,
        up=tuple(float(value) for value in render["up"]),
        vertical_fov_degrees=float(render["vertical_fov_degrees"]),
        near=float(render["near"]),
        far=float(render["far"]),
    )


def run(config_path: Path) -> dict[str, object]:
    config, _ = _load_mapping(config_path, "visual summary config")
    output_root = _repo_path(config["output_root"], "output_root")
    if output_root.exists():
        raise FileExistsError(f"refusing to inherit visual summary output: {output_root}")
    if not torch.cuda.is_available():
        raise RuntimeError("visual summary rendering requires CUDA")
    source = config["source"]
    checked_paths: dict[str, Path] = {}
    for key, value in source.items():
        if key.endswith("_sha256"):
            continue
        path = _repo_path(value, f"source.{key}")
        sha_key = f"{key}_sha256"
        if sha_key in source:
            _verify_sha(path, str(source[sha_key]), key)
        checked_paths[key] = path

    preflight, _ = _load_mapping(checked_paths["preflight_config"], "preflight")
    render_pool, _ = _load_mapping(checked_paths["render_pool_config"], "render pool")
    standard_manifest, _ = _load_mapping(checked_paths["standard_p0_manifest"], "standard P0")
    parent_manifest, _ = _load_mapping(checked_paths["chroma8_parent_manifest"], "chroma8 parent")

    gltf_path = _repo_path(preflight["inputs"]["gltf"], "inputs.gltf")
    core4_dir = _repo_path(preflight["inputs"]["core4_dir"], "inputs.core4_dir")
    core4_manifest = _repo_path(preflight["inputs"]["core4_manifest"], "inputs.core4_manifest")
    mesh = load_gltf_mesh(gltf_path)
    cpu_targets = load_core4_targets(core4_dir, "cpu")
    valid_mask, chart_ids = rasterize_uv_charts(
        mesh.texcoords,
        mesh.triangles,
        height=cpu_targets.height,
        width=cpu_targets.width,
    )
    target_seven = _targets_to_seven(cpu_targets)
    standard_bundle = export_p0_bundle(
        target_seven,
        valid_mask,
        chart_ids,
        margin=float(preflight["p0"]["safety_margin"]),
    )
    if standard_bundle.manifest != standard_manifest:
        raise RuntimeError("reconstructed standard P0 manifest mismatch")
    pca = parent_manifest["pca"]
    parent_bundle = export_p0_enhanced_bundle(
        target_seven,
        valid_mask,
        chart_ids,
        spec=EnhancedPCASpec(
            chroma_tail_strength=float(pca["chroma_tail_strength"]),
            opponent_chroma_weight=float(pca["opponent_chroma_weight"]),
            semantic_group_balance=bool(pca["semantic_group_balance"]),
        ),
        margin=float(preflight["p0"]["safety_margin"]),
    )
    if parent_bundle.manifest != parent_manifest:
        raise RuntimeError("reconstructed chroma8 parent manifest mismatch")

    device = torch.device("cuda")
    standard = _move_p0(standard_bundle.calibration, device)
    parent = _move_p0(parent_bundle.calibration, device)
    models: dict[str, tuple[torch.Tensor, torch.nn.Module]] = {
        "standard_p0": (
            standard.safe.latent_unorm8.to(torch.float32) / 255.0,
            standard.safe_decoder,
        ),
        "chroma8_parent": (
            parent.safe.latent_unorm8.to(torch.float32) / 255.0,
            parent.safe_decoder,
        ),
    }
    checkpoint_hashes: dict[str, str] = {}
    for key, path_key in (
        ("l0_80k", "l0_80k_checkpoint"),
        ("c0_1k", "c0_1k_checkpoint"),
        ("g2_1k", "g2_1k_checkpoint"),
    ):
        latent, decoder, checkpoint_hash = _checkpoint_model(
            checked_paths[path_key], parent.safe_decoder, device
        )
        models[key] = (latent, decoder)
        checkpoint_hashes[key] = checkpoint_hash
    deployed_models = {
        key: (hard_quantize_unorm8(latent), decoder)
        for key, (latent, decoder) in models.items()
    }

    output_root.mkdir(parents=True)
    render = render_pool["render"]
    exposure = float(render["display_exposure"])
    pool_cameras = [
        _orbit_camera_from_spec(value, render) for value in render_pool["train_cameras"]
    ]
    lights = [_light(value) for value in render_pool["train_lights"]]
    textures = load_core4_textures(core4_manifest, device)
    views = []
    for value in config["visual_contract"]["views"]:
        item = dict(value)
        if "camera_index" in item:
            camera_spec = render_pool["train_cameras"][int(item["camera_index"])]
            item["camera_name"] = str(camera_spec["name"])
            detail = f"{camera_spec['name']}\nlight {int(item['light_index'])}"
        else:
            item["camera_name"] = "visualization-only"
            detail = (
                f"elev {float(item['elevation_degrees']):.0f}° / yaw {float(item['yaw_degrees']):.0f}°\n"
                f"light {int(item['light_index'])} · presentation only"
            )
        item["detail"] = detail
        views.append(item)

    columns = [str(value) for value in config["visual_contract"]["columns"]]
    render_images: dict[tuple[str, str], np.ndarray] = {}
    render_cases: list[dict[str, object]] = []
    individual_root = output_root / "renders"
    individual_root.mkdir()
    with torch.no_grad():
        for view in views:
            view_id = str(view["id"])
            camera = _view_camera(view, render, pool_cameras)
            light_index = int(view["light_index"])
            geometry = render_geometry_gbuffer(
                mesh, camera, tuple(config["visual_contract"]["resolution"]), device=device
            )
            reference_material = sample_core4_material(geometry, textures)
            reference = shade_ggx(
                geometry,
                camera,
                lights[light_index],
                material_override=reference_material,
                minimum_roughness=float(render["minimum_roughness"]),
            )
            case_root = individual_root / view_id
            case_root.mkdir()
            reference_u8 = _display_u8(reference, exposure)
            _save_png(case_root / "source.png", reference_u8)
            render_images[(view_id, "source")] = reference_u8
            case: dict[str, object] = {
                "view": view,
                "variants": {},
            }
            for key, (latent, decoder) in deployed_models.items():
                sampled = bilinear_sample_top_down_wrap(latent, geometry.torch_buffers["uv"])
                material = decoded_to_material(geometry, decoder(sampled))
                candidate = shade_ggx(
                    geometry,
                    camera,
                    lights[light_index],
                    material_override=material,
                    minimum_roughness=float(render["minimum_roughness"]),
                )
                candidate_u8 = _display_u8(candidate, exposure)
                _save_png(case_root / f"{key}.png", candidate_u8)
                render_images[(view_id, key)] = candidate_u8
                case["variants"][key] = masked_render_metrics(
                    reference,
                    candidate,
                    geometry.torch_buffers["mask"],
                    linear_psnr_data_range=float(render["linear_psnr_data_range"]),
                    display_exposure=exposure,
                )
            render_cases.append(case)

    figure_records: dict[str, dict[str, object]] = {}
    figure_records["progress_four_views"] = _save_figure(
        _progress_sheet(columns, views, render_images),
        output_root / "progress_four_views.png",
    )

    valid_indices_cpu = torch.nonzero(valid_mask.reshape(-1), as_tuple=False)[:, 0]
    valid_indices = valid_indices_cpu.to(device)
    source_rgb_cpu = cpu_targets.select(valid_indices_cpu).base_color_linear
    source_rgb = source_rgb_cpu.to(device)
    partition = build_color_quantile_partition(source_rgb_cpu).to(device)
    hue = build_color_hue_partition(
        source_rgb_cpu, build_color_quantile_partition(source_rgb_cpu)
    ).to(device)
    predictions: dict[str, torch.Tensor] = {"source": source_rgb}
    with torch.no_grad():
        for key, (latent, decoder) in deployed_models.items():
            predictions[key] = decoder(
                latent.reshape(-1, 4)[valid_indices]
            ).base_color_linear.detach()
    source_opponent = orthogonal_color_coordinates(source_rgb)[:, 1:]
    errors = {
        key: opponent_vector_charbonnier_per_texel(
            orthogonal_color_coordinates(prediction)[:, 1:], source_opponent
        )
        for key, prediction in predictions.items()
    }
    tail_values: dict[str, torch.Tensor] = {}
    hue_values: dict[str, torch.Tensor] = {}
    tail_mass = float(config["visual_contract"]["tail_mass"])
    for key, values in errors.items():
        tail = torch.zeros_like(values)
        for offset, size in zip(partition.bin_offsets, partition.bin_sizes):
            start = int(offset)
            positions = partition.concatenated_valid_positions[start : start + int(size)]
            count = int(np.ceil(int(size) * tail_mass))
            order = torch.sort(values[positions], descending=True, stable=True).indices
            tail[positions[order[:count]]] = values[positions[order[:count]]]
        tail_values[key] = tail
        group_value = torch.zeros_like(values)
        for group_id in range(9):
            selected = hue.valid_group_ids == group_id
            group_value[selected] = values[selected].mean()
        hue_values[key] = group_value

    atlas_images: dict[tuple[str, str], np.ndarray] = {}
    for key in columns:
        atlas_images[("basecolor", key)] = _atlas_u8(
            predictions[key],
            valid_indices,
            height=cpu_targets.height,
            width=cpu_targets.width,
        )
        atlas_images[("yc_tail", key)] = _error_atlas_u8(
            tail_values[key],
            valid_indices,
            height=cpu_targets.height,
            width=cpu_targets.width,
            scale=float(config["visual_contract"]["tail_scale"]),
            palette="tail",
        )
        atlas_images[("hue_group", key)] = _error_atlas_u8(
            hue_values[key],
            valid_indices,
            height=cpu_targets.height,
            width=cpu_targets.width,
            scale=float(config["visual_contract"]["hue_scale"]),
            palette="hue",
        )
    atlas_rows = [
        {"id": "basecolor", "label": "BaseColor atlas", "detail": "linear RGB\nvalid texels"},
        {"id": "yc_tail", "label": "YC tail error", "detail": "worst 25%\ninside each YC bin"},
        {"id": "hue_group", "label": "Hue-group error", "detail": "neutral + hue8\ngroup mean"},
    ]
    figure_records["progress_error_atlases"] = _save_figure(
        _atlas_sheet(columns, atlas_rows, atlas_images),
        output_root / "progress_error_atlases.png",
    )

    figure_records["color_guard_ablation"] = _save_figure(
        _stack_existing(
            "Color-guard ablation — equal-bin opponent objectives",
            "Source / parent / C0 / C1-C2 at r=.10 and .25 · yellow diagnostics are selection_metric=false",
            [
                checked_paths["color_guard_basecolor"],
                checked_paths["color_guard_yellow"],
                checked_paths["color_guard_r_minus_b"],
            ],
        ),
        output_root / "color_guard_ablation.png",
    )
    figure_records["tail_hue_ablation"] = _save_figure(
        _stack_existing(
            "Tail/hue ablation — bounded r=.10",
            "G2 restores worst-hue coverage but transfers error into broader YC tails; all candidates failed the frozen gate",
            [
                checked_paths["color_risk_basecolor"],
                checked_paths["color_risk_yc_tail"],
                checked_paths["color_risk_hue"],
            ],
        ),
        output_root / "tail_hue_ablation.png",
    )

    manifest = {
        "schema_version": 1,
        "config": config_path.relative_to(ROOT).as_posix(),
        "config_sha256": _sha256_file(config_path),
        "visual_contract": config["visual_contract"],
        "node_artifact_hashes": {
            "standard_p0": standard.safe.artifact_hash,
            "chroma8_parent": parent.safe.artifact_hash,
            **checkpoint_hashes,
        },
        "render_cases": render_cases,
        "figures": figure_records,
        "source_files": {
            key: {
                "path": path.relative_to(ROOT).as_posix(),
                "sha256": _sha256_file(path),
            }
            for key, path in checked_paths.items()
        },
        "formal_holdout_accessed": False,
        "training_started": False,
        "ue_started": False,
        "yellow_selection_metric": False,
    }
    _write_new(output_root / "manifest.json", _json_bytes(manifest))
    _write_new(output_root / "manifest.sha256", (_sha256_file(output_root / "manifest.json") + "\n").encode("ascii"))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/eval/scifihelmet_c4_affine_visual_summary_v1.yaml",
    )
    args = parser.parse_args()
    report = run(args.config.resolve())
    print(
        json.dumps(
            {
                "figures": list(report["figures"]),
                "formal_holdout_accessed": report["formal_holdout_accessed"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
