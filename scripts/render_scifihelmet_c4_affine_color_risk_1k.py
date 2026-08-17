"""Render focus views and source-defined tail/hue atlases for color-risk 1k."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

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
    export_p0_enhanced_bundle,
    rasterize_uv_charts,
)
from cg_frontier.compression.affine_regularizers import hard_quantize_unorm8
from cg_frontier.compression.material import load_core4_targets
from cg_frontier.compression.render_loss import (
    bilinear_sample_top_down_wrap,
    decoded_to_material,
    masked_render_metrics,
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


CANDIDATE_IDS = (
    "G0-mean",
    "G1-yc-cvar25",
    "G2-hue8-macro",
    "G3-cvar25-hue8",
)
DISPLAY_NAMES = {
    "source": "Source",
    "parent": "Parent",
    "G0-mean": "G0 mean",
    "G1-yc-cvar25": "G1 YC-CVaR25",
    "G2-hue8-macro": "G2 hue8-macro",
    "G3-cvar25-hue8": "G3 CVaR+hue8",
}
FOCUS_CAMERA_INDICES = (19, 20, 21, 22)
FOCUS_LIGHT_INDEX = 0


def _contact_sheet(
    columns: list[str],
    rows: list[str],
    images: dict[tuple[str, str], np.ndarray],
    output: Path,
) -> None:
    tile = 256
    left = 176
    top = 32
    gap = 2
    canvas = Image.new(
        "RGB",
        (left + len(columns) * (tile + gap), top + len(rows) * (tile + gap)),
        color=(18, 18, 18),
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for column_index, column in enumerate(columns):
        x = left + column_index * (tile + gap)
        draw.text((x + 4, 10), DISPLAY_NAMES[column], fill=(240, 240, 240), font=font)
    for row_index, row in enumerate(rows):
        y = top + row_index * (tile + gap)
        draw.multiline_text((4, y + 8), row, fill=(240, 240, 240), font=font)
        for column_index, column in enumerate(columns):
            x = left + column_index * (tile + gap)
            canvas.paste(Image.fromarray(images[(row, column)], mode="RGB"), (x, y))
    canvas.save(output, format="PNG")


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_path(output_root: Path, candidate_id: str) -> Path:
    return (
        output_root
        / "runs"
        / candidate_id
        / "checkpoints"
        / candidate_id
        / "endpoints"
        / "step-001000"
        / "checkpoint.pt"
    )


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


def _membership_u8(
    group_ids: torch.Tensor,
    valid_indices: torch.Tensor,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    palette = torch.tensor(
        [
            [0.20, 0.20, 0.20],
            [0.90, 0.20, 0.20],
            [0.95, 0.55, 0.15],
            [0.90, 0.85, 0.20],
            [0.25, 0.80, 0.30],
            [0.15, 0.75, 0.80],
            [0.20, 0.40, 0.95],
            [0.55, 0.25, 0.90],
            [0.90, 0.25, 0.70],
        ],
        device=group_ids.device,
    )
    flat = torch.zeros(height * width, 3, device=group_ids.device)
    flat[valid_indices] = palette[group_ids]
    small = F.interpolate(
        flat.reshape(height, width, 3).permute(2, 0, 1)[None],
        size=(256, 256),
        mode="nearest",
    )[0].permute(1, 2, 0)
    return small.mul(255.0).add(0.5).to(torch.uint8).cpu().numpy()


def _reported_case(
    report: dict[str, object], candidate_id: str, camera_index: int, light_index: int
) -> dict[str, object]:
    for value in report["candidates"][candidate_id]["endpoint"]["render_grid"]:
        if (
            int(value["camera_index"]) == camera_index
            and int(value["light_index"]) == light_index
        ):
            return value
    raise ValueError("reported render case is missing")


def run(config_path: Path) -> dict[str, object]:
    config, _ = _load_mapping(config_path, "color risk render config")
    output_root = _repo_path(config["output_root"], "output_root")
    diagnostic_root = output_root / "diagnostics-r2"
    if diagnostic_root.exists():
        raise FileExistsError(f"refusing to inherit diagnostic output: {diagnostic_root}")
    if not torch.cuda.is_available():
        raise RuntimeError("color risk rendering requires CUDA")
    source = config["source"]
    audit_config, _ = _load_mapping(
        _repo_path(source["audit_config"], "source.audit_config"), "audit config"
    )
    audit_source = audit_config["source"]
    preflight, _ = _load_mapping(
        _repo_path(audit_source["preflight_config"], "preflight"), "preflight"
    )
    render_pool, _ = _load_mapping(
        _repo_path(audit_source["render_pool_config"], "render pool"), "render pool"
    )
    parent_manifest, _ = _load_mapping(
        _repo_path(audit_source["parent_manifest"], "parent"), "parent"
    )
    report_path = output_root / "training_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))

    gltf_path = _repo_path(preflight["inputs"]["gltf"], "inputs.gltf")
    core4_dir = _repo_path(preflight["inputs"]["core4_dir"], "inputs.core4_dir")
    core4_manifest = _repo_path(
        preflight["inputs"]["core4_manifest"], "inputs.core4_manifest"
    )
    mesh = load_gltf_mesh(gltf_path)
    cpu_targets = load_core4_targets(core4_dir, "cpu")
    valid_mask, chart_ids = rasterize_uv_charts(
        mesh.texcoords,
        mesh.triangles,
        height=cpu_targets.height,
        width=cpu_targets.width,
    )
    valid_indices_cpu = torch.nonzero(valid_mask.reshape(-1), as_tuple=False)[:, 0]
    source_rgb_cpu = cpu_targets.select(valid_indices_cpu).base_color_linear
    partition_cpu = build_color_quantile_partition(source_rgb_cpu)
    hue_cpu = build_color_hue_partition(source_rgb_cpu, partition_cpu)
    if hue_cpu.group_hash != report["hue_partition_manifest"]["color_group_hash"]:
        raise RuntimeError("reconstructed hue partition hash mismatch")

    pca = parent_manifest["pca"]
    parent_bundle = export_p0_enhanced_bundle(
        _targets_to_seven(cpu_targets),
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
        raise RuntimeError("reconstructed parent manifest mismatch")

    device = torch.device("cuda")
    parent = _move_p0(parent_bundle.calibration, device)
    source_rgb = source_rgb_cpu.to(device)
    valid_indices = valid_indices_cpu.to(device)
    partition = partition_cpu.to(device)
    hue = hue_cpu.to(device)
    models: dict[str, tuple[torch.Tensor, torch.nn.Module]] = {
        "parent": (
            parent.safe.latent_unorm8.to(torch.float32) / 255.0,
            parent.safe_decoder,
        )
    }
    checkpoint_hashes: dict[str, str] = {}
    for candidate_id in CANDIDATE_IDS:
        checkpoint_path = _checkpoint_path(output_root, candidate_id)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        decoder = copy.deepcopy(parent.safe_decoder)
        decoder.load_state_dict(checkpoint["safe_affine_raw_parameters"])
        decoder.to(device).eval()
        models[candidate_id] = (checkpoint["latent"].to(device), decoder)
        checkpoint_hashes[candidate_id] = str(checkpoint["checkpoint_hash"])

    predictions: dict[str, torch.Tensor] = {"source": source_rgb}
    for key, (latent, decoder) in models.items():
        deployed = hard_quantize_unorm8(latent)
        predictions[key] = decoder(
            deployed.reshape(-1, 4)[valid_indices]
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
    for key, values in errors.items():
        tail = torch.zeros_like(values)
        for offset, size in zip(partition.bin_offsets, partition.bin_sizes):
            start = int(offset)
            positions = partition.concatenated_valid_positions[start : start + int(size)]
            count = int(np.ceil(int(size) * 0.25))
            order = torch.sort(values[positions], descending=True, stable=True).indices
            selected = positions[order[:count]]
            tail[selected] = values[selected]
        tail_values[key] = tail
        group_value = torch.zeros_like(values)
        for group_id in range(9):
            selected = hue.valid_group_ids == group_id
            group_value[selected] = values[selected].mean()
        hue_values[key] = group_value
    tail_scale = float(
        torch.quantile(
            torch.cat([value[value > 0] for value in tail_values.values()]), 0.995
        ).detach()
    )
    hue_scale = float(
        torch.stack([value.max() for value in hue_values.values()]).max().detach()
    )

    diagnostic_root.mkdir(parents=True)
    columns = list(predictions)
    base_images = {
        ("BaseColor atlas", key): _atlas_u8(
            prediction,
            valid_indices,
            height=cpu_targets.height,
            width=cpu_targets.width,
        )
        for key, prediction in predictions.items()
    }
    tail_images = {
        ("worst 25% in each YC bin", key): _error_atlas_u8(
            value,
            valid_indices,
            height=cpu_targets.height,
            width=cpu_targets.width,
            scale=tail_scale,
            palette="tail",
        )
        for key, value in tail_values.items()
    }
    hue_images = {
        ("hue-group mean opponent error", key): _error_atlas_u8(
            value,
            valid_indices,
            height=cpu_targets.height,
            width=cpu_targets.width,
            scale=hue_scale,
            palette="hue",
        )
        for key, value in hue_values.items()
    }
    _contact_sheet(
        columns, ["BaseColor atlas"], base_images, diagnostic_root / "basecolor_atlas_comparison.png"
    )
    _contact_sheet(
        columns,
        ["worst 25% in each YC bin"],
        tail_images,
        diagnostic_root / "yc_tail_error_atlas_comparison.png",
    )
    _contact_sheet(
        columns,
        ["hue-group mean opponent error"],
        hue_images,
        diagnostic_root / "hue_group_error_atlas_comparison.png",
    )
    Image.fromarray(
        _membership_u8(
            hue.valid_group_ids,
            valid_indices,
            height=cpu_targets.height,
            width=cpu_targets.width,
        ),
        mode="RGB",
    ).save(diagnostic_root / "hue_group_membership.png")

    render = render_pool["render"]
    exposure = float(render["display_exposure"])
    cameras = [
        _orbit_camera_from_spec(value, render) for value in render_pool["train_cameras"]
    ]
    lights = [_light(value) for value in render_pool["train_lights"]]
    textures = load_core4_textures(core4_manifest, device)
    render_images: dict[tuple[str, str], np.ndarray] = {}
    render_rows: list[str] = []
    cases: list[dict[str, object]] = []
    individual_root = diagnostic_root / "renders"
    individual_root.mkdir()
    for camera_index in FOCUS_CAMERA_INDICES:
        geometry = render_geometry_gbuffer(
            mesh, cameras[camera_index], tuple(render["resolution"]), device=device
        )
        reference_material = sample_core4_material(geometry, textures)
        reference = shade_ggx(
            geometry,
            cameras[camera_index],
            lights[FOCUS_LIGHT_INDEX],
            material_override=reference_material,
            minimum_roughness=float(render["minimum_roughness"]),
        )
        row = f"camera {camera_index} / light {FOCUS_LIGHT_INDEX}"
        render_rows.append(row)
        case_root = individual_root / f"camera-{camera_index:02d}_light-{FOCUS_LIGHT_INDEX:02d}"
        case_root.mkdir()
        reference_u8 = _display_u8(reference, exposure)
        _save_png(case_root / "source.png", reference_u8)
        render_images[(row, "source")] = reference_u8
        case = {"camera_index": camera_index, "light_index": FOCUS_LIGHT_INDEX, "variants": {}}
        for key, (latent, decoder) in models.items():
            sampled = bilinear_sample_top_down_wrap(
                hard_quantize_unorm8(latent), geometry.torch_buffers["uv"]
            )
            material = decoded_to_material(geometry, decoder(sampled))
            candidate = shade_ggx(
                geometry,
                cameras[camera_index],
                lights[FOCUS_LIGHT_INDEX],
                material_override=material,
                minimum_roughness=float(render["minimum_roughness"]),
            )
            candidate_u8 = _display_u8(candidate, exposure)
            _save_png(case_root / f"{key}.png", candidate_u8)
            render_images[(row, key)] = candidate_u8
            metrics = masked_render_metrics(
                reference,
                candidate,
                geometry.torch_buffers["mask"],
                linear_psnr_data_range=float(render["linear_psnr_data_range"]),
                display_exposure=exposure,
            )
            if key in CANDIDATE_IDS:
                expected = _reported_case(
                    report, key, camera_index, FOCUS_LIGHT_INDEX
                )
                for name in (
                    "masked_linear_hdr_mae",
                    "display_ssim",
                    "foreground_pixel_count",
                ):
                    if metrics[name] != expected[name]:
                        raise RuntimeError(f"render replay mismatch: {key}/{name}")
            case["variants"][key] = metrics
        cases.append(case)
    _contact_sheet(
        list(predictions), render_rows, render_images, diagnostic_root / "focus_render_comparison.png"
    )
    manifest = {
        "schema_version": 1,
        "training_report_sha256": _sha256_file(report_path),
        "checkpoint_hashes": checkpoint_hashes,
        "focus_camera_indices": list(FOCUS_CAMERA_INDICES),
        "focus_light_index": FOCUS_LIGHT_INDEX,
        "tail_mass": 0.25,
        "tail_scale": tail_scale,
        "hue_error_scale": hue_scale,
        "render_cases": cases,
        "render_report_exact_match": True,
        "formal_holdout_accessed": False,
        "yellow_selection_metric": False,
    }
    _write_new(diagnostic_root / "render_manifest.json", _json_bytes(manifest))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/train/scifihelmet_c4_affine_color_risk_1k_v1.yaml",
    )
    args = parser.parse_args()
    report = run(args.config.resolve())
    print(
        json.dumps(
            {
                "render_report_exact_match": report["render_report_exact_match"],
                "focus_camera_indices": report["focus_camera_indices"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
