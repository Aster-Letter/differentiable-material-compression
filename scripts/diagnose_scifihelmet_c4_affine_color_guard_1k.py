"""Diagnose yellow-chroma drift and render the frozen color-guard 1k matrix."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for entry in (ROOT, SRC):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from cg_frontier.assets.gltf_mesh import load_gltf_mesh  # noqa: E402
from cg_frontier.compression.affine_color import (  # noqa: E402
    build_color_quantile_partition,
    opponent_vector_charbonnier,
    orthogonal_color_coordinates,
)
from cg_frontier.compression.affine_pca import (  # noqa: E402
    EnhancedPCASpec,
    export_p0_enhanced_bundle,
    rasterize_uv_charts,
)
from cg_frontier.compression.affine_regularizers import (  # noqa: E402
    hard_quantize_unorm8,
)
from cg_frontier.compression.material import Core4Targets, load_core4_targets  # noqa: E402
from cg_frontier.compression.render_loss import (  # noqa: E402
    bilinear_sample_top_down_wrap,
    decoded_to_material,
    display_transform,
    masked_render_metrics,
)
from cg_frontier.render.gbuffer import (  # noqa: E402
    load_core4_textures,
    render_geometry_gbuffer,
    sample_core4_material,
)
from cg_frontier.render.pbr import linear_to_srgb_torch, shade_ggx  # noqa: E402
from scripts.run_scifihelmet_c4_affine_40k import _load_mapping  # noqa: E402
from scripts.run_scifihelmet_c4_affine_chroma8_l0_40k import (  # noqa: E402
    _orbit_camera_from_spec,
)
from scripts.run_scifihelmet_c4_affine_preflight import (  # noqa: E402
    _light,
    _move_p0,
    _repo_path,
    _targets_to_seven,
)


CANDIDATE_KEYS = ("C0", "C1-r010", "C2-r010", "C1-r025", "C2-r025")
DISPLAY_NAMES = {
    "source": "Source",
    "parent": "Parent",
    "C0": "C0",
    "C1-r010": "C1 r=.10",
    "C2-r010": "C2 r=.10",
    "C1-r025": "C1 r=.25",
    "C2-r025": "C2 r=.25",
}
FOCUS_CAMERA_INDICES = (19, 20, 21, 22)


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


def _save_png(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array, mode="RGB").save(path, format="PNG")


def _display_u8(linear: torch.Tensor, exposure: float) -> np.ndarray:
    encoded = torch.floor(display_transform(linear, exposure) * 255.0 + 0.5)
    return encoded.to(torch.uint8).detach().cpu().numpy()


def _atlas_u8(
    valid_rgb: torch.Tensor,
    valid_indices: torch.Tensor,
    *,
    height: int,
    width: int,
    yellow: torch.Tensor | None = None,
) -> np.ndarray:
    flat = torch.zeros(height * width, 3, device=valid_rgb.device, dtype=valid_rgb.dtype)
    flat[valid_indices] = valid_rgb
    atlas = flat.reshape(height, width, 3)
    if yellow is not None:
        full_mask = torch.zeros(height * width, device=valid_rgb.device, dtype=torch.bool)
        full_mask[valid_indices] = yellow
        mask = full_mask.reshape(height, width, 1)
        atlas = torch.where(mask, atlas, atlas * 0.06)
    small = F.interpolate(
        atlas.permute(2, 0, 1)[None],
        size=(256, 256),
        mode="area",
    )[0].permute(1, 2, 0)
    encoded = torch.floor(linear_to_srgb_torch(small).clamp(0.0, 1.0) * 255.0 + 0.5)
    return encoded.to(torch.uint8).cpu().numpy()


def _rb_map_u8(
    valid_rgb: torch.Tensor,
    valid_indices: torch.Tensor,
    yellow: torch.Tensor,
    *,
    height: int,
    width: int,
    scale: float,
) -> np.ndarray:
    rb = (valid_rgb[:, 0] - valid_rgb[:, 2]).clamp(min=0.0, max=scale) / scale
    flat = torch.zeros(height * width, device=valid_rgb.device, dtype=valid_rgb.dtype)
    flat[valid_indices[yellow]] = rb[yellow]
    small = F.interpolate(
        flat.reshape(1, 1, height, width), size=(256, 256), mode="area"
    )[0, 0]
    # A common fixed scale: black means no source-defined yellow support;
    # red-to-yellow brightness directly represents positive linear R-B.
    return torch.stack((small, small * 0.82, small * 0.08), dim=-1).mul(255.0).add(
        0.5
    ).to(torch.uint8).cpu().numpy()


def _contact_sheet(
    columns: list[str],
    rows: list[str],
    images: Mapping[tuple[str, str], np.ndarray],
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
        draw.multiline_text((6, y + 8), row, fill=(230, 230, 230), font=font, spacing=3)
        for column_index, column in enumerate(columns):
            x = left + column_index * (tile + gap)
            canvas.paste(Image.fromarray(images[(row, column)], mode="RGB"), (x, y))
    canvas.save(output, format="PNG")


def _yellow_mask(rgb: torch.Tensor) -> torch.Tensor:
    return (
        (rgb[:, 0] > rgb[:, 1])
        & (rgb[:, 1] > rgb[:, 2])
        & (rgb[:, 0] - rgb[:, 2] > 0.05)
        & (rgb[:, 1] - rgb[:, 2] > 0.02)
    )


def _candidate_checkpoint(output_root: Path, key: str) -> Path:
    candidate_id = key.split("-", maxsplit=1)[0]
    return (
        output_root
        / "runs"
        / key
        / "checkpoints"
        / candidate_id
        / "endpoints"
        / "step-001000"
        / "checkpoint.pt"
    )


def _reported_case(
    report: Mapping[str, object], key: str, camera_index: int, light_index: int
) -> Mapping[str, object]:
    candidate = report["candidates"][key]  # type: ignore[index]
    for value in candidate["endpoint"]["render_grid"]:  # type: ignore[index]
        if value["camera_index"] == camera_index and value["light_index"] == light_index:
            return value
    raise RuntimeError(f"missing reported render case for {key}/{camera_index}/{light_index}")


@torch.no_grad()
def run(config_path: Path) -> dict[str, object]:
    config, _ = _load_mapping(config_path, "color guard diagnostic config")
    output_root = _repo_path(config["output_root"], "output_root")
    diagnostic_root = output_root / "diagnostics-r1"
    if diagnostic_root.exists():
        raise FileExistsError(f"refusing to inherit diagnostic output: {diagnostic_root}")
    if not torch.cuda.is_available():
        raise RuntimeError("color guard rendering diagnostic requires CUDA")

    source = config["source"]
    audit_config, _ = _load_mapping(
        _repo_path(source["audit_config"], "source.audit_config"), "audit config"
    )
    audit_source = audit_config["source"]
    preflight, _ = _load_mapping(
        _repo_path(audit_source["preflight_config"], "preflight config"), "preflight"
    )
    render_pool, _ = _load_mapping(
        _repo_path(audit_source["render_pool_config"], "render pool"), "render pool"
    )
    parent_manifest, _ = _load_mapping(
        _repo_path(audit_source["parent_manifest"], "parent manifest"), "parent"
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
    yellow = _yellow_mask(source_rgb)
    partition = partition_cpu.to(device)

    models: dict[str, tuple[torch.Tensor, torch.nn.Module]] = {
        "parent": (
            parent.safe.latent_unorm8.to(torch.float32) / 255.0,
            parent.safe_decoder,
        )
    }
    checkpoint_hashes: dict[str, str] = {}
    for key in CANDIDATE_KEYS:
        checkpoint_path = _candidate_checkpoint(output_root, key)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        decoder = copy.deepcopy(parent.safe_decoder)
        decoder.load_state_dict(checkpoint["safe_affine_raw_parameters"])
        decoder.to(device).eval()
        models[key] = (checkpoint["latent"].to(device), decoder)
        checkpoint_hashes[key] = str(checkpoint["checkpoint_hash"])
        del checkpoint

    predictions: dict[str, torch.Tensor] = {"source": source_rgb}
    for key, (latent, decoder) in models.items():
        deployed = hard_quantize_unorm8(latent)
        predictions[key] = decoder(deployed.reshape(-1, 4)[valid_indices]).base_color_linear

    logical_by_position = torch.empty(
        source_rgb.shape[0], device=device, dtype=torch.int64
    )
    yellow_bin_rows: list[dict[str, object]] = []
    yellow_count = int(yellow.sum().cpu())
    equal_bin_yellow_probability = 0.0
    for slot in range(partition.active_bin_count):
        start = int(partition.bin_offsets[slot])
        size = int(partition.bin_sizes[slot])
        positions = partition.concatenated_valid_positions[start : start + size]
        logical_id = int(partition.logical_bin_ids[slot])
        logical_by_position[positions] = logical_id
        count = int(yellow[positions].sum().cpu())
        equal_bin_yellow_probability += count / size / partition.active_bin_count
        yellow_bin_rows.append(
            {
                "logical_bin_id": logical_id,
                "bin_size": size,
                "yellow_count": count,
                "yellow_fraction_within_bin": count / size,
                "fraction_of_all_yellow": count / yellow_count,
            }
        )

    epsilon = float(config["training"]["charbonnier_epsilon"])
    atlas_diagnostics: dict[str, object] = {}
    for key, prediction in predictions.items():
        source_coordinates = orthogonal_color_coordinates(source_rgb)[:, 1:]
        prediction_coordinates = orthogonal_color_coordinates(prediction)[:, 1:]
        vector_error = torch.sqrt(
            torch.sum((prediction_coordinates - source_coordinates).square(), dim=-1)
            + epsilon * epsilon
        ) - epsilon
        per_bins: list[dict[str, object]] = []
        for row in yellow_bin_rows:
            logical_id = int(row["logical_bin_id"])
            selected = logical_by_position == logical_id
            per_bins.append(
                {
                    "logical_bin_id": logical_id,
                    "opponent_error": float(vector_error[selected].mean().cpu()),
                    "yellow_count": int(row["yellow_count"]),
                    "yellow_fraction_within_bin": float(row["yellow_fraction_within_bin"]),
                }
            )
        worst = max(per_bins, key=lambda value: float(value["opponent_error"]))
        yellow_source_opponent = source_coordinates[yellow]
        yellow_prediction_opponent = prediction_coordinates[yellow]
        atlas_diagnostics[key] = {
            "yellow_mean_r_minus_b": float(
                (prediction[yellow, 0] - prediction[yellow, 2]).mean().cpu()
            ),
            "yellow_mean_absolute_r_minus_b_error": float(
                torch.abs(
                    (prediction[yellow, 0] - prediction[yellow, 2])
                    - (source_rgb[yellow, 0] - source_rgb[yellow, 2])
                ).mean().cpu()
            ),
            "yellow_opponent_error": float(
                opponent_vector_charbonnier(
                    yellow_prediction_opponent,
                    yellow_source_opponent,
                    epsilon=epsilon,
                ).cpu()
            ),
            "non_yellow_opponent_error": float(vector_error[~yellow].mean().cpu()),
            "uniform_opponent_error": float(vector_error.mean().cpu()),
            "yellow_source_opponent_mean": [
                float(value) for value in yellow_source_opponent.mean(dim=0).cpu()
            ],
            "yellow_prediction_opponent_mean": [
                float(value) for value in yellow_prediction_opponent.mean(dim=0).cpu()
            ],
            "worst_bin": worst,
            "per_bin": per_bins,
        }

    diagnostic_root.mkdir(parents=True)
    atlas_columns = list(predictions)
    base_images: dict[tuple[str, str], np.ndarray] = {}
    masked_images: dict[tuple[str, str], np.ndarray] = {}
    rb_images: dict[tuple[str, str], np.ndarray] = {}
    atlas_row = "full valid atlas"
    yellow_row = "source-defined yellow\n(non-yellow dimmed)"
    rb_row = "yellow linear R-B\nfixed range 0..0.25"
    for key, prediction in predictions.items():
        base_images[(atlas_row, key)] = _atlas_u8(
            prediction,
            valid_indices,
            height=cpu_targets.height,
            width=cpu_targets.width,
        )
        masked_images[(yellow_row, key)] = _atlas_u8(
            prediction,
            valid_indices,
            height=cpu_targets.height,
            width=cpu_targets.width,
            yellow=yellow,
        )
        rb_images[(rb_row, key)] = _rb_map_u8(
            prediction,
            valid_indices,
            yellow,
            height=cpu_targets.height,
            width=cpu_targets.width,
            scale=0.25,
        )
    _contact_sheet(atlas_columns, [atlas_row], base_images, diagnostic_root / "basecolor_atlas_comparison.png")
    _contact_sheet(
        atlas_columns,
        [yellow_row],
        masked_images,
        diagnostic_root / "yellow_basecolor_comparison.png",
    )
    _contact_sheet(
        atlas_columns,
        [rb_row],
        rb_images,
        diagnostic_root / "yellow_r_minus_b_comparison.png",
    )

    render = render_pool["render"]
    exposure = float(render["display_exposure"])
    cameras = [
        _orbit_camera_from_spec(value, render) for value in render_pool["train_cameras"]
    ]
    lights = [_light(value) for value in render_pool["train_lights"]]
    textures = load_core4_textures(core4_manifest, device)
    render_images: dict[tuple[str, str], np.ndarray] = {}
    render_rows: list[str] = []
    render_cases: list[dict[str, object]] = []
    selected_metrics: dict[str, list[dict[str, object]]] = {
        key: [] for key in CANDIDATE_KEYS
    }
    individual_root = diagnostic_root / "renders"
    individual_root.mkdir()
    for camera_index in FOCUS_CAMERA_INDICES:
        geometry = render_geometry_gbuffer(
            mesh, cameras[camera_index], tuple(render["resolution"]), device=device
        )
        reference_material = sample_core4_material(geometry, textures)
        screen_yellow = (
            geometry.torch_buffers["mask"]
            & (reference_material.base_color_linear[..., 0] > reference_material.base_color_linear[..., 1])
            & (reference_material.base_color_linear[..., 1] > reference_material.base_color_linear[..., 2])
            & (
                reference_material.base_color_linear[..., 0]
                - reference_material.base_color_linear[..., 2]
                > 0.05
            )
            & (
                reference_material.base_color_linear[..., 1]
                - reference_material.base_color_linear[..., 2]
                > 0.02
            )
        )
        if not bool(screen_yellow.any()):
            raise RuntimeError(f"focus camera {camera_index} contains no yellow footprint")
        light_candidates: list[tuple[float, int, torch.Tensor]] = []
        for light_index, light in enumerate(lights):
            reference = shade_ggx(
                geometry,
                cameras[camera_index],
                light,
                material_override=reference_material,
                minimum_roughness=float(render["minimum_roughness"]),
            )
            display = display_transform(reference, exposure)
            luminance = torch.sum(
                display * display.new_tensor((0.2126, 0.7152, 0.0722)), dim=-1
            )
            light_candidates.append(
                (float(luminance[screen_yellow].mean().cpu()), light_index, reference)
            )
        _, light_index, reference = max(light_candidates, key=lambda value: value[0])
        camera_name = str(render_pool["train_cameras"][camera_index]["name"])
        light_name = str(render_pool["train_lights"][light_index]["name"])
        row = f"{camera_name}\n{light_name}"
        render_rows.append(row)
        case_root = individual_root / f"camera-{camera_index:02d}_light-{light_index:02d}"
        case_root.mkdir()
        reference_u8 = _display_u8(reference, exposure)
        _save_png(case_root / "source.png", reference_u8)
        render_images[(row, "source")] = reference_u8
        case_report: dict[str, object] = {
            "camera_index": camera_index,
            "camera_name": camera_name,
            "light_index": light_index,
            "light_name": light_name,
            "yellow_screen_pixels": int(screen_yellow.sum().cpu()),
            "variants": {},
        }
        for key, (latent, decoder) in models.items():
            deployed = hard_quantize_unorm8(latent)
            sampled = bilinear_sample_top_down_wrap(
                deployed, geometry.torch_buffers["uv"]
            )
            material = decoded_to_material(geometry, decoder(sampled))
            candidate = shade_ggx(
                geometry,
                cameras[camera_index],
                lights[light_index],
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
            case_report["variants"][key] = metrics  # type: ignore[index]
            if key in CANDIDATE_KEYS:
                expected = _reported_case(report, key, camera_index, light_index)
                for metric_name in (
                    "masked_linear_hdr_mae",
                    "display_ssim",
                    "foreground_pixel_count",
                ):
                    if metrics[metric_name] != expected[metric_name]:
                        raise RuntimeError(
                            f"render reproduction mismatch for {key}/{camera_index}/{light_index}/{metric_name}: "
                            f"{metrics[metric_name]} != {expected[metric_name]}"
                        )
                selected_metrics[key].append(
                    {
                        "camera_index": camera_index,
                        "light_index": light_index,
                        "masked_linear_hdr_mae": metrics["masked_linear_hdr_mae"],
                        "display_ssim": metrics["display_ssim"],
                        "training_report_exact_match": True,
                    }
                )
        render_cases.append(case_report)
        del geometry, reference_material, reference, screen_yellow
        torch.cuda.empty_cache()

    render_columns = ["source", "parent", *CANDIDATE_KEYS]
    _contact_sheet(
        render_columns,
        render_rows,
        render_images,
        diagnostic_root / "render_focus_comparison.png",
    )

    report_value: dict[str, object] = {
        "schema_version": 1,
        "status": "complete",
        "scope": "read-only diagnosis of frozen 1k endpoints",
        "training_report": str(report_path.relative_to(ROOT)).replace("\\", "/"),
        "training_report_sha256": _sha256_file(report_path),
        "checkpoint_hashes": checkpoint_hashes,
        "partition_hash": partition_cpu.partition_hash,
        "active_bin_count": partition_cpu.active_bin_count,
        "yellow_sampling": {
            "valid_texel_count": int(source_rgb.shape[0]),
            "yellow_texel_count": yellow_count,
            "uniform_texel_probability": yellow_count / source_rgb.shape[0],
            "equal_active_bin_probability": equal_bin_yellow_probability,
            "equal_bin_vs_uniform_ratio": equal_bin_yellow_probability
            / (yellow_count / source_rgb.shape[0]),
            "bins": yellow_bin_rows,
        },
        "atlas": atlas_diagnostics,
        "render_cases": render_cases,
        "selected_case_reproduction": selected_metrics,
        "images": {},
    }
    image_names = (
        "render_focus_comparison.png",
        "basecolor_atlas_comparison.png",
        "yellow_basecolor_comparison.png",
        "yellow_r_minus_b_comparison.png",
    )
    for name in image_names:
        report_value["images"][name] = {  # type: ignore[index]
            "sha256": _sha256_file(diagnostic_root / name)
        }
    report_path_out = diagnostic_root / "diagnostic_report.json"
    report_path_out.write_bytes(_json_bytes(report_value))
    return {
        "status": "complete",
        "output_root": str(diagnostic_root.relative_to(ROOT)).replace("\\", "/"),
        "report_sha256": _sha256_file(report_path_out),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs"
        / "train"
        / "scifihelmet_c4_affine_color_guard_dual_ratio_1k_v1.yaml",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
