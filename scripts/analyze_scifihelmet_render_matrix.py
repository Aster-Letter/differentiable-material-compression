"""Render train-derived oracle, fixed-LOD, distance, and oblique diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cg_frontier.assets.gltf_mesh import load_gltf_mesh  # noqa: E402
from cg_frontier.compression.artifact_analysis import deterministic_json, sha256_file, tail_statistics  # noqa: E402
from cg_frontier.compression.render_loss import (  # noqa: E402
    display_transform,
    hard_quantize_unorm8,
    load_latent_unorm8_png,
    masked_render_metrics,
    orbit_camera,
    sample_and_decode_latent,
)
from cg_frontier.render.gbuffer import (  # noqa: E402
    MaterialBuffers,
    load_core4_textures,
    render_geometry_gbuffer,
    sample_core4_material,
)
from cg_frontier.render.pbr import PointLight, shade_ggx  # noqa: E402
from scripts.train_scifihelmet_repair import (  # noqa: E402
    _case_specs,
    _decoder_from_npz,
    _light,
    _prepare_cases,
    _repo_path,
)


def _hybrid(reference: MaterialBuffers, compressed: MaterialBuffers, oracle: str) -> MaterialBuffers:
    """Replace exactly one compressed semantic with reference truth."""

    if oracle not in ("base_color", "normal", "roughness", "metallic"):
        raise ValueError(f"unknown oracle semantic: {oracle}")
    return MaterialBuffers(
        base_color_linear=reference.base_color_linear if oracle == "base_color" else compressed.base_color_linear,
        normal_world=reference.normal_world if oracle == "normal" else compressed.normal_world,
        roughness=reference.roughness if oracle == "roughness" else compressed.roughness,
        metallic=reference.metallic if oracle == "metallic" else compressed.metallic,
    )


def _save_display(path: Path, linear: torch.Tensor, exposure: float) -> None:
    encoded = torch.floor(display_transform(linear, exposure) * 255.0 + 0.5).to(torch.uint8)
    Image.fromarray(encoded.detach().cpu().numpy(), mode="RGB").save(path, format="PNG")


def _error_tails(reference: torch.Tensor, candidate: torch.Tensor, mask: torch.Tensor) -> dict[str, Any]:
    luminance = torch.sum(reference * reference.new_tensor([0.2126, 0.7152, 0.0722]), dim=-1)
    valid_luma = luminance[mask].detach().cpu().numpy()
    low, high = np.percentile(valid_luma, (10.0, 90.0))
    error = torch.max(torch.abs(candidate - reference), dim=-1).values.detach().cpu().numpy()
    mask_np = mask.detach().cpu().numpy()
    luma_np = luminance.detach().cpu().numpy()
    return {
        "all": tail_statistics(error, mask_np),
        "shadow_bottom10": tail_statistics(error, mask_np & (luma_np <= low)),
        "highlight_top10": tail_statistics(error, mask_np & (luma_np >= high)),
        "reference_luminance_p10": float(low),
        "reference_luminance_p90": float(high),
    }


def _mip_proxy(latent: torch.Tensor, levels: int) -> torch.Tensor:
    """Average linear latent texels and requantize each fixed LOD proxy."""

    value = latent.permute(2, 0, 1)[None]
    for _ in range(levels):
        value = F.avg_pool2d(value, kernel_size=2, stride=2)
        value = hard_quantize_unorm8(value)
    return value[0].permute(1, 2, 0).contiguous()


def run(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping) or config.get("schema_version") != 1:
        raise ValueError("unsupported repair config")
    output_dir = _repo_path(config["output_dir"], "output_dir") / "analysis" / "render_matrix"
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    inputs = config["inputs"]
    _, latent = load_latent_unorm8_png(_repo_path(inputs["latent_hard_png"], "inputs.latent_hard_png"), device=device)
    decoder = _decoder_from_npz(_repo_path(inputs["decoder_npz"], "inputs.decoder_npz"), device)
    mesh = load_gltf_mesh(_repo_path(inputs["gltf"], "inputs.gltf"))
    textures = load_core4_textures(_repo_path(inputs["core4_manifest"], "inputs.core4_manifest"), device)
    specs, partitions = _case_specs(config)
    selection_names = partitions["selection"][:3]
    prepared = _prepare_cases(config, selection_names, specs, mesh, textures, device)

    render = config["render"]
    base_camera_spec = next(item for item in config["train_cameras"] if item["name"] == "train_e0_y045")
    base_light = _light(config["train_lights"][0])
    diagnostic_cases: list[tuple[str, Any, Any, Any, torch.Tensor]] = []
    for name, radius, elevation, yaw in (
        ("near", 3.0, 0.0, 45.0),
        ("mid", 4.5, 0.0, 45.0),
        ("far", 7.0, 0.0, 45.0),
        ("oblique", 4.5, 35.0, 225.0),
    ):
        camera = orbit_camera(
            yaw_degrees=yaw,
            elevation_degrees=elevation,
            radius=radius,
            target=tuple(float(value) for value in render["target"]),
            up=tuple(float(value) for value in render["up"]),
            vertical_fov_degrees=float(render["vertical_fov_degrees"]),
            near=float(render["near"]),
            far=float(render["far"]),
        )
        geometry = render_geometry_gbuffer(
            mesh,
            camera,
            tuple(int(value) for value in render["resolution"]),
            device=device,
            cull_backfaces=True,
        )
        reference_material = sample_core4_material(geometry, textures)
        with torch.no_grad():
            reference = shade_ggx(
                geometry,
                camera,
                base_light,
                material_override=reference_material,
                minimum_roughness=float(render["minimum_roughness"]),
            )
        diagnostic_cases.append((f"diagnostic_{name}", geometry, camera, base_light, reference))
    all_cases = prepared + diagnostic_cases
    mip_latents = {"lod0": latent, "lod1_proxy": _mip_proxy(latent, 1), "lod2_proxy": _mip_proxy(latent, 2)}

    case_reports: dict[str, Any] = {}
    generated: dict[str, str] = {}
    aggregate: dict[str, list[float]] = {}
    for case_name, geometry, camera, light, reference_image in all_cases:
        case_dir = output_dir / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        reference_material = sample_core4_material(geometry, textures)
        compressed_material = sample_and_decode_latent(geometry, latent, decoder, quantization="hard")
        materials = {
            "reference": reference_material,
            "baseline": compressed_material,
            "base_color_oracle": _hybrid(reference_material, compressed_material, "base_color"),
            "normal_oracle": _hybrid(reference_material, compressed_material, "normal"),
            "roughness_oracle": _hybrid(reference_material, compressed_material, "roughness"),
            "metallic_oracle": _hybrid(reference_material, compressed_material, "metallic"),
        }
        images: dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, material in materials.items():
                images[name] = shade_ggx(
                    geometry,
                    camera,
                    light,
                    material_override=material,
                    minimum_roughness=float(render["minimum_roughness"]),
                )
            for name, mip_latent in mip_latents.items():
                mip_material = sample_and_decode_latent(geometry, mip_latent, decoder, quantization="hard")
                images[name] = shade_ggx(
                    geometry,
                    camera,
                    light,
                    material_override=mip_material,
                    minimum_roughness=float(render["minimum_roughness"]),
                )
        variants: dict[str, Any] = {}
        for name, image in images.items():
            metrics = masked_render_metrics(
                reference_image,
                image,
                geometry.torch_buffers["mask"],
                linear_psnr_data_range=float(render["linear_psnr_data_range"]),
                display_exposure=float(render["display_exposure"]),
            )
            variants[name] = {"metrics": metrics, "error_tails": _error_tails(reference_image, image, geometry.torch_buffers["mask"])}
            aggregate.setdefault(name, []).append(float(metrics["masked_linear_hdr_mae"]))
            path = case_dir / f"{name}_display.png"
            _save_display(path, image, float(render["display_exposure"]))
            generated[path.relative_to(output_dir).as_posix()] = sha256_file(path)
        case_reports[case_name] = {
            "source": "repair_selection_train_case" if case_name in selection_names else "train_derived_sampling_diagnostic",
            "variants": variants,
        }
    aggregate_mae = {name: float(np.mean(values)) for name, values in aggregate.items()}
    baseline_mae = aggregate_mae["baseline"]
    report = {
        "schema_version": 1,
        "formal_holdout_accessed": False,
        "selection_cases": selection_names,
        "diagnostic_views": ["near", "mid", "far", "oblique"],
        "fixed_lod_note": "lod1/lod2 are deterministic average+UNORM8 proxies, not UE derivative-selected mips",
        "aggregate_hdr_mae": aggregate_mae,
        "oracle_hdr_mae_reduction": {
            name: (baseline_mae - aggregate_mae[name]) / baseline_mae
            for name in ("base_color_oracle", "normal_oracle", "roughness_oracle", "metallic_oracle")
        },
        "cases": case_reports,
        "generated_files": generated,
    }
    (output_dir / "render_matrix.json").write_text(
        deterministic_json(report), encoding="utf-8", newline="\n"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train/scifihelmet_repair.yaml")
    args = parser.parse_args()
    report = run(args.config.resolve())
    print(json.dumps({"status": "complete", "formal_holdout_accessed": False, "oracle_reduction": report["oracle_hdr_mae_reduction"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
