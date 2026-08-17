"""Audit complex CC0 single-material assets before BaseColor-priority training."""

from __future__ import annotations

import argparse
import hashlib
import json
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

from cg_frontier.assets.complex_c4_validation import (  # noqa: E402
    audit_complex_gltf_document,
    select_complex_validators,
)
from cg_frontier.assets.gltf_core4 import load_gltf_core4_asset  # noqa: E402
from cg_frontier.assets.gltf_mesh import GltfMeshError  # noqa: E402
from cg_frontier.assets.preprocess import sha256_file  # noqa: E402
from cg_frontier.compression.affine_pca import (  # noqa: E402
    export_p0_bundle,
    fit_global_valid_pca_oracle,
    rasterize_uv_charts,
)
from cg_frontier.compression.affine_color import linear_srgb_to_oklab  # noqa: E402
from cg_frontier.compression.basecolor_priority import postprocess_affine_output  # noqa: E402
from cg_frontier.compression.render_loss import masked_render_metrics  # noqa: E402
from cg_frontier.render.gbuffer import (  # noqa: E402
    Core4Textures,
    render_geometry_gbuffer,
    sample_core4_material,
)
from cg_frontier.render.generic_c4_rig import (  # noqa: E402
    build_generic_c4_rig,
    instantiate_camera,
    instantiate_lights,
)
from cg_frontier.render.pbr import linear_to_srgb_torch, shade_ggx  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/eval/complex_c4_asset_screen_v1.yaml"
COLUMNS = ("source", "raw_q4", "direct_basecolor", "oracle_q6")


def _repo_path(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty repository-relative path")
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"{label} escapes the repository")
    return path


def _verify(path: Path, expected: object, label: str) -> None:
    if not path.is_file() or sha256_file(path) != str(expected):
        raise ValueError(f"{label} SHA-256 mismatch: {path}")


def _write_json(path: Path, value: object) -> str:
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n", encoding="ascii")
    return digest


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


def _material_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, Any]:
    pred = postprocess_affine_output(
        prediction, compander_parameters=None, straight_through=False
    )
    target_post = postprocess_affine_output(
        target, compander_parameters=None, straight_through=False
    )
    angles = torch.rad2deg(
        torch.acos(
            (pred.normal_xyz * target_post.normal_xyz).sum(dim=-1).clamp(-1.0, 1.0)
        )
    )
    delta_e = torch.linalg.vector_norm(
        linear_srgb_to_oklab(pred.seven[:, :3])
        - linear_srgb_to_oklab(target_post.seven[:, :3]),
        dim=-1,
    )
    base_difference = pred.seven[:, :3] - target_post.seven[:, :3]
    return {
        "seven_channel_mae": float(F.l1_loss(pred.seven, target_post.seven)),
        "base_color_linear_mae": float(base_difference.abs().mean()),
        "base_color_linear_rmse": float(torch.sqrt(base_difference.square().mean())),
        "base_color_linear_psnr": float(
            -10.0 * torch.log10(base_difference.square().mean().clamp_min(1.0e-20))
        ),
        "base_color_charbonnier": float(
            (torch.sqrt(base_difference.square() + 1.0e-6) - 1.0e-3).mean()
        ),
        "oklab_delta_e_mean": float(delta_e.mean()),
        "oklab_delta_e_p95": float(torch.quantile(delta_e, 0.95)),
        "normal_mean_degrees": float(angles.mean()),
        "roughness_mae": float(F.l1_loss(pred.seven[:, 5], target_post.seven[:, 5])),
        "metallic_mae": float(F.l1_loss(pred.seven[:, 6], target_post.seven[:, 6])),
    }


def _prediction_textures(seven: torch.Tensor, hashes: Mapping[str, str]) -> Core4Textures:
    processed = postprocess_affine_output(
        seven, compander_parameters=None, straight_through=False
    )
    return Core4Textures(
        base_color_linear=processed.seven[..., :3].contiguous(),
        normal=((processed.normal_xyz + 1.0) * 0.5).contiguous(),
        roughness=processed.seven[..., 5:6].contiguous(),
        metallic=processed.seven[..., 6:7].contiguous(),
        source_hashes=hashes,
    )


def _display(linear: torch.Tensor, exposure: float) -> np.ndarray:
    mapped = torch.clamp(linear * exposure, min=0.0)
    mapped = mapped / (1.0 + mapped)
    return np.rint(
        linear_to_srgb_torch(mapped).clamp(0.0, 1.0).detach().cpu().numpy() * 255.0
    ).astype(np.uint8)


def _panel(images: list[np.ndarray], labels: tuple[str, ...]) -> Image.Image:
    height, width = images[0].shape[:2]
    panel = Image.new("RGB", (width * len(images), height + 34), (22, 22, 25))
    draw = ImageDraw.Draw(panel)
    for index, (image, label) in enumerate(zip(images, labels, strict=True)):
        panel.paste(Image.fromarray(image), (index * width, 34))
        draw.text((index * width + 10, 10), label, fill=(235, 235, 235))
    return panel


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric = [
        key
        for key, value in rows[0].items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return {
        "case_count": len(rows),
        "mean": {key: float(np.mean([row[key] for row in rows])) for key in numeric},
        "worst": {key: float(max(row[key] for row in rows)) for key in numeric},
    }


def _selection_render_metrics(
    render_metrics: Mapping[str, Mapping[str, Mapping[str, float]]],
) -> dict[str, float]:
    raw_mean = render_metrics["raw_q4"]["mean"]
    oracle_mean = render_metrics["oracle_q6"]["mean"]
    return {
        "q4_hdr_mae": float(raw_mean["masked_linear_hdr_mae"]),
        "q6_hdr_mae": float(oracle_mean["masked_linear_hdr_mae"]),
        "q4_ssim": float(raw_mean["display_ssim"]),
        "q6_ssim": float(oracle_mean["display_ssim"]),
    }


def _static_contract(spec: Mapping[str, Any]) -> dict[str, Any]:
    gltf = _repo_path(spec["gltf"], f"{spec['id']}.gltf")
    metadata = _repo_path(spec["metadata"], f"{spec['id']}.metadata")
    _verify(gltf, spec["gltf_sha256"], f"{spec['id']} glTF")
    _verify(metadata, spec["metadata_sha256"], f"{spec['id']} metadata")
    document = json.loads(gltf.read_text(encoding="utf-8"))
    metadata_document = json.loads(metadata.read_text(encoding="utf-8"))
    declared = {
        str(row.get("spdx"))
        for row in metadata_document.get("legal", [])
        if isinstance(row, Mapping)
    }
    if str(spec["license_spdx"]) not in declared:
        raise ValueError(f"{spec['id']} metadata does not declare configured license")
    contract = audit_complex_gltf_document(
        document,
        license_spdx=str(spec["license_spdx"]),
        emissive_identity_preserved=spec.get("emissive_identity_preserved"),
    )
    return {
        "asset_id": str(spec["id"]),
        "gltf": gltf.relative_to(ROOT).as_posix(),
        "gltf_sha256": sha256_file(gltf),
        "metadata_sha256": sha256_file(metadata),
        "source_url": str(spec["source_url"]),
        "license_spdx": contract.license_spdx,
        "contract_eligible": contract.eligible,
        "emissive_present": contract.emissive_present,
        "emissive_review": contract.emissive_review,
    }


def _process_asset(
    spec: Mapping[str, Any], config: Mapping[str, Any], output_root: Path
) -> dict[str, Any]:
    static = _static_contract(spec)
    asset_id = str(spec["id"])
    gltf = _repo_path(spec["gltf"], f"{asset_id}.gltf")
    asset = load_gltf_core4_asset(
        gltf,
        name=asset_id,
        expected_size=tuple(int(value) for value in config["atlas_resolution"]),
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
    if valid_target.shape[0] < 4096:
        raise ValueError(f"{asset_id} has fewer than 4096 valid UV texels")

    oracle = fit_global_valid_pca_oracle(target_atlas, valid_mask, rank=7)
    ranks: dict[str, Any] = {}
    predictions: dict[int, torch.Tensor] = {}
    for rank in range(1, 8):
        value = oracle.mean + oracle.valid_scores[:, :rank] @ oracle.components[:rank]
        value = postprocess_affine_output(
            value, compander_parameters=None, straight_through=False
        ).seven
        predictions[rank] = value
        ranks[f"q{rank}"] = {
            "deployable": rank <= 4,
            "metrics": _material_metrics(value, valid_target),
        }
    raw_bundle = export_p0_bundle(target_atlas, valid_mask, chart_ids, margin=1.0e-3)
    raw = raw_bundle.calibration.raw
    raw_latent = raw.latent_unorm8[valid_mask].to(torch.float32) / 255.0
    raw_q4 = postprocess_affine_output(
        F.linear(raw_latent, raw.weight, raw.bias),
        compander_parameters=None,
        straight_through=False,
    ).seven
    direct_base = torch.floor(valid_target[:, :3].clamp(0.0, 1.0) * 255.0 + 0.5) / 255.0
    direct = valid_target.clone()
    direct[:, :3] = direct_base
    source_std = valid_target.std(dim=0, unbiased=False).clamp_min(1.0e-6)
    raw_standardized = (raw_q4 - valid_target).abs().mean(dim=0) / source_std
    direct_standardized = (direct - valid_target).abs().mean(dim=0) / source_std

    reconstructed = {name: target_atlas.clone() for name in COLUMNS if name != "source"}
    reconstructed["raw_q4"][valid_mask] = raw_q4
    reconstructed["direct_basecolor"][valid_mask] = direct
    reconstructed["oracle_q6"][valid_mask] = predictions[6]
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
    rig = build_generic_c4_rig()
    rig_config = config["rig"]
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in COLUMNS[1:]}
    presentation: list[Image.Image] = []
    presentation_by_index = {index: name for name, index in rig.presentation_views.items()}
    for camera_index, camera_spec in enumerate(rig.cameras):
        camera, center, radius = instantiate_camera(
            asset.mesh,
            camera_spec,
            vertical_fov_degrees=float(rig_config["vertical_fov_degrees"]),
            distance_padding=float(rig_config["distance_padding"]),
        )
        geometry = render_geometry_gbuffer(
            asset.mesh,
            camera,
            rig.resolution,
            device="cuda",
            cull_backfaces=bool(rig_config["backface_culling"]),
        )
        lights = instantiate_lights(camera, center, radius, rig)
        materials = {
            name: sample_core4_material(geometry, textures)
            for name, textures in device_textures.items()
        }
        view_first_images: dict[str, np.ndarray] = {}
        for light_index, light in enumerate(lights):
            rendered = {
                name: shade_ggx(
                    geometry,
                    camera,
                    light,
                    material_override=material,
                    minimum_roughness=float(rig_config["minimum_roughness"]),
                )
                for name, material in materials.items()
            }
            source_hdr = rendered["source"]
            mask = geometry.torch_buffers["mask"]
            for name in COLUMNS[1:]:
                metric = masked_render_metrics(
                    source_hdr,
                    rendered[name],
                    mask,
                    linear_psnr_data_range=float(rig_config["linear_psnr_data_range"]),
                    display_exposure=float(rig_config["display_exposure"]),
                )
                metric.update(
                    {
                        "camera_index": camera_index,
                        "light_index": light_index,
                        "camera_split": camera_spec.split,
                    }
                )
                rows[name].append(metric)
            if light_index == 0 and camera_index in presentation_by_index:
                view_first_images = {
                    name: _display(hdr, float(rig_config["display_exposure"]))
                    for name, hdr in rendered.items()
                }
        if view_first_images:
            panel = _panel(
                [view_first_images[name] for name in COLUMNS], COLUMNS
            )
            view_name = presentation_by_index[camera_index]
            target = output_root / asset_id / "views" / f"{view_name}.png"
            target.parent.mkdir(parents=True, exist_ok=True)
            panel.save(target, format="PNG")
            presentation.append(panel)

    render_metrics = {name: _aggregate(values) for name, values in rows.items()}
    if len(presentation) != 4:
        raise RuntimeError("generic rig did not produce all four presentation views")
    contact = Image.new(
        "RGB",
        (max(panel.width for panel in presentation), sum(panel.height for panel in presentation)),
        (22, 22, 25),
    )
    offset = 0
    for panel in presentation:
        contact.paste(panel, (0, offset))
        offset += panel.height
    contact_path = output_root / asset_id / "four_view_contact.png"
    contact.save(contact_path, format="PNG")

    q4_metrics = _material_metrics(raw_q4, valid_target)
    q6_metrics = _material_metrics(predictions[6], valid_target)
    report = {
        "schema_version": 1,
        **static,
        "source_contract_eligible": bool(static["contract_eligible"]),
        "runtime_eligible": True,
        "core4_identity_preserved": (
            True
            if not static["emissive_present"]
            else spec.get("emissive_identity_preserved") is True
        ),
        "asset_manifest": asset.manifest,
        "rig_hash": rig.rig_hash,
        "valid_uv": {
            "texels": int(valid_mask.sum()),
            "ratio": float(valid_mask.to(torch.float64).mean()),
            "charts": int(torch.unique(chart_ids[valid_mask]).numel()),
        },
        "rank_distortion": ranks,
        "raw_q4": {
            "artifact_hash": raw.artifact_hash,
            "metrics": q4_metrics,
            "pre_safety_violation_reported": True,
        },
        "direct_basecolor_oracle": {"metrics": _material_metrics(direct, valid_target)},
        "oracle_q6": {"deployable": False, "metrics": q6_metrics},
        "render_31x6": render_metrics,
        "q4_basecolor_error": q4_metrics["base_color_linear_mae"],
        "q6_basecolor_error": q6_metrics["base_color_linear_mae"],
        "q4_seven_error": q4_metrics["seven_channel_mae"],
        "q6_seven_error": q6_metrics["seven_channel_mae"],
        **_selection_render_metrics(render_metrics),
        "basecolor_q4_excess": float(
            raw_standardized[:3].mean() - direct_standardized[:3].mean()
        ),
        "nonbase_standardized_residual": float(raw_standardized[3:].mean()),
        "four_view_contact_sha256": sha256_file(contact_path),
        "selection_metric_yellow": False,
        "formal_holdout_accessed": False,
        "training_started": False,
        "ue_started": False,
    }
    _write_json(output_root / asset_id / "report.json", report)
    del device_textures, reconstructed, target_atlas
    torch.cuda.empty_cache()
    return report


def _process_asset_or_rejection(
    spec: Mapping[str, Any], config: Mapping[str, Any], output_root: Path
) -> dict[str, Any]:
    try:
        return _process_asset(spec, config, output_root)
    except GltfMeshError as error:
        static = _static_contract(spec)
        report = {
            "schema_version": 1,
            **static,
            "source_contract_eligible": bool(static["contract_eligible"]),
            "contract_eligible": False,
            "runtime_eligible": False,
            "core4_identity_preserved": False,
            "screen_rejection": {
                "type": type(error).__name__,
                "message": str(error),
            },
            "formal_holdout_accessed": False,
            "training_started": False,
            "ue_started": False,
        }
        _write_json(output_root / str(spec["id"]) / "report.json", report)
        return report


def run(
    config_path: Path,
    *,
    output_override: Path | None = None,
    contract_only: bool = False,
) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping) or config.get("schema_version") != 1:
        raise ValueError("unsupported complex asset screen config")
    if config.get("formal_holdout_access") != "forbidden":
        raise ValueError("complex asset screen must forbid formal holdout")
    rig = build_generic_c4_rig()
    if tuple(config["rig"]["resolution"]) != rig.resolution:
        raise ValueError("config resolution does not match frozen generic rig")
    contracts = [_static_contract(spec) for spec in config["assets"]]
    if contract_only:
        return {
            "schema_version": 1,
            "status": "complete_contract_only",
            "rig_hash": rig.rig_hash,
            "assets": contracts,
        }
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for 31x6 complex asset screening")
    output = (
        output_override.resolve()
        if output_override is not None
        else _repo_path(config["output_root"], "output_root")
    )
    if not output.is_relative_to(ROOT):
        raise ValueError("output root must remain in the repository")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output root: {output}")
    output.mkdir(parents=True)
    reports = [
        _process_asset_or_rejection(spec, config, output)
        for spec in config["assets"]
    ]
    try:
        selected = select_complex_validators(reports)
        selection: dict[str, Any] | None = {
            "basecolor_dominant": selected.basecolor_dominant,
            "cross_channel_dominant": selected.cross_channel_dominant,
            "eligible_assets": list(selected.eligible_assets),
        }
        status = "complete_two_validators_selected"
    except ValueError as error:
        selection = None
        status = "stopped_no_two_eligible_assets"
        stop_reason = str(error)
    summary = {
        "schema_version": 1,
        "status": status,
        "rig_hash": rig.rig_hash,
        "selection": selection,
        "stop_reason": None if selection is not None else stop_reason,
        "assets": reports,
        "formal_holdout_accessed": False,
        "training_started": False,
        "ue_started": False,
    }
    _write_json(output / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--contract-only", action="store_true")
    arguments = parser.parse_args()
    result = run(
        arguments.config,
        output_override=arguments.output_root,
        contract_only=arguments.contract_only,
    )
    print(json.dumps({"status": result["status"], "rig_hash": result["rig_hash"]}, sort_keys=True))


if __name__ == "__main__":
    main()
