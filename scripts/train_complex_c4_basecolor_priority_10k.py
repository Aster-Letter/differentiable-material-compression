"""Train selected complex C4 validators with the shared BaseColor-priority engine."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for directory in (SRC, SCRIPTS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from cg_frontier.assets.gltf_core4 import load_gltf_core4_asset  # noqa: E402
from cg_frontier.assets.preprocess import sha256_file  # noqa: E402
from cg_frontier.compression.affine_pca import export_p0_bundle, rasterize_uv_charts  # noqa: E402
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
from cg_frontier.render.pbr import shade_ggx  # noqa: E402
from train_scifihelmet_c4_basecolor_priority_10k import (  # noqa: E402
    _candidate_spec,
    _load_config,
    run_audit,
    run_candidate,
)


DEFAULT_CONFIG = ROOT / "configs/train/complex_c4_basecolor_priority_10k_v1.yaml"


def _repo_path(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a repository-relative path")
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"{label} escapes the repository")
    return path


def _selected_spec(
    config: Mapping[str, Any],
    summary: Mapping[str, Any],
    asset_id: str,
) -> Mapping[str, Any]:
    selection = summary.get("selection")
    if summary.get("status") != "complete_two_validators_selected" or not isinstance(selection, Mapping):
        raise ValueError("Phase 0 did not select two complex validators")
    selected = {selection.get("basecolor_dominant"), selection.get("cross_channel_dominant")}
    if asset_id not in selected:
        raise ValueError("asset is not one of the two frozen Phase 0 validators")
    matches = [value for value in config["assets"] if value["id"] == asset_id]
    if len(matches) != 1:
        raise ValueError("selected asset is missing or duplicated in training config")
    return matches[0]


def _seven(targets) -> torch.Tensor:
    return torch.cat(
        (
            targets.base_color_linear.reshape(targets.height, targets.width, 3),
            targets.normal_xyz.reshape(targets.height, targets.width, 3)[..., :2],
            targets.roughness.reshape(targets.height, targets.width, 1),
            targets.metallic.reshape(targets.height, targets.width, 1),
        ),
        dim=-1,
    )


def prepare_generic_asset(
    config_path: Path,
    config: Mapping[str, Any],
    spec: Mapping[str, Any],
    *,
    screen_sha256: str,
) -> dict[str, Any]:
    """Prepare one selected asset with 24 train and 7 read-only audit cameras."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for generic asset preparation")
    gltf = _repo_path(spec["gltf"], f"{spec['id']}.gltf")
    if sha256_file(gltf) != str(spec["gltf_sha256"]):
        raise ValueError("selected complex asset glTF hash mismatch")
    asset = load_gltf_core4_asset(gltf, name=str(spec["id"]), device="cpu")
    target_atlas = _seven(asset.targets)
    valid_mask, chart_ids = rasterize_uv_charts(
        asset.mesh.texcoords,
        asset.mesh.triangles,
        height=asset.targets.height,
        width=asset.targets.width,
    )
    bundle = export_p0_bundle(target_atlas, valid_mask, chart_ids, margin=1.0e-3)
    raw = bundle.calibration.raw
    valid_indices = torch.nonzero(valid_mask.reshape(-1), as_tuple=False).flatten().cuda()
    target_valid = target_atlas.reshape(-1, 7).cuda()[valid_indices]
    rig = build_generic_c4_rig()
    rig_config = config["rig"]
    cameras, geometries, camera_lights = [], [], []
    textures = Core4Textures(
        base_color_linear=asset.textures.base_color_linear.cuda().contiguous(),
        normal=asset.textures.normal.cuda().contiguous(),
        roughness=asset.textures.roughness.cuda().contiguous(),
        metallic=asset.textures.metallic.cuda().contiguous(),
        source_hashes=asset.textures.source_hashes,
    )
    source_materials = []
    for camera_spec in rig.cameras:
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
        cameras.append(camera)
        geometries.append(geometry)
        camera_lights.append(instantiate_lights(camera, center, radius, rig))
        source_materials.append(sample_core4_material(geometry, textures))
    with torch.no_grad():
        references = [
            [
                shade_ggx(
                    geometries[camera_index],
                    cameras[camera_index],
                    camera_lights[camera_index][light_index],
                    material_override=source_materials[camera_index],
                    minimum_roughness=float(rig_config["minimum_roughness"]),
                )
                for camera_index in range(len(cameras))
            ]
            for light_index in range(len(rig.lights))
        ]
    source_hash = hashlib.sha256(
        "".join(asset.manifest["source_hashes"].values()).encode("ascii")
    ).hexdigest()
    return {
        "asset_id": str(spec["id"]),
        "mesh": asset.mesh,
        "textures": textures,
        "target_atlas": target_atlas.cuda(),
        "valid_mask": valid_mask.cuda(),
        "raw": raw,
        "render": rig_config,
        "cameras": cameras,
        "geometries": geometries,
        "lights": rig.lights,
        "camera_lights": camera_lights,
        "references": references,
        "source_materials": source_materials,
        "valid_indices": valid_indices,
        "target_valid": target_valid,
        "training_camera_indices": tuple(
            index for index, value in enumerate(rig.cameras) if value.split == "train"
        ),
        "lineage": {
            "parent_artifact_hash": raw.artifact_hash,
            "config_sha256": sha256_file(config_path),
            "input_sha256": source_hash,
            "screen_summary_sha256": screen_sha256,
        },
        "rig_hash": rig.rig_hash,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--screen-summary", type=Path, required=True)
    parser.add_argument("--screen-summary-sha256", required=True)
    parser.add_argument("--asset", required=True)
    parser.add_argument("--candidate")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--audit-report", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--allow-phase3", action="store_true")
    parser.add_argument("--allow-compander", action="store_true")
    arguments = parser.parse_args()
    if not arguments.allow_phase3:
        raise ValueError("complex asset work requires manual Gate 2 authorization")
    actual_screen_hash = sha256_file(arguments.screen_summary)
    if actual_screen_hash != arguments.screen_summary_sha256:
        raise ValueError("Phase 0 screen summary SHA-256 mismatch")
    summary = json.loads(arguments.screen_summary.read_text(encoding="utf-8"))
    config = _load_config(arguments.config)
    spec = _selected_spec(config, summary, arguments.asset)
    prepared = prepare_generic_asset(
        arguments.config,
        config,
        spec,
        screen_sha256=actual_screen_hash,
    )
    output = arguments.output_root.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("generic training output must remain in the repository")
    if arguments.audit_only:
        result = run_audit(
            arguments.config, output, prepared_override=prepared
        )
    else:
        if arguments.candidate is None or arguments.audit_report is None:
            raise ValueError("generic candidate training requires candidate and audit report")
        candidate = _candidate_spec(config, arguments.candidate)
        if candidate.compander and not arguments.allow_compander:
            raise ValueError("compander requires the manually selected Gate 2 endpoint")
        result = run_candidate(
            arguments.config,
            candidate_id=candidate.candidate_id,
            audit_path=arguments.audit_report.resolve(),
            output=output,
            max_steps=arguments.max_steps,
            resume=arguments.resume.resolve() if arguments.resume else None,
            prepared_override=prepared,
            rig_hash_override=prepared["rig_hash"],
        )
    print(json.dumps({"status": result["status"], "asset": arguments.asset}, sort_keys=True))


if __name__ == "__main__":
    main()
