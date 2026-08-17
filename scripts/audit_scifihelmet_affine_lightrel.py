"""Audit the zero-update 31x6 camera-relative SciFiHelmet reference pool."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cg_frontier.render.camera_relative_lighting import (  # noqa: E402
    build_camera_relative_light_grid,
    parse_camera_relative_light_families,
)
from cg_frontier.assets.gltf_mesh import load_gltf_mesh  # noqa: E402
from cg_frontier.render.gbuffer import (  # noqa: E402
    load_core4_textures,
    render_geometry_gbuffer,
    sample_core4_material,
)
from cg_frontier.render.pbr import shade_ggx  # noqa: E402
from audit_scifihelmet_affine_render_sampling import (  # noqa: E402
    _camera_from_spec,
    _direct_light_statistics,
    _display_encode,
    _mapping,
    _repo_path,
    _sha256,
    build_sample_bundle,
    camera_light_alignment,
    evaluate_camera_relative_audit,
    masked_reference_statistics,
)


DEFAULT_CONFIG = (
    ROOT / "configs/train/scifihelmet_c4_affine_chroma8_l0_lightrel_5k.yaml"
)


@dataclass(frozen=True)
class FrozenLightRelAudit:
    resolution: tuple[int, int]
    camera_count: int
    light_count: int
    pair_count: int
    camera_side_count: int
    rim_count: int


def freeze_lightrel_audit_config(
    config: Mapping[str, object], pool: Mapping[str, object]
) -> FrozenLightRelAudit:
    """Fail closed unless the diagnostic covers all frozen 31x6 references."""

    render = pool.get("render")
    cameras = pool.get("train_cameras")
    specs = config.get("camera_relative_lights")
    gates = config.get("audit_gates")
    if not (
        config.get("experiment")
        == "scifihelmet_c4_affine_chroma8_l0_lightrel_5k"
        and isinstance(render, Mapping)
        and isinstance(cameras, Sequence)
        and isinstance(specs, Sequence)
        and isinstance(gates, Mapping)
        and gates.get("require_all_31x6_pairs") is True
    ):
        raise ValueError("lightrel audit config is incomplete")
    resolution = tuple(int(value) for value in render.get("resolution", ()))
    families = parse_camera_relative_light_families(specs)
    if resolution != (256, 256) or len(cameras) != 31:
        raise ValueError("lightrel audit requires camera31 at 256x256")
    camera_side_count = sum(family.role == "camera_side" for family in families)
    rim_count = sum(family.role == "rim" for family in families)
    return FrozenLightRelAudit(
        resolution=(256, 256),
        camera_count=31,
        light_count=len(families),
        pair_count=31 * len(families),
        camera_side_count=camera_side_count,
        rim_count=rim_count,
    )


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode("utf-8")


def run(config_path: Path) -> dict[str, object]:
    """Render every source reference without model or optimizer updates."""

    config_path = config_path.resolve()
    config = _mapping(config_path, "lightrel experiment")
    source = config.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("lightrel source section is invalid")
    render_pool_path = _repo_path(
        source["render_pool_config"], "source.render_pool_config"
    )
    preflight_path = _repo_path(source["preflight_config"], "source.preflight_config")
    if _sha256(render_pool_path) != str(source["render_pool_config_sha256"]):
        raise ValueError("render pool SHA-256 mismatch")
    if _sha256(preflight_path) != str(source["preflight_config_sha256"]):
        raise ValueError("preflight SHA-256 mismatch")
    pool = _mapping(render_pool_path, "render pool")
    preflight = _mapping(preflight_path, "preflight")
    frozen = freeze_lightrel_audit_config(config, pool)
    if not torch.cuda.is_available():
        raise RuntimeError("camera-relative reference audit requires CUDA")

    render = pool["render"]
    camera_specs = pool["train_cameras"]
    family_specs = config["camera_relative_lights"]
    cameras = [_camera_from_spec(spec, render) for spec in camera_specs]
    families = parse_camera_relative_light_families(family_specs)
    light_grid = build_camera_relative_light_grid(cameras, families)
    gltf_path = _repo_path(preflight["inputs"]["gltf"], "inputs.gltf")
    core4_manifest = _repo_path(
        preflight["inputs"]["core4_manifest"], "inputs.core4_manifest"
    )
    device = torch.device("cuda")
    mesh = load_gltf_mesh(gltf_path)
    textures = load_core4_textures(core4_manifest, device)
    minimum_roughness = float(render["minimum_roughness"])
    exposure = float(render["display_exposure"])
    records: list[dict[str, object]] = []
    samples: list[dict[str, object]] = []
    with torch.no_grad():
        for camera_index, (camera_spec, camera) in enumerate(
            zip(camera_specs, cameras, strict=True)
        ):
            geometry = render_geometry_gbuffer(
                mesh, camera, frozen.resolution, device=device
            )
            material = sample_core4_material(geometry, textures)
            mask = geometry.torch_buffers["mask"]
            mask_numpy = mask.detach().cpu().numpy()
            for light_index, family in enumerate(families):
                light = light_grid[light_index][camera_index]
                hdr = shade_ggx(
                    geometry,
                    camera,
                    light,
                    material_override=material,
                    minimum_roughness=minimum_roughness,
                )
                alignment = camera_light_alignment(
                    camera_eye=camera.eye,
                    light_position=light.position,
                    target=camera.target,
                )
                record = {
                    "camera_index": camera_index,
                    "camera_name": str(camera_spec["name"]),
                    "camera_eye": list(camera.eye),
                    "camera_target": list(camera.target),
                    "light_index": light_index,
                    "light_name": family.name,
                    "role": family.role,
                    "local_position": list(family.local_position),
                    "world_position": list(light.position),
                    **alignment,
                    "reference": masked_reference_statistics(
                        hdr.detach().cpu().numpy(), mask_numpy
                    ),
                    "direct_light": _direct_light_statistics(
                        geometry, material, light
                    ),
                }
                records.append(record)
                samples.append(
                    {
                        "step": camera_index * frozen.light_count + light_index + 1,
                        "camera_name": str(camera_spec["name"]),
                        "light_name": family.name,
                        "role": family.role,
                        "display_rgb": _display_encode(hdr, exposure),
                    }
                )
            del geometry, material
    evaluation = evaluate_camera_relative_audit(
        records,
        camera_count=frozen.camera_count,
        light_count=frozen.light_count,
    )
    output_root = _repo_path(config["audit_output_root"], "audit_output_root")
    output_root.mkdir(parents=True, exist_ok=False)
    render_root = output_root / "renders"
    render_root.mkdir()
    preview_manifest, files = build_sample_bundle(samples, columns=6)
    for name, payload in files.items():
        (render_root / name).write_bytes(payload)
    (render_root / "manifest.json").write_bytes(_json_bytes(preview_manifest))
    report = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "audit_kind": "zero_update_source_reference_31x6",
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "render_pool_sha256": _sha256(render_pool_path),
        "preflight_sha256": _sha256(preflight_path),
        "frozen": frozen.__dict__,
        "families": [family.__dict__ for family in families],
        "evaluation": evaluation,
        "records": records,
        "preview_manifest": preview_manifest,
        "formal_holdout_accessed": False,
        "optimizer_updates": 0,
    }
    (output_root / "audit_report.json").write_bytes(_json_bytes(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    report = run(args.config)
    print(_json_bytes(report["evaluation"]).decode("utf-8").rstrip())
    return 0 if report["evaluation"]["gates_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
