"""Create isolated, correctly aimed RGBA8/BC7 visual comparison maps."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import traceback

import unreal


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[4]
PROJECT_CONTENT = SCRIPT_PATH.parents[1]
CONFIG_PATH = REPO_ROOT / "configs/eval/ue_bc_latent_visual_v1.json"
EVIDENCE_ROOT = REPO_ROOT / "outputs/analysis/ue-bc-latent-feasibility-v1"
REPORT_PATH = EVIDENCE_ROOT / "ue_visual_setup_report.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _umap_path(map_path: str) -> Path:
    return PROJECT_CONTENT / f"{map_path.removeprefix('/Game/')}.umap"


def _write(value: dict) -> None:
    REPORT_PATH.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _prepare_map(config: dict, variant: dict, kind: str, actor_subsystem) -> dict:
    source_map = variant["source_map"]
    destination_map = variant[f"{kind}_map"]
    material_path = variant[f"{kind}_material"]
    source_hash_before = _sha256(_umap_path(source_map))
    resumed = unreal.EditorAssetLibrary.does_asset_exist(destination_map)
    if not resumed:
        unreal.EditorLoadingAndSavingUtils.load_map(source_map)
        if not unreal.EditorAssetLibrary.duplicate_asset(source_map, destination_map):
            raise RuntimeError(f"could not duplicate {source_map} to {destination_map}")
    unreal.EditorLoadingAndSavingUtils.load_map(destination_map)
    actors = list(actor_subsystem.get_all_level_actors())
    targets = [actor for actor in actors if actor.get_actor_label() == variant["source_actor_label"]]
    camera_label = config["camera_contract"]["camera_label"]
    cameras = [
        actor
        for actor in actors
        if isinstance(actor, unreal.CameraActor) and actor.get_actor_label() == camera_label
    ]
    if len(targets) != 1 or not isinstance(targets[0], unreal.StaticMeshActor):
        raise RuntimeError(f"expected one target actor in {destination_map}")
    if len(cameras) != 1:
        raise RuntimeError(f"expected one fixed camera in {destination_map}")
    target = targets[0]
    camera = cameras[0]
    material = unreal.load_asset(material_path)
    if not isinstance(material, unreal.MaterialInterface):
        raise RuntimeError(f"invalid visual material: {material_path}")
    target.static_mesh_component.set_material(0, material)
    target_center = target.get_actor_location()
    offset = config["camera_contract"]["offset"]
    camera_location = unreal.Vector(
        target_center.x + offset[0],
        target_center.y + offset[1],
        target_center.z + offset[2],
    )
    camera_rotation = unreal.MathLibrary.find_look_at_rotation(camera_location, target_center)
    camera.set_actor_location(camera_location, False, False)
    camera.set_actor_rotation(camera_rotation, False)
    camera.camera_component.set_editor_property(
        "field_of_view", float(config["camera_contract"]["field_of_view_degrees"])
    )
    for actor in actors:
        if isinstance(actor, unreal.CameraActor):
            actor.set_editor_property("auto_activate_for_player", unreal.AutoReceiveInput.DISABLED)
    camera.set_editor_property("auto_activate_for_player", unreal.AutoReceiveInput.PLAYER0)
    if not unreal.EditorLoadingAndSavingUtils.save_current_level():
        raise RuntimeError(f"could not save visual map: {destination_map}")
    source_hash_after = _sha256(_umap_path(source_map))
    if source_hash_after != source_hash_before:
        raise RuntimeError(f"source map changed while preparing {destination_map}")
    return {
        "variant_id": variant["id"],
        "kind": kind,
        "resumed_existing_isolated_map": resumed,
        "source_map": source_map,
        "destination_map": destination_map,
        "material": material.get_path_name(),
        "target_center": [target_center.x, target_center.y, target_center.z],
        "camera_location": [camera_location.x, camera_location.y, camera_location.z],
        "camera_rotation": [camera_rotation.pitch, camera_rotation.yaw, camera_rotation.roll],
        "source_map_sha256_before": source_hash_before,
        "source_map_sha256_after": source_hash_after,
        "destination_map_sha256": _sha256(_umap_path(destination_map)),
    }


def setup() -> None:
    report = {
        "schema_version": 1,
        "status": "started",
        "formal_holdout_accessed": False,
        "source_maps_saved": False,
        "maps": [],
    }
    try:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        base_path = REPO_ROOT / config["base_config"]
        if _sha256(base_path) != config["base_config_sha256"]:
            raise RuntimeError("base BC latent config hash mismatch")
        actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        if actor_subsystem is None:
            raise RuntimeError("EditorActorSubsystem is unavailable")
        for variant in config["variants"]:
            for kind in ("rgba8", "bc7"):
                report["maps"].append(_prepare_map(config, variant, kind, actor_subsystem))
        report.update(
            {
                "status": "complete",
                "config_sha256": _sha256(CONFIG_PATH),
                "map_count": len(report["maps"]),
            }
        )
        _write(report)
        unreal.log(f"BC latent visual maps created: {len(report['maps'])}")
    except Exception as exc:
        report.update({"status": "failed", "error": str(exc), "traceback": traceback.format_exc()})
        _write(report)
        unreal.log_error(f"BC latent visual setup failed: {exc}")
        raise


setup()
