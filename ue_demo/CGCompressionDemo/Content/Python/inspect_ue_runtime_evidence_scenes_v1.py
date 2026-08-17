"""Inspect source-map actors without saving any existing UE map."""

from __future__ import annotations

import json
from pathlib import Path
import traceback

import unreal


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[4]
OUTPUT_PATH = (
    REPO_ROOT
    / "outputs/analysis/ue-runtime-evidence-v1/source_scene_inventory.json"
)
MAPS = (
    "/Game/CGCompression/C4LanternRender160k/Maps/C4_Lantern_Source_Raw_20k_160k",
    "/Game/CGCompression/Maps/MaterialLab",
)


def _vec(value: unreal.Vector) -> list[float]:
    return [float(value.x), float(value.y), float(value.z)]


def _rot(value: unreal.Rotator) -> list[float]:
    return [float(value.pitch), float(value.yaw), float(value.roll)]


def _actor_record(actor: unreal.Actor) -> dict:
    transform = actor.get_actor_transform()
    record = {
        "label": actor.get_actor_label(),
        "class": actor.get_class().get_name(),
        "location": _vec(transform.translation),
        "rotation": _rot(transform.rotation.rotator()),
        "scale": _vec(transform.scale3d),
        "hidden": bool(actor.is_hidden_ed()),
    }
    if isinstance(actor, unreal.StaticMeshActor):
        component = actor.static_mesh_component
        mesh = component.get_editor_property("static_mesh")
        record["static_mesh"] = mesh.get_path_name() if mesh else None
        record["materials"] = [
            material.get_path_name() if material else None
            for material in component.get_materials()
        ]
    elif isinstance(actor, unreal.CameraActor):
        component = actor.get_editor_property("camera_component")
        record["field_of_view"] = float(
            component.get_editor_property("field_of_view")
        )
    elif isinstance(actor, unreal.DirectionalLight):
        component = actor.get_editor_property("directional_light_component")
        record["intensity"] = float(component.get_editor_property("intensity"))
    elif isinstance(actor, unreal.PointLight):
        component = actor.get_editor_property("point_light_component")
        record["intensity"] = float(component.get_editor_property("intensity"))
        record["attenuation_radius"] = float(
            component.get_editor_property("attenuation_radius")
        )
    return record


def inspect() -> None:
    result = {
        "schema_version": 1,
        "status": "started",
        "formal_holdout_accessed": False,
        "source_maps_saved": False,
        "maps": {},
    }
    try:
        actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        for map_path in MAPS:
            unreal.EditorLoadingAndSavingUtils.load_map(map_path)
            actors = sorted(
                actor_subsystem.get_all_level_actors(),
                key=lambda actor: actor.get_actor_label(),
            )
            result["maps"][map_path] = {
                "actor_count": len(actors),
                "actors": [_actor_record(actor) for actor in actors],
            }
        result["status"] = "complete"
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        unreal.log(f"UE runtime evidence source scene inventory: {OUTPUT_PATH}")
    except Exception as exc:
        result.update(
            {"status": "failed", "error": str(exc), "traceback": traceback.format_exc()}
        )
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        unreal.log_error(f"UE runtime scene inspection failed: {exc}")
        raise


inspect()
