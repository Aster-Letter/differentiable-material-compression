"""Capture reference/baseline/oracle materials on one fixed UE actor and camera."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
import traceback

import unreal


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[4]
OUTPUT_DIR = REPO_ROOT / "outputs/compression/scifihelmet/repair_v1/analysis/ue_oracle/fixed"
REPORT_PATH = OUTPUT_DIR / "fixed_capture_report.json"

MATERIALS = [
    ("reference", "/Game/CGCompression/Materials/Reference/M_SciFiHelmet_Reference"),
    ("baseline", "/Game/CGCompression/Materials/Compressed/M_SciFiHelmet_Compressed"),
    ("base_color_oracle", "/Game/CGCompression/Materials/Analysis/M_SciFiHelmet_BaseColorOracle"),
    ("metallic_oracle", "/Game/CGCompression/Materials/Analysis/M_SciFiHelmet_MetallicOracle"),
]


def _write_report(value: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def setup_capture() -> None:
    report: dict = {"schema_version": 1, "status": "started", "captures": []}
    try:
        actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        actors = list(actor_subsystem.get_all_level_actors())
        source = next((actor for actor in actors if actor.get_actor_label() == "Helmet_Compressed"), None)
        if source is None:
            raise RuntimeError("Helmet_Compressed is missing")
        capture = next((actor for actor in actors if actor.get_actor_label() == "Helmet_OracleCapture"), None)
        if capture is None:
            duplicated = list(actor_subsystem.duplicate_actors([source], None, unreal.Vector()))
            if len(duplicated) != 1:
                raise RuntimeError("could not duplicate fixed capture actor")
            capture = duplicated[0]
        capture.set_actor_label("Helmet_OracleCapture", mark_dirty=True)
        capture.set_actor_transform(source.get_actor_transform(), False, False)
        component = capture.get_component_by_class(unreal.StaticMeshComponent)
        if not isinstance(component, unreal.StaticMeshComponent):
            raise RuntimeError("fixed capture actor has no StaticMeshComponent")

        hidden_states: dict[str, bool] = {}
        for actor in actors:
            if actor is capture or not actor.get_actor_label().startswith("Helmet_"):
                continue
            hidden_states[actor.get_actor_label()] = bool(actor.is_hidden_ed())
            actor.set_is_temporarily_hidden_in_editor(True)
        capture.set_is_temporarily_hidden_in_editor(False)

        origin, extent = capture.get_actor_bounds(False, True)
        radius = max(float(extent.x), float(extent.y), float(extent.z))
        location = origin + unreal.Vector(radius * 3.1, radius * 2.4, radius * 0.9)
        rotation = unreal.MathLibrary.find_look_at_rotation(location, origin)
        camera = next(
            (
                actor
                for actor in actor_subsystem.get_all_level_actors()
                if isinstance(actor, unreal.CameraActor)
                and actor.get_actor_label() == "Camera_MaterialLab_Oracle_Fixed"
            ),
            None,
        )
        if camera is None:
            camera = actor_subsystem.spawn_actor_from_class(unreal.CameraActor, location, rotation)
        camera.set_actor_label("Camera_MaterialLab_Oracle_Fixed", mark_dirty=True)
        camera.set_actor_location(location, False, False)
        camera.set_actor_rotation(rotation, False)
        camera_component = camera.get_component_by_class(unreal.CameraComponent)
        if isinstance(camera_component, unreal.CameraComponent):
            camera_component.set_editor_property("field_of_view", 38.0)

        loaded_materials: list[tuple[str, unreal.Material]] = []
        for name, path in MATERIALS:
            material = unreal.EditorAssetLibrary.load_asset(path)
            if not isinstance(material, unreal.Material):
                raise RuntimeError(f"capture material missing: {path}")
            loaded_materials.append((name, material))

        state = {
            "index": 0,
            "task": None,
            "callback": None,
            "capture": capture,
            "component": component,
            "camera": camera,
            "materials": loaded_materials,
            "hidden_states": hidden_states,
            "actors": actors,
            "report": report,
        }

        def begin(index: int) -> None:
            name, material = state["materials"][index]
            state["component"].set_material(0, material)
            state["capture"].modify()
            unreal.AutomationLibrary.finish_loading_before_screenshot()
            path = OUTPUT_DIR / f"{index + 1}_{name}.png"
            task = unreal.AutomationLibrary.take_high_res_screenshot(
                1600,
                1200,
                path.as_posix(),
                camera=state["camera"],
                mask_enabled=False,
                capture_hdr=False,
                force_game_view=True,
            )
            if not task.is_valid_task():
                raise RuntimeError(f"UE rejected fixed capture: {name}")
            state["task"] = task
            state["report"]["captures"].append(
                {"name": name, "material": material.get_path_name(), "path": path.as_posix()}
            )
            state["report"]["status"] = f"capturing_{name}"
            _write_report(state["report"])

        def tick(_delta_seconds: float) -> None:
            task = state["task"]
            if task is None or not task.is_task_done():
                return
            state["index"] += 1
            if state["index"] < len(state["materials"]):
                begin(state["index"])
                return
            for actor in state["actors"]:
                label = actor.get_actor_label()
                if label in state["hidden_states"]:
                    actor.set_is_temporarily_hidden_in_editor(state["hidden_states"][label])
            state["report"]["status"] = "complete"
            for record in state["report"]["captures"]:
                path = Path(record["path"])
                record["sha256"] = _sha256(path)
                record["file_bytes"] = path.stat().st_size
            state["report"]["shared_actor"] = state["capture"].get_actor_label()
            state["report"]["shared_transform"] = str(state["capture"].get_actor_transform())
            state["report"]["shared_camera"] = state["camera"].get_actor_label()
            _write_report(state["report"])
            unreal.unregister_slate_post_tick_callback(state["callback"])
            unreal.log(f"CGCompression fixed oracle captures complete: {REPORT_PATH.as_posix()}")

        state["callback"] = unreal.register_slate_post_tick_callback(tick)
        begin(0)
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
        _write_report(report)
        unreal.log_error(f"CGCompression fixed oracle capture failed: {error}")
        raise


setup_capture()
