"""Serially capture reference/baseline/R0b with three fixed close-up cameras."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import traceback

import unreal


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[4]
OUTPUT_DIR = REPO_ROOT / "outputs/deployment/scifihelmet/hybrid_direct_scalars_r0b/ue_evidence/captures"
REPORT_PATH = OUTPUT_DIR / "capture_report.json"
MAP_PATH = "/Game/CGCompression/Maps/HybridInterpolation/R0B_Acceptance"

MATERIALS = [
    ("reference", "/Game/CGCompression/Materials/Reference/M_SciFiHelmet_Reference"),
    ("baseline", "/Game/CGCompression/Materials/Compressed/M_SciFiHelmet_Compressed"),
    ("candidate_r0b", "/Game/CGCompression/Materials/HybridInterpolation/R0B/M_SciFiHelmet_Hybrid_R0B"),
]
CAMERAS = [
    ("d1_metallic_boundary", "Camera_R0B_D1_MetallicBoundary"),
    ("d2_yellow_tube", "Camera_R0B_D2_YellowTube"),
    ("d3_gray_panel", "Camera_R0B_D3_GrayPanel"),
]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_report(value: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def setup_capture() -> None:
    report = {
        "schema_version": 1,
        "status": "started",
        "map": MAP_PATH,
        "one_actor_serial_material_switch": True,
        "shared_lighting_exposure_postprocess": True,
        "formal_holdout_accessed": False,
        "captures": [],
    }
    try:
        if not unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).load_level(MAP_PATH):
            raise RuntimeError(f"could not load R0b acceptance map: {MAP_PATH}")
        actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        actors = list(actor_subsystem.get_all_level_actors())
        capture = next(
            (a for a in actors if a.get_actor_label() == "Helmet_R0B_SequentialCapture"), None
        )
        if capture is None:
            raise RuntimeError("R0b sequential capture actor is missing")
        component = capture.get_component_by_class(unreal.StaticMeshComponent)
        if not isinstance(component, unreal.StaticMeshComponent):
            raise RuntimeError("R0b sequential capture actor has no StaticMeshComponent")

        hidden_states = {}
        for actor in actors:
            label = actor.get_actor_label()
            if actor is capture or not label.startswith("Helmet_"):
                continue
            hidden_states[label] = bool(actor.is_hidden_ed())
            actor.set_is_temporarily_hidden_in_editor(True)
        capture.set_is_temporarily_hidden_in_editor(False)

        materials = []
        for name, path in MATERIALS:
            material = unreal.EditorAssetLibrary.load_asset(path)
            if not isinstance(material, unreal.Material):
                raise RuntimeError(f"capture material missing: {path}")
            materials.append((name, material))
        cameras = []
        for name, label in CAMERAS:
            camera = next(
                (a for a in actors if isinstance(a, unreal.CameraActor) and a.get_actor_label() == label),
                None,
            )
            if camera is None:
                raise RuntimeError(f"capture camera missing: {label}")
            camera_component = camera.get_component_by_class(unreal.CameraComponent)
            cameras.append((name, camera, float(camera_component.get_editor_property("field_of_view"))))

        schedule = []
        for view_name, camera, fov in cameras:
            for material_name, material in materials:
                schedule.append((view_name, camera, fov, material_name, material))

        state = {
            "index": 0,
            "task": None,
            "callback": None,
            "schedule": schedule,
            "component": component,
            "capture": capture,
            "actors": actors,
            "hidden_states": hidden_states,
            "report": report,
        }

        def begin(index: int) -> None:
            view_name, camera, fov, material_name, material = state["schedule"][index]
            state["component"].set_material(0, material)
            state["capture"].modify()
            unreal.AutomationLibrary.finish_loading_before_screenshot()
            path = OUTPUT_DIR / f"{index + 1:02d}_{view_name}_{material_name}.png"
            task = unreal.AutomationLibrary.take_high_res_screenshot(
                1600,
                1200,
                path.as_posix(),
                camera=camera,
                mask_enabled=False,
                capture_hdr=False,
                comparison_tolerance=unreal.ComparisonTolerance.LOW,
                comparison_notes=f"R0b {view_name} {material_name}",
                delay=0.75,
                force_game_view=True,
            )
            if not task.is_valid_task():
                raise RuntimeError(f"UE rejected capture: {view_name}/{material_name}")
            state["task"] = task
            state["report"]["captures"].append(
                {
                    "view": view_name,
                    "camera": camera.get_actor_label(),
                    "camera_location": str(camera.get_actor_location()),
                    "camera_rotation": str(camera.get_actor_rotation()),
                    "field_of_view": fov,
                    "material_role": material_name,
                    "material": material.get_path_name(),
                    "path": path.as_posix(),
                }
            )
            state["report"]["status"] = f"capturing_{view_name}_{material_name}"
            _write_report(state["report"])

        def tick(_delta_seconds: float) -> None:
            task = state["task"]
            if task is None or not task.is_task_done():
                return
            state["index"] += 1
            if state["index"] < len(state["schedule"]):
                begin(state["index"])
                return
            for actor in state["actors"]:
                label = actor.get_actor_label()
                if label in state["hidden_states"]:
                    actor.set_is_temporarily_hidden_in_editor(state["hidden_states"][label])
            state["report"]["status"] = "complete"
            for record in state["report"]["captures"]:
                path = Path(record["path"])
                if not path.is_file():
                    raise RuntimeError(f"capture file missing after valid task: {path}")
                record["sha256"] = _sha256(path)
                record["file_bytes"] = path.stat().st_size
            state["report"]["shared_actor"] = state["capture"].get_actor_label()
            state["report"]["shared_transform"] = str(state["capture"].get_actor_transform())
            _write_report(state["report"])
            unreal.unregister_slate_post_tick_callback(state["callback"])
            unreal.log(f"CGCompression R0b serial captures complete: {REPORT_PATH.as_posix()}")
            unreal.SystemLibrary.quit_editor()

        state["callback"] = unreal.register_slate_post_tick_callback(tick)
        begin(0)
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        _write_report(report)
        unreal.log_error(f"CGCompression R0b serial capture failed: {exc}")
        raise


setup_capture()
