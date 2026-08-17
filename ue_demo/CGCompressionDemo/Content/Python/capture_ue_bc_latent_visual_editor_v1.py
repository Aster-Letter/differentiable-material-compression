"""Capture one fixed-camera Lantern frame through UE editor automation."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import unreal


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[4]
CONFIG_PATH = REPO_ROOT / "configs/eval/ue_bc_latent_visual_v1.json"
EVIDENCE_ROOT = REPO_ROOT / "outputs/analysis/ue-bc-latent-feasibility-v1"
VARIANT_ID = os.environ.get("UE_BC_LATENT_VISUAL_VARIANT", "").strip()
VISUAL_KIND = os.environ.get("UE_BC_LATENT_VISUAL_KIND", "").strip().lower()
REPLICATE_ID = os.environ.get("UE_BC_LATENT_VISUAL_REPLICATE", "").strip()
KEEP_OPEN = os.environ.get("UE_BC_LATENT_VISUAL_KEEP_OPEN") == "1"
OUTPUT_STEM = f"{VARIANT_ID}__{VISUAL_KIND}" + (f"__{REPLICATE_ID}" if REPLICATE_ID else "")
OUTPUT_PATH = EVIDENCE_ROOT / "screenshots" / f"{OUTPUT_STEM}.png"
REPORT_PATH = EVIDENCE_ROOT / "visual_runs" / f"{OUTPUT_STEM}.json"
FIXED_CAMERA_LABEL = "Camera_MaterialLab_Oracle_Fixed"
WARMUP_SECONDS = 30.0
CAPTURE_TIMEOUT_SECONDS = 20.0
EXIT_DELAY_SECONDS = 3.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(value: dict) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


REPORT = {
    "schema_version": 1,
    "status": "starting",
    "variant_id": VARIANT_ID,
    "visual_kind": VISUAL_KIND,
    "replicate_id": REPLICATE_ID or "primary",
    "formal_holdout_accessed": False,
    "map_saved": False,
    "warmup_seconds": WARMUP_SECONDS,
    "capture_api": "AutomationLibrary.take_high_res_screenshot",
    "keep_open_after_capture": KEEP_OPEN,
}
STARTED_AT = None
CAPTURE_REQUESTED_AT = None
CAPTURE_FOUND_AT = None
TICK_HANDLE = None
CAPTURE_TASK = None
WORLD = None
CAMERA = None
VARIANT = None


def _finish(status: str) -> None:
    global TICK_HANDLE
    REPORT["status"] = status
    _write(REPORT)
    if TICK_HANDLE is not None:
        unreal.unregister_slate_post_tick_callback(TICK_HANDLE)
        TICK_HANDLE = None
    unreal.SystemLibrary.execute_console_command(WORLD, "QUIT_EDITOR")


def _complete_without_exit() -> None:
    global TICK_HANDLE
    REPORT["status"] = "complete_kept_open"
    _write(REPORT)
    if TICK_HANDLE is not None:
        unreal.unregister_slate_post_tick_callback(TICK_HANDLE)
        TICK_HANDLE = None
    unreal.log("BC latent visual capture complete; editor kept open for inspection")


def _fail(exc: Exception) -> None:
    REPORT.update({"error": str(exc), "traceback": traceback.format_exc()})
    unreal.log_error(f"BC latent editor visual capture failed: {exc}")
    _finish("failed")


def _tick(_delta_seconds: float) -> None:
    global STARTED_AT, CAPTURE_REQUESTED_AT, CAPTURE_FOUND_AT, CAPTURE_TASK, WORLD, CAMERA
    try:
        now = time.monotonic()
        if STARTED_AT is None:
            actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
            world_subsystem = unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem)
            if actor_subsystem is None or world_subsystem is None:
                return
            WORLD = world_subsystem.get_editor_world()
            actors = list(actor_subsystem.get_all_level_actors())
            cameras = [
                actor
                for actor in actors
                if isinstance(actor, unreal.CameraActor)
                and actor.get_actor_label() == FIXED_CAMERA_LABEL
            ]
            if WORLD is None or len(cameras) != 1:
                return
            CAMERA = cameras[0]
            for command in (
                "r.Streaming.FullyLoadUsedTextures 1",
                "r.ScreenPercentage 100",
                "r.DynamicRes.OperationMode 0",
                "r.VSync 0",
                "t.MaxFPS 0",
            ):
                unreal.SystemLibrary.execute_console_command(WORLD, command)
            STARTED_AT = now
            REPORT.update(
                {
                    "status": "warming_up",
                    "runtime_map": (
                        VARIANT["bc7_map"] if VISUAL_KIND == "bc7" else VARIANT["rgba8_map"]
                    ),
                    "runtime_material": (
                        VARIANT["bc7_material"]
                        if VISUAL_KIND == "bc7"
                        else VARIANT["rgba8_material"]
                    ),
                    "camera_label": CAMERA.get_actor_label(),
                    "config_sha256": _sha256(CONFIG_PATH),
                    "output_path": OUTPUT_PATH.relative_to(REPO_ROOT).as_posix(),
                    "resolution": [1920, 1080],
                }
            )
            _write(REPORT)
            return
        if CAPTURE_REQUESTED_AT is None and now - STARTED_AT >= WARMUP_SECONDS:
            OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
            unreal.AutomationLibrary.finish_loading_before_screenshot()
            CAPTURE_TASK = unreal.AutomationLibrary.take_high_res_screenshot(
                1920,
                1080,
                str(OUTPUT_PATH.resolve()),
                camera=CAMERA,
                delay=0.0,
                force_game_view=True,
            )
            CAPTURE_REQUESTED_AT = now
            REPORT.update(
                {"status": "capture_requested", "actual_warmup_seconds": now - STARTED_AT}
            )
            _write(REPORT)
            return
        if CAPTURE_REQUESTED_AT is not None and CAPTURE_FOUND_AT is None:
            if OUTPUT_PATH.is_file() and OUTPUT_PATH.stat().st_size > 0:
                CAPTURE_FOUND_AT = now
                REPORT.update(
                    {
                        "status": "captured",
                        "screenshot_bytes": OUTPUT_PATH.stat().st_size,
                        "screenshot_sha256": _sha256(OUTPUT_PATH),
                    }
                )
                _write(REPORT)
                return
            if now - CAPTURE_REQUESTED_AT >= CAPTURE_TIMEOUT_SECONDS:
                raise RuntimeError(f"automation screenshot was not created: {OUTPUT_PATH}")
        if CAPTURE_FOUND_AT is not None and now - CAPTURE_FOUND_AT >= EXIT_DELAY_SECONDS:
            if KEEP_OPEN:
                _complete_without_exit()
            else:
                _finish("complete")
    except Exception as exc:
        _fail(exc)


def start() -> None:
    global TICK_HANDLE, VARIANT
    try:
        if not VARIANT_ID or VISUAL_KIND not in {"rgba8", "bc7"}:
            raise RuntimeError("visual variant and kind=rgba8|bc7 are required")
        if OUTPUT_PATH.exists() or REPORT_PATH.exists():
            raise RuntimeError(f"refusing to overwrite visual evidence: {OUTPUT_PATH}")
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        matches = [item for item in config["variants"] if item["id"] == VARIANT_ID]
        if len(matches) != 1:
            raise RuntimeError(f"expected one BC latent variant, got {len(matches)}")
        VARIANT = matches[0]
        REPORT["config_sha256"] = _sha256(CONFIG_PATH)
        _write(REPORT)
        TICK_HANDLE = unreal.register_slate_post_tick_callback(_tick)
    except Exception as exc:
        _fail(exc)


start()
