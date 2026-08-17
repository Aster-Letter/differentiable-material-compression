"""Capture full-RHI per-map residency after a fixed warm-up, then exit read-only."""

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
EVIDENCE_ROOT = REPO_ROOT / "outputs/analysis/ue-runtime-evidence-v1"
CONTRACT_PATH = EVIDENCE_ROOT / "measurement_contract.json"
VARIANT_ID = os.environ.get("UE_RUNTIME_VARIANT", "").strip()
REPORT_PATH = EVIDENCE_ROOT / "residency_runs" / f"{VARIANT_ID or 'missing_variant'}.json"
EXPECTED_CONTRACT_SHA256 = "652fd3761a3d8817e7f026eb067206098a12ad565c6677ab15b13ce679433bb2"
WARMUP_SECONDS = 30.0
EXIT_DELAY_SECONDS = 3.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(value: dict) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


REPORT = {
    "schema_version": 1,
    "status": "waiting_for_map",
    "variant_id": VARIANT_ID,
    "formal_holdout_accessed": False,
    "map_saved": False,
    "streaming_policy": "r.Streaming.FullyLoadUsedTextures=1",
    "warmup_seconds": WARMUP_SECONDS,
}
STARTED_AT = None
CAPTURED_AT = None
TICK_HANDLE = None


def _fail(exc: Exception) -> None:
    global TICK_HANDLE
    REPORT.update({"status": "failed", "error": str(exc), "traceback": traceback.format_exc()})
    _write(REPORT)
    unreal.log_error(f"Residency capture failed: {exc}")
    if TICK_HANDLE is not None:
        unreal.unregister_slate_post_tick_callback(TICK_HANDLE)
        TICK_HANDLE = None
    unreal.SystemLibrary.execute_console_command(None, "QUIT_EDITOR")


def _tick(_delta_seconds: float) -> None:
    global STARTED_AT, CAPTURED_AT, TICK_HANDLE
    try:
        now = time.monotonic()
        if STARTED_AT is None:
            actors = list(unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors())
            target_label = f"PerformanceEvidence_{VARIANT_ID}"
            targets = [actor for actor in actors if actor.get_actor_label() == target_label]
            if len(targets) != 1 or not isinstance(targets[0], unreal.StaticMeshActor):
                return
            target = targets[0]
            transform = target.get_actor_transform()
            target_location = transform.translation
            camera_location = unreal.Vector(
                target_location.x + 452.2026847839356,
                target_location.y + 758.5334865319824,
                target_location.z + 132.50294933156258,
            )
            camera_rotation = unreal.Rotator(
                roll=0.0,
                pitch=-12.929087956085821,
                yaw=-142.25319461272534,
            )
            unreal.get_editor_subsystem(unreal.UnrealEditorSubsystem).set_level_viewport_camera_info(
                camera_location, camera_rotation
            )
            unreal.SystemLibrary.execute_console_command(None, "r.Streaming.FullyLoadUsedTextures 1")
            unreal.SystemLibrary.execute_console_command(None, "r.VSync 0")
            unreal.SystemLibrary.execute_console_command(None, "t.MaxFPS 0")
            STARTED_AT = now
            REPORT.update(
                {
                    "status": "warming_up",
                    "target_label": target_label,
                    "target_location": [target_location.x, target_location.y, target_location.z],
                    "camera_location": [camera_location.x, camera_location.y, camera_location.z],
                    "camera_rotation": [camera_rotation.pitch, camera_rotation.yaw, camera_rotation.roll],
                    "actor_count": len(actors),
                }
            )
            _write(REPORT)
            unreal.log(f"Residency warm-up started: {VARIANT_ID}")
            return
        if CAPTURED_AT is None and now - STARTED_AT >= WARMUP_SECONDS:
            unreal.SystemLibrary.execute_console_command(None, "ListTextures")
            unreal.SystemLibrary.execute_console_command(None, "stat streaming")
            CAPTURED_AT = now
            REPORT.update(
                {
                    "status": "captured",
                    "actual_warmup_seconds": now - STARTED_AT,
                    "commands": ["ListTextures", "stat streaming"],
                }
            )
            _write(REPORT)
            unreal.log(f"Residency captured: {VARIANT_ID}")
            return
        if CAPTURED_AT is not None and now - CAPTURED_AT >= EXIT_DELAY_SECONDS:
            REPORT["status"] = "complete"
            REPORT["contract_sha256"] = EXPECTED_CONTRACT_SHA256
            _write(REPORT)
            unreal.unregister_slate_post_tick_callback(TICK_HANDLE)
            TICK_HANDLE = None
            unreal.SystemLibrary.execute_console_command(None, "QUIT_EDITOR")
    except Exception as exc:
        _fail(exc)


def start() -> None:
    global TICK_HANDLE
    try:
        if not VARIANT_ID:
            raise RuntimeError("UE_RUNTIME_VARIANT is required")
        contract_hash = _sha256(CONTRACT_PATH)
        if contract_hash != EXPECTED_CONTRACT_SHA256:
            raise RuntimeError(f"contract hash mismatch: {contract_hash}")
        contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        matches = [
            variant
            for panel in contract["panels"].values()
            for variant in panel["variants"]
            if variant["id"] == VARIANT_ID
        ]
        if len(matches) != 1:
            raise RuntimeError(f"expected one contract variant, got {len(matches)}")
        REPORT["contract_sha256"] = contract_hash
        _write(REPORT)
        TICK_HANDLE = unreal.register_slate_post_tick_callback(_tick)
    except Exception as exc:
        _fail(exc)


start()
