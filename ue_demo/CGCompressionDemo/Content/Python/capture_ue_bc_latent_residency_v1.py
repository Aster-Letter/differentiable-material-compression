"""Capture warmed texture residency from a prebuilt isolated BC7 runtime map."""

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
CONFIG_PATH = REPO_ROOT / "configs/eval/ue_bc_latent_feasibility_v1.json"
EVIDENCE_ROOT = REPO_ROOT / "outputs/analysis/ue-bc-latent-feasibility-v1"
VARIANT_ID = os.environ.get("UE_BC_LATENT_VARIANT", "").strip()
REPORT_PATH = EVIDENCE_ROOT / "residency_runs" / f"{VARIANT_ID or 'missing_variant'}.json"
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
    REPORT_PATH.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


REPORT = {
    "schema_version": 1,
    "status": "waiting_for_map",
    "variant_id": VARIANT_ID,
    "formal_holdout_accessed": False,
    "map_saved": False,
    "warmup_seconds": WARMUP_SECONDS,
}
STARTED_AT = None
CAPTURED_AT = None
TICK_HANDLE = None
VARIANT = None


def _fail(exc: Exception) -> None:
    global TICK_HANDLE
    REPORT.update({"status": "failed", "error": str(exc), "traceback": traceback.format_exc()})
    _write(REPORT)
    unreal.log_error(f"BC latent residency capture failed: {exc}")
    if TICK_HANDLE is not None:
        unreal.unregister_slate_post_tick_callback(TICK_HANDLE)
        TICK_HANDLE = None
    unreal.SystemLibrary.execute_console_command(None, "QUIT_EDITOR")


def _tick(_delta_seconds: float) -> None:
    global STARTED_AT, CAPTURED_AT, TICK_HANDLE
    try:
        now = time.monotonic()
        if STARTED_AT is None:
            for command in (
                "r.Streaming.FullyLoadUsedTextures 1",
                "r.ScreenPercentage 100",
                "r.DynamicRes.OperationMode 0",
                "r.VSync 0",
                "t.MaxFPS 0",
            ):
                unreal.SystemLibrary.execute_console_command(None, command)
            STARTED_AT = now
            REPORT.update(
                {
                    "status": "warming_up",
                    "runtime_map": VARIANT["destination_map"],
                    "runtime_actor_label": VARIANT["destination_actor_label"],
                    "runtime_material": VARIANT["destination_material"],
                    "config_sha256": _sha256(CONFIG_PATH),
                }
            )
            _write(REPORT)
            return
        if CAPTURED_AT is None and now - STARTED_AT >= WARMUP_SECONDS:
            unreal.SystemLibrary.execute_console_command(None, "ListTextures")
            unreal.SystemLibrary.execute_console_command(None, "stat streaming")
            CAPTURED_AT = now
            REPORT.update({"status": "captured", "actual_warmup_seconds": now - STARTED_AT})
            _write(REPORT)
            return
        if CAPTURED_AT is not None and now - CAPTURED_AT >= EXIT_DELAY_SECONDS:
            REPORT["status"] = "complete"
            _write(REPORT)
            unreal.unregister_slate_post_tick_callback(TICK_HANDLE)
            TICK_HANDLE = None
            unreal.SystemLibrary.execute_console_command(None, "QUIT_EDITOR")
    except Exception as exc:
        _fail(exc)


def start() -> None:
    global TICK_HANDLE, VARIANT
    try:
        if not VARIANT_ID:
            raise RuntimeError("UE_BC_LATENT_VARIANT is required")
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
