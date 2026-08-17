"""Prepare one authored PerformanceEvidenceV1 map for deterministic -game timing."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import traceback

import unreal


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[4]
EVIDENCE_ROOT = REPO_ROOT / "outputs/analysis/ue-runtime-evidence-v1"
CONTRACT_PATH = EVIDENCE_ROOT / "measurement_contract.json"
VARIANT_ID = os.environ.get("UE_RUNTIME_VARIANT", "").strip()
REPORT_PATH = EVIDENCE_ROOT / "timing_map_preparation" / f"{VARIANT_ID or 'missing_variant'}.json"
EXPECTED_CONTRACT_SHA256 = "652fd3761a3d8817e7f026eb067206098a12ad565c6677ab15b13ce679433bb2"
DEST_ROOT = "/Game/CGCompression/PerformanceEvidenceV1/Maps/Single"
FIXED_CAMERA_LABEL = "Camera_MaterialLab_Oracle_Fixed"
CAMERA_OFFSET = (452.2026847839356, 758.5334865319824, 132.50294933156258)
CAMERA_ROTATION = (-12.929087956085821, -142.25319461272534, 0.0)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(value: dict) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def prepare() -> None:
    report = {
        "schema_version": 1,
        "status": "started",
        "variant_id": VARIANT_ID,
        "formal_holdout_accessed": False,
        "authored_content_only": True,
    }
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

        map_asset = f"{DEST_ROOT}/{VARIANT_ID}"
        if not unreal.EditorAssetLibrary.does_asset_exist(map_asset):
            raise RuntimeError(f"map asset missing: {map_asset}")
        unreal.EditorLoadingAndSavingUtils.load_map(map_asset)
        actors = list(unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors())
        target_label = f"PerformanceEvidence_{VARIANT_ID}"
        targets = [actor for actor in actors if actor.get_actor_label() == target_label]
        cameras = [actor for actor in actors if isinstance(actor, unreal.CameraActor)]
        fixed = [actor for actor in cameras if actor.get_actor_label() == FIXED_CAMERA_LABEL]
        if len(targets) != 1 or not isinstance(targets[0], unreal.StaticMeshActor):
            raise RuntimeError(f"expected exactly one static target actor: {target_label}")
        if len(fixed) != 1:
            raise RuntimeError(f"expected exactly one fixed camera: {FIXED_CAMERA_LABEL}")

        target_location = targets[0].get_actor_location()
        camera_location = unreal.Vector(
            target_location.x + CAMERA_OFFSET[0],
            target_location.y + CAMERA_OFFSET[1],
            target_location.z + CAMERA_OFFSET[2],
        )
        camera_rotation = unreal.Rotator(
            roll=CAMERA_ROTATION[2],
            pitch=CAMERA_ROTATION[0],
            yaw=CAMERA_ROTATION[1],
        )
        for camera in cameras:
            camera.set_editor_property("auto_activate_for_player", unreal.AutoReceiveInput.DISABLED)
        fixed[0].set_actor_location(camera_location, False, False)
        fixed[0].set_actor_rotation(camera_rotation, False)
        fixed[0].set_editor_property("auto_activate_for_player", unreal.AutoReceiveInput.PLAYER0)

        saved = unreal.EditorLoadingAndSavingUtils.save_current_level()
        if not saved:
            raise RuntimeError("save_current_level returned false")
        map_file = SCRIPT_PATH.parents[1] / "CGCompression/PerformanceEvidenceV1/Maps/Single" / f"{VARIANT_ID}.umap"
        report.update(
            {
                "status": "complete",
                "contract_sha256": contract_hash,
                "map_asset": map_asset,
                "map_file_sha256": _sha256(map_file),
                "map_saved": True,
                "target_label": target_label,
                "camera_label": FIXED_CAMERA_LABEL,
                "camera_location": [camera_location.x, camera_location.y, camera_location.z],
                "camera_rotation": [camera_rotation.pitch, camera_rotation.yaw, camera_rotation.roll],
                "camera_count": len(cameras),
                "active_camera": "Player0",
            }
        )
        _write(report)
        unreal.log(f"Timing map prepared: {VARIANT_ID}")
    except Exception as exc:
        report.update({"status": "failed", "error": str(exc), "traceback": traceback.format_exc()})
        _write(report)
        unreal.log_error(f"Timing map preparation failed: {exc}")
        raise


prepare()
