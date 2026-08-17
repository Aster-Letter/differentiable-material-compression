"""Read-only structural validation for one PerformanceEvidenceV1 map."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import traceback

import unreal


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[4]
PROJECT_CONTENT = SCRIPT_PATH.parents[1]
EVIDENCE_ROOT = REPO_ROOT / "outputs/analysis/ue-runtime-evidence-v1"
CONTRACT_PATH = EVIDENCE_ROOT / "measurement_contract.json"
VARIANT_ID = os.environ.get("UE_RUNTIME_VARIANT", "").strip()
REPORT_PATH = EVIDENCE_ROOT / "performance_maps_validation" / f"{VARIANT_ID or 'missing_variant'}.json"
EXPECTED_CONTRACT_SHA256 = "652fd3761a3d8817e7f026eb067206098a12ad565c6677ab15b13ce679433bb2"
DEST_ROOT = "/Game/CGCompression/PerformanceEvidenceV1/Maps/Single"
EXPECTED_TARGET_TRANSFORMS = {
    "lantern_fixed_c4": {
        "location": [0.0, -465.0, -1.218291281562573],
        "rotation": [0.0, 0.0, -0.0],
        "scale": [0.06624008288809774, 0.06624008288809774, 0.06624008288809774],
    },
    "scifihelmet_capacity_cost": {
        "location": [0.0, 0.0, 0.0],
        "rotation": [0.0, 0.0, -0.0],
        "scale": [1.0, 1.0, 1.0],
    },
}
EXPECTED_MESHES = {
    "lantern_fixed_c4": "/Game/CGCompression/C4RenderAblation20k/Meshes/Lantern/Lantern/StaticMeshes/Lantern.Lantern",
    "scifihelmet_capacity_cost": "/Game/Imported/SciFiHelmet/SciFiHelmet/StaticMeshes/SciFiHelmet.SciFiHelmet",
}
ALLOWED_STATIC_LABELS = {"Floor", "SM_SkySphere"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _transform(actor: unreal.Actor) -> dict:
    value = actor.get_actor_transform()
    rotation = value.rotation.rotator()
    return {
        "location": [float(value.translation.x), float(value.translation.y), float(value.translation.z)],
        "rotation": [float(rotation.pitch), float(rotation.yaw), float(rotation.roll)],
        "scale": [float(value.scale3d.x), float(value.scale3d.y), float(value.scale3d.z)],
    }


def _write(value: dict) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate() -> None:
    report = {
        "schema_version": 1,
        "status": "started",
        "variant_id": VARIANT_ID,
        "read_only": True,
        "formal_holdout_accessed": False,
        "map_saved": False,
    }
    try:
        if not VARIANT_ID:
            raise RuntimeError("UE_RUNTIME_VARIANT is required")
        contract_hash = _sha256(CONTRACT_PATH)
        if contract_hash != EXPECTED_CONTRACT_SHA256:
            raise RuntimeError(f"contract hash mismatch: {contract_hash}")
        contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        matches = [
            (panel_id, variant)
            for panel_id, panel in contract["panels"].items()
            for variant in panel["variants"]
            if variant["id"] == VARIANT_ID
        ]
        if len(matches) != 1:
            raise RuntimeError(f"expected one contract variant, got {len(matches)}")
        panel_id, variant = matches[0]
        map_asset = f"{DEST_ROOT}/{VARIANT_ID}"
        if not unreal.EditorAssetLibrary.does_asset_exist(map_asset):
            raise RuntimeError(f"map asset missing: {map_asset}")
        unreal.EditorLoadingAndSavingUtils.load_map(map_asset)
        actors = list(unreal.get_editor_subsystem(unreal.EditorActorSubsystem).get_all_level_actors())
        target_label = f"PerformanceEvidence_{VARIANT_ID}"
        targets = [actor for actor in actors if actor.get_actor_label() == target_label]
        if len(targets) != 1 or not isinstance(targets[0], unreal.StaticMeshActor):
            raise RuntimeError(f"expected exactly one static target actor: {target_label}")
        target = targets[0]
        component = target.static_mesh_component
        material = component.get_material(0)
        mesh = component.static_mesh
        expected_material = f"{variant['material']}.{variant['material'].rsplit('/', 1)[-1]}"
        actual_material = material.get_path_name() if material else None
        actual_mesh = mesh.get_path_name() if mesh else None
        transform = _transform(target)
        unexpected_static = sorted(
            actor.get_actor_label()
            for actor in actors
            if isinstance(actor, unreal.StaticMeshActor)
            and actor is not target
            and actor.get_actor_label() not in ALLOWED_STATIC_LABELS
        )
        checks = {
            "contract_hash_matches": contract_hash == EXPECTED_CONTRACT_SHA256,
            "material_matches": actual_material == expected_material,
            "mesh_matches": actual_mesh == EXPECTED_MESHES[panel_id],
            "transform_matches": transform == EXPECTED_TARGET_TRANSFORMS[panel_id],
            "no_unexpected_static_mesh_actors": not unexpected_static,
        }
        if not all(checks.values()):
            raise RuntimeError(f"structural checks failed: {checks}")
        map_file = PROJECT_CONTENT / "CGCompression/PerformanceEvidenceV1/Maps/Single" / f"{VARIANT_ID}.umap"
        report.update(
            {
                "status": "complete",
                "contract_sha256": contract_hash,
                "panel": panel_id,
                "map_asset": map_asset,
                "map_file_sha256": _sha256(map_file),
                "actor_count": len(actors),
                "target_label": target_label,
                "material": actual_material,
                "mesh": actual_mesh,
                "transform": transform,
                "unexpected_static_mesh_actors": unexpected_static,
                "checks": checks,
                "map_check": "not_run; structural validation only",
            }
        )
        _write(report)
        unreal.log(f"PerformanceEvidenceV1 map validated: {VARIANT_ID}")
    except Exception as exc:
        report.update({"status": "failed", "error": str(exc), "traceback": traceback.format_exc()})
        _write(report)
        unreal.log_error(f"PerformanceEvidenceV1 map validation failed: {exc}")
        raise


validate()
