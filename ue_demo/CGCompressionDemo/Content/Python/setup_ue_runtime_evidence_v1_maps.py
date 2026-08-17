"""Create isolated single-object maps under PerformanceEvidenceV1 only."""

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
REPORT_PATH = (
    EVIDENCE_ROOT
    / "performance_maps_setup"
    / f"{VARIANT_ID or 'missing_variant'}.json"
)
EXPECTED_CONTRACT_SHA256 = (
    "652fd3761a3d8817e7f026eb067206098a12ad565c6677ab15b13ce679433bb2"
)
DEST_ROOT = "/Game/CGCompression/PerformanceEvidenceV1/Maps/Single"

PANEL_SOURCE = {
    "lantern_fixed_c4": {
        "source_map": "/Game/CGCompression/C4LanternRender160k/Maps/C4_Lantern_Source_Raw_20k_160k",
        "target_label": "C4_Lantern_SOURCE_REFERENCE",
    },
    "scifihelmet_capacity_cost": {
        "source_map": "/Game/CGCompression/Maps/MaterialLab",
        "target_label": "Helmet_Reference",
    },
}
KEEP_STATIC_LABELS = {"Floor", "SM_SkySphere"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_report(value: dict) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _transform_record(actor: unreal.Actor) -> dict:
    transform = actor.get_actor_transform()
    rotation = transform.rotation.rotator()
    return {
        "location": [
            float(transform.translation.x),
            float(transform.translation.y),
            float(transform.translation.z),
        ],
        "rotation": [float(rotation.pitch), float(rotation.yaw), float(rotation.roll)],
        "scale": [
            float(transform.scale3d.x),
            float(transform.scale3d.y),
            float(transform.scale3d.z),
        ],
    }


def _create_single_map(
    actor_subsystem: unreal.EditorActorSubsystem,
    source_map: str,
    target_label: str,
    variant: dict,
) -> dict:
    dest_map = f"{DEST_ROOT}/{variant['id']}"
    if unreal.EditorAssetLibrary.does_asset_exist(dest_map):
        raise RuntimeError(f"destination already exists; refusing overwrite: {dest_map}")
    unreal.EditorLoadingAndSavingUtils.load_map(source_map)
    if not unreal.EditorAssetLibrary.duplicate_asset(source_map, dest_map):
        raise RuntimeError(f"failed to duplicate {source_map} to {dest_map}")
    unreal.EditorLoadingAndSavingUtils.load_map(dest_map)
    actors = list(actor_subsystem.get_all_level_actors())
    target = next(
        (actor for actor in actors if actor.get_actor_label() == target_label), None
    )
    if not isinstance(target, unreal.StaticMeshActor):
        raise RuntimeError(f"target actor missing in {dest_map}: {target_label}")
    target_transform = _transform_record(target)
    removed = []
    for actor in actors:
        if not isinstance(actor, unreal.StaticMeshActor) or actor is target:
            continue
        if actor.get_actor_label() in KEEP_STATIC_LABELS:
            continue
        removed.append(actor.get_actor_label())
        actor_subsystem.destroy_actor(actor)
    material = unreal.load_asset(variant["material"])
    if not isinstance(material, unreal.MaterialInterface):
        raise RuntimeError(f"invalid material: {variant['material']}")
    target.static_mesh_component.set_material(0, material)
    target.set_actor_label(f"PerformanceEvidence_{variant['id']}")
    if not unreal.EditorAssetLibrary.save_asset(dest_map, only_if_is_dirty=False):
        raise RuntimeError(f"failed to save isolated map: {dest_map}")
    remaining = list(actor_subsystem.get_all_level_actors())
    target_after = next(
        actor
        for actor in remaining
        if actor.get_actor_label() == f"PerformanceEvidence_{variant['id']}"
    )
    if _transform_record(target_after) != target_transform:
        raise RuntimeError(f"target transform changed while isolating {variant['id']}")
    return {
        "variant_id": variant["id"],
        "source_map": source_map,
        "destination_map": dest_map,
        "material": material.get_path_name(),
        "target_transform": target_transform,
        "removed_static_mesh_actors": sorted(removed),
        "remaining_actor_count": len(remaining),
        "source_map_saved": False,
        "map_check": "pending_separate_process",
    }


def setup() -> None:
    report = {
        "schema_version": 1,
        "status": "started",
        "formal_holdout_accessed": False,
        "source_maps_saved": False,
        "destination_root": DEST_ROOT,
        "maps": [],
    }
    try:
        if not VARIANT_ID:
            raise RuntimeError("UE_RUNTIME_VARIANT is required")
        contract_hash = _sha256(CONTRACT_PATH)
        if contract_hash != EXPECTED_CONTRACT_SHA256:
            raise RuntimeError(f"contract hash mismatch: {contract_hash}")
        contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        matched = 0
        for panel_name, panel in contract["panels"].items():
            panel_source = PANEL_SOURCE[panel_name]
            for variant in panel["variants"]:
                if variant["id"] != VARIANT_ID:
                    continue
                report["maps"].append(
                    _create_single_map(
                        actor_subsystem,
                        panel_source["source_map"],
                        panel_source["target_label"],
                        variant,
                    )
                )
                matched += 1
        if matched != 1:
            raise RuntimeError(
                f"expected exactly one contract variant for {VARIANT_ID}, got {matched}"
            )
        report["status"] = "complete"
        report["contract_sha256"] = contract_hash
        report["map_count"] = len(report["maps"])
        _write_report(report)
        unreal.log(f"PerformanceEvidenceV1 maps created: {len(report['maps'])}")
    except Exception as exc:
        report.update(
            {"status": "failed", "error": str(exc), "traceback": traceback.format_exc()}
        )
        _write_report(report)
        unreal.log_error(f"PerformanceEvidenceV1 map setup failed: {exc}")
        raise


setup()
