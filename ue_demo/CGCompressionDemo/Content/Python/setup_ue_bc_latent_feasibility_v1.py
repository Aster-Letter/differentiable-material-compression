"""Create isolated BC7 copies of the frozen Lantern C4 latent endpoints."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import traceback

import unreal


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[4]
PROJECT_CONTENT = SCRIPT_PATH.parents[1]
CONFIG_PATH = REPO_ROOT / "configs/eval/ue_bc_latent_feasibility_v1.json"
EVIDENCE_ROOT = REPO_ROOT / "outputs/analysis/ue-bc-latent-feasibility-v1"
REPORT_PATH = EVIDENCE_ROOT / "ue_setup_report.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _uasset_path(asset_path: str) -> Path:
    if not asset_path.startswith("/Game/"):
        raise RuntimeError(f"asset is outside /Game: {asset_path}")
    return PROJECT_CONTENT / f"{asset_path.removeprefix('/Game/')}.uasset"


def _umap_path(map_path: str) -> Path:
    if not map_path.startswith("/Game/"):
        raise RuntimeError(f"map is outside /Game: {map_path}")
    return PROJECT_CONTENT / f"{map_path.removeprefix('/Game/')}.umap"


def _write(value: dict) -> None:
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _verify_source(variant: dict) -> None:
    for kind in ("texture", "material"):
        asset = variant[f"source_{kind}"]
        path = _uasset_path(asset)
        expected = variant[f"source_{kind}_uasset_sha256"]
        if not path.is_file() or _sha256(path) != expected:
            raise RuntimeError(f"source {kind} hash mismatch: {asset}")


def _duplicate(source_path: str, destination_path: str, expected_type):
    if not unreal.EditorAssetLibrary.does_asset_exist(destination_path):
        if not unreal.EditorAssetLibrary.duplicate_asset(source_path, destination_path):
            raise RuntimeError(f"could not duplicate {source_path} to {destination_path}")
    asset = unreal.EditorAssetLibrary.load_asset(destination_path)
    if not isinstance(asset, expected_type):
        raise RuntimeError(f"unexpected duplicated asset type: {destination_path}")
    return asset


def _prepare_map(variant: dict, actor_subsystem: unreal.EditorActorSubsystem) -> dict:
    source_map = variant["source_map"]
    destination_map = variant["destination_map"]
    source_hash_before = _sha256(_umap_path(source_map))
    created = False
    if not unreal.EditorAssetLibrary.does_asset_exist(destination_map):
        unreal.EditorLoadingAndSavingUtils.load_map(source_map)
        if not unreal.EditorAssetLibrary.duplicate_asset(source_map, destination_map):
            raise RuntimeError(f"could not duplicate {source_map} to {destination_map}")
        created = True
    unreal.EditorLoadingAndSavingUtils.load_map(destination_map)
    actors = list(actor_subsystem.get_all_level_actors())
    labels = {variant["source_actor_label"], variant["destination_actor_label"]}
    targets = [actor for actor in actors if actor.get_actor_label() in labels]
    if len(targets) != 1 or not isinstance(targets[0], unreal.StaticMeshActor):
        raise RuntimeError(
            f"expected one isolated static mesh actor in {destination_map}, got {len(targets)}"
        )
    material = unreal.load_asset(variant["destination_material"])
    if not isinstance(material, unreal.MaterialInterface):
        raise RuntimeError(f"invalid BC7 material: {variant['destination_material']}")
    target = targets[0]
    target.static_mesh_component.set_material(0, material)
    target.set_actor_label(variant["destination_actor_label"])
    if not unreal.EditorAssetLibrary.save_asset(destination_map, only_if_is_dirty=False):
        raise RuntimeError(f"could not save isolated BC7 map: {destination_map}")
    source_hash_after = _sha256(_umap_path(source_map))
    if source_hash_after != source_hash_before:
        raise RuntimeError(f"source map changed while preparing {destination_map}")
    return {
        "path": destination_map,
        "created": created,
        "actor_label": target.get_actor_label(),
        "material": material.get_path_name(),
        "umap_sha256": _sha256(_umap_path(destination_map)),
        "source_map_uasset_sha256_before": source_hash_before,
        "source_map_uasset_sha256_after": source_hash_after,
    }


def _safe_call(obj: object, method: str, *args: object) -> dict:
    fn = getattr(obj, method, None)
    if fn is None:
        return {"available": False, "error": "method not exposed"}
    try:
        value = fn(*args)
        if not isinstance(value, (bool, int, float, str, list, dict, type(None))):
            value = str(value)
        return {"available": True, "value": value}
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def setup() -> None:
    report = {
        "schema_version": 1,
        "status": "started",
        "formal_holdout_accessed": False,
        "source_assets_modified": False,
        "config_sha256": _sha256(CONFIG_PATH),
        "variants": [],
    }
    try:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        if config.get("frozen") is not True or config.get("training_allowed") is not False:
            raise RuntimeError("BC latent contract is not frozen or permits training")
        bc7 = getattr(unreal.TextureCompressionSettings, "TC_BC7", None)
        if bc7 is None:
            available = sorted(
                name for name in dir(unreal.TextureCompressionSettings) if name.startswith("TC_")
            )
            raise RuntimeError(f"TC_BC7 is unavailable; exposed values: {available}")
        actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        if actor_subsystem is None:
            raise RuntimeError("EditorActorSubsystem is unavailable during setup")
        for variant in config["variants"]:
            _verify_source(variant)
            texture = _duplicate(
                variant["source_texture"],
                variant["destination_texture"],
                unreal.Texture2D,
            )
            texture.set_editor_property("srgb", False)
            texture.set_editor_property("compression_settings", bc7)
            texture.set_editor_property("compression_no_alpha", False)
            texture.set_editor_property("mip_gen_settings", unreal.TextureMipGenSettings.TMGS_SIMPLE_AVERAGE)
            texture.set_editor_property("never_stream", False)
            texture.set_editor_property("virtual_texture_streaming", False)
            texture.modify()
            if not unreal.EditorAssetLibrary.save_loaded_asset(texture, only_if_is_dirty=False):
                raise RuntimeError(f"could not save BC7 texture: {variant['destination_texture']}")

            material = _duplicate(
                variant["source_material"],
                variant["destination_material"],
                unreal.MaterialInstanceConstant,
            )
            unreal.MaterialEditingLibrary.set_material_instance_texture_parameter_value(
                material, "LatentRGBA", texture
            )
            unreal.MaterialEditingLibrary.update_material_instance(material)
            material.modify()
            if not unreal.EditorAssetLibrary.save_loaded_asset(material, only_if_is_dirty=False):
                raise RuntimeError(f"could not save BC7 material: {variant['destination_material']}")

            map_record = _prepare_map(variant, actor_subsystem)

            stats = unreal.MaterialEditingLibrary.get_statistics(material)
            mip_count = getattr(unreal, "TextureMipCount", None)
            report["variants"].append(
                {
                    "id": variant["id"],
                    "source_variant_id": variant["source_variant_id"],
                    "texture": texture.get_path_name(),
                    "material": material.get_path_name(),
                    "compression_settings": str(texture.get_editor_property("compression_settings")),
                    "pixel_format": _safe_call(texture, "get_pixel_format"),
                    "all_mips_memory_bytes": (
                        _safe_call(texture, "calc_texture_memory_size_enum", mip_count.TMC_ALL_MIPS)
                        if mip_count is not None
                        else {"available": False, "error": "TextureMipCount not exposed"}
                    ),
                    "pixel_shader_instructions": int(
                        stats.get_editor_property("num_pixel_shader_instructions")
                    ),
                    "pixel_texture_samples": int(
                        stats.get_editor_property("num_pixel_texture_samples")
                    ),
                    "samplers": int(stats.get_editor_property("num_samplers")),
                    "texture_uasset_sha256": _sha256(_uasset_path(variant["destination_texture"])),
                    "material_uasset_sha256": _sha256(_uasset_path(variant["destination_material"])),
                    "map": map_record,
                }
            )
            _verify_source(variant)
        report["status"] = "complete"
        _write(report)
        unreal.log(f"BC latent feasibility setup complete: {len(report['variants'])} variants")
    except Exception as exc:
        report.update({"status": "failed", "error": str(exc), "traceback": traceback.format_exc()})
        _write(report)
        unreal.log_error(f"BC latent feasibility setup failed: {exc}")
        raise


setup()
