"""Create an isolated Lantern Source/Raw/20k/160k Unreal preview map."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import traceback

import unreal


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[4]
DEPLOYMENT = REPO_ROOT / "outputs/deployment/c4_render_ablation_lantern_160k_v1/ue_preview_job_37824"
MANIFEST_PATH = DEPLOYMENT / "preview_manifest.json"
REPORT_PATH = DEPLOYMENT / "ue_setup_report.json"
EVIDENCE_PATH = DEPLOYMENT / "ue_evidence/Lantern/material_render_160k/latent_rgba8_readback.png"
EXPECTED_MANIFEST_SHA256 = "ddfbb57cd454cc506d11861ad632c59f4c298c7b824e3eb6b630f4808f4510bd"

ASSET_ROOT = "/Game/CGCompression/C4LanternRender160k"
SOURCE_MAP = "/Game/CGCompression/C4RenderAblation20k/Maps/C4_Render_Ablation_20k"
PREVIEW_MAP = f"{ASSET_ROOT}/Maps/C4_Lantern_Source_Raw_20k_160k"
MASTER = "/Game/CGCompression/C4RenderAblation20k/Materials/M_C4_Render_Ablation_Master"
MESH = "/Game/CGCompression/C4RenderAblation20k/Meshes/Lantern/Lantern/StaticMeshes/Lantern"
RAW_MATERIAL = "/Game/CGCompression/C4RenderAblation20k/Materials/MI_Lantern_RAW_Q4"
MATERIAL_20K = "/Game/CGCompression/C4RenderAblation20k/Materials/MI_Lantern_MATERIAL_RENDER"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_report(value: dict) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def validate_package() -> dict:
    if sha256(MANIFEST_PATH) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("Lantern 160k UE manifest hash mismatch")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "exported_ue_setup_pending"
        or manifest.get("formal_holdout_accessed") is not False
        or manifest.get("endpoint_step") != 160000
        or manifest.get("arm") != "material_render"
        or manifest.get("ue", {}).get("asset_root") != ASSET_ROOT
        or manifest.get("ue", {}).get("preview_map") != PREVIEW_MAP
    ):
        raise RuntimeError("Lantern 160k UE manifest contract mismatch")
    package = DEPLOYMENT / manifest["package_directory"]
    for filename, expected in manifest["generated_files"].items():
        if sha256(package / filename) != expected:
            raise RuntimeError(f"Lantern 160k package hash mismatch: {filename}")
    for asset in (SOURCE_MAP, MASTER, MESH, RAW_MATERIAL, MATERIAL_20K):
        if not unreal.EditorAssetLibrary.does_asset_exist(asset):
            raise RuntimeError(f"required existing UE asset missing: {asset}")
    return manifest


def ensure_directories() -> None:
    for path in (f"{ASSET_ROOT}/Maps", f"{ASSET_ROOT}/Materials", f"{ASSET_ROOT}/Textures"):
        if not unreal.EditorAssetLibrary.does_directory_exist(path):
            if not unreal.EditorAssetLibrary.make_directory(path):
                raise RuntimeError(f"could not create UE directory: {path}")


def import_texture(source: Path, asset_path: str) -> unreal.Texture2D:
    folder, name = asset_path.rsplit("/", 1)
    task = unreal.AssetImportTask()
    task.set_editor_property("filename", source.as_posix())
    task.set_editor_property("destination_path", folder)
    task.set_editor_property("destination_name", name)
    task.set_editor_property("automated", True)
    task.set_editor_property("replace_existing", True)
    task.set_editor_property("save", False)
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    texture = unreal.EditorAssetLibrary.load_asset(asset_path)
    if not isinstance(texture, unreal.Texture2D):
        raise RuntimeError(f"latent texture import failed: {asset_path}")
    texture.set_editor_property("srgb", False)
    texture.set_editor_property("compression_settings", unreal.TextureCompressionSettings.TC_VECTOR_DISPLACEMENTMAP)
    texture.set_editor_property("filter", unreal.TextureFilter.TF_DEFAULT)
    texture.set_editor_property("address_x", unreal.TextureAddress.TA_WRAP)
    texture.set_editor_property("address_y", unreal.TextureAddress.TA_WRAP)
    texture.set_editor_property("mip_gen_settings", unreal.TextureMipGenSettings.TMGS_SIMPLE_AVERAGE)
    texture.set_editor_property("never_stream", False)
    texture.set_editor_property("virtual_texture_streaming", False)
    texture.modify()
    if not unreal.EditorAssetLibrary.save_loaded_asset(texture, only_if_is_dirty=False):
        raise RuntimeError("could not save 160k latent texture")
    return texture


def create_instance(master, texture, asset_path: str, parameters: dict):
    folder, name = asset_path.rsplit("/", 1)
    instance = (
        unreal.EditorAssetLibrary.load_asset(asset_path)
        if unreal.EditorAssetLibrary.does_asset_exist(asset_path)
        else unreal.AssetToolsHelpers.get_asset_tools().create_asset(
            name, folder, unreal.MaterialInstanceConstant, unreal.MaterialInstanceConstantFactoryNew()
        )
    )
    if not isinstance(instance, unreal.MaterialInstanceConstant):
        raise RuntimeError("invalid 160k material instance")
    library = unreal.MaterialEditingLibrary
    library.set_material_instance_parent(instance, master)
    library.set_material_instance_texture_parameter_value(instance, "LatentRGBA", texture)
    for parameter, values in parameters["vector_parameters"].items():
        library.set_material_instance_vector_parameter_value(
            instance, parameter, unreal.LinearColor(*[float(value) for value in values])
        )
    for parameter, value in parameters["scalar_parameters"].items():
        library.set_material_instance_scalar_parameter_value(instance, parameter, float(value))
    library.update_material_instance(instance)
    instance.modify()
    if not unreal.EditorAssetLibrary.save_loaded_asset(instance, only_if_is_dirty=False):
        raise RuntimeError("could not save 160k material instance")
    return instance


def export_readback(texture: unreal.Texture2D) -> dict:
    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    task = unreal.AssetExportTask()
    task.set_editor_property("object", texture)
    task.set_editor_property("filename", EVIDENCE_PATH.as_posix())
    task.set_editor_property("automated", True)
    task.set_editor_property("prompt", False)
    task.set_editor_property("replace_identical", True)
    task.set_editor_property("write_empty_files", False)
    task.set_editor_property("exporter", unreal.TextureExporterPNG())
    success = bool(unreal.Exporter.run_asset_export_task(task))
    return {
        "path": EVIDENCE_PATH.as_posix(),
        "export_success": success,
        "exists": EVIDENCE_PATH.is_file(),
        "sha256": sha256(EVIDENCE_PATH) if EVIDENCE_PATH.is_file() else None,
    }


def populate_map(material160) -> dict:
    if not unreal.EditorAssetLibrary.does_asset_exist(PREVIEW_MAP):
        if not unreal.EditorAssetLibrary.duplicate_asset(SOURCE_MAP, PREVIEW_MAP):
            raise RuntimeError("could not duplicate the verified 20k preview map")
        unreal.EditorAssetLibrary.save_asset(PREVIEW_MAP, only_if_is_dirty=False)
        return {"fresh_process_required": True, "actors": {}}
    level_editor = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    if not level_editor.load_level(PREVIEW_MAP):
        raise RuntimeError(f"could not load preview map: {PREVIEW_MAP}")
    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    existing = {actor.get_actor_label(): actor for actor in actor_subsystem.get_all_level_actors()}
    keep = {
        "source_reference": "C4_Lantern_SOURCE_REFERENCE",
        "raw_q4": "C4_Lantern_RAW_Q4",
        "material_render_20k": "C4_Lantern_MATERIAL_RENDER",
    }
    for label, actor in existing.items():
        if label.startswith("C4_"):
            visible = label in keep.values()
            actor.set_is_temporarily_hidden_in_editor(not visible)
            actor.set_actor_hidden_in_game(not visible)
    actors = {}
    columns = {
        "source_reference": -465.0,
        "raw_q4": -155.0,
        "material_render_20k": 155.0,
        "material_render_160k": 465.0,
    }
    mesh = unreal.EditorAssetLibrary.load_asset(MESH)
    if not isinstance(mesh, unreal.StaticMesh):
        raise RuntimeError("Lantern mesh could not be loaded")
    template = existing.get(keep["source_reference"])
    if not isinstance(template, unreal.StaticMeshActor):
        raise RuntimeError("Lantern source actor missing from cloned map")
    z = float(template.get_actor_location().z)
    scale = template.get_actor_scale3d()
    materials = {
        "raw_q4": unreal.EditorAssetLibrary.load_asset(RAW_MATERIAL),
        "material_render_20k": unreal.EditorAssetLibrary.load_asset(MATERIAL_20K),
        "material_render_160k": material160,
    }
    for endpoint, y in columns.items():
        label = (
            keep[endpoint]
            if endpoint in keep
            else "C4_Lantern_MATERIAL_RENDER_160K"
        )
        actor = existing.get(label)
        if actor is None:
            actor = actor_subsystem.spawn_actor_from_class(
                unreal.StaticMeshActor, unreal.Vector(0.0, y, z)
            )
        if not isinstance(actor, unreal.StaticMeshActor):
            raise RuntimeError(f"invalid preview actor: {label}")
        actor.set_actor_label(label, mark_dirty=True)
        actor.set_is_temporarily_hidden_in_editor(False)
        actor.set_actor_hidden_in_game(False)
        actor.set_actor_location(unreal.Vector(0.0, y, z), False, False)
        actor.set_actor_scale3d(scale)
        component = actor.get_component_by_class(unreal.StaticMeshComponent)
        component.set_static_mesh(mesh)
        slots = len(mesh.get_editor_property("static_materials"))
        if endpoint == "source_reference":
            for slot in range(slots):
                component.set_material(slot, None)
            material_path = "source_gltf_default_no_emissive"
        else:
            for slot in range(slots):
                component.set_material(slot, materials[endpoint])
            material_path = materials[endpoint].get_path_name()
        actor.modify()
        actors[endpoint] = {
            "label": label,
            "location": [0.0, y, z],
            "material": material_path,
        }
    if not unreal.EditorAssetLibrary.save_asset(PREVIEW_MAP, only_if_is_dirty=False):
        raise RuntimeError("could not save Lantern 160k preview map")
    return {"fresh_process_required": False, "actors": actors}


def setup() -> None:
    report = {
        "schema_version": 1,
        "status": "started",
        "formal_holdout_accessed": False,
        "source_20k_map_preserved": True,
        "preview_map": PREVIEW_MAP,
    }
    try:
        manifest = validate_package()
        ensure_directories()
        package = DEPLOYMENT / manifest["package_directory"]
        texture = import_texture(package / "latent_rgba8.png", manifest["ue"]["latent_texture"])
        master = unreal.EditorAssetLibrary.load_asset(MASTER)
        if not isinstance(master, unreal.Material):
            raise RuntimeError("shared affine master material missing")
        parameters = json.loads((package / "material_parameters.json").read_text(encoding="utf-8"))
        material160 = create_instance(master, texture, manifest["ue"]["material_instance"], parameters)
        report["latent_texture"] = texture.get_path_name()
        report["material_instance"] = material160.get_path_name()
        report["readback"] = export_readback(texture)
        report["scene"] = populate_map(material160)
        report["status"] = (
            "map_duplicated_fresh_process_required"
            if report["scene"]["fresh_process_required"]
            else "complete_ready_for_manual_preview"
        )
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        write_report(report)
        raise
    write_report(report)
    unreal.log(f"Lantern 160k preview setup status: {report['status']}")


if globals().get("C4_LANTERN_RENDER_160K_PREVIEW_AUTORUN", True):
    setup()
