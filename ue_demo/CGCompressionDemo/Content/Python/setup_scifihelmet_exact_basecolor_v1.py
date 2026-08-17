"""Import the strict-BaseColor U0/S/M bundle into an isolated UE preview map."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import traceback

import unreal


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[4]
BUNDLE_ROOT = REPO_ROOT / "outputs/scifihelmet_exact_basecolor_v1/ue_preview_bundle"
MANIFEST_PATH = BUNDLE_ROOT / "ue_preview_manifest.json"
REPORT_PATH = BUNDLE_ROOT / "ue_setup_report.json"

ASSET_ROOT = "/Game/CGCompression/ExactBaseColorV1"
TEXTURE_ROOT = f"{ASSET_ROOT}/Textures"
MATERIAL_ROOT = f"{ASSET_ROOT}/Materials"
MAP_ROOT = f"{ASSET_ROOT}/Maps"
SOURCE_MAP = "/Game/CGCompression/Maps/MaterialLab"
PREVIEW_MAP = f"{MAP_ROOT}/ExactBaseColorV1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_report(report: dict) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _validate_bundle() -> dict:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("experiment") != "scifihelmet_exact_basecolor_v1":
        raise RuntimeError("unexpected UE preview experiment")
    if manifest.get("selection") != "S-separated":
        raise RuntimeError("UE preview selection must remain S-separated")
    if manifest.get("preview_map") != PREVIEW_MAP:
        raise RuntimeError("preview map contract mismatch")
    for record in manifest["candidates"].values():
        for file_record in record["files"].values():
            path = BUNDLE_ROOT / file_record["name"]
            if _sha256(path) != file_record["sha256"]:
                raise RuntimeError(f"bundle hash mismatch: {path.name}")
    return manifest


def _ensure_directories() -> None:
    for path in (ASSET_ROOT, TEXTURE_ROOT, MATERIAL_ROOT, MAP_ROOT):
        if not unreal.EditorAssetLibrary.does_directory_exist(path):
            if not unreal.EditorAssetLibrary.make_directory(path):
                raise RuntimeError(f"could not create UE directory: {path}")


def _import_texture(record: dict) -> tuple[unreal.Texture2D, dict]:
    asset_path = record["ue_assets"]["texture"]
    texture = unreal.EditorAssetLibrary.load_asset(asset_path)
    if texture is None:
        task = unreal.AssetImportTask()
        task.set_editor_property("filename", (BUNDLE_ROOT / record["files"]["texture"]["name"]).as_posix())
        task.set_editor_property("destination_path", TEXTURE_ROOT)
        task.set_editor_property("destination_name", asset_path.rsplit("/", 1)[1])
        task.set_editor_property("automated", True)
        task.set_editor_property("replace_existing", False)
        task.set_editor_property("save", False)
        unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
        texture = unreal.EditorAssetLibrary.load_asset(asset_path)
    if not isinstance(texture, unreal.Texture2D):
        raise RuntimeError(f"latent import did not produce Texture2D: {asset_path}")
    texture.set_editor_property("srgb", False)
    texture.set_editor_property(
        "compression_settings", unreal.TextureCompressionSettings.TC_VECTOR_DISPLACEMENTMAP
    )
    texture.set_editor_property("filter", unreal.TextureFilter.TF_BILINEAR)
    texture.set_editor_property("address_x", unreal.TextureAddress.TA_WRAP)
    texture.set_editor_property("address_y", unreal.TextureAddress.TA_WRAP)
    texture.set_editor_property("mip_gen_settings", unreal.TextureMipGenSettings.TMGS_NO_MIPMAPS)
    texture.set_editor_property("never_stream", True)
    texture.set_editor_property("virtual_texture_streaming", False)
    texture.modify()
    if not unreal.EditorAssetLibrary.save_loaded_asset(texture, only_if_is_dirty=False):
        raise RuntimeError(f"could not save texture: {asset_path}")
    readback_path = BUNDLE_ROOT / "evidence" / f"ue_readback_{record['slug']}.png"
    readback_path.parent.mkdir(parents=True, exist_ok=True)
    export = unreal.AssetExportTask()
    export.set_editor_property("object", texture)
    export.set_editor_property("filename", readback_path.as_posix())
    export.set_editor_property("automated", True)
    export.set_editor_property("prompt", False)
    export.set_editor_property("replace_identical", True)
    export.set_editor_property("write_empty_files", False)
    export.set_editor_property("exporter", unreal.TextureExporterPNG())
    exported = bool(unreal.Exporter.run_asset_export_task(export))
    return texture, {
        "asset": texture.get_path_name(),
        "size": [int(texture.blueprint_get_size_x()), int(texture.blueprint_get_size_y())],
        "srgb": bool(texture.get_editor_property("srgb")),
        "compression_settings": str(texture.get_editor_property("compression_settings")),
        "filter": str(texture.get_editor_property("filter")),
        "address_x": str(texture.get_editor_property("address_x")),
        "address_y": str(texture.get_editor_property("address_y")),
        "mip_gen_settings": str(texture.get_editor_property("mip_gen_settings")),
        "never_stream": bool(texture.get_editor_property("never_stream")),
        "virtual_texture_streaming": bool(texture.get_editor_property("virtual_texture_streaming")),
        "readback": {
            "path": readback_path.as_posix(),
            "exported": exported,
            "exists": readback_path.is_file(),
            "sha256": _sha256(readback_path) if readback_path.is_file() else None,
        },
    }


def _custom_input(name: str) -> unreal.CustomInput:
    value = unreal.CustomInput()
    value.set_editor_property("input_name", name)
    return value


def _custom_output(name: str, output_type) -> unreal.CustomOutput:
    value = unreal.CustomOutput()
    value.set_editor_property("output_name", name)
    value.set_editor_property("output_type", output_type)
    return value


def _create_material(record: dict, texture: unreal.Texture2D) -> tuple[unreal.Material, dict]:
    asset_path = record["ue_assets"]["material"]
    asset_name = asset_path.rsplit("/", 1)[1]
    material = unreal.EditorAssetLibrary.load_asset(asset_path)
    if material is None:
        material = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
            asset_name,
            MATERIAL_ROOT,
            unreal.Material,
            unreal.MaterialFactoryNew(),
        )
    if not isinstance(material, unreal.Material):
        raise RuntimeError(f"asset is not a Material: {asset_path}")
    material.set_editor_property("material_domain", unreal.MaterialDomain.MD_SURFACE)
    material.set_editor_property("blend_mode", unreal.BlendMode.BLEND_OPAQUE)
    material.set_editor_property("two_sided", False)
    unreal.MaterialEditingLibrary.delete_all_material_expressions(material)

    uv = unreal.MaterialEditingLibrary.create_material_expression(
        material, unreal.MaterialExpressionTextureCoordinate, -1100, -100
    )
    uv.set_editor_property("coordinate_index", 0)
    sample = unreal.MaterialEditingLibrary.create_material_expression(
        material, unreal.MaterialExpressionTextureSample, -820, -100
    )
    sample.set_editor_property("texture", texture)
    sample.set_editor_property("sampler_type", unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR)
    normal_y = unreal.MaterialEditingLibrary.create_material_expression(
        material, unreal.MaterialExpressionConstant, -820, 210
    )
    normal_y.set_editor_property("r", -1.0)
    custom = unreal.MaterialEditingLibrary.create_material_expression(
        material, unreal.MaterialExpressionCustom, -350, -80
    )
    custom.set_editor_property(
        "description", f"ExactBaseColorV1 {record['candidate']}; one RGBA8 sample + 4->7 affine"
    )
    custom.set_editor_property(
        "code", (BUNDLE_ROOT / record["files"]["hlsl"]["name"]).read_text(encoding="utf-8")
    )
    custom.set_editor_property("output_type", unreal.CustomMaterialOutputType.CMOT_FLOAT3)
    custom.set_editor_property("inputs", [_custom_input("LatentRGBA"), _custom_input("NormalYSign")])
    custom.set_editor_property(
        "additional_outputs",
        [
            _custom_output("NormalTS", unreal.CustomMaterialOutputType.CMOT_FLOAT3),
            _custom_output("Roughness", unreal.CustomMaterialOutputType.CMOT_FLOAT1),
            _custom_output("Metallic", unreal.CustomMaterialOutputType.CMOT_FLOAT1),
        ],
    )
    connections = {
        "uv_to_sample": unreal.MaterialEditingLibrary.connect_material_expressions(uv, "", sample, "UVs"),
        "sample_to_decoder": unreal.MaterialEditingLibrary.connect_material_expressions(sample, "RGBA", custom, "LatentRGBA"),
        "normal_y_once": unreal.MaterialEditingLibrary.connect_material_expressions(normal_y, "", custom, "NormalYSign"),
        "basecolor": unreal.MaterialEditingLibrary.connect_material_property(custom, "", unreal.MaterialProperty.MP_BASE_COLOR),
        "normal": unreal.MaterialEditingLibrary.connect_material_property(custom, "NormalTS", unreal.MaterialProperty.MP_NORMAL),
        "roughness": unreal.MaterialEditingLibrary.connect_material_property(custom, "Roughness", unreal.MaterialProperty.MP_ROUGHNESS),
        "metallic": unreal.MaterialEditingLibrary.connect_material_property(custom, "Metallic", unreal.MaterialProperty.MP_METALLIC),
    }
    if not all(connections.values()):
        raise RuntimeError(f"material connection failed: {record['candidate']} {connections}")
    unreal.MaterialEditingLibrary.recompile_material(material)
    material.modify()
    if not unreal.EditorAssetLibrary.save_loaded_asset(material, only_if_is_dirty=False):
        raise RuntimeError(f"could not save material: {asset_path}")
    return material, {
        "asset": material.get_path_name(),
        "connections": connections,
        "expression_count": len(unreal.MaterialEditingLibrary.get_material_expressions(material)),
        "outputs": [str(value) for value in unreal.MaterialEditingLibrary.get_material_expression_output_names(custom)],
    }


def _component(actor: unreal.Actor) -> unreal.StaticMeshComponent | None:
    value = actor.get_component_by_class(unreal.StaticMeshComponent)
    return value if isinstance(value, unreal.StaticMeshComponent) else None


def _find_reference(actors: list[unreal.Actor]) -> unreal.Actor:
    for actor in actors:
        if actor.get_actor_label() == "Helmet_Reference" and _component(actor) is not None:
            return actor
    for actor in actors:
        component = _component(actor)
        mesh = component.get_editor_property("static_mesh") if component else None
        if mesh is not None and "SciFiHelmet" in mesh.get_path_name():
            return actor
    raise RuntimeError("MaterialLab has no SciFiHelmet reference actor")


def _ensure_actor(actor_subsystem, actors, reference, record, material, offset):
    label = record["ue_assets"]["actor"]
    actor = next((value for value in actors if value.get_actor_label() == label), None)
    if actor is None:
        actor = actor_subsystem.spawn_actor_from_class(
            unreal.StaticMeshActor, reference.get_actor_location(), reference.get_actor_rotation()
        )
    actor.set_actor_transform(reference.get_actor_transform(), False, False)
    actor.add_actor_world_offset(offset, False, False)
    actor.set_actor_label(label, mark_dirty=True)
    actor.set_actor_rotation(reference.get_actor_rotation(), False)
    actor.set_actor_scale3d(reference.get_actor_scale3d())
    component = _component(actor)
    reference_component = _component(reference)
    if component is None or reference_component is None:
        raise RuntimeError(f"preview actor lacks StaticMeshComponent: {label}")
    component.set_static_mesh(reference_component.get_editor_property("static_mesh"))
    component.set_material(0, material)
    actor.modify()
    return actor


def _ensure_camera(actor_subsystem, actor, slug):
    label = f"Camera_ExactBC_{slug}"
    origin, extent = actor.get_actor_bounds(False, True)
    radius = max(float(extent.x), float(extent.y), float(extent.z))
    location = origin + unreal.Vector(-2.65, 1.75, 0.62) * radius
    target = origin + unreal.Vector(0.0, 0.05, 0.02) * radius
    rotation = unreal.MathLibrary.find_look_at_rotation(location, target)
    camera = next(
        (
            value
            for value in actor_subsystem.get_all_level_actors()
            if isinstance(value, unreal.CameraActor) and value.get_actor_label() == label
        ),
        None,
    )
    if camera is None:
        camera = actor_subsystem.spawn_actor_from_class(unreal.CameraActor, location, rotation)
    camera.set_actor_label(label, mark_dirty=True)
    camera.set_actor_location(location, False, False)
    camera.set_actor_rotation(rotation, False)
    camera_component = camera.get_component_by_class(unreal.CameraComponent)
    if not isinstance(camera_component, unreal.CameraComponent):
        raise RuntimeError(f"preview camera has no CameraComponent: {label}")
    camera_component.set_editor_property("field_of_view", 27.0)
    camera.modify()
    return camera


def _ensure_preview_map(manifest: dict, materials: dict[str, unreal.Material]) -> dict:
    if not unreal.EditorAssetLibrary.does_asset_exist(PREVIEW_MAP):
        if not unreal.EditorAssetLibrary.duplicate_asset(SOURCE_MAP, PREVIEW_MAP):
            raise RuntimeError("could not duplicate MaterialLab into ExactBaseColorV1 map")
        if not unreal.EditorAssetLibrary.save_asset(PREVIEW_MAP, only_if_is_dirty=False):
            raise RuntimeError("could not save duplicated preview map")
        return {"map": PREVIEW_MAP, "duplicated": True, "second_pass_required": True}

    level_editor = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    if not level_editor.load_level(PREVIEW_MAP):
        raise RuntimeError(f"could not load preview map: {PREVIEW_MAP}")
    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actors = list(actor_subsystem.get_all_level_actors())
    reference = _find_reference(actors)
    extent = reference.get_actor_bounds(False, True)[1]
    separation = max(float(extent.x), float(extent.y), float(extent.z)) * 2.8
    preview_actors = {}
    cameras = []
    for index, (candidate, record) in enumerate(manifest["candidates"].items(), start=1):
        actor = _ensure_actor(
            actor_subsystem,
            actors,
            reference,
            record,
            materials[candidate],
            unreal.Vector(0.0, separation * index, 0.0),
        )
        preview_actors[candidate] = actor.get_actor_label()
        cameras.append(_ensure_camera(actor_subsystem, actor, record["slug"]).get_actor_label())
    if not unreal.EditorAssetLibrary.save_asset(PREVIEW_MAP, only_if_is_dirty=False):
        raise RuntimeError("could not save ExactBaseColorV1 preview map")
    return {
        "map": PREVIEW_MAP,
        "source_map": SOURCE_MAP,
        "source_map_preserved": True,
        "duplicated": False,
        "second_pass_required": False,
        "reference": reference.get_actor_label(),
        "actors": preview_actors,
        "cameras": cameras,
        "separation_cm": separation,
    }


def setup() -> None:
    report = {
        "schema_version": 1,
        "status": "started",
        "formal_holdout_accessed": False,
        "source_map_preserved": True,
        "source_map_saved": False,
    }
    try:
        manifest = _validate_bundle()
        report["selection"] = manifest["selection"]
        _ensure_directories()
        materials = {}
        report["candidates"] = {}
        for candidate, record in manifest["candidates"].items():
            texture, texture_report = _import_texture(record)
            material, material_report = _create_material(record, texture)
            materials[candidate] = material
            report["candidates"][candidate] = {
                "texture": texture_report,
                "material": material_report,
            }
        report["preview"] = _ensure_preview_map(manifest, materials)
        if not unreal.EditorLoadingAndSavingUtils.save_dirty_packages(
            save_map_packages=True, save_content_packages=True
        ):
            raise RuntimeError("one or more ExactBaseColorV1 packages failed to save")
        report["status"] = (
            "map_duplicated_second_pass_required"
            if report["preview"]["second_pass_required"]
            else "complete"
        )
        _write_report(report)
        unreal.log(f"ExactBaseColorV1 setup status: {report['status']}")
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        _write_report(report)
        unreal.log_error(f"ExactBaseColorV1 setup failed: {exc}")
        raise


setup()
