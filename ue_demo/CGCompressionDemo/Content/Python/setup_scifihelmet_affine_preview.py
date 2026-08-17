"""Create a shared-affine UE material and eight isolated preview actors."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import traceback

import unreal


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[4]
DEPLOYMENT_ROOT = (
    REPO_ROOT
    / "outputs/scifihelmet_c4_affine_v1/ue_preview/a874ad-progress-r3-enhanced"
)
PREVIEW_MANIFEST_PATH = DEPLOYMENT_ROOT / "preview_manifest.json"
REPORT_PATH = DEPLOYMENT_ROOT / "ue_setup_report.json"
EVIDENCE_ROOT = DEPLOYMENT_ROOT / "ue_evidence"

EXPECTED_PREVIEW_MANIFEST_SHA256 = (
    "07251d55276a885eb928bf13d97d2fc629c4beb5345631c9c2d8d4ceed7d0f5c"
)
EXPECTED_CANDIDATES = (
    "p0_safe",
    "p0_safe_repair",
    "p0_enhanced_chroma4",
    "p0_enhanced_chroma8",
    "l0_s040k",
    "l0_s080k",
    "l1_tv_r005_s040k",
    "l2_cube_r005_s040k",
)

ASSET_ROOT = "/Game/CGCompression/AffinePreview"
SOURCE_MAP = "/Game/CGCompression/Maps/MaterialLab"
PREVIEW_MAP = f"{ASSET_ROOT}/Maps/MaterialLab_Affine_Progress"
MASTER_MATERIAL = f"{ASSET_ROOT}/Materials/M_SciFiHelmet_Affine_Master"

CANDIDATE_LABELS = {
    "p0_safe": "Helmet_Affine_P0_SAFE",
    "p0_safe_repair": "Helmet_Affine_P0_SAFE_REPAIR",
    "p0_enhanced_chroma4": "Helmet_Affine_P0_ENH_CHROMA4",
    "p0_enhanced_chroma8": "Helmet_Affine_P0_ENH_CHROMA8",
    "l0_s040k": "Helmet_Affine_L0_S040K",
    "l0_s080k": "Helmet_Affine_L0_S080K",
    "l1_tv_r005_s040k": "Helmet_Affine_L1_TV_R005_S040K",
    "l2_cube_r005_s040k": "Helmet_Affine_L2_CUBE_R005_S040K",
}


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


def _validate_package() -> tuple[dict, dict[str, dict]]:
    if _sha256(PREVIEW_MANIFEST_PATH) != EXPECTED_PREVIEW_MANIFEST_SHA256:
        raise RuntimeError("affine preview manifest hash mismatch")
    preview = json.loads(PREVIEW_MANIFEST_PATH.read_text(encoding="utf-8"))
    if (
        preview.get("status") != "exported_ue_setup_pending"
        or preview.get("formal_holdout_accessed") is not False
        or set(preview.get("candidates", {})) != set(EXPECTED_CANDIDATES)
        or preview.get("ue_assets", {}).get("asset_root") != ASSET_ROOT
        or preview.get("ue_assets", {}).get("source_map") != SOURCE_MAP
        or preview.get("ue_assets", {}).get("preview_map") != PREVIEW_MAP
        or preview.get("ue_assets", {}).get("master_material") != MASTER_MATERIAL
    ):
        raise RuntimeError("affine preview manifest contract mismatch")
    packages = {}
    for candidate in EXPECTED_CANDIDATES:
        record = preview["candidates"][candidate]
        package_root = DEPLOYMENT_ROOT / record["package_directory"]
        for filename, expected_hash in record["generated_files"].items():
            path = package_root / filename
            if not path.is_file() or _sha256(path) != expected_hash:
                raise RuntimeError(
                    f"affine package file hash mismatch: {candidate}/{filename}"
                )
        parameters = json.loads(
            (package_root / "material_parameters.json").read_text(encoding="utf-8")
        )
        if (
            len(parameters.get("vector_parameters", {})) != 7
            or len(parameters.get("scalar_parameters", {})) != 8
            or parameters["scalar_parameters"].get("NormalYSign") != -1.0
        ):
            raise RuntimeError(f"invalid affine parameters: {candidate}")
        packages[candidate] = {
            "record": record,
            "root": package_root,
            "parameters": parameters,
        }
    return preview, packages


def _ensure_directories() -> None:
    for path in (
        f"{ASSET_ROOT}/Maps",
        f"{ASSET_ROOT}/Materials",
        f"{ASSET_ROOT}/Textures",
    ):
        if not unreal.EditorAssetLibrary.does_directory_exist(path):
            if not unreal.EditorAssetLibrary.make_directory(path):
                raise RuntimeError(f"could not create affine preview directory: {path}")


def _import_texture(source: Path, asset_path: str) -> unreal.Texture2D:
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
        raise RuntimeError(f"affine texture import failed: {asset_path}")
    texture.set_editor_property("srgb", False)
    texture.set_editor_property(
        "compression_settings",
        unreal.TextureCompressionSettings.TC_VECTOR_DISPLACEMENTMAP,
    )
    texture.set_editor_property("filter", unreal.TextureFilter.TF_DEFAULT)
    texture.set_editor_property("address_x", unreal.TextureAddress.TA_WRAP)
    texture.set_editor_property("address_y", unreal.TextureAddress.TA_WRAP)
    texture.set_editor_property(
        "mip_gen_settings", unreal.TextureMipGenSettings.TMGS_SIMPLE_AVERAGE
    )
    texture.set_editor_property("never_stream", False)
    texture.set_editor_property("virtual_texture_streaming", False)
    texture.modify()
    if not unreal.EditorAssetLibrary.save_loaded_asset(texture, only_if_is_dirty=False):
        raise RuntimeError(f"could not save owned affine texture: {asset_path}")
    return texture


def _texture_report(texture: unreal.Texture2D) -> dict:
    return {
        "asset": texture.get_path_name(),
        "size_x": int(texture.blueprint_get_size_x()),
        "size_y": int(texture.blueprint_get_size_y()),
        "srgb": bool(texture.get_editor_property("srgb")),
        "compression_settings": str(texture.get_editor_property("compression_settings")),
        "filter": str(texture.get_editor_property("filter")),
        "address_x": str(texture.get_editor_property("address_x")),
        "address_y": str(texture.get_editor_property("address_y")),
        "mip_gen_settings": str(texture.get_editor_property("mip_gen_settings")),
        "never_stream": bool(texture.get_editor_property("never_stream")),
        "virtual_texture_streaming": bool(
            texture.get_editor_property("virtual_texture_streaming")
        ),
    }


def _export_readback(texture: unreal.Texture2D, path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    task = unreal.AssetExportTask()
    task.set_editor_property("object", texture)
    task.set_editor_property("filename", path.as_posix())
    task.set_editor_property("automated", True)
    task.set_editor_property("prompt", False)
    task.set_editor_property("replace_identical", True)
    task.set_editor_property("write_empty_files", False)
    task.set_editor_property("exporter", unreal.TextureExporterPNG())
    success = bool(unreal.Exporter.run_asset_export_task(task))
    return {
        "path": path.as_posix(),
        "export_success": success,
        "exists": path.is_file(),
        "sha256": _sha256(path) if path.is_file() else None,
        "file_bytes": path.stat().st_size if path.is_file() else None,
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


def _create_master_material(
    hlsl: str, default_texture: unreal.Texture2D
) -> tuple[unreal.Material, dict]:
    folder, name = MASTER_MATERIAL.rsplit("/", 1)
    if unreal.EditorAssetLibrary.does_asset_exist(MASTER_MATERIAL):
        material = unreal.EditorAssetLibrary.load_asset(MASTER_MATERIAL)
    else:
        material = None
    if material is None:
        material = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
            name, folder, unreal.Material, unreal.MaterialFactoryNew()
        )
    if not isinstance(material, unreal.Material):
        raise RuntimeError("affine master material asset is invalid")
    material.set_editor_property("material_domain", unreal.MaterialDomain.MD_SURFACE)
    material.set_editor_property("blend_mode", unreal.BlendMode.BLEND_OPAQUE)
    material.set_editor_property("two_sided", False)
    unreal.MaterialEditingLibrary.delete_all_material_expressions(material)

    latent = unreal.MaterialEditingLibrary.create_material_expression(
        material, unreal.MaterialExpressionTextureSampleParameter2D, -1450, -180
    )
    latent.set_editor_property("parameter_name", "LatentRGBA")
    latent.set_editor_property("texture", default_texture)
    latent.set_editor_property(
        "sampler_type", unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR
    )

    inputs = [_custom_input("LatentRGBA")]
    expressions = [(latent, "RGBA", "LatentRGBA")]
    for row in range(7):
        value = unreal.MaterialEditingLibrary.create_material_expression(
            material, unreal.MaterialExpressionVectorParameter, -1150, -300 + row * 95
        )
        value.set_editor_property("parameter_name", f"AffineW{row}")
        value.set_editor_property("default_value", unreal.LinearColor(0.0, 0.0, 0.0, 0.0))
        inputs.append(_custom_input(f"AffineW{row}"))
        expressions.append((value, "", f"AffineW{row}"))
    for row in range(7):
        value = unreal.MaterialEditingLibrary.create_material_expression(
            material, unreal.MaterialExpressionScalarParameter, -850, -300 + row * 95
        )
        value.set_editor_property("parameter_name", f"AffineB{row}")
        value.set_editor_property("default_value", 0.0)
        inputs.append(_custom_input(f"AffineB{row}"))
        expressions.append((value, "", f"AffineB{row}"))
    normal_sign = unreal.MaterialEditingLibrary.create_material_expression(
        material, unreal.MaterialExpressionScalarParameter, -850, 420
    )
    normal_sign.set_editor_property("parameter_name", "NormalYSign")
    normal_sign.set_editor_property("default_value", -1.0)
    inputs.append(_custom_input("NormalYSign"))
    expressions.append((normal_sign, "", "NormalYSign"))

    custom = unreal.MaterialEditingLibrary.create_material_expression(
        material, unreal.MaterialExpressionCustom, -350, -180
    )
    custom.set_editor_property(
        "description", "Shared C4 affine: one filtered sample, direct 4-to-7 decoder"
    )
    custom.set_editor_property("code", hlsl)
    custom.set_editor_property("output_type", unreal.CustomMaterialOutputType.CMOT_FLOAT3)
    custom.set_editor_property("inputs", inputs)
    custom.set_editor_property(
        "additional_outputs",
        [
            _custom_output("NormalTS", unreal.CustomMaterialOutputType.CMOT_FLOAT3),
            _custom_output("Roughness", unreal.CustomMaterialOutputType.CMOT_FLOAT1),
            _custom_output("Metallic", unreal.CustomMaterialOutputType.CMOT_FLOAT1),
        ],
    )
    connections = {
        input_name: unreal.MaterialEditingLibrary.connect_material_expressions(
            expression, output_name, custom, input_name
        )
        for expression, output_name, input_name in expressions
    }
    connections.update(
        {
            "base_color": unreal.MaterialEditingLibrary.connect_material_property(
                custom, "", unreal.MaterialProperty.MP_BASE_COLOR
            ),
            "normal": unreal.MaterialEditingLibrary.connect_material_property(
                custom, "NormalTS", unreal.MaterialProperty.MP_NORMAL
            ),
            "roughness": unreal.MaterialEditingLibrary.connect_material_property(
                custom, "Roughness", unreal.MaterialProperty.MP_ROUGHNESS
            ),
            "metallic": unreal.MaterialEditingLibrary.connect_material_property(
                custom, "Metallic", unreal.MaterialProperty.MP_METALLIC
            ),
        }
    )
    if not all(connections.values()):
        raise RuntimeError(f"affine master material connection failure: {connections}")
    unreal.MaterialEditingLibrary.recompile_material(material)
    material.modify()
    if not unreal.EditorAssetLibrary.save_loaded_asset(material, only_if_is_dirty=False):
        raise RuntimeError("could not save owned affine master material")
    parameter_names = {
        "texture": [
            str(value)
            for value in unreal.MaterialEditingLibrary.get_texture_parameter_names(material)
        ],
        "vector": [
            str(value)
            for value in unreal.MaterialEditingLibrary.get_vector_parameter_names(material)
        ],
        "scalar": [
            str(value)
            for value in unreal.MaterialEditingLibrary.get_scalar_parameter_names(material)
        ],
    }
    return material, {
        "asset": material.get_path_name(),
        "expression_count": len(
            unreal.MaterialEditingLibrary.get_material_expressions(material)
        ),
        "connections": connections,
        "hlsl_sha256": hashlib.sha256(hlsl.encode("utf-8")).hexdigest(),
        "texture_samples": 1,
        "shader_permutations_by_candidate": 0,
        "uv_source": "default_texcoord0",
        "parameter_names": parameter_names,
    }


def _create_instance(
    master: unreal.Material,
    texture: unreal.Texture2D,
    record: dict,
    parameters: dict,
) -> tuple[unreal.MaterialInstanceConstant, dict]:
    asset_path = record["ue_assets"]["material_instance"]
    folder, name = asset_path.rsplit("/", 1)
    if unreal.EditorAssetLibrary.does_asset_exist(asset_path):
        instance = unreal.EditorAssetLibrary.load_asset(asset_path)
    else:
        instance = None
    if instance is None:
        factory = unreal.MaterialInstanceConstantFactoryNew()
        instance = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
            name,
            folder,
            unreal.MaterialInstanceConstant,
            factory,
        )
    if not isinstance(instance, unreal.MaterialInstanceConstant):
        raise RuntimeError(f"affine material instance is invalid: {asset_path}")
    library = unreal.MaterialEditingLibrary
    library.set_material_instance_parent(instance, master)
    library.update_material_instance(instance)
    instance.modify()
    if not unreal.EditorAssetLibrary.save_loaded_asset(instance, only_if_is_dirty=False):
        raise RuntimeError(f"could not save affine material instance parent: {asset_path}")
    instance = unreal.EditorAssetLibrary.load_asset(asset_path)
    if not isinstance(instance, unreal.MaterialInstanceConstant):
        raise RuntimeError(f"could not reload affine material instance: {asset_path}")
    instance_probe = {
        "parent": str(instance.get_editor_property("parent").get_path_name()),
        "texture": [str(value) for value in library.get_texture_parameter_names(instance)],
        "vector": [str(value) for value in library.get_vector_parameter_names(instance)],
        "scalar": [str(value) for value in library.get_scalar_parameter_names(instance)],
    }
    library.set_material_instance_texture_parameter_value(
        instance, "LatentRGBA", texture
    )
    if not library.is_material_instance_parameter_overridden(
        instance, "LatentRGBA"
    ):
        raise RuntimeError(f"could not bind affine latent texture: {asset_path}")
    for parameter, values in parameters["vector_parameters"].items():
        color = unreal.LinearColor(*[float(value) for value in values])
        library.set_material_instance_vector_parameter_value(
            instance, parameter, color
        )
        if not library.is_material_instance_parameter_overridden(
            instance, parameter
        ):
            raise RuntimeError(f"could not bind {parameter}: {asset_path}")
    for parameter, value in parameters["scalar_parameters"].items():
        library.set_material_instance_scalar_parameter_value(
            instance, parameter, float(value)
        )
        if not library.is_material_instance_parameter_overridden(
            instance, parameter
        ):
            raise RuntimeError(f"could not bind {parameter}: {asset_path}")
    instance.modify()
    if not unreal.EditorAssetLibrary.save_loaded_asset(instance, only_if_is_dirty=False):
        raise RuntimeError(f"could not save affine material instance: {asset_path}")
    return instance, {
        "asset": instance.get_path_name(),
        "parent": master.get_path_name(),
        "texture_parameter": texture.get_path_name(),
        "vector_parameter_count": 7,
        "scalar_parameter_count": 8,
        "normal_y_sign": -1.0,
        "inherited_parameter_names": instance_probe,
    }


def _component(actor) -> unreal.StaticMeshComponent | None:
    value = actor.get_component_by_class(unreal.StaticMeshComponent)
    return value if isinstance(value, unreal.StaticMeshComponent) else None


def _find_reference(actors: list[unreal.Actor]) -> unreal.Actor:
    reference = next((actor for actor in actors if actor.get_actor_label() == "Helmet_Reference"), None)
    if reference is not None and _component(reference) is not None:
        return reference
    for actor in actors:
        component = _component(actor)
        mesh = component.get_editor_property("static_mesh") if component else None
        if mesh is not None and "SciFiHelmet" in mesh.get_path_name():
            return actor
    raise RuntimeError("stored MaterialLab source has no SciFiHelmet reference actor")


def _ensure_actor(actor_subsystem, actors, reference, label, material, offset):
    actor = next((value for value in actors if value.get_actor_label() == label), None)
    if actor is None:
        actor = actor_subsystem.spawn_actor_from_class(
            unreal.StaticMeshActor,
            reference.get_actor_location(),
            reference.get_actor_rotation(),
        )
    if not isinstance(actor, unreal.StaticMeshActor):
        raise RuntimeError(f"could not spawn affine preview actor: {label}")
    actor.set_actor_transform(reference.get_actor_transform(), False, False)
    actor.add_actor_world_offset(offset, False, False)
    actor.set_actor_label(label, mark_dirty=True)
    component = _component(actor)
    reference_component = _component(reference)
    if component is None or reference_component is None:
        raise RuntimeError(f"affine preview actor has no StaticMeshComponent: {label}")
    component.set_static_mesh(reference_component.get_editor_property("static_mesh"))
    component.set_material(0, material)
    actor.modify()
    return actor


def _ensure_preview_map(materials: dict[str, unreal.MaterialInstanceConstant]) -> dict:
    if not unreal.EditorAssetLibrary.does_asset_exist(PREVIEW_MAP):
        if not unreal.EditorAssetLibrary.duplicate_asset(SOURCE_MAP, PREVIEW_MAP):
            raise RuntimeError("could not duplicate stored MaterialLab into affine preview map")
        if not unreal.EditorAssetLibrary.save_asset(PREVIEW_MAP, only_if_is_dirty=False):
            raise RuntimeError("could not save isolated affine preview map")
        return {
            "map": PREVIEW_MAP,
            "source_map": SOURCE_MAP,
            "source_map_saved": False,
            "map_duplicated": True,
            "fresh_process_required": True,
        }
    level_editor = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    if not level_editor.load_level(PREVIEW_MAP):
        raise RuntimeError(f"could not load isolated affine preview map: {PREVIEW_MAP}")
    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actors = list(actor_subsystem.get_all_level_actors())
    reference = _find_reference(actors)
    extent = reference.get_actor_bounds(False, True)[1]
    separation = max(float(extent.x), float(extent.y), float(extent.z)) * 2.8
    created = {}
    for index, candidate in enumerate(EXPECTED_CANDIDATES, start=1):
        actor = _ensure_actor(
            actor_subsystem,
            actors,
            reference,
            CANDIDATE_LABELS[candidate],
            materials[candidate],
            unreal.Vector(0.0, separation * index, 0.0),
        )
        created[candidate] = actor.get_actor_label()
    if not unreal.EditorAssetLibrary.save_asset(PREVIEW_MAP, only_if_is_dirty=False):
        raise RuntimeError("could not save isolated affine preview map")
    return {
        "map": PREVIEW_MAP,
        "source_map": SOURCE_MAP,
        "source_map_saved": False,
        "map_duplicated": False,
        "fresh_process_required": False,
        "reference": reference.get_actor_label(),
        "actors": created,
        "side_by_side_separation_cm": separation,
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
        preview, packages = _validate_package()
        report["preview_manifest_sha256"] = EXPECTED_PREVIEW_MANIFEST_SHA256
        _ensure_directories()
        hlsl_path = DEPLOYMENT_ROOT / "shared_affine_decoder.hlsl"
        hlsl = hlsl_path.read_text(encoding="utf-8")
        if _sha256(hlsl_path) != preview["shared_files"]["shared_affine_decoder.hlsl"]:
            raise RuntimeError("shared affine HLSL hash mismatch")
        default_candidate = EXPECTED_CANDIDATES[0]
        default_package = packages[default_candidate]
        default_record = default_package["record"]
        default_texture = _import_texture(
            default_package["root"] / "latent_rgba8.png",
            default_record["ue_assets"]["latent_texture"],
        )
        textures = {default_candidate: default_texture}
        master, master_report = _create_master_material(hlsl, default_texture)
        report["master_material"] = master_report
        materials = {}
        report["candidates"] = {}
        for candidate in EXPECTED_CANDIDATES:
            package = packages[candidate]
            record = package["record"]
            texture = textures.get(candidate)
            if texture is None:
                texture = _import_texture(
                    package["root"] / "latent_rgba8.png",
                    record["ue_assets"]["latent_texture"],
                )
            instance, instance_report = _create_instance(
                master, texture, record, package["parameters"]
            )
            readback = _export_readback(
                texture, EVIDENCE_ROOT / candidate / "latent_rgba8_readback.png"
            )
            readback["source_sha256"] = record["generated_files"]["latent_rgba8.png"]
            readback["file_sha256_exact_source_match"] = (
                readback["sha256"] == readback["source_sha256"]
            )
            materials[candidate] = instance
            report["candidates"][candidate] = {
                "texture": _texture_report(texture),
                "readback": readback,
                "material_instance": instance_report,
                "artifact_hash": record["artifact_hash"],
                "optimizer_updates": record["optimizer_updates"],
                "objective_id": record["objective_id"],
            }
        report["preview_scene"] = _ensure_preview_map(materials)
        report["status"] = (
            "map_duplicated_fresh_process_required"
            if report["preview_scene"]["fresh_process_required"]
            else "complete_ready_for_manual_preview"
        )
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        _write_report(report)
        raise
    _write_report(report)
    unreal.log(f"Affine preview setup status: {report['status']}")


if globals().get("AFFINE_PREVIEW_AUTORUN", True):
    setup()
