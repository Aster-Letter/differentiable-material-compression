"""Import the verified C4 render-ablation endpoints into an isolated UE map."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import traceback

import unreal


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[4]
DEPLOYMENT_ROOT = REPO_ROOT / "outputs/deployment/c4_render_ablation_20k_v1/ue_preview_job_37489"
MANIFEST_PATH = DEPLOYMENT_ROOT / "preview_manifest.json"
REPORT_PATH = DEPLOYMENT_ROOT / "ue_setup_report.json"
EVIDENCE_ROOT = DEPLOYMENT_ROOT / "ue_evidence"
EXPECTED_MANIFEST_SHA256 = "396b6830e08ead2051d60fc0ba3d1b49021c1778aba595a30927b8f3d0d8bf8b"

ASSET_ROOT = "/Game/CGCompression/C4RenderAblation20k"
SOURCE_MAP = "/Game/CGCompression/Maps/MaterialLab"
PREVIEW_MAP = f"{ASSET_ROOT}/Maps/C4_Render_Ablation_20k"
MASTER_MATERIAL = f"{ASSET_ROOT}/Materials/M_C4_Render_Ablation_Master"
ASSETS = ("Corset", "Lantern", "BoomBox")
ENDPOINTS = ("raw_q4", "material_only", "material_render")
SCENE_COLUMNS = ("source_reference", *ENDPOINTS)


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


def _validate_package() -> dict:
    if _sha256(MANIFEST_PATH) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("C4 render-ablation UE manifest hash mismatch")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "exported_ue_setup_pending"
        or manifest.get("formal_holdout_accessed") is not False
        or set(manifest.get("assets", {})) != set(ASSETS)
        or manifest.get("ue", {}).get("asset_root") != ASSET_ROOT
        or manifest.get("ue", {}).get("preview_map") != PREVIEW_MAP
        or manifest.get("ue", {}).get("master_material") != MASTER_MATERIAL
    ):
        raise RuntimeError("C4 render-ablation UE manifest contract mismatch")
    hlsl = DEPLOYMENT_ROOT / "shared_affine_decoder.hlsl"
    if _sha256(hlsl) != manifest["shared_files"]["shared_affine_decoder.hlsl"]:
        raise RuntimeError("shared affine HLSL hash mismatch")
    for asset_id in ASSETS:
        asset = manifest["assets"][asset_id]
        gltf = REPO_ROOT / asset["gltf"]
        if _sha256(gltf) != asset["gltf_sha256"]:
            raise RuntimeError(f"mesh source hash mismatch: {asset_id}")
        for relative, expected in asset["gltf_directory_files"].items():
            if _sha256(REPO_ROOT / relative) != expected:
                raise RuntimeError(f"mesh dependency hash mismatch: {relative}")
        if set(asset["endpoints"]) != set(ENDPOINTS):
            raise RuntimeError(f"endpoint order mismatch: {asset_id}")
        for endpoint in ENDPOINTS:
            record = asset["endpoints"][endpoint]
            package = DEPLOYMENT_ROOT / record["package_directory"]
            for filename, expected in record["generated_files"].items():
                if _sha256(package / filename) != expected:
                    raise RuntimeError(f"package file hash mismatch: {asset_id}/{endpoint}/{filename}")
    return manifest


def _ensure_directories() -> None:
    for path in (
        f"{ASSET_ROOT}/Maps",
        f"{ASSET_ROOT}/Materials",
        f"{ASSET_ROOT}/Textures",
        f"{ASSET_ROOT}/Meshes",
    ):
        if not unreal.EditorAssetLibrary.does_directory_exist(path):
            if not unreal.EditorAssetLibrary.make_directory(path):
                raise RuntimeError(f"could not create UE directory: {path}")


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
        raise RuntimeError(f"latent texture import failed: {asset_path}")
    texture.set_editor_property("srgb", False)
    texture.set_editor_property(
        "compression_settings", unreal.TextureCompressionSettings.TC_VECTOR_DISPLACEMENTMAP
    )
    texture.set_editor_property("filter", unreal.TextureFilter.TF_DEFAULT)
    texture.set_editor_property("address_x", unreal.TextureAddress.TA_WRAP)
    texture.set_editor_property("address_y", unreal.TextureAddress.TA_WRAP)
    texture.set_editor_property("mip_gen_settings", unreal.TextureMipGenSettings.TMGS_SIMPLE_AVERAGE)
    texture.set_editor_property("never_stream", False)
    texture.set_editor_property("virtual_texture_streaming", False)
    texture.modify()
    if not unreal.EditorAssetLibrary.save_loaded_asset(texture, only_if_is_dirty=False):
        raise RuntimeError(f"could not save latent texture: {asset_path}")
    return texture


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


def _create_master(hlsl: str, default_texture: unreal.Texture2D) -> unreal.Material:
    folder, name = MASTER_MATERIAL.rsplit("/", 1)
    material = (
        unreal.EditorAssetLibrary.load_asset(MASTER_MATERIAL)
        if unreal.EditorAssetLibrary.does_asset_exist(MASTER_MATERIAL)
        else unreal.AssetToolsHelpers.get_asset_tools().create_asset(
            name, folder, unreal.Material, unreal.MaterialFactoryNew()
        )
    )
    if not isinstance(material, unreal.Material):
        raise RuntimeError("invalid shared affine master material")
    material.set_editor_property("material_domain", unreal.MaterialDomain.MD_SURFACE)
    material.set_editor_property("blend_mode", unreal.BlendMode.BLEND_OPAQUE)
    material.set_editor_property("two_sided", False)
    unreal.MaterialEditingLibrary.delete_all_material_expressions(material)
    latent = unreal.MaterialEditingLibrary.create_material_expression(
        material, unreal.MaterialExpressionTextureSampleParameter2D, -1450, -180
    )
    latent.set_editor_property("parameter_name", "LatentRGBA")
    latent.set_editor_property("texture", default_texture)
    latent.set_editor_property("sampler_type", unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR)
    inputs = [_custom_input("LatentRGBA")]
    expressions = [(latent, "RGBA", "LatentRGBA")]
    for row in range(7):
        value = unreal.MaterialEditingLibrary.create_material_expression(
            material, unreal.MaterialExpressionVectorParameter, -1120, -360 + row * 95
        )
        value.set_editor_property("parameter_name", f"AffineW{row}")
        inputs.append(_custom_input(f"AffineW{row}"))
        expressions.append((value, "", f"AffineW{row}"))
    for row in range(7):
        value = unreal.MaterialEditingLibrary.create_material_expression(
            material, unreal.MaterialExpressionScalarParameter, -820, -360 + row * 95
        )
        value.set_editor_property("parameter_name", f"AffineB{row}")
        inputs.append(_custom_input(f"AffineB{row}"))
        expressions.append((value, "", f"AffineB{row}"))
    normal_sign = unreal.MaterialEditingLibrary.create_material_expression(
        material, unreal.MaterialExpressionScalarParameter, -820, 410
    )
    normal_sign.set_editor_property("parameter_name", "NormalYSign")
    normal_sign.set_editor_property("default_value", -1.0)
    inputs.append(_custom_input("NormalYSign"))
    expressions.append((normal_sign, "", "NormalYSign"))
    custom = unreal.MaterialEditingLibrary.create_material_expression(
        material, unreal.MaterialExpressionCustom, -320, -180
    )
    custom.set_editor_property("description", "C4 RGBA8 + affine + fixed deployment safety")
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
    for expression, output_name, input_name in expressions:
        if not unreal.MaterialEditingLibrary.connect_material_expressions(
            expression, output_name, custom, input_name
        ):
            raise RuntimeError(f"could not connect master input: {input_name}")
    for output_name, prop in (
        ("", unreal.MaterialProperty.MP_BASE_COLOR),
        ("NormalTS", unreal.MaterialProperty.MP_NORMAL),
        ("Roughness", unreal.MaterialProperty.MP_ROUGHNESS),
        ("Metallic", unreal.MaterialProperty.MP_METALLIC),
    ):
        if not unreal.MaterialEditingLibrary.connect_material_property(custom, output_name, prop):
            raise RuntimeError(f"could not connect master output: {output_name}")
    unreal.MaterialEditingLibrary.recompile_material(material)
    material.modify()
    if not unreal.EditorAssetLibrary.save_loaded_asset(material, only_if_is_dirty=False):
        raise RuntimeError("could not save shared affine master material")
    return material


def _create_instance(master, texture, record, parameters):
    asset_path = record["ue_assets"]["material_instance"]
    folder, name = asset_path.rsplit("/", 1)
    instance = (
        unreal.EditorAssetLibrary.load_asset(asset_path)
        if unreal.EditorAssetLibrary.does_asset_exist(asset_path)
        else unreal.AssetToolsHelpers.get_asset_tools().create_asset(
            name, folder, unreal.MaterialInstanceConstant, unreal.MaterialInstanceConstantFactoryNew()
        )
    )
    if not isinstance(instance, unreal.MaterialInstanceConstant):
        raise RuntimeError(f"invalid material instance: {asset_path}")
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
        raise RuntimeError(f"could not save material instance: {asset_path}")
    return instance


def _import_mesh(source: Path, folder: str) -> unreal.StaticMesh:
    task = unreal.AssetImportTask()
    task.set_editor_property("filename", source.as_posix())
    task.set_editor_property("destination_path", folder)
    task.set_editor_property("automated", True)
    task.set_editor_property("replace_existing", True)
    task.set_editor_property("save", True)
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    paths = list(task.get_editor_property("imported_object_paths"))
    paths.extend(unreal.EditorAssetLibrary.list_assets(folder, recursive=True, include_folder=False))
    meshes = []
    for path in dict.fromkeys(str(value) for value in paths):
        asset = unreal.EditorAssetLibrary.load_asset(path)
        if isinstance(asset, unreal.StaticMesh):
            meshes.append(asset)
    if not meshes:
        raise RuntimeError(f"glTF import produced no StaticMesh: {source}")
    meshes.sort(
        key=lambda mesh: max(
            float(mesh.get_bounds().box_extent.x),
            float(mesh.get_bounds().box_extent.y),
            float(mesh.get_bounds().box_extent.z),
        ),
        reverse=True,
    )
    return meshes[0]


def _populate_map(meshes: dict, materials: dict) -> dict:
    if not unreal.EditorAssetLibrary.does_asset_exist(PREVIEW_MAP):
        if not unreal.EditorAssetLibrary.duplicate_asset(SOURCE_MAP, PREVIEW_MAP):
            raise RuntimeError("could not duplicate MaterialLab for C4 render preview")
        unreal.EditorAssetLibrary.save_asset(PREVIEW_MAP, only_if_is_dirty=False)
        return {"map": PREVIEW_MAP, "fresh_process_required": True, "actors": {}}
    level_editor = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    if not level_editor.load_level(PREVIEW_MAP):
        raise RuntimeError(f"could not load preview map: {PREVIEW_MAP}")
    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    existing = {actor.get_actor_label(): actor for actor in actor_subsystem.get_all_level_actors()}
    reference = existing.get("Helmet_Reference")
    if reference is not None:
        reference.set_is_temporarily_hidden_in_editor(True)
        reference.set_actor_hidden_in_game(True)
    actors = {}
    row_x = {"Corset": -330.0, "Lantern": 0.0, "BoomBox": 330.0}
    column_y = {
        "source_reference": -465.0,
        "raw_q4": -155.0,
        "material_only": 155.0,
        "material_render": 465.0,
    }
    for asset_id in ASSETS:
        mesh = meshes[asset_id]
        bounds = mesh.get_bounds()
        full_extent = max(
            2.0 * float(bounds.box_extent.x),
            2.0 * float(bounds.box_extent.y),
            2.0 * float(bounds.box_extent.z),
        )
        scale = 170.0 / max(full_extent, 1.0e-6)
        bottom = (float(bounds.origin.z) - float(bounds.box_extent.z)) * scale
        material_slots = len(mesh.get_editor_property("static_materials"))
        for endpoint in SCENE_COLUMNS:
            label = f"C4_{asset_id}_{endpoint.upper()}"
            actor = existing.get(label)
            if actor is None:
                actor = actor_subsystem.spawn_actor_from_class(
                    unreal.StaticMeshActor,
                    unreal.Vector(row_x[asset_id], column_y[endpoint], -bottom),
                )
            if not isinstance(actor, unreal.StaticMeshActor):
                raise RuntimeError(f"invalid preview actor: {label}")
            actor.set_actor_label(label, mark_dirty=True)
            actor.set_actor_location(
                unreal.Vector(row_x[asset_id], column_y[endpoint], -bottom), False, False
            )
            actor.set_actor_scale3d(unreal.Vector(scale, scale, scale))
            component = actor.get_component_by_class(unreal.StaticMeshComponent)
            component.set_static_mesh(mesh)
            if endpoint == "source_reference":
                for slot in range(material_slots):
                    component.set_material(slot, None)
            else:
                for slot in range(material_slots):
                    component.set_material(slot, materials[(asset_id, endpoint)])
            actor.modify()
            actors[f"{asset_id}/{endpoint}"] = {
                "label": label,
                "location": [row_x[asset_id], column_y[endpoint], -bottom],
                "uniform_scale": scale,
                "material": (
                    "source_gltf_default"
                    if endpoint == "source_reference"
                    else materials[(asset_id, endpoint)].get_path_name()
                ),
            }
    if not unreal.EditorAssetLibrary.save_asset(PREVIEW_MAP, only_if_is_dirty=False):
        raise RuntimeError("could not save C4 render preview map")
    return {"map": PREVIEW_MAP, "fresh_process_required": False, "actors": actors}


def setup() -> None:
    report = {
        "schema_version": 1,
        "status": "started",
        "formal_holdout_accessed": False,
        "source_map_preserved": True,
        "preview_asset_root": ASSET_ROOT,
    }
    try:
        manifest = _validate_package()
        report["preview_manifest_sha256"] = EXPECTED_MANIFEST_SHA256
        _ensure_directories()
        hlsl = (DEPLOYMENT_ROOT / "shared_affine_decoder.hlsl").read_text(encoding="utf-8")
        textures = {}
        materials = {}
        report["endpoints"] = {}
        first_texture = None
        for asset_id in ASSETS:
            for endpoint in ENDPOINTS:
                record = manifest["assets"][asset_id]["endpoints"][endpoint]
                package = DEPLOYMENT_ROOT / record["package_directory"]
                texture = _import_texture(package / "latent_rgba8.png", record["ue_assets"]["latent_texture"])
                textures[(asset_id, endpoint)] = texture
                first_texture = first_texture or texture
        master = _create_master(hlsl, first_texture)
        for asset_id in ASSETS:
            for endpoint in ENDPOINTS:
                record = manifest["assets"][asset_id]["endpoints"][endpoint]
                package = DEPLOYMENT_ROOT / record["package_directory"]
                parameters = json.loads((package / "material_parameters.json").read_text(encoding="utf-8"))
                materials[(asset_id, endpoint)] = _create_instance(
                    master, textures[(asset_id, endpoint)], record, parameters
                )
                readback = _export_readback(
                    textures[(asset_id, endpoint)],
                    EVIDENCE_ROOT / asset_id / endpoint / "latent_rgba8_readback.png",
                )
                readback["source"] = (package / "latent_rgba8.png").as_posix()
                readback["source_sha256"] = record["generated_files"]["latent_rgba8.png"]
                report["endpoints"][f"{asset_id}/{endpoint}"] = {
                    "texture": textures[(asset_id, endpoint)].get_path_name(),
                    "material": materials[(asset_id, endpoint)].get_path_name(),
                    "readback": readback,
                }
        meshes = {
            asset_id: _import_mesh(
                REPO_ROOT / manifest["assets"][asset_id]["gltf"],
                manifest["assets"][asset_id]["ue_mesh_folder"],
            )
            for asset_id in ASSETS
        }
        report["meshes"] = {name: mesh.get_path_name() for name, mesh in meshes.items()}
        report["preview_scene"] = _populate_map(meshes, materials)
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
    unreal.log(f"C4 render-ablation preview setup status: {report['status']}")


if globals().get("C4_RENDER_ABLATION_PREVIEW_AUTORUN", True):
    setup()
