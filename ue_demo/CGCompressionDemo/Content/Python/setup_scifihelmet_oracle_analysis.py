"""Create non-destructive BaseColor/Metallic oracle materials for UE attribution."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import traceback

import unreal


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[4]
DEPLOYMENT_ROOT = REPO_ROOT / "outputs/deployment/scifihelmet/ue_pre_qat_hard"
HLSL_PATH = DEPLOYMENT_ROOT / "M_SciFiHelmet_Compressed.custom.hlsl"
REPORT_PATH = REPO_ROOT / "outputs/compression/scifihelmet/repair_v1/analysis/ue_oracle/ue_oracle_report.json"
SCREENSHOT_PATH = REPO_ROOT / "outputs/compression/scifihelmet/repair_v1/analysis/ue_oracle/MaterialLab_Oracle_4Way.png"

MAP_PATH = "/Game/CGCompression/Maps/MaterialLab"
REFERENCE_MATERIAL_PATH = "/Game/CGCompression/Materials/Reference/M_SciFiHelmet_Reference"
LATENT_PATH = "/Game/CGCompression/Textures/T_SciFiHelmet_Latent_RGBA8"
ANALYSIS_FOLDER = "/Game/CGCompression/Materials/Analysis"
BASE_ORACLE_PATH = f"{ANALYSIS_FOLDER}/M_SciFiHelmet_BaseColorOracle"
METAL_ORACLE_PATH = f"{ANALYSIS_FOLDER}/M_SciFiHelmet_MetallicOracle"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _texture_samples(material: unreal.Material) -> dict[str, unreal.Texture2D]:
    result: dict[str, unreal.Texture2D] = {}
    for expression in unreal.MaterialEditingLibrary.get_material_expressions(material):
        if not isinstance(expression, unreal.MaterialExpressionTextureSample):
            continue
        texture = expression.get_editor_property("texture")
        if not isinstance(texture, unreal.Texture2D):
            continue
        lowered = texture.get_name().lower()
        if "basecolor" in lowered or "base_color" in lowered:
            result["base_color"] = texture
        elif "metallicroughness" in lowered or "metallic_roughness" in lowered:
            result["metallic_roughness"] = texture
    missing = {"base_color", "metallic_roughness"}.difference(result)
    if missing:
        raise RuntimeError(f"reference material texture discovery failed: {sorted(missing)}")
    return result


def _new_or_existing_material(asset_path: str) -> unreal.Material:
    material = unreal.EditorAssetLibrary.load_asset(asset_path)
    if material is None:
        material = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
            Path(asset_path).name,
            str(Path(asset_path).parent).replace("\\", "/"),
            unreal.Material,
            unreal.MaterialFactoryNew(),
        )
    if not isinstance(material, unreal.Material):
        raise RuntimeError(f"analysis asset is not a Material: {asset_path}")
    return material


def _build_oracle_material(
    asset_path: str,
    latent: unreal.Texture2D,
    source_texture: unreal.Texture2D,
    hlsl: str,
    oracle: str,
) -> unreal.Material:
    material = _new_or_existing_material(asset_path)
    material.set_editor_property("material_domain", unreal.MaterialDomain.MD_SURFACE)
    material.set_editor_property("blend_mode", unreal.BlendMode.BLEND_OPAQUE)
    material.set_editor_property("two_sided", False)
    unreal.MaterialEditingLibrary.delete_all_material_expressions(material)

    coordinates = unreal.MaterialEditingLibrary.create_material_expression(
        material, unreal.MaterialExpressionTextureCoordinate, -1250, -100
    )
    coordinates.set_editor_property("coordinate_index", 0)
    latent_sample = unreal.MaterialEditingLibrary.create_material_expression(
        material, unreal.MaterialExpressionTextureSample, -950, -180
    )
    latent_sample.set_editor_property("texture", latent)
    latent_sample.set_editor_property("sampler_type", unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR)
    source_sample = unreal.MaterialEditingLibrary.create_material_expression(
        material, unreal.MaterialExpressionTextureSample, -950, 220
    )
    source_sample.set_editor_property("texture", source_texture)
    source_sample.set_editor_property(
        "sampler_type",
        unreal.MaterialSamplerType.SAMPLERTYPE_COLOR
        if oracle == "base_color"
        else unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR,
    )
    normal_y_sign = unreal.MaterialEditingLibrary.create_material_expression(
        material, unreal.MaterialExpressionConstant, -680, 80
    )
    normal_y_sign.set_editor_property("r", -1.0)
    custom = unreal.MaterialEditingLibrary.create_material_expression(
        material, unreal.MaterialExpressionCustom, -380, -100
    )
    custom.set_editor_property("description", f"Frozen baseline with reference {oracle} oracle")
    custom.set_editor_property("code", hlsl)
    custom.set_editor_property("output_type", unreal.CustomMaterialOutputType.CMOT_FLOAT3)
    latent_input = unreal.CustomInput()
    latent_input.set_editor_property("input_name", "LatentRGBA")
    normal_input = unreal.CustomInput()
    normal_input.set_editor_property("input_name", "NormalYSign")
    custom.set_editor_property("inputs", [latent_input, normal_input])
    outputs = []
    for name, output_type in (
        ("NormalTS", unreal.CustomMaterialOutputType.CMOT_FLOAT3),
        ("Roughness", unreal.CustomMaterialOutputType.CMOT_FLOAT1),
        ("Metallic", unreal.CustomMaterialOutputType.CMOT_FLOAT1),
    ):
        output = unreal.CustomOutput()
        output.set_editor_property("output_name", name)
        output.set_editor_property("output_type", output_type)
        outputs.append(output)
    custom.set_editor_property("additional_outputs", outputs)

    connections = [
        unreal.MaterialEditingLibrary.connect_material_expressions(coordinates, "", latent_sample, "UVs"),
        unreal.MaterialEditingLibrary.connect_material_expressions(coordinates, "", source_sample, "UVs"),
        unreal.MaterialEditingLibrary.connect_material_expressions(latent_sample, "RGBA", custom, "LatentRGBA"),
        unreal.MaterialEditingLibrary.connect_material_expressions(normal_y_sign, "", custom, "NormalYSign"),
        unreal.MaterialEditingLibrary.connect_material_property(custom, "NormalTS", unreal.MaterialProperty.MP_NORMAL),
        unreal.MaterialEditingLibrary.connect_material_property(custom, "Roughness", unreal.MaterialProperty.MP_ROUGHNESS),
    ]
    if oracle == "base_color":
        connections.extend(
            [
                unreal.MaterialEditingLibrary.connect_material_property(source_sample, "RGB", unreal.MaterialProperty.MP_BASE_COLOR),
                unreal.MaterialEditingLibrary.connect_material_property(custom, "Metallic", unreal.MaterialProperty.MP_METALLIC),
            ]
        )
    elif oracle == "metallic":
        connections.extend(
            [
                unreal.MaterialEditingLibrary.connect_material_property(custom, "", unreal.MaterialProperty.MP_BASE_COLOR),
                unreal.MaterialEditingLibrary.connect_material_property(source_sample, "B", unreal.MaterialProperty.MP_METALLIC),
            ]
        )
    else:
        raise ValueError(f"unsupported oracle: {oracle}")
    if not all(connections):
        raise RuntimeError(f"oracle material connections failed: {oracle} {connections}")
    unreal.MaterialEditingLibrary.recompile_material(material)
    material.modify()
    if not unreal.EditorAssetLibrary.save_loaded_asset(material, only_if_is_dirty=False):
        raise RuntimeError(f"could not save oracle material: {asset_path}")
    return material


def _component(actor: unreal.Actor) -> unreal.StaticMeshComponent:
    component = actor.get_component_by_class(unreal.StaticMeshComponent)
    if not isinstance(component, unreal.StaticMeshComponent):
        raise RuntimeError(f"actor has no StaticMeshComponent: {actor.get_actor_label()}")
    return component


def _ensure_actor(
    actors: list[unreal.Actor],
    source: unreal.Actor,
    label: str,
    material: unreal.Material,
    offset_y: float,
) -> unreal.Actor:
    subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actor = next((value for value in actors if value.get_actor_label() == label), None)
    if actor is None:
        duplicated = list(subsystem.duplicate_actors([source], None, unreal.Vector(0.0, offset_y, 0.0)))
        if len(duplicated) != 1:
            raise RuntimeError(f"could not duplicate oracle actor: {label}")
        actor = duplicated[0]
    else:
        actor.set_actor_transform(source.get_actor_transform(), False, False)
        actor.add_actor_world_offset(unreal.Vector(0.0, offset_y, 0.0), False, False)
    actor.set_actor_label(label, mark_dirty=True)
    _component(actor).set_material(0, material)
    actor.modify()
    return actor


def setup() -> None:
    report: dict = {"schema_version": 1, "status": "started", "baseline_preserved": True}
    try:
        if not unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).load_level(MAP_PATH):
            raise RuntimeError(f"could not load map: {MAP_PATH}")
        if not unreal.EditorAssetLibrary.does_directory_exist(ANALYSIS_FOLDER):
            unreal.EditorAssetLibrary.make_directory(ANALYSIS_FOLDER)
        reference_material = unreal.EditorAssetLibrary.load_asset(REFERENCE_MATERIAL_PATH)
        latent = unreal.EditorAssetLibrary.load_asset(LATENT_PATH)
        if not isinstance(reference_material, unreal.Material) or not isinstance(latent, unreal.Texture2D):
            raise RuntimeError("reference material or latent texture is missing")
        source_textures = _texture_samples(reference_material)
        hlsl = HLSL_PATH.read_text(encoding="utf-8")
        base_oracle = _build_oracle_material(
            BASE_ORACLE_PATH, latent, source_textures["base_color"], hlsl, "base_color"
        )
        metal_oracle = _build_oracle_material(
            METAL_ORACLE_PATH, latent, source_textures["metallic_roughness"], hlsl, "metallic"
        )
        actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
        actors = list(actor_subsystem.get_all_level_actors())
        baseline = next((actor for actor in actors if actor.get_actor_label() == "Helmet_Compressed"), None)
        if baseline is None:
            raise RuntimeError("Helmet_Compressed baseline actor is missing")
        extent = baseline.get_actor_bounds(False, True)[1]
        separation = max(float(extent.x), float(extent.y), float(extent.z)) * 2.8
        base_actor = _ensure_actor(actors, baseline, "Helmet_BaseColorOracle", base_oracle, separation)
        metal_actor = _ensure_actor(actors, baseline, "Helmet_MetallicOracle", metal_oracle, separation * 2.0)
        if not unreal.EditorLoadingAndSavingUtils.save_dirty_packages(True, True):
            raise RuntimeError("could not save UE oracle analysis packages")
        report.update(
            {
                "status": "assets_saved_screenshot_requested",
                "source_textures": {name: value.get_path_name() for name, value in source_textures.items()},
                "materials": {"base_color_oracle": base_oracle.get_path_name(), "metallic_oracle": metal_oracle.get_path_name()},
                "actors": [base_actor.get_actor_label(), metal_actor.get_actor_label()],
                "hlsl_sha256": _sha256(HLSL_PATH),
            }
        )
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        origins = [actor.get_actor_location() for actor in (baseline, base_actor, metal_actor)]
        target = unreal.Vector(
            sum(float(value.x) for value in origins) / len(origins),
            sum(float(value.y) for value in origins) / len(origins),
            sum(float(value.z) for value in origins) / len(origins),
        )
        radius = max(float(extent.x), float(extent.y), float(extent.z))
        location = target + unreal.Vector(radius * 6.5, radius * 4.5, radius * 1.8)
        camera = actor_subsystem.spawn_actor_from_class(
            unreal.CameraActor, location, unreal.MathLibrary.find_look_at_rotation(location, target)
        )
        camera.set_actor_label("Camera_MaterialLab_Oracle_4Way", mark_dirty=True)
        camera_component = camera.get_component_by_class(unreal.CameraComponent)
        if isinstance(camera_component, unreal.CameraComponent):
            camera_component.set_editor_property("field_of_view", 50.0)
        unreal.AutomationLibrary.finish_loading_before_screenshot()
        SCREENSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
        task = unreal.AutomationLibrary.take_high_res_screenshot(
            1920, 1080, SCREENSHOT_PATH.as_posix(), camera=camera, force_game_view=True
        )
        report["screenshot"] = {"requested_path": SCREENSHOT_PATH.as_posix(), "task_valid": task.is_valid_task()}
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        unreal.log(f"CGCompression oracle analysis ready: {REPORT_PATH.as_posix()}")
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        unreal.log_error(f"CGCompression oracle analysis failed: {error}")
        raise


setup()
