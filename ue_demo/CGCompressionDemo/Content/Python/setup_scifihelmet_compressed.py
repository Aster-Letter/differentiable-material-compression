"""Create the verified content-only UE baseline and fixed MaterialLab A/B scene."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import traceback

import unreal


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[4]
PROJECT_ROOT = SCRIPT_PATH.parents[2]
DEPLOYMENT_ROOT = REPO_ROOT / "outputs/deployment/scifihelmet/ue_pre_qat_hard"
MANIFEST_PATH = DEPLOYMENT_ROOT / "deployment_manifest.json"
HLSL_PATH = DEPLOYMENT_ROOT / "M_SciFiHelmet_Compressed.custom.hlsl"
LATENT_SOURCE_PATH = DEPLOYMENT_ROOT / "T_SciFiHelmet_Latent_RGBA8.png"
REPORT_PATH = DEPLOYMENT_ROOT / "ue_setup_report.json"

MAP_PATH = "/Game/CGCompression/Maps/MaterialLab"
TEXTURE_FOLDER = "/Game/CGCompression/Textures"
TEXTURE_PATH = f"{TEXTURE_FOLDER}/T_SciFiHelmet_Latent_RGBA8"
MATERIAL_FOLDER = "/Game/CGCompression/Materials/Compressed"
MATERIAL_PATH = f"{MATERIAL_FOLDER}/M_SciFiHelmet_Compressed"

EXPECTED_LATENT_SHA256 = "5a1781afc1a877be452a87a3d958e48cab921b45237faebf2be3668a60ae5fdc"
EXPECTED_DECODER_SHA256 = "d676ade8294600eb0064a835eabfe86d4d35e39ee787d512574fbef8d7346baa"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _enum_name(value) -> str:
    return str(value)


def _load_and_validate_generated_package() -> tuple[dict, str]:
    """Reject any package that is not the frozen pre-QAT 103-parameter baseline."""

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest["selection"] != "pre_qat_hard_tiny_mlp":
        raise RuntimeError(f"unexpected deployment selection: {manifest['selection']}")
    if manifest["inputs"]["latent"]["sha256"] != EXPECTED_LATENT_SHA256:
        raise RuntimeError("manifest latent hash is not the frozen pre-QAT artifact")
    if manifest["inputs"]["decoder"]["sha256"] != EXPECTED_DECODER_SHA256:
        raise RuntimeError("manifest decoder hash is not the frozen pre-QAT artifact")
    if manifest["decoder"] != {
        "activation": "ReLU",
        "architecture": "4->8->7",
        "macs_per_pixel": 88,
        "parameter_count": 103,
        "raw_equation": "raw=W2*ReLU(W1*z+b1)+b2",
        "weight_bytes_float32": 412,
    }:
        raise RuntimeError(f"unexpected decoder metadata: {manifest['decoder']}")
    if _sha256(LATENT_SOURCE_PATH) != EXPECTED_LATENT_SHA256:
        raise RuntimeError("generated UE import PNG hash mismatch")
    if _sha256(HLSL_PATH) != manifest["generated_files"][HLSL_PATH.name]:
        raise RuntimeError("generated Custom HLSL hash mismatch")
    return manifest, HLSL_PATH.read_text(encoding="utf-8")


def _ensure_directories() -> None:
    for path in (TEXTURE_FOLDER, MATERIAL_FOLDER):
        if not unreal.EditorAssetLibrary.does_directory_exist(path):
            if not unreal.EditorAssetLibrary.make_directory(path):
                raise RuntimeError(f"could not create content directory: {path}")


def _import_or_load_texture() -> unreal.Texture2D:
    """Import or reuse latent RGBA8, then enforce the faithful sampler settings."""

    texture = unreal.EditorAssetLibrary.load_asset(TEXTURE_PATH)
    if texture is None:
        task = unreal.AssetImportTask()
        task.set_editor_property("filename", LATENT_SOURCE_PATH.as_posix())
        task.set_editor_property("destination_path", TEXTURE_FOLDER)
        task.set_editor_property("destination_name", "T_SciFiHelmet_Latent_RGBA8")
        task.set_editor_property("automated", True)
        task.set_editor_property("replace_existing", False)
        task.set_editor_property("save", False)
        unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
        texture = unreal.EditorAssetLibrary.load_asset(TEXTURE_PATH)
    if not isinstance(texture, unreal.Texture2D):
        raise RuntimeError(f"latent import did not produce Texture2D at {TEXTURE_PATH}")

    texture.set_editor_property("srgb", False)
    texture.set_editor_property(
        "compression_settings",
        unreal.TextureCompressionSettings.TC_VECTOR_DISPLACEMENTMAP,
    )
    texture.set_editor_property("filter", unreal.TextureFilter.TF_BILINEAR)
    texture.set_editor_property("address_x", unreal.TextureAddress.TA_WRAP)
    texture.set_editor_property("address_y", unreal.TextureAddress.TA_WRAP)
    texture.set_editor_property(
        "mip_gen_settings", unreal.TextureMipGenSettings.TMGS_NO_MIPMAPS
    )
    texture.set_editor_property("never_stream", True)
    texture.set_editor_property("virtual_texture_streaming", False)
    texture.modify()
    if not unreal.EditorAssetLibrary.save_loaded_asset(texture, only_if_is_dirty=False):
        raise RuntimeError(f"could not save texture: {TEXTURE_PATH}")
    return texture


def _create_or_update_material(texture: unreal.Texture2D, hlsl: str) -> tuple[unreal.Material, dict]:
    """Build the content-only Custom expression from generated HLSL constants."""

    material = unreal.EditorAssetLibrary.load_asset(MATERIAL_PATH)
    if material is None:
        material = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
            "M_SciFiHelmet_Compressed",
            MATERIAL_FOLDER,
            unreal.Material,
            unreal.MaterialFactoryNew(),
        )
    if not isinstance(material, unreal.Material):
        raise RuntimeError(f"compressed asset is not a Material: {MATERIAL_PATH}")

    material.set_editor_property("material_domain", unreal.MaterialDomain.MD_SURFACE)
    material.set_editor_property("blend_mode", unreal.BlendMode.BLEND_OPAQUE)
    material.set_editor_property("two_sided", False)
    # This function owns only the dedicated compressed material; reference and
    # imported SciFiHelmet materials are never rewritten.
    unreal.MaterialEditingLibrary.delete_all_material_expressions(material)

    coordinates = unreal.MaterialEditingLibrary.create_material_expression(
        material, unreal.MaterialExpressionTextureCoordinate, -1150, -100
    )
    coordinates.set_editor_property("coordinate_index", 0)
    sample = unreal.MaterialEditingLibrary.create_material_expression(
        material, unreal.MaterialExpressionTextureSample, -850, -100
    )
    sample.set_editor_property("texture", texture)
    sample.set_editor_property(
        "sampler_type", unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR
    )
    normal_y_sign = unreal.MaterialEditingLibrary.create_material_expression(
        material, unreal.MaterialExpressionConstant, -850, 220
    )
    # Training truth stays glTF +Y.  Apply the verified UE tangent bridge once.
    normal_y_sign.set_editor_property("r", -1.0)
    custom = unreal.MaterialEditingLibrary.create_material_expression(
        material, unreal.MaterialExpressionCustom, -350, -80
    )
    custom.set_editor_property("description", "Frozen pre-QAT hard 4->8->7 decoder; 103 params, 88 MAC/pixel")
    custom.set_editor_property("code", hlsl)
    custom.set_editor_property("output_type", unreal.CustomMaterialOutputType.CMOT_FLOAT3)
    latent_input = unreal.CustomInput()
    latent_input.set_editor_property("input_name", "LatentRGBA")
    normal_y_input = unreal.CustomInput()
    normal_y_input.set_editor_property("input_name", "NormalYSign")
    custom.set_editor_property(
        "inputs",
        [latent_input, normal_y_input],
    )
    normal_output = unreal.CustomOutput()
    normal_output.set_editor_property("output_name", "NormalTS")
    normal_output.set_editor_property(
        "output_type", unreal.CustomMaterialOutputType.CMOT_FLOAT3
    )
    roughness_output = unreal.CustomOutput()
    roughness_output.set_editor_property("output_name", "Roughness")
    roughness_output.set_editor_property(
        "output_type", unreal.CustomMaterialOutputType.CMOT_FLOAT1
    )
    metallic_output = unreal.CustomOutput()
    metallic_output.set_editor_property("output_name", "Metallic")
    metallic_output.set_editor_property(
        "output_type", unreal.CustomMaterialOutputType.CMOT_FLOAT1
    )
    custom.set_editor_property(
        "additional_outputs",
        [normal_output, roughness_output, metallic_output],
    )

    connections = {
        "texcoord0_to_sample": unreal.MaterialEditingLibrary.connect_material_expressions(
            coordinates, "", sample, "UVs"
        ),
        "sample_rgba_to_decoder": unreal.MaterialEditingLibrary.connect_material_expressions(
            sample, "RGBA", custom, "LatentRGBA"
        ),
        "normal_y_sign_once": unreal.MaterialEditingLibrary.connect_material_expressions(
            normal_y_sign, "", custom, "NormalYSign"
        ),
        "decoder_to_base_color": unreal.MaterialEditingLibrary.connect_material_property(
            custom, "", unreal.MaterialProperty.MP_BASE_COLOR
        ),
        "decoder_to_normal": unreal.MaterialEditingLibrary.connect_material_property(
            custom, "NormalTS", unreal.MaterialProperty.MP_NORMAL
        ),
        "decoder_to_roughness": unreal.MaterialEditingLibrary.connect_material_property(
            custom, "Roughness", unreal.MaterialProperty.MP_ROUGHNESS
        ),
        "decoder_to_metallic": unreal.MaterialEditingLibrary.connect_material_property(
            custom, "Metallic", unreal.MaterialProperty.MP_METALLIC
        ),
    }
    if not all(connections.values()):
        raise RuntimeError(f"one or more material connections failed: {connections}")
    unreal.MaterialEditingLibrary.recompile_material(material)
    material.modify()
    if not unreal.EditorAssetLibrary.save_loaded_asset(material, only_if_is_dirty=False):
        raise RuntimeError(f"could not save material: {MATERIAL_PATH}")
    return material, {
        "connections": connections,
        "expression_count": len(unreal.MaterialEditingLibrary.get_material_expressions(material)),
        "custom_outputs": [
            str(name)
            for name in unreal.MaterialEditingLibrary.get_material_expression_output_names(custom)
        ],
    }


def _actor_component(actor: unreal.Actor) -> unreal.StaticMeshComponent | None:
    component = actor.get_component_by_class(unreal.StaticMeshComponent)
    return component if isinstance(component, unreal.StaticMeshComponent) else None


def _find_reference_actor(actors: list[unreal.Actor]) -> unreal.Actor:
    for actor in actors:
        if actor.get_actor_label() == "Helmet_Reference":
            return actor
    for actor in actors:
        component = _actor_component(actor)
        if component is None:
            continue
        mesh = component.get_editor_property("static_mesh")
        if mesh is not None and "SciFiHelmet" in mesh.get_path_name():
            actor.set_actor_label("Helmet_Reference", mark_dirty=True)
            return actor
    raise RuntimeError("MaterialLab has no SciFiHelmet reference actor")


def _set_up_ab_actor(material: unreal.Material) -> tuple[unreal.Actor, unreal.Actor, dict]:
    level_editor = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    if not level_editor.load_level(MAP_PATH):
        raise RuntimeError(f"could not load map: {MAP_PATH}")
    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actors = list(actor_subsystem.get_all_level_actors())
    reference = _find_reference_actor(actors)
    reference_component = _actor_component(reference)
    if reference_component is None:
        raise RuntimeError("Helmet_Reference has no StaticMeshComponent")

    extent = reference.get_actor_bounds(False, True)[1]
    separation = max(float(extent.x), float(extent.y), float(extent.z)) * 2.8
    offset = unreal.Vector(0.0, separation, 0.0)
    compressed = next(
        (actor for actor in actors if actor.get_actor_label() == "Helmet_Compressed"),
        None,
    )
    if compressed is None:
        duplicated = list(actor_subsystem.duplicate_actors([reference], None, offset))
        if len(duplicated) != 1:
            raise RuntimeError("could not duplicate Helmet_Reference for A/B")
        compressed = duplicated[0]
    else:
        compressed.set_actor_transform(reference.get_actor_transform(), False, False)
        compressed.add_actor_world_offset(offset, False, False)
    compressed.set_actor_label("Helmet_Compressed", mark_dirty=True)
    compressed.set_actor_rotation(reference.get_actor_rotation(), False)
    compressed.set_actor_scale3d(reference.get_actor_scale3d())
    compressed_component = _actor_component(compressed)
    if compressed_component is None:
        raise RuntimeError("Helmet_Compressed has no StaticMeshComponent")
    compressed_component.set_static_mesh(reference_component.get_editor_property("static_mesh"))
    compressed_component.set_material(0, material)
    compressed.modify()

    reference_material = reference_component.get_material(0)
    return reference, compressed, {
        "reference_actor": reference.get_actor_label(),
        "compressed_actor": compressed.get_actor_label(),
        "reference_mesh": reference_component.get_editor_property("static_mesh").get_path_name(),
        "compressed_mesh": compressed_component.get_editor_property("static_mesh").get_path_name(),
        "reference_material": reference_material.get_path_name() if reference_material else None,
        "compressed_material": compressed_component.get_material(0).get_path_name(),
        "shared_rotation": str(reference.get_actor_rotation()),
        "shared_scale": str(reference.get_actor_scale3d()),
        "side_by_side_offset_cm": [0.0, separation, 0.0],
    }


def _camera_for_targets(reference: unreal.Actor, compressed: unreal.Actor) -> unreal.CameraActor:
    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actors = list(actor_subsystem.get_all_level_actors())
    camera = next(
        (
            actor
            for actor in actors
            if isinstance(actor, unreal.CameraActor)
            and actor.get_actor_label() == "Camera_MaterialLab_AB_Fixed"
        ),
        None,
    )
    reference_origin, reference_extent = reference.get_actor_bounds(False, True)
    compressed_origin, compressed_extent = compressed.get_actor_bounds(False, True)
    target = (reference_origin + compressed_origin) * 0.5
    radius = max(
        float(reference_extent.x),
        float(reference_extent.y),
        float(reference_extent.z),
        float(compressed_extent.x),
        float(compressed_extent.y),
        float(compressed_extent.z),
    )
    location = target + unreal.Vector(radius * 4.8, radius * 4.0, radius * 1.4)
    rotation = unreal.MathLibrary.find_look_at_rotation(location, target)
    if camera is None:
        camera = actor_subsystem.spawn_actor_from_class(unreal.CameraActor, location, rotation)
    camera.set_actor_label("Camera_MaterialLab_AB_Fixed", mark_dirty=True)
    camera.set_actor_location(location, False, False)
    camera.set_actor_rotation(rotation, False)
    camera_component = camera.get_component_by_class(unreal.CameraComponent)
    if not isinstance(camera_component, unreal.CameraComponent):
        raise RuntimeError("fixed A/B camera has no CameraComponent")
    camera_component.set_editor_property("field_of_view", 42.0)
    camera.modify()
    return camera


def _texture_settings(texture: unreal.Texture2D) -> dict:
    return {
        "asset": texture.get_path_name(),
        "size_x": int(texture.blueprint_get_size_x()),
        "size_y": int(texture.blueprint_get_size_y()),
        "srgb": bool(texture.get_editor_property("srgb")),
        "compression_settings": _enum_name(texture.get_editor_property("compression_settings")),
        "filter": _enum_name(texture.get_editor_property("filter")),
        "address_x": _enum_name(texture.get_editor_property("address_x")),
        "address_y": _enum_name(texture.get_editor_property("address_y")),
        "mip_gen_settings": _enum_name(texture.get_editor_property("mip_gen_settings")),
        "never_stream": bool(texture.get_editor_property("never_stream")),
        "virtual_texture_streaming": bool(texture.get_editor_property("virtual_texture_streaming")),
    }


def _write_report(report: dict) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def setup() -> None:
    report = {
        "schema_version": 1,
        "status": "started",
        "map": MAP_PATH,
        "reference_assets_preserved": True,
        "ambient_occlusion_connected": False,
        "specular": "Default Lit default 0.5 (unconnected)",
        "normal_y_bridge": {
            "training_truth": "+Y glTF/OpenGL",
            "ue_sign": -1,
            "application_count": 1,
            "visual_verification": "pending oblique-light A/B screenshot review",
        },
    }
    try:
        manifest, hlsl = _load_and_validate_generated_package()
        report["frozen_inputs"] = manifest["inputs"]
        report["decoder"] = manifest["decoder"]
        _ensure_directories()
        texture = _import_or_load_texture()
        report["texture"] = _texture_settings(texture)
        material, material_report = _create_or_update_material(texture, hlsl)
        report["material"] = {
            "asset": material.get_path_name(),
            **material_report,
        }
        reference, compressed, actor_report = _set_up_ab_actor(material)
        report["ab_scene"] = actor_report
        camera = _camera_for_targets(reference, compressed)
        report["fixed_camera"] = {
            "actor": camera.get_actor_label(),
            "location": str(camera.get_actor_location()),
            "rotation": str(camera.get_actor_rotation()),
            "field_of_view": 42.0,
        }
        if not unreal.EditorLoadingAndSavingUtils.save_dirty_packages(
            save_map_packages=True, save_content_packages=True
        ):
            raise RuntimeError("one or more dirty UE packages failed to save")
        report["status"] = "assets_saved_screenshot_pending"
        _write_report(report)

        screenshot_path = DEPLOYMENT_ROOT / "evidence/MaterialLab_AB_Fixed.png"
        screenshot_path.parent.mkdir(parents=True, exist_ok=True)

        unreal.AutomationLibrary.finish_loading_before_screenshot()
        task = unreal.AutomationLibrary.take_high_res_screenshot(
            1600,
            900,
            screenshot_path.as_posix(),
            camera=camera,
            mask_enabled=False,
            capture_hdr=False,
            comparison_tolerance=unreal.ComparisonTolerance.LOW,
            comparison_notes="SciFiHelmet reference/compressed faithful RGBA8 A/B",
            delay=1.0,
            force_game_view=True,
        )
        if not task.is_valid_task():
            raise RuntimeError("UE rejected the fixed A/B screenshot task")
        report["screenshot"] = {
            "requested_path": screenshot_path.as_posix(),
            "exists_at_request": screenshot_path.is_file(),
            "task_valid": True,
            "task_done_at_request": bool(task.is_task_done()),
        }
        report["status"] = "complete_screenshot_requested"
        _write_report(report)
        unreal.log(f"CGCompression setup complete; screenshot requested: {REPORT_PATH.as_posix()}")

    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        _write_report(report)
        unreal.log_error(f"CGCompression setup failed: {exc}")
        raise


setup()
