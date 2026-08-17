"""Import and validate the isolated SciFiHelmet R0b hybrid UE assets.

This script intentionally owns only /HybridInterpolation/R0B assets and an
independent acceptance map.  It never edits the frozen reference or baseline.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import traceback

import unreal


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[4]
DEPLOYMENT_ROOT = REPO_ROOT / "outputs/deployment/scifihelmet/hybrid_direct_scalars_r0b"
MANIFEST_PATH = DEPLOYMENT_ROOT / "deployment_manifest.json"
HLSL_PATH = DEPLOYMENT_ROOT / "M_SciFiHelmet_Hybrid_R0B.custom.hlsl"
TEXTURE_A_SOURCE = DEPLOYMENT_ROOT / "T_SciFiHelmet_Hybrid_R0B_A_RGBA8.png"
TEXTURE_B_SOURCE = DEPLOYMENT_ROOT / "T_SciFiHelmet_Hybrid_R0B_B_RGB8.png"
EVIDENCE_ROOT = DEPLOYMENT_ROOT / "ue_evidence"
REPORT_PATH = EVIDENCE_ROOT / "ue_setup_report.json"

SOURCE_MAP = "/Game/CGCompression/Maps/MaterialLab"
ACCEPTANCE_MAP = "/Game/CGCompression/Maps/HybridInterpolation/R0B_Acceptance"
TEXTURE_FOLDER = "/Game/CGCompression/Textures/HybridInterpolation/R0B"
TEXTURE_A_PATH = f"{TEXTURE_FOLDER}/T_SciFiHelmet_Hybrid_R0B_A_RGBA8"
TEXTURE_B_PATH = f"{TEXTURE_FOLDER}/T_SciFiHelmet_Hybrid_R0B_B_RGB8"
MATERIAL_FOLDER = "/Game/CGCompression/Materials/HybridInterpolation/R0B"
MATERIAL_PATH = f"{MATERIAL_FOLDER}/M_SciFiHelmet_Hybrid_R0B"

EXPECTED = {
    "manifest": "ef87415f97bcb214ae76eb1456c45139e5a7987cb282f8bcadb0f8fe018721b6",
    "texture_a": "4d99be439c7579bc6c42de0f1b9fe48f89fd9a1f0d2a8e9d3e17060fab81e914",
    "texture_b": "70b4fe1289542bd64e45a43ea0ea87e76674586552b0a9ca1af54853e6b01cc1",
    "decoder": "e63a7afd4640fd330b3cdbcfe85121abc477b5f6c0024f1c31a4fdfc57e18b6c",
    "hlsl": "3dd95d5d220b269f7a44315d4230cabe4a746d398f54168d2c08926efe9c7023",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_report(value: dict) -> None:
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _validate_package() -> tuple[dict, str]:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest["selection"] != "r0b_hybrid_direct_scalars_offline_13_of_13":
        raise RuntimeError("deployment manifest is not the offline 13/13 R0b winner")
    if manifest["formal_holdout_accessed"] is not False:
        raise RuntimeError("formal holdout guard is not false")
    if manifest["runtime"] != {
        "ambient_occlusion": "excluded",
        "base_color": "TextureA.rgb direct linear; no decoder/sigmoid/sRGB",
        "metallic": "TextureB.b direct linear",
        "normal": "float2(TextureA.a,TextureB.r)->2x6x2->tanh->+Z->UE Y*-1 once",
        "roughness": "TextureB.g direct linear",
        "same_uv_wrap_bilinear": True,
        "texture_samples": 2,
    }:
        raise RuntimeError("deployment runtime contract changed")
    checks = {
        "texture_a": _sha256(TEXTURE_A_SOURCE),
        "texture_b": _sha256(TEXTURE_B_SOURCE),
        "hlsl": _sha256(HLSL_PATH),
    }
    for key, actual in checks.items():
        if actual != EXPECTED[key]:
            raise RuntimeError(f"{key} hash mismatch: {actual}")
    if manifest["inputs"]["decoder"]["sha256"] != EXPECTED["decoder"]:
        raise RuntimeError("decoder hash mismatch")
    return manifest, HLSL_PATH.read_text(encoding="utf-8")


def _ensure_directories() -> None:
    for path in (TEXTURE_FOLDER, MATERIAL_FOLDER, "/Game/CGCompression/Maps/HybridInterpolation"):
        if not unreal.EditorAssetLibrary.does_directory_exist(path):
            if not unreal.EditorAssetLibrary.make_directory(path):
                raise RuntimeError(f"could not create content directory: {path}")


def _import_texture(source: Path, asset_path: str, asset_name: str) -> unreal.Texture2D:
    texture = unreal.EditorAssetLibrary.load_asset(asset_path)
    if texture is None:
        task = unreal.AssetImportTask()
        task.set_editor_property("filename", source.as_posix())
        task.set_editor_property("destination_path", TEXTURE_FOLDER)
        task.set_editor_property("destination_name", asset_name)
        task.set_editor_property("automated", True)
        task.set_editor_property("replace_existing", False)
        task.set_editor_property("save", False)
        unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
        texture = unreal.EditorAssetLibrary.load_asset(asset_path)
    if not isinstance(texture, unreal.Texture2D):
        raise RuntimeError(f"texture import failed: {asset_path}")
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
    return texture


def _safe_probe(obj, name: str, *args):
    try:
        value = getattr(obj, name)(*args)
        return {"available": True, "value": str(value)}
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def _texture_report(texture: unreal.Texture2D) -> dict:
    report = {
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
        "virtual_texture_streaming": bool(texture.get_editor_property("virtual_texture_streaming")),
        "physical_rgba8_ceiling_bytes": int(texture.blueprint_get_size_x())
        * int(texture.blueprint_get_size_y())
        * 4,
    }
    report["api_probes"] = {
        "get_pixel_format": _safe_probe(texture, "get_pixel_format"),
        "get_resource_size_bytes": _safe_probe(texture, "get_resource_size_bytes"),
        "get_source_disk_and_memory_size": _safe_probe(texture, "get_source_disk_and_memory_size"),
    }
    try:
        report["api_probes"]["calc_texture_memory_size_all_mips"] = _safe_probe(
            texture, "calc_texture_memory_size_enum", unreal.TextureMipCount.TMC_ALL_MIPS
        )
    except Exception as exc:
        report["api_probes"]["calc_texture_memory_size_all_mips"] = {
            "available": False,
            "error": str(exc),
        }
    return report


def _export_texture_readback(texture: unreal.Texture2D, filename: Path) -> dict:
    filename.parent.mkdir(parents=True, exist_ok=True)
    task = unreal.AssetExportTask()
    task.set_editor_property("object", texture)
    task.set_editor_property("filename", filename.as_posix())
    task.set_editor_property("automated", True)
    task.set_editor_property("prompt", False)
    task.set_editor_property("replace_identical", True)
    task.set_editor_property("write_empty_files", False)
    task.set_editor_property("exporter", unreal.TextureExporterPNG())
    success = bool(unreal.Exporter.run_asset_export_task(task))
    return {
        "path": filename.as_posix(),
        "export_success": success,
        "exists": filename.is_file(),
        "sha256": _sha256(filename) if filename.is_file() else None,
        "file_bytes": filename.stat().st_size if filename.is_file() else None,
    }


def _create_material(texture_a: unreal.Texture2D, texture_b: unreal.Texture2D, hlsl: str):
    material = unreal.EditorAssetLibrary.load_asset(MATERIAL_PATH)
    if material is None:
        material = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
            "M_SciFiHelmet_Hybrid_R0B",
            MATERIAL_FOLDER,
            unreal.Material,
            unreal.MaterialFactoryNew(),
        )
    if not isinstance(material, unreal.Material):
        raise RuntimeError(f"candidate material is invalid: {MATERIAL_PATH}")
    material.set_editor_property("material_domain", unreal.MaterialDomain.MD_SURFACE)
    material.set_editor_property("blend_mode", unreal.BlendMode.BLEND_OPAQUE)
    material.set_editor_property("two_sided", False)
    unreal.MaterialEditingLibrary.delete_all_material_expressions(material)

    uv = unreal.MaterialEditingLibrary.create_material_expression(
        material, unreal.MaterialExpressionTextureCoordinate, -1250, -80
    )
    uv.set_editor_property("coordinate_index", 0)
    sample_a = unreal.MaterialEditingLibrary.create_material_expression(
        material, unreal.MaterialExpressionTextureSample, -950, -180
    )
    sample_b = unreal.MaterialEditingLibrary.create_material_expression(
        material, unreal.MaterialExpressionTextureSample, -950, 80
    )
    for sample, texture in ((sample_a, texture_a), (sample_b, texture_b)):
        sample.set_editor_property("texture", texture)
        sample.set_editor_property(
            "sampler_type", unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR
        )
    normal_y_sign = unreal.MaterialEditingLibrary.create_material_expression(
        material, unreal.MaterialExpressionConstant, -650, 280
    )
    normal_y_sign.set_editor_property("r", -1.0)
    custom = unreal.MaterialEditingLibrary.create_material_expression(
        material, unreal.MaterialExpressionCustom, -400, -80
    )
    custom.set_editor_property(
        "description", "R0b hybrid: direct BaseColor/roughness/metallic; normal-only 2->6->2"
    )
    custom.set_editor_property("code", hlsl)
    custom.set_editor_property("output_type", unreal.CustomMaterialOutputType.CMOT_FLOAT3)
    inputs = []
    for name in ("TextureA", "TextureB", "NormalYSign"):
        custom_input = unreal.CustomInput()
        custom_input.set_editor_property("input_name", name)
        inputs.append(custom_input)
    custom.set_editor_property("inputs", inputs)
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

    connections = {
        "one_uv_to_a": unreal.MaterialEditingLibrary.connect_material_expressions(uv, "", sample_a, "UVs"),
        "one_uv_to_b": unreal.MaterialEditingLibrary.connect_material_expressions(uv, "", sample_b, "UVs"),
        "a_to_decoder": unreal.MaterialEditingLibrary.connect_material_expressions(sample_a, "RGBA", custom, "TextureA"),
        "b_to_decoder": unreal.MaterialEditingLibrary.connect_material_expressions(sample_b, "RGB", custom, "TextureB"),
        "normal_y_once": unreal.MaterialEditingLibrary.connect_material_expressions(normal_y_sign, "", custom, "NormalYSign"),
        "base_color": unreal.MaterialEditingLibrary.connect_material_property(custom, "", unreal.MaterialProperty.MP_BASE_COLOR),
        "normal": unreal.MaterialEditingLibrary.connect_material_property(custom, "NormalTS", unreal.MaterialProperty.MP_NORMAL),
        "roughness": unreal.MaterialEditingLibrary.connect_material_property(custom, "Roughness", unreal.MaterialProperty.MP_ROUGHNESS),
        "metallic": unreal.MaterialEditingLibrary.connect_material_property(custom, "Metallic", unreal.MaterialProperty.MP_METALLIC),
    }
    if not all(connections.values()):
        raise RuntimeError(f"candidate material connection failure: {connections}")
    unreal.MaterialEditingLibrary.recompile_material(material)
    material.modify()
    if not unreal.EditorAssetLibrary.save_loaded_asset(material, only_if_is_dirty=False):
        raise RuntimeError(f"could not save candidate material: {MATERIAL_PATH}")
    return material, {
        "asset": material.get_path_name(),
        "connections": connections,
        "expression_count": len(unreal.MaterialEditingLibrary.get_material_expressions(material)),
        "ao_connected": False,
        "base_color_direct_in_hlsl": "return TextureA.rgb;" in hlsl,
        "direct_scalars_in_hlsl": "Roughness = TextureB.g;" in hlsl
        and "Metallic = TextureB.b;" in hlsl,
        "sigmoid_absent": "sigmoid" not in hlsl.lower(),
    }


def _component(actor):
    value = actor.get_component_by_class(unreal.StaticMeshComponent)
    return value if isinstance(value, unreal.StaticMeshComponent) else None


def _ensure_acceptance_map(material: unreal.Material) -> tuple[unreal.Actor | None, list[unreal.CameraActor], dict]:
    if not unreal.EditorAssetLibrary.does_asset_exist(ACCEPTANCE_MAP):
        if not unreal.EditorAssetLibrary.duplicate_asset(SOURCE_MAP, ACCEPTANCE_MAP):
            raise RuntimeError("could not duplicate MaterialLab into isolated R0b acceptance map")
        # UE 5.8 can retain the duplicated UWorld as a standalone reference and
        # fatally reject an immediate LoadLevel in the same process.  Persist it
        # now; the next clean process owns scene authoring.
        if not unreal.EditorAssetLibrary.save_asset(ACCEPTANCE_MAP, only_if_is_dirty=False):
            raise RuntimeError("could not persist duplicated R0b acceptance map")
        return None, [], {
            "map": ACCEPTANCE_MAP,
            "map_duplicated": True,
            "fresh_process_required": True,
        }
    level_editor = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    if not level_editor.load_level(ACCEPTANCE_MAP):
        raise RuntimeError(f"could not load isolated map: {ACCEPTANCE_MAP}")
    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actors = list(actor_subsystem.get_all_level_actors())
    source = next((a for a in actors if a.get_actor_label() == "Helmet_Reference"), None)
    if source is None:
        source = next((a for a in actors if a.get_actor_label() == "Helmet_Compressed"), None)
    if source is None or _component(source) is None:
        raise RuntimeError("isolated map has no source SciFiHelmet actor")
    capture = next((a for a in actors if a.get_actor_label() == "Helmet_R0B_SequentialCapture"), None)
    if capture is None:
        duplicated = list(actor_subsystem.duplicate_actors([source], None, unreal.Vector()))
        if len(duplicated) != 1:
            raise RuntimeError("could not create sequential R0b capture actor")
        capture = duplicated[0]
    capture.set_actor_label("Helmet_R0B_SequentialCapture", mark_dirty=True)
    capture.set_actor_transform(source.get_actor_transform(), False, False)
    component = _component(capture)
    component.set_material(0, material)
    capture.modify()

    origin, extent = capture.get_actor_bounds(False, True)
    radius = max(float(extent.x), float(extent.y), float(extent.z))
    # All three are close-up crops from the well-lit front-left side.  Targets
    # are offset in actor-local world axes to centre the declared diagnostic
    # features rather than merely showing three generic helmet views.
    camera_specs = [
        (
            "Camera_R0B_D1_MetallicBoundary",
            unreal.Vector(-2.65, 1.75, 0.62),
            origin + unreal.Vector(0.0, radius * 0.16, radius * 0.18),
            25.0,
        ),
        (
            "Camera_R0B_D2_YellowTube",
            unreal.Vector(-2.50, 1.65, 0.45),
            origin + unreal.Vector(0.0, 0.0, radius * -0.34),
            24.0,
        ),
        (
            "Camera_R0B_D3_GrayPanel",
            unreal.Vector(-2.70, 1.80, 0.62),
            origin + unreal.Vector(0.0, radius * -0.12, radius * 0.05),
            27.0,
        ),
    ]
    cameras = []
    camera_report = []
    for label, direction, target, fov in camera_specs:
        location = origin + direction * radius
        rotation = unreal.MathLibrary.find_look_at_rotation(location, target)
        camera = next(
            (a for a in actor_subsystem.get_all_level_actors() if isinstance(a, unreal.CameraActor) and a.get_actor_label() == label),
            None,
        )
        if camera is None:
            camera = actor_subsystem.spawn_actor_from_class(unreal.CameraActor, location, rotation)
        camera.set_actor_label(label, mark_dirty=True)
        camera.set_actor_location(location, False, False)
        camera.set_actor_rotation(rotation, False)
        camera_component = camera.get_component_by_class(unreal.CameraComponent)
        if not isinstance(camera_component, unreal.CameraComponent):
            raise RuntimeError(f"camera has no CameraComponent: {label}")
        camera_component.set_editor_property("field_of_view", fov)
        camera.modify()
        cameras.append(camera)
        camera_report.append({
            "label": label,
            "location": str(location),
            "rotation": str(rotation),
            "field_of_view": fov,
        })
    return capture, cameras, {
        "map": ACCEPTANCE_MAP,
        "capture_actor": capture.get_actor_label(),
        "capture_transform": str(capture.get_actor_transform()),
        "cameras": camera_report,
    }


def setup() -> None:
    report = {
        "schema_version": 1,
        "status": "started",
        "formal_holdout_accessed": False,
        "reference_and_baseline_preserved": True,
    }
    try:
        manifest, hlsl = _validate_package()
        report["deployment_manifest"] = manifest
        _ensure_directories()
        texture_a = _import_texture(
            TEXTURE_A_SOURCE, TEXTURE_A_PATH, "T_SciFiHelmet_Hybrid_R0B_A_RGBA8"
        )
        texture_b = _import_texture(
            TEXTURE_B_SOURCE, TEXTURE_B_PATH, "T_SciFiHelmet_Hybrid_R0B_B_RGB8"
        )
        report["textures"] = {
            "a": _texture_report(texture_a),
            "b": _texture_report(texture_b),
        }
        report["readback"] = {
            "a": _export_texture_readback(texture_a, EVIDENCE_ROOT / "readback_texture_a.png"),
            "b": _export_texture_readback(texture_b, EVIDENCE_ROOT / "readback_texture_b.png"),
        }
        material, material_report = _create_material(texture_a, texture_b, hlsl)
        report["material"] = material_report
        _capture, _cameras, map_report = _ensure_acceptance_map(material)
        report["acceptance_scene"] = map_report
        if not unreal.EditorLoadingAndSavingUtils.save_dirty_packages(
            save_map_packages=True, save_content_packages=True
        ):
            raise RuntimeError("one or more isolated R0b UE packages failed to save")
        unreal.SystemLibrary.execute_console_command(None, "ListTextures")
        report["status"] = (
            "map_duplicated_fresh_process_required"
            if map_report.get("fresh_process_required")
            else "complete_readback_pending_external_compare"
        )
        _write_report(report)
        unreal.log(f"CGCompression R0b UE setup complete: {REPORT_PATH.as_posix()}")
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        _write_report(report)
        unreal.log_error(f"CGCompression R0b UE setup failed: {exc}")
        raise


setup()
