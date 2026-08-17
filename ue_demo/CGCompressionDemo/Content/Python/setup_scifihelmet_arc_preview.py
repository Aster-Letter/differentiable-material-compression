"""Import ARC snapshots and author an isolated early-versus-late preview map."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import traceback

import unreal


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[4]
DEPLOYMENT_ROOT = REPO_ROOT / "outputs/deployment/scifihelmet/arc_preview"
REPORT_PATH = DEPLOYMENT_ROOT / "ue_setup_report.json"

SOURCE_MAP = "/Game/CGCompression/Maps/MaterialLab"
PREVIEW_MAP = "/Game/CGCompression/Maps/ARCPreview/ARC_Ablation_10k_81k"
TEXTURE_FOLDER = "/Game/CGCompression/Textures/ARCPreview"
MATERIAL_FOLDER = "/Game/CGCompression/Materials/ARCPreview"

SNAPSHOTS = (
    {
        "selection": "arc_010k_early",
        "label": "ARC_010k",
        "actor": "Helmet_ARC_010k",
    },
    {
        "selection": "arc_081k_user_stopped",
        "label": "ARC_081k",
        "actor": "Helmet_ARC_081k",
    },
)


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


def _load_package(spec: dict) -> tuple[dict, Path, str]:
    root = DEPLOYMENT_ROOT / spec["selection"]
    manifest_path = root / "deployment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["selection"] != spec["selection"]:
        raise RuntimeError(f"selection mismatch: {spec['selection']}")
    if manifest["candidate"] != "arc_relu_fresh":
        raise RuntimeError("preview package is not ARC-ReLU-Fresh")
    if manifest["formal_holdout_accessed"] is not False:
        raise RuntimeError("formal holdout guard changed")
    if manifest["decoder"]["shape"] != "4->8->7":
        raise RuntimeError("preview decoder shape changed")
    latent_name = next(name for name in manifest["files"] if name.endswith("_RGBA8.png"))
    hlsl_name = next(name for name in manifest["files"] if name.endswith(".custom.hlsl"))
    latent_path = root / latent_name
    hlsl_path = root / hlsl_name
    if _sha256(latent_path) != manifest["files"][latent_name]["sha256"]:
        raise RuntimeError(f"latent hash mismatch: {spec['selection']}")
    if _sha256(hlsl_path) != manifest["files"][hlsl_name]:
        raise RuntimeError(f"HLSL hash mismatch: {spec['selection']}")
    return manifest, latent_path, hlsl_path.read_text(encoding="utf-8")


def _ensure_directories() -> None:
    for path in (
        TEXTURE_FOLDER,
        MATERIAL_FOLDER,
        "/Game/CGCompression/Maps/ARCPreview",
    ):
        if not unreal.EditorAssetLibrary.does_directory_exist(path):
            if not unreal.EditorAssetLibrary.make_directory(path):
                raise RuntimeError(f"could not create directory: {path}")


def _import_texture(source: Path, label: str) -> unreal.Texture2D:
    asset_name = f"T_SciFiHelmet_{label}_RGBA8"
    asset_path = f"{TEXTURE_FOLDER}/{asset_name}"
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


def _create_material(texture: unreal.Texture2D, hlsl: str, label: str) -> unreal.Material:
    asset_name = f"M_SciFiHelmet_{label}"
    asset_path = f"{MATERIAL_FOLDER}/{asset_name}"
    material = unreal.EditorAssetLibrary.load_asset(asset_path)
    if material is None:
        material = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
            asset_name, MATERIAL_FOLDER, unreal.Material, unreal.MaterialFactoryNew()
        )
    if not isinstance(material, unreal.Material):
        raise RuntimeError(f"material creation failed: {asset_path}")
    material.set_editor_property("material_domain", unreal.MaterialDomain.MD_SURFACE)
    material.set_editor_property("blend_mode", unreal.BlendMode.BLEND_OPAQUE)
    material.set_editor_property("two_sided", False)
    unreal.MaterialEditingLibrary.delete_all_material_expressions(material)

    uv = unreal.MaterialEditingLibrary.create_material_expression(
        material, unreal.MaterialExpressionTextureCoordinate, -1100, -100
    )
    uv.set_editor_property("coordinate_index", 0)
    sample = unreal.MaterialEditingLibrary.create_material_expression(
        material, unreal.MaterialExpressionTextureSample, -800, -100
    )
    sample.set_editor_property("texture", texture)
    sample.set_editor_property("sampler_type", unreal.MaterialSamplerType.SAMPLERTYPE_LINEAR_COLOR)
    normal_y = unreal.MaterialEditingLibrary.create_material_expression(
        material, unreal.MaterialExpressionConstant, -800, 220
    )
    normal_y.set_editor_property("r", -1.0)
    custom = unreal.MaterialEditingLibrary.create_material_expression(
        material, unreal.MaterialExpressionCustom, -350, -80
    )
    custom.set_editor_property("description", f"Diagnostic {label} ARC 4->8->7 ReLU")
    custom.set_editor_property("code", hlsl)
    custom.set_editor_property("output_type", unreal.CustomMaterialOutputType.CMOT_FLOAT3)
    inputs = []
    for name in ("LatentRGBA", "NormalYSign"):
        value = unreal.CustomInput()
        value.set_editor_property("input_name", name)
        inputs.append(value)
    custom.set_editor_property("inputs", inputs)
    outputs = []
    for name, output_type in (
        ("NormalTS", unreal.CustomMaterialOutputType.CMOT_FLOAT3),
        ("Roughness", unreal.CustomMaterialOutputType.CMOT_FLOAT1),
        ("Metallic", unreal.CustomMaterialOutputType.CMOT_FLOAT1),
    ):
        value = unreal.CustomOutput()
        value.set_editor_property("output_name", name)
        value.set_editor_property("output_type", output_type)
        outputs.append(value)
    custom.set_editor_property("additional_outputs", outputs)
    connections = (
        unreal.MaterialEditingLibrary.connect_material_expressions(uv, "", sample, "UVs"),
        unreal.MaterialEditingLibrary.connect_material_expressions(sample, "RGBA", custom, "LatentRGBA"),
        unreal.MaterialEditingLibrary.connect_material_expressions(normal_y, "", custom, "NormalYSign"),
        unreal.MaterialEditingLibrary.connect_material_property(custom, "", unreal.MaterialProperty.MP_BASE_COLOR),
        unreal.MaterialEditingLibrary.connect_material_property(custom, "NormalTS", unreal.MaterialProperty.MP_NORMAL),
        unreal.MaterialEditingLibrary.connect_material_property(custom, "Roughness", unreal.MaterialProperty.MP_ROUGHNESS),
        unreal.MaterialEditingLibrary.connect_material_property(custom, "Metallic", unreal.MaterialProperty.MP_METALLIC),
    )
    if not all(connections):
        raise RuntimeError(f"material connection failed: {asset_path}")
    unreal.MaterialEditingLibrary.recompile_material(material)
    material.modify()
    if not unreal.EditorAssetLibrary.save_loaded_asset(material, only_if_is_dirty=False):
        raise RuntimeError(f"could not save material: {asset_path}")
    return material


def _component(actor):
    value = actor.get_component_by_class(unreal.StaticMeshComponent)
    return value if isinstance(value, unreal.StaticMeshComponent) else None


def _author_preview_map(materials: list[tuple[dict, unreal.Material]]) -> dict:
    if not unreal.EditorAssetLibrary.does_asset_exist(PREVIEW_MAP):
        if not unreal.EditorAssetLibrary.duplicate_asset(SOURCE_MAP, PREVIEW_MAP):
            raise RuntimeError("could not duplicate MaterialLab into ARCPreview")
        if not unreal.EditorAssetLibrary.save_asset(PREVIEW_MAP, only_if_is_dirty=False):
            raise RuntimeError("could not save duplicated ARC preview map")
        return {"map": PREVIEW_MAP, "fresh_process_required": True}

    if not unreal.get_editor_subsystem(unreal.LevelEditorSubsystem).load_level(PREVIEW_MAP):
        raise RuntimeError(f"could not load preview map: {PREVIEW_MAP}")
    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actors = list(actor_subsystem.get_all_level_actors())
    source = next((actor for actor in actors if actor.get_actor_label() == "Helmet_Reference"), None)
    if source is None or _component(source) is None:
        raise RuntimeError("ARC preview map has no reference helmet")
    extent = source.get_actor_bounds(False, True)[1]
    separation = max(float(extent.x), float(extent.y), float(extent.z)) * 2.8
    authored = []
    for index, (spec, material) in enumerate(materials, start=1):
        actor = next((value for value in actors if value.get_actor_label() == spec["actor"]), None)
        offset = unreal.Vector(0.0, separation * index, 0.0)
        if actor is None:
            duplicated = list(actor_subsystem.duplicate_actors([source], None, offset))
            if len(duplicated) != 1:
                raise RuntimeError(f"could not duplicate actor: {spec['actor']}")
            actor = duplicated[0]
        else:
            actor.set_actor_transform(source.get_actor_transform(), False, False)
            actor.add_actor_world_offset(offset, False, False)
        actor.set_actor_label(spec["actor"], mark_dirty=True)
        component = _component(actor)
        component.set_material(0, material)
        actor.modify()
        authored.append({
            "actor": actor.get_actor_label(),
            "material": material.get_path_name(),
            "offset_cm": [0.0, separation * index, 0.0],
        })
    if not unreal.EditorLoadingAndSavingUtils.save_dirty_packages(
        save_map_packages=True, save_content_packages=True
    ):
        raise RuntimeError("could not save isolated ARC preview packages")
    return {"map": PREVIEW_MAP, "fresh_process_required": False, "actors": authored}


def setup() -> None:
    report = {
        "schema_version": 1,
        "status": "started",
        "formal_holdout_accessed": False,
        "reference_baseline_r0b_preserved": True,
        "diagnostic_only_not_winner": True,
    }
    try:
        _ensure_directories()
        materials = []
        records = []
        for spec in SNAPSHOTS:
            manifest, latent_path, hlsl = _load_package(spec)
            texture = _import_texture(latent_path, spec["label"])
            material = _create_material(texture, hlsl, spec["label"])
            materials.append((spec, material))
            records.append({
                "selection": spec["selection"],
                "checkpoint_step": manifest["checkpoint"]["actual_step"],
                "texture": texture.get_path_name(),
                "material": material.get_path_name(),
            })
        report["snapshots"] = records
        report["preview_scene"] = _author_preview_map(materials)
        report["status"] = (
            "map_duplicated_fresh_process_required"
            if report["preview_scene"]["fresh_process_required"]
            else "complete"
        )
        _write_report(report)
        unreal.log(f"ARC preview setup complete: {REPORT_PATH.as_posix()}")
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        _write_report(report)
        unreal.log_error(f"ARC preview setup failed: {exc}")
        raise


setup()
