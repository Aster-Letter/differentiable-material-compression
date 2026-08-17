"""Create isolated C4/C5 decode-then-filter UE materials and preview actors."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import traceback

import unreal


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[4]
DEPLOYMENT_ROOT = REPO_ROOT / "outputs/deployment/scifihelmet/dtf_preview_v1"
PREVIEW_MANIFEST_PATH = DEPLOYMENT_ROOT / "preview_manifest.json"
REPORT_PATH = DEPLOYMENT_ROOT / "ue_setup_report.json"
EVIDENCE_ROOT = DEPLOYMENT_ROOT / "ue_evidence"

EXPECTED_PREVIEW_MANIFEST_SHA256 = (
    "50f6e80640e51c3dc3841569c8c87fbc3341a9b1fb4537bfff241e9fcdfa925e"
)
EXPECTED_TRAINING_MANIFESTS = {
    "c4_dtf_16_s080k": "0ea2adc0906e5ed3c3311a323252f7a7ad543b32b8ab4900a89246a9a3d28237",
    "c4_dtf_16_s160k": "fbb46b5505259364b3e790d3131a94d5e62a1e6e242ac5bf7f27336745de4f9b",
    "c5_dtf_16_s080k": "58fa74f38671987011941d0c1fd8c0fd1890438b17763d1ca8c0a1d0ff850ce7",
    "c5_dtf_16_s120k": "0ba3797d9528c9a8bad77ac16e508ca480b7cbb29e4cb1c94121e796774c1bcb",
}

ASSET_ROOT = "/Game/CGCompression/DTFPreview"
SOURCE_MAP = "/Game/CGCompression/Maps/MaterialLab"
PREVIEW_MAP = f"{ASSET_ROOT}/Maps/MaterialLab_DTF_Preview"


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
        raise RuntimeError("DTF preview manifest hash mismatch")
    preview = json.loads(PREVIEW_MANIFEST_PATH.read_text(encoding="utf-8"))
    if (
        preview.get("status") != "exported_ue_setup_pending"
        or preview.get("formal_holdout_accessed") is not False
        or set(preview.get("candidates", {}))
        != {
            "c4_dtf_16_s080k",
            "c4_dtf_16_s160k",
            "c5_dtf_16_s080k",
            "c5_dtf_16_s120k",
        }
    ):
        raise RuntimeError("DTF preview manifest is not the frozen step-qualified package")
    packages: dict[str, dict] = {}
    for name, record in preview["candidates"].items():
        if record["source"]["training_manifest_sha256"] != EXPECTED_TRAINING_MANIFESTS[name]:
            raise RuntimeError(f"unexpected frozen training manifest: {name}")
        package_root = DEPLOYMENT_ROOT / record["package_directory"]
        deployment_path = package_root / record["deployment_manifest"]
        if _sha256(deployment_path) != record["deployment_manifest_sha256"]:
            raise RuntimeError(f"DTF deployment manifest hash mismatch: {name}")
        deployment = json.loads(deployment_path.read_text(encoding="utf-8"))
        if (
            deployment.get("candidate") != name
            or deployment.get("formal_holdout_accessed") is not False
            or deployment["runtime"]["filter_order"]
            != "four_point_fetches_per_resource -> per_corner_decode_postprocess -> material_bilinear_filter -> one_normalize"
        ):
            raise RuntimeError(f"DTF deployment contract mismatch: {name}")
        for filename, expected in deployment["generated_files"].items():
            if _sha256(package_root / filename) != expected:
                raise RuntimeError(f"DTF generated file hash mismatch: {name}/{filename}")
        packages[name] = {
            "record": deployment,
            "root": package_root,
            "hlsl": next(package_root.glob("M_*.custom.hlsl")),
            "rgba": next(package_root.glob("T_*_RGBA8.png")),
            "r": next(iter(package_root.glob("T_*_R8.png")), None),
        }
    return preview, packages


def _ensure_directories(packages: dict[str, dict]) -> None:
    paths = {f"{ASSET_ROOT}/Materials", f"{ASSET_ROOT}/Maps"}
    for package in packages.values():
        paths.add(package["record"]["ue_assets"]["texture_folder"])
    for path in sorted(paths):
        if not unreal.EditorAssetLibrary.does_directory_exist(path):
            if not unreal.EditorAssetLibrary.make_directory(path):
                raise RuntimeError(f"could not create DTF preview directory: {path}")


def _import_texture(
    source: Path,
    asset_path: str,
    *,
    grayscale: bool,
) -> unreal.Texture2D:
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
        raise RuntimeError(f"DTF texture import failed: {asset_path}")
    texture.set_editor_property("srgb", False)
    texture.set_editor_property(
        "compression_settings",
        unreal.TextureCompressionSettings.TC_GRAYSCALE
        if grayscale
        else unreal.TextureCompressionSettings.TC_VECTOR_DISPLACEMENTMAP,
    )
    texture.set_editor_property("filter", unreal.TextureFilter.TF_NEAREST)
    texture.set_editor_property("address_x", unreal.TextureAddress.TA_WRAP)
    texture.set_editor_property("address_y", unreal.TextureAddress.TA_WRAP)
    texture.set_editor_property(
        "mip_gen_settings", unreal.TextureMipGenSettings.TMGS_NO_MIPMAPS
    )
    texture.set_editor_property("never_stream", True)
    texture.set_editor_property("virtual_texture_streaming", False)
    texture.modify()
    if not unreal.EditorAssetLibrary.save_loaded_asset(texture, only_if_is_dirty=False):
        raise RuntimeError(f"could not save owned DTF texture: {asset_path}")
    return texture


def _safe_probe(obj, name: str, *args) -> dict:
    try:
        return {"available": True, "value": str(getattr(obj, name)(*args))}
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
        "virtual_texture_streaming": bool(
            texture.get_editor_property("virtual_texture_streaming")
        ),
    }
    report["api_probes"] = {
        "get_pixel_format": _safe_probe(texture, "get_pixel_format"),
        "get_resource_size_bytes": _safe_probe(texture, "get_resource_size_bytes"),
        "get_source_disk_and_memory_size": _safe_probe(
            texture, "get_source_disk_and_memory_size"
        ),
    }
    try:
        report["api_probes"]["calc_texture_memory_size_all_mips"] = _safe_probe(
            texture,
            "calc_texture_memory_size_enum",
            unreal.TextureMipCount.TMC_ALL_MIPS,
        )
    except Exception as exc:
        report["api_probes"]["calc_texture_memory_size_all_mips"] = {
            "available": False,
            "error": str(exc),
        }
    return report


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


def _create_material(
    candidate: str,
    package: dict,
    rgba: unreal.Texture2D,
    red: unreal.Texture2D | None,
) -> tuple[unreal.Material, dict]:
    record = package["record"]
    material_path = record["ue_assets"]["material"]
    folder, name = material_path.rsplit("/", 1)
    material = unreal.EditorAssetLibrary.load_asset(material_path)
    if material is None:
        material = unreal.AssetToolsHelpers.get_asset_tools().create_asset(
            name, folder, unreal.Material, unreal.MaterialFactoryNew()
        )
    if not isinstance(material, unreal.Material):
        raise RuntimeError(f"DTF material asset is invalid: {material_path}")
    material.set_editor_property("material_domain", unreal.MaterialDomain.MD_SURFACE)
    material.set_editor_property("blend_mode", unreal.BlendMode.BLEND_OPAQUE)
    material.set_editor_property("two_sided", False)
    unreal.MaterialEditingLibrary.delete_all_material_expressions(material)

    uv = unreal.MaterialEditingLibrary.create_material_expression(
        material, unreal.MaterialExpressionTextureCoordinate, -1350, -180
    )
    uv.set_editor_property("coordinate_index", 0)
    rgba_object = unreal.MaterialEditingLibrary.create_material_expression(
        material, unreal.MaterialExpressionTextureObject, -1100, -80
    )
    rgba_object.set_editor_property("texture", rgba)
    red_object = None
    if red is not None:
        red_object = unreal.MaterialEditingLibrary.create_material_expression(
            material, unreal.MaterialExpressionTextureObject, -1100, 100
        )
        red_object.set_editor_property("texture", red)
    normal_sign = unreal.MaterialEditingLibrary.create_material_expression(
        material, unreal.MaterialExpressionConstant, -850, 280
    )
    normal_sign.set_editor_property("r", -1.0)
    custom = unreal.MaterialEditingLibrary.create_material_expression(
        material, unreal.MaterialExpressionCustom, -500, -100
    )
    custom.set_editor_property(
        "description",
        f"{candidate}: four LOD0 point corners -> decode/postprocess -> material filter",
    )
    custom.set_editor_property("code", package["hlsl"].read_text(encoding="utf-8"))
    custom.set_editor_property("output_type", unreal.CustomMaterialOutputType.CMOT_FLOAT3)
    inputs = [_custom_input("UV"), _custom_input("LatentRGBA")]
    if red is not None:
        inputs.append(_custom_input("LatentR"))
    inputs.append(_custom_input("NormalYSign"))
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
        "uv": unreal.MaterialEditingLibrary.connect_material_expressions(
            uv, "", custom, "UV"
        ),
        "rgba_texture_object": unreal.MaterialEditingLibrary.connect_material_expressions(
            rgba_object, "", custom, "LatentRGBA"
        ),
        "normal_y_once": unreal.MaterialEditingLibrary.connect_material_expressions(
            normal_sign, "", custom, "NormalYSign"
        ),
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
    if red_object is not None:
        connections["r_texture_object"] = (
            unreal.MaterialEditingLibrary.connect_material_expressions(
                red_object, "", custom, "LatentR"
            )
        )
    if not all(connections.values()):
        raise RuntimeError(f"DTF material connection failure: {candidate}/{connections}")
    unreal.MaterialEditingLibrary.recompile_material(material)
    material.modify()
    if not unreal.EditorAssetLibrary.save_loaded_asset(material, only_if_is_dirty=False):
        raise RuntimeError(f"could not save owned DTF material: {material_path}")
    return material, {
        "asset": material.get_path_name(),
        "connections": connections,
        "expression_count": len(
            unreal.MaterialEditingLibrary.get_material_expressions(material)
        ),
        "texture_object_count": 2 if red is not None else 1,
        "normal_y_sign": -1.0,
        "ao_connected": False,
    }


def _component(actor) -> unreal.StaticMeshComponent | None:
    value = actor.get_component_by_class(unreal.StaticMeshComponent)
    return value if isinstance(value, unreal.StaticMeshComponent) else None


def _find_reference(actors: list[unreal.Actor]) -> unreal.Actor:
    reference = next((a for a in actors if a.get_actor_label() == "Helmet_Reference"), None)
    if reference is not None and _component(reference) is not None:
        return reference
    for actor in actors:
        component = _component(actor)
        mesh = component.get_editor_property("static_mesh") if component else None
        if mesh is not None and "SciFiHelmet" in mesh.get_path_name():
            return actor
    raise RuntimeError("stored MaterialLab source has no SciFiHelmet reference actor")


def _migrate_label_if_unclaimed(
    actors: list[unreal.Actor], old_label: str, new_label: str
) -> None:
    """Retain an owned preview object while replacing its ambiguous legacy label."""

    if any(value.get_actor_label() == new_label for value in actors):
        return
    value = next(
        (item for item in actors if item.get_actor_label() == old_label), None
    )
    if value is not None:
        value.set_actor_label(new_label, mark_dirty=True)
        value.modify()


def _ensure_actor(
    actor_subsystem,
    actors: list[unreal.Actor],
    reference: unreal.Actor,
    *,
    label: str,
    material: unreal.Material,
    offset: unreal.Vector,
) -> unreal.Actor:
    actor = next((value for value in actors if value.get_actor_label() == label), None)
    if actor is None:
        actor = actor_subsystem.spawn_actor_from_class(
            unreal.StaticMeshActor,
            reference.get_actor_location(),
            reference.get_actor_rotation(),
        )
        if not isinstance(actor, unreal.StaticMeshActor):
            raise RuntimeError(f"could not spawn DTF preview actor: {label}")
    actor.set_actor_transform(reference.get_actor_transform(), False, False)
    actor.add_actor_world_offset(offset, False, False)
    actor.set_actor_label(label, mark_dirty=True)
    actor.set_actor_rotation(reference.get_actor_rotation(), False)
    actor.set_actor_scale3d(reference.get_actor_scale3d())
    component = _component(actor)
    reference_component = _component(reference)
    if component is None or reference_component is None:
        raise RuntimeError(f"DTF preview actor has no StaticMeshComponent: {label}")
    component.set_static_mesh(reference_component.get_editor_property("static_mesh"))
    component.set_material(0, material)
    actor.modify()
    return actor


def _ensure_camera(
    actor_subsystem,
    *,
    label: str,
    target_actor: unreal.Actor,
    direction: unreal.Vector,
    target_offset: unreal.Vector,
    field_of_view: float,
) -> unreal.CameraActor:
    origin, extent = target_actor.get_actor_bounds(False, True)
    radius = max(float(extent.x), float(extent.y), float(extent.z))
    location = origin + direction * radius
    target = origin + target_offset * radius
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
        camera = actor_subsystem.spawn_actor_from_class(
            unreal.CameraActor, location, rotation
        )
    camera.set_actor_label(label, mark_dirty=True)
    camera.set_actor_location(location, False, False)
    camera.set_actor_rotation(rotation, False)
    component = camera.get_component_by_class(unreal.CameraComponent)
    if not isinstance(component, unreal.CameraComponent):
        raise RuntimeError(f"DTF preview camera has no CameraComponent: {label}")
    component.set_editor_property("field_of_view", field_of_view)
    camera.modify()
    return camera


def _ensure_preview_map(materials: dict[str, unreal.Material]) -> dict:
    if not unreal.EditorAssetLibrary.does_asset_exist(PREVIEW_MAP):
        if not unreal.EditorAssetLibrary.duplicate_asset(SOURCE_MAP, PREVIEW_MAP):
            raise RuntimeError("could not duplicate stored MaterialLab into isolated DTF preview map")
        if not unreal.EditorAssetLibrary.save_asset(PREVIEW_MAP, only_if_is_dirty=False):
            raise RuntimeError("could not save isolated DTF preview map")
        return {
            "map": PREVIEW_MAP,
            "source_map": SOURCE_MAP,
            "source_map_saved": False,
            "map_duplicated": True,
            "fresh_process_required": True,
        }
    level_editor = unreal.get_editor_subsystem(unreal.LevelEditorSubsystem)
    if not level_editor.load_level(PREVIEW_MAP):
        raise RuntimeError(f"could not load isolated DTF preview map: {PREVIEW_MAP}")
    actor_subsystem = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)
    actors = list(actor_subsystem.get_all_level_actors())
    reference = _find_reference(actors)
    _migrate_label_if_unclaimed(
        actors, "Helmet_C4_DTF16", "Helmet_C4_DTF16_S080K"
    )
    _migrate_label_if_unclaimed(
        actors, "Helmet_C5_DTF16", "Helmet_C5_DTF16_S120K"
    )
    extent = reference.get_actor_bounds(False, True)[1]
    separation = max(float(extent.x), float(extent.y), float(extent.z)) * 2.8
    c4_80k = _ensure_actor(
        actor_subsystem,
        actors,
        reference,
        label="Helmet_C4_DTF16_S080K",
        material=materials["c4_dtf_16_s080k"],
        offset=unreal.Vector(0.0, separation, 0.0),
    )
    c4_160k = _ensure_actor(
        actor_subsystem,
        actors,
        reference,
        label="Helmet_C4_DTF16_S160K",
        material=materials["c4_dtf_16_s160k"],
        offset=unreal.Vector(0.0, separation * 2.0, 0.0),
    )
    c5_80k = _ensure_actor(
        actor_subsystem,
        actors,
        reference,
        label="Helmet_C5_DTF16_S080K",
        material=materials["c5_dtf_16_s080k"],
        offset=unreal.Vector(0.0, separation * 3.0, 0.0),
    )
    c5_120k = _ensure_actor(
        actor_subsystem,
        actors,
        reference,
        label="Helmet_C5_DTF16_S120K",
        material=materials["c5_dtf_16_s120k"],
        offset=unreal.Vector(0.0, separation * 4.0, 0.0),
    )
    camera_specs = [
        ("D1_Metallic", unreal.Vector(-2.65, 1.75, 0.62), unreal.Vector(0.0, 0.16, 0.18), 25.0),
        ("D2_YellowTube", unreal.Vector(-2.50, 1.65, 0.45), unreal.Vector(0.0, 0.0, -0.34), 24.0),
        ("D3_GrayPanel", unreal.Vector(-2.70, 1.80, 0.62), unreal.Vector(0.0, -0.12, 0.05), 27.0),
    ]
    for suffix, _direction, _target_offset, _fov in camera_specs:
        _migrate_label_if_unclaimed(
            actors,
            f"Camera_C4_DTF16_{suffix}",
            f"Camera_C4_DTF16_S080K_{suffix}",
        )
        _migrate_label_if_unclaimed(
            actors,
            f"Camera_C5_DTF16_{suffix}",
            f"Camera_C5_DTF16_S120K_{suffix}",
        )
    cameras = []
    for candidate_name, actor in (
        ("C4_DTF16_S080K", c4_80k),
        ("C4_DTF16_S160K", c4_160k),
        ("C5_DTF16_S080K", c5_80k),
        ("C5_DTF16_S120K", c5_120k),
    ):
        for suffix, direction, target_offset, fov in camera_specs:
            camera = _ensure_camera(
                actor_subsystem,
                label=f"Camera_{candidate_name}_{suffix}",
                target_actor=actor,
                direction=direction,
                target_offset=target_offset,
                field_of_view=fov,
            )
            cameras.append(camera.get_actor_label())
    if not unreal.EditorAssetLibrary.save_asset(PREVIEW_MAP, only_if_is_dirty=False):
        raise RuntimeError("could not save isolated DTF preview map")
    return {
        "map": PREVIEW_MAP,
        "source_map": SOURCE_MAP,
        "source_map_saved": False,
        "map_duplicated": False,
        "fresh_process_required": False,
        "actors": {
            "reference": reference.get_actor_label(),
            "c4_80k": c4_80k.get_actor_label(),
            "c4_160k": c4_160k.get_actor_label(),
            "c5_80k": c5_80k.get_actor_label(),
            "c5_120k": c5_120k.get_actor_label(),
        },
        "cameras": cameras,
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
        report["preview_manifest"] = preview
        _ensure_directories(packages)
        materials = {}
        report["candidates"] = {}
        for name, package in packages.items():
            record = package["record"]
            rgba = _import_texture(
                package["rgba"], record["ue_assets"]["texture_rgba"], grayscale=False
            )
            red = None
            if package["r"] is not None:
                red = _import_texture(
                    package["r"], record["ue_assets"]["texture_r"], grayscale=True
                )
            material, material_report = _create_material(name, package, rgba, red)
            materials[name] = material
            textures = {"rgba": _texture_report(rgba)}
            readback = {
                "rgba": _export_readback(
                    rgba, EVIDENCE_ROOT / name / "readback_rgba.png"
                )
            }
            if red is not None:
                textures["r"] = _texture_report(red)
                readback["r"] = _export_readback(
                    red, EVIDENCE_ROOT / name / "readback_r.png"
                )
            report["candidates"][name] = {
                "textures": textures,
                "readback": readback,
                "material": material_report,
            }
        report["preview_scene"] = _ensure_preview_map(materials)
        unreal.SystemLibrary.execute_console_command(None, "ListTextures")
        report["status"] = (
            "map_duplicated_fresh_process_required"
            if report["preview_scene"].get("fresh_process_required")
            else "complete_ready_for_manual_preview"
        )
        _write_report(report)
        unreal.log(f"CGCompression DTF preview setup: {report['status']}")
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        _write_report(report)
        unreal.log_error(f"CGCompression DTF preview setup failed: {exc}")
        raise


setup()
