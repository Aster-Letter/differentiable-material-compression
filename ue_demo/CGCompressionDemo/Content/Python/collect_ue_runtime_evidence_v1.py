"""Collect read-only UE 5.8 material and texture evidence for tsk:29ea0b."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import traceback

import unreal


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[4]
EVIDENCE_ROOT = REPO_ROOT / "outputs/analysis/ue-runtime-evidence-v1"
CONTRACT_PATH = EVIDENCE_ROOT / "measurement_contract.json"
EXPECTED_CONTRACT_SHA256 = (
    "652fd3761a3d8817e7f026eb067206098a12ad565c6677ab15b13ce679433bb2"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(name: str, value: dict) -> None:
    EVIDENCE_ROOT.mkdir(parents=True, exist_ok=True)
    (EVIDENCE_ROOT / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _safe_call(obj: object, method: str, *args: object) -> dict:
    fn = getattr(obj, method, None)
    if fn is None:
        return {"available": False, "error": "method not exposed"}
    try:
        value = fn(*args)
        if isinstance(value, tuple):
            value = [str(item) for item in value]
        elif not isinstance(value, (bool, int, float, str, list, dict, type(None))):
            value = str(value)
        return {"available": True, "value": value}
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def _texture_record(texture: unreal.Texture) -> dict:
    record = {
        "asset": texture.get_path_name(),
        "class": texture.get_class().get_name(),
        "resource_size_bytes": _safe_call(texture, "get_resource_size_bytes"),
        "pixel_format": _safe_call(texture, "get_pixel_format"),
        "source_disk_and_memory_size": _safe_call(
            texture, "get_source_disk_and_memory_size"
        ),
    }
    if isinstance(texture, unreal.Texture2D):
        mip_count_enum = getattr(unreal, "TextureMipCount", None)
        record.update(
            {
                "size_x": int(texture.blueprint_get_size_x()),
                "size_y": int(texture.blueprint_get_size_y()),
                "srgb": bool(texture.get_editor_property("srgb")),
                "compression_settings": str(
                    texture.get_editor_property("compression_settings")
                ),
                "filter": str(texture.get_editor_property("filter")),
                "address_x": str(texture.get_editor_property("address_x")),
                "address_y": str(texture.get_editor_property("address_y")),
                "mip_gen_settings": str(
                    texture.get_editor_property("mip_gen_settings")
                ),
                "never_stream": bool(texture.get_editor_property("never_stream")),
                "virtual_texture_streaming": bool(
                    texture.get_editor_property("virtual_texture_streaming")
                ),
                "all_mips_memory_bytes": (
                    _safe_call(
                        texture,
                        "calc_texture_memory_size_enum",
                        mip_count_enum.TMC_ALL_MIPS,
                    )
                    if mip_count_enum is not None
                    else {"available": False, "error": "TextureMipCount not exposed"}
                ),
                "resident_mips_memory_bytes": (
                    _safe_call(
                        texture,
                        "calc_texture_memory_size_enum",
                        mip_count_enum.TMC_RESIDENT_MIPS,
                    )
                    if mip_count_enum is not None
                    else {"available": False, "error": "TextureMipCount not exposed"}
                ),
            }
        )
    return record


def _stats_record(material: unreal.MaterialInterface) -> dict:
    stats = unreal.MaterialEditingLibrary.get_statistics(material)
    shaders = unreal.MaterialEditingLibrary.list_shaders(material)
    used_textures = unreal.MaterialEditingLibrary.get_material_used_textures(material)
    shader_rows = sorted(
        {
            (
                str(shader.get_editor_property("vertex_factory_name")),
                str(shader.get_editor_property("shader_type_name")),
            )
            for shader in shaders
        }
    )
    return {
        "material": material.get_path_name(),
        "current_rhi_shader_platform": "PCD3D_SM6",
        "current_material_quality": "High",
        "statistics_semantics": (
            "GetStatistics returns the maximum representative vertex/pixel shader "
            "instruction count for the current RHI shader platform and material quality."
        ),
        "num_vertex_shader_instructions": int(
            stats.get_editor_property("num_vertex_shader_instructions")
        ),
        "num_pixel_shader_instructions": int(
            stats.get_editor_property("num_pixel_shader_instructions")
        ),
        "num_samplers": int(stats.get_editor_property("num_samplers")),
        "num_vertex_texture_samples": int(
            stats.get_editor_property("num_vertex_texture_samples")
        ),
        "num_pixel_texture_samples": int(
            stats.get_editor_property("num_pixel_texture_samples")
        ),
        "num_virtual_texture_samples": int(
            stats.get_editor_property("num_virtual_texture_samples")
        ),
        "num_uv_scalars": int(stats.get_editor_property("num_uv_scalars")),
        "num_interpolator_scalars": int(
            stats.get_editor_property("num_interpolator_scalars")
        ),
        "compiled_shader_entries": [
            {"vertex_factory": vertex_factory, "shader_type": shader_type}
            for vertex_factory, shader_type in shader_rows
        ],
        "used_textures": [texture.get_path_name() for texture in used_textures],
    }


def collect() -> None:
    status = {
        "schema_version": 1,
        "status": "started",
        "formal_holdout_accessed": False,
        "contract_path": CONTRACT_PATH.relative_to(REPO_ROOT).as_posix(),
    }
    try:
        actual_hash = _sha256(CONTRACT_PATH)
        if actual_hash != EXPECTED_CONTRACT_SHA256:
            raise RuntimeError(
                f"measurement contract hash mismatch: {actual_hash}"
            )
        contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        if contract.get("frozen") is not True:
            raise RuntimeError("measurement contract is not frozen")

        unreal.SystemLibrary.execute_console_command(None, "r.MaterialQualityLevel 1")
        variants = []
        for panel_name, panel in contract["panels"].items():
            for variant in panel["variants"]:
                variants.append((panel_name, variant))

        stats_rows = []
        textures_by_path = {}
        for panel_name, variant in variants:
            asset_path = variant["material"]
            if not unreal.EditorAssetLibrary.does_asset_exist(asset_path):
                raise RuntimeError(f"missing material asset: {asset_path}")
            material = unreal.load_asset(asset_path)
            if not isinstance(material, unreal.MaterialInterface):
                raise RuntimeError(f"not a material interface: {asset_path}")
            row = _stats_record(material)
            row.update(
                {
                    "panel": panel_name,
                    "variant_id": variant["id"],
                    "role": variant["role"],
                    "contract_uasset_sha256": variant["uasset_sha256"],
                }
            )
            stats_rows.append(row)
            for texture_path in row["used_textures"]:
                if texture_path in textures_by_path:
                    continue
                texture = unreal.load_asset(texture_path)
                if isinstance(texture, unreal.Texture):
                    textures_by_path[texture_path] = _texture_record(texture)

        if not stats_rows or all(
            row["num_pixel_shader_instructions"] <= 0
            and row["num_pixel_texture_samples"] <= 0
            and row["num_samplers"] <= 0
            for row in stats_rows
        ):
            raise RuntimeError(
                "material statistics are all zero; RHI material resources were not compiled"
            )
        if not textures_by_path:
            raise RuntimeError("material texture dependency inventory is empty")

        material_output = {
            "schema_version": 1,
            "status": "complete",
            "contract_sha256": actual_hash,
            "formal_holdout_accessed": False,
            "rows": stats_rows,
        }
        texture_output = {
            "schema_version": 1,
            "status": "complete",
            "contract_sha256": actual_hash,
            "formal_holdout_accessed": False,
            "textures": [textures_by_path[key] for key in sorted(textures_by_path)],
        }
        _write_json("material_stats_raw.json", material_output)
        _write_json("texture_asset_inventory.json", texture_output)
        status.update(
            {
                "status": "complete",
                "contract_sha256": actual_hash,
                "variant_count": len(stats_rows),
                "unique_texture_count": len(textures_by_path),
            }
        )
        _write_json("ue_readonly_collection_status.json", status)
        unreal.log(
            "UE runtime evidence read-only collection complete: "
            f"{len(stats_rows)} variants, {len(textures_by_path)} textures"
        )
    except Exception as exc:
        status.update(
            {
                "status": "failed",
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        _write_json("ue_readonly_collection_status.json", status)
        unreal.log_error(f"UE runtime evidence collection failed: {exc}")
        raise


collect()
