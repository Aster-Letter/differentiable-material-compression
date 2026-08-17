"""Fail-closed source contract and complementary validator selection for C4."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ComplexAssetContract:
    eligible: bool
    emissive_present: bool
    emissive_review: str
    license_spdx: str


@dataclass(frozen=True)
class ComplexValidatorSelection:
    basecolor_dominant: str
    cross_channel_dominant: str
    eligible_assets: tuple[str, ...]


def _contains_extensions(value: object) -> bool:
    if isinstance(value, Mapping):
        if "extensions" in value:
            return True
        return any(_contains_extensions(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_extensions(child) for child in value)
    return False


def audit_complex_gltf_document(
    document: Mapping[str, Any],
    *,
    license_spdx: str,
    emissive_identity_preserved: bool | None,
) -> ComplexAssetContract:
    """Validate the CC0, opaque, one-material metallic-roughness source contract."""

    if license_spdx != "CC0-1.0":
        raise ValueError("complex C4 assets must be CC0-1.0")
    if document.get("extensionsUsed") or document.get("extensionsRequired"):
        raise ValueError("glTF extensions are outside the complex C4 contract")
    if _contains_extensions(document):
        raise ValueError("nested glTF extensions are outside the complex C4 contract")
    meshes = document.get("meshes")
    materials = document.get("materials")
    if not isinstance(meshes, list) or len(meshes) != 1:
        raise ValueError("complex C4 asset must contain exactly one mesh")
    if not isinstance(materials, list) or len(materials) != 1:
        raise ValueError("complex C4 asset must contain exactly one material")
    primitives = meshes[0].get("primitives") if isinstance(meshes[0], Mapping) else None
    if not isinstance(primitives, list) or len(primitives) != 1:
        raise ValueError("complex C4 asset must contain exactly one primitive")
    primitive = primitives[0]
    if not isinstance(primitive, Mapping) or primitive.get("material") != 0:
        raise ValueError("complex C4 primitive must reference its single material")
    material = materials[0]
    if not isinstance(material, Mapping) or material.get("alphaMode", "OPAQUE") != "OPAQUE":
        raise ValueError("complex C4 material must be opaque")
    pbr = material.get("pbrMetallicRoughness")
    if not isinstance(pbr, Mapping):
        raise ValueError("complex C4 material must use metallic-roughness PBR")
    if not isinstance(pbr.get("baseColorTexture"), Mapping):
        raise ValueError("complex C4 material is missing BaseColor")
    if not isinstance(pbr.get("metallicRoughnessTexture"), Mapping):
        raise ValueError("complex C4 material is missing metallic-roughness")
    if not isinstance(material.get("normalTexture"), Mapping):
        raise ValueError("complex C4 material is missing normal")
    emissive_present = isinstance(material.get("emissiveTexture"), Mapping) or any(
        float(value) != 0.0 for value in material.get("emissiveFactor", (0.0, 0.0, 0.0))
    )
    if emissive_present and emissive_identity_preserved is None:
        return ComplexAssetContract(False, True, "pending", license_spdx)
    if emissive_present and not emissive_identity_preserved:
        return ComplexAssetContract(False, True, "rejected", license_spdx)
    return ComplexAssetContract(
        True,
        emissive_present,
        "accepted" if emissive_present else "not_required",
        license_spdx,
    )


def _finite_metric(report: Mapping[str, Any], name: str) -> float:
    value = float(report[name])
    if not math.isfinite(value):
        raise ValueError(f"non-finite complex asset metric: {name}")
    return value


def _eligible(report: Mapping[str, Any]) -> bool:
    if not bool(report.get("contract_eligible")) or not bool(
        report.get("core4_identity_preserved")
    ):
        return False
    q4_base = _finite_metric(report, "q4_basecolor_error")
    q6_base = _finite_metric(report, "q6_basecolor_error")
    q4_seven = _finite_metric(report, "q4_seven_error")
    q6_seven = _finite_metric(report, "q6_seven_error")
    q4_hdr = _finite_metric(report, "q4_hdr_mae")
    q6_hdr = _finite_metric(report, "q6_hdr_mae")
    q4_ssim = _finite_metric(report, "q4_ssim")
    q6_ssim = _finite_metric(report, "q6_ssim")
    if min(q4_base, q4_seven, q4_hdr) <= 0.0:
        raise ValueError("q4 gate denominators must be positive")
    capacity = max((q4_base - q6_base) / q4_base, (q4_seven - q6_seven) / q4_seven)
    render = (q4_hdr - q6_hdr) / q4_hdr >= 0.10 - 1.0e-12 or q6_ssim - q4_ssim >= 0.01 - 1.0e-12
    return capacity >= 0.25 - 1.0e-12 and render


def select_complex_validators(
    reports: Sequence[Mapping[str, Any]],
) -> ComplexValidatorSelection:
    """Select one BaseColor-tail and one different cross-channel validator."""

    eligible = [report for report in reports if _eligible(report)]
    if len(eligible) < 2:
        raise ValueError("fewer than two complex assets passed the frozen capacity gates")
    base = max(eligible, key=lambda row: _finite_metric(row, "basecolor_q4_excess"))
    others = [row for row in eligible if row["asset_id"] != base["asset_id"]]
    if not others:
        raise ValueError("complex validators must be different assets")
    cross = max(
        others,
        key=lambda row: _finite_metric(row, "nonbase_standardized_residual"),
    )
    return ComplexValidatorSelection(
        basecolor_dominant=str(base["asset_id"]),
        cross_channel_dominant=str(cross["asset_id"]),
        eligible_assets=tuple(str(row["asset_id"]) for row in eligible),
    )
