"""Deterministic Unreal Engine export for frozen C4/C5 DTF candidates."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
from typing import Any

import numpy as np
from PIL import Image

from cg_frontier.compression.ue_export import sha256_file


ARRAY_KEYS = (
    "hidden_in.weight",
    "hidden_in.bias",
    "hidden_mid.weight",
    "hidden_mid.bias",
    "output.weight",
    "output.bias",
)

CANDIDATES = {
    "c4_dtf_16_s080k": {
        "relative_root": Path(
            "outputs/compression/scifihelmet/c4_dtf_v1/c4_dtf_16_selected"
        ),
        "manifest_sha256": "0ea2adc0906e5ed3c3311a323252f7a7ad543b32b8ab4900a89246a9a3d28237",
        "manifest_candidate": "c4_dtf_16_selected",
        "latent_channels": 4,
        "status": "trained_local_80k",
        "selected_step": 80_000,
    },
    "c4_dtf_16_s160k": {
        "relative_root": Path(
            "outputs/compression/scifihelmet/c4_dtf_v1/c4_dtf_16_resume_160k"
        ),
        "manifest_sha256": "fbb46b5505259364b3e790d3131a94d5e62a1e6e242ac5bf7f27336745de4f9b",
        "manifest_candidate": "c4_dtf_16_selected",
        "latent_channels": 4,
        "status": "trained_local_160k_continuation",
        "selected_step": 160_000,
        "best_render_step": 155_000,
        "best_artifact_safe_step": 160_000,
        "cost_source_relative_root": Path(
            "outputs/compression/scifihelmet/c4_dtf_v1/c4_dtf_16_selected"
        ),
        "cost_source_manifest_sha256": (
            "0ea2adc0906e5ed3c3311a323252f7a7ad543b32b8ab4900a89246a9a3d28237"
        ),
    },
    "c5_dtf_16_s080k": {
        "relative_root": Path(
            "outputs/compression/scifihelmet/c4_dtf_v1/c5_dtf_16"
        ),
        "manifest_sha256": "58fa74f38671987011941d0c1fd8c0fd1890438b17763d1ca8c0a1d0ff850ce7",
        "manifest_candidate": "c5_dtf_16",
        "latent_channels": 5,
        "status": "trained_local_80k",
        "selected_step": 80_000,
    },
    "c5_dtf_16_s120k": {
        "relative_root": Path(
            "outputs/compression/scifihelmet/c4_dtf_v1/c5_dtf_16_resume_120k"
        ),
        "manifest_sha256": "0ba3797d9528c9a8bad77ac16e508ca480b7cbb29e4cb1c94121e796774c1bcb",
        "manifest_candidate": "c5_dtf_16",
        "latent_channels": 5,
        "status": "trained_local_120k_continuation",
        "selected_step": 120_000,
        "cost_source_relative_root": Path(
            "outputs/compression/scifihelmet/c4_dtf_v1/c5_dtf_16"
        ),
        "cost_source_manifest_sha256": (
            "58fa74f38671987011941d0c1fd8c0fd1890438b17763d1ca8c0a1d0ff850ce7"
        ),
    },
}

DEFAULT_OUTPUT_RELATIVE_PATH = Path(
    "outputs/deployment/scifihelmet/dtf_preview_v1"
)
PREVIEW_MANIFEST_NAME = "preview_manifest.json"
UV_PROBES = np.asarray(
    [[0.125, 0.25], [0.5, 0.5], [0.9999, 0.0001], [-0.125, 1.125]],
    dtype=np.float32,
)


@dataclass(frozen=True)
class FrozenDTFCandidate:
    """Validated data needed to build one UE DTF material package."""

    name: str
    root: Path
    manifest: dict[str, Any]
    arrays: dict[str, np.ndarray]
    latent_rgba_u8: np.ndarray
    latent_r_u8: np.ndarray | None
    metadata: dict[str, Any]


def _expected_shapes(latent_channels: int) -> dict[str, tuple[int, ...]]:
    return {
        "hidden_in.weight": (16, latent_channels),
        "hidden_in.bias": (16,),
        "hidden_mid.weight": (16, 16),
        "hidden_mid.bias": (16,),
        "output.weight": (7, 16),
        "output.bias": (7,),
    }


def _manifest_resource_paths(root: Path, manifest: dict[str, Any]) -> tuple[Path, Path | None]:
    files = manifest["files"]
    if "latent_resources" in files:
        resources = files["latent_resources"]["resources"]
    else:
        legacy = [
            (name, value)
            for name, value in files.items()
            if name.startswith("latent_") and name.endswith("_rgba_unorm8.png")
        ]
        if len(legacy) != 1 or not isinstance(legacy[0][1], dict):
            raise ValueError("unexpected legacy C4 latent resource layout")
        resources = [
            {
                "file": legacy[0][0],
                "sha256": legacy[0][1]["sha256"],
                "storage_channels": "rgba8",
            }
        ]
    rgba = [item for item in resources if item["storage_channels"] == "rgba8"]
    red = [item for item in resources if item["storage_channels"] == "r8"]
    if len(rgba) != 1 or len(red) > 1:
        raise ValueError("unexpected DTF latent resource layout")
    for item in resources:
        path = root / item["file"]
        if not path.is_file() or sha256_file(path) != item["sha256"]:
            raise ValueError(f"DTF latent resource hash mismatch: {path}")
    return root / rgba[0]["file"], root / red[0]["file"] if red else None


def _validated_cost(
    repo_root: Path, manifest: dict[str, Any], spec: dict[str, Any]
) -> dict[str, Any]:
    if "cost" in manifest:
        return dict(manifest["cost"])
    source_sha256 = spec.get("cost_source_manifest_sha256")
    source_root = spec.get("cost_source_relative_root")
    if source_sha256 is None or source_root is None:
        raise ValueError("DTF manifest is missing frozen deployment cost")
    if manifest.get("lineage", {}).get("source_training_manifest_sha256") != source_sha256:
        raise ValueError("DTF continuation cost lineage mismatch")
    source_path = repo_root / source_root / "training_manifest.json"
    if not source_path.is_file() or sha256_file(source_path) != source_sha256:
        raise ValueError("DTF continuation cost source hash mismatch")
    source_manifest = json.loads(source_path.read_text(encoding="utf-8"))
    return dict(source_manifest["cost"])


def load_frozen_candidate(repo_root: Path, name: str) -> FrozenDTFCandidate:
    """Load a frozen selected candidate and reject manifest or artifact drift."""

    try:
        spec = CANDIDATES[name]
    except KeyError as exc:
        raise ValueError(f"unknown DTF candidate: {name}") from exc
    root = repo_root.resolve() / spec["relative_root"]
    manifest_path = root / "training_manifest.json"
    if sha256_file(manifest_path) != spec["manifest_sha256"]:
        raise ValueError(f"frozen DTF training manifest hash mismatch: {name}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("valid") is not True
        or manifest.get("status") != spec["status"]
        or manifest["candidate"]["name"] != spec["manifest_candidate"]
        or manifest["candidate"]["activation"] != "relu"
        or manifest["training_result"]["completed_steps"] != spec["selected_step"]
        or manifest["checkpoint_selection"]["best_render"]["step"]
        != spec.get("best_render_step", spec["selected_step"])
        or manifest["checkpoint_selection"]["best_artifact_safe"]["step"]
        != spec.get("best_artifact_safe_step", spec["selected_step"])
    ):
        raise ValueError(f"candidate is not the frozen valid selection: {name}")

    decoder_path = root / "decoder_weights.npz"
    if sha256_file(decoder_path) != manifest["files"]["decoder_weights.npz"]:
        raise ValueError(f"DTF decoder hash mismatch: {name}")
    with np.load(decoder_path, allow_pickle=False) as stored:
        if set(stored.files) != set(ARRAY_KEYS):
            raise ValueError(f"unexpected DTF decoder arrays: {sorted(stored.files)}")
        arrays = {key: np.asarray(stored[key]).copy() for key in ARRAY_KEYS}
    expected = _expected_shapes(int(spec["latent_channels"]))
    for key, shape in expected.items():
        value = arrays[key]
        if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
            raise ValueError(f"invalid DTF decoder array {key}: {value.shape}/{value.dtype}")

    rgba_path, r_path = _manifest_resource_paths(root, manifest)
    with Image.open(rgba_path) as image:
        if image.mode != "RGBA" or image.size != (2048, 2048):
            raise ValueError("DTF RGBA resource must be 2048x2048 RGBA8")
        rgba = np.asarray(image, dtype=np.uint8).copy()
    red = None
    if r_path is not None:
        with Image.open(r_path) as image:
            if image.mode != "L" or image.size != (2048, 2048):
                raise ValueError("DTF fifth-channel resource must be 2048x2048 R8")
            red = np.asarray(image, dtype=np.uint8).copy()
    if (red is not None) != (int(spec["latent_channels"]) == 5):
        raise ValueError("DTF resource count does not match latent channel count")

    cost = _validated_cost(repo_root.resolve(), manifest, spec)
    metadata = {
        "latent_channels": int(spec["latent_channels"]),
        "selection_step": int(spec["selected_step"]),
        "texture_resources": int(cost["texture_resources"]),
        "point_texel_loads_per_pixel": int(cost["point_texel_loads_per_pixel"]),
        "theoretical_raw_bytes_unorm8": int(cost["theoretical_raw_bytes_unorm8"]),
        "parameter_count": int(cost["parameters"]),
        "weight_bytes_float32": int(cost["weight_bytes_float32"]),
        "decoder_macs_per_corner": int(cost["decoder_macs_per_corner"]),
        "decoder_macs_per_pixel": int(cost["decoder_macs_per_pixel"]),
        "training_manifest_sha256": spec["manifest_sha256"],
        "decoder_sha256": manifest["files"]["decoder_weights.npz"],
        "rgba_source": rgba_path,
        "r_source": r_path,
    }
    return FrozenDTFCandidate(name, root, manifest, arrays, rgba, red, metadata)


def _float(value: np.float32) -> str:
    result = format(float(np.float32(value)), ".9g")
    return result if "." in result or "e" in result.lower() else result + ".0"


def _values(values: np.ndarray) -> str:
    return ", ".join(_float(value) for value in np.asarray(values).reshape(-1))


def _emit_float4_rows(lines: list[str], prefix: str, values: np.ndarray) -> None:
    for row in range(values.shape[0]):
        for block in range(values.shape[1] // 4):
            suffix = "ABCD"[block]
            start = block * 4
            lines.append(
                f"    const float4 {prefix}_{row}{suffix} = float4("
                f"{_values(values[row, start:start + 4])});"
            )


def generate_candidate_hlsl(candidate: FrozenDTFCandidate) -> str:
    """Generate one Custom-expression body implementing the frozen DTF order."""

    arrays = candidate.arrays
    channels = int(candidate.metadata["latent_channels"])
    lines = [
        "// Deterministically generated C4/C5 decode-then-material-filter shader.",
        f"// candidate={candidate.name}",
        f"// training_manifest_sha256={candidate.metadata['training_manifest_sha256']}",
        f"// decoder_sha256={candidate.metadata['decoder_sha256']}",
        "// Inputs: UV, LatentRGBA texture object, optional LatentR texture object, NormalYSign.",
        "// Textures must be non-sRGB, point filtered, Wrap, LOD0, no mipmaps.",
        "struct CGDDecoder",
        "{",
        "  void CGD_DecodeCorner(float4 Z4, float Z5, out float3 BaseColor, out float3 NormalGLTF, out float RoughnessValue, out float MetallicValue)",
        "  {",
    ]
    first = arrays["hidden_in.weight"]
    for row in range(16):
        lines.append(f"    const float4 CGD_W0_{row}A = float4({_values(first[row, :4])});")
        if channels == 5:
            lines.append(f"    const float CGD_W0X_{row} = {_float(first[row, 4])};")
        lines.append(f"    const float CGD_B0_{row} = {_float(arrays['hidden_in.bias'][row])};")
    _emit_float4_rows(lines, "CGD_W1", arrays["hidden_mid.weight"])
    for row in range(16):
        lines.append(f"    const float CGD_B1_{row} = {_float(arrays['hidden_mid.bias'][row])};")
    _emit_float4_rows(lines, "CGD_W2", arrays["output.weight"])
    for row in range(7):
        lines.append(f"    const float CGD_B2_{row} = {_float(arrays['output.bias'][row])};")
    lines.append("")
    extra = " + Z5 * CGD_W0X_{row}" if channels == 5 else ""
    for row in range(16):
        lines.append(
            f"    float CGD_H0_{row} = max(dot(Z4, CGD_W0_{row}A){extra.format(row=row)} + CGD_B0_{row}, 0.0);"
        )
    for block in range(4):
        start = block * 4
        lines.append(
            f"    float4 CGD_H0{chr(65 + block)} = float4(CGD_H0_{start}, CGD_H0_{start + 1}, CGD_H0_{start + 2}, CGD_H0_{start + 3});"
        )
    for row in range(16):
        terms = " + ".join(
            f"dot(CGD_H0{chr(65 + block)}, CGD_W1_{row}{chr(65 + block)})"
            for block in range(4)
        )
        lines.append(f"    float CGD_H1_{row} = max({terms} + CGD_B1_{row}, 0.0);")
    for block in range(4):
        start = block * 4
        lines.append(
            f"    float4 CGD_H1{chr(65 + block)} = float4(CGD_H1_{start}, CGD_H1_{start + 1}, CGD_H1_{start + 2}, CGD_H1_{start + 3});"
        )
    for row in range(7):
        terms = " + ".join(
            f"dot(CGD_H1{chr(65 + block)}, CGD_W2_{row}{chr(65 + block)})"
            for block in range(4)
        )
        lines.append(f"    float CGD_R{row} = {terms} + CGD_B2_{row};")
    lines.extend(
        [
            "    BaseColor = 1.0 / (1.0 + exp(-float3(CGD_R0, CGD_R1, CGD_R2)));",
            "    float2 CGD_NormalXY = tanh(float2(CGD_R3, CGD_R4));",
            "    CGD_NormalXY /= max(length(CGD_NormalXY) / (1.0 - 1.0e-6), 1.0);",
            "    float CGD_NormalZ = sqrt(max(1.0 - dot(CGD_NormalXY, CGD_NormalXY), 1.0e-8));",
            "    NormalGLTF = float3(CGD_NormalXY, CGD_NormalZ);",
            "    RoughnessValue = 1.0 / (1.0 + exp(-CGD_R5));",
            "    MetallicValue = 1.0 / (1.0 + exp(-CGD_R6));",
            "  }",
            "};",
            "CGDDecoder CGD;",
            "float2 CGD_TexelPosition = UV * 2048.0 - 0.5;",
            "float2 CGD_BaseTexel = floor(CGD_TexelPosition);",
            "float2 CGD_Fraction = CGD_TexelPosition - CGD_BaseTexel;",
            "float2 CGD_UV00 = (CGD_BaseTexel + float2(0.5, 0.5)) / 2048.0;",
            "float2 CGD_UV10 = (CGD_BaseTexel + float2(1.5, 0.5)) / 2048.0;",
            "float2 CGD_UV01 = (CGD_BaseTexel + float2(0.5, 1.5)) / 2048.0;",
            "float2 CGD_UV11 = (CGD_BaseTexel + float2(1.5, 1.5)) / 2048.0;",
        ]
    )
    for corner in ("00", "10", "01", "11"):
        lines.append(
            f"float4 CGD_Z4_{corner} = Texture2DSampleLevel(LatentRGBA, LatentRGBASampler, CGD_UV{corner}, 0.0);"
        )
        if channels == 5:
            lines.append(
                f"float CGD_Z5_{corner} = Texture2DSampleLevel(LatentR, LatentRSampler, CGD_UV{corner}, 0.0).r;"
            )
        else:
            lines.append(f"float CGD_Z5_{corner} = 0.0;")
        lines.append(f"float3 CGD_Base_{corner}, CGD_Normal_{corner};")
        lines.append(f"float CGD_Rough_{corner}, CGD_Metal_{corner};")
        lines.append(
            f"CGD.CGD_DecodeCorner(CGD_Z4_{corner}, CGD_Z5_{corner}, CGD_Base_{corner}, CGD_Normal_{corner}, CGD_Rough_{corner}, CGD_Metal_{corner});"
        )
    lines.extend(
        [
            "float4 CGD_Weights = float4((1.0-CGD_Fraction.x)*(1.0-CGD_Fraction.y), CGD_Fraction.x*(1.0-CGD_Fraction.y), (1.0-CGD_Fraction.x)*CGD_Fraction.y, CGD_Fraction.x*CGD_Fraction.y);",
            "float3 CGD_FilteredBase = CGD_Base_00*CGD_Weights.x + CGD_Base_10*CGD_Weights.y + CGD_Base_01*CGD_Weights.z + CGD_Base_11*CGD_Weights.w;",
            "float3 CGD_FilteredNormalGLTF = normalize(CGD_Normal_00*CGD_Weights.x + CGD_Normal_10*CGD_Weights.y + CGD_Normal_01*CGD_Weights.z + CGD_Normal_11*CGD_Weights.w);",
            "NormalTS = float3(CGD_FilteredNormalGLTF.x, CGD_FilteredNormalGLTF.y * NormalYSign, CGD_FilteredNormalGLTF.z);",
            "Roughness = dot(float4(CGD_Rough_00, CGD_Rough_10, CGD_Rough_01, CGD_Rough_11), CGD_Weights);",
            "Metallic = dot(float4(CGD_Metal_00, CGD_Metal_10, CGD_Metal_01, CGD_Metal_11), CGD_Weights);",
            "return CGD_FilteredBase;",
            "",
        ]
    )
    return "\n".join(lines)


def parse_candidate_hlsl_constants(source: str, latent_channels: int) -> dict[str, np.ndarray]:
    """Reload all decoder constants from generated HLSL for exact verification."""

    def vector_rows(prefix: str, rows: int, width: int) -> np.ndarray:
        blocks = width // 4
        found: dict[tuple[int, str], np.ndarray] = {}
        pattern = rf"{prefix}_(\d+)([A-D])\s*=\s*float4\(([^)]+)\)"
        for match in re.finditer(pattern, source):
            found[(int(match.group(1)), match.group(2))] = np.asarray(
                [np.float32(float(item.strip())) for item in match.group(3).split(",")],
                dtype=np.float32,
            )
        result = []
        for row in range(rows):
            result.append(np.concatenate([found[(row, "ABCD"[block])] for block in range(blocks)]))
        return np.stack(result).astype(np.float32)

    def scalars(prefix: str, count: int) -> np.ndarray:
        found = {
            int(match.group(1)): np.float32(float(match.group(2)))
            for match in re.finditer(rf"{prefix}_(\d+)\s*=\s*([^;]+);", source)
        }
        return np.asarray([found[index] for index in range(count)], dtype=np.float32)

    first = vector_rows("CGD_W0", 16, 4)
    if latent_channels == 5:
        fifth = scalars("CGD_W0X", 16)[:, None]
        first = np.concatenate((first, fifth), axis=1)
    return {
        "hidden_in.weight": first,
        "hidden_in.bias": scalars("CGD_B0", 16),
        "hidden_mid.weight": vector_rows("CGD_W1", 16, 16),
        "hidden_mid.bias": scalars("CGD_B1", 16),
        "output.weight": vector_rows("CGD_W2", 7, 16),
        "output.bias": scalars("CGD_B2", 7),
    }


def _decode_uv_probes(
    candidate: FrozenDTFCandidate,
    arrays: dict[str, np.ndarray],
    uv: np.ndarray,
) -> dict[str, np.ndarray]:
    rgba = candidate.latent_rgba_u8.astype(np.float32) / np.float32(255.0)
    height, width, _ = rgba.shape
    coordinates = np.asarray(uv, dtype=np.float32)
    position = coordinates * np.asarray([width, height], dtype=np.float32) - np.float32(0.5)
    base_floor = np.floor(position)
    fraction = position - base_floor
    base = base_floor.astype(np.int64)
    x0, y0 = base[:, 0] % width, base[:, 1] % height
    x1, y1 = (x0 + 1) % width, (y0 + 1) % height
    corners = np.stack(
        (rgba[y0, x0], rgba[y0, x1], rgba[y1, x0], rgba[y1, x1]), axis=1
    )
    if candidate.latent_r_u8 is not None:
        red = candidate.latent_r_u8.astype(np.float32) / np.float32(255.0)
        r_corners = np.stack(
            (red[y0, x0], red[y0, x1], red[y1, x0], red[y1, x1]), axis=1
        )[..., None]
        corners = np.concatenate((corners, r_corners), axis=-1)
    hidden0 = np.maximum(
        np.sum(
            corners[..., None, :] * arrays["hidden_in.weight"],
            axis=-1,
            dtype=np.float32,
        )
        + arrays["hidden_in.bias"],
        np.float32(0.0),
    )
    hidden1 = np.maximum(
        np.sum(
            hidden0[..., None, :] * arrays["hidden_mid.weight"],
            axis=-1,
            dtype=np.float32,
        )
        + arrays["hidden_mid.bias"],
        np.float32(0.0),
    )
    raw = (
        np.sum(
            hidden1[..., None, :] * arrays["output.weight"],
            axis=-1,
            dtype=np.float32,
        )
        + arrays["output.bias"]
    ).astype(np.float32)
    base_color = (np.float32(1.0) / (np.float32(1.0) + np.exp(-raw[..., :3]))).astype(np.float32)
    normal_xy = np.tanh(raw[..., 3:5]).astype(np.float32)
    radius = np.sqrt(np.sum(normal_xy * normal_xy, axis=-1, keepdims=True, dtype=np.float32))
    normal_xy /= np.maximum(radius / np.float32(1.0 - 1.0e-6), np.float32(1.0))
    normal_z = np.sqrt(
        np.maximum(
            np.float32(1.0)
            - np.sum(normal_xy * normal_xy, axis=-1, keepdims=True, dtype=np.float32),
            np.float32(1.0e-8),
        )
    )
    corner_normal = np.concatenate((normal_xy, normal_z), axis=-1)
    roughness = np.float32(1.0) / (np.float32(1.0) + np.exp(-raw[..., 5:6]))
    metallic = np.float32(1.0) / (np.float32(1.0) + np.exp(-raw[..., 6:7]))
    fx, fy = fraction[:, 0], fraction[:, 1]
    weights = np.stack(
        ((1.0 - fx) * (1.0 - fy), fx * (1.0 - fy), (1.0 - fx) * fy, fx * fy),
        axis=-1,
    ).astype(np.float32)

    def filtered(values: np.ndarray) -> np.ndarray:
        return np.sum(values * weights[..., None], axis=1, dtype=np.float32)

    filtered_normal = filtered(corner_normal)
    filtered_normal /= np.maximum(
        np.sqrt(np.sum(filtered_normal * filtered_normal, axis=-1, keepdims=True, dtype=np.float32)),
        np.float32(1.0e-8),
    )
    return {
        "corner_latent": corners,
        "weights": weights,
        "base_color_linear": filtered(base_color),
        "normal_tangent_gltf_positive_y": filtered_normal,
        "roughness_linear": filtered(roughness),
        "metallic_linear": filtered(metallic),
    }


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def _candidate_export_names(name: str) -> dict[str, str]:
    tokens = {
        "c4_dtf_16_s080k": "C4_DTF16_S080K",
        "c4_dtf_16_s160k": "C4_DTF16_S160K",
        "c5_dtf_16_s080k": "C5_DTF16_S080K",
        "c5_dtf_16_s120k": "C5_DTF16_S120K",
    }
    token = tokens[name]
    return {
        "hlsl": f"M_SciFiHelmet_{token}.custom.hlsl",
        "rgba": f"T_SciFiHelmet_{token}_RGBA8.png",
        "r": f"T_SciFiHelmet_{token}_R8.png",
        "material": f"M_SciFiHelmet_{token}",
        "texture_rgba": f"T_SciFiHelmet_{token}_RGBA8",
        "texture_r": f"T_SciFiHelmet_{token}_R8",
    }


def _export_one_candidate(
    candidate: FrozenDTFCandidate, output_root: Path
) -> dict[str, Any]:
    names = _candidate_export_names(candidate.name)
    package_dir = output_root / candidate.name
    package_dir.mkdir(parents=True, exist_ok=True)
    hlsl_path = package_dir / names["hlsl"]
    rgba_path = package_dir / names["rgba"]
    r_path = package_dir / names["r"]
    probes_path = package_dir / "probe_vectors.json"
    checklist_path = package_dir / "UE_INTEGRATION_CHECKLIST.md"
    manifest_path = package_dir / "deployment_manifest.json"

    hlsl = generate_candidate_hlsl(candidate)
    parsed = parse_candidate_hlsl_constants(
        hlsl, latent_channels=int(candidate.metadata["latent_channels"])
    )
    for key in ARRAY_KEYS:
        if not np.array_equal(candidate.arrays[key], parsed[key]):
            raise ValueError(f"generated HLSL constant reload mismatch: {candidate.name}/{key}")
    _write_text(hlsl_path, hlsl)
    shutil.copyfile(candidate.metadata["rgba_source"], rgba_path)
    if candidate.metadata["r_source"] is not None:
        shutil.copyfile(candidate.metadata["r_source"], r_path)
    elif r_path.exists():
        raise ValueError(f"stale fifth-channel export exists for C4: {r_path}")

    source_probe = _decode_uv_probes(candidate, candidate.arrays, UV_PROBES)
    generated_probe = _decode_uv_probes(candidate, parsed, UV_PROBES)
    probe_max_abs = max(
        float(np.max(np.abs(source_probe[key] - generated_probe[key])))
        for key in source_probe
    )
    probes = {
        "schema_version": 1,
        "candidate": candidate.name,
        "uv_top_down_wrap": UV_PROBES.tolist(),
        "npz_vs_generated_hlsl_constants_max_abs": probe_max_abs,
        "decoded": {key: value.tolist() for key, value in source_probe.items()},
    }
    _write_text(
        probes_path,
        json.dumps(probes, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )

    checklist = f"""# {candidate.name} UE DTF preview checklist

- Import `{names['rgba']}` as a non-sRGB, lossless RGBA8 texture; point filter, Wrap, NoMipmaps, never stream.
"""
    if candidate.latent_r_u8 is not None:
        checklist += f"- Import `{names['r']}` as non-sRGB lossless R8/Grayscale; point filter, Wrap, NoMipmaps, never stream.\n"
    checklist += (
        f"- Create `/Game/CGCompression/DTFPreview/Materials/{names['material']}` with one Custom expression from `{names['hlsl']}`.\n"
        "- Connect TexCoord0 plus Texture Object inputs. The Custom code performs four LOD0 corner fetches per resource, per-corner decode/postprocess, material-domain bilinear filtering, and one filtered-normal normalization.\n"
        "- Apply the glTF +Y to UE tangent bridge exactly once with `NormalYSign=-1`; leave AO and Specular unconnected.\n"
        "- Do not substitute bilinear latent TextureSample nodes for the Texture Object inputs.\n"
    )
    _write_text(checklist_path, checklist)

    asset_root = "/Game/CGCompression/DTFPreview"
    texture_folder = f"{asset_root}/Textures/{candidate.name}"
    material_path = f"{asset_root}/Materials/{names['material']}"
    generated_files = {
        hlsl_path.name: sha256_file(hlsl_path),
        rgba_path.name: sha256_file(rgba_path),
        probes_path.name: sha256_file(probes_path),
        checklist_path.name: sha256_file(checklist_path),
    }
    if candidate.latent_r_u8 is not None:
        generated_files[r_path.name] = sha256_file(r_path)
    manifest = {
        "schema_version": 1,
        "candidate": candidate.name,
        "selection_step": int(candidate.metadata["selection_step"]),
        "formal_holdout_accessed": False,
        "source": {
            "training_manifest_sha256": candidate.metadata["training_manifest_sha256"],
            "decoder_sha256": candidate.metadata["decoder_sha256"],
        },
        "decoder": {
            "architecture": f"{candidate.metadata['latent_channels']}->16->16->7",
            "activation": "ReLU",
            "parameter_count": candidate.metadata["parameter_count"],
            "weight_bytes_float32": candidate.metadata["weight_bytes_float32"],
            "macs_per_corner": candidate.metadata["decoder_macs_per_corner"],
            "macs_per_pixel": candidate.metadata["decoder_macs_per_pixel"],
        },
        "runtime": {
            "filter_order": "four_point_fetches_per_resource -> per_corner_decode_postprocess -> material_bilinear_filter -> one_normalize",
            "texture_resources": candidate.metadata["texture_resources"],
            "point_texel_loads_per_pixel": candidate.metadata["point_texel_loads_per_pixel"],
            "lod": 0,
            "mipmaps": False,
            "address": "Wrap",
            "texture_filter": "Point (manual material-domain bilinear)",
            "base_color_additional_srgb_decode": False,
            "normal_y_bridge_application_count": 1,
            "ambient_occlusion": "excluded",
        },
        "storage": {
            "logical_channels": candidate.metadata["latent_channels"],
            "theoretical_raw_bytes_unorm8": candidate.metadata["theoretical_raw_bytes_unorm8"],
            "actual_resident_bytes": None,
            "actual_resident_measurement": "required_in_UE_after_import",
        },
        "ue_assets": {
            "texture_folder": texture_folder,
            "texture_rgba": f"{texture_folder}/{names['texture_rgba']}",
            "texture_r": (
                f"{texture_folder}/{names['texture_r']}"
                if candidate.latent_r_u8 is not None
                else None
            ),
            "material": material_path,
            "preview_map": f"{asset_root}/Maps/MaterialLab_DTF_Preview",
        },
        "probe_verification": {"npz_vs_generated_hlsl_constants_max_abs": probe_max_abs},
        "generated_files": generated_files,
        "ue_setup_pending": True,
    }
    _write_text(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return {
        **manifest,
        "package_directory": candidate.name,
        "deployment_manifest": manifest_path.name,
        "deployment_manifest_sha256": sha256_file(manifest_path),
    }


def export_dtf_preview_package(repo_root: Path, output_dir: Path) -> dict[str, Any]:
    """Export frozen step-qualified DTF candidates as auditable UE packages."""

    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates = {
        name: _export_one_candidate(load_frozen_candidate(repo_root, name), output_dir)
        for name in (
            "c4_dtf_16_s080k",
            "c4_dtf_16_s160k",
            "c5_dtf_16_s080k",
            "c5_dtf_16_s120k",
        )
    }
    manifest = {
        "schema_version": 1,
        "status": "exported_ue_setup_pending",
        "formal_holdout_accessed": False,
        "candidates": candidates,
        "ue_preview": {
            "asset_root": "/Game/CGCompression/DTFPreview",
            "map": "/Game/CGCompression/DTFPreview/Maps/MaterialLab_DTF_Preview",
            "source_map_preserved": "/Game/CGCompression/Maps/MaterialLab",
            "materials_separate": True,
        },
    }
    _write_text(
        output_dir / PREVIEW_MANIFEST_NAME,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return manifest
