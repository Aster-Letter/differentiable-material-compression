"""Deterministic UE Custom-HLSL export for the frozen pre-QAT baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
from typing import Any

import numpy as np
from PIL import Image


LATENT_RELATIVE_PATH = Path(
    "outputs/compression/scifihelmet/render_quant/baseline/models/tiny_mlp/"
    "latent_pre_qat_rgba_unorm8.png"
)
DECODER_RELATIVE_PATH = Path(
    "outputs/compression/scifihelmet/material_fit/tiny_mlp/decoder_weights.npz"
)
DEFAULT_OUTPUT_RELATIVE_PATH = Path("outputs/deployment/scifihelmet/ue_pre_qat_hard")

EXPECTED_LATENT_SHA256 = "5a1781afc1a877be452a87a3d958e48cab921b45237faebf2be3668a60ae5fdc"
EXPECTED_DECODER_SHA256 = "d676ade8294600eb0064a835eabfe86d4d35e39ee787d512574fbef8d7346baa"
EXPECTED_ARRAYS = {
    "network.0.weight": (8, 4),
    "network.0.bias": (8,),
    "network.2.weight": (7, 8),
    "network.2.bias": (7,),
}
PARAMETER_COUNT = 103
WEIGHT_BYTES_FLOAT32 = 412
MACS_PER_PIXEL = 88

HLSL_FILENAME = "M_SciFiHelmet_Compressed.custom.hlsl"
LATENT_EXPORT_FILENAME = "T_SciFiHelmet_Latent_RGBA8.png"
PROBES_FILENAME = "probe_vectors.json"
MANIFEST_FILENAME = "deployment_manifest.json"
CHECKLIST_FILENAME = "UE_INTEGRATION_CHECKLIST.md"

CANONICAL_PROBES = np.asarray(
    [
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 1.0, 1.0, 1.0],
        [0.5, 0.5, 0.5, 0.5],
        [17.0 / 255.0, 63.0 / 255.0, 129.0 / 255.0, 241.0 / 255.0],
    ],
    dtype=np.float32,
)
UV_PROBES = np.asarray(
    [
        [0.125, 0.25],
        [0.5, 0.5],
        [0.9999, 0.0001],
        [-0.125, 1.125],
    ],
    dtype=np.float32,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _float_literal(value: np.float32) -> str:
    text = format(float(np.float32(value)), ".9g")
    if "e" not in text.lower() and "." not in text:
        text += ".0"
    return text


def _float_list(values: np.ndarray) -> str:
    return ", ".join(_float_literal(value) for value in values.reshape(-1))


def validate_frozen_inputs(
    latent_path: Path, decoder_path: Path
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    """Verify frozen hashes, NPZ layout, dtype, and deployment cost metadata."""

    if not latent_path.is_file():
        raise FileNotFoundError(f"frozen latent is missing: {latent_path}")
    if not decoder_path.is_file():
        raise FileNotFoundError(f"frozen decoder is missing: {decoder_path}")

    latent_sha256 = sha256_file(latent_path)
    decoder_sha256 = sha256_file(decoder_path)
    if latent_sha256 != EXPECTED_LATENT_SHA256:
        raise ValueError(
            f"frozen latent SHA-256 mismatch: {latent_sha256} != {EXPECTED_LATENT_SHA256}"
        )
    if decoder_sha256 != EXPECTED_DECODER_SHA256:
        raise ValueError(
            f"frozen decoder SHA-256 mismatch: {decoder_sha256} != {EXPECTED_DECODER_SHA256}"
        )

    with Image.open(latent_path) as image:
        if image.mode != "RGBA" or image.size != (2048, 2048):
            raise ValueError(
                f"frozen latent must be 2048x2048 RGBA, got {image.size} {image.mode}"
            )
        latent_u8 = np.asarray(image, dtype=np.uint8).copy()

    with np.load(decoder_path, allow_pickle=False) as stored:
        if set(stored.files) != set(EXPECTED_ARRAYS):
            raise ValueError(
                f"unexpected decoder keys: {sorted(stored.files)}; "
                f"expected {sorted(EXPECTED_ARRAYS)}"
            )
        arrays = {name: np.asarray(stored[name]).copy() for name in EXPECTED_ARRAYS}

    for name, expected_shape in EXPECTED_ARRAYS.items():
        array = arrays[name]
        if array.shape != expected_shape:
            raise ValueError(f"{name} shape {array.shape} != {expected_shape}")
        if array.dtype != np.float32:
            raise ValueError(f"{name} dtype {array.dtype} != float32")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains non-finite values")

    parameter_count = sum(int(array.size) for array in arrays.values())
    weight_bytes = sum(int(array.nbytes) for array in arrays.values())
    if parameter_count != PARAMETER_COUNT or weight_bytes != WEIGHT_BYTES_FLOAT32:
        raise ValueError(
            f"decoder metadata mismatch: {parameter_count} parameters/{weight_bytes} bytes"
        )

    metadata = {
        "latent_sha256": latent_sha256,
        "decoder_sha256": decoder_sha256,
        "latent_width": int(latent_u8.shape[1]),
        "latent_height": int(latent_u8.shape[0]),
        "latent_channels": int(latent_u8.shape[2]),
        "latent_raw_gpu_bytes_no_mips": int(latent_u8.nbytes),
        "parameter_count": parameter_count,
        "weight_bytes_float32": weight_bytes,
        "macs_per_pixel": MACS_PER_PIXEL,
    }
    return latent_u8, arrays, metadata


def decoder_raw_forward(latent_rgba: np.ndarray, arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Evaluate the raw 4→8→7 decoder with shader-compatible dot grouping."""

    latent = np.asarray(latent_rgba, dtype=np.float32)
    # Use explicit float32 products/sums instead of NumPy BLAS so this offline
    # verifier remains usable on the project's Windows environment even when an
    # optional BLAS runtime DLL is unavailable. The shader itself uses the same
    # row-wise dot-product grouping emitted below.
    hidden_linear = np.sum(
        latent[..., None, :] * arrays["network.0.weight"],
        axis=-1,
        dtype=np.float32,
    ) + arrays["network.0.bias"]
    hidden = np.maximum(
        hidden_linear,
        np.float32(0.0),
    )
    return np.asarray(
        np.sum(
            hidden[..., None, :] * arrays["network.2.weight"],
            axis=-1,
            dtype=np.float32,
        )
        + arrays["network.2.bias"],
        dtype=np.float32,
    )


def decoder_postprocess(raw: np.ndarray) -> dict[str, np.ndarray]:
    """Apply the seven semantic channel transforms exactly once in float32."""

    raw = np.asarray(raw, dtype=np.float32)
    base_color = np.asarray(1.0 / (1.0 + np.exp(-raw[..., 0:3])), dtype=np.float32)
    normal_xy = np.asarray(np.tanh(raw[..., 3:5]), dtype=np.float32)
    radius = np.linalg.norm(normal_xy, axis=-1, keepdims=True).astype(np.float32)
    divisor = np.maximum(
        radius / np.float32(1.0 - 1.0e-6), np.float32(1.0)
    )
    normal_xy = np.asarray(normal_xy / divisor, dtype=np.float32)
    normal_z = np.sqrt(
        np.maximum(
            np.float32(1.0) - np.sum(normal_xy * normal_xy, axis=-1, keepdims=True),
            np.float32(1.0e-8),
        )
    ).astype(np.float32)
    normal_xyz = np.concatenate((normal_xy, normal_z), axis=-1).astype(np.float32)
    normal_xyz /= np.maximum(
        np.linalg.norm(normal_xyz, axis=-1, keepdims=True).astype(np.float32),
        np.float32(1.0e-8),
    )
    roughness = np.asarray(1.0 / (1.0 + np.exp(-raw[..., 5:6])), dtype=np.float32)
    metallic = np.asarray(1.0 / (1.0 + np.exp(-raw[..., 6:7])), dtype=np.float32)
    return {
        "base_color_linear": base_color,
        "normal_tangent_gltf_positive_y": normal_xyz,
        "roughness_linear": roughness,
        "metallic_linear": metallic,
    }


def bilinear_sample_top_down_wrap(latent_u8: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Dequantize RGBA8 texels, then reproduce wrap-mode GPU bilinear sampling."""

    texture = np.asarray(latent_u8, dtype=np.float32) / np.float32(255.0)
    coordinates = np.asarray(uv, dtype=np.float32)
    height, width, _ = texture.shape
    x = coordinates[..., 0] * np.float32(width) - np.float32(0.5)
    y = coordinates[..., 1] * np.float32(height) - np.float32(0.5)
    x0_floor = np.floor(x)
    y0_floor = np.floor(y)
    wx = (x - x0_floor)[..., None].astype(np.float32)
    wy = (y - y0_floor)[..., None].astype(np.float32)
    x0 = x0_floor.astype(np.int64) % width
    y0 = y0_floor.astype(np.int64) % height
    x1 = (x0 + 1) % width
    y1 = (y0 + 1) % height
    top = texture[y0, x0] * (np.float32(1.0) - wx) + texture[y0, x1] * wx
    bottom = texture[y1, x0] * (np.float32(1.0) - wx) + texture[y1, x1] * wx
    return np.asarray(top * (np.float32(1.0) - wy) + bottom * wy, dtype=np.float32)


def generate_custom_hlsl(arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> str:
    """Serialize NPZ weights into a stable, row-major Custom-expression source."""

    w1 = arrays["network.0.weight"]
    b1 = arrays["network.0.bias"]
    w2 = arrays["network.2.weight"]
    b2 = arrays["network.2.bias"]
    lines = [
        "// Deterministically generated. Do not hand-edit weights.",
        f"// latent_sha256={metadata['latent_sha256']}",
        f"// decoder_sha256={metadata['decoder_sha256']}",
        "// Custom input: LatentRGBA (float4), NormalYSign (float; set -1 for the verified UE bridge).",
        "// Additional outputs: NormalTS (float3), Roughness (float1), Metallic (float1).",
        "// Return value: linear BaseColor RGB.",
        "",
    ]
    for row in range(8):
        lines.append(f"const float4 CGC_W1_{row} = float4({_float_list(w1[row])});")
        lines.append(f"const float CGC_B1_{row} = {_float_literal(b1[row])};")
    lines.append("")
    for row in range(7):
        lines.append(f"const float4 CGC_W2_{row}A = float4({_float_list(w2[row, :4])});")
        lines.append(f"const float4 CGC_W2_{row}B = float4({_float_list(w2[row, 4:])});")
        lines.append(f"const float CGC_B2_{row} = {_float_literal(b2[row])};")
    lines.append("")
    for row in range(8):
        lines.append(
            f"float CGC_H{row} = max(dot(LatentRGBA, CGC_W1_{row}) + CGC_B1_{row}, 0.0);"
        )
    lines.extend(
        [
            "float4 CGC_HA = float4(CGC_H0, CGC_H1, CGC_H2, CGC_H3);",
            "float4 CGC_HB = float4(CGC_H4, CGC_H5, CGC_H6, CGC_H7);",
        ]
    )
    for row in range(7):
        lines.append(
            f"float CGC_R{row} = dot(CGC_HA, CGC_W2_{row}A) + "
            f"dot(CGC_HB, CGC_W2_{row}B) + CGC_B2_{row};"
        )
    lines.extend(
        [
            "",
            "float3 CGC_BaseColorLinear = 1.0 / (1.0 + exp(-float3(CGC_R0, CGC_R1, CGC_R2)));",
            "float2 CGC_NormalXY = tanh(float2(CGC_R3, CGC_R4));",
            "float CGC_NormalRadius = length(CGC_NormalXY);",
            "CGC_NormalXY /= max(CGC_NormalRadius / (1.0 - 1.0e-6), 1.0);",
            "float CGC_NormalZ = sqrt(max(1.0 - dot(CGC_NormalXY, CGC_NormalXY), 1.0e-8));",
            "float3 CGC_NormalGLTF = normalize(float3(CGC_NormalXY, CGC_NormalZ));",
            "NormalTS = float3(CGC_NormalGLTF.x, CGC_NormalGLTF.y * NormalYSign, CGC_NormalGLTF.z);",
            "Roughness = 1.0 / (1.0 + exp(-CGC_R5));",
            "Metallic = 1.0 / (1.0 + exp(-CGC_R6));",
            "return CGC_BaseColorLinear;",
            "",
        ]
    )
    return "\n".join(lines)


def parse_generated_hlsl_constants(source: str) -> dict[str, np.ndarray]:
    """Reload generated constants so export never relies on hand-copied weights."""

    def rows(pattern: str, count: int, width: int) -> np.ndarray:
        matched = {
            int(match.group(1)): np.asarray(
                [np.float32(float(token.strip())) for token in match.group(2).split(",")],
                dtype=np.float32,
            )
            for match in re.finditer(pattern, source)
        }
        if set(matched) != set(range(count)):
            raise ValueError(f"generated HLSL constants are incomplete for {pattern}")
        result = np.stack([matched[index] for index in range(count)])
        if result.shape != (count, width):
            raise ValueError(f"generated HLSL constant shape {result.shape} != {(count, width)}")
        return result

    def scalars(pattern: str, count: int) -> np.ndarray:
        matched = {
            int(match.group(1)): np.float32(float(match.group(2)))
            for match in re.finditer(pattern, source)
        }
        if set(matched) != set(range(count)):
            raise ValueError(f"generated HLSL constants are incomplete for {pattern}")
        return np.asarray([matched[index] for index in range(count)], dtype=np.float32)

    w1 = rows(r"CGC_W1_(\d+)\s*=\s*float4\(([^)]+)\)", 8, 4)
    b1 = scalars(r"CGC_B1_(\d+)\s*=\s*([^;]+);", 8)
    w2_a = rows(r"CGC_W2_(\d+)A\s*=\s*float4\(([^)]+)\)", 7, 4)
    w2_b = rows(r"CGC_W2_(\d+)B\s*=\s*float4\(([^)]+)\)", 7, 4)
    b2 = scalars(r"CGC_B2_(\d+)\s*=\s*([^;]+);", 7)
    return {
        "network.0.weight": w1,
        "network.0.bias": b1,
        "network.2.weight": np.concatenate((w2_a, w2_b), axis=1),
        "network.2.bias": b2,
    }


def _jsonable_array(value: np.ndarray) -> list[Any]:
    return np.asarray(value, dtype=np.float32).tolist()


def _probe_record(name: str, latent: np.ndarray, arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    raw = decoder_raw_forward(latent, arrays)
    post = decoder_postprocess(raw)
    return {
        "name": name,
        "latent_rgba": _jsonable_array(latent),
        "raw_decoder": _jsonable_array(raw),
        "postprocess": {key: _jsonable_array(value) for key, value in post.items()},
    }


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def export_frozen_ue_package(repo_root: Path, output_dir: Path) -> dict[str, Any]:
    """Export and self-verify the complete frozen UE integration package."""

    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve()
    latent_path = repo_root / LATENT_RELATIVE_PATH
    decoder_path = repo_root / DECODER_RELATIVE_PATH
    latent_u8, arrays, metadata = validate_frozen_inputs(latent_path, decoder_path)

    output_dir.mkdir(parents=True, exist_ok=True)
    hlsl_path = output_dir / HLSL_FILENAME
    latent_export_path = output_dir / LATENT_EXPORT_FILENAME
    probes_path = output_dir / PROBES_FILENAME
    checklist_path = output_dir / CHECKLIST_FILENAME
    manifest_path = output_dir / MANIFEST_FILENAME

    hlsl = generate_custom_hlsl(arrays, metadata)
    # Round-trip generated literals before writing any deployable asset.
    parsed_arrays = parse_generated_hlsl_constants(hlsl)
    for name in EXPECTED_ARRAYS:
        if not np.array_equal(arrays[name], parsed_arrays[name]):
            raise ValueError(f"generated HLSL constant reload mismatch for {name}")

    shutil.copyfile(latent_path, latent_export_path)
    if sha256_file(latent_export_path) != EXPECTED_LATENT_SHA256:
        raise ValueError("copied UE latent no longer matches the frozen input")
    _write_text(hlsl_path, hlsl)

    sampled_latent = bilinear_sample_top_down_wrap(latent_u8, UV_PROBES)
    probe_payload = {
        "schema_version": 1,
        "float_comparison": "float32 NPZ versus float32 constants reparsed from generated HLSL",
        "canonical": _probe_record("canonical_rgba", CANONICAL_PROBES, parsed_arrays),
        "real_uv": {
            "uv_top_down_wrap": _jsonable_array(UV_PROBES),
            **_probe_record("frozen_latent_bilinear_then_decode", sampled_latent, parsed_arrays),
        },
    }
    _write_text(
        probes_path,
        json.dumps(probe_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )

    checklist = f"""# SciFiHelmet UE compressed-material integration checklist

Generated from the frozen pre-QAT hard artifacts. Do not hand-copy or edit weights.

1. Import `{LATENT_EXPORT_FILENAME}` as `/Game/CGCompression/Textures/T_SciFiHelmet_Latent_RGBA8`.
2. Set `sRGB=false`; preserve uncompressed RGBA8 (candidate UE setting: `VectorDisplacementmap (RGBA8)` / `TC_VectorDisplacementmap`); set bilinear filtering, U/V Wrap, and `Mip Gen Settings=NoMipmaps`.
3. Create one Custom expression in `/Game/CGCompression/Materials/Compressed/M_SciFiHelmet_Compressed` using `{HLSL_FILENAME}`. Inputs are `LatentRGBA=float4` and `NormalYSign=float`; additional outputs are `NormalTS=float3`, `Roughness=float1`, and `Metallic=float1`; the return type is float3 linear BaseColor.
4. Sample the latent once with the material UVs and feed the bilinear RGBA sample into the Custom expression. Set `NormalYSign=-1` provisionally because the reference normal import flips green; verify exactly once with an oblique light before freezing.
5. Connect return value to Base Color, `NormalTS` to Normal, Roughness to Roughness, and Metallic to Metallic. Leave AO disconnected and Specular at the Default Lit default 0.5.
6. Compile, save under `/Game/CGCompression/`, and place `Helmet_Reference` and `Helmet_Compressed` in the same `MaterialLab`; share mesh, transform except side-by-side translation, lights, exposure, and post process.
7. Save fixed reference, compressed, and side-by-side views plus the actual texture format/settings and shader compile log.
"""
    _write_text(checklist_path, checklist)

    manifest = {
        "schema_version": 1,
        "selection": "pre_qat_hard_tiny_mlp",
        "inputs": {
            "latent": {
                "path": LATENT_RELATIVE_PATH.as_posix(),
                "sha256": metadata["latent_sha256"],
                "format": "RGBA_UNORM8",
                "width": metadata["latent_width"],
                "height": metadata["latent_height"],
                "raw_gpu_bytes_no_mips": metadata["latent_raw_gpu_bytes_no_mips"],
            },
            "decoder": {
                "path": DECODER_RELATIVE_PATH.as_posix(),
                "sha256": metadata["decoder_sha256"],
                "npz_arrays": {
                    name: {"shape": list(EXPECTED_ARRAYS[name]), "dtype": "float32"}
                    for name in EXPECTED_ARRAYS
                },
            },
        },
        "decoder": {
            "architecture": "4->8->7",
            "activation": "ReLU",
            "parameter_count": PARAMETER_COUNT,
            "weight_bytes_float32": WEIGHT_BYTES_FLOAT32,
            "macs_per_pixel": MACS_PER_PIXEL,
            "raw_equation": "raw=W2*ReLU(W1*z+b1)+b2",
        },
        "runtime": {
            "order": [
                "RGBA_UNORM8_texel",
                "UE_bilinear_RGBA",
                "per_pixel_4x8x7_decoder",
                "single_channel_postprocess",
                "normal_positive_Z_and_UE_tangent_bridge",
                "UE_Default_Lit",
            ],
            "postprocess_once": {
                "base_color_linear_rgb": "sigmoid(raw[0:3])",
                "normal_xy_gltf_positive_y": "tanh(raw[3:5]) then unit-disk projection",
                "normal_z": "positive sqrt then normalize",
                "roughness_linear": "sigmoid(raw[5])",
                "metallic_linear": "sigmoid(raw[6])",
            },
            "base_color_additional_srgb_decode": False,
            "ambient_occlusion": "excluded",
        },
        "ue_faithful_baseline": {
            "texture_asset": "/Game/CGCompression/Textures/T_SciFiHelmet_Latent_RGBA8",
            "material_asset": "/Game/CGCompression/Materials/Compressed/M_SciFiHelmet_Compressed",
            "srgb": False,
            "rgba8_lossless_candidate": "TC_VectorDisplacementmap",
            "filter": "bilinear",
            "address_u": "wrap",
            "address_v": "wrap",
            "mips": "disabled",
            "normal_y_bridge": {
                "training_truth": "+Y glTF/OpenGL",
                "provisional_ue_sign": -1,
                "application_count": 1,
                "verification": "must compare against reference under an oblique light",
            },
            "custom_expression": {
                "inputs": {"LatentRGBA": "float4", "NormalYSign": "float1"},
                "return": {"BaseColorLinear": "float3"},
                "additional_outputs": {
                    "NormalTS": "float3",
                    "Roughness": "float1",
                    "Metallic": "float1",
                },
            },
        },
        "generated_files": {
            HLSL_FILENAME: sha256_file(hlsl_path),
            LATENT_EXPORT_FILENAME: sha256_file(latent_export_path),
            PROBES_FILENAME: sha256_file(probes_path),
            CHECKLIST_FILENAME: sha256_file(checklist_path),
        },
        "verification": {
            "generated_constant_reload_max_abs": 0.0,
            "canonical_probe_count": int(CANONICAL_PROBES.shape[0]),
            "real_uv_probe_count": int(UV_PROBES.shape[0]),
            "ue_visual_verification_pending": True,
        },
    }
    _write_text(
        manifest_path,
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export the frozen SciFiHelmet pre-QAT tiny MLP to a UE Custom expression package."
    )
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (repo_root / DEFAULT_OUTPUT_RELATIVE_PATH).resolve()
    )
    manifest = export_frozen_ue_package(repo_root, output_dir)
    print(
        json.dumps(
            {
                "manifest": str(output_dir / MANIFEST_FILENAME),
                "selection": manifest["selection"],
                "parameter_count": manifest["decoder"]["parameter_count"],
                "weight_bytes_float32": manifest["decoder"]["weight_bytes_float32"],
                "macs_per_pixel": manifest["decoder"]["macs_per_pixel"],
                "generated_constant_reload_max_abs": manifest["verification"][
                    "generated_constant_reload_max_abs"
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
