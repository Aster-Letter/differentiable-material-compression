"""Deterministic UE export for the frozen R0b Hybrid direct-scalar winner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shutil
from typing import Any

import numpy as np
from PIL import Image

from cg_frontier.compression.ue_export import bilinear_sample_top_down_wrap, sha256_file


FINAL_REL = Path("outputs/compression/scifihelmet/hybrid_direct_scalars_v1/final_summary.json")
TEXTURE_A_REL = Path("outputs/compression/scifihelmet/hybrid_direct_scalars_v1/r0b/texture_a_base_rgb_normal0_rgba8.png")
TEXTURE_B_REL = Path("outputs/compression/scifihelmet/hybrid_direct_scalars_v1/r0b/texture_b_normal1_roughness_metallic_rgb8.png")
DECODER_REL = Path("outputs/compression/scifihelmet/hybrid_direct_scalars_v1/r0b/decoder_weights.npz")
DEFAULT_OUTPUT_REL = Path("outputs/deployment/scifihelmet/hybrid_direct_scalars_r0b")

EXPECTED = {
    "final": "ef87415f97bcb214ae76eb1456c45139e5a7987cb282f8bcadb0f8fe018721b6",
    "texture_a": "4d99be439c7579bc6c42de0f1b9fe48f89fd9a1f0d2a8e9d3e17060fab81e914",
    "texture_b": "70b4fe1289542bd64e45a43ea0ea87e76674586552b0a9ca1af54853e6b01cc1",
    "decoder": "e63a7afd4640fd330b3cdbcfe85121abc477b5f6c0024f1c31a4fdfc57e18b6c",
}
ARRAYS = {
    "normal_head.0.weight": (6, 2),
    "normal_head.0.bias": (6,),
    "normal_head.2.weight": (2, 6),
    "normal_head.2.bias": (2,),
    "direct_scalars.marker": (1,),
}

HLSL_NAME = "M_SciFiHelmet_Hybrid_R0B.custom.hlsl"
TEXTURE_A_NAME = "T_SciFiHelmet_Hybrid_R0B_A_RGBA8.png"
TEXTURE_B_NAME = "T_SciFiHelmet_Hybrid_R0B_B_RGB8.png"
PROBES_NAME = "probe_vectors.json"
MANIFEST_NAME = "deployment_manifest.json"
CHECKLIST_NAME = "UE_INTEGRATION_CHECKLIST.md"

UV_PROBES = np.asarray([[0.125, 0.25], [0.5, 0.5], [0.9999, 0.0001], [-0.125, 1.125]], dtype=np.float32)


def _float(value: np.float32) -> str:
    text = format(float(np.float32(value)), ".9g")
    return text if "." in text or "e" in text.lower() else text + ".0"


def _list(values: np.ndarray) -> str:
    return ", ".join(_float(value) for value in np.asarray(values).reshape(-1))


def validate(repo: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    paths = {"final": repo / FINAL_REL, "texture_a": repo / TEXTURE_A_REL, "texture_b": repo / TEXTURE_B_REL, "decoder": repo / DECODER_REL}
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    if hashes != EXPECTED:
        raise ValueError(f"frozen R0b hash mismatch: {hashes}")
    summary = json.loads(paths["final"].read_text(encoding="utf-8"))
    if summary.get("winner") != "r0b" or summary.get("formal_holdout_accessed") is not False:
        raise ValueError("final summary does not authorize frozen R0b")
    if summary["candidates"]["r0b"]["offline_pass"] is not True or summary["candidates"]["r0b"]["gate_count"] != 13:
        raise ValueError("R0b is not a 13/13 offline winner")
    with Image.open(paths["texture_a"]) as image:
        if image.mode != "RGBA" or image.size != (2048, 2048):
            raise ValueError("Texture A must be 2048x2048 RGBA8")
        a = np.asarray(image, dtype=np.uint8).copy()
    with Image.open(paths["texture_b"]) as image:
        if image.mode != "RGB" or image.size != (2048, 2048):
            raise ValueError("Texture B must be 2048x2048 logical RGB8")
        b = np.asarray(image, dtype=np.uint8).copy()
    with np.load(paths["decoder"], allow_pickle=False) as stored:
        if set(stored.files) != set(ARRAYS):
            raise ValueError(f"unexpected decoder keys: {stored.files}")
        arrays = {name: np.asarray(stored[name]).copy() for name in ARRAYS}
    for name, shape in ARRAYS.items():
        if arrays[name].shape != shape or arrays[name].dtype != np.float32 or not np.isfinite(arrays[name]).all():
            raise ValueError(f"invalid decoder array {name}")
    parameter_count = sum(arrays[name].size for name in ARRAYS if not name.endswith("marker"))
    weight_bytes = sum(arrays[name].nbytes for name in ARRAYS if not name.endswith("marker"))
    if (parameter_count, weight_bytes) != (32, 128):
        raise ValueError("R0b decoder cost mismatch")
    return a, b, arrays, {"hashes": hashes, "parameter_count": 32, "weight_bytes_float32": 128, "macs_per_pixel": 24}


def raw_forward(normal_latent: np.ndarray, arrays: dict[str, np.ndarray]) -> np.ndarray:
    values = np.asarray(normal_latent, dtype=np.float32)
    hidden = np.maximum(
        np.sum(values[..., None, :] * arrays["normal_head.0.weight"], axis=-1, dtype=np.float32)
        + arrays["normal_head.0.bias"],
        np.float32(0.0),
    )
    return np.sum(hidden[..., None, :] * arrays["normal_head.2.weight"], axis=-1, dtype=np.float32) + arrays["normal_head.2.bias"]


def postprocess(sample_a: np.ndarray, sample_b: np.ndarray, arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    raw = raw_forward(np.stack((sample_a[..., 3], sample_b[..., 0]), axis=-1), arrays)
    xy = np.tanh(raw).astype(np.float32)
    radius = np.sqrt(np.sum(xy * xy, axis=-1, keepdims=True, dtype=np.float32))
    xy = xy / np.maximum(radius / np.float32(1.0 - 1.0e-6), np.float32(1.0))
    z = np.sqrt(np.maximum(np.float32(1.0) - np.sum(xy * xy, axis=-1, keepdims=True, dtype=np.float32), np.float32(1.0e-8)))
    normal = np.concatenate((xy, z), axis=-1).astype(np.float32)
    normal /= np.maximum(np.sqrt(np.sum(normal * normal, axis=-1, keepdims=True, dtype=np.float32)), np.float32(1.0e-8))
    return {
        "base_color_linear": np.asarray(sample_a[..., :3], dtype=np.float32),
        "normal_gltf_positive_y": normal,
        "roughness_linear": np.asarray(sample_b[..., 1:2], dtype=np.float32),
        "metallic_linear": np.asarray(sample_b[..., 2:3], dtype=np.float32),
    }


def generate_hlsl(arrays: dict[str, np.ndarray], metadata: dict[str, Any]) -> str:
    w1, b1 = arrays["normal_head.0.weight"], arrays["normal_head.0.bias"]
    w2, b2 = arrays["normal_head.2.weight"], arrays["normal_head.2.bias"]
    lines = [
        "// Deterministically generated R0b Hybrid direct-scalar decoder.",
        f"// texture_a_sha256={metadata['hashes']['texture_a']}",
        f"// texture_b_sha256={metadata['hashes']['texture_b']}",
        f"// decoder_sha256={metadata['hashes']['decoder']}",
        "// Inputs: TextureA=float4, TextureB=float3, NormalYSign=float.",
        "// Return: direct linear BaseColor. Outputs: NormalTS, direct Roughness, direct Metallic.",
        "",
    ]
    for row in range(6):
        lines.append(f"const float2 CGH_W1_{row} = float2({_list(w1[row])});")
        lines.append(f"const float CGH_B1_{row} = {_float(b1[row])};")
    for row in range(2):
        lines.append(f"const float3 CGH_W2_{row}A = float3({_list(w2[row, :3])});")
        lines.append(f"const float3 CGH_W2_{row}B = float3({_list(w2[row, 3:])});")
        lines.append(f"const float CGH_B2_{row} = {_float(b2[row])};")
    lines.extend(["", "float2 CGH_Z = float2(TextureA.a, TextureB.r);"])
    for row in range(6):
        lines.append(f"float CGH_H{row} = max(dot(CGH_Z, CGH_W1_{row}) + CGH_B1_{row}, 0.0);")
    lines.extend([
        "float3 CGH_HA = float3(CGH_H0, CGH_H1, CGH_H2);",
        "float3 CGH_HB = float3(CGH_H3, CGH_H4, CGH_H5);",
        "float2 CGH_RawNormal = float2(dot(CGH_HA, CGH_W2_0A) + dot(CGH_HB, CGH_W2_0B) + CGH_B2_0, dot(CGH_HA, CGH_W2_1A) + dot(CGH_HB, CGH_W2_1B) + CGH_B2_1);",
        "float2 CGH_NormalXY = tanh(CGH_RawNormal);",
        "CGH_NormalXY /= max(length(CGH_NormalXY) / (1.0 - 1.0e-6), 1.0);",
        "float CGH_NormalZ = sqrt(max(1.0 - dot(CGH_NormalXY, CGH_NormalXY), 1.0e-8));",
        "float3 CGH_NormalGLTF = normalize(float3(CGH_NormalXY, CGH_NormalZ));",
        "NormalTS = float3(CGH_NormalGLTF.x, CGH_NormalGLTF.y * NormalYSign, CGH_NormalGLTF.z);",
        "Roughness = TextureB.g;",
        "Metallic = TextureB.b;",
        "return TextureA.rgb;",
        "",
    ])
    return "\n".join(lines)


def parse_hlsl(source: str) -> dict[str, np.ndarray]:
    def rows(pattern: str, count: int, width: int) -> np.ndarray:
        found = {int(m.group(1)): np.asarray([np.float32(float(x.strip())) for x in m.group(2).split(",")]) for m in re.finditer(pattern, source)}
        result = np.stack([found[index] for index in range(count)])
        if result.shape != (count, width):
            raise ValueError("HLSL rows incomplete")
        return result.astype(np.float32)
    def scalars(pattern: str, count: int) -> np.ndarray:
        found = {int(m.group(1)): np.float32(float(m.group(2))) for m in re.finditer(pattern, source)}
        return np.asarray([found[index] for index in range(count)], dtype=np.float32)
    return {
        "normal_head.0.weight": rows(r"CGH_W1_(\d+)\s*=\s*float2\(([^)]+)\)", 6, 2),
        "normal_head.0.bias": scalars(r"CGH_B1_(\d+)\s*=\s*([^;]+);", 6),
        "normal_head.2.weight": np.concatenate((rows(r"CGH_W2_(\d+)A\s*=\s*float3\(([^)]+)\)", 2, 3), rows(r"CGH_W2_(\d+)B\s*=\s*float3\(([^)]+)\)", 2, 3)), axis=1),
        "normal_head.2.bias": scalars(r"CGH_B2_(\d+)\s*=\s*([^;]+);", 2),
    }


def export(repo: Path, output: Path) -> dict[str, Any]:
    repo, output = repo.resolve(), output.resolve()
    a, b, arrays, metadata = validate(repo)
    output.mkdir(parents=True, exist_ok=True)
    hlsl = generate_hlsl(arrays, metadata)
    parsed = parse_hlsl(hlsl)
    for name in parsed:
        if not np.array_equal(arrays[name], parsed[name]):
            raise ValueError(f"HLSL round-trip mismatch: {name}")
    shutil.copyfile(repo / TEXTURE_A_REL, output / TEXTURE_A_NAME)
    shutil.copyfile(repo / TEXTURE_B_REL, output / TEXTURE_B_NAME)
    (output / HLSL_NAME).write_text(hlsl, encoding="utf-8", newline="\n")
    sampled_a = bilinear_sample_top_down_wrap(a, UV_PROBES)
    sampled_b = bilinear_sample_top_down_wrap(b, UV_PROBES)
    decoded = postprocess(sampled_a, sampled_b, parsed)
    probes = {
        "schema_version": 1,
        "uv_top_down_wrap": UV_PROBES.tolist(),
        "sample_a": sampled_a.tolist(),
        "sample_b": sampled_b.tolist(),
        "decoded": {name: value.tolist() for name, value in decoded.items()},
        "max_abs_npz_vs_hlsl_constants": 0.0,
    }
    (output / PROBES_NAME).write_text(json.dumps(probes, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    checklist = """# R0b UE integration checklist

- Import Texture A/B under `/Game/CGCompression/Textures/HybridInterpolation/R0B/` with sRGB off, VectorDisplacementmap, bilinear, Wrap, NoMipmaps, never stream.
- Texture B is logical RGB8; record the actual UE source/resource format and count RGBA8 expansion as 16 MiB.
- Use one Custom expression under `/Game/CGCompression/Materials/HybridInterpolation/R0B/`; feed both samples from the same TexCoord0.
- Return direct TextureA.rgb, output normal-only decode, direct TextureB.g roughness and TextureB.b metallic. Apply NormalYSign=-1 exactly once; leave AO unconnected.
- Reject any source/readback mismatch or lossy compression. Do not overwrite baseline/reference/imported assets.
"""
    (output / CHECKLIST_NAME).write_text(checklist, encoding="utf-8", newline="\n")
    manifest = {
        "schema_version": 1,
        "selection": "r0b_hybrid_direct_scalars_offline_13_of_13",
        "formal_holdout_accessed": False,
        "inputs": {
            "final_summary": {"path": FINAL_REL.as_posix(), "sha256": EXPECTED["final"]},
            "texture_a": {"path": TEXTURE_A_REL.as_posix(), "sha256": EXPECTED["texture_a"], "logical_format": "RGBA8_UNORM", "raw_bytes": int(a.nbytes)},
            "texture_b": {"path": TEXTURE_B_REL.as_posix(), "sha256": EXPECTED["texture_b"], "logical_format": "RGB8_UNORM", "logical_raw_bytes": int(b.nbytes), "ue_physical_ceiling_bytes": 2048 * 2048 * 4},
            "decoder": {"path": DECODER_REL.as_posix(), "sha256": EXPECTED["decoder"]},
        },
        "decoder": {"architecture": "normal_2->6->2", "activation": "ReLU", "parameter_count": 32, "weight_bytes_float32": 128, "macs_per_pixel": 24},
        "runtime": {
            "texture_samples": 2,
            "same_uv_wrap_bilinear": True,
            "base_color": "TextureA.rgb direct linear; no decoder/sigmoid/sRGB",
            "normal": "float2(TextureA.a,TextureB.r)->2x6x2->tanh->+Z->UE Y*-1 once",
            "roughness": "TextureB.g direct linear",
            "metallic": "TextureB.b direct linear",
            "ambient_occlusion": "excluded",
        },
        "budget": {"logical_raw_bytes": 2048 * 2048 * 7, "ue_physical_ceiling_bytes": 2048 * 2048 * 8},
        "ue_assets": {
            "texture_folder": "/Game/CGCompression/Textures/HybridInterpolation/R0B",
            "material_folder": "/Game/CGCompression/Materials/HybridInterpolation/R0B",
            "map": "/Game/CGCompression/Maps/HybridInterpolation/R0B_Acceptance",
        },
        "generated_files": {},
        "deployment_exported": True,
    }
    for name in (HLSL_NAME, TEXTURE_A_NAME, TEXTURE_B_NAME, PROBES_NAME, CHECKLIST_NAME):
        manifest["generated_files"][name] = sha256_file(output / name)
    (output / MANIFEST_NAME).write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output = args.output_dir or args.repo_root / DEFAULT_OUTPUT_REL
    manifest = export(args.repo_root, output)
    print(json.dumps({"status": "ok", "selection": manifest["selection"], "output": str(output)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
