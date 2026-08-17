"""UE preview bundle for the strict SciFiHelmet BaseColor experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from cg_frontier.compression.exact_basecolor_experiment import (
    stable_json_bytes,
    verify_runtime_export,
)


CANDIDATES = {
    "U0-unconstrained": "U0",
    "S-separated": "S",
    "M-mixed": "M",
}
CAMERAS = (0, 9, 19, 23)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _float(value: float) -> str:
    text = format(float(np.float32(value)), ".9g")
    if "." not in text and "e" not in text:
        text += ".0"
    return text + "f"


def build_exact_affine_hlsl(weight: np.ndarray, bias: np.ndarray) -> str:
    weight = np.asarray(weight, dtype=np.float32)
    bias = np.asarray(bias, dtype=np.float32)
    if weight.shape != (7, 4) or bias.shape != (7,):
        raise ValueError("exact BaseColor UE decoder must be a 4-to-7 affine")
    lines = [
        "// Generated strict-BaseColor 4->7 affine decoder.",
        "// One RGBA8 sample; BaseColor is returned directly in linear RGB.",
        "float4 z = LatentRGBA;",
    ]
    for row in range(7):
        vector = ", ".join(_float(value) for value in weight[row])
        lines.append(f"float raw{row} = dot(z, float4({vector})) + {_float(bias[row])};")
    lines.extend(
        (
            "float2 normalXY = tanh(float2(raw3, raw4));",
            "float normalRadius = length(normalXY);",
            "normalXY /= max(normalRadius / (1.0f - 1.0e-6f), 1.0f);",
            "float normalZ = sqrt(max(1.0f - dot(normalXY, normalXY), 1.0e-8f));",
            "NormalTS = normalize(float3(normalXY.x, normalXY.y * NormalYSign, normalZ));",
            "Roughness = saturate(1.0f / (1.0f + exp(-raw5)));",
            "Metallic = saturate(1.0f / (1.0f + exp(-raw6)));",
            "return float3(raw0, raw1, raw2);",
        )
    )
    return "\n".join(lines) + "\n"


def _copy_verified_candidate(
    *,
    training_root: Path,
    bundle_root: Path,
    candidate: str,
    slug: str,
    final_metrics: dict[str, Any],
) -> dict[str, Any]:
    source = training_root / candidate / "export"
    runtime = verify_runtime_export(source)
    texture_source = source / "latent_rgba_unorm8.png"
    decoder_source = source / "decoder_affine.npz"
    texture_target = bundle_root / f"T_SciFiHelmet_ExactBC_{slug}_RGBA8.png"
    decoder_target = bundle_root / f"D_SciFiHelmet_ExactBC_{slug}_Affine.npz"
    hlsl_target = bundle_root / f"M_SciFiHelmet_ExactBC_{slug}.custom.hlsl"
    shutil.copyfile(texture_source, texture_target)
    shutil.copyfile(decoder_source, decoder_target)
    with np.load(decoder_source, allow_pickle=False) as decoder:
        weight = np.asarray(decoder["weight"], dtype=np.float32)
        bias = np.asarray(decoder["bias"], dtype=np.float32)
    hlsl_target.write_text(build_exact_affine_hlsl(weight, bias), encoding="utf-8")
    return {
        "candidate": candidate,
        "slug": slug,
        "recommended": candidate == "S-separated",
        "runtime_verification": runtime,
        "files": {
            "texture": {"name": texture_target.name, "sha256": _sha256(texture_target)},
            "decoder": {"name": decoder_target.name, "sha256": _sha256(decoder_target)},
            "hlsl": {"name": hlsl_target.name, "sha256": _sha256(hlsl_target)},
        },
        "affine": {"weight": weight.tolist(), "bias": bias.tolist()},
        "postprocess": {
            "basecolor": "direct linear RGB",
            "normal_xy": "tanh + positive-hemisphere projection",
            "normal_y_ue_bridge": -1.0,
            "roughness": "sigmoid",
            "metallic": "sigmoid",
        },
        "offline_10k_metrics": final_metrics,
        "ue_assets": {
            "texture": f"/Game/CGCompression/ExactBaseColorV1/Textures/T_ExactBC_{slug}_RGBA8",
            "material": f"/Game/CGCompression/ExactBaseColorV1/Materials/M_ExactBC_{slug}",
            "actor": f"Helmet_ExactBC_{slug}",
        },
    }


def _render_montage(training_root: Path, output_path: Path) -> None:
    tile = 256
    header = 34
    row_header = 110
    canvas = Image.new(
        "RGB",
        (row_header + tile * len(CAMERAS), header + tile * len(CANDIDATES)),
        "#101418",
    )
    draw = ImageDraw.Draw(canvas)
    for column, camera in enumerate(CAMERAS):
        draw.text((row_header + column * tile + 8, 10), f"camera {camera:02d} / light 0", fill="white")
    for row, (candidate, slug) in enumerate(CANDIDATES.items()):
        y = header + row * tile
        label = f"{slug}: {candidate}"
        draw.text((8, y + 12), label, fill="#7ee787" if slug == "S" else "white")
        for column, camera in enumerate(CAMERAS):
            source = training_root / candidate / "render_10000" / f"camera_{camera:02d}.png"
            image = Image.open(source).convert("RGB").resize((tile, tile), Image.Resampling.LANCZOS)
            canvas.paste(image, (row_header + column * tile, y))
    canvas.save(output_path)


def _selected_render_sheet(training_root: Path, output_path: Path) -> None:
    tile = 256
    label_height = 24
    canvas = Image.new("RGB", (tile * 2, (tile + label_height) * 2), "#101418")
    draw = ImageDraw.Draw(canvas)
    source_root = training_root / "S-separated" / "render_10000"
    for index, camera in enumerate(CAMERAS):
        column, row = index % 2, index // 2
        x, y = column * tile, row * (tile + label_height)
        draw.text((x + 8, y + 5), f"S-separated · camera {camera:02d} · light 0", fill="#7ee787")
        image = Image.open(source_root / f"camera_{camera:02d}.png").convert("RGB")
        canvas.paste(image, (x, y + label_height))
    canvas.save(output_path)


def export_ue_preview_bundle(experiment_root: Path, output_root: Path) -> dict[str, Any]:
    experiment_root = experiment_root.resolve()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = experiment_root / "report" / "final_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    training_root = experiment_root / "training"
    records = {}
    for candidate, slug in CANDIDATES.items():
        records[candidate] = _copy_verified_candidate(
            training_root=training_root,
            bundle_root=output_root,
            candidate=candidate,
            slug=slug,
            final_metrics=summary["10k_main_metrics"][candidate],
        )
    montage = output_root / "SciFiHelmet_ExactBaseColor_10k_RenderMontage.png"
    _render_montage(training_root, montage)
    selected_sheet = output_root / "SciFiHelmet_S_Separated_10k_4View.png"
    _selected_render_sheet(training_root, selected_sheet)
    for source_name in ("basecolor_u0_s_m.png", "training_trajectories.png"):
        shutil.copyfile(experiment_root / "report" / source_name, output_root / source_name)
    manifest = {
        "schema_version": 1,
        "experiment": "scifihelmet_exact_basecolor_v1",
        "selection": "S-separated",
        "selection_reason": "strict BaseColor; mixed advantage gate failed",
        "runtime": {
            "samples_per_pixel": 1,
            "texture_format": "RGBA8 linear",
            "filter": "bilinear",
            "address": "wrap",
            "mips": False,
            "decoder": "single 4-to-7 affine",
        },
        "scope": summary["scope"],
        "preview_map": "/Game/CGCompression/ExactBaseColorV1/Maps/ExactBaseColorV1",
        "candidates": records,
        "visuals": {
            montage.name: _sha256(montage),
            selected_sheet.name: _sha256(selected_sheet),
            "basecolor_u0_s_m.png": _sha256(output_root / "basecolor_u0_s_m.png"),
            "training_trajectories.png": _sha256(output_root / "training_trajectories.png"),
        },
    }
    manifest_path = output_root / "ue_preview_manifest.json"
    manifest_path.write_bytes(stable_json_bytes(manifest))
    readme = output_root / "README_UE.md"
    readme.write_text(
        "# SciFiHelmet Exact BaseColor V1 — UE preview\n\n"
        "Recommended asset: `/Game/CGCompression/ExactBaseColorV1/Materials/M_ExactBC_S`.\n\n"
        "Open `/Game/CGCompression/ExactBaseColorV1/Maps/ExactBaseColorV1` and inspect "
        "`Helmet_Reference`, `Helmet_ExactBC_U0`, `Helmet_ExactBC_S`, and `Helmet_ExactBC_M`.\n\n"
        "Textures must stay linear RGBA8, bilinear, Wrap, no mips, never stream. "
        "The Custom node performs one 4→7 affine decode and applies the UE tangent-Y bridge exactly once.\n",
        encoding="utf-8",
    )
    return manifest


def verify_ue_texture_readbacks(bundle_root: Path) -> dict[str, Any]:
    bundle_root = bundle_root.resolve()
    manifest = json.loads((bundle_root / "ue_preview_manifest.json").read_text(encoding="utf-8"))
    candidates = {}
    for record in manifest["candidates"].values():
        source_path = bundle_root / record["files"]["texture"]["name"]
        readback_path = bundle_root / "evidence" / f"ue_readback_{record['slug']}.png"
        source = np.asarray(Image.open(source_path).convert("RGBA"), dtype=np.uint8)
        readback = np.asarray(Image.open(readback_path).convert("RGBA"), dtype=np.uint8)
        shape_equal = source.shape == readback.shape
        maximum = int(np.abs(source.astype(np.int16) - readback.astype(np.int16)).max()) if shape_equal else None
        candidates[record["candidate"]] = {
            "source": source_path.name,
            "readback": readback_path.relative_to(bundle_root).as_posix(),
            "shape_equal": shape_equal,
            "byte_equal": bool(shape_equal and np.array_equal(source, readback)),
            "max_abs": maximum,
            "readback_sha256": _sha256(readback_path),
        }
    report = {
        "schema_version": 1,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "passed": all(value["byte_equal"] for value in candidates.values()),
    }
    (bundle_root / "ue_readback_verification.json").write_bytes(stable_json_bytes(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path("outputs/scifihelmet_exact_basecolor_v1"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/scifihelmet_exact_basecolor_v1/ue_preview_bundle"),
    )
    parser.add_argument("--verify-readbacks", action="store_true")
    args = parser.parse_args()
    manifest = export_ue_preview_bundle(args.experiment_root, args.output_root)
    result = {"selection": manifest["selection"], "output": str(args.output_root)}
    if args.verify_readbacks:
        result["ue_readbacks"] = verify_ue_texture_readbacks(args.output_root)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
