"""Export the isolated certified PCA repair candidate for SciFiHelmet Core-4."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cg_frontier.assets.gltf_mesh import load_gltf_mesh
from cg_frontier.compression.affine_pca import (
    export_p0_constrained_bundle,
    rasterize_uv_charts,
)
from cg_frontier.compression.material import Core4Targets, load_core4_targets


GLTF_PATH = ROOT / "assets/source/glTF-Sample-Assets/Models/SciFiHelmet/glTF/SciFiHelmet.gltf"
CORE4_ROOT = ROOT / "assets/processed/SciFiHelmet/core4"


def _targets_to_seven(targets: Core4Targets) -> torch.Tensor:
    return torch.cat(
        (
            targets.base_color_linear,
            targets.normal_xyz[:, :2],
            targets.roughness,
            targets.metallic,
        ),
        dim=-1,
    ).reshape(targets.height, targets.width, 7)


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)


def _mae_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    return {
        "seven_mae": float(F.l1_loss(prediction, target)),
        "base_color_mae": float(F.l1_loss(prediction[:, :3], target[:, :3])),
        "normal_xy_mae": float(F.l1_loss(prediction[:, 3:5], target[:, 3:5])),
        "roughness_mae": float(F.l1_loss(prediction[:, 5], target[:, 5])),
        "metallic_mae": float(F.l1_loss(prediction[:, 6], target[:, 6])),
    }


def run(output_root: Path, *, margin: float) -> dict[str, object]:
    output = output_root.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("output root must stay inside the repository")
    lowered = output.as_posix().lower()
    if "formal_holdout" in lowered or "sealed" in lowered:
        raise ValueError("output root points at forbidden evaluation state")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite PCA repair output: {output}")
    output.mkdir(parents=True)

    core4 = load_core4_targets(CORE4_ROOT, "cpu")
    target_atlas = _targets_to_seven(core4)
    mesh = load_gltf_mesh(GLTF_PATH)
    valid_mask, chart_ids = rasterize_uv_charts(
        mesh.texcoords,
        mesh.triangles,
        height=core4.height,
        width=core4.width,
    )
    bundle = export_p0_constrained_bundle(
        target_atlas,
        valid_mask,
        chart_ids,
        margin=margin,
    )

    safe_root = output / "p0-safe-constrained-v1"
    for name, payload in bundle.files.items():
        _write_new(safe_root / name, payload)
    _write_new(safe_root / "manifest.json", _json_bytes(bundle.manifest))

    raw_payload = torch.cat(
        (
            bundle.calibration.raw.weight.float().reshape(-1),
            bundle.calibration.raw.bias.float(),
        )
    ).numpy().tobytes()
    raw_root = output / "p0-raw-constrained-v1"
    _write_new(raw_root / "latent_rgba8.png", bundle.files["latent_rgba8.png"])
    _write_new(raw_root / "decoder.bin", raw_payload)
    _write_new(
        raw_root / "manifest.json",
        _json_bytes(
            {
                "schema_version": 1,
                "pipeline_id": "scifihelmet_c4_affine_pca_repair_v1",
                "artifact_id": bundle.calibration.raw.artifact_id,
                "artifact_hash": bundle.calibration.raw.artifact_hash,
                "deployable": False,
                "target_material_mae": bundle.calibration.raw.material_mae,
                "decoder_sha256": hashlib.sha256(raw_payload).hexdigest(),
                "latent_png_sha256": bundle.manifest["hashes"]["latent_png_sha256"],
            }
        ),
    )

    target = target_atlas[valid_mask]
    latent = bundle.calibration.safe.latent_unorm8[valid_mask].to(target.dtype) / 255.0
    raw_prediction = F.linear(
        latent, bundle.calibration.raw.weight, bundle.calibration.raw.bias
    )
    safe_prediction = F.linear(
        latent, bundle.calibration.safe.weight, bundle.calibration.safe.bias
    )
    yellow = (
        (target[:, 0] > target[:, 1])
        & (target[:, 1] > target[:, 2])
        & (target[:, 0] - target[:, 2] > 0.05)
        & (target[:, 1] - target[:, 2] > 0.02)
    )
    report: dict[str, object] = {
        "schema_version": 1,
        "status": "complete_cpu_read_only_core4",
        "formal_holdout_accessed": False,
        "pipeline_id": bundle.manifest["pipeline_id"],
        "raw": _mae_metrics(raw_prediction, target),
        "safe": _mae_metrics(safe_prediction, target),
        "raw_to_safe_seven_mae_increment": (
            bundle.calibration.safety_material_mae_increment
        ),
        "raw_artifact_hash": bundle.calibration.raw.artifact_hash,
        "safe_artifact_hash": bundle.calibration.safe.artifact_hash,
        "certificate": bundle.calibration.safe.certificate,
        "yellow_diagnostic": {
            "definition": "R>G>B and R-B>0.05 and G-B>0.02",
            "valid_texels": int(torch.count_nonzero(yellow)),
            "source_mean_r_minus_b": float(
                torch.mean(target[yellow, 0] - target[yellow, 2])
            ),
            "raw_base_color_mae": float(
                F.l1_loss(raw_prediction[yellow, :3], target[yellow, :3])
            ),
            "raw_mean_r_minus_b": float(
                torch.mean(raw_prediction[yellow, 0] - raw_prediction[yellow, 2])
            ),
            "safe_base_color_mae": float(
                F.l1_loss(safe_prediction[yellow, :3], target[yellow, :3])
            ),
            "safe_mean_r_minus_b": float(
                torch.mean(safe_prediction[yellow, 0] - safe_prediction[yellow, 2])
            ),
        },
    }
    _write_new(output / "raw_gap_report.json", _json_bytes(report))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=1.0e-3)
    args = parser.parse_args()
    print(json.dumps(run(args.output_root, margin=args.margin), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
