"""Run the bounded train-only C4 PCA representation and safety audit."""

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
    encode_pca_latent,
    export_p0_enhanced_bundle,
    fit_clustered_valid_pca_oracle,
    fit_enhanced_valid_pca,
    fit_global_valid_pca_oracle,
    fit_uniform_valid_pca,
    rasterize_uv_charts,
)
from cg_frontier.compression.affine_pca_audit import (
    cluster_balanced_candidate_specs,
    enhanced_candidate_specs,
    material_region_metrics,
    residual_reweighted_candidate_specs,
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


def _yellow_diagnostic(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, object]:
    yellow = (
        (target[:, 0] > target[:, 1])
        & (target[:, 1] > target[:, 2])
        & (target[:, 0] - target[:, 2] > 0.05)
        & (target[:, 1] - target[:, 2] > 0.02)
    )
    return {
        "selection_metric": False,
        "definition": "R>G>B and R-B>0.05 and G-B>0.02",
        "valid_texels": int(torch.count_nonzero(yellow)),
        "base_color_mae": float(F.l1_loss(prediction[yellow, :3], target[yellow, :3])),
        "source_mean_r_minus_b": float(torch.mean(target[yellow, 0] - target[yellow, 2])),
        "predicted_mean_r_minus_b": float(
            torch.mean(prediction[yellow, 0] - prediction[yellow, 2])
        ),
    }


def _bundle_prediction(bundle, target: torch.Tensor, valid_mask: torch.Tensor, *, safe: bool) -> torch.Tensor:
    artifact = bundle.calibration.safe if safe else bundle.calibration.raw
    latent = artifact.latent_unorm8[valid_mask].to(target.dtype) / 255.0
    return F.linear(latent, artifact.weight, artifact.bias)


def run(
    output_root: Path,
    *,
    margin: float,
    cpca_sample_count: int,
) -> dict[str, object]:
    output = output_root.resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("output root must stay inside the repository")
    lowered = output.as_posix().lower()
    if "formal_holdout" in lowered or "sealed" in lowered:
        raise ValueError("output root points at forbidden evaluation state")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite PCA audit output: {output}")
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
    target = target_atlas[valid_mask]
    report: dict[str, object] = {
        "schema_version": 1,
        "status": "running",
        "formal_holdout_accessed": False,
        "input": {
            "atlas_shape": list(target_atlas.shape),
            "valid_texels": int(target.shape[0]),
            "charts": int(torch.unique(chart_ids[valid_mask]).numel()),
        },
        "source": material_region_metrics(target, target),
        "global_rank_oracles": {},
        "clustered_oracles": {},
        "enhanced_global_q4": {},
        "preview_candidates": [
            "chroma8",
            "chroma8_k4_b05",
            "chroma8_k4_b10",
            "chroma8_resid3",
            "chroma8_resid7",
        ],
    }

    standard = fit_uniform_valid_pca(target_atlas, valid_mask)
    standard_prediction = standard.mean + standard.valid_scores @ standard.components
    report["standard_q4"] = {
        "metrics": material_region_metrics(standard_prediction, target),
        "yellow_diagnostic": _yellow_diagnostic(standard_prediction, target),
    }

    for rank in (5, 6):
        oracle = fit_global_valid_pca_oracle(target_atlas, valid_mask, rank=rank)
        prediction = oracle.mean + oracle.valid_scores @ oracle.components
        report["global_rank_oracles"][f"q{rank}"] = {
            "deployable": False,
            "metrics": material_region_metrics(prediction, target),
        }

    specs = {
        **enhanced_candidate_specs(),
        **cluster_balanced_candidate_specs(),
        **residual_reweighted_candidate_specs(),
    }
    for name, spec in specs.items():
        fitted = fit_enhanced_valid_pca(target_atlas, valid_mask, spec)
        encoding = encode_pca_latent(fitted)
        quantized = torch.floor(encoding.valid_latent * 255.0 + 0.5) / 255.0
        prediction = F.linear(quantized, encoding.weight, encoding.bias)
        report["enhanced_global_q4"][name] = {
            "deployable_family": True,
            "spec": {
                "semantic_group_balance": spec.semantic_group_balance,
                "opponent_chroma_weight": spec.opponent_chroma_weight,
                "chroma_tail_strength": spec.chroma_tail_strength,
                "material_cluster_count": spec.material_cluster_count,
                "material_cluster_balance_power": (
                    spec.material_cluster_balance_power
                ),
                "material_cluster_seed": spec.material_cluster_seed,
                "residual_tail_strength": spec.residual_tail_strength,
                "residual_reweight_iterations": (
                    spec.residual_reweight_iterations
                ),
            },
            "raw_metrics": material_region_metrics(prediction, target),
            "yellow_diagnostic": _yellow_diagnostic(prediction, target),
        }

    for name in report["preview_candidates"]:
        spec = specs[name]
        bundle = export_p0_enhanced_bundle(
            target_atlas,
            valid_mask,
            chart_ids,
            spec=spec,
            margin=margin,
        )
        candidate_root = output / "candidates" / name
        for filename, payload in bundle.files.items():
            _write_new(candidate_root / filename, payload)
        _write_new(candidate_root / "manifest.json", _json_bytes(bundle.manifest))
        raw_prediction = _bundle_prediction(bundle, target, valid_mask, safe=False)
        safe_prediction = _bundle_prediction(bundle, target, valid_mask, safe=True)
        report["enhanced_global_q4"][name]["frame_optimization"] = (
            bundle.manifest["frame_optimization"]
        )
        report["enhanced_global_q4"][name]["raw_post_frame_metrics"] = (
            material_region_metrics(raw_prediction, target)
        )
        report["enhanced_global_q4"][name]["safe_metrics"] = (
            material_region_metrics(safe_prediction, target)
        )
        report["enhanced_global_q4"][name]["safe_yellow_diagnostic"] = (
            _yellow_diagnostic(safe_prediction, target)
        )
        report["enhanced_global_q4"][name]["raw_artifact_hash"] = (
            bundle.calibration.raw.artifact_hash
        )
        report["enhanced_global_q4"][name]["safe_artifact_hash"] = (
            bundle.calibration.safe.artifact_hash
        )
        report["enhanced_global_q4"][name]["certificate"] = (
            bundle.calibration.safe.certificate
        )

    sample_count = min(cpca_sample_count, target.shape[0])
    sample_indices = torch.linspace(
        0, target.shape[0] - 1, sample_count, dtype=torch.float64
    ).round().to(torch.int64)
    sample = target[sample_indices]
    sample_atlas = sample.reshape(1, sample_count, 7)
    sample_valid = torch.ones(1, sample_count, dtype=torch.bool)
    for clusters in (2, 4, 8):
        oracle = fit_clustered_valid_pca_oracle(
            sample_atlas,
            sample_valid,
            clusters=clusters,
            rank=4,
            seed=20260807 + clusters,
            max_iterations=12,
        )
        report["clustered_oracles"][f"k{clusters}_q4"] = {
            "deployable": False,
            "fit_sample_count": sample_count,
            "iterations": oracle.iterations,
            "cluster_counts": [
                int(torch.count_nonzero(oracle.valid_assignments == cluster))
                for cluster in range(clusters)
            ],
            "metrics": material_region_metrics(
                oracle.valid_reconstruction, sample
            ),
        }

    report["status"] = "complete_cpu_train_only"
    payload = _json_bytes(report)
    _write_new(output / "audit_report.json", payload)
    _write_new(
        output / "audit_report.sha256",
        (hashlib.sha256(payload).hexdigest() + "\n").encode("ascii"),
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=1.0e-3)
    parser.add_argument("--cpca-sample-count", type=int, default=32_768)
    args = parser.parse_args()
    print(
        json.dumps(
            run(
                args.output_root,
                margin=args.margin,
                cpca_sample_count=args.cpca_sample_count,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
