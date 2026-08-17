"""Generate the no-winner color/material/render Pareto report for O1/O2."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for directory in (SRC, SCRIPTS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from train_scifihelmet_c4_affine_raw_pca_orthogonal_10k import (  # noqa: E402
    _color_metrics,
    _legacy_atlas_metrics,
    _prepare,
)


DEFAULT_CONFIG = ROOT / "configs/train/scifihelmet_c4_affine_raw_pca_orthogonal_10k_v1.yaml"
DEFAULT_OUTPUT = ROOT / "outputs/scifihelmet_c4_affine_v1/raw_pca_orthogonal/bf215f-pareto-v1"
R0_ROOT = ROOT / "outputs/scifihelmet_c4_affine_v1/raw_pca_train/bf215f-r1-10k"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state(path: Path):
    payload = torch.load(path, map_location="cuda", weights_only=False)
    return payload["latent"].cuda(), payload["weight"].cuda(), payload["bias"].cuda()


def _flatten_metrics(atlas, color, render):
    material = atlas["material"]
    support = atlas["observed_valid_support"]
    full_cube = atlas["full_cube"]
    return {
        "uniform_y_error": color["uniform_y_error"],
        "uniform_opponent_error": color["uniform_opponent_error"],
        "adaptive_macro_opponent_error": color["adaptive_macro_opponent_error"],
        "worst_adaptive_group_opponent_error": color["worst_adaptive_group_opponent_error"],
        "chroma_contrast_retention": color["chroma_contrast_retention"],
        "base_color_linear_mae": material["base_color_linear_mae"],
        "seven_channel_mae": material["seven_channel_mae"],
        "normal_mean_degrees": material["normal_mean_degrees"],
        "roughness_mae": material["roughness_mae"],
        "metallic_mae": material["metallic_mae"],
        "hdr_mae": render["masked_linear_hdr_mae"],
        "hdr_rmse": render["masked_linear_hdr_rmse"],
        "display_ssim": render["display_ssim"],
        "display_psnr_db": render["display_psnr_db"],
        "observed_scalar_violation_fraction": support["scalar_violation_fraction"],
        "observed_scalar_max_violation": support["scalar_max_violation"],
        "observed_normal_violation_fraction": support["normal_violation_fraction"],
        "observed_normal_max_violation": support["normal_max_violation"],
        "full_cube_valid": bool(full_cube["valid"]),
    }


MINIMIZE = (
    "uniform_y_error",
    "uniform_opponent_error",
    "adaptive_macro_opponent_error",
    "worst_adaptive_group_opponent_error",
    "base_color_linear_mae",
    "seven_channel_mae",
    "normal_mean_degrees",
    "roughness_mae",
    "metallic_mae",
    "hdr_mae",
    "hdr_rmse",
)
MAXIMIZE = ("chroma_contrast_retention", "display_ssim", "display_psnr_db")


def _dominates(left, right):
    no_worse = all(left[key] <= right[key] for key in MINIMIZE) and all(
        left[key] >= right[key] for key in MAXIMIZE
    )
    strictly = any(left[key] < right[key] for key in MINIMIZE) or any(
        left[key] > right[key] for key in MAXIMIZE
    )
    return no_worse and strictly


def run(config_path: Path, output: Path):
    config = yaml.safe_load(config_path.read_bytes())
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Pareto root: {output}")
    output.mkdir(parents=True)
    prepared = _prepare(config, config_path)
    endpoint_states = {"R0": _state(R0_ROOT / "checkpoints/step_10000/checkpoint.pt")}
    reports = {"R0": json.loads((R0_ROOT / "training_report.json").read_text(encoding="utf-8"))}
    for candidate_id, candidate in config["candidates"].items():
        root = ROOT / candidate["output_root"]
        endpoint_states[candidate_id] = _state(root / "checkpoints/step_10000/checkpoint.pt")
        reports[candidate_id] = json.loads((root / "training_report.json").read_text(encoding="utf-8"))
    endpoints = {}
    for candidate_id, state in endpoint_states.items():
        atlas = _legacy_atlas_metrics(*state, prepared["valid_indices"], prepared["target_valid"])
        color = _color_metrics(*state, prepared)
        render = reports[candidate_id]["endpoint"]["full_render_31x6"]["mean"]
        endpoints[candidate_id] = _flatten_metrics(atlas, color, render)
    r0 = endpoints["R0"]
    ratios = {}
    for candidate_id, values in endpoints.items():
        ratios[candidate_id] = {
            key: (values[key] / r0[key] if isinstance(values[key], (int, float)) and not isinstance(values[key], bool) and r0[key] != 0 else None)
            for key in values
        }
    pareto = [
        candidate_id
        for candidate_id, values in endpoints.items()
        if not any(
            other_id != candidate_id and _dominates(other, values)
            for other_id, other in endpoints.items()
        )
    ]
    trajectory = {}
    roots = {"R0": R0_ROOT, **{key: ROOT / value["output_root"] for key, value in config["candidates"].items()}}
    for step in (1000, 5000, 10000):
        step_values = {}
        for candidate_id, root in roots.items():
            state = _state(root / "checkpoints" / f"step_{step:05d}" / "checkpoint.pt")
            color = _color_metrics(*state, prepared)
            step_values[candidate_id] = color
        control = step_values["R0"]
        trajectory[str(step)] = {
            candidate_id: {
                "metrics": values,
                "ratios_to_R0_same_step": {
                    key: values[key] / control[key] if control[key] != 0 else None
                    for key in (
                        "uniform_y_error",
                        "uniform_opponent_error",
                        "adaptive_macro_opponent_error",
                        "worst_adaptive_group_opponent_error",
                        "chroma_contrast_retention",
                    )
                },
            }
            for candidate_id, values in step_values.items()
        }
    report = {
        "schema_version": 1,
        "status": "complete_pareto_no_automatic_winner",
        "directions": {"minimize": list(MINIMIZE), "maximize": list(MAXIMIZE)},
        "endpoints": endpoints,
        "ratios_to_R0_10k": ratios,
        "non_dominated_set": pareto,
        "trajectory": trajectory,
        "yellow_diagnostics": {"selection_metric": False},
        "formal_holdout_accessed": False,
        "ue_started": False,
    }
    payload = (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    json_path = output / "pareto_report.json"
    json_path.write_bytes(payload)
    (output / "pareto_report.json.sha256").write_text(hashlib.sha256(payload).hexdigest() + "\n", encoding="ascii")
    lines = [
        "# SciFiHelmet Raw-PCA 正交损失 10k Pareto 报告",
        "",
        "本报告不设置硬 gate、不计算加权总分、不自动选择 winner。黄色诊断不参与 Pareto。",
        "",
        f"非支配集合：`{', '.join(pareto)}`",
        "",
        "| Endpoint | Opponent ↓ | Adaptive macro ↓ | Worst group ↓ | Chroma retention ↑ | Seven-channel ↓ | HDR MAE ↓ | SSIM ↑ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate_id, values in endpoints.items():
        lines.append(
            f"| {candidate_id} | {values['uniform_opponent_error']:.6g} | {values['adaptive_macro_opponent_error']:.6g} | {values['worst_adaptive_group_opponent_error']:.6g} | {values['chroma_contrast_retention']:.4f} | {values['seven_channel_mae']:.6g} | {values['hdr_mae']:.6g} | {values['display_ssim']:.6f} |"
        )
    markdown_path = output / "pareto_report.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json_sha256": _sha256(json_path), "markdown_sha256": _sha256(markdown_path), **report}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    report = run(args.config.resolve(), args.output_root.resolve())
    print(json.dumps({"status": report["status"], "pareto": report["non_dominated_set"]}))


if __name__ == "__main__":
    main()
