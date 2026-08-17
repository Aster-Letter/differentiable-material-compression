"""Freeze C1 near-gate and final bounded Hybrid conclusion."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cg_frontier.compression.artifact_analysis import deterministic_json, sha256_file  # noqa: E402


def _load(path: Path) -> dict[str, Any]:
    if "formal_holdout" in path.as_posix().lower():
        raise ValueError("sealed formal holdout path is forbidden")
    return json.loads(path.read_text(encoding="utf-8"))


def run() -> dict[str, Any]:
    root = ROOT / "outputs/compression/scifihelmet/hybrid_interpolation_v1"
    phase0_path = root / "phase0/phase0_summary.json"
    gates_path = root / "c1/gates.json"
    training_path = root / "c1/training_manifest.json"
    analysis_path = root / "c1/analysis/interpolation_analysis.json"
    analysis_manifest_path = root / "c1/analysis/manifest.json"
    phase0, gates, training, analysis, analysis_manifest = (
        _load(path) for path in (phase0_path, gates_path, training_path, analysis_path, analysis_manifest_path)
    )
    if any(value.get("formal_holdout_accessed") is not False for value in (phase0, gates, training, analysis)):
        raise RuntimeError("Hybrid evidence does not preserve the sealed holdout")
    values = gates["gates"]
    base_names = (
        "D2_novel_dark_relative_reduction",
        "D2_novel_dark_absolute_fraction",
        "D2_max_connected_patch_relative_reduction",
        "D1_novel_dark_relative_reduction",
        "D3_novel_dark_relative_reduction",
        "D2_base_color_p99_relative_reduction",
        "global_texel_base_p99_regression",
    )
    base_all = all(bool(values[name]["passed"]) for name in base_names)
    metal_mae = float(values["metallic_boundary_mae_relative_reduction"]["value"])
    metal_fraction = float(values["metallic_boundary_above_0_1_relative_reduction"]["value"])
    checks = {
        "seven_basecolor_gates_pass": base_all,
        "metallic_both_non_regressing_and_one_improves_15_percent": metal_mae >= 0.0 and metal_fraction >= 0.0 and max(metal_mae, metal_fraction) >= 0.15,
        "normal_p95_regression_at_most_0_5_degrees": float(values["global_texel_normal_p95_regression_degrees"]["value"]) <= 0.5,
        "roughness_mae_regression_at_most_15_percent": float(values["global_texel_roughness_mae_regression"]["value"]) <= 0.15,
        "selection_hdr_mae_regression_at_most_10_percent": float(values["repair_selection_hdr_mae_regression"]["value"]) <= 0.10,
        "selection_ssim_drop_at_most_0_005": float(values["repair_selection_display_ssim_drop"]["value"]) <= 0.005,
        "analysis_two_runs_identical": bool(analysis_manifest["determinism"]["identical"]),
        "sampler_uv_passed": bool(analysis["sampler_uv_contract"]["passed"]),
        "resource_cost_exact": training["cost"] == {
            "parameters": 60,
            "weight_bytes_float32": 240,
            "macs_per_pixel": 48,
            "logical_raw_bytes": 20971520,
            "texture_samples": 2,
        },
    }
    c2_allowed = all(checks.values())
    near_gate = {
        "schema_version": 1,
        "candidate": "c1",
        "formal_holdout_accessed": False,
        "checks": checks,
        "c2_allowed": c2_allowed,
        "policy": "C2 requires every frozen BaseColor, auxiliary, render, determinism, sampler, and resource near-gate",
    }
    near_path = root / "c1/near_gate.json"
    near_path.write_text(deterministic_json(near_gate), encoding="utf-8", newline="\n")
    summary = {
        "schema_version": 1,
        "status": "complete_no_deployable_candidate",
        "formal_holdout_accessed": False,
        "baseline_retained": True,
        "phase0_passed": phase0["gpu_training_allowed"] is True,
        "c1": {
            "completed_step": training["training"]["completed_step"],
            "elapsed_seconds": training["training"]["elapsed_seconds"],
            "offline_pass": gates["offline_pass"],
            "passed_gate_count": sum(bool(item["passed"]) for item in values.values()),
            "gate_count": len(values),
            "c2_near_gate_passed": c2_allowed,
        },
        "c2": {"status": "not_started_by_frozen_near_gate", "initialization": "independent_rank3_phase0_artifacts_retained"},
        "ue": {"started": False, "reason": "no candidate passed every offline gate"},
        "conclusion": "The 5-logical-channel two-sample Hybrid removes BaseColor interpolation artifacts and improves metallic boundaries, but rank-2 auxiliary capacity fails normal, roughness, and render guards; C2 was forbidden by the predeclared near-gate.",
        "evidence": {
            "phase0": {"path": phase0_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(phase0_path)},
            "training": {"path": training_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(training_path)},
            "analysis": {"path": analysis_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(analysis_path)},
            "gates": {"path": gates_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(gates_path)},
            "near_gate": {"path": near_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(near_path)},
        },
        "deployment_exported": False,
    }
    summary_path = root / "final_summary.json"
    summary_path.write_text(deterministic_json(summary), encoding="utf-8", newline="\n")
    print(json.dumps({"status": summary["status"], "c1_passed_gates": summary["c1"]["passed_gate_count"], "c2_allowed": c2_allowed, "ue_started": False, "summary_sha256": sha256_file(summary_path)}, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    run()
