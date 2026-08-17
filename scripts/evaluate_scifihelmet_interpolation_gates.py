"""Evaluate frozen interpolation-repair gates from deterministic reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cg_frontier.compression.artifact_analysis import deterministic_json, sha256_file  # noqa: E402


def _reduction(baseline: float, candidate: float) -> float:
    if baseline <= 0.0:
        raise ValueError("relative reduction requires a positive baseline")
    return 1.0 - candidate / baseline


def _gate(value: float, threshold: float, comparison: str) -> dict[str, Any]:
    passed = value >= threshold if comparison == ">=" else value <= threshold
    return {"value": float(value), "comparison": comparison, "threshold": float(threshold), "passed": bool(passed)}


def evaluate(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    training: Mapping[str, Any],
) -> dict[str, Any]:
    b_random = baseline["probe_families"]["seeded_random"]["scopes"]
    c_random = candidate["probe_families"]["seeded_random"]["scopes"]
    b_fixed = baseline["probe_families"]["fixed_phase_grid"]
    c_fixed = candidate["probe_families"]["fixed_phase_grid"]

    def novel(scopes: Mapping[str, Any], name: str) -> float:
        return float(scopes[name]["dark_fractions"]["0.05"]["novel_dark_fraction"])

    d2_base, d2_candidate = novel(b_random, "D2_yellow_tube"), novel(c_random, "D2_yellow_tube")
    d1_base, d1_candidate = novel(b_random, "D1_metallic_boundary"), novel(c_random, "D1_metallic_boundary")
    d3_base, d3_candidate = novel(b_random, "D3_gray_panel"), novel(c_random, "D3_gray_panel")
    b_patch = float(b_fixed["connected_novel_dark_luminance_gt_0_05"]["D2_yellow_tube"]["max_patch_pixels"])
    c_patch = float(c_fixed["connected_novel_dark_luminance_gt_0_05"]["D2_yellow_tube"]["max_patch_pixels"])
    b_p99 = float(b_random["D2_yellow_tube"]["base_color_max_channel"]["runtime"]["p99_0"])
    c_p99 = float(c_random["D2_yellow_tube"]["base_color_max_channel"]["runtime"]["p99_0"])
    b_metal = b_random["D1_metallic_boundary"]["metallic_boundary"]
    c_metal = c_random["D1_metallic_boundary"]["metallic_boundary"]
    b_center = training["baseline"]["texel_center_material"]["global"]
    c_center = training["candidate_metrics"]["texel_center_material"]["global"]
    b_render = training["baseline"]["repair_selection_render"]
    c_render = training["candidate_metrics"]["repair_selection_render"]
    gates = {
        "D2_novel_dark_relative_reduction": _gate(_reduction(d2_base, d2_candidate), 0.80, ">="),
        "D2_novel_dark_absolute_fraction": _gate(d2_candidate, 0.015, "<="),
        "D2_max_connected_patch_relative_reduction": _gate(_reduction(b_patch, c_patch), 0.70, ">="),
        "D1_novel_dark_relative_reduction": _gate(_reduction(d1_base, d1_candidate), 0.60, ">="),
        "D3_novel_dark_relative_reduction": _gate(_reduction(d3_base, d3_candidate), 0.60, ">="),
        "D2_base_color_p99_relative_reduction": _gate(_reduction(b_p99, c_p99), 0.25, ">="),
        "metallic_boundary_mae_relative_reduction": _gate(
            _reduction(float(b_metal["mae"]), float(c_metal["mae"])), 0.30, ">="
        ),
        "metallic_boundary_above_0_1_relative_reduction": _gate(
            _reduction(float(b_metal["fraction_above_0_1"]), float(c_metal["fraction_above_0_1"])), 0.30, ">="
        ),
        "global_texel_base_p99_regression": _gate(
            float(c_center["base_color_max_channel"]["p99_0"]) / float(b_center["base_color_max_channel"]["p99_0"]) - 1.0,
            0.02,
            "<=",
        ),
        "global_texel_normal_p95_regression_degrees": _gate(
            float(c_center["normal_degrees"]["p95_0"]) - float(b_center["normal_degrees"]["p95_0"]),
            0.1,
            "<=",
        ),
        "global_texel_roughness_mae_regression": _gate(
            float(c_center["roughness"]["mean"]) / float(b_center["roughness"]["mean"]) - 1.0,
            0.05,
            "<=",
        ),
        "repair_selection_hdr_mae_regression": _gate(
            float(c_render["hdr_mae"]) / float(b_render["hdr_mae"]) - 1.0,
            0.02,
            "<=",
        ),
        "repair_selection_display_ssim_drop": _gate(
            float(b_render["display_ssim"]) - float(c_render["display_ssim"]), 0.001, "<="
        ),
    }
    return {
        "schema_version": 1,
        "candidate": training["candidate"],
        "formal_holdout_accessed": False,
        "gate_probe_family": "seeded_random",
        "connected_patch_probe_family": "fixed_phase_grid",
        "gates": gates,
        "offline_pass": all(bool(value["passed"]) for value in gates.values()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument(
        "--baseline-analysis",
        type=Path,
        default=ROOT / "outputs/compression/scifihelmet/interpolation_repair_v1/analysis/interpolation_analysis.json",
    )
    args = parser.parse_args()
    candidate_dir = args.candidate_dir.resolve()
    if not candidate_dir.is_relative_to(ROOT) or "formal_holdout" in candidate_dir.as_posix().lower():
        raise ValueError("candidate directory escapes the allowed repository scope")
    baseline_path = args.baseline_analysis.resolve()
    candidate_path = candidate_dir / "analysis/interpolation_analysis.json"
    training_path = candidate_dir / "training_manifest.json"
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    training = json.loads(training_path.read_text(encoding="utf-8"))
    if any(value.get("formal_holdout_accessed") is not False for value in (baseline, candidate, training)):
        raise RuntimeError("formal holdout access flag is not false")
    result = evaluate(baseline, candidate, training)
    result["inputs"] = {
        "baseline_analysis_sha256": sha256_file(baseline_path),
        "candidate_analysis_sha256": sha256_file(candidate_path),
        "training_manifest_sha256": sha256_file(training_path),
    }
    output_path = candidate_dir / "gates.json"
    output_path.write_text(deterministic_json(result), encoding="utf-8", newline="\n")
    print(json.dumps({"candidate": result["candidate"], "offline_pass": result["offline_pass"], "gates": result["gates"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
