"""Apply legacy 13 and additional filter-aware gates to one valid candidate."""

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
from evaluate_scifihelmet_interpolation_gates import evaluate as evaluate_legacy  # noqa: E402


ROOT_OUTPUT = ROOT / "outputs/compression/scifihelmet/filter_aware_decoder_v1"


def _reduction(baseline: float, candidate: float) -> float:
    if baseline == 0.0:
        return 0.0 if candidate == 0.0 else float("-inf")
    return 1.0 - candidate / baseline


def _gate(value: float, threshold: float, comparison: str) -> dict[str, Any]:
    passed = value >= threshold if comparison == ">=" else value <= threshold
    return {"value": float(value), "comparison": comparison, "threshold": float(threshold), "passed": bool(passed)}


def evaluate_additional(
    baseline_atlas: Mapping[str, Any],
    candidate_atlas: Mapping[str, Any],
    baseline_roi: Mapping[str, Any],
    candidate_roi: Mapping[str, Any],
    training: Mapping[str, Any],
) -> dict[str, Any]:
    b_random = baseline_atlas["probe_families"]["seeded_random"]["scopes"]
    c_random = candidate_atlas["probe_families"]["seeded_random"]["scopes"]
    b_fixed = baseline_atlas["probe_families"]["fixed_phase_grid"]
    c_fixed = candidate_atlas["probe_families"]["fixed_phase_grid"]

    def novel(scopes: Mapping[str, Any], name: str) -> float:
        return float(scopes[name]["dark_fractions"]["0.05"]["novel_dark_fraction"])

    def roi_metric(report: Mapping[str, Any], roi: str, *keys: str) -> float:
        value: Any = report["rois"][roi]["metrics"]
        for key in keys:
            value = value[key]
        return float(value)

    d2_b, d2_c = novel(b_random, "D2_yellow_tube"), novel(c_random, "D2_yellow_tube")
    d1_b, d1_c = novel(b_random, "D1_metallic_boundary"), novel(c_random, "D1_metallic_boundary")
    d3_b, d3_c = novel(b_random, "D3_gray_panel"), novel(c_random, "D3_gray_panel")
    patch_b = float(b_fixed["connected_novel_dark_luminance_gt_0_05"]["D2_yellow_tube"]["max_patch_pixels"])
    patch_c = float(c_fixed["connected_novel_dark_luminance_gt_0_05"]["D2_yellow_tube"]["max_patch_pixels"])
    rectangle_reductions = {
        roi: _reduction(
            roi_metric(baseline_roi, roi, "connected_dark_components", "rectangular_component_max_area"),
            roi_metric(candidate_roi, roi, "connected_dark_components", "rectangular_component_max_area"),
        )
        for roi in ("A_yellow_tube", "B_gray_panel")
    }
    gray_luma = _reduction(
        roi_metric(baseline_roi, "B_gray_panel", "boundary_band_luminance", "absolute_p99"),
        roi_metric(candidate_roi, "B_gray_panel", "boundary_band_luminance", "absolute_p99"),
    )
    halo_reductions = {}
    for sign in ("positive", "negative"):
        baseline_max = max(
            roi_metric(baseline_roi, roi, "boundary_band_luminance", f"{sign}_fraction")
            for roi in ("A_yellow_tube", "B_gray_panel")
        )
        candidate_max = max(
            roi_metric(candidate_roi, roi, "boundary_band_luminance", f"{sign}_fraction")
            for roi in ("A_yellow_tube", "B_gray_panel")
        )
        halo_reductions[sign] = _reduction(baseline_max, candidate_max)
    commute_b = max(
        roi_metric(baseline_roi, roi, "postprocess_commutativity_gap", "composite_max", "p99_0")
        for roi in ("A_yellow_tube", "B_gray_panel")
    )
    commute_c = max(
        roi_metric(candidate_roi, roi, "postprocess_commutativity_gap", "composite_max", "p99_0")
        for roi in ("A_yellow_tube", "B_gray_panel")
    )
    metal_b = b_random["D1_metallic_boundary"]["metallic_boundary"]
    metal_c = c_random["D1_metallic_boundary"]["metallic_boundary"]
    b_center = training["baseline"]["texel_center_material"]["global"]
    c_center = training["candidate_metrics"]["texel_center_material"]["global"]
    b_render = training["baseline"]["repair_selection_render"]
    c_render = training["candidate_metrics"]["repair_selection_render"]
    cost = training["cost"]
    gates = {
        "D2_novel_dark_relative_reduction": _gate(_reduction(d2_b, d2_c), 0.80, ">="),
        "D2_novel_dark_absolute_fraction": _gate(d2_c, 0.015, "<="),
        "D2_max_connected_patch_relative_reduction": _gate(_reduction(patch_b, patch_c), 0.70, ">="),
        "D1_novel_dark_relative_reduction": _gate(_reduction(d1_b, d1_c), 0.60, ">="),
        "D3_novel_dark_relative_reduction": _gate(_reduction(d3_b, d3_c), 0.60, ">="),
        "fixed_roi_A_rectangular_component_reduction": _gate(rectangle_reductions["A_yellow_tube"], 0.70, ">="),
        "fixed_roi_B_rectangular_component_reduction": _gate(rectangle_reductions["B_gray_panel"], 0.70, ">="),
        "gray_panel_boundary_luminance_p99_reduction": _gate(gray_luma, 0.50, ">="),
        "positive_halo_fraction_reduction": _gate(halo_reductions["positive"], 0.60, ">="),
        "negative_halo_fraction_reduction": _gate(halo_reductions["negative"], 0.60, ">="),
        "postprocess_commutativity_p99_reduction": _gate(_reduction(commute_b, commute_c), 0.80, ">="),
        "metallic_boundary_mae_reduction": _gate(
            _reduction(float(metal_b["mae"]), float(metal_c["mae"])), 0.30, ">="
        ),
        "metallic_boundary_above_0_1_reduction": _gate(
            _reduction(float(metal_b["fraction_above_0_1"]), float(metal_c["fraction_above_0_1"])), 0.30, ">="
        ),
        "global_texel_base_p99_regression": _gate(
            float(c_center["base_color_max_channel"]["p99_0"]) / float(b_center["base_color_max_channel"]["p99_0"]) - 1.0,
            0.02,
            "<=",
        ),
        "global_texel_normal_p95_regression_degrees": _gate(
            float(c_center["normal_degrees"]["p95_0"]) - float(b_center["normal_degrees"]["p95_0"]), 0.1, "<="
        ),
        "global_texel_roughness_mae_regression": _gate(
            float(c_center["roughness"]["mean"]) / float(b_center["roughness"]["mean"]) - 1.0, 0.05, "<="
        ),
        "repair_selection_hdr_mae_regression": _gate(
            float(c_render["hdr_mae"]) / float(b_render["hdr_mae"]) - 1.0, 0.02, "<="
        ),
        "repair_selection_display_ssim_drop": _gate(
            float(b_render["display_ssim"]) - float(c_render["display_ssim"]), 0.001, "<="
        ),
        "actual_resident_bytes": _gate(
            float(
                training["files"][
                    next(name for name in training["files"] if name.endswith("rgba_unorm8.png"))
                ]["raw_bytes"]
            ),
            16777216.0,
            "<=",
        ),
        "texture_samples": _gate(1.0, 1.0, "<="),
        "decoder_parameters": _gate(float(cost["parameters"]), 103.0, "<="),
        "decoder_weight_bytes": _gate(float(cost["weight_bytes_float32"]), 412.0, "<="),
        "decoder_macs": _gate(float(cost["macs_per_pixel"]), 88.0, "<="),
    }
    return {"gates": gates, "passed": all(item["passed"] for item in gates.values())}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=("f_relu", "f_softplus", "f_sigmoid"), required=True)
    args = parser.parse_args()
    candidate_root = ROOT_OUTPUT / args.candidate
    baseline_analysis_root = ROOT_OUTPUT / "phase3/baseline/analysis"
    paths = {
        "baseline_atlas": baseline_analysis_root / "interpolation_analysis.json",
        "baseline_roi": baseline_analysis_root / "fixed_roi_analysis.json",
        "candidate_atlas": candidate_root / "analysis/interpolation_analysis.json",
        "candidate_roi": candidate_root / "analysis/fixed_roi_analysis.json",
        "training": candidate_root / "training_manifest.json",
        "analysis_manifest": candidate_root / "analysis/analysis_manifest.json",
    }
    documents = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    if any(document.get("formal_holdout_accessed") is not False for document in documents.values()):
        raise RuntimeError("analysis/training holdout access flag is not false")
    if documents["training"].get("valid") is not True:
        raise RuntimeError("gate evaluation requires a valid training manifest")
    legacy = evaluate_legacy(
        documents["baseline_atlas"], documents["candidate_atlas"], documents["training"]
    )
    additional = evaluate_additional(
        documents["baseline_atlas"],
        documents["candidate_atlas"],
        documents["baseline_roi"],
        documents["candidate_roi"],
        documents["training"],
    )
    result = {
        "schema_version": 1,
        "candidate": args.candidate,
        "formal_holdout_accessed": False,
        "legacy_13": legacy,
        "filter_aware_additional": additional,
        "offline_pass": bool(legacy["offline_pass"] and additional["passed"]),
        "inputs": {name: {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path)} for name, path in paths.items()},
    }
    output_path = candidate_root / "gates.json"
    output_path.write_text(deterministic_json(result), encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                "candidate": args.candidate,
                "legacy_passed": sum(item["passed"] for item in legacy["gates"].values()),
                "legacy_total": len(legacy["gates"]),
                "additional_passed": sum(item["passed"] for item in additional["gates"].values()),
                "additional_total": len(additional["gates"]),
                "offline_pass": result["offline_pass"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
