"""Apply black-block-first and exact-budget gates to one fresh candidate."""

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


ROOT_OUTPUT = ROOT / "outputs/compression/scifihelmet/deployment_parity_arc_v1"
CANDIDATES = ("dp_relu_fresh", "arc_relu_fresh")


def _reduction(baseline: float, candidate: float) -> float:
    if baseline == 0.0:
        return 0.0 if candidate == 0.0 else float("-inf")
    return 1.0 - candidate / baseline


def _gate(value: float, threshold: float, comparison: str) -> dict[str, Any]:
    passed = value >= threshold if comparison == ">=" else value <= threshold
    return {
        "value": float(value),
        "comparison": comparison,
        "threshold": float(threshold),
        "passed": bool(passed),
    }


def _novel(scopes: Mapping[str, Any], name: str) -> float:
    return float(scopes[name]["dark_fractions"]["0.05"]["novel_dark_fraction"])


def _roi(report: Mapping[str, Any], roi: str, *keys: str) -> float:
    value: Any = report["rois"][roi]["metrics"]
    for key in keys:
        value = value[key]
    return float(value)


def evaluate(
    baseline_atlas: Mapping[str, Any],
    candidate_atlas: Mapping[str, Any],
    baseline_roi: Mapping[str, Any],
    candidate_roi: Mapping[str, Any],
    quality: Mapping[str, Any],
    training: Mapping[str, Any],
) -> dict[str, Any]:
    b_random = baseline_atlas["probe_families"]["seeded_random"]["scopes"]
    c_random = candidate_atlas["probe_families"]["seeded_random"]["scopes"]
    b_fixed = baseline_atlas["probe_families"]["fixed_phase_grid"]
    c_fixed = candidate_atlas["probe_families"]["fixed_phase_grid"]
    d2_patch_b = float(
        b_fixed["connected_novel_dark_luminance_gt_0_05"]["D2_yellow_tube"]["max_patch_pixels"]
    )
    d2_patch_c = float(
        c_fixed["connected_novel_dark_luminance_gt_0_05"]["D2_yellow_tube"]["max_patch_pixels"]
    )
    rectangular = {
        roi: _roi(candidate_roi, roi, "connected_dark_components", "rectangular_component_max_area")
        for roi in ("A_yellow_tube", "B_gray_panel")
    }
    gray_reduction = _reduction(
        _roi(baseline_roi, "B_gray_panel", "boundary_band_luminance", "absolute_p99"),
        _roi(candidate_roi, "B_gray_panel", "boundary_band_luminance", "absolute_p99"),
    )
    halo_reduction: dict[str, float] = {}
    for sign in ("positive", "negative"):
        baseline_max = max(
            _roi(baseline_roi, roi, "boundary_band_luminance", f"{sign}_fraction")
            for roi in ("A_yellow_tube", "B_gray_panel")
        )
        candidate_max = max(
            _roi(candidate_roi, roi, "boundary_band_luminance", f"{sign}_fraction")
            for roi in ("A_yellow_tube", "B_gray_panel")
        )
        halo_reduction[sign] = _reduction(baseline_max, candidate_max)
    commute_b = max(
        _roi(baseline_roi, roi, "postprocess_commutativity_gap", "composite_max", "p99_0")
        for roi in ("A_yellow_tube", "B_gray_panel")
    )
    commute_c = max(
        _roi(candidate_roi, roi, "postprocess_commutativity_gap", "composite_max", "p99_0")
        for roi in ("A_yellow_tube", "B_gray_panel")
    )
    metal_b = b_random["D1_metallic_boundary"]["metallic_boundary"]
    metal_c = c_random["D1_metallic_boundary"]["metallic_boundary"]
    b_center = quality["texel_center_material"]["baseline"]["global"]
    c_center = quality["texel_center_material"]["candidate"]["global"]
    b_render = quality["repair_selection_render"]["baseline"]
    c_render = quality["repair_selection_render"]["candidate"]
    cost = quality["cost"]
    black_block_gates = {
        "fixed_roi_A_rectangular_dark_component_max_area": _gate(
            rectangular["A_yellow_tube"], 7.0, "<="
        ),
        "fixed_roi_B_rectangular_dark_component_max_area": _gate(
            rectangular["B_gray_panel"], 7.0, "<="
        ),
        "D2_novel_dark_absolute_fraction": _gate(
            _novel(c_random, "D2_yellow_tube"), 0.005, "<="
        ),
        "D2_max_connected_patch_relative_reduction": _gate(
            _reduction(d2_patch_b, d2_patch_c), 0.70, ">="
        ),
        "D1_novel_dark_relative_reduction": _gate(
            _reduction(
                _novel(b_random, "D1_metallic_boundary"),
                _novel(c_random, "D1_metallic_boundary"),
            ),
            0.60,
            ">=",
        ),
        "D3_novel_dark_relative_reduction": _gate(
            _reduction(
                _novel(b_random, "D3_gray_panel"),
                _novel(c_random, "D3_gray_panel"),
            ),
            0.60,
            ">=",
        ),
        "gray_panel_boundary_luminance_p99_reduction": _gate(
            gray_reduction, 0.50, ">="
        ),
        "positive_halo_fraction_reduction": _gate(
            halo_reduction["positive"], 0.60, ">="
        ),
        "negative_halo_fraction_reduction": _gate(
            halo_reduction["negative"], 0.60, ">="
        ),
        "postprocess_commutativity_p99_reduction": _gate(
            _reduction(commute_b, commute_c), 0.80, ">="
        ),
        "metallic_boundary_mae_reduction": _gate(
            _reduction(float(metal_b["mae"]), float(metal_c["mae"])), 0.30, ">="
        ),
        "metallic_boundary_above_0_1_reduction": _gate(
            _reduction(
                float(metal_b["fraction_above_0_1"]),
                float(metal_c["fraction_above_0_1"]),
            ),
            0.30,
            ">=",
        ),
    }
    quality_gates = {
        "global_texel_base_p99_regression": _gate(
            float(c_center["base_color_max_channel"]["p99_0"])
            / float(b_center["base_color_max_channel"]["p99_0"])
            - 1.0,
            0.02,
            "<=",
        ),
        "global_texel_normal_p95_regression_degrees": _gate(
            float(c_center["normal_degrees"]["p95_0"])
            - float(b_center["normal_degrees"]["p95_0"]),
            0.1,
            "<=",
        ),
        "global_texel_roughness_mae_regression": _gate(
            float(c_center["roughness"]["mean"])
            / float(b_center["roughness"]["mean"])
            - 1.0,
            0.05,
            "<=",
        ),
        "repair_selection_hdr_mae_regression": _gate(
            float(c_render["hdr_mae"]) / float(b_render["hdr_mae"]) - 1.0,
            0.02,
            "<=",
        ),
        "repair_selection_display_ssim_drop": _gate(
            float(b_render["display_ssim"]) - float(c_render["display_ssim"]),
            0.001,
            "<=",
        ),
        "actual_resident_bytes": _gate(
            float(
                training["files"][
                    next(name for name in training["files"] if name.endswith("rgba_unorm8.png"))
                ]["raw_bytes"]
            ),
            16_777_216.0,
            "<=",
        ),
        "texture_samples": _gate(float(cost["texture_samples"]), 1.0, "<="),
        "decoder_shape": {
            "value": cost["shape"],
            "comparison": "==",
            "threshold": "4->8->7",
            "passed": cost["shape"] == "4->8->7",
        },
        "decoder_parameters": _gate(float(cost["parameters"]), 103.0, "<="),
        "decoder_weight_bytes": _gate(float(cost["weight_bytes_float32"]), 412.0, "<="),
        "decoder_macs": _gate(float(cost["macs_per_pixel"]), 88.0, "<="),
    }
    legacy_training = {
        "candidate": training["candidate"],
        "baseline": {
            "texel_center_material": quality["texel_center_material"]["baseline"],
            "repair_selection_render": b_render,
        },
        "candidate_metrics": {
            "texel_center_material": quality["texel_center_material"]["candidate"],
            "repair_selection_render": c_render,
        },
    }
    legacy = evaluate_legacy(baseline_atlas, candidate_atlas, legacy_training)
    black_pass = all(bool(value["passed"]) for value in black_block_gates.values())
    quality_pass = all(bool(value["passed"]) for value in quality_gates.values())
    return {
        "schema_version": 1,
        "candidate": training["candidate"],
        "formal_holdout_accessed": False,
        "black_block_hard_gates": black_block_gates,
        "black_block_hard_pass": black_pass,
        "quality_and_structure_gates": quality_gates,
        "quality_and_structure_pass": quality_pass,
        "legacy_13": legacy,
        "offline_pass": bool(black_pass and quality_pass and legacy["offline_pass"]),
    }


def run(candidate: str) -> dict[str, Any]:
    if candidate not in CANDIDATES:
        raise ValueError(f"unsupported gate candidate: {candidate}")
    root = ROOT_OUTPUT / candidate
    baseline_root = ROOT_OUTPUT / "phase3/baseline/analysis"
    paths = {
        "baseline_atlas": baseline_root / "interpolation_analysis.json",
        "baseline_roi": baseline_root / "fixed_roi_analysis.json",
        "candidate_atlas": root / "analysis/interpolation_analysis.json",
        "candidate_roi": root / "analysis/fixed_roi_analysis.json",
        "analysis_manifest": root / "analysis/analysis_manifest.json",
        "quality": root / "quality_metrics.json",
        "training": root / "training_manifest.json",
    }
    documents = {
        name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()
    }
    if any(document.get("formal_holdout_accessed") is not False for document in documents.values()):
        raise RuntimeError("analysis/training formal holdout flag is not false")
    if documents["training"].get("valid") is not True:
        raise RuntimeError("gate evaluation requires a valid complete training run")
    result = evaluate(
        documents["baseline_atlas"],
        documents["candidate_atlas"],
        documents["baseline_roi"],
        documents["candidate_roi"],
        documents["quality"],
        documents["training"],
    )
    result["inputs"] = {
        name: {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path)}
        for name, path in paths.items()
    }
    (root / "gates.json").write_text(
        deterministic_json(result), encoding="utf-8", newline="\n"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=CANDIDATES, required=True)
    args = parser.parse_args()
    result = run(args.candidate)
    print(
        json.dumps(
            {
                "candidate": args.candidate,
                "black_block_hard_pass": result["black_block_hard_pass"],
                "quality_and_structure_pass": result["quality_and_structure_pass"],
                "legacy_passed": sum(
                    item["passed"] for item in result["legacy_13"]["gates"].values()
                ),
                "offline_pass": result["offline_pass"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
