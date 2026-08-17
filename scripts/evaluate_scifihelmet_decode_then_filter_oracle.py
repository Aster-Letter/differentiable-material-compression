"""Decide whether the structural four-texel oracle earns a runtime follow-up."""

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


REPORT = ROOT / "outputs/compression/scifihelmet/filter_aware_decoder_v1/phase0/phase0_report.json"
OUTPUT = ROOT / "outputs/compression/scifihelmet/filter_aware_decoder_v1/oracle_assessment.json"


def _reduction(baseline: float, candidate: float) -> float:
    if baseline == 0.0:
        return 0.0 if candidate == 0.0 else float("-inf")
    return 1.0 - candidate / baseline


def _gate(value: float, threshold: float, comparison: str) -> dict[str, Any]:
    passed = value >= threshold if comparison == ">=" else value <= threshold
    return {"value": value, "comparison": comparison, "threshold": threshold, "passed": bool(passed)}


def main() -> int:
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    if report.get("formal_holdout_accessed") is not False:
        raise RuntimeError("oracle assessment requires a sealed-data-free baseline report")
    rois = report["rois"]

    def metric(roi: str, *keys: str) -> float:
        value: Any = rois[roi]["metrics"]
        for key in keys:
            value = value[key]
        return float(value)

    d2_novel = metric("A_yellow_tube", "dark", "novel_dark_fraction")
    d2_filter = metric("A_yellow_tube", "dark", "filter_dark_fraction")
    rectangle = {
        roi: _reduction(
            metric(roi, "connected_dark_components", "rectangular_component_max_area"),
            metric(roi, "decode_then_filter_dark_components", "rectangular_component_max_area"),
        )
        for roi in ("A_yellow_tube", "B_gray_panel")
    }
    gray_luma = _reduction(
        metric("B_gray_panel", "boundary_band_luminance", "absolute_p99"),
        metric("B_gray_panel", "decode_then_filter", "boundary_band_luminance", "absolute_p99"),
    )
    halo = {}
    for sign in ("positive", "negative"):
        baseline_max = max(
            metric(roi, "boundary_band_luminance", f"{sign}_fraction")
            for roi in ("A_yellow_tube", "B_gray_panel")
        )
        oracle_max = max(
            metric(roi, "decode_then_filter", "boundary_band_luminance", f"{sign}_fraction")
            for roi in ("A_yellow_tube", "B_gray_panel")
        )
        halo[sign] = _reduction(baseline_max, oracle_max)
    metal = {
        key: _reduction(
            metric("B_gray_panel", "metallic_boundary", key),
            metric("B_gray_panel", "decode_then_filter", "metallic_boundary", key),
        )
        for key in ("mae", "fraction_above_0_1")
    }
    gates = {
        "D2_filter_dark_relative_reduction": _gate(_reduction(d2_novel, d2_filter), 0.80, ">="),
        "D2_filter_dark_absolute_fraction": _gate(d2_filter, 0.015, "<="),
        "fixed_roi_A_rectangular_component_reduction": _gate(rectangle["A_yellow_tube"], 0.70, ">="),
        "fixed_roi_B_rectangular_component_reduction": _gate(rectangle["B_gray_panel"], 0.70, ">="),
        "gray_panel_boundary_luminance_p99_reduction": _gate(gray_luma, 0.50, ">="),
        "positive_halo_fraction_reduction": _gate(halo["positive"], 0.60, ">="),
        "negative_halo_fraction_reduction": _gate(halo["negative"], 0.60, ">="),
        "gray_panel_metallic_boundary_mae_reduction": _gate(metal["mae"], 0.30, ">="),
        "gray_panel_metallic_boundary_above_0_1_reduction": _gate(
            metal["fraction_above_0_1"], 0.30, ">="
        ),
    }
    fixed_roi_pass = all(item["passed"] for item in gates.values())
    result = {
        "schema_version": 1,
        "status": (
            "fixed_roi_artifact_gates_passed_global_render_still_required"
            if fixed_roi_pass
            else "oracle_fails_required_artifact_gates"
        ),
        "formal_holdout_accessed": False,
        "oracle": "single RGBA8 four-texel decode-then-postprocess-then-filter structural upper bound",
        "fixed_roi_required_gates": gates,
        "fixed_roi_pass": fixed_roi_pass,
        "runtime_cost_followup_allowed": False,
        "runtime_cost_followup_reason": (
            "global/render gates remain unevaluated"
            if fixed_roi_pass
            else "the oracle already fails required fixed-ROI artifact gates"
        ),
        "input": {"path": REPORT.relative_to(ROOT).as_posix(), "sha256": sha256_file(REPORT)},
    }
    OUTPUT.write_text(deterministic_json(result), encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "passed": sum(item["passed"] for item in gates.values()),
                "total": len(gates),
                "runtime_cost_followup_allowed": False,
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
