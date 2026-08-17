"""Create the sealed-data-free final summary for the filter-aware decoder task."""

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


OUTPUT_ROOT = ROOT / "outputs/compression/scifihelmet/filter_aware_decoder_v1"
CANDIDATES = ("f_relu", "f_softplus", "f_sigmoid")


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _roi_metrics(report: dict[str, Any], roi: str) -> dict[str, float | int]:
    metrics = report["rois"][roi]["metrics"]
    return {
        "novel_dark_fraction": float(metrics["dark"]["novel_dark_fraction"]),
        "max_connected_dark_area": int(metrics["connected_dark_components"]["max_area"]),
        "rectangular_dark_component_max_area": int(
            metrics["connected_dark_components"]["rectangular_component_max_area"]
        ),
        "commutativity_gap_p99": float(
            metrics["postprocess_commutativity_gap"]["composite_max"]["p99_0"]
        ),
        "positive_halo_fraction": float(metrics["boundary_band_luminance"]["positive_fraction"]),
        "negative_halo_fraction": float(metrics["boundary_band_luminance"]["negative_fraction"]),
        "boundary_luminance_abs_p99": float(metrics["boundary_band_luminance"]["absolute_p99"]),
        "metallic_boundary_mae": float(metrics["metallic_boundary"]["mae"]),
    }


def main() -> int:
    phase0_path = OUTPUT_ROOT / "phase0/phase0_report.json"
    oracle_path = OUTPUT_ROOT / "oracle_assessment.json"
    baseline_roi_path = OUTPUT_ROOT / "phase3/baseline/analysis/fixed_roi_analysis.json"
    phase0 = _read(phase0_path)
    oracle = _read(oracle_path)
    baseline_roi = _read(baseline_roi_path)
    if any(item.get("formal_holdout_accessed") is not False for item in (phase0, oracle, baseline_roi)):
        raise RuntimeError("finalization requires sealed-data-free inputs")

    results: dict[str, Any] = {}
    for candidate in CANDIDATES:
        candidate_root = OUTPUT_ROOT / candidate
        training_path = candidate_root / "training_manifest.json"
        gates_path = candidate_root / "gates.json"
        roi_path = candidate_root / "analysis/fixed_roi_analysis.json"
        training = _read(training_path)
        gates = _read(gates_path)
        roi = _read(roi_path)
        legacy = gates["legacy_13"]["gates"]
        additional = gates["filter_aware_additional"]["gates"]
        results[candidate] = {
            "valid_training": bool(training["valid"]),
            "completed_step": int(training["training"]["completed_step"]),
            "elapsed_seconds": float(training["training"]["elapsed_seconds"]),
            "joint_evaluations": int(training["training"]["joint_evaluations"]),
            "checkpoint_complete": bool(
                training["training"]["checkpoint_contains_optimizer_rng_sampling_generator"]
            ),
            "legacy_gates": {"passed": sum(x["passed"] for x in legacy.values()), "total": len(legacy)},
            "additional_gates": {
                "passed": sum(x["passed"] for x in additional.values()),
                "total": len(additional),
            },
            "offline_pass": bool(gates["offline_pass"]),
            "fixed_roi": {
                "A_yellow_tube": _roi_metrics(roi, "A_yellow_tube"),
                "B_gray_panel": _roi_metrics(roi, "B_gray_panel"),
            },
            "inputs": {
                "training_manifest_sha256": sha256_file(training_path),
                "gate_report_sha256": sha256_file(gates_path),
                "fixed_roi_report_sha256": sha256_file(roi_path),
            },
        }

    winners = [name for name, result in results.items() if result["offline_pass"]]
    summary = {
        "schema_version": 1,
        "status": "complete_no_offline_winner",
        "formal_holdout_accessed": False,
        "baseline_frozen": True,
        "ue_started": False,
        "ue_reason": "zero candidates passed all offline gates",
        "winner": winners[0] if len(winners) == 1 else None,
        "training_extension_authorized": False,
        "training_extension_assessment": (
            "not recommended under the fixed objective: residual artifact and global/halo failures are "
            "large multi-objective tradeoffs rather than a near-converged single metric"
        ),
        "block_artifact_conclusion": (
            "connected black patches were strongly fragmented, but no candidate eliminated the artifacts "
            "while preserving halo, material, and render gates"
        ),
        "baseline_fixed_roi": {
            "A_yellow_tube": _roi_metrics(baseline_roi, "A_yellow_tube"),
            "B_gray_panel": _roi_metrics(baseline_roi, "B_gray_panel"),
        },
        "candidates": results,
        "decode_then_filter_oracle": oracle,
        "inputs": {
            "phase0_report_sha256": sha256_file(phase0_path),
            "baseline_fixed_roi_sha256": sha256_file(baseline_roi_path),
            "oracle_assessment_sha256": sha256_file(oracle_path),
        },
    }
    output = OUTPUT_ROOT / "final_summary.json"
    output.write_text(deterministic_json(summary), encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                "status": summary["status"],
                "winner": summary["winner"],
                "ue_started": summary["ue_started"],
                "sha256": sha256_file(output),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
