"""Validate deterministic Hybrid Phase-0 evidence before any CUDA training."""

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
    root = ROOT / "outputs/compression/scifihelmet/hybrid_interpolation_v1/phase0"
    initialization_path = root / "initialization_report.json"
    initialization = _load(initialization_path)
    checks: dict[str, bool] = {
        "initialization_deterministic": bool(initialization["determinism"]["identical"]),
        "initializer_optimizer_only": initialization["selection_or_validation_used_for_initialization"] is False,
        "initializer_holdout_sealed": initialization["formal_holdout_accessed"] is False,
    }
    candidates: dict[str, Any] = {}
    for candidate in ("c1", "c2"):
        report_path = root / candidate / "analysis/interpolation_analysis.json"
        manifest_path = root / candidate / "analysis/manifest.json"
        report, manifest = _load(report_path), _load(manifest_path)
        scopes = report["probe_families"]["seeded_random"]["scopes"]
        fixed = report["probe_families"]["fixed_phase_grid"]
        direct_equal = all(
            scope["base_color_max_channel"]["runtime"]
            == scope["base_color_max_channel"]["decode_then_filter"]
            for scope in scopes.values()
        )
        novel_zero = all(
            float(scope["dark_fractions"]["0.05"]["novel_dark_fraction"]) == 0.0
            for scope in scopes.values()
        )
        connected_zero = all(
            int(value["max_patch_pixels"]) == 0
            for value in fixed["connected_novel_dark_luminance_gt_0_05"].values()
        )
        candidate_checks = {
            "analysis_deterministic": bool(manifest["determinism"]["identical"]),
            "formal_holdout_sealed": report["formal_holdout_accessed"] is False,
            "sampler_uv": bool(report["sampler_uv_contract"]["passed"]),
            "direct_base_runtime_equals_filter": direct_equal,
            "direct_base_novel_dark_zero": novel_zero,
            "direct_base_connected_novel_dark_zero": connected_zero,
            "base_color_path": report["decoder"].get("base_color_path")
            == "direct_linear_UNORM8_bilinear_no_decoder_no_sigmoid",
        }
        checks.update({f"{candidate}_{name}": value for name, value in candidate_checks.items()})
        candidates[candidate] = {
            "report": {"path": report_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(report_path)},
            "manifest": {"path": manifest_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(manifest_path)},
            "checks": candidate_checks,
            "initial_metrics": {
                "global_normal_p95_degrees": scopes["global"]["normal_degrees"]["p95_0"],
                "global_roughness_mae": scopes["global"]["roughness_absolute_error"]["mean"],
                "metallic_boundary_mae": scopes["D1_metallic_boundary"]["metallic_boundary"]["mae"],
                "metallic_boundary_fraction_above_0_1": scopes["D1_metallic_boundary"]["metallic_boundary"]["fraction_above_0_1"],
            },
        }
    passed = all(checks.values())
    summary = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "formal_holdout_accessed": False,
        "gpu_training_allowed": passed,
        "initialization_report": {
            "path": initialization_path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(initialization_path),
        },
        "checks": checks,
        "candidates": candidates,
    }
    output = root / "phase0_summary.json"
    output.write_text(deterministic_json(summary), encoding="utf-8", newline="\n")
    print(json.dumps({"status": summary["status"], "gpu_training_allowed": passed, "summary_sha256": sha256_file(output)}, ensure_ascii=False))
    if not passed:
        raise RuntimeError("Hybrid Phase 0 failed; GPU/UE are forbidden")
    return summary


if __name__ == "__main__":
    run()
