"""Validate causal Hybrid Phase-0 evidence before CUDA training."""

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
    root = ROOT / "outputs/compression/scifihelmet/hybrid_factorization_v1/phase0"
    initialization_path = root / "initialization_report.json"
    initialization = _load(initialization_path)
    checks: dict[str, bool] = {
        "initialization_deterministic": initialization["determinism"]["identical"] is True,
        "initializer_optimizer_only": initialization["selection_or_validation_used_for_initialization"] is False,
        "initializer_holdout_sealed": initialization["formal_holdout_accessed"] is False,
    }
    candidates: dict[str, Any] = {}
    for candidate in ("d6_s", "d6_h", "d6_p", "d7_p", "o7_direct"):
        report_path = root / candidate / "analysis/interpolation_analysis.json"
        manifest_path = root / candidate / "analysis/manifest.json"
        report, manifest = _load(report_path), _load(manifest_path)
        scopes = report["probe_families"]["seeded_random"]["scopes"]
        fixed = report["probe_families"]["fixed_phase_grid"]
        candidate_checks = {
            "analysis_deterministic": manifest["determinism"]["identical"] is True,
            "formal_holdout_sealed": report["formal_holdout_accessed"] is False,
            "sampler_uv": report["sampler_uv_contract"]["passed"] is True,
            "direct_base_runtime_equals_filter": all(
                scope["base_color_max_channel"]["runtime"] == scope["base_color_max_channel"]["decode_then_filter"]
                for scope in scopes.values()
            ),
            "direct_base_novel_dark_zero": all(
                float(scope["dark_fractions"]["0.05"]["novel_dark_fraction"]) == 0.0
                for scope in scopes.values()
            ),
            "direct_base_connected_novel_dark_zero": all(
                int(value["max_patch_pixels"]) == 0
                for value in fixed["connected_novel_dark_luminance_gt_0_05"].values()
            ),
            "base_color_path": report["decoder"].get("base_color_path") == "direct_linear_UNORM8_bilinear_no_decoder_no_sigmoid",
            "filter_divergence_reported": all("filter_divergence" in scope for scope in scopes.values()),
            "cost_exact": report["decoder"]["parameters"] == initialization["candidates"][candidate]["cost"]["parameters"]
            and report["decoder"]["macs_per_pixel"] == initialization["candidates"][candidate]["cost"]["macs_per_pixel"],
            "gradient_diagnostic": (
                initialization["candidates"][candidate]["gradient_conflict"] is not None
                if candidate != "o7_direct" else initialization["candidates"][candidate]["gradient_conflict"] is None
            ),
        }
        checks.update({f"{candidate}_{name}": value for name, value in candidate_checks.items()})
        candidates[candidate] = {
            "checks": candidate_checks,
            "report": {"path": report_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(report_path)},
            "manifest": {"path": manifest_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(manifest_path)},
            "initial_texel_center": initialization["candidates"][candidate]["initial_texel_center"],
            "gradient_conflict": initialization["candidates"][candidate]["gradient_conflict"],
            "filter_divergence_global": scopes["global"]["filter_divergence"],
        }

    baseline_training = _load(ROOT / "outputs/compression/scifihelmet/hybrid_interpolation_v1/c1/training_manifest.json")
    baseline_analysis = _load(ROOT / "outputs/compression/scifihelmet/interpolation_repair_v1/analysis/interpolation_analysis.json")
    baseline_center = baseline_training["baseline"]["texel_center_material"]["global"]
    oracle_center = initialization["candidates"]["o7_direct"]["initial_texel_center"]
    baseline_metal = baseline_analysis["probe_families"]["seeded_random"]["scopes"]["D1_metallic_boundary"]["metallic_boundary"]
    oracle_metal = _load(root / "o7_direct/analysis/interpolation_analysis.json")["probe_families"]["seeded_random"]["scopes"]["D1_metallic_boundary"]["metallic_boundary"]
    oracle_gates = {
        "base_p99": oracle_center["base_p99"] / float(baseline_center["base_color_max_channel"]["p99_0"]) - 1.0 <= 0.02,
        "normal_p95": oracle_center["normal_p95"] - float(baseline_center["normal_degrees"]["p95_0"]) <= 0.1,
        "roughness_mae": oracle_center["roughness_mae"] / float(baseline_center["roughness"]["mean"]) - 1.0 <= 0.05,
        "metallic_boundary_mae": 1.0 - float(oracle_metal["mae"]) / float(baseline_metal["mae"]) >= 0.30,
        "metallic_boundary_fraction": 1.0 - float(oracle_metal["fraction_above_0_1"]) / float(baseline_metal["fraction_above_0_1"]) >= 0.30,
    }
    passed = all(checks.values())
    summary = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "formal_holdout_accessed": False,
        "gpu_training_allowed": passed,
        "initialization_report": {"path": initialization_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(initialization_path)},
        "checks": checks,
        "candidates": candidates,
        "o7_direct_material_gates": oracle_gates,
        "o7_direct_material_gates_pass": all(oracle_gates.values()),
    }
    output = root / "phase0_summary.json"
    output.write_text(deterministic_json(summary), encoding="utf-8", newline="\n")
    print(json.dumps({"status": summary["status"], "gpu_training_allowed": passed, "o7_material_pass": summary["o7_direct_material_gates_pass"], "summary_sha256": sha256_file(output)}, ensure_ascii=False))
    if not passed:
        raise RuntimeError("causal Hybrid Phase 0 failed; GPU/UE are forbidden")
    return summary


if __name__ == "__main__":
    run()
