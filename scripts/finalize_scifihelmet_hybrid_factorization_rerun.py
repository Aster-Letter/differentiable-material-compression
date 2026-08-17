"""Freeze the mixed-source D7 trigger and final rerun comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cg_frontier.compression.artifact_analysis import deterministic_json, sha256_file  # noqa: E402


RERUN_ROOT = ROOT / "outputs/compression/scifihelmet/hybrid_factorization_rerun_v1"
SOURCE_ROOT = ROOT / "outputs/compression/scifihelmet/hybrid_factorization_v1"


def _load(path: Path) -> dict[str, Any]:
    if "formal_holdout" in path.as_posix().lower():
        raise ValueError("sealed formal holdout path is forbidden")
    return json.loads(path.read_text(encoding="utf-8"))


def _candidate(directory: Path) -> dict[str, Any]:
    paths = {
        "training": directory / "training_manifest.json",
        "analysis": directory / "analysis/interpolation_analysis.json",
        "analysis_manifest": directory / "analysis/manifest.json",
        "gates": directory / "gates.json",
    }
    if not all(path.is_file() for path in paths.values()):
        raise RuntimeError(f"candidate evidence is incomplete: {directory}")
    values = {key: _load(path) for key, path in paths.items()}
    for key in ("training", "analysis", "gates"):
        if values[key].get("formal_holdout_accessed") is not False:
            raise RuntimeError(f"sealed holdout contract failed for {directory.name}:{key}")
    if values["analysis_manifest"]["determinism"]["identical"] is not True:
        raise RuntimeError(f"analysis is not deterministic for {directory.name}")
    training = values["training"]["training"]
    if int(training["completed_step"]) <= int(training["warmup_steps"]):
        raise RuntimeError(f"candidate did not reach joint training: {directory.name}")
    values["paths"] = paths
    return values


def _trigger() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    candidates = {
        "d6_s": _candidate(SOURCE_ROOT / "d6_s"),
        "d6_h": _candidate(RERUN_ROOT / "d6_h"),
        "d6_p": _candidate(RERUN_ROOT / "d6_p"),
    }
    phase0 = _load(SOURCE_ROOT / "phase0/phase0_summary.json")
    d6p_gates = candidates["d6_p"]["gates"]["gates"]
    base_names = (
        "D2_novel_dark_relative_reduction", "D2_novel_dark_absolute_fraction",
        "D2_max_connected_patch_relative_reduction", "D1_novel_dark_relative_reduction",
        "D3_novel_dark_relative_reduction", "D2_base_color_p99_relative_reduction",
        "global_texel_base_p99_regression",
    )
    scalar_or_render = (
        "global_texel_roughness_mae_regression", "metallic_boundary_mae_relative_reduction",
        "metallic_boundary_above_0_1_relative_reduction", "repair_selection_hdr_mae_regression",
        "repair_selection_display_ssim_drop",
    )
    checks = {
        "three_d6_valid": True,
        "no_d6_offline_pass": not any(bool(value["gates"]["offline_pass"]) for value in candidates.values()),
        "d6_p_valid": True,
        "seven_basecolor_gates_pass": all(bool(d6p_gates[name]["passed"]) for name in base_names),
        "normal_regression_at_most_0_5_degrees": float(
            d6p_gates["global_texel_normal_p95_regression_degrees"]["value"]
        ) <= 0.5,
        "scalar_or_render_strict_gate_failed": any(not bool(d6p_gates[name]["passed"]) for name in scalar_or_render),
        "o7_direct_material_gates_pass": phase0["o7_direct_material_gates_pass"] is True,
    }
    trigger = {
        "schema_version": 1,
        "formal_holdout_accessed": False,
        "checks": checks,
        "d7_allowed": all(checks.values()),
        "sources": {
            "d6_s": "hybrid_factorization_v1/d6_s",
            "d6_h": "hybrid_factorization_rerun_v1/d6_h",
            "d6_p": "hybrid_factorization_rerun_v1/d6_p",
            "phase0": "hybrid_factorization_v1/phase0",
        },
    }
    return trigger, candidates


def run(*, final: bool) -> dict[str, Any]:
    RERUN_ROOT.mkdir(parents=True, exist_ok=True)
    trigger, candidates = _trigger()
    trigger_path = RERUN_ROOT / "d7_trigger.json"
    trigger_path.write_text(deterministic_json(trigger), encoding="utf-8", newline="\n")
    if not final:
        print(json.dumps({"d7_allowed": trigger["d7_allowed"], "checks": trigger["checks"], "sha256": sha256_file(trigger_path)}, ensure_ascii=False))
        return trigger

    if trigger["d7_allowed"]:
        candidates["d7_p"] = _candidate(RERUN_ROOT / "d7_p")
    passed = {name: value for name, value in candidates.items() if value["gates"]["offline_pass"] is True}
    winner = min(
        passed,
        key=lambda name: (
            int(passed[name]["training"]["cost"]["physical_ceiling_bytes"]),
            int(passed[name]["training"]["cost"]["macs_per_pixel"]),
            float(passed[name]["training"]["candidate_metrics"]["repair_selection_render"]["hdr_mae"]),
        ),
    ) if passed else None
    results: dict[str, Any] = {}
    evidence: dict[str, Any] = {
        "d7_trigger": {"path": trigger_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(trigger_path)}
    }
    for name, value in candidates.items():
        gates = value["gates"]["gates"]
        key_gate_values = {
            key: float(gates[key]["value"])
            for key in (
                "global_texel_normal_p95_regression_degrees",
                "global_texel_roughness_mae_regression",
                "metallic_boundary_mae_relative_reduction",
                "metallic_boundary_above_0_1_relative_reduction",
                "repair_selection_hdr_mae_regression",
                "repair_selection_display_ssim_drop",
            )
        }
        results[name] = {
            "status": "complete",
            "offline_pass": value["gates"]["offline_pass"],
            "passed_gate_count": sum(bool(item["passed"]) for item in gates.values()),
            "gate_count": len(gates),
            "cost": value["training"]["cost"],
            "training": value["training"]["training"],
            "key_gate_values": key_gate_values,
        }
        evidence[name] = {
            key: {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path)}
            for key, path in value["paths"].items()
        }
    d6s_key = results["d6_s"]["key_gate_values"]
    d6h_key = results["d6_h"]["key_gate_values"]
    d6p_key = results["d6_p"]["key_gate_values"]
    d7_key = results.get("d7_p", {}).get("key_gate_values")
    causal_comparison = {
        "d6_h_vs_d6_s": {
            "normal_regression_reduction_fraction": (
                d6s_key["global_texel_normal_p95_regression_degrees"]
                - d6h_key["global_texel_normal_p95_regression_degrees"]
            ) / d6s_key["global_texel_normal_p95_regression_degrees"],
            "interpretation": "split heads changed the failed metrics only marginally; shared trunk is not the primary bottleneck",
        },
        "d6_p_vs_d6_h": {
            "normal_regression_reduction_fraction": (
                d6h_key["global_texel_normal_p95_regression_degrees"]
                - d6p_key["global_texel_normal_p95_regression_degrees"]
            ) / d6h_key["global_texel_normal_p95_regression_degrees"],
            "interpretation": "partitioning normal from scalar materially reduces normal error; latent task competition is real",
        },
    }
    if d7_key is not None:
        causal_comparison["d7_p_vs_d6_p"] = {
            "roughness_regression_reduction_fraction": (
                d6p_key["global_texel_roughness_mae_regression"]
                - d7_key["global_texel_roughness_mae_regression"]
            ) / d6p_key["global_texel_roughness_mae_regression"],
            "interpretation": "a second scalar latent fixes normal/metallic/render gates but roughness remains above its strict baseline guard",
        }
    summary = {
        "schema_version": 1,
        "status": "offline_winner_pending_ue" if winner else "complete_no_deployable_candidate",
        "formal_holdout_accessed": False,
        "winner": winner,
        "baseline_retained": winner is None,
        "d7_allowed": trigger["d7_allowed"],
        "candidates": results,
        "causal_comparison": causal_comparison,
        "ue": {"started": False, "reason": "winner pending UE" if winner else "no candidate passed every offline gate"},
        "evidence": evidence,
        "deployment_exported": False,
    }
    path = RERUN_ROOT / "final_summary.json"
    path.write_text(deterministic_json(summary), encoding="utf-8", newline="\n")
    print(json.dumps({"status": summary["status"], "winner": winner, "sha256": sha256_file(path)}, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    run(final=args.final)
