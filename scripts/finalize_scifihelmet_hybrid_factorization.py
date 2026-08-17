"""Freeze D7 trigger, winner selection, and causal Hybrid conclusion."""

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


def _candidate(root: Path, name: str) -> dict[str, Any] | None:
    directory = root / name
    required = {key: directory / filename for key, filename in {
        "training": "training_manifest.json", "analysis": "analysis/interpolation_analysis.json",
        "analysis_manifest": "analysis/manifest.json", "gates": "gates.json",
    }.items()}
    if not all(path.is_file() for path in required.values()):
        return None
    values = {key: _load(path) for key, path in required.items()}
    if any(value.get("formal_holdout_accessed") is not False for key, value in values.items() if key != "analysis_manifest"):
        raise RuntimeError(f"{name} evidence does not preserve sealed holdout")
    values["paths"] = required
    return values


def _invalid_run(root: Path, name: str) -> dict[str, Any] | None:
    path = root / name / "invalid_run.json"
    if not path.is_file():
        return None
    value = _load(path)
    if value.get("formal_holdout_accessed") is not False:
        raise RuntimeError(f"{name} invalid-run evidence does not preserve sealed holdout")
    return {"report": value, "path": path}


def run(*, final: bool) -> dict[str, Any]:
    root = ROOT / "outputs/compression/scifihelmet/hybrid_factorization_v1"
    phase0_path = root / "phase0/phase0_summary.json"
    phase0 = _load(phase0_path)
    d6 = {name: _candidate(root, name) for name in ("d6_s", "d6_h", "d6_p")}
    invalid = {name: _invalid_run(root, name) for name in ("d6_s", "d6_h", "d6_p")}
    if any(d6[name] is None and invalid[name] is None for name in d6):
        raise RuntimeError("all three D6 candidates must be complete or carry explicit invalid-run evidence")
    d6p = d6["d6_p"]
    gates = d6p["gates"]["gates"] if d6p is not None else {}
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
    trigger_checks = {
        "three_d6_attempted": True,
        "three_d6_valid": all(value is not None for value in d6.values()),
        "no_d6_offline_pass": not any(bool(value["gates"]["offline_pass"]) for value in d6.values() if value is not None),
        "d6_p_valid": d6p is not None
        and int(d6p["training"]["training"]["completed_step"]) > int(d6p["training"]["training"]["warmup_steps"])
        and d6p["analysis_manifest"]["determinism"]["identical"] is True,
        "seven_basecolor_gates_pass": d6p is not None and all(bool(gates[name]["passed"]) for name in base_names),
        "normal_regression_at_most_0_5_degrees": d6p is not None
        and float(gates["global_texel_normal_p95_regression_degrees"]["value"]) <= 0.5,
        "scalar_or_render_strict_gate_failed": d6p is not None
        and any(not bool(gates[name]["passed"]) for name in scalar_or_render),
        "o7_direct_material_gates_pass": phase0["o7_direct_material_gates_pass"] is True,
    }
    d7_allowed = all(trigger_checks.values())
    trigger = {
        "schema_version": 1, "formal_holdout_accessed": False,
        "checks": trigger_checks, "d7_allowed": d7_allowed,
        "policy": "D7-P adds only the second scalar latent and requires a valid normal-near D6-P failure plus O7 material feasibility",
    }
    trigger_path = root / "d7_trigger.json"
    trigger_path.write_text(deterministic_json(trigger), encoding="utf-8", newline="\n")
    if not final:
        print(json.dumps({"d7_allowed": d7_allowed, "checks": trigger_checks, "trigger_sha256": sha256_file(trigger_path)}, ensure_ascii=False))
        return trigger

    d7 = _candidate(root, "d7_p")
    if d7_allowed and d7 is None:
        raise RuntimeError("D7-P is required by the frozen trigger but has not completed")
    candidates = {**d6, **({"d7_p": d7} if d7 is not None else {})}
    passed = {name: value for name, value in candidates.items() if value is not None and value["gates"]["offline_pass"] is True}
    winner = None
    if passed:
        winner = min(
            passed,
            key=lambda name: (
                int(passed[name]["training"]["cost"]["physical_ceiling_bytes"]),
                int(passed[name]["training"]["cost"]["macs_per_pixel"]),
                float(passed[name]["training"]["candidate_metrics"]["repair_selection_render"]["hdr_mae"]),
            ),
        )
    evidence: dict[str, Any] = {
        "phase0": {"path": phase0_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(phase0_path)},
        "d7_trigger": {"path": trigger_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(trigger_path)},
    }
    results: dict[str, Any] = {}
    for name, value in candidates.items():
        if value is None:
            if name in invalid and invalid[name] is not None:
                results[name] = {
                    "status": "invalid_run",
                    "reason": invalid[name]["report"]["reason"],
                    "quality_interpretation_allowed": False,
                }
                evidence[name] = {
                    "invalid_run": {
                        "path": invalid[name]["path"].relative_to(ROOT).as_posix(),
                        "sha256": sha256_file(invalid[name]["path"]),
                    }
                }
            else:
                results[name] = {"status": "not_started_by_trigger"}
            continue
        gate_values = value["gates"]["gates"]
        results[name] = {
            "status": "complete", "offline_pass": value["gates"]["offline_pass"],
            "passed_gate_count": sum(bool(item["passed"]) for item in gate_values.values()),
            "gate_count": len(gate_values), "cost": value["training"]["cost"],
        }
        evidence[name] = {
            key: {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path)}
            for key, path in value["paths"].items()
        }
    summary = {
        "schema_version": 1,
        "status": "offline_winner_pending_ue" if winner else "stopped_invalid_d6_runs" if any(value is not None for value in invalid.values()) else "complete_no_deployable_candidate",
        "formal_holdout_accessed": False,
        "baseline_retained": winner is None,
        "winner": winner,
        "candidates": results,
        "d7_allowed": d7_allowed,
        "ue": {"started": False, "reason": "winner pending UE" if winner else "no valid offline winner"},
        "conclusion": (
            "offline winner selected; UE validation required"
            if winner
            else "causal ladder inconclusive because one or more fixed D6 runs were invalid; frozen baseline retained"
            if any(value is not None for value in invalid.values())
            else "all valid learned candidates failed at least one offline gate; frozen baseline retained"
        ),
        "evidence": evidence,
        "deployment_exported": False,
    }
    summary_path = root / "final_summary.json"
    summary_path.write_text(deterministic_json(summary), encoding="utf-8", newline="\n")
    print(json.dumps({"status": summary["status"], "winner": winner, "d7_allowed": d7_allowed, "summary_sha256": sha256_file(summary_path)}, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    run(final=args.final)
