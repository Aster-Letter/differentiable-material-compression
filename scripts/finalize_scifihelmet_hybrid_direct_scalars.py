"""Finalize direct-scalar R0 evidence without exporting deployment assets."""

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
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("formal_holdout_accessed") is not False:
        raise RuntimeError(f"formal holdout seal missing from {path}")
    return value


def run() -> dict[str, Any]:
    root = ROOT / "outputs/compression/scifihelmet/hybrid_direct_scalars_v1"
    initialization_path = root / "initialization_report.json"
    render_path = root / "render_verification.json"
    initialization = _load(initialization_path)
    render = _load(render_path)
    candidates: dict[str, Any] = {}
    checks: dict[str, bool] = {
        "initialization_deterministic": initialization["determinism"]["identical"] is True,
        "render_deterministic": render["deterministic"] is True,
    }
    for candidate in ("r0a", "r0b"):
        candidate_dir = root / candidate
        manifest_path = candidate_dir / "training_manifest.json"
        analysis_path = candidate_dir / "analysis/interpolation_analysis.json"
        analysis_manifest_path = candidate_dir / "analysis/manifest.json"
        gates_path = candidate_dir / "gates.json"
        manifest = _load(manifest_path)
        analysis = _load(analysis_path)
        analysis_manifest = _load(analysis_manifest_path)
        gates = _load(gates_path)
        candidate_checks = {
            "offline_13_of_13": gates["offline_pass"] is True and sum(
                bool(value["passed"]) for value in gates["gates"].values()
            ) == 13,
            "analysis_deterministic": analysis_manifest["determinism"]["identical"] is True,
            "sampler_uv": analysis["sampler_uv_contract"]["passed"] is True,
            "base_direct": analysis["decoder"]["base_color_path"]
            == "direct_linear_UNORM8_bilinear_no_decoder_no_sigmoid",
            "scalars_direct": analysis["decoder"]["roughness_metallic_path"]
            == "direct_linear_UNORM8_bilinear_no_decoder_no_sigmoid",
            "cost": (
                analysis["decoder"]["parameters"],
                analysis["decoder"]["weight_bytes_float32"],
                analysis["decoder"]["macs_per_pixel"],
                manifest["cost"]["logical_raw_bytes"],
                manifest["cost"]["physical_ceiling_bytes"],
                manifest["cost"]["texture_samples"],
            )
            == (32, 128, 24, 29360128, 33554432, 2),
            "not_exported": manifest["deployment_exported"] is False,
        }
        checks.update({f"{candidate}_{name}": value for name, value in candidate_checks.items()})
        candidates[candidate] = {
            "deployment_eligible": manifest["deployment_eligible"],
            "initialization": manifest["initialization"],
            "cost": manifest["cost"],
            "cpu_oracle": manifest["cpu_oracle"],
            "repair_selection_render": manifest["candidate_metrics"]["repair_selection_render"],
            "gate_count": 13,
            "offline_pass": gates["offline_pass"],
            "checks": candidate_checks,
            "files": {
                "training_manifest": {"path": manifest_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(manifest_path)},
                "analysis": {"path": analysis_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(analysis_path)},
                "analysis_manifest": {"path": analysis_manifest_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(analysis_manifest_path)},
                "gates": {"path": gates_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(gates_path)},
            },
        }
    checks["r0a_oracle_only"] = candidates["r0a"]["deployment_eligible"] is False
    checks["r0b_independent_initialization"] = (
        candidates["r0b"]["deployment_eligible"] is True
        and candidates["r0b"]["initialization"]["checkpoint_inherited"] is False
        and candidates["r0b"]["initialization"]["source"] == "independent_phase0_rank2_normal"
    )
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise RuntimeError(f"direct-scalar finalization checks failed: {failed}")
    summary = {
        "schema_version": 1,
        "status": "offline_winner_selected_pending_user_confirmation_for_ue",
        "formal_holdout_accessed": False,
        "winner": "r0b",
        "r1_started": False,
        "r1_stop_reason": "r0b_passed_all_13_offline_gates",
        "ue_started": False,
        "deployment_exported": False,
        "baseline_deployment_unchanged": True,
        "checks": checks,
        "candidates": candidates,
        "inputs": {
            "initialization_report": {"path": initialization_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(initialization_path)},
            "render_verification": {"path": render_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(render_path)},
        },
    }
    output = root / "final_summary.json"
    output.write_text(deterministic_json(summary), encoding="utf-8", newline="\n")
    print(json.dumps({"status": summary["status"], "winner": "r0b", "sha256": sha256_file(output)}, ensure_ascii=False))
    return summary


if __name__ == "__main__":
    run()
