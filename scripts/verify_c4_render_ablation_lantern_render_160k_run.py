"""Fail-closed verifier for Lantern material-render 40k-to-160k products."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cg_frontier.compression.render_ablation_long_continuation import (  # noqa: E402
    CHECKPOINT_STEPS,
    OBSERVATION_STEPS,
    load_long_continuation_checkpoint,
)
from cg_frontier.experiment_io import (  # noqa: E402
    build_file_inventory,
    is_finite_tree,
    resolve_within,
    sha256_file,
    write_json_with_sha256,
)


CONFIG = ROOT / "configs/train/c4_render_ablation_lantern_render_160k_v1.yaml"
HISTORICAL_OBSERVATIONS = (1000, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000)
HISTORICAL_CHECKPOINTS = (10000, 20000, 30000, 40000)


def _manifest(result_root: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "result_manifest_verified",
        "result_root": str(result_root.relative_to(ROOT)),
        **build_file_inventory(result_root),
    }


def _load_report(path: Path, *, steps: int, status: str) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("status") != status
        or report.get("experiment") != "c4_render_ablation_lantern_render_160k_v1"
        or report.get("arm") != "material_render"
        or int(report.get("steps", -1)) != steps
        or report.get("formal_holdout_accessed") is not False
        or report.get("audit_used_for_training") is not False
        or report.get("early_stopping") is not False
        or not is_finite_tree(report)
    ):
        raise ValueError(f"invalid 160k report: {path}")
    return report


def verify_preflight(run_root: Path, job_id: str) -> dict[str, object]:
    result_root = run_root / "Lantern"
    summary = json.loads((result_root / "continuation_summary.json").read_text(encoding="utf-8"))
    preparation = json.loads((result_root / "preparation.json").read_text(encoding="utf-8"))
    report = _load_report(
        result_root / "material_render/training_report.json",
        steps=40010,
        status="complete_bounded_preflight",
    )
    evidence = report.get("preflight_checkpoint")
    if (
        summary.get("status") != "complete_material_render_preflight"
        or int(summary.get("steps", -1)) != 40010
        or preparation.get("source_job_id") != "37581"
        or int(preparation.get("endpoint_step", -1)) != 40010
        or preparation.get("formal_holdout_accessed") is not False
        or not isinstance(evidence, dict)
        or evidence.get("reload_verified") is not True
    ):
        raise ValueError("Lantern 160k preflight evidence is incomplete")
    checkpoint = resolve_within(ROOT, ROOT / str(evidence["path"]))
    if sha256_file(checkpoint) != evidence["sha256"]:
        raise ValueError("Lantern 160k preflight checkpoint hash mismatch")
    result = {
        "schema_version": 1,
        "status": "preflight_verified",
        "experiment": "c4_render_ablation_lantern_render_160k_v1",
        "job_id": job_id,
        "source_job_id": "37581",
        "arm": "material_render",
        "steps": 10,
        "checkpoint_sha256": evidence["sha256"],
        "formal_holdout_accessed": False,
    }
    write_json_with_sha256(run_root / "preflight_verified.json", result)
    return result


def verify_formal(
    run_root: Path,
    result_root: Path,
    job_id: str,
    preflight_job_id: str,
) -> dict[str, object]:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config_hash = sha256_file(CONFIG)
    summary = json.loads((result_root / "continuation_summary.json").read_text(encoding="utf-8"))
    preparation = json.loads((result_root / "preparation.json").read_text(encoding="utf-8"))
    report = _load_report(
        result_root / "material_render/training_report.json",
        steps=160000,
        status="complete_160k_continuation",
    )
    if (
        summary.get("status") != "complete_material_render_160k"
        or int(summary.get("steps", -1)) != 160000
        or preparation.get("source_identity") != report.get("source_identity")
        or preparation.get("continuation_config_hash") != config_hash
        or int(preparation.get("endpoint_step", -1)) != 160000
        or summary.get("formal_holdout_accessed") is not False
        or summary.get("audit_used_for_training") is not False
        or report.get("continuation_config_hash") != config_hash
        or report.get("source_job_id") != "37581"
        or report.get("source_checkpoint", {}).get("sha256")
        != config["source"]["checkpoint_sha256"]
        or int(report["endpoint"]["audit_render"]["case_count"]) != 42
        or int(report["endpoint"]["train_render"]["case_count"]) != 144
    ):
        raise ValueError("Lantern 160k formal contract mismatch")
    expected_observations = HISTORICAL_OBSERVATIONS + OBSERVATION_STEPS
    expected_checkpoints = HISTORICAL_CHECKPOINTS + CHECKPOINT_STEPS
    if tuple(report.get("observation_steps", ())) != expected_observations:
        raise ValueError("Lantern 160k observation nodes are incomplete")
    if tuple(sorted(int(step) for step in report.get("checkpoints", {}))) != expected_checkpoints:
        raise ValueError("Lantern 160k checkpoint lineage is incomplete")
    checkpoint_hashes = {}
    for step in CHECKPOINT_STEPS:
        item = report["checkpoints"][str(step)]
        checkpoint = resolve_within(ROOT, ROOT / item["path"])
        if sha256_file(checkpoint) != item["sha256"]:
            raise ValueError(f"Lantern 160k checkpoint hash mismatch: {step}")
        payload = load_long_continuation_checkpoint(
            checkpoint,
            expected_source_identity=report["source_identity"],
            expected_continuation_config_hash=config_hash,
            expected_source_40k_checkpoint_sha256=config["source"]["checkpoint_sha256"],
        )
        if int(payload["step"]) != step:
            raise ValueError(f"Lantern 160k checkpoint step mismatch: {step}")
        checkpoint_hashes[str(step)] = item["sha256"]
    manifest = _manifest(result_root)
    manifest_path = run_root / "result_manifest.json"
    write_json_with_sha256(manifest_path, manifest)
    result = {
        "schema_version": 1,
        "status": "formal_run_verified",
        "experiment": "c4_render_ablation_lantern_render_160k_v1",
        "job_id": job_id,
        "preflight_job_id": preflight_job_id,
        "source_job_id": "37581",
        "arm": "material_render",
        "source_step": 40000,
        "endpoint_step": 160000,
        "result_root": str(result_root.relative_to(ROOT)),
        "source_identity": report["source_identity"],
        "checkpoint_hashes": checkpoint_hashes,
        "audit_cases": 42,
        "train_cases": 144,
        "result_manifest": {
            "path": str(manifest_path.relative_to(ROOT)),
            "sha256": sha256_file(manifest_path),
            "file_count": manifest["file_count"],
            "payload_bytes": manifest["payload_bytes"],
            "verified": True,
        },
        "formal_holdout_accessed": False,
    }
    write_json_with_sha256(run_root / "formal_verified.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "formal"), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--preflight-job-id")
    args = parser.parse_args()
    run_root = resolve_within(ROOT, args.run_root)
    if args.mode == "preflight":
        result = verify_preflight(run_root, args.job_id)
    else:
        if args.preflight_job_id is None:
            raise ValueError("formal verification requires --preflight-job-id")
        result_root = resolve_within(
            ROOT, args.result_root or (run_root / "Lantern")
        )
        result = verify_formal(run_root, result_root, args.job_id, args.preflight_job_id)
    print(json.dumps({"job_id": args.job_id, "status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
