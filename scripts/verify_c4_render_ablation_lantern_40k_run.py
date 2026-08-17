"""Fail-closed verifier for Lantern C4 20k-to-40k continuation products."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from cg_frontier.experiment_io import (
    build_file_inventory,
    is_finite_tree,
    resolve_within,
    sha256_file,
    write_json_with_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/train/c4_render_ablation_lantern_40k_v1.yaml"
ARMS = ("material_only", "material_render")
OBSERVATIONS = (1000, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000)
CHECKPOINTS = (10000, 20000, 30000, 40000)


def _manifest(pair_root: Path) -> dict[str, object]:
    return {
        "schema_version": 1,
        "status": "result_manifest_verified",
        "pair_root": str(pair_root.relative_to(ROOT)),
        **build_file_inventory(pair_root),
    }


def _report(path: Path, *, steps: int, status: str) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("status") != status
        or int(report.get("steps", -1)) != steps
        or report.get("formal_holdout_accessed") is not False
        or report.get("audit_used_for_training") is not False
        or report.get("early_stopping") is not False
        or not is_finite_tree(report)
    ):
        raise ValueError(f"invalid continuation report: {path}")
    return report


def verify_preflight(run_root: Path, job_id: str) -> dict[str, object]:
    pair_root = run_root / "Lantern"
    pair = json.loads((pair_root / "paired_summary.json").read_text(encoding="utf-8"))
    if (
        pair.get("status") != "complete_paired_preflight"
        or int(pair.get("steps", -1)) != 20010
        or pair.get("paired_sampling_evidence", {}).get("identical") is not True
    ):
        raise ValueError("Lantern continuation preflight pair is invalid")
    checkpoints = {}
    for arm in ARMS:
        report = _report(
            pair_root / arm / "training_report.json",
            steps=20010,
            status="complete_bounded_preflight",
        )
        evidence = report.get("preflight_checkpoint")
        if not isinstance(evidence, dict) or evidence.get("reload_verified") is not True:
            raise ValueError(f"preflight checkpoint was not reloaded: {arm}")
        checkpoint = resolve_within(ROOT, ROOT / str(evidence["path"]))
        if sha256_file(checkpoint) != evidence["sha256"]:
            raise ValueError(f"preflight checkpoint hash mismatch: {arm}")
        checkpoints[arm] = evidence["sha256"]
    result = {
        "schema_version": 1,
        "status": "preflight_verified",
        "experiment": "c4_render_ablation_lantern_40k_v1",
        "job_id": job_id,
        "steps_per_arm": 10,
        "checkpoints": checkpoints,
        "formal_holdout_accessed": False,
    }
    write_json_with_sha256(run_root / "preflight_verified.json", result)
    return result


def verify_formal(
    run_root: Path,
    pair_root: Path,
    job_id: str,
    preflight_job_id: str,
) -> dict[str, object]:
    from cg_frontier.compression.render_ablation_continuation import (
        load_continuation_checkpoint,
    )

    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config_hash = sha256_file(CONFIG)
    pair = json.loads((pair_root / "paired_summary.json").read_text(encoding="utf-8"))
    if (
        pair.get("status") != "complete_paired_40k"
        or int(pair.get("steps", -1)) != 40000
        or pair.get("paired_sampling_evidence", {}).get("identical") is not True
        or pair.get("formal_holdout_accessed") is not False
        or pair.get("audit_used_for_training") is not False
    ):
        raise ValueError("Lantern continuation formal pair is invalid")
    identities = []
    reports = {}
    for arm in ARMS:
        report = _report(
            pair_root / arm / "training_report.json",
            steps=40000,
            status="complete_40k_continuation",
        )
        if tuple(report.get("observation_steps", ())) != OBSERVATIONS:
            raise ValueError(f"observation nodes are incomplete: {arm}")
        if tuple(sorted(int(step) for step in report.get("checkpoints", {}))) != CHECKPOINTS:
            raise ValueError(f"checkpoint nodes are incomplete: {arm}")
        if (
            int(report["endpoint"]["audit_render"]["case_count"]) != 42
            or int(report["endpoint"]["train_render"]["case_count"]) != 144
            or report.get("continuation_config_hash") != config_hash
        ):
            raise ValueError(f"rig or config contract mismatch: {arm}")
        identities.append(report["source_identity"])
        new_hashes = {}
        for step in (30000, 40000):
            item = report["checkpoints"][str(step)]
            checkpoint = resolve_within(ROOT, ROOT / item["path"])
            if sha256_file(checkpoint) != item["sha256"]:
                raise ValueError(f"continuation checkpoint hash mismatch: {arm}/{step}")
            payload = load_continuation_checkpoint(
                checkpoint,
                expected_arm=arm,
                expected_source_identity=report["source_identity"],
                expected_continuation_config_hash=config_hash,
                expected_source_checkpoint_sha256=config["source"]["arms"][arm][
                    "checkpoint_sha256"
                ],
            )
            if int(payload["step"]) != step:
                raise ValueError(f"continuation checkpoint step mismatch: {arm}/{step}")
            new_hashes[str(step)] = item["sha256"]
        reports[arm] = {
            "audit_cases": 42,
            "train_cases": 144,
            "checkpoint_hashes": new_hashes,
            "source_checkpoint_sha256": report["source_checkpoint"]["sha256"],
        }
    if identities[0] != identities[1]:
        raise ValueError("paired continuation arms do not share the same frozen source identity")
    manifest = _manifest(pair_root)
    manifest_path = run_root / "result_manifest.json"
    write_json_with_sha256(manifest_path, manifest)
    result = {
        "schema_version": 1,
        "status": "formal_run_verified",
        "experiment": "c4_render_ablation_lantern_40k_v1",
        "job_id": job_id,
        "preflight_job_id": preflight_job_id,
        "source_job_id": str(config["source"]["job_id"]),
        "pair_root": str(pair_root.relative_to(ROOT)),
        "reports": reports,
        "source_identity": identities[0],
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
    parser.add_argument("--pair-root", type=Path)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--preflight-job-id")
    args = parser.parse_args()
    run_root = resolve_within(ROOT, args.run_root)
    if args.mode == "preflight":
        result = verify_preflight(run_root, args.job_id)
    else:
        if args.preflight_job_id is None:
            raise ValueError("formal verification requires --preflight-job-id")
        pair_root = resolve_within(ROOT, args.pair_root or (run_root / "Lantern"))
        result = verify_formal(run_root, pair_root, args.job_id, args.preflight_job_id)
    print(json.dumps({"job_id": args.job_id, "status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
