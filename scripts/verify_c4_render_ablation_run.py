"""Fail-closed verifier for preflight and formal remote run products."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cg_frontier.experiment_io import (
    build_file_inventory,
    resolve_within,
    sha256_file,
    write_json_with_sha256,
)


ROOT = Path(__file__).resolve().parents[1]
ASSETS = ("Corset", "Lantern", "BoomBox")
ARMS = ("material_only", "material_render")


def _result_manifest(pair_root: Path, summary_root: Path) -> dict[str, object]:
    inventory = build_file_inventory({"pair": pair_root, "summary": summary_root})
    entries = inventory["files"]
    for entry in entries:
        source_root = pair_root if entry["root"] == "pair" else summary_root
        path = source_root / str(entry["path"])
        if (
            not path.is_file()
            or path.stat().st_size != entry["bytes"]
            or sha256_file(path) != entry["sha256"]
        ):
            raise ValueError("formal result manifest self-verification failed")
    return {
        "schema_version": 1,
        "status": "result_manifest_verified",
        "roots": {
            "pair": str(pair_root.relative_to(ROOT)),
            "summary": str(summary_root.relative_to(ROOT)),
        },
        **inventory,
    }


def _report(path: Path, *, steps: int) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    expected_status = "complete_20k" if steps == 20000 else "complete_bounded_preflight"
    if (
        value.get("status") != expected_status
        or value.get("steps") != steps
        or value.get("formal_holdout_accessed") is not False
        or value.get("audit_used_for_training") is not False
        or value.get("early_stopping") is not False
    ):
        raise ValueError(f"invalid training report: {path}")
    return value


def verify_preflight(root: Path, job_id: str) -> dict[str, object]:
    reports = {}
    for asset in ASSETS:
        pair = json.loads((root / asset / "paired_summary.json").read_text(encoding="utf-8"))
        if pair.get("paired_sampling_evidence", {}).get("identical") is not True:
            raise ValueError(f"preflight paired sampling mismatch: {asset}")
        reports[asset] = {}
        for arm in ARMS:
            report = _report(root / asset / arm / "training_report.json", steps=10)
            checkpoint = report.get("preflight_checkpoint")
            if not isinstance(checkpoint, dict) or checkpoint.get("reload_verified") is not True:
                raise ValueError(f"preflight checkpoint was not reloaded: {asset}/{arm}")
            reports[asset][arm] = checkpoint["sha256"]
    result = {
        "schema_version": 1,
        "status": "preflight_verified",
        "job_id": job_id,
        "assets": reports,
        "formal_holdout_accessed": False,
    }
    write_json_with_sha256(root / "preflight_verified.json", result)
    return result


def verify_formal(root: Path, pair_root: Path, summary_root: Path, asset: str, job_id: str, preflight_job_id: str) -> dict[str, object]:
    pair = json.loads((pair_root / "paired_summary.json").read_text(encoding="utf-8"))
    if pair.get("paired_sampling_evidence", {}).get("identical") is not True:
        raise ValueError("formal paired sampling evidence failed")
    identities, reports = [], {}
    for arm in ARMS:
        report = _report(pair_root / arm / "training_report.json", steps=20000)
        if tuple(report.get("observation_steps", ())) != (1000, 5000, 10000, 15000, 20000):
            raise ValueError("formal observation nodes are incomplete")
        if set(report.get("checkpoints", {})) != {"10000", "20000"}:
            raise ValueError("formal full checkpoints are incomplete")
        identities.append(report["identity"])
        reports[arm] = {
            "audit_cases": report["endpoint"]["audit_render"]["case_count"],
            "train_cases": report["endpoint"]["train_render"]["case_count"],
            "checkpoint_hashes": {key: value["sha256"] for key, value in report["checkpoints"].items()},
        }
        if reports[arm]["audit_cases"] != 42 or reports[arm]["train_cases"] != 144:
            raise ValueError("formal render case counts differ from the frozen rig")
        for step, item in report["checkpoints"].items():
            checkpoint_path = resolve_within(ROOT, ROOT / item["path"])
            if sha256_file(checkpoint_path) != item["sha256"]:
                raise ValueError("formal checkpoint SHA-256 mismatch")
            payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if payload.get("step") != int(step) or payload.get("identity") != report["identity"]:
                raise ValueError("formal checkpoint reload failed")
    if identities[0] != identities[1]:
        raise ValueError("paired arms do not share parent/config/rig/input identity")
    summary = json.loads((summary_root / "summary.json").read_text(encoding="utf-8"))
    if summary.get("status") != "complete_paired_20k_summary" or summary.get("winner_selected") is not False:
        raise ValueError("formal visual summary is incomplete")
    result_manifest = _result_manifest(pair_root, summary_root)
    manifest_path = root / "result_manifest.json"
    write_json_with_sha256(manifest_path, result_manifest)
    result = {
        "schema_version": 1,
        "status": "formal_run_verified",
        "job_id": job_id,
        "preflight_job_id": preflight_job_id,
        "asset": asset,
        "pair_root": str(pair_root.relative_to(ROOT)),
        "summary_root": str(summary_root.relative_to(ROOT)),
        "result_manifest": {
            "path": str(manifest_path.relative_to(ROOT)),
            "sha256": sha256_file(manifest_path),
            "file_count": result_manifest["file_count"],
            "payload_bytes": result_manifest["payload_bytes"],
            "verified": True,
        },
        "reports": reports,
        "identity": identities[0],
        "formal_holdout_accessed": False,
    }
    write_json_with_sha256(root / "formal_run_verified.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("preflight", "formal"), required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--pair-root", type=Path)
    parser.add_argument("--summary-root", type=Path)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--asset", choices=ASSETS)
    parser.add_argument("--preflight-job-id")
    args = parser.parse_args()
    root = args.run_root.resolve()
    if not root.is_relative_to(ROOT):
        raise ValueError("run root must remain inside the repository")
    if args.mode == "preflight":
        result = verify_preflight(root, args.job_id)
    else:
        if args.asset is None or args.preflight_job_id is None:
            raise ValueError("formal verification requires asset and preflight job id")
        pair_root = (args.pair_root or (root / args.asset)).resolve()
        summary_root = (args.summary_root or (root / f"{args.asset}-summary")).resolve()
        if not pair_root.is_relative_to(ROOT) or not summary_root.is_relative_to(ROOT):
            raise ValueError("formal product roots must remain inside the repository")
        result = verify_formal(root, pair_root, summary_root, args.asset, args.job_id, args.preflight_job_id)
    print(json.dumps({"status": result["status"], "job_id": result["job_id"]}, sort_keys=True))


if __name__ == "__main__":
    main()
