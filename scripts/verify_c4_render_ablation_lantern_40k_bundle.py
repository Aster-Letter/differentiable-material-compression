"""Verify the incremental Lantern 40k SCOW payload and its preserved 20k parent."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import py_compile

import yaml


PAYLOAD_FILES = (
    "configs/train/c4_render_ablation_lantern_40k_v1.yaml",
    "docs/SCOW_C4_RENDER_ABLATION_LANTERN_40K_GUIDE.md",
    "scripts/continue_c4_render_ablation_lantern_40k.py",
    "scripts/remote_run_c4_render_ablation_lantern_40k.sh",
    "scripts/scow_c4_render_ablation_lantern_40k_job.slurm",
    "scripts/scow_c4_render_ablation_lantern_40k_preflight.slurm",
    "scripts/scow_submit_c4_render_ablation_lantern_40k.sh",
    "scripts/verify_c4_render_ablation_lantern_40k_bundle.py",
    "scripts/verify_c4_render_ablation_lantern_40k_run.py",
    "src/cg_frontier/experiment_io.py",
    "src/cg_frontier/compression/render_ablation_continuation.py",
    "tests/test_c4_render_ablation_lantern_40k.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(path: Path) -> dict[str, str]:
    result = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        digest, relative = raw.split("  ", 1)
        result[relative] = digest
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-root", type=Path, default=Path.cwd())
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    args = parser.parse_args()
    bundle_root = args.bundle_root.resolve()
    campaign_root = args.campaign_root.resolve()
    payload_root = bundle_root / "payload"
    expected = {f"payload/{path}" for path in PAYLOAD_FILES}
    manifest = _manifest(bundle_root / "LANTERN40K.MANIFEST.sha256")
    if set(manifest) != expected:
        raise ValueError("incremental payload file set differs from the frozen contract")
    for relative, digest in manifest.items():
        path = bundle_root / relative
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"payload hash mismatch: {relative}")
    baseline = _manifest(bundle_root / "PATCH_BASELINE.sha256")
    for relative, digest in baseline.items():
        path = campaign_root / relative
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"campaign baseline mismatch: {relative}")
    config = yaml.safe_load(
        (payload_root / PAYLOAD_FILES[0]).read_text(encoding="utf-8")
    )
    if (
        config.get("experiment") != "c4_render_ablation_lantern_40k_v1"
        or config.get("formal_holdout_access") != "forbidden"
        or config.get("arms") != ["material_only", "material_render"]
        or config["training"].get("source_step") != 20000
        or config["training"].get("endpoint_step") != 40000
    ):
        raise ValueError("Lantern 40k config contract mismatch")
    source_root = (
        args.source_root.resolve()
        if args.source_root is not None
        else campaign_root / config["source"]["pair_root"]
    )
    if not source_root.resolve().is_relative_to(campaign_root.parent.resolve()):
        raise ValueError("source parent must remain under the project hierarchy")
    required_source = {
        "preparation.json": config["source"]["preparation_sha256"],
        "paired_summary.json": config["source"]["paired_summary_sha256"],
    }
    for arm, arm_spec in config["source"]["arms"].items():
        required_source[f"{arm}/training_report.json"] = arm_spec["report_sha256"]
        required_source[f"{arm}/checkpoints/step_20000/checkpoint.pt"] = arm_spec[
            "checkpoint_sha256"
        ]
        required_source[
            f"{arm}/checkpoints/step_20000/progress_snapshot.json"
        ] = arm_spec["progress_snapshot_sha256"]
    for relative, digest in required_source.items():
        path = source_root / relative
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"preserved 20k parent mismatch: {relative}")
    for relative in PAYLOAD_FILES:
        if relative.endswith(".py"):
            py_compile.compile(str(payload_root / relative), doraise=True)
    result = {
        "schema_version": 1,
        "status": "bundle_verified",
        "payload_files": len(PAYLOAD_FILES),
        "source_job_id": str(config["source"]["job_id"]),
        "preserved_parent_verified": True,
        "formal_holdout_present": False,
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
