"""Verify the incremental Lantern material-render 160k SCOW bundle."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PAYLOAD_FILES = (
    "configs/train/c4_render_ablation_lantern_render_160k_v1.yaml",
    "docs/SCOW_C4_RENDER_ABLATION_LANTERN_RENDER_160K_GUIDE.md",
    "scripts/continue_c4_render_ablation_lantern_render_160k.py",
    "scripts/remote_run_c4_render_ablation_lantern_render_160k.sh",
    "scripts/scow_c4_render_ablation_lantern_render_160k_job.slurm",
    "scripts/scow_c4_render_ablation_lantern_render_160k_preflight.slurm",
    "scripts/scow_submit_c4_render_ablation_lantern_render_160k.sh",
    "scripts/verify_c4_render_ablation_lantern_render_160k_bundle.py",
    "scripts/verify_c4_render_ablation_lantern_render_160k_run.py",
    "src/cg_frontier/experiment_io.py",
    "src/cg_frontier/compression/render_ablation_long_continuation.py",
    "tests/test_c4_render_ablation_lantern_render_160k.py",
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _manifest(data: bytes) -> dict[str, str]:
    result = {}
    for raw in io.StringIO(data.decode("ascii")):
        line = raw.strip()
        if not line:
            continue
        digest, path = line.split(None, 1)
        result[path.strip()] = digest
    return result


def verify(bundle: Path, installed_root: Path | None = None) -> dict[str, object]:
    with zipfile.ZipFile(bundle) as archive:
        names = archive.namelist()
        for name in names:
            pure = PurePosixPath(name)
            if pure.is_absolute() or ".." in pure.parts or "formal_holdout" in name.lower():
                raise ValueError(f"unsafe or forbidden bundle member: {name}")
        expected_members = {f"payload/{item}" for item in PAYLOAD_FILES}
        expected_members.update({"PAYLOAD.MANIFEST.sha256", "PATCH_BASELINE.sha256"})
        if set(names) != expected_members:
            raise ValueError("bundle member list differs from the frozen payload")
        payload_manifest = _manifest(archive.read("PAYLOAD.MANIFEST.sha256"))
        if set(payload_manifest) != {f"payload/{item}" for item in PAYLOAD_FILES}:
            raise ValueError("payload manifest file list mismatch")
        for name, expected in payload_manifest.items():
            if _sha256(archive.read(name)) != expected:
                raise ValueError(f"payload hash mismatch: {name}")
        if installed_root is not None:
            installed_root = installed_root.resolve()
            for relative in PAYLOAD_FILES:
                path = (installed_root / relative).resolve()
                if not path.is_relative_to(installed_root) or not path.is_file():
                    raise ValueError(f"installed payload missing: {relative}")
                if _sha256(path.read_bytes()) != payload_manifest[f"payload/{relative}"]:
                    raise ValueError(f"installed payload hash mismatch: {relative}")
    result = {
        "schema_version": 1,
        "status": "bundle_verified",
        "bundle": bundle.name,
        "bundle_sha256": _sha256(bundle.read_bytes()),
        "payload_files": len(PAYLOAD_FILES),
        "source_job_id": "37581",
        "arm": "material_render",
        "source_step": 40000,
        "endpoint_step": 160000,
        "formal_holdout_present": False,
        "installed_payload_verified": installed_root is not None,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bundle",
        type=Path,
        default=ROOT / "transfers/outgoing/c4-render-ablation-lantern-render-160k-v1.zip",
    )
    parser.add_argument("--installed-root", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.bundle.resolve(), args.installed_root), sort_keys=True))


if __name__ == "__main__":
    main()
