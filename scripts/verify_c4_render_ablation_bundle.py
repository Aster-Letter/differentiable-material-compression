"""Verify the self-contained C4 render-ablation campaign bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "src/cg_frontier/__init__.py",
    "src/cg_frontier/compression/__init__.py",
    "src/cg_frontier/experiment_io.py",
    "src/cg_frontier/compression/render_ablation.py",
    "configs/train/c4_render_ablation_20k_v1.yaml",
    "scripts/train_c4_render_ablation_20k.py",
    "scripts/render_c4_render_ablation_summary.py",
    "scripts/verify_c4_render_ablation_run.py",
    "scripts/remote_run_c4_render_ablation_20k.sh",
    "scripts/scow_c4_render_ablation_20k_preflight.slurm",
    "scripts/scow_c4_render_ablation_20k_job.slurm",
    "scripts/scow_submit_c4_render_ablation_20k.sh",
    "assets/source/c4_render_ablation_20k/Lantern/derived/derived_manifest.json",
    "assets/source/c4_render_ablation_20k/Lantern/derived/glTF/Lantern.gltf",
    "assets/source/simple_nonmetal_khronos/Corset/glTF/Corset.gltf",
    "assets/source/simple_nonmetal_khronos/BoomBox/glTF/BoomBox.gltf",
)
FORBIDDEN_PARTS = {
    ".git",
    ".private",
    ".scratch",
    ".envs",
    ".venv",
    "checkpoints",
    "formal_holdout",
    "logs",
    "outputs",
    "sealed",
    "ue_demo",
}


def _manifest_rows(path: Path) -> list[tuple[str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        digest, separator, relative = line.partition("  ")
        if separator != "  " or len(digest) != 64:
            raise ValueError("invalid payload manifest row")
        pure = PurePosixPath(relative)
        if pure.is_absolute() or ".." in pure.parts or any(part in FORBIDDEN_PARTS for part in pure.parts):
            raise ValueError(f"unsafe or forbidden manifest path: {relative}")
        rows.append((digest, relative))
    if not rows or len(rows) != len({relative for _, relative in rows}):
        raise ValueError("payload manifest is empty or duplicated")
    return rows


def verify(root: Path) -> dict[str, object]:
    root = root.resolve()
    manifest = root / "payload.MANIFEST.sha256"
    if not manifest.is_file():
        raise ValueError("payload.MANIFEST.sha256 is missing")
    rows = _manifest_rows(manifest)
    mismatches = []
    for expected, relative in rows:
        path = root / Path(*PurePosixPath(relative).parts)
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
        if actual != expected:
            mismatches.append(relative)
    if mismatches:
        raise ValueError(f"bundle payload hash mismatch: {mismatches[:5]}")
    actual_files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"payload.MANIFEST.sha256", "bundle.json"}
    }
    expected_files = {relative for _, relative in rows}
    if actual_files != expected_files:
        raise ValueError("bundle contains missing or extra payload files")
    for relative in REQUIRED:
        if relative not in expected_files:
            raise ValueError(f"required bundle file is missing: {relative}")
    config_text = (root / "configs/train/c4_render_ablation_20k_v1.yaml").read_text(encoding="utf-8")
    required_config_lines = (
        "formal_holdout_access: forbidden",
        "  steps: 20000",
        "arms: [material_only, material_render]",
        "  - id: Corset",
        "  - id: Lantern",
        "  - id: BoomBox",
    )
    if any(line not in config_text for line in required_config_lines):
        raise ValueError("bundle config differs from the frozen experiment")
    lantern = json.loads(
        (
            root
            / "assets/source/c4_render_ablation_20k/Lantern/derived/derived_manifest.json"
        ).read_text(encoding="utf-8")
    )
    if (
        lantern.get("derivation", {}).get("source_meshes") != 3
        or lantern.get("derivation", {}).get("output_meshes") != 1
        or lantern.get("derivation", {}).get("emissive", {}).get("policy")
        != "excluded_from_reference_pca_and_both_training_arms"
    ):
        raise ValueError("Lantern derivative contract is invalid")
    for relative in expected_files:
        if relative.endswith(".py"):
            source = (root / relative).read_text(encoding="utf-8")
            compile(source, relative, "exec")
    return {
        "status": "bundle_verified",
        "payload_files": len(rows),
        "payload_bytes": sum((root / Path(*PurePosixPath(relative).parts)).stat().st_size for _, relative in rows),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "formal_holdout_present": False,
        "forbidden_paths_present": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    result = verify(args.root)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
