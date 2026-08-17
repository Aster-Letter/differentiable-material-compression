"""Build the self-contained upload ZIP for the C4 render-ablation campaign."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
NAME = "c4-render-ablation-20k-v1"
STAGING = ROOT / "transfers/outgoing" / NAME
ARCHIVE = ROOT / "transfers/outgoing" / f"{NAME}.zip"


def _tracked_python_sources() -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "src/cg_frontier"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        relative
        for line in result.stdout.splitlines()
        if (relative := line.strip().replace("\\", "/")).endswith(".py")
    ]


def _asset_files(directory: str) -> list[str]:
    root = ROOT / directory
    return [path.relative_to(ROOT).as_posix() for path in root.rglob("*") if path.is_file()]


def payload_files() -> list[str]:
    files = {
        "pyproject.toml",
        "requirements-cpu.txt",
        "requirements-gpu.txt",
        "configs/train/c4_render_ablation_20k_v1.yaml",
        "docs/SCOW_C4_RENDER_ABLATION_20K_GUIDE.md",
        "scripts/train_c4_render_ablation_20k.py",
        "scripts/build_c4_render_ablation_bundle.py",
        "scripts/render_c4_render_ablation_summary.py",
        "scripts/verify_c4_render_ablation_run.py",
        "scripts/verify_c4_render_ablation_bundle.py",
        "scripts/prepare_lantern_c4_asset.py",
        "scripts/remote_run_c4_render_ablation_20k.sh",
        "scripts/scow_c4_render_ablation_20k_preflight.slurm",
        "scripts/scow_c4_render_ablation_20k_job.slurm",
        "scripts/scow_submit_c4_render_ablation_20k.sh",
        "tests/test_c4_render_ablation.py",
        "src/cg_frontier/experiment_io.py",
        "src/cg_frontier/assets/gltf_merge.py",
        "src/cg_frontier/compression/render_ablation.py",
    }
    files.update(_tracked_python_sources())
    files.update(_asset_files("assets/source/simple_nonmetal_khronos/Corset"))
    files.update(_asset_files("assets/source/simple_nonmetal_khronos/BoomBox"))
    files.update(_asset_files("assets/source/c4_render_ablation_20k/Lantern"))
    return sorted(files)


def main() -> None:
    files = payload_files()
    missing = [relative for relative in files if not (ROOT / relative).is_file()]
    if missing:
        raise FileNotFoundError(f"bundle payload is missing: {missing[:5]}")
    forbidden = ("formal_holdout", "sealed", "ue_demo", ".private", ".git", "outputs")
    bad = [relative for relative in files if any(part in forbidden for part in Path(relative).parts)]
    if bad:
        raise ValueError(f"forbidden bundle payload: {bad[:5]}")
    if STAGING.exists() or ARCHIVE.exists():
        raise FileExistsError("refusing to overwrite an existing bundle or staging directory")
    for relative in files:
        target = STAGING / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    rows = []
    for relative in files:
        digest = hashlib.sha256((STAGING / relative).read_bytes()).hexdigest()
        rows.append(f"{digest}  {relative}")
    manifest = STAGING / "payload.MANIFEST.sha256"
    manifest.write_text("\n".join(rows) + "\n", encoding="utf-8", newline="\n")
    bundle = {
        "schema_version": 1,
        "name": NAME,
        "experiment": "c4_render_ablation_20k_v1",
        "payload_files": len(files),
        "payload_bytes": sum((STAGING / relative).stat().st_size for relative in files),
        "payload_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "assets": ["Corset", "Lantern", "BoomBox"],
        "arms": ["material_only", "material_render"],
        "steps_per_arm": 20000,
        "formal_holdout_access": "forbidden",
        "browser_automation": False,
    }
    (STAGING / "bundle.json").write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    subprocess.run(
        [sys.executable, "scripts/verify_c4_render_ablation_bundle.py", "--root", str(STAGING)],
        cwd=STAGING,
        check=True,
    )
    with zipfile.ZipFile(ARCHIVE, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in sorted(STAGING.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(STAGING).as_posix())
    digest = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    ARCHIVE.with_suffix(".zip.sha256").write_text(digest + "\n", encoding="ascii")
    print(json.dumps({
        "status": "bundle_built",
        "archive": str(ARCHIVE.relative_to(ROOT)),
        "archive_bytes": ARCHIVE.stat().st_size,
        "archive_sha256": digest,
        **bundle,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
