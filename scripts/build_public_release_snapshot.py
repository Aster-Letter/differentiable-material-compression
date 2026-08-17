"""Export a fresh-history, public-safe source snapshot.

The export starts from Git-tracked files and a small explicit allowlist of
untracked evidence tooling. It refuses course-delivery, agent state, local
assets, Unreal binaries, and known sealed-local-only paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path


DEFAULT_ROOT = Path(__file__).resolve().parents[1]
ROOT_FILES = {
    ".gitattributes",
    ".gitignore",
    "CONTEXT.md",
    "LICENSE",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "environment.yml",
    "pyproject.toml",
    "requirements-cpu.txt",
    "requirements-gpu.txt",
}
PUBLIC_PREFIXES = (
    ".github/",
    "assets/",
    "configs/",
    "docs/",
    "experiments/",
    "scripts/",
    "src/",
    "tests/",
    "third_party/",
    "ue_demo/",
)
EXCLUDED_PREFIXES = (
    ".agents/",
    ".private/",
    "docs-agent/",
    "docs/agents/",
    "docs/archive/",
    "docs/course-delivery/",
    "docs/obsidian/",
    "docs/obsidian-portfolio-demo/",
    "outputs/",
    "tmp/",
    "transfers/",
)
EXCLUDED_FILES = {
    "AGENTS.md",
    "configs/eval/scifihelmet_repair.yaml",
    "configs/train/scifihelmet_decoder_ablation.yaml",
    "configs/train/scifihelmet_render_quant.yaml",
    "docs/ARTIFACT_RETENTION_POLICY.md",
    "docs/C4_AFFINE_TEACHER_BRIEF_20260809.md",
    "docs/LEARNING_PLAN.md",
    "docs/PROJECT_PLAN.md",
    "scripts/build_course_report_materials.py",
    "scripts/build_course_submission_package.py",
    "scripts/cleanup_course_handoff_artifacts.py",
    "scripts/prepare_project_freeze_archives.py",
    "scripts/render_course_report_scifihelmet_capacity.py",
    "scripts/train_scifihelmet_decoder_ablation.py",
    "scripts/train_scifihelmet_render_quant.py",
    "tests/test_decoder_ablation.py",
    "tests/test_obsidian_dashboard.py",
    "tests/test_render_compression.py",
    "ue_demo/CGCompressionDemo/Content/Python/setup_course_report_video_lantern_map_v1.py",
}
UNTRACKED_ALLOWLIST = {
    "configs/eval/ue_bc_latent_feasibility_v1.json",
    "configs/eval/ue_bc_latent_visual_v1.json",
    "docs/RESULTS.md",
    "scripts/analyze_ue_bc_latent_visual_v1.py",
    "scripts/build_public_release_snapshot.py",
    "scripts/export_all_ue_gpu_events_v1.ps1",
    "scripts/export_ue_bc_latent_gpu_events_v1.ps1",
    "scripts/export_ue_insights_trace_v1.cmd",
    "scripts/finalize_ue_bc_latent_feasibility_v1.py",
    "scripts/finalize_ue_bc_latent_timing_v1.py",
    "scripts/finalize_ue_runtime_evidence_v1.py",
    "scripts/run_ue_bc_latent_feasibility_v1.ps1",
    "scripts/run_ue_bc_latent_timing_v1.ps1",
    "scripts/run_ue_gpu_timing_v1.ps1",
    "ue_demo/CGCompressionDemo/Content/Python/capture_ue_bc_latent_residency_v1.py",
    "ue_demo/CGCompressionDemo/Content/Python/capture_ue_bc_latent_visual_editor_v1.py",
    "ue_demo/CGCompressionDemo/Content/Python/capture_ue_runtime_residency_v1.py",
    "ue_demo/CGCompressionDemo/Content/Python/collect_ue_runtime_evidence_v1.py",
    "ue_demo/CGCompressionDemo/Content/Python/init_unreal.py",
    "ue_demo/CGCompressionDemo/Content/Python/inspect_ue_runtime_evidence_scenes_v1.py",
    "ue_demo/CGCompressionDemo/Content/Python/prepare_ue_runtime_timing_map_v1.py",
    "ue_demo/CGCompressionDemo/Content/Python/setup_ue_bc_latent_feasibility_v1.py",
    "ue_demo/CGCompressionDemo/Content/Python/setup_ue_bc_latent_visual_v1.py",
    "ue_demo/CGCompressionDemo/Content/Python/setup_ue_runtime_evidence_v1_maps.py",
    "ue_demo/CGCompressionDemo/Content/Python/validate_ue_runtime_evidence_v1_map.py",
}
UE_BINARY_SUFFIXES = {".uasset", ".umap"}
PROTECTED_PARTS = {
    "formal_holdout",
    "formal-holdout",
    "sealed",
    "sealed_local_only",
    "sealed-local-only",
}
MAX_PUBLIC_BYTES = 20 * 1024 * 1024


def git_paths(root: Path) -> set[str]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=root,
        check=True,
        capture_output=True,
    )
    return {
        item.decode("utf-8").replace("\\", "/")
        for item in result.stdout.split(b"\0")
        if item
    }


def public_path(relative: str) -> bool:
    normalized = relative.replace("\\", "/")
    lowered = normalized.lower()
    if set(Path(lowered).parts) & PROTECTED_PARTS:
        return False
    if normalized in EXCLUDED_FILES:
        return False
    if lowered.startswith(tuple(prefix.lower() for prefix in EXCLUDED_PREFIXES)):
        return False
    if Path(normalized).suffix.lower() in UE_BINARY_SUFFIXES:
        return False
    if normalized in ROOT_FILES:
        return True
    return normalized.startswith(PUBLIC_PREFIXES)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect(root: Path) -> list[str]:
    paths = git_paths(root) | UNTRACKED_ALLOWLIST
    selected: list[str] = []
    for relative in sorted(paths):
        source = root / relative
        if not public_path(relative) or not source.is_file() or source.is_symlink():
            continue
        if source.stat().st_size > MAX_PUBLIC_BYTES:
            raise ValueError(f"public file exceeds 20 MiB: {relative}")
        selected.append(relative)
    missing = sorted(path for path in UNTRACKED_ALLOWLIST if not (root / path).is_file())
    if missing:
        raise FileNotFoundError(f"public allowlist files are missing: {missing}")
    return selected


def export(root: Path, output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    paths = collect(root)
    for relative in paths:
        source = root / relative
        target = output_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    manifest_entries = []
    for relative in paths:
        target = output_dir / relative
        manifest_entries.append(
            {
                "path": relative,
                "bytes": target.stat().st_size,
                "sha256": sha256(target),
            }
        )
    (output_dir / "PUBLIC_SNAPSHOT_MANIFEST.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "release": "v0.1.0",
                "files": len(manifest_entries),
                "entries": manifest_entries,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    export(root, output_dir)
    print(output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
