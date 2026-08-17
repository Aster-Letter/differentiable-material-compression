"""Validate affine UE texture readbacks by decoded RGBA8 pixels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cg_frontier.compression.affine_ue_export import compare_rgba8_png_bytes


def run(preview_root: Path) -> dict[str, object]:
    root = preview_root.resolve()
    if not root.is_relative_to(ROOT):
        raise ValueError("preview root must stay inside the repository")
    manifest = json.loads((root / "preview_manifest.json").read_text(encoding="utf-8"))
    results = {}
    for name, candidate in manifest["candidates"].items():
        source = root / candidate["package_directory"] / "latent_rgba8.png"
        readback = root / "ue_evidence" / name / "latent_rgba8_readback.png"
        source_bytes = source.read_bytes()
        readback_bytes = readback.read_bytes()
        comparison = compare_rgba8_png_bytes(source_bytes, readback_bytes)
        comparison["file_sha256_match"] = (
            hashlib.sha256(source_bytes).digest()
            == hashlib.sha256(readback_bytes).digest()
        )
        results[name] = comparison
    exact = all(item["exact_pixel_match"] for item in results.values())
    report: dict[str, object] = {
        "schema_version": 1,
        "status": "complete_exact" if exact else "failed_pixel_mismatch",
        "formal_holdout_accessed": False,
        "preview_manifest_sha256": hashlib.sha256(
            (root / "preview_manifest.json").read_bytes()
        ).hexdigest(),
        "candidates": results,
    }
    payload = (
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    report_path = root / "ue_pixel_readback_report.json"
    if report_path.exists():
        if report_path.read_bytes() != payload:
            raise FileExistsError("refusing to overwrite different readback report")
    else:
        with report_path.open("xb") as stream:
            stream.write(payload)
    if not exact:
        raise RuntimeError("one or more UE affine texture readbacks differ")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview-root", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.preview_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
