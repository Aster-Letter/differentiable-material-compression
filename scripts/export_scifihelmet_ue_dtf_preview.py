"""Export frozen C4/C5 decode-then-filter candidates for UE preview."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cg_frontier.compression.ue_decode_then_filter_export import (  # noqa: E402
    DEFAULT_OUTPUT_RELATIVE_PATH,
    PREVIEW_MANIFEST_NAME,
    export_dtf_preview_package,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (repo_root / DEFAULT_OUTPUT_RELATIVE_PATH).resolve()
    )
    manifest = export_dtf_preview_package(repo_root, output_dir)
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "manifest": str(output_dir / PREVIEW_MANIFEST_NAME),
                "candidates": sorted(manifest["candidates"]),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
