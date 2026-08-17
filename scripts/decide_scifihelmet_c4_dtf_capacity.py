"""Freeze C4-DTF-32 diagnostic eligibility from C4 and documented R0b quality."""

from __future__ import annotations

import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cg_frontier.compression.artifact_analysis import (  # noqa: E402
    deterministic_json,
    sha256_file,
)
from cg_frontier.compression.decode_then_filter_training import (  # noqa: E402
    build_capacity_diagnostic_eligibility,
)


OUTPUT_ROOT = ROOT / "outputs/compression/scifihelmet/c4_dtf_v1"
R0B_CONTROL = {
    "display_ssim": 0.999569873,
    "normal_mean_degrees": 0.929772,
    "subpixel_filter_divergence": 0.0,
}


def main() -> int:
    c4_path = OUTPUT_ROOT / "c4_dtf_16_selected/training_manifest.json"
    project_plan = ROOT / "docs/PROJECT_PLAN.md"
    if not c4_path.is_file() or not project_plan.is_file():
        raise FileNotFoundError("C4 manifest and PROJECT_PLAN.md are required")
    c4_manifest = json.loads(c4_path.read_text(encoding="utf-8"))
    record = build_capacity_diagnostic_eligibility(
        c4_manifest,
        r0b_control=R0B_CONTROL,
        c4_manifest_sha256=sha256_file(c4_path),
        r0b_document_sha256=sha256_file(project_plan),
    )
    output = OUTPUT_ROOT / "c4_dtf_32_diagnostic_eligibility.json"
    output.write_text(deterministic_json(record), encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                "eligible": record["eligible"],
                "next_candidate": record["next_candidate"],
                "output": output.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
