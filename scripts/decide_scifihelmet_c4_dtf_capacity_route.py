"""Freeze the width-32 versus fresh C5 route after the 10k diagnostic."""

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
    build_capacity_route_decision,
)


OUTPUT_ROOT = ROOT / "outputs/compression/scifihelmet/c4_dtf_v1"


def main() -> int:
    paths = {
        "c4_dtf_16": OUTPUT_ROOT
        / "c4_dtf_16_relu_precheck/training_manifest.json",
        "c4_dtf_32": OUTPUT_ROOT / "c4_dtf_32_diagnostic/training_manifest.json",
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("both 10k capacity comparison manifests are required")
    manifests = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
    }
    record = build_capacity_route_decision(
        manifests["c4_dtf_16"],
        manifests["c4_dtf_32"],
        c4_16_manifest_sha256=sha256_file(paths["c4_dtf_16"]),
        c4_32_manifest_sha256=sha256_file(paths["c4_dtf_32"]),
    )
    output = OUTPUT_ROOT / "capacity_route.json"
    output.write_text(deterministic_json(record), encoding="utf-8", newline="\n")
    print(
        json.dumps(
            {
                "significant_capacity_benefit": record[
                    "significant_capacity_benefit"
                ],
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
