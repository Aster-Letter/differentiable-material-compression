"""Freeze the paired SciFiHelmet C4-DTF activation decision."""

from __future__ import annotations

import argparse
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
    build_activation_selection_record,
)


OUTPUT_ROOT = ROOT / "outputs/compression/scifihelmet/c4_dtf_v1"
DEFAULT_RATIONALE = (
    "SiLU improved roughness, metallic, generic dark fraction, and dead units, "
    "but worsened base-color MAE, normal mean/P95, and halo while increasing the "
    "measured forward proxy; it therefore lacks a clear overall quality advantage."
)


def run(
    *,
    selected_activation: str,
    silu_clear_quality_advantage: bool,
    rationale: str,
) -> tuple[Path, dict[str, object]]:
    paths = {
        "relu": OUTPUT_ROOT
        / "c4_dtf_16_relu_precheck"
        / "training_manifest.json",
        "silu": OUTPUT_ROOT
        / "c4_dtf_16_silu_precheck"
        / "training_manifest.json",
    }
    if any(not path.is_file() for path in paths.values()):
        raise FileNotFoundError("both paired 10k training manifests are required")
    manifests = {
        name: json.loads(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
    }
    record = build_activation_selection_record(
        manifests["relu"],
        manifests["silu"],
        selected_activation=selected_activation,
        silu_clear_quality_advantage=silu_clear_quality_advantage,
        rationale=rationale,
    )
    record["source_manifests"] = {
        name: {
            "path": path.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(path),
        }
        for name, path in paths.items()
    }
    output = OUTPUT_ROOT / "activation_selection.json"
    output.write_text(deterministic_json(record), encoding="utf-8", newline="\n")
    return output, record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--selected-activation", choices=("relu", "silu"), default="relu"
    )
    parser.add_argument("--silu-clear-quality-advantage", action="store_true")
    parser.add_argument("--rationale", default=DEFAULT_RATIONALE)
    args = parser.parse_args()
    output, record = run(
        selected_activation=args.selected_activation,
        silu_clear_quality_advantage=args.silu_clear_quality_advantage,
        rationale=args.rationale,
    )
    print(
        json.dumps(
            {
                "output": output.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(output),
                "selected_activation": record["selected_activation"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
