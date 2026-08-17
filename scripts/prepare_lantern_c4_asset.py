"""Create the deterministic one-primitive Lantern Core-4 derivative."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cg_frontier.assets.gltf_merge import derive_single_primitive_c4_asset  # noqa: E402


UPSTREAM_COMMIT = "2bac6f8c57bf471df0d2a1e8a8ec023c7801dddf"
DEFAULT_SOURCE = (
    ROOT
    / "assets/source/c4_render_ablation_20k/Lantern/upstream/glTF/Lantern.gltf"
)
DEFAULT_OUTPUT = ROOT / "assets/source/c4_render_ablation_20k/Lantern/derived/glTF"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    arguments = parser.parse_args()
    source = arguments.source.resolve()
    output = arguments.output_directory.resolve()
    for path in (source, output):
        if not path.is_relative_to(ROOT):
            raise ValueError("Lantern preparation paths must remain inside the repository")
    result = derive_single_primitive_c4_asset(
        source, output, upstream_commit=UPSTREAM_COMMIT
    )
    print(json.dumps({
        "status": "complete_lantern_core4_derivation",
        "emissive_fraction": result["derivation"]["emissive"]["max_rgb_gt_0_05_fraction"],
        "triangles": result["derivation"]["triangles"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
