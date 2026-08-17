"""Create deterministic SciFiHelmet Core-4 textures and their manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cg_frontier.assets.preprocess import (  # noqa: E402
    AssetValidationError,
    preprocess_scifihelmet,
)


DEFAULT_GLTF = (
    REPOSITORY_ROOT
    / "assets"
    / "source"
    / "glTF-Sample-Assets"
    / "Models"
    / "SciFiHelmet"
    / "glTF"
    / "SciFiHelmet.gltf"
)
DEFAULT_OUTPUT = REPOSITORY_ROOT / "assets" / "processed" / "SciFiHelmet" / "core4"
DEFAULT_MANIFEST = REPOSITORY_ROOT / "configs" / "assets" / "scifihelmet_core4.yaml"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and preprocess SciFiHelmet Core-4 PNG textures."
    )
    parser.add_argument("--gltf", type=Path, default=DEFAULT_GLTF)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    try:
        manifest = preprocess_scifihelmet(args.gltf, args.output_dir, args.manifest)
    except AssetValidationError as error:
        print(f"asset preprocessing failed: {error}", file=sys.stderr)
        return 2
    print(f"manifest: {args.manifest}")
    for semantic, output in manifest["outputs"].items():
        print(f"{semantic}: {args.output_dir / output['uri']} [{output['sha256']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
