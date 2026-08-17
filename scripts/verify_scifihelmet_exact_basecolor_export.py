from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cg_frontier.compression.exact_basecolor_experiment import verify_runtime_export  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify a SciFiHelmet exact-BaseColor runtime export without source textures."
    )
    parser.add_argument("export_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_runtime_export(args.export_dir.resolve()), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
