"""CLI wrapper for the frozen R0b UE exporter."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cg_frontier.compression.ue_hybrid_direct_scalars_export import main


if __name__ == "__main__":
    raise SystemExit(main())
