"""Build the incremental offline SCOW bundle for Lantern material-render to 160k."""

from __future__ import annotations

import hashlib
from pathlib import Path
import zipfile

from verify_c4_render_ablation_lantern_render_160k_bundle import PAYLOAD_FILES


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "transfers/outgoing/c4-render-ablation-lantern-render-160k-v1.zip"
BASELINE_FILES = (
    "configs/train/c4_render_ablation_20k_v1.yaml",
    "configs/train/c4_render_ablation_lantern_40k_v1.yaml",
    "scripts/train_c4_render_ablation_20k.py",
    "scripts/continue_c4_render_ablation_lantern_40k.py",
    "src/cg_frontier/compression/render_ablation.py",
    "src/cg_frontier/compression/render_ablation_continuation.py",
    "src/cg_frontier/render/gbuffer.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    for relative in (*PAYLOAD_FILES, *BASELINE_FILES):
        if not (ROOT / relative).is_file():
            raise FileNotFoundError(relative)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    payload_manifest = "".join(
        f"{_sha256(ROOT / relative)}  payload/{relative}\n" for relative in PAYLOAD_FILES
    )
    baseline_manifest = "".join(
        f"{_sha256(ROOT / relative)}  {relative}\n" for relative in BASELINE_FILES
    )
    with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for relative in PAYLOAD_FILES:
            archive.write(ROOT / relative, f"payload/{relative}")
        archive.writestr("PAYLOAD.MANIFEST.sha256", payload_manifest)
        archive.writestr("PATCH_BASELINE.sha256", baseline_manifest)
    digest = _sha256(OUTPUT)
    OUTPUT.with_suffix(OUTPUT.suffix + ".sha256").write_text(
        f"{digest}  {OUTPUT.name}\n", encoding="ascii", newline="\n"
    )
    print(f"bundle={OUTPUT}")
    print(f"sha256={digest}")


if __name__ == "__main__":
    main()
