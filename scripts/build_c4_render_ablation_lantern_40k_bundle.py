"""Build the incremental, offline SCOW bundle for Lantern 20k-to-40k continuation."""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import tempfile
import zipfile

from verify_c4_render_ablation_lantern_40k_bundle import PAYLOAD_FILES


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "transfers/outgoing/c4-render-ablation-lantern-40k-v1.zip"
BASELINE_FILES = (
    "configs/train/c4_render_ablation_20k_v1.yaml",
    "scripts/train_c4_render_ablation_20k.py",
    "src/cg_frontier/compression/render_ablation.py",
    "src/cg_frontier/render/gbuffer.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(path: Path, root: Path, files: list[Path]) -> None:
    path.write_text(
        "".join(f"{_sha256(item)}  {item.relative_to(root).as_posix()}\n" for item in files),
        encoding="ascii",
        newline="\n",
    )


def main() -> None:
    for relative in (*PAYLOAD_FILES, *BASELINE_FILES):
        if not (ROOT / relative).is_file():
            raise FileNotFoundError(relative)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="c4-lantern-40k-", dir=ROOT / "outputs") as raw:
        staging = Path(raw)
        copied = []
        for relative in PAYLOAD_FILES:
            destination = staging / "payload" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, destination)
            copied.append(destination)
        _write_manifest(staging / "LANTERN40K.MANIFEST.sha256", staging, copied)
        (staging / "PATCH_BASELINE.sha256").write_text(
            "".join(f"{_sha256(ROOT / item)}  {item}\n" for item in BASELINE_FILES),
            encoding="ascii",
            newline="\n",
        )
        with zipfile.ZipFile(OUTPUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(staging).as_posix())
    digest = _sha256(OUTPUT)
    OUTPUT.with_suffix(OUTPUT.suffix + ".sha256").write_text(
        f"{digest}  {OUTPUT.name}\n", encoding="ascii", newline="\n"
    )
    print(f"bundle={OUTPUT}")
    print(f"sha256={digest}")


if __name__ == "__main__":
    main()
