"""Validate UE latent readbacks for the C4 render-ablation preview."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT = ROOT / "outputs/deployment/c4_render_ablation_20k_v1/ue_preview_job_37489"


def _pixels(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGBA"), dtype=np.int16)


def main() -> None:
    report_path = DEPLOYMENT / "ue_setup_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "complete_ready_for_manual_preview":
        raise ValueError("UE setup is not complete")
    rows = {}
    for name, endpoint in report["endpoints"].items():
        source = Path(endpoint["readback"]["source"])
        readback = Path(endpoint["readback"]["path"])
        if not endpoint["readback"]["export_success"] or not readback.is_file():
            raise ValueError(f"missing UE readback: {name}")
        left, right = _pixels(source), _pixels(readback)
        if left.shape != right.shape:
            raise ValueError(f"readback shape mismatch: {name}")
        difference = np.abs(left - right)
        row = {
            "shape": list(left.shape),
            "max_abs": int(difference.max()),
            "changed_values": int(np.count_nonzero(difference)),
            "exact_pixel_match": not bool(np.count_nonzero(difference)),
        }
        if not row["exact_pixel_match"]:
            raise ValueError(f"UE readback differs from source RGBA8: {name}")
        rows[name] = row
    validation = {
        "schema_version": 1,
        "status": "complete_exact",
        "endpoint_count": len(rows),
        "preview_map": report["preview_scene"]["map"],
        "rows": rows,
    }
    path = DEPLOYMENT / "ue_readback_validation.json"
    payload = (json.dumps(validation, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(payload)
    (DEPLOYMENT / "ue_readback_validation.json.sha256").write_text(
        hashlib.sha256(payload).hexdigest() + "\n", encoding="ascii"
    )
    print(json.dumps({"status": validation["status"], "endpoints": len(rows)}, sort_keys=True))


if __name__ == "__main__":
    main()
