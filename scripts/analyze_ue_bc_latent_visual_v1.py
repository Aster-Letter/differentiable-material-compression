"""Quantify paired UE RGBA8/BC7 frames with an independent-run noise control."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = REPO_ROOT / "outputs/analysis/ue-bc-latent-feasibility-v1"
SCREENSHOT_ROOT = EVIDENCE_ROOT / "screenshots"
VISUAL_CONFIG = REPO_ROOT / "configs/eval/ue_bc_latent_visual_v1.json"
ROI = (850, 40, 1250, 600)
VARIANTS = [
    ("lantern_raw_q4_bc7", "raw PCA"),
    ("lantern_material_render_20k_bc7", "material-render 20k"),
    ("lantern_material_render_160k_bc7", "material-render 160k"),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def metrics(reference: np.ndarray, candidate: np.ndarray) -> dict:
    absolute = np.abs(reference - candidate)
    return {
        "mae": float(absolute.mean()),
        "rmse": float(np.sqrt(np.mean((reference - candidate) ** 2))),
        "p95_absolute_error": float(np.percentile(absolute, 95)),
        "max_absolute_error": float(absolute.max()),
        "changed_pixel_fraction_gt_1_lsb": float(
            (absolute.max(axis=2) > (1.0 / 255.0)).mean()
        ),
        "psnr_db": float(peak_signal_noise_ratio(reference, candidate, data_range=1.0)),
        "ssim": float(
            structural_similarity(reference, candidate, data_range=1.0, channel_axis=2)
        ),
    }


def verify_report(stem: str, config_hash: str) -> dict:
    path = EVIDENCE_ROOT / "visual_runs" / f"{stem}.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") not in {"complete", "complete_kept_open"}:
        raise RuntimeError(f"incomplete visual report: {stem}")
    if report.get("config_sha256") != config_hash:
        raise RuntimeError(f"visual config mismatch: {stem}")
    screenshot = SCREENSHOT_ROOT / f"{stem}.png"
    if sha256(screenshot) != report.get("screenshot_sha256"):
        raise RuntimeError(f"screenshot hash mismatch: {stem}")
    return report


config_hash = sha256(VISUAL_CONFIG)
image_by_stem: dict[str, np.ndarray] = {}
for variant_id, _label in VARIANTS:
    for kind in ("rgba8", "bc7"):
        stem = f"{variant_id}__{kind}"
        verify_report(stem, config_hash)
        image_by_stem[stem] = load_rgb(SCREENSHOT_ROOT / f"{stem}.png")
repeat_stem = "lantern_raw_q4_bc7__rgba8__repeat1"
verify_report(repeat_stem, config_hash)
image_by_stem[repeat_stem] = load_rgb(SCREENSHOT_ROOT / f"{repeat_stem}.png")

reference_shape = next(iter(image_by_stem.values())).shape
if reference_shape != (1080, 1920, 3) or any(
    image.shape != reference_shape for image in image_by_stem.values()
):
    raise RuntimeError(f"unexpected or inconsistent screenshot shape: {reference_shape}")

rows: list[dict] = []
pairs = [
    (
        "repeat_control",
        "lantern_raw_q4_bc7",
        image_by_stem["lantern_raw_q4_bc7__rgba8"],
        image_by_stem[repeat_stem],
    )
]
for variant_id, _label in VARIANTS:
    pairs.append(
        (
            "rgba8_vs_bc7",
            variant_id,
            image_by_stem[f"{variant_id}__rgba8"],
            image_by_stem[f"{variant_id}__bc7"],
        )
    )

for pair_type, variant_id, reference, candidate in pairs:
    for scope, ref_scope, cand_scope in (
        ("full_frame", reference, candidate),
        (
            "object_roi",
            reference[ROI[1] : ROI[3], ROI[0] : ROI[2]],
            candidate[ROI[1] : ROI[3], ROI[0] : ROI[2]],
        ),
    ):
        rows.append(
            {
                "pair_type": pair_type,
                "variant_id": variant_id,
                "scope": scope,
                "roi_x0": ROI[0] if scope == "object_roi" else "",
                "roi_y0": ROI[1] if scope == "object_roi" else "",
                "roi_x1": ROI[2] if scope == "object_roi" else "",
                "roi_y1": ROI[3] if scope == "object_roi" else "",
                **metrics(ref_scope, cand_scope),
                "visual_config_sha256": config_hash,
            }
        )

csv_path = EVIDENCE_ROOT / "visual_metrics.csv"
with csv_path.open("w", encoding="utf-8", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

roi_rows = {
    (row["pair_type"], row["variant_id"]): row
    for row in rows
    if row["scope"] == "object_roi"
}
control = roi_rows[("repeat_control", "lantern_raw_q4_bc7")]
codec_summary = []
for variant_id, label in VARIANTS:
    row = roi_rows[("rgba8_vs_bc7", variant_id)]
    codec_summary.append(
        {
            "variant_id": variant_id,
            "label": label,
            "roi_mae": row["mae"],
            "roi_ssim": row["ssim"],
            "roi_psnr_db": row["psnr_db"],
            "roi_p95_absolute_error": row["p95_absolute_error"],
            "mae_ratio_vs_repeat_control": row["mae"] / control["mae"],
            "mae_excess_over_repeat_control": max(row["mae"] - control["mae"], 0.0),
        }
    )

summary = {
    "schema_version": 1,
    "status": "visual_gate_passed_with_run_noise_caveat",
    "visual_config_sha256": config_hash,
    "resolution": [1920, 1080],
    "object_roi_xyxy": list(ROI),
    "repeat_control": {
        "variant_id": "lantern_raw_q4_bc7",
        "roi_mae": control["mae"],
        "roi_ssim": control["ssim"],
        "roi_psnr_db": control["psnr_db"],
        "purpose": "estimate independent UE editor-run image variation",
    },
    "codec_pairs": codec_summary,
    "manual_assessment": {
        "raw_pca_bc7": "user judged appearance to be well preserved",
        "other_pairs": (
            "agent montage review found no obvious 4x4 blocking, structural failure, or "
            "material-response change in the 20k and 160k pairs"
        ),
    },
    "visual_gate": {
        "decision": "pass",
        "scope": "BC7 deployment feasibility for all three Lantern latent variants",
        "caveat": (
            "The screenshots come from independent editor runs. The raw PCA pair exceeds "
            "the measured repeat noise, while the 20k and 160k pairs are near that baseline."
        ),
    },
    "interpretation_limit": (
        "Independent UE runs are not pixel-deterministic. Compare codec rows against the "
        "RGBA8 repeat control; do not attribute all image delta to BC7."
    ),
}
(EVIDENCE_ROOT / "visual_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

roi_width = ROI[2] - ROI[0]
roi_height = ROI[3] - ROI[1]
header_height = 30
montage = Image.new("RGB", (roi_width * 3, (roi_height + header_height) * 3), "white")
draw = ImageDraw.Draw(montage)
for row_index, (variant_id, label) in enumerate(VARIANTS):
    reference = image_by_stem[f"{variant_id}__rgba8"][ROI[1] : ROI[3], ROI[0] : ROI[2]]
    candidate = image_by_stem[f"{variant_id}__bc7"][ROI[1] : ROI[3], ROI[0] : ROI[2]]
    difference = np.clip(np.abs(reference - candidate) * 16.0, 0.0, 1.0)
    y = row_index * (roi_height + header_height)
    for column, (title, array) in enumerate(
        ((f"{label} | RGBA8", reference), ("BC7", candidate), ("abs diff x16", difference))
    ):
        x = column * roi_width
        draw.text((x + 8, y + 8), title, fill="black")
        tile = Image.fromarray(np.round(array * 255.0).astype(np.uint8), mode="RGB")
        montage.paste(tile, (x, y + header_height))
montage_path = EVIDENCE_ROOT / "visual_comparison_roi.png"
montage.save(montage_path)

manifest_paths = [
    VISUAL_CONFIG,
    csv_path,
    EVIDENCE_ROOT / "visual_summary.json",
    montage_path,
    *(SCREENSHOT_ROOT / f"{stem}.png" for stem in sorted(image_by_stem)),
]
manifest = {
    path.relative_to(REPO_ROOT).as_posix(): {
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }
    for path in manifest_paths
}
(EVIDENCE_ROOT / "visual_analysis_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
