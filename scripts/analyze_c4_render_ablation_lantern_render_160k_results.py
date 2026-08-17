"""Analyze the verified Lantern material-render 40k-to-160k continuation."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any

from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFont


ROOT = Path(__file__).resolve().parents[1]
REPORT_40K = (
    ROOT
    / "outputs/analysis/c4-render-ablation-lantern-40k-job-37581/extracted/outputs/remote"
    / "c4-render-ablation-lantern-40k-v1/37581/Lantern/material_render/training_report.json"
)
REPORT_ONLY_40K = REPORT_40K.parent.parent / "material_only/training_report.json"
RUN_160K = (
    ROOT
    / "outputs/remote-archives/scow-c4-render-ablation-lantern-render-160k-v1-job-37824-archive-38202"
    / "extracted/outputs/remote/c4-render-ablation-lantern-render-160k-v1/37824"
)
LANTERN_160K = RUN_160K / "Lantern"
REPORT_160K = LANTERN_160K / "material_render/training_report.json"
PROGRESS_160K = LANTERN_160K / "material_render/progress.json"
OUTPUT = ROOT / "outputs/analysis/c4-render-ablation-lantern-render-160k-job-37824/report"
SUMMARY_20K = (
    ROOT
    / "outputs/analysis/c4-render-ablation-20k-v1-job-37489/extracted/outputs/remote"
    / "c4-render-ablation-20k-v1/37477/Lantern-summary/endpoint_technical.png"
)
FIXED_20K = (
    ROOT
    / "outputs/analysis/c4-render-ablation-20k-v1-job-37489/extracted/outputs/remote"
    / "c4-render-ablation-20k-v1/37477/Lantern/material_render"
    / "observations/step_20000/fixed_views.png"
)
FIXED_40K = REPORT_40K.parent / "observations/step_40000/fixed_views.png"
FIXED_160K = REPORT_160K.parent / "observations/step_160000/fixed_views.png"

OBSERVATIONS = (
    1000,
    5000,
    10000,
    15000,
    20000,
    25000,
    30000,
    35000,
    40000,
    60000,
    80000,
    100000,
    120000,
    140000,
    160000,
)
CHECKPOINTS = (10000, 20000, 30000, 40000, 80000, 120000, 160000)
NEW_OBSERVATIONS = (60000, 80000, 100000, 120000, 140000, 160000)
EXPECTED_CHECKPOINT_HASHES = {
    "80000": "8a094c1be2a64db013f5802ad40d16f183b752b90bdd93ef18a2828030579f05",
    "120000": "3f9eb9dc90af5afa9f29d01922d17e86c0cc6e0aa3fba15e5b82a6baf444067b",
    "160000": "07c2b5533d2013056699bd0de823a6695576f80980956b56a955e693b7638623",
}

METRICS = (
    ("audit_hdr_mae", "Audit mean HDR MAE", False),
    ("audit_display_ssim", "Audit display SSIM", True),
    ("audit_worst_hdr_mae", "Audit worst HDR MAE", False),
    ("train_hdr_mae", "Train mean HDR MAE", False),
    ("train_display_ssim", "Train display SSIM", True),
    ("base_color_linear_mae", "BaseColor linear MAE", False),
    ("seven_channel_mae", "Seven-channel MAE", False),
    ("normal_mean_degrees", "Normal mean degrees", False),
    ("roughness_mae", "Roughness MAE", False),
    ("metallic_mae", "Metallic MAE", False),
    ("oklab_delta_e_mean", "Oklab delta-E mean", False),
    ("opponent_error", "Opponent error", False),
    ("chroma_magnitude_retention", "Chroma magnitude retention", True),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def finite(value: Any) -> bool:
    if value is None or isinstance(value, (str, bool)):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, dict):
        return all(finite(item) for item in value.values())
    if isinstance(value, list):
        return all(finite(item) for item in value)
    return False


def sidecar(path: Path) -> str:
    return path.read_text(encoding="ascii").strip().split()[0]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def endpoint_row(label: str, step: int, endpoint: dict[str, Any]) -> dict[str, Any]:
    audit = endpoint["audit_render"]
    train = endpoint["train_render"]
    return {
        "endpoint": label,
        "step": step,
        **endpoint["material"],
        "audit_hdr_mae": audit["mean"]["masked_linear_hdr_mae"],
        "audit_display_ssim": audit["mean"]["display_ssim"],
        "audit_worst_hdr_mae": audit["worst"]["masked_linear_hdr_mae"],
        "train_hdr_mae": train["mean"]["masked_linear_hdr_mae"],
        "train_display_ssim": train["mean"]["display_ssim"],
    }


def relative_delta(old: float, new: float) -> float:
    return (new / old - 1.0) * 100.0


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def validate() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    report40 = load_json(REPORT_40K)
    report160 = load_json(REPORT_160K)
    progress = load_json(PROGRESS_160K)
    marker = load_json(RUN_160K / "formal_verified.json")
    manifest = load_json(RUN_160K / "result_manifest.json")

    for path in (REPORT_40K, REPORT_160K, PROGRESS_160K, RUN_160K / "formal_verified.json", RUN_160K / "result_manifest.json"):
        sidecar_path = path.with_suffix(path.suffix + ".sha256")
        if sidecar_path.is_file() and sidecar(sidecar_path) != sha256(path):
            raise ValueError(f"sidecar mismatch: {path}")

    if (
        report40.get("status") != "complete_40k_continuation"
        or report40.get("steps") != 40000
        or report160.get("status") != "complete_160k_continuation"
        or report160.get("steps") != 160000
        or tuple(report160.get("observation_steps", ())) != OBSERVATIONS
        or tuple(sorted(int(step) for step in report160.get("checkpoints", {}))) != CHECKPOINTS
        or report160.get("source_job_id") != "37581"
        or report160.get("source_endpoint_40k") != report40.get("endpoint")
        or report160.get("raw_parent") != report40.get("raw_parent")
        or report160.get("source_identity") != report40.get("source_identity")
        or report160.get("formal_holdout_accessed") is not False
        or report160.get("audit_used_for_training") is not False
        or report160.get("early_stopping") is not False
        or report160["endpoint"]["audit_render"]["case_count"] != 42
        or report160["endpoint"]["train_render"]["case_count"] != 144
        or marker.get("status") != "formal_run_verified"
        or marker.get("job_id") != "37824"
        or marker.get("preflight_job_id") != "37821"
        or marker.get("checkpoint_hashes") != EXPECTED_CHECKPOINT_HASHES
        or marker.get("formal_holdout_accessed") is not False
        or manifest.get("file_count") != 53
        or manifest.get("payload_bytes") != 641388379
        or not finite(report160)
        or not finite(progress)
    ):
        raise ValueError("40k-to-160k formal contract mismatch")

    observations = progress.get("observations", {})
    if tuple(sorted(int(step) for step in observations)) != OBSERVATIONS:
        raise ValueError("material observation sequence mismatch")
    if observations["40000"] != report40["endpoint"]["material"]:
        raise ValueError("40k material endpoint mismatch")
    if observations["160000"] != report160["endpoint"]["material"]:
        raise ValueError("160k material endpoint mismatch")
    for step in NEW_OBSERVATIONS:
        observation = load_json(
            REPORT_160K.parent / f"observations/step_{step}/material_metrics.json"
        )
        if observation != observations[str(step)]:
            raise ValueError(f"observation material mismatch: {step}")

    cases40 = {
        (case["camera_index"], case["light_index"]): case
        for case in report40["endpoint"]["audit_render"]["cases"]
    }
    cases160 = {
        (case["camera_index"], case["light_index"]): case
        for case in report160["endpoint"]["audit_render"]["cases"]
    }
    if cases40.keys() != cases160.keys() or len(cases40) != 42:
        raise ValueError("audit case pairing mismatch")

    validation = {
        "schema_version": 1,
        "status": "complete_exact",
        "formal_job_id": "37824",
        "preflight_job_id": "37821",
        "source_job_id": "37581",
        "archive_job_id": "38202",
        "manifest_files": 53,
        "manifest_payload_bytes": 641388379,
        "observation_steps": list(OBSERVATIONS),
        "checkpoint_steps": list(CHECKPOINTS),
        "audit_cases": 42,
        "train_cases": 144,
        "source_endpoint_40k_exact": True,
        "raw_parent_exact": True,
        "source_identity_exact": True,
        "checkpoint_reload_previously_verified": True,
        "formal_holdout_accessed": False,
        "audit_used_for_training": False,
        "single_seed": True,
    }
    return validation, report40, report160, progress


def build_tables(
    report40: dict[str, Any], report160: dict[str, Any], progress: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    only40 = load_json(REPORT_ONLY_40K)
    endpoints = [
        endpoint_row("raw_q4", 0, report160["raw_parent"]),
        endpoint_row("material_only_40k", 40000, only40["endpoint"]),
        endpoint_row("material_render_40k", 40000, report40["endpoint"]),
        endpoint_row("material_render_160k", 160000, report160["endpoint"]),
    ]
    lookup = {row["endpoint"]: row for row in endpoints}
    comparisons = []
    for comparison, old_name, new_name in (
        ("raw_q4_to_material_render_160k", "raw_q4", "material_render_160k"),
        ("material_render_40k_to_160k", "material_render_40k", "material_render_160k"),
    ):
        old = lookup[old_name]
        new = lookup[new_name]
        for metric, label, higher_better in METRICS:
            delta = relative_delta(float(old[metric]), float(new[metric]))
            comparisons.append(
                {
                    "comparison": comparison,
                    "metric": metric,
                    "metric_label": label,
                    "higher_is_better": higher_better,
                    "old": old[metric],
                    "new": new[metric],
                    "absolute_delta": new[metric] - old[metric],
                    "relative_delta_percent": delta,
                    "improvement_percent": delta if higher_better else -delta,
                }
            )

    trajectory = []
    for step in OBSERVATIONS:
        trajectory.append({"step": step, **progress["observations"][str(step)]})

    cases40 = {
        (case["camera_index"], case["light_index"]): case
        for case in report40["endpoint"]["audit_render"]["cases"]
    }
    cases160 = {
        (case["camera_index"], case["light_index"]): case
        for case in report160["endpoint"]["audit_render"]["cases"]
    }
    cases = []
    for camera, light in sorted(cases40):
        old = cases40[(camera, light)]
        new = cases160[(camera, light)]
        hdr_delta = relative_delta(
            old["masked_linear_hdr_mae"], new["masked_linear_hdr_mae"]
        )
        cases.append(
            {
                "camera_index": camera,
                "light_index": light,
                "hdr_mae_40k": old["masked_linear_hdr_mae"],
                "hdr_mae_160k": new["masked_linear_hdr_mae"],
                "hdr_relative_delta_percent": hdr_delta,
                "hdr_improvement_percent": -hdr_delta,
                "display_ssim_40k": old["display_ssim"],
                "display_ssim_160k": new["display_ssim"],
                "display_ssim_absolute_delta": new["display_ssim"] - old["display_ssim"],
            }
        )
    return endpoints, comparisons, trajectory, cases


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    names = ["seguisb.ttf", "arialbd.ttf"] if bold else ["segoeui.ttf", "arial.ttf"]
    windows_dir = os.environ.get("WINDIR")
    if not windows_dir:
        return ImageFont.load_default()
    for name in names:
        candidate = Path(windows_dir) / "Fonts" / name
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def line_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    rows: list[dict[str, Any]],
    metric: str,
    title: str,
    formatter,
) -> None:
    left, top, right, bottom = box
    plot_left, plot_top, plot_right, plot_bottom = left + 76, top + 52, right - 24, bottom - 54
    values = [float(row[metric]) for row in rows]
    y_min, y_max = min(values), max(values)
    padding = max((y_max - y_min) * 0.14, abs(y_max) * 0.005, 1e-9)
    y_min -= padding
    y_max += padding
    x_min, x_max = 20_000, 160_000

    draw.rounded_rectangle(box, radius=12, fill="#ffffff", outline="#cfd6dc", width=2)
    draw.text((left + 20, top + 16), title, fill="#17212b", font=font(20, True))
    for fraction in (0.0, 0.5, 1.0):
        y = plot_bottom - int((plot_bottom - plot_top) * fraction)
        value = y_min + (y_max - y_min) * fraction
        draw.line((plot_left, y, plot_right, y), fill="#e3e7ea", width=1)
        draw.text((left + 10, y - 9), formatter(value), fill="#66717b", font=font(13))
    for step in (40_000, 80_000, 120_000, 160_000):
        x = plot_left + int((step - x_min) / (x_max - x_min) * (plot_right - plot_left))
        draw.line((x, plot_bottom, x, plot_bottom + 7), fill="#34414d", width=1)
        draw.text((x - 18, plot_bottom + 12), f"{step // 1000}k", fill="#56616b", font=font(13))
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill="#34414d", width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill="#34414d", width=2)

    points = []
    for row in rows:
        x = plot_left + int((row["step"] - x_min) / (x_max - x_min) * (plot_right - plot_left))
        y = plot_bottom - int((float(row[metric]) - y_min) / (y_max - y_min) * (plot_bottom - plot_top))
        points.append((x, y))
    draw.line(points, fill="#2f6b8a", width=4, joint="curve")
    for x, y in points:
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill="#ffffff", outline="#2f6b8a", width=3)
    for step, color in ((40_000, "#d29b27"), (160_000, "#193b51")):
        row = next(item for item in rows if item["step"] == step)
        index = rows.index(row)
        x, y = points[index]
        draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color, outline="#ffffff", width=2)
        label = formatter(float(row[metric]))
        label_x = x - 52 if step == 160_000 else x + 10
        draw.text((label_x, y - 27), label, fill="#17212b", font=font(14, True))


def plot_material_trajectory(rows: list[dict[str, Any]], path: Path) -> None:
    focused = [row for row in rows if row["step"] >= 20_000]
    canvas = Image.new("RGB", (1500, 1040), "#f4f5f3")
    draw = ImageDraw.Draw(canvas)
    draw.text((46, 25), "Lantern material trajectory — material-render", fill="#15202a", font=font(32, True))
    draw.text(
        (46, 68),
        "20k–160k observation nodes · focused y-axes · 40k parent highlighted in gold",
        fill="#56616b",
        font=font(17),
    )
    panels = (
        ("roughness_mae", "Roughness MAE (lower is better)", lambda v: f"{v:.4f}"),
        ("seven_channel_mae", "Seven-channel MAE (lower is better)", lambda v: f"{v:.4f}"),
        ("chroma_magnitude_retention", "Chroma magnitude retention (higher is better)", lambda v: f"{v:.3f}"),
        ("opponent_error", "Opponent error (lower is better)", lambda v: f"{v:.4f}"),
    )
    boxes = (
        (46, 120, 738, 550),
        (762, 120, 1454, 550),
        (46, 580, 738, 1010),
        (762, 580, 1454, 1010),
    )
    for box, panel in zip(boxes, panels, strict=True):
        line_panel(draw, box, focused, *panel)
    canvas.save(path, optimize=True)


def plot_improvements(comparisons: list[dict[str, Any]], path: Path) -> None:
    rows = [row for row in comparisons if row["comparison"] == "material_render_40k_to_160k"]
    canvas = Image.new("RGB", (1500, 980), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    draw.text((48, 25), "Lantern 40k → 160k endpoint changes", fill="#15202a", font=font(32, True))
    draw.text(
        (48, 68),
        "Signed improvement: positive is better; negative is regression. Same source, rig, seed lineage and loss.",
        fill="#56616b",
        font=font(17),
    )
    label_left, zero_x, right = 48, 760, 1450
    top, row_h = 125, 61
    max_abs = max(abs(float(row["improvement_percent"])) for row in rows)
    scale = max(10.0, math.ceil(max_abs * 1.12 / 5.0) * 5.0)
    draw.line((zero_x, top - 10, zero_x, top + row_h * len(rows) - 12), fill="#25313c", width=2)
    for index, row in enumerate(rows):
        y = top + index * row_h
        value = float(row["improvement_percent"])
        draw.text((label_left, y + 8), row["metric_label"], fill="#26323c", font=font(17))
        draw.line((zero_x - 430, y + 38, right, y + 38), fill="#edf0f2", width=1)
        extent = int(value / scale * 560)
        if value >= 0:
            x0, x1 = zero_x, zero_x + extent
            fill, outline = "#2f6b8a", "#193b51"
        else:
            x0, x1 = zero_x + extent, zero_x
            fill, outline = "#f2d9a3", "#b17813"
        draw.rounded_rectangle((x0, y + 11, x1, y + 34), radius=5, fill=fill, outline=outline, width=2)
        text_x = x1 + 10 if value >= 0 else x0 - 70
        draw.text((text_x, y + 8), f"{value:+.2f}%", fill="#17212b", font=font(16, True))
    draw.text((zero_x - 12, top + row_h * len(rows) + 5), "0", fill="#56616b", font=font(14))
    canvas.save(path, optimize=True)


def plot_audit_cases(cases: list[dict[str, Any]], path: Path) -> None:
    ordered = sorted(cases, key=lambda row: row["hdr_improvement_percent"])
    canvas = Image.new("RGB", (1500, 760), "#ffffff")
    draw = ImageDraw.Draw(canvas)
    improved = sum(row["hdr_improvement_percent"] > 0 for row in ordered)
    draw.text((48, 25), "Audit-case HDR MAE changes — 40k to 160k", fill="#15202a", font=font(32, True))
    draw.text(
        (48, 68),
        f"42 fixed camera-light cases · {improved}/42 improve · each point is paired; positive is better",
        fill="#56616b",
        font=font(17),
    )
    left, top, right, bottom = 80, 130, 1450, 650
    values = [float(row["hdr_improvement_percent"]) for row in ordered]
    y_min, y_max = min(values) - 3.0, max(values) + 3.0
    zero_y = bottom - int((0.0 - y_min) / (y_max - y_min) * (bottom - top))
    draw.line((left, zero_y, right, zero_y), fill="#25313c", width=2)
    for index, row in enumerate(ordered):
        x = left + int(index / (len(ordered) - 1) * (right - left))
        value = float(row["hdr_improvement_percent"])
        y = bottom - int((value - y_min) / (y_max - y_min) * (bottom - top))
        color = "#2f6b8a" if value >= 0 else "#d29b27"
        draw.line((x, zero_y, x, y), fill="#cbd3d9", width=1)
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline="#ffffff", width=1)
    for value in (math.ceil(y_min / 10) * 10, 0, math.floor(y_max / 10) * 10):
        y = bottom - int((value - y_min) / (y_max - y_min) * (bottom - top))
        draw.text((22, y - 9), f"{value:+.0f}%", fill="#56616b", font=font(14))
    draw.text((left, bottom + 28), "Cases sorted by paired HDR improvement", fill="#56616b", font=font(15))
    draw.text((48, 704), "Blue = improvement; gold = regression. No variance or significance claim is made from one deterministic seed.", fill="#56616b", font=font(15))
    canvas.save(path, optimize=True)


def preview_source_raw_40_160(path: Path, report40: dict[str, Any], report160: dict[str, Any]) -> None:
    technical = Image.open(SUMMARY_20K).convert("RGB")
    fixed40 = Image.open(FIXED_40K).convert("RGB")
    fixed160 = Image.open(FIXED_160K).convert("RGB")
    views = ("front", "rear", "upper_side", "top")
    tile = 256
    margin, gap, header, row_label, footer = 30, 18, 100, 27, 70
    width = margin * 2 + tile * 4 + gap * 3
    height = header + len(views) * (row_label + tile) + footer
    canvas = Image.new("RGB", (width, height), "#141417")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 16), "Lantern — Source / Raw PCA / material-render 40k / 160k", fill="#f2f2f2", font=font(28, True))
    columns = ("Source", "Raw PCA (raw_q4)", "Render @ 40k", "Render @ 160k")
    for column, label in enumerate(columns):
        draw.text((margin + column * (tile + gap), 62), label, fill="#d7d7d7", font=font(16))
    for row, view in enumerate(views):
        y = header + row * (row_label + tile)
        draw.text((margin, y + 4), view, fill="#a9a9ad", font=font(15))
        image_y = y + row_label
        source = technical.crop((0, row * 310 + 54, tile, row * 310 + 54 + tile))
        raw = technical.crop((tile, row * 310 + 54, tile * 2, row * 310 + 54 + tile))
        trained40 = fixed40.crop((row * tile, 24, (row + 1) * tile, 24 + tile))
        trained160 = fixed160.crop((row * tile, 24, (row + 1) * tile, 24 + tile))
        for column, image in enumerate((source, raw, trained40, trained160)):
            canvas.paste(image, (margin + column * (tile + gap), image_y))
    audit40 = report40["endpoint"]["audit_render"]["mean"]
    audit160 = report160["endpoint"]["audit_render"]["mean"]
    footer_text = (
        f"Audit HDR MAE {audit40['masked_linear_hdr_mae']:.6f} → {audit160['masked_linear_hdr_mae']:.6f} "
        f"(-4.79%) · display SSIM {audit40['display_ssim']:.6f} → {audit160['display_ssim']:.6f}"
    )
    draw.text((margin, height - footer + 18), footer_text, fill="#d7d7d7", font=font(16))
    draw.text((margin, height - 25), "Source reference and both trained endpoints exclude the same ~3.27% emissive area.", fill="#99999f", font=font(14))
    canvas.save(path, optimize=True)


def preview_source_raw_20_160(path: Path) -> None:
    """Show the saved intermediate endpoint on the continued material-render lineage."""
    technical = Image.open(SUMMARY_20K).convert("RGB")
    fixed20 = Image.open(FIXED_20K).convert("RGB")
    fixed160 = Image.open(FIXED_160K).convert("RGB")
    views = ("front", "rear", "upper_side", "top")
    tile = 256
    margin, gap, header, row_label, footer = 30, 18, 100, 27, 66
    width = margin * 2 + tile * 4 + gap * 3
    height = header + len(views) * (row_label + tile) + footer
    canvas = Image.new("RGB", (width, height), "#141417")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (margin, 16),
        "Lantern — Source / Raw PCA / material-render 20k / 160k",
        fill="#f2f2f2",
        font=font(28, True),
    )
    columns = ("Source", "Raw PCA (raw_q4)", "Render @ 20k", "Render @ 160k")
    for column, label in enumerate(columns):
        draw.text((margin + column * (tile + gap), 62), label, fill="#d7d7d7", font=font(16))
    for row, view in enumerate(views):
        y = header + row * (row_label + tile)
        draw.text((margin, y + 4), view, fill="#a9a9ad", font=font(15))
        image_y = y + row_label
        source = technical.crop((0, row * 310 + 54, tile, row * 310 + 54 + tile))
        raw = technical.crop((tile, row * 310 + 54, tile * 2, row * 310 + 54 + tile))
        trained20 = fixed20.crop((row * tile, 24, (row + 1) * tile, 24 + tile))
        trained160 = fixed160.crop((row * tile, 24, (row + 1) * tile, 24 + tile))
        for column, image in enumerate((source, raw, trained20, trained160)):
            canvas.paste(image, (margin + column * (tile + gap), image_y))
    draw.text(
        (margin, height - footer + 17),
        "The intermediate column is the formal material-render@20k endpoint continued through 40k to 160k.",
        fill="#d7d7d7",
        font=font(16),
    )
    draw.text(
        (margin, height - 23),
        "All four columns use the same no-emissive Core-4 scope (~3.27% emissive area excluded).",
        fill="#99999f",
        font=font(14),
    )
    canvas.save(path, optimize=True)


def preview_direct(path: Path) -> None:
    image40 = Image.open(FIXED_40K).convert("RGB")
    image160 = Image.open(FIXED_160K).convert("RGB")
    difference = ImageEnhance.Contrast(ImageChops.difference(image40, image160)).enhance(12.0)
    tile_w, tile_h = image40.size
    margin, header, label_h, footer = 34, 75, 34, 62
    canvas = Image.new("RGB", (tile_w + margin * 2, header + (tile_h + label_h) * 3 + footer), "#f4f4f1")
    draw = ImageDraw.Draw(canvas)
    draw.text((margin, 18), "Lantern material-render — 40k vs 160k fixed views", fill="#171717", font=font(29, True))
    for row, (label, image) in enumerate((("40k", image40), ("160k", image160), ("|40k − 160k| ×12", difference))):
        y = header + label_h + row * (label_h + tile_h)
        draw.text((margin, y - label_h + 7), label, fill="#454545", font=font(18))
        canvas.paste(image, (margin, y))
    draw.text((margin, canvas.height - footer + 15), "Difference is display-space absolute RGB amplified 12×; it is diagnostic, not an HDR metric.", fill="#555555", font=font(15))
    canvas.save(path, optimize=True)


def analyze() -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figures = OUTPUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    validation, report40, report160, progress = validate()
    endpoints, comparisons, trajectory, cases = build_tables(report40, report160, progress)

    write_json(OUTPUT / "validation.json", validation)
    write_csv(OUTPUT / "endpoint_metrics.csv", endpoints)
    write_csv(OUTPUT / "comparisons.csv", comparisons)
    write_csv(OUTPUT / "material_trajectory.csv", trajectory)
    write_csv(OUTPUT / "audit_case_deltas_40k_to_160k.csv", cases)

    comparison40 = {
        row["metric"]: row
        for row in comparisons
        if row["comparison"] == "material_render_40k_to_160k"
    }
    hdr_improvements = [float(row["hdr_improvement_percent"]) for row in cases]
    ssim_deltas = [float(row["display_ssim_absolute_delta"]) for row in cases]
    summary = {
        "schema_version": 1,
        "status": "share_with_caveats",
        "question": "Does Lantern material-render continue to improve from 40k to 160k, and at what material tradeoff?",
        "audit_hdr_mae_improvement_percent": comparison40["audit_hdr_mae"]["improvement_percent"],
        "audit_display_ssim_absolute_delta": comparison40["audit_display_ssim"]["absolute_delta"],
        "audit_worst_hdr_mae_improvement_percent": comparison40["audit_worst_hdr_mae"]["improvement_percent"],
        "train_hdr_mae_improvement_percent": comparison40["train_hdr_mae"]["improvement_percent"],
        "roughness_mae_improvement_percent": comparison40["roughness_mae"]["improvement_percent"],
        "seven_channel_mae_improvement_percent": comparison40["seven_channel_mae"]["improvement_percent"],
        "base_color_mae_improvement_percent": comparison40["base_color_linear_mae"]["improvement_percent"],
        "opponent_error_improvement_percent": comparison40["opponent_error"]["improvement_percent"],
        "chroma_retention_improvement_percent": comparison40["chroma_magnitude_retention"]["improvement_percent"],
        "audit_hdr_cases_improved": sum(value > 0 for value in hdr_improvements),
        "audit_hdr_cases_total": len(hdr_improvements),
        "audit_hdr_case_median_improvement_percent": statistics.median(hdr_improvements),
        "audit_hdr_case_p10_improvement_percent": percentile(hdr_improvements, 0.10),
        "audit_hdr_case_p90_improvement_percent": percentile(hdr_improvements, 0.90),
        "audit_ssim_cases_improved": sum(value > 0 for value in ssim_deltas),
        "intermediate_audit_endpoints_available": False,
        "single_seed_statistical_significance_claimed": False,
        "emissive_excluded_fraction_approx": 0.0327,
        "formal_holdout_accessed": False,
        "conclusion": (
            "160k improves mean audit/train rendering and roughness/seven-channel error, "
            "but slightly worsens worst-case audit HDR, normal, metallic, opponent error and chroma retention. "
            "It is a useful Pareto endpoint, not a uniform dominance result."
        ),
    }
    write_json(OUTPUT / "analysis_summary.json", summary)

    plot_material_trajectory(trajectory, figures / "material_trajectory_20k_160k.png")
    plot_improvements(comparisons, figures / "endpoint_improvements_40k_160k.png")
    plot_audit_cases(cases, figures / "audit_case_hdr_improvements_40k_160k.png")
    preview_source_raw_40_160(figures / "lantern_source_rawpca_40k_160k.png", report40, report160)
    preview_source_raw_20_160(figures / "lantern_source_rawpca_20k_160k.png")
    preview_direct(figures / "lantern_40k_160k_direct_preview.png")

    report = f"""# Lantern material-render 40k → 160k analysis

## Outcome

The 160k endpoint improves mean audit HDR MAE by `{summary['audit_hdr_mae_improvement_percent']:.2f}%` and train HDR MAE by `{summary['train_hdr_mae_improvement_percent']:.2f}%` relative to the exact 40k parent. Audit display SSIM changes by `{summary['audit_display_ssim_absolute_delta']:+.6f}`. Paired HDR MAE improves in `{summary['audit_hdr_cases_improved']}/42` audit cases.

The gain is not uniform. Worst-case audit HDR MAE regresses by `{-summary['audit_worst_hdr_mae_improvement_percent']:.2f}%`. Roughness MAE improves by `{summary['roughness_mae_improvement_percent']:.2f}%` and seven-channel MAE by `{summary['seven_channel_mae_improvement_percent']:.2f}%`, while opponent error regresses by `{-summary['opponent_error_improvement_percent']:.2f}%` and chroma retention declines by `{-summary['chroma_retention_improvement_percent']:.2f}%`.

## Interpretation

`material_render@160k` is a useful render/roughness-oriented Pareto endpoint, not a universal winner over 40k. The fixed-view visual difference is subtle. The saved evidence contains full 42-case audit evaluation at 40k and 160k only, so it cannot determine whether 60k, 80k, 100k, 120k or 140k had a better audit endpoint.

## Scope

- One deterministic seed; no variance or statistical-significance claim.
- Same raw_q4 parent, source identity, rig, loss, learning rates, quantization and safety.
- Audit cases never participate in backpropagation.
- Approximately 3.27% emissive area is excluded from source, PCA and both trained endpoints.
- Formal holdout was not accessed.
"""
    (OUTPUT / "technical_report.md").write_text(report, encoding="utf-8")

    manifest_lines = []
    for path in sorted(OUTPUT.rglob("*")):
        if path.is_file() and path.name != "LOCAL_MANIFEST.sha256":
            manifest_lines.append(f"{sha256(path)}  {path.relative_to(OUTPUT).as_posix()}\n")
    (OUTPUT / "LOCAL_MANIFEST.sha256").write_text("".join(manifest_lines), encoding="ascii")
    return summary


def main() -> None:
    print(json.dumps(analyze(), sort_keys=True))


if __name__ == "__main__":
    main()
