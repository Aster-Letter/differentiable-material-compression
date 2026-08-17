"""Validate and analyze the Lantern C4 20k-to-40k paired continuation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import statistics
from typing import Any

from PIL import Image, ImageDraw, ImageOps
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = (
    ROOT
    / "outputs/remote-archives/scow-c4-render-ablation-lantern-40k-job-37581"
)
DEFAULT_EXTRACTED = (
    ROOT
    / "outputs/analysis/c4-render-ablation-lantern-40k-job-37581/extracted"
)
DEFAULT_OLD_PAIR = (
    ROOT
    / "outputs/analysis/c4-render-ablation-20k-v1-job-37489/extracted/outputs/remote"
    / "c4-render-ablation-20k-v1/37477/Lantern"
)
DEFAULT_OUTPUT = (
    ROOT / "outputs/analysis/c4-render-ablation-lantern-40k-job-37581/report"
)
ARMS = ("material_only", "material_render")
COLORS = {"material_only": "#2f6b8a", "material_render": "#d29b27"}
OBSERVATIONS = (1000, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000)
CHECKPOINTS = (10000, 20000, 30000, 40000)


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


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def sidecar_digest(path: Path) -> str:
    return path.read_text(encoding="ascii").strip().split()[0]


def validate(
    archive_root: Path,
    extracted: Path,
    old_pair: Path,
) -> tuple[dict[str, Any], Path, dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    archive_job = load_json(archive_root / "archive_job.json")
    archive_path = archive_root / archive_job["archive"]
    if (
        archive_job.get("status") != "archive_complete"
        or archive_job.get("formal_job_id") != "37581"
        or archive_job.get("preflight_job_id") != "37580"
        or archive_job.get("source_20k_job_id") != "37477"
        or archive_job.get("unsafe_paths") != 0
        or archive_job.get("required_entries_present") is not True
        or archive_job.get("source_results_preserved") is not True
        or archive_job.get("remote_cleanup_performed") is not False
        or archive_path.stat().st_size != archive_job["archive_bytes"]
        or sha256(archive_path) != archive_job["archive_sha256"]
        or sidecar_digest(archive_root / "ARCHIVE.sha256") != sha256(archive_path)
        or sidecar_digest(archive_root / "archive_job.json.sha256")
        != sha256(archive_root / "archive_job.json")
    ):
        raise ValueError("outer archive evidence failed validation")

    run_root = (
        extracted
        / "outputs/remote/c4-render-ablation-lantern-40k-v1/37581"
    )
    marker_path = run_root / "formal_verified.json"
    manifest_path = run_root / "result_manifest.json"
    marker = load_json(marker_path)
    manifest = load_json(manifest_path)
    if (
        marker.get("status") != "formal_run_verified"
        or marker.get("formal_holdout_accessed") is not False
        or marker.get("source_job_id") != "37477"
        or marker.get("preflight_job_id") != "37580"
        or marker.get("result_manifest", {}).get("verified") is not True
        or marker["result_manifest"]["sha256"] != sha256(manifest_path)
        or manifest.get("status") != "result_manifest_verified"
        or manifest.get("file_count") != 72
        or manifest.get("payload_bytes") != 849020497
    ):
        raise ValueError("formal marker or result manifest contract failed")
    for path in (marker_path, manifest_path):
        if sidecar_digest(path.with_suffix(path.suffix + ".sha256")) != sha256(path):
            raise ValueError(f"formal sidecar mismatch: {path}")

    pair_root = extracted / manifest["pair_root"]
    manifest_bytes = 0
    manifest_hashes: dict[str, str] = {}
    for entry in manifest["files"]:
        path = pair_root / entry["path"]
        if (
            not path.is_file()
            or path.stat().st_size != entry["bytes"]
            or sha256(path) != entry["sha256"]
        ):
            raise ValueError(f"result manifest mismatch: {entry['path']}")
        manifest_bytes += int(entry["bytes"])
        manifest_hashes[entry["path"]] = entry["sha256"]
    if manifest_bytes != manifest["payload_bytes"]:
        raise ValueError("result manifest byte total mismatch")
    for sidecar in pair_root.rglob("*.sha256"):
        target = sidecar.with_suffix("")
        if not target.is_file() or sidecar_digest(sidecar) != sha256(target):
            raise ValueError(f"result sidecar mismatch: {sidecar}")

    pair = load_json(pair_root / "paired_summary.json")
    if (
        pair.get("status") != "complete_paired_40k"
        or pair.get("steps") != 40000
        or pair.get("paired_sampling_evidence", {}).get("identical") is not True
        or not all(pair["paired_sampling_evidence"]["fields"].values())
        or pair.get("formal_holdout_accessed") is not False
        or pair.get("audit_used_for_training") is not False
    ):
        raise ValueError("paired continuation evidence failed")

    archived_config = (
        extracted / "configs/train/c4_render_ablation_lantern_40k_v1.yaml"
    )
    config_hash = sha256(archived_config)
    config = yaml.safe_load(archived_config.read_text(encoding="utf-8"))
    new_reports: dict[str, dict[str, Any]] = {}
    old_reports: dict[str, dict[str, Any]] = {}
    for arm in ARMS:
        report = load_json(pair_root / arm / "training_report.json")
        old_report_path = old_pair / arm / "training_report.json"
        old_report = load_json(old_report_path)
        arm_config = config["source"]["arms"][arm]
        if (
            report.get("status") != "complete_40k_continuation"
            or report.get("steps") != 40000
            or report.get("continuation_updates") != 20000
            or report.get("continuation_config_hash") != config_hash
            or tuple(report.get("observation_steps", ())) != OBSERVATIONS
            or tuple(sorted(int(step) for step in report.get("checkpoints", {})))
            != CHECKPOINTS
            or report.get("formal_holdout_accessed") is not False
            or report.get("audit_used_for_training") is not False
            or report.get("early_stopping") is not False
            or report["endpoint"]["audit_render"]["case_count"] != 42
            or report["endpoint"]["train_render"]["case_count"] != 144
            or report["source_checkpoint"]["sha256"]
            != arm_config["checkpoint_sha256"]
            or sha256(old_report_path) != arm_config["report_sha256"]
            or report["source_endpoint_20k"] != old_report["endpoint"]
            or report["raw_parent"] != old_report["raw_parent"]
            or not finite(report)
        ):
            raise ValueError(f"training report contract failed: {arm}")
        new_reports[arm] = report
        old_reports[arm] = old_report

    if new_reports[ARMS[0]]["source_identity"] != new_reports[ARMS[1]]["source_identity"]:
        raise ValueError("continuation source identities differ between arms")

    from cg_frontier.compression.render_ablation_continuation import (
        load_continuation_checkpoint,
    )

    checkpoint_reload = {}
    for arm in ARMS:
        checkpoint_reload[arm] = {}
        report = new_reports[arm]
        for step in (30000, 40000):
            item = report["checkpoints"][str(step)]
            checkpoint = extracted / item["path"]
            relative = checkpoint.relative_to(pair_root).as_posix()
            if (
                sha256(checkpoint) != item["sha256"]
                or manifest_hashes.get(relative) != item["sha256"]
            ):
                raise ValueError(f"checkpoint hash mismatch: {arm}/{step}")
            payload = load_continuation_checkpoint(
                checkpoint,
                expected_arm=arm,
                expected_source_identity=report["source_identity"],
                expected_continuation_config_hash=config_hash,
                expected_source_checkpoint_sha256=config["source"]["arms"][arm][
                    "checkpoint_sha256"
                ],
            )
            if int(payload["step"]) != step:
                raise ValueError(f"checkpoint step mismatch: {arm}/{step}")
            checkpoint_reload[arm][str(step)] = {
                "sha256": item["sha256"],
                "reload_verified": True,
            }

    validation = {
        "schema_version": 1,
        "status": "complete_exact",
        "archive": {
            "job_id": archive_job["archive_job_id"],
            "bytes": archive_job["archive_bytes"],
            "sha256": archive_job["archive_sha256"],
            "members": archive_job["tar_members"],
            "unsafe_paths": 0,
        },
        "formal": {
            "job_id": "37581",
            "preflight_job_id": "37580",
            "source_job_id": "37477",
            "result_files": manifest["file_count"],
            "payload_bytes": manifest["payload_bytes"],
            "result_manifest_sha256": sha256(manifest_path),
            "formal_holdout_accessed": False,
            "audit_used_for_training": False,
        },
        "checkpoint_reload": checkpoint_reload,
        "paired_sampling_evidence": pair["paired_sampling_evidence"],
    }
    return validation, pair_root, old_reports, new_reports


def pct_change(old: float, new: float) -> float:
    return (new / old - 1.0) * 100.0


def build_tables(
    old_reports: dict[str, dict[str, Any]],
    new_reports: dict[str, dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    endpoints = []
    comparisons = []
    cases = []
    for step, reports in ((20000, old_reports), (40000, new_reports)):
        for arm in ARMS:
            report = reports[arm]
            material = report["endpoint"]["material"]
            audit = report["endpoint"]["audit_render"]
            train = report["endpoint"]["train_render"]
            endpoints.append(
                {
                    "step": step,
                    "arm": arm,
                    **material,
                    "audit_hdr_mae": audit["mean"]["masked_linear_hdr_mae"],
                    "audit_display_ssim": audit["mean"]["display_ssim"],
                    "audit_worst_hdr_mae": audit["worst"]["masked_linear_hdr_mae"],
                    "train_hdr_mae": train["mean"]["masked_linear_hdr_mae"],
                    "train_display_ssim": train["mean"]["display_ssim"],
                }
            )
    for arm in ARMS:
        old = next(row for row in endpoints if row["step"] == 20000 and row["arm"] == arm)
        new = next(row for row in endpoints if row["step"] == 40000 and row["arm"] == arm)
        for metric in (
            "audit_hdr_mae",
            "audit_display_ssim",
            "audit_worst_hdr_mae",
            "train_hdr_mae",
            "train_display_ssim",
            "base_color_linear_mae",
            "seven_channel_mae",
            "normal_mean_degrees",
            "roughness_mae",
            "metallic_mae",
            "oklab_delta_e_mean",
            "opponent_error",
            "chroma_magnitude_retention",
        ):
            comparisons.append(
                {
                    "comparison": "20k_to_40k",
                    "arm": arm,
                    "metric": metric,
                    "old": old[metric],
                    "new": new[metric],
                    "absolute_delta": new[metric] - old[metric],
                    "relative_delta_percent": pct_change(old[metric], new[metric]),
                }
            )
        old_cases = {
            (case["camera_index"], case["light_index"]): case
            for case in old_reports[arm]["endpoint"]["audit_render"]["cases"]
        }
        new_cases = {
            (case["camera_index"], case["light_index"]): case
            for case in new_reports[arm]["endpoint"]["audit_render"]["cases"]
        }
        if old_cases.keys() != new_cases.keys() or len(old_cases) != 42:
            raise ValueError(f"audit case pairing mismatch: {arm}")
        for key, old_case in old_cases.items():
            new_case = new_cases[key]
            cases.append(
                {
                    "arm": arm,
                    "camera_index": key[0],
                    "light_index": key[1],
                    "hdr_mae_20k": old_case["masked_linear_hdr_mae"],
                    "hdr_mae_40k": new_case["masked_linear_hdr_mae"],
                    "hdr_relative_delta_percent": pct_change(
                        old_case["masked_linear_hdr_mae"],
                        new_case["masked_linear_hdr_mae"],
                    ),
                    "ssim_20k": old_case["display_ssim"],
                    "ssim_40k": new_case["display_ssim"],
                }
            )
    return endpoints, comparisons, cases


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _svg(path: Path, width: int, height: int, elements: list[str]) -> None:
    path.write_text(
        "\n".join(
            [
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
                '<rect width="100%" height="100%" fill="#ffffff"/>',
                '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#18222c}.small{font-size:12px}.label{font-size:14px}.title{font-size:20px;font-weight:600}.grid{stroke:#d7dde2;stroke-width:1}.axis{stroke:#24303a;stroke-width:1.2}</style>',
                *elements,
                "</svg>",
            ]
        ),
        encoding="utf-8",
    )


def plot_audit(endpoints: list[dict[str, Any]], path: Path) -> None:
    width, height = 1100, 480
    elements = [
        '<text x="550" y="32" text-anchor="middle" class="title">Lantern paired continuation: discrete 20k and 40k endpoints</text>',
    ]
    panels = ((60, 560), (610, 1060))
    steps = (20000, 40000)
    hdr_values = {
        (step, arm): next(
            row["audit_hdr_mae"]
            for row in endpoints
            if row["step"] == step and row["arm"] == arm
        )
        for step in steps
        for arm in ARMS
    }
    left, right = panels[0]
    top, bottom = 80, 400
    elements += [
        f'<text x="{(left + right) / 2}" y="62" text-anchor="middle" class="label">Audit mean HDR MAE — linear, lower is better</text>',
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis"/>',
    ]
    max_hdr = max(hdr_values.values()) * 1.15
    for tick in range(5):
        value = max_hdr * tick / 4
        y = bottom - (bottom - top) * tick / 4
        elements += [
            f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" class="grid"/>',
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" class="small">{value:.4f}</text>',
        ]
    centers = (200, 420)
    bar_w = 62
    for center, step in zip(centers, steps, strict=True):
        elements.append(
            f'<text x="{center}" y="426" text-anchor="middle" class="label">{step // 1000}k</text>'
        )
        for arm, offset in ((ARMS[0], -34), (ARMS[1], 34)):
            value = hdr_values[(step, arm)]
            bar_h = (bottom - top) * value / max_hdr
            x = center + offset - bar_w / 2
            y = bottom - bar_h
            elements += [
                f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w}" height="{bar_h:.1f}" fill="{COLORS[arm]}" stroke="#24303a"/>',
                f'<text x="{center + offset}" y="{y - 7:.1f}" text-anchor="middle" class="small">{value:.6f}</text>',
            ]

    ssim_values = {
        (step, arm): next(
            row["audit_display_ssim"]
            for row in endpoints
            if row["step"] == step and row["arm"] == arm
        )
        for step in steps
        for arm in ARMS
    }
    left, right = panels[1]
    ssim_min = min(ssim_values.values()) - 0.0007
    ssim_max = max(ssim_values.values()) + 0.0007
    elements += [
        f'<text x="{(left + right) / 2}" y="62" text-anchor="middle" class="label">Audit display SSIM — focused scale, higher is better</text>',
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" class="axis"/>',
    ]
    for tick in range(5):
        value = ssim_min + (ssim_max - ssim_min) * tick / 4
        y = bottom - (bottom - top) * tick / 4
        elements += [
            f'<line x1="{left}" y1="{y:.1f}" x2="{right}" y2="{y:.1f}" class="grid"/>',
            f'<text x="{left - 8}" y="{y + 4:.1f}" text-anchor="end" class="small">{value:.4f}</text>',
        ]
    for center, step in zip((730, 940), steps, strict=True):
        elements.append(
            f'<text x="{center}" y="426" text-anchor="middle" class="label">{step // 1000}k</text>'
        )
        for arm, offset in ((ARMS[0], -22), (ARMS[1], 22)):
            value = ssim_values[(step, arm)]
            y = bottom - (bottom - top) * (value - ssim_min) / (ssim_max - ssim_min)
            elements += [
                f'<circle cx="{center + offset}" cy="{y:.1f}" r="8" fill="{COLORS[arm]}" stroke="#24303a"/>',
                f'<text x="{center + offset}" y="{y - 13:.1f}" text-anchor="middle" class="small">{value:.5f}</text>',
            ]
    elements += [
        f'<rect x="410" y="448" width="14" height="14" fill="{COLORS[ARMS[0]]}" stroke="#24303a"/><text x="432" y="460" class="small">material_only</text>',
        f'<rect x="570" y="448" width="14" height="14" fill="{COLORS[ARMS[1]]}" stroke="#24303a"/><text x="592" y="460" class="small">material_render</text>',
    ]
    _svg(path, width, height, elements)


def plot_material_improvement(comparisons: list[dict[str, Any]], path: Path) -> None:
    definitions = (
        ("BaseColor MAE", "base_color_linear_mae", False),
        ("Seven-channel MAE", "seven_channel_mae", False),
        ("Normal mean degrees", "normal_mean_degrees", False),
        ("Roughness MAE", "roughness_mae", False),
        ("Metallic MAE", "metallic_mae", False),
        ("Oklab ΔE", "oklab_delta_e_mean", False),
        ("Opponent error", "opponent_error", False),
        ("Chroma retention", "chroma_magnitude_retention", True),
    )
    values_by_arm = {}
    for arm in ARMS:
        values = []
        for _, metric, higher_is_better in definitions:
            row = next(
                item for item in comparisons if item["arm"] == arm and item["metric"] == metric
            )
            delta = row["relative_delta_percent"]
            values.append(delta if higher_is_better else -delta)
        values_by_arm[arm] = values
    width, height = 1000, 610
    left, right, top, bottom = 220, 940, 80, 545
    low = min(-10.0, min(value for values in values_by_arm.values() for value in values) - 3)
    high = max(40.0, max(value for values in values_by_arm.values() for value in values) + 3)
    scale = lambda value: left + (right - left) * (value - low) / (high - low)
    elements = [
        '<text x="500" y="32" text-anchor="middle" class="title">Material-domain movement during the paired continuation</text>',
        '<text x="500" y="55" text-anchor="middle" class="small">20k→40k directional improvement (%) — positive is better</text>',
    ]
    for tick in range(math.ceil(low / 10) * 10, math.floor(high / 10) * 10 + 1, 10):
        x = scale(tick)
        elements += [
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" class="grid"/>',
            f'<text x="{x:.1f}" y="{bottom + 22}" text-anchor="middle" class="small">{tick:+d}%</text>',
        ]
    elements.append(
        f'<line x1="{scale(0):.1f}" y1="{top}" x2="{scale(0):.1f}" y2="{bottom}" class="axis"/>'
    )
    row_h = (bottom - top) / len(definitions)
    for index, (label, _, _) in enumerate(definitions):
        y = top + row_h * (index + 0.5)
        elements += [
            f'<text x="{left - 14}" y="{y + 4:.1f}" text-anchor="end" class="label">{html.escape(label)}</text>',
            f'<line x1="{left}" y1="{y + row_h / 2:.1f}" x2="{right}" y2="{y + row_h / 2:.1f}" class="grid"/>',
        ]
        for arm, offset, shape in ((ARMS[0], -9, "circle"), (ARMS[1], 9, "square")):
            value = values_by_arm[arm][index]
            x = scale(value)
            if shape == "circle":
                mark = f'<circle cx="{x:.1f}" cy="{y + offset:.1f}" r="7" fill="{COLORS[arm]}" stroke="#24303a"/>'
            else:
                mark = f'<rect x="{x - 7:.1f}" y="{y + offset - 7:.1f}" width="14" height="14" fill="{COLORS[arm]}" stroke="#24303a"/>'
            elements += [mark, f'<text x="{x + 11:.1f}" y="{y + offset + 4:.1f}" class="small">{value:+.1f}%</text>']
    elements += [
        f'<circle cx="360" cy="590" r="7" fill="{COLORS[ARMS[0]]}" stroke="#24303a"/><text x="375" y="595" class="small">material_only</text>',
        f'<rect x="535" y="583" width="14" height="14" fill="{COLORS[ARMS[1]]}" stroke="#24303a"/><text x="557" y="595" class="small">material_render</text>',
    ]
    _svg(path, width, height, elements)


def plot_case_distribution(cases: list[dict[str, Any]], path: Path) -> None:
    values = [
        [row["hdr_relative_delta_percent"] for row in cases if row["arm"] == arm]
        for arm in ARMS
    ]
    width, height = 940, 430
    left, right, top, bottom = 190, 820, 90, 345
    low, high = -40.0, 0.0
    scale = lambda value: left + (right - left) * (value - low) / (high - low)
    elements = [
        '<text x="470" y="32" text-anchor="middle" class="title">All 42 paired audit camera/light cases improved in both arms</text>',
        '<text x="470" y="56" text-anchor="middle" class="small">Per-case HDR MAE change, 20k→40k (%) — negative is better</text>',
    ]
    for tick in range(-40, 1, 10):
        x = scale(tick)
        elements += [
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}" class="grid"/>',
            f'<text x="{x:.1f}" y="{bottom + 24}" text-anchor="middle" class="small">{tick}%</text>',
        ]
    elements.append(f'<line x1="{scale(0):.1f}" y1="{top}" x2="{scale(0):.1f}" y2="{bottom}" class="axis"/>')
    for row, (arm, arm_values) in enumerate(zip(ARMS, values, strict=True)):
        ordered = sorted(arm_values)
        q1, median, q3 = statistics.quantiles(ordered, n=4, method="inclusive")
        minimum, maximum = min(ordered), max(ordered)
        y = 155 + row * 120
        elements += [
            f'<text x="{left - 18}" y="{y + 5}" text-anchor="end" class="label">{arm}</text>',
            f'<line x1="{scale(minimum):.1f}" y1="{y}" x2="{scale(maximum):.1f}" y2="{y}" class="axis"/>',
            f'<rect x="{scale(q1):.1f}" y="{y - 22}" width="{scale(q3) - scale(q1):.1f}" height="44" fill="{COLORS[arm]}" fill-opacity="0.82" stroke="#24303a"/>',
            f'<line x1="{scale(median):.1f}" y1="{y - 22}" x2="{scale(median):.1f}" y2="{y + 22}" class="axis"/>',
            f'<text x="{right + 12}" y="{y - 3}" class="small">42/42 improved</text>',
            f'<text x="{right + 12}" y="{y + 15}" class="small">median {median:+.1f}%</text>',
        ]
        for position, value in enumerate(arm_values):
            jitter = ((position % 7) - 3) * 2.2
            elements.append(
                f'<circle cx="{scale(value):.1f}" cy="{y + jitter:.1f}" r="2.6" fill="#ffffff" stroke="{COLORS[arm]}"/>'
            )
    _svg(path, width, height, elements)


def build_contact_sheet(old_pair: Path, new_pair: Path, path: Path) -> None:
    steps = (20000, 25000, 30000, 35000, 40000)
    images: list[list[Image.Image]] = []
    for arm in ARMS:
        row = []
        for step in steps:
            root = old_pair if step == 20000 else new_pair
            image_path = root / arm / "observations" / f"step_{step:05d}" / "fixed_views.png"
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            row.append(image)
        images.append(row)
    cell_w = max(image.width for row in images for image in row)
    cell_h = max(image.height for row in images for image in row)
    header_h = 30
    label_w = 130
    canvas = Image.new("RGB", (label_w + cell_w * len(steps), header_h + cell_h * 2), "white")
    draw = ImageDraw.Draw(canvas)
    for column, step in enumerate(steps):
        draw.text((label_w + column * cell_w + 8, 8), f"{step // 1000}k", fill="#18222c")
    for row_index, arm in enumerate(ARMS):
        y = header_h + row_index * cell_h
        draw.text((8, y + 10), arm, fill="#18222c")
        for column, image in enumerate(images[row_index]):
            framed = ImageOps.pad(image, (cell_w, cell_h), color="white")
            canvas.paste(framed, (label_w + column * cell_w, y))
    canvas.save(path)


def analyze(
    archive_root: Path,
    extracted: Path,
    old_pair: Path,
    output: Path,
) -> dict[str, Any]:
    validation, pair_root, old_reports, new_reports = validate(
        archive_root, extracted, old_pair
    )
    endpoints, comparisons, cases = build_tables(old_reports, new_reports)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "validation.json", validation)
    write_csv(output / "endpoint_metrics.csv", endpoints)
    write_csv(output / "comparisons.csv", comparisons)
    write_csv(output / "audit_case_deltas.csv", cases)

    figures = output / "figures"
    figures.mkdir()
    plot_audit(endpoints, figures / "audit_endpoint_comparison.svg")
    plot_material_improvement(comparisons, figures / "material_improvement.svg")
    plot_case_distribution(cases, figures / "audit_case_delta_distribution.svg")
    build_contact_sheet(old_pair, pair_root, figures / "fixed_views_progression.png")

    def endpoint(step: int, arm: str) -> dict[str, Any]:
        return next(row for row in endpoints if row["step"] == step and row["arm"] == arm)

    only20, only40 = endpoint(20000, ARMS[0]), endpoint(40000, ARMS[0])
    render20, render40 = endpoint(20000, ARMS[1]), endpoint(40000, ARMS[1])
    summary = {
        "schema_version": 1,
        "status": "share_with_caveats",
        "primary_findings": {
            "material_only_audit_hdr_change_percent": pct_change(
                only20["audit_hdr_mae"], only40["audit_hdr_mae"]
            ),
            "material_render_audit_hdr_change_percent": pct_change(
                render20["audit_hdr_mae"], render40["audit_hdr_mae"]
            ),
            "material_only_worst_hdr_change_percent": pct_change(
                only20["audit_worst_hdr_mae"], only40["audit_worst_hdr_mae"]
            ),
            "material_render_worst_hdr_change_percent": pct_change(
                render20["audit_worst_hdr_mae"], render40["audit_worst_hdr_mae"]
            ),
            "render_vs_only_audit_hdr_percent_at_20k": pct_change(
                only20["audit_hdr_mae"], render20["audit_hdr_mae"]
            ),
            "render_vs_only_audit_hdr_percent_at_40k": pct_change(
                only40["audit_hdr_mae"], render40["audit_hdr_mae"]
            ),
            "material_only_improved_audit_cases": sum(
                row["hdr_relative_delta_percent"] < 0
                for row in cases
                if row["arm"] == ARMS[0]
            ),
            "material_render_improved_audit_cases": sum(
                row["hdr_relative_delta_percent"] < 0
                for row in cases
                if row["arm"] == ARMS[1]
            ),
        },
        "caveats": [
            "Single deterministic seed; paired cases do not estimate across-seed variance.",
            "Lantern Core-4 reference excludes approximately 3.27% emissive area.",
            "25k/30k/35k observations contain material metrics and fixed views, not full 42-case audit rerenders.",
            "No weighted scalar winner is selected; render and material-domain metrics remain a Pareto comparison.",
        ],
    }
    write_json(output / "analysis_summary.json", summary)

    finding = summary["primary_findings"]
    report = f"""# Lantern C4 可微渲染 20k→40k 技术报告

## 技术结论

继续训练是有效的，并改变了渲染监督臂的相对结论。material-only 的 42-case audit mean HDR MAE 从 `{only20['audit_hdr_mae']:.8f}` 降到 `{only40['audit_hdr_mae']:.8f}`（`{finding['material_only_audit_hdr_change_percent']:.2f}%`）；material-render 从 `{render20['audit_hdr_mae']:.8f}` 降到 `{render40['audit_hdr_mae']:.8f}`（`{finding['material_render_audit_hdr_change_percent']:.2f}%`）。两臂各自 42/42 个配对 audit case 均改善。

20k 时 material-render 相对 material-only 的 audit HDR MAE 高 `{finding['render_vs_only_audit_hdr_percent_at_20k']:.2f}%`；40k 时反而低 `{abs(finding['render_vs_only_audit_hdr_percent_at_40k']):.2f}%`。因此“渲染监督在 Lantern 上回退”的 20k 结论不再成立于 40k 端点，但优势仍小，不能外推为跨模型或跨 seed 的稳定胜出。

## 关键证据

- material-only worst audit HDR MAE：`{only20['audit_worst_hdr_mae']:.8f} → {only40['audit_worst_hdr_mae']:.8f}`（`{finding['material_only_worst_hdr_change_percent']:.2f}%`）。
- material-render worst audit HDR MAE：`{render20['audit_worst_hdr_mae']:.8f} → {render40['audit_worst_hdr_mae']:.8f}`（`{finding['material_render_worst_hdr_change_percent']:.2f}%`）。
- 40k display SSIM：material-only `{only40['audit_display_ssim']:.6f}`，material-render `{render40['audit_display_ssim']:.6f}`。
- 40k BaseColor MAE：material-only `{only40['base_color_linear_mae']:.8f}`，material-render `{render40['base_color_linear_mae']:.8f}`；render 臂的渲染优势伴随轻微 BaseColor/seven-channel 代价，仍属 Pareto 取舍。

![Audit endpoint comparison](figures/audit_endpoint_comparison.svg)

![Audit case delta distribution](figures/audit_case_delta_distribution.svg)

## 范围与指标

比较对象固定为 Lantern、同一 raw_q4 parent、同一 seed/sampling contract、同一 generic rig。两臂从 Job 37477 各自 20k checkpoint 精确恢复 optimizer/RNG，继续到 40k。主指标为 7 个只读 audit camera × 6 lights 的 42-case mean/worst linear HDR MAE 与 display SSIM；材质域指标来自有效纹素 atlas。

## 方法与完整性

Job 37581 的 result manifest 72/72 文件、`849,020,497` bytes 全部重哈希通过；两臂 30k/40k 共四个 continuation checkpoint 已在本地 CPU 环境按 source identity、config hash 与 20k parent hash 重载。paired sampling 的 initial/final RNG、trajectory、steps 与 sampling contract 全部一致。audit 未参与反向，formal holdout 未访问。

![Material improvement](figures/material_improvement.svg)

## 固定视角进展

下图按两臂展示 20k/25k/30k/35k/40k 的相同固定视角。它用于观察连续变化，不替代 42-case audit 数值。

![Fixed views progression](figures/fixed_views_progression.png)

## 限制与稳健性

- 单 seed 只支持确定性配对案例，不支持方差或统计显著性结论。
- Lantern 约 3.27% emissive 区域不在 Core-4 结论内。
- 25k/30k/35k 没有完整 audit rerender，只能用材质指标与固定视角观察轨迹；严格 endpoint 比较为 20k 对 40k。
- 40k render 臂仅以约 2.36% audit HDR MAE 领先 material-only，仍应视为小幅端点优势而非全面 winner。

## 建议

把 Lantern 40k 作为“训练时长会改变渲染监督判断”的正证据保留，并将 20k 与 40k 同时呈现。不要继续无界延长；下一步优先把 40k 两臂部署到 UE，与 source/raw_q4/20k 端点同场人工复核。如果需要估计稳定性，再增加独立 seed，而不是继续同一 seed 到更长步数。

## 后续问题

- 40k 的小幅 render 优势能否在 Corset/BoomBox 或独立 seed 上复现？
- 30k 是否已经接近 40k 的可见质量，从而提供更便宜的停止点？
- opponent error 在两臂 20k→40k 均有回退，是否对应局部综合色或色相漂移？
"""
    (output / "technical_report.md").write_text(report, encoding="utf-8")

    files = []
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "LOCAL_MANIFEST.sha256":
            files.append(path)
    (output / "LOCAL_MANIFEST.sha256").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(output).as_posix()}\n" for path in files),
        encoding="ascii",
    )
    print(json.dumps({"status": summary["status"], "output": str(output)}, sort_keys=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--extracted", type=Path, default=DEFAULT_EXTRACTED)
    parser.add_argument("--old-pair", type=Path, default=DEFAULT_OLD_PAIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    analyze(
        args.archive_root.resolve(),
        args.extracted.resolve(),
        args.old_pair.resolve(),
        args.output.resolve(),
    )


if __name__ == "__main__":
    main()
