"""Analyze the verified three-asset C4 render-ablation campaign.

This script is intentionally CPU-only.  It consumes an already verified local
extraction, recomputes the paired comparisons used in the report, and writes
portable CSV/JSON/Markdown evidence plus static figures.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
from statistics import median

from PIL import Image, ImageDraw, ImageFont


RUNS = (("37474", "Corset"), ("37477", "Lantern"), ("37478", "BoomBox"))
ARMS = ("material_only", "material_render")
ENDPOINTS = ("raw_q4", "material_only", "material_render")
OBSERVATION_STEPS = (1000, 5000, 10000, 15000, 20000)


def _finite_numbers(value: object) -> bool:
    if isinstance(value, dict):
        return all(_finite_numbers(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite_numbers(item) for item in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _write_report_manifest(output: Path) -> None:
    manifest_path = output / "LOCAL_ANALYSIS_MANIFEST.sha256"
    lines = []
    for path in sorted(output.rglob("*")):
        if not path.is_file() or path == manifest_path:
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(output).as_posix()}")
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _pct_delta(new: float, old: float) -> float:
    return (new / old - 1.0) * 100.0


def _endpoint_rows(run_root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    endpoint_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    for job, asset in RUNS:
        pair_root = run_root / job / asset
        reports = {
            arm: _read(pair_root / arm / "training_report.json")
            for arm in ARMS
        }
        endpoints = {
            "raw_q4": reports["material_only"]["raw_parent"],
            "material_only": reports["material_only"]["endpoint"],
            "material_render": reports["material_render"]["endpoint"],
        }
        for endpoint, payload in endpoints.items():
            material = payload["material"]
            audit = payload["audit_render"]
            train = payload["train_render"]
            endpoint_rows.append(
                {
                    "asset": asset,
                    "job_id": job,
                    "endpoint": endpoint,
                    "audit_mean_hdr_mae": audit["mean"]["masked_linear_hdr_mae"],
                    "audit_display_ssim": audit["mean"]["display_ssim"],
                    "audit_worst_hdr_mae": audit["worst"]["masked_linear_hdr_mae"],
                    "train_mean_hdr_mae": train["mean"]["masked_linear_hdr_mae"],
                    "base_color_linear_mae": material["base_color_linear_mae"],
                    "seven_channel_mae": material["seven_channel_mae"],
                    "normal_mean_degrees": material["normal_mean_degrees"],
                    "roughness_mae": material["roughness_mae"],
                    "metallic_mae": material["metallic_mae"],
                    "chroma_magnitude_retention": material["chroma_magnitude_retention"],
                    "oklab_delta_e_mean": material["oklab_delta_e_mean"],
                    "opponent_error": material["opponent_error"],
                }
            )
        for start, end, isolates_render in (
            ("raw_q4", "material_only", False),
            ("raw_q4", "material_render", False),
            ("material_only", "material_render", True),
        ):
            left, right = endpoints[start], endpoints[end]
            comparison_rows.append(
                {
                    "asset": asset,
                    "comparison": f"{start}_to_{end}",
                    "isolates_render_supervision": isolates_render,
                    "audit_mean_hdr_mae_delta": right["audit_render"]["mean"]["masked_linear_hdr_mae"]
                    - left["audit_render"]["mean"]["masked_linear_hdr_mae"],
                    "audit_mean_hdr_mae_delta_pct": _pct_delta(
                        right["audit_render"]["mean"]["masked_linear_hdr_mae"],
                        left["audit_render"]["mean"]["masked_linear_hdr_mae"],
                    ),
                    "audit_display_ssim_delta": right["audit_render"]["mean"]["display_ssim"]
                    - left["audit_render"]["mean"]["display_ssim"],
                    "audit_worst_hdr_mae_delta_pct": _pct_delta(
                        right["audit_render"]["worst"]["masked_linear_hdr_mae"],
                        left["audit_render"]["worst"]["masked_linear_hdr_mae"],
                    ),
                    "train_mean_hdr_mae_delta_pct": _pct_delta(
                        right["train_render"]["mean"]["masked_linear_hdr_mae"],
                        left["train_render"]["mean"]["masked_linear_hdr_mae"],
                    ),
                    "base_color_mae_delta_pct": _pct_delta(
                        right["material"]["base_color_linear_mae"],
                        left["material"]["base_color_linear_mae"],
                    ),
                    "seven_channel_mae_delta_pct": _pct_delta(
                        right["material"]["seven_channel_mae"],
                        left["material"]["seven_channel_mae"],
                    ),
                    "normal_degrees_delta_pct": _pct_delta(
                        right["material"]["normal_mean_degrees"],
                        left["material"]["normal_mean_degrees"],
                    ),
                    "chroma_retention_delta": right["material"]["chroma_magnitude_retention"]
                    - left["material"]["chroma_magnitude_retention"],
                }
            )
    return endpoint_rows, comparison_rows


def _trajectory_rows(run_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for job, asset in RUNS:
        pair_root = run_root / job / asset
        for arm in ARMS:
            progress = _read(pair_root / arm / "progress.json")
            by_step = {int(item["step"]): item for item in progress["sample_metrics"]}
            for step in OBSERVATION_STEPS:
                item = by_step[step]
                rows.append({"asset": asset, "arm": arm, **item})
    return rows


def _case_rows(run_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for job, asset in RUNS:
        pair_root = run_root / job / asset
        reports = {arm: _read(pair_root / arm / "training_report.json") for arm in ARMS}
        endpoints = {
            "raw_q4": reports["material_only"]["raw_parent"],
            "material_only": reports["material_only"]["endpoint"],
            "material_render": reports["material_render"]["endpoint"],
        }
        indexed = {
            endpoint: {
                (int(case["camera_index"]), int(case["light_index"])): case
                for case in payload["audit_render"]["cases"]
            }
            for endpoint, payload in endpoints.items()
        }
        for key in sorted(indexed["raw_q4"]):
            camera, light = key
            row: dict[str, object] = {"asset": asset, "camera_index": camera, "light_index": light}
            for endpoint in ENDPOINTS:
                case = indexed[endpoint][key]
                row[f"{endpoint}_hdr_mae"] = case["masked_linear_hdr_mae"]
                row[f"{endpoint}_display_ssim"] = case["display_ssim"]
            row["render_vs_only_hdr_delta"] = row["material_render_hdr_mae"] - row["material_only_hdr_mae"]
            row["render_vs_only_ssim_delta"] = row["material_render_display_ssim"] - row["material_only_display_ssim"]
            rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


INK = "#111827"
GRID = "#d1d5db"
BLUE = "#2563eb"
ORANGE = "#f97316"
GRAY = "#6b7280"


def _font(size: int, *, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    windows_dir = os.environ.get("WINDIR")
    if not windows_dir:
        return ImageFont.load_default()
    font_root = Path(windows_dir) / "Fonts"
    names = ("arialbd.ttf", "segoeuib.ttf") if bold else ("arial.ttf", "segoeui.ttf")
    candidates = tuple(font_root / name for name in names)
    for path in candidates:
        if path.is_file():
            return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _canvas(title: str, subtitle: str, *, width: int = 1800, height: int = 760) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((70, 42), title, fill=INK, font=_font(34, bold=True))
    draw.text((70, 90), subtitle, fill="#4b5563", font=_font(20))
    return image, draw


def _bar_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    title: str,
    groups: list[str],
    series: list[tuple[str, str, list[float]]],
    *,
    y_min: float,
    y_max: float,
    y_label: str,
) -> None:
    left, top, right, bottom = box
    draw.text((left, top - 45), title, fill=INK, font=_font(24, bold=True))
    plot_left, plot_top, plot_right, plot_bottom = left + 115, top + 10, right - 25, bottom - 70
    for tick in range(6):
        value = y_min + (y_max - y_min) * tick / 5
        y = plot_bottom - (plot_bottom - plot_top) * tick / 5
        draw.line((plot_left, y, plot_right, y), fill=GRID, width=1)
        draw.text((plot_left - 105, y - 10), f"{value:.4f}", fill="#4b5563", font=_font(16))
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=INK, width=2)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=INK, width=2)
    group_width = (plot_right - plot_left) / len(groups)
    bar_width = min(46, group_width / (len(series) + 1))
    for group_index, group in enumerate(groups):
        center = plot_left + (group_index + 0.5) * group_width
        draw.text((center - 45, plot_bottom + 18), group, fill=INK, font=_font(17))
        for series_index, (_, color, values) in enumerate(series):
            value = values[group_index]
            x0 = center + (series_index - (len(series) - 1) / 2) * bar_width * 1.15 - bar_width / 2
            x1 = x0 + bar_width
            y = plot_bottom - (value - y_min) / (y_max - y_min) * (plot_bottom - plot_top)
            draw.rectangle((x0, y, x1, plot_bottom), fill=color, outline=INK, width=1)
    draw.text((left + 5, (plot_top + plot_bottom) / 2), y_label, fill="#4b5563", font=_font(16))
    legend_x = left + 120
    for label, color, _ in series:
        draw.rectangle((legend_x, bottom - 22, legend_x + 18, bottom - 4), fill=color, outline=INK)
        draw.text((legend_x + 26, bottom - 26), label, fill=INK, font=_font(16))
        legend_x += 190


def _plot_endpoint(endpoint_rows: list[dict[str, object]], output: Path) -> None:
    colors = {"raw_q4": "#6b7280", "material_only": "#2563eb", "material_render": "#f97316"}
    labels = {"raw_q4": "raw_q4", "material_only": "material_only", "material_render": "material_render"}
    assets = [asset for _, asset in RUNS]
    by_endpoint = {
        endpoint: {row["asset"]: row for row in endpoint_rows if row["endpoint"] == endpoint}
        for endpoint in ENDPOINTS
    }
    series_hdr = [(labels[e], colors[e], [float(by_endpoint[e][a]["audit_mean_hdr_mae"]) for a in assets]) for e in ENDPOINTS]
    series_ssim = [(labels[e], colors[e], [float(by_endpoint[e][a]["audit_display_ssim"]) for a in assets]) for e in ENDPOINTS]
    image, draw = _canvas("C4 render-ablation endpoints", "Raw PCA and two fresh 20k arms; audit = 7 cameras × 6 lights")
    _bar_panel(draw, (55, 185, 890, 700), "42-case audit mean HDR MAE", assets, series_hdr, y_min=0, y_max=0.014, y_label="lower is better")
    _bar_panel(draw, (920, 185, 1750, 700), "42-case audit display SSIM", assets, series_ssim, y_min=0.88, y_max=1.0, y_label="higher is better")
    image.save(output)


def _plot_render_delta(comparison_rows: list[dict[str, object]], output: Path) -> None:
    rows = [row for row in comparison_rows if row["isolates_render_supervision"]]
    assets = [asset for _, asset in RUNS]
    by_asset = {row["asset"]: row for row in rows}
    image, draw = _canvas("The paired delta isolates render supervision", "material_render minus material_only; negative HDR MAE delta is better")
    _bar_panel(
        draw, (55, 185, 890, 700), "Audit HDR MAE delta", assets,
        [("paired delta (%)", GRAY, [float(by_asset[a]["audit_mean_hdr_mae_delta_pct"]) for a in assets])],
        y_min=-18, y_max=12, y_label="percent",
    )
    _bar_panel(
        draw, (920, 185, 1750, 700), "Train HDR MAE delta", assets,
        [("paired delta (%)", BLUE, [float(by_asset[a]["train_mean_hdr_mae_delta_pct"]) for a in assets])],
        y_min=-70, y_max=5, y_label="percent",
    )
    image.save(output)


def _endpoint_montage(run_root: Path, output: Path) -> None:
    sources = [Image.open(run_root / job / f"{asset}-summary" / "endpoint_technical.png").convert("RGB") for job, asset in RUNS]
    width = sum(image.width for image in sources)
    height = max(image.height for image in sources)
    montage = Image.new("RGB", (width, height), "#111216")
    x = 0
    for source in sources:
        montage.paste(source, (x, 0))
        x += source.width
    montage.save(output)


def _plot_trajectory(trajectory_rows: list[dict[str, object]], output: Path) -> None:
    image, draw = _canvas("Five-node BaseColor trajectory", "Lantern remains steeply improving at 20k; other assets are comparatively flat")
    colors = {"material_only": "#2563eb", "material_render": "#f97316"}
    panel_width = 540
    for index, (_, asset) in enumerate(RUNS):
        left, top, right, bottom = 60 + index * 580, 195, 60 + index * 580 + panel_width, 675
        draw.text((left, top - 42), asset, fill=INK, font=_font(24, bold=True))
        plot_left, plot_top, plot_right, plot_bottom = left + 80, top, right - 25, bottom - 60
        asset_rows = [row for row in trajectory_rows if row["asset"] == asset]
        y_max = max(float(row["base_color_linear_mae"]) for row in asset_rows) * 1.08
        y_min = 0.0
        for tick in range(6):
            value = y_min + (y_max - y_min) * tick / 5
            y = plot_bottom - (plot_bottom - plot_top) * tick / 5
            draw.line((plot_left, y, plot_right, y), fill=GRID, width=1)
            draw.text((plot_left - 75, y - 9), f"{value:.3f}", fill="#4b5563", font=_font(14))
        draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=INK, width=2)
        draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=INK, width=2)
        for arm in ARMS:
            rows = [row for row in trajectory_rows if row["asset"] == asset and row["arm"] == arm]
            points = []
            for row in rows:
                x = plot_left + (float(row["step"]) - 1000) / 19000 * (plot_right - plot_left)
                y = plot_bottom - (float(row["base_color_linear_mae"]) - y_min) / (y_max - y_min) * (plot_bottom - plot_top)
                points.append((x, y))
            draw.line(points, fill=colors[arm], width=4)
            for x, y in points:
                draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=colors[arm], outline=INK)
        for step in OBSERVATION_STEPS:
            x = plot_left + (step - 1000) / 19000 * (plot_right - plot_left)
            draw.text((x - 13, plot_bottom + 16), f"{step // 1000}k", fill=INK, font=_font(14))
    draw.rectangle((670, 690, 688, 708), fill=BLUE, outline=INK)
    draw.text((697, 686), "material_only", fill=INK, font=_font(16))
    draw.rectangle((870, 690, 888, 708), fill=ORANGE, outline=INK)
    draw.text((897, 686), "material_render", fill=INK, font=_font(16))
    image.save(output)


def _summary(endpoint_rows: list[dict[str, object]], comparison_rows: list[dict[str, object]], case_rows: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {"schema_version": 1, "assets": {}}
    for _, asset in RUNS:
        endpoints = {row["endpoint"]: row for row in endpoint_rows if row["asset"] == asset}
        comparisons = {row["comparison"]: row for row in comparison_rows if row["asset"] == asset}
        cases = [row for row in case_rows if row["asset"] == asset]
        render_comparison = comparisons["material_only_to_material_render"]
        hdr_deltas = [float(row["render_vs_only_hdr_delta"]) for row in cases]
        ssim_deltas = [float(row["render_vs_only_ssim_delta"]) for row in cases]
        summary["assets"][asset] = {
            "endpoints": endpoints,
            "comparisons": comparisons,
            "render_supervision_case_evidence": {
                "audit_case_count": len(cases),
                "hdr_mae_improved_cases": sum(delta < 0 for delta in hdr_deltas),
                "display_ssim_improved_cases": sum(delta > 0 for delta in ssim_deltas),
                "hdr_mae_median_delta": median(hdr_deltas),
                "display_ssim_median_delta": median(ssim_deltas),
            },
            "render_supervision_endpoint_delta": render_comparison,
        }
    return summary


def _markdown(summary: dict[str, object]) -> str:
    assets = summary["assets"]
    lines = [
        "# C4 三模型 20k 可微渲染对照分析",
        "",
        "结论：训练对真正的 PCA 难例 Lantern 有显著恢复，但增益主要来自 20k 材质域优化；加入渲染损失本身没有提高 Lantern 的只读 audit，反而使 audit HDR MAE 回退 8.24%。Corset 的渲染监督近似中性；BoomBox 的 train HDR 大幅改善，但 audit 与材质指标混合，属于明显 Pareto 取舍，不能宣布统一 winner。",
        "",
        "| 模型 | raw→material_only audit HDR | raw→material_render audit HDR | render-only audit HDR | render-only audit SSIM | 42 case HDR wins |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, asset in RUNS:
        a = assets[asset]
        c0 = a["comparisons"]["raw_q4_to_material_only"]
        c1 = a["comparisons"]["raw_q4_to_material_render"]
        cr = a["comparisons"]["material_only_to_material_render"]
        ce = a["render_supervision_case_evidence"]
        lines.append(
            f"| {asset} | {c0['audit_mean_hdr_mae_delta_pct']:+.2f}% | {c1['audit_mean_hdr_mae_delta_pct']:+.2f}% | "
            f"{cr['audit_mean_hdr_mae_delta_pct']:+.2f}% | {cr['audit_display_ssim_delta']:+.6f} | "
            f"{ce['hdr_mae_improved_cases']}/42 |"
        )
    lines.extend(
        [
            "",
            "## 解释",
            "",
            "- Lantern：raw q4 的 BaseColor MAE 0.05269 降至 material_only 的 0.01170，audit HDR MAE 降低 63.69%，是三模型中最明确的 PCA 难例恢复。material_render 在训练相机上更好，但对 42 个 audit case 仅 1 个 HDR case 更好，说明附加渲染项在当前固定 rig/权重下过拟合训练视图或改变了 Pareto 分配。",
            "- Corset：material_only 相对 raw 的 audit HDR MAE 改善 9.27%；附加渲染项令 train HDR 再改善 31.49%，但 audit HDR MAE回退 0.85%，SSIM微增 0.000393。不能把它描述为 audit render 净胜。",
            "- BoomBox：raw PCA 本来已经较好。material_only 令 audit HDR MAE恶化 12.97%；material_render 把 audit HDR 拉回并略优于 raw 5.20%，但 display SSIM 和多数材质误差仍退化。它证明训练会重分配误差，不证明复杂模型都受益。",
            "",
            "## 实验边界",
            "",
            "- 每模型单 seed、两臂确定性配对；可做案例级因果归因，但不声称跨 seed 方差或统计显著性。",
            "- audit 的 7 个相机 × 6 个灯光共 42 case 从未反向传播；train 为 24 × 6 共 144 case。",
            "- Lantern 排除了约 3.27% emissive 区域，结论只覆盖 Core-4 非自发光材质。",
            "- 20k 不是统一收敛点：Lantern 两臂的 BaseColor MAE 在 15k→20k 仍下降约 34%，若继续训练必须作为新的步数消融，而不能混入本实验。",
            "",
            "## 建议",
            "",
            "保留 Lantern/material_only@20k 为主要成功案例；Corset/material_only@20k 为温和恢复案例；BoomBox/raw_q4 或 material_render@20k 作为低难度/Pareto 对照。若下一轮专门研究“渲染监督本身”，优先在 Lantern 上调低渲染权重或扩大训练相机覆盖，并至少增加 3 个 seed；不要沿用当前权重直接加步数后宣称渲染监督有效。",
        ]
    )
    return "\n".join(lines) + "\n"


def _artifact(
    endpoint_rows: list[dict[str, object]],
    comparison_rows: list[dict[str, object]],
    trajectory_rows: list[dict[str, object]],
    summary: dict[str, object],
) -> dict[str, object]:
    report_path = "outputs/analysis/c4-render-ablation-20k-v1-job-37489/report/analysis_summary.json"
    source = {
        "id": "verified-results",
        "label": "SCOW C4 render-ablation verified local archive",
        "path": report_path,
    }
    headline_rows = []
    for _, asset in RUNS:
        comparisons = {
            row["comparison"]: row
            for row in comparison_rows
            if row["asset"] == asset
        }
        headline_rows.append(
            {
                "asset": asset,
                "raw_to_material_only_audit_hdr_pct": comparisons["raw_q4_to_material_only"]["audit_mean_hdr_mae_delta_pct"] / 100.0,
                "render_only_audit_hdr_pct": comparisons["material_only_to_material_render"]["audit_mean_hdr_mae_delta_pct"] / 100.0,
                "render_only_train_hdr_pct": comparisons["material_only_to_material_render"]["train_mean_hdr_mae_delta_pct"] / 100.0,
            }
        )
    render_delta = [
        {
            **row,
            "audit_mean_hdr_mae_delta_rate": row["audit_mean_hdr_mae_delta_pct"] / 100.0,
            "train_mean_hdr_mae_delta_rate": row["train_mean_hdr_mae_delta_pct"] / 100.0,
            "base_color_mae_delta_rate": row["base_color_mae_delta_pct"] / 100.0,
            "seven_channel_mae_delta_rate": row["seven_channel_mae_delta_pct"] / 100.0,
        }
        for row in comparison_rows
        if row["isolates_render_supervision"]
    ]
    trajectory = [
        {**row, "step_k": int(row["step"]) / 1000, "series": f"{row['asset']} · {row['arm']}"}
        for row in trajectory_rows
    ]

    def sql_literal(value: object) -> str:
        if value is None:
            return "NULL"
        if isinstance(value, bool):
            return "1" if value else "0"
        if isinstance(value, (int, float)):
            return repr(value)
        return "'" + str(value).replace("'", "''") + "'"

    def literal_query(rows: list[dict[str, object]], fields: list[str]) -> str:
        selects = []
        for index, row in enumerate(rows):
            values = []
            for field in fields:
                value = sql_literal(row[field])
                values.append(f"{value} AS \"{field}\"" if index == 0 else value)
            selects.append("SELECT " + ", ".join(values))
        query = "\nUNION ALL\n".join(selects)
        with sqlite3.connect(":memory:") as connection:
            result = connection.execute(query).fetchall()
        if len(result) != len(rows):
            raise ValueError("literal source SQL did not reproduce the reviewed row count")
        return query

    endpoint_fields = [
        "asset", "endpoint", "audit_mean_hdr_mae", "audit_display_ssim",
        "audit_worst_hdr_mae", "train_mean_hdr_mae", "base_color_linear_mae",
        "seven_channel_mae", "chroma_magnitude_retention",
    ]
    render_fields = [
        "asset", "comparison", "audit_mean_hdr_mae_delta_rate",
        "audit_display_ssim_delta", "audit_worst_hdr_mae_delta_pct",
        "train_mean_hdr_mae_delta_rate", "base_color_mae_delta_rate",
        "seven_channel_mae_delta_rate", "normal_degrees_delta_pct",
        "chroma_retention_delta",
    ]
    trajectory_fields = [
        "asset", "arm", "series", "step_k", "base_color_linear_mae",
        "seven_channel_mae", "normal_mean_degrees", "roughness_mae",
        "metallic_mae", "chroma_magnitude_retention", "oklab_delta_e_mean",
        "opponent_error",
    ]

    def chart_source(label: str, query: str, description: str) -> dict[str, object]:
        return {
            "label": label,
            "path": report_path,
            "query": {
                "engine": "SQLite",
                "language": "sql",
                "sql": query,
                "description": description,
                "filters": ["reviewed local rows only; no external database accessed"],
                "tables_used": ["inline verified result rows"],
            },
        }

    endpoint_sql = literal_query(endpoint_rows, endpoint_fields)
    render_sql = literal_query(render_delta, render_fields)
    trajectory_sql = literal_query(trajectory, trajectory_fields)
    generated_at = datetime.now().astimezone().isoformat(timespec="seconds")
    manifest = {
        "version": 1,
        "surface": "report",
        "title": "C4 三模型 20k 可微渲染对照分析",
        "description": "确定性配对案例：raw_q4、material_only 与 material_render 的 audit、材质和轨迹比较。",
        "generatedAt": generated_at,
        "sources": [source],
        "charts": [
            {
                "id": "endpoint-hdr",
                "title": "三端点的 42-case audit HDR MAE",
                "subtitle": "Lantern 的主要改善来自材质域训练；BoomBox 的 material_only 低于 raw 基线。",
                "type": "bar",
                "dataset": "endpoints",
                "source": chart_source(
                    "Verified endpoint rows",
                    endpoint_sql,
                    "Executed SQLite literal query reproducing the nine reviewed endpoint rows from the verified archive.",
                ),
                "encodings": {
                    "x": {"field": "asset", "type": "nominal", "label": "模型"},
                    "y": {"field": "audit_mean_hdr_mae", "type": "quantitative", "label": "masked linear HDR MAE", "format": "number"},
                    "color": {"field": "endpoint", "type": "nominal", "label": "端点"},
                    "tooltip": [
                        {"field": "audit_display_ssim", "type": "quantitative", "label": "display SSIM", "format": "number"},
                        {"field": "audit_worst_hdr_mae", "type": "quantitative", "label": "worst HDR MAE", "format": "number"},
                        {"field": "base_color_linear_mae", "type": "quantitative", "label": "BaseColor MAE", "format": "number"},
                    ],
                },
                "layout": "full",
                "maxRows": 9,
            },
            {
                "id": "render-delta",
                "title": "渲染监督本身对 audit HDR MAE 的贡献",
                "subtitle": "material_render 相对 material_only；负值改善。仅 BoomBox 的均值改善明显，Lantern 回退。",
                "type": "bar",
                "dataset": "render_delta",
                "source": chart_source(
                    "Verified paired render-supervision deltas",
                    render_sql,
                    "Executed SQLite literal query reproducing the three reviewed material_only-to-material_render rows.",
                ),
                "encodings": {
                    "x": {"field": "asset", "type": "nominal", "label": "模型"},
                    "y": {"field": "audit_mean_hdr_mae_delta_rate", "type": "quantitative", "label": "audit HDR MAE 变化", "format": "percent"},
                    "tooltip": [
                        {"field": "train_mean_hdr_mae_delta_rate", "type": "quantitative", "label": "train HDR MAE 变化", "format": "percent"},
                        {"field": "base_color_mae_delta_rate", "type": "quantitative", "label": "BaseColor MAE 变化", "format": "percent"},
                    ],
                },
                "referenceLines": [{"value": 0, "label": "no change"}],
                "layout": "full",
                "maxRows": 3,
            },
            {
                "id": "basecolor-trajectory",
                "title": "五个观察节点的 BaseColor MAE",
                "subtitle": "Lantern 到 20k 仍持续快速下降；Corset 和 BoomBox 已相对平缓。",
                "type": "line",
                "dataset": "trajectory",
                "source": chart_source(
                    "Verified five-node material trajectories",
                    trajectory_sql,
                    "Executed SQLite literal query reproducing the thirty reviewed observation rows.",
                ),
                "encodings": {
                    "x": {"field": "step_k", "type": "quantitative", "label": "训练步数", "unit": "k"},
                    "y": {"field": "base_color_linear_mae", "type": "quantitative", "label": "BaseColor linear MAE", "format": "number"},
                    "color": {"field": "series", "type": "nominal", "label": "模型 · 训练臂"},
                    "tooltip": [
                        {"field": "seven_channel_mae", "type": "quantitative", "label": "seven-channel MAE", "format": "number"},
                        {"field": "chroma_magnitude_retention", "type": "quantitative", "label": "chroma retention", "format": "number"},
                    ],
                },
                "layout": "full",
                "maxRows": 30,
            },
        ],
        "tables": [
            {
                "id": "render-pareto",
                "title": "material_only → material_render 配对差值",
                "subtitle": "只有这一差值隔离了附加渲染监督；负的误差变化为改善。",
                "dataset": "render_delta",
                "source": chart_source(
                    "Verified paired render-supervision deltas",
                    render_sql,
                    "Executed SQLite literal query reproducing the three reviewed material_only-to-material_render rows.",
                ),
                "defaultSort": {"field": "audit_mean_hdr_mae_delta_rate", "direction": "asc"},
                "density": "dense",
                "layout": "full",
                "columns": [
                    {"field": "asset", "label": "模型", "type": "text"},
                    {"field": "audit_mean_hdr_mae_delta_rate", "label": "audit HDR", "format": "percent", "movement": True},
                    {"field": "audit_display_ssim_delta", "label": "audit SSIM Δ", "format": "number", "movement": True},
                    {"field": "train_mean_hdr_mae_delta_rate", "label": "train HDR", "format": "percent", "movement": True},
                    {"field": "base_color_mae_delta_rate", "label": "BaseColor", "format": "percent", "movement": True},
                    {"field": "seven_channel_mae_delta_rate", "label": "7-channel", "format": "percent", "movement": True},
                ],
            }
        ],
        "blocks": [
            {"id": "title", "type": "markdown", "body": "# C4 三模型 20k 可微渲染对照分析", "layout": "full"},
            {
                "id": "technical-summary",
                "type": "markdown",
                "sourceId": "verified-results",
                "layout": "full",
                "body": "## 技术摘要\n\n**PCA 难例可以恢复，但这批结果没有证明当前渲染损失优于相同 20k 的材质监督。** Lantern 的 material-only 把 audit HDR MAE 相对 raw q4 降低 **63.69%**，BaseColor MAE 降低 **77.79%**，是明确成功案例；然而加入渲染项后，其 audit HDR MAE 相对 material-only 回退 **8.24%**，42 个 audit case 仅 **1 个**更好。Corset 的渲染项近似中性；BoomBox 的 train HDR 大幅改善，但 audit SSIM 与材质指标形成取舍。",
            },
            {"id": "endpoint-heading", "type": "markdown", "body": "## Lantern 是最清晰的 PCA 恢复案例", "layout": "full"},
            {"id": "endpoint-chart", "type": "chart", "chartId": "endpoint-hdr", "layout": "full"},
            {
                "id": "endpoint-interpretation",
                "type": "markdown",
                "sourceId": "verified-results",
                "layout": "full",
                "body": "Lantern 的 raw q4 明显抹平木纹、石材和局部材质；两条训练臂都恢复了大部分结构。Corset 有温和恢复。BoomBox 原始 PCA 已经较好，material-only 反而令 audit HDR MAE 上升 **12.97%**，说明训练并非对所有模型单调有益。",
            },
            {"id": "render-heading", "type": "markdown", "body": "## 渲染监督改善 train，但没有形成跨模型 audit 净胜", "layout": "full"},
            {"id": "render-chart", "type": "chart", "chartId": "render-delta", "layout": "full"},
            {"id": "render-table", "type": "table", "tableId": "render-pareto", "layout": "full"},
            {
                "id": "case-interpretation",
                "type": "markdown",
                "sourceId": "verified-results",
                "layout": "full",
                "body": "Corset 的 render 臂在 42 个 audit case 中有 **33 个 HDR MAE** 和 **34 个 SSIM** case 改善，但少数极端 case 使均值 HDR MAE 回退 0.85%、worst 回退 30.14%。BoomBox 只有 20/42 HDR case 改善，均值收益由少数大幅改善 case 驱动。Lantern 则是 1/42 HDR case 改善，结论最稳定地偏向 material-only。",
            },
            {"id": "trajectory-heading", "type": "markdown", "body": "## 20k 足以比较，但 Lantern 尚未完全平台化", "layout": "full"},
            {"id": "trajectory-chart", "type": "chart", "chartId": "basecolor-trajectory", "layout": "full"},
            {
                "id": "scope",
                "type": "markdown",
                "sourceId": "verified-results",
                "layout": "full",
                "body": "## 范围、方法与指标\n\n三个模型均从同一部署式 raw_q4 定义 fresh 启动，两臂共同优化 RGBA latent 与单个 4→7 affine，各跑 20,000 步。训练相机为 24×6 灯光共 144 case；audit 为独立的 7×6 共 42 case，未参与反向。主指标为 audit mean masked linear HDR MAE 与 foreground display SSIM。`material_only→material_render` 是唯一用于判断渲染监督贡献的差值。",
            },
            {
                "id": "limitations",
                "type": "markdown",
                "sourceId": "verified-results",
                "layout": "full",
                "body": "## 限制与稳健性\n\n这是单 seed 的确定性配对案例，不能声称方差或统计显著性。Lantern 约 **3.27%** emissive 区域被参考、PCA 与两训练臂共同排除，因此结论只覆盖 Core-4 非自发光材质。HDR mean、worst、SSIM 和材质误差并不总同向，应按 Pareto 结果陈述，不能用单一加权总分强选 winner。",
            },
            {
                "id": "recommendations",
                "type": "markdown",
                "sourceId": "verified-results",
                "layout": "full",
                "body": "## 推荐的汇报口径与下一步\n\n1. 把 **Lantern/material_only@20k** 作为 PCA 难例恢复主案例；把 Corset/material_only 作为温和恢复，把 BoomBox 作为低难度/Pareto 对照。\n2. 当前实验应报告为“可微优化有效，但当前附加渲染监督没有稳定 audit 增益”，不能把 material_render 宣布为统一 winner。\n3. 若继续研究渲染监督，优先在 Lantern 上降低渲染权重或扩大训练相机覆盖，并至少增加 3 个 seed；继续加步数必须建立新 lineage。",
            },
            {"id": "questions", "type": "markdown", "body": "## 待回答问题\n\n较弱的渲染权重能否保留 Lantern 的材质恢复同时避免 audit 回退？BoomBox 的均值收益是否在多 seed 下仍由少数极端 case 驱动？课程最终呈现更看重平均观感、worst-case 稳定性，还是材质贴图误差？", "layout": "full"},
        ],
    }
    return {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "status": "ready",
            "generatedAt": generated_at,
            "datasets": {
                "headline": headline_rows,
                "endpoints": endpoint_rows,
                "render_delta": render_delta,
                "trajectory": trajectory,
            },
        },
        "sources": [source],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extracted-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    run_root = args.extracted_root.resolve() / "outputs/remote/c4-render-ablation-20k-v1"
    output = args.output_root.resolve()
    figures = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    endpoint_rows, comparison_rows = _endpoint_rows(run_root)
    trajectory_rows = _trajectory_rows(run_root)
    case_rows = _case_rows(run_root)
    summary = _summary(endpoint_rows, comparison_rows, case_rows)
    _write_csv(output / "endpoint_metrics.csv", endpoint_rows)
    _write_csv(output / "paired_comparisons.csv", comparison_rows)
    _write_csv(output / "observation_trajectory.csv", trajectory_rows)
    _write_csv(output / "audit_case_pairs.csv", case_rows)
    (output / "analysis_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "analysis_report.md").write_text(_markdown(summary), encoding="utf-8")
    (output / "artifact.json").write_text(
        json.dumps(_artifact(endpoint_rows, comparison_rows, trajectory_rows, summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _plot_endpoint(endpoint_rows, figures / "endpoint_audit_comparison.png")
    _plot_render_delta(comparison_rows, figures / "render_supervision_delta.png")
    _plot_trajectory(trajectory_rows, figures / "basecolor_trajectory.png")
    _endpoint_montage(run_root, figures / "endpoint_technical_montage.png")
    validation = {
        "schema_version": 1,
        "status": "analysis_outputs_valid",
        "all_numeric_values_finite": _finite_numbers(summary),
        "endpoint_rows": len(endpoint_rows),
        "paired_comparison_rows": len(comparison_rows),
        "observation_rows": len(trajectory_rows),
        "audit_case_rows": len(case_rows),
        "assets": [asset for _, asset in RUNS],
        "arms": list(ARMS),
        "observation_steps": list(OBSERVATION_STEPS),
        "scientific_readiness": "share_with_caveats",
        "caveats": [
            "single deterministic seed; no variance or significance claim",
            "Lantern excludes approximately 3.27% emissive area",
            "Lantern material trajectories were still improving at 20k",
        ],
    }
    if not validation["all_numeric_values_finite"]:
        raise ValueError("analysis summary contains a non-finite numeric value")
    (output / "analysis_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_report_manifest(output)
    print(json.dumps({"status": "analysis_complete", "output_root": str(output)}, sort_keys=True))


if __name__ == "__main__":
    main()
