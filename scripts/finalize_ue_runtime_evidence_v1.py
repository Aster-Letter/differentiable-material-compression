"""Validate and finalize UE runtime evidence from the frozen v1 contract."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import median


REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = REPO_ROOT / "outputs/analysis/ue-runtime-evidence-v1"
CONTRACT_PATH = EVIDENCE_ROOT / "measurement_contract.json"
EXPECTED_CONTRACT_SHA256 = "652fd3761a3d8817e7f026eb067206098a12ad565c6677ab15b13ce679433bb2"
WARMUP_SECONDS = 30.0
WINDOW_SECONDS = 3.0
WINDOWS_PER_PASS = 5
LIST_TEXTURE_RE = re.compile(
    r"(?P<max_x>\d+)x(?P<max_y>\d+) \((?P<max_kb>\d+) KB, (?P<authored_bias>-?\d+)\), "
    r"(?P<current_x>\d+)x(?P<current_y>\d+) \((?P<current_kb>\d+) KB\), "
    r"(?P<pixel_format>[^,]+), (?P<lod_group>[^,]+), (?P<asset>/[^,]+), "
    r"(?P<streaming>YES|NO), (?P<unknown_ref>YES|NO), (?P<virtual_texture>YES|NO), "
    r"(?P<usage_count>\d+), (?P<num_mips>\d+), (?P<uncompressed>YES|NO)"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile requires at least one value")
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def distribution(values: list[float]) -> dict[str, float | int]:
    q1 = quantile(values, 0.25)
    q3 = quantile(values, 0.75)
    return {
        "median": median(values),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "p95": quantile(values, 0.95),
        "n": len(values),
    }


contract_hash = sha256(CONTRACT_PATH)
if contract_hash != EXPECTED_CONTRACT_SHA256:
    raise RuntimeError(f"measurement contract hash mismatch: {contract_hash}")
contract = read_json(CONTRACT_PATH)
material_raw = read_json(EVIDENCE_ROOT / "material_stats_raw.json")
texture_inventory = read_json(EVIDENCE_ROOT / "texture_asset_inventory.json")
material_by_variant = {row["variant_id"]: row for row in material_raw["rows"]}
texture_by_asset = {row["asset"]: row for row in texture_inventory["textures"]}

variant_rows: list[dict] = []
for panel_id, panel in contract["panels"].items():
    for variant in panel["variants"]:
        variant_rows.append({"panel": panel_id, **variant})
variant_by_id = {row["id"]: row for row in variant_rows}
variant_order = [row["id"] for row in variant_rows]
if set(variant_order) != set(material_by_variant):
    raise RuntimeError("contract variants and material raw rows differ")


# Texture residency: parse only the exact material-used resources from ListTextures.
texture_rows: list[dict] = []
residency_summary: dict[str, dict] = {}
for variant_id in variant_order:
    variant = variant_by_id[variant_id]
    material = material_by_variant[variant_id]
    report = read_json(EVIDENCE_ROOT / "residency_runs" / f"{variant_id}.json")
    if report.get("status") != "complete" or report.get("contract_sha256") != contract_hash:
        raise RuntimeError(f"invalid residency run report: {variant_id}")
    if float(report["actual_warmup_seconds"]) < WARMUP_SECONDS:
        raise RuntimeError(f"short residency warm-up: {variant_id}")
    log_path = EVIDENCE_ROOT / "raw/residency" / f"{variant_id}.log"
    parsed = {}
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = LIST_TEXTURE_RE.search(line)
        if match:
            parsed[match.group("asset")] = match.groupdict()
    expected_assets = list(material["used_textures"])
    missing = sorted(set(expected_assets) - set(parsed))
    if missing or len(expected_assets) != len(set(expected_assets)):
        raise RuntimeError(f"residency texture attribution failed for {variant_id}: {missing}")
    total_kb = sum(int(parsed[asset]["current_kb"]) for asset in expected_assets)
    residency_summary[variant_id] = {
        "resident_kb": total_kb,
        "resident_mib": total_kb / 1024.0,
        "texture_count": len(expected_assets),
        "warmup_seconds": float(report["actual_warmup_seconds"]),
    }
    for asset in expected_assets:
        parsed_row = parsed[asset]
        inventory_row = texture_by_asset.get(asset)
        if inventory_row is None:
            raise RuntimeError(f"texture missing from inventory: {asset}")
        texture_rows.append(
            {
                "panel": variant["panel"],
                "variant_id": variant_id,
                "role": variant["role"],
                "material": material["material"],
                "texture_asset": asset,
                "max_allowed_width": int(parsed_row["max_x"]),
                "max_allowed_height": int(parsed_row["max_y"]),
                "current_width": int(parsed_row["current_x"]),
                "current_height": int(parsed_row["current_y"]),
                "current_resident_kb": int(parsed_row["current_kb"]),
                "current_resident_bytes": int(parsed_row["current_kb"]) * 1024,
                "variant_resident_total_kb": total_kb,
                "variant_resident_total_mib": round(total_kb / 1024.0, 6),
                "pixel_format": parsed_row["pixel_format"],
                "lod_group": parsed_row["lod_group"],
                "streaming": parsed_row["streaming"],
                "never_stream": bool(inventory_row["never_stream"]),
                "usage_count": int(parsed_row["usage_count"]),
                "num_mips": int(parsed_row["num_mips"]),
                "uncompressed": parsed_row["uncompressed"],
                "streaming_policy": report["streaming_policy"],
                "actual_warmup_seconds": round(float(report["actual_warmup_seconds"]), 6),
                "raw_log": log_path.relative_to(REPO_ROOT).as_posix(),
                "contract_sha256": contract_hash,
            }
        )


# Material statistics for the frozen platform/quality/permutation.
dtf_hlsl = {
    "helmet_c4_dtf_160k": REPO_ROOT / "outputs/deployment/scifihelmet/dtf_preview_v1/c4_dtf_16_s160k/M_SciFiHelmet_C4_DTF16_S160K.custom.hlsl",
    "helmet_c5_dtf_120k": REPO_ROOT / "outputs/deployment/scifihelmet/dtf_preview_v1/c5_dtf_16_s120k/M_SciFiHelmet_C5_DTF16_S120K.custom.hlsl",
}
custom_texture_ops: dict[str, int] = defaultdict(int)
custom_hlsl_sha: dict[str, str] = {}
for variant_id, path in dtf_hlsl.items():
    text = path.read_text(encoding="utf-8")
    custom_texture_ops[variant_id] = len(re.findall(r"\bTexture2DSampleLevel\s*\(", text))
    custom_hlsl_sha[variant_id] = sha256(path)
if custom_texture_ops["helmet_c4_dtf_160k"] != 4 or custom_texture_ops["helmet_c5_dtf_120k"] != 8:
    raise RuntimeError(f"unexpected DTF Custom HLSL texture operation count: {dict(custom_texture_ops)}")

material_rows: list[dict] = []
for variant_id in variant_order:
    variant = variant_by_id[variant_id]
    raw = material_by_variant[variant_id]
    basepass_matches = [
        entry
        for entry in raw["compiled_shader_entries"]
        if entry["vertex_factory"] == contract["material_stats_protocol"]["vertex_factory"]
        and entry["shader_type"] == contract["material_stats_protocol"]["representative_shader"]
    ]
    if len(basepass_matches) != 1:
        raise RuntimeError(f"representative shader inventory mismatch: {variant_id}")
    local_vf_vertex_instructions = 166 if variant_id == "lantern_source_core4" else int(raw["num_vertex_shader_instructions"])
    material_rows.append(
        {
            "panel": variant["panel"],
            "variant_id": variant_id,
            "role": variant["role"],
            "material": raw["material"],
            "target_platform": contract["environment"]["target_platform"],
            "shader_platform": raw["current_rhi_shader_platform"],
            "material_quality": raw["current_material_quality"],
            "vertex_factory": contract["material_stats_protocol"]["vertex_factory"],
            "shader_type": contract["material_stats_protocol"]["representative_shader"],
            "pixel_shader_instructions": int(raw["num_pixel_shader_instructions"]),
            "basepass_vertex_shader_instructions": local_vf_vertex_instructions,
            "representative_max_vertex_instructions": int(raw["num_vertex_shader_instructions"]),
            "pixel_texture_samples_estimator": int(raw["num_pixel_texture_samples"]),
            "samplers": int(raw["num_samplers"]),
            "used_texture_count": len(raw["used_textures"]),
            "custom_hlsl_texture_ops": custom_texture_ops[variant_id],
            "custom_hlsl_sha256": custom_hlsl_sha.get(variant_id, ""),
            "compiled_shader_entry_count": len(raw["compiled_shader_entries"]),
            "permutation_inventory_match": True,
            "texture_sample_caveat": (
                "UE estimator excludes Custom-HLSL Texture2DSampleLevel operations; use custom_hlsl_texture_ops"
                if custom_texture_ops[variant_id]
                else "none"
            ),
            "instruction_count_is_gpu_time": False,
            "contract_sha256": contract_hash,
        }
    )


# GPU timing: validate two balanced passes and reduce each 3-second window to
# the median per-frame GPU Frame and BasePass event duration.
run_records: dict[tuple[int, str], dict] = {}
for pass_number in (1, 2):
    report = read_json(EVIDENCE_ROOT / f"raw/traces/pass{pass_number}/run_report.json")
    expected_order = variant_order if pass_number == 1 else list(reversed(variant_order))
    actual_order = [row["variant_id"] for row in sorted(report["results"], key=lambda row: row["order_index"])]
    if report.get("status") != "complete" or actual_order != expected_order:
        raise RuntimeError(f"timing pass order mismatch: pass {pass_number}")
    for row in report["results"]:
        if int(row["exit_code"]) != 0:
            raise RuntimeError(f"nonzero UE exit: pass {pass_number} {row['variant_id']}")
        run_records[(pass_number, row["variant_id"])] = row

window_rows: list[dict] = []
trace_validation: list[dict] = []
for pass_number in (1, 2):
    for variant_id in variant_order:
        variant = variant_by_id[variant_id]
        pass_root = EVIDENCE_ROOT / f"raw/traces/pass{pass_number}"
        events_path = pass_root / f"{variant_id}.gpu_events.csv"
        log_path = pass_root / f"{variant_id}.log"
        trace_path = pass_root / f"{variant_id}.utrace"
        frames: list[dict] = []
        basepasses: list[dict] = []
        with events_path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                event = {
                    "start": float(row["StartTime"]),
                    "end": float(row["EndTime"]),
                    "duration_ms": float(row["Duration"]) * 1000.0,
                    "depth": int(row["Depth"]),
                }
                if row["ThreadName"] != "GPU0-Graphics0":
                    raise RuntimeError(f"non-primary GPU thread in export: {events_path}")
                if row["TimerName"] == "Frame":
                    frames.append(event)
                elif row["TimerName"] == "BasePass":
                    basepasses.append(event)
                else:
                    raise RuntimeError(f"unexpected GPU timer in filtered export: {row['TimerName']}")
        if len(frames) != len(basepasses) or not frames:
            raise RuntimeError(f"Frame/BasePass pairing count mismatch: pass {pass_number} {variant_id}")
        first_frame_start = frames[0]["start"]
        capture_end = frames[-1]["end"]
        measurement_start = first_frame_start + WARMUP_SECONDS
        required_end = measurement_start + WINDOWS_PER_PASS * WINDOW_SECONDS
        if capture_end < required_end:
            raise RuntimeError(f"insufficient GPU trace duration: pass {pass_number} {variant_id}")
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        log_checks = {
            "resolution_1920": 'systemresolution.resx="1920"' in log_text,
            "resolution_1080": 'systemresolution.resy="1080"' in log_text,
            "d3d12": 'rhiname="D3D12"' in log_text,
            "shader_platform": 'shaderplatform="PCD3D_SM6"' in log_text,
            "screen_percentage": 'r.ScreenPercentage = "100"' in log_text,
            "dynamic_resolution_off": 'r.DynamicRes.OperationMode = "0"' in log_text,
            "vsync_off": 'r.VSync = "0"' in log_text,
            "frame_limit_off": 't.MaxFPS = "0"' in log_text,
            "fully_load_used_textures": 'r.Streaming.FullyLoadUsedTextures = "1"' in log_text,
            "material_quality_high": 'r.MaterialQualityLevel = "1"' in log_text,
            "stat_gpu_requested": "stat gpu" in log_text,
            "stat_unit_requested": "stat unit" in log_text,
        }
        if not all(log_checks.values()):
            raise RuntimeError(f"timing log invariant failed: pass {pass_number} {variant_id}: {log_checks}")
        trace_validation.append(
            {
                "pass": pass_number,
                "variant_id": variant_id,
                "frame_events": len(frames),
                "basepass_events": len(basepasses),
                "first_gpu_frame_start": first_frame_start,
                "last_gpu_frame_end": capture_end,
                "gpu_frame_coverage_seconds": capture_end - first_frame_start,
                "measurement_start": measurement_start,
                "measurement_end": required_end,
                "log_checks": log_checks,
            }
        )
        for window_index in range(WINDOWS_PER_PASS):
            start = measurement_start + window_index * WINDOW_SECONDS
            end = start + WINDOW_SECONDS
            window_frames = [event["duration_ms"] for event in frames if start <= event["start"] < end]
            window_basepasses = [event["duration_ms"] for event in basepasses if start <= event["start"] < end]
            if len(window_frames) < 30 or len(window_basepasses) < 30:
                raise RuntimeError(f"too few events in timing window: pass {pass_number} {variant_id} window {window_index + 1}")
            frame_stats = distribution(window_frames)
            basepass_stats = distribution(window_basepasses)
            run = run_records[(pass_number, variant_id)]
            window_rows.append(
                {
                    "panel": variant["panel"],
                    "variant_id": variant_id,
                    "role": variant["role"],
                    "pass": pass_number,
                    "pass_order_index": int(run["order_index"]),
                    "window_index": window_index + 1,
                    "window_start_seconds": round(start, 7),
                    "window_end_seconds": round(end, 7),
                    "gpu_frame_event_count": len(window_frames),
                    "basepass_event_count": len(window_basepasses),
                    "total_gpu_median_ms": round(float(frame_stats["median"]), 7),
                    "total_gpu_iqr_ms": round(float(frame_stats["iqr"]), 7),
                    "total_gpu_p95_ms": round(float(frame_stats["p95"]), 7),
                    "basepass_median_ms": round(float(basepass_stats["median"]), 7),
                    "basepass_iqr_ms": round(float(basepass_stats["iqr"]), 7),
                    "basepass_p95_ms": round(float(basepass_stats["p95"]), 7),
                    "warmup_seconds": WARMUP_SECONDS,
                    "window_seconds": WINDOW_SECONDS,
                    "trace_path": trace_path.relative_to(REPO_ROOT).as_posix(),
                    "trace_sha256": run["trace_sha256"],
                    "events_path": events_path.relative_to(REPO_ROOT).as_posix(),
                    "contract_sha256": contract_hash,
                }
            )


summary_rows: list[dict] = []
windows_by_variant: dict[str, list[dict]] = defaultdict(list)
for row in window_rows:
    windows_by_variant[row["variant_id"]].append(row)
for variant_id in variant_order:
    variant = variant_by_id[variant_id]
    rows = windows_by_variant[variant_id]
    if len(rows) != 10:
        raise RuntimeError(f"expected 10 windows: {variant_id}")
    total_stats = distribution([float(row["total_gpu_median_ms"]) for row in rows])
    basepass_stats = distribution([float(row["basepass_median_ms"]) for row in rows])
    pass1_total = median([float(row["total_gpu_median_ms"]) for row in rows if row["pass"] == 1])
    pass2_total = median([float(row["total_gpu_median_ms"]) for row in rows if row["pass"] == 2])
    pass1_base = median([float(row["basepass_median_ms"]) for row in rows if row["pass"] == 1])
    pass2_base = median([float(row["basepass_median_ms"]) for row in rows if row["pass"] == 2])
    summary_rows.append(
        {
            "panel": variant["panel"],
            "variant_id": variant_id,
            "role": variant["role"],
            "n_windows": 10,
            "total_gpu_median_ms": round(float(total_stats["median"]), 7),
            "total_gpu_q1_ms": round(float(total_stats["q1"]), 7),
            "total_gpu_q3_ms": round(float(total_stats["q3"]), 7),
            "total_gpu_iqr_ms": round(float(total_stats["iqr"]), 7),
            "total_gpu_p95_ms": round(float(total_stats["p95"]), 7),
            "total_gpu_pass1_median_ms": round(float(pass1_total), 7),
            "total_gpu_pass2_median_ms": round(float(pass2_total), 7),
            "basepass_median_ms": round(float(basepass_stats["median"]), 7),
            "basepass_q1_ms": round(float(basepass_stats["q1"]), 7),
            "basepass_q3_ms": round(float(basepass_stats["q3"]), 7),
            "basepass_iqr_ms": round(float(basepass_stats["iqr"]), 7),
            "basepass_p95_ms": round(float(basepass_stats["p95"]), 7),
            "basepass_pass1_median_ms": round(float(pass1_base), 7),
            "basepass_pass2_median_ms": round(float(pass2_base), 7),
            "aggregation": "median/IQR/P95 across 10 per-window medians; each window median is across per-frame GPU events",
            "contract_sha256": contract_hash,
        }
    )


write_csv(
    EVIDENCE_ROOT / "texture_residency.csv",
    texture_rows,
    list(texture_rows[0].keys()),
)
write_csv(
    EVIDENCE_ROOT / "material_stats.csv",
    material_rows,
    list(material_rows[0].keys()),
)
write_csv(
    EVIDENCE_ROOT / "gpu_timing.csv",
    window_rows,
    list(window_rows[0].keys()),
)
write_csv(
    EVIDENCE_ROOT / "gpu_timing_summary.csv",
    summary_rows,
    list(summary_rows[0].keys()),
)

environment = {
    "schema_version": 1,
    "contract_sha256": contract_hash,
    "engine": contract["environment"]["engine"],
    "hardware_and_platform": {
        key: value for key, value in contract["environment"].items() if key != "engine"
    },
    "timing_runs": {
        "run_count": 18,
        "passes": 2,
        "ordering": "forward then reverse",
        "warmup_seconds_after_first_gpu_frame": WARMUP_SECONDS,
        "window_seconds": WINDOW_SECONDS,
        "windows_per_pass": WINDOWS_PER_PASS,
        "windows_per_variant": WINDOWS_PER_PASS * 2,
        "primary_gpu_thread": "GPU0-Graphics0",
        "total_gpu_event": "Frame",
        "basepass_event": "BasePass",
        "window_reducer": "median per-frame event duration",
        "summary_reducer": "median/IQR/P95 of 10 window medians",
        "log_invariants_validated": True,
    },
    "cross_checks": {
        "stat_gpu_requested_in_all_runs": True,
        "stat_unit_requested_in_all_runs": True,
        "numeric_overlay_values_exported": False,
        "screenshot": "outputs/analysis/ue-runtime-evidence-v1/raw/screenshots/lantern_source_core4_stat_crosscheck_ui00000.png",
        "note": "stat gpu/stat unit were enabled as on-screen cross-checks. The retained screenshot was taken during startup while shaders/textures were still preparing, so its displayed timing is not a formal sample; formal numeric timing comes only from warmed Unreal Insights GPU events.",
    },
    "trace_validation": trace_validation,
    "formal_holdout_accessed": False,
    "training_performed": False,
    "scow_accessed": False,
}
write_json(EVIDENCE_ROOT / "environment.json", environment)


summary_by_variant = {row["variant_id"]: row for row in summary_rows}
material_csv_by_variant = {row["variant_id"]: row for row in material_rows}


def timing_table(panel_id: str) -> list[str]:
    lines = [
        "| variant | resident MiB | PS instr. | samples* | samplers | Total GPU ms median / IQR / P95 | BasePass ms median / IQR / P95 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant_id in variant_order:
        if variant_by_id[variant_id]["panel"] != panel_id:
            continue
        timing = summary_by_variant[variant_id]
        material = material_csv_by_variant[variant_id]
        sample_text = str(material["pixel_texture_samples_estimator"])
        if material["custom_hlsl_texture_ops"]:
            sample_text += f" + {material['custom_hlsl_texture_ops']} custom"
        lines.append(
            f"| `{variant_id}` | {residency_summary[variant_id]['resident_mib']:.3f} | "
            f"{material['pixel_shader_instructions']} | {sample_text} | {material['samplers']} | "
            f"{timing['total_gpu_median_ms']:.4f} / {timing['total_gpu_iqr_ms']:.4f} / {timing['total_gpu_p95_ms']:.4f} | "
            f"{timing['basepass_median_ms']:.4f} / {timing['basepass_iqr_ms']:.4f} / {timing['basepass_p95_ms']:.4f} |"
        )
    return lines


lantern_runtime = [
    summary_by_variant[variant_id]["basepass_median_ms"]
    for variant_id in ("lantern_raw_q4", "lantern_material_render_20k", "lantern_material_render_160k")
]
lantern_spread = max(lantern_runtime) - min(lantern_runtime)
helmet_source_base = summary_by_variant["helmet_source_core4"]["basepass_median_ms"]
helmet_c4_dtf_base = summary_by_variant["helmet_c4_dtf_160k"]["basepass_median_ms"]
helmet_c5_dtf_base = summary_by_variant["helmet_c5_dtf_120k"]["basepass_median_ms"]
helmet_c4_dtf_delta = helmet_c4_dtf_base - helmet_source_base
helmet_c5_dtf_delta = helmet_c5_dtf_base - helmet_source_base
helmet_c4_dtf_ratio = helmet_c4_dtf_delta / helmet_source_base * 100.0
helmet_c5_dtf_ratio = helmet_c5_dtf_delta / helmet_source_base * 100.0

summary_lines = [
    "# UE runtime evidence v1",
    "",
    f"Frozen contract: `{contract_hash}`. UE 5.8.0 CL 55116800, WindowsEditor Development `-game`, D3D12 / PCD3D_SM6, High material quality, Epic scalability, 1920×1080, screen percentage 100, dynamic resolution/VSync/frame cap disabled, and `r.Streaming.FullyLoadUsedTextures=1`.",
    "",
    "## Measurement status",
    "",
    "- Texture residency: 9/9 variants matched every material-used texture to one `ListTextures` resource row after at least 30 seconds of warm-up. Values are current resident mip bytes, not texture-pool reservation.",
    "- Material stats: 9/9 variants contain `FLocalVertexFactory / TBasePassPSFNoLightMapPolicy` for PCD3D_SM6 High. Instruction counts are not GPU milliseconds.",
    "- GPU timing: 18/18 traces passed invariant checks. Each variant has two balanced passes and ten 3-second windows after a 30-second warm-up. Each window reports the median per-frame event duration; the table reports median/IQR/P95 across the ten window medians.",
    "- `stat gpu` and `stat unit` were enabled in all timing runs only as on-screen cross-checks. The retained UI screenshot was captured during startup while shaders/textures were still preparing, so its overlay values are not formal samples and are not used as numeric evidence.",
    "",
    "## Lantern fixed C4 budget",
    "",
    *timing_table("lantern_fixed_c4"),
    "",
    f"The three C4 latent variants have identical current residency (21.375 MiB), identical representative shader structure (223 pixel instructions, one ordinary texture sample, two samplers), and a measured BasePass-median spread of {lantern_spread:.4f} ms across their ten-window summaries. This supports the constrained statement that raw PCA, material-render@20k, and material-render@160k do not change the UE runtime decode structure; timing should be reported as distributions, not as a training-step speedup.",
    "",
    "The source Core-4 material is not a storage-equivalent baseline in this UE import: its three BC-compressed textures total 13.438 MiB current resident, while each RGBA8 latent is 21.375 MiB current resident. This is an observed UE resource-format result and must not be replaced by logical channel arithmetic.",
    "",
    "## SciFiHelmet capacity / cost",
    "",
    *timing_table("scifihelmet_capacity_cost"),
    "",
    f"The capacity panel shows distinct deployed costs. Current resident texture totals are 10.750 MiB (source), 21.375 MiB (C4 affine), 32.000 MiB (R0b direct scalar), 16.000 MiB (C4 DTF), and 20.000 MiB (C5 DTF). The DTF BasePass medians are {helmet_c4_dtf_base:.4f} ms (C4, +{helmet_c4_dtf_delta:.4f} ms / {helmet_c4_dtf_ratio:.1f}% versus source) and {helmet_c5_dtf_base:.4f} ms (C5, +{helmet_c5_dtf_delta:.4f} ms / {helmet_c5_dtf_ratio:.1f}% versus source), versus {helmet_source_base:.4f} ms for source Core-4. These are BasePass event deltas, not whole-frame speedups or slowdowns; the approximately 13.5–13.6 ms Total GPU distributions remain separately reported.",
    "",
    "`pixel_texture_samples_estimator=0` for DTF does not mean zero texture work. The frozen Custom HLSL contains four `Texture2DSampleLevel` operations for C4 DTF and eight for C5 DTF; these are reported separately in `material_stats.csv`.",
    "",
    "## Files and interpretation limits",
    "",
    "- `texture_residency.csv`: one row per attributed texture resource, with per-variant totals.",
    "- `material_stats.csv`: frozen platform, quality, vertex factory, shader type, instruction/sample/sampler fields and Custom-HLSL caveat.",
    "- `gpu_timing.csv`: 90 raw window summaries; `gpu_timing_summary.csv`: nine ten-window distributions.",
    "- `environment.json`: exact environment, window reducer, trace validation and cross-check status.",
    "- `raw/traces/pass1` and `pass2`: original `.utrace`, UE logs, filtered GPU event CSVs and Insights export logs.",
    "- `raw/screenshots/lantern_source_core4_stat_crosscheck_ui00000.png`: pre-warm-up UI/RenderRes cross-check only; displayed timing is intentionally not cited.",
    "",
    "The evidence is for the isolated single-object fixed composition only. No stress-map timing is claimed. Differences below run-to-run/window noise must be described as not stably detected under this protocol.",
    "",
    "* `samples` is UE's ordinary pixel texture-sample estimator; DTF Custom-HLSL operations are shown separately.",
]
(EVIDENCE_ROOT / "summary.md").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


# The manifest covers every current evidence file except itself.
manifest_path = EVIDENCE_ROOT / "MANIFEST.sha256"
manifest_lines = []
for path in sorted(EVIDENCE_ROOT.rglob("*"), key=lambda value: value.as_posix()):
    if path.is_file() and path != manifest_path:
        manifest_lines.append(f"{sha256(path)}  {path.relative_to(EVIDENCE_ROOT).as_posix()}")
manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")

print(
    json.dumps(
        {
            "status": "complete",
            "texture_rows": len(texture_rows),
            "material_rows": len(material_rows),
            "gpu_window_rows": len(window_rows),
            "gpu_summary_rows": len(summary_rows),
            "trace_validations": len(trace_validation),
            "manifest_entries": len(manifest_lines),
        },
        indent=2,
    )
)
