"""Validate and summarize the balanced RGBA8/BC7 Lantern 160k UE timing pair."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median


REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = REPO_ROOT / "outputs/analysis/ue-bc-latent-feasibility-v1"
CONFIG_PATH = REPO_ROOT / "configs/eval/ue_bc_latent_visual_v1.json"
EXPECTED_CONFIG_SHA256 = "d7ca91b31bc9d66899e8abfd709bbef7a5252ca165ee8f3f3c99afc92f998071"
VARIANTS = ["lantern_material_render_160k_rgba8", "lantern_material_render_160k_bc7"]
WARMUP_SECONDS = 30.0
WINDOW_SECONDS = 3.0
WINDOWS_PER_PASS = 5


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


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * probability
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def distribution(values: list[float]) -> dict[str, float]:
    q1 = quantile(values, 0.25)
    q3 = quantile(values, 0.75)
    return {
        "median": median(values),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "p95": quantile(values, 0.95),
    }


config_hash = sha256(CONFIG_PATH)
if config_hash != EXPECTED_CONFIG_SHA256:
    raise RuntimeError(f"visual config hash mismatch: {config_hash}")

run_records: dict[tuple[int, str], dict] = {}
for pass_number in (1, 2):
    pass_root = EVIDENCE_ROOT / f"raw/timing/pass{pass_number}"
    report = read_json(pass_root / "run_report.json")
    expected = VARIANTS if pass_number == 1 else list(reversed(VARIANTS))
    actual = [row["variant_id"] for row in sorted(report["results"], key=lambda row: row["order_index"])]
    if report.get("status") != "complete" or actual != expected:
        raise RuntimeError(f"timing pass order mismatch: pass {pass_number}")
    for row in report["results"]:
        if int(row["exit_code"]) != 0 or row["visual_config_sha256"] != config_hash:
            raise RuntimeError(f"invalid run record: pass {pass_number} {row['variant_id']}")
        run_records[(pass_number, row["variant_id"])] = row

window_rows: list[dict] = []
trace_validation: list[dict] = []
for pass_number in (1, 2):
    pass_root = EVIDENCE_ROOT / f"raw/timing/pass{pass_number}"
    for variant_id in VARIANTS:
        events_path = pass_root / f"{variant_id}.gpu_events.csv"
        log_path = pass_root / f"{variant_id}.log"
        frames: list[dict] = []
        basepasses: list[dict] = []
        with events_path.open("r", encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                event = {
                    "start": float(row["StartTime"]),
                    "end": float(row["EndTime"]),
                    "duration_ms": float(row["Duration"]) * 1000.0,
                }
                if row["ThreadName"] != "GPU0-Graphics0":
                    raise RuntimeError(f"non-primary GPU thread: {events_path}")
                if row["TimerName"] == "Frame":
                    frames.append(event)
                elif row["TimerName"] == "BasePass":
                    basepasses.append(event)
                else:
                    raise RuntimeError(f"unexpected GPU timer: {row['TimerName']}")
        if not frames or len(frames) != len(basepasses):
            raise RuntimeError(f"Frame/BasePass pairing mismatch: pass {pass_number} {variant_id}")
        measurement_start = frames[0]["start"] + WARMUP_SECONDS
        measurement_end = measurement_start + WINDOWS_PER_PASS * WINDOW_SECONDS
        if frames[-1]["end"] < measurement_end:
            raise RuntimeError(f"insufficient trace duration: pass {pass_number} {variant_id}")
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        checks = {
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
        }
        if not all(checks.values()):
            raise RuntimeError(f"timing log invariant failed: pass {pass_number} {variant_id}: {checks}")
        trace_validation.append({
            "pass": pass_number,
            "variant_id": variant_id,
            "frame_events": len(frames),
            "basepass_events": len(basepasses),
            "measurement_start": measurement_start,
            "measurement_end": measurement_end,
            "log_checks": checks,
        })
        for window_index in range(WINDOWS_PER_PASS):
            start = measurement_start + window_index * WINDOW_SECONDS
            end = start + WINDOW_SECONDS
            window_frames = [event["duration_ms"] for event in frames if start <= event["start"] < end]
            window_basepasses = [event["duration_ms"] for event in basepasses if start <= event["start"] < end]
            if len(window_frames) < 30 or len(window_basepasses) < 30:
                raise RuntimeError(f"too few events: pass {pass_number} {variant_id} window {window_index + 1}")
            frame_stats = distribution(window_frames)
            basepass_stats = distribution(window_basepasses)
            run = run_records[(pass_number, variant_id)]
            window_rows.append({
                "variant_id": variant_id,
                "latent_format": "BC7" if variant_id.endswith("_bc7") else "RGBA8",
                "pass": pass_number,
                "pass_order_index": int(run["order_index"]),
                "window_index": window_index + 1,
                "gpu_frame_event_count": len(window_frames),
                "basepass_event_count": len(window_basepasses),
                "total_gpu_median_ms": round(frame_stats["median"], 7),
                "total_gpu_iqr_ms": round(frame_stats["iqr"], 7),
                "basepass_median_ms": round(basepass_stats["median"], 7),
                "basepass_iqr_ms": round(basepass_stats["iqr"], 7),
                "warmup_seconds": WARMUP_SECONDS,
                "window_seconds": WINDOW_SECONDS,
                "trace_sha256": run["trace_sha256"],
                "visual_config_sha256": config_hash,
            })

by_variant: dict[str, list[dict]] = defaultdict(list)
for row in window_rows:
    by_variant[row["variant_id"]].append(row)
summary_rows: list[dict] = []
for variant_id in VARIANTS:
    rows = by_variant[variant_id]
    if len(rows) != 10:
        raise RuntimeError(f"expected 10 windows: {variant_id}")
    total = distribution([float(row["total_gpu_median_ms"]) for row in rows])
    base = distribution([float(row["basepass_median_ms"]) for row in rows])
    summary_rows.append({
        "variant_id": variant_id,
        "latent_format": "BC7" if variant_id.endswith("_bc7") else "RGBA8",
        "n_windows": 10,
        "total_gpu_median_ms": round(total["median"], 7),
        "total_gpu_q1_ms": round(total["q1"], 7),
        "total_gpu_q3_ms": round(total["q3"], 7),
        "total_gpu_iqr_ms": round(total["iqr"], 7),
        "total_gpu_p95_ms": round(total["p95"], 7),
        "basepass_median_ms": round(base["median"], 7),
        "basepass_q1_ms": round(base["q1"], 7),
        "basepass_q3_ms": round(base["q3"], 7),
        "basepass_iqr_ms": round(base["iqr"], 7),
        "basepass_p95_ms": round(base["p95"], 7),
        "aggregation": "median/IQR/P95 across 10 per-window medians",
    })

by_format = {row["latent_format"]: row for row in summary_rows}
rgba8 = by_format["RGBA8"]
bc7 = by_format["BC7"]
basepass_delta = bc7["basepass_median_ms"] - rgba8["basepass_median_ms"]
total_delta = bc7["total_gpu_median_ms"] - rgba8["total_gpu_median_ms"]
summary = {
    "schema_version": 1,
    "status": "complete",
    "visual_config_sha256": config_hash,
    "scope": "representative Lantern material-render 160k RGBA8 versus BC7 latent",
    "protocol": {
        "passes": 2,
        "ordering": "forward then reverse",
        "warmup_seconds_after_first_gpu_frame": WARMUP_SECONDS,
        "windows_per_pass": WINDOWS_PER_PASS,
        "window_seconds": WINDOW_SECONDS,
        "resolution": [1920, 1080],
        "rhi": "D3D12",
        "material_quality": "High",
        "scalability": "Epic",
    },
    "variants": summary_rows,
    "bc7_minus_rgba8": {
        "total_gpu_median_ms": round(total_delta, 7),
        "total_gpu_percent": round(total_delta / rgba8["total_gpu_median_ms"] * 100.0, 4),
        "basepass_median_ms": round(basepass_delta, 7),
        "basepass_percent": round(basepass_delta / rgba8["basepass_median_ms"] * 100.0, 4),
    },
    "interpretation_limit": (
        "This is a same-scene distribution comparison, not proof that a sub-IQR delta is causal. "
        "BC7 changes resource format while the affine decoder shader structure is unchanged."
    ),
    "trace_validation": trace_validation,
}

write_csv(EVIDENCE_ROOT / "gpu_timing.csv", window_rows)
write_csv(EVIDENCE_ROOT / "gpu_timing_summary.csv", summary_rows)
write_json(EVIDENCE_ROOT / "timing_summary.json", summary)
print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
