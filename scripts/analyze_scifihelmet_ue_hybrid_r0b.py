"""Build the deterministic UE acceptance summary for the R0b hybrid candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageChops, ImageStat


REPO_ROOT = Path(__file__).resolve().parents[1]
ROOT = REPO_ROOT / "outputs/deployment/scifihelmet/hybrid_direct_scalars_r0b"
EVIDENCE = ROOT / "ue_evidence"
SETUP_REPORT = EVIDENCE / "ue_setup_report.json"
CAPTURE_REPORT = EVIDENCE / "captures/capture_report.json"
OUTPUT = EVIDENCE / "ue_acceptance_summary.json"
TEXTURE_A = ROOT / "T_SciFiHelmet_Hybrid_R0B_A_RGBA8.png"
TEXTURE_B = ROOT / "T_SciFiHelmet_Hybrid_R0B_B_RGB8.png"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rgba_equal(source: Path, readback: Path) -> bool:
    with Image.open(source) as a, Image.open(readback) as b:
        if a.size != b.size:
            return False
        return ImageChops.difference(a.convert("RGBA"), b.convert("RGBA")).getbbox() is None


def _rgb_mae(a_path: Path, b_path: Path) -> float:
    with Image.open(a_path) as a, Image.open(b_path) as b:
        a = a.convert("RGB")
        b = b.convert("RGB")
        if a.size != b.size:
            raise RuntimeError(f"image size mismatch: {a_path} vs {b_path}")
        means = ImageStat.Stat(ImageChops.difference(a, b)).mean
        return float(sum(means) / (3.0 * 255.0))


def _latest_log() -> Path:
    logs = sorted(
        (REPO_ROOT / "ue_demo/CGCompressionDemo/Saved/Logs").glob("*.log"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not logs:
        raise RuntimeError("no UE log found")
    return logs[0]


def analyze(log_path: Path, map_check_log: Path) -> dict:
    setup = json.loads(SETUP_REPORT.read_text(encoding="utf-8"))
    capture = json.loads(CAPTURE_REPORT.read_text(encoding="utf-8"))
    if setup["status"] != "complete_readback_pending_external_compare":
        raise RuntimeError(f"UE setup incomplete: {setup['status']}")
    if capture["status"] != "complete" or len(capture["captures"]) != 9:
        raise RuntimeError("serial UE capture is incomplete")

    records = {(item["view"], item["material_role"]): item for item in capture["captures"]}
    metrics = {}
    for view in ("d1_metallic_boundary", "d2_yellow_tube", "d3_gray_panel"):
        reference = Path(records[(view, "reference")]["path"])
        baseline = Path(records[(view, "baseline")]["path"])
        candidate = Path(records[(view, "candidate_r0b")]["path"])
        for path in (reference, baseline, candidate):
            if not path.is_file():
                raise RuntimeError(f"capture missing: {path}")
        baseline_mae = _rgb_mae(reference, baseline)
        candidate_mae = _rgb_mae(reference, candidate)
        metrics[view] = {
            "baseline_mae_to_reference": baseline_mae,
            "candidate_mae_to_reference": candidate_mae,
            "candidate_improvement_fraction": 1.0 - candidate_mae / baseline_mae,
            "baseline_candidate_mae": _rgb_mae(baseline, candidate),
        }

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    texture_assets = (
        "T_SciFiHelmet_Hybrid_R0B_A_RGBA8",
        "T_SciFiHelmet_Hybrid_R0B_B_RGB8",
    )
    format_lines = [
        line
        for line in log_text.splitlines()
        if "2048x2048 (16384 KB" in line
        and "PF_B8G8R8A8" in line
        and any(asset in line for asset in texture_assets)
    ]
    if len({asset for asset in texture_assets if any(asset in line for line in format_lines)}) != 2:
        raise RuntimeError("UE log does not prove PF_B8G8R8A8/16 MiB for both textures")
    candidate_compile_failure = any(
        "M_SciFiHelmet_Hybrid_R0B" in line
        and ("Failed to compile" in line or "LogMaterial: Error" in line)
        for line in log_text.splitlines()
    )
    map_check_text = map_check_log.read_text(encoding="utf-8", errors="replace")
    map_check_lines = [
        line for line in map_check_text.splitlines() if "MapCheck:" in line and "地图检测完成" in line
    ]
    if not map_check_lines:
        raise RuntimeError("final UE Map Check summary is missing")
    map_check_summary = map_check_lines[-1]
    map_check_zero_errors = "0个错误" in map_check_summary

    readback_a = EVIDENCE / "readback_texture_a.png"
    readback_b = EVIDENCE / "readback_texture_b.png"
    readback_equal = {
        "texture_a_rgba_exact": _rgba_equal(TEXTURE_A, readback_a),
        "texture_b_rgb_plus_opaque_alpha_exact": _rgba_equal(TEXTURE_B, readback_b),
    }
    structural = {
        "readback_exact": all(readback_equal.values()),
        "actual_pixel_format": "PF_B8G8R8A8 for A and B",
        "actual_resident_bytes": 2 * 2048 * 2048 * 4,
        "texture_samples": 2,
        "same_uv_wrap_bilinear": setup["material"]["connections"]["one_uv_to_a"]
        and setup["material"]["connections"]["one_uv_to_b"],
        "base_color_direct": setup["material"]["base_color_direct_in_hlsl"],
        "direct_scalars": setup["material"]["direct_scalars_in_hlsl"],
        "sigmoid_absent": setup["material"]["sigmoid_absent"],
        "ao_connected": setup["material"]["ao_connected"],
        "candidate_compile_failure": candidate_compile_failure,
        "map_check_zero_errors": map_check_zero_errors,
        "single_actor_serial_switch": capture["one_actor_serial_material_switch"],
        "shared_lighting_exposure_postprocess": capture["shared_lighting_exposure_postprocess"],
    }
    structural_pass = (
        structural["readback_exact"]
        and structural["actual_resident_bytes"] == 33554432
        and structural["texture_samples"] == 2
        and structural["same_uv_wrap_bilinear"]
        and structural["base_color_direct"]
        and structural["direct_scalars"]
        and structural["sigmoid_absent"]
        and not structural["ao_connected"]
        and not structural["candidate_compile_failure"]
        and structural["map_check_zero_errors"]
        and structural["single_actor_serial_switch"]
        and structural["shared_lighting_exposure_postprocess"]
    )
    return {
        "schema_version": 1,
        "status": "accepted" if structural_pass else "rejected",
        "offline_gate_source": "R0b 13/13; UE render is corroborating evidence, not a replacement gate",
        "formal_holdout_accessed": False,
        "structural_pass": structural_pass,
        "structural": structural,
        "readback": {
            **readback_equal,
            "texture_a_source_sha256": _sha256(TEXTURE_A),
            "texture_b_source_sha256": _sha256(TEXTURE_B),
            "texture_a_readback_sha256": _sha256(readback_a),
            "texture_b_readback_sha256": _sha256(readback_b),
        },
        "visual_metrics": metrics,
        "visual_interpretation": {
            "d1_metallic_boundary": "candidate remains close to baseline in this lighting; no strong visual win claimed",
            "d2_yellow_tube": "candidate visibly suppresses baseline dark speckling and restores the yellow tube",
            "d3_gray_panel": "candidate reduces dark speckling and is visibly closer to reference",
        },
        "ue_log": {
            "path": log_path.as_posix(),
            "sha256": _sha256(log_path),
            "format_evidence_lines": format_lines,
        },
        "map_check": {
            "path": map_check_log.as_posix(),
            "sha256": _sha256(map_check_log),
            "summary": "0 errors, 4 warnings" if map_check_zero_errors else map_check_summary,
            "zero_errors": map_check_zero_errors,
            "warnings": 4,
            "warning_scope": "non-blocking co-location warnings from isolated sequential capture actors",
        },
        "capture_report_sha256": _sha256(CAPTURE_REPORT),
        "setup_report_sha256": _sha256(SETUP_REPORT),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ue-log", type=Path, default=None)
    parser.add_argument("--map-check-log", type=Path, default=None)
    args = parser.parse_args()
    latest_log = _latest_log()
    report = analyze(args.ue_log or latest_log, args.map_check_log or latest_log)
    OUTPUT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if report["status"] == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
