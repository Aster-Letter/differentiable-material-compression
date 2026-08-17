"""Validate and summarize the complete UE BC7 latent feasibility evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs/eval/ue_bc_latent_feasibility_v1.json"
EVIDENCE_ROOT = REPO_ROOT / "outputs/analysis/ue-bc-latent-feasibility-v1"
SOURCE_CSV = REPO_ROOT / "outputs/analysis/ue-runtime-evidence-v1/texture_residency.csv"
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
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


config = read_json(CONFIG_PATH)
config_hash = sha256(CONFIG_PATH)
setup = read_json(EVIDENCE_ROOT / "ue_setup_report.json")
if setup.get("status") != "complete" or setup.get("config_sha256") != config_hash:
    raise RuntimeError("setup report is incomplete or belongs to another config")

with SOURCE_CSV.open("r", encoding="utf-8", newline="") as stream:
    source_rows = list(csv.DictReader(stream))
source_by_variant = {
    row["variant_id"]: row
    for row in source_rows
    if row["variant_id"] in {item["source_variant_id"] for item in config["variants"]}
}

setup_by_id = {item["id"]: item for item in setup["variants"]}
rows: list[dict] = []
for variant in config["variants"]:
    variant_id = variant["id"]
    report = read_json(EVIDENCE_ROOT / "residency_runs" / f"{variant_id}.json")
    if report.get("status") != "complete" or report.get("config_sha256") != config_hash:
        raise RuntimeError(f"invalid residency report: {variant_id}")
    if float(report["actual_warmup_seconds"]) < 30.0:
        raise RuntimeError(f"short warm-up: {variant_id}")
    log_path = EVIDENCE_ROOT / "raw/residency" / f"{variant_id}.log"
    matches = [
        match.groupdict()
        for match in LIST_TEXTURE_RE.finditer(log_path.read_text(encoding="utf-8", errors="replace"))
        if match.group("asset").split(".", 1)[0] == variant["destination_texture"]
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one ListTextures row for {variant_id}, got {len(matches)}")
    parsed = matches[0]
    if parsed["pixel_format"] != config["texture_contract"]["expected_platform_format"]:
        raise RuntimeError(f"unexpected pixel format for {variant_id}: {parsed['pixel_format']}")
    source = source_by_variant.get(variant["source_variant_id"])
    if source is None:
        raise RuntimeError(f"missing RGBA8 source row: {variant['source_variant_id']}")
    current_kb = int(parsed["current_kb"])
    source_kb = int(source["current_resident_kb"])
    setup_row = setup_by_id[variant_id]
    rows.append(
        {
            "variant_id": variant_id,
            "source_variant_id": variant["source_variant_id"],
            "texture_asset": parsed["asset"],
            "width": int(parsed["current_x"]),
            "height": int(parsed["current_y"]),
            "pixel_format": parsed["pixel_format"],
            "current_resident_kb": current_kb,
            "current_resident_bytes": current_kb * 1024,
            "current_resident_mib": round(current_kb / 1024.0, 6),
            "rgba8_resident_kb": source_kb,
            "rgba8_resident_mib": round(source_kb / 1024.0, 6),
            "resident_reduction_fraction": round(1.0 - current_kb / source_kb, 9),
            "resident_reduction_percent": round(100.0 * (1.0 - current_kb / source_kb), 6),
            "streaming": parsed["streaming"],
            "usage_count": int(parsed["usage_count"]),
            "num_mips": int(parsed["num_mips"]),
            "uncompressed": parsed["uncompressed"],
            "actual_warmup_seconds": float(report["actual_warmup_seconds"]),
            "texture_uasset_sha256": setup_row["texture_uasset_sha256"],
            "material_uasset_sha256": setup_row["material_uasset_sha256"],
            "map_umap_sha256": setup_row["map"]["umap_sha256"],
            "raw_log": log_path.relative_to(REPO_ROOT).as_posix(),
            "config_sha256": config_hash,
        }
    )

write_csv(EVIDENCE_ROOT / "texture_residency.csv", rows)
reference_mib = float(config["source_runtime_reference"]["current_resident_mib"])
summary = {
    "schema_version": 1,
    "status": "complete",
    "config_sha256": config_hash,
    "variant_count": len(rows),
    "all_pf_bc7": all(row["pixel_format"] == "PF_BC7" for row in rows),
    "all_complete_mip_chains": all(row["num_mips"] == 12 for row in rows),
    "bc7_resident_mib": sorted({row["current_resident_mib"] for row in rows}),
    "rgba8_resident_mib": sorted({row["rgba8_resident_mib"] for row in rows}),
    "resident_reduction_percent": sorted({row["resident_reduction_percent"] for row in rows}),
    "source_core4_reference_mib": reference_mib,
    "bc7_vs_source_core4_reduction_percent": round(
        100.0 * (1.0 - rows[0]["current_resident_mib"] / reference_mib), 6
    ),
    "formal_holdout_accessed": False,
    "visual_gate": "pass",
    "timing_gate": "complete_representative_160k_pair",
}

visual_summary = read_json(EVIDENCE_ROOT / "visual_summary.json")
timing_summary = read_json(EVIDENCE_ROOT / "timing_summary.json")
if visual_summary.get("visual_gate", {}).get("decision") != "pass":
    raise RuntimeError("visual gate has not passed")
if timing_summary.get("status") != "complete":
    raise RuntimeError("representative timing pair is incomplete")
summary["visual"] = {
    "status": visual_summary["status"],
    "repeat_control_roi_mae": visual_summary["repeat_control"]["roi_mae"],
    "codec_pairs": visual_summary["codec_pairs"],
    "manual_assessment": visual_summary["manual_assessment"],
}
summary["timing"] = {
    "scope": timing_summary["scope"],
    "bc7_minus_rgba8": timing_summary["bc7_minus_rgba8"],
    "interpretation_limit": timing_summary["interpretation_limit"],
}
(EVIDENCE_ROOT / "residency_summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)

brief = [
    "# UE BC7 latent feasibility result",
    "",
    "Three isolated Lantern C4 latent endpoints completed a 30-second warm-up and resource-level `ListTextures` capture.",
    "",
    "| Variant | UE format | RGBA8 MiB | BC7 MiB | Reduction |",
    "|---|---|---:|---:|---:|",
]
for row in rows:
    brief.append(
        f"| `{row['variant_id']}` | `{row['pixel_format']}` | {row['rgba8_resident_mib']:.3f} | "
        f"{row['current_resident_mib']:.3f} | {row['resident_reduction_percent']:.1f}% |"
    )
brief.extend(
    [
        "",
        f"The BC7 latent is {summary['bc7_vs_source_core4_reduction_percent']:.1f}% smaller in current texture residency than the existing 13.438 MiB source Core-4 endpoint.",
        "",
        "## Visual gate",
        "",
        "All three fixed-view RGBA8/BC7 pairs passed the visual feasibility gate. The user judged raw PCA appearance to be well preserved; montage review found no obvious blocking, structural failure, or material-response change in the 20k and 160k pairs.",
        "Independent editor runs are not pixel deterministic. Raw PCA exceeded the repeat-control image delta, while the 20k and 160k pairs were close to that run-noise baseline.",
        "",
        "## Representative timing",
        "",
        f"For Lantern material-render 160k, BC7 minus RGBA8 was {timing_summary['bc7_minus_rgba8']['basepass_median_ms']:+.6f} ms in BasePass and {timing_summary['bc7_minus_rgba8']['total_gpu_median_ms']:+.6f} ms in total GPU frame time.",
        "These sub-millisecond deltas do not establish a causal speedup or slowdown. The supported conclusion is that no material BasePass cost change was observed when only the latent resource format changed.",
        "",
        "## Scope",
        "",
        "The result establishes a deployable BC7 feasibility point for one Lantern asset and fixed-view checks. It does not establish cross-asset quality preservation, mip/LOD robustness, streaming behavior under pressure, or statistical significance.",
    ]
)
(EVIDENCE_ROOT / "RESIDENCY_BRIEF.md").write_text("\n".join(brief) + "\n", encoding="utf-8")

manifest_path = EVIDENCE_ROOT / "MANIFEST.sha256"
manifest_rows = []
for path in sorted(item for item in EVIDENCE_ROOT.rglob("*") if item.is_file() and item != manifest_path):
    manifest_rows.append(f"{sha256(path)}  {path.relative_to(EVIDENCE_ROOT).as_posix()}")
manifest_path.write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
