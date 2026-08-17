"""Build the canonical portable-report payload for the Lantern 40k analysis."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = (
    ROOT
    / "outputs"
    / "analysis"
    / "c4-render-ablation-lantern-40k-job-37581"
    / "report"
)


SOURCE_SQL = {
    "endpoint_csv": "Read validated 20k and 40k endpoint reports for Lantern and normalize shared metrics.",
    "comparison_csv": "Compute paired absolute and relative 20k-to-40k changes for both training arms.",
    "case_csv": "Join the same 42 audit camera-light cases by case identity and calculate per-case change.",
    "validation_json": "Verify archive and result manifests, safe paths, checkpoint reloads, and paired sampling evidence.",
}


def source(source_id: str, label: str, path: str) -> dict[str, object]:
    return {
        "id": source_id,
        "label": label,
        "path": path,
        "query": {
            "engine": "python",
            "sql": SOURCE_SQL[source_id],
            "description": label,
            "executed_at": "2026-08-14T00:00:00Z",
        },
    }


def main() -> None:
    endpoint_rows = [
        {"step": 20_000, "arm": "material_only", "hdr_mae_x1e3": 4.674222, "display_ssim": 0.969308},
        {"step": 40_000, "arm": "material_only", "hdr_mae_x1e3": 4.271624, "display_ssim": 0.970615},
        {"step": 20_000, "arm": "material_render", "hdr_mae_x1e3": 5.059503, "display_ssim": 0.968392},
        {"step": 40_000, "arm": "material_render", "hdr_mae_x1e3": 4.170954, "display_ssim": 0.971311},
    ]
    material_rows = [
        {"metric": "BaseColor MAE", "arm": "material_only", "improvement_pct": 15.7644},
        {"metric": "BaseColor MAE", "arm": "material_render", "improvement_pct": 33.5692},
        {"metric": "Seven-channel MAE", "arm": "material_only", "improvement_pct": 6.2355},
        {"metric": "Seven-channel MAE", "arm": "material_render", "improvement_pct": 11.8396},
        {"metric": "Normal angle", "arm": "material_only", "improvement_pct": 0.1032},
        {"metric": "Normal angle", "arm": "material_render", "improvement_pct": 0.1049},
        {"metric": "Roughness MAE", "arm": "material_only", "improvement_pct": 14.2872},
        {"metric": "Roughness MAE", "arm": "material_render", "improvement_pct": 15.6556},
        {"metric": "Metallic MAE", "arm": "material_only", "improvement_pct": -2.2345},
        {"metric": "Metallic MAE", "arm": "material_render", "improvement_pct": 9.6104},
        {"metric": "Oklab delta-E", "arm": "material_only", "improvement_pct": 11.4264},
        {"metric": "Oklab delta-E", "arm": "material_render", "improvement_pct": 25.6982},
        {"metric": "Opponent error", "arm": "material_only", "improvement_pct": -7.1567},
        {"metric": "Opponent error", "arm": "material_render", "improvement_pct": -4.2278},
        {"metric": "Chroma retention", "arm": "material_only", "improvement_pct": 2.1412},
        {"metric": "Chroma retention", "arm": "material_render", "improvement_pct": 5.4729},
    ]
    case_rows = [
        {
            "arm": "material_only",
            "improved_cases": 42,
            "total_cases": 42,
            "median_hdr_change_pct": -5.7336,
            "best_hdr_change_pct": -30.5980,
            "smallest_hdr_change_pct": -0.8790,
        },
        {
            "arm": "material_render",
            "improved_cases": 42,
            "total_cases": 42,
            "median_hdr_change_pct": -14.7785,
            "best_hdr_change_pct": -36.4900,
            "smallest_hdr_change_pct": -6.8810,
        },
    ]
    endpoint_detail = [
        {
            "arm": "material_only",
            "audit_hdr_mae": 0.004271624,
            "worst_hdr_mae": 0.013612648,
            "display_ssim": 0.970615,
            "basecolor_mae": 0.009857741,
            "seven_channel_mae": 0.021234466,
            "roughness_mae": 0.026055306,
        },
        {
            "arm": "material_render",
            "audit_hdr_mae": 0.004170954,
            "worst_hdr_mae": 0.012884066,
            "display_ssim": 0.971311,
            "basecolor_mae": 0.009989993,
            "seven_channel_mae": 0.021330012,
            "roughness_mae": 0.025831223,
        },
    ]

    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Lantern C4 differentiable rendering: 20k to 40k",
            "description": "Exact-resume validation and paired endpoint analysis for SCOW Job 37581.",
            "generatedAt": "2026-08-14T00:00:00Z",
            "cards": [
                {
                    "id": "render_gain",
                    "description": "Reduction in the material-render arm's 42-case audit HDR MAE.",
                    "dataset": "headline",
                    "sourceId": "comparison_csv",
                    "metrics": [{"label": "Render arm HDR improvement", "field": "render_gain", "format": "percent", "signed": True}],
                },
                {
                    "id": "relative_edge",
                    "description": "Lower audit HDR MAE for material-render relative to material-only at 40k.",
                    "dataset": "headline",
                    "sourceId": "comparison_csv",
                    "metrics": [{"label": "Render arm edge at 40k", "field": "relative_edge", "format": "percent", "signed": True}],
                },
                {
                    "id": "paired_cases",
                    "description": "Paired audit cases whose HDR MAE improved from 20k to 40k, per arm.",
                    "dataset": "headline",
                    "sourceId": "case_csv",
                    "metrics": [{"label": "Improved cases per arm", "field": "paired_cases", "format": "number"}],
                },
                {
                    "id": "integrity",
                    "description": "Formal result files whose manifest hashes were independently verified.",
                    "dataset": "headline",
                    "sourceId": "validation_json",
                    "metrics": [{"label": "Verified result files", "field": "verified_files", "format": "number"}],
                },
            ],
            "charts": [
                {
                    "id": "audit_endpoints",
                    "title": "Audit HDR error continues to fall",
                    "subtitle": "Mean linear HDR MAE across 42 read-only audit cases; lower is better.",
                    "headerMarkdown": "The render-supervised arm improves faster and crosses below material-only at **40k**.",
                    "type": "line",
                    "dataset": "endpoint",
                    "sourceId": "endpoint_csv",
                    "encodings": {
                        "x": {"field": "step", "type": "quantitative", "label": "Training step"},
                        "y": {"field": "hdr_mae_x1e3", "type": "quantitative", "label": "Audit HDR MAE (x1e-3)"},
                        "color": {"field": "arm", "type": "nominal", "label": "Training arm"},
                        "tooltip": [{"field": "display_ssim", "type": "quantitative", "label": "Display SSIM"}],
                    },
                },
                {
                    "id": "material_improvement",
                    "title": "Material-domain change from 20k to 40k",
                    "subtitle": "Positive means improvement after accounting for each metric's direction; negative means regression.",
                    "headerMarkdown": "Most material metrics improve, but opponent error regresses in both arms and metallic MAE slightly regresses for **material-only**.",
                    "type": "bar",
                    "dataset": "material_improvement",
                    "sourceId": "comparison_csv",
                    "encodings": {
                        "x": {"field": "metric", "type": "nominal", "label": "Metric"},
                        "y": {"field": "improvement_pct", "type": "quantitative", "label": "Improvement (%)"},
                        "color": {"field": "arm", "type": "nominal", "label": "Training arm"},
                    },
                },
            ],
            "tables": [
                {
                    "id": "case_summary",
                    "title": "Paired audit-case distribution",
                    "subtitle": "All 42 camera-light cases improve in both arms; negative HDR change is favorable.",
                    "dataset": "case_summary",
                    "sourceId": "case_csv",
                    "columns": [
                        {"field": "arm", "label": "Arm", "type": "text"},
                        {"field": "improved_cases", "label": "Improved", "format": "number"},
                        {"field": "total_cases", "label": "Total", "format": "number"},
                        {"field": "median_hdr_change_pct", "label": "Median HDR change (%)", "format": "number"},
                        {"field": "best_hdr_change_pct", "label": "Largest improvement (%)", "format": "number"},
                        {"field": "smallest_hdr_change_pct", "label": "Smallest improvement (%)", "format": "number"},
                    ],
                },
                {
                    "id": "endpoint_detail",
                    "title": "40k endpoint trade-off",
                    "subtitle": "Render quality slightly favors material-render; BaseColor and seven-channel errors slightly favor material-only.",
                    "dataset": "endpoint_detail",
                    "sourceId": "endpoint_csv",
                    "columns": [
                        {"field": "arm", "label": "Arm", "type": "text"},
                        {"field": "audit_hdr_mae", "label": "Audit HDR MAE", "format": "number"},
                        {"field": "worst_hdr_mae", "label": "Worst HDR MAE", "format": "number"},
                        {"field": "display_ssim", "label": "Display SSIM", "format": "number"},
                        {"field": "basecolor_mae", "label": "BaseColor MAE", "format": "number"},
                        {"field": "seven_channel_mae", "label": "Seven-channel MAE", "format": "number"},
                        {"field": "roughness_mae", "label": "Roughness MAE", "format": "number"},
                    ],
                },
            ],
            "sources": [
                source("endpoint_csv", "Validated endpoint metrics", "report/endpoint_metrics.csv"),
                source("comparison_csv", "Validated 20k-to-40k comparisons", "report/comparisons.csv"),
                source("case_csv", "Paired audit-case deltas", "report/audit_case_deltas.csv"),
                source("validation_json", "Archive and checkpoint validation", "report/validation.json"),
            ],
            "blocks": [
                {
                    "id": "executive_summary",
                    "type": "markdown",
                    "sourceId": "comparison_csv",
                    "body": (
                        "## Executive Summary\n\n"
                        "Continuing Lantern from 20k to 40k is beneficial and changes the endpoint interpretation. "
                        "Material-only audit HDR MAE falls **8.61%**; material-render falls **17.56%**. "
                        "At 20k, material-render is 8.24% worse than material-only; at 40k it is **2.36% better**. "
                        "This is a small, single-seed endpoint advantage—not evidence of a universal winner."
                    ),
                },
                {"id": "headline_metrics", "type": "metric-strip", "cardIds": ["render_gain", "relative_edge", "paired_cases", "integrity"]},
                {"id": "audit_chart_block", "type": "chart", "chartId": "audit_endpoints"},
                {"id": "case_table_block", "type": "table", "tableId": "case_summary"},
                {
                    "id": "technical_readout",
                    "type": "markdown",
                    "sourceId": "endpoint_csv",
                    "body": (
                        "## Technical readout\n\n"
                        "Worst-case audit HDR also improves: **9.38%** for material-only and **14.88%** for material-render. "
                        "At 40k, material-render has the better mean/worst render error and display SSIM; "
                        "material-only retains a slight BaseColor and seven-channel advantage. The result remains a Pareto trade-off."
                    ),
                },
                {"id": "endpoint_table_block", "type": "table", "tableId": "endpoint_detail"},
                {"id": "material_chart_block", "type": "chart", "chartId": "material_improvement"},
                {
                    "id": "methodology",
                    "type": "markdown",
                    "sourceId": "validation_json",
                    "body": (
                        "## Methodology and integrity\n\n"
                        "SCOW formal Job 37581 resumes both arms from Job 37477's exact 20k optimizer/RNG state. "
                        "The formal manifest contains **72/72 verified files** (849,020,497 bytes). All four 30k/40k checkpoints "
                        "reload locally with matching source identity, config hash, and parent hash. Initial/final RNG, sampling trajectory, "
                        "step count, and sampling contract match across arms. Audit views were read-only and formal holdout data was not accessed."
                    ),
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "body": (
                        "## Limitations\n\n"
                        "- One deterministic seed does not estimate across-seed variance or statistical significance.\n"
                        "- Lantern's Core-4 reference excludes approximately 3.27% emissive area.\n"
                        "- Intermediate 25k/30k/35k observations have material metrics and fixed views, not full 42-case audit rerenders.\n"
                        "- Both arms regress on opponent error from 20k to 40k; this needs UE inspection for localized hue or composite-color drift."
                    ),
                },
                {
                    "id": "recommendation",
                    "type": "markdown",
                    "body": (
                        "## Recommendation\n\n"
                        "Keep the 40k Lantern result as evidence that training duration can reverse a 20k judgment about render supervision. "
                        "Deploy both 40k endpoints beside source/raw_q4/20k in UE before extending further. If stability matters, spend the next "
                        "budget on independent seeds or another hard asset rather than extending this same seed without a bound."
                    ),
                },
                {
                    "id": "further_questions",
                    "type": "markdown",
                    "body": (
                        "## Further questions\n\n"
                        "- Does the small 40k render advantage reproduce on Corset, BoomBox, or a second seed?\n"
                        "- Is 30k visually indistinguishable from 40k, allowing a cheaper stopping point?\n"
                        "- Does the opponent-error regression map to a visible local color shift in UE?"
                    ),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": "2026-08-14T00:00:00Z",
            "status": "ready",
            "datasets": {
                "headline": [{"render_gain": 0.17561984, "relative_edge": 0.02356697, "paired_cases": 42, "verified_files": 72}],
                "endpoint": endpoint_rows,
                "material_improvement": material_rows,
                "case_summary": case_rows,
                "endpoint_detail": endpoint_detail,
            },
        },
        "sources": [
            {
                "id": "endpoint_csv",
                "query": {
                    "engine": "python",
                    "sql": "Read validated 20k and 40k endpoint reports for Lantern and normalize shared metrics.",
                    "description": "Exact endpoint extraction from the archived reports.",
                    "executed_at": "2026-08-14T00:00:00Z",
                },
            },
            {
                "id": "comparison_csv",
                "query": {
                    "engine": "python",
                    "sql": "Compute paired absolute and relative 20k-to-40k changes for both training arms.",
                    "description": "Deterministic endpoint comparison.",
                    "executed_at": "2026-08-14T00:00:00Z",
                },
            },
            {
                "id": "case_csv",
                "query": {
                    "engine": "python",
                    "sql": "Join the same 42 audit camera-light cases by case identity and calculate per-case change.",
                    "description": "Paired audit-case analysis.",
                    "executed_at": "2026-08-14T00:00:00Z",
                },
            },
            {
                "id": "validation_json",
                "query": {
                    "engine": "python",
                    "sql": "Verify archive and result manifests, safe paths, checkpoint reloads, and paired sampling evidence.",
                    "description": "Independent local integrity validation.",
                    "executed_at": "2026-08-14T00:00:00Z",
                },
            },
        ],
    }

    destination = REPORT / "artifact.json"
    destination.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
