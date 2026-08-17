"""Analyze and render the verified O2-r050 10k/20k/30k continuation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for directory in (SRC, SCRIPTS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from cg_frontier.compression.raw_orthogonal_training import (  # noqa: E402
    load_raw_orthogonal_checkpoint,
)
from render_scifihelmet_c4_affine_raw_pca import _camera  # noqa: E402
from render_scifihelmet_raw_pca_orthogonal_summary import (  # noqa: E402
    _panel,
    _render_state,
    _sha256,
    _state,
)
from train_scifihelmet_c4_affine_raw_pca_orthogonal_10k import (  # noqa: E402
    _color_metrics,
    _legacy_atlas_metrics,
    _prepare,
    _render_metrics,
)


DEFAULT_CONFIG = ROOT / "configs/train/scifihelmet_c4_affine_raw_pca_orthogonal_10k_v1.yaml"
DEFAULT_CONTINUATION = ROOT / (
    "outputs/scifihelmet_c4_affine_v1/raw_pca_orthogonal/"
    "bf215f-o2-r050-30k-r1"
)
DEFAULT_OUTPUT = ROOT / (
    "outputs/scifihelmet_c4_affine_v1/raw_pca_orthogonal/"
    "bf215f-o2-r050-30k-analysis-r1"
)


def _write_json(path: Path, value: object) -> str:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n", encoding="ascii")
    return digest


def run(config_path: Path, continuation: Path, output: Path) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite analysis root: {output}")
    output.mkdir(parents=True)
    config = yaml.safe_load(config_path.read_bytes())
    report_path = continuation / "training_report.json"
    training_report = json.loads(report_path.read_text(encoding="utf-8"))
    if training_report.get("status") != "complete_30k":
        raise ValueError("continuation training report is not complete_30k")
    lineage = training_report["lineage"]
    prepared = _prepare(config, config_path)
    if prepared["lineage"] != lineage:
        raise ValueError("continuation lineage mismatch")

    checkpoints: dict[str, dict[str, str]] = {}
    states = {}
    for step in (10000, 20000, 30000):
        checkpoint = continuation / "checkpoints" / f"step_{step:05d}" / "checkpoint.pt"
        payload = load_raw_orthogonal_checkpoint(
            checkpoint,
            expected_candidate_id="O2-r050",
            expected_objective_id="O2",
            expected_ratio=0.5,
            expected_lineage=lineage,
        )
        if int(payload["step"]) != step:
            raise ValueError(f"checkpoint step mismatch at {step}")
        states[str(step)] = _state(checkpoint)
        checkpoints[str(step)] = {
            "path": checkpoint.relative_to(ROOT).as_posix(),
            "sha256": _sha256(checkpoint),
        }

    raw = prepared["raw"]
    parent = (
        raw.latent_unorm8.cuda().float() / 255.0,
        raw.weight.cuda().float(),
        raw.bias.cuda().float(),
    )
    views = []
    for value in config["visual_views"]:
        views.append(
            (
                str(value["id"]),
                str(value["label"]),
                _camera(value, prepared["render"], prepared["cameras"]),
                prepared["lights"][int(value["light_index"])],
            )
        )
    rows = [(view_id, label) for view_id, label, _, _ in views]
    render_states = {"source": None, "parent": parent, **states}
    images = {}
    for view_id, _, camera, light in views:
        for state_id, state in render_states.items():
            images[(view_id, state_id)] = _render_state(prepared, camera, light, state)
    figure = output / "o2_r050_10k_20k_30k_four_views.png"
    _panel(
        "SciFiHelmet O2-r050 continuation",
        "Same objective/LR/RNG lineage; 10k vs 20k vs 30k; no automatic winner",
        rows,
        [
            ("source", "Source"),
            ("parent", "Raw parent"),
            ("10000", "10k"),
            ("20000", "20k"),
            ("30000", "30k"),
        ],
        images,
        figure,
    )

    all_pairs = [
        (camera, light)
        for light in range(len(prepared["lights"]))
        for camera in range(len(prepared["cameras"]))
    ]
    metrics = {}
    for step in (10000, 20000, 30000):
        state = states[str(step)]
        metrics[str(step)] = {
            **_legacy_atlas_metrics(
                *state, prepared["valid_indices"], prepared["target_valid"]
            ),
            "color": _color_metrics(*state, prepared),
            "full_render_31x6": _render_metrics(
                *state,
                prepared["geometries"],
                prepared["cameras"],
                prepared["lights"],
                prepared["references"],
                all_pairs,
                prepared["render"],
            ),
        }
    analysis = {
        "schema_version": 1,
        "status": "complete_10k_20k_30k_analysis",
        "candidate_id": "O2-r050",
        "objective_id": "O2",
        "ratio": 0.5,
        "lineage": lineage,
        "checkpoints": checkpoints,
        "metrics": metrics,
        "artifacts": [
            {
                "path": figure.relative_to(ROOT).as_posix(),
                "sha256": _sha256(figure),
            }
        ],
        "training_report_sha256": _sha256(report_path),
        "yellow_diagnostics": {"selection_metric": False},
        "formal_holdout_accessed": False,
        "ue_started": False,
    }
    _write_json(output / "analysis_report.json", analysis)
    return analysis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--continuation-root", type=Path, default=DEFAULT_CONTINUATION)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(
        args.config.resolve(),
        args.continuation_root.resolve(),
        args.output_root.resolve(),
    )
    print(json.dumps({"status": result["status"], "artifacts": len(result["artifacts"])}))


if __name__ == "__main__":
    main()
