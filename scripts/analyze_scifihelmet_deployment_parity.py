"""Run byte-identical full-atlas and fixed-ROI analysis for a fresh candidate."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import analyze_scifihelmet_filter_aware as roi_analysis  # noqa: E402
import analyze_scifihelmet_interpolation as atlas_analysis  # noqa: E402
from analyze_scifihelmet_filter_aware_candidates import (  # noqa: E402
    decoder_raw_activation,
    load_filter_aware_arrays,
)
from cg_frontier.compression.artifact_analysis import deterministic_json, sha256_file  # noqa: E402


FROZEN_LATENT = ROOT / "outputs/compression/scifihelmet/render_quant/baseline/models/tiny_mlp/latent_pre_qat_rgba_unorm8.png"
FROZEN_DECODER = ROOT / "outputs/compression/scifihelmet/material_fit/tiny_mlp/decoder_weights.npz"
ROOT_OUTPUT = ROOT / "outputs/compression/scifihelmet/deployment_parity_arc_v1"
CANDIDATES = ("dp_relu_fresh", "arc_relu_fresh")
RUNTIME_CONTRACT = (
    "RGBA8 texel -> one bilinear latent sample -> 4->8->7 ReLU decoder -> one Core4 postprocess"
)


def _relative(path: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT) or "formal_holdout" in resolved.as_posix().lower():
        raise ValueError("analysis path escapes the repository or enters a sealed path")
    return resolved.relative_to(ROOT).as_posix()


def _candidate_inputs(name: str) -> tuple[Path, Path, Path]:
    if name == "baseline":
        return FROZEN_LATENT, FROZEN_DECODER, ROOT_OUTPUT / "phase3/baseline/analysis"
    if name not in CANDIDATES:
        raise ValueError(f"unsupported deployment-parity analysis target: {name}")
    root = ROOT_OUTPUT / name
    return (
        root / f"latent_{name}_rgba_unorm8.png",
        root / "decoder_weights.npz",
        root / "analysis",
    )


def _resolved_config(name: str, template_path: Path) -> tuple[dict[str, Any], Path]:
    if "formal_holdout" in template_path.as_posix().lower():
        raise ValueError("sealed analysis config is forbidden")
    config = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    latent, decoder, output_dir = _candidate_inputs(name)
    if not latent.is_file() or not decoder.is_file():
        raise FileNotFoundError(f"analysis inputs are missing for {name}")
    module_identifier = (
        "legacy_single_sample_baseline"
        if name == "baseline"
        else f"deployment_parity_arc_v1.{name}.relu_w8"
    )
    config["candidate"] = {
        "name": name,
        "hidden_activation": "f_relu",
        "module_identifier": module_identifier,
    }
    config["inputs"]["latent_hard_png"] = _relative(latent)
    config["inputs"]["decoder_npz"] = _relative(decoder)
    config["frozen_sha256"] = {
        "latent_hard_png": sha256_file(latent),
        "decoder_npz": sha256_file(decoder),
    }
    config["output_dir"] = _relative(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = output_dir / "resolved_analysis_config.yaml"
    resolved_path.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )
    return config, resolved_path


def _decorate(report: dict[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(report)
    result["scope"] = "scifihelmet_deployment_parity_candidate_analysis"
    result["candidate"] = dict(config["candidate"])
    result["decoder"]["hidden_activation"] = "f_relu"
    result["decoder"]["module_identifier"] = config["candidate"]["module_identifier"]
    result["runtime_order"] = RUNTIME_CONTRACT
    return result


def run(name: str, template_path: Path) -> dict[str, Any]:
    config, resolved_path = _resolved_config(name, template_path)
    original_load = atlas_analysis._load_decoder
    original_raw = atlas_analysis._decoder_raw
    original_roi_load = roi_analysis._load_decoder
    atlas_analysis._load_decoder = load_filter_aware_arrays
    atlas_analysis._decoder_raw = lambda latent, arrays: decoder_raw_activation(
        latent, arrays, "f_relu"
    )
    roi_analysis._load_decoder = load_filter_aware_arrays
    try:
        atlas_a = deterministic_json(_decorate(atlas_analysis.run(resolved_path), config)).encode("utf-8")
        atlas_b = deterministic_json(_decorate(atlas_analysis.run(resolved_path), config)).encode("utf-8")
        first_roi = roi_analysis.run(resolved_path)
        first_roi["candidate"] = dict(config["candidate"])
        first_roi["runtime_contract"] = RUNTIME_CONTRACT
        roi_a = deterministic_json(first_roi).encode("utf-8")
        second_roi = roi_analysis.run(resolved_path)
        second_roi["candidate"] = dict(config["candidate"])
        second_roi["runtime_contract"] = first_roi["runtime_contract"]
        roi_b = deterministic_json(second_roi).encode("utf-8")
    finally:
        atlas_analysis._load_decoder = original_load
        atlas_analysis._decoder_raw = original_raw
        roi_analysis._load_decoder = original_roi_load
    if atlas_a != atlas_b or roi_a != roi_b:
        raise RuntimeError(f"two analysis runs disagree for {name}")
    output_dir = resolved_path.parent
    atlas_path = output_dir / "interpolation_analysis.json"
    roi_path = output_dir / "fixed_roi_analysis.json"
    atlas_path.write_bytes(atlas_a)
    roi_path.write_bytes(roi_a)
    manifest = {
        "schema_version": 1,
        "candidate": name,
        "status": "complete",
        "formal_holdout_accessed": False,
        "determinism": {
            "two_runs_requested": True,
            "atlas_byte_identical": True,
            "fixed_roi_byte_identical": True,
        },
        "reports": {
            "interpolation_analysis.json": sha256_file(atlas_path),
            "fixed_roi_analysis.json": sha256_file(roi_path),
        },
        "resolved_config_sha256": sha256_file(resolved_path),
        "sampler_uv_contract_passed": True,
    }
    (output_dir / "analysis_manifest.json").write_text(
        deterministic_json(manifest), encoding="utf-8", newline="\n"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=("baseline",) + CANDIDATES, required=True)
    parser.add_argument(
        "--template",
        type=Path,
        default=ROOT / "configs/eval/scifihelmet_deployment_parity_phase3.yaml",
    )
    args = parser.parse_args()
    manifest = run(args.candidate, args.template.resolve())
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
