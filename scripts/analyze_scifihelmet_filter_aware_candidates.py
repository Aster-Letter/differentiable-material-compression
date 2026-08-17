"""Run byte-identical full-atlas and fixed-ROI analysis for one candidate."""

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
from cg_frontier.compression.artifact_analysis import deterministic_json, sha256_file  # noqa: E402


FROZEN_LATENT = ROOT / "outputs/compression/scifihelmet/render_quant/baseline/models/tiny_mlp/latent_pre_qat_rgba_unorm8.png"
FROZEN_DECODER = ROOT / "outputs/compression/scifihelmet/material_fit/tiny_mlp/decoder_weights.npz"
ROOT_OUTPUT = ROOT / "outputs/compression/scifihelmet/filter_aware_decoder_v1"
ACTIVATIONS = ("f_relu", "f_softplus", "f_sigmoid")


def _relative(path: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT) or "formal_holdout" in resolved.as_posix().lower():
        raise ValueError("analysis path escapes the repository or enters a sealed path")
    return resolved.relative_to(ROOT).as_posix()


def _candidate_inputs(name: str) -> tuple[Path, Path, str, Path]:
    if name == "baseline":
        return FROZEN_LATENT, FROZEN_DECODER, "f_relu", ROOT_OUTPUT / "phase3/baseline/analysis"
    if name not in ACTIVATIONS:
        raise ValueError(f"unsupported candidate analysis target: {name}")
    root = ROOT_OUTPUT / name
    latent = root / f"latent_{name}_rgba_unorm8.png"
    decoder = root / "decoder_weights.npz"
    return latent, decoder, name, root / "analysis"


def load_filter_aware_arrays(path: Path) -> dict[str, np.ndarray]:
    """Load frozen network.* or candidate hidden/output arrays into one schema."""

    with np.load(path, allow_pickle=False) as stored:
        arrays = {name: np.asarray(stored[name], dtype=np.float32).copy() for name in stored.files}
    frozen = {
        "network.0.weight": (8, 4),
        "network.0.bias": (8,),
        "network.2.weight": (7, 8),
        "network.2.bias": (7,),
    }
    candidate = {
        "hidden.weight": (8, 4),
        "hidden.bias": (8,),
        "output.weight": (7, 8),
        "output.bias": (7,),
    }
    if set(arrays) == set(frozen):
        normalized = arrays
    elif set(arrays) == set(candidate):
        normalized = {
            "network.0.weight": arrays["hidden.weight"],
            "network.0.bias": arrays["hidden.bias"],
            "network.2.weight": arrays["output.weight"],
            "network.2.bias": arrays["output.bias"],
        }
    else:
        raise ValueError(f"unexpected filter-aware decoder arrays: {sorted(arrays)}")
    for name, shape in frozen.items():
        if normalized[name].shape != shape or not np.isfinite(normalized[name]).all():
            raise ValueError(f"invalid filter-aware array {name}")
    return normalized


def decoder_raw_activation(
    latent: np.ndarray, arrays: Mapping[str, np.ndarray], activation: str
) -> np.ndarray:
    """Evaluate the fixed affine layers with the declared hidden activation."""

    values = np.asarray(latent, dtype=np.float32)
    preactivation = np.sum(
        values[..., None, :] * arrays["network.0.weight"], axis=-1, dtype=np.float32
    ) + arrays["network.0.bias"]
    if activation == "f_relu":
        hidden = np.maximum(preactivation, np.float32(0.0))
    elif activation == "f_softplus":
        hidden = np.asarray(np.logaddexp(np.float32(0.0), preactivation * np.float32(4.0)) / np.float32(4.0), dtype=np.float32)
    elif activation == "f_sigmoid":
        hidden = np.asarray(1.0 / (1.0 + np.exp(-preactivation)), dtype=np.float32)
    else:
        raise ValueError(f"unknown activation: {activation}")
    return np.asarray(
        np.sum(hidden[..., None, :] * arrays["network.2.weight"], axis=-1, dtype=np.float32)
        + arrays["network.2.bias"],
        dtype=np.float32,
    )


def _resolved_config(name: str, template_path: Path) -> tuple[dict[str, Any], Path]:
    config = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    latent, decoder, activation, output_dir = _candidate_inputs(name)
    if not latent.is_file() or not decoder.is_file():
        raise FileNotFoundError(f"analysis inputs are missing for {name}")
    config["candidate"] = {
        "name": name,
        "hidden_activation": activation,
        "module_identifier": (
            "legacy_single_sample_baseline" if name == "baseline" else f"filter_aware_decoder_v1.{name}"
        ),
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
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8", newline="\n"
    )
    return config, resolved_path


def _decorate(report: dict[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(report)
    result["scope"] = "scifihelmet_filter_aware_candidate_analysis"
    result["candidate"] = dict(config["candidate"])
    result["decoder"]["hidden_activation"] = config["candidate"]["hidden_activation"]
    result["decoder"]["module_identifier"] = config["candidate"]["module_identifier"]
    result["runtime_order"] = (
        "RGBA8 texel -> one bilinear latent sample -> 4->8->7 declared-activation decoder -> one channel postprocess"
    )
    return result


def run(name: str, template_path: Path) -> dict[str, Any]:
    config, resolved_path = _resolved_config(name, template_path)
    activation = str(config["candidate"]["hidden_activation"])
    original_load = atlas_analysis._load_decoder
    original_raw = atlas_analysis._decoder_raw
    atlas_analysis._load_decoder = load_filter_aware_arrays
    atlas_analysis._decoder_raw = lambda latent, arrays: decoder_raw_activation(latent, arrays, activation)
    roi_analysis._load_decoder = load_filter_aware_arrays
    try:
        first_atlas = _decorate(atlas_analysis.run(resolved_path), config)
        atlas_bytes_a = deterministic_json(first_atlas).encode("utf-8")
        second_atlas = _decorate(atlas_analysis.run(resolved_path), config)
        atlas_bytes_b = deterministic_json(second_atlas).encode("utf-8")
        first_roi = roi_analysis.run(resolved_path)
        first_roi["candidate"] = dict(config["candidate"])
        first_roi["runtime_contract"] = (
            "RGBA8 texel -> one bilinear latent sample -> 4->8->7 declared-activation decoder -> one postprocess"
        )
        roi_bytes_a = deterministic_json(first_roi).encode("utf-8")
        second_roi = roi_analysis.run(resolved_path)
        second_roi["candidate"] = dict(config["candidate"])
        second_roi["runtime_contract"] = first_roi["runtime_contract"]
        roi_bytes_b = deterministic_json(second_roi).encode("utf-8")
    finally:
        atlas_analysis._load_decoder = original_load
        atlas_analysis._decoder_raw = original_raw
    if atlas_bytes_a != atlas_bytes_b or roi_bytes_a != roi_bytes_b:
        raise RuntimeError(f"two analysis runs disagree for {name}")
    output_dir = resolved_path.parent
    atlas_path = output_dir / "interpolation_analysis.json"
    roi_path = output_dir / "fixed_roi_analysis.json"
    atlas_path.write_bytes(atlas_bytes_a)
    roi_path.write_bytes(roi_bytes_a)
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
    parser.add_argument("--candidate", choices=("baseline",) + ACTIVATIONS, required=True)
    parser.add_argument(
        "--template",
        type=Path,
        default=ROOT / "configs/eval/scifihelmet_filter_aware_phase3_template.yaml",
    )
    args = parser.parse_args()
    manifest = run(args.candidate, args.template.resolve())
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
