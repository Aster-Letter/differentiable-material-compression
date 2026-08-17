"""Build deterministic texel-center and repair-selection quality evidence."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cg_frontier.assets.gltf_mesh import load_gltf_mesh  # noqa: E402
from cg_frontier.compression.artifact_analysis import (  # noqa: E402
    deterministic_json,
    sha256_file,
)
from cg_frontier.compression.deployment_parity import DeploymentParityDecoder  # noqa: E402
from cg_frontier.compression.material import load_core4_targets  # noqa: E402
from cg_frontier.compression.render_loss import load_latent_unorm8_png  # noqa: E402
from cg_frontier.render.gbuffer import load_core4_textures  # noqa: E402
from train_scifihelmet_interpolation_repair import _load_config  # noqa: E402
from train_scifihelmet_repair import (  # noqa: E402
    _case_specs,
    _decode_mapping,
    _decoder_from_npz,
    _material_report,
    _prepare_cases,
    _reference_mapping,
    _render_summary,
    _repo_path,
)


ROOT_OUTPUT = ROOT / "outputs/compression/scifihelmet/deployment_parity_arc_v1"
BASELINE_LATENT = ROOT / "outputs/compression/scifihelmet/render_quant/baseline/models/tiny_mlp/latent_pre_qat_rgba_unorm8.png"
BASELINE_DECODER = ROOT / "outputs/compression/scifihelmet/material_fit/tiny_mlp/decoder_weights.npz"
CANDIDATES = ("dp_relu_fresh", "arc_relu_fresh")


def _candidate_decoder(path: Path, device: torch.device) -> DeploymentParityDecoder:
    with np.load(path, allow_pickle=False) as stored:
        arrays = {name: torch.from_numpy(np.asarray(stored[name])).to(device) for name in stored.files}
    decoder = DeploymentParityDecoder(width=8).to(device)
    decoder.load_state_dict(arrays)
    return decoder


def _evaluate(
    config: Mapping[str, Any], candidate_name: str, device: torch.device
) -> dict[str, Any]:
    candidate_root = ROOT_OUTPUT / candidate_name
    latent_path = candidate_root / f"latent_{candidate_name}_rgba_unorm8.png"
    decoder_path = candidate_root / "decoder_weights.npz"
    training_path = candidate_root / "training_manifest.json"
    for path in (latent_path, decoder_path, training_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    training = json.loads(training_path.read_text(encoding="utf-8"))
    if training.get("valid") is not True or training.get("formal_holdout_accessed") is not False:
        raise RuntimeError("quality evaluation requires a valid sealed-safe training manifest")
    _, baseline_latent = load_latent_unorm8_png(BASELINE_LATENT, device=device)
    baseline_decoder = _decoder_from_npz(BASELINE_DECODER, device)
    _, candidate_latent = load_latent_unorm8_png(latent_path, device=device)
    candidate_decoder = _candidate_decoder(decoder_path, device)
    targets = load_core4_targets(
        _repo_path(config["inputs"]["core4_dir"], "inputs.core4_dir"), device
    )
    reference = _reference_mapping(targets)
    chunk_size = int(config["training"]["decode_chunk_size"])
    baseline_mapping = _decode_mapping(baseline_latent, baseline_decoder, chunk_size=chunk_size)
    candidate_mapping = _decode_mapping(candidate_latent, candidate_decoder, chunk_size=chunk_size)
    report_config = copy.deepcopy(config)
    report_config["rois_xyxy"] = {
        "D1_metallic_boundary_full": [0, 0, targets.width, targets.height],
        "D2_yellow_tube": [1664, 0, 2048, 512],
        "D3_gray_panel": [1024, 512, 1664, 1152],
    }
    mesh = load_gltf_mesh(_repo_path(config["inputs"]["gltf"], "inputs.gltf"))
    textures = load_core4_textures(
        _repo_path(config["inputs"]["core4_manifest"], "inputs.core4_manifest"), device
    )
    specs, partitions = _case_specs(config)
    names = partitions["selection"][: int(config["training"]["selection_render_case_limit"])]
    cases = _prepare_cases(config, names, specs, mesh, textures, device)
    return {
        "schema_version": 1,
        "candidate": candidate_name,
        "formal_holdout_accessed": False,
        "texel_center_material": {
            "baseline": _material_report(reference, baseline_mapping, report_config),
            "candidate": _material_report(reference, candidate_mapping, report_config),
        },
        "repair_selection_render": {
            "baseline": _render_summary(cases, baseline_latent, baseline_decoder, config),
            "candidate": _render_summary(cases, candidate_latent, candidate_decoder, config),
        },
        "cost": training["cost"],
        "files": training["files"],
        "input_hashes": {
            "training_manifest": sha256_file(training_path),
            "candidate_latent": sha256_file(latent_path),
            "candidate_decoder": sha256_file(decoder_path),
            "baseline_latent": sha256_file(BASELINE_LATENT),
            "baseline_decoder": sha256_file(BASELINE_DECODER),
        },
    }


def run(config_path: Path, candidate_name: str) -> dict[str, Any]:
    if "formal_holdout" in config_path.as_posix().lower():
        raise ValueError("sealed evaluation config is forbidden")
    config = _load_config(config_path)
    if candidate_name not in CANDIDATES:
        raise ValueError(f"unsupported quality candidate: {candidate_name}")
    if not torch.cuda.is_available():
        raise RuntimeError("quality render evaluation requires CUDA")
    first = deterministic_json(_evaluate(config, candidate_name, torch.device("cuda"))).encode("utf-8")
    second = deterministic_json(_evaluate(config, candidate_name, torch.device("cuda"))).encode("utf-8")
    if first != second:
        raise RuntimeError("two quality evaluations are not byte-identical")
    output_path = ROOT_OUTPUT / candidate_name / "quality_metrics.json"
    output_path.write_bytes(first)
    result = json.loads(first)
    result["determinism"] = {
        "two_runs_requested": True,
        "byte_identical": True,
        "sha256": sha256_file(output_path),
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", choices=CANDIDATES, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/train/scifihelmet_deployment_parity_arc_v1.yaml",
    )
    args = parser.parse_args()
    result = run(args.config.resolve(), args.candidate)
    print(
        json.dumps(
            {
                "candidate": args.candidate,
                "byte_identical": result["determinism"]["byte_identical"],
                "formal_holdout_accessed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
