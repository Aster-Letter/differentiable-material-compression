"""Create deterministic optimizer-only C1/C2 Hybrid initialization artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cg_frontier.compression.artifact_analysis import (  # noqa: E402
    deterministic_json,
    deterministic_tile_partitions,
    sha256_file,
)
from cg_frontier.compression.hybrid import (  # noqa: E402
    HybridInitialization,
    deterministic_pca_initialization,
    export_hybrid_textures,
)
from cg_frontier.compression.material import load_core4_targets  # noqa: E402


def _repo_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"config field {label} must be a non-empty path")
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT) or "formal_holdout" in path.as_posix().lower():
        raise ValueError(f"forbidden repository path: {label}")
    return path


def _tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def _content_report(initialization: HybridInitialization) -> dict[str, Any]:
    arrays = {
        name: _tensor_sha256(value)
        for name, value in initialization.decoder.state_dict().items()
    }
    decoder = initialization.decoder
    return {
        "direct_base_sha256": _tensor_sha256(initialization.direct_base_linear),
        "auxiliary_latent_sha256": _tensor_sha256(initialization.auxiliary_latent),
        "decoder_array_sha256": arrays,
        "initializer": initialization.metadata,
        "cost": {
            "parameters": decoder.parameter_count,
            "weight_bytes_float32": decoder.weight_bytes_float32,
            "macs_per_pixel": decoder.macs_per_pixel,
        },
    }


def run(config_path: Path, verify_twice: bool) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping) or config.get("schema_version") != 1:
        raise ValueError("unsupported Hybrid Phase-0 config")
    core4_dir = _repo_path(config["inputs"]["core4_dir"], "inputs.core4_dir")
    output_dir = _repo_path(config["output_dir"], "output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = load_core4_targets(core4_dir, "cpu")
    if [targets.height, targets.width] != list(config["expected"]["atlas_hw"]):
        raise ValueError("Core-4 atlas shape differs from Hybrid contract")
    initialization_config = config["initialization"]
    partitions = deterministic_tile_partitions(
        targets.height,
        targets.width,
        tile_size=int(initialization_config["tile_size"]),
        seed=int(initialization_config["split_seed"]),
    )
    optimizer = partitions["optimizer"]
    first: dict[int, HybridInitialization] = {}
    content: dict[str, Any] = {}
    for rank in (int(value) for value in initialization_config["ranks"]):
        candidate = f"c{rank - 1}"
        first[rank] = deterministic_pca_initialization(
            targets,
            optimizer,
            rank,
            epsilon=float(initialization_config["inverse_postprocess_epsilon"]),
        )
        content[candidate] = _content_report(first[rank])
    first_digest = hashlib.sha256(deterministic_json(content).encode("utf-8")).hexdigest()
    run_hashes = [first_digest]
    if verify_twice:
        repeated: dict[str, Any] = {}
        for rank in (int(value) for value in initialization_config["ranks"]):
            candidate = f"c{rank - 1}"
            repeated[candidate] = _content_report(
                deterministic_pca_initialization(
                    targets,
                    optimizer,
                    rank,
                    epsilon=float(initialization_config["inverse_postprocess_epsilon"]),
                )
            )
        second_digest = hashlib.sha256(deterministic_json(repeated).encode("utf-8")).hexdigest()
        run_hashes.append(second_digest)
        if content != repeated or first_digest != second_digest:
            raise RuntimeError(f"Hybrid initialization runs disagree: {run_hashes}")
    else:
        raise RuntimeError("Hybrid Phase 0 requires --verify-twice")

    files: dict[str, Any] = {}
    for rank, initialization in first.items():
        candidate = f"c{rank - 1}"
        candidate_dir = output_dir / candidate
        candidate_dir.mkdir(parents=True, exist_ok=True)
        texture_a = candidate_dir / "texture_a_base_rgb_aux0_rgba8.png"
        texture_b = candidate_dir / ("texture_b_aux1_r8.png" if rank == 2 else "texture_b_aux12_rg8.png")
        packing = export_hybrid_textures(
            initialization.direct_base_linear,
            initialization.auxiliary_latent,
            texture_a,
            texture_b,
        )
        decoder_path = candidate_dir / "decoder_weights.npz"
        np.savez(
            decoder_path,
            **{
                name: value.detach().cpu().numpy()
                for name, value in initialization.decoder.state_dict().items()
            },
        )
        expected = config["expected"][candidate]
        actual = {
            "logical_channels": 3 + rank,
            "raw_bytes": packing["logical_raw_bytes"],
            "parameters": initialization.decoder.parameter_count,
            "weight_bytes": initialization.decoder.weight_bytes_float32,
            "macs": initialization.decoder.macs_per_pixel,
        }
        if actual != dict(expected):
            raise RuntimeError(f"{candidate.upper()} cost mismatch: {actual}")
        files[candidate] = {
            "texture_a": {"path": texture_a.relative_to(ROOT).as_posix(), "sha256": sha256_file(texture_a)},
            "texture_b": {"path": texture_b.relative_to(ROOT).as_posix(), "sha256": sha256_file(texture_b)},
            "decoder": {"path": decoder_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(decoder_path)},
            "packing": packing,
        }
    core4_hashes = {
        name: sha256_file(core4_dir / filename)
        for name, filename in {
            "base_color": "base_color.png",
            "normal": "normal.png",
            "roughness": "roughness.png",
            "metallic": "metallic.png",
        }.items()
    }
    report = {
        "schema_version": 1,
        "status": "phase0_initialization_complete",
        "formal_holdout_accessed": False,
        "config": {"path": config_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(config_path)},
        "core4_sha256": core4_hashes,
        "partition_pixels": {name: int(mask.sum()) for name, mask in partitions.items()},
        "selection_or_validation_used_for_initialization": False,
        "determinism": {"two_runs_requested": True, "run_content_sha256": run_hashes, "identical": True},
        "candidates": content,
        "files": files,
        "runtime_contract": {
            "base_color": "linear RGB direct bilinear; no decoder, sigmoid, or additional sRGB transform",
            "auxiliary": "UNORM8 texels -> bilinear -> 2/3->8->4 -> tanh/tanh/sigmoid/sigmoid",
            "texture_samples": 2,
        },
    }
    report_path = output_dir / "initialization_report.json"
    report_path.write_text(deterministic_json(report), encoding="utf-8", newline="\n")
    print(json.dumps({"status": report["status"], "report_sha256": sha256_file(report_path), "deterministic": True}, ensure_ascii=False))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/eval/scifihelmet_hybrid_phase0.yaml")
    parser.add_argument("--verify-twice", action="store_true")
    args = parser.parse_args()
    run(args.config.resolve(), args.verify_twice)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
