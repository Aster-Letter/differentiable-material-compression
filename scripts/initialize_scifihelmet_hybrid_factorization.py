"""Create deterministic causal Hybrid initializations and analysis configs."""

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
    material_error_maps,
    sha256_file,
)
from cg_frontier.compression.hybrid import decode_auxiliary, export_hybrid_textures  # noqa: E402
from cg_frontier.compression.hybrid_factorization import (  # noqa: E402
    CausalHybridInitialization,
    candidate_aux_channels,
    deterministic_causal_initialization,
    direct_semantic_material,
    gradient_conflict_report,
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


@torch.no_grad()
def _mapping(initialization: CausalHybridInitialization) -> dict[str, np.ndarray]:
    direct = initialization.direct_base_linear
    auxiliary = initialization.auxiliary_latent
    height, width = auxiliary.shape[:2]
    if initialization.decoder is None:
        normal, roughness, metallic = direct_semantic_material(auxiliary)
        decoded = (normal, roughness, metallic)
    else:
        flat_aux = auxiliary.reshape(-1, auxiliary.shape[-1])
        flat_base = direct.reshape(-1, 3)
        pieces = []
        for start in range(0, flat_aux.shape[0], 262144):
            pieces.append(decode_auxiliary(initialization.decoder, flat_aux[start:start + 262144], flat_base[start:start + 262144]))
        normal = torch.cat([item.normal_xyz for item in pieces])
        roughness = torch.cat([item.roughness for item in pieces])
        metallic = torch.cat([item.metallic for item in pieces])
        decoded = (normal, roughness, metallic)
    normal, roughness, metallic = decoded
    return {
        "base_color_linear": direct.cpu().numpy(),
        "normal_xyz": normal.reshape(height, width, 3).cpu().numpy(),
        "roughness_linear": roughness.reshape(height, width).cpu().numpy(),
        "metallic_linear": metallic.reshape(height, width).cpu().numpy(),
    }


def _content_report(
    initialization: CausalHybridInitialization,
    gradient: dict[str, Any] | None,
    errors: dict[str, Any],
) -> dict[str, Any]:
    decoder = initialization.decoder
    arrays = {} if decoder is None else {
        name: _tensor_sha256(value) for name, value in decoder.state_dict().items()
    }
    return {
        "direct_base_sha256": _tensor_sha256(initialization.direct_base_linear),
        "auxiliary_latent_sha256": _tensor_sha256(initialization.auxiliary_latent),
        "decoder_array_sha256": arrays,
        "initializer": initialization.metadata,
        "gradient_conflict": gradient,
        "initial_texel_center": {
            "base_p99": float(np.quantile(errors["base_color_max_channel"], 0.99)),
            "normal_p95": float(np.quantile(errors["normal_degrees"], 0.95)),
            "roughness_mae": float(np.mean(errors["roughness"])),
            "metallic_mae": float(np.mean(errors["metallic"])),
        },
        "cost": {
            "parameters": 0 if decoder is None else decoder.parameter_count,
            "weight_bytes_float32": 0 if decoder is None else decoder.weight_bytes_float32,
            "macs_per_pixel": 0 if decoder is None else decoder.macs_per_pixel,
        },
    }


def _analysis_config(
    config: Mapping[str, Any],
    candidate: str,
    candidate_dir: Path,
    texture_a: Path,
    texture_b: Path,
    decoder: Path,
) -> dict[str, Any]:
    channels = candidate_aux_channels(candidate)
    return {
        "schema_version": 1,
        "inputs": {
            "core4_dir": config["inputs"]["core4_dir"],
            "texture_a_png": texture_a.relative_to(ROOT).as_posix(),
            "texture_b_png": texture_b.relative_to(ROOT).as_posix(),
            "decoder_npz": decoder.relative_to(ROOT).as_posix(),
        },
        "frozen_sha256": {
            "texture_a_png": sha256_file(texture_a),
            "texture_b_png": sha256_file(texture_b),
            "decoder_npz": sha256_file(decoder),
        },
        "representation": {
            "candidate": candidate,
            "architecture": str(candidate),
            "texture_a": "RGBA8_linear_BaseColor_RGB_plus_aux0",
            "texture_b": "RG8_aux12" if channels == 3 else "logical_RGB8_aux123",
            "aux_channels": channels,
            "logical_channels": 3 + channels,
            "theoretical_raw_bytes_no_mips": 2048 * 2048 * (3 + channels),
            "physical_ceiling_bytes": 2048 * 2048 * (4 + (2 if channels == 3 else 4)),
            "texture_samples": 2,
        },
        "analysis": dict(config["analysis"]),
        "scopes": dict(config["scopes"]),
        "output_dir": (candidate_dir / "analysis").relative_to(ROOT).as_posix(),
    }


def run(config_path: Path, verify_twice: bool) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping) or config.get("schema_version") != 1:
        raise ValueError("unsupported causal Hybrid Phase-0 config")
    if not verify_twice:
        raise RuntimeError("causal Hybrid Phase 0 requires --verify-twice")
    core4_dir = _repo_path(config["inputs"]["core4_dir"], "inputs.core4_dir")
    output_dir = _repo_path(config["output_dir"], "output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = load_core4_targets(core4_dir, "cpu")
    if [targets.height, targets.width] != list(config["expected"]["atlas_hw"]):
        raise ValueError("Core-4 atlas shape differs from causal Hybrid contract")
    init_config = config["initialization"]
    partitions = deterministic_tile_partitions(
        targets.height, targets.width,
        tile_size=int(init_config["tile_size"]), seed=int(init_config["split_seed"]),
    )
    optimizer = partitions["optimizer"]
    optimizer_ids = np.flatnonzero(optimizer.reshape(-1))
    rng = np.random.default_rng(int(init_config["gradient_probe_seed"]))
    probe_ids_np = rng.choice(optimizer_ids, size=int(init_config["gradient_probe_count"]), replace=False)
    probe_ids = torch.from_numpy(probe_ids_np.astype(np.int64))
    reference = {
        "base_color_linear": targets.base_color_linear.reshape(targets.height, targets.width, 3).numpy(),
        "normal_xyz": targets.normal_xyz.reshape(targets.height, targets.width, 3).numpy(),
        "roughness_linear": targets.roughness.reshape(targets.height, targets.width).numpy(),
        "metallic_linear": targets.metallic.reshape(targets.height, targets.width).numpy(),
    }

    first: dict[str, CausalHybridInitialization] = {}
    content: dict[str, Any] = {}
    for candidate in config["candidates"]:
        initialization = deterministic_causal_initialization(
            targets, optimizer, candidate,
            epsilon=float(init_config["inverse_postprocess_epsilon"]),
        )
        first[candidate] = initialization
        gradient = None
        if initialization.decoder is not None:
            gradient = gradient_conflict_report(
                initialization.decoder,
                initialization.auxiliary_latent.reshape(-1, initialization.auxiliary_latent.shape[-1])[probe_ids],
                targets.select(probe_ids),
            )
        errors = material_error_maps(reference, _mapping(initialization))
        content[candidate] = _content_report(initialization, gradient, errors)
    first_digest = hashlib.sha256(deterministic_json(content).encode("utf-8")).hexdigest()
    repeated: dict[str, Any] = {}
    for candidate in config["candidates"]:
        initialization = deterministic_causal_initialization(
            targets, optimizer, candidate,
            epsilon=float(init_config["inverse_postprocess_epsilon"]),
        )
        gradient = None
        if initialization.decoder is not None:
            gradient = gradient_conflict_report(
                initialization.decoder,
                initialization.auxiliary_latent.reshape(-1, initialization.auxiliary_latent.shape[-1])[probe_ids],
                targets.select(probe_ids),
            )
        repeated[candidate] = _content_report(
            initialization, gradient, material_error_maps(reference, _mapping(initialization))
        )
    second_digest = hashlib.sha256(deterministic_json(repeated).encode("utf-8")).hexdigest()
    if content != repeated or first_digest != second_digest:
        raise RuntimeError(f"causal Hybrid initialization runs disagree: {[first_digest, second_digest]}")

    files: dict[str, Any] = {}
    for candidate, initialization in first.items():
        candidate_dir = output_dir / candidate
        candidate_dir.mkdir(parents=True, exist_ok=True)
        channels = initialization.auxiliary_latent.shape[-1]
        texture_a = candidate_dir / "texture_a_base_rgb_aux0_rgba8.png"
        texture_b = candidate_dir / ("texture_b_aux12_rg8.png" if channels == 3 else "texture_b_aux123_rgb8.png")
        packing = export_hybrid_textures(
            initialization.direct_base_linear, initialization.auxiliary_latent, texture_a, texture_b
        )
        decoder_path = candidate_dir / "decoder_weights.npz"
        if initialization.decoder is None:
            np.savez(decoder_path, **{"direct_semantic.marker": np.asarray([1.0], dtype=np.float32)})
        else:
            np.savez(decoder_path, **{
                name: value.detach().cpu().numpy() for name, value in initialization.decoder.state_dict().items()
            })
        expected = dict(config["expected"][candidate])
        actual = {
            "logical_channels": 3 + channels,
            "raw_bytes": packing["logical_raw_bytes"],
            "parameters": content[candidate]["cost"]["parameters"],
            "weight_bytes": content[candidate]["cost"]["weight_bytes_float32"],
            "macs": content[candidate]["cost"]["macs_per_pixel"],
        }
        if actual != expected:
            raise RuntimeError(f"{candidate} cost mismatch: {actual} != {expected}")
        analysis_config = _analysis_config(config, candidate, candidate_dir, texture_a, texture_b, decoder_path)
        analysis_config_path = candidate_dir / "analysis_config.yaml"
        analysis_config_path.write_text(yaml.safe_dump(analysis_config, sort_keys=False), encoding="utf-8", newline="\n")
        files[candidate] = {
            "texture_a": {"path": texture_a.relative_to(ROOT).as_posix(), "sha256": sha256_file(texture_a)},
            "texture_b": {"path": texture_b.relative_to(ROOT).as_posix(), "sha256": sha256_file(texture_b)},
            "decoder": {"path": decoder_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(decoder_path)},
            "analysis_config": {"path": analysis_config_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(analysis_config_path)},
            "packing": packing,
        }
    report = {
        "schema_version": 1,
        "status": "phase0_initialization_complete",
        "formal_holdout_accessed": False,
        "config": {"path": config_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(config_path)},
        "core4_sha256": {
            name: sha256_file(core4_dir / filename)
            for name, filename in {"base_color": "base_color.png", "normal": "normal.png", "roughness": "roughness.png", "metallic": "metallic.png"}.items()
        },
        "partition_pixels": {name: int(mask.sum()) for name, mask in partitions.items()},
        "selection_or_validation_used_for_initialization": False,
        "determinism": {"two_runs_requested": True, "run_content_sha256": [first_digest, second_digest], "identical": True},
        "candidates": content,
        "files": files,
        "runtime_contract": {
            "base_color": "linear RGB direct bilinear; no decoder, sigmoid, or additional sRGB transform",
            "auxiliary": "UNORM8 texels -> identical bilinear samples -> causal auxiliary path -> one postprocess",
            "texture_samples": 2,
        },
    }
    report_path = output_dir / "initialization_report.json"
    report_path.write_text(deterministic_json(report), encoding="utf-8", newline="\n")
    print(json.dumps({"status": report["status"], "report_sha256": sha256_file(report_path), "deterministic": True}, ensure_ascii=False))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/eval/scifihelmet_hybrid_factorization_phase0.yaml")
    parser.add_argument("--verify-twice", action="store_true")
    args = parser.parse_args()
    run(args.config.resolve(), args.verify_twice)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
