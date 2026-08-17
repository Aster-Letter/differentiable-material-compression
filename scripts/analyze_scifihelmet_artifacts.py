"""Analyze frozen SciFiHelmet material artifacts without opening formal holdout data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any, Mapping

import numpy as np
from PIL import Image
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cg_frontier.compression.artifact_analysis import (  # noqa: E402
    bilinear_sample_float_top_down_wrap,
    connected_patch_statistics,
    cross_channel_error_correlation,
    deterministic_json,
    deterministic_tile_partitions,
    low_gradient_mask,
    material_error_maps,
    metallic_boundary_mask,
    roi_mask,
    roi_material_metrics,
    sha256_file,
    tail_statistics,
    worst_tiles,
)
from cg_frontier.compression.material import (  # noqa: E402
    MaterialDecoder,
    decode_material,
    load_core4_targets,
)
from cg_frontier.compression.render_loss import load_latent_unorm8_png  # noqa: E402
from cg_frontier.compression.ue_export import (  # noqa: E402
    bilinear_sample_top_down_wrap as bilinear_sample_u8,
    decoder_postprocess,
    decoder_raw_forward,
)


def _repo_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"config field {label} must be a non-empty path")
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"config field {label} escapes the repository")
    if "formal_holdout" in path.as_posix().lower():
        raise ValueError(f"sealed formal holdout path is forbidden: {label}")
    return path


def _load_decoder(path: Path) -> MaterialDecoder:
    decoder = MaterialDecoder("tiny_mlp")
    with np.load(path, allow_pickle=False) as stored:
        state = {name: torch.from_numpy(np.asarray(stored[name])) for name in stored.files}
    decoder.load_state_dict(state)
    decoder.eval()
    return decoder


@torch.no_grad()
def _decode_image(latent: torch.Tensor, decoder: MaterialDecoder, chunk: int) -> dict[str, np.ndarray]:
    height, width = latent.shape[:2]
    flat = latent.reshape(-1, 4)
    pieces: dict[str, list[np.ndarray]] = {
        "base_color_linear": [],
        "normal_xyz": [],
        "roughness_linear": [],
        "metallic_linear": [],
    }
    for start in range(0, flat.shape[0], chunk):
        decoded = decode_material(decoder, flat[start : start + chunk])
        pieces["base_color_linear"].append(decoded.base_color_linear.numpy())
        pieces["normal_xyz"].append(decoded.normal_xyz.numpy())
        pieces["roughness_linear"].append(decoded.roughness.numpy()[..., 0])
        pieces["metallic_linear"].append(decoded.metallic.numpy()[..., 0])
    return {
        name: np.concatenate(values, axis=0).reshape(height, width, -1 if name in ("base_color_linear", "normal_xyz") else 1)[..., :]
        if name in ("base_color_linear", "normal_xyz")
        else np.concatenate(values, axis=0).reshape(height, width)
        for name, values in pieces.items()
    }


def _reference(core4_dir: Path) -> dict[str, np.ndarray]:
    targets = load_core4_targets(core4_dir)
    return {
        "base_color_linear": targets.base_color_linear.numpy().reshape(targets.height, targets.width, 3),
        "normal_xyz": targets.normal_xyz.numpy().reshape(targets.height, targets.width, 3),
        "roughness_linear": targets.roughness.numpy().reshape(targets.height, targets.width),
        "metallic_linear": targets.metallic.numpy().reshape(targets.height, targets.width),
    }


def _save_error_png(path: Path, values: np.ndarray, gain: float) -> None:
    encoded = np.floor(np.clip(values * gain, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    Image.fromarray(encoded, mode="L").save(path, format="PNG")


def _copy_screenshot_evidence(paths: list[str], output_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    evidence_dir = output_dir / "screenshots"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    for index, value in enumerate(paths, start=1):
        source = Path(value)
        if not source.is_file():
            records.append({"source": value, "status": "missing_recreate_in_ue"})
            continue
        destination = evidence_dir / f"defect_{index}_{source.name}"
        shutil.copyfile(source, destination)
        records.append(
            {
                "source": value,
                "status": "copied",
                "file": destination.relative_to(output_dir).as_posix(),
                "sha256": sha256_file(destination),
            }
        )
    return records


def run(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping) or config.get("schema_version") != 1:
        raise ValueError("unsupported artifact-analysis config schema")
    inputs = config["inputs"]
    output_dir = _repo_path(config["output_dir"], "output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    core4_dir = _repo_path(inputs["core4_dir"], "inputs.core4_dir")
    latent_float_path = _repo_path(inputs["latent_float"], "inputs.latent_float")
    latent_hard_path = _repo_path(inputs["latent_hard_png"], "inputs.latent_hard_png")
    decoder_path = _repo_path(inputs["decoder_npz"], "inputs.decoder_npz")
    expected_hashes = config["frozen_sha256"]
    if sha256_file(latent_hard_path) != expected_hashes["latent_hard_png"]:
        raise ValueError("frozen latent hash mismatch")
    if sha256_file(decoder_path) != expected_hashes["decoder_npz"]:
        raise ValueError("frozen decoder hash mismatch")

    reference = _reference(core4_dir)
    decoder = _load_decoder(decoder_path)
    latent_float = torch.from_numpy(np.load(latent_float_path, allow_pickle=False)).to(torch.float32)
    _, latent_hard = load_latent_unorm8_png(latent_hard_path)
    chunk = int(config["analysis"]["decode_chunk_size"])
    candidates = {
        "float": _decode_image(latent_float, decoder, chunk),
        "hard": _decode_image(latent_hard, decoder, chunk),
    }
    height, width = reference["metallic_linear"].shape
    partitions = deterministic_tile_partitions(
        height,
        width,
        tile_size=int(config["analysis"]["tile_size"]),
        seed=int(config["analysis"]["split_seed"]),
    )
    edges = metallic_boundary_mask(
        reference["metallic_linear"], float(config["analysis"]["metallic_edge_threshold"])
    )
    smooth_base = low_gradient_mask(reference["base_color_linear"], 0.01)
    roi_definitions = {
        name: list(values) for name, values in config["rois_xyxy"].items()
    }
    roi_masks = {name: roi_mask((height, width), values) for name, values in roi_definitions.items()}

    candidate_reports: dict[str, Any] = {}
    generated: dict[str, str] = {}
    error_dir = output_dir / "error_maps"
    error_dir.mkdir(parents=True, exist_ok=True)
    for branch, candidate in candidates.items():
        errors = material_error_maps(reference, candidate)
        branch_report: dict[str, Any] = {
            "global": {name: tail_statistics(values) for name, values in errors.items()},
            "metallic_boundary": {
                **tail_statistics(errors["metallic"], edges),
                "fraction_above_0_1": float(np.mean(errors["metallic"][edges] > 0.1)),
            },
            "rois": {
                name: roi_material_metrics(reference, candidate, mask, metallic_edges=edges)
                for name, mask in roi_masks.items()
            },
            "cross_channel_error_correlation": cross_channel_error_correlation(errors),
            "low_gradient_base_color": tail_statistics(errors["base_color_max_channel"], smooth_base),
            "worst_tiles": {
                name: worst_tiles(
                    values,
                    tile_size=int(config["analysis"]["tile_size"]),
                    limit=int(config["analysis"]["worst_tile_count"]),
                )
                for name, values in errors.items()
            },
        }
        truth_luma = reference["base_color_linear"] @ np.asarray([0.2126, 0.7152, 0.0722])
        candidate_luma = candidate["base_color_linear"] @ np.asarray([0.2126, 0.7152, 0.0722])
        black_spots = (truth_luma > 1.0e-6) & (candidate_luma < truth_luma * 0.5)
        branch_report["black_spots"] = {
            "fraction": float(np.mean(black_spots)),
            **connected_patch_statistics(black_spots),
        }
        for name, values in errors.items():
            path = error_dir / f"{branch}_{name}.png"
            gain = 1.0 / 180.0 if name == "normal_degrees" else float(config["analysis"]["error_display_gain"])
            _save_error_png(path, values, gain)
            generated[path.relative_to(output_dir).as_posix()] = sha256_file(path)
        candidate_reports[branch] = branch_report

    with np.load(decoder_path, allow_pickle=False) as stored:
        decoder_arrays = {name: np.asarray(stored[name]).copy() for name in stored.files}
    encoded_hard = np.asarray(Image.open(latent_hard_path).convert("RGBA"), dtype=np.uint8)
    axis = (np.arange(64, dtype=np.float32) + np.float32(0.25)) / np.float32(64.0)
    grid_y, grid_x = np.meshgrid(axis, axis, indexing="ij")
    probe_uv = np.stack((grid_x, grid_y), axis=-1)
    sampled_rgba = bilinear_sample_u8(encoded_hard, probe_uv)
    correct_raw = decoder_raw_forward(sampled_rgba, decoder_arrays)
    correct_post = decoder_postprocess(correct_raw)
    decoded_then_sampled = {
        name: bilinear_sample_float_top_down_wrap(
            values if values.ndim == 3 else values[..., None], probe_uv
        )
        for name, values in candidates["hard"].items()
    }
    report_sampling = {
        "probe_count": int(probe_uv.shape[0] * probe_uv.shape[1]),
        "correct_order": "RGBA8 texel -> bilinear RGBA -> decoder -> postprocess",
        "decode_before_bilinear_is_not_equivalent": {
            "base_color_max_abs": float(np.max(np.abs(correct_post["base_color_linear"] - decoded_then_sampled["base_color_linear"]))),
            "normal_max_abs": float(np.max(np.abs(correct_post["normal_tangent_gltf_positive_y"] - decoded_then_sampled["normal_xyz"]))),
            "roughness_max_abs": float(np.max(np.abs(correct_post["roughness_linear"] - decoded_then_sampled["roughness_linear"]))),
            "metallic_max_abs": float(np.max(np.abs(correct_post["metallic_linear"] - decoded_then_sampled["metallic_linear"]))),
        },
    }

    split_path = output_dir / "tile_partitions.npz"
    np.savez_compressed(split_path, **partitions, metallic_boundary=edges)
    generated[split_path.relative_to(output_dir).as_posix()] = sha256_file(split_path)
    report: dict[str, Any] = {
        "schema_version": 1,
        "scope": "material_domain_tail_and_spatial_baseline",
        "formal_holdout_accessed": False,
        "inputs": {
            "core4_dir": inputs["core4_dir"],
            "latent_float": {"path": inputs["latent_float"], "sha256": sha256_file(latent_float_path)},
            "latent_hard_png": {"path": inputs["latent_hard_png"], "sha256": sha256_file(latent_hard_path)},
            "decoder_npz": {"path": inputs["decoder_npz"], "sha256": sha256_file(decoder_path)},
        },
        "decoder": {"architecture": "4->8->7", "parameters": 103, "bytes": 412, "macs_per_pixel": 88},
        "runtime_order": "RGBA8 texel -> bilinear RGBA -> per-pixel decoder -> one channel postprocess",
        "tile_partitions": {name: int(mask.sum()) for name, mask in partitions.items()},
        "rois_xyxy": roi_definitions,
        "metallic_boundary_pixels": int(edges.sum()),
        "sampling_probe": report_sampling,
        "candidates": candidate_reports,
        "screenshot_evidence": _copy_screenshot_evidence(list(config.get("screenshot_sources", [])), output_dir),
        "generated_files": generated,
    }
    report_path = output_dir / "baseline_analysis.json"
    report_path.write_text(deterministic_json(report), encoding="utf-8", newline="\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/eval/scifihelmet_repair.yaml")
    args = parser.parse_args()
    report = run(args.config.resolve())
    print(json.dumps({"status": "complete", "candidates": sorted(report["candidates"]), "formal_holdout_accessed": False}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
