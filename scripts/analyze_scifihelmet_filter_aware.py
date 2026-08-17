"""Deterministic fixed-ROI diagnosis for the filter-aware decoder task."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
from PIL import Image
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from analyze_scifihelmet_interpolation import (  # noqa: E402
    _decode_texels,
    _load_decoder,
    _load_reference,
    _runtime_material,
    _sample_material,
    _sampler_contract,
)
from cg_frontier.compression.artifact_analysis import (  # noqa: E402
    deterministic_json,
    metallic_boundary_mask,
    sha256_file,
    tail_statistics,
)
from cg_frontier.compression.filter_aware import (  # noqa: E402
    commutativity_statistics,
    component_rectangularity,
    dilate_mask,
    halo_statistics,
    replacement_oracle_statistics,
)
from cg_frontier.compression.interpolation_analysis import (  # noqa: E402
    LUMINANCE_WEIGHTS,
    activation_crossings,
    bilinear_footprint_top_down_wrap,
    normal_angular_error,
)


def _repo_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty repository-relative path")
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT) or "formal_holdout" in path.as_posix().lower():
        raise ValueError(f"forbidden path for {label}")
    return path


def _luminance(base: np.ndarray) -> np.ndarray:
    return np.sum(np.asarray(base, dtype=np.float32) * LUMINANCE_WEIGHTS, axis=-1, dtype=np.float32)


def _normal_mapping(material: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {name: np.asarray(value) for name, value in material.items()}


def _reference_boundary_masks(
    reference_texture: Mapping[str, np.ndarray], config: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    luma = _luminance(reference_texture["base_color_linear"])
    gradient = np.zeros(luma.shape, dtype=np.float32)
    horizontal = np.abs(luma[:, 1:] - luma[:, :-1])
    vertical = np.abs(luma[1:, :] - luma[:-1, :])
    gradient[:, 1:] = np.maximum(gradient[:, 1:], horizontal)
    gradient[:, :-1] = np.maximum(gradient[:, :-1], horizontal)
    gradient[1:, :] = np.maximum(gradient[1:, :], vertical)
    gradient[:-1, :] = np.maximum(gradient[:-1, :], vertical)
    base_edge = gradient > float(config["analysis"]["luminance_boundary_threshold"])
    metal_edge = metallic_boundary_mask(
        reference_texture["metallic_linear"],
        float(config["analysis"]["metallic_boundary_threshold"]),
    )
    band = dilate_mask(
        base_edge | metal_edge, radius=int(config["analysis"]["boundary_dilation_radius"])
    )
    return band, metal_edge


def _sample_roi(
    bbox: list[int], count: int, seed: int, atlas_hw: tuple[int, int]
) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    height, width = atlas_hw
    rng = np.random.default_rng(seed)
    x = rng.uniform(x0, x1, size=count).astype(np.float32)
    y = rng.uniform(y0, y1, size=count).astype(np.float32)
    return np.stack((x / np.float32(width), y / np.float32(height)), axis=-1)


def _fixed_roi_uv(
    bbox: list[int], phase: list[float], atlas_hw: tuple[int, int]
) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    height, width = atlas_hw
    x = (np.arange(x0, x1, dtype=np.float32) + np.float32(phase[0])) / np.float32(width)
    y = (np.arange(y0, y1, dtype=np.float32) + np.float32(phase[1])) / np.float32(height)
    grid_y, grid_x = np.meshgrid(y, x, indexing="ij")
    return np.stack((grid_x, grid_y), axis=-1).reshape(-1, 2)


def _metrics(
    reference: Mapping[str, np.ndarray],
    runtime: Mapping[str, np.ndarray],
    filtered: Mapping[str, np.ndarray],
    boundary_band: np.ndarray,
    metallic_boundary: np.ndarray,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    ref_luma = _luminance(reference["base_color_linear"])
    run_luma = _luminance(runtime["base_color_linear"])
    filter_luma = _luminance(filtered["base_color_linear"])
    threshold = float(config["analysis"]["luminance_threshold"])
    ratio = float(config["analysis"]["dark_ratio"])
    eligible = ref_luma > threshold
    run_dark = run_luma < ref_luma * ratio
    filter_dark = filter_luma < ref_luma * ratio
    novel = eligible & run_dark & ~filter_dark
    under_runtime = np.maximum(ref_luma - run_luma, 0.0)
    under_filter = np.maximum(ref_luma - filter_luma, 0.0)
    base_error = np.max(
        np.abs(np.asarray(runtime["base_color_linear"]) - np.asarray(reference["base_color_linear"])), axis=-1
    )
    metallic_error = np.abs(
        np.asarray(runtime["metallic_linear"]) - np.asarray(reference["metallic_linear"])
    )
    filter_base_error = np.max(
        np.abs(np.asarray(filtered["base_color_linear"]) - np.asarray(reference["base_color_linear"])), axis=-1
    )
    filter_normal_error = normal_angular_error(reference["normal_xyz"], filtered["normal_xyz"])
    filter_roughness_error = np.abs(
        np.asarray(filtered["roughness_linear"]) - np.asarray(reference["roughness_linear"])
    )
    filter_metallic_error = np.abs(
        np.asarray(filtered["metallic_linear"]) - np.asarray(reference["metallic_linear"])
    )
    band_selected = np.asarray(boundary_band, dtype=bool)
    metallic_selected = np.asarray(metallic_boundary, dtype=bool)
    if not np.any(band_selected) or not np.any(metallic_selected):
        raise RuntimeError("fixed ROI boundary band is empty")
    halo = halo_statistics(
        ref_luma,
        run_luma,
        band_selected,
        threshold=float(config["analysis"]["halo_threshold"]),
    )
    return {
        "reference": {"luminance": tail_statistics(ref_luma)},
        "runtime": {
            "luminance": tail_statistics(run_luma),
            "luminance_underprediction": tail_statistics(under_runtime),
            "base_color_max_channel": tail_statistics(base_error),
        },
        "decode_then_filter": {
            "luminance": tail_statistics(filter_luma),
            "luminance_underprediction": tail_statistics(under_filter),
            "base_color_max_channel": tail_statistics(filter_base_error),
            "normal_degrees": tail_statistics(filter_normal_error),
            "roughness_absolute_error": tail_statistics(filter_roughness_error),
            "metallic_absolute_error": tail_statistics(filter_metallic_error),
            "metallic_boundary": {
                "count": int(metallic_selected.sum()),
                "mae": float(filter_metallic_error[metallic_selected].mean(dtype=np.float64)),
                "fraction_above_0_1": float(np.mean(filter_metallic_error[metallic_selected] > 0.1)),
            },
            "boundary_band_luminance": halo_statistics(
                ref_luma,
                filter_luma,
                band_selected,
                threshold=float(config["analysis"]["halo_threshold"]),
            ),
        },
        "dark": {
            "eligible_count": int(eligible.sum()),
            "runtime_dark_fraction": float(np.mean(run_dark[eligible])) if np.any(eligible) else 0.0,
            "filter_dark_fraction": float(np.mean(filter_dark[eligible])) if np.any(eligible) else 0.0,
            "novel_dark_fraction": float(np.mean(novel[eligible])) if np.any(eligible) else 0.0,
        },
        "metallic_boundary": {
            "count": int(metallic_selected.sum()),
            "mae": float(metallic_error[metallic_selected].mean(dtype=np.float64)),
            "fraction_above_0_1": float(np.mean(metallic_error[metallic_selected] > 0.1)),
        },
        "boundary_band_luminance": halo,
        "postprocess_commutativity_gap": commutativity_statistics(runtime, filtered),
        "reference_channel_replacement_oracles": _replacement_artifact_oracles(
            reference,
            runtime,
            filtered,
            band_selected,
            config,
        ),
        "decode_then_filter_oracle": replacement_oracle_statistics(reference, filtered)["runtime"],
    }


def _replacement_artifact_oracles(
    reference: Mapping[str, np.ndarray],
    runtime: Mapping[str, np.ndarray],
    filtered: Mapping[str, np.ndarray],
    boundary_band: np.ndarray,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    report = replacement_oracle_statistics(reference, runtime)
    ref_luma = _luminance(reference["base_color_linear"])
    filter_luma = _luminance(filtered["base_color_linear"])
    threshold = float(config["analysis"]["luminance_threshold"])
    ratio = float(config["analysis"]["dark_ratio"])
    eligible = ref_luma > threshold
    filter_dark = filter_luma < ref_luma * ratio
    channel_names = ("base_color_linear", "normal_xyz", "roughness_linear", "metallic_linear")
    for channel in channel_names:
        base = (
            np.asarray(reference["base_color_linear"])
            if channel == "base_color_linear"
            else np.asarray(runtime["base_color_linear"])
        )
        luma = _luminance(base)
        novel = eligible & (luma < ref_luma * ratio) & ~filter_dark
        report["replacements"][channel]["novel_dark_fraction"] = (
            float(np.mean(novel[eligible])) if np.any(eligible) else 0.0
        )
        report["replacements"][channel]["boundary_band_luminance"] = halo_statistics(
            ref_luma,
            luma,
            boundary_band,
            threshold=float(config["analysis"]["halo_threshold"]),
        )
    return report


def _activation_report(
    latent: np.ndarray,
    arrays: Mapping[str, np.ndarray],
    uv: np.ndarray,
    novel: np.ndarray,
) -> dict[str, Any]:
    crossing = activation_crossings(
        latent, uv, arrays["network.0.weight"], arrays["network.0.bias"]
    )
    novel_flat = np.asarray(novel, dtype=bool).reshape(-1)
    any_cross = np.any(crossing, axis=-1)
    units = {}
    for unit in range(crossing.shape[1]):
        selected = crossing[:, unit]
        joint = int(np.sum(selected & novel_flat))
        units[str(unit)] = {
            "crossing_count": int(selected.sum()),
            "novel_dark_crossing_count": joint,
            "novel_dark_coverage": float(joint / novel_flat.sum()) if np.any(novel_flat) else 0.0,
        }
    return {
        "any_crossing_fraction": float(np.mean(any_cross)),
        "novel_dark_with_any_crossing_fraction": float(np.mean(any_cross[novel_flat])) if np.any(novel_flat) else 0.0,
        "units": units,
    }


def run(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != 1 or "formal_holdout" in config_path.as_posix().lower():
        raise ValueError("unsupported or forbidden Phase-0 config")
    inputs = config["inputs"]
    latent_path = _repo_path(inputs["latent_hard_png"], "inputs.latent_hard_png")
    decoder_path = _repo_path(inputs["decoder_npz"], "inputs.decoder_npz")
    actual_hashes = {
        "latent_hard_png": sha256_file(latent_path),
        "decoder_npz": sha256_file(decoder_path),
    }
    if actual_hashes != dict(config["frozen_sha256"]):
        raise ValueError("frozen baseline hash mismatch")
    with Image.open(latent_path) as image:
        latent_u8 = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    latent = latent_u8.astype(np.float32) / np.float32(255.0)
    expected_hw = tuple(int(value) for value in config["analysis"]["expected_atlas_hw"])
    if latent.shape[:2] != expected_hw:
        raise ValueError("latent atlas dimensions differ from the frozen contract")
    arrays = _load_decoder(decoder_path)
    reference_texture = _load_reference(_repo_path(inputs["core4_dir"], "inputs.core4_dir"))
    decoded_texels = _decode_texels(
        latent, arrays, int(config["analysis"]["decode_chunk_size"])
    )
    full_boundary_band, full_metallic_boundary = _reference_boundary_masks(reference_texture, config)
    rois: dict[str, Any] = {}
    for roi_index, (name, anchor) in enumerate(config["anchors"].items()):
        bbox = [int(value) for value in anchor["atlas_roi_xyxy"]]
        uv = _sample_roi(
            bbox,
            int(config["analysis"]["seeded_probe_count_per_roi"]),
            int(config["analysis"]["random_seed"]) + roi_index,
            expected_hw,
        )
        reference = _sample_material(reference_texture, uv)
        runtime = _runtime_material(latent, arrays, uv)
        filtered = _sample_material(decoded_texels, uv)
        _, _, base_xy = bilinear_footprint_top_down_wrap(expected_hw, uv)
        boundary_band = full_boundary_band[base_xy[:, 1], base_xy[:, 0]]
        metallic_boundary = full_metallic_boundary[base_xy[:, 1], base_xy[:, 0]]
        metrics = _metrics(
            reference,
            runtime,
            filtered,
            boundary_band,
            metallic_boundary,
            config,
        )
        ref_luma = _luminance(reference["base_color_linear"])
        run_luma = _luminance(runtime["base_color_linear"])
        filter_luma = _luminance(filtered["base_color_linear"])
        novel = (
            (ref_luma > float(config["analysis"]["luminance_threshold"]))
            & (run_luma < ref_luma * float(config["analysis"]["dark_ratio"]))
            & ~(filter_luma < ref_luma * float(config["analysis"]["dark_ratio"]))
        )
        metrics["relu_activation_crossing"] = _activation_report(latent, arrays, uv, novel)
        components = []
        filter_components = []
        x0, y0, x1, y1 = bbox
        for phase in config["analysis"]["fixed_phases_xy"]:
            phase_uv = _fixed_roi_uv(bbox, phase, expected_hw)
            phase_reference = _sample_material(reference_texture, phase_uv)
            phase_runtime = _runtime_material(latent, arrays, phase_uv)
            phase_filtered = _sample_material(decoded_texels, phase_uv)
            phase_ref_luma = _luminance(phase_reference["base_color_linear"])
            phase_run_luma = _luminance(phase_runtime["base_color_linear"])
            phase_filter_luma = _luminance(phase_filtered["base_color_linear"])
            phase_novel = (
                (phase_ref_luma > float(config["analysis"]["luminance_threshold"]))
                & (phase_run_luma < phase_ref_luma * float(config["analysis"]["dark_ratio"]))
                & ~(phase_filter_luma < phase_ref_luma * float(config["analysis"]["dark_ratio"]))
            ).reshape(y1 - y0, x1 - x0)
            phase_filter_dark = (
                (phase_ref_luma > float(config["analysis"]["luminance_threshold"]))
                & (phase_filter_luma < phase_ref_luma * float(config["analysis"]["dark_ratio"]))
            ).reshape(y1 - y0, x1 - x0)
            components.append(
                {
                    "phase_xy": [float(phase[0]), float(phase[1])],
                    **component_rectangularity(
                        phase_novel,
                        rectangularity_threshold=float(config["analysis"]["rectangularity_threshold"]),
                    ),
                }
            )
            filter_components.append(
                {
                    "phase_xy": [float(phase[0]), float(phase[1])],
                    **component_rectangularity(
                        phase_filter_dark,
                        rectangularity_threshold=float(config["analysis"]["rectangularity_threshold"]),
                    ),
                }
            )
        metrics["connected_dark_components"] = {
            "per_phase": components,
            "max_area": max(item["max_area"] for item in components),
            "rectangular_component_max_area": max(
                item["rectangular_component_max_area"] for item in components
            ),
        }
        metrics["decode_then_filter_dark_components"] = {
            "per_phase": filter_components,
            "max_area": max(item["max_area"] for item in filter_components),
            "rectangular_component_max_area": max(
                item["rectangular_component_max_area"] for item in filter_components
            ),
        }
        rois[name] = {
            "anchor": anchor,
            "seeded_probe_count": int(uv.shape[0]),
            "metrics": metrics,
        }
    sampler = _sampler_contract(latent_u8)
    if not sampler["passed"]:
        raise RuntimeError("sampler/UV contract failed")
    return {
        "schema_version": 1,
        "scope": "scifihelmet_filter_aware_phase0_fixed_rois",
        "formal_holdout_accessed": False,
        "inputs": {
            "latent_hard_png": {"path": inputs["latent_hard_png"], "sha256": actual_hashes["latent_hard_png"]},
            "decoder_npz": {"path": inputs["decoder_npz"], "sha256": actual_hashes["decoder_npz"]},
            "config": {"path": config_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(config_path)},
        },
        "runtime_contract": "RGBA8 texel -> one bilinear latent sample -> 4->8->7 ReLU decoder -> one postprocess",
        "decode_then_filter_oracle_contract": "four RGBA8 texels -> four 4->8->7 decodes -> four postprocess operations -> bilinear material filter; structural oracle only",
        "sampler_uv_contract": sampler,
        "analysis_contract": config["analysis"],
        "rois": rois,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/eval/scifihelmet_filter_aware_phase0.yaml"
    )
    parser.add_argument("--verify-twice", action="store_true")
    args = parser.parse_args()
    if not args.verify_twice:
        raise RuntimeError("Phase 0 requires --verify-twice")
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir = _repo_path(config["output_dir"], "output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "phase0_report.json"
    first = run(config_path)
    first_bytes = deterministic_json(first).encode("utf-8")
    report_path.write_bytes(first_bytes)
    first_hash = sha256_file(report_path)
    second = run(config_path)
    second_bytes = deterministic_json(second).encode("utf-8")
    if first_bytes != second_bytes:
        raise RuntimeError("two Phase-0 reports are not byte-identical")
    report_path.write_bytes(second_bytes)
    second_hash = sha256_file(report_path)
    if first_hash != second_hash:
        raise RuntimeError("two Phase-0 report hashes disagree")
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "formal_holdout_accessed": False,
        "report": {"path": "phase0_report.json", "sha256": first_hash},
        "determinism": {"run_sha256": [first_hash, second_hash], "byte_identical": True},
        "sampler_uv_contract_passed": True,
        "gpu_started": False,
    }
    (output_dir / "manifest.json").write_text(
        deterministic_json(manifest), encoding="utf-8", newline="\n"
    )
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
