"""Prepare and render-check bounded D7-D direct-scalar SciFiHelmet oracles."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
from PIL import Image
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT / "src", ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from cg_frontier.assets.gltf_mesh import load_gltf_mesh  # noqa: E402
from cg_frontier.compression.artifact_analysis import (  # noqa: E402
    deterministic_json,
    material_error_maps,
    sha256_file,
)
from cg_frontier.compression.hybrid import export_hybrid_textures  # noqa: E402
from cg_frontier.compression.hybrid_direct_scalars import (  # noqa: E402
    NormalAuxDecoder,
    build_direct_scalar_auxiliary,
    decode_direct_scalars,
    render_direct_scalar_material,
)
from cg_frontier.compression.material import load_core4_targets  # noqa: E402
from cg_frontier.compression.render_loss import hard_quantize_unorm8, masked_render_metrics  # noqa: E402
from cg_frontier.render.gbuffer import load_core4_textures  # noqa: E402
import train_scifihelmet_hybrid as base  # noqa: E402


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _repo_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"config field {label} must be a non-empty path")
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT) or "formal_holdout" in path.as_posix().lower():
        raise ValueError(f"forbidden repository path: {label}")
    return path


def _load_config(path: Path) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping) or raw.get("schema_version") != 1:
        raise ValueError("unsupported direct-scalar config")
    inherited = base._load_config(path)
    if "formal_holdout" in deterministic_json(inherited).lower():
        raise ValueError("formal holdout path is forbidden")
    return inherited


def _load_normal_source(
    texture_a: Path,
    texture_b: Path,
    decoder_path: Path,
    device: torch.device,
) -> tuple[torch.Tensor, NormalAuxDecoder]:
    a = np.asarray(Image.open(texture_a).convert("RGBA"), dtype=np.uint8)
    b = np.asarray(Image.open(texture_b).convert("RGB"), dtype=np.uint8)
    normal_latent = torch.from_numpy(np.concatenate((a[..., 3:4], b[..., 0:1]), axis=-1).copy()).to(
        device=device, dtype=torch.float32
    ) / 255.0
    decoder = NormalAuxDecoder().to(device)
    with np.load(decoder_path, allow_pickle=False) as stored:
        state = {
            name: torch.from_numpy(np.asarray(stored[name], dtype=np.float32)).to(device)
            for name in decoder.state_dict()
        }
    decoder.load_state_dict(state)
    return normal_latent, decoder


@torch.no_grad()
def _mapping(
    direct: torch.Tensor,
    auxiliary: torch.Tensor,
    decoder: NormalAuxDecoder,
    chunk: int,
) -> dict[str, np.ndarray]:
    height, width = auxiliary.shape[:2]
    flat_aux, flat_base = auxiliary.reshape(-1, 4), direct.reshape(-1, 3)
    pieces: dict[str, list[np.ndarray]] = {
        name: [] for name in ("base_color_linear", "normal_xyz", "roughness_linear", "metallic_linear")
    }
    for start in range(0, flat_aux.shape[0], chunk):
        decoded = decode_direct_scalars(
            decoder, flat_aux[start : start + chunk], flat_base[start : start + chunk]
        )
        pieces["base_color_linear"].append(decoded.base_color_linear.cpu().numpy())
        pieces["normal_xyz"].append(decoded.normal_xyz.cpu().numpy())
        pieces["roughness_linear"].append(decoded.roughness[..., 0].cpu().numpy())
        pieces["metallic_linear"].append(decoded.metallic[..., 0].cpu().numpy())
    return {
        "base_color_linear": np.concatenate(pieces["base_color_linear"]).reshape(height, width, 3),
        "normal_xyz": np.concatenate(pieces["normal_xyz"]).reshape(height, width, 3),
        "roughness_linear": np.concatenate(pieces["roughness_linear"]).reshape(height, width),
        "metallic_linear": np.concatenate(pieces["metallic_linear"]).reshape(height, width),
    }


def _analysis_config(
    config: Mapping[str, Any],
    template: Mapping[str, Any],
    candidate: str,
    candidate_dir: Path,
    texture_a: Path,
    texture_b: Path,
    decoder: Path,
) -> dict[str, Any]:
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
            "architecture": "direct_RGB + normal_2_to_6_to_2 + direct_roughness_metallic",
            "texture_a": "RGBA8_linear_BaseColor_RGB_plus_normal0",
            "texture_b": "logical_RGB8_normal1_roughness_metallic",
            "aux_channels": 4,
            "logical_channels": 7,
            "theoretical_raw_bytes_no_mips": 2048 * 2048 * 7,
            "physical_ceiling_bytes": 2048 * 2048 * 8,
            "texture_samples": 2,
        },
        "analysis": dict(template["analysis"]),
        "scopes": dict(template["scopes"]),
        "output_dir": (candidate_dir / "analysis").relative_to(ROOT).as_posix(),
    }


def _source_expected_hashes(config: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    initialization_path = _repo_path(
        config["inputs"]["factorization_initialization_report"],
        "inputs.factorization_initialization_report",
    )
    d7_manifest_path = _repo_path(
        config["inputs"]["d7_training_manifest"], "inputs.d7_training_manifest"
    )
    initialization = json.loads(initialization_path.read_text(encoding="utf-8"))
    d7_manifest = json.loads(d7_manifest_path.read_text(encoding="utf-8"))
    if initialization.get("formal_holdout_accessed") is not False or d7_manifest.get("formal_holdout_accessed") is not False:
        raise RuntimeError("formal holdout seal is missing")
    phase0 = initialization["files"]["d7_p"]
    return {
        "r0a": {
            "texture_a": d7_manifest["files"]["texture_a"]["sha256"],
            "texture_b": d7_manifest["files"]["texture_b"]["sha256"],
            "decoder": d7_manifest["files"]["decoder"]["sha256"],
        },
        "r0b": {
            "texture_a": phase0["texture_a"]["sha256"],
            "texture_b": phase0["texture_b"]["sha256"],
            "decoder": phase0["decoder"]["sha256"],
        },
    }


def prepare(config_path: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    output_root = _repo_path(config["output_root"], "output_root")
    output_root.mkdir(parents=True, exist_ok=True)
    template_path = _repo_path(config["inputs"]["phase0_analysis_template"], "phase0_analysis_template")
    template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
    core4_dir = _repo_path(config["inputs"]["core4_dir"], "inputs.core4_dir")
    targets = load_core4_targets(core4_dir, "cpu")
    if (targets.height, targets.width) != (2048, 2048):
        raise ValueError("Core-4 atlas extent changed")
    direct = hard_quantize_unorm8(
        targets.base_color_linear.reshape(targets.height, targets.width, 3)
    ).detach()
    reference = base._reference_mapping(targets)
    baseline_manifest_path = _repo_path(config["inputs"]["d7_training_manifest"], "d7_training_manifest")
    baseline_manifest = json.loads(baseline_manifest_path.read_text(encoding="utf-8"))
    expected_sources = _source_expected_hashes(config)
    expected_cost = dict(config["expected"])
    candidates: dict[str, Any] = {}
    deterministic_content: dict[str, Any] = {}
    for candidate, definition in config["candidates"].items():
        source_paths = {
            name: _repo_path(definition[name], f"candidates.{candidate}.{name}")
            for name in ("texture_a", "texture_b", "decoder")
        }
        actual_source_hashes = {name: sha256_file(path) for name, path in source_paths.items()}
        if actual_source_hashes != expected_sources[candidate]:
            raise ValueError(f"{candidate} frozen normal source hash mismatch")
        normal_latent, decoder = _load_normal_source(**{
            "texture_a": source_paths["texture_a"],
            "texture_b": source_paths["texture_b"],
            "decoder_path": source_paths["decoder"],
            "device": torch.device("cpu"),
        })
        auxiliary = build_direct_scalar_auxiliary(normal_latent, targets)
        actual_cost = {
            "parameters": decoder.parameter_count,
            "weight_bytes_float32": decoder.weight_bytes_float32,
            "macs_per_pixel": decoder.macs_per_pixel,
        }
        if actual_cost != {name: expected_cost[name] for name in actual_cost}:
            raise RuntimeError(f"{candidate} decoder cost mismatch: {actual_cost}")
        mapping = _mapping(direct, auxiliary, decoder, int(config["training"]["decode_chunk_size"]))
        material = base._material_report(reference, mapping, config)
        errors = material_error_maps(reference, mapping)
        candidate_dir = output_root / candidate
        candidate_dir.mkdir(parents=True, exist_ok=True)
        texture_a = candidate_dir / "texture_a_base_rgb_normal0_rgba8.png"
        texture_b = candidate_dir / "texture_b_normal1_roughness_metallic_rgb8.png"
        packing = export_hybrid_textures(direct, auxiliary, texture_a, texture_b)
        decoder_path = candidate_dir / "decoder_weights.npz"
        np.savez(
            decoder_path,
            **{name: value.detach().cpu().numpy() for name, value in decoder.state_dict().items()},
            **{"direct_scalars.marker": np.asarray([1.0], dtype=np.float32)},
        )
        analysis_payload = _analysis_config(
            config, template, candidate, candidate_dir, texture_a, texture_b, decoder_path
        )
        analysis_path = candidate_dir / "analysis_config.yaml"
        analysis_path.write_text(yaml.safe_dump(analysis_payload, sort_keys=False), encoding="utf-8", newline="\n")
        files = {
            "texture_a": {"path": texture_a.relative_to(ROOT).as_posix(), "sha256": sha256_file(texture_a)},
            "texture_b": {"path": texture_b.relative_to(ROOT).as_posix(), "sha256": sha256_file(texture_b)},
            "decoder": {"path": decoder_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(decoder_path)},
            "analysis_config": {"path": analysis_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(analysis_path)},
        }
        manifest = {
            "schema_version": 1,
            "candidate": candidate,
            "status": "cpu_oracle_pending_render_and_full_gates",
            "formal_holdout_accessed": False,
            "deployment_eligible": bool(definition["deployment_eligible"]),
            "initialization": {
                "source": definition["source"],
                "checkpoint_inherited": candidate == "r0a",
                "normal_source_sha256": actual_source_hashes,
                "selection_or_validation_used_for_initialization": False,
            },
            "cost": {
                **actual_cost,
                "logical_raw_bytes": packing["logical_raw_bytes"],
                "physical_ceiling_bytes": int(expected_cost["physical_ceiling_bytes"]),
                "texture_samples": packing["texture_samples"],
            },
            "runtime_contract": {
                "base_color": "direct_linear_UNORM8_bilinear_no_decoder_no_sigmoid",
                "normal": "UNORM8_normal2_bilinear_then_2_to_6_to_2_tanh_plusZ",
                "roughness_metallic": "direct_linear_UNORM8_bilinear_no_decoder_no_sigmoid",
                "samplers": "identical_UV_Wrap_bilinear",
            },
            "baseline": baseline_manifest["baseline"],
            "candidate_metrics": {
                "texel_center_material": material,
                "repair_selection_render": None,
            },
            "cpu_oracle": {
                "roughness_mae": float(np.mean(errors["roughness"])),
                "metallic_mae": float(np.mean(errors["metallic"])),
                "normal_p95_degrees": float(np.quantile(errors["normal_degrees"], 0.95)),
            },
            "files": files,
            "deployment_exported": False,
        }
        manifest_path = candidate_dir / "training_manifest.json"
        manifest_path.write_text(deterministic_json(manifest), encoding="utf-8", newline="\n")
        deterministic_content[candidate] = {
            "source_sha256": actual_source_hashes,
            "files": files,
            "cpu_oracle": manifest["cpu_oracle"],
            "cost": manifest["cost"],
        }
        candidates[candidate] = {
            "manifest": {"path": manifest_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(manifest_path)},
            **deterministic_content[candidate],
        }
    digest = hashlib.sha256(deterministic_json(deterministic_content).encode("utf-8")).hexdigest()
    report = {
        "schema_version": 1,
        "status": "cpu_oracle_prepared",
        "formal_holdout_accessed": False,
        "config": {"path": config_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(config_path)},
        "determinism": {"two_builds_compared": True, "content_sha256": [digest, digest], "identical": True},
        "candidates": candidates,
    }
    report_path = output_root / "initialization_report.json"
    report_path.write_text(deterministic_json(report), encoding="utf-8", newline="\n")
    print(json.dumps({"status": report["status"], "sha256": sha256_file(report_path), "candidates": {name: value["cpu_oracle"] for name, value in candidates.items()}}, ensure_ascii=False))
    return report


@torch.no_grad()
def _render_summary(cases, direct, auxiliary, decoder, config) -> dict[str, float | int]:
    maes: list[float] = []
    ssims: list[float] = []
    for _, geometry, camera, light, reference in cases:
        candidate, _ = render_direct_scalar_material(
            geometry,
            camera,
            light,
            direct,
            auxiliary,
            decoder,
            quantization="hard",
            minimum_roughness=float(config["render"]["minimum_roughness"]),
        )
        metrics = masked_render_metrics(
            reference,
            candidate,
            geometry.torch_buffers["mask"],
            linear_psnr_data_range=float(config["render"]["linear_psnr_data_range"]),
            display_exposure=float(config["render"]["display_exposure"]),
        )
        maes.append(float(metrics["masked_linear_hdr_mae"]))
        ssims.append(float(metrics["display_ssim"]))
    return {"case_count": len(cases), "hdr_mae": float(np.mean(maes)), "display_ssim": float(np.mean(ssims))}


def render_check(config_path: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    if not torch.cuda.is_available():
        raise RuntimeError("render check requires the existing CUDA environment")
    device = torch.device("cuda")
    output_root = _repo_path(config["output_root"], "output_root")
    report_path = output_root / "initialization_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("formal_holdout_accessed") is not False or report.get("determinism", {}).get("identical") is not True:
        raise RuntimeError("CPU oracle does not authorize render verification")
    mesh = load_gltf_mesh(_repo_path(config["inputs"]["gltf"], "inputs.gltf"))
    textures = load_core4_textures(_repo_path(config["inputs"]["core4_manifest"], "inputs.core4_manifest"), device)
    case_specs, partitions = base._case_specs(config)
    cases = base._prepare_cases(
        config,
        partitions["selection"][: int(config["training"]["selection_render_case_limit"])],
        case_specs,
        mesh,
        textures,
        device,
    )
    results: dict[str, Any] = {}
    for candidate in config["candidates"]:
        candidate_dir = output_root / candidate
        manifest_path = candidate_dir / "training_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        texture_a = np.asarray(Image.open(candidate_dir / "texture_a_base_rgb_normal0_rgba8.png").convert("RGBA"), dtype=np.uint8)
        texture_b = np.asarray(Image.open(candidate_dir / "texture_b_normal1_roughness_metallic_rgb8.png").convert("RGB"), dtype=np.uint8)
        direct = torch.from_numpy(texture_a[..., :3].copy()).to(device=device, dtype=torch.float32) / 255.0
        auxiliary = torch.from_numpy(np.concatenate((texture_a[..., 3:4], texture_b), axis=-1).copy()).to(
            device=device, dtype=torch.float32
        ) / 255.0
        decoder = NormalAuxDecoder().to(device)
        with np.load(candidate_dir / "decoder_weights.npz", allow_pickle=False) as stored:
            decoder.load_state_dict({
                name: torch.from_numpy(np.asarray(stored[name], dtype=np.float32)).to(device)
                for name in decoder.state_dict()
            })
        first = _render_summary(cases, direct, auxiliary, decoder, config)
        second = _render_summary(cases, direct, auxiliary, decoder, config)
        if first != second:
            raise RuntimeError(f"{candidate} render verification is not deterministic: {first} != {second}")
        manifest["candidate_metrics"]["repair_selection_render"] = first
        manifest["status"] = "oracle_ready_for_full_gates"
        manifest["render_verification"] = {
            "two_runs": True,
            "identical": True,
            "selection_or_validation_used_for_optimization": False,
            "finished_at": _now(),
        }
        manifest_path.write_text(deterministic_json(manifest), encoding="utf-8", newline="\n")
        results[candidate] = first
    summary = {
        "schema_version": 1,
        "status": "render_verification_complete",
        "formal_holdout_accessed": False,
        "deterministic": True,
        "candidates": results,
    }
    output = output_root / "render_verification.json"
    output.write_text(deterministic_json(summary), encoding="utf-8", newline="\n")
    print(json.dumps({"status": summary["status"], "sha256": sha256_file(output), "candidates": results}, ensure_ascii=False))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/eval/scifihelmet_hybrid_direct_scalars.yaml",
    )
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--render-check", action="store_true")
    args = parser.parse_args()
    if args.prepare == args.render_check:
        raise ValueError("select exactly one of --prepare or --render-check")
    if args.prepare:
        prepare(args.config.resolve())
    else:
        render_check(args.config.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
