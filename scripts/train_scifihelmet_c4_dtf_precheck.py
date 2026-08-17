"""Train paired activation prechecks or the gated C4-DTF-32 diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cg_frontier.compression.artifact_analysis import (  # noqa: E402
    deterministic_json,
    sha256_file,
)
from cg_frontier.compression.decode_then_filter import (  # noqa: E402
    DecodeThenFilterDecoder,
    calculate_decode_then_filter_cost,
    decode_then_filter_sample,
    instantiate_paired_precheck_candidate,
    make_paired_precheck_initialization,
)
from cg_frontier.compression.decode_then_filter_training import (  # noqa: E402
    build_decode_then_filter_manifest,
    decode_then_filter_batch,
    paired_precheck_objective,
    precheck_learning_rates_at_step,
    validate_decode_then_filter_protocol,
)
from cg_frontier.compression.filter_aware import (  # noqa: E402
    bilinear_corners_top_down_wrap_torch,
)
from cg_frontier.compression.material import (  # noqa: E402
    Core4Targets,
    DecodedMaterial,
    keep_system_awake,
    load_core4_targets,
)
from cg_frontier.compression.render_loss import (  # noqa: E402
    bilinear_sample_top_down_wrap,
    export_latent_unorm8_png,
    fake_quantize_unorm8,
)


PRECHECKS = {
    "c4_dtf_16_relu_precheck": "relu",
    "c4_dtf_16_silu_precheck": "silu",
    "c4_dtf_32_diagnostic": "selected_precheck_winner",
}


def _repo_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"config field {label} must be a path")
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"config field {label} escapes the repository")
    if "formal_holdout" in path.as_posix().lower():
        raise ValueError(f"sealed evaluation path is forbidden: {label}")
    return path


def _atomic_save(value: Any, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _reference_textures(targets: Core4Targets) -> dict[str, torch.Tensor]:
    return {
        "base_color": targets.base_color_linear.reshape(targets.height, targets.width, 3),
        "normal": targets.normal_xyz.reshape(targets.height, targets.width, 3),
        "roughness": targets.roughness.reshape(targets.height, targets.width, 1),
        "metallic": targets.metallic.reshape(targets.height, targets.width, 1),
    }


def _sample_reference(
    textures: Mapping[str, torch.Tensor], uv: torch.Tensor
) -> Core4Targets:
    base = bilinear_sample_top_down_wrap(textures["base_color"], uv)
    normal = F.normalize(
        bilinear_sample_top_down_wrap(textures["normal"], uv),
        dim=-1,
        eps=1.0e-8,
    )
    roughness = bilinear_sample_top_down_wrap(textures["roughness"], uv)
    metallic = bilinear_sample_top_down_wrap(textures["metallic"], uv)
    return Core4Targets(base, normal, roughness, metallic, height=1, width=base.shape[0])


def _generic_high_gradient_pool(
    targets: Core4Targets, *, top_fraction: float
) -> np.ndarray:
    if not (0.0 < top_fraction < 1.0):
        raise ValueError("high-gradient top fraction must be within (0, 1)")
    height, width = targets.height, targets.width
    base = targets.base_color_linear.detach().cpu().numpy().reshape(height, width, 3)
    normal = targets.normal_xyz.detach().cpu().numpy().reshape(height, width, 3)
    roughness = targets.roughness.detach().cpu().numpy().reshape(height, width)
    metallic = targets.metallic.detach().cpu().numpy().reshape(height, width)
    luma = np.sum(
        base * np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32), axis=-1
    )
    score = np.zeros((height, width), dtype=np.float32)
    for values in (luma[..., None], normal, roughness[..., None], metallic[..., None]):
        horizontal = np.linalg.norm(values[:, 1:] - values[:, :-1], axis=-1)
        vertical = np.linalg.norm(values[1:] - values[:-1], axis=-1)
        score[:, 1:] = np.maximum(score[:, 1:], horizontal)
        score[:, :-1] = np.maximum(score[:, :-1], horizontal)
        score[1:] = np.maximum(score[1:], vertical)
        score[:-1] = np.maximum(score[:-1], vertical)
    positive = score[score > 0.0]
    if positive.size == 0:
        raise RuntimeError("generic high-gradient pool is empty")
    threshold = float(np.quantile(positive, 1.0 - top_fraction))
    pool = np.flatnonzero(score.reshape(-1) >= threshold).astype(np.int64)
    if pool.size == 0:
        raise RuntimeError("generic high-gradient pool selection failed")
    return pool


def _checkpoint_payload(
    *,
    candidate: str,
    step: int,
    latent: torch.Tensor,
    decoder: DecodeThenFilterDecoder,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    initialization_sha256: str,
    input_hashes: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "candidate": candidate,
        "step": step,
        "latent": latent.detach(),
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all(),
            "sampling_generator": generator.get_state(),
        },
        "initialization_sha256": initialization_sha256,
        "input_hashes": dict(input_hashes),
    }


@torch.no_grad()
def _evaluate_precheck(
    latent: torch.Tensor,
    decoder: DecodeThenFilterDecoder,
    reference_textures: Mapping[str, torch.Tensor],
    *,
    sample_count: int,
    seed: int,
) -> dict[str, Any]:
    device = latent.device
    generator = torch.Generator(device=device).manual_seed(seed)
    uv = torch.rand((sample_count, 2), generator=generator, device=device)
    reference = _sample_reference(reference_textures, uv)
    deployed = fake_quantize_unorm8(latent).detach()
    prediction = decode_then_filter_sample(
        deployed, uv, decoder, quantization="prequantized"
    ).material
    base_error = torch.abs(prediction.base_color_linear - reference.base_color_linear)
    normal_dot = torch.sum(prediction.normal_xyz * reference.normal_xyz, dim=-1).clamp(-1.0, 1.0)
    normal_angle = torch.rad2deg(torch.acos(normal_dot))
    roughness_error = torch.abs(prediction.roughness - reference.roughness)
    metallic_error = torch.abs(prediction.metallic - reference.metallic)
    luma_weights = latent.new_tensor((0.2126, 0.7152, 0.0722))
    reference_luma = torch.sum(reference.base_color_linear * luma_weights, dim=-1)
    prediction_luma = torch.sum(prediction.base_color_linear * luma_weights, dim=-1)
    eligible = reference_luma > 0.05
    dark = eligible & (prediction_luma < reference_luma * 0.5)
    bright_halo = eligible & (prediction_luma > reference_luma + 0.05)

    corners, _ = bilinear_corners_top_down_wrap_torch(deployed, uv[: min(sample_count, 65_536)])
    first = decoder._activate(decoder.hidden_in(corners))
    second = decoder._activate(decoder.hidden_mid(first))
    first_dead = torch.amax(torch.abs(first), dim=(0, 1)) <= 1.0e-8
    second_dead = torch.amax(torch.abs(second), dim=(0, 1)) <= 1.0e-8

    timing_uv = uv[: min(sample_count, 65_536)]
    for _ in range(3):
        decode_then_filter_sample(deployed, timing_uv, decoder, quantization="prequantized")
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(10):
        decode_then_filter_sample(deployed, timing_uv, decoder, quantization="prequantized")
    stop.record()
    torch.cuda.synchronize()
    return {
        "sample_count": sample_count,
        "base_color_linear_mae": float(base_error.mean().cpu()),
        "base_color_max_channel_p99": float(
            torch.quantile(base_error.amax(dim=-1), 0.99).cpu()
        ),
        "normal_mean_degrees": float(normal_angle.mean().cpu()),
        "normal_p95_degrees": float(torch.quantile(normal_angle, 0.95).cpu()),
        "roughness_mae": float(roughness_error.mean().cpu()),
        "metallic_mae": float(metallic_error.mean().cpu()),
        "generic_dark_fraction": float(dark[eligible].to(torch.float32).mean().cpu())
        if torch.any(eligible)
        else 0.0,
        "generic_positive_halo_fraction": float(
            bright_halo[eligible].to(torch.float32).mean().cpu()
        )
        if torch.any(eligible)
        else 0.0,
        "dead_units": {
            "hidden_in": int(first_dead.sum().cpu()),
            "hidden_mid": int(second_dead.sum().cpu()),
        },
        "forward_timing": {
            "pixels": int(timing_uv.shape[0]),
            "repeats": 10,
            "median_proxy_ms_per_repeat": float(start.elapsed_time(stop) / 10.0),
        },
    }


def run(
    config_path: Path,
    *,
    candidate_name: str,
    preflight_only: bool,
) -> dict[str, Any]:
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    if not isinstance(config, Mapping):
        raise ValueError("DTF config must be a mapping")
    validate_decode_then_filter_protocol(config)
    if config.get("schema_version") != 1 or config.get("experiment") != "scifihelmet_c4_dtf_v1":
        raise ValueError("unsupported DTF training config")
    if candidate_name not in PRECHECKS:
        raise ValueError(f"unsupported paired precheck: {candidate_name}")
    if not torch.cuda.is_available():
        raise RuntimeError("DTF paired precheck requires CUDA")
    candidate_config = config["candidates"][candidate_name]
    if int(candidate_config["max_steps"]) != 10_000:
        raise ValueError("material diagnostic must use the frozen 10k budget")

    output_root = _repo_path(config["output_root"], "output_root")
    activation_selection_sha256: str | None = None
    eligibility_sha256: str | None = None
    if candidate_name == "c4_dtf_32_diagnostic":
        selection_path = output_root / "activation_selection.json"
        eligibility_path = output_root / "c4_dtf_32_diagnostic_eligibility.json"
        if not selection_path.is_file() or not eligibility_path.is_file():
            raise RuntimeError("C4-DTF-32 requires signed activation and eligibility evidence")
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        eligibility = json.loads(eligibility_path.read_text(encoding="utf-8"))
        if (
            selection.get("selected_activation") != "relu"
            or not eligibility.get("eligible")
            or eligibility.get("next_candidate") != candidate_name
        ):
            raise RuntimeError("C4-DTF-32 eligibility or activation evidence is invalid")
        activation = "relu"
        activation_selection_sha256 = sha256_file(selection_path)
        eligibility_sha256 = sha256_file(eligibility_path)
        if candidate_config["activation"] != "selected_precheck_winner":
            raise ValueError("capacity diagnostic activation source differs from protocol")
    else:
        activation = PRECHECKS[candidate_name]
        if candidate_config["activation"] != activation:
            raise ValueError("paired precheck candidate differs from the frozen protocol")

    training = config["training"]
    seed = int(training["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)

    inputs = {
        name: _repo_path(value, f"inputs.{name}")
        for name, value in config["inputs"].items()
    }
    input_hashes = {
        name: sha256_file(path)
        for name, path in inputs.items()
        if path.is_file()
    }
    targets = load_core4_targets(inputs["core4_dir"], device)
    reference_textures = _reference_textures(targets)
    decoder_width = int(candidate_config["decoder_width"])
    initialization = make_paired_precheck_initialization(
        height=targets.height,
        width=targets.width,
        latent_channels=4,
        decoder_width=decoder_width,
        seed=seed,
    )
    fresh = instantiate_paired_precheck_candidate(
        initialization, activation=activation, device=device
    )
    latent, decoder = fresh.latent, fresh.decoder

    output_dir = output_root / candidate_name
    if preflight_only:
        output_dir = output_root / "preflight" / candidate_name
    log_path = output_dir / "train.jsonl"
    if log_path.exists():
        raise FileExistsError(f"refusing to inherit an existing DTF run: {log_path}")
    if candidate_name.endswith("silu_precheck") and not preflight_only:
        relu_manifest = output_root / "c4_dtf_16_relu_precheck" / "training_manifest.json"
        if not relu_manifest.is_file():
            raise RuntimeError("paired prechecks must run serially: ReLU is not complete")
    if candidate_name == "c4_dtf_32_diagnostic" and not preflight_only:
        preflight_manifest = (
            output_root / "preflight" / candidate_name / "training_manifest.json"
        )
        if not preflight_manifest.is_file():
            raise RuntimeError("C4-DTF-32 requires a successful independent preflight")
        preflight = json.loads(preflight_manifest.read_text(encoding="utf-8"))
        if (
            not preflight.get("valid")
            or preflight.get("initialization", {}).get("sha256")
            != initialization.sha256
            or preflight.get("activation_selection_sha256")
            != activation_selection_sha256
            or preflight.get("capacity_eligibility_sha256") != eligibility_sha256
        ):
            raise RuntimeError("C4-DTF-32 preflight evidence is mismatched")
    output_dir.mkdir(parents=True, exist_ok=True)

    planned_manifest = build_decode_then_filter_manifest(
        config,
        candidate=candidate_name,
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        input_hashes=input_hashes,
        git_commit=str(config["version_control"]["git_commit"]),
        selected_activation=activation
        if candidate_name == "c4_dtf_32_diagnostic"
        else None,
        activation_selection_sha256=activation_selection_sha256,
    )
    (output_dir / "run_manifest.json").write_text(
        deterministic_json(planned_manifest), encoding="utf-8", newline="\n"
    )

    screen_uv_array = np.load(inputs["precheck_screen_uv"])
    screen_mask = np.load(inputs["precheck_screen_mask"]).astype(bool)
    if screen_uv_array.shape[:2] != screen_mask.shape or screen_uv_array.shape[-1] != 2:
        raise ValueError("precheck screen UV/mask shapes differ")
    screen_uv = torch.from_numpy(screen_uv_array[screen_mask].astype(np.float32)).to(device)
    uniform_pool = torch.arange(targets.texel_count, device=device)
    gradient_np = _generic_high_gradient_pool(
        targets, top_fraction=float(training["high_gradient_top_fraction"])
    )
    high_gradient_pool = torch.from_numpy(gradient_np).to(device)

    if training.get("optimizer") != "Adam" or training.get("optimizer_implementation") != "fused_cuda_direct_unorm":
        raise ValueError("DTF paired precheck requires fused CUDA Adam")
    optimizer = torch.optim.Adam(
        [
            {"params": [latent], "lr": 0.0},
            {"params": list(decoder.parameters()), "lr": 0.0},
        ],
        fused=True,
    )
    max_steps = int(training["preflight_steps"] if preflight_only else candidate_config["max_steps"])
    batch_size = int(training["batch_size"])
    checkpoint_path = output_dir / "checkpoint.pt"
    started = time.monotonic()
    completed_step = 0
    with keep_system_awake():
        for step in range(1, max_steps + 1):
            latent_lr, decoder_lr = precheck_learning_rates_at_step(step)
            optimizer.param_groups[0]["lr"] = latent_lr
            optimizer.param_groups[1]["lr"] = decoder_lr
            optimizer.zero_grad(set_to_none=True)
            batch = decode_then_filter_batch(
                screen_uv=screen_uv,
                uniform_pool=uniform_pool,
                high_gradient_pool=high_gradient_pool,
                batch_size=batch_size,
                height=targets.height,
                width=targets.width,
                generator=generator,
            )
            target = _sample_reference(reference_textures, batch.uv)
            sample = decode_then_filter_sample(
                latent, batch.uv, decoder, quantization="fake"
            )
            raw_corners, _ = bilinear_corners_top_down_wrap_torch(latent, batch.uv)
            quantization_error = F.mse_loss(raw_corners, sample.corners.detach())
            total, terms = paired_precheck_objective(
                sample.material,
                target,
                anchor_slice=batch.slices["texel_center_quantization_anchor"],
                quantization_error=quantization_error,
                loss_config=config["loss"],
            )
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite DTF precheck loss at step {step}")
            total.backward()
            optimizer.step()
            with torch.no_grad():
                latent.clamp_(0.0, 1.0)
            completed_step = step
            elapsed = time.monotonic() - started
            record = {
                "step": step,
                "elapsed_seconds": round(elapsed, 3),
                "latent_lr": latent_lr,
                "decoder_lr": decoder_lr,
                "total": float(total.detach().cpu()),
                **{name: float(value.detach().cpu()) for name, value in terms.items()},
            }
            if step == 1 or step % int(training["log_interval"]) == 0 or step == max_steps:
                with log_path.open("a", encoding="utf-8", newline="\n") as stream:
                    stream.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
                print(json.dumps({"candidate": candidate_name, **record}, sort_keys=True), flush=True)
            if step % int(training["checkpoint_interval"]) == 0 or step == max_steps:
                _atomic_save(
                    _checkpoint_payload(
                        candidate=candidate_name,
                        step=step,
                        latent=latent,
                        decoder=decoder,
                        optimizer=optimizer,
                        generator=generator,
                        initialization_sha256=initialization.sha256,
                        input_hashes=input_hashes,
                    ),
                    checkpoint_path,
                )

    elapsed = time.monotonic() - started
    evaluation_samples = int(
        training["preflight_evaluation_samples"]
        if preflight_only
        else training["evaluation_samples"]
    )
    evaluation = _evaluate_precheck(
        latent.detach(),
        decoder,
        reference_textures,
        sample_count=evaluation_samples,
        seed=seed + 99,
    )
    latent_name = f"latent_{candidate_name}_rgba_unorm8.png"
    latent_metadata = export_latent_unorm8_png(
        fake_quantize_unorm8(latent.detach()), output_dir / latent_name
    )
    decoder_path = output_dir / "decoder_weights.npz"
    np.savez(
        decoder_path,
        **{
            name: value.detach().cpu().numpy()
            for name, value in decoder.state_dict().items()
        },
    )
    valid = completed_step == max_steps and checkpoint_path.is_file()
    manifest = {
        **planned_manifest,
        "status": "preflight_complete"
        if preflight_only and valid
        else (
            "trained_capacity_diagnostic"
            if candidate_name == "c4_dtf_32_diagnostic"
            else "trained_precheck"
        ),
        "valid": valid,
        "activation_selection_sha256": activation_selection_sha256,
        "capacity_eligibility_sha256": eligibility_sha256,
        "initialization": {
            "kind": "fresh_capacity_diagnostic"
            if candidate_name == "c4_dtf_32_diagnostic"
            else "paired_fresh_random_state",
            "seed": seed,
            "sha256": initialization.sha256,
            "shared_mutable_storage": False,
        },
        "training_result": {
            "completed_steps": completed_step,
            "required_steps": max_steps,
            "elapsed_seconds": round(elapsed, 3),
            "batch_size": batch_size,
            "optimizer": "Adam",
            "optimizer_implementation": "fused_cuda_direct_unorm",
            "checkpoint_contains_optimizer_rng_sampling_generator": True,
        },
        "evaluation": evaluation,
        "files": {
            latent_name: latent_metadata,
            "decoder_weights.npz": sha256_file(decoder_path),
            "checkpoint.pt": sha256_file(checkpoint_path),
            "train.jsonl": sha256_file(log_path),
            "run_manifest.json": sha256_file(output_dir / "run_manifest.json"),
        },
    }
    (output_dir / "training_manifest.json").write_text(
        deterministic_json(manifest), encoding="utf-8", newline="\n"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/train/scifihelmet_c4_dtf_v1.yaml",
    )
    parser.add_argument("--candidate", choices=tuple(PRECHECKS), required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    result = run(
        args.config.resolve(),
        candidate_name=args.candidate,
        preflight_only=args.preflight_only,
    )
    print(
        json.dumps(
            {
                "candidate": args.candidate,
                "preflight_only": args.preflight_only,
                "valid": result["valid"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
