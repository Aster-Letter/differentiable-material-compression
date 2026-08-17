"""Frozen CPU-side protocol for SciFiHelmet quality-first DTF experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Mapping

import numpy as np
import torch

from cg_frontier.compression.decode_then_filter import (
    DecodeThenFilterDecoder,
    calculate_decode_then_filter_cost,
)
from cg_frontier.compression.material import (
    Core4Targets,
    DecodedMaterial,
    material_loss,
)


DTF_BATCH_MIX = {
    "screen_space_render": 0.45,
    "uniform_uv_chart_subpixel": 0.35,
    "generic_high_gradient_boundary": 0.10,
    "texel_center_quantization_anchor": 0.10,
}

DTF_LOSS = {
    "render_hdr_l1": 1.0,
    "render_display_l1": 0.25,
    "base_color_linear_l1": 0.5,
    "normal_cosine": 0.5,
    "roughness_l1": 0.25,
    "metallic_l1": 0.25,
    "subpixel_material_anchor": 0.5,
    "texel_center_anchor": 0.25,
    "unorm8_quantization": 0.25,
}

DTF_PHASES = [
    {"name": "material_continuous_pretrain", "start": 0, "stop": 15_000},
    {"name": "render_first_joint", "start": 15_000, "stop": 65_000},
    {"name": "low_lr_polish", "start": 65_000, "stop": 80_000},
]

DTF_CANDIDATES = {
    "c4_dtf_16_relu_precheck": {
        "latent_channels": 4,
        "decoder_width": 16,
        "activation": "relu",
        "role": "paired_precheck",
        "max_steps": 10_000,
    },
    "c4_dtf_16_silu_precheck": {
        "latent_channels": 4,
        "decoder_width": 16,
        "activation": "silu",
        "role": "paired_precheck",
        "max_steps": 10_000,
    },
    "c4_dtf_16_selected": {
        "latent_channels": 4,
        "decoder_width": 16,
        "activation": "selected_precheck_winner",
        "role": "primary_full_training",
        "max_steps": 80_000,
    },
    "c4_dtf_32_diagnostic": {
        "latent_channels": 4,
        "decoder_width": 32,
        "activation": "selected_precheck_winner",
        "role": "conditional_capacity_diagnostic",
        "max_steps": 10_000,
    },
    "c4_dtf_32_selected": {
        "latent_channels": 4,
        "decoder_width": 32,
        "activation": "selected_precheck_winner",
        "role": "conditional_full_capacity",
        "max_steps": 80_000,
    },
    "c5_dtf_16": {
        "latent_channels": 5,
        "decoder_width": 16,
        "activation": "selected_precheck_winner",
        "role": "conditional_representation_upper_bound",
        "max_steps": 80_000,
    },
}


_PRECHECK_METRICS = (
    "base_color_linear_mae",
    "normal_mean_degrees",
    "normal_p95_degrees",
    "roughness_mae",
    "metallic_mae",
    "generic_dark_fraction",
    "generic_positive_halo_fraction",
)


@dataclass(frozen=True)
class DecodeThenFilterBatch:
    """One generic quality-first DTF sample mixture."""

    uv: torch.Tensor
    slices: Mapping[str, slice]
    screen_positions: torch.Tensor


def resolve_dtf_camera_spec(
    spec: Mapping[str, Any], render: Mapping[str, Any]
) -> dict[str, Any]:
    """Resolve one orbit camera, including an optional local focus override."""

    target_values = spec.get("target", render["target"])
    if not isinstance(target_values, (list, tuple)) or len(target_values) != 3:
        raise ValueError("DTF camera target must contain three coordinates")
    up_values = render["up"]
    if not isinstance(up_values, (list, tuple)) or len(up_values) != 3:
        raise ValueError("DTF camera up vector must contain three coordinates")
    radius = float(spec.get("radius", render["camera_radius"]))
    if radius <= 0.0:
        raise ValueError("DTF camera radius must be positive")
    return {
        "name": str(spec["name"]),
        "yaw_degrees": float(spec["yaw_degrees"]),
        "elevation_degrees": float(spec["elevation_degrees"]),
        "radius": radius,
        "target": tuple(float(value) for value in target_values),
        "up": tuple(float(value) for value in up_values),
        "vertical_fov_degrees": float(render["vertical_fov_degrees"]),
        "near": float(render["near"]),
        "far": float(render["far"]),
    }


def _draw_rows(
    pool: torch.Tensor, count: int, generator: torch.Generator
) -> torch.Tensor:
    if pool.ndim != 1 or pool.numel() == 0:
        raise ValueError("DTF texel pool must be a non-empty vector")
    positions = torch.randint(
        0,
        pool.shape[0],
        (count,),
        generator=generator,
        device=pool.device,
    )
    return pool[positions]


def _cell_uv(
    ids: torch.Tensor,
    *,
    height: int,
    width: int,
    generator: torch.Generator,
    centered: bool,
) -> torch.Tensor:
    if centered:
        offsets = torch.full((ids.numel(), 2), 0.5, device=ids.device)
    else:
        offsets = torch.rand((ids.numel(), 2), generator=generator, device=ids.device)
    x = ids.remainder(width).to(torch.float32) + offsets[:, 0]
    y = torch.div(ids, width, rounding_mode="floor").to(torch.float32) + offsets[:, 1]
    return torch.stack((x / float(width), y / float(height)), dim=-1)


def decode_then_filter_batch(
    *,
    screen_uv: torch.Tensor,
    uniform_pool: torch.Tensor,
    high_gradient_pool: torch.Tensor,
    batch_size: int,
    height: int,
    width: int,
    generator: torch.Generator,
) -> DecodeThenFilterBatch:
    """Draw the frozen generic 45/35/10/10 DTF training mixture."""

    if batch_size <= 0 or batch_size % 20:
        raise ValueError("DTF batch size must be a positive multiple of 20")
    if screen_uv.ndim != 2 or screen_uv.shape[-1] != 2 or screen_uv.shape[0] == 0:
        raise ValueError("DTF screen-space pool must have shape Nx2 and be non-empty")
    if height <= 0 or width <= 0:
        raise ValueError("DTF atlas dimensions must be positive")
    unit = batch_size // 20
    counts = (9 * unit, 7 * unit, 2 * unit, 2 * unit)
    screen_positions = torch.randint(
        0,
        screen_uv.shape[0],
        (counts[0],),
        generator=generator,
        device=screen_uv.device,
    )
    uniform_ids = _draw_rows(uniform_pool, counts[1] + counts[3], generator)
    gradient_ids = _draw_rows(high_gradient_pool, counts[2], generator)
    chunks = (
        screen_uv[screen_positions],
        _cell_uv(
            uniform_ids[: counts[1]],
            height=height,
            width=width,
            generator=generator,
            centered=False,
        ),
        _cell_uv(
            gradient_ids,
            height=height,
            width=width,
            generator=generator,
            centered=False,
        ),
        _cell_uv(
            uniform_ids[counts[1] :],
            height=height,
            width=width,
            generator=generator,
            centered=True,
        ),
    )
    names = tuple(DTF_BATCH_MIX)
    offset = 0
    slices: dict[str, slice] = {}
    for name, count in zip(names, counts, strict=True):
        slices[name] = slice(offset, offset + count)
        offset += count
    return DecodeThenFilterBatch(torch.cat(chunks, dim=0), slices, screen_positions)


def precheck_learning_rates_at_step(step: int) -> tuple[float, float]:
    """Return the paired 10k material-precheck latent and decoder rates."""

    if not (0 <= step <= 10_000):
        raise ValueError("DTF precheck step must be within [0, 10000]")
    if step == 10_000:
        return 2.0e-3, 1.0e-4
    if step <= 500:
        fraction = step / 500.0
        return 2.0e-2 * fraction, 1.0e-3 * fraction
    fraction = (step - 500) / 9_500.0
    blend = 0.5 * (1.0 - math.cos(math.pi * fraction))
    return (
        2.0e-2 + (2.0e-3 - 2.0e-2) * blend,
        1.0e-3 + (1.0e-4 - 1.0e-3) * blend,
    )


def full_training_phase_at_step(step: int) -> str:
    """Return the frozen phase for one 1-based step in the 80k run."""

    if not (1 <= step <= 80_000):
        raise ValueError("DTF full-training step must be within [1, 80000]")
    if step <= 15_000:
        return "material_continuous_pretrain"
    if step <= 65_000:
        return "render_first_joint"
    return "low_lr_polish"


def _cosine_rate(
    step: int,
    *,
    start: int,
    stop: int,
    initial: float,
    final: float,
) -> float:
    if step == stop:
        return final
    fraction = (step - start) / float(stop - start)
    blend = 0.5 * (1.0 - math.cos(math.pi * fraction))
    return initial + (final - initial) * blend


def full_training_learning_rates_at_step(step: int) -> tuple[float, float]:
    """Return fresh-run latent/decoder rates for the frozen 80k phases."""

    full_training_phase_at_step(step)
    if step <= 500:
        fraction = step / 500.0
        return 2.0e-2 * fraction, 1.0e-3 * fraction
    if step <= 15_000:
        return (
            _cosine_rate(
                step,
                start=500,
                stop=15_000,
                initial=2.0e-2,
                final=5.0e-3,
            ),
            _cosine_rate(
                step,
                start=500,
                stop=15_000,
                initial=1.0e-3,
                final=2.5e-4,
            ),
        )
    if step <= 65_000:
        return (
            _cosine_rate(
                step,
                start=15_000,
                stop=65_000,
                initial=5.0e-3,
                final=2.0e-3,
            ),
            _cosine_rate(
                step,
                start=15_000,
                stop=65_000,
                initial=2.5e-4,
                final=1.0e-4,
            ),
        )
    return (
        _cosine_rate(
            step,
            start=65_000,
            stop=80_000,
            initial=2.0e-3,
            final=1.0e-3,
        ),
        _cosine_rate(
            step,
            start=65_000,
            stop=80_000,
            initial=1.0e-4,
            final=5.0e-5,
        ),
    )


def continuation_learning_rates_at_step(
    step: int, *, stop_step: int = 120_000
) -> tuple[float, float]:
    """Continue polish to an approved endpoint without an LR discontinuity."""

    if stop_step not in {120_000, 160_000}:
        raise ValueError("DTF continuation stop must be 120000 or 160000")
    if not (80_000 <= step <= stop_step):
        raise ValueError(
            f"DTF continuation step must be within [80000, {stop_step}]"
        )
    return (
        _cosine_rate(
            step,
            start=80_000,
            stop=stop_step,
            initial=1.0e-3,
            final=2.5e-4,
        ),
        _cosine_rate(
            step,
            start=80_000,
            stop=stop_step,
            initial=5.0e-5,
            final=1.25e-5,
        ),
    )


def camera_finetune_learning_rates_at_step(step: int) -> tuple[float, float]:
    """Decay smoothly from the frozen C4-160k endpoint during camera fine-tuning."""

    if not (160_000 <= step <= 180_000):
        raise ValueError("DTF camera fine-tune step must be within [160000, 180000]")
    return (
        _cosine_rate(
            step,
            start=160_000,
            stop=180_000,
            initial=2.5e-4,
            final=1.0e-4,
        ),
        _cosine_rate(
            step,
            start=160_000,
            stop=180_000,
            initial=1.25e-5,
            final=5.0e-6,
        ),
    )


def validate_decode_then_filter_resume_payload(
    payload: Mapping[str, Any],
    *,
    expected_candidate: str,
    expected_start_step: int,
    expected_latent_shape: tuple[int, ...],
    expected_decoder_keys: set[str],
    expected_input_hashes: Mapping[str, str],
    expected_camera_names: list[str],
) -> dict[str, Any]:
    """Reject incomplete or mismatched state before a DTF continuation starts."""

    if payload.get("schema_version") not in {1, 2}:
        raise ValueError("unsupported DTF resume checkpoint schema")
    if payload.get("candidate") != expected_candidate:
        raise ValueError("DTF resume candidate mismatch")
    if int(payload.get("step", -1)) != expected_start_step:
        raise ValueError("DTF resume start step mismatch")
    latent = payload.get("latent")
    if not isinstance(latent, torch.Tensor) or tuple(latent.shape) != expected_latent_shape:
        raise ValueError("DTF resume latent shape mismatch")
    decoder = payload.get("decoder")
    if not isinstance(decoder, Mapping) or set(decoder) != expected_decoder_keys:
        raise ValueError("DTF resume decoder state mismatch")
    optimizer = payload.get("optimizer")
    if (
        not isinstance(optimizer, Mapping)
        or not optimizer.get("state")
        or len(optimizer.get("param_groups", [])) != 2
    ):
        raise ValueError("DTF resume Adam state is incomplete")
    rng = payload.get("rng")
    required_rng = {"python", "numpy", "torch_cpu", "torch_cuda", "sampling_generator"}
    if not isinstance(rng, Mapping) or not required_rng.issubset(rng):
        raise ValueError("DTF resume RNG state is incomplete")
    if dict(payload.get("input_hashes", {})) != dict(expected_input_hashes):
        raise ValueError("DTF resume input hashes mismatch")
    camera_names = list(dict(payload.get("camera_selection", {})).get("selected_names", []))
    if camera_names != expected_camera_names:
        raise ValueError("DTF resume camera selection mismatch")
    initialization_sha256 = str(payload.get("initialization_sha256", ""))
    activation_selection_sha256 = str(payload.get("activation_selection_sha256", ""))
    _require_sha256(initialization_sha256, name="initialization hash")
    _require_sha256(activation_selection_sha256, name="activation-selection hash")
    return {
        "source_candidate": expected_candidate,
        "source_step": expected_start_step,
        "initialization_sha256": initialization_sha256,
        "activation_selection_sha256": activation_selection_sha256,
    }


def restore_decode_then_filter_resume_state(
    payload: Mapping[str, Any],
    *,
    latent: torch.nn.Parameter,
    decoder: DecodeThenFilterDecoder,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    restore_cuda_rng: bool = True,
) -> None:
    """Restore model, Adam, and stochastic streams from a validated checkpoint."""

    with torch.no_grad():
        latent.copy_(payload["latent"].to(device=latent.device, dtype=latent.dtype))
    decoder.load_state_dict(payload["decoder"])
    optimizer.load_state_dict(payload["optimizer"])
    for state in optimizer.state.values():
        for name, value in tuple(state.items()):
            if isinstance(value, torch.Tensor):
                state[name] = value.to(device=latent.device)
    rng = payload["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch_cpu"].cpu())
    generator.set_state(rng["sampling_generator"].cpu())
    if restore_cuda_rng and torch.cuda.is_available():
        cuda_states = list(rng["torch_cuda"])
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError("DTF resume CUDA RNG device count mismatch")
        torch.cuda.set_rng_state_all(cuda_states)


def full_training_objective(
    material_total: torch.Tensor,
    *,
    reference_hdr: torch.Tensor,
    candidate_hdr: torch.Tensor,
    reference_display: torch.Tensor,
    candidate_display: torch.Tensor,
    phase: str,
    loss_config: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Add only the frozen generic render terms after the 15k pretrain."""

    valid_phases = {phase_spec["name"] for phase_spec in DTF_PHASES} | {
        "low_lr_camera_distribution_finetune"
    }
    if phase not in valid_phases:
        raise ValueError(f"unknown DTF full-training phase: {phase}")
    if phase == "material_continuous_pretrain":
        return material_total, {}
    if reference_hdr.shape != candidate_hdr.shape:
        raise ValueError("reference and candidate HDR render shapes differ")
    if reference_display.shape != candidate_display.shape:
        raise ValueError("reference and candidate display render shapes differ")
    render_hdr = torch.mean(torch.abs(candidate_hdr - reference_hdr))
    render_display = torch.mean(torch.abs(candidate_display - reference_display))
    total = (
        material_total
        + float(loss_config["render_hdr_l1"]) * render_hdr
        + float(loss_config["render_display_l1"]) * render_display
    )
    return total, {
        "render_hdr_l1": render_hdr,
        "render_display_l1": render_display,
    }


@torch.no_grad()
def material_subset_metrics(
    reference: Core4Targets,
    prediction: DecodedMaterial,
    mask: torch.Tensor,
) -> dict[str, float | int]:
    """Measure an evaluation-only material subset, including dark outliers."""

    selected = mask.to(dtype=torch.bool).reshape(-1)
    if selected.numel() != reference.texel_count:
        raise ValueError("material subset mask size does not match the reference")
    sample_count = int(selected.sum().item())
    if sample_count == 0:
        raise ValueError("material subset mask must select at least one texel")

    reference_base = reference.base_color_linear[selected]
    prediction_base = prediction.base_color_linear[selected]
    reference_normal = torch.nn.functional.normalize(
        reference.normal_xyz[selected], dim=-1
    )
    prediction_normal = torch.nn.functional.normalize(
        prediction.normal_xyz[selected], dim=-1
    )
    normal_cosine = torch.sum(reference_normal * prediction_normal, dim=-1)
    normal_degrees = torch.rad2deg(torch.acos(normal_cosine.clamp(-1.0, 1.0)))

    luminance_weights = reference_base.new_tensor([0.2126, 0.7152, 0.0722])
    reference_luminance = reference_base @ luminance_weights
    prediction_luminance = prediction_base @ luminance_weights
    visible_reference = reference_luminance > 0.05
    novel_dark = visible_reference & (
        prediction_luminance < 0.5 * reference_luminance
    )
    dark_denominator = torch.clamp(visible_reference.sum(), min=1)

    return {
        "sample_count": sample_count,
        "novel_dark_fraction": float(
            novel_dark.sum().float().div(dark_denominator).item()
        ),
        "base_color_linear_mae": float(
            torch.mean(torch.abs(prediction_base - reference_base)).item()
        ),
        "normal_mean_degrees": float(torch.mean(normal_degrees).item()),
        "normal_p95_degrees": float(
            torch.quantile(normal_degrees, 0.95).item()
        ),
        "roughness_mae": float(
            torch.mean(
                torch.abs(
                    prediction.roughness[selected] - reference.roughness[selected]
                )
            ).item()
        ),
        "metallic_mae": float(
            torch.mean(
                torch.abs(
                    prediction.metallic[selected] - reference.metallic[selected]
                )
            ).item()
        ),
    }


def select_camera_triangle_coverage(
    candidate_triangle_ids: list[torch.Tensor],
    *,
    triangle_count: int,
    increment_stop: float,
    camera_limit: int,
) -> dict[str, Any]:
    """Select ordered training cameras until new triangle coverage is negligible."""

    if triangle_count <= 0:
        raise ValueError("triangle count must be positive")
    if not (0.0 <= increment_stop < 1.0):
        raise ValueError("camera coverage increment stop must be within [0, 1)")
    if not (1 <= camera_limit <= 48):
        raise ValueError("DTF camera limit must be within [1, 48]")
    if not candidate_triangle_ids:
        raise ValueError("DTF camera candidates must be non-empty")
    covered: set[int] = set()
    selected: list[int] = []
    records: list[dict[str, Any]] = []
    for index, values in enumerate(candidate_triangle_ids):
        ids = {int(value) for value in values.detach().cpu().reshape(-1).tolist()}
        if any(value < 0 or value >= triangle_count for value in ids):
            raise ValueError("camera triangle coverage contains an invalid triangle id")
        new_ids = ids.difference(covered)
        increment = len(new_ids) / float(triangle_count)
        stopped = bool(selected and increment < increment_stop)
        records.append(
            {
                "candidate_index": index,
                "new_triangles": len(new_ids),
                "increment": increment,
                "stopped": stopped,
            }
        )
        if stopped:
            break
        selected.append(index)
        covered.update(new_ids)
        if len(selected) >= camera_limit:
            break
    return {
        "selected_indices": selected,
        "covered_triangles": len(covered),
        "coverage_fraction": len(covered) / float(triangle_count),
        "records": records,
    }


def select_explicit_dtf_camera_pool(
    candidate_triangle_ids: list[torch.Tensor],
    *,
    camera_names: list[str],
    triangle_count: int,
    camera_limit: int,
) -> dict[str, Any]:
    """Retain every view in a bounded camera pool selected by an external audit."""

    if triangle_count <= 0:
        raise ValueError("triangle count must be positive")
    if not (1 <= camera_limit <= 48):
        raise ValueError("DTF camera limit must be within [1, 48]")
    if not candidate_triangle_ids or len(candidate_triangle_ids) != len(camera_names):
        raise ValueError("explicit DTF camera ids and names must be non-empty and aligned")
    if len(camera_names) > camera_limit:
        raise ValueError("explicit DTF camera pool exceeds its bounded limit")
    if len(set(camera_names)) != len(camera_names):
        raise ValueError("explicit DTF camera names must be unique")

    covered: set[int] = set()
    records: list[dict[str, Any]] = []
    for index, values in enumerate(candidate_triangle_ids):
        ids = {int(value) for value in values.detach().cpu().reshape(-1).tolist()}
        if any(value < 0 or value >= triangle_count for value in ids):
            raise ValueError("camera triangle coverage contains an invalid triangle id")
        new_ids = ids.difference(covered)
        records.append(
            {
                "candidate_index": index,
                "new_triangles": len(new_ids),
                "increment": len(new_ids) / float(triangle_count),
                "stopped": False,
            }
        )
        covered.update(new_ids)
    return {
        "strategy": "explicit_audited_pool_v1",
        "selected_indices": list(range(len(camera_names))),
        "selected_names": list(camera_names),
        "covered_triangles": len(covered),
        "coverage_fraction": len(covered) / float(triangle_count),
        "records": records,
    }


def _precheck_evidence(
    manifest: Mapping[str, Any], *, activation: str
) -> tuple[str, dict[str, Any]]:
    if not bool(manifest.get("valid")):
        raise ValueError(f"{activation} paired precheck is not valid")
    candidate = dict(manifest.get("candidate", {}))
    expected_candidate = {
        "name": f"c4_dtf_16_{activation}_precheck",
        "latent_channels": 4,
        "decoder_width": 16,
        "activation": activation,
        "role": "paired_precheck",
        "max_steps": 10_000,
    }
    if candidate != expected_candidate:
        raise ValueError(f"{activation} paired precheck candidate metadata is invalid")
    initialization = dict(manifest.get("initialization", {}))
    sha256 = _require_sha256(
        str(initialization.get("sha256", "")),
        name=f"{activation} initialization hash",
    )
    if initialization.get("shared_mutable_storage") is not False:
        raise ValueError("paired precheck must not share mutable storage")
    evaluation = dict(manifest.get("evaluation", {}))
    metrics = {name: float(evaluation[name]) for name in _PRECHECK_METRICS}
    forward = dict(evaluation.get("forward_timing", {}))
    dead = dict(evaluation.get("dead_units", {}))
    metrics["forward_median_proxy_ms_per_repeat"] = float(
        forward["median_proxy_ms_per_repeat"]
    )
    metrics["dead_hidden_in"] = int(dead["hidden_in"])
    metrics["dead_hidden_mid"] = int(dead["hidden_mid"])
    return sha256, {"valid": True, "metrics": metrics}


def build_activation_selection_record(
    relu_manifest: Mapping[str, Any],
    silu_manifest: Mapping[str, Any],
    *,
    selected_activation: str,
    silu_clear_quality_advantage: bool,
    rationale: str,
) -> dict[str, Any]:
    """Bind the paired 10k evidence to the conservative activation decision."""

    relu_hash, relu = _precheck_evidence(relu_manifest, activation="relu")
    silu_hash, silu = _precheck_evidence(silu_manifest, activation="silu")
    if relu_hash != silu_hash:
        raise ValueError("paired prechecks use different initialization hashes")
    if selected_activation not in {"relu", "silu"}:
        raise ValueError("selected activation must be relu or silu")
    if (selected_activation == "silu") != bool(silu_clear_quality_advantage):
        raise ValueError("SiLU may be selected only with a clear quality advantage")
    if not rationale.strip():
        raise ValueError("activation selection requires a rationale")
    return {
        "schema_version": 1,
        "decision": "c4_dtf_16_activation_selection",
        "policy": "silu_only_with_clear_quality_advantage",
        "selected_activation": selected_activation,
        "silu_clear_quality_advantage": bool(silu_clear_quality_advantage),
        "paired_initialization_sha256": relu_hash,
        "prechecks": {"relu": relu, "silu": silu},
        "rationale": rationale.strip(),
    }


def build_capacity_diagnostic_eligibility(
    c4_manifest: Mapping[str, Any],
    *,
    r0b_control: Mapping[str, Any],
    c4_manifest_sha256: str,
    r0b_document_sha256: str,
) -> dict[str, Any]:
    """Bind the documented R0b quality gap that gates the C4 width diagnostic."""

    if not bool(c4_manifest.get("valid")):
        raise ValueError("C4-DTF-16 full candidate is not valid")
    candidate = dict(c4_manifest.get("candidate", {}))
    expected = {
        "name": "c4_dtf_16_selected",
        "latent_channels": 4,
        "decoder_width": 16,
        "activation": "relu",
        "role": "primary_full_training",
        "max_steps": 80_000,
        "activation_source": "paired_10k_precheck",
    }
    if candidate != expected:
        raise ValueError("capacity diagnostic requires the selected C4-DTF-16 run")
    evaluation = dict(c4_manifest.get("final_evaluation", {}))
    render = dict(evaluation.get("render", {}))
    artifact = dict(evaluation.get("artifact", {}))
    material = dict(evaluation.get("material", {}))
    c4 = {
        "display_ssim": float(render["display_ssim"]),
        "normal_mean_degrees": float(material["normal_mean_degrees"]),
        "novel_dark_fraction": float(artifact["novel_dark_fraction"]),
        "halo_fraction": float(artifact["halo_fraction"]),
    }
    control = {
        "display_ssim": float(r0b_control["display_ssim"]),
        "normal_mean_degrees": float(r0b_control["normal_mean_degrees"]),
        "subpixel_filter_divergence": float(
            r0b_control["subpixel_filter_divergence"]
        ),
    }
    gaps: dict[str, Any] = {}
    if c4["display_ssim"] < control["display_ssim"]:
        gaps["display_ssim"] = {
            "c4": c4["display_ssim"],
            "r0b": control["display_ssim"],
        }
    if c4["normal_mean_degrees"] > control["normal_mean_degrees"]:
        gaps["normal_mean_degrees"] = {
            "c4": c4["normal_mean_degrees"],
            "r0b": control["normal_mean_degrees"],
        }
    if max(c4["novel_dark_fraction"], c4["halo_fraction"]) > control[
        "subpixel_filter_divergence"
    ]:
        gaps["subpixel_filter_safety"] = {
            "c4_novel_dark_fraction": c4["novel_dark_fraction"],
            "c4_halo_fraction": c4["halo_fraction"],
            "r0b_subpixel_filter_divergence": control[
                "subpixel_filter_divergence"
            ],
        }
    eligible = bool(gaps)
    return {
        "schema_version": 1,
        "decision": "c4_dtf_32_capacity_diagnostic_eligibility",
        "eligible": eligible,
        "next_candidate": "c4_dtf_32_diagnostic" if eligible else None,
        "c4_manifest_sha256": _require_sha256(
            c4_manifest_sha256, name="C4 manifest hash"
        ),
        "r0b_control_source": {
            "document": "docs/PROJECT_PLAN.md",
            "sha256": _require_sha256(
                r0b_document_sha256, name="R0b control document hash"
            ),
        },
        "c4_quality": c4,
        "r0b_control": control,
        "quality_gaps": gaps,
    }


def build_capacity_route_decision(
    c4_16_manifest: Mapping[str, Any],
    c4_32_manifest: Mapping[str, Any],
    *,
    c4_16_manifest_sha256: str,
    c4_32_manifest_sha256: str,
) -> dict[str, Any]:
    """Decide full width-32 versus the fresh learned C5 representation upper bound."""

    manifests = {"c4_dtf_16": c4_16_manifest, "c4_dtf_32": c4_32_manifest}
    expected = {
        "c4_dtf_16": {
            "name": "c4_dtf_16_relu_precheck",
            "latent_channels": 4,
            "decoder_width": 16,
            "activation": "relu",
            "role": "paired_precheck",
            "max_steps": 10_000,
        },
        "c4_dtf_32": {
            "name": "c4_dtf_32_diagnostic",
            "latent_channels": 4,
            "decoder_width": 32,
            "activation": "relu",
            "role": "conditional_capacity_diagnostic",
            "max_steps": 10_000,
        },
    }
    metrics: dict[str, dict[str, float]] = {}
    macs: dict[str, int] = {}
    for label, manifest in manifests.items():
        if not bool(manifest.get("valid")):
            raise ValueError(f"{label} capacity comparison run is invalid")
        candidate = dict(manifest.get("candidate", {}))
        for name, value in expected[label].items():
            if candidate.get(name) != value:
                raise ValueError(f"{label} candidate metadata differs from protocol")
        evaluation = dict(manifest.get("evaluation", {}))
        metrics[label] = {
            "base_color_linear_mae": float(evaluation["base_color_linear_mae"]),
            "normal_mean_degrees": float(evaluation["normal_mean_degrees"]),
            "roughness_mae": float(evaluation["roughness_mae"]),
            "metallic_mae": float(evaluation["metallic_mae"]),
            "generic_dark_fraction": float(evaluation["generic_dark_fraction"]),
            "generic_positive_halo_fraction": float(
                evaluation["generic_positive_halo_fraction"]
            ),
        }
        macs[label] = int(manifest["cost"]["decoder_macs_per_pixel"])

    def composite(values: Mapping[str, float]) -> float:
        return (
            values["base_color_linear_mae"]
            + math.radians(values["normal_mean_degrees"])
            + values["roughness_mae"]
            + values["metallic_mae"]
        )

    baseline = metrics["c4_dtf_16"]
    widened = metrics["c4_dtf_32"]
    baseline_composite = composite(baseline)
    widened_composite = composite(widened)
    composite_improvement = (
        baseline_composite - widened_composite
    ) / baseline_composite
    relative_changes = {
        name: (widened[name] - baseline[name]) / max(abs(baseline[name]), 1.0e-12)
        for name in baseline
    }
    minimum_composite_improvement = 0.10
    maximum_metric_regression = 0.05
    significant = (
        composite_improvement >= minimum_composite_improvement
        and all(change <= maximum_metric_regression for change in relative_changes.values())
    )
    return {
        "schema_version": 1,
        "decision": "c4_dtf_capacity_route",
        "significant_capacity_benefit": significant,
        "next_candidate": "c4_dtf_32_selected" if significant else "c5_dtf_16",
        "c5_is_channel_upper_bound": True,
        "policy": {
            "minimum_composite_improvement": minimum_composite_improvement,
            "maximum_per_metric_regression": maximum_metric_regression,
            "material_composite": "base_mae + radians(normal_mean_deg) + roughness_mae + metallic_mae",
        },
        "inputs": {
            "c4_dtf_16_manifest_sha256": _require_sha256(
                c4_16_manifest_sha256, name="C4-DTF-16 diagnostic manifest hash"
            ),
            "c4_dtf_32_manifest_sha256": _require_sha256(
                c4_32_manifest_sha256, name="C4-DTF-32 diagnostic manifest hash"
            ),
        },
        "comparison": {
            "c4_dtf_16": baseline,
            "c4_dtf_32": widened,
            "material_composite_c4_dtf_16": baseline_composite,
            "material_composite_c4_dtf_32": widened_composite,
            "material_composite_improvement": composite_improvement,
            "relative_metric_changes": relative_changes,
            "decoder_macs_ratio": macs["c4_dtf_32"] / float(macs["c4_dtf_16"]),
        },
    }


def _decoded_slice(value: DecodedMaterial, selected: slice) -> DecodedMaterial:
    return DecodedMaterial(
        value.base_color_linear[selected],
        value.normal_xy[selected],
        value.normal_xyz[selected],
        value.roughness[selected],
        value.metallic[selected],
    )


def _target_slice(value: Core4Targets, selected: slice) -> Core4Targets:
    base = value.base_color_linear[selected]
    return Core4Targets(
        base,
        value.normal_xyz[selected],
        value.roughness[selected],
        value.metallic[selected],
        height=1,
        width=base.shape[0],
    )


def paired_precheck_objective(
    prediction: DecodedMaterial,
    target: Core4Targets,
    *,
    anchor_slice: slice,
    quantization_error: torch.Tensor,
    loss_config: Mapping[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the generic material-only objective used by paired 10k runs."""

    count = prediction.base_color_linear.shape[0]
    if anchor_slice.start is None or anchor_slice.stop != count:
        raise ValueError("DTF precheck anchor slice must end the batch")
    if not (0 < anchor_slice.start < count):
        raise ValueError("DTF precheck requires non-empty subpixel and anchor slices")
    material_weights = {
        "base_color_l1": float(loss_config["base_color_linear_l1"]),
        "normal_cosine": float(loss_config["normal_cosine"]),
        "roughness_l1": float(loss_config["roughness_l1"]),
        "metallic_l1": float(loss_config["metallic_l1"]),
    }
    material, channels = material_loss(prediction, target, material_weights)
    subpixel_slice = slice(0, anchor_slice.start)
    subpixel, _ = material_loss(
        _decoded_slice(prediction, subpixel_slice),
        _target_slice(target, subpixel_slice),
        material_weights,
    )
    anchor, _ = material_loss(
        _decoded_slice(prediction, anchor_slice),
        _target_slice(target, anchor_slice),
        material_weights,
    )
    total = (
        material
        + float(loss_config["subpixel_material_anchor"]) * subpixel
        + float(loss_config["texel_center_anchor"]) * anchor
        + float(loss_config["unorm8_quantization"]) * quantization_error
    )
    return total, {
        "material": material,
        "subpixel_material_anchor": subpixel,
        "texel_center_anchor": anchor,
        "unorm8_quantization": quantization_error,
        **channels,
    }


def validate_decode_then_filter_protocol(config: Mapping[str, Any]) -> dict[str, Any]:
    """Reject changes to the approved DTF candidate and training contract."""

    if config.get("renderer") != "decode_then_filter_renderer_v1":
        raise ValueError("DTF renderer identifier differs from the frozen protocol")
    representation = dict(config.get("representation", {}))
    expected_representation = {
        "texture_resolution": [2048, 2048],
        "storage": "unorm8",
        "lod": 0,
        "mipmaps": False,
        "address_mode": "wrap",
        "latent_channel_upper_bound": 5,
    }
    if representation != expected_representation:
        raise ValueError("DTF representation differs from the frozen protocol")
    if dict(config.get("candidates", {})) != DTF_CANDIDATES:
        raise ValueError("DTF candidate matrix differs from the frozen protocol")
    training = dict(config.get("training", {}))
    if int(training.get("full_training_steps", -1)) != 80_000:
        raise ValueError("DTF full training must run for 80000 steps")
    if list(training.get("phases", [])) != DTF_PHASES:
        raise ValueError("DTF phase boundaries differ from the frozen protocol")
    if dict(config.get("batch_mix", {})) != DTF_BATCH_MIX:
        raise ValueError("DTF batch mix differs from the frozen protocol")
    if dict(config.get("loss", {})) != DTF_LOSS:
        raise ValueError("DTF loss differs from the frozen generic objective")
    pool = dict(config.get("training_pool", {}))
    frozen_pool = {
        name: pool.get(name)
        for name in (
            "camera_limit",
            "light_limit",
            "camera_triangle_coverage_increment_stop",
        )
    }
    if frozen_pool != {
        "camera_limit": 48,
        "light_limit": 6,
        "camera_triangle_coverage_increment_stop": 0.005,
    }:
        raise ValueError("DTF camera/light pool differs from the frozen protocol")
    checkpoint_tracks = list(config.get("selection", {}).get("checkpoint_tracks", []))
    if checkpoint_tracks != ["best-render", "best-artifact-safe"]:
        raise ValueError("DTF checkpoint tracks differ from the frozen protocol")
    return {
        "renderer": "decode_then_filter_renderer_v1",
        "quantization": "unorm8_ste_before_corner_fetch",
        "full_training_steps": 80_000,
        "phases": [dict(phase) for phase in DTF_PHASES],
        "batch_mix": dict(DTF_BATCH_MIX),
        "loss": dict(DTF_LOSS),
        "camera_pool_limit": 48,
        "light_pool_limit": 6,
        "camera_triangle_coverage_increment_stop": 0.005,
        "checkpoint_tracks": checkpoint_tracks,
    }


def _require_sha256(value: str, *, name: str) -> str:
    lowered = value.lower()
    if len(lowered) != 64 or any(character not in "0123456789abcdef" for character in lowered):
        raise ValueError(f"{name} must be a lowercase-compatible SHA-256 digest")
    return lowered


def build_decode_then_filter_manifest(
    config: Mapping[str, Any],
    *,
    candidate: str,
    config_sha256: str,
    input_hashes: Mapping[str, str],
    git_commit: str,
    selected_activation: str | None = None,
    activation_selection_sha256: str | None = None,
) -> dict[str, Any]:
    """Build a deterministic, output-free DTF run manifest."""

    protocol = validate_decode_then_filter_protocol(config)
    candidates = dict(config["candidates"])
    if candidate not in candidates:
        raise ValueError(f"unknown DTF candidate: {candidate}")
    candidate_spec = dict(candidates[candidate])
    activation = str(candidate_spec["activation"])
    selection_hash: str | None = None
    if activation == "selected_precheck_winner":
        if selected_activation not in {"relu", "silu"}:
            raise ValueError("full DTF manifest requires the selected ReLU/SiLU activation")
        if activation_selection_sha256 is None:
            raise ValueError("full DTF manifest requires activation-selection evidence")
        activation = selected_activation
        selection_hash = _require_sha256(
            activation_selection_sha256,
            name="activation selection hash",
        )
        candidate_spec["activation"] = activation
        candidate_spec["activation_source"] = "paired_10k_precheck"
    elif activation not in {"relu", "silu"}:
        raise ValueError("DTF manifest contains an unsupported activation")
    elif selected_activation is not None or activation_selection_sha256 is not None:
        raise ValueError("paired precheck manifests cannot override their activation")
    decoder = DecodeThenFilterDecoder(
        latent_channels=int(candidate_spec["latent_channels"]),
        width=int(candidate_spec["decoder_width"]),
        activation=activation,
    )
    texture_height, texture_width = config["representation"]["texture_resolution"]
    hashes = {
        name: _require_sha256(str(value), name=f"input hash {name}")
        for name, value in sorted(input_hashes.items())
    }
    if not hashes:
        raise ValueError("DTF manifest requires at least one input hash")
    if not git_commit:
        raise ValueError("DTF manifest requires a Git commit identifier")
    selection = dict(config["selection"])
    output_root = str(config["output_root"]).rstrip("/\\").replace("\\", "/")
    manifest_inputs: dict[str, Any] = {
        "config_sha256": _require_sha256(config_sha256, name="config hash"),
        "artifacts": hashes,
    }
    if selection_hash is not None:
        manifest_inputs["activation_selection_sha256"] = selection_hash
    return {
        "schema_version": 1,
        "experiment": str(config["experiment"]),
        "candidate": {"name": candidate, **candidate_spec},
        "renderer": {
            "identifier": "decode_then_filter_renderer_v1",
            "filter_order": "decode_four_corners_then_material_filter",
            "quantization": "unorm8_before_corner_fetch",
            "lod": 0,
            "mipmaps": False,
            "address_mode": "wrap",
        },
        "representation": dict(config["representation"]),
        "protocol": protocol,
        "cost": calculate_decode_then_filter_cost(
            decoder,
            height=int(texture_height),
            width=int(texture_width),
        ),
        "selection": selection,
        "retention": {
            "keep": [
                "rolling-resume",
                "best-render",
                "best-artifact-safe",
                "final",
            ],
            "cloud_scratch_upper_bound_bytes": 3 * 1024**3,
        },
        "inputs": manifest_inputs,
        "version_control": {
            "git_commit": git_commit,
            "repository_visibility": "private",
        },
        "output_dir": f"{output_root}/{candidate}",
    }


def build_decode_then_filter_continuation_manifest(
    config: Mapping[str, Any],
    *,
    candidate_name: str = "c5_dtf_16",
    stop_step: int = 120_000,
    source_training_manifest_sha256: str,
    source_checkpoint_sha256: str,
    config_sha256: str,
    input_hashes: Mapping[str, str],
    output_dir: str,
) -> dict[str, Any]:
    """Describe an approved isolated DTF continuation before it runs."""

    validate_decode_then_filter_protocol(config)
    approved = {
        ("c5_dtf_16", 120_000): (5, "c5_dtf_16_resume_120k"),
        ("c4_dtf_16_selected", 160_000): (4, "c4_dtf_16_resume_160k"),
    }
    try:
        latent_channels, output_name = approved[(candidate_name, stop_step)]
    except KeyError as exc:
        raise ValueError("unsupported DTF continuation candidate/endpoint") from exc
    normalized_output = output_dir.rstrip("/\\").replace("\\", "/")
    output_root = str(config["output_root"]).rstrip("/\\").replace("\\", "/")
    source_output = f"{output_root}/{candidate_name}"
    if normalized_output == source_output or not normalized_output.endswith(
        f"/{output_name}"
    ):
        raise ValueError("DTF continuation output must be its isolated resume directory")
    hashes = {
        name: _require_sha256(str(value), name=f"input hash {name}")
        for name, value in sorted(input_hashes.items())
    }
    if not hashes:
        raise ValueError("DTF continuation requires input hashes")
    training = dict(config["training"])
    return {
        "schema_version": 1,
        "experiment": f"scifihelmet_{output_name}_v1",
        "status": "planned_continuation",
        "candidate": {
            "name": candidate_name,
            "latent_channels": latent_channels,
            "decoder_width": 16,
            "activation": "relu",
        },
        "lineage": {
            "source_candidate": candidate_name,
            "source_step": 80_000,
            "source_training_manifest_sha256": _require_sha256(
                source_training_manifest_sha256,
                name="source training manifest hash",
            ),
            "source_checkpoint_sha256": _require_sha256(
                source_checkpoint_sha256,
                name="source checkpoint hash",
            ),
        },
        "training": {
            "start_step": 80_000,
            "stop_step": stop_step,
            "phase": "low_lr_continuation",
            "schedule": "cosine_1x_to_0.25x",
            "batch_mix": dict(config["batch_mix"]),
            "loss": dict(config["loss"]),
        },
        "selection": {
            "interval": int(training["selection_interval"]),
            "checkpoint_tracks": ["best-render", "best-artifact-safe"],
            "inherit_step_80k_as_initial_best": True,
        },
        "tracking": {
            "global_selection_evaluation": True,
            "yellow_material_subset": {
                "identifier": "reference_chroma_mask_v1",
                "evaluation_only": True,
                "affects_loss_or_sampling": False,
            },
        },
        "inputs": {
            "config_sha256": _require_sha256(config_sha256, name="config hash"),
            "artifacts": hashes,
        },
        "output_dir": normalized_output,
    }


def build_decode_then_filter_camera_finetune_manifest(
    config: Mapping[str, Any],
    *,
    source_training_manifest_sha256: str,
    source_checkpoint_sha256: str,
    config_sha256: str,
    input_hashes: Mapping[str, str],
    output_dir: str,
) -> dict[str, Any]:
    """Describe the isolated C4 camera-distribution fine-tune before it runs."""

    validate_decode_then_filter_protocol(config)
    profile = dict(config.get("camera_finetune", {}))
    expected = {
        "source_candidate": "c4_dtf_16_selected",
        "source_step": 160_000,
        "stop_step": 180_000,
        "output_name": "c4_dtf_16_camera31_ft_180k",
        "exact_continuation": False,
    }
    for name, value in expected.items():
        if profile.get(name) != value:
            raise ValueError(f"invalid DTF camera fine-tune field: {name}")
    strategy = str(
        dict(config["training_pool"]).get("camera_selection_strategy", "")
    )
    if strategy != "explicit_audited_pool_v1":
        raise ValueError("DTF camera fine-tune requires an explicit audited pool")
    camera_names = [str(spec["name"]) for spec in config["train_cameras"]]
    camera_limit = int(config["training_pool"]["camera_limit"])
    if not camera_names or len(camera_names) > camera_limit:
        raise ValueError("DTF camera fine-tune pool is empty or exceeds its limit")
    if len(set(camera_names)) != len(camera_names):
        raise ValueError("DTF camera fine-tune names must be unique")
    audit = dict(profile.get("audit", {}))
    if int(audit.get("selected_camera_count", -1)) != len(camera_names):
        raise ValueError("DTF camera audit count does not match the selected pool")

    normalized_output = output_dir.rstrip("/\\").replace("\\", "/")
    output_root = str(config["output_root"]).rstrip("/\\").replace("\\", "/")
    output_name = str(profile["output_name"])
    if not normalized_output.endswith(f"/{output_name}"):
        raise ValueError("DTF camera fine-tune output must use its isolated directory")
    if normalized_output == f"{output_root}/{profile['source_output_name']}":
        raise ValueError("DTF camera fine-tune must not overwrite its parent")
    hashes = {
        name: _require_sha256(str(value), name=f"input hash {name}")
        for name, value in sorted(input_hashes.items())
    }
    if not hashes:
        raise ValueError("DTF camera fine-tune requires input hashes")
    training = dict(config["training"])
    return {
        "schema_version": 1,
        "experiment": "scifihelmet_c4_dtf_camera31_finetune_v1",
        "status": "planned_camera_distribution_finetune",
        "candidate": {
            "name": "c4_dtf_16_selected",
            "latent_channels": 4,
            "decoder_width": 16,
            "activation": "relu",
        },
        "lineage": {
            "kind": "changed_camera_distribution_finetune",
            "exact_continuation": False,
            "source_candidate": "c4_dtf_16_selected",
            "source_step": 160_000,
            "source_output_name": str(profile["source_output_name"]),
            "source_training_manifest_sha256": _require_sha256(
                source_training_manifest_sha256,
                name="source training manifest hash",
            ),
            "source_checkpoint_sha256": _require_sha256(
                source_checkpoint_sha256,
                name="source checkpoint hash",
            ),
        },
        "training": {
            "start_step": 160_000,
            "stop_step": 180_000,
            "phase": str(profile["phase"]),
            "schedule": str(profile["schedule"]),
            "batch_mix": dict(config["batch_mix"]),
            "loss": dict(config["loss"]),
        },
        "camera_distribution": {
            "identifier": str(profile["identifier"]),
            "strategy": strategy,
            "camera_count": len(camera_names),
            "camera_limit": camera_limit,
            "selected_names": camera_names,
            "audit": audit,
            "yellow_audit_influenced_selection": True,
            "dynamic_roi_sampling": False,
            "loss_changed": False,
        },
        "selection": {
            "interval": int(training["selection_interval"]),
            "checkpoint_tracks": ["best-render", "best-artifact-safe"],
            "inherit_step_160k_as_initial_best": True,
        },
        "inputs": {
            "config_sha256": _require_sha256(config_sha256, name="config hash"),
            "artifacts": hashes,
        },
        "output_dir": normalized_output,
    }


def build_decode_then_filter_camera31_fresh_manifest(
    config: Mapping[str, Any],
    *,
    config_sha256: str,
    input_hashes: Mapping[str, str],
    git_commit: str,
    selected_activation: str,
    activation_selection_sha256: str,
    output_dir: str,
) -> dict[str, Any]:
    """Describe an isolated fresh C4 run using the audited 31-camera pool."""

    if selected_activation != "relu":
        raise ValueError("camera31 fresh C4 requires the frozen ReLU selection")
    base = build_decode_then_filter_manifest(
        config,
        candidate="c4_dtf_16_selected",
        config_sha256=config_sha256,
        input_hashes=input_hashes,
        git_commit=git_commit,
        selected_activation=selected_activation,
        activation_selection_sha256=activation_selection_sha256,
    )
    strategy = str(
        dict(config["training_pool"]).get("camera_selection_strategy", "")
    )
    if strategy != "explicit_audited_pool_v1":
        raise ValueError("camera31 fresh C4 requires an explicit audited pool")
    camera_names = [str(spec["name"]) for spec in config["train_cameras"]]
    camera_limit = int(config["training_pool"]["camera_limit"])
    if len(camera_names) != 31 or len(set(camera_names)) != len(camera_names):
        raise ValueError("camera31 fresh C4 requires 31 uniquely named cameras")
    if len(camera_names) > camera_limit:
        raise ValueError("camera31 fresh C4 exceeds the frozen camera limit")
    audit = dict(config.get("camera_finetune", {}).get("audit", {}))
    if int(audit.get("selected_camera_count", -1)) != len(camera_names):
        raise ValueError("camera31 fresh C4 audit count does not match the pool")
    output_name = "c4_dtf_16_camera31_fresh_80k"
    normalized_output = output_dir.rstrip("/\\").replace("\\", "/")
    if not normalized_output.endswith(f"/{output_name}"):
        raise ValueError("camera31 fresh C4 output must use its isolated directory")
    training = dict(config["training"])
    base.update(
        {
            "experiment": "scifihelmet_c4_dtf_camera31_fresh_80k_v1",
            "status": "planned_camera31_fresh_training",
            "candidate": {
                "name": "c4_dtf_16_selected",
                "latent_channels": 4,
                "decoder_width": 16,
                "activation": "relu",
            },
            "lineage": {
                "kind": "fresh_random_initialization",
                "source_step": 0,
                "parent_checkpoint_used": False,
            },
            "training": {
                "start_step": 0,
                "stop_step": 80_000,
                "phases": [dict(phase) for phase in training["phases"]],
                "batch_mix": dict(config["batch_mix"]),
                "loss": dict(config["loss"]),
            },
            "camera_distribution": {
                "identifier": "c4_dtf_camera31_distribution_v1",
                "strategy": strategy,
                "camera_count": len(camera_names),
                "camera_limit": camera_limit,
                "selected_names": camera_names,
                "audit": audit,
                "yellow_audit_influenced_selection": True,
                "dynamic_roi_sampling": False,
                "loss_changed": False,
            },
            "tracking": {
                "global_selection_evaluation": True,
                "yellow_material_subset": {
                    "identifier": "reference_chroma_mask_v1",
                    "evaluation_only": True,
                    "affects_loss_or_sampling": False,
                },
            },
            "output_dir": normalized_output,
        }
    )
    return base
