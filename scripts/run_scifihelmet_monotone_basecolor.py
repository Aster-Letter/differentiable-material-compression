"""Run monotone BaseColor continuation directly from raw PCA to 10k."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cg_frontier.assets.gltf_mesh import load_gltf_mesh  # noqa: E402
from cg_frontier.compression.exact_basecolor_experiment import (  # noqa: E402
    RenderCase,
    atomic_torch_save,
    candidate_forward_parity,
    candidate_state_hash,
    display_transform,
    evaluate_render_pool,
    evaluate_texels,
    load_texel_targets,
    orbit_camera,
    stable_json_bytes,
    tensor_sha256,
)
from cg_frontier.compression.monotone_basecolor import (  # noqa: E402
    BaseColorMetrics,
    basecolor_tail_ids,
    balanced_basecolor_loss,
    build_color_partition,
    checkpoint_payload,
    clone_training_state,
    composite_curve_acceptance,
    composite_curve_alpha_for_value,
    composite_curve_target,
    evaluate_basecolor_constraints,
    export_candidate,
    interpolate_candidate_state_,
    load_raw_pca_candidate,
    restore_training_state,
    sample_balanced_ids,
    train_stochastic_step,
    validate_checkpoint,
)
from cg_frontier.render.gbuffer import (  # noqa: E402
    load_core4_textures,
    render_geometry_gbuffer,
    sample_core4_material,
)
from cg_frontier.render.pbr import PointLight, linear_to_srgb_torch, shade_ggx  # noqa: E402


def _path(value: str) -> Path:
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"path escapes repository: {value}")
    return path


def _read_yaml(path: Path) -> Mapping[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return value


def _load_config(path: Path) -> tuple[Mapping[str, Any], Mapping[str, Any], str]:
    raw = path.read_bytes()
    config = _read_yaml(path)
    if config.get("schema_version") != 1 or config.get("experiment") != "scifihelmet_monotone_basecolor_curve_v7":
        raise ValueError("unsupported monotone BaseColor config")
    scope = config.get("scope")
    if not isinstance(scope, Mapping) or scope.get("formal_holdout_accessed") is not False:
        raise ValueError("formal holdout must remain sealed")
    contract = _read_yaml(_path(str(config["inputs"]["camera_contract"])))
    if len(contract.get("train_cameras", [])) != 31 or len(contract.get("train_lights", [])) != 6:
        raise ValueError("camera/light contract must remain 31x6")
    checkpoints = [int(value) for value in config["training"]["checkpoints"]]
    if checkpoints != [5000, 10000] or int(config["training"]["final_steps"]) != 10000:
        raise ValueError("C-monotone v1 uses 5k/10k as a maximum local training budget")
    curve = config["constraints"]["composite_curve"]
    if curve.get("kind") != "normalized_composite_smoothstep_power":
        raise ValueError("unsupported BaseColor composite curve")
    alpha = float(curve["initial_alpha"])
    delta = float(curve["initial_delta_alpha"])
    minimum = float(curve["minimum_delta_alpha"])
    maximum = float(curve["maximum_delta_alpha"])
    if not 0.0 <= alpha <= 1.0 or not 0.0 <= minimum <= delta <= maximum <= 1.0:
        raise ValueError("invalid preset-curve alpha schedule")
    if not 0.0 < float(curve["backoff_factor"]) < 1.0 or float(curve["growth_factor"]) < 1.0:
        raise ValueError("invalid preset-curve growth/backoff factors")
    return config, contract, hashlib.sha256(raw).hexdigest()


def _camera(spec: Mapping[str, Any], render: Mapping[str, Any]):
    return orbit_camera(
        yaw_degrees=float(spec["yaw_degrees"]),
        elevation_degrees=float(spec["elevation_degrees"]),
        radius=float(spec.get("radius", render["camera_radius"])),
        target=spec.get("target", render["target"]),
        up=render["up"],
        vertical_fov_degrees=float(render["vertical_fov_degrees"]),
        near=float(render["near"]),
        far=float(render["far"]),
    )


def _light(spec: Mapping[str, Any]) -> PointLight:
    return PointLight(
        position=tuple(float(value) for value in spec["position"]),
        color=tuple(float(value) for value in spec["color"]),
        radiant_intensity=float(spec["radiant_intensity"]),
        ambient_intensity=float(spec["ambient_intensity"]),
    )


def _build_render_cases(config: Mapping[str, Any], contract: Mapping[str, Any], textures, device: torch.device):
    mesh = load_gltf_mesh(_path(str(config["inputs"]["gltf"])))
    render = contract["render"]
    resolution = tuple(int(value) for value in render["resolution"])
    cases = []
    for spec in contract["train_cameras"]:
        camera = _camera(spec, render)
        geometry = render_geometry_gbuffer(mesh, camera, resolution, device=device)
        reference = sample_core4_material(geometry, textures)
        valid = torch.nonzero(geometry.torch_buffers["mask"].reshape(-1), as_tuple=False).reshape(-1)
        if valid.numel() == 0:
            raise ValueError(f"camera has no visible pixels: {spec['name']}")
        cases.append(RenderCase(str(spec["name"]), camera, geometry, reference, valid))
    return cases, [_light(spec) for spec in contract["train_lights"]]


def _load_state(config: Mapping[str, Any], device: torch.device):
    textures_cpu = load_core4_textures(_path(str(config["inputs"]["core4_manifest"])), device="cpu")
    targets_cpu = load_texel_targets(textures_cpu)
    candidate, artifact = load_raw_pca_candidate(
        _path(str(config["inputs"]["raw_pca_artifact"])),
        targets_cpu,
        expected_artifact_hash=str(config["inputs"]["raw_pca_artifact_hash"]),
        device=device,
    )
    partition_cpu = build_color_partition(targets_cpu.base_q8, bins=int(config["constraints"]["bins"]))
    return targets_cpu, candidate, partition_cpu.to(device), artifact


def _optimizers(candidate):
    return (
        torch.optim.Adam(candidate.code_parameters(), lr=0.02),
        torch.optim.Adam(candidate.decoder_parameters(), lr=0.001),
    )


def _gradient_stats(loss: torch.Tensor, parameters, *, retain_graph: bool) -> dict[str, Any]:
    gradients = torch.autograd.grad(loss, tuple(parameters), allow_unused=True, retain_graph=retain_graph)
    present = [value for value in gradients if value is not None]
    l1 = sum(float(value.abs().sum()) for value in present)
    return {
        "finite": bool(present) and all(bool(torch.isfinite(value).all()) for value in present),
        "nonzero": l1 > 0.0,
        "l1": l1,
        "max_abs": max((float(value.abs().max()) for value in present), default=0.0),
    }


def _zero_update_audit(candidate, targets, partition, cases, lights, contract, seed):
    generator = torch.Generator(device=candidate.latent_byte.device).manual_seed(seed)
    ids = sample_balanced_ids(partition, sample_count=4096, generator=generator)
    prediction = candidate.decoder(candidate.latent_for_ids(ids, ste=True)).base_color_linear
    target = targets.base_q8[ids].to(torch.float32) / 255.0
    color, _ = balanced_basecolor_loss(prediction, target, partition.group_ids[ids])
    parameters = list(candidate.parameters())
    color_gradient = _gradient_stats(color, parameters, retain_graph=False)
    case = cases[0]
    screen = case.valid_flat_indices[:4096]
    from cg_frontier.compression.exact_basecolor_experiment import render_pair_loss

    hdr, display, _ = render_pair_loss(
        candidate,
        case,
        lights[0],
        screen,
        height=targets.height,
        width=targets.width,
        minimum_roughness=float(contract["render"]["minimum_roughness"]),
        display_exposure=float(contract["render"]["display_exposure"]),
        ste=True,
    )
    render_gradient = _gradient_stats(hdr + 0.25 * display, parameters, retain_graph=False)
    if not all(item["finite"] and item["nonzero"] for item in (color_gradient, render_gradient)):
        raise AssertionError("C-monotone zero-update gradient audit failed")
    return {"basecolor": color_gradient, "render": render_gradient}


def _controller(config: Mapping[str, Any], initial: BaseColorMetrics) -> dict[str, Any]:
    curve = config["constraints"]["composite_curve"]
    return {
        "base_scale": float(config["training"]["base_scale"]),
        "learning_rate_scale": 1.0,
        "accepted_step": 0,
        "initial_metrics": initial.as_dict(),
        "accepted_metrics": initial.as_dict(),
        "curve_alpha": float(curve["initial_alpha"]),
        "delta_alpha": float(curve["initial_delta_alpha"]),
        "converged_early": False,
        "convergence_step": None,
        "consecutive_hold_audits": 0,
        "audits": [],
        "rejected_attempts": 0,
    }


def _run_until(
    candidate,
    targets,
    partition,
    cases,
    lights,
    config,
    contract,
    generator,
    code_optimizer,
    decoder_optimizer,
    controller,
    initial_metrics,
    *,
    start: int,
    stop: int,
    log_path: Path | None = None,
):
    training = config["training"]
    constraints = config["constraints"]
    audit_interval = int(training["audit_interval"])
    final_step = int(training["final_steps"])
    current = start
    accepted_metrics = BaseColorMetrics(**controller["accepted_metrics"])
    last = None
    while current < stop:
        block_stop = min(stop, ((current // audit_interval) + 1) * audit_interval)
        snapshot = clone_training_state(candidate, code_optimizer, decoder_optimizer, generator)
        tail_ids = basecolor_tail_ids(
            candidate,
            targets,
            tail_fraction=float(constraints["tail_fraction"]),
        )
        attempts = []
        accepted = False
        accepted_logs = []
        disposition = "rejected"
        curve = constraints["composite_curve"]
        minimum_delta = float(curve["minimum_delta_alpha"])
        attempted_delta = float(controller["delta_alpha"])
        for attempt in range(int(training["maximum_attempts"])):
            if attempt:
                restore_training_state(snapshot, candidate, code_optimizer, decoder_optimizer, generator)
            scale = float(controller["base_scale"]) * math.pow(float(training["retry_multiplier"]), attempt)
            lr_scale = float(controller["learning_rate_scale"]) * math.pow(float(training["retry_lr_factor"]), attempt)
            scale = min(float(training["maximum_base_scale"]), scale)
            lr_scale = max(float(training["minimum_learning_rate_scale"]), lr_scale)
            logs = []
            for step in range(current + 1, block_stop + 1):
                last = train_stochastic_step(
                    candidate,
                    targets,
                    partition,
                    tail_ids,
                    cases,
                    lights,
                    generator=generator,
                    texel_batch_size=int(training["texel_batch_size"]),
                    color_batch_size=int(training["color_batch_size"]),
                    screen_batch_size=int(training["screen_batch_size"]),
                    minimum_roughness=float(contract["render"]["minimum_roughness"]),
                    display_exposure=float(contract["render"]["display_exposure"]),
                    code_optimizer=code_optimizer,
                    decoder_optimizer=decoder_optimizer,
                    step=step,
                    stop=final_step,
                    base_scale=scale,
                    learning_rate_scale=lr_scale,
                )
                if step == current + 1 or step % int(training["log_interval"]) == 0:
                    logs.append(last)
            measured = evaluate_basecolor_constraints(
                candidate,
                targets,
                partition,
                tail_fraction=float(constraints["tail_fraction"]),
            )
            raw_delta = float(controller["delta_alpha"]) * math.pow(float(curve["backoff_factor"]), attempt)
            attempted_delta = 0.0 if float(controller["delta_alpha"]) == 0.0 else max(minimum_delta, raw_delta)
            target_alpha = min(1.0, float(controller["curve_alpha"]) + attempted_delta)
            target_composite = composite_curve_target(
                alpha=target_alpha,
                floor=float(curve["floor"]),
                exponent=float(curve["exponent"]),
            )
            passed, checks = composite_curve_acceptance(
                measured,
                initial_metrics,
                weights=curve["weights"],
                target=target_composite,
                guard_multipliers=curve["guard_multipliers"],
                relative_tolerance=float(constraints["relative_tolerance"]),
                absolute_tolerance=float(constraints["absolute_tolerance"]),
            )
            attempts.append(
                {
                    "attempt": attempt + 1,
                    "base_scale": scale,
                    "learning_rate_scale": lr_scale,
                    "curve_alpha_before": float(controller["curve_alpha"]),
                    "attempted_delta_alpha": attempted_delta,
                    "target_alpha": target_alpha,
                    "target_composite": target_composite,
                    "metrics": measured.as_dict(),
                    "checks": checks,
                    "accepted": passed,
                }
            )
            if passed:
                accepted = True
                disposition = "advance"
                accepted_logs = logs
                accepted_metrics = measured
                achieved_alpha = composite_curve_alpha_for_value(
                    float(checks["composite"]["value"]),
                    floor=float(curve["floor"]),
                    exponent=float(curve["exponent"]),
                )
                controller["base_scale"] = min(
                    float(training["maximum_base_scale"]),
                    max(float(training["base_scale"]), scale * 0.5 if attempt else scale),
                )
                controller["learning_rate_scale"] = max(
                    float(training["minimum_learning_rate_scale"]),
                    min(1.0, lr_scale * 2.0 if attempt else lr_scale),
                )
                controller["curve_alpha"] = max(target_alpha, achieved_alpha)
                controller["delta_alpha"] = min(
                    float(curve["maximum_delta_alpha"]),
                    attempted_delta * float(curve["growth_factor"]),
                )
                controller["consecutive_hold_audits"] = 0
                attempts[-1]["achieved_alpha"] = achieved_alpha
                break
            controller["rejected_attempts"] += 1
            if (
                attempted_delta <= minimum_delta + 1.0e-12
                and scale >= float(training["maximum_base_scale"])
                and lr_scale <= float(training["minimum_learning_rate_scale"])
            ):
                break
        if not accepted:
            if attempted_delta > minimum_delta + 1.0e-12:
                raise RuntimeError(
                    "maximum_attempts did not reach minimum_delta_alpha; curve schedule is internally inconsistent"
                )
            hold_target = composite_curve_target(
                alpha=float(controller["curve_alpha"]),
                floor=float(curve["floor"]),
                exponent=float(curve["exponent"]),
            )
            hold_passed, hold_checks = composite_curve_acceptance(
                measured,
                initial_metrics,
                weights=curve["weights"],
                target=hold_target,
                guard_multipliers=curve["guard_multipliers"],
                relative_tolerance=float(constraints["relative_tolerance"]),
                absolute_tolerance=float(constraints["absolute_tolerance"]),
            )
            projection_attempts = []
            projection_fraction = 1.0
            if not hold_passed:
                end_state = {
                    name: value.detach().clone()
                    for name, value in candidate.state_dict().items()
                }
                for projection_attempt in range(1, int(training["maximum_projection_attempts"]) + 1):
                    projection_fraction = math.pow(float(training["projection_backoff_factor"]), projection_attempt)
                    interpolate_candidate_state_(
                        candidate,
                        snapshot["candidate_state"],
                        end_state,
                        projection_fraction,
                    )
                    projected_metrics = evaluate_basecolor_constraints(
                        candidate,
                        targets,
                        partition,
                        tail_fraction=float(constraints["tail_fraction"]),
                    )
                    projected_passed, projected_checks = composite_curve_acceptance(
                        projected_metrics,
                        initial_metrics,
                        weights=curve["weights"],
                        target=hold_target,
                        guard_multipliers=curve["guard_multipliers"],
                        relative_tolerance=float(constraints["relative_tolerance"]),
                        absolute_tolerance=float(constraints["absolute_tolerance"]),
                    )
                    projection_attempts.append(
                        {
                            "fraction": projection_fraction,
                            "metrics": projected_metrics.as_dict(),
                            "checks": projected_checks,
                            "accepted": projected_passed,
                        }
                    )
                    if projected_passed:
                        measured = projected_metrics
                        hold_checks = projected_checks
                        hold_passed = True
                        break
            hold_count = int(controller["consecutive_hold_audits"])
            achieved_alpha = (
                composite_curve_alpha_for_value(
                    float(hold_checks["composite"]["value"]),
                    floor=float(curve["floor"]),
                    exponent=float(curve["exponent"]),
                )
                if hold_passed
                else None
            )
            advanced = bool(
                achieved_alpha is not None
                and achieved_alpha > float(controller["curve_alpha"]) + 1.0e-12
            )
            if hold_passed and (advanced or hold_count < int(training["maximum_hold_audits"])):
                accepted = True
                disposition = "projected_advance" if advanced else "hold"
                accepted_logs = logs
                accepted_metrics = measured
                controller["base_scale"] = min(
                    float(training["maximum_base_scale"]),
                    max(float(training["base_scale"]), scale * 0.5),
                )
                controller["learning_rate_scale"] = max(
                    float(training["minimum_learning_rate_scale"]),
                    min(1.0, lr_scale * 2.0),
                )
                controller["delta_alpha"] = minimum_delta
                controller["curve_alpha"] = max(float(controller["curve_alpha"]), float(achieved_alpha))
                controller["consecutive_hold_audits"] = 0 if advanced else hold_count + 1
            else:
                restore_training_state(snapshot, candidate, code_optimizer, decoder_optimizer, generator)
                disposition = "converged"
                controller["converged_early"] = True
                controller["convergence_step"] = current
                controller["failed_target_step"] = block_stop
            attempts[-1]["hold_target_composite"] = hold_target
            attempts[-1]["hold_checks"] = hold_checks
            attempts[-1]["hold_accepted"] = disposition in ("hold", "projected_advance")
            attempts[-1]["achieved_alpha"] = (
                achieved_alpha if disposition in ("hold", "projected_advance") else None
            )
            attempts[-1]["hold_projection_fraction"] = projection_fraction if hold_passed else None
            attempts[-1]["hold_projection_attempts"] = projection_attempts
        controller["audits"].append(
            {"step": block_stop, "attempts": attempts, "accepted": accepted, "disposition": disposition}
        )
        if not accepted:
            break
        controller["accepted_step"] = block_stop
        controller["accepted_metrics"] = accepted_metrics.as_dict()
        if log_path is not None:
            with log_path.open("a", encoding="utf-8") as stream:
                for item in accepted_logs:
                    stream.write(json.dumps(item, sort_keys=True, separators=(",", ":")) + "\n")
        current = block_stop
    return {"last": last, "step": current, "converged": bool(controller["converged_early"])}


def _save_basecolor(candidate, targets, path: Path) -> None:
    pieces = []
    with torch.no_grad():
        for start in range(0, candidate.texel_count, 262144):
            ids = torch.arange(start, min(start + 262144, candidate.texel_count), device=candidate.latent_byte.device)
            pieces.append(candidate.decoder(candidate.latent_for_ids(ids, ste=False)).base_color_linear.cpu())
    linear = torch.cat(pieces).reshape(targets.height, targets.width, 3).clamp(0.0, 1.0)
    Image.fromarray(torch.floor(linear_to_srgb_torch(linear) * 255.0 + 0.5).to(torch.uint8).numpy(), mode="RGB").save(path)


def _save_reference_views(cases, light, contract, indices: Sequence[int], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for index in indices:
            case = cases[index]
            hdr = shade_ggx(
                case.geometry,
                case.camera,
                light,
                material_override=case.reference,
                minimum_roughness=float(contract["render"]["minimum_roughness"]),
            )
            display = display_transform(hdr, float(contract["render"]["display_exposure"]))
            image = torch.floor(display.clamp(0, 1) * 255.0 + 0.5).to(torch.uint8).cpu().numpy()
            Image.fromarray(image, mode="RGB").save(output / f"camera_{index:02d}.png")


def _evaluate(candidate, targets_cpu, cases, lights, config, contract, *, step: int, output: Path, full: bool):
    indices = None if full else [int(value) for value in config["evaluation"]["intermediate_camera_indices"]]
    return {
        "texel": evaluate_texels(candidate, targets_cpu),
        "basecolor_constraints": evaluate_basecolor_constraints(
            candidate,
            targets_cpu,
            build_color_partition(targets_cpu.base_q8, bins=int(config["constraints"]["bins"])).to(candidate.latent_byte.device),
            tail_fraction=float(config["constraints"]["tail_fraction"]),
        ).as_dict(),
        "render": evaluate_render_pool(
            candidate,
            cases,
            lights,
            height=targets_cpu.height,
            width=targets_cpu.width,
            minimum_roughness=float(contract["render"]["minimum_roughness"]),
            display_exposure=float(contract["render"]["display_exposure"]),
            case_indices=indices,
            save_images=output / f"render_{step:05d}",
        ),
    }


def run_preflight(config, contract, config_hash, device):
    textures = load_core4_textures(_path(str(config["inputs"]["core4_manifest"])), device=device)
    cases, lights = _build_render_cases(config, contract, textures, device)
    targets_cpu, direct, partition, artifact = _load_state(config, device)
    targets = targets_cpu.to(device)
    initial = evaluate_basecolor_constraints(direct, targets_cpu, partition, tail_fraction=float(config["constraints"]["tail_fraction"]))
    audit = _zero_update_audit(direct, targets, partition, cases, lights, contract, int(config["training"]["seed"]))
    generator = torch.Generator(device=device).manual_seed(int(config["training"]["seed"]))
    code_optimizer, decoder_optimizer = _optimizers(direct)
    preflight_config = copy.deepcopy(config)
    preflight_config["training"]["audit_interval"] = 10
    preflight_config["constraints"]["relative_tolerance"] = 0.02
    preflight_curve = preflight_config["constraints"]["composite_curve"]
    preflight_curve["initial_delta_alpha"] = 0.0
    preflight_curve["minimum_delta_alpha"] = 0.0
    preflight_curve["maximum_delta_alpha"] = 0.0
    preflight_curve["growth_factor"] = 1.0
    controller = _controller(preflight_config, initial)
    _run_until(
        direct, targets, partition, cases, lights, preflight_config, contract, generator, code_optimizer, decoder_optimizer,
        controller, initial, start=0, stop=11,
    )
    direct_hash = candidate_state_hash(direct)

    _, staged, staged_partition, _ = _load_state(config, device)
    staged_initial = evaluate_basecolor_constraints(staged, targets_cpu, staged_partition, tail_fraction=float(config["constraints"]["tail_fraction"]))
    staged_generator = torch.Generator(device=device).manual_seed(int(config["training"]["seed"]))
    staged_code, staged_decoder = _optimizers(staged)
    staged_controller = _controller(preflight_config, staged_initial)
    _run_until(
        staged, targets, staged_partition, cases, lights, preflight_config, contract, staged_generator, staged_code, staged_decoder,
        staged_controller, staged_initial, start=0, stop=10,
    )
    target_hash = tensor_sha256(targets_cpu.base_q8)
    payload = checkpoint_payload(
        candidate=staged,
        step=10,
        code_optimizer=staged_code,
        decoder_optimizer=staged_decoder,
        generator=staged_generator,
        config_hash=config_hash,
        target_hash=target_hash,
        artifact_hash=artifact["artifact_hash"],
        partition=staged_partition,
        initial_metrics=staged_initial,
        accepted_metrics=BaseColorMetrics(**staged_controller["accepted_metrics"]),
        controller=staged_controller,
    )
    validate_checkpoint(payload, config_hash=config_hash, target_hash=target_hash, artifact_hash=artifact["artifact_hash"])
    _, resumed, resumed_partition, _ = _load_state(config, device)
    resumed_generator = torch.Generator(device=device)
    resumed_code, resumed_decoder = _optimizers(resumed)
    restore_training_state(payload, resumed, resumed_code, resumed_decoder, resumed_generator)
    resumed_controller = dict(payload["controller"])
    _run_until(
        resumed, targets, resumed_partition, cases, lights, preflight_config, contract, resumed_generator, resumed_code, resumed_decoder,
        resumed_controller, BaseColorMetrics(**payload["initial_metrics"]), start=10, stop=11,
    )
    resumed_hash = candidate_state_hash(resumed)
    if direct_hash != resumed_hash:
        raise AssertionError("C-monotone 10->checkpoint->11 exact resume mismatch")
    report = {
        "schema_version": 1,
        "artifact": artifact,
        "initial_metrics": initial.as_dict(),
        "partition": partition.specification(),
        "zero_update_gradient_audit": audit,
        "preflight_constraint_policy": {
            "purpose": "finite-gradient, rollback, and exact-resume correctness only",
            "audit_interval": 10,
            "relative_tolerance": 0.02,
            "curve_alpha_held_at": float(preflight_curve["initial_alpha"]),
            "formal_composite_curve_unchanged": dict(config["constraints"]["composite_curve"]),
        },
        "hard_fake_parity": candidate_forward_parity(staged),
        "direct_step11_hash": direct_hash,
        "resumed_step11_hash": resumed_hash,
        "exact_resume": True,
        "scope": dict(config["scope"]),
    }
    output = _path(str(config["output_root"])) / "preflight"
    output.mkdir(parents=True, exist_ok=True)
    (output / "preflight_report.json").write_bytes(stable_json_bytes(report))
    return report


def run_train(config, contract, config_hash, device):
    textures = load_core4_textures(_path(str(config["inputs"]["core4_manifest"])), device=device)
    cases, lights = _build_render_cases(config, contract, textures, device)
    targets_cpu, candidate, partition, artifact = _load_state(config, device)
    targets = targets_cpu.to(device)
    initial = evaluate_basecolor_constraints(candidate, targets_cpu, partition, tail_fraction=float(config["constraints"]["tail_fraction"]))
    generator = torch.Generator(device=device).manual_seed(int(config["training"]["seed"]))
    code_optimizer, decoder_optimizer = _optimizers(candidate)
    controller = _controller(config, initial)
    target_hash = tensor_sha256(targets_cpu.base_q8)
    output = _path(str(config["output_root"])) / "training" / "C-monotone"
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "candidate": candidate.name,
        "artifact": artifact,
        "partition": partition.specification(),
        "constraint_policy": {
            "kind": "adaptive_progress_on_fixed_normalized_composite_curve",
            "composite_curve": dict(config["constraints"]["composite_curve"]),
            "alpha_is_constraint_progress_not_step_fraction": True,
            "hold_audits_before_convergence": int(config["training"]["maximum_hold_audits"]),
            "no_fixed_10k_endpoint": True,
        },
        "step_metrics": {"0": _evaluate(candidate, targets_cpu, cases, lights, config, contract, step=0, output=output, full=True)},
        "checkpoints": {},
        "scope": dict(config["scope"]),
    }
    current = 0
    for stop in [int(value) for value in config["training"]["checkpoints"]]:
        outcome = _run_until(
            candidate, targets, partition, cases, lights, config, contract, generator, code_optimizer, decoder_optimizer,
            controller, initial, start=current, stop=stop, log_path=output / "train.jsonl",
        )
        current = int(outcome["step"])
        payload = checkpoint_payload(
            candidate=candidate,
            step=current,
            code_optimizer=code_optimizer,
            decoder_optimizer=decoder_optimizer,
            generator=generator,
            config_hash=config_hash,
            target_hash=target_hash,
            artifact_hash=artifact["artifact_hash"],
            partition=partition,
            initial_metrics=initial,
            accepted_metrics=BaseColorMetrics(**controller["accepted_metrics"]),
            controller=controller,
        )
        checkpoint = output / f"checkpoint_{current:05d}.pt"
        atomic_torch_save(payload, checkpoint, immutable=True)
        report["checkpoints"][str(current)] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        report["step_metrics"][str(current)] = _evaluate(
            candidate,
            targets_cpu,
            cases,
            lights,
            config,
            contract,
            step=current,
            output=output,
            full=bool(outcome["converged"]) or current == 10000,
        )
        (output / "training_report.partial.json").write_bytes(stable_json_bytes(report | {"controller": controller}))
        if bool(outcome["converged"]):
            break
    final_metrics = BaseColorMetrics(**controller["accepted_metrics"])
    final_target = composite_curve_target(
        alpha=float(controller["curve_alpha"]),
        floor=float(config["constraints"]["composite_curve"]["floor"]),
        exponent=float(config["constraints"]["composite_curve"]["exponent"]),
    )
    passed, checks = composite_curve_acceptance(
        final_metrics,
        initial,
        weights=config["constraints"]["composite_curve"]["weights"],
        target=final_target,
        guard_multipliers=config["constraints"]["composite_curve"]["guard_multipliers"],
        relative_tolerance=float(config["constraints"]["relative_tolerance"]),
        absolute_tolerance=float(config["constraints"]["absolute_tolerance"]),
    )
    if not passed:
        raise AssertionError("C-monotone final hard constraint gate failed")
    runtime = export_candidate(candidate, output / "export")
    (output / "export" / "export_manifest.json").write_bytes(stable_json_bytes(runtime))
    _save_basecolor(candidate, targets_cpu, output / f"basecolor_decoded_{current:05d}.png")
    _save_reference_views(cases, lights[0], contract, config["evaluation"]["intermediate_camera_indices"], output / "render_reference")
    report.update(
        {
            "controller": controller,
            "endpoint_step": current,
            "endpoint_curve_alpha": float(controller["curve_alpha"]),
            "endpoint_composite_target": final_target,
            "final_constraint_checks": checks,
            "achieved_relative_improvement": {
                name: (initial.constrained()[name] - final_metrics.constrained()[name])
                / max(initial.constrained()[name], 1.0e-12)
                for name in initial.constrained()
            },
            "monotone_certificate_passed": True,
            "runtime_export": runtime,
            "formal_holdout_accessed": False,
        }
    )
    (output / "training_report.json").write_bytes(stable_json_bytes(report))
    return report


def _contact_sheet(columns: Sequence[tuple[str, Path]], cameras: Sequence[int], output: Path) -> None:
    cell = 384
    header = 44
    canvas = Image.new("RGB", (cell * len(columns), header + cell * len(cameras)), "#111111")
    draw = ImageDraw.Draw(canvas)
    for column, (label, root) in enumerate(columns):
        draw.text((column * cell + 8, 12), label, fill="white")
        for row, camera in enumerate(cameras):
            image = Image.open(root / f"camera_{camera:02d}.png").convert("RGB").resize((cell, cell))
            canvas.paste(image, (column * cell, header + row * cell))
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def run_report(config):
    root = _path(str(config["output_root"]))
    training_path = root / "training" / "C-monotone" / "training_report.json"
    training = json.loads(training_path.read_text(encoding="utf-8"))
    if not training.get("monotone_certificate_passed") or training.get("formal_holdout_accessed") is not False:
        raise ValueError("C-monotone training report is incomplete")
    exact_root = _path("outputs/scifihelmet_exact_basecolor_v1/training")
    cameras = [int(value) for value in config["evaluation"]["intermediate_camera_indices"]]
    endpoint = int(training["endpoint_step"])
    previous_root = _path("outputs/scifihelmet_monotone_basecolor_curve_v5/training/C-monotone")
    columns = [
        ("Source", root / "training" / "C-monotone" / "render_reference"),
        ("PCA raw step-0", root / "training" / "C-monotone" / "render_00000"),
        ("C-six-metric v5", previous_root / "render_01750"),
        (f"C-composite v7 step-{endpoint}", root / "training" / "C-monotone" / f"render_{endpoint:05d}"),
        ("U0 10k", exact_root / "U0-unconstrained" / "render_10000"),
        ("S exact 10k", exact_root / "S-separated" / "render_10000"),
    ]
    report_dir = root / "report"
    _contact_sheet(columns, cameras, report_dir / "render_comparison_4view.png")
    exact_summary = json.loads(_path("outputs/scifihelmet_exact_basecolor_v1/report/final_summary.json").read_text(encoding="utf-8"))
    previous = json.loads(
        _path("outputs/scifihelmet_monotone_basecolor_curve_v5/training/C-monotone/training_report.json").read_text(
            encoding="utf-8"
        )
    )
    summary = {
        "schema_version": 1,
        "decision_scope": "train-only 31 cameras x 6 lights",
        "candidate": "C-monotone",
        "monotone_certificate_passed": True,
        "step0": training["step_metrics"]["0"],
        "endpoint_step": endpoint,
        "endpoint": training["step_metrics"][str(endpoint)],
        "step_metrics": training["step_metrics"],
        "previous_six_metric_curve": {
            "endpoint_step": previous["endpoint_step"],
            "endpoint_curve_alpha": previous["endpoint_curve_alpha"],
            "endpoint": previous["step_metrics"][str(previous["endpoint_step"])],
        },
        "controller": training["controller"],
        "comparison_exact_10k": exact_summary["10k_main_metrics"],
        "formal_holdout_accessed": False,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "final_summary.json").write_bytes(stable_json_bytes(summary))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("preflight", "train", "report", "all"))
    parser.add_argument("--config", default="configs/train/scifihelmet_monotone_basecolor_v1.yaml")
    args = parser.parse_args()
    config, contract, config_hash = _load_config(_path(args.config))
    if args.phase == "report":
        run_report(config)
        print(json.dumps({"phase": "report", "status": "complete"}, separators=(",", ":")))
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("C-monotone preflight and training require CUDA")
    phases = ("preflight", "train", "report") if args.phase == "all" else (args.phase,)
    for phase in phases:
        if phase == "preflight":
            run_preflight(config, contract, config_hash, torch.device("cuda"))
        elif phase == "train":
            run_train(config, contract, config_hash, torch.device("cuda"))
        else:
            run_report(config)
        print(json.dumps({"phase": phase, "status": "complete"}, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
