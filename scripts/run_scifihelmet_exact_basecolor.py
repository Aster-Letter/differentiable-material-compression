"""Run the fresh U0/S/M strict-BaseColor experiment lineage."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
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
from cg_frontier.compression.exact_basecolor import (  # noqa: E402
    AffineLatticeCodec,
    enumerate_lattice_capacity,
)
from cg_frontier.compression.exact_basecolor_experiment import (  # noqa: E402
    CANDIDATE_NAMES,
    LatticeOracleRecord,
    RenderCase,
    atomic_torch_save,
    candidate_forward_parity,
    candidate_state_hash,
    checkpoint_payload,
    conditional_rank_one_initialization,
    evaluate_render_pool,
    evaluate_texels,
    export_candidate,
    initialize_candidates,
    load_texel_targets,
    orbit_camera,
    render_pair_loss,
    search_lattice_oracles,
    sha256_bytes,
    stable_json_bytes,
    strict_candidate_certificate,
    tensor_sha256,
    train_one_step,
    validate_checkpoint,
    verify_runtime_export,
    write_runtime_material_diagnostics,
)
from cg_frontier.render.gbuffer import (  # noqa: E402
    load_core4_textures,
    render_geometry_gbuffer,
    sample_core4_material,
)
from cg_frontier.render.pbr import PointLight, linear_to_srgb_torch  # noqa: E402


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
    if config.get("schema_version") != 1 or config.get("experiment") != "scifihelmet_exact_basecolor_v1":
        raise ValueError("unsupported exact BaseColor config")
    scope = config.get("scope")
    if not isinstance(scope, Mapping) or scope.get("formal_holdout_accessed") is not False:
        raise ValueError("exact BaseColor run must explicitly keep formal holdout sealed")
    camera_contract = _read_yaml(_path(str(config["inputs"]["camera_contract"])))
    if len(camera_contract.get("train_cameras", [])) != 31 or len(camera_contract.get("train_lights", [])) != 6:
        raise ValueError("camera contract must contain exactly 31 train cameras and 6 train lights")
    return config, camera_contract, hashlib.sha256(raw).hexdigest()


def _camera(spec: Mapping[str, Any], render: Mapping[str, Any]):
    return orbit_camera(
        yaw_degrees=float(spec["yaw_degrees"]), elevation_degrees=float(spec["elevation_degrees"]),
        radius=float(spec.get("radius", render["camera_radius"])), target=spec.get("target", render["target"]),
        up=render["up"], vertical_fov_degrees=float(render["vertical_fov_degrees"]),
        near=float(render["near"]), far=float(render["far"]),
    )


def _light(spec: Mapping[str, Any]) -> PointLight:
    return PointLight(
        position=tuple(float(value) for value in spec["position"]),
        color=tuple(float(value) for value in spec["color"]),
        radiant_intensity=float(spec["radiant_intensity"]), ambient_intensity=float(spec["ambient_intensity"]),
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


def _input_state(config: Mapping[str, Any], device: torch.device):
    textures = load_core4_textures(_path(str(config["inputs"]["core4_manifest"])), device=device)
    targets = load_texel_targets(textures)
    return textures, targets


def _lattice_payload(path: Path) -> Mapping[str, Any]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if value.get("schema_version") != 1:
        raise ValueError("unsupported lattice oracle payload")
    return value


def _record_from_payload(value: Mapping[str, Any]) -> LatticeOracleRecord:
    from cg_frontier.compression.exact_basecolor import LatticeCapacity

    capacity = LatticeCapacity(**value["capacity"])
    return LatticeOracleRecord(
        capacity=capacity, material_score=float(value["material_score"]),
        auxiliary_weight=value["auxiliary_weight"], auxiliary_bias=value["auxiliary_bias"],
        audit_residual=value["audit_residual"],
    )


def _serialize_record(record: LatticeOracleRecord) -> dict[str, Any]:
    return {
        "capacity": record.capacity.as_dict(), "material_score": record.material_score,
        "auxiliary_weight": record.auxiliary_weight, "auxiliary_bias": record.auxiliary_bias,
        "audit_residual": record.audit_residual,
    }


def _lattice_search_evidence(targets, search, winner):
    unique_colors, frequencies = np.unique(targets.base_q8.numpy(), axis=0, return_counts=True)
    capacity_ranking = enumerate_lattice_capacity(
        unique_colors,
        frequencies,
        min_states=int(search["min_states"]),
        top_k=64 * 256,
    )
    winner_codec = AffineLatticeCodec(winner.capacity.kernel_rgb, winner.capacity.t0)
    k_lower, k_upper = winner_codec.valid_bounds(torch.from_numpy(unique_colors.astype(np.int64)))
    k_states = (k_upper - k_lower + 1).numpy()
    state_values, state_color_counts = np.unique(k_states, return_counts=True)
    state_texel_counts = [int(frequencies[k_states == value].sum()) for value in state_values]
    return unique_colors, {
        "search": {
            "enumerated_count": 64 * 256,
            "feasible_count": len(capacity_ranking),
            "capacity_top_k": int(search["capacity_top_k"]),
            "material_top_k": int(search["material_top_k"]),
            "capacity_ranking": [item.as_dict() for item in capacity_ranking],
        },
        "winner_k_distribution": {
            "histogram": [
                {"states": int(value), "unique_color_count": int(color_count), "texel_count": texel_count}
                for value, color_count, texel_count in zip(state_values, state_color_counts, state_texel_counts, strict=True)
            ],
            "per_color": [
                {"rgb": color.tolist(), "frequency": int(frequency), "states": int(states)}
                for color, frequency, states in zip(unique_colors, frequencies, k_states, strict=True)
            ],
        },
    }


def run_audit(config: Mapping[str, Any], contract: Mapping[str, Any], config_hash: str, device: torch.device) -> Mapping[str, Any]:
    output = _path(str(config["output_root"])) / "audit"
    output.mkdir(parents=True, exist_ok=True)
    textures_cpu, targets = _input_state(config, torch.device("cpu"))
    search = config["lattice_search"]
    initialization = conditional_rank_one_initialization(
        targets, sample_count=int(search["sample_count"]), seed=int(search["seed"]),
    )
    records = search_lattice_oracles(
        targets, initialization, min_states=int(search["min_states"]),
        capacity_top_k=int(search["capacity_top_k"]), material_top_k=int(search["material_top_k"]),
    )
    textures_gpu = load_core4_textures(_path(str(config["inputs"]["core4_manifest"])), device=device)
    cases, lights = _build_render_cases(config, contract, textures_gpu, device)
    render_scores = []
    camera_indices = [int(value) for value in search["render_camera_indices"]]
    for record in records:
        candidate = initialize_candidates(targets, initialization, record, device=device)["M-mixed"]
        render_metrics = evaluate_render_pool(
            candidate, cases, lights, height=targets.height, width=targets.width,
            minimum_roughness=float(contract["render"]["minimum_roughness"]),
            display_exposure=float(contract["render"]["display_exposure"]), case_indices=camera_indices,
        )
        render_scores.append({"record": record, "render": render_metrics})
        del candidate
        torch.cuda.empty_cache()
    best_hdr = min(float(item["render"]["mean_hdr_mae"]) for item in render_scores)
    eligible = [item for item in render_scores if float(item["render"]["mean_hdr_mae"]) <= best_hdr * 1.005]
    eligible.sort(key=lambda item: (item["record"].material_score, item["record"].capacity.capacity_sort_key))
    winner = eligible[0]["record"]
    winner_render = next(item["render"] for item in render_scores if item["record"] is winner)
    oracle_payload = {
        "schema_version": 1, "config_hash": config_hash,
        "target_hash": tensor_sha256(targets.base_q8),
        "winner": _serialize_record(winner),
        "records": [_serialize_record(item["record"]) for item in render_scores],
        "render_scores": [
            {"capacity": item["record"].capacity.as_dict(), "material_score": item["record"].material_score, "render": item["render"]}
            for item in render_scores
        ],
    }
    oracle_path = output / "lattice_oracles.pt"
    atomic_torch_save(oracle_payload, oracle_path)
    oracle_hash = hashlib.sha256(oracle_path.read_bytes()).hexdigest()
    candidates = initialize_candidates(targets, initialization, winner, device=device)
    certificates = {}
    for name in ("S-separated", "M-mixed"):
        certificates[name] = strict_candidate_certificate(
            candidates[name], targets, uv_count=int(config["evaluation"]["strict_uv_probes_gate"]), seed=int(search["seed"]),
        )
    if not all(item["passed"] for item in certificates.values()):
        raise AssertionError("strict codec audit certificate failed")
    q8_floor = torch.abs(targets.base_float - targets.base_q8.to(torch.float32) / 255.0)
    unique_colors, search_evidence = _lattice_search_evidence(targets, search, winner)
    centered = unique_colors.astype(np.float64) - unique_colors.mean(axis=0, keepdims=True)
    manifest = {
        "schema_version": 1, "experiment": "scifihelmet_exact_basecolor_v1",
        "config_hash": config_hash, "lattice_oracle_sha256": oracle_hash,
        "target_basecolor_sha256": tensor_sha256(targets.base_q8),
        "basecolor": {
            "unique_linear_q8": int(unique_colors.shape[0]), "affine_rank": int(np.linalg.matrix_rank(centered)),
            "linear_q8_floor_mae": float(q8_floor.mean()), "linear_q8_floor_max_abs": float(q8_floor.max()),
        },
        "search": search_evidence["search"],
        "winner": {**winner.summary(), "render_oracle": winner_render},
        "ranked_candidates": [
            {**item["record"].summary(), "render_oracle": item["render"]} for item in render_scores
        ],
        "certificates": certificates,
        "winner_k_distribution": search_evidence["winner_k_distribution"],
        "scope": dict(config["scope"]),
    }
    (output / "lattice_manifest.json").write_bytes(stable_json_bytes(manifest))
    return manifest


def _load_winner(config: Mapping[str, Any]):
    oracle_path = _path(str(config["output_root"])) / "audit" / "lattice_oracles.pt"
    payload = _lattice_payload(oracle_path)
    return _record_from_payload(payload["winner"]), payload, hashlib.sha256(oracle_path.read_bytes()).hexdigest()


def _optimizers(candidate):
    return torch.optim.Adam(candidate.code_parameters(), lr=0.02), torch.optim.Adam(candidate.decoder_parameters(), lr=0.001)


def _restore(payload, candidate, code_optimizer, decoder_optimizer, generator):
    candidate.load_state_dict(payload["candidate_state"])
    code_optimizer.load_state_dict(payload["code_optimizer"])
    decoder_optimizer.load_state_dict(payload["decoder_optimizer"])
    generator.set_state(payload["generator_state"])
    random.setstate(payload["python_random_state"])
    np.random.set_state(payload["numpy_random_state"])
    torch.set_rng_state(payload["torch_rng_state"])
    if torch.cuda.is_available() and payload["cuda_rng_state"]:
        torch.cuda.set_rng_state_all(payload["cuda_rng_state"])


def _run_steps(candidate, targets, cases, lights, config, contract, generator, code_optimizer, decoder_optimizer, start, stop, log_path=None):
    training = config["training"]
    last = None
    for step in range(start + 1, stop + 1):
        last = train_one_step(
            candidate, targets, cases, lights, generator=generator,
            texel_batch_size=int(training["texel_batch_size"]), screen_batch_size=int(training["screen_batch_size"]),
            minimum_roughness=float(contract["render"]["minimum_roughness"]),
            display_exposure=float(contract["render"]["display_exposure"]),
            code_optimizer=code_optimizer, decoder_optimizer=decoder_optimizer,
            step=step, stop=int(training["final_steps"]),
        )
        if log_path is not None and (step == 1 or step % int(training["log_interval"]) == 0):
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(last, sort_keys=True, separators=(",", ":")) + "\n")
    return last


def _gradient_stats(loss: torch.Tensor, parameters, *, retain_graph: bool) -> dict[str, Any]:
    gradients = torch.autograd.grad(
        loss,
        tuple(parameters),
        allow_unused=True,
        retain_graph=retain_graph,
    )
    present = [gradient for gradient in gradients if gradient is not None]
    finite = all(bool(torch.isfinite(gradient).all()) for gradient in present)
    maximum = max((float(gradient.abs().max()) for gradient in present), default=0.0)
    l1 = sum(float(gradient.abs().sum()) for gradient in present)
    return {"finite": finite, "max_abs": maximum, "l1": l1, "nonzero": l1 > 0.0}


def _zero_update_gradient_audit(candidate, targets, cases, lights, contract):
    device = candidate.colors_u8.device
    texel_ids = torch.arange(min(1024, candidate.texel_count), device=device)
    decoded = candidate.decoder(candidate.latent_for_ids(texel_ids, ste=True))
    auxiliary_loss = (
        decoded.normal_xyz[..., :2].sum()
        + decoded.roughness.sum()
        + decoded.metallic.sum()
    )
    trainable = list(candidate.code_parameters()) + list(candidate.decoder_parameters())
    auxiliary = _gradient_stats(auxiliary_loss, trainable, retain_graph=False)

    case = cases[0]
    selected = case.valid_flat_indices[: min(4096, case.valid_flat_indices.numel())]
    hdr, display, _ = render_pair_loss(
        candidate,
        case,
        lights[0],
        selected,
        height=targets.height,
        width=targets.width,
        minimum_roughness=float(contract["render"]["minimum_roughness"]),
        display_exposure=float(contract["render"]["display_exposure"]),
        ste=True,
    )
    render = _gradient_stats(hdr + 0.25 * display, trainable, retain_graph=False)
    if not auxiliary["finite"] or not auxiliary["nonzero"]:
        raise AssertionError(f"non-BaseColor gradient audit failed: {candidate.name}")
    if not render["finite"] or not render["nonzero"]:
        raise AssertionError(f"render gradient audit failed: {candidate.name}")

    strict = None
    if candidate.strict:
        decoded = candidate.decoder(candidate.latent_for_ids(texel_ids, ste=True))
        constrained_parameters = [candidate.residual_byte, candidate.decoder.auxiliary.weight, candidate.decoder.auxiliary.bias]
        strict = _gradient_stats(decoded.base_color_linear.sum(), constrained_parameters, retain_graph=False)
        if strict["max_abs"] > 1e-7:
            raise AssertionError(f"BaseColor Jacobian audit failed: {candidate.name}")
    return {"basecolor_constrained": strict, "auxiliary": auxiliary, "render": render}


def run_preflight(config, contract, config_hash, device):
    winner, oracle_payload, oracle_hash = _load_winner(config)
    textures, targets_cpu = _input_state(config, torch.device("cpu"))
    initialization = conditional_rank_one_initialization(targets_cpu, sample_count=int(config["lattice_search"]["sample_count"]), seed=int(config["lattice_search"]["seed"]))
    textures_gpu = load_core4_textures(_path(str(config["inputs"]["core4_manifest"])), device=device)
    cases, lights = _build_render_cases(config, contract, textures_gpu, device)
    targets = targets_cpu.to(device)
    report = {"schema_version": 1, "candidate_results": {}, "scope": dict(config["scope"])}
    for name in CANDIDATE_NAMES:
        fresh_a = initialize_candidates(targets_cpu, initialization, winner, device=device)[name]
        gradient_audit = _zero_update_gradient_audit(fresh_a, targets, cases, lights, contract)
        gen_a = torch.Generator(device=device).manual_seed(int(config["training"]["seed"]))
        opt_ca, opt_da = _optimizers(fresh_a)
        _run_steps(fresh_a, targets, cases, lights, config, contract, gen_a, opt_ca, opt_da, 0, 11)
        direct_hash = candidate_state_hash(fresh_a)
        fresh_b = initialize_candidates(targets_cpu, initialization, winner, device=device)[name]
        gen_b = torch.Generator(device=device).manual_seed(int(config["training"]["seed"]))
        opt_cb, opt_db = _optimizers(fresh_b)
        _run_steps(fresh_b, targets, cases, lights, config, contract, gen_b, opt_cb, opt_db, 0, 10)
        payload = checkpoint_payload(
            candidate=fresh_b, step=10, code_optimizer=opt_cb, decoder_optimizer=opt_db, generator=gen_b,
            config_hash=config_hash, lattice_manifest_hash=oracle_hash, target_hash=str(oracle_payload["target_hash"]),
        )
        resumed = initialize_candidates(targets_cpu, initialization, winner, device=device)[name]
        gen_r = torch.Generator(device=device)
        opt_cr, opt_dr = _optimizers(resumed)
        validate_checkpoint(payload, candidate_name=name, config_hash=config_hash, lattice_manifest_hash=oracle_hash, target_hash=str(oracle_payload["target_hash"]))
        _restore(payload, resumed, opt_cr, opt_dr, gen_r)
        _run_steps(resumed, targets, cases, lights, config, contract, gen_r, opt_cr, opt_dr, 10, 11)
        resumed_hash = candidate_state_hash(resumed)
        if direct_hash != resumed_hash:
            raise AssertionError(f"exact resume mismatch: {name}")
        strict_gradient = None
        if resumed.strict:
            ids = torch.arange(1024, device=device)
            base = resumed.decoder(resumed.latent_for_ids(ids, ste=True)).base_color_linear.sum()
            gradient = torch.autograd.grad(base, resumed.residual_byte, allow_unused=True)[0]
            strict_gradient = 0.0 if gradient is None else float(gradient.abs().max())
            if strict_gradient > 1e-7:
                raise AssertionError(f"BaseColor null gradient failed: {name}")
        report["candidate_results"][name] = {
            "direct_step11_hash": direct_hash,
            "resumed_step11_hash": resumed_hash,
            "exact_resume": True,
            "strict_basecolor_gradient_max_abs": strict_gradient,
            "zero_update_gradient_audit": gradient_audit,
        }
        del fresh_a, fresh_b, resumed
        torch.cuda.empty_cache()
    output = _path(str(config["output_root"])) / "preflight"
    output.mkdir(parents=True, exist_ok=True)
    (output / "preflight_report.json").write_bytes(stable_json_bytes(report))
    return report


def _save_decoded_base(candidate, targets, path):
    pieces = []
    with torch.no_grad():
        for start in range(0, candidate.texel_count, 262_144):
            ids = torch.arange(start, min(start + 262_144, candidate.texel_count), device=candidate.colors_u8.device)
            pieces.append(candidate.decoder(candidate.latent_for_ids(ids, ste=False)).base_color_linear.cpu())
    linear = torch.cat(pieces).reshape(targets.height, targets.width, 3).clamp(0, 1)
    display = linear_to_srgb_torch(linear)
    Image.fromarray(torch.floor(display * 255 + 0.5).to(torch.uint8).numpy(), mode="RGB").save(path)


def run_train(config, contract, config_hash, device):
    winner, oracle_payload, oracle_hash = _load_winner(config)
    _, targets_cpu = _input_state(config, torch.device("cpu"))
    initialization = conditional_rank_one_initialization(targets_cpu, sample_count=int(config["lattice_search"]["sample_count"]), seed=int(config["lattice_search"]["seed"]))
    textures_gpu = load_core4_textures(_path(str(config["inputs"]["core4_manifest"])), device=device)
    cases, lights = _build_render_cases(config, contract, textures_gpu, device)
    targets = targets_cpu.to(device)
    candidates = initialize_candidates(targets_cpu, initialization, winner, device=device)
    output_root = _path(str(config["output_root"])) / "training"
    report = {"schema_version": 1, "candidates": {}, "scope": dict(config["scope"])}
    runtime = {}
    for name in CANDIDATE_NAMES:
        candidate = candidates[name]
        output = output_root / name
        output.mkdir(parents=True, exist_ok=True)
        generator = torch.Generator(device=device).manual_seed(int(config["training"]["seed"]))
        code_optimizer, decoder_optimizer = _optimizers(candidate)
        candidate_report = {"step_metrics": {}, "checkpoints": {}, "initial_state_hash": candidate_state_hash(candidate)}
        candidate_report["step_metrics"]["0"] = {
            "texel": evaluate_texels(candidate, targets_cpu),
            "render": evaluate_render_pool(candidate, cases, lights, height=targets.height, width=targets.width, minimum_roughness=float(contract["render"]["minimum_roughness"]), display_exposure=float(contract["render"]["display_exposure"])),
        }
        report["candidates"][name] = candidate_report
        runtime[name] = {
            "candidate": candidate,
            "output": output,
            "generator": generator,
            "code_optimizer": code_optimizer,
            "decoder_optimizer": decoder_optimizer,
            "current": 0,
        }

    stops = [int(value) for value in config["training"]["checkpoints"]]
    gate = int(config["training"]["gate_steps"])
    final = int(config["training"]["final_steps"])
    if not stops or stops[0] != gate or stops[-1] != final:
        raise ValueError("checkpoint schedule must start at the 1k gate and end at final_steps")

    # Stage every paired candidate through the same stop before any candidate
    # may proceed.  In particular, all three 1k correctness gates are closed
    # before the first 5k update is taken.
    for stop in stops:
        for name in CANDIDATE_NAMES:
            state = runtime[name]
            candidate = state["candidate"]
            output = state["output"]
            candidate_report = report["candidates"][name]
            _run_steps(
                candidate,
                targets,
                cases,
                lights,
                config,
                contract,
                state["generator"],
                state["code_optimizer"],
                state["decoder_optimizer"],
                state["current"],
                stop,
                output / "train.jsonl",
            )
            payload = checkpoint_payload(
                candidate=candidate,
                step=stop,
                code_optimizer=state["code_optimizer"],
                decoder_optimizer=state["decoder_optimizer"],
                generator=state["generator"],
                config_hash=config_hash, lattice_manifest_hash=oracle_hash, target_hash=str(oracle_payload["target_hash"]),
            )
            checkpoint = output / f"checkpoint_{stop:05d}.pt"
            atomic_torch_save(payload, checkpoint, immutable=True)
            candidate_report["checkpoints"][str(stop)] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            case_indices = None if stop == final else [int(value) for value in config["evaluation"]["intermediate_camera_indices"]]
            metrics = {
                "texel": evaluate_texels(candidate, targets_cpu),
                "render": evaluate_render_pool(
                    candidate, cases, lights, height=targets.height, width=targets.width,
                    minimum_roughness=float(contract["render"]["minimum_roughness"]), display_exposure=float(contract["render"]["display_exposure"]),
                    case_indices=case_indices, save_images=output / f"render_{stop:05d}",
                ),
            }
            if candidate.strict:
                probes = int(config["evaluation"]["strict_uv_probes_final"] if stop == final else config["evaluation"]["strict_uv_probes_gate"])
                metrics["strict_certificate"] = strict_candidate_certificate(candidate, targets_cpu, uv_count=probes, seed=int(config["training"]["seed"]))
                if not metrics["strict_certificate"]["passed"]:
                    raise AssertionError(f"strict correctness gate failed at {stop}: {name}")
            if stop == gate:
                metrics["hard_fake_parity"] = candidate_forward_parity(
                    candidate,
                    seed=int(config["training"]["seed"]),
                )
                if not metrics["hard_fake_parity"]["passed"]:
                    raise AssertionError(f"hard/fake parity gate failed at {stop}: {name}")
            candidate_report["step_metrics"][str(stop)] = metrics
            state["current"] = stop

        if stop == gate:
            gate_passed = all(
                report["candidates"][name]["step_metrics"][str(gate)]["hard_fake_parity"]["passed"]
                and (
                    not runtime[name]["candidate"].strict
                    or report["candidates"][name]["step_metrics"][str(gate)]["strict_certificate"]["passed"]
                )
                for name in CANDIDATE_NAMES
            )
            if not gate_passed:
                raise AssertionError("not all paired candidates passed the 1k correctness gate")
            report["all_candidates_passed_1k_correctness_gate"] = True

    for name in CANDIDATE_NAMES:
        state = runtime[name]
        candidate = state["candidate"]
        output = state["output"]
        candidate_report = report["candidates"][name]
        export_candidate(candidate, output_dir=output / "export", height=targets.height, width=targets.width)
        candidate_report["runtime_diagnostics"] = write_runtime_material_diagnostics(
            output / "export",
            targets_cpu,
            output_dir=output / "diagnostics",
        )
        _save_decoded_base(candidate, targets_cpu, output / "basecolor_decoded.png")
        (output / "candidate_report.json").write_bytes(stable_json_bytes(candidate_report))
    (output_root / "training_report.json").write_bytes(stable_json_bytes(report))
    return report


def _bootstrap_mixed(training, samples, seed):
    separated = training["candidates"]["S-separated"]["step_metrics"]["10000"]["render"]["pairs"]
    mixed = training["candidates"]["M-mixed"]["step_metrics"]["10000"]["render"]["pairs"]
    s = np.asarray([item["hdr_mae"] for item in separated], dtype=np.float64)
    m = np.asarray([item["hdr_mae"] for item in mixed], dtype=np.float64)
    if s.shape != m.shape:
        raise ValueError("paired render arrays differ")
    relative = (s - m) / np.maximum(s, 1e-12)
    generator = np.random.default_rng(seed)
    means = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 1000):
        count = min(1000, samples - start)
        indices = generator.integers(0, relative.size, size=(count, relative.size))
        means[start:start + count] = relative[indices].mean(axis=1)
    return {"relative_mean_improvement": float(relative.mean()), "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]}


def _write_training_trajectory(root: Path, output_path: Path) -> None:
    names = ("total", "hdr", "base_color", "normal", "roughness", "metallic")
    colors = {
        "total": "#ffffff",
        "hdr": "#58a6ff",
        "base_color": "#ff7b72",
        "normal": "#d2a8ff",
        "roughness": "#ffa657",
        "metallic": "#7ee787",
    }
    canvas = Image.new("RGB", (1536, 512), "#0d1117")
    draw = ImageDraw.Draw(canvas)
    for candidate_index, candidate_name in enumerate(CANDIDATE_NAMES):
        left = candidate_index * 512
        draw.rectangle((left + 45, 25, left + 495, 470), outline="#8b949e")
        records = [
            json.loads(line)
            for line in (root / "training" / candidate_name / "train.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        max_step = max(int(record["step"]) for record in records)
        for loss_name in names:
            points = []
            for record in records:
                x = left + 45 + 450 * float(record["step"]) / max_step
                log_value = math.log10(max(float(record[loss_name]), 1e-6))
                y = 470 - 445 * (log_value + 6.0) / 7.0
                points.append((x, min(470, max(25, y))))
            if len(points) >= 2:
                draw.line(points, fill=colors[loss_name], width=2)
        draw.text((left + 50, 32), candidate_name, fill="white")
        for index, loss_name in enumerate(names):
            draw.text((left + 50 + (index % 3) * 145, 480 + (index // 3) * 14), loss_name, fill=colors[loss_name])
    canvas.save(output_path)


def _write_report_diagnostics(config, root: Path, report_dir: Path) -> Mapping[str, Any]:
    _, targets = _input_state(config, torch.device("cpu"))
    height, width = targets.height, targets.width
    source = targets.base_float.reshape(height, width, 3)
    q8 = targets.base_q8.reshape(height, width, 3).to(torch.float32) / 255.0
    floor_error = torch.mean(torch.abs(source - q8), dim=-1)
    source_path = report_dir / "source_basecolor_float.png"
    q8_path = report_dir / "basecolor_q8_target.png"
    floor_path = report_dir / "basecolor_q8_floor_error.png"
    Image.fromarray(torch.floor(linear_to_srgb_torch(source).clamp(0, 1) * 255 + 0.5).to(torch.uint8).numpy()).save(source_path)
    Image.fromarray(torch.floor(linear_to_srgb_torch(q8).clamp(0, 1) * 255 + 0.5).to(torch.uint8).numpy()).save(q8_path)
    Image.fromarray(torch.floor((floor_error * 255.0).clamp(0, 1) * 255 + 0.5).to(torch.uint8).numpy()).save(floor_path)
    candidates = {
        name: write_runtime_material_diagnostics(
            root / "training" / name / "export",
            targets,
            output_dir=root / "training" / name / "diagnostics",
        )
        for name in CANDIDATE_NAMES
    }
    trajectory_path = report_dir / "training_trajectories.png"
    _write_training_trajectory(root, trajectory_path)
    return {
        "source_basecolor_float": source_path.name,
        "basecolor_q8_target": q8_path.name,
        "basecolor_q8_floor_error": floor_path.name,
        "q8_floor_error_visualization_scale": 255.0,
        "training_trajectories": trajectory_path.name,
        "candidates": candidates,
    }


def _verify_training_checkpoints(root: Path, audit, training) -> Mapping[str, Any]:
    result = {"files": {}, "paired_generator_state": {}}
    generator_hashes: dict[int, list[str]] = {}
    for name in CANDIDATE_NAMES:
        result["files"][name] = {}
        for step_text, expected_hash in training["candidates"][name]["checkpoints"].items():
            step = int(step_text)
            path = root / "training" / name / f"checkpoint_{step:05d}.pt"
            actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual_hash != expected_hash:
                raise ValueError(f"checkpoint file hash mismatch: {name}@{step}")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            validate_checkpoint(
                payload,
                candidate_name=name,
                config_hash=str(audit["config_hash"]),
                lattice_manifest_hash=str(audit["lattice_oracle_sha256"]),
                target_hash=str(audit["target_basecolor_sha256"]),
            )
            if int(payload["step"]) != step:
                raise ValueError(f"checkpoint step mismatch: {name}@{step}")
            generator_hash = tensor_sha256(payload["generator_state"])
            generator_hashes.setdefault(step, []).append(generator_hash)
            result["files"][name][step_text] = {
                "sha256": actual_hash,
                "metadata_valid": True,
                "generator_state_sha256": generator_hash,
            }
            del payload
    for step, hashes in generator_hashes.items():
        paired = len(set(hashes)) == 1
        if not paired:
            raise ValueError(f"paired RNG state mismatch at step {step}")
        result["paired_generator_state"][str(step)] = paired
    result["passed"] = True
    return result


def run_report(config):
    root = _path(str(config["output_root"]))
    training = json.loads((root / "training" / "training_report.json").read_text(encoding="utf-8"))
    audit = json.loads((root / "audit" / "lattice_manifest.json").read_text(encoding="utf-8"))
    checkpoint_verification = _verify_training_checkpoints(root, audit, training)
    runtime_exports = {
        name: verify_runtime_export(root / "training" / name / "export")
        for name in CANDIDATE_NAMES
    }
    target_hash = str(audit["target_basecolor_sha256"])
    strict_export_hashes_match = all(
        runtime_exports[name]["decoded_basecolor_u8_sha256"] == target_hash
        for name in ("S-separated", "M-mixed")
    )
    evaluation = config["evaluation"]
    bootstrap = _bootstrap_mixed(training, int(evaluation["bootstrap_samples"]), int(evaluation["bootstrap_seed"]))
    s = training["candidates"]["S-separated"]["step_metrics"]["10000"]
    m = training["candidates"]["M-mixed"]["step_metrics"]["10000"]
    u0_start = training["candidates"]["U0-unconstrained"]["step_metrics"]["0"]["texel"]
    u0_final = training["candidates"]["U0-unconstrained"]["step_metrics"]["10000"]["texel"]
    report_dir = root / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = _write_report_diagnostics(config, root, report_dir)
    s_runtime = diagnostics["candidates"]["S-separated"]["metrics"]
    m_runtime = diagnostics["candidates"]["M-mixed"]["metrics"]
    guards = {
        "mean_hdr_improvement_1pct": bootstrap["relative_mean_improvement"] >= 0.01,
        "bootstrap_ci_excludes_zero": bootstrap["ci95"][0] > 0.0,
        "worst_hdr_guard": m["render"]["worst_hdr_mae"] <= 1.02 * s["render"]["worst_hdr_mae"],
        "display_ssim_guard": m["render"]["mean_display_ssim"] >= s["render"]["mean_display_ssim"] - 0.001,
        "normal_guard": m_runtime["normal_angle_degrees"] <= 1.05 * s_runtime["normal_angle_degrees"],
        "roughness_guard": m_runtime["roughness_mae"] <= 1.05 * s_runtime["roughness_mae"],
        "metallic_guard": m_runtime["metallic_mae"] <= 1.05 * s_runtime["metallic_mae"],
        "strict_certificates": bool(
            s["strict_certificate"]["passed"]
            and m["strict_certificate"]["passed"]
            and strict_export_hashes_match
        ),
    }
    mixed_advantage = all(guards.values())
    gray_reproduced = bool(u0_final["generic_chroma_retention"] < 0.5 and u0_final["chromatic_texel_fraction_losing_25pct"] > 0.5)
    summary = {
        "schema_version": 1, "mixed_bootstrap": bootstrap, "mixed_acceptance_gates": guards,
        "mixed_advantage_demonstrated": mixed_advantage,
        "decision": "mixed lattice advantage demonstrated" if mixed_advantage else "no demonstrated mixed lattice advantage; prefer separated",
        "u0_gray_reproduced": gray_reproduced,
        "u0_step0_chroma_retention": u0_start["generic_chroma_retention"],
        "u0_10k_chroma_retention": u0_final["generic_chroma_retention"],
        "runtime_export_strict_hashes_match": strict_export_hashes_match,
        "checkpoint_and_paired_rng_verification": bool(checkpoint_verification["passed"]),
        "scope": dict(config["scope"]),
    }
    summary["10k_main_metrics"] = {}
    for name in CANDIDATE_NAMES:
        texel = training["candidates"][name]["step_metrics"]["10000"]["texel"]
        render = training["candidates"][name]["step_metrics"]["10000"]["render"]
        runtime_metrics = diagnostics["candidates"][name]["metrics"]
        summary["10k_main_metrics"][name] = {
            **runtime_metrics,
            "generic_chroma_retention": texel["generic_chroma_retention"],
            "chromatic_texel_fraction_losing_25pct": texel["chromatic_texel_fraction_losing_25pct"],
            "mean_hdr_mae": render["mean_hdr_mae"],
            "worst_hdr_mae": render["worst_hdr_mae"],
            "mean_display_ssim": render["mean_display_ssim"],
        }
    (report_dir / "final_summary.json").write_bytes(stable_json_bytes(summary))
    (report_dir / "runtime_export_verification.json").write_bytes(
        stable_json_bytes({"target_basecolor_u8_sha256": target_hash, "candidates": runtime_exports})
    )
    (report_dir / "diagnostics_manifest.json").write_bytes(stable_json_bytes(diagnostics))
    (report_dir / "checkpoint_verification.json").write_bytes(stable_json_bytes(checkpoint_verification))
    images = [Image.open(root / "training" / name / "basecolor_decoded.png").convert("RGB") for name in CANDIDATE_NAMES]
    thumb = (512, 512)
    canvas = Image.new("RGB", (512 * 3, 512), "black")
    for index, image in enumerate(images):
        canvas.paste(image.resize(thumb), (index * 512, 0))
    canvas.save(report_dir / "basecolor_u0_s_m.png")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("audit", "preflight", "train", "report", "all"))
    parser.add_argument("--config", default="configs/train/scifihelmet_exact_basecolor_v1.yaml")
    args = parser.parse_args()
    config, contract, config_hash = _load_config(_path(args.config))
    if args.phase == "report":
        result = run_report(config)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("audit render oracle and training require CUDA")
    device = torch.device("cuda")
    result = None
    phases = ("audit", "preflight", "train", "report") if args.phase == "all" else (args.phase,)
    for phase in phases:
        if phase == "audit":
            result = run_audit(config, contract, config_hash, device)
        elif phase == "preflight":
            result = run_preflight(config, contract, config_hash, device)
        elif phase == "train":
            result = run_train(config, contract, config_hash, device)
        else:
            result = run_report(config)
        print(json.dumps({"phase": phase, "status": "complete"}, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
