"""Run the shared C0 and dual-ratio C1/C2 SciFiHelmet color-guard 1k matrix."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Mapping, Sequence

import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from audit_scifihelmet_c4_affine_color_guard import (
    _sha256_bytes,
    certified_parent_hash,
)
from cg_frontier.assets.gltf_mesh import load_gltf_mesh
from cg_frontier.compression.affine_color import (
    ColorGuardObjective,
    ColorMetricPairs,
    ColorQuantilePartition,
    build_color_quantile_partition,
    color_quality_metrics,
    oklab_mean_delta_e,
)
from cg_frontier.compression.affine_gradient_audit import audit_gradient_objectives
from cg_frontier.compression.affine_material import certify_affine
from cg_frontier.compression.affine_pca import (
    EnhancedPCASpec,
    export_p0_enhanced_bundle,
    rasterize_uv_charts,
)
from cg_frontier.compression.affine_regularizers import fake_quantize_unorm8
from cg_frontier.compression.affine_training import (
    AffineCandidateState,
    AffineTrainingBatch,
    TrainingObservationPlan,
    checkpoint_candidate,
    create_color_candidates,
    resume_candidate,
    run_candidate_training,
    time_candidate_steps,
)
from cg_frontier.compression.material import Core4Targets, load_core4_targets
from cg_frontier.render.gbuffer import (
    load_core4_textures,
    render_geometry_gbuffer,
    sample_core4_material,
)
from cg_frontier.render.pbr import shade_ggx
from run_scifihelmet_c4_affine_40k import (
    _append_json_line,
    _export_endpoint,
)
from run_scifihelmet_c4_affine_chroma8_l0_40k import (
    _endpoint_metrics_for_pool,
    _mean_render,
    _orbit_camera_from_spec,
    freeze_render_coverage,
)
from run_scifihelmet_c4_affine_preflight import (
    PreflightObjective,
    _json_bytes,
    _light,
    _move_p0,
    _repo_path,
    _sha256_file,
    _targets_to_seven,
    _write_new,
)
from run_scifihelmet_c4_affine_40k import _load_mapping


@dataclass(frozen=True)
class FrozenColorGuardTrainingSpec:
    ratios: tuple[float, ...]
    candidate_keys: tuple[str, ...]
    preflight_steps: int
    total_steps: int
    checkpoint_steps: tuple[int, ...]
    metric_interval: int
    progress_interval: int
    trajectory_gradient_batches: int
    material_batch_size: int
    color_batch_size: int
    color_seed_offset: int
    charbonnier_epsilon: float


def _ratio_key(value: float) -> str:
    return f"r{round(value * 100):03d}"


def freeze_color_guard_training_spec(
    config: Mapping[str, object],
) -> FrozenColorGuardTrainingSpec:
    """Freeze the user-authorized shared-C0, r=0.10/0.25 comparison."""

    training = config.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("training section is missing")
    ratios = tuple(float(value) for value in training["ratios"])
    if ratios != (0.10, 0.25):
        raise ValueError("training ratios must be the frozen (0.10, 0.25) pair")
    checkpoint_steps = tuple(int(value) for value in training["checkpoint_steps"])
    candidate_keys = ("C0",) + tuple(
        f"{candidate}-{_ratio_key(ratio)}"
        for ratio in ratios
        for candidate in ("C1", "C2")
    )
    spec = FrozenColorGuardTrainingSpec(
        ratios=ratios,
        candidate_keys=candidate_keys,
        preflight_steps=int(training["preflight_steps"]),
        total_steps=int(training["total_steps"]),
        checkpoint_steps=checkpoint_steps,
        metric_interval=int(training["metric_interval"]),
        progress_interval=int(training["progress_interval"]),
        trajectory_gradient_batches=int(training["trajectory_gradient_batches"]),
        material_batch_size=int(training["material_batch_size"]),
        color_batch_size=int(training["color_batch_size"]),
        color_seed_offset=int(training["color_seed_offset"]),
        charbonnier_epsilon=float(training["charbonnier_epsilon"]),
    )
    if spec.preflight_steps != 10 or spec.total_steps != 1000:
        raise ValueError("training must freeze 10-step preflight and 1k endpoint")
    if spec.checkpoint_steps != (250, 500, 750, 1000):
        raise ValueError("training checkpoint steps must be 250/500/750/1000")
    if spec.metric_interval != 250 or spec.progress_interval != 100:
        raise ValueError("training metric/progress intervals are invalid")
    if spec.trajectory_gradient_batches != 5:
        raise ValueError("trajectory gradient audit must use five frozen batches")
    if spec.material_batch_size != 4096 or spec.color_batch_size != 4096:
        raise ValueError("training batch sizes must remain 4096")
    if spec.color_seed_offset != 23 or spec.charbonnier_epsilon != 1.0e-3:
        raise ValueError("training color RNG/Charbonnier contract is invalid")
    return spec


def freeze_candidate_matrix(
    audit_report: Mapping[str, object], ratios: tuple[float, ...]
) -> dict[str, dict[str, object]]:
    """Bind the authorized matrix to the immutable zero-update calibration."""

    if (
        audit_report.get("status") != "complete_waiting_for_ratio_confirmation"
        or audit_report.get("training_started") is not False
        or audit_report.get("selected_ratio") is not None
    ):
        raise ValueError("candidate matrix requires the completed zero-update audit")
    calibration = audit_report.get("calibration")
    if not isinstance(calibration, Mapping) or calibration.get("selected_ratio") is not None:
        raise ValueError("zero-update calibration must not preselect a ratio")
    candidates = calibration.get("candidates")
    if not isinstance(candidates, Mapping):
        raise ValueError("zero-update calibration candidates are missing")
    matrix: dict[str, dict[str, object]] = {
        "C0": {
            "candidate_id": "C0",
            "ratio": None,
            "opponent_lambda": 0.0,
            "pair_lambda": 0.0,
        }
    }
    for ratio in ratios:
        ratio_key = f"{ratio:.6f}"
        calibrated = candidates.get(ratio_key)
        if not isinstance(calibrated, Mapping):
            raise ValueError(f"audit calibration is missing ratio {ratio_key}")
        for candidate_id in ("C1", "C2"):
            weights = calibrated.get(candidate_id)
            if not isinstance(weights, Mapping):
                raise ValueError(f"audit calibration is missing {candidate_id}/{ratio_key}")
            opponent_lambda = float(weights["opponent_lambda"])
            pair_lambda = float(weights["pair_lambda"])
            if opponent_lambda <= 0.0 or pair_lambda < 0.0:
                raise ValueError("audit calibration contains invalid color weights")
            if candidate_id == "C1" and pair_lambda != 0.0:
                raise ValueError("C1 must not carry a pair loss")
            matrix[f"{candidate_id}-{_ratio_key(ratio)}"] = {
                "candidate_id": candidate_id,
                "ratio": ratio,
                "opponent_lambda": opponent_lambda,
                "pair_lambda": pair_lambda,
            }
    return matrix


def _number(mapping: Mapping[str, object], key: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"missing numeric metric: {key}")
    return float(value)


def evaluate_balanced_color_gate(
    parent: Mapping[str, object],
    c0: Mapping[str, object],
    candidate: Mapping[str, object],
    policy: Mapping[str, object],
) -> dict[str, object]:
    """Evaluate the frozen Pareto gate without consulting yellow diagnostics."""

    for endpoint in (parent, c0, candidate):
        if endpoint.get("yellow_diagnostic", {}).get("selection_metric") is not False:
            raise ValueError("yellow diagnostic must remain selection_metric=false")
    parent_color = parent["color"]
    c0_color = c0["color"]
    candidate_color = candidate["color"]
    parent_material = parent["material"]
    c0_material = c0["material"]
    candidate_material = candidate["material"]
    parent_render = parent["render"]
    c0_render = c0["render"]
    candidate_render = candidate["render"]
    if not all(
        isinstance(value, Mapping)
        for value in (
            parent_color,
            c0_color,
            candidate_color,
            parent_material,
            c0_material,
            candidate_material,
            parent_render,
            c0_render,
            candidate_render,
        )
    ):
        raise ValueError("balanced gate endpoint metrics are incomplete")

    color_names = tuple(str(value) for value in policy["primary_color_metrics"])
    improvement_limit = 1.0 - float(policy["color_improvement_vs_c0"])
    regression_limit = 1.0 + float(policy["color_regression_vs_c0"])
    color_ratios = {
        name: _number(candidate_color, name) / _number(c0_color, name)
        for name in color_names
    }
    improved = {name: ratio <= improvement_limit for name, ratio in color_ratios.items()}
    protected = {name: ratio <= regression_limit for name, ratio in color_ratios.items()}

    multiplier = float(policy["global_error_multiplier"])
    error_sources = {
        "uniform_base_color_l1": (parent_color, c0_color, candidate_color),
        "seven_channel_mae": (parent_material, c0_material, candidate_material),
        "masked_linear_hdr_mae": (parent_render, c0_render, candidate_render),
        "normal_cosine": (parent_material, c0_material, candidate_material),
        "roughness_l1": (parent_material, c0_material, candidate_material),
        "metallic_l1": (parent_material, c0_material, candidate_material),
    }
    gates: dict[str, bool] = {
        "certificate": candidate.get("certificate", {}).get("valid") is True,
        "chroma_retention": _number(candidate_color, "chroma_contrast_retention")
        >= float(policy["parent_chroma_retention_floor"])
        * _number(parent_color, "chroma_contrast_retention"),
        "color_improvement": any(improved.values()),
        "color_no_regression": all(protected.values()),
    }
    for name, (parent_values, c0_values, candidate_values) in error_sources.items():
        candidate_value = _number(candidate_values, name)
        gates[name] = (
            candidate_value <= multiplier * _number(parent_values, name)
            and candidate_value <= multiplier * _number(c0_values, name)
        )
    ssim_drop = float(policy["display_ssim_absolute_drop"])
    candidate_ssim = _number(candidate_render, "display_ssim")
    gates["display_ssim"] = (
        candidate_ssim >= _number(parent_render, "display_ssim") - ssim_drop
        and candidate_ssim >= _number(c0_render, "display_ssim") - ssim_drop
    )
    return {
        "schema_version": 1,
        "passed": all(gates.values()),
        "gates": gates,
        "color_ratios_vs_c0": color_ratios,
        "color_improvement_count": sum(improved.values()),
        "selection_values": {
            "candidate_color": dict(candidate_color),
            "candidate_material": dict(candidate_material),
            "candidate_render": dict(candidate_render),
        },
    }


def _tensor_sha256(value: torch.Tensor) -> str:
    host = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(host.dtype).encode("ascii"))
    digest.update(str(tuple(host.shape)).encode("ascii"))
    digest.update(host.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _load_json(path: Path) -> tuple[dict[str, object], bytes]:
    payload = path.read_bytes()
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON mapping: {path}")
    return value, payload


@torch.no_grad()
def _atlas_endpoint(
    state: AffineCandidateState,
    targets: Core4Targets,
    valid_indices: torch.Tensor,
    source_base_color: torch.Tensor,
    partition: ColorQuantilePartition,
    metric_pairs: ColorMetricPairs,
) -> dict[str, object]:
    deployed = fake_quantize_unorm8(state.latent.detach())
    decoded = state.decoder(deployed.reshape(-1, 4)[valid_indices])
    selected = targets.select(valid_indices)
    predicted_seven = torch.cat(
        (
            decoded.base_color_linear,
            decoded.normal_xy,
            decoded.roughness,
            decoded.metallic,
        ),
        dim=-1,
    )
    source_seven = torch.cat(
        (
            selected.base_color_linear,
            selected.normal_xyz[:, :2],
            selected.roughness,
            selected.metallic,
        ),
        dim=-1,
    )
    color = color_quality_metrics(
        decoded.base_color_linear, source_base_color, partition, metric_pairs
    )
    color["oklab_mean_delta_e_report_only"] = float(
        oklab_mean_delta_e(decoded.base_color_linear, source_base_color).cpu()
    )
    material = {
        "base_color_l1": float(
            F.l1_loss(decoded.base_color_linear, selected.base_color_linear).cpu()
        ),
        "seven_channel_mae": float(
            torch.mean(torch.abs(predicted_seven - source_seven)).cpu()
        ),
        "normal_cosine": float(
            torch.mean(
                1.0 - torch.sum(decoded.normal_xyz * selected.normal_xyz, dim=-1)
            ).cpu()
        ),
        "roughness_l1": float(F.l1_loss(decoded.roughness, selected.roughness).cpu()),
        "metallic_l1": float(F.l1_loss(decoded.metallic, selected.metallic).cpu()),
    }
    weight, bias = state.decoder.fold_affine()
    certificate = certify_affine(weight, bias, margin=state.decoder.margin)
    yellow = (
        (source_base_color[:, 0] > source_base_color[:, 1])
        & (source_base_color[:, 1] > source_base_color[:, 2])
        & (source_base_color[:, 0] - source_base_color[:, 2] > 0.05)
        & (source_base_color[:, 1] - source_base_color[:, 2] > 0.02)
    )
    return {
        "step": state.optimizer_updates,
        "color": color,
        "material": material,
        "certificate": certificate,
        "yellow_diagnostic": {
            "selection_metric": False,
            "valid_texels": int(torch.count_nonzero(yellow).cpu()),
            "source_mean_r_minus_b": float(
                torch.mean(
                    source_base_color[yellow, 0] - source_base_color[yellow, 2]
                ).cpu()
            ),
            "predicted_mean_r_minus_b": float(
                torch.mean(
                    decoded.base_color_linear[yellow, 0]
                    - decoded.base_color_linear[yellow, 2]
                ).cpu()
            ),
        },
    }


def _with_render_metrics(
    atlas: Mapping[str, object], render_metrics: Mapping[str, object]
) -> dict[str, object]:
    return {
        **dict(atlas),
        "render": {
            "masked_linear_hdr_mae": _mean_render(
                render_metrics, "masked_linear_hdr_mae"
            ),
            "display_ssim": _mean_render(render_metrics, "display_ssim"),
            "pair_count": len(render_metrics["helmet_render"]),
        },
        "render_grid": render_metrics["helmet_render"],
    }


def _gradient_summary(
    state: AffineCandidateState,
    batches: Sequence[AffineTrainingBatch],
    audit_base_objective: PreflightObjective,
    color_objective: ColorGuardObjective,
    loss_config: Mapping[str, object],
) -> dict[str, object]:
    def terms(batch: AffineTrainingBatch) -> dict[str, torch.Tensor]:
        base, base_terms = audit_base_objective(state, batch)
        color = color_objective.color_terms(state, batch)
        candidate_color = (
            color["opponent"] * color_objective.opponent_lambda
            + color["pair"] * color_objective.pair_lambda
        )
        return {
            "base": base,
            "render": base_terms["helmet_charbonnier"]
            * float(loss_config["helmet_charbonnier"])
            + base_terms["helmet_log1p"] * float(loss_config["helmet_log1p"]),
            "rgb": base_terms["base_color_l1"]
            * float(loss_config["base_color_l1"]),
            "opponent": color["opponent"],
            "pair": color["pair"],
            "candidate_color": candidate_color,
            "normal": base_terms["normal_cosine"]
            * float(loss_config["normal_cosine"]),
            "roughness": base_terms["roughness_l1"]
            * float(loss_config["roughness_l1"]),
            "metallic": base_terms["metallic_l1"]
            * float(loss_config["metallic_l1"]),
        }

    audit = audit_gradient_objectives(
        batches=batches,
        objective_terms=terms,
        parameter_groups={
            "latent": (state.latent,),
            "affine": tuple(state.decoder.parameters()),
        },
        epsilon=1.0e-12,
    )
    groups: dict[str, object] = {}
    for group in ("latent", "affine"):
        cosines = {
            color: [
                float(batch["parameter_groups"][group]["cosine"]["base"][color])
                for batch in audit["batches"]
            ]
            for color in ("opponent", "pair", "candidate_color")
        }
        norms = {
            objective: statistics.median(
                float(batch["parameter_groups"][group]["norms"][objective])
                for batch in audit["batches"]
            )
            for objective in audit["objective_names"]
        }
        groups[group] = {
            "base_color_negative_counts": {
                color: sum(value < 0.0 for value in values)
                for color, values in cosines.items()
            },
            "base_color_cosine_medians": {
                color: statistics.median(values) for color, values in cosines.items()
            },
            "gradient_norm_medians": norms,
        }
    return {
        "schema_version": 1,
        "step": state.optimizer_updates,
        "batch_count": len(batches),
        "groups": groups,
    }


def _candidate_state(
    candidate_id: str,
    parent: object,
    *,
    seed: int,
    color_seed_offset: int,
    partition_hash: str,
    config_hash: str,
    input_hash: str,
    latent_learning_rate: float,
    affine_learning_rate: float,
) -> AffineCandidateState:
    return create_color_candidates(
        parent,
        core_seed=seed + 11,
        color_seed=seed + color_seed_offset,
        color_partition_hash=partition_hash,
        config_hash=config_hash,
        input_hash=input_hash,
        latent_learning_rate=latent_learning_rate,
        affine_learning_rate=affine_learning_rate,
    )[candidate_id]


def _candidate_objective(
    base_objective: PreflightObjective,
    *,
    valid_indices: torch.Tensor,
    source_base_color: torch.Tensor,
    weights: Mapping[str, object],
    epsilon: float,
) -> ColorGuardObjective:
    return ColorGuardObjective(
        base_objective,
        valid_flat_indices=valid_indices,
        source_base_color=source_base_color,
        opponent_lambda=float(weights["opponent_lambda"]),
        pair_lambda=float(weights["pair_lambda"]),
        epsilon=epsilon,
    )


def run(config_path: Path) -> dict[str, object]:
    config, config_bytes = _load_mapping(config_path, "color guard 1k config")
    if config.get("experiment") != "scifihelmet_c4_affine_color_guard_dual_ratio_1k":
        raise ValueError("unsupported color guard 1k experiment")
    spec = freeze_color_guard_training_spec(config)
    source = config.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("color guard source section is invalid")

    audit_config_path = _repo_path(source["audit_config"], "source.audit_config")
    audit_report_path = _repo_path(source["audit_report"], "source.audit_report")
    gate_policy_path = _repo_path(source["gate_policy"], "source.gate_policy")
    audit_batches_path = _repo_path(source["audit_batches"], "source.audit_batches")
    metric_pairs_path = _repo_path(source["metric_pairs"], "source.metric_pairs")
    audit_config, audit_config_bytes = _load_mapping(audit_config_path, "audit config")
    audit_report, audit_report_bytes = _load_json(audit_report_path)
    gate_policy, gate_policy_bytes = _load_json(gate_policy_path)
    expected_files = {
        "audit_config": str(source["audit_config_sha256"]),
        "audit_report": str(source["audit_report_sha256"]),
        "gate_policy": str(source["gate_policy_sha256"]),
        "audit_batches": str(source["audit_batches_sha256"]),
        "metric_pairs": str(source["metric_pairs_sha256"]),
    }
    actual_files = {
        "audit_config": _sha256_bytes(audit_config_bytes),
        "audit_report": _sha256_bytes(audit_report_bytes),
        "gate_policy": _sha256_bytes(gate_policy_bytes),
        "audit_batches": _sha256_file(audit_batches_path),
        "metric_pairs": _sha256_file(metric_pairs_path),
    }
    if actual_files != expected_files:
        raise ValueError("color guard audit evidence SHA-256 mismatch")
    if (
        audit_report.get("config_sha256") != actual_files["audit_config"]
        or audit_report.get("gate_policy_sha256") != actual_files["gate_policy"]
        or audit_report.get("audit_batches_file_sha256")
        != actual_files["audit_batches"]
        or audit_report.get("metric_pairs_file_sha256")
        != actual_files["metric_pairs"]
    ):
        raise ValueError("color guard audit evidence lineage mismatch")
    matrix = freeze_candidate_matrix(audit_report, spec.ratios)

    audit_source = audit_config.get("source")
    if not isinstance(audit_source, Mapping):
        raise ValueError("audit source section is missing")
    source_paths = {
        name: _repo_path(audit_source[path_key], f"audit.source.{path_key}")
        for name, path_key in {
            "preflight": "preflight_config",
            "render_pool": "render_pool_config",
            "parent": "parent_manifest",
        }.items()
    }
    source_values: dict[str, dict[str, object]] = {}
    source_payloads: dict[str, bytes] = {}
    for name, path in source_paths.items():
        source_values[name], source_payloads[name] = _load_mapping(path, name)
    for name, hash_key in {
        "preflight": "preflight_config_sha256",
        "render_pool": "render_pool_config_sha256",
        "parent": "parent_manifest_sha256",
    }.items():
        if _sha256_bytes(source_payloads[name]) != str(audit_source[hash_key]):
            raise ValueError(f"color guard {name} source SHA-256 mismatch")
    preflight = source_values["preflight"]
    render_pool = source_values["render_pool"]
    parent_manifest = source_values["parent"]
    parent_hash = str(audit_source["parent_p0_hash"])
    if certified_parent_hash(parent_manifest) != parent_hash:
        raise ValueError("certified color guard parent hash mismatch")
    coverage = freeze_render_coverage(render_pool)
    if (
        coverage.camera_count != 31
        or coverage.light_count != 6
        or coverage.resolution != (256, 256)
    ):
        raise ValueError("color guard training requires camera31/light6 at 256x256")

    output_root = _repo_path(config["output_root"], "output_root")
    if output_root.exists():
        raise FileExistsError(f"refusing to inherit color guard training output: {output_root}")

    gltf_path = _repo_path(preflight["inputs"]["gltf"], "inputs.gltf")
    core4_dir = _repo_path(preflight["inputs"]["core4_dir"], "inputs.core4_dir")
    core4_manifest = _repo_path(
        preflight["inputs"]["core4_manifest"], "inputs.core4_manifest"
    )
    mesh = load_gltf_mesh(gltf_path)
    cpu_targets = load_core4_targets(core4_dir, "cpu")
    valid_mask, chart_ids = rasterize_uv_charts(
        mesh.texcoords,
        mesh.triangles,
        height=cpu_targets.height,
        width=cpu_targets.width,
    )
    valid_indices_cpu = torch.nonzero(valid_mask.reshape(-1), as_tuple=False)[:, 0]
    source_base_color_cpu = cpu_targets.select(valid_indices_cpu).base_color_linear
    partition_cpu = build_color_quantile_partition(source_base_color_cpu)
    if partition_cpu.partition_hash != audit_report["partition_manifest"]["partition_hash"]:
        raise ValueError("reconstructed color partition hash mismatch")
    metric_payload = torch.load(metric_pairs_path, map_location="cpu", weights_only=False)
    metric_pairs_cpu = ColorMetricPairs(
        left_valid_positions=metric_payload["left_valid_positions"],
        right_valid_positions=metric_payload["right_valid_positions"],
        left_logical_bin_ids=metric_payload["left_logical_bin_ids"],
        right_logical_bin_ids=metric_payload["right_logical_bin_ids"],
        pairs_per_bin_pair=int(
            audit_report["partition_manifest"]["metric_pairs"][
                "pairs_per_unordered_active_bin_pair"
            ]
        ),
        seed=int(audit_report["partition_manifest"]["metric_pairs"]["seed"]),
        pair_hash=str(metric_payload["pair_hash"]),
    )
    if (
        metric_payload["partition_hash"] != partition_cpu.partition_hash
        or metric_pairs_cpu.pair_hash
        != audit_report["partition_manifest"]["metric_pairs"]["pair_hash"]
    ):
        raise ValueError("frozen metric pair lineage mismatch")

    pca = parent_manifest.get("pca")
    if not isinstance(pca, Mapping):
        raise ValueError("certified parent PCA spec is missing")
    parent_bundle = export_p0_enhanced_bundle(
        _targets_to_seven(cpu_targets),
        valid_mask,
        chart_ids,
        spec=EnhancedPCASpec(
            chroma_tail_strength=float(pca["chroma_tail_strength"]),
            opponent_chroma_weight=float(pca["opponent_chroma_weight"]),
            semantic_group_balance=bool(pca["semantic_group_balance"]),
        ),
        margin=float(preflight["p0"]["safety_margin"]),
    )
    if parent_bundle.manifest != parent_manifest:
        raise ValueError("reconstructed chroma8 parent manifest mismatch")
    if parent_bundle.calibration.safe.artifact_hash != parent_hash:
        raise ValueError("reconstructed chroma8 parent hash mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("color guard 1k training requires CUDA")

    output_root.mkdir(parents=True)
    config_hash = _sha256_bytes(config_bytes)
    decision = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "authorized_ratios": list(spec.ratios),
        "candidate_matrix": matrix,
        "candidate_order": list(spec.candidate_keys),
        "shared_control": "C0",
        "single_gpu_serial_execution": True,
        "config_sha256": config_hash,
        "audit_evidence_sha256": actual_files,
        "gate_policy": gate_policy,
        "formal_holdout_accessed": False,
        "ue_authorized": False,
        "training_beyond_1k_authorized": False,
    }
    _write_new(output_root / "decision.json", _json_bytes(decision))

    device = torch.device("cuda")
    seed = int(preflight["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    parent = _move_p0(parent_bundle.calibration, device)
    targets = Core4Targets(
        base_color_linear=cpu_targets.base_color_linear.to(device),
        normal_xyz=cpu_targets.normal_xyz.to(device),
        roughness=cpu_targets.roughness.to(device),
        metallic=cpu_targets.metallic.to(device),
        height=cpu_targets.height,
        width=cpu_targets.width,
    )
    valid_indices = valid_indices_cpu.to(device)
    source_base_color = source_base_color_cpu.to(device)
    partition = partition_cpu.to(device)
    metric_pairs = ColorMetricPairs(
        left_valid_positions=metric_pairs_cpu.left_valid_positions.to(device),
        right_valid_positions=metric_pairs_cpu.right_valid_positions.to(device),
        left_logical_bin_ids=metric_pairs_cpu.left_logical_bin_ids.to(device),
        right_logical_bin_ids=metric_pairs_cpu.right_logical_bin_ids.to(device),
        pairs_per_bin_pair=metric_pairs_cpu.pairs_per_bin_pair,
        seed=metric_pairs_cpu.seed,
        pair_hash=metric_pairs_cpu.pair_hash,
    )
    valid_cuda = valid_mask.to(device)
    charts_cuda = chart_ids.to(device)

    render = render_pool["render"]
    cameras = [
        _orbit_camera_from_spec(value, render) for value in render_pool["train_cameras"]
    ]
    geometries = [
        render_geometry_gbuffer(mesh, camera, tuple(render["resolution"]), device=device)
        for camera in cameras
    ]
    textures = load_core4_textures(core4_manifest, device)
    lights = [_light(value) for value in render_pool["train_lights"]]
    minimum_roughness = float(render["minimum_roughness"])
    reference_grid = [
        [
            shade_ggx(
                geometry,
                camera,
                light,
                material_override=sample_core4_material(geometry, textures),
                minimum_roughness=minimum_roughness,
            ).detach()
            for geometry, camera in zip(geometries, cameras, strict=True)
        ]
        for light in lights
    ]
    reference_textures = {
        "base_color": targets.base_color_linear.reshape(targets.height, targets.width, 3),
        "normal": targets.normal_xyz.reshape(targets.height, targets.width, 3),
        "roughness": targets.roughness.reshape(targets.height, targets.width, 1),
        "metallic": targets.metallic.reshape(targets.height, targets.width, 1),
    }
    loss_config = dict(preflight["loss"])
    loss_config["minimum_roughness"] = minimum_roughness

    def make_base_objective() -> PreflightObjective:
        return PreflightObjective(
            target=targets,
            valid_indices=valid_indices,
            valid_mask=valid_cuda,
            chart_ids=charts_cuda,
            helmet_geometries=geometries,
            helmet_cameras=cameras,
            helmet_references=reference_grid[0],
            helmet_light=lights[0],
            helmet_lights=lights,
            helmet_reference_grid=reference_grid,
            reference_textures=reference_textures,
            cube_resolution=int(preflight["cube"]["resolution"]),
            cube_config=preflight["cube"],
            loss_config=loss_config,
            tv_lambda=0.0,
            cube_lambda=0.0,
        )

    training_defaults = preflight["training"]
    input_hash = hashlib.sha256(
        (
            _sha256_file(gltf_path)
            + _sha256_file(core4_manifest)
            + parent_hash
            + str(audit_report["input_hash"])
            + partition.partition_hash
            + actual_files["audit_report"]
        ).encode("ascii")
    ).hexdigest()
    state_kwargs = {
        "parent": parent,
        "seed": seed,
        "color_seed_offset": spec.color_seed_offset,
        "partition_hash": partition.partition_hash,
        "config_hash": config_hash,
        "input_hash": input_hash,
        "latent_learning_rate": float(training_defaults["latent_learning_rate"]),
        "affine_learning_rate": float(training_defaults["affine_learning_rate"]),
    }

    frozen_payload = torch.load(audit_batches_path, map_location="cpu", weights_only=False)
    if not isinstance(frozen_payload, list) or len(frozen_payload) < spec.trajectory_gradient_batches:
        raise ValueError("frozen trajectory gradient batches are incomplete")
    trajectory_batches = tuple(
        AffineTrainingBatch(
            core_indices=value["core_indices"].to(device),
            cube_samples=None,
            color_indices=value["color_indices"].to(device),
            color_bin_ids=value["color_bin_ids"].to(device),
        )
        for value in frozen_payload[: spec.trajectory_gradient_batches]
    )

    parent_state = _candidate_state("C0", **state_kwargs)
    parent_atlas = _atlas_endpoint(
        parent_state, targets, valid_indices, source_base_color, partition, metric_pairs
    )
    parent_render_metrics = _endpoint_metrics_for_pool(
        parent_state,
        targets,
        valid_indices,
        geometries,
        cameras,
        reference_grid,
        lights,
        minimum_roughness,
    )
    parent_endpoint = _with_render_metrics(parent_atlas, parent_render_metrics)
    _write_new(output_root / "parent_endpoint.json", _json_bytes(parent_endpoint))
    replayed_parent_color = dict(parent_endpoint["color"])
    replayed_parent_color.pop("oklab_mean_delta_e_report_only")
    if replayed_parent_color != audit_report["parent_color_metrics"]["generic"]:
        raise ValueError("training parent color metrics do not replay the zero-update audit")

    preflight_reports: dict[str, object] = {}
    for key in spec.candidate_keys:
        weights = matrix[key]
        preflight_state = _candidate_state(str(weights["candidate_id"]), **state_kwargs)
        base_objective = make_base_objective()
        objective = _candidate_objective(
            base_objective,
            valid_indices=valid_indices,
            source_base_color=source_base_color,
            weights=weights,
            epsilon=spec.charbonnier_epsilon,
        )
        timing = time_candidate_steps(
            preflight_state,
            objective,
            texel_count=valid_indices.numel(),
            batch_size=spec.material_batch_size,
            cube_sample_count=0,
            warmup_steps=0,
            measured_steps=spec.preflight_steps,
            color_partition=partition,
            color_batch_size=spec.color_batch_size,
        )
        weight, bias = preflight_state.decoder.fold_affine()
        certificate = certify_affine(weight, bias, margin=preflight_state.decoder.margin)
        if (
            not all(math.isfinite(value) for value in timing.mean_loss_terms.values())
            or certificate["valid"] is not True
        ):
            raise RuntimeError(f"{key} 10-step CUDA correctness failed")
        preflight_reports[key] = {
            **asdict(timing),
            "candidate_key": key,
            "candidate": weights,
            "certificate": certificate,
            "final_checkpoint_hash": checkpoint_candidate(preflight_state)[
                "checkpoint_hash"
            ],
        }
        _write_new(
            output_root / "preflight" / f"{key}.json",
            _json_bytes(preflight_reports[key]),
        )
        del preflight_state, objective, base_objective
        torch.cuda.empty_cache()

    candidate_reports: dict[str, object] = {}
    pair_sequence_hashes: dict[str, str] = {}
    rng_fingerprints: dict[str, dict[str, str]] = {}
    started_all = time.perf_counter()
    for key in spec.candidate_keys:
        weights = matrix[key]
        state = _candidate_state(str(weights["candidate_id"]), **state_kwargs)
        training_base = make_base_objective()
        audit_base = make_base_objective()
        objective = _candidate_objective(
            training_base,
            valid_indices=valid_indices,
            source_base_color=source_base_color,
            weights=weights,
            epsilon=spec.charbonnier_epsilon,
        )
        audit_color = _candidate_objective(
            audit_base,
            valid_indices=valid_indices,
            source_base_color=source_base_color,
            weights=weights,
            epsilon=spec.charbonnier_epsilon,
        )
        candidate_root = output_root / "runs" / key
        curve_path = candidate_root / "curve.jsonl"
        trajectory: dict[str, object] = {}
        pair_sequence: list[tuple[int, int]] = []
        latest_pair: list[tuple[int, int]] = []

        def record_render_pair(camera_index: int, light_index: int) -> None:
            value = (int(camera_index), int(light_index))
            pair_sequence.append(value)
            latest_pair[:] = [value]

        training_base.on_render_pair = record_render_pair

        def capture_trajectory() -> dict[str, object]:
            atlas = _atlas_endpoint(
                state,
                targets,
                valid_indices,
                source_base_color,
                partition,
                metric_pairs,
            )
            gradient = _gradient_summary(
                state, trajectory_batches, audit_base, audit_color, loss_config
            )
            return {"atlas": atlas, "gradient": gradient}

        trajectory["step_000000"] = capture_trajectory()

        def report_step(point: dict[str, object]) -> None:
            if not latest_pair:
                raise RuntimeError("training step did not report a camera/light pair")
            enriched = {
                **point,
                "camera_index": latest_pair[0][0],
                "light_index": latest_pair[0][1],
            }
            _append_json_line(curve_path, enriched)
            if int(point["step"]) % spec.progress_interval == 0:
                print(
                    _json_bytes(
                        {
                            "candidate": key,
                            "step": point["step"],
                            "loss": point["loss"],
                            "camera_index": latest_pair[0][0],
                            "light_index": latest_pair[0][1],
                        }
                    ).decode().rstrip(),
                    flush=True,
                )

        def report_trend(point: dict[str, object]) -> None:
            step_key = f"step_{int(point['step']):06d}"
            trajectory[step_key] = {
                "parameter_trend": point,
                **capture_trajectory(),
            }
            _write_new(
                candidate_root / "trajectory" / f"{step_key}.json",
                _json_bytes(trajectory[step_key]),
            )

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        training_run = run_candidate_training(
            state,
            objective,
            parent,
            output_root=candidate_root / "checkpoints",
            observation_plan=TrainingObservationPlan(
                total_steps=spec.total_steps,
                checkpoint_steps=spec.checkpoint_steps,
                trend_steps=spec.checkpoint_steps,
            ),
            texel_count=valid_indices.numel(),
            batch_size=spec.material_batch_size,
            cube_sample_count=0,
            color_partition=partition,
            color_batch_size=spec.color_batch_size,
            on_step=report_step,
            on_trend=report_trend,
        )
        torch.cuda.synchronize(device)
        wall_seconds = time.perf_counter() - started
        training_base.on_render_pair = None
        if len(pair_sequence) != spec.total_steps:
            raise RuntimeError(f"{key} render-pair sample count mismatch")
        pair_sequence_hash = hashlib.sha256(
            json.dumps(pair_sequence, separators=(",", ":")).encode("ascii")
        ).hexdigest()
        pair_sequence_hashes[key] = pair_sequence_hash

        checkpoint_path = (
            candidate_root
            / "checkpoints"
            / str(weights["candidate_id"])
            / "endpoints"
            / "step-001000"
            / "checkpoint.pt"
        )
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        resumed = resume_candidate(
            checkpoint,
            parent,
            expected_parent_p0_hash=parent_hash,
            expected_config_hash=config_hash,
            expected_input_hash=input_hash,
            expected_color_partition_hash=partition.partition_hash,
        )
        resumed_hash = str(checkpoint_candidate(resumed)["checkpoint_hash"])
        if resumed_hash != checkpoint["checkpoint_hash"]:
            raise RuntimeError(f"{key} exact-resume fingerprint mismatch")
        endpoint_atlas = _atlas_endpoint(
            state, targets, valid_indices, source_base_color, partition, metric_pairs
        )
        endpoint_render_metrics = _endpoint_metrics_for_pool(
            state,
            targets,
            valid_indices,
            geometries,
            cameras,
            reference_grid,
            lights,
            minimum_roughness,
        )
        endpoint = _with_render_metrics(endpoint_atlas, endpoint_render_metrics)
        endpoint_artifact = _export_endpoint(
            candidate_root,
            state,
            endpoint_render_metrics,
            str(checkpoint["checkpoint_hash"]),
        )
        rng_fingerprints[key] = {
            "core_rng": _tensor_sha256(checkpoint["core_rng_state"]),
            "color_rng": _tensor_sha256(checkpoint["color_rng_state"]),
        }
        counts = [[0 for _ in lights] for _ in cameras]
        for camera_index, light_index in pair_sequence:
            counts[camera_index][light_index] += 1
        candidate_report = {
            "schema_version": 1,
            "candidate_key": key,
            "candidate": weights,
            "manifest": training_run.manifest,
            "checkpoint_hash": checkpoint["checkpoint_hash"],
            "checkpoint_file_sha256": _sha256_file(checkpoint_path),
            "exact_resume": {
                "resumed_checkpoint_hash": resumed_hash,
                "hash_match": True,
            },
            "endpoint": endpoint,
            "endpoint_artifact": endpoint_artifact,
            "trajectory": trajectory,
            "render_pair_sampling": {
                "sample_count": len(pair_sequence),
                "sequence_hash": pair_sequence_hash,
                "counts_camera_by_light": counts,
                "observed_pair_count": sum(count > 0 for row in counts for count in row),
                "minimum_pair_count": min(count for row in counts for count in row),
                "maximum_pair_count": max(count for row in counts for count in row),
            },
            "sampler_rng_fingerprints": rng_fingerprints[key],
            "curve_points": len(training_run.curve),
            "parameter_trend_points": len(training_run.parameter_trends),
            "wall_seconds": wall_seconds,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
        candidate_reports[key] = candidate_report
        _write_new(candidate_root / "candidate_report.json", _json_bytes(candidate_report))
        del state, resumed, objective, audit_color, training_base, audit_base
        torch.cuda.empty_cache()

    if len(set(pair_sequence_hashes.values())) != 1:
        raise RuntimeError("candidate camera/light sequences are not paired")
    if len({value["core_rng"] for value in rng_fingerprints.values()}) != 1:
        raise RuntimeError("candidate core RNG endpoints are not identical")
    if len({value["color_rng"] for value in rng_fingerprints.values()}) != 1:
        raise RuntimeError("candidate color RNG endpoints are not identical")

    c0_endpoint = candidate_reports["C0"]["endpoint"]
    gates: dict[str, object] = {}
    for key in spec.candidate_keys[1:]:
        gates[key] = evaluate_balanced_color_gate(
            parent_endpoint,
            c0_endpoint,
            candidate_reports[key]["endpoint"],
            gate_policy,
        )
    passing = [key for key, value in gates.items() if value["passed"] is True]
    ranking = sorted(
        passing,
        key=lambda key: (
            max(gates[key]["color_ratios_vs_c0"].values()),
            candidate_reports[key]["endpoint"]["render"]["masked_linear_hdr_mae"],
            candidate_reports[key]["endpoint"]["material"]["seven_channel_mae"],
        ),
    )
    trajectory_conflict = {
        key: {
            group: [
                candidate_reports[key]["trajectory"][f"step_{step:06d}"]["gradient"][
                    "groups"
                ][group]["base_color_negative_counts"]["candidate_color"]
                for step in (0, 250, 500, 750, 1000)
            ]
            for group in ("latent", "affine")
        }
        for key in ("C2-r010", "C2-r025")
    }
    parent_negative = {
        group: {
            color: sum(
                float(batch["parameter_groups"][group]["cosine"]["base"][color])
                < 0.0
                for batch in audit_report["gradient_audit"]["batches"]
            )
            for color in ("opponent", "pair")
        }
        for group in ("latent", "affine")
    }
    parent_requirement_met = all(
        count >= 6
        for groups in parent_negative.values()
        for count in groups.values()
    )
    c3_eligible = parent_requirement_met and any(
        all(count >= 3 for count in groups["latent"])
        and all(count >= 3 for count in groups["affine"])
        and gates[key]["passed"] is False
        for key, groups in trajectory_conflict.items()
    )
    report = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "status": "completed_1k_stop_before_5k_or_ue",
        "config_sha256": config_hash,
        "input_hash": input_hash,
        "parent_p0_hash": parent_hash,
        "render_coverage": asdict(coverage),
        "decision": decision,
        "parent_endpoint": parent_endpoint,
        "preflight": preflight_reports,
        "candidates": candidate_reports,
        "paired_sampling": {
            "pair_sequence_hash": next(iter(pair_sequence_hashes.values())),
            "core_rng_hash": next(iter(rng_fingerprints.values()))["core_rng"],
            "color_rng_hash": next(iter(rng_fingerprints.values()))["color_rng"],
            "all_candidate_sequences_equal": True,
        },
        "gates": gates,
        "passing_candidates": passing,
        "ranking": ranking,
        "preferred_candidate": ranking[0] if ranking else None,
        "c3_trigger": {
            "eligible": c3_eligible,
            "parent_audit_negative_counts": parent_negative,
            "parent_audit_both_groups_requirement_met": parent_requirement_met,
            "c2_trajectory_negative_counts": trajectory_conflict,
        },
        "wall_seconds": time.perf_counter() - started_all,
        "formal_holdout_accessed": False,
        "ue_started": False,
        "training_beyond_1k_started": False,
        "commit_or_push_performed": False,
    }
    _write_new(output_root / "training_report.json", _json_bytes(report))
    return report


def cli_summary(report: Mapping[str, object]) -> dict[str, object]:
    """Keep terminal output bounded; the complete evidence remains in JSON files."""

    return {
        "status": report.get("status"),
        "passing_candidates": report.get("passing_candidates"),
        "preferred_candidate": report.get("preferred_candidate"),
        "wall_seconds": report.get("wall_seconds"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/train/scifihelmet_c4_affine_color_guard_dual_ratio_1k_v1.yaml",
    )
    arguments = parser.parse_args()
    report = run(arguments.config.resolve())
    print(_json_bytes(cli_summary(report)).decode().rstrip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
