"""Audit and run the bounded SciFiHelmet C4 tail/hue color-risk matrix."""

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
for entry in (ROOT, SRC):
    if str(entry) not in sys.path:
        sys.path.insert(0, str(entry))

from audit_scifihelmet_c4_affine_color_guard import (
    _sha256_bytes,
    certified_parent_hash,
)
from cg_frontier.assets.gltf_mesh import load_gltf_mesh
from cg_frontier.compression.affine_color import (
    ColorHuePartition,
    ColorMetricPairs,
    ColorQuantilePartition,
    ColorRiskObjective,
    build_color_hue_partition,
    build_color_quantile_partition,
    color_risk_quality_metrics,
    oklab_mean_delta_e,
)
from cg_frontier.compression.affine_gradient_audit import (
    audit_gradient_objectives,
    calibrate_static_risk_budgets,
)
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
    create_color_risk_candidates,
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
    _load_mapping,
)
from run_scifihelmet_c4_affine_chroma8_l0_40k import (
    _endpoint_metrics_for_pool,
    _orbit_camera_from_spec,
    freeze_render_coverage,
)
from run_scifihelmet_c4_affine_color_guard_1k import (
    _load_json,
    _tensor_sha256,
    _with_render_metrics,
    evaluate_balanced_color_gate,
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


@dataclass(frozen=True)
class FrozenColorRiskTrainingSpec:
    candidate_ids: tuple[str, ...]
    total_ratio: float
    tail_mass: float
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
    hue_group_count: int
    hue_min_group_size: int


def freeze_color_risk_training_spec(
    config: Mapping[str, object],
) -> FrozenColorRiskTrainingSpec:
    """Freeze the four fresh r=0.10 candidates and bounded 1k protocol."""

    training = config.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("training section is missing")
    spec = FrozenColorRiskTrainingSpec(
        candidate_ids=(
            "G0-mean",
            "G1-yc-cvar25",
            "G2-hue8-macro",
            "G3-cvar25-hue8",
        ),
        total_ratio=float(training["total_ratio"]),
        tail_mass=float(training["tail_mass"]),
        preflight_steps=int(training["preflight_steps"]),
        total_steps=int(training["total_steps"]),
        checkpoint_steps=tuple(int(value) for value in training["checkpoint_steps"]),
        metric_interval=int(training["metric_interval"]),
        progress_interval=int(training["progress_interval"]),
        trajectory_gradient_batches=int(training["trajectory_gradient_batches"]),
        material_batch_size=int(training["material_batch_size"]),
        color_batch_size=int(training["color_batch_size"]),
        color_seed_offset=int(training["color_seed_offset"]),
        charbonnier_epsilon=float(training["charbonnier_epsilon"]),
        hue_group_count=int(training["hue_group_count"]),
        hue_min_group_size=int(training["hue_min_group_size"]),
    )
    if spec.total_ratio != 0.10:
        raise ValueError("total color-risk ratio must be 0.10")
    if spec.tail_mass != 0.25:
        raise ValueError("tail mass must be frozen at 0.25")
    if spec.preflight_steps != 10 or spec.total_steps != 1_000:
        raise ValueError("training must freeze 10-step preflight and 1k endpoint")
    if spec.checkpoint_steps != (250, 500, 750, 1_000):
        raise ValueError("checkpoint steps must be 250/500/750/1000")
    if spec.metric_interval != 250 or spec.progress_interval != 100:
        raise ValueError("metric/progress intervals are invalid")
    if spec.trajectory_gradient_batches != 5:
        raise ValueError("trajectory gradient summary must use five batches")
    if spec.material_batch_size != 4_096 or spec.color_batch_size != 4_096:
        raise ValueError("material and color batches must remain 4096")
    if spec.color_seed_offset != 23 or spec.charbonnier_epsilon != 1.0e-3:
        raise ValueError("color RNG or Charbonnier contract changed")
    if spec.hue_group_count != 8 or spec.hue_min_group_size != 4_096:
        raise ValueError("hue partition must use eight groups with 4096 minimum")
    return spec


def freeze_tail_hue_gate_policy() -> dict[str, object]:
    """Freeze the original balanced gate plus generic tail/hue protection."""

    return {
        "schema_version": 1,
        "policy": "balanced_color_tail_hue_protection",
        "parent_chroma_retention_floor": 0.90,
        "global_error_multiplier": 1.05,
        "color_improvement_vs_c0": 0.05,
        "color_regression_vs_c0": 0.02,
        "display_ssim_absolute_drop": 0.005,
        "primary_color_metrics": [
            "uniform_opponent_error",
            "macro_bin_opponent_error",
            "worst_bin_opponent_error",
            "fixed_pair_opponent_error",
        ],
        "new_color_metrics": [
            "macro_bin_cvar25_opponent_error",
            "worst_bin_cvar25_opponent_error",
            "hue_macro_opponent_error",
            "worst_hue_group_opponent_error",
        ],
        "yellow_diagnostic": {"selection_metric": False},
    }


def _metric(mapping: Mapping[str, object], name: str) -> float:
    value = mapping.get(name)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"missing numeric metric: {name}")
    return float(value)


def validate_g0_mean_scale_replay(actual: float, old_raw: float) -> None:
    """Require exact raw replay and the frozen rounded report value."""

    if actual != old_raw or round(actual, 10) != 8.4928470181:
        raise RuntimeError(
            f"G0 mean scale replay mismatch: old={old_raw}, new={actual}"
        )


def evaluate_color_risk_gate(
    parent: Mapping[str, object],
    c0: Mapping[str, object],
    g0: Mapping[str, object],
    candidate: Mapping[str, object],
    policy: Mapping[str, object],
) -> dict[str, object]:
    """Require both the original Pareto gate and the new tail/hue gate."""

    original = evaluate_balanced_color_gate(parent, c0, candidate, policy)
    candidate_color = candidate.get("color")
    g0_color = g0.get("color")
    if not isinstance(candidate_color, Mapping) or not isinstance(g0_color, Mapping):
        raise ValueError("color-risk endpoint metrics are incomplete")
    names = tuple(str(value) for value in policy["new_color_metrics"])
    ratios = {
        name: _metric(candidate_color, name) / _metric(g0_color, name)
        for name in names
    }
    improvement_limit = 1.0 - float(policy["color_improvement_vs_c0"])
    regression_limit = 1.0 + float(policy["color_regression_vs_c0"])
    improved = {name: ratio <= improvement_limit for name, ratio in ratios.items()}
    protected = {name: ratio <= regression_limit for name, ratio in ratios.items()}
    new_gates = {
        "new_metric_improvement": any(improved.values()),
        "new_metric_no_regression": all(protected.values()),
    }
    return {
        **original,
        "passed": bool(original["passed"]) and all(new_gates.values()),
        "gates": {**original["gates"], **new_gates},
        "new_metric_ratios_vs_g0": ratios,
        "new_metric_improvement_count": sum(improved.values()),
    }


def _candidate_state(
    candidate_id: str,
    parent: object,
    *,
    seed: int,
    color_seed_offset: int,
    partition_hash: str,
    group_hash: str,
    config_hash: str,
    input_hash: str,
    latent_learning_rate: float,
    affine_learning_rate: float,
) -> AffineCandidateState:
    return create_color_risk_candidates(
        parent,
        core_seed=seed + 11,
        color_seed=seed + color_seed_offset,
        color_partition_hash=partition_hash,
        color_group_hash=group_hash,
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
    partition: ColorQuantilePartition,
    hue_partition: ColorHuePartition,
    scales: Mapping[str, object],
    ratio: float,
    tail_mass: float,
    epsilon: float,
) -> ColorRiskObjective:
    return ColorRiskObjective(
        base_objective,
        valid_flat_indices=valid_indices,
        source_base_color=source_base_color,
        yc_partition=partition,
        hue_partition=hue_partition,
        mean_scale=float(scales["mean"]),
        cvar_scale=float(scales["yc_cvar25"]),
        hue_scale=float(scales["hue_macro"]),
        total_ratio=ratio,
        tail_mass=tail_mass,
        epsilon=epsilon,
    )


@torch.no_grad()
def _risk_atlas_endpoint(
    state: AffineCandidateState,
    targets: Core4Targets,
    valid_indices: torch.Tensor,
    source_base_color: torch.Tensor,
    partition: ColorQuantilePartition,
    hue_partition: ColorHuePartition,
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
    color = color_risk_quality_metrics(
        decoded.base_color_linear,
        source_base_color,
        partition,
        hue_partition,
        metric_pairs,
    )
    color["oklab_mean_delta_e_report_only"] = float(
        oklab_mean_delta_e(decoded.base_color_linear, source_base_color).cpu()
    )
    weight, bias = state.decoder.fold_affine()
    yellow = (
        (source_base_color[:, 0] > source_base_color[:, 1])
        & (source_base_color[:, 1] > source_base_color[:, 2])
        & (source_base_color[:, 0] - source_base_color[:, 2] > 0.05)
        & (source_base_color[:, 1] - source_base_color[:, 2] > 0.02)
    )
    return {
        "step": state.optimizer_updates,
        "color": color,
        "material": {
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
            "roughness_l1": float(
                F.l1_loss(decoded.roughness, selected.roughness).cpu()
            ),
            "metallic_l1": float(
                F.l1_loss(decoded.metallic, selected.metallic).cpu()
            ),
        },
        "certificate": certify_affine(weight, bias, margin=state.decoder.margin),
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


def _risk_gradient_summary(
    state: AffineCandidateState,
    batches: Sequence[AffineTrainingBatch],
    base_objective: PreflightObjective,
    color_objective: ColorRiskObjective,
    loss_config: Mapping[str, object],
) -> dict[str, object]:
    weights = color_objective.weights_for_candidate(state.candidate_id)

    def terms(batch: AffineTrainingBatch) -> dict[str, torch.Tensor]:
        base, base_terms = base_objective(state, batch)
        risks = color_objective.risk_terms(state, batch)
        candidate_color = sum(risks[name] * weight for name, weight in weights.items())
        return {
            "base": base,
            "render": base_terms["helmet_charbonnier"]
            * float(loss_config["helmet_charbonnier"])
            + base_terms["helmet_log1p"] * float(loss_config["helmet_log1p"]),
            "rgb": base_terms["base_color_l1"]
            * float(loss_config["base_color_l1"]),
            "mean": risks["mean"],
            "yc_cvar25": risks["yc_cvar25"],
            "hue_macro": risks["hue_macro"],
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
    for group_name in ("latent", "affine"):
        cosines = {
            name: [
                float(batch["parameter_groups"][group_name]["cosine"]["base"][name])
                for batch in audit["batches"]
            ]
            for name in ("mean", "yc_cvar25", "hue_macro", "candidate_color")
        }
        groups[group_name] = {
            "base_color_negative_counts": {
                name: sum(value < 0.0 for value in values)
                for name, values in cosines.items()
            },
            "base_color_cosine_medians": {
                name: statistics.median(values) for name, values in cosines.items()
            },
            "gradient_norm_medians": {
                name: statistics.median(
                    float(batch["parameter_groups"][group_name]["norms"][name])
                    for batch in audit["batches"]
                )
                for name in audit["objective_names"]
            },
        }
    return {
        "schema_version": 1,
        "step": state.optimizer_updates,
        "batch_count": len(batches),
        "weights": weights,
        "groups": groups,
    }


def _nested_equal(left: object, right: object) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return set(left) == set(right) and all(
            _nested_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _nested_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def _hue_manifest(
    hue_partition: ColorHuePartition,
    source_base_color: torch.Tensor,
    frozen_batches: Sequence[AffineTrainingBatch],
) -> dict[str, object]:
    source = source_base_color.detach().cpu()
    yellow = (
        (source[:, 0] > source[:, 1])
        & (source[:, 1] > source[:, 2])
        & (source[:, 0] - source[:, 2] > 0.05)
        & (source[:, 1] - source[:, 2] > 0.02)
    )
    group_ids = hue_partition.valid_group_ids.detach().cpu()
    group_yellow = [
        int(torch.count_nonzero(yellow & (group_ids == group_id)))
        for group_id in range(9)
    ]
    batch_group_counts = []
    for batch in frozen_batches:
        indices = batch.color_indices.detach().cpu()
        batch_group_counts.append(
            [
                int(torch.count_nonzero(group_ids[indices] == group_id))
                for group_id in range(9)
            ]
        )
    return {
        "schema_version": 1,
        "definition": "source-opponent-neutral-median-plus-eight-equal-mass-hue-groups",
        "chroma_threshold": hue_partition.chroma_threshold,
        "circular_mean": hue_partition.circular_mean,
        "circular_resultant": hue_partition.circular_resultant,
        "seam": hue_partition.seam,
        "rotated_hue_edges": list(hue_partition.rotated_hue_edges),
        "group_sizes": [int(value) for value in hue_partition.group_sizes],
        "minimum_group_size": int(hue_partition.group_sizes.min()),
        "base_partition_hash": hue_partition.base_partition_hash,
        "color_group_hash": hue_partition.group_hash,
        "frozen_batch_group_counts": batch_group_counts,
        "minimum_frozen_batch_group_count": min(min(row) for row in batch_group_counts),
        "yellow_diagnostic": {
            "selection_metric": False,
            "group_counts": group_yellow,
            "total": int(torch.count_nonzero(yellow)),
        },
    }


def run(config_path: Path) -> dict[str, object]:
    config, config_bytes = _load_mapping(config_path, "color risk 1k config")
    if config.get("experiment") != "scifihelmet_c4_affine_color_risk_1k_v1":
        raise ValueError("unsupported color risk experiment")
    spec = freeze_color_risk_training_spec(config)
    source = config.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("color risk source section is missing")

    source_keys = {
        "audit_config": "audit_config",
        "audit_report": "audit_report",
        "audit_batches": "audit_batches",
        "metric_pairs": "metric_pairs",
        "old_training_config": "old_training_config",
        "old_training_report": "old_training_report",
        "old_g0_checkpoint": "old_g0_checkpoint",
        "old_g0_curve": "old_g0_curve",
    }
    source_paths = {
        name: _repo_path(source[path_key], f"source.{path_key}")
        for name, path_key in source_keys.items()
    }
    actual_hashes = {
        name: _sha256_file(path) for name, path in source_paths.items()
    }
    expected_hashes = {
        name: str(source[f"{path_key}_sha256"])
        for name, path_key in source_keys.items()
    }
    if actual_hashes != expected_hashes:
        raise ValueError("color risk source SHA-256 mismatch")
    audit_config, audit_config_bytes = _load_mapping(
        source_paths["audit_config"], "source audit config"
    )
    audit_report, audit_report_bytes = _load_json(source_paths["audit_report"])
    old_training_report, _ = _load_json(source_paths["old_training_report"])
    if audit_report.get("config_sha256") != _sha256_bytes(audit_config_bytes):
        raise ValueError("source audit config lineage mismatch")
    if (
        audit_report.get("audit_batches_file_sha256")
        != actual_hashes["audit_batches"]
        or audit_report.get("metric_pairs_file_sha256")
        != actual_hashes["metric_pairs"]
    ):
        raise ValueError("source audit payload lineage mismatch")
    old_g0_report = old_training_report.get("candidates", {}).get("C1-r010")
    if not isinstance(old_g0_report, Mapping):
        raise ValueError("old C1-r010 replay report is missing")
    if old_g0_report.get("checkpoint_file_sha256") != actual_hashes["old_g0_checkpoint"]:
        raise ValueError("old G0 checkpoint lineage mismatch")

    audit_source = audit_config.get("source")
    if not isinstance(audit_source, Mapping):
        raise ValueError("source audit inputs are missing")
    upstream_paths = {
        name: _repo_path(audit_source[path_key], f"audit.source.{path_key}")
        for name, path_key in {
            "preflight": "preflight_config",
            "render_pool": "render_pool_config",
            "parent": "parent_manifest",
        }.items()
    }
    upstream_values: dict[str, dict[str, object]] = {}
    upstream_payloads: dict[str, bytes] = {}
    for name, path in upstream_paths.items():
        upstream_values[name], upstream_payloads[name] = _load_mapping(path, name)
    for name, hash_key in {
        "preflight": "preflight_config_sha256",
        "render_pool": "render_pool_config_sha256",
        "parent": "parent_manifest_sha256",
    }.items():
        if _sha256_bytes(upstream_payloads[name]) != str(audit_source[hash_key]):
            raise ValueError(f"color risk {name} source SHA-256 mismatch")
    preflight = upstream_values["preflight"]
    render_pool = upstream_values["render_pool"]
    parent_manifest = upstream_values["parent"]
    parent_hash = str(audit_source["parent_p0_hash"])
    if certified_parent_hash(parent_manifest) != parent_hash:
        raise ValueError("certified parent hash mismatch")
    coverage = freeze_render_coverage(render_pool)
    if (
        coverage.camera_count != 31
        or coverage.light_count != 6
        or coverage.resolution != (256, 256)
    ):
        raise ValueError("color risk requires camera31/light6 at 256x256")

    output_root = _repo_path(config["output_root"], "output_root")
    if output_root.exists():
        raise FileExistsError(f"refusing to inherit color risk output: {output_root}")

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
        raise ValueError("reconstructed YC partition hash mismatch")
    hue_partition_cpu = build_color_hue_partition(
        source_base_color_cpu,
        partition_cpu,
        min_group_size=spec.hue_min_group_size,
    )
    metric_payload = torch.load(
        source_paths["metric_pairs"], map_location="cpu", weights_only=False
    )
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
        raise ValueError("reconstructed parent manifest mismatch")
    if parent_bundle.calibration.safe.artifact_hash != parent_hash:
        raise ValueError("reconstructed parent artifact hash mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("color risk audit and training require CUDA")

    output_root.mkdir(parents=True)
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
    hue_partition = hue_partition_cpu.to(device)
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

    frozen_payload = torch.load(
        source_paths["audit_batches"], map_location="cpu", weights_only=False
    )
    if not isinstance(frozen_payload, list) or len(frozen_payload) != 8:
        raise ValueError("risk audit requires the original eight frozen batches")
    frozen_batches = tuple(
        AffineTrainingBatch(
            core_indices=value["core_indices"].to(device),
            cube_samples=None,
            color_indices=value["color_indices"].to(device),
            color_bin_ids=value["color_bin_ids"].to(device),
        )
        for value in frozen_payload
    )
    trajectory_batches = frozen_batches[: spec.trajectory_gradient_batches]
    hue_manifest = _hue_manifest(
        hue_partition_cpu, source_base_color_cpu, frozen_batches
    )
    if hue_manifest["minimum_frozen_batch_group_count"] <= 0:
        raise RuntimeError("a frozen batch is missing a hue group")

    config_hash = _sha256_bytes(config_bytes)
    input_hash = hashlib.sha256(
        (
            _sha256_file(gltf_path)
            + _sha256_file(core4_manifest)
            + parent_hash
            + partition.partition_hash
            + hue_partition.group_hash
            + actual_hashes["audit_batches"]
            + actual_hashes["old_training_report"]
        ).encode("ascii")
    ).hexdigest()
    training_defaults = preflight["training"]
    state_kwargs = {
        "parent": parent,
        "seed": seed,
        "color_seed_offset": spec.color_seed_offset,
        "partition_hash": partition.partition_hash,
        "group_hash": hue_partition.group_hash,
        "config_hash": config_hash,
        "input_hash": input_hash,
        "latent_learning_rate": float(training_defaults["latent_learning_rate"]),
        "affine_learning_rate": float(training_defaults["affine_learning_rate"]),
    }

    gate_policy = freeze_tail_hue_gate_policy()
    _write_new(output_root / "hue_partition_manifest.json", _json_bytes(hue_manifest))
    _write_new(output_root / "gate_policy.json", _json_bytes(gate_policy))
    decision = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "candidate_order": list(spec.candidate_ids),
        "total_ratio": spec.total_ratio,
        "tail_mass": spec.tail_mass,
        "single_gpu_serial_execution": True,
        "fresh_from_parent": True,
        "old_failed_checkpoint_used_for_training": False,
        "g0_reference_only": str(source_paths["old_g0_checkpoint"]),
        "config_sha256": config_hash,
        "input_hash": input_hash,
        "source_hashes": actual_hashes,
        "gate_policy_sha256": _sha256_bytes(_json_bytes(gate_policy)),
        "formal_holdout_accessed": False,
        "ue_authorized": False,
        "training_beyond_1k_authorized": False,
    }
    _write_new(output_root / "decision.json", _json_bytes(decision))

    audit_state = _candidate_state("G0-mean", **state_kwargs)
    audit_base = make_base_objective()
    unit_scales = {"mean": 1.0, "yc_cvar25": 1.0, "hue_macro": 1.0}
    audit_color = _candidate_objective(
        audit_base,
        valid_indices=valid_indices,
        source_base_color=source_base_color,
        partition=partition,
        hue_partition=hue_partition,
        scales=unit_scales,
        ratio=spec.total_ratio,
        tail_mass=spec.tail_mass,
        epsilon=spec.charbonnier_epsilon,
    )
    checkpoint_before = checkpoint_candidate(audit_state)

    def audit_terms(batch: AffineTrainingBatch) -> dict[str, torch.Tensor]:
        base, terms = audit_base(audit_state, batch)
        risks = audit_color.risk_terms(audit_state, batch)
        return {
            "base": base,
            "render": terms["helmet_charbonnier"]
            * float(loss_config["helmet_charbonnier"])
            + terms["helmet_log1p"] * float(loss_config["helmet_log1p"]),
            "rgb": terms["base_color_l1"] * float(loss_config["base_color_l1"]),
            "mean": risks["mean"],
            "yc_cvar25": risks["yc_cvar25"],
            "hue_macro": risks["hue_macro"],
            "normal": terms["normal_cosine"] * float(loss_config["normal_cosine"]),
            "roughness": terms["roughness_l1"] * float(loss_config["roughness_l1"]),
            "metallic": terms["metallic_l1"] * float(loss_config["metallic_l1"]),
        }

    zero_update_audit = audit_gradient_objectives(
        batches=frozen_batches,
        objective_terms=audit_terms,
        parameter_groups={
            "latent": (audit_state.latent,),
            "affine": tuple(audit_state.decoder.parameters()),
        },
        epsilon=1.0e-12,
    )
    calibration = calibrate_static_risk_budgets(
        zero_update_audit, total_ratio=spec.total_ratio, epsilon=1.0e-12
    )
    checkpoint_after = checkpoint_candidate(audit_state)
    if checkpoint_before["checkpoint_hash"] != checkpoint_after["checkpoint_hash"]:
        raise RuntimeError("zero-update risk audit mutated candidate state")
    old_mean_scale = float(audit_report["calibration"]["scales"]["opponent"])
    new_mean_scale = float(calibration["scales"]["mean"])
    validate_g0_mean_scale_replay(new_mean_scale, old_mean_scale)
    audit_result = {
        "schema_version": 1,
        "status": "passed_zero_update_audit",
        "batch_count": 8,
        "checkpoint_hash_before": checkpoint_before["checkpoint_hash"],
        "checkpoint_hash_after": checkpoint_after["checkpoint_hash"],
        "optimizer_updates_before": 0,
        "optimizer_updates_after": audit_state.optimizer_updates,
        "g0_mean_scale_expected": 8.4928470181,
        "g0_mean_scale_actual": new_mean_scale,
        "gradient_audit": zero_update_audit,
        "calibration": calibration,
        "training_started": False,
    }
    _write_new(output_root / "zero_update_gradient_audit.json", _json_bytes(audit_result))
    del audit_state, audit_base, audit_color
    torch.cuda.empty_cache()

    scales = calibration["scales"]
    parent_state = _candidate_state("G0-mean", **state_kwargs)
    parent_atlas = _risk_atlas_endpoint(
        parent_state,
        targets,
        valid_indices,
        source_base_color,
        partition,
        hue_partition,
        metric_pairs,
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

    preflight_reports: dict[str, object] = {}
    for candidate_id in spec.candidate_ids:
        state = _candidate_state(candidate_id, **state_kwargs)
        base_objective = make_base_objective()
        objective = _candidate_objective(
            base_objective,
            valid_indices=valid_indices,
            source_base_color=source_base_color,
            partition=partition,
            hue_partition=hue_partition,
            scales=scales,
            ratio=spec.total_ratio,
            tail_mass=spec.tail_mass,
            epsilon=spec.charbonnier_epsilon,
        )
        timing = time_candidate_steps(
            state,
            objective,
            texel_count=valid_indices.numel(),
            batch_size=spec.material_batch_size,
            cube_sample_count=0,
            warmup_steps=0,
            measured_steps=spec.preflight_steps,
            color_partition=partition,
            color_batch_size=spec.color_batch_size,
        )
        weight, bias = state.decoder.fold_affine()
        certificate = certify_affine(weight, bias, margin=state.decoder.margin)
        if (
            not all(math.isfinite(value) for value in timing.mean_loss_terms.values())
            or certificate["valid"] is not True
        ):
            raise RuntimeError(f"{candidate_id} 10-step CUDA correctness failed")
        preflight_reports[candidate_id] = {
            **asdict(timing),
            "candidate_id": candidate_id,
            "weights": objective.weights_for_candidate(candidate_id),
            "certificate": certificate,
            "checkpoint_hash": checkpoint_candidate(state)["checkpoint_hash"],
        }
        _write_new(
            output_root / "preflight" / f"{candidate_id}.json",
            _json_bytes(preflight_reports[candidate_id]),
        )
        del state, objective, base_objective
        torch.cuda.empty_cache()

    candidate_reports: dict[str, object] = {}
    pair_sequence_hashes: dict[str, str] = {}
    rng_fingerprints: dict[str, dict[str, str]] = {}
    g0_replay: dict[str, object] | None = None
    started_all = time.perf_counter()
    for candidate_id in spec.candidate_ids:
        state = _candidate_state(candidate_id, **state_kwargs)
        training_base = make_base_objective()
        trajectory_base = make_base_objective()
        objective = _candidate_objective(
            training_base,
            valid_indices=valid_indices,
            source_base_color=source_base_color,
            partition=partition,
            hue_partition=hue_partition,
            scales=scales,
            ratio=spec.total_ratio,
            tail_mass=spec.tail_mass,
            epsilon=spec.charbonnier_epsilon,
        )
        trajectory_color = _candidate_objective(
            trajectory_base,
            valid_indices=valid_indices,
            source_base_color=source_base_color,
            partition=partition,
            hue_partition=hue_partition,
            scales=scales,
            ratio=spec.total_ratio,
            tail_mass=spec.tail_mass,
            epsilon=spec.charbonnier_epsilon,
        )
        candidate_root = output_root / "runs" / candidate_id
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
            return {
                "atlas": _risk_atlas_endpoint(
                    state,
                    targets,
                    valid_indices,
                    source_base_color,
                    partition,
                    hue_partition,
                    metric_pairs,
                ),
                "gradient": _risk_gradient_summary(
                    state,
                    trajectory_batches,
                    trajectory_base,
                    trajectory_color,
                    loss_config,
                ),
            }

        trajectory["step_000000"] = capture_trajectory()
        _write_new(
            candidate_root / "trajectory" / "step_000000.json",
            _json_bytes(trajectory["step_000000"]),
        )

        def report_step(point: dict[str, object]) -> None:
            if not latest_pair:
                raise RuntimeError("training step did not report a camera/light pair")
            enriched = {
                **point,
                "camera_index": latest_pair[0][0],
                "light_index": latest_pair[0][1],
            }
            if not math.isfinite(float(point["loss"])) or not all(
                math.isfinite(float(value)) for value in point["terms"].values()
            ):
                raise RuntimeError(f"{candidate_id} emitted a non-finite loss")
            _append_json_line(curve_path, enriched)
            if int(point["step"]) % spec.progress_interval == 0:
                print(
                    _json_bytes(
                        {
                            "candidate": candidate_id,
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
            raise RuntimeError(f"{candidate_id} render-pair sample count mismatch")
        pair_sequence_hash = hashlib.sha256(
            json.dumps(pair_sequence, separators=(",", ":")).encode("ascii")
        ).hexdigest()
        pair_sequence_hashes[candidate_id] = pair_sequence_hash

        checkpoint_path = (
            candidate_root
            / "checkpoints"
            / candidate_id
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
            expected_color_group_hash=hue_partition.group_hash,
        )
        resumed_hash = str(checkpoint_candidate(resumed)["checkpoint_hash"])
        if resumed_hash != checkpoint["checkpoint_hash"]:
            raise RuntimeError(f"{candidate_id} exact-resume fingerprint mismatch")

        if candidate_id == "G0-mean":
            old_checkpoint = torch.load(
                source_paths["old_g0_checkpoint"], map_location=device, weights_only=False
            )
            replay_fields = (
                "latent",
                "safe_affine_raw_parameters",
                "latent_optimizer",
                "affine_optimizer",
                "core_rng_state",
                "color_rng_state",
            )
            field_matches = {
                field: _nested_equal(checkpoint[field], old_checkpoint[field])
                for field in replay_fields
            }
            old_curve = [
                json.loads(line)
                for line in source_paths["old_g0_curve"].read_text(
                    encoding="utf-8"
                ).splitlines()
                if line
            ]
            curve_match = len(old_curve) == len(training_run.curve) and all(
                int(old["step"]) == int(new["step"])
                and old["loss"] == new["loss"]
                and old["terms"] == new["terms"]
                for old, new in zip(old_curve, training_run.curve, strict=True)
            )
            sequence_match = (
                pair_sequence_hash
                == old_g0_report["render_pair_sampling"]["sequence_hash"]
            )
            g0_replay = {
                "schema_version": 1,
                "reference_candidate": "C1-r010",
                "reference_checkpoint_hash": old_checkpoint["checkpoint_hash"],
                "field_matches": field_matches,
                "public_loss_curve_match": curve_match,
                "render_pair_sequence_match": sequence_match,
                "passed": all(field_matches.values()) and curve_match and sequence_match,
            }
            _write_new(output_root / "g0_exact_replay.json", _json_bytes(g0_replay))
            if g0_replay["passed"] is not True:
                raise RuntimeError("G0 exact replay failed; refusing to run G1-G3")

        endpoint_atlas = _risk_atlas_endpoint(
            state,
            targets,
            valid_indices,
            source_base_color,
            partition,
            hue_partition,
            metric_pairs,
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
        rng_fingerprints[candidate_id] = {
            "core_rng": _tensor_sha256(checkpoint["core_rng_state"]),
            "color_rng": _tensor_sha256(checkpoint["color_rng_state"]),
        }
        counts = [[0 for _ in lights] for _ in cameras]
        for camera_index, light_index in pair_sequence:
            counts[camera_index][light_index] += 1
        candidate_report = {
            "schema_version": 1,
            "candidate_id": candidate_id,
            "weights": objective.weights_for_candidate(candidate_id),
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
                "observed_pair_count": sum(
                    count > 0 for row in counts for count in row
                ),
                "minimum_pair_count": min(count for row in counts for count in row),
                "maximum_pair_count": max(count for row in counts for count in row),
            },
            "sampler_rng_fingerprints": rng_fingerprints[candidate_id],
            "curve_points": len(training_run.curve),
            "parameter_trend_points": len(training_run.parameter_trends),
            "wall_seconds": wall_seconds,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
        candidate_reports[candidate_id] = candidate_report
        _write_new(
            candidate_root / "candidate_report.json", _json_bytes(candidate_report)
        )
        del state, resumed, objective, trajectory_color, training_base, trajectory_base
        torch.cuda.empty_cache()

    if len(set(pair_sequence_hashes.values())) != 1:
        raise RuntimeError("candidate camera/light sequences are not paired")
    if len({value["core_rng"] for value in rng_fingerprints.values()}) != 1:
        raise RuntimeError("candidate core RNG endpoints are not identical")
    if len({value["color_rng"] for value in rng_fingerprints.values()}) != 1:
        raise RuntimeError("candidate color RNG endpoints are not identical")
    if g0_replay is None or g0_replay["passed"] is not True:
        raise RuntimeError("G0 replay evidence is missing")

    old_c0 = old_training_report["candidates"]["C0"]["endpoint"]
    g0_endpoint = candidate_reports["G0-mean"]["endpoint"]
    gates = {
        candidate_id: evaluate_color_risk_gate(
            parent_endpoint,
            old_c0,
            g0_endpoint,
            candidate_reports[candidate_id]["endpoint"],
            gate_policy,
        )
        for candidate_id in spec.candidate_ids[1:]
    }
    passing = [
        candidate_id
        for candidate_id, value in gates.items()
        if value["passed"] is True
    ]
    ranking = sorted(
        passing,
        key=lambda candidate_id: (
            max(
                *gates[candidate_id]["color_ratios_vs_c0"].values(),
                *gates[candidate_id]["new_metric_ratios_vs_g0"].values(),
            ),
            -float(
                candidate_reports[candidate_id]["endpoint"]["render"]["display_ssim"]
            ),
            float(
                candidate_reports[candidate_id]["endpoint"]["render"]
                ["masked_linear_hdr_mae"]
            ),
            float(
                candidate_reports[candidate_id]["endpoint"]["material"]
                ["seven_channel_mae"]
            ),
        ),
    )
    report = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "status": "completed_bounded_1k_stop",
        "config_sha256": config_hash,
        "input_hash": input_hash,
        "parent_p0_hash": parent_hash,
        "render_coverage": asdict(coverage),
        "decision": decision,
        "hue_partition_manifest": hue_manifest,
        "zero_update_audit": audit_result,
        "g0_exact_replay": g0_replay,
        "parent_endpoint": parent_endpoint,
        "old_c0_endpoint": old_c0,
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
        "wall_seconds": time.perf_counter() - started_all,
        "formal_holdout_accessed": False,
        "ue_started": False,
        "training_beyond_1k_started": False,
        "commit_or_push_performed": False,
    }
    _write_new(output_root / "training_report.json", _json_bytes(report))
    return report


def cli_summary(report: Mapping[str, object]) -> dict[str, object]:
    return {
        "status": report.get("status"),
        "g0_exact_replay": report.get("g0_exact_replay", {}).get("passed"),
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
        / "configs/train/scifihelmet_c4_affine_color_risk_1k_v1.yaml",
    )
    arguments = parser.parse_args()
    report = run(arguments.config.resolve())
    print(json.dumps(cli_summary(report), ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
