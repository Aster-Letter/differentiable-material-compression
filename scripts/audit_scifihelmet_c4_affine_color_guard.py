"""Audit generic color losses at the frozen SciFiHelmet chroma8 parent."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import sys
from typing import Mapping

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cg_frontier.assets.gltf_mesh import load_gltf_mesh
from cg_frontier.compression.affine_color import (
    ColorGuardObjective,
    ColorMetricPairs,
    ColorQuantilePartition,
    build_color_quantile_partition,
    color_quality_metrics,
    freeze_color_metric_pairs,
)
from cg_frontier.compression.affine_gradient_audit import (
    audit_gradient_objectives,
    calibrate_static_color_budgets,
)
from cg_frontier.compression.affine_pca import (
    EnhancedPCASpec,
    export_p0_enhanced_bundle,
    rasterize_uv_charts,
)
from cg_frontier.compression.affine_regularizers import fake_quantize_unorm8
from cg_frontier.compression.affine_training import (
    checkpoint_candidate,
    create_color_candidates,
    draw_training_batch,
)
from cg_frontier.compression.material import Core4Targets, load_core4_targets
from cg_frontier.render.gbuffer import (
    load_core4_textures,
    render_geometry_gbuffer,
    sample_core4_material,
)
from cg_frontier.render.pbr import shade_ggx
from run_scifihelmet_c4_affine_40k import _load_mapping
from run_scifihelmet_c4_affine_chroma8_l0_40k import (
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


@dataclass(frozen=True)
class FrozenColorAuditSpec:
    batch_count: int
    material_batch_size: int
    color_batch_size: int
    color_seed_offset: int
    metric_seed_offset: int
    pairs_per_bin_pair: int
    quantiles: tuple[float, ...]
    charbonnier_epsilon: float
    gradient_epsilon: float
    ratios: tuple[float, ...]
    selected_ratio: None


def freeze_color_audit_spec(config: Mapping[str, object]) -> FrozenColorAuditSpec:
    """Validate the bounded zero-update audit and its manual ratio stop."""

    audit = config.get("audit")
    if not isinstance(audit, Mapping):
        raise ValueError("audit section is missing")
    if audit.get("selected_ratio") is not None:
        raise ValueError("zero-update audit must not select a ratio")
    quantiles = tuple(float(value) for value in audit["quantiles"])
    ratios = tuple(float(value) for value in audit["ratios"])
    spec = FrozenColorAuditSpec(
        batch_count=int(audit["batch_count"]),
        material_batch_size=int(audit["material_batch_size"]),
        color_batch_size=int(audit["color_batch_size"]),
        color_seed_offset=int(audit["color_seed_offset"]),
        metric_seed_offset=int(audit["metric_seed_offset"]),
        pairs_per_bin_pair=int(audit["pairs_per_bin_pair"]),
        quantiles=quantiles,
        charbonnier_epsilon=float(audit["charbonnier_epsilon"]),
        gradient_epsilon=float(audit["gradient_epsilon"]),
        ratios=ratios,
        selected_ratio=None,
    )
    if spec.batch_count != 8:
        raise ValueError("color audit must use exactly eight batches")
    if spec.material_batch_size <= 0 or spec.color_batch_size <= 0:
        raise ValueError("audit batch sizes must be positive")
    if spec.color_batch_size % 2 != 0:
        raise ValueError("color batch size must be even")
    if spec.quantiles != (0.25, 0.50, 0.75):
        raise ValueError("color audit must use the frozen quartile grid")
    if spec.ratios != (0.10, 0.25, 0.50):
        raise ValueError("color audit ratios do not match the frozen sweep")
    if spec.charbonnier_epsilon != 1.0e-3 or spec.gradient_epsilon <= 0.0:
        raise ValueError("color audit epsilon values are invalid")
    if spec.pairs_per_bin_pair != 32:
        raise ValueError("metric audit must freeze 32 pairs per active-bin pair")
    return spec


def freeze_balanced_gate_policy() -> dict[str, object]:
    """Return the pre-training balanced Pareto gate without semantic color masks."""

    return {
        "schema_version": 1,
        "policy": "balanced_color_protection",
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
        "yellow_diagnostic": {"selection_metric": False},
        "selected_ratio": None,
    }


def certified_parent_hash(manifest: Mapping[str, object]) -> str:
    """Read the deployable hash from the manifest's certified safe artifact."""

    safe_artifact = manifest.get("safe_artifact")
    if not isinstance(safe_artifact, Mapping):
        raise ValueError("parent manifest safe artifact is missing")
    value = safe_artifact.get("artifact_hash")
    if not isinstance(value, str) or not value:
        raise ValueError("parent manifest safe artifact hash is missing")
    return value


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _partition_manifest(
    partition: ColorQuantilePartition, metric_pairs: ColorMetricPairs
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "definition": "source_valid_linear_rgb_yorth_chroma_quartiles",
        "yorth_definition": "(R+G+B)/sqrt(3); orthogonal achromatic axis, not photometric luminance",
        "opponent_definition": ["(R-G)/sqrt(2)", "(R+G-2B)/sqrt(6)"],
        "quantiles": [0.25, 0.50, 0.75],
        "yorth_edges": list(partition.yorth_edges),
        "chroma_edges": list(partition.chroma_edges),
        "logical_bin_ids": [int(value) for value in partition.logical_bin_ids],
        "active_bin_count": partition.active_bin_count,
        "active_bin_sizes": [int(value) for value in partition.bin_sizes],
        "minimum_active_bin_size": int(partition.bin_sizes.min()),
        "valid_texel_count": int(partition.bin_sizes.sum()),
        "partition_hash": partition.partition_hash,
        "metric_pairs": {
            "seed": metric_pairs.seed,
            "pairs_per_unordered_active_bin_pair": metric_pairs.pairs_per_bin_pair,
            "pair_count": int(metric_pairs.left_valid_positions.numel()),
            "pair_hash": metric_pairs.pair_hash,
        },
        "uses_semantic_mask": False,
        "yellow_selection_metric": False,
    }


def _torch_payload(value: object) -> bytes:
    buffer = io.BytesIO()
    torch.save(value, buffer)
    return buffer.getvalue()


@torch.no_grad()
def _parent_color_report(
    state: object,
    source_base_color: torch.Tensor,
    valid_indices: torch.Tensor,
    partition: ColorQuantilePartition,
    metric_pairs: ColorMetricPairs,
) -> dict[str, object]:
    deployed = fake_quantize_unorm8(getattr(state, "latent").detach())
    decoded = getattr(state, "decoder")(
        deployed.reshape(-1, 4)[valid_indices]
    )
    prediction = decoded.base_color_linear
    metrics = color_quality_metrics(
        prediction,
        source_base_color,
        partition,
        metric_pairs,
    )
    yellow = (
        (source_base_color[:, 0] > source_base_color[:, 1])
        & (source_base_color[:, 1] > source_base_color[:, 2])
        & (source_base_color[:, 0] - source_base_color[:, 2] > 0.05)
        & (source_base_color[:, 1] - source_base_color[:, 2] > 0.02)
    )
    return {
        "generic": metrics,
        "yellow_diagnostic": {
            "selection_metric": False,
            "valid_texels": int(torch.count_nonzero(yellow).cpu()),
            "source_mean_r_minus_b": float(
                torch.mean(source_base_color[yellow, 0] - source_base_color[yellow, 2]).cpu()
            ),
            "predicted_mean_r_minus_b": float(
                torch.mean(prediction[yellow, 0] - prediction[yellow, 2]).cpu()
            ),
        },
    }


def run(config_path: Path) -> dict[str, object]:
    config, config_bytes = _load_mapping(config_path, "color guard audit config")
    if config.get("experiment") != "scifihelmet_c4_affine_color_guard_audit_v1":
        raise ValueError("unsupported color guard audit experiment")
    spec = freeze_color_audit_spec(config)
    source = config.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("source section is invalid")
    source_paths = {
        "preflight": _repo_path(source["preflight_config"], "source.preflight_config"),
        "render_pool": _repo_path(
            source["render_pool_config"], "source.render_pool_config"
        ),
        "audit": _repo_path(source["pca_audit_report"], "source.pca_audit_report"),
        "parent": _repo_path(source["parent_manifest"], "source.parent_manifest"),
    }
    source_values: dict[str, dict[str, object]] = {}
    source_bytes: dict[str, bytes] = {}
    for name, path in source_paths.items():
        source_values[name], source_bytes[name] = _load_mapping(path, name)
    expected_hashes = {
        "preflight": str(source["preflight_config_sha256"]),
        "render_pool": str(source["render_pool_config_sha256"]),
        "audit": str(source["pca_audit_report_sha256"]),
        "parent": str(source["parent_manifest_sha256"]),
    }
    actual_hashes = {name: _sha256_bytes(payload) for name, payload in source_bytes.items()}
    if actual_hashes != expected_hashes:
        raise ValueError("color audit source SHA-256 mismatch")
    parent_manifest = source_values["parent"]
    parent_hash = str(source["parent_p0_hash"])
    if certified_parent_hash(parent_manifest) != parent_hash:
        raise ValueError("certified parent artifact hash mismatch")
    coverage = freeze_render_coverage(source_values["render_pool"])
    if coverage.camera_count != 31 or coverage.light_count != 6:
        raise ValueError("color audit requires the frozen camera31/light6 pool")
    if coverage.resolution != (256, 256):
        raise ValueError("color audit requires the frozen 256x256 render pool")

    output_root = _repo_path(config["output_root"], "output_root")
    if output_root.exists():
        raise FileExistsError(f"refusing to inherit color audit output: {output_root}")

    preflight = source_values["preflight"]
    render_pool = source_values["render_pool"]
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
    partition_cpu = build_color_quantile_partition(
        source_base_color_cpu, quantiles=spec.quantiles
    )
    metric_pairs_cpu = freeze_color_metric_pairs(
        partition_cpu,
        seed=int(preflight["seed"]) + spec.metric_seed_offset,
        pairs_per_bin_pair=spec.pairs_per_bin_pair,
    )
    partition_manifest = _partition_manifest(partition_cpu, metric_pairs_cpu)
    gate_policy = freeze_balanced_gate_policy()

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
        raise RuntimeError("color guard gradient audit requires CUDA")

    output_root.mkdir(parents=True)
    _write_new(output_root / "partition_manifest.json", _json_bytes(partition_manifest))
    metric_pairs_payload = _torch_payload(
        {
            "schema_version": 1,
            "partition_hash": partition_cpu.partition_hash,
            "pair_hash": metric_pairs_cpu.pair_hash,
            "left_valid_positions": metric_pairs_cpu.left_valid_positions,
            "right_valid_positions": metric_pairs_cpu.right_valid_positions,
            "left_logical_bin_ids": metric_pairs_cpu.left_logical_bin_ids,
            "right_logical_bin_ids": metric_pairs_cpu.right_logical_bin_ids,
        }
    )
    _write_new(output_root / "metric_pairs.pt", metric_pairs_payload)
    _write_new(output_root / "gate_policy.json", _json_bytes(gate_policy))

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
    valid_cuda = valid_mask.to(device)
    charts_cuda = chart_ids.to(device)
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

    render_config = render_pool["render"]
    cameras = [
        _orbit_camera_from_spec(value, render_config)
        for value in render_pool["train_cameras"]
    ]
    geometries = [
        render_geometry_gbuffer(
            mesh, camera, tuple(render_config["resolution"]), device=device
        )
        for camera in cameras
    ]
    textures = load_core4_textures(core4_manifest, device)
    lights = [_light(value) for value in render_pool["train_lights"]]
    minimum_roughness = float(render_config["minimum_roughness"])
    reference_grid = [
        [
            shade_ggx(
                geometry,
                camera,
                light,
                material_override=sample_core4_material(geometry, textures),
                minimum_roughness=minimum_roughness,
            ).detach()
            for geometry, camera in zip(geometries, cameras)
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
    base_objective = PreflightObjective(
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
    )
    config_hash = _sha256_bytes(config_bytes)
    input_hash = hashlib.sha256(
        (
            _sha256_file(gltf_path)
            + _sha256_file(core4_manifest)
            + parent_hash
            + actual_hashes["render_pool"]
            + partition.partition_hash
        ).encode("ascii")
    ).hexdigest()
    candidate_kwargs = {
        "core_seed": seed + 11,
        "color_seed": seed + spec.color_seed_offset,
        "color_partition_hash": partition.partition_hash,
        "config_hash": config_hash,
        "input_hash": input_hash,
        "latent_learning_rate": float(preflight["training"]["latent_learning_rate"]),
        "affine_learning_rate": float(preflight["training"]["affine_learning_rate"]),
    }
    batch_source = create_color_candidates(parent, **candidate_kwargs)["C0"]
    batches = tuple(
        draw_training_batch(
            batch_source,
            texel_count=valid_indices.numel(),
            batch_size=spec.material_batch_size,
            cube_sample_count=0,
            color_partition=partition,
            color_batch_size=spec.color_batch_size,
        )
        for _ in range(spec.batch_count)
    )
    batches_payload = _torch_payload(
        [
            {
                "core_indices": batch.core_indices.detach().cpu(),
                "color_indices": batch.color_indices.detach().cpu(),
                "color_bin_ids": batch.color_bin_ids.detach().cpu(),
            }
            for batch in batches
        ]
    )
    _write_new(output_root / "audit_batches.pt", batches_payload)

    audit_state = create_color_candidates(parent, **candidate_kwargs)["C0"]
    color_objective = ColorGuardObjective(
        base_objective,
        valid_flat_indices=valid_indices,
        source_base_color=source_base_color,
        opponent_lambda=0.0,
        pair_lambda=0.0,
        epsilon=spec.charbonnier_epsilon,
    )
    checkpoint_before = checkpoint_candidate(audit_state)

    def objective_terms(batch: object) -> dict[str, torch.Tensor]:
        base, terms = base_objective(audit_state, batch)
        color = color_objective.color_terms(audit_state, batch)
        return {
            "base": base,
            "render": terms["helmet_charbonnier"]
            * float(loss_config["helmet_charbonnier"])
            + terms["helmet_log1p"] * float(loss_config["helmet_log1p"]),
            "rgb": terms["base_color_l1"] * float(loss_config["base_color_l1"]),
            "opponent": color["opponent"],
            "pair": color["pair"],
            "normal": terms["normal_cosine"] * float(loss_config["normal_cosine"]),
            "roughness": terms["roughness_l1"] * float(loss_config["roughness_l1"]),
            "metallic": terms["metallic_l1"] * float(loss_config["metallic_l1"]),
        }

    gradient_audit = audit_gradient_objectives(
        batches=batches,
        objective_terms=objective_terms,
        parameter_groups={
            "latent": (audit_state.latent,),
            "affine": tuple(audit_state.decoder.parameters()),
        },
        epsilon=spec.gradient_epsilon,
    )
    calibration = calibrate_static_color_budgets(
        gradient_audit, ratios=spec.ratios, epsilon=spec.gradient_epsilon
    )
    checkpoint_after = checkpoint_candidate(audit_state)
    if checkpoint_before["checkpoint_hash"] != checkpoint_after["checkpoint_hash"]:
        raise RuntimeError("zero-update gradient audit mutated candidate state")
    if audit_state.optimizer_updates != 0:
        raise RuntimeError("zero-update gradient audit performed an optimizer update")
    parent_color = _parent_color_report(
        audit_state,
        source_base_color,
        valid_indices,
        partition,
        metric_pairs,
    )
    report = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "status": "complete_waiting_for_ratio_confirmation",
        "parent_p0_hash": parent_hash,
        "config_sha256": config_hash,
        "input_hash": input_hash,
        "source_hashes": actual_hashes,
        "render_coverage": {
            "camera_count": coverage.camera_count,
            "light_count": coverage.light_count,
            "resolution": list(coverage.resolution),
            "world_space_lights": True,
        },
        "partition_manifest": partition_manifest,
        "metric_pairs_file_sha256": _sha256_bytes(metric_pairs_payload),
        "audit_batches_file_sha256": _sha256_bytes(batches_payload),
        "gate_policy_sha256": _sha256_bytes(_json_bytes(gate_policy)),
        "candidate_checkpoint_hash_before": checkpoint_before["checkpoint_hash"],
        "candidate_checkpoint_hash_after": checkpoint_after["checkpoint_hash"],
        "optimizer_updates_before": 0,
        "optimizer_updates_after": audit_state.optimizer_updates,
        "parent_color_metrics": parent_color,
        "gradient_audit": gradient_audit,
        "calibration": calibration,
        "selected_ratio": None,
        "formal_holdout_accessed": False,
        "training_started": False,
        "ue_started": False,
    }
    _write_new(output_root / "gradient_audit.json", _json_bytes(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/train/scifihelmet_c4_affine_color_guard_audit_v1.yaml",
    )
    arguments = parser.parse_args()
    report = run(arguments.config.resolve())
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
