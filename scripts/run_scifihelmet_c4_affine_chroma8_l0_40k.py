"""Train an authorized camera31/light6 L0-only 80k child from certified chroma8."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cg_frontier.compression.affine_training import (
    TrainingObservationPlan,
    create_paired_candidates,
    plan_training_observations,
    resume_candidate,
    run_candidate_training,
    select_render_pair,
    time_candidate_steps,
)
from cg_frontier.assets.gltf_mesh import load_gltf_mesh
from cg_frontier.compression.affine_material import AFFINE_STATIC_COST
from cg_frontier.compression.affine_pca import (
    EnhancedPCASpec,
    export_p0_enhanced_bundle,
    rasterize_uv_charts,
)
from cg_frontier.compression.affine_pca_audit import material_region_metrics
from cg_frontier.compression.affine_regularizers import fake_quantize_unorm8
from cg_frontier.compression.material import Core4Targets, load_core4_targets
from cg_frontier.compression.render_loss import orbit_camera
from cg_frontier.render.gbuffer import (
    load_core4_textures,
    render_geometry_gbuffer,
    sample_core4_material,
)
from cg_frontier.render.pbr import shade_ggx
from run_scifihelmet_c4_affine_40k import (
    _append_json_line,
    _endpoint_metrics,
    _export_endpoint,
    _line_count,
    _load_mapping,
    _read_json,
)
from run_scifihelmet_c4_affine_preflight import (
    PreflightObjective,
    _camera,
    _json_bytes,
    _light,
    _move_p0,
    _repo_path,
    _sha256_file,
    _targets_to_seven,
    _write_new,
)


CHECKPOINT_STEPS = (1_000, *range(5_000, 80_001, 5_000))


@dataclass(frozen=True)
class FrozenChroma8L0Decision:
    candidate_id: str
    parent_p0_hash: str
    parent_manifest_hash: str
    source_audit_hash: str
    render_pool_hash: str
    observation_plan: TrainingObservationPlan


@dataclass(frozen=True)
class FrozenRenderCoverage:
    resolution: tuple[int, int]
    camera_count: int
    light_count: int
    randomize_lights_online: bool
    camera_selection_strategy: str


def freeze_render_coverage(pool_config: Mapping[str, object]) -> FrozenRenderCoverage:
    """Freeze the audited camera31/light6 coverage used by affine L0."""

    training_pool = pool_config.get("training_pool")
    render = pool_config.get("render")
    audit = pool_config.get("camera_finetune")
    cameras = pool_config.get("train_cameras")
    lights = pool_config.get("train_lights")
    if not (
        isinstance(training_pool, Mapping)
        and isinstance(render, Mapping)
        and isinstance(audit, Mapping)
        and isinstance(cameras, Sequence)
        and isinstance(lights, Sequence)
    ):
        raise ValueError("affine render coverage config is incomplete")
    audit_values = audit.get("audit")
    resolution = tuple(int(value) for value in render.get("resolution", ()))
    camera_names = [str(value.get("name", "")) for value in cameras if isinstance(value, Mapping)]
    light_names = [str(value.get("name", "")) for value in lights if isinstance(value, Mapping)]
    strategy = str(training_pool.get("camera_selection_strategy", ""))
    randomize_lights = training_pool.get("randomize_lights_online") is True
    valid = (
        resolution == (256, 256)
        and len(camera_names) == 31
        and len(set(camera_names)) == 31
        and all(camera_names)
        and len(light_names) == 6
        and len(set(light_names)) == 6
        and all(light_names)
        and strategy == "explicit_audited_pool_v1"
        and randomize_lights
        and isinstance(audit_values, Mapping)
        and int(audit_values.get("selected_camera_count", -1)) == 31
        and int(training_pool.get("camera_limit", 0)) >= 31
        and int(training_pool.get("light_limit", 0)) == 6
    )
    if not valid:
        raise ValueError("affine L0 requires audited camera31/light6 coverage at 256x256")
    return FrozenRenderCoverage(
        resolution=(256, 256),
        camera_count=31,
        light_count=6,
        randomize_lights_online=True,
        camera_selection_strategy=strategy,
    )


def freeze_chroma8_l0_decision(
    config: Mapping[str, object],
    parent_manifest: Mapping[str, object],
) -> FrozenChroma8L0Decision:
    """Fail closed unless this is the authorized chroma8→L0-only 80k run."""

    authorization = config.get("authorization")
    source = config.get("source")
    training = config.get("training")
    if not all(
        isinstance(value, Mapping)
        for value in (authorization, source, training)
    ):
        raise ValueError("chroma8 L0 config sections are invalid")
    if authorization.get("authorized_chroma8_l0_80k") is not True:
        raise ValueError("chroma8 L0 80k is not authorized")
    if authorization.get("retain_p0_raw_gap_report") is not True:
        raise ValueError("P0 raw gap report must be retained")
    candidate_id = str(authorization.get("candidate_id"))
    if candidate_id != "L0":
        raise ValueError("only L0 is authorized")

    pca = parent_manifest.get("pca")
    safe = parent_manifest.get("safe_artifact")
    if not isinstance(pca, Mapping) or not isinstance(safe, Mapping):
        raise ValueError("parent manifest is incomplete")
    expected_pca = (
        parent_manifest.get("pipeline_id") == "scifihelmet_c4_affine_pca_enhanced_v1"
        and pca.get("rank") == 4
        and float(pca.get("chroma_tail_strength", -1.0)) == 7.0
        and float(pca.get("opponent_chroma_weight", -1.0)) == 2.0
        and pca.get("semantic_group_balance") is True
        and int(pca.get("material_cluster_count", -1)) == 0
        and int(pca.get("residual_reweight_iterations", -1)) == 0
        and isinstance(safe.get("certificate"), Mapping)
        and safe["certificate"].get("valid") is True
    )
    parent_p0_hash = str(source.get("parent_p0_hash", ""))
    if not expected_pca or safe.get("artifact_hash") != parent_p0_hash:
        raise ValueError("source is not the authorized certified chroma8 parent")

    total_steps = int(training.get("total_steps", -1))
    checkpoint_steps = tuple(int(v) for v in training.get("checkpoint_steps", ()))
    if total_steps != 80_000 or checkpoint_steps != CHECKPOINT_STEPS:
        raise ValueError("chroma8 L0 requires the frozen 80k checkpoint plan")
    observation_plan = plan_training_observations(
        total_steps=total_steps,
        checkpoint_steps=checkpoint_steps,
        trend_interval=int(training.get("trend_interval", 0)),
    )
    return FrozenChroma8L0Decision(
        candidate_id=candidate_id,
        parent_p0_hash=parent_p0_hash,
        parent_manifest_hash=str(source.get("parent_manifest_sha256", "")),
        source_audit_hash=str(source.get("pca_audit_report_sha256", "")),
        render_pool_hash=str(source.get("render_pool_config_sha256", "")),
        observation_plan=observation_plan,
    )


def freeze_lightrel_decision(
    config: Mapping[str, object],
    parent_manifest: Mapping[str, object],
) -> FrozenChroma8L0Decision:
    """Freeze the short L0 camera-relative-light diagnostic and forbid long training."""

    authorization = config.get("authorization")
    source = config.get("source")
    training = config.get("training")
    if not all(isinstance(value, Mapping) for value in (authorization, source, training)):
        raise ValueError("lightrel config sections are invalid")
    if (
        authorization.get("authorized_lightrel_diagnostic") is not True
        or authorization.get("candidate_id") != "L0"
        or authorization.get("retain_p0_raw_gap_report") is not True
        or authorization.get("long_training_authorized") is not False
    ):
        raise ValueError("only the short L0 lightrel diagnostic is authorized")
    pca = parent_manifest.get("pca")
    safe = parent_manifest.get("safe_artifact")
    parent_p0_hash = str(source.get("parent_p0_hash", ""))
    valid_parent = (
        parent_manifest.get("pipeline_id") == "scifihelmet_c4_affine_pca_enhanced_v1"
        and isinstance(pca, Mapping)
        and pca.get("rank") == 4
        and float(pca.get("chroma_tail_strength", -1.0)) == 7.0
        and float(pca.get("opponent_chroma_weight", -1.0)) == 2.0
        and pca.get("semantic_group_balance") is True
        and int(pca.get("material_cluster_count", -1)) == 0
        and int(pca.get("residual_reweight_iterations", -1)) == 0
        and isinstance(safe, Mapping)
        and safe.get("artifact_hash") == parent_p0_hash
        and isinstance(safe.get("certificate"), Mapping)
        and safe["certificate"].get("valid") is True
    )
    if not valid_parent:
        raise ValueError("source is not the authorized certified chroma8 parent")
    total_steps = int(training.get("total_steps", -1))
    checkpoint_steps = tuple(int(value) for value in training.get("checkpoint_steps", ()))
    if total_steps != 5_000 or checkpoint_steps != (1_000, 5_000):
        raise ValueError("lightrel diagnostic must stop at checkpoints 1k and 5k")
    observation_plan = plan_training_observations(
        total_steps=total_steps,
        checkpoint_steps=checkpoint_steps,
        trend_interval=int(training.get("trend_interval", 0)),
    )
    return FrozenChroma8L0Decision(
        candidate_id="L0",
        parent_p0_hash=parent_p0_hash,
        parent_manifest_hash=str(source.get("parent_manifest_sha256", "")),
        source_audit_hash=str(source.get("pca_audit_report_sha256", "")),
        render_pool_hash=str(source.get("render_pool_config_sha256", "")),
        observation_plan=observation_plan,
    )


def validate_completed_chroma8_l0_report(
    report: dict[str, object],
    decision: FrozenChroma8L0Decision,
) -> dict[str, object]:
    """Validate and return an already complete chroma8 L0 report unchanged."""

    candidate = report.get("candidate")
    manifest = candidate.get("manifest") if isinstance(candidate, Mapping) else None
    valid = (
        report.get("experiment")
        == "scifihelmet_c4_affine_chroma8_l0_camera31_light6_80k"
        and report.get("endpoint") == decision.observation_plan.total_steps
        and report.get("source_pca_audit_report_sha256") == decision.source_audit_hash
        and report.get("source_parent_manifest_sha256")
        == decision.parent_manifest_hash
        and report.get("source_render_pool_config_sha256")
        == decision.render_pool_hash
        and isinstance(report.get("p0_raw_gap_report"), Mapping)
        and isinstance(manifest, Mapping)
        and manifest.get("candidate_id") == "L0"
        and manifest.get("objective_id") == "material+helmet"
        and manifest.get("parent_p0_hash") == decision.parent_p0_hash
        and manifest.get("optimizer_updates") == 80_000
    )
    if not valid:
        raise ValueError("completed chroma8 L0 lineage mismatch")
    return report


@torch.no_grad()
def _region_metrics(
    state: object,
    targets: Core4Targets,
    valid_indices: torch.Tensor,
) -> dict[str, object]:
    deployed = fake_quantize_unorm8(state.latent.detach())
    decoded = state.decoder(deployed.reshape(-1, 4)[valid_indices])
    selected = targets.select(valid_indices)
    prediction = torch.cat(
        (
            decoded.base_color_linear,
            decoded.normal_xy,
            decoded.roughness,
            decoded.metallic,
        ),
        dim=-1,
    )
    target = torch.cat(
        (
            selected.base_color_linear,
            selected.normal_xyz[:, :2],
            selected.roughness,
            selected.metallic,
        ),
        dim=-1,
    )
    yellow = (
        (target[:, 0] > target[:, 1])
        & (target[:, 1] > target[:, 2])
        & (target[:, 0] - target[:, 2] > 0.05)
        & (target[:, 1] - target[:, 2] > 0.02)
    )
    return {
        "generic": material_region_metrics(prediction, target),
        "yellow_diagnostic": {
            "selection_metric": False,
            "valid_texels": int(torch.count_nonzero(yellow).cpu()),
            "base_color_mae": float(
                F.l1_loss(prediction[yellow, :3], target[yellow, :3]).cpu()
            ),
            "source_mean_r_minus_b": float(
                torch.mean(target[yellow, 0] - target[yellow, 2]).cpu()
            ),
            "predicted_mean_r_minus_b": float(
                torch.mean(prediction[yellow, 0] - prediction[yellow, 2]).cpu()
            ),
        },
    }


def _mean_render(metrics: Mapping[str, object], name: str) -> float:
    renders = metrics["helmet_render"]
    if not isinstance(renders, list) or not renders:
        raise ValueError("helmet render metrics are incomplete")
    return sum(float(value[name]) for value in renders) / len(renders)


def _orbit_camera_from_spec(
    spec: Mapping[str, object], render: Mapping[str, object]
) -> object:
    target = spec.get("target", render["target"])
    if not isinstance(target, Sequence) or len(target) != 3:
        raise ValueError("camera target must contain three coordinates")
    up = render["up"]
    if not isinstance(up, Sequence) or len(up) != 3:
        raise ValueError("camera up must contain three coordinates")
    return orbit_camera(
        yaw_degrees=float(spec["yaw_degrees"]),
        elevation_degrees=float(spec["elevation_degrees"]),
        radius=float(spec.get("radius", render["camera_radius"])),
        target=tuple(float(value) for value in target),
        up=tuple(float(value) for value in up),
        vertical_fov_degrees=float(render["vertical_fov_degrees"]),
        near=float(render["near"]),
        far=float(render["far"]),
    )


def _endpoint_metrics_for_pool(
    state: object,
    target: Core4Targets,
    valid_indices: torch.Tensor,
    geometries: list[object],
    cameras: list[object],
    reference_grid: list[list[torch.Tensor]],
    lights: list[object],
    minimum_roughness: float,
) -> dict[str, object]:
    material: object | None = None
    renders: list[dict[str, object]] = []
    for light_index, (light, references) in enumerate(zip(lights, reference_grid)):
        metrics = _endpoint_metrics(
            state,
            target,
            valid_indices,
            geometries,
            cameras,
            references,
            light,
            minimum_roughness,
        )
        material = metrics["material"]
        for camera_index, value in enumerate(metrics["helmet_render"]):
            renders.append(
                {
                    **value,
                    "camera_index": camera_index,
                    "light_index": light_index,
                }
            )
    if material is None:
        raise ValueError("render coverage pool is empty")
    return {"material": material, "helmet_render": renders}


def _comparison(
    parent_metrics: Mapping[str, object],
    final_metrics: Mapping[str, object],
    parent_regions: Mapping[str, object],
    final_regions: Mapping[str, object],
) -> dict[str, object]:
    parent_material = parent_metrics["material"]
    final_material = final_metrics["material"]
    parent_generic = parent_regions["generic"]
    final_generic = final_regions["generic"]
    parent_yellow = parent_regions["yellow_diagnostic"]
    final_yellow = final_regions["yellow_diagnostic"]
    if not all(
        isinstance(value, Mapping)
        for value in (
            parent_material,
            final_material,
            parent_generic,
            final_generic,
            parent_yellow,
            final_yellow,
        )
    ):
        raise ValueError("parent/final comparison metrics are incomplete")
    return {
        "source_step": 0,
        "final_step": 80_000,
        "seven_channel_mae": {
            "parent": float(parent_material["seven_channel_mae"]),
            "final": float(final_material["seven_channel_mae"]),
        },
        "helmet_render_mean": {
            name: {
                "parent": _mean_render(parent_metrics, name),
                "final": _mean_render(final_metrics, name),
            }
            for name in ("masked_linear_hdr_mae", "display_ssim")
        },
        "chroma_contrast_retention": {
            "parent": float(parent_generic["chroma_contrast_retention"]),
            "final": float(final_generic["chroma_contrast_retention"]),
        },
        "yellow_base_color_mae": {
            "parent": float(parent_yellow["base_color_mae"]),
            "final": float(final_yellow["base_color_mae"]),
        },
        "yellow_mean_r_minus_b": {
            "parent": float(parent_yellow["predicted_mean_r_minus_b"]),
            "final": float(final_yellow["predicted_mean_r_minus_b"]),
            "source": float(final_yellow["source_mean_r_minus_b"]),
        },
    }


def run(config_path: Path, *, resume_existing: bool = False) -> dict[str, object]:
    config, config_bytes = _load_mapping(config_path, "chroma8 L0 config")
    if config.get("experiment") != "scifihelmet_c4_affine_chroma8_l0_camera31_light6_80k":
        raise ValueError("unsupported chroma8 L0 experiment")
    source = config.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("source section is invalid")
    parent_manifest_path = _repo_path(source["parent_manifest"], "source.parent_manifest")
    audit_path = _repo_path(source["pca_audit_report"], "source.pca_audit_report")
    preflight_path = _repo_path(source["preflight_config"], "source.preflight_config")
    render_pool_path = _repo_path(
        source["render_pool_config"], "source.render_pool_config"
    )
    parent_manifest, parent_bytes = _load_mapping(parent_manifest_path, "parent manifest")
    audit, audit_bytes = _load_mapping(audit_path, "PCA audit")
    preflight, preflight_bytes = _load_mapping(preflight_path, "preflight config")
    render_pool, render_pool_bytes = _load_mapping(render_pool_path, "render pool config")
    hashes = {
        "parent": hashlib.sha256(parent_bytes).hexdigest(),
        "audit": hashlib.sha256(audit_bytes).hexdigest(),
        "preflight": hashlib.sha256(preflight_bytes).hexdigest(),
        "render_pool": hashlib.sha256(render_pool_bytes).hexdigest(),
    }
    if hashes["parent"] != str(source["parent_manifest_sha256"]):
        raise ValueError("parent manifest SHA-256 mismatch")
    if hashes["audit"] != str(source["pca_audit_report_sha256"]):
        raise ValueError("PCA audit SHA-256 mismatch")
    if hashes["preflight"] != str(source["preflight_config_sha256"]):
        raise ValueError("preflight config SHA-256 mismatch")
    if hashes["render_pool"] != str(source["render_pool_config_sha256"]):
        raise ValueError("render pool config SHA-256 mismatch")
    decision = freeze_chroma8_l0_decision(config, parent_manifest)
    coverage = freeze_render_coverage(render_pool)
    if decision.render_pool_hash != hashes["render_pool"]:
        raise ValueError("render pool decision hash mismatch")
    enhanced = audit.get("enhanced_global_q4")
    raw_gap = enhanced.get("chroma8") if isinstance(enhanced, Mapping) else None
    if not isinstance(raw_gap, Mapping):
        raise ValueError("chroma8 raw/safe evidence is missing")

    output_root = _repo_path(config["output_root"], "output_root")
    if output_root.exists():
        if not resume_existing:
            raise FileExistsError(f"refusing to inherit chroma8 L0 output: {output_root}")
        existing_decision = _read_json(output_root / "decision.json")
        if existing_decision.get("parent_p0_hash") != decision.parent_p0_hash:
            raise ValueError("existing output parent lineage mismatch")
        if existing_decision.get("render_pool_hash") != decision.render_pool_hash:
            raise ValueError("existing output render-pool lineage mismatch")
        if _read_json(output_root / "p0_raw_gap_report.json") != dict(raw_gap):
            raise ValueError("existing raw gap report mismatch")
        completed_path = output_root / "training_report.json"
        if completed_path.exists():
            return validate_completed_chroma8_l0_report(
                _read_json(completed_path), decision
            )
    else:
        output_root.mkdir(parents=True)
        _write_new(
            output_root / "decision.json",
            _json_bytes(
                {
                    **asdict(decision),
                    "render_coverage": asdict(coverage),
                    "authorized_80k": True,
                    "formal_holdout_accessed": False,
                }
            ),
        )
        _write_new(output_root / "p0_raw_gap_report.json", _json_bytes(raw_gap))

    gltf_path = _repo_path(preflight["inputs"]["gltf"], "inputs.gltf")
    core4_dir = _repo_path(preflight["inputs"]["core4_dir"], "inputs.core4_dir")
    core4_manifest = _repo_path(
        preflight["inputs"]["core4_manifest"], "inputs.core4_manifest"
    )
    mesh = load_gltf_mesh(gltf_path)
    cpu_targets = load_core4_targets(core4_dir, "cpu")
    target_seven = _targets_to_seven(cpu_targets)
    valid_mask, chart_ids = rasterize_uv_charts(
        mesh.texcoords,
        mesh.triangles,
        height=cpu_targets.height,
        width=cpu_targets.width,
    )
    pca = parent_manifest["pca"]
    spec = EnhancedPCASpec(
        chroma_tail_strength=float(pca["chroma_tail_strength"]),
        opponent_chroma_weight=float(pca["opponent_chroma_weight"]),
        semantic_group_balance=bool(pca["semantic_group_balance"]),
    )
    parent_bundle = export_p0_enhanced_bundle(
        target_seven,
        valid_mask,
        chart_ids,
        spec=spec,
        margin=float(preflight["p0"]["safety_margin"]),
    )
    if parent_bundle.manifest != parent_manifest:
        raise ValueError("reconstructed chroma8 parent manifest mismatch")
    if parent_bundle.calibration.safe.artifact_hash != decision.parent_p0_hash:
        raise ValueError("reconstructed chroma8 parent hash mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("chroma8 L0 80k requires CUDA")

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
    valid_indices = torch.nonzero(valid_cuda.reshape(-1), as_tuple=False)[:, 0]
    reference_textures = {
        "base_color": targets.base_color_linear.reshape(targets.height, targets.width, 3),
        "normal": targets.normal_xyz.reshape(targets.height, targets.width, 3),
        "roughness": targets.roughness.reshape(targets.height, targets.width, 1),
        "metallic": targets.metallic.reshape(targets.height, targets.width, 1),
    }
    render_config = render_pool["render"]
    camera_specs = render_pool["train_cameras"]
    light_specs = render_pool["train_lights"]
    cameras = [_orbit_camera_from_spec(value, render_config) for value in camera_specs]
    geometries = [
        render_geometry_gbuffer(
            mesh, camera, tuple(render_config["resolution"]), device=device
        )
        for camera in cameras
    ]
    textures = load_core4_textures(core4_manifest, device)
    lights = [_light(value) for value in light_specs]
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
    loss_config = dict(preflight["loss"])
    loss_config["minimum_roughness"] = minimum_roughness
    training_config = preflight["training"]
    formal_training = config["training"]
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    input_hash = hashlib.sha256(
        (
            _sha256_file(gltf_path)
            + _sha256_file(core4_manifest)
            + decision.parent_p0_hash
            + decision.render_pool_hash
        ).encode("ascii")
    ).hexdigest()
    initial = create_paired_candidates(
        parent,
        core_seed=seed + 11,
        cube_seed=seed + 17,
        config_hash=config_hash,
        input_hash=input_hash,
        latent_learning_rate=float(training_config["latent_learning_rate"]),
        affine_learning_rate=float(training_config["affine_learning_rate"]),
    )["L0"]
    parent_metrics = _endpoint_metrics_for_pool(
        initial,
        targets,
        valid_indices,
        geometries,
        cameras,
        reference_grid,
        lights,
        minimum_roughness,
    )
    parent_regions = _region_metrics(initial, targets, valid_indices)
    state = initial
    endpoint_root = output_root / "checkpoints/L0/endpoints"
    existing = sorted(endpoint_root.glob("step-*/checkpoint.pt")) if resume_existing else []
    if existing:
        checkpoint = torch.load(existing[-1], map_location=device, weights_only=False)
        state = resume_candidate(
            checkpoint,
            parent,
            expected_parent_p0_hash=decision.parent_p0_hash,
            expected_config_hash=config_hash,
            expected_input_hash=input_hash,
        )
        print(
            _json_bytes(
                {"candidate": "L0", "status": "resumed", "step": state.optimizer_updates}
            ).decode().rstrip(),
            flush=True,
        )
    candidate_root = output_root / "candidate"
    curve_path = candidate_root / "curve.jsonl"
    trend_path = candidate_root / "parameter_trends.jsonl"
    if _line_count(curve_path) != state.optimizer_updates:
        raise ValueError("L0 curve length does not match resume step")
    if _line_count(trend_path) != state.optimizer_updates // int(
        formal_training["trend_interval"]
    ):
        raise ValueError("L0 trend length does not match resume step")
    objective = PreflightObjective(
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
    preflight_path = output_root / "cuda_preflight.json"
    if preflight_path.exists():
        cuda_preflight = _read_json(preflight_path)
        if (
            cuda_preflight.get("parent_p0_hash") != decision.parent_p0_hash
            or cuda_preflight.get("render_pool_hash") != decision.render_pool_hash
            or cuda_preflight.get("camera_count") != coverage.camera_count
            or cuda_preflight.get("light_count") != coverage.light_count
        ):
            raise ValueError("existing CUDA preflight lineage mismatch")
    else:
        preflight_state = create_paired_candidates(
            parent,
            core_seed=seed + 11,
            cube_seed=seed + 17,
            config_hash=config_hash,
            input_hash=input_hash,
            latent_learning_rate=float(training_config["latent_learning_rate"]),
            affine_learning_rate=float(training_config["affine_learning_rate"]),
        )["L0"]
        timing = time_candidate_steps(
            preflight_state,
            objective,
            texel_count=valid_indices.numel(),
            batch_size=int(training_config["material_batch_size"]),
            cube_sample_count=1,
            warmup_steps=int(formal_training["preflight_warmup_steps"]),
            measured_steps=int(formal_training["preflight_measured_steps"]),
        )
        if not all(math.isfinite(value) for value in timing.mean_loss_terms.values()):
            raise RuntimeError("CUDA preflight produced non-finite loss terms")
        cuda_preflight = {
            **asdict(timing),
            "parent_p0_hash": decision.parent_p0_hash,
            "render_pool_hash": decision.render_pool_hash,
            "camera_count": coverage.camera_count,
            "light_count": coverage.light_count,
            "resolution": list(coverage.resolution),
        }
        _write_new(preflight_path, _json_bytes(cuda_preflight))
        del preflight_state
        torch.cuda.empty_cache()
    progress_interval = int(formal_training.get("progress_interval", 250))

    def report_progress(point: dict[str, object]) -> None:
        _append_json_line(curve_path, point)
        if int(point["step"]) % progress_interval == 0:
            print(
                _json_bytes(
                    {
                        "candidate": "L0",
                        "step": point["step"],
                        "loss": point["loss"],
                        "phase": point["phase"],
                    }
                ).decode().rstrip(),
                flush=True,
            )

    def report_trend(point: dict[str, object]) -> None:
        _append_json_line(trend_path, point)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    run_report = run_candidate_training(
        state,
        objective,
        parent,
        output_root=output_root / "checkpoints",
        observation_plan=decision.observation_plan,
        texel_count=valid_indices.numel(),
        batch_size=int(training_config["material_batch_size"]),
        cube_sample_count=1,
        on_step=report_progress,
        on_trend=report_trend,
    )
    torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - started
    final_metrics = _endpoint_metrics_for_pool(
        state,
        targets,
        valid_indices,
        geometries,
        cameras,
        reference_grid,
        lights,
        minimum_roughness,
    )
    final_regions = _region_metrics(state, targets, valid_indices)
    all_checkpoint_paths = sorted(endpoint_root.glob("step-*/checkpoint.pt"))
    checkpoint_hashes: dict[str, str] = {}
    for path in all_checkpoint_paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint_hashes[path.parent.name] = str(checkpoint["checkpoint_hash"])
    if tuple(int(name.removeprefix("step-")) for name in checkpoint_hashes) != CHECKPOINT_STEPS:
        raise ValueError("completed checkpoint schedule mismatch")
    final_hash = checkpoint_hashes["step-080000"]
    endpoint = _export_endpoint(
        output_root,
        state,
        final_metrics,
        final_hash,
    )
    _write_new(
        candidate_root / "parameter_trends.json",
        _json_bytes(
            [json.loads(line) for line in trend_path.read_text(encoding="utf-8").splitlines()]
        ),
    )
    candidate_report = {
        "manifest": run_report.manifest,
        "endpoint": endpoint,
        "parent_metrics": parent_metrics,
        "parent_regions": parent_regions,
        "final_regions": final_regions,
        "comparison": _comparison(
            parent_metrics, final_metrics, parent_regions, final_regions
        ),
        "checkpoint_hashes": checkpoint_hashes,
        "curve_points": _line_count(curve_path),
        "parameter_trend_points": _line_count(trend_path),
        "wall_seconds": wall_seconds,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    _write_new(candidate_root / "training_report.json", _json_bytes(candidate_report))
    final_report = {
        "schema_version": 1,
        "experiment": "scifihelmet_c4_affine_chroma8_l0_camera31_light6_80k",
        "endpoint": 80_000,
        "config_sha256": config_hash,
        "source_pca_audit_report_sha256": decision.source_audit_hash,
        "source_parent_manifest_sha256": decision.parent_manifest_hash,
        "source_render_pool_config_sha256": decision.render_pool_hash,
        "parent_p0_hash": decision.parent_p0_hash,
        "render_coverage": asdict(coverage),
        "cuda_preflight": cuda_preflight,
        "p0_raw_gap_report": dict(raw_gap),
        "candidate": candidate_report,
        "formal_holdout_accessed": False,
        "authorized_80k": True,
        "static_cost": dict(AFFINE_STATIC_COST),
    }
    _write_new(output_root / "training_report.json", _json_bytes(final_report))
    return final_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT
        / "configs/train/scifihelmet_c4_affine_chroma8_l0_camera31_light6_80k.yaml",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = run(args.config.resolve(), resume_existing=args.resume)
    print(
        _json_bytes(
            {
                "complete": True,
                "experiment": report["experiment"],
                "endpoint": report["endpoint"],
            }
        ).decode(),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
