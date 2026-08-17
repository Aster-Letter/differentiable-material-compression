"""Run the gated chroma8 L0 camera-relative-light diagnostic to 1k/5k."""

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

from cg_frontier.assets.gltf_mesh import load_gltf_mesh  # noqa: E402
from cg_frontier.compression.affine_material import (  # noqa: E402
    AFFINE_STATIC_COST,
    certify_affine,
)
from cg_frontier.compression.affine_pca import (  # noqa: E402
    EnhancedPCASpec,
    export_p0_enhanced_bundle,
    rasterize_uv_charts,
)
from cg_frontier.compression.affine_regularizers import (  # noqa: E402
    fake_quantize_unorm8,
)
from cg_frontier.compression.affine_training import (  # noqa: E402
    TrainingObservationPlan,
    begin_candidate_continuation,
    checkpoint_candidate,
    create_paired_candidates,
    resume_candidate,
    run_candidate_training,
    plan_training_observations,
    time_candidate_steps,
)
from cg_frontier.compression.material import (  # noqa: E402
    Core4Targets,
    load_core4_targets,
)
from cg_frontier.compression.render_loss import (  # noqa: E402
    bilinear_sample_top_down_wrap,
    decoded_to_material,
    masked_render_metrics,
)
from cg_frontier.render.camera_relative_lighting import (  # noqa: E402
    build_camera_relative_light_grid,
    parse_camera_relative_light_families,
)
from cg_frontier.render.gbuffer import (  # noqa: E402
    load_core4_textures,
    render_geometry_gbuffer,
    sample_core4_material,
)
from cg_frontier.render.pbr import shade_ggx  # noqa: E402
from run_scifihelmet_c4_affine_40k import (  # noqa: E402
    _append_json_line,
    _export_endpoint,
    _line_count,
    _load_mapping,
    _read_json,
)
from run_scifihelmet_c4_affine_chroma8_l0_40k import (  # noqa: E402
    _mean_render,
    _orbit_camera_from_spec,
    _region_metrics,
    freeze_lightrel_decision,
    freeze_render_coverage,
)
from run_scifihelmet_c4_affine_preflight import (  # noqa: E402
    PreflightObjective,
    _json_bytes,
    _move_p0,
    _repo_path,
    _sha256_file,
    _targets_to_seven,
    _write_new,
)


DEFAULT_CONFIG = (
    ROOT / "configs/train/scifihelmet_c4_affine_chroma8_l0_lightrel_5k.yaml"
)


@dataclass(frozen=True)
class FrozenLightrel40kContinuation:
    candidate_id: str
    source_step: int
    source_checkpoint_hash: str
    parent_p0_hash: str
    parent_manifest_hash: str
    source_audit_hash: str
    render_pool_hash: str
    observation_plan: TrainingObservationPlan


def freeze_lightrel_40k_continuation(
    config: Mapping[str, object],
) -> FrozenLightrel40kContinuation:
    """Freeze the user-authorized camera-relative L0 1k→40k continuation."""

    authorization = config.get("authorization")
    source = config.get("source")
    training = config.get("training")
    if not all(isinstance(value, Mapping) for value in (authorization, source, training)):
        raise ValueError("lightrel continuation config sections are invalid")
    if (
        authorization.get("authorized_lightrel_l0_40k") is not True
        or authorization.get("candidate_id") != "L0"
        or authorization.get("retain_p0_raw_gap_report") is not True
    ):
        raise ValueError("only the authorized lightrel L0 40k continuation is allowed")
    source_step = int(source.get("source_step", -1))
    total_steps = int(training.get("total_steps", -1))
    checkpoint_steps = tuple(int(value) for value in training.get("checkpoint_steps", ()))
    expected_checkpoints = tuple(range(5_000, 40_001, 5_000))
    if source_step != 1_000 or total_steps != 40_000:
        raise ValueError("lightrel L0 continuation must be exactly 1k→40k")
    if checkpoint_steps != expected_checkpoints:
        raise ValueError("lightrel L0 40k checkpoint schedule mismatch")
    observation_plan = plan_training_observations(
        total_steps=total_steps,
        checkpoint_steps=checkpoint_steps,
        trend_interval=int(training.get("trend_interval", 0)),
    )
    observation_plan = TrainingObservationPlan(
        total_steps=observation_plan.total_steps,
        checkpoint_steps=observation_plan.checkpoint_steps,
        trend_steps=tuple(
            step for step in observation_plan.trend_steps if step > source_step
        ),
    )
    return FrozenLightrel40kContinuation(
        candidate_id="L0",
        source_step=source_step,
        source_checkpoint_hash=str(source.get("source_checkpoint_hash", "")),
        parent_p0_hash=str(source.get("parent_p0_hash", "")),
        parent_manifest_hash=str(source.get("parent_manifest_sha256", "")),
        source_audit_hash=str(source.get("pca_audit_report_sha256", "")),
        render_pool_hash=str(source.get("render_pool_config_sha256", "")),
        observation_plan=observation_plan,
    )


def evaluate_lightrel_training_gate(
    parent: Mapping[str, object],
    candidate: Mapping[str, object],
    thresholds: Mapping[str, object],
) -> dict[str, object]:
    """Decide whether the 1k endpoint may exact-resume to 5k."""

    chroma_fraction = float(candidate["chroma_contrast_retention"]) / float(
        parent["chroma_contrast_retention"]
    )
    yellow_fraction = float(candidate["yellow_mean_r_minus_b"]) / float(
        parent["yellow_mean_r_minus_b"]
    )
    seven_multiplier = float(candidate["seven_channel_mae"]) / float(
        parent["seven_channel_mae"]
    )
    hdr_multiplier = float(candidate["masked_linear_hdr_mae"]) / float(
        parent["masked_linear_hdr_mae"]
    )
    gates = {
        "generic_chroma_retention": chroma_fraction
        >= float(thresholds["generic_chroma_retention_fraction_of_parent_min"]),
        "yellow_r_minus_b": yellow_fraction
        >= float(thresholds["yellow_r_minus_b_fraction_of_parent_min"]),
        "seven_channel_mae": seven_multiplier
        <= float(thresholds["seven_channel_mae_multiplier_max"]),
        "masked_linear_hdr_mae": hdr_multiplier
        <= float(thresholds["masked_linear_hdr_mae_multiplier_max"]),
        "certificate_valid": (
            bool(candidate["certificate_valid"])
            if thresholds.get("certificate_valid_required") is True
            else True
        ),
    }
    return {
        "ratios": {
            "generic_chroma_retention_fraction_of_parent": chroma_fraction,
            "yellow_r_minus_b_fraction_of_parent": yellow_fraction,
            "seven_channel_mae_multiplier": seven_multiplier,
            "masked_linear_hdr_mae_multiplier": hdr_multiplier,
        },
        "gates": gates,
        "passed": all(gates.values()),
    }


def certified_parent_certificate(
    parent_manifest: Mapping[str, object],
) -> Mapping[str, object]:
    """Return the certificate bound into the hashed parent manifest."""

    safe = parent_manifest.get("safe_artifact")
    certificate = safe.get("certificate") if isinstance(safe, Mapping) else None
    if not isinstance(certificate, Mapping) or certificate.get("valid") is not True:
        raise ValueError("certified parent manifest is missing its valid certificate")
    return certificate


@torch.no_grad()
def _endpoint_metrics_light_grid(
    state: object,
    target: Core4Targets,
    valid_indices: torch.Tensor,
    geometries: Sequence[object],
    cameras: Sequence[object],
    reference_grid: Sequence[Sequence[torch.Tensor]],
    light_grid: Sequence[Sequence[object]],
    minimum_roughness: float,
) -> dict[str, object]:
    deployed = fake_quantize_unorm8(state.latent.detach())
    decoded = state.decoder(deployed.reshape(-1, 4)[valid_indices])
    selected = target.select(valid_indices)
    material = {
        "base_color_l1": float(
            F.l1_loss(decoded.base_color_linear, selected.base_color_linear).cpu()
        ),
        "normal_cosine": float(
            torch.mean(
                1.0 - torch.sum(decoded.normal_xyz * selected.normal_xyz, dim=-1)
            ).cpu()
        ),
        "roughness_l1": float(
            F.l1_loss(decoded.roughness, selected.roughness).cpu()
        ),
        "metallic_l1": float(F.l1_loss(decoded.metallic, selected.metallic).cpu()),
    }
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
    material["seven_channel_mae"] = float(
        torch.mean(torch.abs(predicted_seven - source_seven)).cpu()
    )
    renders: list[dict[str, object]] = []
    for camera_index, (geometry, camera) in enumerate(
        zip(geometries, cameras, strict=True)
    ):
        sampled = bilinear_sample_top_down_wrap(
            deployed, geometry.torch_buffers["uv"]
        )
        candidate_material = decoded_to_material(geometry, state.decoder(sampled))
        for light_index, (references, lights) in enumerate(
            zip(reference_grid, light_grid, strict=True)
        ):
            candidate_hdr = shade_ggx(
                geometry,
                camera,
                lights[camera_index],
                material_override=candidate_material,
                minimum_roughness=minimum_roughness,
            )
            renders.append(
                {
                    **masked_render_metrics(
                        references[camera_index],
                        candidate_hdr,
                        geometry.torch_buffers["mask"],
                        linear_psnr_data_range=2.0,
                        display_exposure=1.5,
                    ),
                    "camera_index": camera_index,
                    "light_index": light_index,
                }
            )
    return {"material": material, "helmet_render": renders}


def _gate_values(
    metrics: Mapping[str, object],
    regions: Mapping[str, object],
    certificate: Mapping[str, object],
) -> dict[str, object]:
    material = metrics["material"]
    generic = regions["generic"]
    yellow = regions["yellow_diagnostic"]
    if not all(isinstance(value, Mapping) for value in (material, generic, yellow)):
        raise ValueError("lightrel endpoint metrics are incomplete")
    return {
        "chroma_contrast_retention": float(generic["chroma_contrast_retention"]),
        "yellow_mean_r_minus_b": float(yellow["predicted_mean_r_minus_b"]),
        "seven_channel_mae": float(material["seven_channel_mae"]),
        "masked_linear_hdr_mae": _mean_render(metrics, "masked_linear_hdr_mae"),
        "certificate_valid": certificate.get("valid") is True,
    }


def _comparison(
    *,
    step: int,
    parent_metrics: Mapping[str, object],
    endpoint_metrics: Mapping[str, object],
    parent_regions: Mapping[str, object],
    endpoint_regions: Mapping[str, object],
) -> dict[str, object]:
    parent_certificate = {"valid": True}
    endpoint_certificate = {"valid": True}
    return {
        "source_step": 0,
        "final_step": step,
        "parent": _gate_values(parent_metrics, parent_regions, parent_certificate),
        "endpoint": _gate_values(
            endpoint_metrics, endpoint_regions, endpoint_certificate
        ),
    }


def _write_summary(
    output_root: Path,
    *,
    status: str,
    decision: object,
    coverage: object,
    config_hash: str,
    audit_hash: str,
    raw_gap: Mapping[str, object],
    cuda_preflight: Mapping[str, object],
    parent_metrics: Mapping[str, object],
    parent_regions: Mapping[str, object],
    step_reports: Mapping[str, object],
    exact_resume: Mapping[str, object] | None,
) -> dict[str, object]:
    report = {
        "schema_version": 1,
        "experiment": "scifihelmet_c4_affine_chroma8_l0_lightrel_5k",
        "status": status,
        "config_sha256": config_hash,
        "source_lightrel_audit_sha256": audit_hash,
        "parent_p0_hash": decision.parent_p0_hash,
        "render_coverage": asdict(coverage),
        "cuda_preflight": dict(cuda_preflight),
        "p0_raw_gap_report": dict(raw_gap),
        "parent_metrics": dict(parent_metrics),
        "parent_regions": dict(parent_regions),
        "step_reports": dict(step_reports),
        "exact_resume": dict(exact_resume) if exact_resume is not None else None,
        "formal_holdout_accessed": False,
        "long_training_authorized": False,
        "static_cost": dict(AFFINE_STATIC_COST),
    }
    _write_new(output_root / "training_report.json", _json_bytes(report))
    return report


def run(config_path: Path) -> dict[str, object]:
    config, config_bytes = _load_mapping(config_path, "lightrel L0 config")
    experiment = config.get("experiment")
    continuation_40k = (
        experiment == "scifihelmet_c4_affine_chroma8_l0_lightrel_40k_continuation"
    )
    if experiment not in {
        "scifihelmet_c4_affine_chroma8_l0_lightrel_5k",
        "scifihelmet_c4_affine_chroma8_l0_lightrel_40k_continuation",
    }:
        raise ValueError("unsupported lightrel experiment")
    source = config.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("lightrel source section is invalid")
    parent_manifest_path = _repo_path(source["parent_manifest"], "source.parent_manifest")
    audit_path = _repo_path(source["pca_audit_report"], "source.pca_audit_report")
    preflight_path = _repo_path(source["preflight_config"], "source.preflight_config")
    render_pool_path = _repo_path(
        source["render_pool_config"], "source.render_pool_config"
    )
    parent_manifest, parent_bytes = _load_mapping(parent_manifest_path, "parent manifest")
    pca_audit, audit_bytes = _load_mapping(audit_path, "PCA audit")
    preflight, preflight_bytes = _load_mapping(preflight_path, "preflight config")
    render_pool, render_pool_bytes = _load_mapping(render_pool_path, "render pool")
    expected_hashes = {
        "parent": str(source["parent_manifest_sha256"]),
        "audit": str(source["pca_audit_report_sha256"]),
        "preflight": str(source["preflight_config_sha256"]),
        "render_pool": str(source["render_pool_config_sha256"]),
    }
    actual_hashes = {
        "parent": hashlib.sha256(parent_bytes).hexdigest(),
        "audit": hashlib.sha256(audit_bytes).hexdigest(),
        "preflight": hashlib.sha256(preflight_bytes).hexdigest(),
        "render_pool": hashlib.sha256(render_pool_bytes).hexdigest(),
    }
    if actual_hashes != expected_hashes:
        raise ValueError("lightrel source SHA-256 mismatch")
    decision = (
        freeze_lightrel_40k_continuation(config)
        if continuation_40k
        else freeze_lightrel_decision(config, parent_manifest)
    )
    coverage = freeze_render_coverage(render_pool)
    enhanced = pca_audit.get("enhanced_global_q4")
    raw_gap = enhanced.get("chroma8") if isinstance(enhanced, Mapping) else None
    if not isinstance(raw_gap, Mapping):
        raise ValueError("chroma8 raw/safe gap evidence is missing")
    audit_root = _repo_path(config["audit_output_root"], "audit_output_root")
    lightrel_audit_path = audit_root / "audit_report.json"
    lightrel_audit = _read_json(lightrel_audit_path)
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    audit_config_hash = (
        str(source.get("source_training_config_sha256", ""))
        if continuation_40k
        else config_hash
    )
    if (
        lightrel_audit.get("config_sha256") != audit_config_hash
        or lightrel_audit.get("evaluation", {}).get("gates_passed") is not True
        or int(lightrel_audit.get("optimizer_updates", -1)) != 0
    ):
        raise ValueError("zero-update lightrel audit is missing or failed")
    lightrel_audit_hash = _sha256_file(lightrel_audit_path)

    output_root = _repo_path(config["output_root"], "output_root")
    output_root.mkdir(parents=True, exist_ok=False)
    _write_new(
        output_root / "decision.json",
        _json_bytes(
            {
                **asdict(decision),
                "render_coverage": asdict(coverage),
                "source_lightrel_audit_sha256": lightrel_audit_hash,
                "long_training_authorized": continuation_40k,
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
        raise RuntimeError("lightrel L0 diagnostic requires CUDA")

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
    render = render_pool["render"]
    camera_specs = render_pool["train_cameras"]
    cameras = [_orbit_camera_from_spec(value, render) for value in camera_specs]
    geometries = [
        render_geometry_gbuffer(mesh, camera, tuple(render["resolution"]), device=device)
        for camera in cameras
    ]
    families = parse_camera_relative_light_families(config["camera_relative_lights"])
    light_grid = build_camera_relative_light_grid(cameras, families)
    textures = load_core4_textures(core4_manifest, device)
    minimum_roughness = float(render["minimum_roughness"])
    reference_grid = [
        [
            shade_ggx(
                geometry,
                camera,
                light_grid[light_index][camera_index],
                material_override=sample_core4_material(geometry, textures),
                minimum_roughness=minimum_roughness,
            ).detach()
            for camera_index, (geometry, camera) in enumerate(
                zip(geometries, cameras, strict=True)
            )
        ]
        for light_index in range(len(families))
    ]
    loss_config = dict(preflight["loss"])
    loss_config["minimum_roughness"] = minimum_roughness
    training = preflight["training"]
    formal = config["training"]
    input_hash = hashlib.sha256(
        (
            _sha256_file(gltf_path)
            + _sha256_file(core4_manifest)
            + decision.parent_p0_hash
            + decision.render_pool_hash
            + lightrel_audit_hash
        ).encode("ascii")
    ).hexdigest()
    source_checkpoint: dict[str, object] | None = None
    source_report: Mapping[str, object] | None = None
    if continuation_40k:
        source_config_path = _repo_path(
            source["source_training_config"], "source.source_training_config"
        )
        source_report_path = _repo_path(
            source["source_training_report"], "source.source_training_report"
        )
        source_checkpoint_path = _repo_path(
            source["source_checkpoint"], "source.source_checkpoint"
        )
        source_config_bytes = source_config_path.read_bytes()
        source_report_bytes = source_report_path.read_bytes()
        source_config_hash = hashlib.sha256(source_config_bytes).hexdigest()
        source_report_hash = hashlib.sha256(source_report_bytes).hexdigest()
        if (
            source_config_hash != str(source["source_training_config_sha256"])
            or source_report_hash != str(source["source_training_report_sha256"])
            or _sha256_file(source_checkpoint_path)
            != str(source["source_checkpoint_file_sha256"])
        ):
            raise ValueError("lightrel 1k continuation source SHA-256 mismatch")
        source_report = json.loads(source_report_bytes)
        step_report = source_report.get("step_reports", {}).get("step_001000", {})
        source_manifest = step_report.get("manifest", {})
        if (
            source_report.get("status") != "stopped_at_1k_gate"
            or source_report.get("config_sha256") != source_config_hash
            or source_report.get("parent_p0_hash") != decision.parent_p0_hash
            or source_manifest.get("candidate_id") != "L0"
            or source_manifest.get("objective_id") != "material+helmet"
            or source_manifest.get("optimizer_updates") != decision.source_step
            or source_manifest.get("input_hash") != input_hash
        ):
            raise ValueError("lightrel 1k continuation source lineage mismatch")
        source_checkpoint = torch.load(
            source_checkpoint_path, map_location=device, weights_only=False
        )
        state = resume_candidate(
            source_checkpoint,
            parent,
            expected_parent_p0_hash=decision.parent_p0_hash,
            expected_config_hash=source_config_hash,
            expected_input_hash=input_hash,
        )
        resumed_source_hash = str(checkpoint_candidate(state)["checkpoint_hash"])
        if (
            resumed_source_hash != decision.source_checkpoint_hash
            or source_checkpoint.get("checkpoint_hash")
            != decision.source_checkpoint_hash
        ):
            raise ValueError("lightrel 1k exact-resume fingerprint mismatch")
        begin_candidate_continuation(
            state,
            source_checkpoint=source_checkpoint,
            continuation_config_hash=config_hash,
        )
        parent_metrics = source_report["parent_metrics"]
        parent_regions = source_report["parent_regions"]
    else:
        state = create_paired_candidates(
            parent,
            core_seed=seed + 11,
            cube_seed=seed + 17,
            config_hash=config_hash,
            input_hash=input_hash,
            latent_learning_rate=float(training["latent_learning_rate"]),
            affine_learning_rate=float(training["affine_learning_rate"]),
        )["L0"]
        parent_metrics = _endpoint_metrics_light_grid(
            state,
            targets,
            valid_indices,
            geometries,
            cameras,
            reference_grid,
            light_grid,
            minimum_roughness,
        )
        parent_regions = _region_metrics(state, targets, valid_indices)
    objective = PreflightObjective(
        target=targets,
        valid_indices=valid_indices,
        valid_mask=valid_cuda,
        chart_ids=charts_cuda,
        helmet_geometries=geometries,
        helmet_cameras=cameras,
        helmet_references=reference_grid[0],
        helmet_light=light_grid[0][0],
        helmet_lights=[row[0] for row in light_grid],
        helmet_reference_grid=[list(row) for row in reference_grid],
        helmet_light_grid=light_grid,
        reference_textures=reference_textures,
        cube_resolution=int(preflight["cube"]["resolution"]),
        cube_config=preflight["cube"],
        loss_config=loss_config,
        tv_lambda=0.0,
        cube_lambda=0.0,
    )
    if continuation_40k:
        if source_checkpoint is None:
            raise AssertionError("continuation source checkpoint was not loaded")
        preflight_state = resume_candidate(
            source_checkpoint,
            parent,
            expected_parent_p0_hash=decision.parent_p0_hash,
            expected_config_hash=str(source["source_training_config_sha256"]),
            expected_input_hash=input_hash,
        )
        begin_candidate_continuation(
            preflight_state,
            source_checkpoint=source_checkpoint,
            continuation_config_hash=config_hash,
        )
    else:
        preflight_state = create_paired_candidates(
            parent,
            core_seed=seed + 11,
            cube_seed=seed + 17,
            config_hash=config_hash,
            input_hash=input_hash,
            latent_learning_rate=float(training["latent_learning_rate"]),
            affine_learning_rate=float(training["affine_learning_rate"]),
        )["L0"]
    timing = time_candidate_steps(
        preflight_state,
        objective,
        texel_count=valid_indices.numel(),
        batch_size=int(training["material_batch_size"]),
        cube_sample_count=1,
        warmup_steps=int(formal["preflight_warmup_steps"]),
        measured_steps=int(formal["preflight_measured_steps"]),
    )
    if not all(math.isfinite(value) for value in timing.mean_loss_terms.values()):
        raise RuntimeError("lightrel CUDA preflight produced non-finite losses")
    cuda_preflight = {
        **asdict(timing),
        "parent_p0_hash": decision.parent_p0_hash,
        "source_lightrel_audit_sha256": lightrel_audit_hash,
        "camera_count": coverage.camera_count,
        "light_count": coverage.light_count,
        "resolution": list(coverage.resolution),
    }
    _write_new(output_root / "cuda_preflight.json", _json_bytes(cuda_preflight))
    del preflight_state
    torch.cuda.empty_cache()

    candidate_root = output_root / "candidate"
    curve_path = candidate_root / "curve.jsonl"
    trend_path = candidate_root / "parameter_trends.jsonl"
    progress_interval = int(formal["progress_interval"])

    def report_progress(point: dict[str, object]) -> None:
        _append_json_line(curve_path, point)
        if int(point["step"]) % progress_interval == 0:
            print(
                _json_bytes(
                    {"candidate": "L0", "step": point["step"], "loss": point["loss"]}
                ).decode().rstrip(),
                flush=True,
            )

    def report_trend(point: dict[str, object]) -> None:
        _append_json_line(trend_path, point)

    if continuation_40k:
        source_metrics = _endpoint_metrics_light_grid(
            state,
            targets,
            valid_indices,
            geometries,
            cameras,
            reference_grid,
            light_grid,
            minimum_roughness,
        )
        source_regions = _region_metrics(state, targets, valid_indices)
        pair_counts = [
            [0 for _ in range(len(families))] for _ in range(len(cameras))
        ]

        def record_render_pair(camera_index: int, light_index: int) -> None:
            pair_counts[camera_index][light_index] += 1

        objective.on_render_pair = record_render_pair
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
            batch_size=int(training["material_batch_size"]),
            cube_sample_count=1,
            on_step=report_progress,
            on_trend=report_trend,
        )
        torch.cuda.synchronize(device)
        wall_seconds = time.perf_counter() - started
        actual_updates = decision.observation_plan.total_steps - decision.source_step
        total_pair_samples = sum(sum(row) for row in pair_counts)
        if total_pair_samples != actual_updates:
            raise ValueError("render-pair count does not match optimizer updates")
        flat_pair_counts = [count for row in pair_counts for count in row]
        if not flat_pair_counts or min(flat_pair_counts) == 0:
            raise ValueError("40k continuation did not cover every camera/light pair")
        pair_report = {
            "source_step": decision.source_step,
            "final_step": decision.observation_plan.total_steps,
            "sample_count": total_pair_samples,
            "camera_count": len(cameras),
            "light_count": len(families),
            "camera_names": [str(value["name"]) for value in camera_specs],
            "light_names": [family.name for family in families],
            "counts_camera_by_light": pair_counts,
            "minimum_pair_count": min(flat_pair_counts),
            "maximum_pair_count": max(flat_pair_counts),
            "mean_pair_count": total_pair_samples / len(flat_pair_counts),
            "all_pairs_observed": True,
        }
        _write_new(
            output_root / "render_pair_counts.json", _json_bytes(pair_report)
        )
        final_metrics = _endpoint_metrics_light_grid(
            state,
            targets,
            valid_indices,
            geometries,
            cameras,
            reference_grid,
            light_grid,
            minimum_roughness,
        )
        final_regions = _region_metrics(state, targets, valid_indices)
        weight, bias = state.decoder.fold_affine()
        final_certificate = certify_affine(weight, bias, margin=state.decoder.margin)
        final_checkpoint_path = (
            output_root / "checkpoints/L0/endpoints/step-040000/checkpoint.pt"
        )
        final_checkpoint = torch.load(
            final_checkpoint_path, map_location="cpu", weights_only=False
        )
        final_checkpoint_hash = str(final_checkpoint["checkpoint_hash"])
        final_endpoint = _export_endpoint(
            output_root, state, final_metrics, final_checkpoint_hash
        )
        _write_new(
            candidate_root / "parameter_trends.json",
            _json_bytes(
                [
                    json.loads(line)
                    for line in trend_path.read_text(encoding="utf-8").splitlines()
                ]
            ),
        )
        source_values = _gate_values(
            source_metrics, source_regions, {"valid": True}
        )
        final_values = _gate_values(
            final_metrics, final_regions, final_certificate
        )
        report = {
            "schema_version": 1,
            "experiment": experiment,
            "status": "completed_40k_stop_before_ue_or_further_training",
            "config_sha256": config_hash,
            "source_training_config_sha256": str(
                source["source_training_config_sha256"]
            ),
            "source_training_report_sha256": str(
                source["source_training_report_sha256"]
            ),
            "source_checkpoint_hash": decision.source_checkpoint_hash,
            "exact_resume": {
                "source_step": decision.source_step,
                "source_checkpoint_hash": decision.source_checkpoint_hash,
                "resumed_checkpoint_hash": resumed_source_hash,
                "hash_match": resumed_source_hash == decision.source_checkpoint_hash,
            },
            "parent_p0_hash": decision.parent_p0_hash,
            "render_coverage": asdict(coverage),
            "render_pair_sampling": pair_report,
            "cuda_preflight": cuda_preflight,
            "p0_raw_gap_report": source_report["p0_raw_gap_report"],
            "source_1k": {
                "metrics": source_metrics,
                "regions": source_regions,
                "gate_values": source_values,
            },
            "final_40k": {
                "manifest": run_report.manifest,
                "endpoint": final_endpoint,
                "regions": final_regions,
                "gate_values": final_values,
                "certificate": final_certificate,
                "checkpoint_hash": final_checkpoint_hash,
            },
            "parent_to_40k": _comparison(
                step=40_000,
                parent_metrics=parent_metrics,
                endpoint_metrics=final_metrics,
                parent_regions=parent_regions,
                endpoint_regions=final_regions,
            ),
            "source_1k_to_40k": {
                "source": source_values,
                "final": final_values,
                "ratios": {
                    key: (
                        float(final_values[key]) / float(source_values[key])
                        if isinstance(source_values[key], (int, float))
                        and not isinstance(source_values[key], bool)
                        and float(source_values[key]) != 0.0
                        else None
                    )
                    for key in source_values
                },
            },
            "wall_seconds": wall_seconds,
            "curve_points": _line_count(curve_path),
            "parameter_trend_points": _line_count(trend_path),
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
            "formal_holdout_accessed": False,
            "static_cost": dict(AFFINE_STATIC_COST),
        }
        _write_new(output_root / "training_report.json", _json_bytes(report))
        return report

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    first_phase = run_candidate_training(
        state,
        objective,
        parent,
        output_root=output_root / "checkpoints",
        observation_plan=TrainingObservationPlan(
            total_steps=1_000,
            checkpoint_steps=(1_000,),
            trend_steps=(1_000,),
        ),
        texel_count=valid_indices.numel(),
        batch_size=int(training["material_batch_size"]),
        cube_sample_count=1,
        on_step=report_progress,
        on_trend=report_trend,
    )
    torch.cuda.synchronize(device)
    step_1k_metrics = _endpoint_metrics_light_grid(
        state,
        targets,
        valid_indices,
        geometries,
        cameras,
        reference_grid,
        light_grid,
        minimum_roughness,
    )
    step_1k_regions = _region_metrics(state, targets, valid_indices)
    weight, bias = state.decoder.fold_affine()
    step_1k_certificate = certify_affine(weight, bias, margin=state.decoder.margin)
    checkpoint_path = (
        output_root / "checkpoints/L0/endpoints/step-001000/checkpoint.pt"
    )
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_hash = str(checkpoint["checkpoint_hash"])
    step_1k_endpoint = _export_endpoint(
        output_root, state, step_1k_metrics, checkpoint_hash
    )
    parent_gate_values = _gate_values(
        parent_metrics, parent_regions, certified_parent_certificate(parent_manifest)
    )
    step_1k_gate_values = _gate_values(
        step_1k_metrics, step_1k_regions, step_1k_certificate
    )
    continuation_gate = evaluate_lightrel_training_gate(
        parent_gate_values, step_1k_gate_values, config["continuation_gates"]
    )
    step_1k_report = {
        "manifest": first_phase.manifest,
        "endpoint": step_1k_endpoint,
        "regions": step_1k_regions,
        "comparison": _comparison(
            step=1_000,
            parent_metrics=parent_metrics,
            endpoint_metrics=step_1k_metrics,
            parent_regions=parent_regions,
            endpoint_regions=step_1k_regions,
        ),
        "continuation_gate": continuation_gate,
        "checkpoint_hash": checkpoint_hash,
    }
    _write_new(output_root / "step_1k_report.json", _json_bytes(step_1k_report))
    if continuation_gate["passed"] is not True:
        return _write_summary(
            output_root,
            status="stopped_at_1k_gate",
            decision=decision,
            coverage=coverage,
            config_hash=config_hash,
            audit_hash=lightrel_audit_hash,
            raw_gap=raw_gap,
            cuda_preflight=cuda_preflight,
            parent_metrics=parent_metrics,
            parent_regions=parent_regions,
            step_reports={"step_001000": step_1k_report},
            exact_resume=None,
        )

    resumed = resume_candidate(
        checkpoint,
        parent,
        expected_parent_p0_hash=decision.parent_p0_hash,
        expected_config_hash=config_hash,
        expected_input_hash=input_hash,
    )
    resumed_hash = str(checkpoint_candidate(resumed)["checkpoint_hash"])
    exact_resume = {
        "source_step": 1_000,
        "source_checkpoint_hash": checkpoint_hash,
        "resumed_checkpoint_hash": resumed_hash,
        "hash_match": resumed_hash == checkpoint_hash,
    }
    if not exact_resume["hash_match"]:
        raise ValueError("1k exact-resume fingerprint mismatch")
    second_phase = run_candidate_training(
        resumed,
        objective,
        parent,
        output_root=output_root / "checkpoints",
        observation_plan=TrainingObservationPlan(
            total_steps=5_000,
            checkpoint_steps=(5_000,),
            trend_steps=(2_000, 3_000, 4_000, 5_000),
        ),
        texel_count=valid_indices.numel(),
        batch_size=int(training["material_batch_size"]),
        cube_sample_count=1,
        on_step=report_progress,
        on_trend=report_trend,
    )
    torch.cuda.synchronize(device)
    step_5k_metrics = _endpoint_metrics_light_grid(
        resumed,
        targets,
        valid_indices,
        geometries,
        cameras,
        reference_grid,
        light_grid,
        minimum_roughness,
    )
    step_5k_regions = _region_metrics(resumed, targets, valid_indices)
    final_checkpoint_path = (
        output_root / "checkpoints/L0/endpoints/step-005000/checkpoint.pt"
    )
    final_checkpoint = torch.load(
        final_checkpoint_path, map_location="cpu", weights_only=False
    )
    step_5k_endpoint = _export_endpoint(
        output_root,
        resumed,
        step_5k_metrics,
        str(final_checkpoint["checkpoint_hash"]),
    )
    step_5k_report = {
        "manifest": second_phase.manifest,
        "endpoint": step_5k_endpoint,
        "regions": step_5k_regions,
        "comparison": _comparison(
            step=5_000,
            parent_metrics=parent_metrics,
            endpoint_metrics=step_5k_metrics,
            parent_regions=parent_regions,
            endpoint_regions=step_5k_regions,
        ),
        "checkpoint_hash": str(final_checkpoint["checkpoint_hash"]),
    }
    _write_new(output_root / "step_5k_report.json", _json_bytes(step_5k_report))
    _write_new(
        candidate_root / "parameter_trends.json",
        _json_bytes(
            [
                json.loads(line)
                for line in trend_path.read_text(encoding="utf-8").splitlines()
            ]
        ),
    )
    wall_seconds = time.perf_counter() - started
    final_report = _write_summary(
        output_root,
        status="completed_5k_stop_before_long_training",
        decision=decision,
        coverage=coverage,
        config_hash=config_hash,
        audit_hash=lightrel_audit_hash,
        raw_gap=raw_gap,
        cuda_preflight=cuda_preflight,
        parent_metrics=parent_metrics,
        parent_regions=parent_regions,
        step_reports={
            "step_001000": step_1k_report,
            "step_005000": step_5k_report,
        },
        exact_resume=exact_resume,
    )
    final_report["wall_seconds"] = wall_seconds
    final_report["curve_points"] = _line_count(curve_path)
    final_report["parameter_trend_points"] = _line_count(trend_path)
    final_report["peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
    final_report["peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
    return final_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    report = run(args.config.resolve())
    print(
        _json_bytes(
            {
                "status": report["status"],
                "parent_p0_hash": report["parent_p0_hash"],
                "exact_resume": report["exact_resume"],
            }
        ).decode().rstrip()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
