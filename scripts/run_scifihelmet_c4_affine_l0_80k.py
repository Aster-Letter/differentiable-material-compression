"""Continue the authorized SciFiHelmet C4-affine L0 from 40k to 80k."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Mapping

import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cg_frontier.compression.affine_training import (
    begin_candidate_continuation,
    resume_candidate,
    run_candidate_training,
    TrainingObservationPlan,
    plan_training_observations,
)
from cg_frontier.assets.gltf_mesh import load_gltf_mesh
from cg_frontier.compression.affine_material import AFFINE_STATIC_COST
from cg_frontier.compression.affine_pca import export_p0_bundle, rasterize_uv_charts
from cg_frontier.compression.material import Core4Targets, load_core4_targets
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


CONTINUATION_CHECKPOINT_STEPS = tuple(range(45_000, 80_001, 5_000))


@dataclass(frozen=True)
class FrozenContinuationDecision:
    candidate_id: str
    source_step: int
    source_report_hash: str
    source_checkpoint_hash: str
    parent_p0_hash: str
    observation_plan: TrainingObservationPlan


def freeze_continuation_decision(
    config: Mapping[str, object],
) -> FrozenContinuationDecision:
    """Fail closed unless the user-authorized L0 40k→80k plan is exact."""

    authorization = config.get("authorization")
    source = config.get("source")
    training = config.get("training")
    if not all(isinstance(value, Mapping) for value in (authorization, source, training)):
        raise ValueError("continuation config sections are invalid")
    if authorization.get("authorized_l0_80k") is not True:
        raise ValueError("L0 80k continuation is not authorized")
    if authorization.get("retain_p0_raw_gap_report") is not True:
        raise ValueError("P0 raw gap report must be retained")
    candidate_id = str(authorization.get("candidate_id"))
    source_step = int(training.get("source_step", -1))
    total_steps = int(training.get("total_steps", -1))
    checkpoint_steps = tuple(int(value) for value in training.get("checkpoint_steps", ()))
    if candidate_id != "L0":
        raise ValueError("only L0 continuation is authorized")
    if source_step != 40_000 or total_steps != 80_000:
        raise ValueError("L0 continuation must be exactly 40k→80k")
    if checkpoint_steps != CONTINUATION_CHECKPOINT_STEPS:
        raise ValueError("L0 continuation checkpoint plan mismatch")
    observation_plan = plan_training_observations(
        total_steps=total_steps,
        checkpoint_steps=checkpoint_steps,
        trend_interval=int(training.get("trend_interval", 0)),
    )
    return FrozenContinuationDecision(
        candidate_id=candidate_id,
        source_step=source_step,
        source_report_hash=str(source.get("common_40k_report_sha256", "")),
        source_checkpoint_hash=str(source.get("checkpoint_hash", "")),
        parent_p0_hash=str(source.get("parent_p0_hash", "")),
        observation_plan=observation_plan,
    )


def validate_source_40k_report(
    report: dict[str, object],
    decision: FrozenContinuationDecision,
) -> dict[str, object]:
    """Validate the immutable common-40k L0 source selected for continuation."""

    candidates = report.get("candidates")
    candidate = candidates.get("L0") if isinstance(candidates, Mapping) else None
    manifest = candidate.get("manifest") if isinstance(candidate, Mapping) else None
    endpoint = candidate.get("endpoint") if isinstance(candidate, Mapping) else None
    raw_gap = report.get("p0_raw_gap_report")
    valid = (
        report.get("experiment") == "scifihelmet_c4_affine_v1_40k"
        and report.get("common_endpoint") == decision.source_step
        and isinstance(raw_gap, Mapping)
        and isinstance(manifest, Mapping)
        and manifest.get("candidate_id") == decision.candidate_id
        and manifest.get("objective_id") == "material+helmet"
        and manifest.get("optimizer_updates") == decision.source_step
        and manifest.get("parent_p0_hash") == decision.parent_p0_hash
        and isinstance(endpoint, Mapping)
        and endpoint.get("checkpoint_hash") == decision.source_checkpoint_hash
    )
    if not valid:
        raise ValueError("common L0@40k source lineage mismatch")
    return report


def validate_completed_continuation_report(
    report: dict[str, object],
    decision: FrozenContinuationDecision,
) -> dict[str, object]:
    """Validate and return an already-complete L0 continuation unchanged."""

    candidate = report.get("candidate")
    manifest = candidate.get("manifest") if isinstance(candidate, Mapping) else None
    continuation = manifest.get("continuation") if isinstance(manifest, Mapping) else None
    valid = (
        report.get("experiment") == "scifihelmet_c4_affine_v1_l0_80k_continuation"
        and report.get("endpoint") == decision.observation_plan.total_steps
        and report.get("source_common_40k_report_sha256") == decision.source_report_hash
        and isinstance(report.get("p0_raw_gap_report"), Mapping)
        and isinstance(manifest, Mapping)
        and manifest.get("candidate_id") == decision.candidate_id
        and manifest.get("objective_id") == "material+helmet"
        and manifest.get("optimizer_updates") == decision.observation_plan.total_steps
        and manifest.get("parent_p0_hash") == decision.parent_p0_hash
        and isinstance(continuation, Mapping)
        and continuation.get("source_checkpoint_hash") == decision.source_checkpoint_hash
        and continuation.get("source_step") == decision.source_step
    )
    if not valid:
        raise ValueError("completed L0 80k continuation lineage mismatch")
    return report


def _mean_render_metric(metrics: Mapping[str, object], name: str) -> float:
    renders = metrics["helmet_render"]
    if not isinstance(renders, list) or not renders:
        raise ValueError("helmet render metrics are incomplete")
    return sum(float(render[name]) for render in renders) / len(renders)


def _endpoint_comparison(
    source_metrics: Mapping[str, object],
    final_metrics: Mapping[str, object],
) -> dict[str, object]:
    source_material = source_metrics["material"]
    final_material = final_metrics["material"]
    if not isinstance(source_material, Mapping) or not isinstance(final_material, Mapping):
        raise ValueError("material metrics are incomplete")
    names = ("masked_linear_hdr_mae", "display_ssim")
    source_render = {name: _mean_render_metric(source_metrics, name) for name in names}
    final_render = {name: _mean_render_metric(final_metrics, name) for name in names}
    source_mae = float(source_material["seven_channel_mae"])
    final_mae = float(final_material["seven_channel_mae"])
    return {
        "source_step": 40_000,
        "final_step": 80_000,
        "seven_channel_mae": {
            "source": source_mae,
            "final": final_mae,
            "absolute_delta": final_mae - source_mae,
            "relative_delta": (final_mae - source_mae) / source_mae,
        },
        "helmet_render_mean": {
            name: {
                "source": source_render[name],
                "final": final_render[name],
                "absolute_delta": final_render[name] - source_render[name],
                "relative_delta": (
                    (final_render[name] - source_render[name]) / source_render[name]
                    if source_render[name] != 0.0
                    else None
                ),
            }
            for name in names
        },
    }


def run(config_path: Path, *, resume_existing: bool = False) -> dict[str, object]:
    config, config_bytes = _load_mapping(config_path, "L0 continuation config")
    if config.get("experiment") != "scifihelmet_c4_affine_v1_l0_80k_continuation":
        raise ValueError("unsupported affine continuation experiment")
    decision = freeze_continuation_decision(config)
    source = config["source"]
    if not isinstance(source, Mapping):
        raise ValueError("source section is invalid")

    common_config_path = _repo_path(source["common_40k_config"], "source.common_40k_config")
    common_report_path = _repo_path(source["common_40k_report"], "source.common_40k_report")
    source_checkpoint_path = _repo_path(source["checkpoint"], "source.checkpoint")
    common_config, common_config_bytes = _load_mapping(common_config_path, "common 40k config")
    common_report, common_report_bytes = _load_mapping(common_report_path, "common 40k report")
    if hashlib.sha256(common_report_bytes).hexdigest() != decision.source_report_hash:
        raise ValueError("common 40k report SHA-256 mismatch")
    validate_source_40k_report(common_report, decision)
    common_config_hash = hashlib.sha256(common_config_bytes).hexdigest()
    source_manifest = common_report["candidates"]["L0"]["manifest"]
    if source_manifest.get("config_hash") != common_config_hash:
        raise ValueError("common 40k config lineage mismatch")
    config_hash = hashlib.sha256(config_bytes).hexdigest()

    output_root = _repo_path(config["output_root"], "output_root")
    if output_root.exists():
        if not resume_existing:
            raise FileExistsError(f"refusing to inherit continuation output: {output_root}")
        existing_decision = _read_json(output_root / "decision.json")
        if (
            existing_decision.get("continuation_config_sha256") != config_hash
            or existing_decision.get("source_report_hash") != decision.source_report_hash
            or existing_decision.get("source_checkpoint_hash") != decision.source_checkpoint_hash
        ):
            raise ValueError("existing continuation output lineage mismatch")
        completed_path = output_root / "training_report.json"
        if completed_path.exists():
            return validate_completed_continuation_report(
                _read_json(completed_path), decision
            )
    else:
        output_root.mkdir(parents=True)
        _write_new(
            output_root / "decision.json",
            _json_bytes(
                {
                    **asdict(decision),
                    "continuation_config_sha256": config_hash,
                    "source_report_hash": decision.source_report_hash,
                    "source_checkpoint_hash": decision.source_checkpoint_hash,
                    "formal_holdout_accessed": False,
                    "l1_or_l2_started": False,
                }
            ),
        )
        _write_new(
            output_root / "p0_raw_gap_report.json",
            _json_bytes(common_report["p0_raw_gap_report"]),
        )

    common_source = common_config["source"]
    preflight_config_path = _repo_path(
        common_source["preflight_config"], "source.preflight_config"
    )
    preflight_config, _ = _load_mapping(preflight_config_path, "preflight config")
    gltf_path = _repo_path(preflight_config["inputs"]["gltf"], "inputs.gltf")
    core4_dir = _repo_path(preflight_config["inputs"]["core4_dir"], "inputs.core4_dir")
    core4_manifest = _repo_path(
        preflight_config["inputs"]["core4_manifest"], "inputs.core4_manifest"
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
    p0_bundle = export_p0_bundle(
        target_seven,
        valid_mask,
        chart_ids,
        margin=float(preflight_config["p0"]["safety_margin"]),
    )
    if p0_bundle.calibration.safe.artifact_hash != decision.parent_p0_hash:
        raise ValueError("reconstructed P0-safe parent hash mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("L0 80k continuation requires CUDA")

    device = torch.device("cuda")
    seed = int(preflight_config["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    p0 = _move_p0(p0_bundle.calibration, device)
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
    render_config = preflight_config["render"]
    cameras = [_camera(value) for value in render_config["cameras"]]
    geometries = [
        render_geometry_gbuffer(mesh, camera, tuple(render_config["resolution"]), device=device)
        for camera in cameras
    ]
    textures = load_core4_textures(core4_manifest, device)
    light = _light(render_config["light"])
    minimum_roughness = float(render_config["minimum_roughness"])
    reference_hdr = [
        shade_ggx(
            geometry,
            camera,
            light,
            material_override=sample_core4_material(geometry, textures),
            minimum_roughness=minimum_roughness,
        ).detach()
        for geometry, camera in zip(geometries, cameras)
    ]
    loss_config = dict(preflight_config["loss"])
    loss_config["minimum_roughness"] = minimum_roughness
    input_hash = hashlib.sha256(
        (_sha256_file(gltf_path) + _sha256_file(core4_manifest)).encode("ascii")
    ).hexdigest()
    if source_manifest.get("input_hash") != input_hash:
        raise ValueError("source input lineage mismatch")

    source_checkpoint = torch.load(
        source_checkpoint_path, map_location=device, weights_only=False
    )
    state = resume_candidate(
        source_checkpoint,
        p0,
        expected_parent_p0_hash=decision.parent_p0_hash,
        expected_config_hash=common_config_hash,
        expected_input_hash=input_hash,
    )
    if (
        source_checkpoint.get("checkpoint_hash") != decision.source_checkpoint_hash
        or state.candidate_id != "L0"
        or state.objective_id != "material+helmet"
        or state.optimizer_updates != decision.source_step
    ):
        raise ValueError("source L0 checkpoint lineage mismatch")
    begin_candidate_continuation(
        state,
        source_checkpoint=source_checkpoint,
        continuation_config_hash=config_hash,
    )

    endpoint_root = output_root / "checkpoints" / "L0" / "endpoints"
    existing_checkpoints = sorted(endpoint_root.glob("step-*/checkpoint.pt")) if resume_existing else []
    if existing_checkpoints:
        state = resume_candidate(
            torch.load(existing_checkpoints[-1], map_location=device, weights_only=False),
            p0,
            expected_parent_p0_hash=decision.parent_p0_hash,
            expected_config_hash=config_hash,
            expected_input_hash=input_hash,
        )
        if (
            state.continuation_from_checkpoint_hash != decision.source_checkpoint_hash
            or state.continuation_from_step != decision.source_step
        ):
            raise ValueError("resumed continuation source lineage mismatch")
        print(
            _json_bytes({"candidate": "L0", "status": "resumed", "step": state.optimizer_updates}).decode().rstrip(),
            flush=True,
        )

    candidate_root = output_root / "candidates" / "L0"
    curve_path = candidate_root / "curve.jsonl"
    trend_path = candidate_root / "parameter_trends.jsonl"
    expected_curve_points = state.optimizer_updates - decision.source_step
    expected_trend_points = expected_curve_points // int(config["training"]["trend_interval"])
    if _line_count(curve_path) != expected_curve_points:
        raise ValueError("L0 continuation curve length does not match resume step")
    if _line_count(trend_path) != expected_trend_points:
        raise ValueError("L0 continuation trend length does not match resume step")

    objective = PreflightObjective(
        target=targets,
        valid_indices=valid_indices,
        valid_mask=valid_cuda,
        chart_ids=charts_cuda,
        helmet_geometries=geometries,
        helmet_cameras=cameras,
        helmet_references=reference_hdr,
        helmet_light=light,
        reference_textures=reference_textures,
        cube_resolution=int(preflight_config["cube"]["resolution"]),
        cube_config=preflight_config["cube"],
        loss_config=loss_config,
        tv_lambda=0.0,
        cube_lambda=0.0,
    )
    progress_interval = int(config["training"].get("progress_interval", 250))

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

    def report_trend(trend: dict[str, object]) -> None:
        _append_json_line(trend_path, trend)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    run_report = run_candidate_training(
        state,
        objective,
        p0,
        output_root=output_root / "checkpoints",
        observation_plan=decision.observation_plan,
        texel_count=int(valid_indices.numel()),
        batch_size=int(preflight_config["training"]["material_batch_size"]),
        cube_sample_count=0,
        on_step=report_progress,
        on_trend=report_trend,
    )
    torch.cuda.synchronize(device)
    wall_seconds = time.perf_counter() - started
    final_metrics = _endpoint_metrics(
        state,
        targets,
        valid_indices,
        geometries,
        cameras,
        reference_hdr,
        light,
        minimum_roughness,
    )
    final_checkpoint_path = endpoint_root / "step-080000" / "checkpoint.pt"
    final_checkpoint = torch.load(final_checkpoint_path, map_location="cpu", weights_only=False)
    final_checkpoint_hash = str(final_checkpoint["checkpoint_hash"])
    endpoint_manifest = _export_endpoint(
        output_root, state, final_metrics, final_checkpoint_hash
    )

    checkpoint_hashes: dict[str, str] = {}
    for step in decision.observation_plan.checkpoint_steps:
        path = endpoint_root / f"step-{step:06d}" / "checkpoint.pt"
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        checkpoint_hashes[f"step-{step:06d}"] = str(checkpoint["checkpoint_hash"])
    trends = [json.loads(line) for line in trend_path.read_text(encoding="utf-8").splitlines()]
    _write_new(candidate_root / "parameter_trends.json", _json_bytes(trends))
    source_endpoint = common_report["candidates"]["L0"]["endpoint"]
    candidate_report = {
        "manifest": run_report.manifest,
        "endpoint": endpoint_manifest,
        "source_endpoint": source_endpoint,
        "checkpoint_steps": list(decision.observation_plan.checkpoint_steps),
        "checkpoint_hashes": checkpoint_hashes,
        "curve_points": _line_count(curve_path),
        "parameter_trend_points": _line_count(trend_path),
        "wall_seconds": wall_seconds,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "comparison_40k_to_80k": _endpoint_comparison(
            source_endpoint["metrics"], endpoint_manifest["metrics"]
        ),
    }
    final_report = {
        "schema_version": 1,
        "experiment": "scifihelmet_c4_affine_v1_l0_80k_continuation",
        "endpoint": 80_000,
        "source_common_40k_report_sha256": decision.source_report_hash,
        "source_checkpoint_hash": decision.source_checkpoint_hash,
        "continuation_config_sha256": config_hash,
        "p0_raw_gap_report": common_report["p0_raw_gap_report"],
        "candidate": candidate_report,
        "formal_holdout_accessed": False,
        "l1_or_l2_started": False,
        "static_cost": dict(AFFINE_STATIC_COST),
    }
    validate_completed_continuation_report(final_report, decision)
    _write_new(output_root / "training_report.json", _json_bytes(final_report))
    return final_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/train/scifihelmet_c4_affine_v1_l0_80k.yaml",
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
