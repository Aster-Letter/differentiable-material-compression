"""Run the authorized SciFiHelmet C4-affine L0/L1/L2 common 40k experiment."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import io
import json
from pathlib import Path
import sys
import time
from typing import Mapping

from PIL import Image
import torch
import torch.nn.functional as F
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cg_frontier.compression.affine_training import (
    AffineCandidateState,
    TrainingObservationPlan,
    candidate_manifest,
    create_paired_candidates,
    plan_training_observations,
    resume_candidate,
    run_candidate_training,
)
from cg_frontier.assets.gltf_mesh import load_gltf_mesh
from cg_frontier.compression.affine_material import (
    AFFINE_STATIC_COST,
    certify_affine,
    export_affine_decoder,
)
from cg_frontier.compression.affine_pca import export_p0_bundle, rasterize_uv_charts
from cg_frontier.compression.affine_regularizers import fake_quantize_unorm8
from cg_frontier.compression.material import Core4Targets, load_core4_targets
from cg_frontier.compression.render_loss import (
    bilinear_sample_top_down_wrap,
    decoded_to_material,
    masked_render_metrics,
)
from cg_frontier.render.gbuffer import (
    load_core4_textures,
    render_geometry_gbuffer,
    sample_core4_material,
)
from cg_frontier.render.pbr import shade_ggx
from run_scifihelmet_c4_affine_preflight import (
    PreflightObjective,
    _camera,
    _json_bytes,
    _light,
    _move_p0,
    _raw_safe_metrics,
    _repo_path,
    _sha256_file,
    _targets_to_seven,
    _write_new,
)


OFFICIAL_CHECKPOINT_STEPS = (1_000, 5_000, 10_000, 20_000, 30_000, 35_000, 40_000)


@dataclass(frozen=True)
class FrozenTrainingDecision:
    parent_p0_hash: str
    raw_p0_hash: str
    raw_gap_report: dict[str, object]
    tv_ratio: float
    cube_ratio: float
    tv_lambda: float
    cube_lambda: float
    candidate_order: tuple[str, ...]
    observation_plan: TrainingObservationPlan


def freeze_training_decision(
    config: Mapping[str, object],
    preflight_report: Mapping[str, object],
) -> FrozenTrainingDecision:
    """Fail closed unless the authorized M6 decision is completely frozen."""

    authorization = config["authorization"]
    selection = config["selection"]
    training = config["training"]
    if not isinstance(authorization, Mapping) or not isinstance(selection, Mapping) or not isinstance(training, Mapping):
        raise ValueError("formal training config sections are invalid")
    if authorization.get("authorized_40k") is not True:
        raise ValueError("40k training is not authorized")
    if authorization.get("retain_p0_raw_gap_report") is not True:
        raise ValueError("P0 raw gap report must be retained")
    candidate_order = tuple(str(value) for value in authorization["candidate_order"])
    if candidate_order != ("L0", "L1", "L2"):
        raise ValueError("candidate order must be L0, L1, L2")
    tv_ratio = float(selection["tv_ratio"])
    cube_ratio = float(selection["cube_ratio"])
    if tv_ratio != 0.05 or cube_ratio != 0.05:
        raise ValueError("authorized TV/cube ratios must both be 0.05")
    total_steps = int(training["total_steps"])
    checkpoint_steps = tuple(int(value) for value in training["checkpoint_steps"])
    if total_steps != 40_000 or checkpoint_steps != OFFICIAL_CHECKPOINT_STEPS:
        raise ValueError("formal training requires the frozen common 40k checkpoint plan")
    observation_plan = plan_training_observations(
        total_steps=total_steps,
        checkpoint_steps=checkpoint_steps,
        trend_interval=int(training["trend_interval"]),
    )

    p0 = preflight_report["p0"]
    calibration = preflight_report["calibration"]
    if not isinstance(p0, Mapping) or not isinstance(calibration, Mapping):
        raise ValueError("preflight evidence is incomplete")
    raw = p0["raw"]
    safe = p0["safe"]
    tv = calibration["tv"]
    cube = calibration["cube"]
    if not all(isinstance(value, Mapping) for value in (raw, safe, tv, cube)):
        raise ValueError("preflight lineage or calibration is invalid")
    tv_lambda = float(tv["lambdas"]["0.05"])
    cube_lambda = float(cube["lambdas"]["0.05"])
    return FrozenTrainingDecision(
        parent_p0_hash=str(safe["artifact_hash"]),
        raw_p0_hash=str(raw["artifact_hash"]),
        raw_gap_report=dict(p0),
        tv_ratio=tv_ratio,
        cube_ratio=cube_ratio,
        tv_lambda=tv_lambda,
        cube_lambda=cube_lambda,
        candidate_order=candidate_order,
        observation_plan=observation_plan,
    )


def validate_completed_report(
    report: dict[str, object],
    decision: FrozenTrainingDecision,
    *,
    expected_preflight_hash: str,
) -> dict[str, object]:
    """Validate and return an already-complete common 40k report unchanged."""

    if report.get("experiment") != "scifihelmet_c4_affine_v1_40k":
        raise ValueError("completed report experiment mismatch")
    if report.get("common_endpoint") != decision.observation_plan.total_steps:
        raise ValueError("completed report endpoint mismatch")
    if report.get("source_preflight_report_sha256") != expected_preflight_hash:
        raise ValueError("completed report preflight lineage mismatch")
    candidates = report.get("candidates")
    if not isinstance(candidates, Mapping) or tuple(candidates) != decision.candidate_order:
        raise ValueError("completed report candidate order mismatch")
    for candidate_id in decision.candidate_order:
        candidate = candidates[candidate_id]
        manifest = candidate.get("manifest") if isinstance(candidate, Mapping) else None
        if (
            not isinstance(manifest, Mapping)
            or manifest.get("candidate_id") != candidate_id
            or manifest.get("parent_p0_hash") != decision.parent_p0_hash
            or manifest.get("optimizer_updates") != decision.observation_plan.total_steps
        ):
            raise ValueError(f"completed {candidate_id} lineage mismatch")
    return report


def _load_mapping(path: Path, label: str) -> tuple[dict[str, object], bytes]:
    payload = path.read_bytes()
    value = yaml.safe_load(payload) if path.suffix.lower() in {".yaml", ".yml"} else json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a mapping")
    return value, payload


@torch.no_grad()
def _endpoint_metrics(
    state: AffineCandidateState,
    target: Core4Targets,
    valid_indices: torch.Tensor,
    geometries: list[object],
    cameras: list[object],
    reference_hdr: list[torch.Tensor],
    light: object,
    minimum_roughness: float,
) -> dict[str, object]:
    deployed = fake_quantize_unorm8(state.latent.detach())
    decoded = state.decoder(deployed.reshape(-1, 4)[valid_indices])
    selected = target.select(valid_indices)
    material = {
        "base_color_l1": float(F.l1_loss(decoded.base_color_linear, selected.base_color_linear).cpu()),
        "normal_cosine": float(torch.mean(1.0 - torch.sum(decoded.normal_xyz * selected.normal_xyz, dim=-1)).cpu()),
        "roughness_l1": float(F.l1_loss(decoded.roughness, selected.roughness).cpu()),
        "metallic_l1": float(F.l1_loss(decoded.metallic, selected.metallic).cpu()),
    }
    material["seven_channel_mae"] = float(
        torch.mean(
            torch.abs(
                torch.cat(
                    (
                        decoded.base_color_linear,
                        decoded.normal_xy,
                        decoded.roughness,
                        decoded.metallic,
                    ),
                    dim=-1,
                )
                - torch.cat(
                    (
                        selected.base_color_linear,
                        selected.normal_xyz[:, :2],
                        selected.roughness,
                        selected.metallic,
                    ),
                    dim=-1,
                )
            )
        ).cpu()
    )
    renders: list[dict[str, object]] = []
    for geometry, camera, reference in zip(geometries, cameras, reference_hdr):
        sampled = bilinear_sample_top_down_wrap(deployed, geometry.torch_buffers["uv"])
        candidate_material = decoded_to_material(geometry, state.decoder(sampled))
        candidate_hdr = shade_ggx(
            geometry,
            camera,
            light,
            material_override=candidate_material,
            minimum_roughness=minimum_roughness,
        )
        renders.append(
            masked_render_metrics(
                reference,
                candidate_hdr,
                geometry.torch_buffers["mask"],
                linear_psnr_data_range=2.0,
                display_exposure=1.5,
            )
        )
    return {"material": material, "helmet_render": renders}


def _export_endpoint(
    output_root: Path,
    state: AffineCandidateState,
    metrics: dict[str, object],
    checkpoint_hash: str,
) -> dict[str, object]:
    endpoint = endpoint_artifact_directory(output_root, state)
    latent_unorm8 = torch.floor(state.latent.detach().clamp(0.0, 1.0) * 255.0 + 0.5).to(torch.uint8)
    image = Image.fromarray(latent_unorm8.cpu().numpy(), mode="RGBA")
    stream = io.BytesIO()
    image.save(stream, format="PNG", compress_level=9, optimize=False)
    latent_payload = stream.getvalue()
    decoder = export_affine_decoder(state.decoder)
    weight, bias = state.decoder.fold_affine()
    certificate = certify_affine(weight, bias, margin=state.decoder.margin)
    if certificate["valid"] is not True:
        raise ValueError(f"{state.candidate_id} endpoint certificate failed")
    latent_hash = hashlib.sha256(latent_payload).hexdigest()
    decoder_hash = hashlib.sha256(decoder.payload).hexdigest()
    artifact_hash = hashlib.sha256(
        (state.candidate_id + checkpoint_hash + latent_hash + decoder_hash).encode("ascii")
    ).hexdigest()
    manifest = {
        **candidate_manifest(state),
        "artifact_hash": artifact_hash,
        "checkpoint_hash": checkpoint_hash,
        "hashes": {
            "latent_png_sha256": latent_hash,
            "decoder_sha256": decoder_hash,
        },
        "certificate": certificate,
        "static_cost": dict(AFFINE_STATIC_COST),
        "decoder_manifest": decoder.manifest,
        "metrics": metrics,
    }
    _write_new(endpoint / "latent_rgba8.png", latent_payload)
    _write_new(endpoint / "decoder.bin", decoder.payload)
    _write_new(endpoint / "manifest.json", _json_bytes(manifest))
    return manifest


def endpoint_artifact_directory(
    output_root: Path, state: AffineCandidateState
) -> Path:
    """Return the immutable artifact directory for the state's actual step."""

    return (
        output_root
        / "candidates"
        / state.candidate_id
        / f"step-{state.optimizer_updates:06d}"
    )


def _json_lines(points: tuple[dict[str, object], ...]) -> bytes:
    return b"".join(_json_bytes(point) for point in points)


def _append_json_line(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as stream:
        stream.write(_json_bytes(value))


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("rb") as stream:
        return sum(1 for _ in stream)


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON mapping: {path}")
    return value


def run(config_path: Path, *, resume_existing: bool = False) -> dict[str, object]:
    config, config_bytes = _load_mapping(config_path, "formal config")
    if config.get("experiment") != "scifihelmet_c4_affine_v1_40k":
        raise ValueError("unsupported formal affine experiment")
    source = config["source"]
    if not isinstance(source, Mapping):
        raise ValueError("source section is invalid")
    preflight_config_path = _repo_path(source["preflight_config"], "source.preflight_config")
    preflight_report_path = _repo_path(source["preflight_report"], "source.preflight_report")
    preflight_config, preflight_config_bytes = _load_mapping(preflight_config_path, "preflight config")
    preflight_report, preflight_report_bytes = _load_mapping(preflight_report_path, "preflight report")
    expected_report_hash = str(source["preflight_report_sha256"])
    if hashlib.sha256(preflight_report_bytes).hexdigest() != expected_report_hash:
        raise ValueError("preflight report SHA-256 mismatch")
    if hashlib.sha256(preflight_config_bytes).hexdigest() != preflight_report.get("config_sha256"):
        raise ValueError("preflight config lineage mismatch")
    decision = freeze_training_decision(config, preflight_report)

    output_root = _repo_path(config["output_root"], "output_root")
    if output_root.exists():
        if not resume_existing:
            raise FileExistsError(f"refusing to inherit formal training output: {output_root}")
        existing_decision = _read_json(output_root / "decision.json")
        if existing_decision.get("parent_p0_hash") != decision.parent_p0_hash:
            raise ValueError("existing output parent P0 lineage mismatch")
        if existing_decision.get("source_preflight_report_sha256") != expected_report_hash:
            raise ValueError("existing output preflight lineage mismatch")
        if _read_json(output_root / "p0_raw_gap_report.json") != decision.raw_gap_report:
            raise ValueError("existing output P0 raw gap report mismatch")
        completed_report_path = output_root / "training_report.json"
        if completed_report_path.exists():
            return validate_completed_report(
                _read_json(completed_report_path),
                decision,
                expected_preflight_hash=expected_report_hash,
            )
    else:
        output_root.mkdir(parents=True)
        _write_new(
            output_root / "decision.json",
            _json_bytes(
                {
                    **asdict(decision),
                    "source_preflight_report": str(source["preflight_report"]),
                    "source_preflight_report_sha256": expected_report_hash,
                    "authorized_40k": True,
                    "formal_holdout_accessed": False,
                }
            ),
        )
        _write_new(output_root / "p0_raw_gap_report.json", _json_bytes(decision.raw_gap_report))

    gltf_path = _repo_path(preflight_config["inputs"]["gltf"], "inputs.gltf")
    core4_dir = _repo_path(preflight_config["inputs"]["core4_dir"], "inputs.core4_dir")
    core4_manifest = _repo_path(preflight_config["inputs"]["core4_manifest"], "inputs.core4_manifest")
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
    if p0_bundle.calibration.raw.artifact_hash != decision.raw_p0_hash:
        raise ValueError("reconstructed P0-raw hash mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("formal affine 40k training requires CUDA")

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
    training_config = preflight_config["training"]
    formal_training = config["training"]
    batch_size = int(training_config["material_batch_size"])
    config_hash = hashlib.sha256(config_bytes).hexdigest()
    input_hash = hashlib.sha256(
        (_sha256_file(gltf_path) + _sha256_file(core4_manifest)).encode("ascii")
    ).hexdigest()

    p0_metrics = _raw_safe_metrics(
        p0_bundle,
        target_seven,
        valid_mask,
        geometries[0],
        cameras[0],
        light,
        reference_hdr[0],
        minimum_roughness,
    )
    if p0_metrics != decision.raw_gap_report:
        raise ValueError("P0 raw gap report changed since preflight")

    candidate_reports: dict[str, object] = {}
    for candidate_id in decision.candidate_order:
        candidate_root = output_root / "candidates" / candidate_id
        completed_report_path = candidate_root / "training_report.json"
        if resume_existing and completed_report_path.exists():
            completed = _read_json(completed_report_path)
            manifest = completed.get("manifest")
            if not isinstance(manifest, Mapping) or manifest.get("parent_p0_hash") != decision.parent_p0_hash or manifest.get("optimizer_updates") != 40_000:
                raise ValueError(f"completed {candidate_id} report lineage mismatch")
            candidate_reports[candidate_id] = completed
            print(_json_bytes({"candidate": candidate_id, "status": "validated-complete"}).decode().rstrip(), flush=True)
            continue
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        initial = create_paired_candidates(
                p0,
                core_seed=seed + 11,
                cube_seed=seed + 17,
                config_hash=config_hash,
                input_hash=input_hash,
                latent_learning_rate=float(training_config["latent_learning_rate"]),
                affine_learning_rate=float(training_config["affine_learning_rate"]),
            )[candidate_id]
        state = initial
        endpoint_root = output_root / "checkpoints" / candidate_id / "endpoints"
        existing_checkpoints = sorted(endpoint_root.glob("step-*/checkpoint.pt")) if resume_existing else []
        if existing_checkpoints:
            checkpoint = torch.load(existing_checkpoints[-1], map_location=device, weights_only=False)
            state = resume_candidate(
                checkpoint,
                p0,
                expected_parent_p0_hash=decision.parent_p0_hash,
                expected_config_hash=config_hash,
                expected_input_hash=input_hash,
            )
            del initial
            print(
                _json_bytes(
                    {"candidate": candidate_id, "status": "resumed", "step": state.optimizer_updates}
                ).decode().rstrip(),
                flush=True,
            )
        curve_path = candidate_root / "curve.jsonl"
        trend_path = candidate_root / "parameter_trends.jsonl"
        if _line_count(curve_path) != state.optimizer_updates:
            raise ValueError(f"{candidate_id} curve length does not match resume step")
        if _line_count(trend_path) != state.optimizer_updates // int(formal_training["trend_interval"]):
            raise ValueError(f"{candidate_id} trend length does not match resume step")
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
            tv_lambda=decision.tv_lambda if candidate_id == "L1" else 0.0,
            cube_lambda=decision.cube_lambda if candidate_id == "L2" else 0.0,
        )
        progress_interval = int(formal_training.get("progress_interval", 250))

        def report_progress(point: dict[str, object], *, name: str = candidate_id) -> None:
            _append_json_line(curve_path, point)
            if int(point["step"]) % progress_interval == 0:
                print(
                    _json_bytes(
                        {
                            "candidate": name,
                            "step": point["step"],
                            "loss": point["loss"],
                            "phase": point["phase"],
                        }
                    ).decode().rstrip(),
                    flush=True,
                )

        def report_trend(trend: dict[str, object]) -> None:
            _append_json_line(trend_path, trend)

        started = time.perf_counter()
        run_report = run_candidate_training(
            state,
            objective,
            p0,
            output_root=output_root / "checkpoints",
            observation_plan=decision.observation_plan,
            texel_count=valid_indices.numel(),
            batch_size=batch_size,
            cube_sample_count=int(formal_training.get("cube_sample_count", 1)),
            on_step=report_progress,
            on_trend=report_trend,
        )
        torch.cuda.synchronize(device)
        wall_seconds = time.perf_counter() - started
        endpoint_metrics = _endpoint_metrics(
            state,
            targets,
            valid_indices,
            geometries,
            cameras,
            reference_hdr,
            light,
            minimum_roughness,
        )
        final_checkpoint = run_report.checkpoints[-1]
        endpoint_manifest = _export_endpoint(
            output_root,
            state,
            endpoint_metrics,
            final_checkpoint.checkpoint_hash,
        )
        _write_new(
            candidate_root / "parameter_trends.json",
            _json_bytes(
                [json.loads(line) for line in trend_path.read_text(encoding="utf-8").splitlines()]
            ),
        )
        report = {
            "manifest": run_report.manifest,
            "endpoint": endpoint_manifest,
            "checkpoint_steps": [
                int(write.path.parent.name.removeprefix("step-"))
                for write in run_report.checkpoints
            ],
            "checkpoint_hashes": {
                write.path.parent.name: write.checkpoint_hash
                for write in run_report.checkpoints
            },
            "curve_points": _line_count(curve_path),
            "parameter_trend_points": _line_count(trend_path),
            "wall_seconds": wall_seconds,
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
        _write_new(candidate_root / "training_report.json", _json_bytes(report))
        candidate_reports[candidate_id] = report
        del state, objective, run_report
        torch.cuda.empty_cache()

    final_report = {
        "schema_version": 1,
        "experiment": "scifihelmet_c4_affine_v1_40k",
        "config_sha256": config_hash,
        "source_preflight_report_sha256": expected_report_hash,
        "decision": asdict(decision),
        "p0_raw_gap_report": p0_metrics,
        "candidates": candidate_reports,
        "candidate_order": list(decision.candidate_order),
        "common_endpoint": 40_000,
        "formal_holdout_accessed": False,
        "authorized_80k": False,
        "static_cost": dict(AFFINE_STATIC_COST),
    }
    _write_new(output_root / "training_report.json", _json_bytes(final_report))
    return final_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/train/scifihelmet_c4_affine_v1_40k.yaml",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = run(args.config.resolve(), resume_existing=args.resume)
    print(
        _json_bytes(
            {
                "complete": True,
                "experiment": report["experiment"],
                "common_endpoint": report["common_endpoint"],
            }
        ).decode(),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
