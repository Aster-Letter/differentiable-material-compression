"""Exact-resume Lantern material-render from the verified 40k endpoint to 160k."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Mapping

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
for candidate in (ROOT / "src", ROOT / "scripts"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from cg_frontier.assets.preprocess import sha256_file  # noqa: E402
from cg_frontier.compression.render_ablation import (  # noqa: E402
    LossWeights,
    compose_ablation_loss,
    sampling_trajectory_hash,
    tensor_sha256,
)
from cg_frontier.compression.render_ablation_continuation import (  # noqa: E402
    load_continuation_checkpoint,
)
from cg_frontier.compression.render_ablation_long_continuation import (  # noqa: E402
    CHECKPOINT_STEPS,
    OBSERVATION_STEPS,
    load_long_continuation_checkpoint,
    save_long_continuation_checkpoint,
)
import train_c4_render_ablation_20k as base  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/train/c4_render_ablation_lantern_render_160k_v1.yaml"
SOURCE_STEP = 40000
ENDPOINT_STEP = 160000
ARM = "material_render"


def _repo_path(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a repository-relative path")
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"{label} escapes the repository")
    return path


def _load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != 1
        or value.get("experiment") != "c4_render_ablation_lantern_render_160k_v1"
        or value.get("formal_holdout_access") != "forbidden"
        or value.get("asset") != "Lantern"
        or value.get("arm") != ARM
    ):
        raise ValueError("invalid Lantern material-render 160k contract")
    training = value["training"]
    if (
        int(training["source_step"]) != SOURCE_STEP
        or int(training["endpoint_step"]) != ENDPOINT_STEP
        or tuple(training["continuation_observation_steps"]) != OBSERVATION_STEPS
        or tuple(training["continuation_checkpoint_steps"]) != CHECKPOINT_STEPS
        or float(training["latent_learning_rate"]) != 2.0e-4
        or float(training["affine_learning_rate"]) != 2.0e-5
    ):
        raise ValueError("Lantern 160k schedule differs from the frozen contract")
    LossWeights.from_mapping(value["loss"])
    if not all(bool(item) for item in value["postprocess"].values()):
        raise ValueError("Lantern 160k safety differs from the frozen contract")
    base_path = _repo_path(value["base_config"]["path"], "base config")
    if sha256_file(base_path) != value["base_config"]["sha256"]:
        raise ValueError("base 20k config hash mismatch")
    return value


def _effective_base_config(config: Mapping[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = _repo_path(config["base_config"]["path"], "base config")
    effective = copy.deepcopy(base._config(path))
    effective["training"].update(
        {
            "steps": ENDPOINT_STEP,
            "observation_steps": list(OBSERVATION_STEPS),
            "full_checkpoint_steps": list(CHECKPOINT_STEPS),
        }
    )
    return path, effective


def _validate_source(config: Mapping[str, Any], prepared: Mapping[str, Any]) -> dict[str, Any]:
    source = config["source"]
    paths = {
        "arm_root": _repo_path(source["arm_root"], "source arm root"),
        "preparation": _repo_path(source["preparation_path"], "source preparation"),
        "formal_marker": _repo_path(source["formal_marker_path"], "source formal marker"),
        "result_manifest": _repo_path(source["result_manifest_path"], "source result manifest"),
    }
    arm_root = paths["arm_root"]
    paths.update(
        {
            "report": arm_root / "training_report.json",
            "checkpoint": arm_root / "checkpoints/step_40000/checkpoint.pt",
            "progress_snapshot": arm_root / "checkpoints/step_40000/progress_snapshot.json",
        }
    )
    expected_hashes = {
        "preparation": source["preparation_sha256"],
        "formal_marker": source["formal_marker_sha256"],
        "result_manifest": source["result_manifest_sha256"],
        "report": source["report_sha256"],
        "checkpoint": source["checkpoint_sha256"],
        "progress_snapshot": source["progress_snapshot_sha256"],
    }
    for name, expected in expected_hashes.items():
        path = paths[name]
        if not path.is_file() or sha256_file(path) != expected:
            raise ValueError(f"source 40k {name} hash mismatch")
    preparation = json.loads(paths["preparation"].read_text(encoding="utf-8"))
    marker = json.loads(paths["formal_marker"].read_text(encoding="utf-8"))
    report = json.loads(paths["report"].read_text(encoding="utf-8"))
    if (
        preparation.get("source_identity") != prepared["identity"]
        or marker.get("status") != "formal_run_verified"
        or marker.get("job_id") != str(source["job_id"])
        or marker.get("formal_holdout_accessed") is not False
        or report.get("status") != "complete_40k_continuation"
        or report.get("arm") != ARM
        or int(report.get("steps", -1)) != SOURCE_STEP
        or report.get("source_identity") != prepared["identity"]
        or report.get("continuation_config_hash") != source["prior_continuation_config_sha256"]
    ):
        raise ValueError("source 40k report lineage mismatch")
    payload = load_continuation_checkpoint(
        paths["checkpoint"],
        expected_arm=ARM,
        expected_source_identity=prepared["identity"],
        expected_continuation_config_hash=source["prior_continuation_config_sha256"],
        expected_source_checkpoint_sha256=source["source_20k_checkpoint_sha256"],
    )
    if int(payload["step"]) != SOURCE_STEP:
        raise ValueError("source checkpoint is not the verified 40k endpoint")
    return {
        **paths,
        "checkpoint_sha256": source["checkpoint_sha256"],
        "report_value": report,
        "checkpoint_payload": payload,
    }


def _restore_state(prepared, effective, source, config_hash: str, resume: Path | None):
    state, optimizers, rng = base._new_state(prepared, effective)
    if resume is None:
        payload = source["checkpoint_payload"]
        continuation_initial_rng_hash = str(payload["current_rng_hash"])
        original_initial_rng_hash = str(payload["initial_rng_hash"])
    else:
        payload = load_long_continuation_checkpoint(
            resume,
            expected_source_identity=prepared["identity"],
            expected_continuation_config_hash=config_hash,
            expected_source_40k_checkpoint_sha256=source["checkpoint_sha256"],
        )
        continuation_initial_rng_hash = str(payload["continuation_initial_rng_hash"])
        original_initial_rng_hash = str(payload["original_initial_rng_hash"])
    for parameter, name in zip(state, ("latent", "weight", "bias"), strict=True):
        parameter.data.copy_(payload[name].to(parameter))
    for optimizer, name in zip(
        optimizers, ("latent_optimizer", "affine_optimizer"), strict=True
    ):
        optimizer.load_state_dict(payload[name])
    rng.set_state(payload["rng_state"])
    return (
        state,
        optimizers,
        rng,
        payload,
        continuation_initial_rng_hash,
        original_initial_rng_hash,
    )


def _preflight_checkpoint(
    path: Path,
    *,
    state,
    optimizers,
    rng,
    step: int,
    prepared,
    config_hash: str,
    source,
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "checkpoint_type": "c4_render_ablation_lantern_render_160k_preflight_v1",
        "asset": "Lantern",
        "arm": ARM,
        "step": step,
        "source_identity": prepared["identity"],
        "continuation_config_hash": config_hash,
        "source_40k_checkpoint_sha256": source["checkpoint_sha256"],
        "latent": state[0].detach().cpu(),
        "weight": state[1].detach().cpu(),
        "bias": state[2].detach().cpu(),
        "latent_optimizer": optimizers[0].state_dict(),
        "affine_optimizer": optimizers[1].state_dict(),
        "rng_state": rng.get_state().cpu(),
    }
    torch.save(payload, path)
    reloaded = torch.load(path, map_location="cpu", weights_only=False)
    if reloaded["step"] != step or not torch.equal(reloaded["latent"], payload["latent"]):
        raise RuntimeError("160k preflight checkpoint reload failed")
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": sha256_file(path),
        "reload_verified": True,
    }


def run(
    config_path: Path,
    output: Path,
    *,
    max_continuation_steps: int | None = None,
    resume: Path | None = None,
) -> dict[str, Any]:
    config = _load_config(config_path)
    config_hash = sha256_file(config_path)
    base_path, effective = _effective_base_config(config)
    prepared = base.prepare_asset(base_path, base._config(base_path), "Lantern")
    source = _validate_source(config, prepared)
    endpoint = (
        ENDPOINT_STEP
        if max_continuation_steps is None
        else min(ENDPOINT_STEP, SOURCE_STEP + int(max_continuation_steps))
    )
    (
        state,
        optimizers,
        rng,
        payload,
        continuation_initial_rng_hash,
        original_initial_rng_hash,
    ) = _restore_state(prepared, effective, source, config_hash, resume)
    checkpoint_step = int(payload["step"])
    if endpoint <= checkpoint_step or endpoint > ENDPOINT_STEP:
        raise ValueError("endpoint must be after the selected checkpoint and at most 160k")
    if resume is None:
        if output.exists():
            raise FileExistsError(f"refusing to overwrite 160k output: {output}")
        output.mkdir(parents=True)
        prefix = json.loads(source["progress_snapshot"].read_text(encoding="utf-8"))
    else:
        if not output.is_dir() or not resume.resolve().is_relative_to(output.resolve()):
            raise ValueError("resume checkpoint must belong to the selected output")
        prefix = json.loads((resume.parent / "progress_snapshot.json").read_text(encoding="utf-8"))
        stale = []
        for directory_name in ("observations", "checkpoints"):
            directory = output / directory_name
            if directory.is_dir():
                stale.extend(
                    child
                    for child in directory.iterdir()
                    if child.is_dir()
                    and child.name.startswith("step_")
                    and int(child.name.split("_", 1)[1]) > checkpoint_step
                )
        if stale:
            quarantine = output / f"superseded_after_step_{checkpoint_step:06d}"
            if quarantine.exists():
                raise FileExistsError("resume quarantine already exists")
            for item in stale:
                destination = quarantine / item.relative_to(output)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(item), str(destination))
    preparation_path = output.parent / "preparation.json"
    preparation = {
        "schema_version": 1,
        "experiment": "c4_render_ablation_lantern_render_160k_v1",
        "asset": "Lantern",
        "arm": ARM,
        "source_job_id": str(config["source"]["job_id"]),
        "source_step": SOURCE_STEP,
        "endpoint_step": endpoint,
        "source_checkpoint_sha256": source["checkpoint_sha256"],
        "source_identity": prepared["identity"],
        "source_identity_hash": prepared["identity_hash"],
        "continuation_config_hash": config_hash,
        "continuation_initial_rng_hash": continuation_initial_rng_hash,
        "formal_holdout_accessed": False,
        "audit_used_for_training": False,
    }
    if resume is None:
        base._write_json(preparation_path, preparation)
    else:
        existing_preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
        if existing_preparation != preparation:
            raise ValueError("resume preparation evidence differs from the frozen lineage")
    curve = list(prefix["curve"])
    sample_metrics = list(prefix["sample_metrics"])
    observations = dict(prefix["observations"])
    checkpoints = dict(prefix["checkpoints"])
    weights = LossWeights.from_mapping(effective["loss"])
    log_interval = int(effective["training"]["log_interval"])
    metric_interval = int(effective["training"]["metric_interval"])
    started = time.perf_counter()
    last_checkpoint = None
    for step in range(checkpoint_step + 1, endpoint + 1):
        terms, context = base._terms(
            state,
            prepared,
            effective,
            base._draw_batch(prepared, effective, rng),
            compute_render=True,
        )
        total, pieces = compose_ablation_loss(terms, arm=ARM, weights=weights)
        if not bool(torch.isfinite(total)):
            raise FloatingPointError(f"non-finite 160k objective at step {step}")
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        total.backward()
        for parameter in state:
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError(f"non-finite 160k gradient at step {step}")
        for optimizer in optimizers:
            optimizer.step()
        with torch.no_grad():
            state[0].clamp_(0.0, 1.0)
        if step % log_interval == 0 or step == endpoint:
            row = {
                "step": step,
                "loss": float(total.detach()),
                **{name: float(value.detach()) for name, value in terms.items()},
                "weighted_material": float(pieces["material"].detach()),
                "weighted_render": float(pieces["render"].detach()),
                "diagnostic_render": float(pieces["diagnostic_render"].detach()),
                **context,
                "finite": True,
            }
            curve.append(row)
            print(json.dumps({"arm": ARM, **row}, sort_keys=True), flush=True)
        if step % metric_interval == 0 or step == endpoint:
            sample_metrics.append({"step": step, **base._atlas_metrics(state, prepared)})
        if step in OBSERVATION_STEPS or step == endpoint:
            observations[str(step)] = base._save_observation(
                output, step, state, prepared, effective
            )
        if step in CHECKPOINT_STEPS:
            checkpoint = output / "checkpoints" / f"step_{step:06d}" / "checkpoint.pt"
            digest = save_long_continuation_checkpoint(
                checkpoint,
                step=step,
                latent=state[0],
                weight=state[1],
                bias=state[2],
                latent_optimizer=optimizers[0],
                affine_optimizer=optimizers[1],
                rng=rng,
                source_identity=prepared["identity"],
                continuation_initial_rng_hash=continuation_initial_rng_hash,
                original_initial_rng_hash=original_initial_rng_hash,
                continuation_config_hash=config_hash,
                source_40k_checkpoint_sha256=source["checkpoint_sha256"],
            )
            checkpoints[str(step)] = {
                "path": str(checkpoint.relative_to(ROOT)),
                "sha256": digest,
            }
            last_checkpoint = checkpoint
        if (
            step % log_interval == 0
            or step in OBSERVATION_STEPS
            or step in CHECKPOINT_STEPS
            or step == endpoint
        ):
            progress = {
                "schema_version": 1,
                "experiment": "c4_render_ablation_lantern_render_160k_v1",
                "asset": "Lantern",
                "arm": ARM,
                "steps": step,
                "curve": curve,
                "sample_metrics": sample_metrics,
                "observations": observations,
                "checkpoints": checkpoints,
                "source_identity": prepared["identity"],
                "continuation_config_hash": config_hash,
                "continuation_initial_rng_hash": continuation_initial_rng_hash,
                "original_initial_rng_hash": original_initial_rng_hash,
                "final_rng_hash": tensor_sha256(rng.get_state()),
            }
            base._write_json(output / "progress.json", progress)
            if step in CHECKPOINT_STEPS:
                base._write_json(
                    output / "checkpoints" / f"step_{step:06d}" / "progress_snapshot.json",
                    progress,
                )
    formal = endpoint == ENDPOINT_STEP
    preflight_checkpoint = None
    if not formal:
        preflight_checkpoint = _preflight_checkpoint(
            output / "preflight_checkpoint.pt",
            state=state,
            optimizers=optimizers,
            rng=rng,
            step=endpoint,
            prepared=prepared,
            config_hash=config_hash,
            source=source,
        )
    train_pairs = (
        tuple(
            (camera, light)
            for camera in prepared["training_camera_indices"]
            for light in range(len(prepared["rig"].lights))
        )
        if formal
        else ((prepared["training_camera_indices"][0], 0),)
    )
    audit_pairs = (
        tuple(
            (camera, light)
            for camera in prepared["audit_camera_indices"]
            for light in range(len(prepared["rig"].lights))
        )
        if formal
        else ((prepared["audit_camera_indices"][0], 0),)
    )
    report = {
        "schema_version": 1,
        "status": "complete_160k_continuation" if formal else "complete_bounded_preflight",
        "experiment": "c4_render_ablation_lantern_render_160k_v1",
        "asset": "Lantern",
        "arm": ARM,
        "steps": endpoint,
        "continuation_updates": endpoint - SOURCE_STEP,
        "continuation_updates_this_run": endpoint - checkpoint_step,
        "elapsed_seconds": time.perf_counter() - started,
        "source_identity": prepared["identity"],
        "source_identity_hash": prepared["identity_hash"],
        "continuation_config_hash": config_hash,
        "source_job_id": str(config["source"]["job_id"]),
        "source_checkpoint": {
            "step": SOURCE_STEP,
            "path": str(source["checkpoint"].relative_to(ROOT)),
            "sha256": source["checkpoint_sha256"],
        },
        "source_20k_checkpoint_sha256": config["source"]["source_20k_checkpoint_sha256"],
        "prior_continuation_config_sha256": config["source"]["prior_continuation_config_sha256"],
        "sampling_contract_hash": prepared["identity"]["sampling_contract_hash"],
        "continuation_initial_rng_hash": continuation_initial_rng_hash,
        "original_initial_rng_hash": original_initial_rng_hash,
        "final_rng_hash": tensor_sha256(rng.get_state()),
        "sampling_trajectory_hash": sampling_trajectory_hash(
            sampling_contract=prepared["identity"]["sampling_contract_hash"],
            initial_rng=continuation_initial_rng_hash,
            final_rng=tensor_sha256(rng.get_state()),
            steps=endpoint - SOURCE_STEP,
        ),
        "raw_parent": source["report_value"]["raw_parent"],
        "source_endpoint_40k": source["report_value"]["endpoint"],
        "endpoint": {
            "material": base._atlas_metrics(state, prepared),
            "train_render": base._render_metrics(state, prepared, effective, train_pairs),
            "audit_render": base._render_metrics(state, prepared, effective, audit_pairs),
        },
        "observation_steps": sorted(int(value) for value in observations),
        "checkpoints": checkpoints,
        "last_checkpoint": str(last_checkpoint.relative_to(ROOT)) if last_checkpoint else None,
        "preflight_checkpoint": preflight_checkpoint,
        "formal_holdout_accessed": False,
        "audit_used_for_training": False,
        "early_stopping": False,
        "single_seed_statistical_significance_claimed": False,
    }
    base._write_json(output / "training_report.json", report)
    base._write_json(
        output.parent / "continuation_summary.json",
        {
            "schema_version": 1,
            "status": "complete_material_render_160k" if formal else "complete_material_render_preflight",
            "experiment": "c4_render_ablation_lantern_render_160k_v1",
            "asset": "Lantern",
            "arm": ARM,
            "source_job_id": str(config["source"]["job_id"]),
            "source_step": SOURCE_STEP,
            "steps": endpoint,
            "report": str((output / "training_report.json").relative_to(ROOT)),
            "formal_holdout_accessed": False,
            "audit_used_for_training": False,
        },
    )
    print(json.dumps({"status": report["status"], "output": str(output)}, sort_keys=True))
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-continuation-steps", type=int)
    parser.add_argument("--resume-checkpoint", type=Path)
    args = parser.parse_args()
    if args.resume_checkpoint is not None and args.max_continuation_steps is not None:
        parser.error("bounded preflight and formal resume are mutually exclusive")
    run(
        args.config.resolve(),
        args.output_root.resolve(),
        max_continuation_steps=args.max_continuation_steps,
        resume=args.resume_checkpoint.resolve() if args.resume_checkpoint else None,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
