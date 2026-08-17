"""Exact-resume both Lantern C4 ablation arms from 20k to 40k."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Mapping

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from cg_frontier.assets.preprocess import sha256_file  # noqa: E402
from cg_frontier.compression.render_ablation import (  # noqa: E402
    ARMS,
    LossWeights,
    compose_ablation_loss,
    load_ablation_checkpoint,
    paired_sampling_evidence,
    sampling_trajectory_hash,
    tensor_sha256,
)
from cg_frontier.compression.render_ablation_continuation import (  # noqa: E402
    CHECKPOINT_STEPS,
    OBSERVATION_STEPS,
    load_continuation_checkpoint,
    save_continuation_checkpoint,
)
import train_c4_render_ablation_20k as base  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/train/c4_render_ablation_lantern_40k_v1.yaml"


def _repo_path(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a repository-relative path")
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"{label} escapes the repository")
    return path


def _load_config(path: Path) -> Mapping[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (
        not isinstance(value, Mapping)
        or value.get("schema_version") != 1
        or value.get("experiment") != "c4_render_ablation_lantern_40k_v1"
        or value.get("formal_holdout_access") != "forbidden"
        or value.get("asset") != "Lantern"
        or tuple(value.get("arms", ())) != ARMS
    ):
        raise ValueError("invalid Lantern 40k continuation contract")
    training = value["training"]
    if (
        int(training["source_step"]) != 20000
        or int(training["endpoint_step"]) != 40000
        or tuple(training["continuation_observation_steps"]) != OBSERVATION_STEPS
        or tuple(training["continuation_checkpoint_steps"]) != CHECKPOINT_STEPS
        or float(training["latent_learning_rate"]) != 2.0e-4
        or float(training["affine_learning_rate"]) != 2.0e-5
    ):
        raise ValueError("Lantern continuation schedule differs from the frozen contract")
    LossWeights.from_mapping(value["loss"])
    if not all(bool(item) for item in value["postprocess"].values()):
        raise ValueError("Lantern continuation safety differs from the frozen contract")
    base_path = _repo_path(value["base_config"]["path"], "base config")
    if sha256_file(base_path) != value["base_config"]["sha256"]:
        raise ValueError("base 20k config hash mismatch")
    return value


def _effective_base_config(config: Mapping[str, Any]) -> tuple[Path, Mapping[str, Any]]:
    path = _repo_path(config["base_config"]["path"], "base config")
    base_config = copy.deepcopy(base._config(path))
    base_config["training"].update(
        {
            "steps": 40000,
            "observation_steps": list(OBSERVATION_STEPS),
            "full_checkpoint_steps": list(CHECKPOINT_STEPS),
        }
    )
    return path, base_config


def _validate_source(config: Mapping[str, Any], prepared: Mapping[str, Any]) -> dict[str, Any]:
    source = config["source"]
    pair_root = _repo_path(source["pair_root"], "source pair root")
    preparation_path = pair_root / "preparation.json"
    paired_path = pair_root / "paired_summary.json"
    if sha256_file(preparation_path) != source["preparation_sha256"]:
        raise ValueError("source preparation hash mismatch")
    if sha256_file(paired_path) != source["paired_summary_sha256"]:
        raise ValueError("source paired summary hash mismatch")
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    paired = json.loads(paired_path.read_text(encoding="utf-8"))
    if (
        preparation.get("asset") != "Lantern"
        or preparation.get("identity") != prepared["identity"]
        or paired.get("status") != "complete_paired_20k"
        or paired.get("paired_sampling_evidence", {}).get("identical") is not True
    ):
        raise ValueError("source Lantern pair contract mismatch")
    arms = {}
    for arm in ARMS:
        arm_spec = source["arms"][arm]
        arm_root = pair_root / arm
        report_path = arm_root / "training_report.json"
        checkpoint = arm_root / "checkpoints/step_20000/checkpoint.pt"
        progress_snapshot = arm_root / "checkpoints/step_20000/progress_snapshot.json"
        if sha256_file(report_path) != arm_spec["report_sha256"]:
            raise ValueError(f"source report hash mismatch: {arm}")
        if sha256_file(checkpoint) != arm_spec["checkpoint_sha256"]:
            raise ValueError(f"source checkpoint hash mismatch: {arm}")
        if sha256_file(progress_snapshot) != arm_spec["progress_snapshot_sha256"]:
            raise ValueError(f"source progress snapshot hash mismatch: {arm}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("status") != "complete_20k" or int(report.get("steps", -1)) != 20000:
            raise ValueError(f"source arm is not a complete 20k run: {arm}")
        arms[arm] = {
            "root": arm_root,
            "report": report,
            "checkpoint": checkpoint,
            "checkpoint_sha256": arm_spec["checkpoint_sha256"],
            "progress_snapshot": progress_snapshot,
        }
    return {"pair_root": pair_root, "preparation": preparation, "paired": paired, "arms": arms}


def _restore_state(prepared, base_config, source, arm, continuation_config_hash, resume):
    state, optimizers, rng = base._new_state(prepared, base_config)
    if resume is None:
        payload = load_ablation_checkpoint(
            source["checkpoint"],
            expected_asset="Lantern",
            expected_arm=arm,
            expected_identity=prepared["identity"],
        )
    else:
        payload = load_continuation_checkpoint(
            resume,
            expected_arm=arm,
            expected_source_identity=prepared["identity"],
            expected_continuation_config_hash=continuation_config_hash,
            expected_source_checkpoint_sha256=source["checkpoint_sha256"],
        )
    for parameter, name in zip(state, ("latent", "weight", "bias"), strict=True):
        parameter.data.copy_(payload[name].to(parameter))
    for optimizer, name in zip(
        optimizers, ("latent_optimizer", "affine_optimizer"), strict=True
    ):
        optimizer.load_state_dict(payload[name])
    rng.set_state(payload["rng_state"])
    return state, optimizers, rng, payload


def _preflight_checkpoint(path, *, state, optimizers, rng, arm, step, prepared, config_hash, source):
    payload = {
        "schema_version": 1,
        "checkpoint_type": "c4_render_ablation_lantern_40k_preflight_v1",
        "asset": "Lantern",
        "arm": arm,
        "step": step,
        "source_identity": prepared["identity"],
        "continuation_config_hash": config_hash,
        "source_checkpoint_sha256": source["checkpoint_sha256"],
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
        raise RuntimeError("continuation preflight checkpoint reload failed")
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path), "reload_verified": True}


def run_arm(
    prepared,
    base_config,
    continuation_config,
    source,
    *,
    arm: str,
    output: Path,
    endpoint_step: int,
    resume: Path | None = None,
) -> dict[str, Any]:
    config_hash = sha256_file(DEFAULT_CONFIG if continuation_config is None else continuation_config["_path"])
    state, optimizers, rng, payload = _restore_state(
        prepared, base_config, source, arm, config_hash, resume
    )
    checkpoint_step = int(payload["step"])
    if endpoint_step <= checkpoint_step or endpoint_step > 40000:
        raise ValueError("continuation endpoint must be after its checkpoint and at most 40k")
    if output.exists() and resume is None:
        raise FileExistsError(f"refusing to overwrite continuation output: {output}")
    output.mkdir(parents=True, exist_ok=resume is not None)
    source_snapshot = source["progress_snapshot"]
    prefix = json.loads(source_snapshot.read_text(encoding="utf-8"))
    if resume is None:
        curve = list(prefix["curve"])
        sample_metrics = list(prefix["sample_metrics"])
        observations = dict(prefix["observations"])
        checkpoints = dict(prefix["checkpoints"])
    else:
        snapshot = resume.parent / "progress_snapshot.json"
        progress = json.loads(snapshot.read_text(encoding="utf-8"))
        curve = list(progress["curve"])
        sample_metrics = list(progress["sample_metrics"])
        observations = dict(progress["observations"])
        checkpoints = dict(progress["checkpoints"])
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
            quarantine = output / f"superseded_after_step_{checkpoint_step:05d}"
            if quarantine.exists():
                raise FileExistsError("continuation resume quarantine already exists")
            for item in stale:
                destination = quarantine / item.relative_to(output)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(item), str(destination))
    weights = LossWeights.from_mapping(base_config["loss"])
    started = time.perf_counter()
    last_checkpoint = None
    log_interval = int(base_config["training"]["log_interval"])
    metric_interval = int(base_config["training"]["metric_interval"])
    for step in range(checkpoint_step + 1, endpoint_step + 1):
        compute_render = arm == "material_render" or step % log_interval == 0 or step == endpoint_step
        terms, context = base._terms(
            state,
            prepared,
            base_config,
            base._draw_batch(prepared, base_config, rng),
            compute_render=compute_render,
        )
        total, pieces = compose_ablation_loss(terms, arm=arm, weights=weights)
        if not bool(torch.isfinite(total)):
            raise FloatingPointError(f"non-finite continuation objective at step {step}")
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        total.backward()
        for parameter in state:
            if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                raise FloatingPointError(f"non-finite continuation gradient at step {step}")
        for optimizer in optimizers:
            optimizer.step()
        with torch.no_grad():
            state[0].clamp_(0.0, 1.0)
        if step % log_interval == 0 or step == endpoint_step:
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
            print(json.dumps({"arm": arm, **row}, sort_keys=True), flush=True)
        if step % metric_interval == 0 or step == endpoint_step:
            sample_metrics.append({"step": step, **base._atlas_metrics(state, prepared)})
        if step in OBSERVATION_STEPS or step == endpoint_step:
            observations[str(step)] = base._save_observation(
                output, step, state, prepared, base_config
            )
        if step in CHECKPOINT_STEPS:
            path = output / "checkpoints" / f"step_{step:05d}" / "checkpoint.pt"
            digest = save_continuation_checkpoint(
                path,
                arm=arm,
                step=step,
                latent=state[0],
                weight=state[1],
                bias=state[2],
                latent_optimizer=optimizers[0],
                affine_optimizer=optimizers[1],
                rng=rng,
                source_identity=prepared["identity"],
                initial_rng_hash=str(payload["initial_rng_hash"]),
                continuation_config_hash=config_hash,
                source_checkpoint_sha256=source["checkpoint_sha256"],
            )
            checkpoints[str(step)] = {"path": str(path.relative_to(ROOT)), "sha256": digest}
            last_checkpoint = path
        if (
            step % log_interval == 0
            or step in OBSERVATION_STEPS
            or step in CHECKPOINT_STEPS
            or step == endpoint_step
        ):
            progress = {
                "schema_version": 1,
                "experiment": "c4_render_ablation_lantern_40k_v1",
                "asset": "Lantern",
                "arm": arm,
                "steps": step,
                "curve": curve,
                "sample_metrics": sample_metrics,
                "observations": observations,
                "checkpoints": checkpoints,
                "source_identity": prepared["identity"],
                "continuation_config_hash": config_hash,
                "initial_rng_hash": str(payload["initial_rng_hash"]),
                "final_rng_hash": tensor_sha256(rng.get_state()),
            }
            base._write_json(output / "progress.json", progress)
            if step in CHECKPOINT_STEPS:
                base._write_json(
                    output / "checkpoints" / f"step_{step:05d}" / "progress_snapshot.json",
                    progress,
                )
    formal = endpoint_step == 40000
    preflight_checkpoint = None
    if not formal:
        preflight_checkpoint = _preflight_checkpoint(
            output / "preflight_checkpoint.pt",
            state=state,
            optimizers=optimizers,
            rng=rng,
            arm=arm,
            step=endpoint_step,
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
        "status": "complete_40k_continuation" if formal else "complete_bounded_preflight",
        "experiment": "c4_render_ablation_lantern_40k_v1",
        "asset": "Lantern",
        "arm": arm,
        "steps": endpoint_step,
        "continuation_updates": endpoint_step - checkpoint_step,
        "elapsed_seconds": time.perf_counter() - started,
        "source_identity": prepared["identity"],
        "source_identity_hash": prepared["identity_hash"],
        "continuation_config_hash": config_hash,
        "source_checkpoint": {
            "step": 20000,
            "path": str(source["checkpoint"].relative_to(ROOT)),
            "sha256": source["checkpoint_sha256"],
        },
        "sampling_contract_hash": prepared["identity"]["sampling_contract_hash"],
        "initial_rng_hash": str(payload["initial_rng_hash"]),
        "final_rng_hash": tensor_sha256(rng.get_state()),
        "sampling_trajectory_hash": sampling_trajectory_hash(
            sampling_contract=prepared["identity"]["sampling_contract_hash"],
            initial_rng=str(payload["initial_rng_hash"]),
            final_rng=tensor_sha256(rng.get_state()),
            steps=endpoint_step,
        ),
        "raw_parent": source["report"]["raw_parent"],
        "source_endpoint_20k": source["report"]["endpoint"],
        "endpoint": {
            "material": base._atlas_metrics(state, prepared),
            "train_render": base._render_metrics(state, prepared, base_config, train_pairs),
            "audit_render": base._render_metrics(state, prepared, base_config, audit_pairs),
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
    return report


def run_pair(config_path: Path, output: Path, *, max_steps: int | None = None) -> dict[str, Any]:
    config = dict(_load_config(config_path))
    config["_path"] = config_path
    base_path, base_config = _effective_base_config(config)
    prepared = base.prepare_asset(base_path, base._config(base_path), "Lantern")
    source = _validate_source(config, prepared)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite Lantern continuation pair: {output}")
    output.mkdir(parents=True)
    endpoint = 40000 if max_steps is None else min(40000, 20000 + int(max_steps))
    base._write_json(
        output / "preparation.json",
        {
            "schema_version": 1,
            "experiment": "c4_render_ablation_lantern_40k_v1",
            "asset": "Lantern",
            "source_job_id": str(config["source"]["job_id"]),
            "source_pair_root": str(source["pair_root"].relative_to(ROOT)),
            "source_identity": prepared["identity"],
            "source_identity_hash": prepared["identity_hash"],
            "continuation_config_hash": sha256_file(config_path),
            "source_step": 20000,
            "endpoint_step": endpoint,
            "formal_holdout_accessed": False,
            "audit_used_for_training": False,
        },
    )
    reports = {
        arm: run_arm(
            prepared,
            base_config,
            config,
            source["arms"][arm],
            arm=arm,
            output=output / arm,
            endpoint_step=endpoint,
        )
        for arm in ARMS
    }
    evidence = paired_sampling_evidence(reports[ARMS[0]], reports[ARMS[1]])
    if not evidence["identical"]:
        raise RuntimeError("Lantern continuation arms did not preserve paired sampling")
    summary = {
        "schema_version": 1,
        "status": "complete_paired_40k" if endpoint == 40000 else "complete_paired_preflight",
        "experiment": "c4_render_ablation_lantern_40k_v1",
        "asset": "Lantern",
        "steps": endpoint,
        "paired_sampling_evidence": evidence,
        "source_job_id": str(config["source"]["job_id"]),
        "formal_holdout_accessed": False,
        "audit_used_for_training": False,
        "reports": {
            arm: str((output / arm / "training_report.json").relative_to(ROOT))
            for arm in ARMS
        },
    }
    base._write_json(output / "paired_summary.json", summary)
    print(json.dumps({"status": summary["status"], "output": str(output)}, sort_keys=True))
    return summary


def resume_pair(
    config_path: Path,
    pair_root: Path,
    *,
    arm: str,
    checkpoint: Path,
) -> dict[str, Any]:
    """Resume exactly one failed arm from a frozen 30k continuation checkpoint."""
    if arm not in ARMS:
        raise ValueError(f"unknown continuation arm: {arm}")
    config = dict(_load_config(config_path))
    config["_path"] = config_path
    base_path, base_config = _effective_base_config(config)
    prepared = base.prepare_asset(base_path, base._config(base_path), "Lantern")
    source = _validate_source(config, prepared)
    if not pair_root.is_dir() or not checkpoint.is_file():
        raise FileNotFoundError("resume pair root and checkpoint must already exist")
    if not checkpoint.resolve().is_relative_to((pair_root / arm).resolve()):
        raise ValueError("resume checkpoint must belong to the selected arm")
    payload = load_continuation_checkpoint(
        checkpoint,
        expected_arm=arm,
        expected_source_identity=prepared["identity"],
        expected_continuation_config_hash=sha256_file(config_path),
        expected_source_checkpoint_sha256=source["arms"][arm]["checkpoint_sha256"],
    )
    if int(payload["step"]) != 30000:
        raise ValueError("formal recovery must restart from the frozen 30k checkpoint")
    run_arm(
        prepared,
        base_config,
        config,
        source["arms"][arm],
        arm=arm,
        output=pair_root / arm,
        endpoint_step=40000,
        resume=checkpoint,
    )
    other_arm = ARMS[1] if arm == ARMS[0] else ARMS[0]
    other_report = pair_root / other_arm / "training_report.json"
    if not other_report.is_file():
        other_root = pair_root / other_arm
        if other_root.exists():
            raise RuntimeError(
                f"the other arm is partial; resume it separately before finalization: {other_root}"
            )
        run_arm(
            prepared,
            base_config,
            config,
            source["arms"][other_arm],
            arm=other_arm,
            output=other_root,
            endpoint_step=40000,
        )
    reports = {
        name: json.loads((pair_root / name / "training_report.json").read_text(encoding="utf-8"))
        for name in ARMS
    }
    if any(
        report.get("status") != "complete_40k_continuation"
        or int(report.get("steps", -1)) != 40000
        for report in reports.values()
    ):
        raise RuntimeError("both continuation arms must be complete at 40k before finalization")
    evidence = paired_sampling_evidence(reports[ARMS[0]], reports[ARMS[1]])
    if not evidence["identical"]:
        raise RuntimeError("resumed Lantern continuation lost paired sampling")
    summary = {
        "schema_version": 1,
        "status": "complete_paired_40k",
        "experiment": "c4_render_ablation_lantern_40k_v1",
        "asset": "Lantern",
        "steps": 40000,
        "paired_sampling_evidence": evidence,
        "source_job_id": str(config["source"]["job_id"]),
        "resumed_arm": arm,
        "resume_checkpoint_step": 30000,
        "formal_holdout_accessed": False,
        "audit_used_for_training": False,
        "reports": {
            name: str((pair_root / name / "training_report.json").relative_to(ROOT))
            for name in ARMS
        },
    }
    base._write_json(pair_root / "paired_summary.json", summary)
    print(json.dumps({"status": summary["status"], "output": str(pair_root)}, sort_keys=True))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-continuation-steps", type=int)
    parser.add_argument("--resume-arm", choices=ARMS)
    parser.add_argument("--resume-checkpoint", type=Path)
    args = parser.parse_args()
    if (args.resume_arm is None) != (args.resume_checkpoint is None):
        parser.error("--resume-arm and --resume-checkpoint must be provided together")
    if args.resume_arm is not None:
        if args.max_continuation_steps is not None:
            parser.error("bounded preflight and formal resume are mutually exclusive")
        resume_pair(
            args.config.resolve(),
            args.output_root.resolve(),
            arm=args.resume_arm,
            checkpoint=args.resume_checkpoint.resolve(),
        )
    else:
        run_pair(
            args.config.resolve(),
            args.output_root.resolve(),
            max_steps=args.max_continuation_steps,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
