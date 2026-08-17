from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for directory in (SRC, SCRIPTS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from cg_frontier.compression.basecolor_dynamic_constraint import (  # noqa: E402
    DynamicBaseColorConstraintConfig,
    compose_dynamic_basecolor_constraint,
    update_dynamic_multipliers,
)
from cg_frontier.compression.basecolor_priority import (  # noqa: E402
    C4PostprocessConfig,
    decoder_instruction_audit,
    postprocess_affine_output,
)
from train_scifihelmet_c4_basecolor_priority_10k import (  # noqa: E402
    _atlas_metrics,
    _canonical_hash,
    _draw_batch,
    _load_config,
    _loss_terms,
    _new_state,
    _prepare,
    _render_metrics,
    _repo_output_path,
    _residual,
    _write_json,
)


DEFAULT_CONFIG = ROOT / "configs/train/scifihelmet_c4_basecolor_dynamic_constraint_10k_v1.yaml"
CHECKPOINT_TYPE = "c4_basecolor_dynamic_constraint_v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_dynamic_config(path: Path) -> tuple[Mapping[str, Any], Path, Mapping[str, Any]]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping) or config.get("schema_version") != 1:
        raise ValueError("unsupported dynamic BaseColor constraint config")
    if config.get("formal_holdout_access") != "forbidden":
        raise ValueError("formal holdout must remain forbidden")
    if config.get("candidate_id") != "BC-dynamic":
        raise ValueError("dynamic BaseColor candidate identity changed")
    if config.get("objective_id") != "dynamic_rgb_chroma_constraint":
        raise ValueError("dynamic BaseColor objective identity changed")
    training = config.get("training")
    if not isinstance(training, Mapping) or int(training.get("steps", 0)) != 10000:
        raise ValueError("dynamic BaseColor training must run 10k")
    checkpoints = [int(value) for value in training.get("checkpoint_steps", [])]
    if checkpoints != [1000, 5000, 10000]:
        raise ValueError("dynamic BaseColor checkpoints must be 1k/5k/10k")
    base_path = (ROOT / str(config.get("base_config", ""))).resolve()
    if not base_path.is_file() or not base_path.is_relative_to(ROOT):
        raise ValueError("dynamic BaseColor base config must remain in the repository")
    if _sha256(base_path) != str(config.get("base_config_sha256")):
        raise ValueError("dynamic BaseColor base config hash mismatch")
    base_config = _load_config(base_path)
    _constraint_config(config)
    return config, base_path, base_config


def _constraint_config(config: Mapping[str, Any]) -> DynamicBaseColorConstraintConfig:
    values = config.get("constraints")
    if not isinstance(values, Mapping):
        raise ValueError("dynamic BaseColor constraints are required")
    return DynamicBaseColorConstraintConfig(
        rgb_rmse_ceiling=float(values.get("rgb_rmse_ceiling", 0.0)),
        chroma_retention_floor=float(values.get("chroma_retention_floor", 0.0)),
        penalty_rho=float(values.get("penalty_rho", 0.0)),
        dual_learning_rate=float(values.get("dual_learning_rate", 0.0)),
        multiplier_max=float(values.get("multiplier_max", 0.0)),
    )


def _identity(
    config_path: Path,
    config: Mapping[str, Any],
    prepared: Mapping[str, Any],
    base_config: Mapping[str, Any],
) -> dict[str, str]:
    objective = {
        "candidate_id": config["candidate_id"],
        "objective_id": config["objective_id"],
        "constraints": config["constraints"],
        "residual_loss": base_config["loss"],
    }
    return {
        "parent_hash": prepared["raw"].artifact_hash,
        "input_hash": prepared["lineage"]["input_sha256"],
        "base_config_hash": prepared["lineage"]["config_sha256"],
        "dynamic_config_hash": _sha256(config_path),
        "objective_hash": _canonical_hash(objective),
        "postprocess_hash": _canonical_hash(dict(base_config["postprocess"], compander=False)),
        "rig_hash": str(base_config["source"]["render_pool_config_sha256"]),
    }


def _save_checkpoint(
    path: Path,
    *,
    step: int,
    state,
    optimizers,
    core_rng: torch.Generator,
    multipliers: torch.Tensor,
    identity: Mapping[str, str],
    constraints: DynamicBaseColorConstraintConfig,
) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "checkpoint_type": CHECKPOINT_TYPE,
            "step": int(step),
            "candidate_id": "BC-dynamic",
            "objective_id": "dynamic_rgb_chroma_constraint",
            "identity": dict(identity),
            "constraints": asdict(constraints),
            "multipliers": multipliers.detach().cpu(),
            "latent": state[0].detach().cpu(),
            "weight": state[1].detach().cpu(),
            "bias": state[2].detach().cpu(),
            "compander_parameters": state[3].detach().cpu(),
            "latent_optimizer": optimizers[0].state_dict(),
            "affine_optimizer": optimizers[1].state_dict(),
            "core_rng_state": core_rng.get_state().cpu(),
        },
        path,
    )
    return _sha256(path)


def _reload_checkpoint(path: Path, identity: Mapping[str, str]) -> Mapping[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if (
        not isinstance(payload, Mapping)
        or payload.get("checkpoint_type") != CHECKPOINT_TYPE
        or payload.get("candidate_id") != "BC-dynamic"
        or payload.get("objective_id") != "dynamic_rgb_chroma_constraint"
        or payload.get("identity") != dict(identity)
    ):
        raise ValueError("dynamic BaseColor checkpoint identity mismatch")
    for name in ("latent", "weight", "bias", "compander_parameters", "multipliers"):
        value = payload.get(name)
        if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
            raise ValueError(f"dynamic BaseColor checkpoint field is invalid: {name}")
    return payload


def run(
    config_path: Path,
    output: Path,
    *,
    max_steps: int | None = None,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("dynamic BaseColor training requires CUDA")
    config, base_path, base_config = _load_dynamic_config(config_path)
    output = _repo_output_path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite dynamic BaseColor output: {output}")
    output.mkdir(parents=True)
    prepared = _prepare(base_config, base_path)
    state, optimizers, core_rng = _new_state(prepared, base_config)
    constraints = _constraint_config(config)
    multipliers = torch.zeros(2, device="cuda", dtype=state[0].dtype)
    identity = _identity(config_path, config, prepared, base_config)

    formal_steps = int(config["training"]["steps"])
    steps = formal_steps if max_steps is None else min(formal_steps, int(max_steps))
    if steps <= 0:
        raise ValueError("dynamic BaseColor steps must be positive")
    checkpoint_steps = set(int(value) for value in config["training"]["checkpoint_steps"])
    checkpoint_steps.add(steps)
    metric_count = min(262144, prepared["valid_indices"].numel())
    metric_positions = torch.linspace(
        0,
        prepared["valid_indices"].numel() - 1,
        metric_count,
        device=prepared["valid_indices"].device,
    ).round().to(torch.int64)
    audit_pairs = [tuple(int(value) for value in pair) for pair in base_config["audit_pairs"]]
    curve: list[dict[str, Any]] = []
    sample_metrics: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    last_checkpoint: Path | None = None
    started = time.perf_counter()

    for step in range(1, steps + 1):
        batch = _draw_batch(prepared, base_config, core_rng)
        terms, context = _loss_terms(
            state,
            prepared,
            base_config,
            batch,
            compander_enabled=False,
        )
        target = postprocess_affine_output(
            prepared["target_valid"][batch[0]],
            compander_parameters=None,
            straight_through=False,
        )
        dynamic = compose_dynamic_basecolor_constraint(
            context["processed_material"].seven[:, :3],
            target.seven[:, :3],
            _residual(terms),
            multipliers,
            constraints,
        )
        if not torch.isfinite(dynamic.total):
            raise FloatingPointError(f"non-finite dynamic BaseColor loss at step {step}")
        optimizers[0].zero_grad(set_to_none=True)
        optimizers[1].zero_grad(set_to_none=True)
        dynamic.total.backward()
        optimizers[0].step()
        optimizers[1].step()
        with torch.no_grad():
            state[0].clamp_(0.0, 1.0)
        update_dynamic_multipliers(
            multipliers,
            dynamic.rgb_violation.detach(),
            dynamic.chroma_violation.detach(),
            constraints,
        )

        if step == 1 or step % int(base_config["training"]["log_interval"]) == 0:
            point = {
                "step": step,
                "loss": float(dynamic.total.detach()),
                "residual": float(dynamic.residual.detach()),
                "rgb_rmse": float(dynamic.rgb_rmse.detach()),
                "chroma_retention": float(dynamic.chroma_retention.detach()),
                "rgb_violation": float(dynamic.rgb_violation.detach()),
                "chroma_violation": float(dynamic.chroma_violation.detach()),
                "rgb_penalty": float(dynamic.rgb_penalty.detach()),
                "chroma_penalty": float(dynamic.chroma_penalty.detach()),
                "rgb_multiplier": float(multipliers[0]),
                "chroma_multiplier": float(multipliers[1]),
                "camera_index": context["camera_index"],
                "light_index": context["light_index"],
                "finite": True,
            }
            curve.append(point)
            print(json.dumps(point, sort_keys=True), flush=True)
        if step % int(base_config["training"]["metric_interval"]) == 0 or step == steps:
            sample_metrics.append(
                {
                    "step": step,
                    **_atlas_metrics(
                        state,
                        prepared,
                        compander_enabled=False,
                        valid_positions=metric_positions,
                    ),
                }
            )
        if step in checkpoint_steps:
            checkpoint = output / "checkpoints" / f"step_{step:05d}" / "checkpoint.pt"
            digest = _save_checkpoint(
                checkpoint,
                step=step,
                state=state,
                optimizers=optimizers,
                core_rng=core_rng,
                multipliers=multipliers,
                identity=identity,
                constraints=constraints,
            )
            last_checkpoint = checkpoint
            trajectory.append(
                {
                    "step": step,
                    "checkpoint": checkpoint.relative_to(output).as_posix(),
                    "checkpoint_sha256": digest,
                    "atlas": _atlas_metrics(state, prepared, compander_enabled=False),
                    "audit_render": _render_metrics(
                        state,
                        prepared,
                        audit_pairs,
                        compander_enabled=False,
                    ),
                    "multipliers": [float(value) for value in multipliers.detach().cpu()],
                    "compander_parameters": [1.0, 0.0],
                }
            )
            _write_json(
                output / "progress.json",
                {
                    "candidate_id": "BC-dynamic",
                    "curve": curve,
                    "trajectory": trajectory,
                    "sample_metrics": sample_metrics,
                },
            )
            print(json.dumps({"checkpoint": step, "sha256": digest}), flush=True)

    if last_checkpoint is None:
        raise RuntimeError("dynamic BaseColor training produced no checkpoint")
    reloaded = _reload_checkpoint(last_checkpoint, identity)
    if int(reloaded["step"]) != steps:
        raise ValueError("dynamic BaseColor final checkpoint step mismatch")
    all_pairs = [
        (camera, light)
        for light in range(len(prepared["lights"]))
        for camera in range(len(prepared["cameras"]))
    ]
    endpoint_render = _render_metrics(
        state,
        prepared,
        all_pairs if steps == formal_steps else audit_pairs,
        compander_enabled=False,
    )
    report = {
        "schema_version": 1,
        "status": f"complete_{steps}_steps",
        "candidate": {
            "candidate_id": "BC-dynamic",
            "objective_id": "dynamic_rgb_chroma_constraint",
            "output_root": output.relative_to(ROOT).as_posix(),
            "compander": False,
        },
        "steps": steps,
        "identity": identity,
        "constraints": asdict(constraints),
        "final_multipliers": [float(value) for value in multipliers.detach().cpu()],
        "runtime_contract": {
            "texture": "2048x2048 linear RGBA8",
            "filtered_samples_per_pixel": 1,
            "decoder": "single unconstrained 4_to_7 affine plus bounded postprocess",
            "instruction_audit": decoder_instruction_audit(C4PostprocessConfig(compander=False)),
            "ue_gpu_timing_validated": False,
        },
        "trajectory": trajectory,
        "sample_metrics": sample_metrics,
        "curve": curve,
        "endpoint": {**trajectory[-1], "full_render_31x6": endpoint_render},
        "wall_seconds": time.perf_counter() - started,
        "final_reload_passed": True,
        "formal_holdout_accessed": False,
        "ue_started": False,
        "yellow_diagnostics": {"selection_metric": False},
    }
    _write_json(output / "training_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--max-steps", type=int)
    arguments = parser.parse_args()
    config, _, _ = _load_dynamic_config(arguments.config)
    output = arguments.output_root or Path(str(config["output_root"]))
    result = run(arguments.config, output, max_steps=arguments.max_steps)
    print(json.dumps({"status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
