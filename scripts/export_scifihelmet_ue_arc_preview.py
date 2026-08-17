"""Export isolated UE packages for the ARC 10k and user-stopped 81k snapshots."""

from __future__ import annotations

import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cg_frontier.compression.artifact_analysis import deterministic_json, sha256_file  # noqa: E402
from cg_frontier.compression.deployment_parity import (  # noqa: E402
    DeploymentParityDecoder,
    calculate_deployment_parity_cost,
)
from cg_frontier.compression.render_loss import (  # noqa: E402
    export_latent_unorm8_png,
    fake_quantize_unorm8,
)
from cg_frontier.compression.ue_export import (  # noqa: E402
    EXPECTED_ARRAYS,
    generate_custom_hlsl,
    parse_generated_hlsl_constants,
)


RUN_ROOT = ROOT / "outputs/compression/scifihelmet/deployment_parity_arc_v1/arc_relu_fresh"
OUTPUT_ROOT = ROOT / "outputs/deployment/scifihelmet/arc_preview"
LOG_PATH = RUN_ROOT / "train.jsonl"
SNAPSHOTS = (
    {
        "selection": "arc_010k_early",
        "checkpoint": RUN_ROOT / "checkpoints/step_010000.pt",
        "evaluation_step": 10_000,
        "label": "ARC_010k",
    },
    {
        "selection": "arc_081k_user_stopped",
        "checkpoint": RUN_ROOT / "checkpoint.pt",
        "evaluation_step": 80_000,
        "label": "ARC_081k",
    },
)


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8", newline="\n")


def _full_atlas_record(step: int) -> dict[str, Any]:
    for line in LOG_PATH.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if int(record["step"]) == step and "full_atlas_evaluation" in record:
            return record["full_atlas_evaluation"]
    raise RuntimeError(f"full-atlas evaluation is missing for step {step}")


def _export_snapshot(spec: dict[str, Any]) -> dict[str, Any]:
    checkpoint_path = Path(spec["checkpoint"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint.get("candidate") != "arc_relu_fresh":
        raise ValueError(f"unexpected checkpoint candidate: {checkpoint.get('candidate')}")
    actual_step = int(checkpoint["step"])
    evaluation_step = int(spec["evaluation_step"])
    if actual_step < evaluation_step:
        raise ValueError(f"checkpoint step {actual_step} predates evaluation {evaluation_step}")

    decoder = DeploymentParityDecoder(width=8)
    decoder.load_state_dict(checkpoint["decoder"])
    state = decoder.state_dict()
    arrays = {
        "network.0.weight": state["hidden.weight"].detach().cpu().numpy().astype(np.float32, copy=True),
        "network.0.bias": state["hidden.bias"].detach().cpu().numpy().astype(np.float32, copy=True),
        "network.2.weight": state["output.weight"].detach().cpu().numpy().astype(np.float32, copy=True),
        "network.2.bias": state["output.bias"].detach().cpu().numpy().astype(np.float32, copy=True),
    }
    if set(arrays) != set(EXPECTED_ARRAYS):
        raise ValueError(f"unexpected decoder arrays: {sorted(arrays)}")
    for name, shape in EXPECTED_ARRAYS.items():
        if arrays[name].shape != shape or not np.isfinite(arrays[name]).all():
            raise ValueError(f"invalid decoder array {name}: {arrays[name].shape}")

    label = str(spec["label"])
    output_dir = OUTPUT_ROOT / str(spec["selection"])
    output_dir.mkdir(parents=True, exist_ok=True)
    latent_name = f"T_SciFiHelmet_{label}_RGBA8.png"
    decoder_name = "decoder_weights.npz"
    hlsl_name = f"M_SciFiHelmet_{label}.custom.hlsl"
    latent_path = output_dir / latent_name
    decoder_path = output_dir / decoder_name
    hlsl_path = output_dir / hlsl_name

    latent_metadata = export_latent_unorm8_png(
        fake_quantize_unorm8(checkpoint["latent"].to(torch.float32)), latent_path
    )
    np.savez(decoder_path, **arrays)
    decoder_sha = sha256_file(decoder_path)
    metadata = {
        "latent_sha256": latent_metadata["sha256"],
        "decoder_sha256": decoder_sha,
    }
    hlsl = generate_custom_hlsl(arrays, metadata)
    reparsed = parse_generated_hlsl_constants(hlsl)
    for name in EXPECTED_ARRAYS:
        if not np.array_equal(arrays[name], reparsed[name]):
            raise ValueError(f"HLSL round-trip mismatch for {name}")
    _write(hlsl_path, hlsl)

    cost = calculate_deployment_parity_cost(decoder)
    if cost["parameters"] != 103 or cost["weight_bytes_float32"] != 412:
        raise ValueError(f"unexpected ARC decoder cost: {cost}")
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "selection": spec["selection"],
        "candidate": "arc_relu_fresh",
        "status": "diagnostic_preview_not_deployment_winner",
        "formal_holdout_accessed": False,
        "checkpoint": {
            "repo_relative_path": checkpoint_path.relative_to(ROOT).as_posix(),
            "actual_step": actual_step,
            "sha256": sha256_file(checkpoint_path),
        },
        "referenced_full_atlas_step": evaluation_step,
        "full_atlas_evaluation": _full_atlas_record(evaluation_step),
        "initialization_sha256": checkpoint["initialization_sha256"],
        "decoder": cost,
        "runtime": {
            "texture_samples": 1,
            "order": "RGBA8 bilinear -> 4x8x7 ReLU -> Core4 postprocess",
            "actual_resident_target_bytes": 16 * 1024 * 1024,
            "ambient_occlusion": "excluded",
            "normal_y_bridge": "UE applies Y * -1 exactly once",
        },
        "files": {
            latent_name: latent_metadata,
            decoder_name: decoder_sha,
            hlsl_name: sha256_file(hlsl_path),
        },
    }
    _write(output_dir / "deployment_manifest.json", deterministic_json(manifest))
    return {
        "selection": spec["selection"],
        "actual_step": actual_step,
        "evaluation_step": evaluation_step,
        "output_dir": output_dir.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256_file(output_dir / "deployment_manifest.json"),
    }


def main() -> int:
    exported = [_export_snapshot(spec) for spec in SNAPSHOTS]
    index = {
        "schema_version": 1,
        "purpose": "isolated ARC early-versus-late UE diagnostic preview",
        "formal_holdout_accessed": False,
        "snapshots": exported,
    }
    _write(OUTPUT_ROOT / "preview_index.json", deterministic_json(index))
    print(json.dumps(index, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
