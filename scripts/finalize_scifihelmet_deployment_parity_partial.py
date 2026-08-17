"""Export an explicitly invalid diagnostic run stopped by user authorization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

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


ROOT_OUTPUT = ROOT / "outputs/compression/scifihelmet/deployment_parity_arc_v1"


def run(*, requested_stop_step: int) -> dict[str, object]:
    root = ROOT_OUTPUT / "dp_relu_fresh"
    checkpoint_path = root / "checkpoint.pt"
    log_path = root / "train.jsonl"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    actual_step = int(checkpoint["step"])
    if checkpoint.get("candidate") != "dp_relu_fresh":
        raise ValueError("partial finalizer only accepts the DP-ReLU checkpoint")
    if actual_step < requested_stop_step:
        raise ValueError("checkpoint predates the requested diagnostic stop")
    decoder = DeploymentParityDecoder(width=8)
    decoder.load_state_dict(checkpoint["decoder"])
    latent = fake_quantize_unorm8(checkpoint["latent"].to(torch.float32))
    latent_name = "latent_dp_relu_fresh_rgba_unorm8.png"
    latent_metadata = export_latent_unorm8_png(latent, root / latent_name)
    decoder_path = root / "decoder_weights.npz"
    np.savez(
        decoder_path,
        **{name: value.detach().cpu().numpy() for name, value in decoder.state_dict().items()},
    )
    requested_record = None
    for line in log_path.read_text(encoding="utf-8").splitlines():
        value = json.loads(line)
        if int(value["step"]) == requested_stop_step:
            requested_record = value
    if requested_record is None or "full_atlas_evaluation" not in requested_record:
        raise RuntimeError("requested diagnostic full-atlas record is missing")
    manifest: dict[str, object] = {
        "schema_version": 1,
        "candidate": "dp_relu_fresh",
        "status": "invalid_partial_user_authorized_diagnostic_stop",
        "valid": False,
        "formal_holdout_accessed": False,
        "stop": {
            "requested_full_atlas_step": requested_stop_step,
            "actual_checkpoint_step": actual_step,
            "reason": "user prioritized ARC after DP metrics plateaued and worsened",
        },
        "initialization": {
            "kind": "shared_fresh_random_state",
            "sha256": checkpoint["initialization_sha256"],
            "legacy_or_rejected_checkpoint_used": False,
        },
        "cost": calculate_deployment_parity_cost(decoder),
        "requested_step_full_atlas_evaluation": requested_record["full_atlas_evaluation"],
        "files": {
            latent_name: latent_metadata,
            "decoder_weights.npz": sha256_file(decoder_path),
            "checkpoint.pt": sha256_file(checkpoint_path),
            "train.jsonl": sha256_file(log_path),
        },
        "deployment_exported": False,
    }
    (root / "training_manifest.json").write_text(
        deterministic_json(manifest), encoding="utf-8", newline="\n"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--requested-stop-step", type=int, default=40_000)
    args = parser.parse_args()
    result = run(requested_stop_step=args.requested_stop_step)
    print(json.dumps({"status": result["status"], "valid": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
