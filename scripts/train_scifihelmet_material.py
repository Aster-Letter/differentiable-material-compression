"""CLI entry point for bounded material-domain latent/decoder fitting."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cg_frontier.compression.material import keep_system_awake, load_core4_targets, train_material_model


def main() -> int:
    """Train only explicitly selected configured models under their time limits."""

    parser = argparse.ArgumentParser(description="Train bounded RGBA-latent SciFiHelmet material models.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train/scifihelmet_material.yaml")
    parser.add_argument("--models", nargs="+", choices=("linear", "tiny_mlp"))
    parser.add_argument("--steps", type=int, help="Override steps for smoke/diagnostics.")
    parser.add_argument("--output-dir", type=Path, help="Override output root.")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    device_name = config["training"]["device"]
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("training config requires CUDA, but torch.cuda.is_available() is false")
    device = torch.device(device_name)
    core4_dir = (ROOT / config["asset"]["core4_dir"]).resolve()
    output_root = (args.output_dir or (ROOT / config["output_dir"])).resolve()
    with keep_system_awake():
        targets = load_core4_targets(core4_dir, device=device)
        selected = args.models or [entry["kind"] for entry in config["models"]]
        model_configs = {entry["kind"]: entry for entry in config["models"]}
        for kind in selected:
            entry = model_configs[kind]
            result = train_material_model(
                kind=kind,
                targets=targets,
                output_dir=output_root / kind,
                seed=int(config["training"]["seed"]),
                steps=args.steps or int(entry["steps"]),
                batch_size=int(config["training"]["batch_size"]),
                latent_learning_rate=float(entry["latent_learning_rate"]),
                decoder_learning_rate=float(entry["decoder_learning_rate"]),
                weights={name: float(value) for name, value in config["loss"].items()},
                log_interval=int(config["training"]["log_interval"]),
                checkpoint_interval=int(config["training"]["checkpoint_interval"]),
                evaluation_chunk_size=int(config["training"]["evaluation_chunk_size"]),
                max_minutes=float(entry["max_minutes"]),
                resume=args.resume,
            )
            print(yaml.safe_dump(result, sort_keys=False, allow_unicode=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
