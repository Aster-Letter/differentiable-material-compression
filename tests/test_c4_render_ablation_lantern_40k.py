from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest
import torch
import yaml

from cg_frontier.compression.render_ablation_continuation import (
    CHECKPOINT_STEPS,
    OBSERVATION_STEPS,
    load_continuation_checkpoint,
    save_continuation_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/train/c4_render_ablation_lantern_40k_v1.yaml"


def _runner():
    path = ROOT / "scripts/continue_c4_render_ablation_lantern_40k.py"
    spec = importlib.util.spec_from_file_location("lantern_40k_runner_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _bundle_verifier():
    path = ROOT / "scripts/verify_c4_render_ablation_lantern_40k_bundle.py"
    spec = importlib.util.spec_from_file_location("lantern_40k_bundle_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _state():
    latent = torch.nn.Parameter(torch.tensor([[0.2, 0.4, 0.6, 0.8]]))
    weight = torch.nn.Parameter(torch.eye(4))
    bias = torch.nn.Parameter(torch.zeros(4))
    latent_optimizer = torch.optim.Adam([latent], lr=2.0e-4)
    affine_optimizer = torch.optim.Adam([weight, bias], lr=2.0e-5)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(20260814)
    loss = latent.square().sum() + weight.square().sum() + bias.square().sum()
    loss.backward()
    latent_optimizer.step()
    affine_optimizer.step()
    latent_optimizer.zero_grad(set_to_none=True)
    affine_optimizer.zero_grad(set_to_none=True)
    return latent, weight, bias, latent_optimizer, affine_optimizer, rng


def test_frozen_continuation_schedule_and_loss_contract():
    value = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert tuple(value["training"]["continuation_observation_steps"]) == OBSERVATION_STEPS
    assert tuple(value["training"]["continuation_checkpoint_steps"]) == CHECKPOINT_STEPS
    assert value["training"]["source_step"] == 20000
    assert value["training"]["endpoint_step"] == 40000
    assert value["training"]["latent_learning_rate"] == pytest.approx(2.0e-4)
    assert value["training"]["affine_learning_rate"] == pytest.approx(2.0e-5)
    assert value["loss"] == {
        "base_color_l1": 1.0,
        "render_linear": 1.0,
        "render_log": 0.25,
        "normal_cosine": 0.25,
        "roughness_l1": 0.5,
        "metallic_l1": 0.5,
    }
    assert value["arms"] == ["material_only", "material_render"]


def test_continuation_checkpoint_exact_reload_and_lineage_rejection(tmp_path):
    latent, weight, bias, latent_opt, affine_opt, rng = _state()
    identity = {
        "asset_hash": "asset-hash",
        "config_hash": "base-config-hash",
        "parent_hash": "raw-q4-parent-hash",
        "rig_hash": "generic-c4-rig-hash",
        "sampling_contract_hash": "sampling-contract-hash",
    }
    path = tmp_path / "checkpoint.pt"
    digest = save_continuation_checkpoint(
        path,
        arm="material_only",
        step=30000,
        latent=latent,
        weight=weight,
        bias=bias,
        latent_optimizer=latent_opt,
        affine_optimizer=affine_opt,
        rng=rng,
        source_identity=identity,
        initial_rng_hash="initial-rng",
        continuation_config_hash="config-hash",
        source_checkpoint_sha256="source-hash",
    )
    assert len(digest) == 64
    payload = load_continuation_checkpoint(
        path,
        expected_arm="material_only",
        expected_source_identity=identity,
        expected_continuation_config_hash="config-hash",
        expected_source_checkpoint_sha256="source-hash",
    )
    assert payload["step"] == 30000
    assert torch.equal(payload["latent"], latent.detach())
    assert torch.equal(payload["rng_state"], rng.get_state())
    with pytest.raises(ValueError, match="lineage"):
        load_continuation_checkpoint(
            path,
            expected_arm="material_render",
            expected_source_identity=identity,
            expected_continuation_config_hash="config-hash",
            expected_source_checkpoint_sha256="source-hash",
        )


def test_config_rejects_formal_holdout_or_schedule_drift(tmp_path):
    runner = _runner()
    value = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    for mutation in ("holdout", "steps"):
        candidate = copy.deepcopy(value)
        if mutation == "holdout":
            candidate["formal_holdout_access"] = "allowed"
        else:
            candidate["training"]["endpoint_step"] = 39999
        path = tmp_path / f"{mutation}.yaml"
        path.write_text(yaml.safe_dump(candidate, sort_keys=False), encoding="utf-8")
        with pytest.raises(ValueError):
            runner._load_config(path)


def test_remote_scripts_preserve_scow_resource_contract():
    preflight = (ROOT / "scripts/scow_c4_render_ablation_lantern_40k_preflight.slurm").read_text()
    formal = (ROOT / "scripts/scow_c4_render_ablation_lantern_40k_job.slurm").read_text()
    for text in (preflight, formal):
        assert "#SBATCH --partition=Students" in text
        assert "#SBATCH --gres=gpu:RTX5090:1" in text
        assert "#SBATCH --cpus-per-task=4" in text
        assert "--mem" not in text
        assert "--account" not in text
        assert "--qos" not in text
    assert "--time=00:30:00" in preflight
    assert "--time=04:00:00" in formal


def test_incremental_bundle_carries_shared_experiment_io_dependency():
    verifier = _bundle_verifier()

    assert "src/cg_frontier/experiment_io.py" in verifier.PAYLOAD_FILES
