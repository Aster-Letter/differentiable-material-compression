from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import pytest
import torch
import yaml

from cg_frontier.compression.render_ablation_long_continuation import (
    CHECKPOINT_STEPS,
    OBSERVATION_STEPS,
    load_long_continuation_checkpoint,
    save_long_continuation_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/train/c4_render_ablation_lantern_render_160k_v1.yaml"


def _runner():
    path = ROOT / "scripts/continue_c4_render_ablation_lantern_render_160k.py"
    spec = importlib.util.spec_from_file_location("lantern_render_160k_runner_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _bundle_verifier():
    path = ROOT / "scripts/verify_c4_render_ablation_lantern_render_160k_bundle.py"
    spec = importlib.util.spec_from_file_location("lantern_render_160k_bundle_test", path)
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
    return latent, weight, bias, latent_optimizer, affine_optimizer, rng


def test_frozen_single_arm_schedule_and_loss_contract():
    value = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert value["arm"] == "material_render"
    assert value["training"]["source_step"] == 40000
    assert value["training"]["endpoint_step"] == 160000
    assert tuple(value["training"]["continuation_observation_steps"]) == OBSERVATION_STEPS
    assert tuple(value["training"]["continuation_checkpoint_steps"]) == CHECKPOINT_STEPS
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


def test_long_checkpoint_exact_reload_and_lineage_rejection():
    latent, weight, bias, latent_opt, affine_opt, rng = _state()
    identity = {
        "asset_hash": "asset-hash",
        "config_hash": "base-config-hash",
        "parent_hash": "raw-q4-parent-hash",
        "rig_hash": "generic-c4-rig-hash",
        "sampling_contract_hash": "sampling-contract-hash",
    }
    path = ROOT / "outputs/.test-lantern-render-160k-checkpoint.pt"
    try:
        digest = save_long_continuation_checkpoint(
            path,
            step=80000,
            latent=latent,
            weight=weight,
            bias=bias,
            latent_optimizer=latent_opt,
            affine_optimizer=affine_opt,
            rng=rng,
            source_identity=identity,
            continuation_initial_rng_hash="rng-at-40k",
            original_initial_rng_hash="rng-at-zero",
            continuation_config_hash="config-hash",
            source_40k_checkpoint_sha256="source-40k-hash",
        )
        assert len(digest) == 64
        payload = load_long_continuation_checkpoint(
            path,
            expected_source_identity=identity,
            expected_continuation_config_hash="config-hash",
            expected_source_40k_checkpoint_sha256="source-40k-hash",
        )
        assert payload["step"] == 80000
        assert torch.equal(payload["latent"], latent.detach())
        assert torch.equal(payload["rng_state"], rng.get_state())
        with pytest.raises(ValueError, match="lineage"):
            load_long_continuation_checkpoint(
                path,
                expected_source_identity=identity,
                expected_continuation_config_hash="config-hash",
                expected_source_40k_checkpoint_sha256="wrong-source",
            )
    finally:
        path.unlink(missing_ok=True)


def test_config_rejects_arm_or_endpoint_drift():
    runner = _runner()
    runner._load_config(CONFIG)
    value = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    paths = []
    try:
        for mutation in ("arm", "endpoint"):
            candidate = copy.deepcopy(value)
            if mutation == "arm":
                candidate["arm"] = "material_only"
            else:
                candidate["training"]["endpoint_step"] = 159999
            path = ROOT / "outputs" / f".test-lantern-render-160k-{mutation}.yaml"
            paths.append(path)
            path.write_text(yaml.safe_dump(candidate, sort_keys=False), encoding="utf-8")
            with pytest.raises(ValueError):
                runner._load_config(path)
    finally:
        for path in paths:
            path.unlink(missing_ok=True)


def test_scow_scripts_preserve_single_job_resource_contract():
    preflight = (
        ROOT / "scripts/scow_c4_render_ablation_lantern_render_160k_preflight.slurm"
    ).read_text()
    formal = (
        ROOT / "scripts/scow_c4_render_ablation_lantern_render_160k_job.slurm"
    ).read_text()
    submit = (
        ROOT / "scripts/scow_submit_c4_render_ablation_lantern_render_160k.sh"
    ).read_text()
    for text in (preflight, formal):
        assert "#SBATCH --partition=Students" in text
        assert "#SBATCH --gres=gpu:RTX5090:1" in text
        assert "#SBATCH --cpus-per-task=4" in text
        assert "--mem" not in text
        assert "--account" not in text
        assert "--qos" not in text
    assert "--time=00:30:00" in preflight
    assert "--time=04:00:00" in formal
    assert "Refusing to submit: an online job already exists" in submit
    assert "material_only" not in submit


def test_incremental_bundle_carries_shared_experiment_io_dependency():
    verifier = _bundle_verifier()

    assert "src/cg_frontier/experiment_io.py" in verifier.PAYLOAD_FILES
