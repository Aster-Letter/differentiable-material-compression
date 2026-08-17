from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from cg_frontier.assets.gltf_core4 import load_gltf_core4_asset
from cg_frontier.assets.gltf_merge import merge_shared_material_meshes
from cg_frontier.compression.render_ablation import (
    FULL_CHECKPOINT_STEPS,
    OBSERVATION_STEPS,
    LossWeights,
    compose_ablation_loss,
    load_ablation_checkpoint,
    paired_sampling_evidence,
    sampling_contract_hash,
    sampling_trajectory_hash,
    save_ablation_checkpoint,
)
from cg_frontier.render import gbuffer as gbuffer_module
import train_c4_render_ablation_20k as runner
import render_c4_render_ablation_summary as summary_runner
import build_c4_render_ablation_bundle as bundle_builder
import verify_c4_render_ablation_bundle as bundle_verifier
import verify_c4_render_ablation_run as run_verifier


CONFIG = ROOT / "configs/train/c4_render_ablation_20k_v1.yaml"
LANTERN_ROOT = ROOT / "assets/source/c4_render_ablation_20k/Lantern"


def test_loss_arms_have_exact_frozen_terms_and_weights() -> None:
    terms = {
        "base_color_l1": torch.tensor(2.0),
        "render_linear": torch.tensor(3.0),
        "render_log": torch.tensor(5.0),
        "normal_cosine": torch.tensor(7.0),
        "roughness_l1": torch.tensor(11.0),
        "metallic_l1": torch.tensor(13.0),
    }
    weights = LossWeights()
    material_only, material_parts = compose_ablation_loss(
        terms, arm="material_only", weights=weights
    )
    material_render, render_parts = compose_ablation_loss(
        terms, arm="material_render", weights=weights
    )
    material = 2.0 + 0.25 * 7.0 + 0.5 * 11.0 + 0.5 * 13.0
    render = 3.0 + 0.25 * 5.0

    assert float(material_only) == pytest.approx(material)
    assert float(material_render) == pytest.approx(material + render)
    assert float(material_parts["render"]) == 0.0
    assert float(material_parts["diagnostic_render"]) == pytest.approx(render)
    assert float(render_parts["render"]) == pytest.approx(render)


@pytest.mark.asset
def test_config_freezes_three_assets_20k_nodes_and_deployment_contract() -> None:
    config = runner._config(CONFIG)
    assert tuple(item["id"] for item in config["assets"]) == (
        "Corset",
        "Lantern",
        "BoomBox",
    )
    assert tuple(config["training"]["observation_steps"]) == OBSERVATION_STEPS
    assert tuple(config["training"]["full_checkpoint_steps"]) == FULL_CHECKPOINT_STEPS
    assert config["formal_holdout_access"] == "forbidden"
    assert "support" not in config["loss"]
    assert tuple(config["arms"]) == ("material_only", "material_render")
    assert tuple(item["tangent_source"] for item in config["assets"]) == (
        "reconstructed_uv",
        "source_gltf",
        "reconstructed_uv",
    )
    assert tuple(item["degenerate_uv_triangles"] for item in config["assets"]) == (
        0,
        3,
        0,
    )
    for item in config["assets"]:
        gltf = ROOT / item["gltf"]
        metadata = ROOT / item["metadata"]
        assert hashlib.sha256(gltf.read_bytes()).hexdigest() == item["gltf_sha256"]
        assert hashlib.sha256(metadata.read_bytes()).hexdigest() == item["metadata_sha256"]


def test_bundle_contains_regular_package_root_for_isolated_campaign_imports() -> None:
    files = bundle_builder.payload_files()
    assert "src/cg_frontier/__init__.py" in files
    assert "src/cg_frontier/compression/__init__.py" in files
    assert "src/cg_frontier/experiment_io.py" in files
    assert "src/cg_frontier/compression/render_ablation.py" in files
    assert "src/cg_frontier/__init__.py" in bundle_verifier.REQUIRED
    assert "src/cg_frontier/experiment_io.py" in bundle_verifier.REQUIRED
    remote_runner = (ROOT / "scripts/remote_run_c4_render_ablation_20k.sh").read_text(
        encoding="utf-8"
    )
    assert 'export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"' in remote_runner
    assert 'assert package.parent == expected' in remote_runner


def test_config_rejects_loss_weight_drift() -> None:
    value = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    value["loss"]["normal_cosine"] = 0.5
    path = ROOT / "tests/.c4-render-ablation-loss-drift.yaml"
    try:
        path.write_text(yaml.safe_dump(value), encoding="utf-8")
        with pytest.raises(ValueError, match="loss weights"):
            runner._config(path)
    finally:
        path.unlink(missing_ok=True)


def test_sampling_contract_and_pair_evidence_are_arm_independent() -> None:
    first = sampling_contract_hash(
        seed=20260813,
        valid_texels=12345,
        training_camera_indices=tuple(range(24)),
        lights=6,
    )
    second = sampling_contract_hash(
        seed=20260813,
        valid_texels=12345,
        training_camera_indices=tuple(range(24)),
        lights=6,
    )
    left = {
        "sampling_contract_hash": first,
        "initial_rng_hash": "initial",
        "final_rng_hash": "final",
        "sampling_trajectory_hash": sampling_trajectory_hash(
            sampling_contract=first,
            initial_rng="initial",
            final_rng="final",
            steps=20000,
        ),
        "steps": 20000,
    }
    right = dict(left, sampling_contract_hash=second)
    assert paired_sampling_evidence(left, right)["identical"] is True
    assert paired_sampling_evidence(left, dict(right, final_rng_hash="different"))["identical"] is False


def test_summary_delta_uses_actual_render_metric_field_names() -> None:
    left = {
        "masked_linear_hdr_mae": 0.02,
        "display_ssim": 0.90,
        "linear_psnr_db": 30.0,
        "display_psnr_db": 35.0,
    }
    right = {
        "masked_linear_hdr_mae": 0.01,
        "display_ssim": 0.95,
        "linear_psnr_db": 33.0,
        "display_psnr_db": 38.0,
    }
    assert summary_runner._delta(left, right) == pytest.approx(
        {
            "masked_linear_hdr_mae": -0.01,
            "display_ssim": 0.05,
            "linear_psnr_db": 3.0,
            "display_psnr_db": 3.0,
        }
    )


def test_10k_checkpoint_exactly_restores_parameters_optimizers_and_rng() -> None:
    def make_state():
        latent = torch.nn.Parameter(torch.tensor([0.2, 0.7], dtype=torch.float64))
        weight = torch.nn.Parameter(torch.tensor([0.5], dtype=torch.float64))
        bias = torch.nn.Parameter(torch.tensor([0.1], dtype=torch.float64))
        latent_optimizer = torch.optim.Adam((latent,), lr=1.0e-3)
        affine_optimizer = torch.optim.Adam((weight, bias), lr=2.0e-4)
        return (latent, weight, bias), (latent_optimizer, affine_optimizer)

    def update(state, optimizers, rng):
        draw = torch.rand((), generator=rng, dtype=torch.float64)
        loss = draw * sum(value.square().sum() for value in state)
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
        return draw, loss.detach()

    identity = {
        "asset_hash": "asset",
        "config_hash": "config",
        "parent_hash": "parent",
        "rig_hash": "rig",
        "sampling_contract_hash": "sampling",
    }
    state, optimizers = make_state()
    rng = torch.Generator().manual_seed(123)
    update(state, optimizers, rng)
    path = ROOT / "tests/.c4-render-ablation-checkpoint.pt"
    try:
        save_ablation_checkpoint(
            path,
            asset="Corset",
            arm="material_only",
            step=10000,
            latent=state[0],
            weight=state[1],
            bias=state[2],
            latent_optimizer=optimizers[0],
            affine_optimizer=optimizers[1],
            rng=rng,
            identity=identity,
            initial_rng_hash="initial",
        )
        continuous = update(state, optimizers, rng)

        resumed_state, resumed_optimizers = make_state()
        resumed_rng = torch.Generator()
        payload = load_ablation_checkpoint(
            path,
            expected_asset="Corset",
            expected_arm="material_only",
            expected_identity=identity,
        )
        for parameter, name in zip(resumed_state, ("latent", "weight", "bias"), strict=True):
            parameter.data.copy_(payload[name])
        for optimizer, name in zip(
            resumed_optimizers, ("latent_optimizer", "affine_optimizer"), strict=True
        ):
            optimizer.load_state_dict(payload[name])
        resumed_rng.set_state(payload["rng_state"])
        resumed = update(resumed_state, resumed_optimizers, resumed_rng)
        assert all(torch.equal(left, right) for left, right in zip(continuous, resumed, strict=True))
        assert all(torch.equal(left, right) for left, right in zip(state, resumed_state, strict=True))
    finally:
        path.unlink(missing_ok=True)


def test_formal_verifier_rejects_a_replaced_checkpoint_with_valid_metadata(tmp_path) -> None:
    pair_root = tmp_path / "Corset"
    summary_root = tmp_path / "Corset-summary"
    identity = {"parent": "raw-q4", "config": "frozen", "rig": "generic"}
    for arm in ("material_only", "material_render"):
        arm_root = pair_root / arm
        arm_root.mkdir(parents=True)
        checkpoints = {}
        for step in (10000, 20000):
            checkpoint_path = arm_root / f"checkpoint-{step}.pt"
            torch.save({"step": step, "identity": identity}, checkpoint_path)
            checkpoints[str(step)] = {
                "path": checkpoint_path.relative_to(ROOT).as_posix(),
                "sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
            }
        report = {
            "status": "complete_20k",
            "steps": 20000,
            "formal_holdout_accessed": False,
            "audit_used_for_training": False,
            "early_stopping": False,
            "observation_steps": [1000, 5000, 10000, 15000, 20000],
            "checkpoints": checkpoints,
            "identity": identity,
            "endpoint": {
                "audit_render": {"case_count": 42},
                "train_render": {"case_count": 144},
            },
        }
        (arm_root / "training_report.json").write_text(
            json.dumps(report), encoding="utf-8"
        )
    (pair_root / "paired_summary.json").write_text(
        json.dumps({"paired_sampling_evidence": {"identical": True}}),
        encoding="utf-8",
    )
    summary_root.mkdir()
    (summary_root / "summary.json").write_text(
        json.dumps(
            {"status": "complete_paired_20k_summary", "winner_selected": False}
        ),
        encoding="utf-8",
    )

    replaced = pair_root / "material_only" / "checkpoint-10000.pt"
    torch.save({"step": 10000, "identity": identity, "replacement": True}, replaced)

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        run_verifier.verify_formal(
            tmp_path,
            pair_root,
            summary_root,
            "Corset",
            "formal-job",
            "preflight-job",
        )


@pytest.mark.asset
def test_lantern_merge_bakes_three_meshes_and_preserves_valid_geometry() -> None:
    source = LANTERN_ROOT / "upstream/glTF/Lantern.gltf"
    mesh, manifest = merge_shared_material_meshes(source)
    assert manifest["source_meshes"] == 3
    assert manifest["source_primitives"] == 3
    assert mesh.positions.shape == (4145, 3)
    assert mesh.triangles.shape == (5394, 3)
    assert int(mesh.triangles.min()) == 0
    assert int(mesh.triangles.max()) < mesh.positions.shape[0]
    assert np.max(np.abs(np.linalg.norm(mesh.normals, axis=1) - 1.0)) < 2.0e-6
    assert np.max(np.abs(np.linalg.norm(mesh.tangents[:, :3], axis=1) - 1.0)) < 2.0e-6
    assert set(np.unique(mesh.tangents[:, 3])).issubset({-1.0, 1.0})


@pytest.mark.asset
def test_lantern_derived_asset_excludes_emissive_and_hashes_are_frozen() -> None:
    manifest_path = LANTERN_ROOT / "derived/derived_manifest.json"
    manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert manifest["derivation"]["output_meshes"] == 1
    assert manifest["derivation"]["output_primitives"] == 1
    assert manifest["derivation"]["texture_resampling"] is False
    assert manifest["derivation"]["emissive"]["policy"] == (
        "excluded_from_reference_pca_and_both_training_arms"
    )
    assert manifest["derivation"]["emissive"]["max_rgb_gt_0_05_fraction"] == pytest.approx(
        0.03266143798828125
    )
    derived = LANTERN_ROOT / "derived/glTF"
    for name, expected in manifest["derived_files"].items():
        assert hashlib.sha256((derived / name).read_bytes()).hexdigest() == expected
    asset = load_gltf_core4_asset(derived / "Lantern.gltf")
    assert asset.mesh.positions.shape == (4145, 3)
    assert asset.manifest["core4"]["emissive"] == "excluded"
    tangents, metadata = gbuffer_module.select_render_tangents(
        asset.mesh, source="source_gltf"
    )
    assert tangents is asset.mesh.tangents
    assert metadata["render_source"] == "source_gltf"
    assert metadata["uv_reconstruction_attempted"] is False


@pytest.mark.asset
def test_source_tangent_policy_fails_closed_for_invalid_or_unknown_inputs() -> None:
    asset = load_gltf_core4_asset(
        LANTERN_ROOT / "derived/glTF/Lantern.gltf"
    )
    invalid = type(asset.mesh)(
        positions=asset.mesh.positions,
        normals=asset.mesh.normals,
        tangents=np.array(asset.mesh.tangents, copy=True),
        texcoords=asset.mesh.texcoords,
        triangles=asset.mesh.triangles,
    )
    invalid.tangents[0, 3] = 0.0
    with pytest.raises(ValueError, match="missing or invalid"):
        gbuffer_module.select_render_tangents(invalid, source="source_gltf")
    with pytest.raises(ValueError, match="unsupported tangent source"):
        gbuffer_module.select_render_tangents(asset.mesh, source="automatic")


def test_repository_path_and_formal_holdout_guards_fail_closed() -> None:
    with pytest.raises(ValueError, match="escapes"):
        runner._repo_path("../formal_holdout/secret.json", "test")
    value = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    value["formal_holdout_access"] = "allowed"
    path = ROOT / "tests/.c4-render-ablation-holdout-drift.yaml"
    try:
        path.write_text(yaml.safe_dump(value), encoding="utf-8")
        with pytest.raises(ValueError, match="formal holdout"):
            runner._config(path)
    finally:
        path.unlink(missing_ok=True)
