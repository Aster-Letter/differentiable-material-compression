from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cg_frontier.compression.affine_pca import (
    calibrate_pca_safe,
    encode_pca_latent,
    fit_uniform_valid_pca,
)
from cg_frontier.compression.affine_material import certify_affine
from cg_frontier.compression.affine_color import (
    ColorGuardObjective,
    ColorRiskObjective,
    build_color_hue_partition,
    build_color_quantile_partition,
)
from cg_frontier.compression.affine_training import (
    InMemoryCheckpointStorage,
    begin_candidate_continuation,
    candidate_parameter_trend,
    candidate_manifest,
    checkpoint_candidate,
    create_color_candidates,
    create_color_risk_candidates,
    create_paired_candidates,
    draw_training_batch,
    plan_training_observations,
    resume_candidate,
    run_candidate_training,
    train_candidate_step,
    time_candidate_steps,
    write_candidate_checkpoint,
)


def _synthetic_p0():
    generator = torch.Generator().manual_seed(151)
    target = torch.rand(5, 6, 7, generator=generator, dtype=torch.float64)
    valid = torch.ones(5, 6, dtype=torch.bool)
    return calibrate_pca_safe(
        encode_pca_latent(fit_uniform_valid_pca(target, valid)), margin=1.0e-3
    )


def test_paired_candidates_are_independent_children_of_one_frozen_p0() -> None:
    p0 = _synthetic_p0()
    children = create_paired_candidates(
        p0,
        core_seed=157,
        cube_seed=163,
        config_hash="config-a",
        input_hash="input-a",
        latent_learning_rate=1.0e-3,
        affine_learning_rate=2.0e-3,
    )

    assert set(children) == {"L0", "L1", "L2"}
    assert all(child.parent_p0_hash == p0.safe.artifact_hash for child in children.values())
    assert [children[name].objective_id for name in ("L0", "L1", "L2")] == [
        "material+helmet",
        "material+helmet+tv",
        "material+helmet+cube",
    ]
    assert all(child.optimizer_updates == 0 for child in children.values())
    assert all(
        torch.equal(
            child.latent.detach(), p0.safe.latent_unorm8.to(torch.float64) / 255.0
        )
        for child in children.values()
    )

    assert len({child.latent.data_ptr() for child in children.values()}) == 3
    parameter_ptrs = [
        {parameter.data_ptr() for parameter in child.decoder.parameters()}
        for child in children.values()
    ]
    assert parameter_ptrs[0].isdisjoint(parameter_ptrs[1])
    assert parameter_ptrs[0].isdisjoint(parameter_ptrs[2])
    assert parameter_ptrs[1].isdisjoint(parameter_ptrs[2])
    assert len({id(child.latent_optimizer) for child in children.values()}) == 3
    assert len({id(child.affine_optimizer) for child in children.values()}) == 3
    assert len({id(child.core_rng) for child in children.values()}) == 3
    first_batches = [
        torch.randint(0, 30, (8,), generator=child.core_rng)
        for child in children.values()
    ]
    assert torch.equal(first_batches[0], first_batches[1])
    assert torch.equal(first_batches[0], first_batches[2])
    assert children["L0"].cube_rng is None
    assert children["L1"].cube_rng is None
    assert children["L2"].cube_rng is not None


def test_legacy_checkpoint_schema_and_hash_are_repeatable() -> None:
    state = create_paired_candidates(
        _synthetic_p0(),
        core_seed=157,
        cube_seed=163,
        config_hash="config-a",
        input_hash="input-a",
        latent_learning_rate=1.0e-3,
        affine_learning_rate=2.0e-3,
    )["L0"]

    checkpoint = checkpoint_candidate(state)

    assert checkpoint["schema_version"] == 1
    assert "color_rng_state" not in checkpoint
    assert "color_partition_hash" not in checkpoint
    repeated = checkpoint_candidate(state)
    assert checkpoint["checkpoint_hash"] == repeated["checkpoint_hash"]
    assert len(str(checkpoint["checkpoint_hash"])) == 64


def test_color_candidates_are_independent_schema_v2_children() -> None:
    p0 = _synthetic_p0()
    children = create_color_candidates(
        p0,
        core_seed=211,
        color_seed=223,
        color_partition_hash="partition-a",
        config_hash="config-color",
        input_hash="input-a",
        latent_learning_rate=1.0e-3,
        affine_learning_rate=2.0e-3,
    )

    assert set(children) == {"C0", "C1", "C2"}
    assert [children[name].objective_id for name in ("C0", "C1", "C2")] == [
        "material+helmet",
        "material+helmet+opponent",
        "material+helmet+opponent+pair",
    ]
    assert len({id(child.color_rng) for child in children.values()}) == 3
    first_draws = [
        torch.rand(8, generator=child.color_rng) for child in children.values()
    ]
    assert torch.equal(first_draws[0], first_draws[1])
    assert torch.equal(first_draws[0], first_draws[2])
    checkpoint = checkpoint_candidate(children["C0"])
    assert checkpoint["schema_version"] == 2
    assert checkpoint["color_partition_hash"] == "partition-a"
    assert isinstance(checkpoint["color_rng_state"], torch.Tensor)
    repeated = checkpoint_candidate(children["C0"])
    assert checkpoint["checkpoint_hash"] == repeated["checkpoint_hash"]
    assert len(str(checkpoint["checkpoint_hash"])) == 64


def test_color_risk_candidates_are_independent_schema_v3_children() -> None:
    p0 = _synthetic_p0()
    children = create_color_risk_candidates(
        p0,
        core_seed=211,
        color_seed=223,
        color_partition_hash="partition-a",
        color_group_hash="groups-a",
        config_hash="config-risk",
        input_hash="input-a",
        latent_learning_rate=1.0e-3,
        affine_learning_rate=2.0e-3,
    )

    assert set(children) == {
        "G0-mean",
        "G1-yc-cvar25",
        "G2-hue8-macro",
        "G3-cvar25-hue8",
    }
    assert len({id(child.color_rng) for child in children.values()}) == 4
    core_states = [child.core_rng.get_state() for child in children.values()]
    color_states = [child.color_rng.get_state() for child in children.values()]
    assert all(torch.equal(core_states[0], value) for value in core_states[1:])
    assert all(torch.equal(color_states[0], value) for value in color_states[1:])
    checkpoint = checkpoint_candidate(children["G0-mean"])
    assert checkpoint["schema_version"] == 3
    assert checkpoint["color_partition_hash"] == "partition-a"
    assert checkpoint["color_group_hash"] == "groups-a"

    resumed = resume_candidate(
        checkpoint,
        p0,
        expected_parent_p0_hash=p0.safe.artifact_hash,
        expected_config_hash="config-risk",
        expected_input_hash="input-a",
        expected_color_partition_hash="partition-a",
        expected_color_group_hash="groups-a",
    )
    assert resumed.color_group_hash == "groups-a"
    with pytest.raises(ValueError, match="color group hash mismatch"):
        resume_candidate(
            checkpoint,
            p0,
            expected_parent_p0_hash=p0.safe.artifact_hash,
            expected_config_hash="config-risk",
            expected_input_hash="input-a",
            expected_color_partition_hash="partition-a",
            expected_color_group_hash="wrong",
        )


def test_color_checkpoint_resume_reproduces_next_core_and_color_batch() -> None:
    p0 = _synthetic_p0()
    partition = build_color_quantile_partition(
        torch.rand(30, 3, generator=torch.Generator().manual_seed(227))
    )
    common = dict(
        core_seed=229,
        color_seed=233,
        color_partition_hash=partition.partition_hash,
        config_hash="config-color",
        input_hash="input-a",
        latent_learning_rate=1.0e-3,
        affine_learning_rate=2.0e-3,
    )
    continuous = create_color_candidates(p0, **common)["C0"]
    interrupted = create_color_candidates(p0, **common)["C0"]
    for _ in range(3):
        continuous_batch = draw_training_batch(
            continuous,
            texel_count=30,
            batch_size=8,
            cube_sample_count=0,
            color_partition=partition,
            color_batch_size=8,
        )
        interrupted_batch = draw_training_batch(
            interrupted,
            texel_count=30,
            batch_size=8,
            cube_sample_count=0,
            color_partition=partition,
            color_batch_size=8,
        )
        assert torch.equal(continuous_batch.core_indices, interrupted_batch.core_indices)
        assert torch.equal(continuous_batch.color_indices, interrupted_batch.color_indices)
    resumed = resume_candidate(
        checkpoint_candidate(interrupted),
        p0,
        expected_parent_p0_hash=p0.safe.artifact_hash,
        expected_config_hash="config-color",
        expected_input_hash="input-a",
        expected_color_partition_hash=partition.partition_hash,
    )

    continuous_next = draw_training_batch(
        continuous,
        texel_count=30,
        batch_size=8,
        cube_sample_count=0,
        color_partition=partition,
        color_batch_size=8,
    )
    resumed_next = draw_training_batch(
        resumed,
        texel_count=30,
        batch_size=8,
        cube_sample_count=0,
        color_partition=partition,
        color_batch_size=8,
    )

    assert torch.equal(continuous_next.core_indices, resumed_next.core_indices)
    assert torch.equal(continuous_next.color_indices, resumed_next.color_indices)
    assert torch.equal(continuous_next.color_bin_ids, resumed_next.color_bin_ids)


def test_cube_sampling_only_advances_l2_cube_rng_and_not_core_rng() -> None:
    children = create_paired_candidates(
        _synthetic_p0(),
        core_seed=167,
        cube_seed=173,
        config_hash="config-a",
        input_hash="input-a",
        latent_learning_rate=1.0e-3,
        affine_learning_rate=2.0e-3,
    )
    cube_state_before = children["L2"].cube_rng.get_state().clone()

    batches = {
        name: draw_training_batch(
            child, texel_count=30, batch_size=8, cube_sample_count=5
        )
        for name, child in children.items()
    }

    assert torch.equal(batches["L0"].core_indices, batches["L1"].core_indices)
    assert torch.equal(batches["L0"].core_indices, batches["L2"].core_indices)
    assert batches["L0"].cube_samples is None
    assert batches["L1"].cube_samples is None
    assert batches["L2"].cube_samples is not None
    assert not torch.equal(cube_state_before, children["L2"].cube_rng.get_state())
    next_core = [
        torch.randint(0, 30, (8,), generator=child.core_rng)
        for child in children.values()
    ]
    assert torch.equal(next_core[0], next_core[1])
    assert torch.equal(next_core[0], next_core[2])


def _base_objective(state, batch):
    decoded = state.decoder(state.latent[batch.core_indices])
    seven = torch.cat(
        (
            decoded.base_color_linear,
            decoded.normal_xy,
            decoded.roughness,
            decoded.metallic,
        ),
        dim=-1,
    )
    target = batch.core_indices.to(seven.dtype)[:, None] / 30.0
    return torch.mean((seven - target) ** 2), {"base": torch.mean((seven - target) ** 2)}


def _color_objective(state, batch):
    base, terms = _base_objective(state, batch)
    if batch.color_indices is None:
        raise AssertionError("color batch is required")
    color = torch.mean(state.latent.reshape(-1, 4)[batch.color_indices].square())
    return base + 0.25 * color, {**terms, "color": color}


def _assert_nested_equal(first, second) -> None:
    if isinstance(first, torch.Tensor):
        assert torch.equal(first, second)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            _assert_nested_equal(first[key], second[key])
    elif isinstance(first, (list, tuple)):
        assert len(first) == len(second)
        for left, right in zip(first, second):
            _assert_nested_equal(left, right)
    else:
        assert first == second


def test_checkpoint_resume_next_step_matches_continuous_training_exactly() -> None:
    p0 = _synthetic_p0()
    common = dict(
        core_seed=179,
        cube_seed=181,
        config_hash="config-a",
        input_hash="input-a",
        latent_learning_rate=1.0e-3,
        affine_learning_rate=2.0e-3,
    )
    continuous = create_paired_candidates(p0, **common)["L0"]
    interrupted = create_paired_candidates(p0, **common)["L0"]

    for _ in range(3):
        train_candidate_step(
            continuous, _base_objective, texel_count=30, batch_size=8, cube_sample_count=0
        )
        train_candidate_step(
            interrupted, _base_objective, texel_count=30, batch_size=8, cube_sample_count=0
        )
    checkpoint = checkpoint_candidate(interrupted)
    resumed = resume_candidate(
        checkpoint,
        p0,
        expected_parent_p0_hash=p0.safe.artifact_hash,
        expected_config_hash="config-a",
        expected_input_hash="input-a",
    )
    continuous_step = train_candidate_step(
        continuous, _base_objective, texel_count=30, batch_size=8, cube_sample_count=0
    )
    resumed_step = train_candidate_step(
        resumed, _base_objective, texel_count=30, batch_size=8, cube_sample_count=0
    )

    assert torch.equal(continuous_step.batch.core_indices, resumed_step.batch.core_indices)
    assert continuous_step.loss == resumed_step.loss
    assert torch.equal(continuous.latent, resumed.latent)
    for left, right in zip(continuous.decoder.parameters(), resumed.decoder.parameters()):
        assert torch.equal(left, right)
    _assert_nested_equal(
        continuous.latent_optimizer.state_dict(), resumed.latent_optimizer.state_dict()
    )
    _assert_nested_equal(
        continuous.affine_optimizer.state_dict(), resumed.affine_optimizer.state_dict()
    )


def test_color_training_resume_matches_continuous_update_exactly() -> None:
    p0 = _synthetic_p0()
    partition = build_color_quantile_partition(
        torch.rand(30, 3, generator=torch.Generator().manual_seed(239))
    )
    common = dict(
        core_seed=241,
        color_seed=251,
        color_partition_hash=partition.partition_hash,
        config_hash="config-color",
        input_hash="input-a",
        latent_learning_rate=1.0e-3,
        affine_learning_rate=2.0e-3,
    )
    continuous = create_color_candidates(p0, **common)["C2"]
    interrupted = create_color_candidates(p0, **common)["C2"]
    for _ in range(3):
        train_candidate_step(
            continuous,
            _color_objective,
            texel_count=30,
            batch_size=8,
            cube_sample_count=0,
            color_partition=partition,
            color_batch_size=8,
        )
        train_candidate_step(
            interrupted,
            _color_objective,
            texel_count=30,
            batch_size=8,
            cube_sample_count=0,
            color_partition=partition,
            color_batch_size=8,
        )
    resumed = resume_candidate(
        checkpoint_candidate(interrupted),
        p0,
        expected_parent_p0_hash=p0.safe.artifact_hash,
        expected_config_hash="config-color",
        expected_input_hash="input-a",
        expected_color_partition_hash=partition.partition_hash,
    )

    continuous_step = train_candidate_step(
        continuous,
        _color_objective,
        texel_count=30,
        batch_size=8,
        cube_sample_count=0,
        color_partition=partition,
        color_batch_size=8,
    )
    resumed_step = train_candidate_step(
        resumed,
        _color_objective,
        texel_count=30,
        batch_size=8,
        cube_sample_count=0,
        color_partition=partition,
        color_batch_size=8,
    )

    assert torch.equal(continuous_step.batch.core_indices, resumed_step.batch.core_indices)
    assert torch.equal(continuous_step.batch.color_indices, resumed_step.batch.color_indices)
    assert continuous_step.loss == resumed_step.loss
    assert torch.equal(continuous.latent, resumed.latent)
    for left, right in zip(continuous.decoder.parameters(), resumed.decoder.parameters()):
        assert torch.equal(left, right)
    _assert_nested_equal(
        continuous.latent_optimizer.state_dict(), resumed.latent_optimizer.state_dict()
    )
    _assert_nested_equal(
        continuous.affine_optimizer.state_dict(), resumed.affine_optimizer.state_dict()
    )


def test_color_risk_v3_resume_matches_continuous_update_exactly() -> None:
    p0 = _synthetic_p0()
    source = torch.rand(30, 3, generator=torch.Generator().manual_seed(313))
    partition = build_color_quantile_partition(source)
    hue_partition = build_color_hue_partition(source, partition, min_group_size=1)
    common = dict(
        core_seed=317,
        color_seed=331,
        color_partition_hash=partition.partition_hash,
        color_group_hash=hue_partition.group_hash,
        config_hash="config-risk",
        input_hash="input-a",
        latent_learning_rate=1.0e-3,
        affine_learning_rate=2.0e-3,
    )
    objective = ColorRiskObjective(
        _base_objective,
        valid_flat_indices=torch.arange(30),
        source_base_color=source,
        yc_partition=partition,
        hue_partition=hue_partition,
        mean_scale=8.4928470181,
        cvar_scale=3.0,
        hue_scale=4.0,
    )
    continuous = create_color_risk_candidates(p0, **common)["G0-mean"]
    interrupted = create_color_risk_candidates(p0, **common)["G0-mean"]
    for _ in range(3):
        train_candidate_step(
            continuous,
            objective,
            texel_count=30,
            batch_size=8,
            cube_sample_count=0,
            color_partition=partition,
            color_batch_size=8,
        )
        train_candidate_step(
            interrupted,
            objective,
            texel_count=30,
            batch_size=8,
            cube_sample_count=0,
            color_partition=partition,
            color_batch_size=8,
        )
    resumed = resume_candidate(
        checkpoint_candidate(interrupted),
        p0,
        expected_parent_p0_hash=p0.safe.artifact_hash,
        expected_config_hash="config-risk",
        expected_input_hash="input-a",
        expected_color_partition_hash=partition.partition_hash,
        expected_color_group_hash=hue_partition.group_hash,
    )

    continuous_step = train_candidate_step(
        continuous,
        objective,
        texel_count=30,
        batch_size=8,
        cube_sample_count=0,
        color_partition=partition,
        color_batch_size=8,
    )
    resumed_step = train_candidate_step(
        resumed,
        objective,
        texel_count=30,
        batch_size=8,
        cube_sample_count=0,
        color_partition=partition,
        color_batch_size=8,
    )

    assert torch.equal(continuous_step.batch.core_indices, resumed_step.batch.core_indices)
    assert torch.equal(continuous_step.batch.color_indices, resumed_step.batch.color_indices)
    assert continuous_step.loss == resumed_step.loss
    assert continuous_step.terms == resumed_step.terms
    assert torch.equal(continuous.latent, resumed.latent)
    assert torch.equal(continuous.core_rng.get_state(), resumed.core_rng.get_state())
    assert torch.equal(continuous.color_rng.get_state(), resumed.color_rng.get_state())
    for left, right in zip(continuous.decoder.parameters(), resumed.decoder.parameters()):
        assert torch.equal(left, right)
    _assert_nested_equal(
        continuous.latent_optimizer.state_dict(), resumed.latent_optimizer.state_dict()
    )
    _assert_nested_equal(
        continuous.affine_optimizer.state_dict(), resumed.affine_optimizer.state_dict()
    )


def test_c0_color_wrapper_matches_legacy_objective_update_exactly() -> None:
    p0 = _synthetic_p0()
    source_rgb = torch.rand(30, 3, generator=torch.Generator().manual_seed(257))
    partition = build_color_quantile_partition(source_rgb)
    legacy = create_paired_candidates(
        p0,
        core_seed=263,
        cube_seed=269,
        config_hash="config-legacy",
        input_hash="input-a",
        latent_learning_rate=1.0e-3,
        affine_learning_rate=2.0e-3,
    )["L0"]
    color = create_color_candidates(
        p0,
        core_seed=263,
        color_seed=271,
        color_partition_hash=partition.partition_hash,
        config_hash="config-color",
        input_hash="input-a",
        latent_learning_rate=1.0e-3,
        affine_learning_rate=2.0e-3,
    )["C0"]
    objective = ColorGuardObjective(
        _base_objective,
        valid_flat_indices=torch.arange(30),
        source_base_color=source_rgb,
        opponent_lambda=0.75,
        pair_lambda=0.50,
    )

    legacy_step = train_candidate_step(
        legacy,
        _base_objective,
        texel_count=30,
        batch_size=8,
        cube_sample_count=0,
    )
    color_step = train_candidate_step(
        color,
        objective,
        texel_count=30,
        batch_size=8,
        cube_sample_count=0,
        color_partition=partition,
        color_batch_size=8,
    )

    assert torch.equal(legacy_step.batch.core_indices, color_step.batch.core_indices)
    assert legacy_step.loss == color_step.loss
    assert legacy_step.terms == color_step.terms
    assert torch.equal(legacy.latent, color.latent)
    for left, right in zip(legacy.decoder.parameters(), color.decoder.parameters()):
        assert torch.equal(left, right)


def test_g0_mean_loss_and_one_update_exactly_replay_old_c1() -> None:
    p0 = _synthetic_p0()
    source_rgb = torch.rand(30, 3, generator=torch.Generator().manual_seed(277))
    partition = build_color_quantile_partition(source_rgb)
    hue_partition = build_color_hue_partition(
        source_rgb, partition, min_group_size=1
    )
    common = dict(
        core_seed=281,
        color_seed=283,
        color_partition_hash=partition.partition_hash,
        config_hash="config-replay",
        input_hash="input-a",
        latent_learning_rate=1.0e-3,
        affine_learning_rate=2.0e-3,
    )
    old = create_color_candidates(p0, **common)["C1"]
    new = create_color_risk_candidates(
        p0, color_group_hash=hue_partition.group_hash, **common
    )["G0-mean"]
    scale = 8.4928470181
    old_objective = ColorGuardObjective(
        _base_objective,
        valid_flat_indices=torch.arange(30),
        source_base_color=source_rgb,
        opponent_lambda=0.10 * scale,
        pair_lambda=0.0,
    )
    new_objective = ColorRiskObjective(
        _base_objective,
        valid_flat_indices=torch.arange(30),
        source_base_color=source_rgb,
        yc_partition=partition,
        hue_partition=hue_partition,
        mean_scale=scale,
        cvar_scale=3.0,
        hue_scale=4.0,
        total_ratio=0.10,
    )

    old_step = train_candidate_step(
        old,
        old_objective,
        texel_count=30,
        batch_size=8,
        cube_sample_count=0,
        color_partition=partition,
        color_batch_size=8,
    )
    new_step = train_candidate_step(
        new,
        new_objective,
        texel_count=30,
        batch_size=8,
        cube_sample_count=0,
        color_partition=partition,
        color_batch_size=8,
    )

    assert torch.equal(old_step.batch.core_indices, new_step.batch.core_indices)
    assert torch.equal(old_step.batch.color_indices, new_step.batch.color_indices)
    assert old_step.loss == new_step.loss
    assert old_step.terms == new_step.terms
    assert torch.equal(old.latent, new.latent)
    for left, right in zip(old.decoder.parameters(), new.decoder.parameters()):
        assert torch.equal(left, right)
    _assert_nested_equal(old.latent_optimizer.state_dict(), new.latent_optimizer.state_dict())
    _assert_nested_equal(old.affine_optimizer.state_dict(), new.affine_optimizer.state_dict())


def test_color_risk_candidate_weights_share_one_point_one_gradient_budget() -> None:
    source = torch.rand(64, 3, generator=torch.Generator().manual_seed(293))
    partition = build_color_quantile_partition(source)
    hue_partition = build_color_hue_partition(source, partition, min_group_size=1)
    objective = ColorRiskObjective(
        _base_objective,
        valid_flat_indices=torch.arange(64),
        source_base_color=source,
        yc_partition=partition,
        hue_partition=hue_partition,
        mean_scale=2.0,
        cvar_scale=4.0,
        hue_scale=5.0,
        total_ratio=0.10,
    )

    expected = {
        "G0-mean": {"mean": 0.2},
        "G1-yc-cvar25": {"yc_cvar25": 0.4},
        "G2-hue8-macro": {"hue_macro": 0.5},
        "G3-cvar25-hue8": {"yc_cvar25": 0.2, "hue_macro": 0.25},
    }
    assert all(
        objective.weights_for_candidate(candidate_id) == pytest.approx(weights)
        for candidate_id, weights in expected.items()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_resume_accepts_rng_state_mapped_to_candidate_cuda_device() -> None:
    p0 = _synthetic_p0()
    p0 = replace(
        p0,
        safe=replace(
            p0.safe,
            latent_unorm8=p0.safe.latent_unorm8.to("cuda"),
            weight=p0.safe.weight.to("cuda"),
            bias=p0.safe.bias.to("cuda"),
        ),
        safe_decoder=copy.deepcopy(p0.safe_decoder).to("cuda"),
    )
    state = create_paired_candidates(
        p0,
        core_seed=182,
        cube_seed=184,
        config_hash="config-cuda",
        input_hash="input-cuda",
        latent_learning_rate=1.0e-3,
        affine_learning_rate=2.0e-3,
    )["L0"]
    checkpoint = checkpoint_candidate(state)
    checkpoint["core_rng_state"] = checkpoint["core_rng_state"].to("cuda")
    checkpoint["checkpoint_hash"] = checkpoint_candidate(state)["checkpoint_hash"]

    resumed = resume_candidate(
        checkpoint,
        p0,
        expected_parent_p0_hash=p0.safe.artifact_hash,
        expected_config_hash="config-cuda",
        expected_input_hash="input-cuda",
    )

    assert resumed.latent.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_two_cuda_resumes_from_one_checkpoint_are_independent() -> None:
    p0 = _synthetic_p0()
    p0 = replace(
        p0,
        safe=replace(
            p0.safe,
            latent_unorm8=p0.safe.latent_unorm8.to("cuda"),
            weight=p0.safe.weight.to("cuda"),
            bias=p0.safe.bias.to("cuda"),
        ),
        safe_decoder=copy.deepcopy(p0.safe_decoder).to("cuda"),
    )
    source = create_paired_candidates(
        p0,
        core_seed=185,
        cube_seed=186,
        config_hash="config-cuda",
        input_hash="input-cuda",
        latent_learning_rate=1.0e-3,
        affine_learning_rate=2.0e-3,
    )["L0"]
    train_candidate_step(
        source, _base_objective, texel_count=30, batch_size=8, cube_sample_count=0
    )
    checkpoint = checkpoint_candidate(source)
    resumed = [
        resume_candidate(
            checkpoint,
            p0,
            expected_parent_p0_hash=p0.safe.artifact_hash,
            expected_config_hash="config-cuda",
            expected_input_hash="input-cuda",
        )
        for _ in range(2)
    ]

    steps = [
        train_candidate_step(
            state, _base_objective, texel_count=30, batch_size=8, cube_sample_count=0
        )
        for state in resumed
    ]

    assert torch.equal(steps[0].batch.core_indices, steps[1].batch.core_indices)
    assert steps[0].loss == steps[1].loss
    assert torch.equal(resumed[0].latent, resumed[1].latent)
    assert all(
        torch.equal(left, right)
        for left, right in zip(resumed[0].decoder.parameters(), resumed[1].decoder.parameters())
    )


def test_continuation_rebinds_config_and_preserves_exact_source_state() -> None:
    p0 = _synthetic_p0()
    source = create_paired_candidates(
        p0,
        core_seed=183,
        cube_seed=187,
        config_hash="common-40k-config",
        input_hash="input-a",
        latent_learning_rate=1.0e-3,
        affine_learning_rate=2.0e-3,
    )["L0"]
    for _ in range(3):
        train_candidate_step(
            source, _base_objective, texel_count=30, batch_size=8, cube_sample_count=0
        )
    source_checkpoint = checkpoint_candidate(source)
    exact_before = checkpoint_candidate(source)

    begin_candidate_continuation(
        source,
        source_checkpoint=source_checkpoint,
        continuation_config_hash="l0-80k-config",
    )

    assert source.config_hash == "l0-80k-config"
    assert source.continuation_from_checkpoint_hash == source_checkpoint["checkpoint_hash"]
    assert source.continuation_from_step == 3
    assert torch.equal(source.latent, exact_before["latent"])
    assert torch.equal(source.core_rng.get_state(), exact_before["core_rng_state"])
    _assert_nested_equal(source.latent_optimizer.state_dict(), exact_before["latent_optimizer"])
    _assert_nested_equal(source.affine_optimizer.state_dict(), exact_before["affine_optimizer"])


def test_continuation_lineage_survives_checkpoint_and_exact_resume() -> None:
    p0 = _synthetic_p0()
    state = create_paired_candidates(
        p0,
        core_seed=189,
        cube_seed=190,
        config_hash="common-40k-config",
        input_hash="input-a",
        latent_learning_rate=1.0e-3,
        affine_learning_rate=2.0e-3,
    )["L0"]
    source_checkpoint = checkpoint_candidate(state)
    begin_candidate_continuation(
        state,
        source_checkpoint=source_checkpoint,
        continuation_config_hash="l0-80k-config",
    )

    continuation_checkpoint = checkpoint_candidate(state)
    resumed = resume_candidate(
        continuation_checkpoint,
        p0,
        expected_parent_p0_hash=p0.safe.artifact_hash,
        expected_config_hash="l0-80k-config",
        expected_input_hash="input-a",
    )

    assert continuation_checkpoint["continuation_from_checkpoint_hash"] == source_checkpoint["checkpoint_hash"]
    assert continuation_checkpoint["continuation_from_step"] == 0
    assert candidate_manifest(resumed)["continuation"] == {
        "source_checkpoint_hash": source_checkpoint["checkpoint_hash"],
        "source_step": 0,
    }


def test_checkpoint_and_manifest_capture_complete_training_lineage() -> None:
    p0 = _synthetic_p0()
    state = create_paired_candidates(
        p0,
        core_seed=191,
        cube_seed=193,
        config_hash="config-b",
        input_hash="input-b",
        latent_learning_rate=1.0e-3,
        affine_learning_rate=2.0e-3,
    )["L2"]
    state.best_metadata = {"best_loss": 0.125, "best_step": 1}
    train_candidate_step(
        state, _base_objective, texel_count=30, batch_size=8, cube_sample_count=4
    )

    checkpoint = checkpoint_candidate(state)
    manifest = candidate_manifest(state)

    assert set(checkpoint) == {
        "schema_version",
        "candidate_id",
        "objective_id",
        "parent_p0_hash",
        "config_hash",
        "input_hash",
        "optimizer_updates",
        "phase",
        "latent",
        "safe_affine_raw_parameters",
        "latent_optimizer",
        "affine_optimizer",
        "core_rng_state",
        "cube_rng_state",
        "best_metadata",
        "checkpoint_hash",
    }
    assert checkpoint["cube_rng_state"] is not None
    assert checkpoint["optimizer_updates"] == 1
    assert checkpoint["best_metadata"] == state.best_metadata
    assert manifest == {
        "schema_version": 1,
        "candidate_id": "L2",
        "objective_id": "material+helmet+cube",
        "parent_p0_hash": p0.safe.artifact_hash,
        "config_hash": "config-b",
        "input_hash": "input-b",
        "optimizer_updates": 1,
        "phase": "warmup",
        "learned_linear": True,
    }


def test_resume_rejects_lineage_or_checkpoint_integrity_mismatch() -> None:
    p0 = _synthetic_p0()
    state = create_paired_candidates(
        p0,
        core_seed=197,
        cube_seed=199,
        config_hash="config-c",
        input_hash="input-c",
        latent_learning_rate=1.0e-3,
        affine_learning_rate=2.0e-3,
    )["L0"]
    checkpoint = checkpoint_candidate(state)
    assert "checkpoint_hash" in checkpoint

    for field in ("parent_p0_hash", "config_hash", "input_hash"):
        mismatched = copy.deepcopy(checkpoint)
        mismatched[field] = "wrong"
        with pytest.raises(ValueError, match="mismatch"):
            resume_candidate(
                mismatched,
                p0,
                expected_parent_p0_hash=p0.safe.artifact_hash,
                expected_config_hash="config-c",
                expected_input_hash="input-c",
            )

    corrupted = copy.deepcopy(checkpoint)
    corrupted["latent"][0, 0] += 1.0
    with pytest.raises(ValueError, match="checkpoint_hash mismatch"):
        resume_candidate(
            corrupted,
            p0,
            expected_parent_p0_hash=p0.safe.artifact_hash,
            expected_config_hash="config-c",
            expected_input_hash="input-c",
        )


def test_step_qualified_endpoints_are_immutable_and_separate_from_rolling() -> None:
    state = create_paired_candidates(
        _synthetic_p0(),
        core_seed=211,
        cube_seed=223,
        config_hash="config-d",
        input_hash="input-d",
        latent_learning_rate=1.0e-3,
        affine_learning_rate=2.0e-3,
    )["L0"]
    root = Path("isolated-affine-output")
    storage = InMemoryCheckpointStorage()
    state.optimizer_updates = 40_000
    rolling = write_candidate_checkpoint(root, state, endpoint=False, storage=storage)
    endpoint_40k = write_candidate_checkpoint(root, state, endpoint=True, storage=storage)

    assert rolling.path == root / "L0" / "rolling" / "checkpoint.pt"
    assert endpoint_40k.path == (
        root / "L0" / "endpoints" / "step-040000" / "checkpoint.pt"
    )
    assert rolling.path != endpoint_40k.path
    with pytest.raises(FileExistsError):
        write_candidate_checkpoint(root, state, endpoint=True, storage=storage)

    state.optimizer_updates = 80_000
    endpoint_80k = write_candidate_checkpoint(root, state, endpoint=True, storage=storage)
    state.optimizer_updates = 120_000
    endpoint_120k = write_candidate_checkpoint(root, state, endpoint=True, storage=storage)
    assert len(
        {endpoint_40k.path, endpoint_80k.path, endpoint_120k.path}
    ) == 3


def test_fixed_step_timing_reports_distribution_through_public_runner() -> None:
    state = create_paired_candidates(
        _synthetic_p0(),
        core_seed=227,
        cube_seed=229,
        config_hash="config-timing",
        input_hash="input-timing",
        latent_learning_rate=1.0e-3,
        affine_learning_rate=2.0e-3,
    )["L0"]

    report = time_candidate_steps(
        state,
        _base_objective,
        texel_count=30,
        batch_size=8,
        cube_sample_count=0,
        warmup_steps=2,
        measured_steps=3,
    )

    assert state.optimizer_updates == 5
    assert report.warmup_steps == 2
    assert report.measured_steps == 3
    assert len(report.step_times_ms) == 3
    assert report.median_step_ms > 0.0
    assert report.p95_step_ms >= report.median_step_ms
    assert report.samples_per_second > 0.0
    assert report.peak_allocated_bytes == 0
    assert report.peak_reserved_bytes == 0
    assert set(report.mean_loss_terms) == {"base"}


def test_optimizer_update_keeps_rgba_bounded_and_safe_affine_certified() -> None:
    p0 = _synthetic_p0()
    state = create_paired_candidates(
        p0,
        core_seed=233,
        cube_seed=239,
        config_hash="config-bounds",
        input_hash="input-bounds",
        latent_learning_rate=0.1,
        affine_learning_rate=1.0e-3,
    )["L0"]

    def outward_objective(candidate, batch):
        loss = -candidate.latent.sum()
        return loss, {"outward": loss}

    train_candidate_step(
        state,
        outward_objective,
        texel_count=30,
        batch_size=8,
        cube_sample_count=0,
    )
    weight, bias = state.decoder.fold_affine()

    assert torch.all(state.latent >= 0.0)
    assert torch.all(state.latent <= 1.0)
    assert certify_affine(weight, bias, margin=state.decoder.margin)["valid"] is True


def test_training_observation_plan_keeps_dense_trends_and_immutable_milestones() -> None:
    plan = plan_training_observations(
        total_steps=40_000,
        checkpoint_steps=(1_000, 5_000, 10_000, 20_000, 30_000, 35_000, 40_000),
        trend_interval=1_000,
    )

    assert plan.checkpoint_steps == (
        1_000,
        5_000,
        10_000,
        20_000,
        30_000,
        35_000,
        40_000,
    )
    assert plan.trend_steps == tuple(range(1_000, 40_001, 1_000))


def test_parameter_trend_is_comparable_to_the_frozen_parent() -> None:
    p0 = _synthetic_p0()
    state = create_paired_candidates(
        p0,
        core_seed=241,
        cube_seed=251,
        config_hash="config-trend",
        input_hash="input-trend",
        latent_learning_rate=1.0e-3,
        affine_learning_rate=2.0e-3,
    )["L0"]

    initial = candidate_parameter_trend(state, p0)
    train_candidate_step(
        state, _base_objective, texel_count=30, batch_size=8, cube_sample_count=0
    )
    learned = candidate_parameter_trend(state, p0)

    assert initial["parent_p0_hash"] == learned["parent_p0_hash"] == p0.safe.artifact_hash
    assert initial["step"] == 0
    assert initial["latent_delta_rmse"] == 0.0
    assert initial["affine_weight_delta_l2"] < 1.0e-12
    assert initial["affine_bias_delta_l2"] < 1.0e-12
    assert learned["step"] == 1
    assert learned["latent_delta_rmse"] > 0.0
    assert learned["affine_weight_delta_l2"] > 0.0
    assert learned["affine_bias_delta_l2"] > 0.0
    assert len(learned["latent_channel_mean"]) == 4
    assert len(learned["latent_channel_std"]) == 4
    assert 0.0 <= learned["latent_saturation_fraction"] <= 1.0
    assert learned["latent_learning_rate"] == 1.0e-3
    assert learned["affine_learning_rate"] == 2.0e-3
    assert learned["certificate"]["valid"] is True


def test_candidate_run_emits_curve_trends_and_scheduled_checkpoints() -> None:
    p0 = _synthetic_p0()
    state = create_paired_candidates(
        p0,
        core_seed=257,
        cube_seed=263,
        config_hash="config-run",
        input_hash="input-run",
        latent_learning_rate=1.0e-3,
        affine_learning_rate=2.0e-3,
    )["L0"]
    plan = plan_training_observations(
        total_steps=5,
        checkpoint_steps=(2, 5),
        trend_interval=1,
    )
    storage = InMemoryCheckpointStorage()
    observed_steps: list[int] = []
    observed_trends: list[int] = []

    report = run_candidate_training(
        state,
        _base_objective,
        p0,
        output_root=Path("isolated-affine-run"),
        observation_plan=plan,
        texel_count=30,
        batch_size=8,
        cube_sample_count=0,
        storage=storage,
        on_step=lambda step: observed_steps.append(step["step"]),
        on_trend=lambda trend: observed_trends.append(trend["step"]),
    )

    assert state.optimizer_updates == 5
    assert [point["step"] for point in report.curve] == [1, 2, 3, 4, 5]
    assert [point["step"] for point in report.parameter_trends] == [1, 2, 3, 4, 5]
    assert [write.path for write in report.checkpoints] == [
        Path("isolated-affine-run/L0/endpoints/step-000002/checkpoint.pt"),
        Path("isolated-affine-run/L0/endpoints/step-000005/checkpoint.pt"),
    ]
    assert Path("isolated-affine-run/L0/rolling/checkpoint.pt") in storage.files
    assert observed_steps == [1, 2, 3, 4, 5]
    assert observed_trends == [1, 2, 3, 4, 5]
    assert state.best_metadata["best_step"] in range(1, 6)
    assert report.manifest["optimizer_updates"] == 5
