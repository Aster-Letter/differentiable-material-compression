from __future__ import annotations

import pytest
import torch
from torch import nn
from pathlib import Path

from cg_frontier.compression.basecolor_priority import (
    BaseColorObjectiveConfig,
    C4PostprocessConfig,
    apply_chroma_compander,
    apply_material_safety,
    basecolor_charbonnier,
    calibrate_basecolor_gradient_budget,
    compose_basecolor_priority_objective,
    decoder_instruction_audit,
    load_basecolor_priority_checkpoint,
    postprocess_affine_output,
    save_basecolor_priority_checkpoint,
)


def test_basecolor_charbonnier_is_zero_at_identity_and_averages_channels() -> None:
    source = torch.tensor([[0.2, 0.4, 0.8]], dtype=torch.float64)
    prediction = torch.tensor(
        [[0.3, 0.2, 0.8]], dtype=torch.float64, requires_grad=True
    )

    identity = basecolor_charbonnier(source, source)
    loss = basecolor_charbonnier(prediction, source)
    expected = (
        (0.1**2 + 1.0e-6) ** 0.5
        + (0.2**2 + 1.0e-6) ** 0.5
        + (0.0**2 + 1.0e-6) ** 0.5
    ) / 3.0 - 1.0e-3
    loss.backward()

    assert float(identity) == pytest.approx(0.0, abs=1.0e-15)
    assert float(loss.detach()) == pytest.approx(expected, abs=1.0e-15)
    assert prediction.grad is not None
    assert bool(torch.isfinite(prediction.grad).all())
    assert float(torch.linalg.vector_norm(prediction.grad)) > 0.0


def test_material_safety_is_forward_legal_and_uses_ste_backward() -> None:
    raw = torch.tensor(
        [[-0.4, 0.3, 1.4, 3.0, 4.0, -0.2, 1.2]],
        dtype=torch.float64,
        requires_grad=True,
    )

    final = apply_material_safety(raw, straight_through=True)
    final.sum().backward()

    assert final[0, [0, 1, 2, 5, 6]].tolist() == pytest.approx(
        [0.0, 0.3, 1.0, 0.0, 1.0]
    )
    assert float(torch.linalg.vector_norm(final[0, 3:5]).detach()) == pytest.approx(1.0)
    assert raw.grad is not None
    assert bool(torch.isfinite(raw.grad).all())
    assert bool(torch.all(raw.grad != 0.0))


def test_chroma_compander_identity_preserves_y_hue_and_bounds_gain() -> None:
    rgb = torch.tensor(
        [[0.45, 0.35, 0.25], [0.6, 0.3, 0.3]],
        dtype=torch.float64,
        requires_grad=True,
    )
    identity = apply_chroma_compander(rgb, torch.tensor(1.0), torch.tensor(0.0))
    expanded = apply_chroma_compander(rgb, torch.tensor(1.4), torch.tensor(0.8))

    source_y = rgb.mean(dim=-1)
    expanded_y = expanded.rgb.mean(dim=-1)
    source_direction = rgb - source_y[:, None]
    expanded_direction = expanded.rgb - expanded_y[:, None]
    expanded.rgb.sum().backward()

    assert torch.equal(identity.rgb, rgb)
    assert torch.equal(identity.gain, torch.ones_like(identity.gain))
    assert torch.allclose(expanded_y, source_y, atol=1.0e-15, rtol=0.0)
    assert torch.allclose(
        expanded_direction,
        source_direction * expanded.gain,
        atol=1.0e-15,
        rtol=0.0,
    )
    assert bool(torch.all((expanded.gain >= 0.5) & (expanded.gain <= 2.0)))
    assert rgb.grad is not None
    assert bool(torch.isfinite(rgb.grad).all())


def test_decoder_instruction_audit_is_bounded_and_explicit() -> None:
    safety_only = decoder_instruction_audit(C4PostprocessConfig(compander=False))
    compander = decoder_instruction_audit(C4PostprocessConfig(compander=True))

    assert safety_only["total_scalar_instruction_equivalents"] == 46
    assert compander["total_scalar_instruction_equivalents"] == 70
    assert compander["within_budget"] is True
    assert compander["budget"] == 80
    assert compander["breakdown"]["affine_mac"] == 28
    assert compander["measured_gpu_cost"] is False

    with pytest.raises(ValueError):
        decoder_instruction_audit(C4PostprocessConfig(compander=True), budget=69)


def test_gradient_budget_calibrates_both_parameter_groups_without_updates() -> None:
    latent = torch.nn.Parameter(torch.tensor([0.4, -0.2], dtype=torch.float64))
    weight = torch.nn.Parameter(torch.tensor([0.3, 0.7], dtype=torch.float64))
    bias = torch.nn.Parameter(torch.tensor([0.1], dtype=torch.float64))
    before = [value.detach().clone() for value in (latent, weight, bias)]

    def losses(scale: float) -> tuple[torch.Tensor, torch.Tensor]:
        base = scale * (latent.square().sum() + 0.25 * weight.square().sum() + bias.square().sum())
        residual = scale * (
            2.0 * latent.square().sum() + weight.square().sum() + 3.0 * bias.square().sum()
        )
        return base, residual

    audit = calibrate_basecolor_gradient_budget(
        [float(index + 1) for index in range(8)],
        losses,
        parameter_groups={"latent": (latent,), "affine": (weight, bias)},
        target_share=0.8,
    )

    assert audit.batch_count == 8
    assert audit.lambda_value == pytest.approx(max(row.lambda_candidate for row in audit.groups.values()))
    assert audit.groups[audit.limiting_group].achieved_share == pytest.approx(0.8)
    assert all(row.achieved_share >= 0.8 for row in audit.groups.values())
    assert all(parameter.grad is None for parameter in (latent, weight, bias))
    assert all(torch.equal(current, expected) for current, expected in zip((latent, weight, bias), before, strict=True))


def test_basecolor_v3_checkpoint_exactly_resumes_three_optimizers_and_rng() -> None:
    def make_state():
        latent = nn.Parameter(torch.tensor([0.2, 0.7], dtype=torch.float64))
        weight = nn.Parameter(torch.tensor([0.5], dtype=torch.float64))
        bias = nn.Parameter(torch.tensor([0.1], dtype=torch.float64))
        compander = nn.Parameter(torch.tensor([1.0, 0.0], dtype=torch.float64))
        optimizers = (
            torch.optim.Adam([latent], lr=1.0e-2),
            torch.optim.Adam([weight, bias], lr=2.0e-2),
            torch.optim.Adam([compander], lr=3.0e-2),
        )
        return (latent, weight, bias, compander), optimizers

    def update(state, optimizers, rng):
        draw = torch.rand((), generator=rng, dtype=torch.float64)
        loss = draw * sum(value.square().sum() for value in state)
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
        return loss.detach(), draw

    identity = {
        "parent_hash": "parent",
        "input_hash": "input",
        "config_hash": "config",
        "objective_hash": "objective",
        "postprocess_hash": "postprocess",
        "rig_hash": "rig",
    }
    state, optimizers = make_state()
    rng = torch.Generator().manual_seed(17)
    update(state, optimizers, rng)
    path = Path("tests/.basecolor-priority-v3-test.pt")
    path.unlink(missing_ok=True)
    save_basecolor_priority_checkpoint(
        path,
        step=1,
        candidate_id="BC80",
        objective_id="basecolor_priority",
        target_share=0.8,
        lambda_value=4.0,
        actual_shares={"latent": 0.8, "affine": 0.83},
        latent=state[0],
        weight=state[1],
        bias=state[2],
        compander_parameters=state[3],
        latent_optimizer=optimizers[0],
        affine_optimizer=optimizers[1],
        compander_optimizer=optimizers[2],
        core_rng=rng,
        identity=identity,
    )
    continuous = update(state, optimizers, rng)

    resumed_state, resumed_optimizers = make_state()
    resumed_rng = torch.Generator()
    payload = load_basecolor_priority_checkpoint(
        path,
        expected_candidate_id="BC80",
        expected_objective_id="basecolor_priority",
        expected_target_share=0.8,
        expected_identity=identity,
    )
    for parameter, field in zip(
        resumed_state, ("latent", "weight", "bias", "compander_parameters"), strict=True
    ):
        parameter.data.copy_(payload[field])
    for optimizer, field in zip(
        resumed_optimizers,
        ("latent_optimizer", "affine_optimizer", "compander_optimizer"),
        strict=True,
    ):
        optimizer.load_state_dict(payload[field])
    resumed_rng.set_state(payload["core_rng_state"])
    after_resume = update(resumed_state, resumed_optimizers, resumed_rng)

    assert all(torch.equal(left, right) for left, right in zip(continuous, after_resume, strict=True))
    assert all(torch.equal(left, right) for left, right in zip(state, resumed_state, strict=True))

    bad_identity = dict(identity, rig_hash="wrong")
    with pytest.raises(ValueError):
        load_basecolor_priority_checkpoint(
            path,
            expected_candidate_id="BC80",
            expected_objective_id="basecolor_priority",
            expected_target_share=0.8,
            expected_identity=bad_identity,
        )
    path.unlink()


def test_objective_keeps_r0_residual_weights_and_has_no_support_term() -> None:
    terms = {
        "base_color_l1": torch.tensor(2.0),
        "base_color_charbonnier": torch.tensor(3.0),
        "render_linear": torch.tensor(5.0),
        "render_log": torch.tensor(7.0),
        "normal_cosine": torch.tensor(11.0),
        "roughness_l1": torch.tensor(13.0),
        "metallic_l1": torch.tensor(17.0),
    }
    control, control_pieces = compose_basecolor_priority_objective(
        terms, BaseColorObjectiveConfig(candidate_id="N0-control", target_share=0.0, lambda_value=1.0)
    )
    bc80, bc80_pieces = compose_basecolor_priority_objective(
        terms, BaseColorObjectiveConfig(candidate_id="BC80", target_share=0.8, lambda_value=4.0)
    )
    oracle, oracle_pieces = compose_basecolor_priority_objective(
        terms,
        BaseColorObjectiveConfig(candidate_id="BC-only", target_share=1.0, lambda_value=1.0),
    )
    residual = 5.0 + 0.25 * 7.0 + 0.25 * 11.0 + 0.5 * 13.0 + 0.5 * 17.0

    assert float(control) == pytest.approx(2.0 + residual)
    assert float(bc80) == pytest.approx(4.0 * 3.0 + residual)
    assert float(oracle) == pytest.approx(3.0)
    assert float(oracle_pieces["residual"]) == pytest.approx(0.0)
    assert set(control_pieces) == {"base_color", "residual", "total"}
    assert set(bc80_pieces) == {"base_color", "residual", "total"}
    assert set(oracle_pieces) == {"base_color", "residual", "total"}


def test_basecolor_only_checkpoint_identity_is_explicit() -> None:
    latent = nn.Parameter(torch.tensor([0.2, 0.7], dtype=torch.float64))
    weight = nn.Parameter(torch.tensor([0.5], dtype=torch.float64))
    bias = nn.Parameter(torch.tensor([0.1], dtype=torch.float64))
    compander = nn.Parameter(torch.tensor([1.0, 0.0], dtype=torch.float64))
    optimizers = (
        torch.optim.Adam([latent], lr=1.0e-2),
        torch.optim.Adam([weight, bias], lr=2.0e-2),
        torch.optim.Adam([compander], lr=3.0e-2),
    )
    identity = {
        "parent_hash": "parent",
        "input_hash": "input",
        "config_hash": "config",
        "objective_hash": "objective",
        "postprocess_hash": "postprocess",
        "rig_hash": "rig",
    }
    path = Path("tests/.basecolor-only-oracle-test.pt")
    path.unlink(missing_ok=True)

    save_basecolor_priority_checkpoint(
        path,
        step=1,
        candidate_id="BC-only",
        objective_id="basecolor_only_oracle",
        target_share=1.0,
        lambda_value=1.0,
        actual_shares={"latent": 1.0, "affine": 1.0},
        latent=latent,
        weight=weight,
        bias=bias,
        compander_parameters=compander,
        latent_optimizer=optimizers[0],
        affine_optimizer=optimizers[1],
        compander_optimizer=optimizers[2],
        core_rng=torch.Generator().manual_seed(23),
        identity=identity,
    )
    payload = load_basecolor_priority_checkpoint(
        path,
        expected_candidate_id="BC-only",
        expected_objective_id="basecolor_only_oracle",
        expected_target_share=1.0,
        expected_identity=identity,
    )

    assert payload["target_share"] == pytest.approx(1.0)
    assert payload["lambda_value"] == pytest.approx(1.0)
    path.unlink()


def test_postprocessed_affine_output_is_the_single_legal_loss_and_render_surface() -> None:
    raw = torch.tensor(
        [[-0.1, 0.5, 1.2, 2.0, 0.0, -0.3, 1.4]], dtype=torch.float64
    )
    result = postprocess_affine_output(
        raw,
        compander_parameters=torch.tensor([1.0, 0.0], dtype=torch.float64),
        straight_through=False,
    )

    assert result.seven.shape == raw.shape
    assert result.normal_xyz.shape == (1, 3)
    assert torch.equal(result.seven[..., :3], torch.tensor([[0.0, 0.5, 1.0]], dtype=torch.float64))
    assert float(torch.linalg.vector_norm(result.normal_xyz, dim=-1)) == pytest.approx(1.0)
    assert bool(torch.all((result.seven[..., [0, 1, 2, 5, 6]] >= 0.0) & (result.seven[..., [0, 1, 2, 5, 6]] <= 1.0)))


def test_postprocess_normal_z_has_finite_training_gradient_at_disk_boundary() -> None:
    raw = torch.tensor(
        [[0.2, 0.3, 0.4, 1.0, 0.0, 0.5, 0.6]],
        dtype=torch.float64,
        requires_grad=True,
    )

    training = postprocess_affine_output(
        raw,
        compander_parameters=None,
        straight_through=True,
    )
    deployed = postprocess_affine_output(
        raw.detach(),
        compander_parameters=None,
        straight_through=False,
    )
    training.normal_xyz.sum().backward()

    assert torch.equal(training.normal_xyz.detach(), deployed.normal_xyz)
    assert training.normal_xyz.detach().tolist() == [[1.0, 0.0, 0.0]]
    assert raw.grad is not None
    assert bool(torch.isfinite(raw.grad).all())


def test_safety_legal_set_is_closed_under_filtered_convex_combinations() -> None:
    raw = torch.tensor(
        [
            [-1.0, 0.2, 1.5, 2.0, 0.0, -0.2, 1.2],
            [0.4, 1.4, 0.1, 0.0, 3.0, 0.6, 0.4],
            [0.3, 0.4, 0.5, -2.0, 1.0, 0.2, 0.8],
            [1.4, -0.1, 0.7, 0.4, -3.0, 1.3, -0.3],
        ],
        dtype=torch.float64,
    )
    legal = apply_material_safety(raw, straight_through=False)
    weights = torch.tensor([0.1, 0.2, 0.3, 0.4], dtype=torch.float64)
    filtered = (legal * weights[:, None]).sum(dim=0)

    assert bool(torch.all((filtered[[0, 1, 2, 5, 6]] >= 0.0) & (filtered[[0, 1, 2, 5, 6]] <= 1.0)))
    assert float(torch.linalg.vector_norm(filtered[3:5])) <= 1.0 + 1.0e-12
