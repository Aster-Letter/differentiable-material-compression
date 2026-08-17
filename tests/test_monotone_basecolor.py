from __future__ import annotations

import torch

from cg_frontier.compression.exact_basecolor_experiment import TexelTargets
from cg_frontier.compression.monotone_basecolor import (
    BaseColorMetrics,
    ConstraintFractions,
    MonotoneBaseColorCandidate,
    audit_acceptance,
    balanced_basecolor_loss,
    build_color_partition,
    clone_training_state,
    composite_curve_acceptance,
    composite_curve_alpha_for_value,
    composite_curve_target,
    constraint_targets,
    interpolate_candidate_state_,
    restore_training_state,
    sample_balanced_ids,
    normalized_basecolor_merit,
    normalized_basecolor_composite,
    preset_curve_acceptance,
    preset_curve_targets,
    trust_region_acceptance,
)


def _candidate() -> tuple[MonotoneBaseColorCandidate, TexelTargets]:
    colors = torch.tensor(
        [
            [0, 0, 0],
            [255, 0, 0],
            [0, 255, 0],
            [0, 0, 255],
            [255, 255, 255],
            [128, 80, 20],
            [30, 60, 90],
            [220, 180, 40],
        ],
        dtype=torch.uint8,
    )
    latent = torch.cat((colors, torch.arange(8, dtype=torch.uint8)[:, None]), dim=-1).reshape(2, 4, 4)
    weight = torch.zeros((7, 4))
    weight[:3, :3] = torch.eye(3)
    weight[5, 3] = 0.1
    weight[6, 3] = 0.2
    bias = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 0.4, 0.2])
    candidate = MonotoneBaseColorCandidate(
        latent_u8=latent,
        weight=weight,
        bias=bias,
        colors_u8=colors,
    )
    targets = TexelTargets(
        base_float=colors.to(torch.float32) / 255.0,
        base_q8=colors,
        normal_xyz=torch.tensor([[0.0, 0.0, 1.0]]).repeat(8, 1),
        roughness=torch.full((8, 1), 0.5),
        metallic=torch.full((8, 1), 0.2),
        height=2,
        width=4,
    )
    return candidate, targets


def test_color_partition_and_balanced_sampling_are_deterministic_and_cover_groups() -> None:
    _, targets = _candidate()
    first = build_color_partition(targets.base_q8, bins=3)
    second = build_color_partition(targets.base_q8, bins=3)
    torch.testing.assert_close(first.group_ids, second.group_ids, rtol=0.0, atol=0.0)
    generator_a = torch.Generator().manual_seed(7)
    generator_b = torch.Generator().manual_seed(7)
    ids_a = sample_balanced_ids(first, sample_count=32, generator=generator_a)
    ids_b = sample_balanced_ids(first, sample_count=32, generator=generator_b)
    torch.testing.assert_close(ids_a, ids_b, rtol=0.0, atol=0.0)
    assert set(first.group_ids[ids_a].tolist()) == set(range(first.group_count))


def test_balanced_loss_is_zero_only_for_exact_spatial_color_recovery() -> None:
    _, targets = _candidate()
    partition = build_color_partition(targets.base_q8, bins=3)
    exact = targets.base_q8.to(torch.float32) / 255.0
    loss, terms = balanced_basecolor_loss(exact, exact, partition.group_ids)
    assert float(loss) == 0.0
    assert all(float(value) == 0.0 for value in terms.values())
    shuffled = exact.roll(1, dims=0)
    shifted, _ = balanced_basecolor_loss(shuffled, exact, partition.group_ids)
    assert float(shifted) > 0.0


def test_geometric_targets_and_acceptance_enforce_monotone_multi_metric_progress() -> None:
    initial = BaseColorMetrics(0.04, 0.05, 0.1, 0.03, 0.05, 0.4, 0.5, 0.4, 0.5)
    fractions = ConstraintFractions(0.1, 0.2, 0.15, 0.15, 0.15, 0.15)
    final = constraint_targets(initial, fractions, step=100, final_step=100)
    assert final == {
        "rgb_mean": 0.004,
        "rgb_tail": 0.020000000000000004,
        "opponent_mean": 0.0045,
        "opponent_macro": 0.0075,
        "opponent_relative_mean": 0.06,
        "opponent_relative_macro": 0.075,
    }
    start = constraint_targets(initial, fractions, step=0, final_step=100)
    assert start == initial.constrained()
    midpoint = constraint_targets(initial, fractions, step=50, final_step=100)
    assert abs(midpoint["rgb_mean"] - 0.022) < 1.0e-12
    improved = BaseColorMetrics(0.003, 0.01, 0.019, 0.004, 0.007, 0.04, 0.05, 0.9, 0.0)
    passed, checks = audit_acceptance(
        improved,
        initial,
        final,
        relative_tolerance=0.0,
        absolute_tolerance=0.0,
    )
    assert passed and all(item["monotone"] and item["target"] for item in checks.values())
    regressed = BaseColorMetrics(0.003, 0.01, 0.019, 0.004, 0.007, 0.4, 0.5, 0.9, 0.0)
    assert not audit_acceptance(regressed, initial, final, relative_tolerance=0.0, absolute_tolerance=0.0)[0]


def test_trust_region_requires_merit_progress_and_guards_historical_best() -> None:
    initial = BaseColorMetrics(0.04, 0.05, 0.1, 0.03, 0.05, 0.4, 0.5, 0.4, 0.5)
    current = BaseColorMetrics(0.039, 0.049, 0.099, 0.029, 0.049, 0.38, 0.48, 0.45, 0.4)
    merit = normalized_basecolor_merit(current, initial)
    passed, checks = trust_region_acceptance(
        current,
        initial,
        initial.constrained(),
        target_merit=merit + 1.0e-12,
        relative_tolerance=0.2,
        absolute_tolerance=0.0,
    )
    assert passed and checks["merit"]["passed"]
    failed, _ = trust_region_acceptance(
        current,
        initial,
        initial.constrained(),
        target_merit=merit - 1.0e-6,
        relative_tolerance=0.2,
        absolute_tolerance=0.0,
    )
    assert not failed


def test_preset_curve_is_monotone_and_reaches_declared_floors() -> None:
    initial = BaseColorMetrics(0.04, 0.05, 0.1, 0.03, 0.05, 0.4, 0.5, 0.4, 0.5)
    exponents = {
        "rgb_mean": 1.0,
        "rgb_tail": 0.4,
        "opponent_mean": 0.8,
        "opponent_macro": 0.65,
        "opponent_relative_mean": 0.35,
        "opponent_relative_macro": 0.35,
    }
    floors = {name: 0.0 for name in initial.constrained()}
    start = preset_curve_targets(initial, alpha=0.0, exponents=exponents, floors=floors)
    quarter = preset_curve_targets(initial, alpha=0.25, exponents=exponents, floors=floors)
    half = preset_curve_targets(initial, alpha=0.5, exponents=exponents, floors=floors)
    final = preset_curve_targets(initial, alpha=1.0, exponents=exponents, floors=floors)
    assert start == initial.constrained()
    assert final == floors
    for name in start:
        assert start[name] > quarter[name] > half[name] > final[name]
    # A smaller exponent deliberately makes the tail constraint advance more gently.
    assert quarter["rgb_tail"] / start["rgb_tail"] > quarter["rgb_mean"] / start["rgb_mean"]


def test_preset_curve_acceptance_checks_every_metric_without_historical_best() -> None:
    target = {
        "rgb_mean": 0.03,
        "rgb_tail": 0.09,
        "opponent_mean": 0.02,
        "opponent_macro": 0.04,
        "opponent_relative_mean": 0.3,
        "opponent_relative_macro": 0.4,
    }
    inside = BaseColorMetrics(0.029, 0.05, 0.089, 0.019, 0.039, 0.299, 0.399, 0.4, 0.5)
    passed, checks = preset_curve_acceptance(
        inside,
        target,
        relative_tolerance=0.0,
        absolute_tolerance=0.0,
    )
    assert passed and all(item["passed"] for item in checks.values())
    outside = BaseColorMetrics(0.029, 0.05, 0.091, 0.019, 0.039, 0.299, 0.399, 0.4, 0.5)
    assert not preset_curve_acceptance(
        outside,
        target,
        relative_tolerance=0.0,
        absolute_tolerance=0.0,
    )[0]


def test_preset_curve_can_delay_difficult_metrics_without_changing_endpoint() -> None:
    initial = BaseColorMetrics(0.04, 0.05, 0.1, 0.03, 0.05, 0.4, 0.5, 0.4, 0.5)
    names = initial.constrained()
    exponents = {name: 1.0 for name in names}
    floors = {name: 0.0 for name in names}
    starts = {name: (0.1 if name == "opponent_macro" else 0.0) for name in names}
    early = preset_curve_targets(initial, alpha=0.05, exponents=exponents, floors=floors, starts=starts)
    assert early["opponent_macro"] == initial.opponent_macro
    assert early["rgb_mean"] < initial.rgb_mean
    final = preset_curve_targets(initial, alpha=1.0, exponents=exponents, floors=floors, starts=starts)
    assert final == floors
    headrooms = {name: (0.01 if name == "opponent_macro" else 0.0) for name in names}
    corridor = preset_curve_targets(
        initial,
        alpha=0.05,
        exponents=exponents,
        floors=floors,
        starts=starts,
        headrooms=headrooms,
    )
    assert abs(corridor["opponent_macro"] - initial.opponent_macro * 1.01) < 1.0e-12


def test_candidate_hard_forward_and_training_snapshot_restore_are_exact() -> None:
    candidate, targets = _candidate()
    ids = torch.arange(8)
    decoded = candidate.decoder(candidate.latent_for_ids(ids, ste=False))
    torch.testing.assert_close(
        decoded.base_color_linear,
        targets.base_q8.to(torch.float32) / 255.0,
        rtol=0.0,
        atol=0.0,
    )
    code = torch.optim.Adam(candidate.code_parameters(), lr=0.02)
    decoder = torch.optim.Adam(candidate.decoder_parameters(), lr=0.001)
    generator = torch.Generator().manual_seed(9)
    snapshot = clone_training_state(candidate, code, decoder, generator)
    expected_rng = generator.get_state().clone()
    with torch.no_grad():
        candidate.latent_byte.add_(13.0)
        candidate.decoder.weight.mul_(2.0)
    _ = torch.rand((10,), generator=generator)
    restore_training_state(snapshot, candidate, code, decoder, generator)
    torch.testing.assert_close(candidate.latent_byte, snapshot["candidate_state"]["latent_byte"])
    torch.testing.assert_close(candidate.decoder.weight, snapshot["candidate_state"]["decoder.weight"])
    torch.testing.assert_close(generator.get_state(), expected_rng)


def test_candidate_state_interpolation_projects_parameters_exactly() -> None:
    candidate, _ = _candidate()
    start = {name: value.detach().clone() for name, value in candidate.state_dict().items()}
    with torch.no_grad():
        candidate.latent_byte.add_(8.0)
        candidate.decoder.weight.add_(2.0)
    end = {name: value.detach().clone() for name, value in candidate.state_dict().items()}
    interpolate_candidate_state_(candidate, start, end, 0.25)
    torch.testing.assert_close(candidate.latent_byte, start["latent_byte"] + 2.0, rtol=0.0, atol=0.0)
    torch.testing.assert_close(candidate.decoder.weight, start["decoder.weight"] + 0.5, rtol=0.0, atol=1.0e-7)


def test_composite_curve_is_normalized_monotone_and_guarded() -> None:
    initial = BaseColorMetrics(0.04, 0.05, 0.1, 0.03, 0.05, 0.4, 0.5, 0.4, 0.5)
    weights = {
        "rgb_mean": 0.35,
        "rgb_tail": 0.15,
        "opponent_relative_mean": 0.25,
        "opponent_relative_macro": 0.25,
    }
    assert abs(normalized_basecolor_composite(initial, initial, weights) - 1.0) < 1.0e-12
    targets = [composite_curve_target(alpha=value, floor=0.0, exponent=1.0) for value in (0.0, 0.25, 0.5, 1.0)]
    assert targets[0] == 1.0 and targets[-1] == 0.0
    assert targets[0] > targets[1] > targets[2] > targets[3]
    for alpha, target in zip((0.0, 0.25, 0.5, 1.0), targets, strict=True):
        assert abs(composite_curve_alpha_for_value(target, floor=0.0, exponent=1.0) - alpha) < 1.0e-12
    guards = {name: 1.05 for name in initial.constrained()}
    improved = BaseColorMetrics(0.03, 0.05, 0.08, 0.031, 0.052, 0.3, 0.4, 0.4, 0.5)
    passed, checks = composite_curve_acceptance(
        improved,
        initial,
        weights=weights,
        target=0.9,
        guard_multipliers=guards,
        relative_tolerance=0.0,
        absolute_tolerance=0.0,
    )
    assert passed and checks["composite"]["passed"]
    guard_violation = BaseColorMetrics(0.03, 0.05, 0.08, 0.04, 0.052, 0.3, 0.4, 0.4, 0.5)
    failed, failed_checks = composite_curve_acceptance(
        guard_violation,
        initial,
        weights=weights,
        target=0.9,
        guard_multipliers=guards,
        relative_tolerance=0.0,
        absolute_tolerance=0.0,
    )
    assert not failed and not failed_checks["opponent_mean"]["passed"]
