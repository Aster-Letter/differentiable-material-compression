from __future__ import annotations

import torch
import pytest

from cg_frontier.compression.affine_color import (
    build_color_quantile_partition,
    build_color_hue_partition,
    color_quality_metrics,
    color_risk_quality_metrics,
    draw_color_batch,
    empirical_cvar,
    freeze_color_metric_pairs,
    grouped_empirical_cvar,
    grouped_mean,
    opponent_vector_charbonnier_per_texel,
    opponent_vector_charbonnier,
    oklab_mean_delta_e,
    orthogonal_color_coordinates,
)


def test_oklab_is_report_only_identity_zero_and_color_change_positive() -> None:
    source = torch.tensor([[0.8, 0.1, 0.05], [0.2, 0.2, 0.2]], dtype=torch.float64)

    assert oklab_mean_delta_e(source, source) == pytest.approx(0.0, abs=1.0e-12)
    changed = source.clone()
    changed[0] = torch.tensor([0.05, 0.8, 0.1], dtype=torch.float64)
    value = oklab_mean_delta_e(changed, source)
    assert torch.isfinite(value)
    assert float(value) > 0.0


def test_identity_base_color_has_zero_opponent_loss() -> None:
    rgb = torch.tensor(
        [[0.05, 0.20, 0.80], [0.60, 0.10, 0.35]], dtype=torch.float64
    )

    coordinates = orthogonal_color_coordinates(rgb)
    loss = opponent_vector_charbonnier(
        coordinates[..., 1:], coordinates[..., 1:]
    )

    assert torch.equal(loss, torch.zeros_like(loss))


def test_rgb_permutation_preserves_chroma_norm_and_opponent_error() -> None:
    source = torch.tensor(
        [[0.80, 0.10, 0.25], [0.04, 0.60, 0.20]], dtype=torch.float64
    )
    prediction = torch.tensor(
        [[0.70, 0.15, 0.30], [0.08, 0.50, 0.25]], dtype=torch.float64
    )
    permutation = torch.tensor([2, 0, 1])

    source_coordinates = orthogonal_color_coordinates(source)
    prediction_coordinates = orthogonal_color_coordinates(prediction)
    permuted_source = orthogonal_color_coordinates(source[:, permutation])
    permuted_prediction = orthogonal_color_coordinates(prediction[:, permutation])

    assert torch.allclose(
        torch.linalg.vector_norm(source_coordinates[:, 1:], dim=-1),
        torch.linalg.vector_norm(permuted_source[:, 1:], dim=-1),
        atol=1.0e-12,
        rtol=0.0,
    )
    assert torch.allclose(
        opponent_vector_charbonnier(
            prediction_coordinates[:, 1:], source_coordinates[:, 1:]
        ),
        opponent_vector_charbonnier(
            permuted_prediction[:, 1:], permuted_source[:, 1:]
        ),
        atol=1.0e-12,
        rtol=0.0,
    )


def test_non_yellow_red_to_green_error_has_finite_nonzero_gradient() -> None:
    source = torch.tensor([[0.90, 0.05, 0.05]], dtype=torch.float64)
    prediction = torch.tensor(
        [[0.05, 0.90, 0.05]], dtype=torch.float64, requires_grad=True
    )

    loss = opponent_vector_charbonnier(
        orthogonal_color_coordinates(prediction)[..., 1:],
        orthogonal_color_coordinates(source)[..., 1:],
    )
    loss.backward()

    assert float(loss.detach()) > 0.0
    assert prediction.grad is not None
    assert bool(torch.isfinite(prediction.grad).all())
    assert float(torch.linalg.vector_norm(prediction.grad)) > 0.0


def test_empirical_cvar_constant_matches_mean_and_fractional_tail_is_exact() -> None:
    constant = torch.full((8,), 2.5, dtype=torch.float64)
    assert empirical_cvar(constant, tail_mass=0.25) == pytest.approx(2.5)

    errors = torch.tensor([4.0, 3.0, 2.0, 1.0], dtype=torch.float64)
    # 37.5% of four samples is 1.5 samples: all of 4 and half of 3.
    assert empirical_cvar(errors, tail_mass=0.375) == pytest.approx((4.0 + 1.5) / 1.5)


def test_empirical_cvar_gradient_is_finite_and_only_reaches_fractional_tail() -> None:
    prediction = torch.tensor(
        [[4.0, 0.0], [3.0, 0.0], [2.0, 0.0], [1.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    target = torch.zeros_like(prediction)
    errors = opponent_vector_charbonnier_per_texel(prediction, target)

    empirical_cvar(errors, tail_mass=0.375).backward()

    assert prediction.grad is not None
    assert bool(torch.isfinite(prediction.grad).all())
    assert bool(torch.all(prediction.grad[:2, 0] > 0.0))
    assert torch.equal(prediction.grad[2:], torch.zeros_like(prediction.grad[2:]))


def test_grouped_risks_are_invariant_to_group_label_reordering() -> None:
    errors = torch.tensor([1.0, 4.0, 2.0, 3.0, 8.0, 6.0, 7.0, 5.0])
    groups = torch.tensor([10, 10, 10, 10, 20, 20, 20, 20])
    relabelled = torch.where(groups == 10, 7, 3)

    assert grouped_mean(errors, groups, torch.tensor([10, 20])) == pytest.approx(
        grouped_mean(errors, relabelled, torch.tensor([3, 7]))
    )
    assert grouped_empirical_cvar(
        errors, groups, torch.tensor([10, 20]), tail_mass=0.25
    ) == pytest.approx(
        grouped_empirical_cvar(
            errors, relabelled, torch.tensor([3, 7]), tail_mass=0.25
        )
    )


def test_grouped_risks_fail_closed_on_missing_nonfinite_or_short_groups() -> None:
    errors = torch.tensor([1.0, 2.0, 3.0, 4.0])
    groups = torch.tensor([0, 0, 0, 0])
    with pytest.raises(ValueError, match="missing required group"):
        grouped_mean(errors, groups, torch.tensor([0, 1]))
    with pytest.raises(ValueError, match="finite"):
        grouped_mean(
            torch.tensor([1.0, float("nan"), 3.0, 4.0]),
            groups,
            torch.tensor([0]),
        )


def test_hue_partition_is_deterministic_neutral_and_rgb_permutation_equivariant() -> None:
    source = torch.rand(
        2_048, 3, generator=torch.Generator().manual_seed(307), dtype=torch.float64
    )
    source[0] = 0.4
    base = build_color_quantile_partition(source)

    first = build_color_hue_partition(source, base, min_group_size=16)
    second = build_color_hue_partition(source, base, min_group_size=16)
    permutation = torch.tensor([2, 0, 1])
    permuted_base = build_color_quantile_partition(source[:, permutation])
    permuted = build_color_hue_partition(
        source[:, permutation], permuted_base, min_group_size=16
    )

    assert first.group_hash == second.group_hash
    assert torch.equal(first.valid_group_ids, second.valid_group_ids)
    assert int(first.valid_group_ids[0]) == 0
    assert first.group_count == 9
    assert torch.equal(
        first.valid_group_ids[:, None] == first.valid_group_ids[None, :],
        permuted.valid_group_ids[:, None] == permuted.valid_group_ids[None, :],
    )


def test_hue_partition_fails_closed_for_low_resultant_or_small_groups() -> None:
    angles = torch.arange(16, dtype=torch.float64) * (2.0 * torch.pi / 16.0)
    high_opponent = torch.stack((angles.cos(), angles.sin()), dim=-1) * 0.1
    high_rgb = torch.stack(
        (
            0.5
            + high_opponent[:, 0] / torch.sqrt(torch.tensor(2.0))
            + high_opponent[:, 1] / torch.sqrt(torch.tensor(6.0)),
            0.5
            - high_opponent[:, 0] / torch.sqrt(torch.tensor(2.0))
            + high_opponent[:, 1] / torch.sqrt(torch.tensor(6.0)),
            0.5 - 2.0 * high_opponent[:, 1] / torch.sqrt(torch.tensor(6.0)),
        ),
        dim=-1,
    )
    source = torch.cat((torch.full((16, 3), 0.5, dtype=torch.float64), high_rgb))
    base = build_color_quantile_partition(source)
    with pytest.raises(ValueError, match="resultant"):
        build_color_hue_partition(source, base, min_group_size=1)

    random_source = torch.rand(
        256, 3, generator=torch.Generator().manual_seed(311), dtype=torch.float64
    )
    random_base = build_color_quantile_partition(random_source)
    with pytest.raises(ValueError, match="minimum hue group"):
        build_color_hue_partition(random_source, random_base, min_group_size=64)
    with pytest.raises(ValueError, match="tail samples"):
        grouped_empirical_cvar(
            torch.tensor([1.0, 2.0, 3.0]),
            torch.tensor([0, 0, 0]),
            torch.tensor([0]),
            tail_mass=0.25,
        )


def test_quantile_color_batches_are_reproducible_balanced_and_cross_bin() -> None:
    source_generator = torch.Generator().manual_seed(9)
    source_rgb = torch.rand(20_000, 3, generator=source_generator, dtype=torch.float64)
    partition = build_color_quantile_partition(source_rgb)
    first_generator = torch.Generator().manual_seed(27)
    second_generator = torch.Generator().manual_seed(27)

    first = draw_color_batch(
        partition, generator=first_generator, batch_size=32_768
    )
    second = draw_color_batch(
        partition, generator=second_generator, batch_size=32_768
    )

    assert torch.equal(first.valid_positions, second.valid_positions)
    assert torch.equal(first.logical_bin_ids, second.logical_bin_ids)
    half = first.logical_bin_ids.numel() // 2
    assert bool(
        torch.all(first.logical_bin_ids[:half] != first.logical_bin_ids[half:])
    )
    counts = torch.bincount(
        first.active_bin_slots, minlength=partition.active_bin_count
    ).to(torch.float64)
    assert float((counts.max() - counts.min()) / counts.mean()) < 0.12


def test_metric_pairs_cover_every_active_bin_pair_without_training_rng() -> None:
    source_rgb = torch.rand(4_096, 3, generator=torch.Generator().manual_seed(31))
    partition = build_color_quantile_partition(source_rgb)
    training_rng = torch.Generator().manual_seed(41)
    training_state_before = training_rng.get_state().clone()

    first = freeze_color_metric_pairs(partition, seed=73, pairs_per_bin_pair=4)
    second = freeze_color_metric_pairs(partition, seed=73, pairs_per_bin_pair=4)

    expected = partition.active_bin_count * (partition.active_bin_count - 1) // 2 * 4
    assert first.left_valid_positions.numel() == expected
    assert torch.equal(first.left_valid_positions, second.left_valid_positions)
    assert torch.equal(first.right_valid_positions, second.right_valid_positions)
    assert first.pair_hash == second.pair_hash
    assert bool(torch.all(first.left_logical_bin_ids != first.right_logical_bin_ids))
    assert torch.equal(training_rng.get_state(), training_state_before)


def test_identity_color_metrics_are_zero_with_unit_contrast_retention() -> None:
    source = torch.rand(8_192, 3, generator=torch.Generator().manual_seed(101))
    partition = build_color_quantile_partition(source)
    pairs = freeze_color_metric_pairs(partition, seed=103, pairs_per_bin_pair=2)

    metrics = color_quality_metrics(source, source, partition, pairs)

    assert metrics["uniform_base_color_l1"] == 0.0
    assert metrics["uniform_opponent_error"] == 0.0
    assert metrics["macro_bin_opponent_error"] == 0.0
    assert metrics["worst_bin_opponent_error"] == 0.0
    assert metrics["fixed_pair_opponent_error"] == 0.0
    assert abs(metrics["chroma_contrast_retention"] - 1.0) < 1.0e-6


def test_identity_color_risk_metrics_are_zero() -> None:
    source = torch.rand(8_192, 3, generator=torch.Generator().manual_seed(337))
    partition = build_color_quantile_partition(source)
    hue_partition = build_color_hue_partition(source, partition, min_group_size=32)
    pairs = freeze_color_metric_pairs(partition, seed=347, pairs_per_bin_pair=2)

    metrics = color_risk_quality_metrics(
        source, source, partition, hue_partition, pairs
    )

    assert metrics["color_group_hash"] == hue_partition.group_hash
    assert metrics["tail_mass"] == 0.25
    for name in (
        "macro_bin_cvar25_opponent_error",
        "worst_bin_cvar25_opponent_error",
        "hue_macro_opponent_error",
        "worst_hue_group_opponent_error",
    ):
        assert metrics[name] == 0.0
