from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cg_frontier.compression.affine_pca import (
    EnhancedPCASpec,
    fit_global_valid_pca_oracle,
    fit_clustered_valid_pca_oracle,
    optimize_pca_latent_frame,
    calibrate_pca_safe_constrained,
    calibrate_pca_safe_enhanced,
    calibrate_pca_safe,
    encode_pca_latent,
    export_p0_constrained_bundle,
    export_p0_enhanced_bundle,
    export_p0_bundle,
    fit_uniform_valid_pca,
    fit_enhanced_valid_pca,
    raw_pca_hash,
    rasterize_uv_charts,
    reload_p0_bundle,
)


def test_enhanced_pca_preserves_generic_minority_chroma_better_than_uniform() -> None:
    generator = torch.Generator().manual_seed(97)
    count = 2048
    target = torch.zeros(count, 7, dtype=torch.float64)
    nuisance = torch.randn(count, 4, generator=generator, dtype=torch.float64)
    target[:, 0:3] = 0.12 + nuisance[:, :1] * 0.015
    target[:, 3] = nuisance[:, 0] * 0.20
    target[:, 4] = nuisance[:, 1] * 0.20
    target[:, 5] = 0.5 + nuisance[:, 2] * 0.20
    target[:, 6] = 0.5 + nuisance[:, 3] * 0.20
    minority = torch.zeros(count, dtype=torch.bool)
    minority[:96] = True
    target[minority, 0] += 0.32
    target[minority, 1] += 0.12
    target[minority, 2] -= 0.10
    atlas = target.reshape(32, 64, 7)
    valid = torch.ones(32, 64, dtype=torch.bool)

    uniform = fit_uniform_valid_pca(atlas, valid)
    enhanced = fit_enhanced_valid_pca(
        atlas,
        valid,
        EnhancedPCASpec(chroma_tail_strength=7.0),
    )
    uniform_prediction = uniform.mean + uniform.valid_scores @ uniform.components
    enhanced_prediction = enhanced.mean + enhanced.valid_scores @ enhanced.components
    uniform_error = torch.mean(
        torch.abs(uniform_prediction[minority, :3] - target[minority, :3])
    )
    enhanced_error = torch.mean(
        torch.abs(enhanced_prediction[minority, :3] - target[minority, :3])
    )

    assert enhanced_error < uniform_error * 0.7


def test_enhanced_pca_opponent_metric_preserves_color_separation() -> None:
    generator = torch.Generator().manual_seed(99)
    count = 2048
    target = torch.zeros(count, 7, dtype=torch.float64)
    nuisance = torch.randn(count, 4, generator=generator, dtype=torch.float64)
    target[:, :3] = 0.15 + nuisance[:, :1] * 0.01
    target[:, 3] = nuisance[:, 0] * 0.20
    target[:, 4] = nuisance[:, 1] * 0.20
    target[:, 5] = 0.5 + nuisance[:, 2] * 0.20
    target[:, 6] = 0.5 + nuisance[:, 3] * 0.20
    minority = torch.zeros(count, dtype=torch.bool)
    minority[:128] = True
    target[minority, 0] += 0.28
    target[minority, 1] += 0.08
    target[minority, 2] -= 0.12
    atlas = target.reshape(32, 64, 7)
    valid = torch.ones(32, 64, dtype=torch.bool)

    uniform = fit_uniform_valid_pca(atlas, valid)
    enhanced = fit_enhanced_valid_pca(
        atlas,
        valid,
        EnhancedPCASpec(opponent_chroma_weight=4.0),
    )
    uniform_prediction = uniform.mean + uniform.valid_scores @ uniform.components
    enhanced_prediction = enhanced.mean + enhanced.valid_scores @ enhanced.components
    source_separation = torch.mean(target[minority, 0] - target[minority, 2])
    uniform_separation = torch.mean(
        uniform_prediction[minority, 0] - uniform_prediction[minority, 2]
    )
    enhanced_separation = torch.mean(
        enhanced_prediction[minority, 0] - enhanced_prediction[minority, 2]
    )

    assert torch.abs(source_separation - enhanced_separation) < torch.abs(
        source_separation - uniform_separation
    ) * 0.5


def test_enhanced_pca_semantic_group_balance_protects_low_variance_rgb() -> None:
    generator = torch.Generator().manual_seed(100)
    rows = torch.randn(4096, 5, generator=generator, dtype=torch.float64)
    target = torch.zeros(4096, 7, dtype=torch.float64)
    target[:, 0] = 0.2 + rows[:, 0] * 0.025
    target[:, 1] = 0.2 - rows[:, 0] * 0.015
    target[:, 2] = 0.2 - rows[:, 0] * 0.010
    target[:, 3] = rows[:, 1] * 0.30
    target[:, 4] = rows[:, 2] * 0.30
    target[:, 5] = 0.5 + rows[:, 3] * 0.30
    target[:, 6] = 0.5 + rows[:, 4] * 0.30
    atlas = target.reshape(64, 64, 7)
    valid = torch.ones(64, 64, dtype=torch.bool)

    uniform = fit_uniform_valid_pca(atlas, valid)
    balanced = fit_enhanced_valid_pca(
        atlas,
        valid,
        EnhancedPCASpec(semantic_group_balance=True),
    )
    uniform_prediction = uniform.mean + uniform.valid_scores @ uniform.components
    balanced_prediction = balanced.mean + balanced.valid_scores @ balanced.components
    uniform_rgb_error = torch.mean(torch.abs(uniform_prediction[:, :3] - target[:, :3]))
    balanced_rgb_error = torch.mean(torch.abs(balanced_prediction[:, :3] - target[:, :3]))

    assert balanced_rgb_error < uniform_rgb_error * 0.1


def test_enhanced_pca_material_cluster_balance_protects_a_minority_regime() -> None:
    generator = torch.Generator().manual_seed(101)
    count = 4096
    target = torch.zeros(count, 7, dtype=torch.float64)
    nuisance = torch.randn(count, 4, generator=generator, dtype=torch.float64)
    target[:, :3] = 0.12 + nuisance[:, :1] * 0.008
    target[:, 3] = nuisance[:, 0] * 0.24
    target[:, 4] = nuisance[:, 1] * 0.24
    target[:, 5] = 0.5 + nuisance[:, 2] * 0.24
    target[:, 6] = 0.5 + nuisance[:, 3] * 0.24
    minority = torch.zeros(count, dtype=torch.bool)
    minority[:128] = True
    target[minority, 0] += 0.28
    target[minority, 1] += 0.10
    target[minority, 2] -= 0.10
    atlas = target.reshape(64, 64, 7)
    valid = torch.ones(64, 64, dtype=torch.bool)
    base_spec = EnhancedPCASpec()
    cluster_spec = EnhancedPCASpec(
        material_cluster_count=2,
        material_cluster_balance_power=1.0,
        material_cluster_seed=101,
    )

    base = fit_enhanced_valid_pca(atlas, valid, base_spec)
    first = fit_enhanced_valid_pca(atlas, valid, cluster_spec)
    second = fit_enhanced_valid_pca(atlas, valid, cluster_spec)
    base_prediction = base.mean + base.valid_scores @ base.components
    cluster_prediction = first.mean + first.valid_scores @ first.components
    base_error = torch.mean(
        torch.abs(base_prediction[minority, :3] - target[minority, :3])
    )
    cluster_error = torch.mean(
        torch.abs(cluster_prediction[minority, :3] - target[minority, :3])
    )

    assert cluster_error < base_error * 0.6
    assert raw_pca_hash(first) == raw_pca_hash(second)


def test_enhanced_pca_residual_tail_reweighting_reduces_worst_regime_error() -> None:
    generator = torch.Generator().manual_seed(103)
    count = 4096
    target = torch.zeros(count, 7, dtype=torch.float64)
    factors = torch.randn(count, 4, generator=generator, dtype=torch.float64)
    target[:, 0] = 0.20 + factors[:, 0] * 0.01
    target[:, 1] = 0.20 - factors[:, 0] * 0.01
    target[:, 2] = 0.20
    target[:, 3] = factors[:, 0] * 0.30
    target[:, 4] = factors[:, 1] * 0.30
    target[:, 5] = 0.5 + factors[:, 2] * 0.30
    target[:, 6] = 0.5 + factors[:, 3] * 0.30
    minority = torch.zeros(count, dtype=torch.bool)
    minority[:128] = True
    target[minority, 0] += 0.24
    target[minority, 1] += 0.08
    target[minority, 2] -= 0.08
    atlas = target.reshape(64, 64, 7)
    valid = torch.ones(64, 64, dtype=torch.bool)

    base = fit_enhanced_valid_pca(atlas, valid, EnhancedPCASpec())
    robust = fit_enhanced_valid_pca(
        atlas,
        valid,
        EnhancedPCASpec(
            residual_tail_strength=7.0,
            residual_reweight_iterations=2,
        ),
    )
    base_prediction = base.mean + base.valid_scores @ base.components
    robust_prediction = robust.mean + robust.valid_scores @ robust.components
    base_error = torch.mean(torch.abs(base_prediction[minority] - target[minority]))
    robust_error = torch.mean(
        torch.abs(robust_prediction[minority] - target[minority])
    )

    assert robust_error < base_error * 0.9


def test_global_rank_oracle_distinguishes_rank_four_from_rank_six_capacity() -> None:
    generator = torch.Generator().manual_seed(102)
    factors = torch.randn(1024, 6, generator=generator, dtype=torch.float64)
    basis = torch.randn(6, 7, generator=generator, dtype=torch.float64)
    target = (factors @ basis).reshape(32, 32, 7)
    valid = torch.ones(32, 32, dtype=torch.bool)

    rank_four = fit_global_valid_pca_oracle(target, valid, rank=4)
    rank_six = fit_global_valid_pca_oracle(target, valid, rank=6)
    rows = target[valid]
    error_four = torch.mean(
        torch.abs(rank_four.mean + rank_four.valid_scores @ rank_four.components - rows)
    )
    error_six = torch.mean(
        torch.abs(rank_six.mean + rank_six.valid_scores @ rank_six.components - rows)
    )

    assert rank_four.deployable is False
    assert rank_six.deployable is False
    assert error_four > 1.0e-2
    assert error_six < 1.0e-12


def test_clustered_pca_oracle_recovers_piecewise_rank_four_material() -> None:
    generator = torch.Generator().manual_seed(104)
    left_factor = torch.randn(512, 4, generator=generator, dtype=torch.float64)
    right_factor = torch.randn(512, 4, generator=generator, dtype=torch.float64)
    left_basis = torch.zeros(4, 7, dtype=torch.float64)
    right_basis = torch.zeros(4, 7, dtype=torch.float64)
    left_basis[:, :4] = torch.eye(4, dtype=torch.float64)
    right_basis[:, 3:] = torch.eye(4, dtype=torch.float64)
    left = left_factor @ left_basis
    right = right_factor @ right_basis + torch.tensor(
        (4.0, -3.0, 2.0, 0.0, 0.0, 0.0, 0.0), dtype=torch.float64
    )
    target = torch.cat((left, right), dim=0).reshape(32, 32, 7)
    valid = torch.ones(32, 32, dtype=torch.bool)

    global_four = fit_global_valid_pca_oracle(target, valid, rank=4)
    clustered = fit_clustered_valid_pca_oracle(
        target,
        valid,
        clusters=2,
        rank=4,
        seed=104,
    )
    rows = target[valid]
    global_error = torch.mean(
        torch.abs(global_four.mean + global_four.valid_scores @ global_four.components - rows)
    )
    clustered_error = torch.mean(torch.abs(clustered.valid_reconstruction - rows))

    assert clustered.deployable is False
    assert set(clustered.valid_assignments.tolist()) == {0, 1}
    assert clustered_error < global_error * 0.1


def test_enhanced_safe_calibration_reuses_chroma_weights_under_certificate() -> None:
    coordinate = torch.linspace(0.0, 1.0, 1024, dtype=torch.float64)
    target = torch.stack(
        (
            0.10 + 0.15 * coordinate,
            0.10 + 0.15 * coordinate,
            0.10 + 0.15 * coordinate,
            0.30 * torch.sin(coordinate * 11.0),
            0.30 * torch.cos(coordinate * 13.0),
            coordinate,
            coordinate.square(),
        ),
        dim=-1,
    )
    minority = torch.zeros(1024, dtype=torch.bool)
    minority[:64] = True
    target[minority, 0] += 0.45
    target[minority, 1] += 0.20
    target[minority, 2] -= 0.05
    atlas = target.reshape(32, 32, 7)
    valid = torch.ones(32, 32, dtype=torch.bool)
    spec = EnhancedPCASpec(
        chroma_tail_strength=7.0,
        opponent_chroma_weight=2.0,
        semantic_group_balance=True,
    )
    encoded = encode_pca_latent(fit_enhanced_valid_pca(atlas, valid, spec))

    uniform = calibrate_pca_safe_constrained(encoded, target, margin=1.0e-3)
    enhanced = calibrate_pca_safe_enhanced(
        encoded,
        target,
        spec,
        margin=1.0e-3,
    )
    latent = enhanced.safe.latent_unorm8.to(target.dtype) / 255.0
    uniform_prediction = torch.nn.functional.linear(
        latent, uniform.safe.weight, uniform.safe.bias
    )
    enhanced_prediction = torch.nn.functional.linear(
        latent, enhanced.safe.weight, enhanced.safe.bias
    )
    uniform_error = torch.mean(
        torch.abs(uniform_prediction[minority, :3] - target[minority, :3])
    )
    enhanced_error = torch.mean(
        torch.abs(enhanced_prediction[minority, :3] - target[minority, :3])
    )

    assert enhanced.safe.certificate["valid"] is True
    assert enhanced_error < uniform_error


def test_enhanced_safe_calibration_reuses_material_cluster_weights() -> None:
    generator = torch.Generator().manual_seed(105)
    target = torch.rand(2048, 7, generator=generator, dtype=torch.float64)
    target[:, :3] *= 0.18
    minority = torch.zeros(2048, dtype=torch.bool)
    minority[:64] = True
    target[minority, 0] = 0.92
    target[minority, 1] = 0.58
    target[minority, 2] = 0.08
    atlas = target.reshape(32, 64, 7)
    valid = torch.ones(32, 64, dtype=torch.bool)
    spec = EnhancedPCASpec(
        material_cluster_count=2,
        material_cluster_balance_power=1.0,
        material_cluster_seed=105,
    )
    encoded = encode_pca_latent(fit_enhanced_valid_pca(atlas, valid, spec))

    uniform = calibrate_pca_safe_constrained(encoded, target, margin=1.0e-3)
    enhanced = calibrate_pca_safe_enhanced(
        encoded,
        target,
        spec,
        margin=1.0e-3,
    )
    latent = enhanced.safe.latent_unorm8.to(target.dtype) / 255.0
    uniform_prediction = torch.nn.functional.linear(
        latent, uniform.safe.weight, uniform.safe.bias
    )
    enhanced_prediction = torch.nn.functional.linear(
        latent, enhanced.safe.weight, enhanced.safe.bias
    )
    uniform_error = torch.mean(
        torch.abs(uniform_prediction[minority, :3] - target[minority, :3])
    )
    enhanced_error = torch.mean(
        torch.abs(enhanced_prediction[minority, :3] - target[minority, :3])
    )

    assert enhanced.safe.certificate["valid"] is True
    assert enhanced_error < uniform_error * 0.8


def test_enhanced_safe_calibration_reuses_residual_tail_weights() -> None:
    generator = torch.Generator().manual_seed(107)
    target = torch.rand(2048, 7, generator=generator, dtype=torch.float64)
    target[:, :3] *= 0.16
    minority = torch.zeros(2048, dtype=torch.bool)
    minority[:64] = True
    target[minority, :3] = torch.tensor((0.88, 0.54, 0.06), dtype=torch.float64)
    atlas = target.reshape(32, 64, 7)
    valid = torch.ones(32, 64, dtype=torch.bool)
    spec = EnhancedPCASpec(
        residual_tail_strength=7.0,
        residual_reweight_iterations=2,
    )
    encoded = encode_pca_latent(fit_enhanced_valid_pca(atlas, valid, spec))

    uniform = calibrate_pca_safe_constrained(encoded, target, margin=1.0e-3)
    enhanced = calibrate_pca_safe_enhanced(
        encoded,
        target,
        spec,
        margin=1.0e-3,
    )
    latent = enhanced.safe.latent_unorm8.to(target.dtype) / 255.0
    uniform_prediction = torch.nn.functional.linear(
        latent, uniform.safe.weight, uniform.safe.bias
    )
    enhanced_prediction = torch.nn.functional.linear(
        latent, enhanced.safe.weight, enhanced.safe.bias
    )
    uniform_error = torch.mean(
        torch.abs(uniform_prediction[minority] - target[minority])
    )
    enhanced_error = torch.mean(
        torch.abs(enhanced_prediction[minority] - target[minority])
    )

    assert enhanced.safe.certificate["valid"] is True
    assert enhanced_error < uniform_error * 0.9


def test_safety_aware_frame_preserves_raw_reconstruction_and_never_worsens_proxy() -> None:
    generator = torch.Generator().manual_seed(106)
    target = torch.rand(24, 24, 7, generator=generator, dtype=torch.float64)
    target[..., 3:5] = (target[..., 3:5] - 0.5) * 1.2
    valid = torch.ones(24, 24, dtype=torch.bool)
    fitted = fit_enhanced_valid_pca(
        target,
        valid,
        EnhancedPCASpec(
            chroma_tail_strength=7.0,
            opponent_chroma_weight=2.0,
            semantic_group_balance=True,
        ),
    )
    reference = fitted.mean + fitted.valid_scores @ fitted.components

    optimized = optimize_pca_latent_frame(fitted, margin=1.0e-3)
    reconstructed = (
        optimized.pca.mean
        + optimized.pca.valid_scores @ optimized.pca.components
    )

    assert torch.allclose(reconstructed, reference, atol=1.0e-12, rtol=1.0e-12)
    assert optimized.optimized_proxy <= optimized.identity_proxy
    assert optimized.rotation.shape == (4, 4)


def test_enhanced_p0_bundle_is_distinct_deterministic_and_reloadable() -> None:
    generator = torch.Generator().manual_seed(108)
    target = torch.rand(12, 13, 7, generator=generator, dtype=torch.float64)
    target[..., 3:5] = (target[..., 3:5] - 0.5) * 1.4
    valid = torch.rand(12, 13, generator=generator) > 0.15
    charts = torch.full((12, 13), -1, dtype=torch.int64)
    charts[valid] = 0
    spec = EnhancedPCASpec(
        chroma_tail_strength=7.0,
        opponent_chroma_weight=2.0,
        semantic_group_balance=True,
        material_cluster_count=2,
        material_cluster_balance_power=0.5,
        material_cluster_seed=108,
        residual_tail_strength=3.0,
        residual_reweight_iterations=1,
    )

    first = export_p0_enhanced_bundle(
        target, valid, charts, spec=spec, margin=1.0e-3
    )
    second = export_p0_enhanced_bundle(
        target.clone(), valid.clone(), charts.clone(), spec=spec, margin=1.0e-3
    )
    reloaded = reload_p0_bundle(first)

    assert first.manifest == second.manifest
    assert first.files == second.files
    assert first.manifest["pipeline_id"] == "scifihelmet_c4_affine_pca_enhanced_v1"
    assert first.manifest["pca"]["semantic_group_balance"] is True
    assert first.manifest["pca"]["opponent_chroma_weight"] == 2.0
    assert first.manifest["pca"]["chroma_tail_strength"] == 7.0
    assert first.manifest["pca"]["material_cluster_count"] == 2
    assert first.manifest["pca"]["material_cluster_balance_power"] == 0.5
    assert first.manifest["pca"]["material_cluster_seed"] == 108
    assert first.manifest["pca"]["residual_tail_strength"] == 3.0
    assert first.manifest["pca"]["residual_reweight_iterations"] == 1
    assert first.manifest["frame_optimization"]["optimized_proxy"] <= (
        first.manifest["frame_optimization"]["identity_proxy"]
    )
    assert first.calibration.raw.artifact_id == "p0-raw-enhanced-v1"
    assert first.calibration.safe.artifact_id == "p0-safe-enhanced-v1"
    assert first.calibration.safe.certificate["valid"] is True
    assert reloaded.latent_rgba_unorm8.shape == (12, 13, 4)
    assert len(first.files["decoder.bin"]) == 140


def test_pca_ignores_invalid_atlas_values_and_matches_valid_rows() -> None:
    generator = torch.Generator().manual_seed(101)
    atlas = torch.randn(4, 5, 7, generator=generator, dtype=torch.float64)
    valid = torch.tensor(
        [
            [True, True, False, False, True],
            [True, False, False, True, True],
            [False, True, True, True, False],
            [True, True, False, True, False],
        ]
    )
    changed_invalid = atlas.clone()
    changed_invalid[~valid] = torch.randn(
        changed_invalid[~valid].shape, generator=generator, dtype=torch.float64
    ) * 1.0e6

    original = fit_uniform_valid_pca(atlas, valid)
    changed = fit_uniform_valid_pca(changed_invalid, valid)
    explicit = fit_uniform_valid_pca(
        atlas[valid].reshape(-1, 1, 7),
        torch.ones(atlas[valid].shape[0], 1, dtype=torch.bool),
    )

    assert torch.equal(original.mean, changed.mean)
    assert torch.equal(original.components, changed.components)
    assert torch.equal(original.valid_scores, changed.valid_scores)
    assert torch.equal(original.mean, explicit.mean)
    assert torch.equal(original.components, explicit.components)
    assert torch.equal(original.valid_scores, explicit.valid_scores)


def test_pca_is_sign_canonical_deterministic_and_unscaled() -> None:
    generator = torch.Generator().manual_seed(103)
    target = torch.randn(8, 9, 7, generator=generator, dtype=torch.float64)
    target[..., 0] *= 100.0
    valid = torch.ones(8, 9, dtype=torch.bool)

    first = fit_uniform_valid_pca(target, valid)
    second = fit_uniform_valid_pca(target.clone(), valid.clone())
    pivot_indices = torch.argmax(torch.abs(first.components), dim=-1)
    pivots = first.components[
        torch.arange(first.components.shape[0]), pivot_indices
    ]

    assert torch.all(pivots > 0.0)
    assert torch.equal(first.mean, second.mean)
    assert torch.equal(first.components, second.components)
    assert torch.equal(first.valid_scores, second.valid_scores)
    assert raw_pca_hash(first) == raw_pca_hash(second)
    assert len(raw_pca_hash(first)) == 64
    assert torch.argmax(torch.abs(first.components[0])).item() == 0


def test_pca_degenerate_inputs_have_finite_zero_filled_missing_dimensions() -> None:
    constant = torch.full((3, 3, 7), 0.25, dtype=torch.float64)
    constant_fit = fit_uniform_valid_pca(
        constant, torch.ones(3, 3, dtype=torch.bool)
    )

    coordinates = torch.linspace(-1.0, 1.0, 12, dtype=torch.float64)
    rank_two_rows = torch.stack(
        (
            coordinates,
            coordinates.square(),
            2.0 * coordinates,
            -coordinates.square(),
            torch.zeros_like(coordinates),
            coordinates,
            coordinates.square(),
        ),
        dim=-1,
    )
    rank_two_fit = fit_uniform_valid_pca(
        rank_two_rows.reshape(3, 4, 7), torch.ones(3, 4, dtype=torch.bool)
    )

    single_mask = torch.zeros(2, 2, dtype=torch.bool)
    single_mask[1, 0] = True
    single_fit = fit_uniform_valid_pca(
        torch.arange(28, dtype=torch.float64).reshape(2, 2, 7), single_mask
    )

    for fitted, count in ((constant_fit, 9), (rank_two_fit, 12), (single_fit, 1)):
        assert fitted.components.shape == (4, 7)
        assert fitted.valid_scores.shape == (count, 4)
        assert torch.all(torch.isfinite(fitted.mean))
        assert torch.all(torch.isfinite(fitted.components))
        assert torch.all(torch.isfinite(fitted.valid_scores))
    assert torch.count_nonzero(constant_fit.components) == 0
    assert torch.count_nonzero(single_fit.components) == 0
    assert torch.count_nonzero(rank_two_fit.components[2:]) == 0
    assert torch.count_nonzero(rank_two_fit.valid_scores[:, 2:]) == 0


def test_pca_scores_map_exactly_to_unorm_latent_and_record_quantization_error() -> None:
    generator = torch.Generator().manual_seed(107)
    target = torch.rand(7, 8, 7, generator=generator, dtype=torch.float64)
    valid = torch.rand(7, 8, generator=generator) > 0.2
    fitted = fit_uniform_valid_pca(target, valid)
    encoded = encode_pca_latent(fitted)

    active = encoded.score_span > 0.0
    assert torch.equal(encoded.valid_latent[:, active].amin(dim=0), torch.zeros(4, dtype=torch.float64)[active])
    assert torch.equal(encoded.valid_latent[:, active].amax(dim=0), torch.ones(4, dtype=torch.float64)[active])
    assert torch.all(encoded.valid_latent[:, ~active] == 0.5)

    pca_reconstruction = fitted.mean + fitted.valid_scores @ fitted.components
    affine_reconstruction = torch.nn.functional.linear(
        encoded.valid_latent, encoded.weight, encoded.bias
    )
    assert torch.allclose(
        affine_reconstruction, pca_reconstruction, atol=1.0e-12, rtol=1.0e-12
    )

    quantized_latent = (
        torch.floor(torch.clamp(encoded.valid_latent, 0.0, 1.0) * 255.0 + 0.5)
        / 255.0
    )
    quantized_reconstruction = torch.nn.functional.linear(
        quantized_latent, encoded.weight, encoded.bias
    )
    expected_quantization_mae = torch.mean(
        torch.abs(quantized_reconstruction - affine_reconstruction)
    )
    assert encoded.quantization_material_mae == pytest.approx(
        expected_quantization_mae.item(), abs=1.0e-15
    )

    constant = fit_uniform_valid_pca(
        torch.ones(2, 2, 7, dtype=torch.float64),
        torch.ones(2, 2, dtype=torch.bool),
    )
    assert torch.all(encode_pca_latent(constant).valid_latent == 0.5)


def test_raw_and_continuously_safe_pca_are_distinct_hash_bound_artifacts() -> None:
    generator = torch.Generator().manual_seed(109)
    target = torch.rand(9, 10, 7, generator=generator, dtype=torch.float64)
    target[..., 3:5] = (target[..., 3:5] - 0.5) * 1.8
    valid = torch.rand(9, 10, generator=generator) > 0.15
    encoded = encode_pca_latent(fit_uniform_valid_pca(target, valid))

    calibration = calibrate_pca_safe(encoded, margin=1.0e-3)

    assert calibration.raw.artifact_id == "p0-raw"
    assert calibration.safe.artifact_id == "p0-safe"
    assert calibration.raw.artifact_hash != calibration.safe.artifact_hash
    assert calibration.raw.certificate is None
    assert calibration.safe.certificate is not None
    assert calibration.safe.certificate["valid"] is True
    assert not torch.equal(calibration.raw.weight, calibration.safe.weight)
    assert calibration.safety_material_mae_increment == pytest.approx(
        calibration.safe.material_mae - calibration.raw.material_mae,
        abs=1.0e-15,
    )
    folded_weight, folded_bias = calibration.safe_decoder.fold_affine()
    assert torch.allclose(
        folded_weight, calibration.safe.weight, atol=1.0e-12, rtol=1.0e-12
    )
    assert torch.allclose(
        folded_bias, calibration.safe.bias, atol=1.0e-12, rtol=1.0e-12
    )


def test_constrained_pca_calibration_preserves_valid_targets_better_than_radial() -> None:
    coordinate = torch.linspace(0.0, 1.0, 257, dtype=torch.float64)
    target = torch.stack(
        (
            coordinate,
            coordinate,
            coordinate,
            torch.zeros_like(coordinate),
            torch.zeros_like(coordinate),
            coordinate,
            coordinate,
        ),
        dim=-1,
    ).reshape(1, -1, 7)
    valid = torch.ones(target.shape[:2], dtype=torch.bool)
    encoded = encode_pca_latent(fit_uniform_valid_pca(target, valid))

    radial = calibrate_pca_safe(encoded, margin=1.0e-3)
    constrained = calibrate_pca_safe_constrained(
        encoded,
        target[valid],
        margin=1.0e-3,
    )

    assert constrained.safe.certificate["valid"] is True
    assert constrained.safe.material_mae < radial.safe.material_mae * 0.1
    assert constrained.safety_material_mae_increment >= 0.0


def test_p0_rgba8_decoder_manifest_export_is_repeatable_and_reloadable() -> None:
    generator = torch.Generator().manual_seed(113)
    target = torch.rand(6, 7, 7, generator=generator, dtype=torch.float64)
    valid = torch.rand(6, 7, generator=generator) > 0.2
    charts = torch.full((6, 7), -1, dtype=torch.int64)
    charts[valid] = torch.arange(torch.count_nonzero(valid), dtype=torch.int64) % 3

    first = export_p0_bundle(target, valid, charts, margin=1.0e-3)
    second = export_p0_bundle(target.clone(), valid.clone(), charts.clone(), margin=1.0e-3)
    reloaded = reload_p0_bundle(first)

    assert first.manifest == second.manifest
    assert first.files == second.files
    assert set(first.manifest["hashes"]) == {
        "input_valid_sha256",
        "valid_mask_sha256",
        "chart_ids_sha256",
        "raw_pca_sha256",
        "safe_calibration_sha256",
        "latent_png_sha256",
        "decoder_sha256",
    }
    assert first.manifest["raw_artifact"]["artifact_id"] == "p0-raw"
    assert first.manifest["safe_artifact"]["artifact_id"] == "p0-safe"
    assert reloaded.latent_rgba_unorm8.shape == (6, 7, 4)
    assert reloaded.latent_rgba_unorm8.dtype == torch.uint8
    assert torch.all(reloaded.latent_rgba_unorm8[~valid] == 128)
    assert first.calibration.raw.latent_unorm8.shape == (6, 7, 4)
    assert first.calibration.safe.latent_unorm8.shape == (6, 7, 4)
    assert torch.all(first.calibration.safe.latent_unorm8[~valid] == 128)
    assert torch.equal(
        reloaded.weight, first.calibration.safe.weight.float()
    )
    assert torch.equal(reloaded.bias, first.calibration.safe.bias.float())


def test_constrained_p0_bundle_reports_raw_gap_against_source_target() -> None:
    coordinate = torch.linspace(0.0, 1.0, 35, dtype=torch.float64).reshape(5, 7)
    target = torch.stack(
        (
            coordinate,
            coordinate.square(),
            coordinate,
            torch.zeros_like(coordinate),
            torch.zeros_like(coordinate),
            coordinate,
            coordinate.square(),
        ),
        dim=-1,
    )
    valid = torch.ones(target.shape[:2], dtype=torch.bool)
    charts = torch.zeros(target.shape[:2], dtype=torch.int64)

    bundle = export_p0_constrained_bundle(
        target,
        valid,
        charts,
        margin=1.0e-3,
    )

    metrics = bundle.manifest["target_error_metrics"]
    assert bundle.manifest["pipeline_id"] == "scifihelmet_c4_affine_pca_repair_v1"
    assert metrics["reference"] == "valid_source_seven"
    assert metrics["raw_target_material_mae"] == bundle.calibration.raw.material_mae
    assert metrics["safe_target_material_mae"] == bundle.calibration.safe.material_mae
    assert metrics["safety_target_material_mae_increment"] == (
        bundle.calibration.safety_material_mae_increment
    )
    assert metrics["quantization_material_mae_vs_pca"] >= 0.0


def test_constrained_p0_boundary_solution_exports_finite_decoder() -> None:
    coordinate = torch.linspace(0.0, 1.0, 257, dtype=torch.float32)
    target = torch.stack(
        (
            coordinate,
            coordinate,
            coordinate,
            torch.zeros_like(coordinate),
            torch.zeros_like(coordinate),
            coordinate,
            coordinate,
        ),
        dim=-1,
    ).reshape(1, -1, 7)
    valid = torch.ones(target.shape[:2], dtype=torch.bool)
    charts = torch.zeros(target.shape[:2], dtype=torch.int64)

    bundle = export_p0_constrained_bundle(
        target,
        valid,
        charts,
        margin=1.0e-3,
    )
    reloaded = reload_p0_bundle(bundle)

    assert len(bundle.files["decoder.bin"]) == 140
    assert torch.all(torch.isfinite(reloaded.weight))
    assert torch.all(torch.isfinite(reloaded.bias))


def test_uv_triangle_rasterization_produces_deterministic_two_chart_valid_mask() -> None:
    texcoords = np.asarray(
        [
            [0.05, 0.05], [0.40, 0.05], [0.40, 0.40],
            [0.05, 0.05], [0.40, 0.40], [0.05, 0.40],
            [0.60, 0.60], [0.95, 0.60], [0.95, 0.95],
            [0.60, 0.60], [0.95, 0.95], [0.60, 0.95],
        ],
        dtype=np.float32,
    )
    triangles = np.arange(12, dtype=np.int32).reshape(-1, 3)

    first_mask, first_charts = rasterize_uv_charts(
        texcoords, triangles, height=32, width=32
    )
    second_mask, second_charts = rasterize_uv_charts(
        texcoords.copy(), triangles.copy(), height=32, width=32
    )

    assert torch.equal(first_mask, second_mask)
    assert torch.equal(first_charts, second_charts)
    assert set(torch.unique(first_charts[first_mask]).tolist()) == {0, 1}
    assert torch.all(first_charts[~first_mask] == -1)
    assert torch.count_nonzero(first_mask) > 0
