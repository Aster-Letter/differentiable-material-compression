from __future__ import annotations

import pytest
import torch
from torch import nn
from pathlib import Path

from cg_frontier.compression.adaptive_basecolor import (
    RenderColorVisibility,
    adaptive_group_chroma_loss,
    build_adaptive_basecolor_profile,
    build_render_color_visibility,
    draw_adaptive_color_batch,
    estimate_neutral_chroma_threshold,
    orthogonal_basecolor_losses,
    orthogonal_error_components,
    weighted_orthogonal_domain_loss,
    weighted_lbg,
    visibility_corrected_group_loss,
)
from cg_frontier.compression.raw_orthogonal_training import (
    load_raw_orthogonal_checkpoint,
    save_raw_orthogonal_checkpoint,
)


def test_orthogonal_basecolor_losses_separate_achromatic_and_chroma_error() -> None:
    source = torch.tensor([[0.8, 0.2, 0.2]], dtype=torch.float64)
    prediction = source.mean(dim=-1, keepdim=True).expand_as(source).clone()

    y_error, chroma_error = orthogonal_basecolor_losses(prediction, source)

    assert y_error == pytest.approx(0.0, abs=1.0e-12)
    assert float(chroma_error) > 0.0


def test_orthogonal_basecolor_losses_are_zero_at_identity_and_permutation_invariant() -> None:
    source = torch.tensor(
        [[0.8, 0.1, 0.25], [0.04, 0.6, 0.2]], dtype=torch.float64
    )
    prediction = torch.tensor(
        [[0.7, 0.15, 0.3], [0.08, 0.5, 0.25]],
        dtype=torch.float64,
        requires_grad=True,
    )
    permutation = torch.tensor([2, 0, 1])

    identity_y, identity_chroma = orthogonal_basecolor_losses(source, source)
    y_error, chroma_error = orthogonal_basecolor_losses(prediction, source)
    permuted_y, permuted_chroma = orthogonal_basecolor_losses(
        prediction[:, permutation], source[:, permutation]
    )
    (y_error + chroma_error).backward()

    assert identity_y == pytest.approx(0.0, abs=1.0e-12)
    assert identity_chroma == pytest.approx(0.0, abs=1.0e-12)
    assert float(y_error.detach()) == pytest.approx(
        float(permuted_y.detach()), abs=1.0e-12
    )
    assert float(chroma_error.detach()) == pytest.approx(
        float(permuted_chroma.detach()), abs=1.0e-12
    )
    assert prediction.grad is not None
    assert bool(torch.isfinite(prediction.grad).all())
    assert float(torch.linalg.vector_norm(prediction.grad)) > 0.0


def test_otsu_neutral_threshold_separates_repeated_log_chroma_modes_deterministically() -> None:
    chroma = torch.cat(
        (
            torch.full((5000,), 1.0e-3, dtype=torch.float64),
            torch.full((7000,), 2.0e-1, dtype=torch.float64),
        )
    )

    first = estimate_neutral_chroma_threshold(chroma, bins=2048, min_side_size=4096)
    second = estimate_neutral_chroma_threshold(chroma, bins=2048, min_side_size=4096)

    assert 1.0e-3 <= first.threshold < 2.0e-1
    assert first.neutral_count == 5000
    assert first.colored_count == 7000
    assert first == second


@pytest.mark.parametrize(
    "chroma",
    [
        torch.ones(9000, dtype=torch.float64),
        torch.cat(
            (
                torch.full((100,), 1.0e-3, dtype=torch.float64),
                torch.full((9000,), 2.0e-1, dtype=torch.float64),
            )
        ),
        torch.tensor([1.0e-3, float("nan"), 0.2], dtype=torch.float64),
    ],
)
def test_otsu_neutral_threshold_fails_closed_on_degenerate_inputs(
    chroma: torch.Tensor,
) -> None:
    with pytest.raises(ValueError):
        estimate_neutral_chroma_threshold(chroma, bins=2048, min_side_size=4096)


def test_weighted_lbg_matches_explicit_duplicates_and_canonicalizes_labels() -> None:
    unique = torch.tensor(
        [[-1.0, 0.0], [-0.9, 0.05], [0.9, -0.05], [1.0, 0.0]],
        dtype=torch.float64,
    )
    multiplicity = torch.tensor([11, 7, 5, 13], dtype=torch.int64)
    expanded = torch.repeat_interleave(unique, multiplicity, dim=0)

    weighted = weighted_lbg(unique, multiplicity, clusters=2, seed=17, restarts=4)
    explicit = weighted_lbg(
        expanded,
        torch.ones(expanded.shape[0], dtype=torch.int64),
        clusters=2,
        seed=17,
        restarts=4,
    )

    assert torch.allclose(weighted.centroids, explicit.centroids, atol=1.0e-12, rtol=0.0)
    assert weighted.distortion == pytest.approx(explicit.distortion, abs=1.0e-14)
    angles = torch.atan2(weighted.centroids[:, 1], weighted.centroids[:, 0])
    assert bool(torch.all(angles[1:] >= angles[:-1]))
    assert torch.equal(weighted.centroids, weighted_lbg(unique, multiplicity, clusters=2, seed=17, restarts=4).centroids)


def _opponent_to_rgb(coordinates: torch.Tensor) -> torch.Tensor:
    transform = torch.tensor(
        [
            [1.0 / 3.0**0.5, 1.0 / 2.0**0.5, 1.0 / 6.0**0.5],
            [1.0 / 3.0**0.5, -1.0 / 2.0**0.5, 1.0 / 6.0**0.5],
            [1.0 / 3.0**0.5, 0.0, -2.0 / 6.0**0.5],
        ],
        dtype=coordinates.dtype,
    )
    return coordinates @ transform.T


def test_adaptive_profile_selects_rate_distortion_k_and_hashes_membership() -> None:
    generator = torch.Generator().manual_seed(41)
    neutral = torch.randn((90, 2), generator=generator, dtype=torch.float64) * 1.0e-4
    colored = torch.cat(
        [
            center + torch.randn((60, 2), generator=generator, dtype=torch.float64) * 2.0e-3
            for center in (
                torch.tensor([0.18, 0.0], dtype=torch.float64),
                torch.tensor([-0.09, 0.156], dtype=torch.float64),
                torch.tensor([-0.09, -0.156], dtype=torch.float64),
            )
        ]
    )
    opponent = torch.cat((neutral, colored))
    y = torch.full((opponent.shape[0], 1), 0.75, dtype=torch.float64)
    source = _opponent_to_rgb(torch.cat((y, opponent), dim=-1))

    profile = build_adaptive_basecolor_profile(
        source,
        source_hash="source",
        input_hash="input",
        config_hash="config",
        bins=64,
        min_group_size=20,
        max_clusters=3,
        restarts=4,
        seed=13,
    )
    repeated = build_adaptive_basecolor_profile(
        source,
        source_hash="source",
        input_hash="input",
        config_hash="config",
        bins=64,
        min_group_size=20,
        max_clusters=3,
        restarts=4,
        seed=13,
    )

    assert profile.k == 3
    assert profile.group_sizes.tolist() == [90, 60, 60, 60]
    assert len(profile.distortion_curve) == 3
    assert len(profile.jump_curve) == 2
    assert profile.profile_hash == repeated.profile_hash
    assert torch.equal(profile.valid_group_ids, repeated.valid_group_ids)


def test_render_visibility_uses_bilinear_source_samples_and_pixel_support() -> None:
    opponent = torch.cat(
        (
            torch.zeros((12, 2), dtype=torch.float64),
            torch.tensor([[0.2, 0.0]], dtype=torch.float64).repeat(12, 1),
        )
    )
    source = _opponent_to_rgb(
        torch.cat((torch.full((24, 1), 0.75, dtype=torch.float64), opponent), dim=-1)
    )
    profile = build_adaptive_basecolor_profile(
        source,
        source_hash="source",
        input_hash="input",
        config_hash="config",
        bins=16,
        min_group_size=4,
        max_clusters=1,
        restarts=2,
        seed=3,
    )
    atlas = source[[0, 12, 0, 12]].reshape(2, 2, 3)
    uv = torch.tensor(
        [[[0.25, 0.25], [0.75, 0.25]], [[0.25, 0.75], [0.75, 0.75]]],
        dtype=torch.float64,
    )
    masks = [
        torch.tensor([[True, True], [False, False]]),
        torch.tensor([[False, False], [True, True]]),
    ]

    visibility = build_render_color_visibility(
        atlas,
        [uv, uv],
        masks,
        profile,
        min_pixels=1,
        min_cameras=2,
    )

    assert visibility.counts.tolist() == [[1, 1], [1, 1]]
    assert visibility.active_mask.tolist() == [[True, True], [True, True]]
    assert visibility.visible_camera_counts.tolist() == [2, 2]


def test_inverse_visibility_correction_is_unbiased_for_camera_macro_average() -> None:
    visibility = RenderColorVisibility(
        counts=torch.tensor([[5, 5, 0], [5, 0, 5]], dtype=torch.int64),
        active_mask=torch.tensor([[True, True, False], [True, False, True]]),
        visible_camera_counts=torch.tensor([2, 1, 1], dtype=torch.int64),
        min_pixels=1,
        min_cameras=1,
        profile_hash="profile",
        visibility_hash="visibility",
    )
    camera_zero = visibility_corrected_group_loss(
        torch.tensor([2.0, 4.0]),
        torch.tensor([0, 1]),
        camera_index=0,
        visibility=visibility,
    )
    camera_one = visibility_corrected_group_loss(
        torch.tensor([2.0, 8.0]),
        torch.tensor([0, 2]),
        camera_index=1,
        visibility=visibility,
    )

    explicit_macro = 0.5 * 2.0 + 0.25 * (4.0 + 8.0)
    assert float((camera_zero + camera_one) / 2.0) == pytest.approx(explicit_macro)


def test_adaptive_color_sampler_is_balanced_reproducible_and_rng_independent() -> None:
    profile_source = _opponent_to_rgb(
        torch.cat(
            (
                torch.full((24, 1), 0.75, dtype=torch.float64),
                torch.cat(
                    (
                        torch.zeros((12, 2), dtype=torch.float64),
                        torch.tensor([[0.2, 0.0]], dtype=torch.float64).repeat(12, 1),
                    )
                ),
            ),
            dim=-1,
        )
    )
    profile = build_adaptive_basecolor_profile(
        profile_source,
        source_hash="source",
        input_hash="input",
        config_hash="config",
        bins=16,
        min_group_size=4,
        max_clusters=1,
        restarts=2,
        seed=3,
    )
    core_rng = torch.Generator().manual_seed(10)
    core_before = core_rng.get_state().clone()
    first_rng = torch.Generator().manual_seed(33)
    second_rng = torch.Generator().manual_seed(33)

    first = draw_adaptive_color_batch(profile, samples_per_group=7, generator=first_rng)
    second = draw_adaptive_color_batch(profile, samples_per_group=7, generator=second_rng)

    assert torch.equal(first.valid_positions, second.valid_positions)
    assert torch.equal(first.group_ids, second.group_ids)
    assert torch.bincount(first.group_ids).tolist() == [7, 7]
    assert torch.equal(core_before, core_rng.get_state())


def test_adaptive_chroma_macro_keeps_neutral_and_colored_top_weights_constant_across_k() -> None:
    one_color = adaptive_group_chroma_loss(
        torch.tensor([1.0, 3.0]), torch.tensor([0, 1]), colored_group_count=1
    )
    three_colors = adaptive_group_chroma_loss(
        torch.tensor([1.0, 3.0, 3.0, 3.0]),
        torch.tensor([0, 1, 2, 3]),
        colored_group_count=3,
    )

    assert float(one_color) == pytest.approx(2.0)
    assert float(three_colors) == pytest.approx(2.0)


def test_weighted_orthogonal_domain_loss_uses_frozen_y_c_budget_formula() -> None:
    prediction = torch.tensor([[0.7, 0.2, 0.1], [0.1, 0.6, 0.3]], dtype=torch.float64)
    source = torch.tensor([[0.6, 0.25, 0.15], [0.2, 0.5, 0.3]], dtype=torch.float64)
    y_errors, chroma_errors = orthogonal_error_components(prediction, source)

    total, terms = weighted_orthogonal_domain_loss(
        y_errors,
        chroma_errors.mean(),
        ratio=0.25,
        y_scale=2.0,
        chroma_scale=3.0,
    )
    expected = 0.75 * 2.0 * y_errors.mean() + 0.25 * 3.0 * chroma_errors.mean()

    assert torch.equal(total, expected)
    assert torch.equal(terms["y"], y_errors.mean())
    assert torch.equal(terms["chroma"], chroma_errors.mean())


def test_raw_v2_checkpoint_exactly_resumes_parameters_optimizers_and_both_rngs() -> None:
    def make_state():
        latent = nn.Parameter(torch.tensor([0.2, 0.7], dtype=torch.float64))
        affine = nn.Parameter(torch.tensor([0.5], dtype=torch.float64))
        latent_optimizer = torch.optim.Adam([latent], lr=1.0e-2)
        affine_optimizer = torch.optim.Adam([affine], lr=2.0e-2)
        return latent, affine, latent_optimizer, affine_optimizer

    def update(state, core_rng, color_rng):
        latent, affine, latent_optimizer, affine_optimizer = state
        core = torch.rand((), generator=core_rng, dtype=torch.float64)
        color = torch.rand((), generator=color_rng, dtype=torch.float64)
        loss = ((latent * core).sum() + affine.sum() * color).square()
        latent_optimizer.zero_grad(set_to_none=True)
        affine_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        latent_optimizer.step()
        affine_optimizer.step()
        return loss.detach(), core, color

    lineage = {
        "parent_artifact_hash": "parent",
        "config_sha256": "config",
        "input_sha256": "input",
        "basecolor_profile_hash": "profile",
        "visibility_hash": "visibility",
    }
    state = make_state()
    core_rng = torch.Generator().manual_seed(5)
    color_rng = torch.Generator().manual_seed(28)
    update(state, core_rng, color_rng)
    path = Path("tests/.raw-v2-checkpoint-test.pt")
    path.unlink(missing_ok=True)
    save_raw_orthogonal_checkpoint(
        path,
        step=1,
        candidate_id="O2-r025",
        objective_id="O2",
        ratio=0.25,
        latent=state[0],
        weight=state[1],
        bias=nn.Parameter(torch.zeros(1, dtype=torch.float64)),
        latent_optimizer=state[2],
        affine_optimizer=state[3],
        core_rng=core_rng,
        color_rng=color_rng,
        lineage=lineage,
    )
    continuous = update(state, core_rng, color_rng)

    resumed = make_state()
    resumed_core = torch.Generator()
    resumed_color = torch.Generator()
    payload = load_raw_orthogonal_checkpoint(
        path,
        expected_candidate_id="O2-r025",
        expected_objective_id="O2",
        expected_ratio=0.25,
        expected_lineage=lineage,
    )
    resumed[0].data.copy_(payload["latent"])
    resumed[1].data.copy_(payload["weight"])
    resumed[2].load_state_dict(payload["latent_optimizer"])
    resumed[3].load_state_dict(payload["affine_optimizer"])
    resumed_core.set_state(payload["rng_state"])
    resumed_color.set_state(payload["color_rng_state"])
    after_resume = update(resumed, resumed_core, resumed_color)

    for expected, actual in zip(continuous, after_resume, strict=True):
        assert torch.equal(expected, actual)
    assert torch.equal(state[0], resumed[0])
    assert torch.equal(state[1], resumed[1])
    path.unlink()
