from __future__ import annotations

from pathlib import Path
import sys

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cg_frontier.compression.affine_regularizers import (
    calibrate_gradient_ratios,
    chart_aware_quantized_tv,
    fake_quantize_unorm8,
    hard_quantize_unorm8,
)


def test_constant_fake_quantized_latent_has_zero_finite_tv_gradient() -> None:
    latent = torch.full((5, 6, 4), 0.375, dtype=torch.float64, requires_grad=True)
    valid = torch.ones(5, 6, dtype=torch.bool)
    charts = torch.zeros(5, 6, dtype=torch.int64)

    result = chart_aware_quantized_tv(latent, valid, charts)
    result.loss.backward()

    assert result.edge_count == 49
    assert result.loss.item() == 0.0
    assert latent.grad is not None
    assert torch.all(torch.isfinite(latent.grad))


def test_tv_excludes_invalid_and_cross_chart_edges_from_loss_and_gradient() -> None:
    latent = torch.tensor(
        [[[0.0], [1.0], [0.0], [1.0]]], dtype=torch.float64, requires_grad=True
    )
    valid = torch.tensor([[True, True, True, False]])
    charts = torch.tensor([[0, 1, 1, 1]])

    result = chart_aware_quantized_tv(latent, valid, charts)
    result.loss.backward()

    assert result.edge_count == 1
    assert latent.grad is not None
    assert latent.grad[0, 0, 0] == 0.0
    assert latent.grad[0, 3, 0] == 0.0
    assert latent.grad[0, 1, 0] != 0.0
    assert latent.grad[0, 2, 0] != 0.0


def test_tv_does_not_cross_island_seams_and_is_invariant_to_atlas_copy() -> None:
    island = torch.tensor(
        [[[0.1], [0.3]], [[0.7], [0.9]]], dtype=torch.float64
    )
    single = chart_aware_quantized_tv(
        island,
        torch.ones(2, 2, dtype=torch.bool),
        torch.zeros(2, 2, dtype=torch.int64),
    )
    duplicated = torch.cat(
        (island, torch.full((2, 1, 1), 1000.0, dtype=torch.float64), island),
        dim=1,
    )
    duplicated_valid = torch.tensor(
        [[True, True, False, True, True], [True, True, False, True, True]]
    )
    duplicated_charts = torch.tensor([[0, 0, -1, 1, 1], [0, 0, -1, 1, 1]])
    copied = chart_aware_quantized_tv(
        duplicated, duplicated_valid, duplicated_charts
    )

    assert copied.edge_count == 2 * single.edge_count
    assert torch.equal(copied.loss, single.loss)


def test_fake_quant_forward_matches_hard_unorm8_and_ste_backward_is_finite() -> None:
    latent = torch.tensor(
        [0.0, 0.1, 0.5, 127.5 / 255.0, 0.9, 1.0],
        dtype=torch.float64,
        requires_grad=True,
    )

    hard = hard_quantize_unorm8(latent)
    fake = fake_quantize_unorm8(latent)

    assert torch.equal(fake, hard)
    fake.sum().backward()
    assert latent.grad is not None
    assert torch.all(torch.isfinite(latent.grad))
    assert torch.all(latent.grad != 0.0)


def test_gradient_ratio_calibration_uses_eight_batch_median_without_update() -> None:
    latent = torch.linspace(0.1, 0.9, 12, dtype=torch.float64).requires_grad_(True)
    before = latent.detach().clone()
    batches = tuple(range(8))

    def base_objective(value: torch.Tensor, batch: int) -> torch.Tensor:
        return torch.mean((value - batch * 0.01) ** 2)

    def extension_objective(value: torch.Tensor, batch: int) -> torch.Tensor:
        return torch.mean(torch.sqrt((value + batch * 0.02) ** 2 + 0.01))

    report = calibrate_gradient_ratios(
        latent,
        batches,
        base_objective,
        extension_objective,
        target_ratios=(0.02, 0.05, 0.10),
    )

    assert report.batch_count == 8
    assert report.target_ratios == (0.02, 0.05, 0.10)
    assert report.lambdas[0.05] == pytest.approx(
        report.lambdas[0.02] * 2.5, rel=1.0e-12
    )
    assert report.lambdas[0.10] == pytest.approx(
        report.lambdas[0.02] * 5.0, rel=1.0e-12
    )
    assert torch.equal(latent.detach(), before)
    assert latent.grad is None
