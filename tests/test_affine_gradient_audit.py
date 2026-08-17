from __future__ import annotations

import pytest
import torch
from torch import nn

from cg_frontier.compression.affine_gradient_audit import (
    audit_gradient_objectives,
    calibrate_static_color_budgets,
    calibrate_static_risk_budgets,
)


def test_gradient_audit_reports_two_groups_without_mutating_parameters() -> None:
    latent = nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float64))
    affine = nn.Parameter(torch.tensor([0.5, 1.5], dtype=torch.float64))
    latent_before = latent.detach().clone()
    affine_before = affine.detach().clone()

    def terms(batch: torch.Tensor) -> dict[str, torch.Tensor]:
        base = torch.sum(latent * batch) + torch.sum(affine * batch.flip(0))
        opponent = torch.sum(latent * batch.flip(0)) - torch.sum(affine * batch)
        pair = torch.sum((latent + affine) * batch.square())
        return {"base": base, "opponent": opponent, "pair": pair}

    report = audit_gradient_objectives(
        batches=(
            torch.tensor([1.0, 2.0], dtype=torch.float64),
            torch.tensor([2.0, -1.0], dtype=torch.float64),
        ),
        objective_terms=terms,
        parameter_groups={"latent": (latent,), "affine": (affine,)},
    )

    assert report["batch_count"] == 2
    assert report["objective_names"] == ["base", "opponent", "pair"]
    for batch in report["batches"]:
        for group in ("latent", "affine"):
            cosine = batch["parameter_groups"][group]["cosine"]
            assert cosine["base"]["base"] == 1.0
            assert cosine["base"]["opponent"] == cosine["opponent"]["base"]
    assert torch.equal(latent, latent_before)
    assert torch.equal(affine, affine_before)
    assert latent.grad is None
    assert affine.grad is None

    calibration = calibrate_static_color_budgets(
        report, ratios=(0.10, 0.25, 0.50)
    )
    assert calibration["ratios"] == [0.1, 0.25, 0.5]
    assert set(calibration["candidates"]) == {"0.100000", "0.250000", "0.500000"}
    assert calibration["scales"]["opponent"] > 0.0
    assert calibration["scales"]["pair"] > 0.0


def test_risk_budget_calibration_assigns_equal_point_one_total_budget() -> None:
    latent = nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.float64))
    affine = nn.Parameter(torch.tensor([0.5, 1.5], dtype=torch.float64))

    def terms(batch: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "base": torch.sum(latent * batch) + torch.sum(affine * batch.flip(0)),
            "mean": torch.sum(latent * batch.flip(0)) - torch.sum(affine * batch),
            "yc_cvar25": torch.sum((latent + affine) * batch.square()),
            "hue_macro": torch.sum((latent - affine) * (batch + 0.5)),
        }

    audit = audit_gradient_objectives(
        batches=(
            torch.tensor([1.0, 2.0], dtype=torch.float64),
            torch.tensor([2.0, -1.0], dtype=torch.float64),
        ),
        objective_terms=terms,
        parameter_groups={"latent": (latent,), "affine": (affine,)},
    )
    calibration = calibrate_static_risk_budgets(audit, total_ratio=0.10)

    assert set(calibration["scales"]) == {"mean", "yc_cvar25", "hue_macro"}
    candidates = calibration["candidates"]
    assert candidates["G0-mean"]["weights"]["mean"] == pytest.approx(
        0.10 * calibration["scales"]["mean"]
    )
    assert candidates["G3-cvar25-hue8"]["weights"] == pytest.approx(
        {
            "yc_cvar25": 0.05 * calibration["scales"]["yc_cvar25"],
            "hue_macro": 0.05 * calibration["scales"]["hue_macro"],
        }
    )
