from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cg_frontier.compression.interpolation_analysis import (
    activation_crossings,
    bilinear_footprint_top_down_wrap,
    dark_fraction_counts,
    fraction_report,
    hidden_unit_attribution,
    phase0_manifest,
    sample_float_top_down_wrap,
    subpixel_boundary_mask,
)


LATENT_SHA256 = "5a1781afc1a877be452a87a3d958e48cab921b45237faebf2be3668a60ae5fdc"
DECODER_SHA256 = "d676ade8294600eb0064a835eabfe86d4d35e39ee787d512574fbef8d7346baa"


def test_sampler_uses_top_down_half_texel_centers_and_wrap() -> None:
    texture = np.asarray([[[0.0], [1.0]], [[2.0], [3.0]]], dtype=np.float32)
    uv = np.asarray([[0.25, 0.25], [0.75, 0.75], [1.25, 0.25]], dtype=np.float32)
    sampled = sample_float_top_down_wrap(texture, uv)[:, 0]
    assert np.array_equal(sampled, np.asarray([0.0, 3.0, 0.0], dtype=np.float32))
    corners, weights, base_xy = bilinear_footprint_top_down_wrap((2, 2), uv)
    assert corners.shape == (3, 4, 2)
    assert np.array_equal(weights[0], np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32))
    assert base_xy.tolist() == [[0, 0], [1, 1], [0, 0]]


def test_activation_crossing_ignores_zero_weight_corners() -> None:
    latent = np.zeros((2, 2, 4), dtype=np.float32)
    latent[0, 1, 0] = 1.0
    latent[1, 1, 0] = 1.0
    weight = np.zeros((1, 4), dtype=np.float32)
    weight[0, 0] = 1.0
    bias = np.asarray([-0.5], dtype=np.float32)
    center = activation_crossings(latent, np.asarray([[0.25, 0.25]], dtype=np.float32), weight, bias)
    between = activation_crossings(latent, np.asarray([[0.50, 0.25]], dtype=np.float32), weight, bias)
    assert not center[0, 0]
    assert between[0, 0]


def test_subpixel_boundary_requires_contributing_corner_span() -> None:
    metallic = np.asarray([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    uv = np.asarray([[0.25, 0.25], [0.50, 0.25]], dtype=np.float32)
    boundary = subpixel_boundary_mask(metallic, uv, threshold=0.1)
    assert boundary.tolist() == [False, True]


def test_dark_fraction_separates_filter_and_interpolation_novel_samples() -> None:
    truth = np.asarray([0.1, 0.1, 0.01], dtype=np.float32)
    runtime = np.asarray([0.04, 0.04, 0.0], dtype=np.float32)
    filtered = np.asarray([0.08, 0.03, 0.0], dtype=np.float32)
    report = fraction_report(dark_fraction_counts(truth, runtime, filtered, [0.05]))
    assert report["0.05"]["eligible"] == 2
    assert report["0.05"]["runtime_dark_fraction"] == 1.0
    assert report["0.05"]["filter_dark_fraction"] == 0.5
    assert report["0.05"]["novel_dark_fraction"] == 0.5


def test_hidden_attribution_reports_coverage_and_conditional_rate() -> None:
    crossings = np.asarray([[True, False], [True, True], [False, True]], dtype=bool)
    novel = np.asarray([True, True, False], dtype=bool)
    report = hidden_unit_attribution(crossings, novel)
    assert report["novel_dark_with_any_crossing_fraction"] == 1.0
    assert report["units"]["0"]["novel_dark_coverage"] == 1.0
    assert report["units"]["1"]["novel_dark_coverage"] == 0.5
    assert report["units"]["1"]["conditional_novel_dark_rate"] == 0.5


def test_phase0_manifest_requires_two_identical_hashes_and_sampler_pass() -> None:
    digest = "a" * 64
    manifest = phase0_manifest(digest, [digest, digest], {"latent": LATENT_SHA256}, sampler_uv_contract_passed=True)
    assert manifest["determinism"]["identical"] is True
    assert manifest["formal_holdout_accessed"] is False
    with pytest.raises(ValueError, match="two byte-identical"):
        phase0_manifest(digest, [digest, "b" * 64], {}, sampler_uv_contract_passed=True)
    with pytest.raises(ValueError, match="sampler/UV"):
        phase0_manifest(digest, [digest, digest], {}, sampler_uv_contract_passed=False)


def test_phase0_config_freezes_inputs_and_avoids_formal_holdout() -> None:
    path = ROOT / "configs/eval/scifihelmet_interpolation_repair.yaml"
    text = path.read_text(encoding="utf-8")
    config = yaml.safe_load(text)
    assert "formal_holdout" not in text.lower()
    assert config["frozen_sha256"]["latent_hard_png"] == LATENT_SHA256
    assert config["frozen_sha256"]["decoder_npz"] == DECODER_SHA256
    assert config["output_dir"].endswith("interpolation_repair_v1/analysis")
    assert config["analysis"]["random_seed"] == 20260803


def test_s1_batch_is_exact_50_25_25_and_uv_stays_in_selected_texels() -> None:
    scripts = ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from train_scifihelmet_interpolation_repair import _subpixel_batch

    generator = torch.Generator().manual_seed(11)
    ids, uv, slices = _subpixel_batch(
        torch.arange(0, 8),
        torch.arange(8, 12),
        torch.arange(12, 16),
        batch_size=16,
        height=4,
        width=4,
        generator=generator,
    )
    assert ids[slices["uniform"]].numel() == 8
    assert ids[slices["base_hard_bright"]].numel() == 4
    assert ids[slices["metallic_boundary"]].numel() == 4
    recovered_x = torch.floor(uv[:, 0] * 4).to(torch.int64)
    recovered_y = torch.floor(uv[:, 1] * 4).to(torch.int64)
    assert torch.equal(recovered_y * 4 + recovered_x, ids)


def test_s1_config_freezes_schedule_learning_rates_and_loss_weights() -> None:
    path = ROOT / "configs/train/scifihelmet_interpolation_repair.yaml"
    text = path.read_text(encoding="utf-8")
    config = yaml.safe_load(text)
    assert "formal_holdout" not in text.lower()
    assert config["training"]["warmup_steps"] == 1000
    assert config["training"]["max_steps"] == 10000
    assert config["training"]["evaluation_interval"] == 250
    assert config["training"]["latent_learning_rate"] == 2.0e-4
    assert config["training"]["decoder_learning_rate"] == 2.0e-6
    assert config["loss"]["subpixel_material_weight"] == 0.5
    assert config["loss"]["luminance_underprediction_top_tail_weight"] == 0.5
    assert config["loss"]["subpixel_metallic_boundary_l1_weight"] == 0.25


def test_s2_config_inherits_frozen_matrix_with_only_bounded_overrides() -> None:
    scripts = ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from train_scifihelmet_interpolation_repair import _load_config

    config = _load_config(ROOT / "configs/train/scifihelmet_interpolation_repair_s2.yaml")
    assert config["candidate"] == "s2"
    assert config["training"]["warmup_steps"] == 500
    assert config["training"]["max_steps"] == 10000
    assert config["training"]["latent_learning_rate"] == 2.0e-4
    assert config["training"]["decoder_learning_rate"] == 2.0e-6
    assert config["output_dir"].endswith("interpolation_repair_v1/s2")
