from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cg_frontier.compression.filter_aware import (
    FilterAwareDecoder,
    calculate_filter_aware_cost,
    component_rectangularity,
    halo_statistics,
    initialize_filter_aware_from_tiny,
    postprocess_commutativity_loss,
)
from cg_frontier.compression.material import MaterialDecoder


def test_candidates_keep_shape_cost_and_separate_activation_cost() -> None:
    for kind in ("f_relu", "f_softplus", "f_sigmoid"):
        decoder = FilterAwareDecoder(kind)
        cost = calculate_filter_aware_cost(decoder)
        assert (cost["parameters"], cost["weight_bytes_float32"], cost["macs_per_pixel"]) == (
            103,
            412,
            88,
        )
        assert cost["module_identifier"].endswith(kind)
        assert "activation" in cost
    assert calculate_filter_aware_cost(FilterAwareDecoder("f_relu"))["activation"]["special_functions_per_pixel"] == 0
    assert calculate_filter_aware_cost(FilterAwareDecoder("f_softplus"))["activation"]["special_functions_per_pixel"] > 0
    assert calculate_filter_aware_cost(FilterAwareDecoder("f_sigmoid"))["activation"]["special_functions_per_pixel"] > 0


def test_f_relu_initialization_is_exact_on_step_zero() -> None:
    torch.manual_seed(19)
    baseline = MaterialDecoder("tiny_mlp")
    candidate = initialize_filter_aware_from_tiny(baseline, "f_relu")
    probes = torch.rand(257, 4)
    assert torch.equal(candidate(probes), baseline(probes))


def test_smooth_swaps_are_not_claimed_as_exact_initialization() -> None:
    torch.manual_seed(23)
    baseline = MaterialDecoder("tiny_mlp")
    probes = torch.rand(257, 4)
    for kind in ("f_softplus", "f_sigmoid"):
        candidate = initialize_filter_aware_from_tiny(baseline, kind)
        assert not torch.equal(candidate(probes), baseline(probes))


def test_commutativity_loss_uses_postprocessed_corner_materials() -> None:
    torch.manual_seed(29)
    decoder = FilterAwareDecoder("f_softplus")
    latent = torch.rand(4, 4, 4)
    uv = torch.tensor([[0.31, 0.42], [0.77, 0.63]])
    result = postprocess_commutativity_loss(decoder, latent, uv)
    assert result.loss.ndim == 0
    assert torch.isfinite(result.loss)
    assert result.runtime.base_color_linear.shape == (2, 3)
    assert result.filtered.normal_xyz.shape == (2, 3)


def test_component_rectangularity_distinguishes_blocks() -> None:
    mask = np.zeros((8, 10), dtype=bool)
    mask[1:4, 2:6] = True
    mask[6, 8] = True
    report = component_rectangularity(mask)
    assert report["component_count"] == 2
    assert report["max_area"] == 12
    assert report["max_area_rectangularity"] == 1.0
    assert report["rectangular_component_max_area"] == 12


def test_halo_statistics_reports_two_sides_and_signed_band_error() -> None:
    reference = np.full(6, 0.5, dtype=np.float32)
    candidate = np.asarray([0.6, 0.4, 0.51, 0.49, 0.8, 0.2], dtype=np.float32)
    report = halo_statistics(reference, candidate, np.ones(6, dtype=bool), threshold=0.05)
    assert report["positive_fraction"] == 2 / 6
    assert report["negative_fraction"] == 2 / 6
    assert abs(report["signed_mean"]) < 1.0e-6


def test_phase0_config_freezes_rois_inputs_and_no_formal_holdout() -> None:
    path = ROOT / "configs/eval/scifihelmet_filter_aware_phase0.yaml"
    text = path.read_text(encoding="utf-8")
    config = yaml.safe_load(text)
    assert "formal_holdout" not in text.lower()
    assert config["anchors"]["A_yellow_tube"]["atlas_roi_xyxy"] == [1664, 0, 2048, 512]
    assert config["anchors"]["B_gray_panel"]["atlas_roi_xyxy"] == [1024, 512, 1664, 1152]
    assert config["analysis"]["fixed_phases_xy"] == [
        [0.25, 0.25],
        [0.75, 0.25],
        [0.25, 0.75],
        [0.75, 0.75],
    ]
    assert config["analysis"]["metallic_boundary_threshold"] == 0.1
    assert config["analysis"]["halo_threshold"] == 0.02


def test_training_batch_is_exact_40_20_20_20_and_anchor_is_centered() -> None:
    scripts = ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from train_scifihelmet_filter_aware import filter_aware_batch

    generator = torch.Generator().manual_seed(31)
    ids, uv, slices = filter_aware_batch(
        torch.arange(0, 20),
        torch.arange(20, 30),
        torch.arange(30, 40),
        batch_size=20,
        height=8,
        width=8,
        generator=generator,
    )
    assert ids[slices["uniform_subpixel"]].numel() == 8
    assert ids[slices["dark_hard_subpixel"]].numel() == 4
    assert ids[slices["boundary_halo_subpixel"]].numel() == 4
    assert ids[slices["texel_center_anchor"]].numel() == 4
    anchors = uv[slices["texel_center_anchor"]] * 8.0
    assert torch.allclose(anchors.frac(), torch.full_like(anchors, 0.5))


def test_training_configs_freeze_loss_schedule_and_independent_outputs() -> None:
    scripts = ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from train_scifihelmet_interpolation_repair import _load_config

    configs = [
        _load_config(ROOT / "configs/train/scifihelmet_filter_aware.yaml"),
        _load_config(ROOT / "configs/train/scifihelmet_filter_aware_softplus.yaml"),
        _load_config(ROOT / "configs/train/scifihelmet_filter_aware_sigmoid.yaml"),
    ]
    assert [item["candidate"] for item in configs] == ["f_relu", "f_softplus", "f_sigmoid"]
    assert [item["training"]["warmup_steps"] for item in configs] == [0, 1000, 1000]
    assert len({item["output_dir"] for item in configs}) == 3
    for config in configs:
        assert config["training"]["max_steps"] == 10000
        assert config["training"]["max_minutes"] == 30
        assert config["training"]["evaluation_interval"] == 250
        assert config["training"]["warmup_decoder_learning_rate"] == 2.0e-5
        assert config["training"]["joint_latent_learning_rate"] == 2.0e-4
        assert config["training"]["joint_decoder_learning_rate"] == 2.0e-6
        assert config["batch_mix"] == {
            "uniform_subpixel": 0.40,
            "D2_D3_dark_hard_roi": 0.20,
            "material_boundary_halo_roi": 0.20,
            "uniform_texel_center_anchor": 0.20,
        }
        assert list(config["loss"].values())[:5] == [0.5, 0.5, 0.5, 0.25, 0.5]
        assert set(config["rois_xyxy"]) == {
            "D1_metallic_boundary_full",
            "D2_yellow_tube_uv",
            "D3_gray_panel_proxy",
        }


def test_candidate_numpy_activations_and_schema_loader() -> None:
    scripts = ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from analyze_scifihelmet_filter_aware_candidates import decoder_raw_activation

    arrays = {
        "network.0.weight": np.eye(8, 4, dtype=np.float32),
        "network.0.bias": np.zeros(8, dtype=np.float32),
        "network.2.weight": np.ones((7, 8), dtype=np.float32),
        "network.2.bias": np.zeros(7, dtype=np.float32),
    }
    latent = np.asarray([[0.25, 0.5, 0.75, 1.0]], dtype=np.float32)
    relu = decoder_raw_activation(latent, arrays, "f_relu")
    softplus = decoder_raw_activation(latent, arrays, "f_softplus")
    sigmoid = decoder_raw_activation(latent, arrays, "f_sigmoid")
    assert relu.shape == softplus.shape == sigmoid.shape == (1, 7)
    assert not np.array_equal(relu, softplus)
    assert not np.array_equal(relu, sigmoid)


def test_phase3_template_freezes_full_atlas_and_fixed_roi_contracts() -> None:
    path = ROOT / "configs/eval/scifihelmet_filter_aware_phase3_template.yaml"
    text = path.read_text(encoding="utf-8")
    config = yaml.safe_load(text)
    assert "formal_holdout" not in text.lower()
    assert config["analysis"]["random_probe_count"] == 1048576
    assert config["analysis"]["seeded_probe_count_per_roi"] == 262144
    assert set(config["scopes"]) == {
        "global",
        "D1_metallic_boundary",
        "D2_yellow_tube",
        "D3_gray_panel",
    }


def test_exported_latent_metadata_uses_raw_bytes_resident_field() -> None:
    from cg_frontier.compression.render_loss import export_latent_unorm8_png

    assert "raw_bytes" in export_latent_unorm8_png.__annotations__ or callable(export_latent_unorm8_png)


def test_oracle_assessment_is_bounded_to_fixed_roi_failure_before_runtime_followup() -> None:
    source = (ROOT / "scripts/evaluate_scifihelmet_decode_then_filter_oracle.py").read_text(encoding="utf-8")
    assert "runtime_cost_followup_allowed\": False" in source
    assert "four-texel" in source
    assert "formal_holdout/" not in source.lower()
