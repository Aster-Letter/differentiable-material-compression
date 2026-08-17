from __future__ import annotations

import sys
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cg_frontier.compression.deployment_parity import (  # noqa: E402
    DeploymentParityDecoder,
    activation_region_coherence,
    calculate_deployment_parity_cost,
    instantiate_fresh_candidate,
    dark_envelope_loss,
    deployment_parity_sample,
    make_fresh_initialization,
)
from cg_frontier.compression.deployment_parity_training import (  # noqa: E402
    deployment_parity_batch,
    learning_rates_at_step,
    validate_fixed_protocol,
)


def _checkerboard_black_hole() -> tuple[torch.Tensor, DeploymentParityDecoder]:
    latent = torch.zeros((2, 2, 4), dtype=torch.float32)
    latent[..., 0] = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    decoder = DeploymentParityDecoder(width=8)
    with torch.no_grad():
        decoder.hidden.weight.zero_()
        decoder.hidden.bias.fill_(-1.0)
        decoder.output.weight.zero_()
        decoder.output.bias.zero_()
        decoder.hidden.weight[0, 0] = 1.0
        decoder.hidden.bias[0] = -0.75
        decoder.hidden.weight[1, 0] = -1.0
        decoder.hidden.bias[1] = 0.25
        decoder.output.bias[0:3] = -6.0
        decoder.output.weight[0:3, 0] = 48.0
        decoder.output.weight[0:3, 1] = 48.0
    return latent, decoder


def test_deployment_parity_sample_reproduces_a_subpixel_relu_black_hole() -> None:
    latent, decoder = _checkerboard_black_hole()
    sample = deployment_parity_sample(
        latent,
        torch.tensor([[0.5, 0.5]], dtype=torch.float32),
        decoder,
        quantization="hard",
    )

    assert torch.max(sample.runtime.base_color_linear) < 0.01
    assert torch.min(sample.decode_then_filter.base_color_linear) > 0.99
    assert sample.postprocess_commutativity_l1 > 0.9


def test_activation_region_coherence_distinguishes_crossing_and_safe_cells() -> None:
    latent, decoder = _checkerboard_black_hole()
    unsafe = deployment_parity_sample(
        latent,
        torch.tensor([[0.5, 0.5]], dtype=torch.float32),
        decoder,
        quantization="hard",
    )
    unsafe_region = activation_region_coherence(decoder, unsafe.corners)

    safe_latent = latent.clone()
    safe_latent[..., 0] = 0.0
    safe = deployment_parity_sample(
        safe_latent,
        torch.tensor([[0.5, 0.5]], dtype=torch.float32),
        decoder,
        quantization="hard",
    )
    safe_region = activation_region_coherence(decoder, safe.corners)

    assert unsafe_region.crossing_fraction == 0.25
    assert unsafe_region.loss > 0.0
    assert safe_region.crossing_fraction == 0.0
    assert safe_region.loss == 0.0


def test_dark_envelope_penalizes_only_runtime_undercut() -> None:
    latent, decoder = _checkerboard_black_hole()
    unsafe = deployment_parity_sample(
        latent,
        torch.tensor([[0.5, 0.5]], dtype=torch.float32),
        decoder,
        quantization="hard",
    )
    unsafe_loss = dark_envelope_loss(unsafe.runtime, unsafe.decode_then_filter)

    safe_latent = latent.clone()
    safe_latent[..., 0] = 0.0
    safe = deployment_parity_sample(
        safe_latent,
        torch.tensor([[0.5, 0.5]], dtype=torch.float32),
        decoder,
        quantization="hard",
    )
    safe_loss = dark_envelope_loss(safe.runtime, safe.decode_then_filter)

    assert unsafe_loss > 0.9
    assert safe_loss == 0.0


def test_fresh_candidates_share_step_zero_but_not_mutable_storage() -> None:
    initialization = make_fresh_initialization(height=2, width=3, decoder_width=8, seed=20260804)
    dp = instantiate_fresh_candidate(initialization, candidate="dp_relu_fresh")
    arc = instantiate_fresh_candidate(initialization, candidate="arc_relu_fresh")
    probes = torch.tensor([[0.0, 0.25, 0.5, 1.0], [1.0, 0.5, 0.25, 0.0]])

    assert initialization.sha256 == make_fresh_initialization(
        height=2, width=3, decoder_width=8, seed=20260804
    ).sha256
    assert torch.equal(dp.decoder(probes), arc.decoder(probes))
    assert torch.equal(dp.latent, arc.latent)
    assert dp.latent.data_ptr() != arc.latent.data_ptr()

    with torch.no_grad():
        dp.latent.add_(1.0)
        next(dp.decoder.parameters()).add_(1.0)

    assert not torch.equal(dp.latent, arc.latent)
    assert not torch.equal(dp.decoder(probes), arc.decoder(probes))


def test_deployment_parity_cost_keeps_width12_diagnostic_out_of_exact_budget() -> None:
    exact = calculate_deployment_parity_cost(DeploymentParityDecoder(width=8))
    diagnostic = calculate_deployment_parity_cost(DeploymentParityDecoder(width=12))

    assert exact == {
        "shape": "4->8->7",
        "parameters": 103,
        "weight_bytes_float32": 412,
        "macs_per_pixel": 88,
        "texture_samples": 1,
        "actual_resident_bytes": 16 * 1024 * 1024,
        "activation": {"kind": "relu", "hidden_units": 8, "special_functions_per_pixel": 0},
        "deployable_exact_budget": True,
    }
    assert diagnostic["shape"] == "4->12->7"
    assert diagnostic["parameters"] == 151
    assert diagnostic["weight_bytes_float32"] == 604
    assert diagnostic["macs_per_pixel"] == 132
    assert diagnostic["deployable_exact_budget"] is False


def test_deployment_batch_is_exact_40_20_15_15_10() -> None:
    generator = torch.Generator().manual_seed(101)
    batch = deployment_parity_batch(
        screen_uv=torch.tensor([[0.125, 0.25], [0.75, 0.875]]),
        uniform_pool=torch.arange(0, 16),
        dark_pool=torch.arange(16, 24),
        boundary_pool=torch.arange(24, 32),
        batch_size=20,
        height=8,
        width=8,
        generator=generator,
    )

    assert {name: value.stop - value.start for name, value in batch.slices.items()} == {
        "visible_screen_subpixel": 8,
        "uniform_atlas_subpixel": 4,
        "dark_hard_cell": 3,
        "material_boundary_halo_cell": 3,
        "texel_center_reference_anchor": 2,
    }
    anchors = batch.uv[batch.slices["texel_center_reference_anchor"]] * 8.0
    assert torch.allclose(anchors.frac(), torch.full_like(anchors, 0.5))


def test_training_protocol_weights_and_lr_schedule_are_frozen() -> None:
    protocol = validate_fixed_protocol(
        {
            "batch_mix": {
                "visible_screen_subpixel": 0.40,
                "uniform_atlas_subpixel": 0.20,
                "D2_D3_dark_hard_cell": 0.15,
                "material_boundary_halo_cell": 0.15,
                "texel_center_reference_anchor": 0.10,
            },
            "loss": {
                "subpixel_reference_material": 0.5,
                "postprocess_filter_commutativity": 0.5,
                "luminance_underprediction_top_tail": 0.5,
                "metallic_boundary_halo": 0.25,
                "texel_center_reference_anchor": 0.5,
                "dark_envelope": 0.5,
                "activation_region": 0.5,
                "activation_region_margin": 2.0 / 255.0,
            },
        }
    )

    assert protocol["max_steps"] == 120_000
    assert learning_rates_at_step(0) == (0.0, 0.0)
    assert learning_rates_at_step(5_000) == (2.0e-2, 1.0e-3)
    assert learning_rates_at_step(120_000) == (5.0e-4, 2.0e-5)


def test_public_training_config_freezes_candidate_matrix_and_excludes_holdout() -> None:
    path = ROOT / "configs/train/scifihelmet_deployment_parity_arc_v1.yaml"
    text = path.read_text(encoding="utf-8")
    config = yaml.safe_load(text)

    validate_fixed_protocol(config)
    assert "formal_holdout" not in text.lower()
    assert list(config["candidates"]) == [
        "dp_relu_fresh",
        "arc_relu_fresh",
        "arc12_diagnostic",
    ]
    assert config["candidates"]["dp_relu_fresh"]["decoder_width"] == 8
    assert config["candidates"]["arc_relu_fresh"]["decoder_width"] == 8
    assert config["candidates"]["arc12_diagnostic"]["decoder_width"] == 12
    assert config["training"]["max_steps"] == 120_000
    assert config["training"]["max_minutes"] == 120


def test_phase3_config_and_analyzer_keep_fixed_rois_and_two_runs() -> None:
    config_path = ROOT / "configs/eval/scifihelmet_deployment_parity_phase3.yaml"
    script_path = ROOT / "scripts/analyze_scifihelmet_deployment_parity.py"
    config_text = config_path.read_text(encoding="utf-8")
    script_text = script_path.read_text(encoding="utf-8")
    config = yaml.safe_load(config_text)

    assert "formal_holdout" not in config_text.lower()
    assert config["analysis"]["fixed_phases_xy"] == [
        [0.25, 0.25],
        [0.75, 0.25],
        [0.25, 0.75],
        [0.75, 0.75],
    ]
    assert config["anchors"]["A_yellow_tube"]["atlas_roi_xyxy"] == [1664, 0, 2048, 512]
    assert config["anchors"]["B_gray_panel"]["atlas_roi_xyxy"] == [1024, 512, 1664, 1152]
    assert 'atlas_a != atlas_b or roi_a != roi_b' in script_text


def test_gate_evaluator_prioritizes_zero_tolerance_blocks_before_quality() -> None:
    path = ROOT / "scripts/evaluate_scifihelmet_deployment_parity_gates.py"
    source = path.read_text(encoding="utf-8")

    assert '"fixed_roi_A_rectangular_dark_component_max_area"' in source
    assert '"fixed_roi_B_rectangular_dark_component_max_area"' in source
    assert '"D2_novel_dark_absolute_fraction"' in source
    assert '0.005, "<="' in source
    assert '"black_block_hard_pass"' in source
    assert 'black_pass and quality_pass and legacy["offline_pass"]' in source
