from __future__ import annotations

import sys
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from cg_frontier.compression.decode_then_filter import (  # noqa: E402
    DecodeThenFilterDecoder,
    calculate_decode_then_filter_cost,
    decode_then_filter_sample,
    export_decode_then_filter_latent_unorm8,
    pack_decode_then_filter_latent_unorm8,
    instantiate_paired_precheck_candidate,
    make_paired_precheck_initialization,
)
from cg_frontier.compression.filter_aware import postprocess_raw_torch  # noqa: E402
from cg_frontier.compression.decode_then_filter_training import (  # noqa: E402
    build_activation_selection_record,
    build_capacity_diagnostic_eligibility,
    build_capacity_route_decision,
    build_decode_then_filter_continuation_manifest,
    build_decode_then_filter_camera_finetune_manifest,
    build_decode_then_filter_camera31_fresh_manifest,
    build_decode_then_filter_manifest,
    camera_finetune_learning_rates_at_step,
    continuation_learning_rates_at_step,
    decode_then_filter_batch,
    full_training_objective,
    full_training_learning_rates_at_step,
    full_training_phase_at_step,
    material_subset_metrics,
    paired_precheck_objective,
    precheck_learning_rates_at_step,
    resolve_dtf_camera_spec,
    select_explicit_dtf_camera_pool,
    select_camera_triangle_coverage,
    restore_decode_then_filter_resume_state,
    validate_decode_then_filter_resume_payload,
    validate_decode_then_filter_protocol,
)
from cg_frontier.compression.material import Core4Targets, DecodedMaterial  # noqa: E402
from cg_frontier.render.decode_then_filter import (  # noqa: E402
    DTFLatentMaterialSource,
    DTFReferenceMaterialSource,
    DecodeThenFilterRenderer,
)
from cg_frontier.render.gbuffer import Camera, GBufferResult, MaterialBuffers  # noqa: E402
from cg_frontier.render.pbr import PointLight  # noqa: E402
from train_scifihelmet_c4_dtf_full import (  # noqa: E402
    recover_logged_checkpoint_selection,
    summarize_resume_training_result,
)


def _checkerboard_black_hole() -> tuple[torch.Tensor, DecodeThenFilterDecoder]:
    latent = torch.zeros((2, 2, 4), dtype=torch.float32)
    latent[..., 0] = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    decoder = DecodeThenFilterDecoder(latent_channels=4, width=16, activation="relu")
    with torch.no_grad():
        for parameter in decoder.parameters():
            parameter.zero_()
        decoder.hidden_in.weight[0, 0] = 1.0
        decoder.hidden_in.bias[0] = -0.75
        decoder.hidden_in.weight[1, 0] = -1.0
        decoder.hidden_in.bias[1] = 0.25
        decoder.hidden_mid.weight[0, 0] = 1.0
        decoder.hidden_mid.weight[1, 1] = 1.0
        decoder.output.bias[0:3] = -6.0
        decoder.output.weight[0:3, 0:2] = 48.0
    return latent, decoder


def test_decode_then_filter_eliminates_nonlinear_center_black_hole() -> None:
    latent, decoder = _checkerboard_black_hole()
    uv = torch.tensor([[0.5, 0.5]], dtype=torch.float32)

    result = decode_then_filter_sample(latent, uv, decoder, quantization="hard")
    old_center = postprocess_raw_torch(decoder(result.filtered_latent))

    assert torch.max(old_center.base_color_linear) < 0.01
    assert torch.min(result.material.base_color_linear) > 0.99


def _geometry() -> GBufferResult:
    return GBufferResult(
        buffers={},
        torch_buffers={
            "mask": torch.tensor([[True]]),
            "uv": torch.tensor([[[0.5, 0.5]]], dtype=torch.float32),
            "position_world": torch.zeros((1, 1, 3), dtype=torch.float32),
            "vertex_normal_world": torch.tensor([[[0.0, 0.0, 1.0]]]),
            "tangent_world": torch.tensor([[[1.0, 0.0, 0.0]]]),
            "bitangent_world": torch.tensor([[[0.0, 1.0, 0.0]]]),
        },
        metadata={"fixture": "one_pixel"},
    )


def test_dtf_renderer_exposes_the_four_corner_material_filter_contract() -> None:
    latent, decoder = _checkerboard_black_hole()
    renderer = DecodeThenFilterRenderer(display_exposure=1.0, minimum_roughness=0.045)
    camera = Camera(
        eye=(0.0, 0.0, 4.5),
        target=(0.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
        vertical_fov_degrees=45.0,
        near=0.1,
        far=10.0,
    )
    light = PointLight(
        position=(2.5, 3.0, 4.0),
        color=(1.0, 0.98, 0.95),
        radiant_intensity=90.0,
        ambient_intensity=0.04,
    )

    bundle = renderer.render(
        _geometry(),
        camera,
        light,
        DTFLatentMaterialSource(latent, decoder, quantization="hard"),
        input_hashes={"fixture": "sha256:test"},
    )

    assert bundle.renderer_identifier == "decode_then_filter_renderer_v1"
    assert torch.min(bundle.material.base_color_linear) > 0.99
    assert torch.isfinite(bundle.linear_hdr).all()
    assert torch.isfinite(bundle.display_rgb).all()
    assert bundle.metadata["deployment_order"] == [
        "unorm8_quantize_corner_texels",
        "four_point_corner_fetches_lod0_wrap",
        "shared_decoder_per_corner",
        "core4_postprocess_per_corner",
        "bilinear_filter_material_semantics",
        "normalize_tangent_normal_once",
        "shared_tbn_ggx_pbr",
        "display_transform",
    ]


def test_c4_dtf16_cost_separates_four_corner_work_from_actual_residency() -> None:
    decoder = DecodeThenFilterDecoder(latent_channels=4, width=16, activation="relu")

    cost = calculate_decode_then_filter_cost(decoder, height=2048, width=2048)

    assert cost == {
        "module_identifier": "decode_then_filter_decoder_v1.c4_w16_relu",
        "shape": "4->16->16->7",
        "parameters": 471,
        "weight_bytes_float32": 1884,
        "decoder_macs_per_corner": 432,
        "decoder_macs_per_pixel": 1728,
        "texture_resources": 1,
        "point_texel_loads_per_pixel": 4,
        "material_filter_scalar_channels": 8,
        "normal_normalizations_per_pixel": 1,
        "theoretical_raw_bytes_unorm8": 16 * 1024 * 1024,
        "actual_resident_bytes": None,
        "actual_resident_measurement": "required_in_target_runtime",
        "activation": {
            "kind": "relu",
            "hidden_units_per_layer": 16,
            "hidden_layers": 2,
            "activation_evaluations_per_pixel": 128,
            "special_functions_per_pixel": 0,
        },
    }


def test_public_dtf_config_freezes_quality_first_protocol_without_arc_losses() -> None:
    path = ROOT / "configs/train/scifihelmet_c4_dtf_v1.yaml"
    text = path.read_text(encoding="utf-8")
    config = yaml.safe_load(text)

    protocol = validate_decode_then_filter_protocol(config)

    assert list(config["candidates"]) == [
        "c4_dtf_16_relu_precheck",
        "c4_dtf_16_silu_precheck",
        "c4_dtf_16_selected",
        "c4_dtf_32_diagnostic",
        "c4_dtf_32_selected",
        "c5_dtf_16",
    ]
    assert protocol["full_training_steps"] == 80_000
    assert protocol["phases"] == [
        {"name": "material_continuous_pretrain", "start": 0, "stop": 15_000},
        {"name": "render_first_joint", "start": 15_000, "stop": 65_000},
        {"name": "low_lr_polish", "start": 65_000, "stop": 80_000},
    ]
    assert protocol["batch_mix"] == {
        "screen_space_render": 0.45,
        "uniform_uv_chart_subpixel": 0.35,
        "generic_high_gradient_boundary": 0.10,
        "texel_center_quantization_anchor": 0.10,
    }
    assert protocol["camera_pool_limit"] == 48
    assert protocol["light_pool_limit"] == 6
    assert config["training"]["batch_size"] > 0
    assert config["training"]["batch_size"] % 20 == 0
    assert protocol["checkpoint_tracks"] == ["best-render", "best-artifact-safe"]
    lowered = text.lower()
    for forbidden in (
        "formal_holdout",
        "commutativity",
        "activation_region",
        "dark_tail",
        "d2_",
        "d3_",
    ):
        assert forbidden not in lowered


def test_camera31_finetune_config_is_bounded_and_preserves_the_frozen_loss() -> None:
    base = yaml.safe_load(
        (ROOT / "configs/train/scifihelmet_c4_dtf_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    config = yaml.safe_load(
        (ROOT / "configs/train/scifihelmet_c4_dtf_camera31_v1.yaml").read_text(
            encoding="utf-8"
        )
    )

    assert config["experiment"] == "scifihelmet_c4_dtf_camera31_v1"
    assert config["training_pool"]["camera_selection_strategy"] == (
        "explicit_audited_pool_v1"
    )
    assert len(config["train_cameras"]) == 31
    assert len({camera["name"] for camera in config["train_cameras"]}) == 31
    assert config["batch_mix"] == base["batch_mix"]
    assert config["loss"] == base["loss"]
    assert config["camera_finetune"]["exact_continuation"] is False
    assert config["camera_finetune"]["source_step"] == 160_000
    assert config["camera_finetune"]["stop_step"] == 180_000
    focus = next(
        camera
        for camera in config["train_cameras"]
        if camera["name"] == "focus_ep20_y150"
    )
    resolved = resolve_dtf_camera_spec(focus, config["render"])
    assert resolved["radius"] == 2.8
    assert resolved["target"] == (0.0, -0.67, -0.72)


def test_dtf_manifest_binds_candidate_inputs_cost_and_dual_track_retention() -> None:
    path = ROOT / "configs/train/scifihelmet_c4_dtf_v1.yaml"
    text = path.read_text(encoding="utf-8")
    config = yaml.safe_load(text)

    manifest = build_decode_then_filter_manifest(
        config,
        candidate="c4_dtf_16_relu_precheck",
        config_sha256="a" * 64,
        input_hashes={"core4_manifest": "b" * 64, "gltf": "c" * 64},
        git_commit="08ebac3",
    )

    assert manifest["candidate"] == {
        "name": "c4_dtf_16_relu_precheck",
        "latent_channels": 4,
        "decoder_width": 16,
        "activation": "relu",
        "role": "paired_precheck",
        "max_steps": 10_000,
    }
    assert manifest["renderer"]["identifier"] == "decode_then_filter_renderer_v1"
    assert manifest["renderer"]["filter_order"] == "decode_four_corners_then_material_filter"
    assert manifest["cost"]["decoder_macs_per_pixel"] == 1728
    assert manifest["cost"]["actual_resident_bytes"] is None
    assert manifest["selection"]["checkpoint_tracks"] == [
        "best-render",
        "best-artifact-safe",
    ]
    assert manifest["retention"]["keep"] == [
        "rolling-resume",
        "best-render",
        "best-artifact-safe",
        "final",
    ]
    assert manifest["output_dir"].endswith("/c4_dtf_16_relu_precheck")
    assert manifest["inputs"]["config_sha256"] == "a" * 64
    serialized = json.dumps(manifest, sort_keys=True).lower()
    for forbidden in ("commutativity", "activation_region", "dark_tail", "d2_", "d3_"):
        assert forbidden not in serialized


def test_full_candidate_manifest_requires_and_binds_precheck_activation_selection() -> None:
    path = ROOT / "configs/train/scifihelmet_c4_dtf_v1.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))

    manifest = build_decode_then_filter_manifest(
        config,
        candidate="c4_dtf_16_selected",
        config_sha256="a" * 64,
        input_hashes={"core4_manifest": "b" * 64},
        git_commit="08ebac3",
        selected_activation="silu",
        activation_selection_sha256="d" * 64,
    )

    assert manifest["candidate"]["activation"] == "silu"
    assert manifest["candidate"]["activation_source"] == "paired_10k_precheck"
    assert manifest["inputs"]["activation_selection_sha256"] == "d" * 64


def test_constant_corner_dtf_matches_the_direct_reference_renderer() -> None:
    latent = torch.rand((2, 2, 4), generator=torch.Generator().manual_seed(17))
    decoder = DecodeThenFilterDecoder(latent_channels=4, width=16, activation="relu")
    expected_base = torch.tensor([0.2, 0.4, 0.6])
    with torch.no_grad():
        for parameter in decoder.parameters():
            parameter.zero_()
        decoder.output.bias[0:3] = torch.logit(expected_base)
        decoder.output.bias[5] = torch.logit(torch.tensor(0.4))
        decoder.output.bias[6] = torch.logit(torch.tensor(0.7))
    reference_material = MaterialBuffers(
        base_color_linear=expected_base.reshape(1, 1, 3),
        normal_world=torch.tensor([[[0.0, 0.0, 1.0]]]),
        roughness=torch.tensor([[0.4]]),
        metallic=torch.tensor([[0.7]]),
        normal_ts_raw=torch.tensor([[[0.0, 0.0, 1.0]]]),
        normal_ts_unit=torch.tensor([[[0.0, 0.0, 1.0]]]),
    )
    renderer = DecodeThenFilterRenderer(display_exposure=1.0, minimum_roughness=0.045)
    camera = Camera(
        eye=(0.0, 0.0, 4.5),
        target=(0.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
        vertical_fov_degrees=45.0,
        near=0.1,
        far=10.0,
    )
    light = PointLight(
        position=(2.5, 3.0, 4.0),
        color=(1.0, 0.98, 0.95),
        radiant_intensity=90.0,
        ambient_intensity=0.04,
    )

    candidate = renderer.render(
        _geometry(),
        camera,
        light,
        DTFLatentMaterialSource(latent, decoder, quantization="hard"),
        input_hashes={"fixture": "sha256:candidate"},
    )
    reference = renderer.render(
        _geometry(),
        camera,
        light,
        DTFReferenceMaterialSource(reference_material),
        input_hashes={"fixture": "sha256:reference"},
    )

    assert torch.allclose(candidate.material.base_color_linear, reference.material.base_color_linear)
    assert torch.allclose(candidate.material.normal_world, reference.material.normal_world)
    assert torch.allclose(candidate.material.roughness, reference.material.roughness)
    assert torch.allclose(candidate.material.metallic, reference.material.metallic)
    assert torch.allclose(candidate.linear_hdr, reference.linear_hdr)
    assert torch.allclose(candidate.display_rgb, reference.display_rgb)


def test_dtf_sampler_uses_half_texel_centers_and_wrap_addressing() -> None:
    latent = torch.zeros((2, 2, 4), dtype=torch.float32)
    latent[..., 0] = torch.tensor([[0.0, 1.0 / 3.0], [2.0 / 3.0, 1.0]])
    decoder = DecodeThenFilterDecoder(latent_channels=4, width=16, activation="relu")

    center = decode_then_filter_sample(
        latent,
        torch.tensor([[0.25, 0.25]]),
        decoder,
        quantization="hard",
    )
    wrapped = decode_then_filter_sample(
        latent,
        torch.tensor([[1.25, 0.25]]),
        decoder,
        quantization="hard",
    )

    assert torch.equal(center.weights, torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    assert torch.equal(center.corners, wrapped.corners)
    assert torch.equal(center.weights, wrapped.weights)


def test_dtf_filters_corner_normals_then_normalizes_exactly_once() -> None:
    latent = torch.zeros((2, 2, 4), dtype=torch.float32)
    latent[..., 0] = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    decoder = DecodeThenFilterDecoder(latent_channels=4, width=16, activation="relu")
    with torch.no_grad():
        for parameter in decoder.parameters():
            parameter.zero_()
        decoder.hidden_in.weight[0, 0] = 1.0
        decoder.hidden_mid.weight[0, 0] = 1.0
        decoder.output.weight[3, 0] = 2.0
        decoder.output.bias[3] = -1.0

    result = decode_then_filter_sample(
        latent,
        torch.tensor([[0.5, 0.5]]),
        decoder,
        quantization="hard",
    )
    unnormalized = torch.sum(
        result.corner_material.normal_xyz * result.weights[..., None], dim=1
    )

    assert torch.linalg.vector_norm(unnormalized, dim=-1).item() < 0.9
    assert torch.allclose(
        torch.linalg.vector_norm(result.material.normal_xyz, dim=-1),
        torch.ones(1),
    )
    assert torch.allclose(result.material.normal_xyz, torch.tensor([[0.0, 0.0, 1.0]]))


def test_dtf_fake_quantization_matches_hard_forward_and_keeps_gradients() -> None:
    generator = torch.Generator().manual_seed(20260804)
    latent = torch.rand((2, 2, 4), generator=generator, requires_grad=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260805)
        decoder = DecodeThenFilterDecoder(
            latent_channels=4, width=16, activation="silu"
        )
    uv = torch.tensor([[0.11, 0.73], [0.88, 0.26]], dtype=torch.float32)

    hard = decode_then_filter_sample(latent, uv, decoder, quantization="hard")
    fake = decode_then_filter_sample(latent, uv, decoder, quantization="fake")
    fake.material.base_color_linear.sum().backward()

    assert torch.equal(fake.corners, hard.corners)
    assert torch.equal(fake.material.base_color_linear, hard.material.base_color_linear)
    assert latent.grad is not None
    assert torch.isfinite(latent.grad).all()
    assert torch.count_nonzero(latent.grad) > 0


def test_dtf_batch_is_exact_generic_45_35_10_10() -> None:
    batch = decode_then_filter_batch(
        screen_uv=torch.tensor([[0.125, 0.25], [0.75, 0.875]]),
        uniform_pool=torch.arange(0, 32),
        high_gradient_pool=torch.arange(32, 48),
        batch_size=20,
        height=8,
        width=8,
        generator=torch.Generator().manual_seed(101),
    )

    assert {name: value.stop - value.start for name, value in batch.slices.items()} == {
        "screen_space_render": 9,
        "uniform_uv_chart_subpixel": 7,
        "generic_high_gradient_boundary": 2,
        "texel_center_quantization_anchor": 2,
    }
    anchors = batch.uv[batch.slices["texel_center_quantization_anchor"]] * 8.0
    assert torch.allclose(anchors.frac(), torch.full_like(anchors, 0.5))
    assert set(batch.slices) == {
        "screen_space_render",
        "uniform_uv_chart_subpixel",
        "generic_high_gradient_boundary",
        "texel_center_quantization_anchor",
    }


def test_relu_and_silu_prechecks_clone_one_initial_state_without_shared_storage() -> None:
    initialization = make_paired_precheck_initialization(
        height=2,
        width=3,
        latent_channels=4,
        decoder_width=16,
        seed=20260804,
    )
    relu = instantiate_paired_precheck_candidate(initialization, activation="relu")
    silu = instantiate_paired_precheck_candidate(initialization, activation="silu")

    repeated = make_paired_precheck_initialization(
        height=2,
        width=3,
        latent_channels=4,
        decoder_width=16,
        seed=20260804,
    )
    assert initialization.sha256 == repeated.sha256
    assert torch.equal(relu.latent, silu.latent)
    assert relu.latent.data_ptr() != silu.latent.data_ptr()
    assert all(
        torch.equal(left, right)
        for left, right in zip(relu.decoder.parameters(), silu.decoder.parameters(), strict=True)
    )
    assert all(
        left.data_ptr() != right.data_ptr()
        for left, right in zip(relu.decoder.parameters(), silu.decoder.parameters(), strict=True)
    )

    with torch.no_grad():
        relu.latent.add_(1.0)
        next(relu.decoder.parameters()).add_(1.0)

    assert not torch.equal(relu.latent, silu.latent)
    assert not torch.equal(next(relu.decoder.parameters()), next(silu.decoder.parameters()))


def test_paired_precheck_lr_schedule_warms_up_then_cosine_decays() -> None:
    assert precheck_learning_rates_at_step(0) == (0.0, 0.0)
    assert precheck_learning_rates_at_step(500) == (2.0e-2, 1.0e-3)
    assert precheck_learning_rates_at_step(10_000) == (2.0e-3, 1.0e-4)


def test_c5_continuation_schedule_is_continuous_and_decays_through_120k() -> None:
    assert continuation_learning_rates_at_step(80_000) == (1.0e-3, 5.0e-5)
    latent_mid, decoder_mid = continuation_learning_rates_at_step(100_000)
    assert 2.5e-4 < latent_mid < 1.0e-3
    assert 1.25e-5 < decoder_mid < 5.0e-5
    assert continuation_learning_rates_at_step(120_000) == (2.5e-4, 1.25e-5)
    with pytest.raises(ValueError, match=r"\[80000, 120000\]"):
        continuation_learning_rates_at_step(120_001)


def test_c4_continuation_schedule_is_continuous_and_decays_through_160k() -> None:
    assert continuation_learning_rates_at_step(80_000, stop_step=160_000) == (
        1.0e-3,
        5.0e-5,
    )
    latent_mid, decoder_mid = continuation_learning_rates_at_step(
        120_000, stop_step=160_000
    )
    assert 2.5e-4 < latent_mid < 1.0e-3
    assert 1.25e-5 < decoder_mid < 5.0e-5
    assert continuation_learning_rates_at_step(160_000, stop_step=160_000) == (
        2.5e-4,
        1.25e-5,
    )
    with pytest.raises(ValueError, match=r"\[80000, 160000\]"):
        continuation_learning_rates_at_step(160_001, stop_step=160_000)


def test_camera_distribution_finetune_schedule_is_continuous_from_c4_160k() -> None:
    assert camera_finetune_learning_rates_at_step(160_000) == (
        2.5e-4,
        1.25e-5,
    )
    latent_mid, decoder_mid = camera_finetune_learning_rates_at_step(170_000)
    assert 1.0e-4 < latent_mid < 2.5e-4
    assert 5.0e-6 < decoder_mid < 1.25e-5
    assert camera_finetune_learning_rates_at_step(180_000) == (1.0e-4, 5.0e-6)
    with pytest.raises(ValueError, match=r"\[160000, 180000\]"):
        camera_finetune_learning_rates_at_step(180_001)


def test_resume_payload_binds_the_complete_c5_training_state() -> None:
    decoder = DecodeThenFilterDecoder(latent_channels=5, width=16, activation="relu")
    latent = torch.nn.Parameter(torch.zeros((2, 3, 5)))
    optimizer = torch.optim.Adam(
        [
            {"params": [latent]},
            {"params": list(decoder.parameters())},
        ]
    )
    (latent.sum() + sum(parameter.sum() for parameter in decoder.parameters())).backward()
    optimizer.step()
    payload = {
        "schema_version": 1,
        "candidate": "c5_dtf_16",
        "step": 80_000,
        "latent": torch.zeros((2, 3, 5)),
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "rng": {
            "python": object(),
            "numpy": object(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": [],
            "sampling_generator": torch.Generator().get_state(),
        },
        "initialization_sha256": "1" * 64,
        "activation_selection_sha256": "2" * 64,
        "input_hashes": {"core4": "3" * 64},
        "camera_selection": {"selected_names": ["camera_a", "camera_b"]},
    }

    lineage = validate_decode_then_filter_resume_payload(
        payload,
        expected_candidate="c5_dtf_16",
        expected_start_step=80_000,
        expected_latent_shape=(2, 3, 5),
        expected_decoder_keys=set(decoder.state_dict()),
        expected_input_hashes={"core4": "3" * 64},
        expected_camera_names=["camera_a", "camera_b"],
    )

    assert lineage == {
        "source_candidate": "c5_dtf_16",
        "source_step": 80_000,
        "initialization_sha256": "1" * 64,
        "activation_selection_sha256": "2" * 64,
    }

    payload["rng"] = {"torch_cpu": torch.get_rng_state()}
    with pytest.raises(ValueError, match="RNG state"):
        validate_decode_then_filter_resume_payload(
            payload,
            expected_candidate="c5_dtf_16",
            expected_start_step=80_000,
            expected_latent_shape=(2, 3, 5),
            expected_decoder_keys=set(decoder.state_dict()),
            expected_input_hashes={"core4": "3" * 64},
            expected_camera_names=["camera_a", "camera_b"],
        )


def test_resume_restores_parameters_adam_and_every_cpu_rng_stream() -> None:
    random.seed(17)
    np.random.seed(18)
    torch.manual_seed(19)
    source_generator = torch.Generator().manual_seed(20)
    source_latent = torch.nn.Parameter(torch.rand((2, 3, 5)))
    source_decoder = DecodeThenFilterDecoder(latent_channels=5, width=16, activation="relu")
    source_optimizer = torch.optim.Adam(
        [
            {"params": [source_latent], "lr": 1.0e-3},
            {"params": list(source_decoder.parameters()), "lr": 5.0e-5},
        ]
    )
    loss = source_latent.square().mean() + sum(
        parameter.square().mean() for parameter in source_decoder.parameters()
    )
    loss.backward()
    source_optimizer.step()
    payload = {
        "latent": source_latent.detach().clone(),
        "decoder": source_decoder.state_dict(),
        "optimizer": source_optimizer.state_dict(),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": [],
            "sampling_generator": source_generator.get_state(),
        },
    }
    expected_random = random.random()
    expected_numpy = float(np.random.random())
    expected_torch = torch.rand(4)
    expected_generator = torch.rand(4, generator=source_generator)

    restored_latent = torch.nn.Parameter(torch.zeros_like(source_latent))
    restored_decoder = DecodeThenFilterDecoder(latent_channels=5, width=16, activation="relu")
    restored_optimizer = torch.optim.Adam(
        [
            {"params": [restored_latent], "lr": 0.0},
            {"params": list(restored_decoder.parameters()), "lr": 0.0},
        ]
    )
    restored_generator = torch.Generator().manual_seed(999)
    restore_decode_then_filter_resume_state(
        payload,
        latent=restored_latent,
        decoder=restored_decoder,
        optimizer=restored_optimizer,
        generator=restored_generator,
        restore_cuda_rng=False,
    )

    assert torch.equal(restored_latent, source_latent)
    assert all(
        torch.equal(restored, source)
        for restored, source in zip(
            restored_decoder.parameters(), source_decoder.parameters(), strict=True
        )
    )
    restored_state = restored_optimizer.state_dict()
    source_state = source_optimizer.state_dict()
    assert restored_state["param_groups"] == source_state["param_groups"]
    assert all(
        torch.equal(restored_state["state"][key][name], value)
        if isinstance(value, torch.Tensor)
        else restored_state["state"][key][name] == value
        for key, state in source_state["state"].items()
        for name, value in state.items()
    )
    assert random.random() == expected_random
    assert float(np.random.random()) == expected_numpy
    assert torch.equal(torch.rand(4), expected_torch)
    assert torch.equal(torch.rand(4, generator=restored_generator), expected_generator)


def test_interrupted_finetune_recovers_improved_dual_track_selection_from_log() -> None:
    recovered = recover_logged_checkpoint_selection(
        {
            "best_render": {"step": 155_000, "score": [0.0011, -0.9955, 0.029]},
            "best_artifact_safe": {
                "step": 160_000,
                "score": [0.00158, 0.00175, 0.0050],
            },
        },
        [
            {
                "step": 165_000,
                "selection_evaluation": {
                    "render": {
                        "multi_light_hdr_mae": 0.0012,
                        "display_ssim": 0.9954,
                        "material_error": 0.030,
                    },
                    "artifact": {
                        "obvious_rectangular_black_blocks": False,
                        "novel_dark_fraction": 0.00153,
                        "halo_fraction": 0.00180,
                        "metallic_boundary_proxy_mae": 0.0049,
                    },
                },
            }
        ],
    )

    assert recovered["best_render"]["step"] == 155_000
    assert recovered["best_artifact_safe"] == {
        "step": 165_000,
        "score": [0.00153, 0.0018, 0.0049],
    }


def test_interrupted_finetune_reports_full_lineage_steps_and_final_segment() -> None:
    result = summarize_resume_training_result(
        planned_start_step=160_000,
        segment_start_step=170_000,
        completed_step=180_000,
        required_step=180_000,
        segment_elapsed_seconds=699.5,
    )

    assert result == {
        "completed_steps": 180_000,
        "required_steps": 180_000,
        "source_step": 160_000,
        "resume_segment_start_step": 170_000,
        "steps_executed": 20_000,
        "segment_steps_executed": 10_000,
        "elapsed_seconds": 699.5,
        "elapsed_scope": "final_process_segment",
    }


def test_continuation_manifest_is_isolated_and_binds_80k_lineage() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/train/scifihelmet_c4_dtf_v1.yaml").read_text(encoding="utf-8")
    )
    manifest = build_decode_then_filter_continuation_manifest(
        config,
        source_training_manifest_sha256="1" * 64,
        source_checkpoint_sha256="2" * 64,
        config_sha256="3" * 64,
        input_hashes={"core4_manifest": "4" * 64},
        output_dir="outputs/compression/scifihelmet/c4_dtf_v1/c5_dtf_16_resume_120k",
    )

    assert manifest["candidate"] == {
        "name": "c5_dtf_16",
        "latent_channels": 5,
        "decoder_width": 16,
        "activation": "relu",
    }
    assert manifest["lineage"] == {
        "source_candidate": "c5_dtf_16",
        "source_step": 80_000,
        "source_training_manifest_sha256": "1" * 64,
        "source_checkpoint_sha256": "2" * 64,
    }
    assert manifest["training"] == {
        "start_step": 80_000,
        "stop_step": 120_000,
        "phase": "low_lr_continuation",
        "schedule": "cosine_1x_to_0.25x",
        "batch_mix": config["batch_mix"],
        "loss": config["loss"],
    }
    assert manifest["selection"]["interval"] == 5_000
    assert manifest["selection"]["checkpoint_tracks"] == [
        "best-render",
        "best-artifact-safe",
    ]
    assert manifest["tracking"]["yellow_material_subset"]["evaluation_only"] is True
    assert manifest["output_dir"].endswith("/c5_dtf_16_resume_120k")


def test_c4_160k_continuation_manifest_is_isolated_from_c4_80k() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/train/scifihelmet_c4_dtf_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    manifest = build_decode_then_filter_continuation_manifest(
        config,
        candidate_name="c4_dtf_16_selected",
        stop_step=160_000,
        source_training_manifest_sha256="1" * 64,
        source_checkpoint_sha256="2" * 64,
        config_sha256="3" * 64,
        input_hashes={"core4_manifest": "4" * 64},
        output_dir=(
            "outputs/compression/scifihelmet/c4_dtf_v1/"
            "c4_dtf_16_resume_160k"
        ),
    )

    assert manifest["candidate"] == {
        "name": "c4_dtf_16_selected",
        "latent_channels": 4,
        "decoder_width": 16,
        "activation": "relu",
    }
    assert manifest["lineage"]["source_candidate"] == "c4_dtf_16_selected"
    assert manifest["lineage"]["source_step"] == 80_000
    assert manifest["training"]["stop_step"] == 160_000
    assert manifest["training"]["batch_mix"] == config["batch_mix"]
    assert manifest["training"]["loss"] == config["loss"]
    assert manifest["tracking"]["yellow_material_subset"]["evaluation_only"]
    assert manifest["output_dir"].endswith("/c4_dtf_16_resume_160k")


def test_camera31_manifest_marks_the_160k_parent_as_a_non_exact_finetune() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/train/scifihelmet_c4_dtf_camera31_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    manifest = build_decode_then_filter_camera_finetune_manifest(
        config,
        source_training_manifest_sha256="1" * 64,
        source_checkpoint_sha256="2" * 64,
        config_sha256="3" * 64,
        input_hashes={"core4_manifest": "4" * 64},
        output_dir=(
            "outputs/compression/scifihelmet/c4_dtf_v1/"
            "c4_dtf_16_camera31_ft_180k"
        ),
    )

    assert manifest["lineage"]["source_step"] == 160_000
    assert manifest["lineage"]["kind"] == "changed_camera_distribution_finetune"
    assert manifest["lineage"]["exact_continuation"] is False
    assert manifest["training"]["start_step"] == 160_000
    assert manifest["training"]["stop_step"] == 180_000
    assert manifest["training"]["batch_mix"] == config["batch_mix"]
    assert manifest["training"]["loss"] == config["loss"]
    assert manifest["camera_distribution"]["strategy"] == (
        "explicit_audited_pool_v1"
    )
    assert manifest["camera_distribution"]["camera_count"] == 31
    assert manifest["camera_distribution"]["yellow_audit_influenced_selection"]
    assert manifest["camera_distribution"]["dynamic_roi_sampling"] is False
    assert manifest["output_dir"].endswith("/c4_dtf_16_camera31_ft_180k")


def test_camera31_fresh_manifest_is_an_isolated_zero_to_80k_lineage() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/train/scifihelmet_c4_dtf_camera31_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    manifest = build_decode_then_filter_camera31_fresh_manifest(
        config,
        config_sha256="3" * 64,
        input_hashes={"core4_manifest": "4" * 64},
        git_commit="dirty-private-worktree",
        selected_activation="relu",
        activation_selection_sha256="5" * 64,
        output_dir=(
            "outputs/compression/scifihelmet/c4_dtf_v1/"
            "c4_dtf_16_camera31_fresh_80k"
        ),
    )

    assert manifest["candidate"] == {
        "name": "c4_dtf_16_selected",
        "latent_channels": 4,
        "decoder_width": 16,
        "activation": "relu",
    }
    assert manifest["lineage"] == {
        "kind": "fresh_random_initialization",
        "source_step": 0,
        "parent_checkpoint_used": False,
    }
    assert manifest["training"]["start_step"] == 0
    assert manifest["training"]["stop_step"] == 80_000
    assert manifest["training"]["phases"] == config["training"]["phases"]
    assert manifest["training"]["batch_mix"] == config["batch_mix"]
    assert manifest["training"]["loss"] == config["loss"]
    assert manifest["camera_distribution"]["strategy"] == (
        "explicit_audited_pool_v1"
    )
    assert manifest["camera_distribution"]["camera_count"] == 31
    assert manifest["camera_distribution"]["yellow_audit_influenced_selection"]
    assert manifest["camera_distribution"]["dynamic_roi_sampling"] is False
    assert manifest["tracking"]["yellow_material_subset"]["evaluation_only"]
    assert manifest["tracking"]["yellow_material_subset"]["affects_loss_or_sampling"] is False
    assert manifest["output_dir"].endswith("/c4_dtf_16_camera31_fresh_80k")


def test_material_subset_metrics_track_dark_tail_without_affecting_training() -> None:
    reference = Core4Targets(
        base_color_linear=torch.tensor(
            [[0.2, 0.2, 0.0], [0.4, 0.4, 0.0], [0.1, 0.1, 0.1]]
        ),
        normal_xyz=torch.tensor(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
        ),
        roughness=torch.tensor([[0.5], [0.5], [0.5]]),
        metallic=torch.tensor([[0.0], [0.0], [0.0]]),
        height=1,
        width=3,
    )
    prediction = DecodedMaterial(
        base_color_linear=torch.tensor(
            [[0.2, 0.2, 0.0], [0.0, 0.0, 0.0], [0.9, 0.9, 0.9]]
        ),
        normal_xy=torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]),
        normal_xyz=torch.tensor(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        roughness=torch.tensor([[0.5], [0.1], [0.5]]),
        metallic=torch.tensor([[0.0], [0.0], [1.0]]),
    )

    metrics = material_subset_metrics(
        reference,
        prediction,
        torch.tensor([True, True, False]),
    )

    assert metrics["sample_count"] == 2
    assert metrics["novel_dark_fraction"] == 0.5
    assert metrics["base_color_linear_mae"] == pytest.approx(0.4 / 3.0)
    assert metrics["normal_mean_degrees"] == pytest.approx(45.0)
    assert metrics["roughness_mae"] == pytest.approx(0.2)
    assert metrics["metallic_mae"] == 0.0


def test_paired_precheck_objective_contains_only_generic_material_terms() -> None:
    base = torch.tensor([[0.2, 0.4, 0.6], [0.7, 0.3, 0.1]])
    normal = torch.tensor([[0.0, 0.0, 1.0], [0.6, 0.0, 0.8]])
    roughness = torch.tensor([[0.4], [0.8]])
    metallic = torch.tensor([[0.7], [0.1]])
    prediction = DecodedMaterial(base, normal[:, :2], normal, roughness, metallic)
    target = Core4Targets(base, normal, roughness, metallic, height=1, width=2)

    total, terms = paired_precheck_objective(
        prediction,
        target,
        anchor_slice=slice(1, 2),
        quantization_error=torch.tensor(0.2),
        loss_config={
            "base_color_linear_l1": 0.5,
            "normal_cosine": 0.5,
            "roughness_l1": 0.25,
            "metallic_l1": 0.25,
            "subpixel_material_anchor": 0.5,
            "texel_center_anchor": 0.25,
            "unorm8_quantization": 0.25,
        },
    )

    assert torch.allclose(total, torch.tensor(0.05))
    assert set(terms) == {
        "material",
        "subpixel_material_anchor",
        "texel_center_anchor",
        "unorm8_quantization",
        "base_color_l1",
        "normal_cosine",
        "roughness_l1",
        "metallic_l1",
    }


def _precheck_manifest(
    activation: str,
    *,
    initialization_sha256: str = "1" * 64,
) -> dict[str, object]:
    return {
        "valid": True,
        "candidate": {
            "name": f"c4_dtf_16_{activation}_precheck",
            "latent_channels": 4,
            "decoder_width": 16,
            "activation": activation,
            "role": "paired_precheck",
            "max_steps": 10_000,
        },
        "initialization": {
            "sha256": initialization_sha256,
            "shared_mutable_storage": False,
        },
        "evaluation": {
            "base_color_linear_mae": 0.01,
            "normal_mean_degrees": 3.0,
            "normal_p95_degrees": 9.0,
            "roughness_mae": 0.02,
            "metallic_mae": 0.02,
            "generic_dark_fraction": 0.1,
            "generic_positive_halo_fraction": 0.005,
            "forward_timing": {"median_proxy_ms_per_repeat": 2.0},
            "dead_units": {"hidden_in": 0, "hidden_mid": 0},
        },
    }


def test_activation_selection_record_binds_paired_evidence_and_conservative_relu_rule() -> None:
    relu = _precheck_manifest("relu")
    silu = _precheck_manifest("silu")
    silu["evaluation"]["normal_p95_degrees"] = 11.0  # type: ignore[index]
    silu["evaluation"]["forward_timing"]["median_proxy_ms_per_repeat"] = 4.0  # type: ignore[index]

    record = build_activation_selection_record(
        relu,
        silu,
        selected_activation="relu",
        silu_clear_quality_advantage=False,
        rationale="SiLU does not show a clear overall quality advantage.",
    )

    assert record["selected_activation"] == "relu"
    assert record["paired_initialization_sha256"] == "1" * 64
    assert record["silu_clear_quality_advantage"] is False
    assert record["prechecks"]["relu"]["valid"] is True
    assert record["prechecks"]["silu"]["metrics"]["normal_p95_degrees"] == 11.0
    assert record["policy"] == "silu_only_with_clear_quality_advantage"
    assert "formal_holdout" not in json.dumps(record).lower()


def test_activation_selection_rejects_unpaired_initializations() -> None:
    try:
        build_activation_selection_record(
            _precheck_manifest("relu", initialization_sha256="1" * 64),
            _precheck_manifest("silu", initialization_sha256="2" * 64),
            selected_activation="relu",
            silu_clear_quality_advantage=False,
            rationale="not paired",
        )
    except ValueError as error:
        assert "initialization" in str(error).lower()
    else:
        raise AssertionError("mismatched paired initialization must be rejected")


def test_full_training_schedule_matches_the_three_frozen_phases() -> None:
    assert full_training_phase_at_step(1) == "material_continuous_pretrain"
    assert full_training_phase_at_step(15_000) == "material_continuous_pretrain"
    assert full_training_phase_at_step(15_001) == "render_first_joint"
    assert full_training_phase_at_step(65_000) == "render_first_joint"
    assert full_training_phase_at_step(65_001) == "low_lr_polish"
    assert full_training_phase_at_step(80_000) == "low_lr_polish"
    assert full_training_learning_rates_at_step(500) == (2.0e-2, 1.0e-3)
    assert full_training_learning_rates_at_step(80_000) == (1.0e-3, 5.0e-5)


def test_camera_pool_stops_when_triangle_coverage_increment_falls_below_half_percent() -> None:
    selection = select_camera_triangle_coverage(
        [
            torch.tensor([0, 1, 2, 3]),
            torch.tensor([3, 4, 5]),
            torch.tensor([5]),
            torch.tensor([6, 7, 8]),
        ],
        triangle_count=100,
        increment_stop=0.005,
        camera_limit=48,
    )

    assert selection["selected_indices"] == [0, 1]
    assert selection["records"][-1]["stopped"] is True
    assert selection["records"][-1]["increment"] == 0.0


def test_dtf_camera_spec_uses_render_defaults_and_allows_a_local_focus_override() -> None:
    render = {
        "camera_radius": 4.5,
        "target": [0.0, 0.0, 0.0],
        "up": [0.0, 1.0, 0.0],
        "vertical_fov_degrees": 45.0,
        "near": 0.1,
        "far": 10.0,
    }

    default = resolve_dtf_camera_spec(
        {"name": "orbit", "elevation_degrees": 0.0, "yaw_degrees": 180.0},
        render,
    )
    focused = resolve_dtf_camera_spec(
        {
            "name": "yellow_focus",
            "elevation_degrees": 20.0,
            "yaw_degrees": 150.0,
            "radius": 2.8,
            "target": [0.0, -0.67, -0.72],
        },
        render,
    )

    assert default["radius"] == 4.5
    assert default["target"] == (0.0, 0.0, 0.0)
    assert focused["radius"] == 2.8
    assert focused["target"] == (0.0, -0.67, -0.72)
    assert focused["yaw_degrees"] == 150.0


def test_explicit_audited_camera_pool_keeps_low_triangle_increment_views() -> None:
    selection = select_explicit_dtf_camera_pool(
        [
            torch.tensor([0, 1, 2, 3]),
            torch.tensor([3, 4, 5]),
            torch.tensor([5]),
        ],
        camera_names=["original", "coverage", "focused_repeat"],
        triangle_count=100,
        camera_limit=48,
    )

    assert selection["strategy"] == "explicit_audited_pool_v1"
    assert selection["selected_indices"] == [0, 1, 2]
    assert selection["selected_names"] == [
        "original",
        "coverage",
        "focused_repeat",
    ]
    assert selection["records"][-1]["increment"] == 0.0
    assert selection["records"][-1]["stopped"] is False


def test_full_training_objective_turns_on_only_generic_render_terms_after_pretrain() -> None:
    reference = torch.zeros((1, 2, 3))
    candidate = torch.ones((1, 2, 3))
    material_total = torch.tensor(0.2)
    pretrain, pretrain_terms = full_training_objective(
        material_total,
        reference_hdr=reference,
        candidate_hdr=candidate,
        reference_display=reference,
        candidate_display=candidate,
        phase="material_continuous_pretrain",
        loss_config={"render_hdr_l1": 1.0, "render_display_l1": 0.25},
    )
    joint, joint_terms = full_training_objective(
        material_total,
        reference_hdr=reference,
        candidate_hdr=candidate,
        reference_display=reference,
        candidate_display=candidate,
        phase="render_first_joint",
        loss_config={"render_hdr_l1": 1.0, "render_display_l1": 0.25},
    )

    assert torch.allclose(pretrain, torch.tensor(0.2))
    assert pretrain_terms == {}
    assert torch.allclose(joint, torch.tensor(1.45))
    assert set(joint_terms) == {"render_hdr_l1", "render_display_l1"}


def test_camera_distribution_finetune_uses_the_same_generic_render_objective() -> None:
    reference = torch.zeros((1, 2, 3))
    candidate = torch.ones((1, 2, 3))
    material_total = torch.tensor(0.2)

    total, terms = full_training_objective(
        material_total,
        reference_hdr=reference,
        candidate_hdr=candidate,
        reference_display=reference,
        candidate_display=candidate,
        phase="low_lr_camera_distribution_finetune",
        loss_config={"render_hdr_l1": 1.0, "render_display_l1": 0.25},
    )

    assert torch.isclose(total, torch.tensor(1.45))
    assert set(terms) == {"render_hdr_l1", "render_display_l1"}


def test_full_trainer_freezes_camera_light_and_dual_track_boundaries() -> None:
    config = yaml.safe_load(
        (ROOT / "configs/train/scifihelmet_c4_dtf_v1.yaml").read_text(
            encoding="utf-8"
        )
    )
    trainer = (ROOT / "scripts/train_scifihelmet_c4_dtf_full.py").read_text(
        encoding="utf-8"
    )

    assert 1 <= len(config["train_cameras"]) <= 48
    assert len(config["train_lights"]) == 6
    assert config["training_pool"]["cache_camera_gbuffer_only"] is True
    assert config["training_pool"]["randomize_lights_online"] is True
    assert config["training"]["selection_start"] == 20_000
    assert config["selection"]["checkpoint_tracks"] == [
        "best-render",
        "best-artifact-safe",
    ]
    assert "precheck_checkpoint_used" in trainer
    lowered = trainer.lower()
    for forbidden in (
        "commutativity",
        "activation_region",
        "dark_tail",
        "d2_",
        "d3_",
    ):
        assert forbidden not in lowered


def test_capacity_diagnostic_is_triggered_only_by_signed_c4_vs_r0b_quality_evidence() -> None:
    c4_manifest = {
        "valid": True,
        "candidate": {
            "name": "c4_dtf_16_selected",
            "latent_channels": 4,
            "decoder_width": 16,
            "activation": "relu",
            "role": "primary_full_training",
            "max_steps": 80_000,
            "activation_source": "paired_10k_precheck",
        },
        "final_evaluation": {
            "render": {"display_ssim": 0.9895},
            "artifact": {
                "novel_dark_fraction": 0.0036,
                "halo_fraction": 0.0076,
            },
            "material": {"normal_mean_degrees": 1.386},
        },
    }
    decision = build_capacity_diagnostic_eligibility(
        c4_manifest,
        r0b_control={
            "display_ssim": 0.99957,
            "normal_mean_degrees": 0.9298,
            "subpixel_filter_divergence": 0.0,
        },
        c4_manifest_sha256="a" * 64,
        r0b_document_sha256="b" * 64,
    )

    assert decision["eligible"] is True
    assert decision["next_candidate"] == "c4_dtf_32_diagnostic"
    assert decision["c4_manifest_sha256"] == "a" * 64
    assert decision["r0b_control_source"]["sha256"] == "b" * 64
    assert set(decision["quality_gaps"]) == {
        "display_ssim",
        "normal_mean_degrees",
        "subpixel_filter_safety",
    }
    assert "formal_holdout" not in json.dumps(decision).lower()


def test_weak_width32_capacity_result_routes_to_fresh_c5_upper_bound() -> None:
    def manifest(name: str, width: int, values: tuple[float, ...]) -> dict[str, object]:
        base, normal, rough, metal, dark, halo = values
        return {
            "valid": True,
            "candidate": {
                "name": name,
                "latent_channels": 4,
                "decoder_width": width,
                "activation": "relu",
                "role": "paired_precheck"
                if width == 16
                else "conditional_capacity_diagnostic",
                "max_steps": 10_000,
            },
            "evaluation": {
                "base_color_linear_mae": base,
                "normal_mean_degrees": normal,
                "roughness_mae": rough,
                "metallic_mae": metal,
                "generic_dark_fraction": dark,
                "generic_positive_halo_fraction": halo,
            },
            "cost": {"decoder_macs_per_pixel": 1_728 if width == 16 else 5_504},
        }

    decision = build_capacity_route_decision(
        manifest(
            "c4_dtf_16_relu_precheck",
            16,
            (0.0083, 3.43, 0.0211, 0.0232, 0.1655, 0.0061),
        ),
        manifest(
            "c4_dtf_32_diagnostic",
            32,
            (0.0099, 3.47, 0.0243, 0.0224, 0.2032, 0.0096),
        ),
        c4_16_manifest_sha256="a" * 64,
        c4_32_manifest_sha256="b" * 64,
    )

    assert decision["significant_capacity_benefit"] is False
    assert decision["next_candidate"] == "c5_dtf_16"
    assert decision["c5_is_channel_upper_bound"] is True
    assert decision["comparison"]["decoder_macs_ratio"] > 3.0
    assert decision["policy"]["minimum_composite_improvement"] == 0.10


def test_c5_unorm8_pack_uses_one_rgba_and_one_r_texture() -> None:
    latent = torch.linspace(0.0, 1.0, 30).reshape(2, 3, 5)

    packed = pack_decode_then_filter_latent_unorm8(latent)

    assert [(item["storage_channels"], item["mode"]) for item in packed] == [
        ("rgba8", "RGBA"),
        ("r8", "L"),
    ]
    assert packed[0]["encoded"].shape == (2, 3, 4)
    assert packed[1]["encoded"].shape == (2, 3)
    assert packed[0]["encoded"].dtype.name == "uint8"
