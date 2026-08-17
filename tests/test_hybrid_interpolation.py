from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cg_frontier.compression.hybrid import (
    AuxMaterialDecoder,
    decode_auxiliary,
    deterministic_pca_initialization,
    export_hybrid_textures,
    pack_hybrid_textures,
)
from cg_frontier.compression.material import Core4Targets
from cg_frontier.compression.render_loss import bilinear_sample_top_down_wrap


def _targets() -> Core4Targets:
    normal = torch.tensor(
        [[0.0, 0.0, 1.0], [0.2, 0.0, 0.9797959], [0.0, -0.3, 0.9539392], [0.2, -0.3, 0.9327379]],
        dtype=torch.float32,
    )
    return Core4Targets(
        base_color_linear=torch.tensor(
            [[0.0, 0.1, 0.2], [0.2, 0.3, 0.4], [0.4, 0.5, 0.6], [0.6, 0.7, 0.8]],
            dtype=torch.float32,
        ),
        normal_xyz=torch.nn.functional.normalize(normal, dim=-1),
        roughness=torch.tensor([[0.1], [0.3], [0.7], [0.9]], dtype=torch.float32),
        metallic=torch.tensor([[0.0], [0.2], [0.8], [1.0]], dtype=torch.float32),
        height=2,
        width=2,
    )


@pytest.mark.parametrize("channels,cost", [(2, (60, 240, 48)), (3, (68, 272, 56))])
def test_aux_decoder_cost_is_derived_from_module(channels: int, cost: tuple[int, int, int]) -> None:
    decoder = AuxMaterialDecoder(channels)
    assert (decoder.parameter_count, decoder.weight_bytes_float32, decoder.macs_per_pixel) == cost


def test_basecolor_is_direct_and_has_no_decoder_gradient() -> None:
    decoder = AuxMaterialDecoder(2)
    direct = torch.rand((2, 2, 3), requires_grad=True)
    latent = torch.rand((2, 2, 2), requires_grad=True)
    texture_a, texture_b = pack_hybrid_textures(direct, latent)
    uv = torch.tensor([[[0.5, 0.5]]])
    sampled_a = bilinear_sample_top_down_wrap(texture_a, uv)
    sampled_b = bilinear_sample_top_down_wrap(texture_b, uv)
    decoded = decode_auxiliary(
        decoder, torch.cat((sampled_a[..., 3:4], sampled_b), dim=-1), sampled_a[..., :3]
    )
    torch.testing.assert_close(decoded.base_color_linear, sampled_a[..., :3], rtol=0.0, atol=0.0)
    decoded.base_color_linear.sum().backward()
    assert direct.grad is None
    assert latent.grad is not None
    torch.testing.assert_close(latent.grad, torch.zeros_like(latent), rtol=0.0, atol=0.0)
    assert all(parameter.grad is None for parameter in decoder.parameters())


def test_two_texture_sampler_uses_identical_uv_contract() -> None:
    direct = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3) / 12.0
    latent = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2) / 8.0
    texture_a, texture_b = pack_hybrid_textures(direct, latent)
    uv = torch.tensor([[0.25, 0.25], [1.25, 0.25], [0.5, 0.5]])
    sampled_a = bilinear_sample_top_down_wrap(texture_a, uv)
    sampled_b = bilinear_sample_top_down_wrap(texture_b, uv)
    torch.testing.assert_close(sampled_a[..., 3], bilinear_sample_top_down_wrap(latent[..., :1], uv)[..., 0])
    torch.testing.assert_close(sampled_b, bilinear_sample_top_down_wrap(latent[..., 1:], uv))


def test_pca_initialization_is_deterministic_and_optimizer_only() -> None:
    targets = _targets()
    mask = np.asarray([[True, True], [True, False]])
    first = deterministic_pca_initialization(targets, mask, 2)
    second = deterministic_pca_initialization(targets, mask, 2)
    torch.testing.assert_close(first.direct_base_linear, second.direct_base_linear, rtol=0.0, atol=0.0)
    torch.testing.assert_close(first.auxiliary_latent, second.auxiliary_latent, rtol=0.0, atol=0.0)
    for left, right in zip(first.decoder.parameters(), second.decoder.parameters(), strict=True):
        torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)
    assert first.metadata["optimizer_texels"] == 3


@pytest.mark.parametrize("channels,mode,logical", [(2, "L", 20), (3, "LA", 24)])
def test_hybrid_png_reload_and_logical_bytes(
    tmp_path: Path, channels: int, mode: str, logical: int
) -> None:
    targets = _targets()
    init = deterministic_pca_initialization(targets, np.ones((2, 2), dtype=bool), channels)
    a_path, b_path = tmp_path / "a.png", tmp_path / "b.png"
    report = export_hybrid_textures(
        init.direct_base_linear, init.auxiliary_latent, a_path, b_path
    )
    assert Image.open(a_path).mode == "RGBA"
    assert Image.open(b_path).mode == mode
    assert report["logical_raw_bytes"] == logical
    assert report["texture_samples"] == 2


def test_source_contains_no_basecolor_sigmoid_path() -> None:
    source = (SRC / "cg_frontier/compression/hybrid.py").read_text(encoding="utf-8")
    body = source.split("def decode_auxiliary", 1)[1].split("def pack_hybrid_textures", 1)[0]
    assert "direct_base_linear" in body
    assert "sigmoid(raw[..., 0" not in body


def test_phase0_configs_freeze_hybrid_layout_and_sealed_data() -> None:
    for candidate, channels, raw_bytes in (("c1", 5, 20 * 1024 * 1024), ("c2", 6, 24 * 1024 * 1024)):
        path = ROOT / f"configs/eval/scifihelmet_hybrid_phase0_{candidate}.yaml"
        text = path.read_text(encoding="utf-8")
        config = yaml.safe_load(text)
        assert "formal_holdout" not in text.lower()
        assert config["representation"]["logical_channels"] == channels
        assert config["representation"]["theoretical_raw_bytes_no_mips"] == raw_bytes
        assert config["representation"]["texture_samples"] == 2
        assert len(config["frozen_sha256"]["texture_a_png"]) == 64
        assert len(config["frozen_sha256"]["texture_b_png"]) == 64
        assert len(config["frozen_sha256"]["decoder_npz"]) == 64


def test_training_configs_freeze_schedule_and_independent_initializers() -> None:
    scripts = ROOT / "scripts"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))
    from train_scifihelmet_hybrid import _load_config

    c1 = _load_config(ROOT / "configs/train/scifihelmet_hybrid_c1.yaml")
    c2 = _load_config(ROOT / "configs/train/scifihelmet_hybrid_c2.yaml")
    for config, candidate, channels in ((c1, "c1", 2), (c2, "c2", 3)):
        text = str(config).lower()
        assert "formal_holdout" not in text
        assert config["candidate"] == candidate
        assert config["aux_channels"] == channels
        assert config["training"]["warmup_steps"] == 500
        assert config["training"]["max_steps"] == 10000
        assert config["training"]["max_minutes"] == 30
        assert config["training"]["latent_learning_rate"] == 2.0e-4
        assert config["training"]["decoder_learning_rate"] == 2.0e-5
        assert config["training"]["evaluation_interval"] == 250
    assert "phase0/c2" in c2["inputs"]["initialization_texture_a"]
    assert "hybrid_interpolation_v1/c1/" not in c2["inputs"]["initialization_texture_a"]
