from __future__ import annotations

import itertools

import numpy as np
import pytest
import torch

from cg_frontier.compression.exact_basecolor import (
    AffineLatticeCodec,
    ExactAffineMaterialDecoder,
    LatticeCapacity,
    SeparatedCodec,
    bilinear_sample_wrap,
    enumerate_lattice_capacity,
    half_up_unorm8,
)


def test_half_up_unorm8_has_exact_byte_forward_and_ste_gradient() -> None:
    values = torch.tensor(
        [-0.1, 0.0, 0.5 / 255.0, 1.5 / 255.0, 1.0, 1.1],
        dtype=torch.float64,
        requires_grad=True,
    )
    quantized = half_up_unorm8(values, ste=True)
    expected = torch.tensor([0, 0, 1, 2, 255, 255], dtype=torch.float64) / 255.0
    torch.testing.assert_close(quantized, expected, rtol=0.0, atol=0.0)
    quantized.sum().backward()
    torch.testing.assert_close(values.grad[1:-1], torch.ones(4, dtype=torch.float64))


@pytest.mark.parametrize("codec", [SeparatedCodec(), AffineLatticeCodec((-1, -1, -1), 128)])
def test_exact_codecs_recover_random_basecolor_bytes(codec: object) -> None:
    generator = torch.Generator().manual_seed(20260811)
    colors = torch.randint(0, 256, (100_000, 3), generator=generator, dtype=torch.int64)
    colors = torch.cat(
        (
            colors,
            torch.tensor(list(itertools.product((0, 255), repeat=3)), dtype=torch.int64),
        )
    )
    lower, upper = codec.valid_bounds(colors)
    residual = torch.div(lower + upper, 2, rounding_mode="floor")
    encoded = codec.encode_hard(colors, residual)
    decoded = codec.decode_basecolor(encoded.to(torch.float32) / 255.0)
    torch.testing.assert_close(decoded, colors.to(torch.float32) / 255.0, rtol=0.0, atol=1e-7)
    assert int(encoded.min()) >= 0
    assert int(encoded.max()) <= 255


def test_mixed_codec_moves_every_stored_channel_along_null_direction() -> None:
    codec = AffineLatticeCodec((-1, -1, -1), 128)
    color = torch.tensor([[40, 80, 120]], dtype=torch.int64)
    lower, upper = codec.valid_bounds(color)
    assert int(upper - lower) >= 1
    first = codec.encode_hard(color, lower)
    second = codec.encode_hard(color, lower + 1)
    torch.testing.assert_close(
        (second - first).reshape(-1), torch.tensor([-1, -1, -1, 1], dtype=torch.int64)
    )


@pytest.mark.parametrize("codec", [SeparatedCodec(), AffineLatticeCodec((-1, -1, -1), 128)])
def test_affine_decode_commutes_with_wrap_bilinear(codec: object) -> None:
    generator = torch.Generator().manual_seed(83)
    colors = torch.randint(0, 256, (9, 11, 3), generator=generator, dtype=torch.int64)
    lower, upper = codec.valid_bounds(colors)
    residual = torch.div(lower + upper, 2, rounding_mode="floor")
    encoded = codec.encode_hard(colors, residual).to(torch.float32) / 255.0
    uv = torch.cat(
        (
            torch.rand((10_000, 2), generator=generator),
            torch.tensor(
                [
                    [0.0, 0.0],
                    [1.0, 1.0],
                    [-1e-7, 0.5],
                    [0.5, 1.0 + 1e-7],
                    [0.5 / 11.0, 0.5 / 9.0],
                    [10.5 / 11.0, 8.5 / 9.0],
                ]
            ),
        )
    )
    decoded = codec.decode_basecolor(bilinear_sample_wrap(encoded, uv))
    direct = bilinear_sample_wrap(colors.to(torch.float32) / 255.0, uv)
    assert float((decoded - direct).abs().max()) <= 2e-6


@pytest.mark.parametrize("codec", [SeparatedCodec(), AffineLatticeCodec((-1, -1, -1), 128)])
def test_fake_and_hard_codec_forwards_are_value_identical(codec: object) -> None:
    generator = torch.Generator().manual_seed(47)
    colors = torch.randint(0, 256, (10_000, 3), generator=generator, dtype=torch.int64)
    lower, upper = codec.valid_bounds(colors)
    residual = lower.to(torch.float32) + torch.rand((10_000,), generator=generator) * (upper - lower).to(torch.float32)
    fake = codec.encode_fake(colors, residual)
    hard = codec.encode_hard(colors, residual).to(torch.float32) / 255.0
    torch.testing.assert_close(fake, hard, rtol=0.0, atol=0.0)


def test_exact_decoder_blocks_basecolor_gradient_but_keeps_auxiliary_gradient() -> None:
    codec = AffineLatticeCodec((-1, -1, -1), 128)
    decoder = ExactAffineMaterialDecoder(codec=codec, train_basecolor=False)
    color = torch.tensor([[40, 80, 120]], dtype=torch.int64)
    residual = torch.tensor([10.25], requires_grad=True)
    latent = codec.encode_fake(color, residual)
    decoded = decoder(latent)
    base_gradient = torch.autograd.grad(decoded.base_color_linear.sum(), residual, retain_graph=True)[0]
    torch.testing.assert_close(base_gradient, torch.zeros_like(base_gradient), rtol=0.0, atol=1e-7)
    auxiliary_loss = decoded.normal_xyz[..., 0].sum() + decoded.roughness.sum() + decoded.metallic.sum()
    auxiliary_gradient = torch.autograd.grad(auxiliary_loss, residual)[0]
    assert torch.isfinite(auxiliary_gradient).all()
    assert float(torch.linalg.vector_norm(auxiliary_gradient)) > 0.0
    assert decoder.base_weight.requires_grad is False
    assert decoder.base_bias.requires_grad is False


def test_lattice_capacity_enumeration_is_deterministic_and_filters_small_fibers() -> None:
    colors = np.array([[0, 0, 0], [10, 30, 80], [128, 128, 128], [255, 255, 255]], dtype=np.int16)
    frequencies = np.array([4, 3, 2, 1], dtype=np.int64)
    left = enumerate_lattice_capacity(colors, frequencies, min_states=2, top_k=16)
    right = enumerate_lattice_capacity(colors, frequencies, min_states=2, top_k=16)
    assert left == right
    assert len(left) == 16
    assert all(item.min_states >= 2 for item in left)
    assert left == sorted(left, key=lambda item: item.capacity_sort_key)


def test_lattice_ties_end_in_dictionary_order_without_hidden_t0_preference() -> None:
    shared = dict(min_states=64, max_states=256, mean_states=200.0, p05_states=100.0, weighted_mean_log2_states=7.0)
    values = [
        LatticeCapacity(kernel_rgb=(1, 1, 1), t0=128, **shared),
        LatticeCapacity(kernel_rgb=(-1, -1, -1), t0=255, **shared),
        LatticeCapacity(kernel_rgb=(-1, -1, -1), t0=0, **shared),
    ]
    ordered = sorted(values, key=lambda item: item.capacity_sort_key)
    assert [(item.kernel_rgb, item.t0) for item in ordered] == [
        ((-1, -1, -1), 0),
        ((-1, -1, -1), 255),
        ((1, 1, 1), 128),
    ]
