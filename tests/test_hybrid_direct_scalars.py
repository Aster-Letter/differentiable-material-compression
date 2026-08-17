from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT / "src", ROOT / "scripts"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from analyze_scifihelmet_interpolation import (  # noqa: E402
    _decoder_raw,
    _direct_scalars_postprocess,
    _load_decoder,
)
from cg_frontier.compression.hybrid import export_hybrid_textures, pack_hybrid_textures  # noqa: E402
from cg_frontier.compression.hybrid_direct_scalars import (  # noqa: E402
    NormalAuxDecoder,
    build_direct_scalar_auxiliary,
    decode_direct_scalars,
)
from cg_frontier.compression.material import Core4Targets  # noqa: E402


def _targets() -> Core4Targets:
    roughness = torch.tensor([[0.2], [0.4], [0.6], [0.8]], requires_grad=True)
    metallic = torch.tensor([[0.0], [0.1], [0.9], [1.0]], requires_grad=True)
    return Core4Targets(
        base_color_linear=torch.full((4, 3), 0.5),
        normal_xyz=torch.tensor([[0.0, 0.0, 1.0]]).repeat(4, 1),
        roughness=roughness,
        metallic=metallic,
        height=2,
        width=2,
    )


def test_direct_scalar_cost_is_derived() -> None:
    decoder = NormalAuxDecoder()
    assert (decoder.parameter_count, decoder.weight_bytes_float32, decoder.macs_per_pixel) == (32, 128, 24)


def test_direct_scalars_are_half_up_quantized_and_gradient_isolated() -> None:
    targets = _targets()
    normal_latent = torch.rand((2, 2, 2), requires_grad=True)
    auxiliary = build_direct_scalar_auxiliary(normal_latent, targets)
    expected = torch.floor(
        torch.cat((targets.roughness, targets.metallic), dim=-1).reshape(2, 2, 2).detach() * 255.0
        + 0.5
    ) / 255.0
    torch.testing.assert_close(auxiliary[..., 2:], expected, rtol=0.0, atol=0.0)
    assert auxiliary.requires_grad is False
    assert normal_latent.grad is None
    assert targets.roughness.grad is None
    assert targets.metallic.grad is None


def test_direct_scalar_decode_bypasses_scalar_postprocess() -> None:
    decoder = NormalAuxDecoder()
    auxiliary = torch.tensor([[0.5, 0.5, 0.25, 0.75]], dtype=torch.float32)
    direct = torch.tensor([[0.1, 0.2, 0.3]], dtype=torch.float32)
    decoded = decode_direct_scalars(decoder, auxiliary, direct)
    torch.testing.assert_close(decoded.base_color_linear, direct)
    torch.testing.assert_close(decoded.roughness, auxiliary[..., 2:3])
    torch.testing.assert_close(decoded.metallic, auxiliary[..., 3:4])


def test_numpy_analyzer_matches_direct_scalar_torch_path(tmp_path: Path) -> None:
    decoder = NormalAuxDecoder()
    path = tmp_path / "decoder.npz"
    arrays = {name: value.detach().numpy() for name, value in decoder.state_dict().items()}
    arrays["direct_scalars.marker"] = np.asarray([1.0], dtype=np.float32)
    np.savez(path, **arrays)
    loaded = _load_decoder(path)
    auxiliary = np.random.default_rng(19).random((13, 4), dtype=np.float32)
    direct = np.random.default_rng(23).random((13, 3), dtype=np.float32)
    expected = decode_direct_scalars(
        decoder, torch.from_numpy(auxiliary), torch.from_numpy(direct)
    )
    actual = _direct_scalars_postprocess(_decoder_raw(auxiliary, loaded), auxiliary, direct)
    np.testing.assert_allclose(actual["normal_xyz"], expected.normal_xyz.detach().numpy(), rtol=2e-6, atol=2e-6)
    np.testing.assert_array_equal(actual["roughness_linear"], auxiliary[:, 2])
    np.testing.assert_array_equal(actual["metallic_linear"], auxiliary[:, 3])


def test_direct_scalar_layout_is_seven_logical_channels_and_two_samples(tmp_path: Path) -> None:
    direct = torch.rand((2, 2, 3))
    auxiliary = torch.rand((2, 2, 4))
    texture_a, texture_b = pack_hybrid_textures(direct, auxiliary)
    assert texture_a.shape[-1] == 4 and texture_b.shape[-1] == 3
    report = export_hybrid_textures(direct, auxiliary, tmp_path / "a.png", tmp_path / "b.png")
    assert report["logical_raw_bytes"] == 2 * 2 * 7
    assert report["texture_samples"] == 2
    assert report["texture_b"]["format"] == "RGB8_UNORM_LOGICAL"
