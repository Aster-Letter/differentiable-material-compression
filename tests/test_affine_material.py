from __future__ import annotations

from itertools import product
from pathlib import Path
import sys

import pytest
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cg_frontier.compression.affine_material import (
    SafeAffineMaterialDecoder,
    certify_affine,
    decode_affine_material,
    decode_filtered_affine_material,
    export_affine_decoder,
    reconstruct_positive_normal,
    reload_affine_decoder,
)


def test_safe_affine_is_valid_on_latent_domain_and_folds_exactly() -> None:
    torch.manual_seed(7)
    decoder = SafeAffineMaterialDecoder(margin=1.0e-3).double()
    latent_corners = torch.tensor(
        list(product((0.0, 1.0), repeat=4)), dtype=torch.float64
    )

    decoded = decoder(latent_corners)
    weight, bias = decoder.fold_affine()
    folded = F.linear(latent_corners, weight, bias)
    seven = torch.cat(
        (
            decoded.base_color_linear,
            decoded.normal_xy,
            decoded.roughness,
            decoded.metallic,
        ),
        dim=-1,
    )

    assert torch.allclose(seven, folded, atol=1.0e-12, rtol=1.0e-12)
    assert torch.all((seven[:, [0, 1, 2, 5, 6]] >= 0.0))
    assert torch.all((seven[:, [0, 1, 2, 5, 6]] <= 1.0))
    assert torch.all(torch.linalg.vector_norm(decoded.normal_xy, dim=-1) < 1.0)
    assert torch.allclose(
        torch.linalg.vector_norm(decoded.normal_xyz, dim=-1),
        torch.ones(16, dtype=torch.float64),
        atol=1.0e-12,
        rtol=1.0e-12,
    )
    seven.square().mean().backward()
    assert all(
        parameter.grad is not None and torch.all(torch.isfinite(parameter.grad))
        for parameter in decoder.parameters()
    )


def test_affine_certificate_bounds_domain_and_fails_closed() -> None:
    torch.manual_seed(11)
    decoder = SafeAffineMaterialDecoder(margin=2.0e-3).double()
    with torch.no_grad():
        for parameter in decoder.parameters():
            parameter.normal_()
    weight, bias = decoder.fold_affine()
    certificate = certify_affine(weight, bias, margin=decoder.margin)
    corners = torch.tensor(
        list(product((0.0, 1.0), repeat=4)), dtype=torch.float64
    )
    values = F.linear(corners, weight, bias)
    scalar_rows = [0, 1, 2, 5, 6]

    assert certificate["valid"] is True
    assert certificate["finite"] is True
    assert certificate["dtype"] == "float64"
    assert certificate["margin"] == decoder.margin
    assert torch.all(
        torch.tensor(certificate["scalar_lower_bounds"], dtype=torch.float64)
        <= values[:, scalar_rows].amin(dim=0)
    )
    assert torch.all(
        torch.tensor(certificate["scalar_upper_bounds"], dtype=torch.float64)
        >= values[:, scalar_rows].amax(dim=0)
    )
    assert certificate["normal_max_radius"] < 1.0
    assert certificate["certificate_margin"] > 0.0

    with pytest.raises(ValueError, match="shape"):
        certify_affine(weight[:6], bias, margin=decoder.margin)
    with pytest.raises(ValueError, match="finite"):
        certify_affine(weight.clone().fill_(float("nan")), bias, margin=decoder.margin)
    with pytest.raises(ValueError, match="margin"):
        certify_affine(weight, bias, margin=0.0)


def test_affine_certificate_rejects_scalar_clearance_below_requested_margin() -> None:
    weight = torch.zeros((7, 4), dtype=torch.float64)
    bias = torch.tensor([0.0005, 0.5, 0.5, 0.0, 0.0, 0.5, 0.5], dtype=torch.float64)

    with pytest.raises(ValueError, match="full latent domain"):
        certify_affine(weight, bias, margin=0.001)


def test_affine_certificate_rejects_normal_clearance_below_requested_margin() -> None:
    weight = torch.zeros((7, 4), dtype=torch.float64)
    bias = torch.tensor([0.5, 0.5, 0.5, 0.9995, 0.0, 0.5, 0.5], dtype=torch.float64)

    with pytest.raises(ValueError, match="full latent domain"):
        certify_affine(weight, bias, margin=0.001)


def test_deployment_affine_outputs_direct_material_semantics() -> None:
    decoder = SafeAffineMaterialDecoder(margin=1.0e-3).double()
    with torch.no_grad():
        for parameter in decoder.parameters():
            parameter.uniform_(-2.0, 2.0)
    weight, bias = decoder.fold_affine()
    latent = torch.tensor(
        [[0.0, 0.25, 0.75, 1.0], [1.0, 0.5, 0.125, 0.0]],
        dtype=torch.float64,
    )

    decoded = decode_affine_material(latent, weight, bias)
    seven = torch.cat(
        (
            decoded.base_color_linear,
            decoded.normal_xy,
            decoded.roughness,
            decoded.metallic,
        ),
        dim=-1,
    )

    assert torch.equal(seven, F.linear(latent, weight, bias))


def test_positive_normal_reconstruction_is_finite_unit_and_has_no_normalize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    generator = torch.Generator().manual_seed(23)
    random_xy = torch.randn(32, 2, generator=generator, dtype=torch.float64)
    random_xy = random_xy / torch.linalg.vector_norm(
        random_xy, dim=-1, keepdim=True
    ) * torch.rand(32, 1, generator=generator, dtype=torch.float64) * 0.98
    normal_xy = torch.cat(
        (
            torch.zeros(1, 2, dtype=torch.float64),
            torch.tensor([[1.0 - 1.0e-10, 0.0]], dtype=torch.float64),
            random_xy,
        ),
        dim=0,
    )

    monkeypatch.setattr(
        F,
        "normalize",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("final normalize is forbidden")
        ),
    )
    normal_xyz = reconstruct_positive_normal(normal_xy)

    assert torch.all(torch.isfinite(normal_xyz))
    assert torch.all(normal_xyz[..., 2] > 0.0)
    assert torch.allclose(
        torch.linalg.vector_norm(normal_xyz, dim=-1),
        torch.ones(normal_xyz.shape[0], dtype=torch.float64),
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_affine_commutes_with_linear_latent_filtering_before_normal_z() -> None:
    torch.manual_seed(31)
    decoder = SafeAffineMaterialDecoder(margin=1.0e-3).double()
    with torch.no_grad():
        for parameter in decoder.parameters():
            parameter.normal_()
    weight, bias = decoder.fold_affine()
    latent_corners = torch.rand(6, 4, 4, dtype=torch.float64)
    filter_weights = torch.rand(6, 4, dtype=torch.float64)
    filter_weights /= filter_weights.sum(dim=-1, keepdim=True)

    filtered = decode_filtered_affine_material(
        latent_corners, filter_weights, weight, bias
    )
    filtered_seven = torch.cat(
        (
            filtered.base_color_linear,
            filtered.normal_xy,
            filtered.roughness,
            filtered.metallic,
        ),
        dim=-1,
    )
    corner_seven = F.linear(latent_corners, weight, bias)
    material_filtered = torch.sum(
        corner_seven * filter_weights[..., None], dim=-2
    )

    assert torch.allclose(
        filtered_seven, material_filtered, atol=1.0e-12, rtol=1.0e-12
    )


def test_affine_export_reload_preserves_fp32_payload_semantics_and_cost() -> None:
    torch.manual_seed(41)
    decoder = SafeAffineMaterialDecoder(margin=1.0e-3).double()
    with torch.no_grad():
        for parameter in decoder.parameters():
            parameter.normal_()
    expected_weight, expected_bias = decoder.fold_affine()

    artifact = export_affine_decoder(decoder)
    weight, bias = reload_affine_decoder(artifact)

    assert len(artifact.payload) == 140
    assert torch.equal(weight, expected_weight.float())
    assert torch.equal(bias, expected_bias.float())
    assert artifact.manifest["semantics"] == [
        "base_color_linear_r",
        "base_color_linear_g",
        "base_color_linear_b",
        "normal_tangent_x",
        "normal_tangent_y",
        "roughness_linear",
        "metallic_linear",
    ]
    assert artifact.manifest["certificate"]["valid"] is True
    assert artifact.manifest["cost"] == {
        "parameters": 35,
        "weight_bytes_fp32": 140,
        "decoder_macs_per_pixel": 28,
        "texture_resources": 1,
        "filtered_samples_per_pixel": 1,
    }

    corrupt = type(artifact)(
        payload=artifact.payload[:-1] + bytes([artifact.payload[-1] ^ 1]),
        manifest=artifact.manifest,
    )
    with pytest.raises(ValueError, match="SHA-256"):
        reload_affine_decoder(corrupt)
