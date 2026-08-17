"""Minimal GGX contracts and optional real Core-4 texture gradient test."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cg_frontier.assets.gltf_mesh import load_gltf_mesh  # noqa: E402
from cg_frontier.render.gbuffer import (  # noqa: E402
    Camera,
    Core4Textures,
    GBufferResult,
    MaterialBuffers,
    material_from_gbuffer,
    load_core4_textures,
    render_gbuffer,
    srgb_to_linear_torch,
)
from cg_frontier.render.pbr import PointLight, render_reference_variants, shade_ggx  # noqa: E402


REAL_GLTF = (
    REPOSITORY_ROOT
    / "assets"
    / "source"
    / "glTF-Sample-Assets"
    / "Models"
    / "SciFiHelmet"
    / "glTF"
    / "SciFiHelmet.gltf"
)
CORE4_MANIFEST = REPOSITORY_ROOT / "configs" / "assets" / "scifihelmet_core4.yaml"
PROCESSED_CORE4 = REPOSITORY_ROOT / "assets" / "processed" / "SciFiHelmet" / "core4"


def _camera() -> Camera:
    return Camera(
        eye=(0.0, 0.0, 4.5),
        target=(0.0, 0.0, 0.0),
        up=(0.0, 1.0, 0.0),
        vertical_fov_degrees=45.0,
        near=0.1,
        far=10.0,
    )


def _light() -> PointLight:
    return PointLight(
        position=(2.5, 3.0, 4.0),
        color=(1.0, 0.98, 0.95),
        radiant_intensity=90.0,
        ambient_intensity=0.04,
    )


def test_torch_srgb_decode_anchor_points() -> None:
    encoded = torch.tensor([0.0, 0.04045, 0.5, 1.0], dtype=torch.float64)
    decoded = srgb_to_linear_torch(encoded)
    expected = np.array(
        [0.0, 0.04045 / 12.92, ((0.5 + 0.055) / 1.055) ** 2.4, 1.0]
    )
    np.testing.assert_allclose(decoded.numpy(), expected, rtol=0.0, atol=1e-12)


def test_ggx_single_pixel_is_finite_and_respects_mask() -> None:
    mask = torch.tensor([[True, False]])
    vector = lambda values: torch.tensor([[values, values]], dtype=torch.float32)
    scalar = lambda value: torch.tensor([[value, value]], dtype=torch.float32)
    buffers = {
        "mask": mask,
        "position_world": vector([0.0, 0.0, 0.0]),
        "normal_world": vector([0.0, 0.0, 1.0]),
        "normal_world_y_flipped": vector([0.0, 0.0, 1.0]),
        "base_color_linear": vector([0.5, 0.25, 0.125]),
        "roughness": scalar(0.5),
        "metallic": scalar(0.0),
    }
    gbuffer = GBufferResult(buffers={}, torch_buffers=buffers, metadata={})
    image = shade_ggx(gbuffer, _camera(), _light())
    assert torch.isfinite(image).all()
    assert torch.all(image[0, 0] > 0.0)
    assert torch.all(image[0, 1] == 0.0)


def test_complete_material_override_is_strictly_equivalent_to_legacy_buffers() -> None:
    mask = torch.tensor([[True, False]])
    vector = lambda values: torch.tensor([[values, values]], dtype=torch.float32)
    scalar = lambda value: torch.tensor([[value, value]], dtype=torch.float32)
    buffers = {
        "mask": mask,
        "position_world": vector([0.0, 0.0, 0.0]),
        "normal_world": vector([0.0, 0.0, 1.0]),
        "base_color_linear": vector([0.5, 0.25, 0.125]),
        "roughness": scalar(0.5),
        "metallic": scalar(0.2),
    }
    gbuffer = GBufferResult(buffers={}, torch_buffers=buffers, metadata={})
    legacy = shade_ggx(gbuffer, _camera(), _light())
    explicit = shade_ggx(
        gbuffer,
        _camera(),
        _light(),
        material_override=material_from_gbuffer(gbuffer),
    )
    torch.testing.assert_close(explicit, legacy, rtol=0.0, atol=0.0)


def test_base_color_material_override_is_already_linear() -> None:
    mask = torch.tensor([[True]])
    buffers = {
        "mask": mask,
        "position_world": torch.zeros((1, 1, 3)),
    }
    gbuffer = GBufferResult(buffers={}, torch_buffers=buffers, metadata={})
    base = torch.tensor([[[0.5, 0.25, 0.125]]])
    material = MaterialBuffers(
        base_color_linear=base,
        normal_world=torch.tensor([[[0.0, 0.0, 1.0]]]),
        roughness=torch.tensor([[0.5]]),
        metallic=torch.tensor([[0.0]]),
    )
    image = shade_ggx(gbuffer, _camera(), _light(), material_override=material)
    assert torch.isfinite(image).all()
    assert material.base_color_linear.data_ptr() == base.data_ptr()


REAL_PBR_AVAILABLE = (
    torch.cuda.is_available()
    and REAL_GLTF.is_file()
    and CORE4_MANIFEST.is_file()
    and all(
        (PROCESSED_CORE4 / name).is_file()
        for name in ("base_color.png", "normal.png", "roughness.png", "metallic.png")
    )
)


@pytest.mark.skipif(
    not REAL_PBR_AVAILABLE,
    reason="CUDA or ignored SciFiHelmet source/processed assets are absent",
)
def test_real_core4_textures_receive_finite_nonzero_render_gradients() -> None:
    mesh = load_gltf_mesh(REAL_GLTF)
    loaded = load_core4_textures(CORE4_MANIFEST, "cuda")
    textures = Core4Textures(
        base_color_linear=loaded.base_color_linear.detach().clone().requires_grad_(True),
        normal=loaded.normal.detach().clone().requires_grad_(True),
        roughness=loaded.roughness.detach().clone().requires_grad_(True),
        metallic=loaded.metallic.detach().clone().requires_grad_(True),
        source_hashes=loaded.source_hashes,
    )
    gbuffer = render_gbuffer(mesh, textures, _camera(), (64, 64), device="cuda")
    variants = render_reference_variants(gbuffer, _camera(), _light())
    image = variants.images["reference_linear"]
    mask = gbuffer.torch_buffers["mask"]
    loss = image[mask].square().mean()
    loss.backward()

    for name in ("base_color_linear", "normal", "roughness", "metallic"):
        tensor = getattr(textures, name)
        assert tensor.grad is not None, name
        assert torch.isfinite(tensor.grad).all(), name
        assert tensor.grad.abs().sum().item() > 0.0, name
