"""Minimal CUDA smoke test for the differentiable rasterization toolchain."""

import pytest
import torch


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA GPU is required for nvdiffrast"
)


def test_raster_interpolate_texture_backward() -> None:
    import nvdiffrast.torch as dr

    device = torch.device("cuda")
    context = dr.RasterizeCudaContext(device=device)

    clip_positions = torch.tensor(
        [[[-0.8, -0.8, 0.0, 1.0], [0.8, -0.8, 0.0, 1.0], [0.0, 0.8, 0.0, 1.0]]],
        dtype=torch.float32,
        device=device,
    )
    triangles = torch.tensor([[0, 1, 2]], dtype=torch.int32, device=device)
    vertex_uv = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [0.5, 1.0]]],
        dtype=torch.float32,
        device=device,
    )
    texture = torch.rand(
        [1, 8, 8, 3], dtype=torch.float32, device=device, requires_grad=True
    )

    raster, _ = dr.rasterize(context, clip_positions, triangles, resolution=[32, 32])
    interpolated_uv, _ = dr.interpolate(vertex_uv, raster, triangles)
    image = dr.texture(texture, interpolated_uv, filter_mode="linear")
    covered = raster[..., 3:] > 0
    loss = image.square().mul(covered).sum() / covered.sum().clamp_min(1)
    loss.backward()

    assert covered.any()
    assert texture.grad is not None
    assert torch.isfinite(texture.grad).all()
    assert texture.grad.abs().sum() > 0
