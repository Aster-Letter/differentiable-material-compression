"""Stage-B camera/TBN contracts and optional real CUDA GBuffer integration."""

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
    load_core4_textures,
    look_at_matrix,
    orthonormal_tangent_frame,
    perspective_matrix,
    render_gbuffer,
)


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


def test_look_at_and_projection_follow_opengl_camera_convention() -> None:
    camera = _camera()
    view = look_at_matrix(camera)
    eye = torch.tensor([[0.0, 0.0, 4.5, 1.0]])
    target = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
    np.testing.assert_allclose((eye @ view.T).numpy(), [[0.0, 0.0, 0.0, 1.0]], atol=1e-7)
    np.testing.assert_allclose(
        (target @ view.T).numpy(), [[0.0, 0.0, -4.5, 1.0]], atol=1e-7
    )
    projection = perspective_matrix(90.0, 1.0, 0.1, 10.0)
    assert projection[0, 0].item() == pytest.approx(1.0)
    assert projection[1, 1].item() == pytest.approx(1.0)
    assert projection[3, 2].item() == -1.0


def test_tangent_frame_uses_gltf_handedness_and_gram_schmidt() -> None:
    normal = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    tangent = torch.tensor([[2.0, 0.0, 0.5, 1.0], [2.0, 0.0, -0.5, -1.0]])
    normal_unit, tangent_unit, bitangent_unit = orthonormal_tangent_frame(normal, tangent)
    np.testing.assert_allclose(normal_unit.numpy(), [[0, 0, 1], [0, 0, 1]], atol=1e-7)
    np.testing.assert_allclose(tangent_unit.numpy(), [[1, 0, 0], [1, 0, 0]], atol=1e-7)
    np.testing.assert_allclose(bitangent_unit.numpy(), [[0, 1, 0], [0, -1, 0]], atol=1e-7)
    assert torch.max(torch.abs(torch.sum(normal_unit * tangent_unit, dim=1))).item() < 1e-7


REAL_GBUFFER_AVAILABLE = (
    torch.cuda.is_available()
    and REAL_GLTF.is_file()
    and CORE4_MANIFEST.is_file()
    and all(
        (PROCESSED_CORE4 / name).is_file()
        for name in ("base_color.png", "normal.png", "roughness.png", "metallic.png")
    )
)


@pytest.mark.skipif(
    not REAL_GBUFFER_AVAILABLE,
    reason="CUDA or ignored SciFiHelmet source/processed assets are absent",
)
def test_real_scifihelmet_gbuffer_contract() -> None:
    mesh = load_gltf_mesh(REAL_GLTF)
    textures = load_core4_textures(CORE4_MANIFEST, "cuda")
    result = render_gbuffer(mesh, textures, _camera(), (128, 128), device="cuda")
    buffers = result.buffers
    mask = buffers["mask"]
    assert mask.shape == (128, 128)
    assert 0.1 < float(mask.mean()) < 0.9
    assert np.isfinite(buffers["position_world"]).all()
    assert np.all(buffers["depth_camera"][mask] > 0.0)
    assert np.all((buffers["roughness"][mask] >= 0.0) & (buffers["roughness"][mask] <= 1.0))
    assert np.all((buffers["metallic"][mask] >= 0.0) & (buffers["metallic"][mask] <= 1.0))
    for name in (
        "vertex_normal_world",
        "tangent_world",
        "bitangent_world",
        "normal_ts_unit",
        "normal_world",
        "normal_world_y_flipped",
    ):
        lengths = np.linalg.norm(buffers[name][mask], axis=1)
        np.testing.assert_allclose(lengths, 1.0, rtol=0.0, atol=2e-5)
    pixel_dots = result.metadata["tangent_basis"]["pixel_max_abs_dot"]
    assert max(pixel_dots.values()) < 2e-5
    angles = buffers["normal_y_flip_angle_degrees"][mask]
    assert float(angles.mean()) > 1.0
