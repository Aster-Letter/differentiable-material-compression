"""Frozen six-face canonical cube used by the L2 auxiliary objective."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class CanonicalCube:
    face_names: tuple[str, ...]
    uv: torch.Tensor
    positions: torch.Tensor
    tangents: torch.Tensor
    bitangents: torch.Tensor
    normals: torch.Tensor
    camera_positions: torch.Tensor
    camera_targets: torch.Tensor
    camera_up: torch.Tensor


@dataclass(frozen=True)
class CubeAtlasSamples:
    material: torch.Tensor
    valid: torch.Tensor


@dataclass(frozen=True)
class MaskedCubeLoss:
    loss: torch.Tensor
    valid_pixel_count: int


def make_cube_rng(*, seed: int, device: torch.device | str = "cpu") -> torch.Generator:
    """Create the cube-only RNG stream, independent of all core samplers."""

    return torch.Generator(device=device).manual_seed(seed)


def cube_tangent_normal_to_world(
    cube: CanonicalCube, normal_tangent: torch.Tensor
) -> torch.Tensor:
    """Apply the frozen glTF-space TBN without an engine-boundary Y flip."""

    return (
        cube.tangents * normal_tangent[..., 0]
        + cube.bitangents * normal_tangent[..., 1]
        + cube.normals * normal_tangent[..., 2]
    )


def sample_cube_atlas(
    material_atlas: torch.Tensor,
    valid_atlas: torch.Tensor,
    cube: CanonicalCube,
) -> CubeAtlasSamples:
    """Sample material and mask with the exact same face-local UV grid."""

    face_count = len(cube.face_names)
    material_nchw = material_atlas.permute(2, 0, 1).unsqueeze(0).expand(
        face_count, -1, -1, -1
    )
    valid_nchw = valid_atlas.to(material_atlas.dtype)[None, None].expand(
        face_count, -1, -1, -1
    )
    grid = cube.uv * 2.0 - 1.0
    sampled_material = F.grid_sample(
        material_nchw,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).permute(0, 2, 3, 1)
    sampled_valid = F.grid_sample(
        valid_nchw,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[:, 0]
    return CubeAtlasSamples(
        material=sampled_material,
        valid=sampled_valid >= 1.0 - 1.0e-6,
    )


def masked_cube_l1(
    prediction: torch.Tensor,
    reference: torch.Tensor,
    valid: torch.Tensor,
) -> MaskedCubeLoss:
    """Normalize cube material error by accepted screen pixels."""

    per_pixel = torch.mean(torch.abs(prediction - reference), dim=-1)
    valid_pixel_count = int(valid.sum())
    if valid_pixel_count == 0:
        loss = prediction.sum() * 0.0
    else:
        loss = per_pixel[valid].sum() / valid_pixel_count
    return MaskedCubeLoss(loss=loss, valid_pixel_count=valid_pixel_count)


def build_canonical_cube(
    *,
    resolution: int,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str = "cpu",
) -> CanonicalCube:
    """Build six explicitly oriented faces with full face-local UV coverage."""

    if resolution < 2:
        raise ValueError("cube face resolution must be at least 2")
    face_names = ("+X", "-X", "+Y", "-Y", "+Z", "-Z")
    normals = torch.tensor(
        ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)),
        dtype=dtype,
        device=device,
    )
    tangents = torch.tensor(
        ((0, 0, -1), (0, 0, 1), (1, 0, 0), (1, 0, 0), (1, 0, 0), (-1, 0, 0)),
        dtype=dtype,
        device=device,
    )
    bitangents = torch.tensor(
        ((0, 1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1), (0, 1, 0), (0, 1, 0)),
        dtype=dtype,
        device=device,
    )
    coordinate = torch.linspace(0.0, 1.0, resolution, dtype=dtype, device=device)
    v, u = torch.meshgrid(coordinate, coordinate, indexing="ij")
    face_uv = torch.stack((u, v), dim=-1)
    uv = face_uv.unsqueeze(0).expand(6, -1, -1, -1).clone()
    positions = (
        0.5 * normals[:, None, None, :]
        + (uv[..., 0:1] - 0.5) * tangents[:, None, None, :]
        + (uv[..., 1:2] - 0.5) * bitangents[:, None, None, :]
    )
    return CanonicalCube(
        face_names=face_names,
        uv=uv,
        positions=positions,
        tangents=tangents,
        bitangents=bitangents,
        normals=normals,
        camera_positions=2.0 * normals,
        camera_targets=torch.zeros_like(normals),
        camera_up=bitangents.clone(),
    )
