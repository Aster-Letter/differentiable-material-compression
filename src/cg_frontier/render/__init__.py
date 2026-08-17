"""Minimal differentiable rendering building blocks."""

from .gbuffer import (
    Camera,
    Core4Textures,
    GBufferResult,
    export_gbuffer,
    load_core4_textures,
    look_at_matrix,
    orthonormal_tangent_frame,
    perspective_matrix,
    render_gbuffer,
    srgb_to_linear_torch,
)
from .pbr import (
    PointLight,
    ReferenceResult,
    export_reference,
    render_reference_variants,
    shade_ggx,
)

__all__ = [
    "Camera",
    "Core4Textures",
    "GBufferResult",
    "export_gbuffer",
    "load_core4_textures",
    "look_at_matrix",
    "orthonormal_tangent_frame",
    "perspective_matrix",
    "render_gbuffer",
    "srgb_to_linear_torch",
    "PointLight",
    "ReferenceResult",
    "export_reference",
    "render_reference_variants",
    "shade_ggx",
]
