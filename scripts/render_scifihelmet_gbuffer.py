"""Render and export the fixed SciFiHelmet Stage-B GBuffer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cg_frontier.assets.gltf_mesh import load_gltf_mesh  # noqa: E402
from cg_frontier.render.gbuffer import (  # noqa: E402
    Camera,
    export_gbuffer,
    load_core4_textures,
    render_gbuffer,
)


DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "render" / "scifihelmet_gbuffer.yaml"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render fixed-camera SciFiHelmet GBuffer diagnostics."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"config field {label} must be an object")
    return value


def _repo_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"config field {label} must be a non-empty path")
    path = (REPOSITORY_ROOT / value).resolve()
    if not path.is_relative_to(REPOSITORY_ROOT):
        raise ValueError(f"config field {label} escapes the repository")
    return path


def main() -> int:
    """Validate the frozen render contract, then export one deterministic GBuffer."""

    args = _arguments()
    try:
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        if not isinstance(config, Mapping) or config.get("schema_version") != 1:
            raise ValueError("unsupported GBuffer config schema")
        asset = _mapping(config.get("asset"), "asset")
        render = _mapping(config.get("render"), "render")
        camera_config = _mapping(render.get("camera"), "render.camera")
        resolution_value = render.get("resolution")
        if (
            not isinstance(resolution_value, list)
            or len(resolution_value) != 2
            or not all(isinstance(value, int) for value in resolution_value)
        ):
            raise ValueError("render.resolution must contain two integers")
        if render.get("coordinate_system") != "glTF_right_handed_y_up_front_plus_z":
            raise ValueError("Stage B only supports the frozen native glTF coordinate system")
        if render.get("tangent_basis") != "reconstructed_uv":
            raise ValueError("Stage B requires reconstructed_uv because source tangents are invalid")
        if render.get("normal_y_flip") is not False:
            raise ValueError("primary Stage-B normal path must not flip Y")
        if render.get("texture_filter") != "linear":
            raise ValueError("Stage B currently supports linear texture filtering only")
        camera = Camera(
            eye=tuple(float(value) for value in camera_config["eye"]),
            target=tuple(float(value) for value in camera_config["target"]),
            up=tuple(float(value) for value in camera_config["up"]),
            vertical_fov_degrees=float(camera_config["vertical_fov_degrees"]),
            near=float(camera_config["near"]),
            far=float(camera_config["far"]),
        )
        gltf_path = _repo_path(asset.get("gltf"), "asset.gltf")
        manifest_path = _repo_path(
            asset.get("core4_manifest"), "asset.core4_manifest"
        )
        output_dir = (
            args.output_dir.resolve()
            if args.output_dir is not None
            else _repo_path(config.get("output_dir"), "output_dir")
        )
        mesh = load_gltf_mesh(gltf_path)
        textures = load_core4_textures(manifest_path, "cuda")
        result = render_gbuffer(
            mesh,
            textures,
            camera,
            (resolution_value[0], resolution_value[1]),
            device="cuda",
            cull_backfaces=bool(render.get("backface_culling", True)),
        )
        metadata = export_gbuffer(result, output_dir, mesh)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        print(f"GBuffer render failed: {error}", file=sys.stderr)
        return 2

    coverage = metadata["coverage"]
    depth = metadata["valid_pixel_statistics"]["depth_camera"]
    tangent = metadata["tangent_basis"]
    print(f"output: {output_dir}")
    print(
        f"mesh: {metadata['mesh']['vertex_count']} vertices, "
        f"{metadata['mesh']['triangle_count']} triangles, "
        f"{metadata['mesh']['rendered_triangle_count']} after culling"
    )
    print(f"coverage: {coverage['pixels']} pixels ({coverage['ratio']:.6f})")
    print(f"depth: [{depth['min']:.6f}, {depth['max']:.6f}] meters")
    print(
        "source/rebuilt max |N·T|: "
        f"{tangent['source_abs_normal_dot_tangent']['max']:.6f} / "
        f"{tangent['reconstructed_abs_normal_dot_tangent']['max']:.6e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
