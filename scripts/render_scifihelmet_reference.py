"""Render fixed SciFiHelmet GGX reference and error-control images."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cg_frontier.assets.gltf_mesh import load_gltf_mesh  # noqa: E402
from cg_frontier.assets.preprocess import sha256_file  # noqa: E402
from cg_frontier.render.gbuffer import (  # noqa: E402
    Camera,
    load_core4_textures,
    render_geometry_gbuffer,
    sample_core4_material,
)
from cg_frontier.render.pbr import (  # noqa: E402
    PointLight,
    export_reference,
    render_reference_variants,
)


DEFAULT_CONFIG = REPOSITORY_ROOT / "configs" / "render" / "scifihelmet_reference.yaml"


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the fixed SciFiHelmet Cook-Torrance GGX reference."
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


def _triple(values: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(values, list) or len(values) != 3:
        raise ValueError(f"config field {label} must contain three numbers")
    return tuple(float(value) for value in values)


def main() -> int:
    """Render configured views while keeping geometry, light, and conventions fixed."""

    args = _arguments()
    try:
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        if not isinstance(config, Mapping) or config.get("schema_version") != 1:
            raise ValueError("unsupported reference config schema")
        asset = _mapping(config.get("asset"), "asset")
        render = _mapping(config.get("render"), "render")
        light_config = _mapping(config.get("light"), "light")
        brdf_config = _mapping(config.get("brdf"), "brdf")
        display_config = _mapping(config.get("display"), "display")
        resolution = render.get("resolution")
        if (
            not isinstance(resolution, list)
            or len(resolution) != 2
            or not all(isinstance(value, int) for value in resolution)
        ):
            raise ValueError("render.resolution must contain two integers")
        if render.get("coordinate_system") != "glTF_right_handed_y_up_front_plus_z":
            raise ValueError("reference renderer only supports the frozen glTF coordinates")
        if render.get("tangent_basis") != "reconstructed_uv":
            raise ValueError("reference renderer requires reconstructed_uv tangents")
        if render.get("normal_y_flip") is not False:
            raise ValueError("reference primary branch must keep glTF normal +Y")
        if light_config.get("type") != "point":
            raise ValueError("minimal reference renderer currently supports one point light")
        if brdf_config.get("model") != "cook_torrance_ggx":
            raise ValueError("unsupported BRDF model")
        if display_config.get("tone_mapping") != "reinhard":
            raise ValueError("unsupported display tone mapping")
        if display_config.get("encoding") != "sRGB":
            raise ValueError("display output must use sRGB encoding")

        views = render.get("views")
        if not isinstance(views, list) or not views:
            raise ValueError("render.views must contain at least one camera")
        light = PointLight(
            position=_triple(light_config["position"], "light.position"),
            color=_triple(light_config["color"], "light.color"),
            radiant_intensity=float(light_config["radiant_intensity"]),
            ambient_intensity=float(light_config["ambient_intensity"]),
        )
        output_dir = (
            args.output_dir.resolve()
            if args.output_dir is not None
            else _repo_path(config.get("output_dir"), "output_dir")
        )
        mesh = load_gltf_mesh(_repo_path(asset.get("gltf"), "asset.gltf"))
        textures = load_core4_textures(
            _repo_path(asset.get("core4_manifest"), "asset.core4_manifest"), "cuda"
        )
        view_results: list[dict[str, Any]] = []
        seen_names: set[str] = set()
        for view_value in views:
            view_config = _mapping(view_value, "render.views[]")
            view_name = view_config.get("name")
            if (
                not isinstance(view_name, str)
                or re.fullmatch(r"[a-z0-9_]+", view_name) is None
                or view_name in seen_names
            ):
                raise ValueError("view names must be unique lowercase identifiers")
            seen_names.add(view_name)
            camera = Camera(
                eye=_triple(view_config["eye"], f"render.views.{view_name}.eye"),
                target=_triple(view_config["target"], f"render.views.{view_name}.target"),
                up=_triple(view_config["up"], f"render.views.{view_name}.up"),
                vertical_fov_degrees=float(view_config["vertical_fov_degrees"]),
                near=float(view_config["near"]),
                far=float(view_config["far"]),
            )
            gbuffer = render_geometry_gbuffer(
                mesh,
                camera,
                (resolution[0], resolution[1]),
                device="cuda",
                cull_backfaces=bool(render.get("backface_culling", True)),
            )
            reference = render_reference_variants(
                gbuffer,
                camera,
                light,
                material=sample_core4_material(gbuffer, textures),
                minimum_roughness=float(brdf_config["minimum_roughness"]),
            )
            view_dir = output_dir / view_name
            metadata = export_reference(
                reference,
                view_dir,
                display_exposure=float(display_config["exposure"]),
            )
            view_manifest = view_dir / "reference.yaml"
            view_results.append(
                {
                    "name": view_name,
                    "directory": view_name,
                    "eye": list(camera.eye),
                    "reference_manifest_sha256": sha256_file(view_manifest),
                    "variant_error": metadata["variant_error"],
                }
            )
        set_metadata = {
            "schema_version": 1,
            "coordinate_system": render["coordinate_system"],
            "resolution": resolution,
            "shared_light": {
                "type": "point",
                "position": list(light.position),
                "color": list(light.color),
                "radiant_intensity": light.radiant_intensity,
                "ambient_intensity": light.ambient_intensity,
            },
            "views": view_results,
        }
        output_dir.mkdir(parents=True, exist_ok=True)
        set_text = yaml.safe_dump(
            set_metadata,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )
        with (output_dir / "reference_set.yaml").open(
            "w", encoding="utf-8", newline="\n"
        ) as stream:
            stream.write(set_text)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        print(f"reference render failed: {error}", file=sys.stderr)
        return 2

    print(f"output: {output_dir}")
    for view in view_results:
        print(f"view: {view['name']} eye={view['eye']}")
        for name, statistics in view["variant_error"].items():
            print(
                f"  {name}: MAE={statistics['mae_linear']:.6f}, "
                f"RMSE={statistics['rmse_linear']:.6f}, "
                f"max={statistics['max_abs_linear']:.6f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
