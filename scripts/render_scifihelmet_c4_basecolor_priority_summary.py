"""Render technical and anonymous Gate figures for C4 BaseColor candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for directory in (SRC, SCRIPTS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from cg_frontier.compression.affine_material import AffineDecodedMaterial  # noqa: E402
from cg_frontier.compression.basecolor_priority import postprocess_affine_output  # noqa: E402
from cg_frontier.compression.render_loss import (  # noqa: E402
    bilinear_sample_top_down_wrap,
    decoded_to_material,
    hard_quantize_unorm8,
    orbit_camera,
)
from cg_frontier.render.gbuffer import (  # noqa: E402
    Core4Textures,
    render_geometry_gbuffer,
    sample_core4_material,
)
from cg_frontier.render.pbr import linear_to_srgb_torch, shade_ggx  # noqa: E402
from train_scifihelmet_c4_basecolor_priority_10k import _load_config, _prepare  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/train/scifihelmet_c4_basecolor_priority_10k_v1.yaml"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> str:
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n", encoding="ascii")
    return digest


def _parse_candidate(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError("candidate input must use ID=output-root")
    candidate_id, raw_path = value.split("=", 1)
    path = Path(raw_path).resolve()
    if not candidate_id or not path.is_dir() or not path.is_relative_to(ROOT):
        raise ValueError("candidate root must exist inside the repository")
    return candidate_id, path


def _repo_output_path(value: Path) -> Path:
    path = value.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("summary output must remain inside the repository")
    return path


def _visual_views(config: Mapping[str, Any]) -> list[tuple[str, int | None, int, bool, float, float]]:
    rows = config.get("visual_views")
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError("visual_views must contain the frozen four-view set")
    result = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("visual view entries must be mappings")
        view_id = str(row.get("id", ""))
        camera_index = row.get("camera_index")
        result.append(
            (
                view_id,
                None if camera_index is None else int(camera_index),
                int(row.get("light_index", 2)),
                bool(row.get("selection_metric", True)),
                float(row.get("yaw_degrees", 0.0)),
                float(row.get("elevation_degrees", 78.0)),
            )
        )
    if tuple(value[0] for value in result) != (
        "front",
        "upper_side",
        "rear_pipeline",
        "top",
    ):
        raise ValueError("visual view IDs or order changed")
    if result[2][3] or result[3][3]:
        raise ValueError("rear and top views must remain presentation-only")
    return result


def _load_candidate(candidate_id: str, root: Path) -> tuple[dict[str, Any], dict[int, tuple]]:
    report = json.loads((root / "training_report.json").read_text(encoding="utf-8"))
    if report.get("candidate", {}).get("candidate_id") != candidate_id:
        raise ValueError("candidate report identity mismatch")
    states = {}
    for row in report["trajectory"]:
        step = int(row["step"])
        path = root / row["checkpoint"]
        if _sha(path) != row["checkpoint_sha256"]:
            raise ValueError(f"candidate checkpoint SHA mismatch: {candidate_id}@{step}")
        payload = torch.load(path, map_location="cuda", weights_only=False)
        if payload.get("checkpoint_type") not in {
            "c4_basecolor_priority_v3",
            "c4_basecolor_dynamic_constraint_v1",
        }:
            raise ValueError("unexpected checkpoint type")
        states[step] = (
            payload["latent"].cuda(),
            payload["weight"].cuda(),
            payload["bias"].cuda(),
            payload["compander_parameters"].cuda(),
            bool(report["candidate"]["compander"]),
        )
    required = {1000, 5000, 10000}
    if not required.issubset(states):
        raise ValueError(f"candidate lacks 1k/5k/10k checkpoints: {candidate_id}")
    return report, states


def _material(geometry, processed):
    return decoded_to_material(
        geometry,
        AffineDecodedMaterial(
            base_color_linear=processed.seven[..., :3],
            normal_xy=processed.seven[..., 3:5],
            normal_xyz=processed.normal_xyz,
            roughness=processed.seven[..., 5:6],
            metallic=processed.seven[..., 6:7],
        ),
    )


def _render_state(state, geometry, camera, light, render):
    latent, weight, bias, compander, enabled = state
    sampled = bilinear_sample_top_down_wrap(
        hard_quantize_unorm8(latent), geometry.torch_buffers["uv"]
    )
    processed = postprocess_affine_output(
        F.linear(sampled, weight, bias),
        compander_parameters=compander if enabled else None,
        straight_through=False,
    )
    return shade_ggx(
        geometry,
        camera,
        light,
        material_override=_material(geometry, processed),
        minimum_roughness=float(render["minimum_roughness"]),
    )


def _display(linear, exposure):
    mapped = (linear.clamp_min(0.0) * exposure)
    mapped = mapped / (1.0 + mapped)
    return np.rint(
        linear_to_srgb_torch(mapped).clamp(0.0, 1.0).cpu().numpy() * 255.0
    ).astype(np.uint8)


def _panel(images: list[np.ndarray], labels: list[str], title: str) -> Image.Image:
    height, width = images[0].shape[:2]
    header = 58
    result = Image.new("RGB", (width * len(images), height + header), (22, 22, 25))
    draw = ImageDraw.Draw(result)
    draw.text((10, 8), title, fill=(245, 245, 245))
    for index, (image, label) in enumerate(zip(images, labels, strict=True)):
        result.paste(Image.fromarray(image), (index * width, header))
        draw.text((index * width + 10, 31), label, fill=(220, 220, 220))
    return result


def _stack(panels: list[Image.Image]) -> Image.Image:
    width = max(panel.width for panel in panels)
    result = Image.new("RGB", (width, sum(panel.height for panel in panels)), (22, 22, 25))
    offset = 0
    for panel in panels:
        result.paste(panel, (0, offset))
        offset += panel.height
    return result


def _save(image: Image.Image, path: Path) -> dict[str, Any]:
    image.save(path, format="PNG")
    return {"path": path.name, "sha256": _sha(path), "size": list(image.size)}


def _atlas_image(rgb: torch.Tensor) -> np.ndarray:
    encoded = linear_to_srgb_torch(rgb.clamp(0.0, 1.0)).cpu().numpy()
    return np.rint(encoded * 255.0).astype(np.uint8)


def _error_image(prediction: torch.Tensor, source: torch.Tensor) -> np.ndarray:
    error = (prediction - source).abs().mean(dim=-1).clamp(0.0, 0.25) / 0.25
    red = error
    green = torch.clamp(1.5 * error - 0.25, 0.0, 1.0)
    blue = torch.clamp(1.0 - 2.0 * error, 0.0, 1.0)
    return np.rint(torch.stack((red, green, blue), dim=-1).cpu().numpy() * 255.0).astype(np.uint8)


def run(config_path: Path, candidate_values: list[str], output: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("summary rendering requires CUDA")
    output = _repo_output_path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite summary root: {output}")
    output.mkdir(parents=True)
    config = _load_config(config_path)
    prepared = _prepare(config, config_path)
    parsed = [_parse_candidate(value) for value in candidate_values]
    candidates = dict(parsed)
    if not candidates or len(candidates) != len(parsed):
        raise ValueError("at least one uniquely named candidate is required")
    loaded = {name: _load_candidate(name, root) for name, root in candidates.items()}
    raw = prepared["raw"]
    parent = (
        raw.latent_unorm8.cuda().to(torch.float32) / 255.0,
        raw.weight.cuda(),
        raw.bias.cuda(),
        torch.tensor((1.0, 0.0), device="cuda"),
        False,
    )
    source_textures = prepared["textures"]
    direct_textures = Core4Textures(
        base_color_linear=(
            torch.floor(source_textures.base_color_linear * 255.0 + 0.5) / 255.0
        ).contiguous(),
        normal=source_textures.normal,
        roughness=source_textures.roughness,
        metallic=source_textures.metallic,
        source_hashes=source_textures.source_hashes,
    )
    render = prepared["render"]
    views = _visual_views(config)

    def view_data(camera_index, light_index, yaw_degrees, elevation_degrees):
        if camera_index is not None:
            return (
                prepared["geometries"][camera_index],
                prepared["cameras"][camera_index],
                prepared["lights"][light_index],
                prepared["references"][light_index][camera_index],
            )
        camera = orbit_camera(
            yaw_degrees=yaw_degrees,
            elevation_degrees=elevation_degrees,
            radius=float(render["camera_radius"]),
            target=tuple(float(value) for value in render["target"]),
            up=tuple(float(value) for value in render["up"]),
            vertical_fov_degrees=float(render["vertical_fov_degrees"]),
            near=float(render["near"]),
            far=float(render["far"]),
        )
        geometry = render_geometry_gbuffer(
            prepared["mesh"], camera, (256, 256), device="cuda"
        )
        light = prepared["lights"][light_index]
        source = shade_ggx(
            geometry,
            camera,
            light,
            material_override=sample_core4_material(geometry, source_textures),
            minimum_roughness=float(render["minimum_roughness"]),
        )
        return geometry, camera, light, source

    figures = []
    exposure = float(render["display_exposure"])
    for candidate_id, (_, states) in loaded.items():
        panels = []
        for view_name, camera_index, light_index, selection_metric, yaw, elevation in views:
            geometry, camera, light, source = view_data(
                camera_index, light_index, yaw, elevation
            )
            images = [_display(source, exposure), _display(_render_state(parent, geometry, camera, light, render), exposure)]
            images.extend(
                _display(_render_state(states[step], geometry, camera, light, render), exposure)
                for step in (1000, 5000, 10000)
            )
            panels.append(
                _panel(
                    images,
                    ["Source", "Raw PCA", "1k", "5k", "10k"],
                    f"{candidate_id} | {view_name} | selection_metric={str(selection_metric).lower()}",
                )
            )
        figures.append(_save(_stack(panels), output / f"trajectory_{candidate_id}.png"))

    endpoint_labels = ["Source", "Raw PCA", "Direct BaseColor"] + list(loaded)
    endpoint_panels = []
    anonymous_panels = []
    aliases = {name: chr(ord("A") + index) for index, name in enumerate(sorted(loaded))}
    for view_name, camera_index, light_index, selection_metric, yaw, elevation in views:
        geometry, camera, light, source = view_data(
            camera_index, light_index, yaw, elevation
        )
        direct = shade_ggx(
            geometry,
            camera,
            light,
            material_override=sample_core4_material(geometry, direct_textures),
            minimum_roughness=float(render["minimum_roughness"]),
        )
        images = [
            _display(source, exposure),
            _display(_render_state(parent, geometry, camera, light, render), exposure),
            _display(direct, exposure),
        ] + [
            _display(_render_state(states[10000], geometry, camera, light, render), exposure)
            for _, states in loaded.values()
        ]
        endpoint_panels.append(
            _panel(images, endpoint_labels, f"{view_name} | selection_metric={str(selection_metric).lower()}")
        )
        anonymous_panels.append(
            _panel(
                images,
                ["Reference", "PCA", "Direct"] + [aliases[name] for name in loaded],
                f"View {view_name}",
            )
        )
    figures.append(_save(_stack(endpoint_panels), output / "endpoint_technical.png"))
    figures.append(_save(_stack(anonymous_panels), output / "endpoint_anonymous.png"))

    source_atlas = source_textures.base_color_linear
    atlas_images = [_atlas_image(source_atlas)]
    atlas_labels = ["Source"]
    error_images = [np.zeros_like(atlas_images[0])]
    atlas_states = {"Raw PCA": parent}
    atlas_states.update({name: states[10000] for name, (_, states) in loaded.items()})
    atlas_images.append(_atlas_image(direct_textures.base_color_linear))
    atlas_labels.append("Direct BaseColor")
    error_images.append(_error_image(direct_textures.base_color_linear, source_atlas))
    for label, state in atlas_states.items():
        latent, weight, bias, compander, enabled = state
        processed = postprocess_affine_output(
            F.linear(hard_quantize_unorm8(latent), weight, bias),
            compander_parameters=compander if enabled else None,
            straight_through=False,
        )
        atlas_images.append(_atlas_image(processed.seven[..., :3]))
        atlas_labels.append(label)
        error_images.append(_error_image(processed.seven[..., :3], source_atlas))
    resized_atlas = [np.asarray(Image.fromarray(value).resize((384, 384), Image.Resampling.LANCZOS)) for value in atlas_images]
    resized_error = [np.asarray(Image.fromarray(value).resize((384, 384), Image.Resampling.BILINEAR)) for value in error_images]
    atlas_sheet = _stack(
        [
            _panel(resized_atlas, atlas_labels, "BaseColor atlas"),
            _panel(resized_error, atlas_labels, "Mean absolute RGB error, fixed 0..0.25 scale"),
        ]
    )
    figures.append(_save(atlas_sheet, output / "basecolor_atlas_and_error.png"))
    manifest = {
        "schema_version": 1,
        "status": "complete_basecolor_gate_figures",
        "candidates": list(loaded),
        "anonymous_aliases": aliases,
        "figures": figures,
        "top_and_rear_selection_metric": False,
        "formal_holdout_accessed": False,
        "training_started": False,
        "ue_started": False,
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(arguments.config, arguments.candidate, arguments.output_root.resolve())
    print(json.dumps({"status": result["status"], "figures": len(result["figures"])}, sort_keys=True))


if __name__ == "__main__":
    main()
