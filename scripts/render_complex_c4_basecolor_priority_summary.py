"""Render Phase 3 technical and anonymous summaries for one complex C4 asset."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for directory in (SRC, SCRIPTS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from cg_frontier.assets.preprocess import sha256_file  # noqa: E402
from cg_frontier.compression.basecolor_priority import postprocess_affine_output  # noqa: E402
from cg_frontier.compression.render_loss import hard_quantize_unorm8  # noqa: E402
from cg_frontier.render.gbuffer import Core4Textures, sample_core4_material  # noqa: E402
from cg_frontier.render.generic_c4_rig import build_generic_c4_rig  # noqa: E402
from cg_frontier.render.pbr import shade_ggx  # noqa: E402
from render_scifihelmet_c4_basecolor_priority_summary import (  # noqa: E402
    _atlas_image,
    _display,
    _error_image,
    _load_candidate,
    _panel,
    _render_state,
    _repo_output_path,
    _save,
    _stack,
    _write_json,
)
from train_complex_c4_basecolor_priority_10k import (  # noqa: E402
    _load_config,
    _selected_spec,
    prepare_generic_asset,
)


DEFAULT_CONFIG = ROOT / "configs/train/complex_c4_basecolor_priority_10k_v1.yaml"


def _candidate_matrix(values: list[str]) -> tuple[dict[str, Path], str]:
    parsed: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("candidate input must use ID=output-root")
        candidate_id, raw_path = value.split("=", 1)
        path = Path(raw_path).resolve()
        if not candidate_id or not path.is_dir() or not path.is_relative_to(ROOT):
            raise ValueError("candidate root must exist inside the repository")
        if candidate_id in parsed:
            raise ValueError("candidate IDs must be unique")
        parsed[candidate_id] = path
    base = [name for name in parsed if name in {"BC80", "BC90"}]
    expected = {
        "BC80": "BC80-compander",
        "BC90": "BC90-compander",
    }
    if set(parsed) != {"N0-control", *base} | {expected[name] for name in base} or len(base) != 1:
        raise ValueError("summary requires N0-control, one B*, and its compander")
    return parsed, base[0]


def _direct_textures(prepared: dict[str, Any]) -> Core4Textures:
    textures = prepared["textures"]
    return Core4Textures(
        base_color_linear=(
            torch.floor(textures.base_color_linear.clamp(0.0, 1.0) * 255.0 + 0.5)
            / 255.0
        ).contiguous(),
        normal=textures.normal,
        roughness=textures.roughness,
        metallic=textures.metallic,
        source_hashes=textures.source_hashes,
    )


def _validate_candidate_report(
    candidate_id: str,
    report: dict[str, Any],
    prepared: dict[str, Any],
) -> None:
    identity = report.get("identity", {})
    if identity.get("input_hash") != prepared["lineage"]["input_sha256"]:
        raise ValueError(f"candidate input mismatch: {candidate_id}")
    if identity.get("rig_hash") != prepared["rig_hash"]:
        raise ValueError(f"candidate rig mismatch: {candidate_id}")


def run(
    config_path: Path,
    screen_summary: Path,
    screen_sha256: str,
    asset_id: str,
    candidate_values: list[str],
    output: Path,
) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("complex summary rendering requires CUDA")
    output = _repo_output_path(output)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite summary root: {output}")
    if sha256_file(screen_summary) != screen_sha256:
        raise ValueError("Phase 0 screen summary SHA-256 mismatch")
    summary = json.loads(screen_summary.read_text(encoding="utf-8"))
    config = _load_config(config_path)
    spec = _selected_spec(config, summary, asset_id)
    prepared = prepare_generic_asset(
        config_path,
        config,
        spec,
        screen_sha256=screen_sha256,
    )
    candidate_roots, selected = _candidate_matrix(candidate_values)
    loaded = {
        name: _load_candidate(name, root)
        for name, root in candidate_roots.items()
    }
    for name, (report, _) in loaded.items():
        _validate_candidate_report(name, report, prepared)
    output.mkdir(parents=True)

    raw = prepared["raw"]
    parent = (
        raw.latent_unorm8.cuda().to(torch.float32) / 255.0,
        raw.weight.cuda(),
        raw.bias.cuda(),
        torch.tensor((1.0, 0.0), device="cuda"),
        False,
    )
    direct_textures = _direct_textures(prepared)
    rig = build_generic_c4_rig()
    if rig.rig_hash != prepared["rig_hash"]:
        raise RuntimeError("generic rig changed during summary rendering")
    view_lights = {
        "front": 0,
        "top": 2,
        "upper_side": 4,
        "rear": 2,
    }
    views = [
        (name, rig.presentation_views[name], view_lights[name], name not in {"top", "rear"})
        for name in ("front", "top", "upper_side", "rear")
    ]
    exposure = float(config["rig"]["display_exposure"])
    minimum_roughness = float(config["rig"]["minimum_roughness"])
    render_config = config["rig"]

    figures = []
    for candidate_id, (_, states) in loaded.items():
        panels = []
        for view_name, camera_index, light_index, selection_metric in views:
            geometry = prepared["geometries"][camera_index]
            camera = prepared["cameras"][camera_index]
            light = prepared["camera_lights"][camera_index][light_index]
            source = prepared["references"][light_index][camera_index]
            images = [
                _display(source, exposure),
                _display(_render_state(parent, geometry, camera, light, render_config), exposure),
            ]
            images.extend(
                _display(_render_state(states[step], geometry, camera, light, render_config), exposure)
                for step in (1000, 5000, 10000)
            )
            panels.append(
                _panel(
                    images,
                    ["Source", "Raw PCA", "1k", "5k", "10k"],
                    f"{asset_id} | {candidate_id} | {view_name} | selection_metric={str(selection_metric).lower()}",
                )
            )
        figures.append(_save(_stack(panels), output / f"trajectory_{candidate_id}.png"))

    endpoint_order = ["N0-control", selected, f"{selected}-compander"]
    aliases = {name: chr(ord("A") + index) for index, name in enumerate(endpoint_order)}
    technical, anonymous = [], []
    for view_name, camera_index, light_index, selection_metric in views:
        geometry = prepared["geometries"][camera_index]
        camera = prepared["cameras"][camera_index]
        light = prepared["camera_lights"][camera_index][light_index]
        source = prepared["references"][light_index][camera_index]
        direct = shade_ggx(
            geometry,
            camera,
            light,
            material_override=sample_core4_material(geometry, direct_textures),
            minimum_roughness=minimum_roughness,
        )
        images = [
            _display(source, exposure),
            _display(_render_state(parent, geometry, camera, light, render_config), exposure),
            _display(direct, exposure),
        ]
        images.extend(
            _display(
                _render_state(loaded[name][1][10000], geometry, camera, light, render_config),
                exposure,
            )
            for name in endpoint_order
        )
        technical.append(
            _panel(
                images,
                ["Source", "Raw PCA", "Direct BaseColor", *endpoint_order],
                f"{asset_id} | {view_name} | selection_metric={str(selection_metric).lower()}",
            )
        )
        anonymous.append(
            _panel(
                images,
                ["Reference", "PCA", "Direct", *[aliases[name] for name in endpoint_order]],
                f"{asset_id} | View {view_name}",
            )
        )
    figures.append(_save(_stack(technical), output / "endpoint_technical.png"))
    figures.append(_save(_stack(anonymous), output / "endpoint_anonymous.png"))

    source_atlas = prepared["target_atlas"][..., :3]
    atlas_states = {"Raw PCA": parent}
    atlas_states.update({name: loaded[name][1][10000] for name in endpoint_order})
    atlas_images = [_atlas_image(source_atlas), _atlas_image(direct_textures.base_color_linear)]
    atlas_labels = ["Source", "Direct BaseColor"]
    error_images = [np.zeros_like(atlas_images[0]), _error_image(direct_textures.base_color_linear, source_atlas)]
    for label, state in atlas_states.items():
        latent, weight, bias, compander, enabled = state
        processed = postprocess_affine_output(
            F.linear(hard_quantize_unorm8(latent), weight, bias),
            compander_parameters=compander if enabled else None,
            straight_through=False,
        )
        prediction = processed.seven[..., :3]
        atlas_images.append(_atlas_image(prediction))
        atlas_labels.append(label)
        error_images.append(_error_image(prediction, source_atlas))
    resized_atlas = [
        np.asarray(Image.fromarray(value).resize((384, 384), Image.Resampling.LANCZOS))
        for value in atlas_images
    ]
    resized_error = [
        np.asarray(Image.fromarray(value).resize((384, 384), Image.Resampling.BILINEAR))
        for value in error_images
    ]
    figures.append(
        _save(
            _stack(
                [
                    _panel(resized_atlas, atlas_labels, f"{asset_id} BaseColor atlas"),
                    _panel(resized_error, atlas_labels, "Mean absolute RGB error, fixed 0..0.25 scale"),
                ]
            ),
            output / "basecolor_atlas_and_error.png",
        )
    )
    manifest = {
        "schema_version": 1,
        "status": "complete_complex_basecolor_gate_figures",
        "asset_id": asset_id,
        "screen_summary_sha256": screen_sha256,
        "rig_hash": prepared["rig_hash"],
        "selected_basecolor_candidate": selected,
        "candidates": endpoint_order,
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
    parser.add_argument("--screen-summary", type=Path, required=True)
    parser.add_argument("--screen-summary-sha256", required=True)
    parser.add_argument("--asset", required=True)
    parser.add_argument("--candidate", action="append", default=[])
    parser.add_argument("--output-root", type=Path, required=True)
    arguments = parser.parse_args()
    result = run(
        arguments.config,
        arguments.screen_summary.resolve(),
        arguments.screen_summary_sha256,
        arguments.asset,
        arguments.candidate,
        arguments.output_root,
    )
    print(json.dumps({"status": result["status"], "figures": len(result["figures"])}, sort_keys=True))


if __name__ == "__main__":
    main()
