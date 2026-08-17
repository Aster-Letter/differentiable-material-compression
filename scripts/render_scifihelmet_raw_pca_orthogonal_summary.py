"""Render four-view trajectories and endpoint overview for raw-PCA O1/O2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

from PIL import Image, ImageDraw, ImageFont
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for directory in (SRC, SCRIPTS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from cg_frontier.compression.affine_material import decode_affine_material  # noqa: E402
from cg_frontier.compression.render_loss import (  # noqa: E402
    bilinear_sample_top_down_wrap,
    decoded_to_material,
    hard_quantize_unorm8,
)
from cg_frontier.render.gbuffer import render_geometry_gbuffer, sample_core4_material  # noqa: E402
from cg_frontier.render.pbr import shade_ggx  # noqa: E402
from render_scifihelmet_c4_affine_raw_pca import _camera, _display  # noqa: E402
from train_scifihelmet_c4_affine_raw_pca_orthogonal_10k import _prepare  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/train/scifihelmet_c4_affine_raw_pca_orthogonal_10k_v1.yaml"
DEFAULT_OUTPUT = ROOT / "outputs/scifihelmet_c4_affine_v1/raw_pca_orthogonal/bf215f-summary-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _font(size: int, *, bold: bool = False):
    names = ("arialbd.ttf", "segoeuib.ttf") if bold else ("arial.ttf", "segoeui.ttf")
    windows_dir = os.environ.get("WINDIR")
    if not windows_dir:
        return ImageFont.load_default()
    for name in names:
        path = Path(windows_dir) / "Fonts" / name
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _state(path: Path):
    payload = torch.load(path, map_location="cuda", weights_only=False)
    return payload["latent"].cuda(), payload["weight"].cuda(), payload["bias"].cuda()


@torch.no_grad()
def _render_state(prepared, camera, light, state):
    geometry = render_geometry_gbuffer(prepared["mesh"], camera, (256, 256), device="cuda")
    if state is None:
        material = sample_core4_material(geometry, prepared["textures"])
    else:
        latent, weight, bias = state
        sampled = bilinear_sample_top_down_wrap(hard_quantize_unorm8(latent), geometry.torch_buffers["uv"])
        material = decoded_to_material(geometry, decode_affine_material(sampled, weight, bias))
    image = shade_ggx(
        geometry,
        camera,
        light,
        material_override=material,
        minimum_roughness=float(prepared["render"]["minimum_roughness"]),
    )
    return Image.fromarray(_display(image, float(prepared["render"]["display_exposure"])))


def _panel(title, subtitle, rows, columns, images, output):
    tile, left, top, gap = 256, 190, 94, 4
    width = left + len(columns) * (tile + gap) - gap
    height = top + len(rows) * (tile + gap) - gap + 42
    canvas = Image.new("RGB", (width, height), (247, 247, 245))
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 14), title, fill=(28, 31, 35), font=_font(23, bold=True))
    draw.text((20, 49), subtitle, fill=(78, 83, 90), font=_font(13))
    for index, (_, label) in enumerate(columns):
        draw.text((left + index * (tile + gap) + 4, 73), label, fill=(35, 38, 42), font=_font(12, bold=True))
    for row_index, (row_id, row_label) in enumerate(rows):
        y = top + row_index * (tile + gap)
        draw.text((18, y + 105), row_label, fill=(35, 38, 42), font=_font(15, bold=True))
        for column_index, (column_id, _) in enumerate(columns):
            x = left + column_index * (tile + gap)
            canvas.paste(images[(row_id, column_id)], (x, y))
            draw.rectangle((x, y, x + tile - 1, y + tile - 1), outline=(185, 187, 190), width=1)
    draw.text((20, height - 27), "Top/rear/yellow diagnostics are presentation-only (selection_metric=false).", fill=(100, 84, 38), font=_font(12))
    canvas.save(output, optimize=True)


def run(config_path: Path, output: Path) -> dict:
    config = yaml.safe_load(config_path.read_bytes())
    if output.exists():
        raise FileExistsError(f"refusing to overwrite summary root: {output}")
    output.mkdir(parents=True)
    prepared = _prepare(config, config_path)
    raw = prepared["raw"]
    parent = (
        raw.latent_unorm8.cuda().float() / 255.0,
        raw.weight.cuda().float(),
        raw.bias.cuda().float(),
    )
    views = []
    for value in config["visual_views"]:
        camera = _camera(value, prepared["render"], prepared["cameras"])
        views.append((str(value["id"]), str(value["label"]), camera, prepared["lights"][int(value["light_index"])]))
    rows = [(view_id, label) for view_id, label, _, _ in views]
    artifacts = []
    endpoint_states = {}
    for candidate_id, candidate in config["candidates"].items():
        root = ROOT / candidate["output_root"]
        states = {"source": None, "parent": parent}
        for step in (1000, 5000, 10000):
            states[str(step)] = _state(root / "checkpoints" / f"step_{step:05d}" / "checkpoint.pt")
        endpoint_states[candidate_id] = states["10000"]
        images = {}
        for view_id, _, camera, light in views:
            for state_id, state in states.items():
                images[(view_id, state_id)] = _render_state(prepared, camera, light, state)
        path = output / f"{candidate_id.lower()}_trajectory_four_views.png"
        _panel(
            f"SciFiHelmet {candidate_id} — raw PCA orthogonal trajectory",
            "2048² RGBA8 · one filtered sample · single 4→7 affine · source-only adaptive profile K=4",
            rows,
            [("source", "Source"), ("parent", "Raw parent"), ("1000", "1k"), ("5000", "5k"), ("10000", "10k")],
            images,
            path,
        )
        artifacts.append({"path": path.relative_to(ROOT).as_posix(), "sha256": _sha256(path)})
    r0_path = ROOT / "outputs/scifihelmet_c4_affine_v1/raw_pca_train/bf215f-r1-10k/checkpoints/step_10000/checkpoint.pt"
    r0 = _state(r0_path)
    endpoint_columns = [("source", "Source"), ("parent", "Raw parent"), ("R0", "R0 render 10k")] + [
        (candidate_id, candidate_id) for candidate_id in config["candidates"]
    ]
    endpoint_state_map = {"source": None, "parent": parent, "R0": r0, **endpoint_states}
    images = {}
    for view_id, _, camera, light in views:
        for state_id, state in endpoint_state_map.items():
            images[(view_id, state_id)] = _render_state(prepared, camera, light, state)
    path = output / "endpoint_overview_four_views.png"
    _panel(
        "SciFiHelmet raw-PCA orthogonal endpoints",
        "R0 control vs O1 coordinate measure vs O2 adaptive BaseColor macro; no automatic winner",
        rows,
        endpoint_columns,
        images,
        path,
    )
    artifacts.append({"path": path.relative_to(ROOT).as_posix(), "sha256": _sha256(path)})
    manifest = {
        "schema_version": 1,
        "artifacts": artifacts,
        "views": [value[0] for value in rows],
        "yellow_diagnostics": {"selection_metric": False},
        "formal_holdout_accessed": False,
        "ue_started": False,
    }
    payload = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    (output / "manifest.json").write_bytes(payload)
    (output / "manifest.json.sha256").write_text(hashlib.sha256(payload).hexdigest() + "\n", encoding="ascii")
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.config.resolve(), args.output_root.resolve())
    print(json.dumps({"status": "complete", "artifacts": len(result["artifacts"])}))


if __name__ == "__main__":
    main()
