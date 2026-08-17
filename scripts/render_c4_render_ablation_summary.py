"""Render paired 20k C4 ablation evidence and metric deltas for one asset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for directory in (SRC, SCRIPTS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from cg_frontier.compression.basecolor_priority import postprocess_affine_output  # noqa: E402
from cg_frontier.compression.render_ablation import ARMS, OBSERVATION_STEPS, load_ablation_checkpoint  # noqa: E402
from cg_frontier.compression.render_loss import bilinear_sample_top_down_wrap, decoded_to_material, hard_quantize_unorm8  # noqa: E402
from cg_frontier.render.pbr import shade_ggx  # noqa: E402
from train_c4_render_ablation_20k import (  # noqa: E402
    _asset_spec,
    _config,
    _display,
    _material_override,
    _write_json,
    prepare_asset,
)


def _load_report(root: Path, arm: str) -> dict[str, Any]:
    report = json.loads((root / arm / "training_report.json").read_text(encoding="utf-8"))
    if (
        report.get("status") != "complete_20k"
        or report.get("arm") != arm
        or report.get("steps") != 20000
        or report.get("audit_used_for_training") is not False
        or report.get("formal_holdout_accessed") is not False
    ):
        raise ValueError(f"{arm} is not a valid formal 20k report")
    if report["endpoint"]["audit_render"]["case_count"] != 42:
        raise ValueError(f"{arm} does not contain the frozen 42-case audit")
    if report["endpoint"]["train_render"]["case_count"] != 144:
        raise ValueError(f"{arm} does not contain the frozen 144-case train report")
    if tuple(report["observation_steps"]) != OBSERVATION_STEPS:
        raise ValueError(f"{arm} observation trajectory is incomplete")
    return report


def _state(root: Path, report: Mapping[str, Any], prepared: Mapping[str, Any], arm: str):
    checkpoint = root / arm / "checkpoints/step_20000/checkpoint.pt"
    payload = load_ablation_checkpoint(
        checkpoint,
        expected_asset=prepared["asset_id"],
        expected_arm=arm,
        expected_identity=prepared["identity"],
    )
    expected = report["checkpoints"]["20000"]["sha256"]
    from cg_frontier.assets.preprocess import sha256_file

    if sha256_file(checkpoint) != expected:
        raise ValueError(f"{arm} endpoint checkpoint hash mismatch")
    return tuple(payload[name].cuda().float() for name in ("latent", "weight", "bias"))


def _raw_state(prepared: Mapping[str, Any]):
    raw = prepared["raw"]
    return raw.latent_unorm8.cuda().float() / 255.0, raw.weight.cuda().float(), raw.bias.cuda().float()


@torch.no_grad()
def _render(state, prepared: Mapping[str, Any], config: Mapping[str, Any], camera_index: int):
    latent, weight, bias = state
    geometry = prepared["geometries"][camera_index]
    sampled = bilinear_sample_top_down_wrap(hard_quantize_unorm8(latent), geometry.torch_buffers["uv"])
    processed = postprocess_affine_output(F.linear(sampled, weight, bias), compander_parameters=None, straight_through=False)
    return shade_ggx(
        geometry,
        prepared["cameras"][camera_index],
        prepared["camera_lights"][camera_index][0],
        material_override=decoded_to_material(geometry, _material_override(processed)),
        minimum_roughness=float(config["rig"]["minimum_roughness"]),
    )


def _panel(images: list[Image.Image], labels: list[str], title: str) -> Image.Image:
    label_height, title_height = 24, 30
    width = sum(image.width for image in images)
    height = max(image.height for image in images) + label_height + title_height
    canvas = Image.new("RGB", (width, height), (20, 20, 23))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 7), title, fill=(242, 242, 242))
    x = 0
    for image, label in zip(images, labels, strict=True):
        draw.text((x + 6, title_height + 5), label, fill=(225, 225, 225))
        canvas.paste(image, (x, title_height + label_height))
        x += image.width
    return canvas


def _stack(images: list[Image.Image]) -> Image.Image:
    canvas = Image.new("RGB", (max(image.width for image in images), sum(image.height for image in images)), (20, 20, 23))
    y = 0
    for image in images:
        canvas.paste(image, (0, y))
        y += image.height
    return canvas


def _save(image: Image.Image, path: Path) -> dict[str, Any]:
    from cg_frontier.assets.preprocess import sha256_file

    image.save(path, format="PNG")
    return {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path), "size": list(image.size)}


def _source_image(prepared: Mapping[str, Any], config: Mapping[str, Any], camera_index: int) -> Image.Image:
    return _display(prepared["references"][0][camera_index], float(config["rig"]["display_exposure"]))


def _atlas(state, prepared: Mapping[str, Any]) -> np.ndarray:
    latent, weight, bias = state
    raw = F.linear(hard_quantize_unorm8(latent), weight, bias)
    return postprocess_affine_output(raw, compander_parameters=None, straight_through=False).seven[..., :3].detach().cpu().numpy()


def _atlas_image(value: np.ndarray) -> Image.Image:
    value = np.clip(value, 0.0, 1.0)
    srgb = np.where(value <= 0.0031308, value * 12.92, 1.055 * np.power(value, 1.0 / 2.4) - 0.055)
    return Image.fromarray(np.clip(np.rint(srgb * 255.0), 0, 255).astype(np.uint8), mode="RGB").resize((512, 512), Image.Resampling.LANCZOS)


def _error_image(value: np.ndarray, source: np.ndarray) -> Image.Image:
    error = np.mean(np.abs(value - source), axis=-1)
    scale = max(float(np.quantile(error, 0.99)), 1.0e-6)
    normalized = np.clip(error / scale, 0.0, 1.0)
    color = np.stack((normalized, np.sqrt(normalized) * 0.45, np.zeros_like(normalized)), axis=-1)
    return Image.fromarray(np.rint(color * 255.0).astype(np.uint8), mode="RGB").resize((512, 512), Image.Resampling.LANCZOS)


def _delta(left: Mapping[str, float], right: Mapping[str, float]) -> dict[str, float]:
    keys = (
        "masked_linear_hdr_mae",
        "display_ssim",
        "linear_psnr_db",
        "display_psnr_db",
    )
    return {name: float(right[name] - left[name]) for name in keys}


def run(config_path: Path, pair_root: Path, output: Path, asset_id: str) -> dict[str, Any]:
    config = _config(config_path)
    _asset_spec(config, asset_id)
    pair_root = pair_root.resolve()
    output = output.resolve()
    for path in (pair_root, output):
        if not path.is_relative_to(ROOT):
            raise ValueError("summary paths must remain inside the repository")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite summary output: {output}")
    output.mkdir(parents=True)
    prepared = prepare_asset(config_path, config, asset_id)
    reports = {arm: _load_report(pair_root, arm) for arm in ARMS}
    pair = json.loads((pair_root / "paired_summary.json").read_text(encoding="utf-8"))
    if pair.get("paired_sampling_evidence", {}).get("identical") is not True:
        raise ValueError("paired sampling evidence is missing")
    states = {"Raw PCA": _raw_state(prepared)}
    states.update({arm: _state(pair_root, reports[arm], prepared, arm) for arm in ARMS})
    figures: list[dict[str, Any]] = []
    technical_rows, anonymous_rows = [], []
    aliases = {"material_only": "A", "material_render": "B"}
    for view_name, camera_index in prepared["rig"].presentation_views.items():
        source = _source_image(prepared, config, camera_index)
        rendered = {
            name: _display(_render(state, prepared, config, camera_index), float(config["rig"]["display_exposure"]))
            for name, state in states.items()
        }
        technical_rows.append(_panel(
            [source, rendered["Raw PCA"], rendered["material_only"], rendered["material_render"]],
            ["Source", "Raw PCA", "material_only", "material_render"],
            f"{asset_id} | {view_name}",
        ))
        anonymous_rows.append(_panel(
            [rendered["material_only"], rendered["material_render"]],
            [aliases["material_only"], aliases["material_render"]],
            f"{asset_id} | {view_name} | paired endpoint",
        ))
    figures.append(_save(_stack(technical_rows), output / "endpoint_technical.png"))
    figures.append(_save(_stack(anonymous_rows), output / "endpoint_anonymous.png"))

    for arm in ARMS:
        trajectory_images = []
        for step in OBSERVATION_STEPS:
            image = Image.open(pair_root / arm / "observations" / f"step_{step:05d}" / "fixed_views.png").convert("RGB")
            trajectory_images.append(image)
        figures.append(_save(
            _panel(trajectory_images, [f"{step // 1000}k" for step in OBSERVATION_STEPS], f"{asset_id} | {arm} trajectory"),
            output / f"trajectory_{arm}.png",
        ))

    source_atlas = prepared["target_atlas"][..., :3].detach().cpu().numpy()
    atlas_values = {name: _atlas(state, prepared) for name, state in states.items()}
    atlas_panel = _panel(
        [_atlas_image(source_atlas), *(_atlas_image(atlas_values[name]) for name in states)],
        ["Source", *states.keys()],
        f"{asset_id} | BaseColor atlas",
    )
    error_panel = _panel(
        [Image.new("RGB", (512, 512), (0, 0, 0)), *(_error_image(atlas_values[name], source_atlas) for name in states)],
        ["Source", *states.keys()],
        f"{asset_id} | BaseColor absolute error (per-column p99 scale)",
    )
    figures.append(_save(atlas_panel, output / "basecolor_atlas.png"))
    figures.append(_save(error_panel, output / "basecolor_error.png"))

    raw_audit = reports["material_only"]["raw_parent"]["audit_render"]
    raw_train = reports["material_only"]["raw_parent"]["train_render"]
    for arm in ARMS:
        if reports[arm]["raw_parent"] != reports["material_only"]["raw_parent"]:
            raise ValueError("paired arms do not report an identical raw parent")
    comparisons = {
        "raw_q4_to_material_only": {
            "audit_mean_delta": _delta(raw_audit["mean"], reports["material_only"]["endpoint"]["audit_render"]["mean"]),
            "train_mean_delta": _delta(raw_train["mean"], reports["material_only"]["endpoint"]["train_render"]["mean"]),
        },
        "raw_q4_to_material_render": {
            "audit_mean_delta": _delta(raw_audit["mean"], reports["material_render"]["endpoint"]["audit_render"]["mean"]),
            "train_mean_delta": _delta(raw_train["mean"], reports["material_render"]["endpoint"]["train_render"]["mean"]),
        },
        "material_only_to_material_render": {
            "interpretation": "only this paired delta isolates render-supervision contribution",
            "audit_mean_delta": _delta(
                reports["material_only"]["endpoint"]["audit_render"]["mean"],
                reports["material_render"]["endpoint"]["audit_render"]["mean"],
            ),
            "train_mean_delta": _delta(
                reports["material_only"]["endpoint"]["train_render"]["mean"],
                reports["material_render"]["endpoint"]["train_render"]["mean"],
            ),
        },
    }
    result = {
        "schema_version": 1,
        "status": "complete_paired_20k_summary",
        "asset": asset_id,
        "primary_metrics": [
            "audit_mean_masked_linear_hdr_mae",
            "audit_mean_display_ssim",
        ],
        "comparisons": comparisons,
        "figures": figures,
        "anonymous_aliases": aliases,
        "winner_selected": False,
        "single_seed_statistical_significance_claimed": False,
        "formal_holdout_accessed": False,
        "audit_used_for_training": False,
        "emissive_scope": {
            "policy": prepared["spec"]["emissive_policy"],
            "max_rgb_gt_0_05_fraction": prepared["spec"].get("emissive_max_rgb_gt_0_05_fraction"),
        },
    }
    _write_json(output / "summary.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/train/c4_render_ablation_20k_v1.yaml")
    parser.add_argument("--asset", required=True)
    parser.add_argument("--pair-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config.resolve(), args.pair_root.resolve(), args.output_root.resolve(), args.asset)
    print(json.dumps({"status": result["status"], "asset": result["asset"]}, sort_keys=True))


if __name__ == "__main__":
    main()
