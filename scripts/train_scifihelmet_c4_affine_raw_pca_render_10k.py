"""Train uniform raw PCA with a bounded train-only differentiable render objective."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torch import nn
import torch.nn.functional as F
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for directory in (SRC, SCRIPTS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from cg_frontier.assets.gltf_mesh import load_gltf_mesh  # noqa: E402
from cg_frontier.compression.affine_material import SCALAR_ROWS, decode_affine_material  # noqa: E402
from cg_frontier.compression.affine_pca import export_p0_bundle, rasterize_uv_charts  # noqa: E402
from cg_frontier.compression.affine_regularizers import fake_quantize_unorm8  # noqa: E402
from cg_frontier.compression.material import load_core4_targets  # noqa: E402
from cg_frontier.compression.render_loss import (  # noqa: E402
    bilinear_sample_top_down_wrap,
    decoded_to_material,
    hard_quantize_unorm8,
    masked_render_metrics,
    sparse_fake_quantized_bilinear_sample_top_down_wrap,
)
from cg_frontier.render.gbuffer import (  # noqa: E402
    load_core4_textures,
    render_geometry_gbuffer,
    sample_core4_material,
)
from cg_frontier.render.pbr import shade_ggx  # noqa: E402
from render_scifihelmet_c4_affine_raw_pca import (  # noqa: E402
    _camera,
    _cube_diagnostic,
    _display,
    _material_metrics,
    _repo_path,
    _sha256,
    _support,
    _verify,
)
from run_scifihelmet_c4_affine_chroma8_l0_40k import _orbit_camera_from_spec  # noqa: E402
from run_scifihelmet_c4_affine_preflight import _light, _targets_to_seven  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/train/scifihelmet_c4_affine_raw_pca_render_10k_v1.yaml"


def _load(path: Path) -> Mapping[str, Any]:
    payload = path.read_bytes()
    value = yaml.safe_load(payload) if path.suffix.lower() in {".yaml", ".yml"} else json.loads(payload)
    if not isinstance(value, Mapping):
        raise ValueError(f"mapping expected: {path}")
    return value


def _write_json(path: Path, value: Any) -> str:
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n", encoding="ascii")
    return digest


def _support_penalty(seven: torch.Tensor, margin: float) -> torch.Tensor:
    scalar = seven[:, list(SCALAR_ROWS)]
    scalar_penalty = F.relu(margin - scalar).mean() + F.relu(scalar - (1.0 - margin)).mean()
    normal_penalty = F.relu(torch.linalg.vector_norm(seven[:, 3:5], dim=-1) - (1.0 - margin)).mean()
    return scalar_penalty + normal_penalty


def _checkpoint(
    path: Path,
    *,
    step: int,
    latent: nn.Parameter,
    weight: nn.Parameter,
    bias: nn.Parameter,
    latent_optimizer: torch.optim.Optimizer,
    affine_optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
    lineage: Mapping[str, str],
) -> str:
    path.parent.mkdir(parents=True, exist_ok=False)
    torch.save(
        {
            "schema_version": 1,
            "candidate_id": "R0_raw_render",
            "step": step,
            **lineage,
            "latent": latent.detach().cpu(),
            "weight": weight.detach().cpu(),
            "bias": bias.detach().cpu(),
            "latent_optimizer": latent_optimizer.state_dict(),
            "affine_optimizer": affine_optimizer.state_dict(),
            "rng_state": generator.get_state().cpu(),
        },
        path,
    )
    return _sha256(path)


@torch.no_grad()
def _atlas_metrics(latent, weight, bias, valid_indices, target_valid):
    deployed = hard_quantize_unorm8(latent)
    prediction = F.linear(deployed.reshape(-1, 4)[valid_indices], weight, bias)
    return {
        "material": _material_metrics(prediction, target_valid),
        "observed_valid_support": _support(prediction),
        "full_cube": _cube_diagnostic(weight, bias),
    }


@torch.no_grad()
def _render_metrics(latent, weight, bias, geometries, cameras, lights, references, pairs, render):
    deployed = hard_quantize_unorm8(latent)
    rows = []
    for camera_index, light_index in pairs:
        geometry = geometries[camera_index]
        sampled = bilinear_sample_top_down_wrap(deployed, geometry.torch_buffers["uv"])
        candidate = shade_ggx(
            geometry,
            cameras[camera_index],
            lights[light_index],
            material_override=decoded_to_material(geometry, decode_affine_material(sampled, weight, bias)),
            minimum_roughness=float(render["minimum_roughness"]),
        )
        rows.append(
            masked_render_metrics(
                references[light_index][camera_index],
                candidate,
                geometry.torch_buffers["mask"],
                linear_psnr_data_range=float(render["linear_psnr_data_range"]),
                display_exposure=float(render["display_exposure"]),
            )
        )
    numeric = [key for key, value in rows[0].items() if isinstance(value, (int, float)) and not isinstance(value, bool)]
    return {
        "case_count": len(rows),
        "mean": {key: float(np.mean([row[key] for row in rows])) for key in numeric},
        "cases": rows,
    }


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


@torch.no_grad()
def _trajectory_visual(mesh, textures, parent, checkpoints, cameras, lights, render, views, output):
    columns = ["source", "raw_parent", "step_1000", "step_5000", "step_10000"]
    labels = ["Source", "Raw PCA parent", "Render train 1k", "Render train 5k", "Render train 10k"]
    states = {"raw_parent": parent}
    for step in (1000, 5000, 10000):
        checkpoint = torch.load(checkpoints[step], map_location="cuda", weights_only=False)
        states[f"step_{step}"] = (checkpoint["latent"].cuda(), checkpoint["weight"].cuda(), checkpoint["bias"].cuda())
    images = {}
    for view in views:
        camera = _camera(view, render, cameras)
        geometry = render_geometry_gbuffer(mesh, camera, (256, 256), device="cuda")
        light = lights[int(view["light_index"])]
        reference = shade_ggx(
            geometry,
            camera,
            light,
            material_override=sample_core4_material(geometry, textures),
            minimum_roughness=float(render["minimum_roughness"]),
        )
        images[(view["id"], "source")] = _display(reference, float(render["display_exposure"]))
        for key, (latent, weight, bias) in states.items():
            sampled = bilinear_sample_top_down_wrap(hard_quantize_unorm8(latent), geometry.torch_buffers["uv"])
            candidate = shade_ggx(
                geometry,
                camera,
                light,
                material_override=decoded_to_material(geometry, decode_affine_material(sampled, weight, bias)),
                minimum_roughness=float(render["minimum_roughness"]),
            )
            images[(view["id"], key)] = _display(candidate, float(render["display_exposure"]))
    tile, left, top, gap = 256, 190, 92, 4
    width = left + len(columns) * (tile + gap) - gap
    height = top + len(views) * (tile + gap) - gap + 42
    canvas = Image.new("RGB", (width, height), (247, 247, 245))
    draw = ImageDraw.Draw(canvas)
    draw.text((20, 14), "SciFiHelmet raw PCA — differentiable-render trajectory", fill=(28, 31, 35), font=_font(23, bold=True))
    draw.text((20, 48), "RGBA8 · one filtered sample · unconstrained single 4→7 affine · camera31/light6", fill=(78, 83, 90), font=_font(14))
    for index, label in enumerate(labels):
        draw.text((left + index * (tile + gap) + 5, 72), label, fill=(35, 38, 42), font=_font(13, bold=True))
    for row, view in enumerate(views):
        y = top + row * (tile + gap)
        draw.text((20, y + 105), str(view["label"]), fill=(35, 38, 42), font=_font(17, bold=True))
        for column, key in enumerate(columns):
            x = left + column * (tile + gap)
            canvas.paste(Image.fromarray(images[(view["id"], key)]), (x, y))
            draw.rectangle((x, y, x + tile - 1, y + tile - 1), outline=(185, 187, 190), width=1)
    draw.text((20, height - 28), "No full-cube certificate; top/rear are presentation-only; no formal holdout accessed.", fill=(100, 84, 38), font=_font(12))
    path = output / "raw_pca_render_training_trajectory.png"
    canvas.save(path, optimize=True)
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": _sha256(path)}


def run(config_path: Path, *, output_override: str | None = None, max_steps: int | None = None):
    config = copy.deepcopy(_load(config_path))
    if config.get("formal_holdout_access") != "forbidden":
        raise ValueError("formal holdout must remain forbidden")
    if output_override is not None:
        config["output_root"] = output_override
    if max_steps is not None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        config["training"]["steps"] = max_steps
        config["training"]["checkpoint_interval"] = min(int(config["training"]["checkpoint_interval"]), max_steps)
    output = _repo_path(config["output_root"], "output_root")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output root: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    source = config["source"]
    preflight_path = _repo_path(source["preflight_config"], "source.preflight_config")
    pool_path = _repo_path(source["render_pool_config"], "source.render_pool_config")
    manifest_path = _repo_path(source["standard_p0_manifest"], "source.standard_p0_manifest")
    _verify(preflight_path, source["preflight_config_sha256"], "preflight config")
    _verify(pool_path, source["render_pool_config_sha256"], "render pool config")
    _verify(manifest_path, source["standard_p0_manifest_sha256"], "standard P0 manifest")
    preflight, pool = _load(preflight_path), _load(pool_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mesh = load_gltf_mesh(_repo_path(preflight["inputs"]["gltf"], "inputs.gltf"))
    core4_dir = _repo_path(preflight["inputs"]["core4_dir"], "inputs.core4_dir")
    core4_manifest = _repo_path(preflight["inputs"]["core4_manifest"], "inputs.core4_manifest")
    targets_cpu = load_core4_targets(core4_dir, "cpu")
    target_seven_cpu = _targets_to_seven(targets_cpu)
    valid_mask, chart_ids = rasterize_uv_charts(mesh.texcoords, mesh.triangles, height=targets_cpu.height, width=targets_cpu.width)
    bundle = export_p0_bundle(target_seven_cpu, valid_mask, chart_ids, margin=float(preflight["p0"]["safety_margin"]))
    if bundle.manifest != manifest:
        raise RuntimeError("reconstructed standard P0 manifest mismatch")
    if bundle.calibration.raw.artifact_hash != source["raw_parent_artifact_hash"]:
        raise RuntimeError("raw parent artifact hash mismatch")

    device = torch.device("cuda")
    render = pool["render"]
    if tuple(render["resolution"]) != (256, 256) or len(pool["train_cameras"]) != 31 or len(pool["train_lights"]) != 6:
        raise ValueError("runner requires frozen camera31/light6 at 256x256")
    cameras = [_orbit_camera_from_spec(value, render) for value in pool["train_cameras"]]
    geometries = [render_geometry_gbuffer(mesh, camera, (256, 256), device=device) for camera in cameras]
    lights = [_light(value) for value in pool["train_lights"]]
    textures = load_core4_textures(core4_manifest, device)
    with torch.no_grad():
        references = [
            [
                shade_ggx(
                    geometry,
                    camera,
                    light,
                    material_override=sample_core4_material(geometry, textures),
                    minimum_roughness=float(render["minimum_roughness"]),
                )
                for geometry, camera in zip(geometries, cameras, strict=True)
            ]
            for light in lights
        ]
    raw = bundle.calibration.raw
    latent = nn.Parameter(raw.latent_unorm8.to(device=device, dtype=torch.float32) / 255.0)
    weight = nn.Parameter(raw.weight.to(device=device, dtype=torch.float32).clone())
    bias = nn.Parameter(raw.bias.to(device=device, dtype=torch.float32).clone())
    parent = (latent.detach().clone(), weight.detach().clone(), bias.detach().clone())
    training, loss_config = config["training"], config["loss"]
    latent_optimizer = torch.optim.Adam((latent,), lr=float(training["latent_learning_rate"]))
    affine_optimizer = torch.optim.Adam((weight, bias), lr=float(training["affine_learning_rate"]))
    generator = torch.Generator(device=device).manual_seed(int(config["seed"]))
    valid_indices = torch.nonzero(valid_mask.reshape(-1), as_tuple=False)[:, 0].to(device)
    target_valid = target_seven_cpu.reshape(-1, 7).to(device)[valid_indices]
    audit_pairs = [tuple(int(value) for value in pair) for pair in config["audit_pairs"]]
    lineage = {
        "parent_artifact_hash": raw.artifact_hash,
        "config_sha256": _sha256(config_path),
        "input_sha256": hashlib.sha256((_sha256(_repo_path(preflight["inputs"]["gltf"], "inputs.gltf")) + _sha256(core4_manifest)).encode("ascii")).hexdigest(),
    }
    output.mkdir(parents=True)
    parent_report = {
        **_atlas_metrics(*parent, valid_indices, target_valid),
        "audit_render": _render_metrics(*parent, geometries, cameras, lights, references, audit_pairs, render),
    }
    curve, trajectory, checkpoints = [], [], {}
    started = time.perf_counter()
    for step in range(1, int(training["steps"]) + 1):
        positions = torch.randint(0, valid_indices.numel(), (int(training["material_batch_size"]),), generator=generator, device=device)
        flat_indices = valid_indices[positions]
        camera_index = int(torch.randint(0, len(cameras), (1,), generator=generator, device=device))
        light_index = int(torch.randint(0, len(lights), (1,), generator=generator, device=device))
        material_seven = F.linear(fake_quantize_unorm8(latent.reshape(-1, 4)[flat_indices]), weight, bias)
        target_batch = target_valid[positions]
        pred_xy, target_xy = material_seven[:, 3:5], target_batch[:, 3:5]
        pred_z = torch.sqrt(torch.clamp(1.0 - pred_xy.square().sum(-1, keepdim=True), min=1e-8))
        target_z = torch.sqrt(torch.clamp(1.0 - target_xy.square().sum(-1, keepdim=True), min=1e-8))
        pred_normal = F.normalize(torch.cat((pred_xy, pred_z), dim=-1), dim=-1)
        target_normal = F.normalize(torch.cat((target_xy, target_z), dim=-1), dim=-1)
        material_terms = {
            "base_color_l1": F.l1_loss(material_seven[:, :3], target_batch[:, :3]),
            "normal_cosine": torch.mean(1.0 - (pred_normal * target_normal).sum(-1)),
            "roughness_l1": F.l1_loss(material_seven[:, 5], target_batch[:, 5]),
            "metallic_l1": F.l1_loss(material_seven[:, 6], target_batch[:, 6]),
        }
        material_loss = sum(material_terms[name] * float(loss_config[name]) for name in material_terms)
        geometry = geometries[camera_index]
        sampled = sparse_fake_quantized_bilinear_sample_top_down_wrap(latent, geometry.torch_buffers["uv"])
        sampled_seven = F.linear(sampled, weight, bias)
        candidate = shade_ggx(
            geometry,
            cameras[camera_index],
            lights[light_index],
            material_override=decoded_to_material(geometry, decode_affine_material(sampled, weight, bias)),
            minimum_roughness=float(render["minimum_roughness"]),
        )
        mask = geometry.torch_buffers["mask"]
        reference = references[light_index][camera_index]
        epsilon = float(loss_config["charbonnier_epsilon"])
        difference = candidate[mask] - reference[mask]
        helmet = torch.sqrt(difference.square() + epsilon * epsilon).mean()
        log_helmet = torch.abs(torch.log1p(candidate[mask].clamp_min(0.0)) - torch.log1p(reference[mask].clamp_min(0.0))).mean()
        support = _support_penalty(material_seven, float(training["support_margin"])) + _support_penalty(sampled_seven[mask], float(training["support_margin"]))
        total = material_loss + helmet * float(loss_config["helmet_charbonnier"]) + log_helmet * float(loss_config["helmet_log1p"]) + support * float(training["support_penalty_weight"])
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite loss at step {step}")
        latent_optimizer.zero_grad(set_to_none=True)
        affine_optimizer.zero_grad(set_to_none=True)
        total.backward()
        latent_optimizer.step()
        affine_optimizer.step()
        with torch.no_grad():
            latent.clamp_(0.0, 1.0)
        if step == 1 or step % int(training["log_interval"]) == 0:
            point = {"step": step, "loss": float(total.detach()), "material": float(material_loss.detach()), "helmet": float(helmet.detach()), "log_helmet": float(log_helmet.detach()), "support": float(support.detach()), "camera_index": camera_index, "light_index": light_index}
            curve.append(point)
            print(json.dumps(point, sort_keys=True), flush=True)
        if step % int(training["checkpoint_interval"]) == 0:
            checkpoint_path = output / "checkpoints" / f"step_{step:05d}" / "checkpoint.pt"
            digest = _checkpoint(checkpoint_path, step=step, latent=latent, weight=weight, bias=bias, latent_optimizer=latent_optimizer, affine_optimizer=affine_optimizer, generator=generator, lineage=lineage)
            checkpoints[step] = checkpoint_path
            trajectory.append({"step": step, "checkpoint": checkpoint_path.relative_to(output).as_posix(), "checkpoint_sha256": digest, **_atlas_metrics(latent, weight, bias, valid_indices, target_valid), "audit_render": _render_metrics(latent, weight, bias, geometries, cameras, lights, references, audit_pairs, render)})
            print(json.dumps({"checkpoint": step, "sha256": digest}, sort_keys=True), flush=True)
    torch.cuda.synchronize(device)
    all_pairs = (
        [(camera_index, light_index) for light_index in range(len(lights)) for camera_index in range(len(cameras))]
        if int(training["steps"]) >= 10000
        else audit_pairs
    )
    endpoint_full = _render_metrics(latent, weight, bias, geometries, cameras, lights, references, all_pairs, render)
    parent_full = _render_metrics(*parent, geometries, cameras, lights, references, all_pairs, render)
    visual = None
    if int(training["steps"]) >= 10000:
        visual = _trajectory_visual(mesh, textures, parent, checkpoints, cameras, lights, render, [dict(value) for value in config["visual_views"]], output)
    report = {
        "schema_version": 1,
        "status": "complete_raw_pca_render_10k" if int(training["steps"]) == 10000 else "complete_preflight",
        "candidate_id": "R0_raw_render",
        "steps": int(training["steps"]),
        "lineage": lineage,
        "runtime_contract": {"texture": "2048x2048 RGBA8", "filtered_samples_per_pixel": 1, "decoder": "unconstrained single 4_to_7 affine", "decoder_macs_per_pixel": 28, "full_cube_certificate": False},
        "parent": {**parent_report, "full_render_31x6": parent_full},
        "trajectory": trajectory,
        "endpoint": {**trajectory[-1], "full_render_31x6": endpoint_full},
        "curve": curve,
        "wall_seconds": time.perf_counter() - started,
        "visual": visual,
        "formal_holdout_accessed": False,
        "ue_started": False,
    }
    _write_json(output / "training_report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root")
    parser.add_argument("--max-steps", type=int)
    args = parser.parse_args()
    report = run(args.config.resolve(), output_override=args.output_root, max_steps=args.max_steps)
    print(json.dumps({"status": report["status"], "steps": report["steps"]}, sort_keys=True))


if __name__ == "__main__":
    main()
