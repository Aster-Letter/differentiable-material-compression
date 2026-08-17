"""Export deterministic renders from the affine L0 training sampler."""

from __future__ import annotations

import argparse
import json
import math
import hashlib
import io
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from cg_frontier.assets.gltf_mesh import load_gltf_mesh  # noqa: E402
from cg_frontier.compression.affine_pca import rasterize_uv_charts  # noqa: E402
from cg_frontier.compression.material import load_core4_targets  # noqa: E402
from cg_frontier.compression.render_loss import orbit_camera  # noqa: E402
from cg_frontier.render.gbuffer import (  # noqa: E402
    load_core4_textures,
    render_geometry_gbuffer,
    sample_core4_material,
)
from cg_frontier.render.pbr import (  # noqa: E402
    PointLight,
    linear_to_srgb_torch,
    shade_ggx,
)


DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "configs"
    / "train"
    / "scifihelmet_c4_affine_chroma8_l0_camera31_light6_80k.yaml"
)
DEFAULT_CHECKPOINT = (
    REPOSITORY_ROOT
    / "outputs"
    / "scifihelmet_c4_affine_v1"
    / "train80k"
    / "a874ad-chroma8-l0-camera31-light6-r1"
    / "checkpoints"
    / "L0"
    / "endpoints"
    / "step-080000"
    / "checkpoint.pt"
)


def plan_sample_pairs(
    draw_batches: Sequence[Sequence[int]],
    *,
    camera_names: Sequence[str],
    light_names: Sequence[str],
    first_step: int,
) -> list[dict[str, object]]:
    """Map the first two independent draws of each batch to the render pools."""

    if not camera_names or not light_names:
        raise ValueError("camera and light pools must be non-empty")
    records: list[dict[str, object]] = []
    for offset, draws in enumerate(draw_batches):
        if len(draws) < 2:
            raise ValueError("each training batch must contain at least two draws")
        camera_draw = int(draws[0])
        light_draw = int(draws[1])
        camera_index = camera_draw % len(camera_names)
        light_index = light_draw % len(light_names)
        records.append(
            {
                "step": first_step + offset,
                "camera_draw": camera_draw,
                "light_draw": light_draw,
                "camera_index": camera_index,
                "light_index": light_index,
                "camera_name": str(camera_names[camera_index]),
                "light_name": str(light_names[light_index]),
            }
        )
    return records


def camera_light_alignment(
    *,
    camera_eye: Sequence[float],
    light_position: Sequence[float],
    target: Sequence[float],
) -> dict[str, object]:
    """Describe the light position relative to the camera-facing hemisphere."""

    camera = [float(camera_eye[index]) - float(target[index]) for index in range(3)]
    light = [float(light_position[index]) - float(target[index]) for index in range(3)]
    camera_length = math.sqrt(sum(value * value for value in camera))
    light_length = math.sqrt(sum(value * value for value in light))
    if camera_length <= 0.0 or light_length <= 0.0:
        raise ValueError("camera and light must not coincide with the target")
    cosine = sum(a * b for a, b in zip(camera, light)) / (
        camera_length * light_length
    )
    cosine = max(-1.0, min(1.0, cosine))
    return {
        "cosine": cosine,
        "angle_degrees": math.degrees(math.acos(cosine)),
        "hemisphere": "camera_side" if cosine >= 0.0 else "opposite_side",
    }


def masked_reference_statistics(
    hdr: np.ndarray,
    mask: np.ndarray,
    *,
    dark_threshold: float = 0.02,
) -> dict[str, object]:
    """Summarize only visible helmet pixels from a linear-HDR reference render."""

    hdr = np.asarray(hdr, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    if hdr.ndim != 3 or hdr.shape[-1] != 3 or mask.shape != hdr.shape[:2]:
        raise ValueError("HDR and mask shapes are incompatible")
    pixels = hdr[mask]
    if pixels.size == 0:
        raise ValueError("reference mask contains no visible pixels")
    luminance = pixels @ np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return {
        "valid_pixels": int(pixels.shape[0]),
        "mean_rgb": [float(value) for value in pixels.mean(axis=0)],
        "luminance_mean": float(luminance.mean()),
        "luminance_p10": float(np.quantile(luminance, 0.10)),
        "luminance_p50": float(np.quantile(luminance, 0.50)),
        "luminance_p90": float(np.quantile(luminance, 0.90)),
        "dark_threshold": float(dark_threshold),
        "dark_fraction": float(np.mean(luminance < dark_threshold)),
    }


def evaluate_camera_relative_audit(
    records: Sequence[Mapping[str, object]],
    *,
    camera_count: int,
    light_count: int,
) -> dict[str, object]:
    """Evaluate the frozen 31x6 camera-relative visibility gates."""

    expected_pairs = camera_count * light_count
    pairs = {
        (int(record["camera_index"]), int(record["light_index"]))
        for record in records
    }
    if len(records) != expected_pairs or len(pairs) != expected_pairs:
        raise ValueError("camera-relative audit requires every camera/light pair exactly once")
    opposite = [record for record in records if record["hemisphere"] == "opposite_side"]
    non_rim = [record for record in records if record["role"] != "rim"]
    if not non_rim:
        raise ValueError("camera-relative audit requires non-rim light families")
    non_rim_opposite = [
        record for record in non_rim if record["hemisphere"] == "opposite_side"
    ]
    mean_dark = float(
        np.mean([float(record["reference"]["dark_fraction"]) for record in non_rim])
    )
    mean_positive = float(
        np.mean(
            [float(record["direct_light"]["mean_positive"]) for record in non_rim]
        )
    )
    opposite_probability = len(opposite) / len(records)
    gates = {
        "weighted_opposite_probability": opposite_probability <= (1.0 / 6.0 + 1.0e-12),
        "no_non_rim_opposite": not non_rim_opposite,
        "non_rim_mean_dark_fraction": mean_dark <= 0.70,
        "non_rim_mean_positive_n_dot_l": mean_positive >= 0.30,
    }
    return {
        "pair_count": len(records),
        "weighted_opposite_probability": opposite_probability,
        "non_rim_opposite_count": len(non_rim_opposite),
        "non_rim_mean_dark_fraction": mean_dark,
        "non_rim_mean_positive_n_dot_l": mean_positive,
        "gates": gates,
        "gates_passed": all(gates.values()),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_path(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty repository path")
    path = (REPOSITORY_ROOT / value).resolve()
    if not path.is_relative_to(REPOSITORY_ROOT):
        raise ValueError(f"{label} escapes the repository")
    return path


def _mapping(path: Path, label: str) -> Mapping[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _camera_from_spec(
    spec: Mapping[str, object], render: Mapping[str, object]
) -> object:
    target_value = spec.get("target", render["target"])
    up_value = render["up"]
    if not isinstance(target_value, Sequence) or not isinstance(up_value, Sequence):
        raise ValueError("camera target/up must be sequences")
    return orbit_camera(
        yaw_degrees=float(spec["yaw_degrees"]),
        elevation_degrees=float(spec["elevation_degrees"]),
        radius=float(spec.get("radius", render["camera_radius"])),
        target=tuple(float(value) for value in target_value),
        up=tuple(float(value) for value in up_value),
        vertical_fov_degrees=float(render["vertical_fov_degrees"]),
        near=float(render["near"]),
        far=float(render["far"]),
    )


def _light_from_spec(spec: Mapping[str, object]) -> PointLight:
    position = spec["position"]
    color = spec["color"]
    if not isinstance(position, Sequence) or not isinstance(color, Sequence):
        raise ValueError("light position/color must be sequences")
    return PointLight(
        position=tuple(float(value) for value in position),
        color=tuple(float(value) for value in color),
        radiant_intensity=float(spec["radiant_intensity"]),
        ambient_intensity=float(spec["ambient_intensity"]),
    )


def _display_encode(linear: torch.Tensor, exposure: float) -> np.ndarray:
    mapped = linear.clamp_min(0.0) * exposure
    mapped = mapped / (1.0 + mapped)
    encoded = linear_to_srgb_torch(mapped).clamp(0.0, 1.0)
    return np.rint(encoded.detach().cpu().numpy() * 255.0).astype(np.uint8)


def _direct_light_statistics(
    geometry: object, material: object, light: PointLight
) -> dict[str, float]:
    buffers = geometry.torch_buffers
    position = buffers["position_world"]
    mask = buffers["mask"]
    light_position = torch.tensor(
        light.position, dtype=position.dtype, device=position.device
    )
    direction = torch.nn.functional.normalize(light_position - position, dim=-1)
    n_dot_l = torch.sum(material.normal_world * direction, dim=-1)[mask]
    return {
        "positive_fraction": float((n_dot_l > 0.0).float().mean().cpu()),
        "mean_positive": float(n_dot_l.clamp_min(0.0).mean().cpu()),
        "p50": float(torch.quantile(n_dot_l, 0.50).cpu()),
    }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replay post-80k affine training camera/light samples as references."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument("--columns", type=int, default=4)
    return parser.parse_args()


def build_sample_bundle(
    samples: Sequence[dict[str, object]],
    *,
    columns: int = 4,
) -> tuple[dict[str, object], dict[str, bytes]]:
    """Build individual PNG renders and one labeled contact sheet in memory."""

    if not samples or columns <= 0:
        raise ValueError("sample export requires images and positive columns")
    first = np.asarray(samples[0]["display_rgb"])
    if first.ndim != 3 or first.shape[2] != 3 or first.dtype != np.uint8:
        raise ValueError("display_rgb must be an HxWx3 uint8 array")
    height, width = first.shape[:2]
    label_height = 24
    rows = math.ceil(len(samples) / columns)
    sheet = Image.new("RGB", (width * columns, (height + label_height) * rows))
    draw = ImageDraw.Draw(sheet)
    exported: list[dict[str, object]] = []
    files: dict[str, bytes] = {}
    for index, sample in enumerate(samples):
        array = np.asarray(sample["display_rgb"])
        if array.shape != first.shape or array.dtype != np.uint8:
            raise ValueError("all display renders must share shape and uint8 dtype")
        step = int(sample["step"])
        camera_name = str(sample["camera_name"])
        light_name = str(sample["light_name"])
        file_name = f"sample-{step:06d}-{camera_name}__{light_name}.png"
        image = Image.fromarray(array, mode="RGB")
        stream = io.BytesIO()
        image.save(stream, format="PNG")
        files[file_name] = stream.getvalue()
        row, column = divmod(index, columns)
        x = column * width
        y = row * (height + label_height)
        sheet.paste(image, (x, y + label_height))
        draw.text((x + 3, y + 3), f"{step} {camera_name} / {light_name}", fill="white")
        exported.append(
            {
                **{key: value for key, value in sample.items() if key != "display_rgb"},
                "file": file_name,
                "sha256": hashlib.sha256(files[file_name]).hexdigest(),
            }
        )
    sheet_stream = io.BytesIO()
    sheet.save(sheet_stream, format="PNG")
    files["contact_sheet.png"] = sheet_stream.getvalue()
    manifest = {
        "sample_count": len(samples),
        "samples": exported,
        "contact_sheet": {
            "file": "contact_sheet.png",
            "sha256": hashlib.sha256(files["contact_sheet.png"]).hexdigest(),
            "size": list(sheet.size),
            "columns": columns,
        },
    }
    return manifest, files


def export_sample_bundle(
    samples: Sequence[dict[str, object]],
    output_dir: Path | str,
    *,
    columns: int = 4,
) -> dict[str, object]:
    """Write a previously validated in-memory sample bundle to a fresh directory."""

    manifest, files = build_sample_bundle(samples, columns=columns)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    for name, payload in files.items():
        (output_dir / name).write_bytes(payload)
    return manifest


def main() -> int:
    """Replay the exact post-80k sampler state without updating model parameters."""

    args = _arguments()
    try:
        if args.sample_count <= 0:
            raise ValueError("sample-count must be positive")
        config_path = args.config.resolve()
        checkpoint_path = args.checkpoint.resolve()
        config = _mapping(config_path, "affine audit config")
        source = config.get("source")
        if not isinstance(source, Mapping):
            raise ValueError("config source section is invalid")
        render_pool_path = _repo_path(
            source["render_pool_config"], "source.render_pool_config"
        )
        if _sha256(render_pool_path) != str(source["render_pool_config_sha256"]):
            raise ValueError("render pool hash does not match the completed run")
        preflight_path = _repo_path(source["preflight_config"], "source.preflight_config")
        render_pool = _mapping(render_pool_path, "render pool")
        preflight = _mapping(preflight_path, "preflight config")
        render = render_pool.get("render")
        camera_specs = render_pool.get("train_cameras")
        light_specs = render_pool.get("train_lights")
        if not (
            isinstance(render, Mapping)
            and isinstance(camera_specs, Sequence)
            and isinstance(light_specs, Sequence)
            and len(camera_specs) == 31
            and len(light_specs) == 6
        ):
            raise ValueError("expected the frozen camera31/light6 render pool")
        if not torch.cuda.is_available():
            raise RuntimeError("render sampling audit requires CUDA")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if (
            int(checkpoint.get("optimizer_updates", -1)) != 80_000
            or checkpoint.get("parent_p0_hash") != source.get("parent_p0_hash")
        ):
            raise ValueError("checkpoint is not the completed chroma8 camera31 run")

        device = torch.device("cuda")
        gltf_path = _repo_path(preflight["inputs"]["gltf"], "inputs.gltf")
        core4_dir = _repo_path(preflight["inputs"]["core4_dir"], "inputs.core4_dir")
        core4_manifest = _repo_path(
            preflight["inputs"]["core4_manifest"], "inputs.core4_manifest"
        )
        mesh = load_gltf_mesh(gltf_path)
        targets = load_core4_targets(core4_dir, "cpu")
        valid_mask, _ = rasterize_uv_charts(
            mesh.texcoords,
            mesh.triangles,
            height=targets.height,
            width=targets.width,
        )
        texel_count = int(valid_mask.sum())
        training = preflight.get("training")
        if not isinstance(training, Mapping):
            raise ValueError("preflight training section is invalid")
        batch_size = int(training["material_batch_size"])
        generator = torch.Generator(device=device)
        generator.set_state(checkpoint["core_rng_state"])
        draw_batches: list[list[int]] = []
        for _ in range(args.sample_count):
            draws = torch.randint(
                0,
                texel_count,
                (batch_size,),
                generator=generator,
                device=device,
            )
            draw_batches.append([int(value) for value in draws[:2].cpu().tolist()])
        records = plan_sample_pairs(
            draw_batches,
            camera_names=[str(value["name"]) for value in camera_specs],
            light_names=[str(value["name"]) for value in light_specs],
            first_step=80_001,
        )

        cameras = [_camera_from_spec(value, render) for value in camera_specs]
        lights = [_light_from_spec(value) for value in light_specs]
        full_pool_alignment: list[dict[str, object]] = []
        for camera_index, camera in enumerate(cameras):
            for light_index, light in enumerate(lights):
                full_pool_alignment.append(
                    {
                        "camera_index": camera_index,
                        "camera_name": str(camera_specs[camera_index]["name"]),
                        "light_index": light_index,
                        "light_name": str(light_specs[light_index]["name"]),
                        **camera_light_alignment(
                            camera_eye=camera.eye,
                            light_position=light.position,
                            target=camera.target,
                        ),
                    }
                )

        textures = load_core4_textures(core4_manifest, device)
        resolution = tuple(int(value) for value in render["resolution"])
        minimum_roughness = float(render["minimum_roughness"])
        exposure = float(render["display_exposure"])
        geometry_cache: dict[int, object] = {}
        samples: list[dict[str, object]] = []
        with torch.no_grad():
            for record in records:
                camera_index = int(record["camera_index"])
                light_index = int(record["light_index"])
                camera = cameras[camera_index]
                light = lights[light_index]
                geometry = geometry_cache.get(camera_index)
                if geometry is None:
                    geometry = render_geometry_gbuffer(
                        mesh, camera, resolution, device=device
                    )
                    geometry_cache[camera_index] = geometry
                material = sample_core4_material(geometry, textures)
                hdr = shade_ggx(
                    geometry,
                    camera,
                    light,
                    material_override=material,
                    minimum_roughness=minimum_roughness,
                )
                mask = geometry.torch_buffers["mask"]
                alignment = camera_light_alignment(
                    camera_eye=camera.eye,
                    light_position=light.position,
                    target=camera.target,
                )
                samples.append(
                    {
                        **record,
                        "camera_eye": [float(value) for value in camera.eye],
                        "camera_target": [float(value) for value in camera.target],
                        "light_position": [float(value) for value in light.position],
                        "alignment": alignment,
                        "direct_light": _direct_light_statistics(
                            geometry, material, light
                        ),
                        "reference": masked_reference_statistics(
                            hdr.detach().cpu().numpy(), mask.detach().cpu().numpy()
                        ),
                        "display_rgb": _display_encode(hdr, exposure),
                    }
                )

        export = export_sample_bundle(samples, args.output_dir.resolve(), columns=args.columns)
        sampled_opposite = [
            sample for sample in export["samples"]
            if sample["alignment"]["hemisphere"] == "opposite_side"
        ]
        by_light: dict[str, dict[str, object]] = {}
        for light_name in [str(value["name"]) for value in light_specs]:
            selected = [
                sample for sample in export["samples"] if sample["light_name"] == light_name
            ]
            by_light[light_name] = {
                "count": len(selected),
                "mean_reference_luminance": (
                    sum(float(sample["reference"]["luminance_mean"]) for sample in selected)
                    / len(selected)
                    if selected
                    else None
                ),
                "opposite_side_count": sum(
                    sample["alignment"]["hemisphere"] == "opposite_side"
                    for sample in selected
                ),
            }
        manifest = {
            "schema_version": 1,
            "experiment": "scifihelmet_affine_render_sampling_audit_v1",
            "source": {
                "config": str(config_path.relative_to(REPOSITORY_ROOT)).replace("\\", "/"),
                "config_sha256": _sha256(config_path),
                "render_pool": str(render_pool_path.relative_to(REPOSITORY_ROOT)).replace("\\", "/"),
                "render_pool_sha256": _sha256(render_pool_path),
                "checkpoint": str(checkpoint_path.relative_to(REPOSITORY_ROOT)).replace("\\", "/"),
                "checkpoint_sha256": _sha256(checkpoint_path),
                "checkpoint_step": 80_000,
                "parent_p0_hash": checkpoint["parent_p0_hash"],
                "rng_position": "exact_post_step_80000_core_rng_state",
            },
            "sampler": {
                "texel_count": texel_count,
                "batch_size": batch_size,
                "camera_count": len(cameras),
                "light_count": len(lights),
                "selection": "first_draw_mod_camera_count__second_draw_mod_light_count",
                "sample_steps": [int(value["step"]) for value in records],
            },
            "render": {
                "resolution": list(resolution),
                "display_exposure": exposure,
                "tone_mapping": "Reinhard x/(1+x)",
                "minimum_roughness": minimum_roughness,
            },
            "summary": {
                "sample_count": len(samples),
                "sampled_opposite_side_count": len(sampled_opposite),
                "sampled_opposite_side_fraction": len(sampled_opposite) / len(samples),
                "full_pool_opposite_side_count": sum(
                    value["hemisphere"] == "opposite_side"
                    for value in full_pool_alignment
                ),
                "full_pool_pair_count": len(full_pool_alignment),
                "by_light": by_light,
            },
            "full_pool_alignment": full_pool_alignment,
            "export": export,
            "formal_holdout_accessed": False,
            "optimizer_updates_performed": 0,
        }
        manifest_path = args.output_dir.resolve() / "sampling_audit.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest["sampling_audit_sha256"] = _sha256(manifest_path)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as error:
        print(f"render sampling audit failed: {error}", file=sys.stderr)
        return 2

    print(json.dumps(manifest["summary"], ensure_ascii=False, sort_keys=True))
    print(f"output: {args.output_dir.resolve()}")
    print(f"manifest_sha256: {manifest['sampling_audit_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
