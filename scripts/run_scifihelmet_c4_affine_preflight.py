"""Run the bounded SciFiHelmet C4-affine M5 P0/CUDA/timing preflight."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Mapping

import torch
import torch.nn.functional as F
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cg_frontier.assets.gltf_mesh import load_gltf_mesh
from cg_frontier.compression.affine_material import (
    AFFINE_STATIC_COST,
    SafeAffineMaterialDecoder,
    certify_affine,
    decode_affine_material,
)
from cg_frontier.compression.affine_pca import (
    P0AffineArtifact,
    P0Bundle,
    P0Calibration,
    export_p0_bundle,
    rasterize_uv_charts,
)
from cg_frontier.compression.affine_regularizers import (
    calibrate_gradient_ratios,
    chart_aware_quantized_tv,
    fake_quantize_unorm8,
)
from cg_frontier.compression.affine_training import (
    AffineCandidateState,
    AffineTrainingBatch,
    candidate_manifest,
    checkpoint_candidate,
    create_paired_candidates,
    resume_candidate,
    select_render_pair,
    time_candidate_steps,
    train_candidate_step,
    write_candidate_checkpoint,
)
from cg_frontier.compression.material import Core4Targets, load_core4_targets
from cg_frontier.compression.render_loss import (
    bilinear_sample_top_down_wrap,
    decoded_to_material,
    masked_render_metrics,
)
from cg_frontier.render.canonical_cube import (
    build_canonical_cube,
    cube_tangent_normal_to_world,
    masked_cube_l1,
    sample_cube_atlas,
)
from cg_frontier.render.gbuffer import (
    Camera,
    GBufferResult,
    MaterialBuffers,
    load_core4_textures,
    render_geometry_gbuffer,
    sample_core4_material,
)
from cg_frontier.render.pbr import PointLight, shade_ggx


def _repo_path(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty repository path")
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"{label} escapes the repository")
    lowered = path.as_posix().lower()
    if "formal_holdout" in lowered or "sealed" in lowered:
        raise ValueError(f"{label} points at forbidden evaluation state")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n"
    ).encode("utf-8")


def _write_new(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(payload)


def _camera(value: Mapping[str, object]) -> Camera:
    return Camera(
        eye=tuple(float(item) for item in value["eye"]),
        target=tuple(float(item) for item in value["target"]),
        up=tuple(float(item) for item in value["up"]),
        vertical_fov_degrees=float(value["vertical_fov_degrees"]),
        near=float(value["near"]),
        far=float(value["far"]),
    )


def _light(value: Mapping[str, object]) -> PointLight:
    return PointLight(
        position=tuple(float(item) for item in value["position"]),
        color=tuple(float(item) for item in value["color"]),
        radiant_intensity=float(value["radiant_intensity"]),
        ambient_intensity=float(value["ambient_intensity"]),
    )


def _targets_to_seven(targets: Core4Targets) -> torch.Tensor:
    return torch.cat(
        (
            targets.base_color_linear,
            targets.normal_xyz[:, :2],
            targets.roughness,
            targets.metallic,
        ),
        dim=-1,
    ).reshape(targets.height, targets.width, 7)


def _move_artifact(artifact: P0AffineArtifact, device: torch.device) -> P0AffineArtifact:
    return replace(
        artifact,
        latent_unorm8=artifact.latent_unorm8.to(device),
        weight=artifact.weight.to(device),
        bias=artifact.bias.to(device),
    )


def _move_p0(p0: P0Calibration, device: torch.device) -> P0Calibration:
    raw = _move_artifact(p0.raw, device)
    safe = _move_artifact(p0.safe, device)
    decoder = SafeAffineMaterialDecoder.from_safe_affine(
        safe.weight, safe.bias, margin=float(safe.certificate["margin"])
    )
    return P0Calibration(
        raw=raw,
        safe=safe,
        safe_decoder=decoder,
        safety_material_mae_increment=p0.safety_material_mae_increment,
    )


def _material_terms(
    decoded: object,
    target: Core4Targets,
    weights: Mapping[str, object],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    terms = {
        "base_color_l1": F.l1_loss(decoded.base_color_linear, target.base_color_linear),
        "normal_cosine": torch.mean(
            1.0 - torch.sum(decoded.normal_xyz * target.normal_xyz, dim=-1)
        ),
        "roughness_l1": F.l1_loss(decoded.roughness, target.roughness),
        "metallic_l1": F.l1_loss(decoded.metallic, target.metallic),
    }
    total = sum(terms[name] * float(weights[name]) for name in terms)
    return total, terms


class PreflightObjective:
    def __init__(
        self,
        *,
        target: Core4Targets,
        valid_indices: torch.Tensor,
        valid_mask: torch.Tensor,
        chart_ids: torch.Tensor,
        helmet_geometries: list[GBufferResult],
        helmet_cameras: list[Camera],
        helmet_references: list[torch.Tensor],
        helmet_light: PointLight,
        reference_textures: dict[str, torch.Tensor],
        cube_resolution: int,
        cube_config: Mapping[str, object],
        loss_config: Mapping[str, object],
        tv_lambda: float = 0.0,
        cube_lambda: float = 0.0,
        helmet_lights: list[PointLight] | None = None,
        helmet_reference_grid: list[list[torch.Tensor]] | None = None,
        helmet_light_grid: Sequence[Sequence[PointLight]] | None = None,
        on_render_pair: Callable[[int, int], None] | None = None,
    ) -> None:
        self.target = target
        self.valid_indices = valid_indices
        self.valid_mask = valid_mask
        self.chart_ids = chart_ids
        self.helmet_geometries = helmet_geometries
        self.helmet_cameras = helmet_cameras
        self.helmet_references = helmet_references
        self.helmet_light = helmet_light
        self.helmet_lights = list(helmet_lights) if helmet_lights is not None else [helmet_light]
        self.helmet_light_grid = (
            [list(row) for row in helmet_light_grid]
            if helmet_light_grid is not None
            else None
        )
        self.on_render_pair = on_render_pair
        self.helmet_reference_grid = (
            helmet_reference_grid
            if helmet_reference_grid is not None
            else [helmet_references]
        )
        if not self.helmet_lights or len(self.helmet_reference_grid) != len(self.helmet_lights):
            raise ValueError("helmet light/reference pools must be non-empty and aligned")
        if any(len(row) != len(helmet_geometries) for row in self.helmet_reference_grid):
            raise ValueError("every helmet light requires one reference per camera")
        if self.helmet_light_grid is not None and (
            len(self.helmet_light_grid) != len(self.helmet_lights)
            or any(len(row) != len(helmet_geometries) for row in self.helmet_light_grid)
        ):
            raise ValueError("camera-relative light grid must align with light families and cameras")
        self.reference_textures = reference_textures
        self.cube = build_canonical_cube(
            resolution=cube_resolution,
            dtype=target.base_color_linear.dtype,
            device=target.base_color_linear.device,
        )
        self.cube_config = cube_config
        self.loss_config = loss_config
        self.tv_lambda = float(tv_lambda)
        self.cube_lambda = float(cube_lambda)
        self.height = target.height
        self.width = target.width
        self._cube_reference = {
            name: sample_cube_atlas(texture, valid_mask, self.cube)
            for name, texture in reference_textures.items()
        }

    def resolve_helmet_light(
        self, *, camera_index: int, light_index: int
    ) -> PointLight:
        """Resolve a family draw to either its camera-relative or legacy global light."""

        if self.helmet_light_grid is not None:
            return self.helmet_light_grid[light_index][camera_index]
        return self.helmet_lights[light_index]

    def observe_render_pair(
        self, *, camera_index: int, light_index: int
    ) -> tuple[int, int]:
        """Report the selected pair without transforming either index."""

        pair = (camera_index, light_index)
        if self.on_render_pair is not None:
            self.on_render_pair(*pair)
        return pair

    def _base(
        self,
        latent: torch.Tensor,
        decoder: SafeAffineMaterialDecoder,
        core_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        flat_indices = self.valid_indices[core_indices]
        deployed = fake_quantize_unorm8(latent)
        decoded_texels = decoder(deployed.reshape(-1, 4)[flat_indices])
        material_total, material_terms = _material_terms(
            decoded_texels, self.target.select(flat_indices), self.loss_config
        )
        camera_index, light_index = select_render_pair(
            core_indices[:2].detach().cpu().tolist(),
            camera_count=len(self.helmet_geometries),
            light_count=len(self.helmet_lights),
        )
        camera_index, light_index = self.observe_render_pair(
            camera_index=camera_index, light_index=light_index
        )
        geometry = self.helmet_geometries[camera_index]
        sampled = bilinear_sample_top_down_wrap(
            deployed, geometry.torch_buffers["uv"]
        )
        material = decoded_to_material(geometry, decoder(sampled))
        candidate_hdr = shade_ggx(
            geometry,
            self.helmet_cameras[camera_index],
            self.resolve_helmet_light(
                camera_index=camera_index, light_index=light_index
            ),
            material_override=material,
            minimum_roughness=float(self.loss_config.get("minimum_roughness", 0.045)),
        )
        reference_hdr = self.helmet_reference_grid[light_index][camera_index]
        mask = geometry.torch_buffers["mask"]
        difference = candidate_hdr[mask] - reference_hdr[mask]
        helmet_charbonnier = torch.sqrt(
            difference.square()
            + float(self.loss_config["helmet_charbonnier_epsilon"]) ** 2
        ).mean()
        helmet_log1p = torch.abs(
            torch.log1p(candidate_hdr[mask].clamp_min(0.0))
            - torch.log1p(reference_hdr[mask].clamp_min(0.0))
        ).mean()
        total = (
            material_total
            + helmet_charbonnier * float(self.loss_config["helmet_charbonnier"])
            + helmet_log1p * float(self.loss_config["helmet_log1p"])
        )
        return total, {
            **material_terms,
            "material_total": material_total,
            "helmet_charbonnier": helmet_charbonnier,
            "helmet_log1p": helmet_log1p,
        }

    def _cube_light(self, samples: torch.Tensor) -> PointLight:
        azimuth = float(samples[0, 0].detach().cpu()) * 2.0 * math.pi
        height_alpha = float(samples[0, 1].detach().cpu())
        radius = float(self.cube_config["light_radius"])
        low = float(self.cube_config["light_height_min"])
        high = float(self.cube_config["light_height_max"])
        return PointLight(
            position=(radius * math.cos(azimuth), low + (high - low) * height_alpha, radius * math.sin(azimuth)),
            color=(1.0, 0.98, 0.95),
            radiant_intensity=35.0,
            ambient_intensity=0.04,
        )

    def _cube_loss(
        self,
        latent: torch.Tensor,
        decoder: SafeAffineMaterialDecoder,
        cube_samples: torch.Tensor,
    ) -> torch.Tensor:
        deployed = fake_quantize_unorm8(latent)
        sampled_latent = sample_cube_atlas(deployed, self.valid_mask, self.cube)
        decoded = decoder(sampled_latent.material)
        reference_normal = F.normalize(
            self._cube_reference["normal"].material, dim=-1, eps=1.0e-8
        )
        decoded_world = (
            self.cube.tangents[:, None, None, :] * decoded.normal_xyz[..., 0:1]
            + self.cube.bitangents[:, None, None, :] * decoded.normal_xyz[..., 1:2]
            + self.cube.normals[:, None, None, :] * decoded.normal_xyz[..., 2:3]
        )
        reference_world = (
            self.cube.tangents[:, None, None, :] * reference_normal[..., 0:1]
            + self.cube.bitangents[:, None, None, :] * reference_normal[..., 1:2]
            + self.cube.normals[:, None, None, :] * reference_normal[..., 2:3]
        )
        light = self._cube_light(cube_samples)
        candidate_faces: list[torch.Tensor] = []
        reference_faces: list[torch.Tensor] = []
        for face in range(6):
            valid = sampled_latent.valid[face]
            position = self.cube.positions[face]
            normal = self.cube.normals[face].expand_as(position)
            tangent = self.cube.tangents[face].expand_as(position)
            bitangent = self.cube.bitangents[face].expand_as(position)
            geometry = GBufferResult(
                buffers={},
                torch_buffers={
                    "mask": valid,
                    "uv": self.cube.uv[face],
                    "position_world": position,
                    "vertex_normal_world": normal,
                    "tangent_world": tangent,
                    "bitangent_world": bitangent,
                },
                metadata={"canonical_cube_face": self.cube.face_names[face]},
            )
            camera = Camera(
                eye=tuple(float(value) for value in self.cube.camera_positions[face].detach().cpu()),
                target=(0.0, 0.0, 0.0),
                up=tuple(float(value) for value in self.cube.camera_up[face].detach().cpu()),
                vertical_fov_degrees=45.0,
                near=0.1,
                far=10.0,
            )
            candidate_material = MaterialBuffers(
                base_color_linear=decoded.base_color_linear[face],
                normal_world=decoded_world[face],
                roughness=decoded.roughness[face, ..., 0],
                metallic=decoded.metallic[face, ..., 0],
                normal_ts_raw=decoded.normal_xyz[face],
                normal_ts_unit=decoded.normal_xyz[face],
            )
            reference_material = MaterialBuffers(
                base_color_linear=self._cube_reference["base_color"].material[face],
                normal_world=reference_world[face],
                roughness=self._cube_reference["roughness"].material[face, ..., 0],
                metallic=self._cube_reference["metallic"].material[face, ..., 0],
                normal_ts_raw=reference_normal[face],
                normal_ts_unit=reference_normal[face],
            )
            candidate_faces.append(
                shade_ggx(
                    geometry,
                    camera,
                    light,
                    material_override=candidate_material,
                    minimum_roughness=float(self.loss_config.get("minimum_roughness", 0.045)),
                )
            )
            reference_faces.append(
                shade_ggx(
                    geometry,
                    camera,
                    light,
                    material_override=reference_material,
                    minimum_roughness=float(self.loss_config.get("minimum_roughness", 0.045)),
                )
            )
        return masked_cube_l1(
            torch.stack(candidate_faces),
            torch.stack(reference_faces),
            sampled_latent.valid,
        ).loss

    def __call__(
        self, state: AffineCandidateState, batch: AffineTrainingBatch
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        total, terms = self._base(state.latent, state.decoder, batch.core_indices)
        if state.candidate_id == "L1":
            tv = chart_aware_quantized_tv(
                state.latent, self.valid_mask, self.chart_ids
            ).loss
            total = total + self.tv_lambda * tv
            terms = {**terms, "tv": tv, "weighted_tv": tv * self.tv_lambda}
        elif state.candidate_id == "L2":
            if batch.cube_samples is None:
                raise RuntimeError("L2 requires cube RNG samples")
            cube = self._cube_loss(state.latent, state.decoder, batch.cube_samples)
            total = total + self.cube_lambda * cube
            terms = {**terms, "cube": cube, "weighted_cube": cube * self.cube_lambda}
        return total, terms


def _raw_safe_metrics(
    bundle: P0Bundle,
    target: torch.Tensor,
    valid: torch.Tensor,
    geometry: GBufferResult,
    camera: Camera,
    light: PointLight,
    reference_hdr: torch.Tensor,
    minimum_roughness: float,
) -> dict[str, object]:
    device = geometry.torch_buffers["uv"].device
    latent = bundle.calibration.safe.latent_unorm8.to(
        device=device, dtype=target.dtype
    ) / 255.0
    target = target.to(device)
    valid = valid.to(device)
    metrics: dict[str, object] = {}
    for name, artifact in (("raw", bundle.calibration.raw), ("safe", bundle.calibration.safe)):
        weight = artifact.weight.to(device)
        bias = artifact.bias.to(device)
        seven = F.linear(latent[valid], weight, bias)
        material_mae = float(torch.mean(torch.abs(seven - target[valid])))
        sampled = bilinear_sample_top_down_wrap(latent, geometry.torch_buffers["uv"])
        decoded = decode_affine_material(sampled, weight, bias)
        material = decoded_to_material(geometry, decoded)
        hdr = shade_ggx(
            geometry,
            camera,
            light,
            material_override=material,
            minimum_roughness=minimum_roughness,
        )
        render_metrics = masked_render_metrics(
            reference_hdr,
            hdr,
            geometry.torch_buffers["mask"],
            linear_psnr_data_range=2.0,
            display_exposure=1.5,
        )
        metrics[name] = {
            "artifact_hash": artifact.artifact_hash,
            "material_mae": material_mae,
            "render": render_metrics,
            "certificate": artifact.certificate,
        }
    raw = metrics["raw"]
    safe = metrics["safe"]
    metrics["increment"] = {
        "material_mae": safe["material_mae"] - raw["material_mae"],
        "masked_linear_hdr_mae": safe["render"]["masked_linear_hdr_mae"] - raw["render"]["masked_linear_hdr_mae"],
    }
    return metrics


def run(config_path: Path) -> dict[str, object]:
    config_bytes = config_path.read_bytes()
    config = yaml.safe_load(config_bytes)
    if not isinstance(config, Mapping) or config.get("experiment") != "scifihelmet_c4_affine_v1":
        raise ValueError("unsupported affine preflight config")
    if config["calibration"].get("selected_tv_ratio") is not None or config["calibration"].get("selected_cube_ratio") is not None:
        raise ValueError("preflight must not select TV/cube ratios")
    output_root = _repo_path(config["output_root"], "output_root")
    if output_root.exists():
        raise FileExistsError(f"refusing to inherit preflight output: {output_root}")
    output_root.mkdir(parents=True)

    gltf_path = _repo_path(config["inputs"]["gltf"], "inputs.gltf")
    core4_dir = _repo_path(config["inputs"]["core4_dir"], "inputs.core4_dir")
    core4_manifest = _repo_path(config["inputs"]["core4_manifest"], "inputs.core4_manifest")
    mesh = load_gltf_mesh(gltf_path)
    cpu_targets = load_core4_targets(core4_dir, "cpu")
    target_seven = _targets_to_seven(cpu_targets)
    valid_mask, chart_ids = rasterize_uv_charts(
        mesh.texcoords,
        mesh.triangles,
        height=cpu_targets.height,
        width=cpu_targets.width,
    )
    p0_bundle = export_p0_bundle(
        target_seven,
        valid_mask,
        chart_ids,
        margin=float(config["p0"]["safety_margin"]),
    )
    for name, payload in p0_bundle.files.items():
        _write_new(output_root / "p0-safe" / name, payload)
    _write_new(output_root / "p0-safe" / "manifest.json", _json_bytes(p0_bundle.manifest))
    raw_payload = torch.cat(
        (
            p0_bundle.calibration.raw.weight.float().reshape(-1),
            p0_bundle.calibration.raw.bias.float(),
        )
    ).numpy().tobytes()
    _write_new(output_root / "p0-raw" / "latent_rgba8.png", p0_bundle.files["latent_rgba8.png"])
    _write_new(output_root / "p0-raw" / "decoder.bin", raw_payload)
    _write_new(
        output_root / "p0-raw" / "manifest.json",
        _json_bytes(
            {
                "artifact_id": "p0-raw",
                "artifact_hash": p0_bundle.calibration.raw.artifact_hash,
                "deployable": False,
                "decoder_sha256": hashlib.sha256(raw_payload).hexdigest(),
                "latent_png_sha256": p0_bundle.manifest["hashes"]["latent_png_sha256"],
            }
        ),
    )

    if not torch.cuda.is_available():
        raise RuntimeError("affine M5 preflight requires CUDA")
    device = torch.device("cuda")
    torch.manual_seed(int(config["seed"]))
    torch.cuda.manual_seed_all(int(config["seed"]))
    p0 = _move_p0(p0_bundle.calibration, device)
    targets = Core4Targets(
        base_color_linear=cpu_targets.base_color_linear.to(device),
        normal_xyz=cpu_targets.normal_xyz.to(device),
        roughness=cpu_targets.roughness.to(device),
        metallic=cpu_targets.metallic.to(device),
        height=cpu_targets.height,
        width=cpu_targets.width,
    )
    valid_cuda = valid_mask.to(device)
    charts_cuda = chart_ids.to(device)
    valid_indices = torch.nonzero(valid_cuda.reshape(-1), as_tuple=False)[:, 0]
    reference_textures = {
        "base_color": targets.base_color_linear.reshape(targets.height, targets.width, 3),
        "normal": targets.normal_xyz.reshape(targets.height, targets.width, 3),
        "roughness": targets.roughness.reshape(targets.height, targets.width, 1),
        "metallic": targets.metallic.reshape(targets.height, targets.width, 1),
    }
    render_config = config["render"]
    cameras = [_camera(value) for value in render_config["cameras"]]
    geometries = [
        render_geometry_gbuffer(mesh, camera, tuple(render_config["resolution"]), device=device)
        for camera in cameras
    ]
    textures = load_core4_textures(core4_manifest, device)
    light = _light(render_config["light"])
    reference_hdr = [
        shade_ggx(
            geometry,
            camera,
            light,
            material_override=sample_core4_material(geometry, textures),
            minimum_roughness=float(render_config["minimum_roughness"]),
        ).detach()
        for geometry, camera in zip(geometries, cameras)
    ]
    loss_config = dict(config["loss"])
    loss_config["minimum_roughness"] = float(render_config["minimum_roughness"])
    base_objective = PreflightObjective(
        target=targets,
        valid_indices=valid_indices,
        valid_mask=valid_cuda,
        chart_ids=charts_cuda,
        helmet_geometries=geometries,
        helmet_cameras=cameras,
        helmet_references=reference_hdr,
        helmet_light=light,
        reference_textures=reference_textures,
        cube_resolution=int(config["cube"]["resolution"]),
        cube_config=config["cube"],
        loss_config=loss_config,
    )

    batch_size = int(config["training"]["material_batch_size"])
    core_generator = torch.Generator(device=device).manual_seed(int(config["seed"]))
    cube_generator = torch.Generator(device=device).manual_seed(int(config["seed"]) + 1)
    calibration_batches = []
    for _ in range(8):
        calibration_batches.append(
            {
                "core_indices": torch.randint(
                    0, valid_indices.numel(), (batch_size,), generator=core_generator, device=device
                ),
                "cube_samples": torch.rand((1, 2), generator=cube_generator, device=device),
            }
        )
    calibration_latent = (p0.safe.latent_unorm8.to(torch.float32) / 255.0).detach().requires_grad_(True)
    fixed_decoder = SafeAffineMaterialDecoder.from_safe_affine(
        p0.safe.weight, p0.safe.bias, margin=float(config["p0"]["safety_margin"])
    )

    def calibration_base(latent: torch.Tensor, batch: object) -> torch.Tensor:
        return base_objective._base(latent, fixed_decoder, batch["core_indices"])[0]

    def calibration_tv(latent: torch.Tensor, batch: object) -> torch.Tensor:
        return chart_aware_quantized_tv(latent, valid_cuda, charts_cuda).loss

    def calibration_cube(latent: torch.Tensor, batch: object) -> torch.Tensor:
        return base_objective._cube_loss(latent, fixed_decoder, batch["cube_samples"])

    target_ratios = tuple(float(value) for value in config["calibration"]["target_ratios"])
    tv_calibration = calibrate_gradient_ratios(
        calibration_latent,
        calibration_batches,
        calibration_base,
        calibration_tv,
        target_ratios=target_ratios,
    )
    cube_calibration = calibrate_gradient_ratios(
        calibration_latent,
        calibration_batches,
        calibration_base,
        calibration_cube,
        target_ratios=target_ratios,
    )
    calibration_report = {
        "selected_ratio": None,
        "tv": asdict(tv_calibration),
        "cube": asdict(cube_calibration),
    }
    _write_new(output_root / "calibration.json", _json_bytes(calibration_report))

    config_hash = hashlib.sha256(config_bytes).hexdigest()
    input_hash = hashlib.sha256(
        (_sha256_file(gltf_path) + _sha256_file(core4_manifest)).encode("ascii")
    ).hexdigest()
    training_config = config["training"]

    def new_state(candidate: str) -> AffineCandidateState:
        return create_paired_candidates(
            p0,
            core_seed=int(config["seed"]) + 11,
            cube_seed=int(config["seed"]) + 17,
            config_hash=config_hash,
            input_hash=input_hash,
            latent_learning_rate=float(training_config["latent_learning_rate"]),
            affine_learning_rate=float(training_config["affine_learning_rate"]),
        )[candidate]

    variants = [("L0", None, 0.0)]
    variants.extend(("L1", ratio, tv_calibration.lambdas[ratio]) for ratio in target_ratios)
    variants.extend(("L2", ratio, cube_calibration.lambdas[ratio]) for ratio in target_ratios)
    correctness: dict[str, object] = {}
    timings: dict[str, object] = {}
    for candidate, ratio, coefficient in variants:
        label = candidate if ratio is None else f"{candidate}-r{int(round(ratio * 100)):03d}"
        objective = PreflightObjective(
            target=targets,
            valid_indices=valid_indices,
            valid_mask=valid_cuda,
            chart_ids=charts_cuda,
            helmet_geometries=geometries,
            helmet_cameras=cameras,
            helmet_references=reference_hdr,
            helmet_light=light,
            reference_textures=reference_textures,
            cube_resolution=int(config["cube"]["resolution"]),
            cube_config=config["cube"],
            loss_config=loss_config,
            tv_lambda=coefficient if candidate == "L1" else 0.0,
            cube_lambda=coefficient if candidate == "L2" else 0.0,
        )
        state = new_state(candidate)
        losses = []
        for _ in range(int(training_config["correctness_steps"])):
            result = train_candidate_step(
                state,
                objective,
                texel_count=valid_indices.numel(),
                batch_size=batch_size,
                cube_sample_count=1,
            )
            losses.append(result.loss)
        weight, bias = state.decoder.fold_affine()
        certificate = certify_affine(weight, bias, margin=state.decoder.margin)
        checkpoint_write = write_candidate_checkpoint(
            output_root / "correctness" / label, state, endpoint=False
        )
        correctness[label] = {
            "candidate": candidate,
            "ratio": ratio,
            "lambda": coefficient,
            "losses": losses,
            "all_finite": all(math.isfinite(value) for value in losses),
            "latent_min": float(state.latent.detach().amin().cpu()),
            "latent_max": float(state.latent.detach().amax().cpu()),
            "certificate": certificate,
            "manifest": candidate_manifest(state),
            "checkpoint_hash": checkpoint_write.checkpoint_hash,
        }
        del state
        torch.cuda.empty_cache()

        timing_state = new_state(candidate)
        timing = time_candidate_steps(
            timing_state,
            objective,
            texel_count=valid_indices.numel(),
            batch_size=batch_size,
            cube_sample_count=1,
            warmup_steps=int(training_config["timing_warmup_steps"]),
            measured_steps=int(training_config["timing_measured_steps"]),
        )
        timings[label] = asdict(timing)
        del timing_state, objective
        torch.cuda.empty_cache()

    resume_source = new_state("L0")
    for _ in range(3):
        train_candidate_step(
            resume_source,
            base_objective,
            texel_count=valid_indices.numel(),
            batch_size=batch_size,
            cube_sample_count=1,
        )
    resume_checkpoint = checkpoint_candidate(resume_source)
    resumed = resume_candidate(
        resume_checkpoint,
        p0,
        expected_parent_p0_hash=p0.safe.artifact_hash,
        expected_config_hash=config_hash,
        expected_input_hash=input_hash,
    )
    continuous_step = train_candidate_step(
        resume_source,
        base_objective,
        texel_count=valid_indices.numel(),
        batch_size=batch_size,
        cube_sample_count=1,
    )
    resumed_step = train_candidate_step(
        resumed,
        base_objective,
        texel_count=valid_indices.numel(),
        batch_size=batch_size,
        cube_sample_count=1,
    )
    resume_evidence = {
        "next_batch_exact": torch.equal(
            continuous_step.batch.core_indices, resumed_step.batch.core_indices
        ),
        "next_loss_exact": continuous_step.loss == resumed_step.loss,
        "latent_exact": torch.equal(resume_source.latent, resumed.latent),
        "checkpoint_hash": resume_checkpoint["checkpoint_hash"],
    }

    raw_safe = _raw_safe_metrics(
        p0_bundle,
        target_seven,
        valid_mask,
        geometries[0],
        cameras[0],
        light,
        reference_hdr[0],
        float(render_config["minimum_roughness"]),
    )
    report = {
        "schema_version": 1,
        "experiment": "scifihelmet_c4_affine_v1",
        "config_sha256": config_hash,
        "inputs": {
            "gltf_sha256": _sha256_file(gltf_path),
            "core4_manifest_sha256": _sha256_file(core4_manifest),
            "formal_holdout_accessed": False,
        },
        "atlas": {
            "height": targets.height,
            "width": targets.width,
            "valid_texels": int(valid_mask.sum()),
            "chart_count": int(torch.unique(chart_ids[valid_mask]).numel()),
        },
        "p0": raw_safe,
        "calibration": calibration_report,
        "correctness": correctness,
        "timing": timings,
        "resume": resume_evidence,
        "static_cost": dict(AFFINE_STATIC_COST),
        "selected_tv_ratio": None,
        "selected_cube_ratio": None,
        "authorized_40k": False,
    }
    _write_new(output_root / "preflight_report.json", _json_bytes(report))
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/train/scifihelmet_c4_affine_v1_preflight.yaml",
    )
    args = parser.parse_args()
    report = run(args.config.resolve())
    print(_json_bytes({"output": str(report["experiment"]), "complete": True}).decode(), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
