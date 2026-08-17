"""Train SciFiHelmet N0/BC80/BC90 with deployment-consistent postprocessing."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
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

from cg_frontier.compression.affine_color import (  # noqa: E402
    linear_srgb_to_oklab,
    orthogonal_color_coordinates,
)
from cg_frontier.assets.gltf_mesh import load_gltf_mesh  # noqa: E402
from cg_frontier.compression.affine_pca import P0AffineArtifact, rasterize_uv_charts  # noqa: E402
from cg_frontier.compression.affine_material import AffineDecodedMaterial  # noqa: E402
from cg_frontier.compression.material import load_core4_targets  # noqa: E402
from cg_frontier.compression.affine_regularizers import fake_quantize_unorm8  # noqa: E402
from cg_frontier.compression.basecolor_priority import (  # noqa: E402
    BaseColorObjectiveConfig,
    C4PostprocessConfig,
    basecolor_charbonnier,
    calibrate_basecolor_gradient_budget,
    compose_basecolor_priority_objective,
    decoder_instruction_audit,
    load_basecolor_priority_checkpoint,
    postprocess_affine_output,
    save_basecolor_priority_checkpoint,
)
from cg_frontier.compression.render_loss import (  # noqa: E402
    bilinear_sample_top_down_wrap,
    decoded_to_material,
    hard_quantize_unorm8,
    masked_render_metrics,
    orbit_camera,
    sparse_fake_quantized_bilinear_sample_top_down_wrap,
)
from cg_frontier.render.gbuffer import (  # noqa: E402
    load_core4_textures,
    render_geometry_gbuffer,
    sample_core4_material,
)
from cg_frontier.render.pbr import PointLight, shade_ggx  # noqa: E402
DEFAULT_CONFIG = ROOT / "configs/train/scifihelmet_c4_basecolor_priority_10k_v1.yaml"


@dataclass(frozen=True)
class CandidateSpec:
    candidate_id: str
    objective_id: str
    target_share: float
    compander: bool
    output_root: str


def _orbit_camera_from_spec(
    spec: Mapping[str, object], render: Mapping[str, object]
) -> object:
    target = spec.get("target", render["target"])
    if not isinstance(target, Sequence) or len(target) != 3:
        raise ValueError("camera target must contain three coordinates")
    up = render["up"]
    if not isinstance(up, Sequence) or len(up) != 3:
        raise ValueError("camera up must contain three coordinates")
    return orbit_camera(
        yaw_degrees=float(spec["yaw_degrees"]),
        elevation_degrees=float(spec["elevation_degrees"]),
        radius=float(spec.get("radius", render["camera_radius"])),
        target=tuple(float(value) for value in target),
        up=tuple(float(value) for value in up),
        vertical_fov_degrees=float(render["vertical_fov_degrees"]),
        near=float(render["near"]),
        far=float(render["far"]),
    )


def _light(value: Mapping[str, object]) -> PointLight:
    return PointLight(
        position=tuple(float(item) for item in value["position"]),
        color=tuple(float(item) for item in value["color"]),
        radiant_intensity=float(value["radiant_intensity"]),
        ambient_intensity=float(value["ambient_intensity"]),
    )


def _targets_to_seven(targets: Any) -> torch.Tensor:
    return torch.cat(
        (
            targets.base_color_linear,
            targets.normal_xyz[:, :2],
            targets.roughness,
            targets.metallic,
        ),
        dim=-1,
    ).reshape(targets.height, targets.width, 7)


def _candidate_spec(config: Mapping[str, Any], candidate_id: str) -> CandidateSpec:
    candidates = config.get("candidates")
    if not isinstance(candidates, Mapping) or candidate_id not in candidates:
        raise ValueError(f"unknown candidate: {candidate_id}")
    value = candidates[candidate_id]
    if not isinstance(value, Mapping):
        raise ValueError("candidate specification must be a mapping")
    objective_id = str(value.get("objective_id"))
    target_share = float(value.get("target_share", -1.0))
    compander = bool(value.get("compander", False))
    expected: tuple[str, float, bool]
    if candidate_id == "N0-control":
        expected = ("r0_control", 0.0, False)
    elif candidate_id in {"BC80", "BC90"}:
        expected = (
            "basecolor_priority",
            0.8 if candidate_id == "BC80" else 0.9,
            False,
        )
    elif candidate_id in {"BC80-compander", "BC90-compander"}:
        expected = (
            "basecolor_priority_compander",
            0.8 if candidate_id.startswith("BC80") else 0.9,
            True,
        )
    elif candidate_id == "BC-only":
        expected = ("basecolor_only_oracle", 1.0, False)
    else:
        raise ValueError("candidate is outside the frozen C4 matrix")
    if (objective_id, target_share, compander) != expected:
        raise ValueError("candidate identity does not match the frozen C4 matrix")
    output_root = str(value.get("output_root", ""))
    if not output_root:
        raise ValueError("candidate output root is required")
    return CandidateSpec(
        candidate_id=candidate_id,
        objective_id=objective_id,
        target_share=target_share,
        compander=compander,
        output_root=output_root,
    )


def _checkpoint_steps(training: Mapping[str, Any]) -> frozenset[int]:
    total = int(training.get("steps", 0))
    values = [int(value) for value in training.get("checkpoint_steps", [])]
    if (
        total != 10000
        or values != sorted(values)
        or len(values) != len(set(values))
        or values != [1000, 5000, 10000]
        or values[-1] != total
    ):
        raise ValueError("C4 BaseColor checkpoints must be exactly 1k/5k/10k")
    return frozenset(values)


def _canonical_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value: object) -> str:
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n", encoding="ascii")
    return digest


def _load_config(path: Path) -> Mapping[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping) or config.get("schema_version") != 1:
        raise ValueError("unsupported BaseColor-priority config")
    if config.get("formal_holdout_access") != "forbidden":
        raise ValueError("formal holdout must remain forbidden")
    if "support_penalty_weight" in config["training"] or "support" in config["loss"]:
        raise ValueError("support penalty is forbidden in the BaseColor-priority lineage")
    _checkpoint_steps(config["training"])
    expected_loss = {
        "render_linear": 1.0,
        "render_log": 0.25,
        "normal_cosine": 0.25,
        "roughness_l1": 0.5,
        "metallic_l1": 0.5,
        "charbonnier_epsilon": 0.001,
    }
    if config["loss"] != expected_loss:
        raise ValueError("loss weights differ from the frozen R0 residual contract")
    postprocess = config["postprocess"]
    if (
        postprocess.get("scalar_saturate") is not True
        or postprocess.get("normal_disk_projection") is not True
        or postprocess.get("normal_z_reconstruction") is not True
        or postprocess.get("straight_through_backward") is not True
        or float(postprocess.get("compander_gain_min")) != 0.5
        or float(postprocess.get("compander_gain_max")) != 2.0
    ):
        raise ValueError("postprocess config differs from the deployment contract")
    for candidate_id in config["candidates"]:
        _candidate_spec(config, str(candidate_id))
    return config


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_path(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a repository-relative path")
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"{label} escapes the repository")
    return path


def _verify_file(path: Path, expected: object, label: str) -> None:
    if not path.is_file() or _sha256_file(path) != str(expected):
        raise ValueError(f"{label} SHA-256 mismatch")


def _p0_artifact_hash(
    artifact_id: str,
    latent_unorm8: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> str:
    digest = hashlib.sha256(artifact_id.encode("ascii"))
    for tensor in (latent_unorm8, weight, bias):
        host = tensor.detach().cpu().contiguous()
        digest.update(str(host.dtype).encode("ascii"))
        digest.update(str(tuple(host.shape)).encode("ascii"))
        digest.update(host.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _load_frozen_raw_parent(source: Mapping[str, Any]) -> P0AffineArtifact:
    root = _repo_path(source["raw_parent_directory"], "source.raw_parent_directory")
    manifest_path = root / "manifest.json"
    _verify_file(
        manifest_path,
        source["raw_parent_manifest_sha256"],
        "raw parent manifest",
    )
    manifest = _load_mapping(manifest_path)
    artifact_id = str(manifest.get("artifact_id"))
    artifact_hash = str(manifest.get("artifact_hash"))
    if (
        manifest.get("schema_version") != 1
        or artifact_id != "p0-raw"
        or artifact_hash != source["raw_parent_artifact_hash"]
        or manifest.get("deployable") is not False
        or manifest.get("latent_shape") != [2048, 2048, 4]
    ):
        raise ValueError("raw parent manifest differs from the frozen contract")
    latent_path = root / "latent_rgba8.png"
    decoder_path = root / "decoder.bin"
    _verify_file(latent_path, manifest["latent_png_sha256"], "raw parent latent")
    _verify_file(decoder_path, manifest["decoder_sha256"], "raw parent decoder")
    with Image.open(latent_path) as image:
        if image.mode != "RGBA" or image.size != (2048, 2048):
            raise ValueError("raw parent latent must be a 2048x2048 RGBA image")
        latent = torch.from_numpy(np.array(image, dtype=np.uint8, copy=True))
    packed = np.frombuffer(decoder_path.read_bytes(), dtype="<f4").copy()
    if packed.shape != (35,) or not np.isfinite(packed).all():
        raise ValueError("raw parent decoder must contain 35 finite float32 values")
    weight = torch.from_numpy(packed[:28].reshape(7, 4))
    bias = torch.from_numpy(packed[28:])
    if _p0_artifact_hash(artifact_id, latent, weight, bias) != artifact_hash:
        raise ValueError("raw parent artifact hash mismatch")
    return P0AffineArtifact(
        artifact_id=artifact_id,
        latent_unorm8=latent,
        weight=weight,
        bias=bias,
        material_mae=float(manifest["material_mae"]),
        artifact_hash=artifact_hash,
        certificate=None,
    )


def _load_mapping(path: Path) -> Mapping[str, Any]:
    payload = path.read_text(encoding="utf-8")
    value = yaml.safe_load(payload) if path.suffix.lower() in {".yaml", ".yml"} else json.loads(payload)
    if not isinstance(value, Mapping):
        raise ValueError(f"mapping expected: {path}")
    return value


def _prepare(config: Mapping[str, Any], config_path: Path) -> dict[str, Any]:
    """Load the raw-PCA parent and frozen render pool without adaptive-color metadata."""

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    source = config["source"]
    forbidden = {
        "adaptive_profile",
        "adaptive_profile_sha256",
        "basecolor_profile_hash",
        "visibility_hash",
    }
    if forbidden.intersection(source):
        raise ValueError("BaseColor-priority source must not depend on adaptive profiles")
    preflight_path = _repo_path(source["preflight_config"], "source.preflight_config")
    pool_path = _repo_path(source["render_pool_config"], "source.render_pool_config")
    _verify_file(preflight_path, source["preflight_config_sha256"], "preflight config")
    _verify_file(pool_path, source["render_pool_config_sha256"], "render pool config")
    preflight = _load_mapping(preflight_path)
    pool = _load_mapping(pool_path)
    mesh = load_gltf_mesh(_repo_path(preflight["inputs"]["gltf"], "inputs.gltf"))
    core4_dir = _repo_path(preflight["inputs"]["core4_dir"], "inputs.core4_dir")
    core4_manifest = _repo_path(
        preflight["inputs"]["core4_manifest"], "inputs.core4_manifest"
    )
    targets_cpu = load_core4_targets(core4_dir, "cpu")
    target_seven_cpu = _targets_to_seven(targets_cpu)
    valid_mask, chart_ids = rasterize_uv_charts(
        mesh.texcoords,
        mesh.triangles,
        height=targets_cpu.height,
        width=targets_cpu.width,
    )
    raw = _load_frozen_raw_parent(source)
    if raw.latent_unorm8.shape[:2] != (targets_cpu.height, targets_cpu.width):
        raise ValueError("raw parent latent dimensions do not match Core-4 targets")
    render = pool["render"]
    if (
        tuple(render["resolution"]) != (256, 256)
        or len(pool["train_cameras"]) != 31
        or len(pool["train_lights"]) != 6
    ):
        raise ValueError("runner requires frozen camera31/light6 at 256x256")
    device = torch.device("cuda")
    cameras = [_orbit_camera_from_spec(value, render) for value in pool["train_cameras"]]
    geometries = [
        render_geometry_gbuffer(mesh, camera, (256, 256), device=device)
        for camera in cameras
    ]
    lights = [_light(value) for value in pool["train_lights"]]
    textures = load_core4_textures(core4_manifest, device)
    source_materials = [
        sample_core4_material(geometry, textures) for geometry in geometries
    ]
    with torch.no_grad():
        references = [
            [
                shade_ggx(
                    geometry,
                    camera,
                    light,
                    material_override=source_material,
                    minimum_roughness=float(render["minimum_roughness"]),
                )
                for geometry, camera, source_material in zip(
                    geometries, cameras, source_materials, strict=True
                )
            ]
            for light in lights
        ]
    valid_positions_cpu = torch.nonzero(
        valid_mask.reshape(-1), as_tuple=False
    ).flatten()
    valid_indices = valid_positions_cpu.to(device)
    target_valid = target_seven_cpu.reshape(-1, 7).to(device)[valid_indices]
    lineage = {
        "parent_artifact_hash": raw.artifact_hash,
        "config_sha256": _sha256_file(config_path),
        "input_sha256": hashlib.sha256(
            (
                _sha256_file(_repo_path(preflight["inputs"]["gltf"], "inputs.gltf"))
                + _sha256_file(core4_manifest)
            ).encode("ascii")
        ).hexdigest(),
    }
    return {
        "mesh": mesh,
        "textures": textures,
        "raw": raw,
        "render": render,
        "cameras": cameras,
        "geometries": geometries,
        "lights": lights,
        "references": references,
        "source_materials": source_materials,
        "valid_indices": valid_indices,
        "target_valid": target_valid,
        "lineage": lineage,
    }


def _new_state(prepared: Mapping[str, Any], config: Mapping[str, Any]):
    raw = prepared["raw"]
    device = torch.device("cuda")
    latent = nn.Parameter(raw.latent_unorm8.to(device=device, dtype=torch.float32) / 255.0)
    weight = nn.Parameter(raw.weight.to(device=device, dtype=torch.float32).clone())
    bias = nn.Parameter(raw.bias.to(device=device, dtype=torch.float32).clone())
    compander = nn.Parameter(torch.tensor((1.0, 0.0), device=device))
    training = config["training"]
    optimizers = (
        torch.optim.Adam((latent,), lr=float(training["latent_learning_rate"])),
        torch.optim.Adam((weight, bias), lr=float(training["affine_learning_rate"])),
        torch.optim.Adam((compander,), lr=float(training["compander_learning_rate"])),
    )
    rng = torch.Generator(device=device).manual_seed(int(config["seed"]))
    return (latent, weight, bias, compander), optimizers, rng


def _state_hash(state, optimizers, rng) -> str:
    stream = io.BytesIO()
    torch.save(
        {
            "parameters": [value.detach().cpu() for value in state],
            "optimizers": [value.state_dict() for value in optimizers],
            "rng": rng.get_state().cpu(),
        },
        stream,
    )
    return hashlib.sha256(stream.getvalue()).hexdigest()


def _draw_batch(prepared, config, rng):
    device = prepared["valid_indices"].device
    positions = torch.randint(
        0,
        prepared["valid_indices"].numel(),
        (int(config["training"]["material_batch_size"]),),
        generator=rng,
        device=device,
    )
    training_cameras = prepared.get(
        "training_camera_indices", tuple(range(len(prepared["cameras"])))
    )
    camera_position = int(
        torch.randint(0, len(training_cameras), (1,), generator=rng, device=device)
    )
    camera_index = int(training_cameras[camera_position])
    light_index = int(
        torch.randint(0, len(prepared["lights"]), (1,), generator=rng, device=device)
    )
    return positions, camera_index, light_index


def _selected_light(prepared: Mapping[str, Any], camera_index: int, light_index: int):
    camera_lights = prepared.get("camera_lights")
    if camera_lights is not None:
        return camera_lights[camera_index][light_index]
    return prepared["lights"][light_index]


def _material_override(geometry, processed):
    decoded = AffineDecodedMaterial(
        base_color_linear=processed.seven[..., :3],
        normal_xy=processed.seven[..., 3:5],
        normal_xyz=processed.normal_xyz,
        roughness=processed.seven[..., 5:6],
        metallic=processed.seven[..., 6:7],
    )
    return decoded_to_material(geometry, decoded)


def _loss_terms(state, prepared, config, batch, *, compander_enabled: bool):
    latent, weight, bias, compander = state
    positions, camera_index, light_index = batch
    flat_indices = prepared["valid_indices"][positions]
    raw_material = F.linear(
        fake_quantize_unorm8(latent.reshape(-1, 4)[flat_indices]), weight, bias
    )
    material = postprocess_affine_output(
        raw_material,
        compander_parameters=compander if compander_enabled else None,
        straight_through=True,
    )
    target = postprocess_affine_output(
        prepared["target_valid"][positions],
        compander_parameters=None,
        straight_through=False,
    )
    geometry = prepared["geometries"][camera_index]
    sampled = sparse_fake_quantized_bilinear_sample_top_down_wrap(
        latent, geometry.torch_buffers["uv"]
    )
    raw_render_material = F.linear(sampled, weight, bias)
    render_material = postprocess_affine_output(
        raw_render_material,
        compander_parameters=compander if compander_enabled else None,
        straight_through=True,
    )
    candidate = shade_ggx(
        geometry,
        prepared["cameras"][camera_index],
        _selected_light(prepared, camera_index, light_index),
        material_override=_material_override(geometry, render_material),
        minimum_roughness=float(prepared["render"]["minimum_roughness"]),
    )
    mask = geometry.torch_buffers["mask"]
    prediction_rgb = candidate[mask]
    source_rgb = prepared["references"][light_index][camera_index][mask]
    difference = prediction_rgb - source_rgb
    epsilon = float(config["loss"]["charbonnier_epsilon"])
    terms = {
        "base_color_l1": F.l1_loss(material.seven[:, :3], target.seven[:, :3]),
        "base_color_charbonnier": basecolor_charbonnier(
            material.seven[:, :3], target.seven[:, :3], epsilon=epsilon
        ),
        "render_linear": torch.sqrt(difference.square() + epsilon * epsilon).mean(),
        "render_log": torch.abs(
            torch.log1p(prediction_rgb.clamp_min(0.0))
            - torch.log1p(source_rgb.clamp_min(0.0))
        ).mean(),
        "normal_cosine": torch.mean(
            1.0 - (material.normal_xyz * target.normal_xyz).sum(dim=-1)
        ),
        "roughness_l1": F.l1_loss(material.seven[:, 5], target.seven[:, 5]),
        "metallic_l1": F.l1_loss(material.seven[:, 6], target.seven[:, 6]),
    }
    return terms, {
        "camera_index": camera_index,
        "light_index": light_index,
        "raw_material": raw_material,
        "processed_material": material,
        "raw_render_material": raw_render_material,
        "processed_render_material": render_material,
        "mask": mask,
    }


def _residual(terms: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return (
        terms["render_linear"]
        + 0.25 * terms["render_log"]
        + 0.25 * terms["normal_cosine"]
        + 0.5 * terms["roughness_l1"]
        + 0.5 * terms["metallic_l1"]
    )


def run_audit(
    config_path: Path,
    output: Path,
    *,
    prepared_override: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    config = _load_config(config_path)
    if output.exists():
        raise FileExistsError(f"refusing to overwrite audit root: {output}")
    output.mkdir(parents=True)
    prepared = prepared_override or _prepare(config, config_path)
    state, optimizers, core_rng = _new_state(prepared, config)
    before = _state_hash(state, optimizers, core_rng)
    audit_rng = torch.Generator(device="cuda").manual_seed(int(config["seed"]))
    batches = [
        _draw_batch(prepared, config, audit_rng)
        for _ in range(int(config["audit"]["batches"]))
    ]

    def builder(batch):
        terms, _ = _loss_terms(state, prepared, config, batch, compander_enabled=False)
        return terms["base_color_charbonnier"], _residual(terms)

    audits = {
        str(share): calibrate_basecolor_gradient_budget(
            batches,
            builder,
            parameter_groups={
                "latent": (state[0],),
                "affine": (state[1], state[2]),
            },
            target_share=share,
        )
        for share in (0.8, 0.9)
    }
    after = _state_hash(state, optimizers, core_rng)
    if before != after:
        raise RuntimeError("zero-update gradient audit changed training state")
    instruction = {
        "safety_only": decoder_instruction_audit(C4PostprocessConfig(compander=False)),
        "with_compander": decoder_instruction_audit(C4PostprocessConfig(compander=True)),
    }
    report = {
        "schema_version": 1,
        "status": "complete_zero_update_gradient_audit",
        "lineage": prepared["lineage"],
        "parent_artifact_hash": prepared["raw"].artifact_hash,
        "batch_count": len(batches),
        "batch_identity": [
            {
                "positions_sha256": hashlib.sha256(batch[0].cpu().numpy().tobytes()).hexdigest(),
                "camera_index": batch[1],
                "light_index": batch[2],
            }
            for batch in batches
        ],
        "targets": {key: asdict(value) for key, value in audits.items()},
        "instruction_audit": instruction,
        "state_hash_before": before,
        "state_hash_after": after,
        "state_unchanged": True,
        "optimizer_updates": 0,
        "formal_holdout_accessed": False,
        "training_started": False,
        "ue_started": False,
    }
    _write_json(output / "gradient_audit.json", report)
    return report


def _checkpoint_identity(
    config: Mapping[str, Any],
    prepared: Mapping[str, Any],
    candidate: CandidateSpec,
    lambda_value: float,
    actual_shares: Mapping[str, float],
    rig_hash: str,
) -> dict[str, str]:
    objective = {
        "candidate_id": candidate.candidate_id,
        "objective_id": candidate.objective_id,
        "target_share": candidate.target_share,
        "lambda_value": lambda_value,
        "actual_shares": actual_shares,
        "loss": config["loss"],
    }
    postprocess = dict(config["postprocess"], compander=candidate.compander)
    return {
        "parent_hash": prepared["raw"].artifact_hash,
        "input_hash": prepared["lineage"]["input_sha256"],
        "config_hash": prepared["lineage"]["config_sha256"],
        "objective_hash": _canonical_hash(objective),
        "postprocess_hash": _canonical_hash(postprocess),
        "rig_hash": str(rig_hash),
    }


def _repo_output_path(value: Path) -> Path:
    path = (ROOT / value).resolve() if not value.is_absolute() else value.resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("output must remain inside the repository")
    return path


def _audit_target(audit: Mapping[str, Any], candidate: CandidateSpec):
    if candidate.target_share == 0.0:
        return 1.0, {"latent": 0.0, "affine": 0.0}
    if candidate.target_share == 1.0:
        return 1.0, {"latent": 1.0, "affine": 1.0}
    row = audit["targets"][str(candidate.target_share)]
    return float(row["lambda_value"]), {
        name: float(row["groups"][name]["achieved_share"])
        for name in ("latent", "affine")
    }


@torch.no_grad()
def _atlas_metrics(
    state,
    prepared,
    *,
    compander_enabled: bool,
    valid_positions: torch.Tensor | None = None,
) -> dict[str, Any]:
    latent, weight, bias, compander = state
    deployed = hard_quantize_unorm8(latent).reshape(-1, 4)
    prediction_chunks: list[torch.Tensor] = []
    raw_chunks: list[torch.Tensor] = []
    safety_chunks: list[torch.Tensor] = []
    target_chunks: list[torch.Tensor] = []
    gain_chunks: list[torch.Tensor] = []
    selected_indices = (
        prepared["valid_indices"]
        if valid_positions is None
        else prepared["valid_indices"][valid_positions]
    )
    selected_targets = (
        prepared["target_valid"]
        if valid_positions is None
        else prepared["target_valid"][valid_positions]
    )
    for start in range(0, selected_indices.numel(), 262144):
        indices = selected_indices[start : start + 262144]
        raw = F.linear(deployed[indices], weight, bias)
        processed = postprocess_affine_output(
            raw,
            compander_parameters=compander if compander_enabled else None,
            straight_through=False,
        )
        safety_only = postprocess_affine_output(
            raw,
            compander_parameters=None,
            straight_through=False,
        )
        raw_chunks.append(raw.cpu())
        safety_chunks.append(safety_only.seven.cpu())
        prediction_chunks.append(processed.seven.cpu())
        target_chunks.append(selected_targets[start : start + 262144].cpu())
        if processed.compander_gain is not None:
            gain_chunks.append(processed.compander_gain.cpu())
    raw = torch.cat(raw_chunks)
    safety_only = torch.cat(safety_chunks)
    prediction = torch.cat(prediction_chunks)
    target = torch.cat(target_chunks)
    difference = prediction - target
    base_difference = difference[:, :3]
    pred_oklab = linear_srgb_to_oklab(prediction[:, :3])
    source_oklab = linear_srgb_to_oklab(target[:, :3])
    delta_e = torch.linalg.vector_norm(pred_oklab - source_oklab, dim=-1)
    pred_opponent = orthogonal_color_coordinates(prediction[:, :3])
    source_opponent = orthogonal_color_coordinates(target[:, :3])
    pred_chroma = torch.linalg.vector_norm(pred_opponent[:, 1:], dim=-1)
    source_chroma = torch.linalg.vector_norm(source_opponent[:, 1:], dim=-1)
    raw_scalar = raw[:, [0, 1, 2, 5, 6]]
    raw_radius = torch.linalg.vector_norm(raw[:, 3:5], dim=-1)
    angles = torch.rad2deg(
        torch.acos(
            (
                postprocess_affine_output(raw, compander_parameters=None, straight_through=False).normal_xyz
                * postprocess_affine_output(target, compander_parameters=None, straight_through=False).normal_xyz
            ).sum(dim=-1).clamp(-1.0, 1.0)
        )
    )
    metrics: dict[str, Any] = {
        "seven_channel_mae": float(difference.abs().mean()),
        "base_color_linear_mae": float(base_difference.abs().mean()),
        "base_color_linear_rmse": float(torch.sqrt(base_difference.square().mean())),
        "base_color_linear_psnr": float(-10.0 * torch.log10(base_difference.square().mean().clamp_min(1.0e-20))),
        "base_color_charbonnier": float((torch.sqrt(base_difference.square() + 1.0e-6) - 1.0e-3).mean()),
        "oklab_delta_e_mean": float(delta_e.mean()),
        "oklab_delta_e_p95": float(torch.quantile(delta_e, 0.95)),
        "opponent_error": float((pred_opponent[:, 1:] - source_opponent[:, 1:]).abs().mean()),
        "chroma_magnitude_retention": float(pred_chroma.std() / source_chroma.std().clamp_min(1.0e-12)),
        "normal_mean_degrees": float(angles.mean()),
        "roughness_mae": float(difference[:, 5].abs().mean()),
        "metallic_mae": float(difference[:, 6].abs().mean()),
        "pre_safety_scalar_violation_fraction": float(((raw_scalar < 0.0) | (raw_scalar > 1.0)).to(torch.float64).mean()),
        "pre_safety_scalar_violation_max": float(torch.maximum((-raw_scalar).clamp_min(0.0), (raw_scalar - 1.0).clamp_min(0.0)).max()),
        "pre_safety_normal_violation_fraction": float((raw_radius > 1.0).to(torch.float64).mean()),
        "pre_safety_normal_violation_max": float((raw_radius - 1.0).clamp_min(0.0).max()),
        "post_safety_scalar_violation_fraction": 0.0,
        "post_safety_normal_violation_fraction": 0.0,
        "saturate_fraction": float(
            (
                safety_only[:, [0, 1, 2, 5, 6]]
                != raw_scalar
            ).to(torch.float64).mean()
        ),
    }
    if gain_chunks:
        gain = torch.cat(gain_chunks).flatten()
        metrics["compander_gain"] = {
            "min": float(gain.min()),
            "mean": float(gain.mean()),
            "p95": float(torch.quantile(gain, 0.95)),
            "max": float(gain.max()),
        }
    return metrics


@torch.no_grad()
def _render_metrics(state, prepared, pairs, *, compander_enabled: bool) -> dict[str, Any]:
    latent, weight, bias, compander = state
    deployed = hard_quantize_unorm8(latent)
    rows: list[dict[str, Any]] = []
    for camera_index, light_index in pairs:
        geometry = prepared["geometries"][camera_index]
        sampled = bilinear_sample_top_down_wrap(deployed, geometry.torch_buffers["uv"])
        raw = F.linear(sampled, weight, bias)
        processed = postprocess_affine_output(
            raw,
            compander_parameters=compander if compander_enabled else None,
            straight_through=False,
        )
        candidate = shade_ggx(
            geometry,
            prepared["cameras"][camera_index],
            _selected_light(prepared, camera_index, light_index),
            material_override=_material_override(geometry, processed),
            minimum_roughness=float(prepared["render"]["minimum_roughness"]),
        )
        row = masked_render_metrics(
            prepared["references"][light_index][camera_index],
            candidate,
            geometry.torch_buffers["mask"],
            linear_psnr_data_range=float(prepared["render"]["linear_psnr_data_range"]),
            display_exposure=float(prepared["render"]["display_exposure"]),
        )
        row.update({"camera_index": camera_index, "light_index": light_index})
        rows.append(row)
    numeric = [
        key
        for key, value in rows[0].items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    return {
        "case_count": len(rows),
        "mean": {key: float(np.mean([row[key] for row in rows])) for key in numeric},
        "worst": {key: float(max(row[key] for row in rows)) for key in numeric},
        "cases": rows,
    }


def run_candidate(
    config_path: Path,
    *,
    candidate_id: str,
    audit_path: Path,
    output: Path,
    max_steps: int | None = None,
    resume: Path | None = None,
    prepared_override: Mapping[str, Any] | None = None,
    rig_hash_override: str | None = None,
) -> dict[str, Any]:
    config = _load_config(config_path)
    candidate = _candidate_spec(config, candidate_id)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    prepared = prepared_override or _prepare(config, config_path)
    if audit.get("lineage") != prepared["lineage"] or audit.get("state_unchanged") is not True:
        raise ValueError("gradient audit does not match the prepared parent")
    lambda_value, actual_shares = _audit_target(audit, candidate)
    rig_hash = rig_hash_override or str(config["source"]["render_pool_config_sha256"])
    identity = _checkpoint_identity(
        config, prepared, candidate, lambda_value, actual_shares, rig_hash
    )
    if resume is None:
        if output.exists():
            raise FileExistsError(f"refusing to overwrite candidate root: {output}")
        output.mkdir(parents=True)
    elif not output.is_dir():
        raise ValueError("resume requires the existing candidate output root")
    state, optimizers, core_rng = _new_state(prepared, config)
    start_step = 1
    curve: list[dict[str, Any]] = []
    trajectory: list[dict[str, Any]] = []
    sample_metrics: list[dict[str, Any]] = []
    gradient_summaries: list[dict[str, Any]] = []
    if resume is not None:
        payload = load_basecolor_priority_checkpoint(
            resume,
            expected_candidate_id=candidate.candidate_id,
            expected_objective_id=candidate.objective_id,
            expected_target_share=candidate.target_share,
            expected_identity=identity,
        )
        for parameter, field in zip(
            state,
            ("latent", "weight", "bias", "compander_parameters"),
            strict=True,
        ):
            parameter.data.copy_(payload[field].to(parameter))
        for optimizer, field in zip(
            optimizers,
            ("latent_optimizer", "affine_optimizer", "compander_optimizer"),
            strict=True,
        ):
            optimizer.load_state_dict(payload[field])
        core_rng.set_state(payload["core_rng_state"])
        start_step = int(payload["step"]) + 1
        progress = json.loads((output / "progress.json").read_text(encoding="utf-8"))
        curve = list(progress["curve"])
        trajectory = list(progress["trajectory"])
        sample_metrics = list(progress.get("sample_metrics", []))
        gradient_summaries = list(progress.get("gradient_summaries", []))
    formal_steps = int(config["training"]["steps"])
    steps = formal_steps if max_steps is None else min(formal_steps, int(max_steps))
    if start_step > steps:
        raise ValueError("resume checkpoint is at or beyond requested endpoint")
    checkpoint_steps = set(_checkpoint_steps(config["training"]))
    checkpoint_steps.add(steps)
    metric_count = min(262144, prepared["valid_indices"].numel())
    metric_positions = torch.linspace(
        0,
        prepared["valid_indices"].numel() - 1,
        metric_count,
        device=prepared["valid_indices"].device,
    ).round().to(torch.int64)
    started = time.perf_counter()
    last_checkpoint = None
    for step in range(start_step, steps + 1):
        batch = _draw_batch(prepared, config, core_rng)
        terms, context = _loss_terms(
            state, prepared, config, batch, compander_enabled=candidate.compander
        )
        objective = BaseColorObjectiveConfig(
            candidate_id=candidate.candidate_id,
            target_share=candidate.target_share,
            lambda_value=lambda_value,
        )
        total, pieces = compose_basecolor_priority_objective(terms, objective)
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite loss at step {step}")
        for optimizer in optimizers:
            optimizer.zero_grad(set_to_none=True)
        total.backward()
        if step in {1, 1000, 5000, 10000}:
            gradient_summaries.append(
                {
                    "step": 0 if step == 1 else step,
                    "latent": float(
                        torch.linalg.vector_norm(state[0].grad).detach()
                    ),
                    "affine": float(
                        torch.sqrt(
                            state[1].grad.square().sum()
                            + state[2].grad.square().sum()
                        ).detach()
                    ),
                    "compander": (
                        float(torch.linalg.vector_norm(state[3].grad))
                        if state[3].grad is not None
                        else 0.0
                    ),
                }
            )
        optimizers[0].step()
        optimizers[1].step()
        if candidate.compander:
            optimizers[2].step()
        with torch.no_grad():
            state[0].clamp_(0.0, 1.0)
        if step == 1 or step % int(config["training"]["log_interval"]) == 0:
            point = {
                "step": step,
                "loss": float(total.detach()),
                **{name: float(value.detach()) for name, value in terms.items()},
                "weighted_base_color": float(pieces["base_color"].detach()),
                "weighted_residual": float(pieces["residual"].detach()),
                "camera_index": context["camera_index"],
                "light_index": context["light_index"],
                "finite": True,
            }
            curve.append(point)
            print(json.dumps(point, sort_keys=True), flush=True)
        if step % int(config["training"]["metric_interval"]) == 0 or step == steps:
            sample_metrics.append(
                {
                    "step": step,
                    **_atlas_metrics(
                        state,
                        prepared,
                        compander_enabled=candidate.compander,
                        valid_positions=metric_positions,
                    ),
                }
            )
        if step in checkpoint_steps:
            checkpoint = output / "checkpoints" / f"step_{step:05d}" / "checkpoint.pt"
            digest = save_basecolor_priority_checkpoint(
                checkpoint,
                step=step,
                candidate_id=candidate.candidate_id,
                objective_id=candidate.objective_id,
                target_share=candidate.target_share,
                lambda_value=lambda_value,
                actual_shares=actual_shares,
                latent=state[0],
                weight=state[1],
                bias=state[2],
                compander_parameters=state[3],
                latent_optimizer=optimizers[0],
                affine_optimizer=optimizers[1],
                compander_optimizer=optimizers[2],
                core_rng=core_rng,
                identity=identity,
            )
            last_checkpoint = checkpoint
            audit_pairs = [tuple(int(value) for value in pair) for pair in config["audit_pairs"]]
            trajectory.append(
                {
                    "step": step,
                    "checkpoint": checkpoint.relative_to(output).as_posix(),
                    "checkpoint_sha256": digest,
                    "atlas": _atlas_metrics(
                        state, prepared, compander_enabled=candidate.compander
                    ),
                    "audit_render": _render_metrics(
                        state,
                        prepared,
                        audit_pairs,
                        compander_enabled=candidate.compander,
                    ),
                    "compander_parameters": [float(value) for value in state[3].detach().cpu()],
                }
            )
            _write_json(
                output / "progress.json",
                {
                    "candidate_id": candidate.candidate_id,
                    "curve": curve,
                    "trajectory": trajectory,
                    "sample_metrics": sample_metrics,
                    "gradient_summaries": gradient_summaries,
                },
            )
            print(json.dumps({"checkpoint": step, "sha256": digest}), flush=True)
    assert last_checkpoint is not None
    load_basecolor_priority_checkpoint(
        last_checkpoint,
        expected_candidate_id=candidate.candidate_id,
        expected_objective_id=candidate.objective_id,
        expected_target_share=candidate.target_share,
        expected_identity=identity,
    )
    all_pairs = [
        (camera, light)
        for light in range(len(prepared["lights"]))
        for camera in range(len(prepared["cameras"]))
    ]
    endpoint_render = _render_metrics(
        state,
        prepared,
        all_pairs if steps == formal_steps else [tuple(pair) for pair in config["audit_pairs"]],
        compander_enabled=candidate.compander,
    )
    report = {
        "schema_version": 3,
        "status": f"complete_{steps}_steps",
        "candidate": asdict(candidate),
        "steps": steps,
        "lambda_value": lambda_value,
        "actual_gradient_shares": actual_shares,
        "identity": identity,
        "runtime_contract": {
            "texture": "2048x2048 linear RGBA8",
            "filtered_samples_per_pixel": 1,
            "decoder": "single unconstrained 4_to_7 affine plus bounded postprocess",
            "instruction_audit": decoder_instruction_audit(
                C4PostprocessConfig(compander=candidate.compander)
            ),
            "ue_gpu_timing_validated": False,
        },
        "trajectory": trajectory,
        "sample_metrics": sample_metrics,
        "gradient_summaries": gradient_summaries,
        "endpoint": {
            **trajectory[-1],
            "full_render_31x6": endpoint_render,
        },
        "curve": curve,
        "wall_seconds": time.perf_counter() - started,
        "formal_holdout_accessed": False,
        "ue_started": False,
        "yellow_diagnostics": {"selection_metric": False},
    }
    _write_json(output / "training_report.json", report)
    return report


def run_posthoc_oracle(
    config_path: Path,
    *,
    source_candidate_id: str,
    source_checkpoint: Path,
    audit_path: Path,
    output: Path,
    steps: int = 1000,
) -> dict[str, Any]:
    """Optimize only G0/G1 on eight repeating frozen batches."""

    if source_candidate_id not in {"BC80", "BC90"} or steps <= 0:
        raise ValueError("post-hoc oracle requires BC80/BC90 and positive steps")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite oracle root: {output}")
    output.mkdir(parents=True)
    config = _load_config(config_path)
    candidate = _candidate_spec(config, source_candidate_id)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    prepared = _prepare(config, config_path)
    if audit.get("lineage") != prepared["lineage"]:
        raise ValueError("oracle gradient audit lineage mismatch")
    lambda_value, actual_shares = _audit_target(audit, candidate)
    identity = _checkpoint_identity(
        config,
        prepared,
        candidate,
        lambda_value,
        actual_shares,
        str(config["source"]["render_pool_config_sha256"]),
    )
    payload = load_basecolor_priority_checkpoint(
        source_checkpoint,
        expected_candidate_id=candidate.candidate_id,
        expected_objective_id=candidate.objective_id,
        expected_target_share=candidate.target_share,
        expected_identity=identity,
    )
    state, _, _ = _new_state(prepared, config)
    for parameter, field in zip(
        state,
        ("latent", "weight", "bias", "compander_parameters"),
        strict=True,
    ):
        parameter.data.copy_(payload[field].to(parameter))
    with torch.no_grad():
        state[3].copy_(torch.tensor((1.0, 0.0), device=state[3].device))
    for parameter in state[:3]:
        parameter.requires_grad_(False)
    frozen_before = hashlib.sha256(
        b"".join(
            value.detach().cpu().contiguous().numpy().tobytes() for value in state[:3]
        )
    ).hexdigest()
    optimizer = torch.optim.Adam(
        (state[3],), lr=float(config["training"]["compander_learning_rate"])
    )
    audit_rng = torch.Generator(device="cuda").manual_seed(int(config["seed"]))
    batches = [
        _draw_batch(prepared, config, audit_rng)
        for _ in range(int(config["audit"]["batches"]))
    ]
    curve = []
    for step in range(1, steps + 1):
        terms, context = _loss_terms(
            state,
            prepared,
            config,
            batches[(step - 1) % len(batches)],
            compander_enabled=True,
        )
        objective = BaseColorObjectiveConfig(
            candidate_id=candidate.candidate_id,
            target_share=candidate.target_share,
            lambda_value=lambda_value,
        )
        total, pieces = compose_basecolor_priority_objective(terms, objective)
        if not torch.isfinite(total):
            raise FloatingPointError(f"non-finite post-hoc oracle loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        optimizer.step()
        if step == 1 or step % 100 == 0:
            curve.append(
                {
                    "step": step,
                    "loss": float(total.detach()),
                    "base_color": float(pieces["base_color"].detach()),
                    "residual": float(pieces["residual"].detach()),
                    "g0": float(state[3][0].detach()),
                    "g1": float(state[3][1].detach()),
                    "camera_index": context["camera_index"],
                    "light_index": context["light_index"],
                }
            )
    frozen_after = hashlib.sha256(
        b"".join(
            value.detach().cpu().contiguous().numpy().tobytes() for value in state[:3]
        )
    ).hexdigest()
    if frozen_before != frozen_after:
        raise RuntimeError("post-hoc oracle changed frozen BaseColor checkpoint state")
    oracle_path = output / "compander_oracle.pt"
    torch.save(
        {
            "schema_version": 1,
            "source_checkpoint_sha256": hashlib.sha256(source_checkpoint.read_bytes()).hexdigest(),
            "source_candidate_id": source_candidate_id,
            "steps": steps,
            "compander_parameters": state[3].detach().cpu(),
            "compander_optimizer": optimizer.state_dict(),
            "identity": identity,
        },
        oracle_path,
    )
    report = {
        "schema_version": 1,
        "status": "complete_posthoc_compander_oracle",
        "source_candidate_id": source_candidate_id,
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256": hashlib.sha256(source_checkpoint.read_bytes()).hexdigest(),
        "steps": steps,
        "frozen_batch_count": len(batches),
        "lambda_value": lambda_value,
        "compander_parameters": [float(value) for value in state[3].detach().cpu()],
        "atlas": _atlas_metrics(state, prepared, compander_enabled=True),
        "audit_render": _render_metrics(
            state,
            prepared,
            [tuple(int(value) for value in pair) for pair in config["audit_pairs"]],
            compander_enabled=True,
        ),
        "curve": curve,
        "frozen_state_hash_before": frozen_before,
        "frozen_state_hash_after": frozen_after,
        "frozen_state_unchanged": True,
        "oracle_sha256": hashlib.sha256(oracle_path.read_bytes()).hexdigest(),
        "selection_metric_yellow": False,
        "formal_holdout_accessed": False,
        "ue_started": False,
    }
    _write_json(output / "oracle_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--candidate")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--allow-compander", action="store_true")
    parser.add_argument("--audit-report", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--posthoc-oracle-from", type=Path)
    parser.add_argument("--source-candidate")
    parser.add_argument("--oracle-steps", type=int, default=1000)
    arguments = parser.parse_args()
    config = _load_config(arguments.config)
    candidate = (
        _candidate_spec(config, arguments.candidate)
        if arguments.candidate is not None
        else None
    )
    if candidate is not None and candidate.compander and not arguments.allow_compander:
        raise ValueError("compander candidates require the manual Gate 1 authorization flag")
    if arguments.posthoc_oracle_from is not None:
        if not arguments.allow_compander or arguments.source_candidate is None:
            raise ValueError("post-hoc oracle requires Gate 1 authorization and source candidate")
        if arguments.audit_report is None or arguments.output_root is None:
            raise ValueError("post-hoc oracle requires audit report and output root")
        output_value = arguments.output_root
        output = _repo_output_path(output_value)
        result = run_posthoc_oracle(
            arguments.config,
            source_candidate_id=arguments.source_candidate,
            source_checkpoint=arguments.posthoc_oracle_from.resolve(),
            audit_path=arguments.audit_report.resolve(),
            output=output,
            steps=arguments.oracle_steps,
        )
    elif arguments.audit_only:
        output = _repo_output_path(
            arguments.output_root or Path(config["audit"]["output_root"])
        )
        result = run_audit(arguments.config, output)
    else:
        if candidate is None:
            raise ValueError("candidate training requires --candidate")
        if arguments.audit_report is None:
            raise ValueError("candidate training requires --audit-report")
        output_value = arguments.output_root or Path(candidate.output_root)
        output = _repo_output_path(output_value)
        result = run_candidate(
            arguments.config,
            candidate_id=candidate.candidate_id,
            audit_path=arguments.audit_report.resolve(),
            output=output,
            max_steps=arguments.max_steps,
            resume=arguments.resume.resolve() if arguments.resume else None,
        )
    print(json.dumps({"status": result["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()
