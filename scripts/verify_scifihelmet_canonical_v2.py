"""Verify canonical renderer v2 against the frozen legacy deployment path."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cg_frontier.assets.gltf_mesh import load_gltf_mesh  # noqa: E402
from cg_frontier.compression.artifact_analysis import sha256_file  # noqa: E402
from cg_frontier.compression.deployment_parity import (  # noqa: E402
    DeploymentParityDecoder,
)
from cg_frontier.compression.filter_aware import postprocess_raw_torch  # noqa: E402
from cg_frontier.compression.material import decode_material  # noqa: E402
from cg_frontier.compression.render_loss import (  # noqa: E402
    load_latent_unorm8_png,
    orbit_camera,
    render_latent_material,
)
from cg_frontier.render.canonical_v2 import (  # noqa: E402
    CanonicalRendererV2,
    LatentMaterialSource,
    ReferenceMaterialSource,
    compare_gradient_probes,
    compare_render_bundles,
)
from cg_frontier.render.gbuffer import render_geometry_gbuffer  # noqa: E402
from cg_frontier.render.pbr import PointLight  # noqa: E402
from train_scifihelmet_repair import _decoder_from_npz  # noqa: E402


def _repo_path(value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"config field {label} must be a path")
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f"config field {label} escapes the repository")
    if "formal_holdout" in path.as_posix().lower():
        raise ValueError(f"sealed evaluation path is forbidden: {label}")
    return path


def _load_config(path: Path) -> dict[str, Any]:
    if "formal_holdout" in path.as_posix().lower():
        raise ValueError("sealed evaluation config is forbidden")
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError("unsupported canonical renderer parity config")
    return value


def _deployment_decoder(legacy: torch.nn.Module) -> DeploymentParityDecoder:
    state = legacy.state_dict()
    decoder = DeploymentParityDecoder(width=8).to(next(legacy.parameters()).device)
    decoder.load_state_dict(
        {
            "hidden.weight": state["network.0.weight"],
            "hidden.bias": state["network.0.bias"],
            "output.weight": state["network.2.weight"],
            "output.bias": state["network.2.bias"],
        }
    )
    return decoder


def _camera(spec: Mapping[str, Any], render: Mapping[str, Any]):
    return orbit_camera(
        yaw_degrees=float(spec["yaw_degrees"]),
        elevation_degrees=float(spec["elevation_degrees"]),
        radius=float(render["camera_radius"]),
        target=tuple(float(value) for value in render["target"]),
        up=tuple(float(value) for value in render["up"]),
        vertical_fov_degrees=float(render["vertical_fov_degrees"]),
        near=float(render["near"]),
        far=float(render["far"]),
    )


def _light(spec: Mapping[str, Any]) -> PointLight:
    return PointLight(
        position=tuple(float(value) for value in spec["position"]),
        color=tuple(float(value) for value in spec["color"]),
        radiant_intensity=float(spec["radiant_intensity"]),
        ambient_intensity=float(spec["ambient_intensity"]),
    )


def _stable_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")


def _evaluate(config: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    inputs = config["inputs"]
    latent_path = _repo_path(inputs["latent_hard_png"], "inputs.latent_hard_png")
    decoder_path = _repo_path(inputs["decoder_npz"], "inputs.decoder_npz")
    actual_hashes = {
        "latent_hard_png": sha256_file(latent_path),
        "decoder_npz": sha256_file(decoder_path),
    }
    if actual_hashes != dict(config["frozen_sha256"]):
        raise ValueError("frozen baseline hashes differ from the approved parity inputs")
    mesh_path = _repo_path(inputs["gltf"], "inputs.gltf")
    mesh = load_gltf_mesh(mesh_path)
    _, latent = load_latent_unorm8_png(latent_path, device=device)
    legacy_decoder = _decoder_from_npz(decoder_path, device)
    decoder = _deployment_decoder(legacy_decoder)
    render = config["render"]
    renderer = CanonicalRendererV2(
        display_exposure=float(render["display_exposure"]),
        minimum_roughness=float(render["minimum_roughness"]),
    )
    light = _light(config["train_lights"][0])
    resolution = tuple(int(value) for value in render["resolution"])
    source_hashes = {
        **actual_hashes,
        "gltf": sha256_file(mesh_path),
        "canonical_v2_source": sha256_file(ROOT / "src/cg_frontier/render/canonical_v2.py"),
    }
    cases: dict[str, Any] = {}
    for camera_spec in config["train_cameras"]:
        camera = _camera(camera_spec, render)
        geometry = render_geometry_gbuffer(
            mesh,
            camera,
            resolution,
            device=device,
            cull_backfaces=True,
        )
        legacy_hdr, legacy_material = render_latent_material(
            geometry,
            camera,
            light,
            latent,
            legacy_decoder,
            quantization="hard",
            minimum_roughness=float(render["minimum_roughness"]),
        )
        legacy_bundle = renderer.render(
            geometry,
            camera,
            light,
            ReferenceMaterialSource(legacy_material),
            input_hashes=source_hashes,
        )
        candidate_bundle = renderer.render(
            geometry,
            camera,
            light,
            LatentMaterialSource(latent, decoder, quantization="hard"),
            input_hashes=source_hashes,
        )
        report = compare_render_bundles(legacy_bundle, candidate_bundle)
        cases[str(camera_spec["name"])] = {
            "passed": report.passed,
            "metrics": dict(report.metrics),
            "thresholds": dict(report.thresholds),
            "legacy_wrapper_hdr_max_abs": float(
                (legacy_bundle.linear_hdr - legacy_hdr).abs().max().detach().cpu()
            ),
        }

    generator = torch.Generator(device="cpu").manual_seed(int(config["analysis"]["seed"]))
    probes = torch.rand((32, 4), generator=generator, dtype=torch.float32).to(device)

    def legacy_probe(values: torch.Tensor) -> torch.Tensor:
        decoded = decode_material(legacy_decoder, values)  # type: ignore[arg-type]
        return torch.cat(
            (
                decoded.base_color_linear,
                decoded.normal_xyz,
                decoded.roughness,
                decoded.metallic,
            ),
            dim=-1,
        )

    def canonical_probe(values: torch.Tensor) -> torch.Tensor:
        decoded = postprocess_raw_torch(decoder(values))
        return torch.cat(
            (
                decoded.base_color_linear,
                decoded.normal_xyz,
                decoded.roughness,
                decoded.metallic,
            ),
            dim=-1,
        )

    gradients = compare_gradient_probes(legacy_probe, canonical_probe, probes)
    passed = all(case["passed"] for case in cases.values()) and bool(gradients["passed"])
    return {
        "schema_version": 1,
        "renderer": CanonicalRendererV2.renderer_identifier,
        "passed": passed,
        "formal_holdout_accessed": False,
        "input_hashes": source_hashes,
        "cases": cases,
        "gradient_probes": gradients,
    }


def run(config_path: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    if not torch.cuda.is_available():
        raise RuntimeError("canonical renderer parity requires the existing CUDA environment")
    device = torch.device("cuda")
    first = _stable_json(_evaluate(config, device))
    second = _stable_json(_evaluate(config, device))
    output_dir = _repo_path(config["output_dir"], "output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "parity_run_a.json").write_bytes(first)
    (output_dir / "parity_run_b.json").write_bytes(second)
    identical = first == second
    summary = {
        "schema_version": 1,
        "passed": bool(identical and json.loads(first)["passed"]),
        "byte_identical": identical,
        "report_sha256": hashlib.sha256(first).hexdigest() if identical else None,
        "formal_holdout_accessed": False,
    }
    (output_dir / "parity_summary.json").write_bytes(_stable_json(summary))
    if not identical:
        raise RuntimeError("canonical renderer parity reports are not byte-identical")
    if not summary["passed"]:
        raise RuntimeError("canonical renderer v2 failed its immutable takeover gates")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/eval/scifihelmet_canonical_v2_parity.yaml",
    )
    args = parser.parse_args()
    print(json.dumps(run(args.config.resolve()), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
