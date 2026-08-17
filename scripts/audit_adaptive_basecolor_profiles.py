"""Build source-only adaptive BaseColor and render-visibility profiles."""

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
SCRIPTS = ROOT / "scripts"
for directory in (SRC, SCRIPTS):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from cg_frontier.assets.gltf_core4 import load_gltf_core4_asset  # noqa: E402
from cg_frontier.compression.adaptive_basecolor import (  # noqa: E402
    AdaptiveBaseColorProfile,
    RenderColorVisibility,
    build_adaptive_basecolor_profile,
    build_render_color_visibility,
)
from cg_frontier.compression.affine_pca import rasterize_uv_charts  # noqa: E402
from cg_frontier.render.gbuffer import render_geometry_gbuffer  # noqa: E402
from run_scifihelmet_c4_affine_chroma8_l0_40k import _orbit_camera_from_spec  # noqa: E402
from train_simple_nonmetal_c4_affine_render import _camera as _simple_camera  # noqa: E402


DEFAULT_CONFIG = ROOT / "configs/eval/adaptive_basecolor_profiles_v1.yaml"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(json.dumps(list(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _load(path: Path) -> Mapping[str, Any]:
    value = yaml.safe_load(path.read_bytes())
    if not isinstance(value, Mapping):
        raise ValueError(f"mapping expected: {path}")
    return value


def _repo_path(value: Any, label: str) -> Path:
    path = (ROOT / str(value)).resolve()
    if not path.is_relative_to(ROOT) or not path.exists():
        raise ValueError(f"{label} must resolve inside the repository: {value}")
    lowered = {part.lower() for part in path.parts}
    if "formal_holdout" in lowered or "sealed" in lowered:
        raise ValueError(f"{label} touches forbidden data")
    return path


def _profile_manifest(profile: AdaptiveBaseColorProfile) -> dict[str, Any]:
    return {
        "neutral_threshold": profile.neutral_threshold,
        "otsu": profile.otsu.__dict__,
        "k": profile.k,
        "opponent_centroids": profile.opponent_centroids.tolist(),
        "group_sizes": profile.group_sizes.tolist(),
        "group_distortions": list(profile.group_distortions),
        "group_p95_radii": list(profile.group_p95_radii),
        "distortion_curve": list(profile.distortion_curve),
        "jump_curve": list(profile.jump_curve),
        "selected_jump_k": profile.selected_jump_k,
        "source_hash": profile.source_hash,
        "input_hash": profile.input_hash,
        "config_hash": profile.config_hash,
        "profile_hash": profile.profile_hash,
    }


def _visibility_manifest(visibility: RenderColorVisibility) -> dict[str, Any]:
    return {
        "counts": visibility.counts.tolist(),
        "active_mask": visibility.active_mask.tolist(),
        "visible_camera_counts": visibility.visible_camera_counts.tolist(),
        "min_pixels": visibility.min_pixels,
        "min_cameras": visibility.min_cameras,
        "profile_hash": visibility.profile_hash,
        "visibility_hash": visibility.visibility_hash,
    }


def _write_json(path: Path, value: Any) -> str:
    payload = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    digest = hashlib.sha256(payload).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(digest + "\n", encoding="ascii")
    return digest


def run(
    config_path: Path,
    *,
    output_override: str | None = None,
    asset_filter: str | None = None,
) -> dict[str, Any]:
    config = _load(config_path)
    if config.get("formal_holdout_access") != "forbidden":
        raise ValueError("formal holdout must remain forbidden")
    output = (ROOT / str(output_override or config["output_root"])).resolve()
    if not output.is_relative_to(ROOT):
        raise ValueError("output must stay inside the repository")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite output root: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for frozen visibility renders")
    output.mkdir(parents=True)
    profile_config = config["profile"]
    profile_config_hash = hashlib.sha256(
        json.dumps(profile_config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    simple_config = _load(_repo_path(config["simple_render_config"], "simple_render_config"))
    simple_specs = {str(value["id"]): value for value in simple_config["assets"]}
    helmet_pool = _load(_repo_path(config["helmet_render_pool"], "helmet_render_pool"))
    rows: list[dict[str, Any]] = []

    for spec in config["assets"]:
        asset_id = str(spec["id"])
        if asset_filter is not None and asset_id != asset_filter:
            continue
        print(json.dumps({"asset": asset_id, "stage": "load"}), flush=True)
        gltf = _repo_path(spec["gltf"], f"assets.{asset_id}.gltf")
        asset = load_gltf_core4_asset(
            gltf,
            name=asset_id,
            expected_size=tuple(int(v) for v in config["atlas_resolution"]),
            device="cpu",
        )
        valid_mask, _ = rasterize_uv_charts(
            asset.mesh.texcoords,
            asset.mesh.triangles,
            height=asset.targets.height,
            width=asset.targets.width,
        )
        valid_positions = torch.nonzero(valid_mask.reshape(-1), as_tuple=False).flatten()
        source_valid = asset.targets.base_color_linear.reshape(-1, 3)[valid_positions]
        source_hash = _tensor_sha256(source_valid)
        input_hash = hashlib.sha256(
            (_sha256(gltf) + "".join(sorted(asset.textures.source_hashes.values()))).encode("ascii")
        ).hexdigest()
        base_kwargs = {
            "source_hash": source_hash,
            "input_hash": input_hash,
            "config_hash": profile_config_hash,
            "bins": int(profile_config["otsu_bins"]),
            "min_group_size": int(profile_config["min_group_size"]),
            "max_clusters": int(profile_config["max_clusters"]),
            "restarts": int(profile_config["restarts"]),
            "max_iterations": int(profile_config["max_iterations"]),
            "relative_tolerance": float(profile_config["relative_tolerance"]),
            "seed": int(config["seed"]),
        }
        profile = build_adaptive_basecolor_profile(source_valid, **base_kwargs)
        initial_jump_k = profile.selected_jump_k

        kind = str(spec["kind"])
        visibility_config = config["visibility"][kind]
        resolution = tuple(int(v) for v in visibility_config["resolution"])
        camera_uvs: list[torch.Tensor] = []
        camera_masks: list[torch.Tensor] = []
        if kind == "helmet":
            render = helmet_pool["render"]
            cameras = [
                _orbit_camera_from_spec(value, render)
                for value in helmet_pool["train_cameras"]
            ]
        else:
            render = simple_config["render"]
            cameras = [
                _simple_camera(asset.mesh, direction, render)[0]
                for direction in simple_specs[asset_id]["camera_directions"]
            ]
        for camera in cameras:
            geometry = render_geometry_gbuffer(
                asset.mesh,
                camera,
                resolution,
                device="cuda",
                cull_backfaces=(kind == "simple"),
            )
            camera_uvs.append(geometry.torch_buffers["uv"])
            camera_masks.append(geometry.torch_buffers["mask"])

        visibility: RenderColorVisibility | None = None
        last_error: str | None = None
        for candidate_k in range(profile.k, 0, -1):
            if candidate_k != profile.k:
                profile = build_adaptive_basecolor_profile(
                    source_valid, selected_k_override=candidate_k, **base_kwargs
                )
            try:
                visibility = build_render_color_visibility(
                    asset.textures.base_color_linear.to("cuda"),
                    camera_uvs,
                    camera_masks,
                    profile,
                    min_pixels=int(visibility_config["min_pixels"]),
                    min_cameras=int(visibility_config["min_cameras"]),
                )
                break
            except ValueError as error:
                last_error = str(error)
        asset_root = output / asset_id
        asset_root.mkdir()
        if visibility is None:
            row = {
                "asset_id": asset_id,
                "status": "profile_invalid",
                "error": last_error,
                "initial_jump_k": initial_jump_k,
            }
            _write_json(asset_root / "manifest.json", row)
            rows.append(row)
            continue

        torch.save(
            {
                "schema_version": 1,
                "profile": profile,
                "visibility": visibility,
                "valid_positions": valid_positions,
            },
            asset_root / "profile.pt",
        )
        row = {
            "asset_id": asset_id,
            "status": "valid",
            "kind": kind,
            "initial_jump_k": initial_jump_k,
            "selected_k": profile.k,
            "profile": _profile_manifest(profile),
            "visibility": _visibility_manifest(visibility),
            "profile_artifact": "profile.pt",
            "profile_artifact_sha256": _sha256(asset_root / "profile.pt"),
        }
        _write_json(asset_root / "manifest.json", row)
        rows.append(row)
        print(json.dumps({"asset": asset_id, "status": "valid", "k": profile.k}), flush=True)

    report = {
        "schema_version": 1,
        "experiment": config["experiment"],
        "config_sha256": _sha256(config_path),
        "assets": rows,
        "formal_holdout_accessed": False,
        "ue_started": False,
    }
    _write_json(output / "profile_audit_report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root")
    parser.add_argument("--asset")
    args = parser.parse_args()
    report = run(
        args.config.resolve(),
        output_override=args.output_root,
        asset_filter=args.asset,
    )
    print(json.dumps({"status": "complete", "assets": len(report["assets"])}))


if __name__ == "__main__":
    main()
